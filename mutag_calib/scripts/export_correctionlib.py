#!/usr/bin/env python3

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class POIResult:
    name: str
    value: float
    err_up: float
    err_down: float

    @property
    def up(self) -> float:
        return float(self.value + self.err_up)

    @property
    def down(self) -> float:
        return float(self.value - self.err_down)


def _find_unique_dir(parent: str, pattern: str) -> str:
    matches = sorted([p for p in glob.glob(os.path.join(parent, pattern)) if os.path.isdir(p)])
    if len(matches) == 0:
        raise FileNotFoundError(f"No directory matching '{pattern}' under: {parent}")
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple directories match '{pattern}' under {parent}.\n"
            "Refine the pattern or clean the folder:\n" + "\n".join(matches)
        )
    return matches[0]


def _find_fit_file(fit_dir: str) -> str:
    patterns = [
        os.path.join(fit_dir, "fitDiagnostics*.root"),
        os.path.join(fit_dir, "fitDiagnostics_*.root"),
        os.path.join(fit_dir, "fitDiagnostics.root"),
    ]
    matches: List[str] = []
    for pat in patterns:
        matches.extend(glob.glob(pat))
    matches = sorted(set(matches))

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No fitDiagnostics ROOT file found in: {fit_dir}\nTried: {patterns}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            "Multiple fitDiagnostics ROOT files found. Please clean up or choose explicitly:\n"
            + "\n".join(matches)
        )
    return matches[0]


def _load_fit_s(fit_file: str):
    try:
        import ROOT  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Failed to import ROOT (PyROOT). Source your runtime first, e.g.\n"
            "  source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc11-opt/setup.sh\n"
            "and activate your venv if needed."
        ) from exc

    tf = ROOT.TFile.Open(fit_file)
    if not tf or tf.IsZombie():
        raise RuntimeError(f"Failed to open ROOT file: {fit_file}")

    fit_s = tf.Get("fit_s")
    if not fit_s:
        keys = [k.GetName() for k in tf.GetListOfKeys()]
        raise KeyError(f"Object 'fit_s' not found in {fit_file}. Keys include: {keys[:20]}")

    return tf, fit_s


def _get_poi_result(fit_s, poi: str) -> POIResult:
    par = fit_s.floatParsFinal().find(poi)
    if not par:
        raise KeyError(f"POI '{poi}' not found in fit_s.floatParsFinal()")

    val = float(par.getVal())

    # Prefer asymmetric errors if present
    try:
        err_hi = float(par.getErrorHi())
        err_lo = float(abs(par.getErrorLo()))
        if err_hi == 0.0 and err_lo == 0.0:
            raise RuntimeError
        return POIResult(name=poi, value=val, err_up=err_hi, err_down=err_lo)
    except Exception:
        err = float(par.getError())
        return POIResult(name=poi, value=val, err_up=err, err_down=err)


def _systematic_category(res: POIResult, tau21_unc: Optional[float] = None) -> dict:
    content = [
        {"key": "nominal", "value": float(res.value)},
        {"key": "up", "value": float(res.up)},
        {"key": "down", "value": float(res.down)},
    ]

    if tau21_unc is not None:
        tau21_unc = float(tau21_unc)
        content.extend(
            [
                {"key": "tau21Up", "value": float(res.value + tau21_unc)},
                {"key": "tau21Down", "value": float(res.value - tau21_unc)},
            ]
        )

    return {
        "nodetype": "category",
        "input": "systematic",
        "content": content,
        "default": None,
    }


def _pt_binning(
    pt_edges: List[float], res: POIResult, tau21_unc: Optional[float] = None
) -> dict:
    if len(pt_edges) != 2:
        raise ValueError("This exporter expects exactly one pT bin (2 edges)")
    return {
        "nodetype": "binning",
        "input": "pt",
        "edges": [float(pt_edges[0]), float(pt_edges[1])],
        "content": [_systematic_category(res, tau21_unc=tau21_unc)],
        "flow": "clamp",
    }


def build_correction(
    name: str,
    description: str,
    poi_name: str,
    nominal: POIResult,
    pt_edges: List[float],
    tau21_unc: float,
) -> dict:
    data = _pt_binning(pt_edges, nominal, tau21_unc=tau21_unc)
    inputs = [
        {
            "name": "pt",
            "type": "real",
            "description": "AK8 jet pT (GeV)",
        },
        {
            "name": "systematic",
            "type": "string",
            "description": "nominal|up|down|tau21Up|tau21Down",
        },
    ]

    return {
        "name": name,
        "description": f"{description} | {poi_name}",
        "version": 1,
        "inputs": inputs,
        "output": {"name": "sf", "type": "real", "description": "scale factor"},
        "generic_formulas": None,
        "data": data,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export correctionlib JSONs (schema v2) for Run-3 HHbbww SFs, "
            "one JSON per era (2022/2023), using ONLY the Pt-300toInf datacard directories."
        )
    )
    parser.add_argument(
        "--datacards-dir",
        default=os.path.join(
            os.path.dirname(__file__), "fit_templates", "datacards"
        ),
        help="Base datacards directory",
    )
    parser.add_argument(
        "--output",
        default="sf_corrections_run3",
        help=(
            "Output base name used as a stem. Files are written as: "
            "<output-stem>_<era>.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write per-era output JSONs",
    )
    parser.add_argument(
        "--pt-pattern",
        default="*Pt-300toInf_*",
        help="Glob pattern for the pT-bin directory to use (default selects Pt-300toInf)",
    )
    parser.add_argument(
        "--tau21-nominal",
        default="0p30",
        help=(
            "Nominal tau21 working point label used for the central value, "
            "corresponding to a folder tau21_<label> (default: 0p30)"
        ),
    )
    parser.add_argument(
        "--tau21-alternatives",
        default="0p20,0p25,0p35,0p40",
        help=(
            "Comma-separated list of tau21 alternative working point labels used to derive "
            "the tau21 systematic uncertainty as max(|SF_alt - SF_nom|). "
            "Each label must correspond to a folder tau21_<label>."
        ),
    )
    parser.add_argument(
        "--eras",
        default="2022_preEE,2022_postEE,2023_preBPix,2023_postBPix",
        help=(
            "Comma-separated list of eras."
        ),
    )
    parser.add_argument(
        "--poi",
        action="append",
        default=None,
        help=(
            "POI name to export from fit_s (repeatable). "
            "If omitted, exports both bb and cc rateParams: r (bb) and SF_c (cc)."
        ),
    )
    parser.add_argument(
        "--description",
        default="HHbbww Run3 scale factors (from Combine FitDiagnostics) | Pt-300toInf only",
        help="Top-level description",
    )

    args = parser.parse_args()

    # By default export both bb and cc scale factors as defined in the datacards:
    #   r      rateParam * b_*  (bb)
    #   SF_c   rateParam * c_*  (cc)
    pois = args.poi if args.poi else ["r", "SF_c"]

    poi_alias = {
        "r": "SF_bb",
        "SF_c": "SF_cc",
    }

    pt_edges = [300,20000]
    if len(pt_edges) != 2:
        raise SystemExit("--pt-edges must have exactly 2 numbers, e.g. 300,20000")

    years = [era for era in args.eras.split(",")]

    # values_by_poi[poi][year][tau21] = POIResult
    values_by_poi: Dict[str, Dict[str, Dict[str, POIResult]]] = {p: {} for p in pois}

    tau21_nominal = args.tau21_nominal.strip()
    tau21_alternatives = [t.strip() for t in args.tau21_alternatives.split(",") if t.strip()]
    if len(tau21_alternatives) == 0:
        raise SystemExit("--tau21-alternatives must contain at least one label")
    required_tau21 = [tau21_nominal, *tau21_alternatives]

    for year in years:
        year_dir = os.path.join(args.datacards_dir, year)
        if not os.path.isdir(year_dir):
            raise FileNotFoundError(f"Missing year directory: {year_dir}")

        pt_dir = _find_unique_dir(year_dir, args.pt_pattern)
        tau_dirs = sorted(
            [
                p
                for p in glob.glob(os.path.join(pt_dir, "tau21_*"))
                if os.path.isdir(p)
            ]
        )
        if len(tau_dirs) == 0:
            raise FileNotFoundError(f"No tau21_* subdirectories found under: {pt_dir}")

        found_tau21: set[str] = set()

        for tau_dir in tau_dirs:
            tau21_label = os.path.basename(tau_dir).replace("tau21_", "")
            found_tau21.add(tau21_label)
            fit_file = _find_fit_file(tau_dir)

            tf, fit_s = _load_fit_s(fit_file)
            try:
                for poi in pois:
                    values_by_poi.setdefault(poi, {}).setdefault(year, {})[tau21_label] = _get_poi_result(
                        fit_s, poi
                    )
            finally:
                tf.Close()

        missing = [t for t in required_tau21 if t not in found_tau21]
        if missing:
            raise FileNotFoundError(
                f"Missing required tau21 folders under: {pt_dir}\n"
                f"Missing: {missing}\n"
                f"Found: {sorted(found_tau21)}"
            )

    def _validate_and_write(path: str, cset_obj: dict, corr_names: List[str]) -> None:
        # Optional validation
        try:
            import correctionlib.schemav2 as cs  # type: ignore

            cs.CorrectionSet.model_validate(cset_obj)
        except Exception:
            pass

        with open(path, "w") as f:
            json.dump(cset_obj, f, indent=2)
        print(f"Wrote: {path}")
        for n in corr_names:
            print(" -", n)

    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.basename(args.output)
    stem, ext = os.path.splitext(base)
    if ext.lower() != ".json":
        ext = ".json"

    for year in years:
        per_year_corrections: List[dict] = []
        for poi in pois:
            poi_label = poi_alias.get(poi, poi)
            corr_name = f"HHbbww_{year}_{poi_label}"

            nominal = values_by_poi[poi][year][tau21_nominal]
            tau21_unc = 0.0
            for alt in tau21_alternatives:
                alt_res = values_by_poi[poi][year][alt]
                tau21_unc = max(tau21_unc, abs(float(alt_res.value) - float(nominal.value)))

            per_year_corrections.append(
                build_correction(
                    name=corr_name,
                    description=f"{args.description} | {year}",
                    poi_name=poi,
                    nominal=nominal,
                    pt_edges=pt_edges,
                    tau21_unc=tau21_unc,
                )
            )

        out_path = os.path.join(args.output_dir, f"{stem}_{year}{ext}")
        cset = {
            "schema_version": 2,
            "description": f"{args.description} | {year}",
            "corrections": per_year_corrections,
            "compound_corrections": None,
        }
        _validate_and_write(out_path, cset, [c["name"] for c in per_year_corrections])


if __name__ == "__main__":
    main()
