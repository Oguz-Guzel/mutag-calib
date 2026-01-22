#!/usr/bin/env python3
"""Combine AK8 SF correctionlib JSONs across pT bins into a single file per era.

Input files (already produced) must follow the pattern:
    ak8_sf_jsons/ak8_sf_msdtest_Pt-<ptTag>__<era>.json
where ptTag in {300to350, 350to425, 425toInf} and era in
{2022_preEE, 2022_postEE, 2023_preBPix, 2023_postBPix}.

Output files:
    ak8_sf_jsons/ak8_sf_msdtest_Pt-<ptTag>__<era>.json
Each output contains two corrections (bb and cc) with a binning over pT:
    edges: 300, 350, 425, 20000
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent / "ak8_sf_jsons"
PT_BINS = [
    ("300to350", 300.0, 350.0),
    ("350to425", 350.0, 425.0),
    ("425toInf", 425.0, 20000.0),
]
ERAS = [
    "2022_preEE",
    "2022_postEE",
    "2023_preBPix",
    "2023_postBPix",
]
SYSTEMATICS = ["nominal", "up", "down", "tau21Up", "tau21Down", "msdUp", "msdDown"]


def load_values(pt_tag: str, era: str) -> Dict[str, Dict[str, float]]:
    """Load SF values for one (pt_tag, era); returns mapping corr_name -> syst -> val."""
    path = BASE_DIR / f"ak8_sf_msdtest_Pt-{pt_tag}__{era}.json"
    with path.open() as f:
        data = json.load(f)

    values: Dict[str, Dict[str, float]] = {}
    for corr in data.get("corrections", []):
        syst_entries = corr["data"]["content"][0]["content"]
        values[corr["name"]] = {entry["key"]: entry["value"] for entry in syst_entries}
    return values


def build_correction(name: str, descr: str, per_bin_values: List[Dict[str, float]]) -> Dict:
    """Build a correctionlib binning over pT with category over systematic."""
    content = []
    for syst_map in per_bin_values:
        content.append(
            {
                "nodetype": "category",
                "input": "systematic",
                "content": [{"key": k, "value": syst_map[k]} for k in SYSTEMATICS],
                "default": None,
            }
        )

    return {
        "name": name,
        "description": descr,
        "version": 1,
        "inputs": [
            {"name": "pt", "type": "real", "description": "AK8 jet pT (GeV)"},
            {
                "name": "systematic",
                "type": "string",
                "description": "nominal|up|down|tau21Up|tau21Down|msdUp|msdDown",
            },
        ],
        "output": {"name": "sf", "type": "real", "description": "scale factor"},
        "generic_formulas": None,
        "data": {
            "nodetype": "binning",
            "input": "pt",
            "edges": [b[1] for b in PT_BINS] + [PT_BINS[-1][2]],
            "content": content,
            "flow": "clamp",
        },
    }


def combine_era(era: str) -> None:
    """Combine three pT-bin files into a single correctionlib JSON for one era."""
    per_bin_values_bb: List[Dict[str, float]] = []
    per_bin_values_cc: List[Dict[str, float]] = []

    for pt_tag, _, _ in PT_BINS:
        vals = load_values(pt_tag, era)
        # Find the bb and cc corrections in this file
        corr_bb = next((v for k, v in vals.items() if k.endswith("_SF_bb")), None)
        corr_cc = next((v for k, v in vals.items() if k.endswith("_SF_cc")), None)
        if corr_bb is None or corr_cc is None:
            raise RuntimeError(f"Missing bb/cc corrections in {pt_tag} {era}")
        per_bin_values_bb.append(corr_bb)
        per_bin_values_cc.append(corr_cc)

    corr_bb_out = build_correction(
        name=f"HHbbww_{era}_SF_bb",
        descr=f"HHbbww Run3 scale factors | combined pT bins | {era} | r",
        per_bin_values=per_bin_values_bb,
    )
    corr_cc_out = build_correction(
        name=f"HHbbww_{era}_SF_cc",
        descr=f"HHbbww Run3 scale factors | combined pT bins | {era} | SF_c",
        per_bin_values=per_bin_values_cc,
    )

    output = {
        "schema_version": 2,
        "description": f"HHbbww Run3 scale factors combined pT bins (300-350, 350-425, 425-Inf) for {era}",
        "corrections": [corr_bb_out, corr_cc_out],
        "compound_corrections": None,
    }

    out_path = BASE_DIR / f"ak8_sf_msdtest_Pt-combined_{era}.json"
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {out_path}")


def main() -> None:
    for era in ERAS:
        combine_era(era)


if __name__ == "__main__":
    main()
