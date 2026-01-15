"""
Utilities to apply the HHbbww AK8 scale factors (per era) derived from the
`sf_corrections_run3_<era>.json` correctionlib files.

The corrections exported by `export_correctionlib_all.py` have inputs:
  - pt (float, GeV)
  - systematic (string): nominal | up | down | tau21Up | tau21Down

Each era file contains two corrections:
  - HHbbww_<era>_SF_bb
  - HHbbww_<era>_SF_cc

This helper loads a chosen era file and evaluates per-jet or per-event weights.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict

import awkward as ak
import correctionlib


@lru_cache(maxsize=None)
def _load_corrections(sf_path: str, year: str):
    """Load correction objects for a given year, with a small cache."""
    if not os.path.exists(sf_path):
        raise FileNotFoundError(f"Scale factor file not found: {sf_path}")
    cset = correctionlib.CorrectionSet.from_file(sf_path)
    bb_name = f"HHbbww_{year}_SF_bb"
    cc_name = f"HHbbww_{year}_SF_cc"
    if bb_name not in cset or cc_name not in cset:
        raise KeyError(
            f"Corrections {bb_name} / {cc_name} not found in {sf_path}. "
            "Check that the file corresponds to the requested era."
        )
    return cset[bb_name], cset[cc_name]


def hhbbww_sf_per_jet(
    events, year: str, sf_path: str, systematic: str = "nominal"
) -> Dict[str, ak.Array]:
    """Return per-jet SFs for bb and cc hypotheses.

    Parameters
    ----------
    events : awkward.Array
        Event record with `FatJetGood.pt` (uses all jets in the collection).
    year : str
        Era label matching the file, e.g. 2022_preEE, 2022_postEE, 2023_preBPix, 2023_postBPix.
    sf_path : str
        Path to the per-era correctionlib JSON file (e.g. sf_corrections_run3_2022_preEE.json).
    systematic : str
        One of: nominal, up, down, tau21Up, tau21Down.

    Returns
    -------
    dict
        {"bb": per-jet SF array, "cc": per-jet SF array}, each shaped like FatJetGood.pt.
    """
    corr_bb, corr_cc = _load_corrections(sf_path, year)
    pt_flat = ak.flatten(events.FatJetGood.pt)
    counts = ak.num(events.FatJetGood.pt)

    sf_bb = ak.unflatten(corr_bb.evaluate(pt_flat, systematic), counts)
    sf_cc = ak.unflatten(corr_cc.evaluate(pt_flat, systematic), counts)
    return {"bb": sf_bb, "cc": sf_cc}


def hhbbww_sf_event_weight(
    events, year: str, sf_path: str, systematic: str = "nominal", flavor: str = "bb"
) -> ak.Array:
    """Return an event-level SF by multiplying the per-jet SFs along the jet axis.

    Parameters
    ----------
    events : awkward.Array
        Event record with `FatJetGood.pt`.
    year : str
        Era label matching the file.
    sf_path : str
        Path to the per-era correctionlib JSON file.
    systematic : str
        One of: nominal, up, down, tau21Up, tau21Down.
    flavor : str
        Which correction to use: "bb" or "cc".
    """
    per_jet = hhbbww_sf_per_jet(events, year, sf_path, systematic)
    if flavor not in per_jet:
        raise ValueError("flavor must be 'bb' or 'cc'")
    # Product over jets in the event (if you only want the leading jet, slice before calling)
    return ak.prod(per_jet[flavor], axis=1)


__all__ = [
    "hhbbww_sf_per_jet",
    "hhbbww_sf_event_weight",
]
