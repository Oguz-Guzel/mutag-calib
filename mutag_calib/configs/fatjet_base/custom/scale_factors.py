import awkward as ak
import numpy as np

import correctionlib
from functools import lru_cache

#from mutag_calib.configs.fatjet_base.custom.parameters.pt_reweighting.pt_reweighting import pt_corrections, pteta_corrections

def pt_reweighting(events, year):
    '''Reweighting scale factor based on the leading fatjet pT'''
    cat = 'pt350msd40'
    cset = correctionlib.CorrectionSet.from_file(pt_corrections[year])
    pt_corr = cset[f'pt_corr_{year}']

    '''In case the jet pt is higher than 1500 GeV, the pt is padded to 0
    and a correction SF of 1 is returned.'''
    pt = events.FatJetGood.pt[:,0]
    pt = ak.where(pt < 1500, pt, 0)

    return pt_corr.evaluate(cat, pt)

def pteta_reweighting(events, year):
    '''Reweighting scale factor based on the leading fatjet pT'''
    cat = 'pt350msd40'
    cset = correctionlib.CorrectionSet.from_file(pteta_corrections[year])
    pteta_corr = cset[f'pt_eta_2D_corr_{year}']

    '''In case the jet pt is higher than 1500 GeV, the pt is padded to 0
    and a correction SF of 1 is returned.'''
    pt  = events.FatJetGood.pt[:,0]
    eta = events.FatJetGood.eta[:,0]
    pt = ak.where(pt < 1500, pt, 0)

    return pteta_corr.evaluate(cat, pt, eta)

def sf_trigger_prescale(events, year, params):
    '''Trigger prescale factor'''
    # Here we assume that both BTagMu_AK4Jet300_Mu5 and BTagMu_AK8Jet170_DoubleMu5 triggers have a prescale of 1
    sf = ak.Array(len(events)*[1.0])
    pass_unprescaled_triggers = events.HLT["BTagMu_AK4Jet300_Mu5"] | events.HLT["BTagMu_AK8Jet170_DoubleMu5"]
    sf = ak.where(events.HLT["BTagMu_AK8Jet300_Mu5"] & (~pass_unprescaled_triggers), 1. / params["HLT_triggers_prescales"][year]["BTagMu"]["BTagMu_AK8Jet300_Mu5"], sf)
    sf = ak.where(events.HLT["BTagMu_AK8DiJet170_Mu5"] & (~events.HLT["BTagMu_AK8Jet300_Mu5"]) & (~pass_unprescaled_triggers), 1. / params["HLT_triggers_prescales"][year]["BTagMu"]["BTagMu_AK8DiJet170_Mu5"], sf)

    return sf

def sf_ptetatau21_reweighting(events, year, params):
    '''Correction of jets observable by a 3D reweighting based on (pT, eta, tau21).
    The function returns the nominal, up and down weights, where the up/down variations are computed considering the statistical uncertainty on data and MC.'''


    cset = correctionlib.CorrectionSet.from_file(params["ptetatau21_reweighting"][year])
    key = list(cset.keys())[0]
    corr = cset[key]

    cat = "inclusive"
    nfatjet  = ak.num(events.FatJetGood.pt)
    pos = ak.flatten(ak.local_index(events.FatJetGood.pt))
    pt = ak.flatten(events.FatJetGood.pt)
    eta = ak.flatten(events.FatJetGood.eta)
    tau21 = ak.flatten(events.FatJetGood.tau21)

    weight = {}
    for var in ["nominal", "statUp", "statDown"]:
        w = corr.evaluate(cat, var, pos, pt, eta, tau21)
        weight[var] = ak.unflatten(w, nfatjet)

    return weight["nominal"], weight["statUp"], weight["statDown"]


# HHbbww AK8 scale factors (per era) from correctionlib JSONs produced by export_correctionlib_all.py

_HHBBWW_SF_PATHS = {
    "2022_preEE": "/afs/cern.ch/work/a/aguzel/private/bbww_ak8_sf_derivation/ak8_sf_jsons/sf_corrections_run3_2022_preEE.json",
    "2022_postEE": "/afs/cern.ch/work/a/aguzel/private/bbww_ak8_sf_derivation/ak8_sf_jsons/sf_corrections_run3_2022_postEE.json",
    "2023_preBPix": "/afs/cern.ch/work/a/aguzel/private/bbww_ak8_sf_derivation/ak8_sf_jsons/sf_corrections_run3_2023_preBPix.json",
    "2023_postBPix": "/afs/cern.ch/work/a/aguzel/private/bbww_ak8_sf_derivation/ak8_sf_jsons/sf_corrections_run3_2023_postBPix.json",
}


@lru_cache(maxsize=None)
def _hhbbww_load(year: str):
    path = _HHBBWW_SF_PATHS.get(year)
    if path is None:
        raise KeyError(f"No HHbbww SF path configured for year: {year}")
    cset = correctionlib.CorrectionSet.from_file(path)
    bb = cset[f"HHbbww_{year}_SF_bb"]
    cc = cset[f"HHbbww_{year}_SF_cc"]
    return bb, cc


def _hhbbww_per_jet_variations(corr, pt_flat, counts):
    # Evaluate all needed systematics to build a total symmetric uncertainty
    nominal = corr.evaluate(pt_flat, "nominal")
    up = corr.evaluate(pt_flat, "up")
    down = corr.evaluate(pt_flat, "down")
    tau21_up = corr.evaluate(pt_flat, "tau21Up")
    tau21_down = corr.evaluate(pt_flat, "tau21Down")

    delta_sq = (
        (up - nominal) ** 2
        + (nominal - down) ** 2
        + (tau21_up - nominal) ** 2
        + (nominal - tau21_down) ** 2
    )
    sigma = np.sqrt(delta_sq)

    variants = {
        "nominal": nominal,
        "totalUp": nominal + sigma,
        "totalDown": nominal - sigma,
    }

    # Unflatten back to jet structure
    return {k: ak.unflatten(v, counts) for k, v in variants.items()}


def sf_hhbbww(events, year, systematic="nominal", flavor="bb"):
    """Event-level HHbbww SF using leading+subleading jets independently.

    systematic: nominal|totalUp|totalDown (total built from up/down/tau21Up/tau21Down in quadrature)
    flavor: bb (uses HHbbww_*_SF_bb) or cc (uses HHbbww_*_SF_cc)
    """
    bb, cc = _hhbbww_load(year)
    corr = bb if flavor == "bb" else cc if flavor == "cc" else None
    if corr is None:
        raise ValueError("flavor must be 'bb' or 'cc'")

    pt = events.FatJetGood.pt
    counts = ak.num(pt)
    pt_flat = ak.flatten(pt)

    per_jet_variants = _hhbbww_per_jet_variations(corr, pt_flat, counts)
    if systematic not in per_jet_variants:
        raise ValueError("systematic must be one of: nominal, totalUp, totalDown")

    per_jet = per_jet_variants[systematic]
    # Extract leading and subleading; default to 1 when jet is missing
    leading = ak.fill_none(ak.firsts(per_jet), 1.0)
    subleading = ak.fill_none(ak.pad_none(per_jet, 2)[:, 1], 1.0)

    return leading * subleading
