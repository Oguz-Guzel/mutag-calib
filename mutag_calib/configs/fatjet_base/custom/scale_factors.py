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
    "2022_preEE": "/afs/cern.ch/work/a/aguzel/private/bbww_ak8_sf_derivation/ak8_sf_jsons/ak8_sf_corrections_bbww_combined_2022_preEE.json",
    "2022_postEE": "/afs/cern.ch/work/a/aguzel/private/bbww_ak8_sf_derivation/ak8_sf_jsons/ak8_sf_corrections_bbww_combined_2022_postEE.json",
    "2023_preBPix": "/afs/cern.ch/work/a/aguzel/private/bbww_ak8_sf_derivation/ak8_sf_jsons/ak8_sf_corrections_bbww_combined_2023_preBPix.json",
    "2023_postBPix": "/afs/cern.ch/work/a/aguzel/private/bbww_ak8_sf_derivation/ak8_sf_jsons/ak8_sf_corrections_bbww_combined_2023_postBPix.json",
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


def _hhbbww_variations(corr_bb, corr_cc, pt_flat, counts, flavor_flat, nbh_flat, nch_flat):
    """Per-jet SF choosing bb or cc map based on gen flavor; others get weight 1."""
    mask_bb = (flavor_flat == 5) & (nbh_flat >= 2)
    mask_cc = (flavor_flat == 4) & (nch_flat >= 2) & (nbh_flat == 0)

    # Evaluate both correction sets
    bb_vars = _hhbbww_per_jet_variations(corr_bb, pt_flat, counts)
    cc_vars = _hhbbww_per_jet_variations(corr_cc, pt_flat, counts)

    variants = {}
    for key in ["nominal", "totalUp", "totalDown"]:
        v_bb = bb_vars[key]
        v_cc = cc_vars[key]
        # Start from 1 for non-bb/cc jets
        ones = ak.ones_like(v_bb, dtype=float)
        v = ak.where(mask_bb, v_bb, ones)
        v = ak.where(mask_cc, v_cc, v)
        variants[key] = v
    return variants

def sf_hhbbww(events, year, systematic="nominal"):
    """Event-level HHbbww SF that dispatches bb/cc corrections per jet by flavor.

    Jets identified as bb use the bb map; cc jets use the cc map; others get weight 1.
    """
    corr_bb, corr_cc = _hhbbww_load(year)

    pt = events.FatJetGood.pt
    counts = ak.num(pt)
    pt_flat = ak.flatten(pt)

    flavor_flat = ak.flatten(events.FatJetGood.hadronFlavour)
    nbh_flat = ak.flatten(events.FatJetGood.nBHadrons)
    nch_flat = ak.flatten(events.FatJetGood.nCHadrons)

    per_jet_variants = _hhbbww_variations(corr_bb, corr_cc, pt_flat, counts, flavor_flat, nbh_flat, nch_flat)
    if systematic not in per_jet_variants:
        raise ValueError("systematic must be one of: nominal, totalUp, totalDown")

    per_jet = per_jet_variants[systematic]
    leading = ak.fill_none(ak.firsts(per_jet), 1.0)
    subleading = ak.fill_none(ak.pad_none(per_jet, 2)[:, 1], 1.0)
    return leading * subleading
