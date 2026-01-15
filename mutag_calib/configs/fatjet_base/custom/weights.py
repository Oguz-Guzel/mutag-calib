from pocket_coffea.lib.weights.weights import WeightLambda
from mutag_calib.configs.fatjet_base.custom.scale_factors import (
    pt_reweighting,
    pteta_reweighting,
    sf_ptetatau21_reweighting,
    sf_trigger_prescale,
    sf_hhbbww,
)

pt_weight = WeightLambda.wrap_func(
    name="pt_reweighting",
    function= lambda events, size, metadata, shape_variation: [("pt_reweighting", pt_reweighting(events, metadata['year']))]
)

pteta_weight = WeightLambda.wrap_func(
    name="pteta_reweighting",
    function= lambda events, size, metadata, shape_variation: [("pteta_reweighting", pteta_reweighting(events, metadata['year']))]
)

SF_trigger_prescale = WeightLambda.wrap_func(
    name="sf_trigger_prescale",
    function=lambda params, metadata, events, size, shape_variations:
        sf_trigger_prescale(events, metadata['year'], params),
    has_variations=False,
    )

SF_ptetatau21_reweighting = WeightLambda.wrap_func(
    name="sf_ptetatau21_reweighting",
    function=lambda params, metadata, events, size, shape_variations:
        sf_ptetatau21_reweighting(events, metadata['year'], params),
    has_variations=True
)

# HHbbww AK8 scale factors (per era) from correctionlib JSONs under ak8_sf_jsons/
# Variations include: nominal, totalUp, totalDown (total combines up/down/tau21Up/tau21Down in quadrature).
SF_hhbbww_bb = WeightLambda.wrap_func(
    name="sf_hhbbww_bb",
    function=lambda params, metadata, events, size, shape_variations:
        [
            (
                "sf_hhbbww_bb",
                sf_hhbbww(
                    events,
                    metadata['year'],
                    systematic=shape_variations.get("sf_hhbbww_bb", "nominal"),
                    flavor="bb",
                ),
            )
        ],
    has_variations=True,
)

SF_hhbbww_cc = WeightLambda.wrap_func(
    name="sf_hhbbww_cc",
    function=lambda params, metadata, events, size, shape_variations:
        [
            (
                "sf_hhbbww_cc",
                sf_hhbbww(
                    events,
                    metadata['year'],
                    systematic=shape_variations.get("sf_hhbbww_cc", "nominal"),
                    flavor="cc",
                ),
            )
        ],
    has_variations=True,
)
