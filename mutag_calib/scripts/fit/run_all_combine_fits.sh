#!/bin/bash

# filepath: run_combine_all.sh
# Usage: ./run_cards.sh [datacards directory] [channel]
# Both base_dir and channel are required. If not provided, print error and exit.
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Error: Usage: $0 [base_dir] [channel]" >&2
  return 1 2>/dev/null || exit 1
fi
base_dir="$1"
channel="$2"

for d in $base_dir/202*/*/tau21*; do
  if [ -d "$d" ]; then
    echo "Processing $d"
    cd "$d"
    if [ -f combine_cards.sh ]; then
      source combine_cards.sh
      combine -M FitDiagnostics -d workspace.root --saveWorkspace \
        --name .msd-80to170_Pt-300toInf_particleNet_XbbVsQCD-${channel} --cminDefaultMinimizerStrategy 2 \
        --robustFit=1 --saveShapes --saveWithUncertainties --saveOverallShapes \
        --redefineSignalPOIs=r,SF_c,SF_light --setParameters SF_light=1 --freezeParameters SF_light \
        --robustHesse=1 --stepSize=0.001 --X-rtd=MINIMIZER_analytic --X-rtd MINIMIZER_MaxCalls=9999999 \
        --cminFallbackAlgo Minuit2,Migrad,0:0.2 --X-rtd FITTER_NEW_CROSSING_ALGO --X-rtd FITTER_NEVER_GIVE_UP --X-rtd FITTER_BOUND
    else
      echo "combine_cards.sh not found in $d"
    fi
    cd - > /dev/null
  fi
done