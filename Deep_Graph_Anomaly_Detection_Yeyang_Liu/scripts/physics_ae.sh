#!/usr/bin/env bash
set -euo pipefail

# AE trained on LHCO sidebands, frozen, then an SR/SB tree readout for each signal.
seed="${SEED:-42}"
variant="${VARIANT:-ratios}"           # predicted, graph_mass, errors, ratios
device="${DEVICE:-cpu}"                # cpu, cuda:0, ...
run_dir="${RUN_DIR:-runs/physics_ae/sb}"
signals=("$@")                         # default: all five signals
if (( ${#signals[@]} == 0 )); then
    signals=(LHCO XtoWRto3W XtoYYprime ZpToTpTp YtoHHto4T)
fi

cd "$(dirname "$0")/.."

python -m scripts.prepare_kitchensink --signals LHCO "${signals[@]}" --feature-sets subjettiness

for signal in LHCO "${signals[@]}"; do
    if [[ ! -f "data/packed/$signal/test.pt" ]]; then
        python -m scripts.physics_ae pack --signal "$signal"
    fi
done

model_dir="$run_dir/seed$seed"
if [[ "$variant" != ratios ]]; then
    model_dir="$model_dir/physics"
fi
if [[ ! -f "$model_dir/ae.pt" ]]; then
    python -u -m scripts.physics_ae train --variant "$variant" --seed "$seed" --device "$device" --run-dir "$run_dir"
fi

for signal in "${signals[@]}"; do
    python -u -m scripts.physics_ae evaluate --variant "$variant" --signal "$signal" --seed "$seed" --device "$device" --run-dir "$run_dir"
done
