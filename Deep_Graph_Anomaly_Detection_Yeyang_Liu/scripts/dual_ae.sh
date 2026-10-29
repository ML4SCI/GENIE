#!/usr/bin/env bash
set -euo pipefail

seed="${SEED:-42}"
device="${DEVICE:-cpu}"
signals=("${@:-LHCO}")  # LHCO XtoWRto3W XtoYYprime ZpToTpTp YtoHHto4T

cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2

python -m scripts.prepare_kitchensink --signals "${signals[@]}" --feature-sets subjettiness

for signal in "${signals[@]}"; do
    if [[ ! -f "data/packed/$signal/test.pt" ]]; then
        python -m scripts.physics_ae pack --signal "$signal"
    fi
    python -u -m scripts.dual_ae prepare --signal "$signal" --device "$device"
    python -u -m scripts.dual_ae train --signal "$signal" --seed "$seed" --device "$device"
    python -u -m scripts.dual_ae evaluate --signal "$signal" --seed "$seed" --device "$device"
done
