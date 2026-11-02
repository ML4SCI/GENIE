#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

variant="${VARIANT:-energy}"       # original, energy, dense, control, dense_control
signal="${SIGNAL:-LHCO}"           # LHCO, XtoWRto3W, XtoYYprime, ZpToTpTp, YtoHHto4T
nsig="${NSIG:-1000}"               # 0, 50, 100, 500, 1000
sample="${SAMPLE:-reference}"      # reference = SB, SR
device="${DEVICE:-cuda:0}"
split="data/splits/CWoLa_${signal}_nsig${nsig}.npz"
run="${RUN_DIR:-runs/ati_${sample}_nsig${nsig}/seed42/${variant}/${signal}}"

python -m scripts.train --model sdm_ati --sdm-variant "$variant" \
    --split-file "$split" --sample "$sample" --output-dir "$run" --device "$device"
python -m scripts.inference --checkpoint "$run/last.pt" --split-file "$split" \
    --output "$run/test_scores.npz" --device "$device"
