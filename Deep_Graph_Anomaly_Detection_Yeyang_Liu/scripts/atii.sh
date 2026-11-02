#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

variant="${VARIANT:-energy}"       # original, energy, context
subjets="${SUBJETS:-30}"           # 30, 60 (energy/context)
signal="${SIGNAL:-LHCO}"           # LHCO, XtoWRto3W, XtoYYprime, ZpToTpTp, YtoHHto4T
nsig="${NSIG:-1000}"               # 0, 50, 100, 500, 1000
sample="${SAMPLE:-reference}"      # reference = SB, SR
region=sb
if [[ "$sample" == SR ]]; then region=sr; fi
device="${DEVICE:-cuda:0}"
split="data/splits/CWoLa_${signal}_nsig${nsig}.npz"
case "$subjets" in
    30) graphs="data/graphs" ;;
    60) graphs="data/graphs60" ;;
    *) echo 'SUBJETS: 30 or 60'; exit 1 ;;
esac
run="${RUN_DIR:-runs/atii_${region}_nsig${nsig}/seed42/${variant}${subjets}/${signal}}"

case "$variant" in
    original)
        [[ "$subjets" == 30 ]] || { echo 'Original ATII uses 30 subjets'; exit 1; }
        options=(--model sdm_atii)
        ;;
    energy) options=(--model sdm_energy) ;;
    context) options=(--model sdm_energy --sdm-critic-context) ;;
    *) echo 'VARIANT: original, energy, context'; exit 1 ;;
esac

python -m scripts.train "${options[@]}" --split-file "$split" --sample "$sample" \
    --graph-dir "$graphs" --output-dir "$run" --device "$device"
python -m scripts.inference --checkpoint "$run/last.pt" --split-file "$split" \
    --output "$run/test_scores.npz" --device "$device"
