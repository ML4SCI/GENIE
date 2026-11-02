#!/usr/bin/env bash
set -e

split_file="${SPLIT_FILE:-data/splits/CWoLa_LHCO_nsig1000.npz}"
sample="${SAMPLE:-reference}"          # reference = CWoLa SB; SR
model="${MODEL:-edgeconv}"             # edgeconv, gladc, netge, sdm_atii, sdm_energy, sdm_ati, sdm_nat
reg_weight="${REG_WEIGHT:-0}"          # 0 = baseline, 1 = EB3
region=sb
if [[ "$sample" == SR ]]; then region=sr; fi
run_dir="${RUN_DIR:-runs/edgeconv_${region}}"
device="${DEVICE:-cpu}"                # cpu, cuda:0, ...
resume="${RESUME:-}"                   # optional last.pt; keeps saved settings

cd "$(dirname "$0")/.."

if [[ -n "$resume" ]]; then
    options=(--resume "$resume")
    if [[ -n "${DEVICE:-}" ]]; then options+=(--device "$device"); fi
    exec python -m scripts.train "${options[@]}" "$@"
fi

exec python -m scripts.train \
    --split-file "$split_file" \
    --sample "$sample" \
    --model "$model" \
    --reg-weight "$reg_weight" \
    --output-dir "$run_dir" \
    --device "$device" "$@"
