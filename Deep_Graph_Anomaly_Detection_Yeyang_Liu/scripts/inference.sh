#!/usr/bin/env bash
set -e

split_file="${SPLIT_FILE:-data/splits/CWoLa_LHCO_nsig1000.npz}"
run_dir="${RUN_DIR:-runs/edgeconv_sb}"  # trained model's run folder
device="${DEVICE:-cpu}"                # cpu, cuda:0, ...
calibration="${CALIBRATION:-}"         # optional calibration.pt for gated scoring

cd "$(dirname "$0")/.."

model=(--checkpoint "$run_dir/last.pt")
output="$run_dir/test_scores.npz"
if [[ -n "$calibration" ]]; then
    model=(--calibration "$calibration")
    output="$run_dir/gated_scores.npz"
fi

exec python -m scripts.inference "${model[@]}" \
    --split-file "$split_file" \
    --output "$output" \
    --device "$device" "$@"
