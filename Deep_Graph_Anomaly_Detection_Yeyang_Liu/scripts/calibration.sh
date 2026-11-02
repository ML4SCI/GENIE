#!/usr/bin/env bash
set -euo pipefail

n_signal="${NSIG:-1000}"             # 0, 50, 100, 500, 1000
sample="${SAMPLE:-reference}"        # reference = SB, SR
region=sb
if [[ "$sample" == SR ]]; then region=sr; fi
device="${DEVICE:-cuda:0}"           # cpu, cuda:0, cuda:1, ...
num_workers="${NUM_WORKERS:-8}"
runs_dir="${RUNS_DIR:-runs}"
signals=("$@")                       # default: all five signals
if (( ${#signals[@]} == 0 )); then
    signals=(LHCO XtoWRto3W XtoYYprime ZpToTpTp YtoHHto4T)
fi

cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

for signal in "${signals[@]}"; do
    split="data/splits/CWoLa_${signal}_nsig${n_signal}.npz"
    baseline="$runs_dir/baseline_cwola_${region}_nsig${n_signal}/seed42/$signal"
    output="$runs_dir/reg_calibration_cwola_${region}_nsig${n_signal}/seed42/$signal"
    baseline_checkpoint="${BASELINE_CHECKPOINT:-$baseline/last.pt}"
    regularized_checkpoint="${REGULARIZED_CHECKPOINT:-$output/reg/last.pt}"
    calibration_dir="${RUN_DIR:-$output/calibration}"

    python -u -m scripts.calibrate \
        --baseline-checkpoint "$baseline_checkpoint" \
        --reg-checkpoint "$regularized_checkpoint" \
        --split-file "$split" --sample "$sample" \
        --batch-size 256 --device "$device" --num-workers "$num_workers" \
        --output "$calibration_dir/calibration.pt"

    python -u -m scripts.inference \
        --calibration "$calibration_dir/calibration.pt" --split-file "$split" \
        --batch-size 256 --device "$device" --num-workers "$num_workers" \
        --output "$calibration_dir/test_scores.npz"
done
