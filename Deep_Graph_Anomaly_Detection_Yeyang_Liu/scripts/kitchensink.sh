#!/usr/bin/env bash
set -euo pipefail

read -r -a features <<< "${FEATURES:-baseline subjettiness efp combined}"
read -r -a n_signal <<< "${NSIG:-1000}"  # space-separated values: 0, 50, 100, 500, 1000
feature_dir="${FEATURE_DIR:-data/features/kitchensink}"
split_dir="${SPLIT_DIR:-data/splits/kitchensink}"
threads="${THREADS:-8}"
ensemble="${ENSEMBLE:-0}"              # 0 = single classifier, 1 = 50-model ensemble
runs="${RUNS:-1}"                      # number of ensemble runs
recipe=seed42
options=()
if [[ "$ensemble" == 1 ]]; then
    recipe=ensemble
    options=(--ensemble --runs "$runs")
fi
run_dir="${RUN_DIR:-runs/kitchensink_cwola/$recipe}"
signals=("$@")                       # default: all five signals
if (( ${#signals[@]} == 0 )); then
    signals=(LHCO XtoWRto3W XtoYYprime ZpToTpTp YtoHHto4T)
fi

cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python -m scripts.prepare_kitchensink \
    --feature-sets "${features[@]}" --n-signal "${n_signal[@]}" --signals "${signals[@]}" \
    --feature-dir "$feature_dir" --split-dir "$split_dir"

exec python -u -m scripts.kitchensink \
    --feature-sets "${features[@]}" --n-signal "${n_signal[@]}" --signals "${signals[@]}" \
    --feature-dir "$feature_dir" --split-dir "$split_dir" --output-dir "$run_dir" \
    --threads "$threads" "${options[@]}"
