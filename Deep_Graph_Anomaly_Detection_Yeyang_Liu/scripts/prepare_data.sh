#!/usr/bin/env bash
set -e

mode="${MODE:-CWoLa}"      # CWoLa, IAD
n_signal="${NSIG:-1000}"   # 0, 50, 100, 500, 1000

datasets=("$@")           # LHCO, extra_qcd, BB1, XtoWRto3W, XtoYYprime, ZpToTpTp, YtoHHto4T; default: all
if (( ${#datasets[@]} == 0 )); then
    datasets=(LHCO extra_qcd BB1 XtoWRto3W XtoYYprime ZpToTpTp YtoHHto4T)
fi
mapfile -t datasets < <(printf '%s\n' "${datasets[@]}" | sort -u)

signals=()
sources=("${datasets[@]}")
for dataset in "${datasets[@]}"; do
    case "$dataset" in
        LHCO|XtoWRto3W|XtoYYprime|ZpToTpTp|YtoHHto4T) signals+=("$dataset") ;;
        BB1|extra_qcd) ;;
        *) printf 'Unknown dataset: %s\n' "$dataset" >&2; exit 2 ;;
    esac
done

# Signal splits share the LHCO and extra-QCD background sources.
if (( ${#signals[@]} > 0 )); then
    sources+=(LHCO extra_qcd)
fi
mapfile -t sources < <(printf '%s\n' "${sources[@]}" | sort -u)

cd "$(dirname "$0")/.."

bash scripts/download_data.sh data/raw "${sources[@]}"
for signal in "${signals[@]}"; do
    python -m scripts.make_splits --mode "$mode" --signal "$signal" --n-signal "$n_signal"
done
python -m scripts.preprocess --datasets "${sources[@]}"
if [[ " ${sources[*]} " == *" BB1 "* ]]; then
    python -m scripts.make_bb1_splits
fi
python -m scripts.build_graph --datasets "${sources[@]}"
