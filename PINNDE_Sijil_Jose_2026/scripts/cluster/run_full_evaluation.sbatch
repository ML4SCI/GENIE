#!/bin/bash
#SBATCH -p gpu2                          # GPU partition -- NOT regular1: an earlier
                                          # eval job on regular1 got zero GPUs allocated
                                          # (see logs/error.err), which is likely why
                                          # classifier training never finished on cluster.
#SBATCH -t 15:00:00                      # generous; classifier training x3 modes + PRDC
                                          # can run long. Adjust to your queue's max.
#SBATCH --cpus-per-task=8
#SBATCH --mem=56000
#SBATCH --gpus-per-task=1
#SBATCH -J calo_full_eval
#SBATCH -o logs/full_eval_%j.out
#SBATCH -e logs/full_eval_%j.err

# ==============================================================================
# run_full_evaluation.sbatch
# ==============================================================================
# Full CaloChallenge Dataset-2 evaluation + comparison pipeline. Meant to run
# unattended on the cluster -- this class of job (multi-hour, sustained
# GPU+CPU) is exactly what does NOT belong on the laptop: that machine's
# thermald reports "Unsupported cpu model or platform" (no active thermal
# management) and its thermal_zone0 sensor can't even be read by the kernel
# (errno -61). It hard-crashed mid-run more than once running jobs like this.
#
# Does, in order:
#   1. Downloads the official reference file (dataset_2_2.hdf5) from Zenodo,
#      if not already present, and checks its byte size against the known-good
#      value so a truncated download can't silently propagate downstream.
#   2. Regenerates ae_recon_ds2_full.hdf5 from generate_caloeval_hdf5.py --
#      ALWAYS regenerates, never reuses a stale file (see "why regenerate"
#      note below), and gates on a physical sanity check before proceeding.
#   3. Runs the official evaluate.py for every mode needed:
#        mode=all       (avg, avg-E, hist, cls-low classifier, fpd, kpd, prdc)
#        mode=cls-high
#        mode=cls-resnet
#   4. Builds the comparison plots against the 18 official submissions.
#
# Adjust REPO_DIR / CONDA_ENV below to match your cluster layout before
# submitting -- these were inferred from local paths and past job logs, not
# a live look at the cluster filesystem.
#
# Submit with:  sbatch run_full_evaluation.sbatch
# ==============================================================================

set -euo pipefail
mkdir -p logs

echo '------------------------------------------------------'
echo 'This job is allocated on '$SLURM_JOB_CPUS_PER_NODE' cpu(s)'
echo 'Job is running on node(s): '$SLURM_JOB_NODELIST
echo 'SLURM: executing on cluster     '$SLURM_CLUSTER_NAME
echo 'SLURM: executing on partition   '$SLURM_JOB_PARTITION
echo 'SLURM: number of gpus/node is   '$SLURM_GPUS_PER_NODE
echo '------------------------------------------------------'
nvidia-smi || echo "WARNING: nvidia-smi failed -- no GPU visible to this job."

# ---- Environment -------------------------------------------------------
source ~/.bashrc
CONDA_ENV="ML"                            # matches launch_job.sh / run_calo_eval.sbatch;
                                           # change if your env is named differently
conda activate "$CONDA_ENV"

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available in this job -- check the partition/gpu request above.'"

# fpd/kpd/prdc need these; harmless if already installed.
python -c "import jetnet" 2>/dev/null || pip install -q jetnet
python -c "import prdc"   2>/dev/null || pip install -q prdc

# ---- Paths -- ADJUST to your cluster layout -----------------------------
REPO_DIR="$(pwd)"                                    # run this script from the dir
                                                      # containing generate_caloeval_hdf5.py
CODE_DIR="$REPO_DIR/CaloChallenge/code"              # wherever evaluate.py lives
REF_FILE="$REPO_DIR/dataset_2_2.hdf5"
GEN_FILE="$REPO_DIR/ae_recon_ds2_full.hdf5"
OUT_DIR="$REPO_DIR/evaluation_results_ds2"
WEIGHTS="$REPO_DIR/best_calo_ae_cond_improved.pt"
DATA_PT="$REPO_DIR/data/Dataset2.pt"
COMPARISON_OUTDIR="$REPO_DIR/../comparison_plots"

# ---- 1. Reference file ---------------------------------------------------
if [ ! -f "$REF_FILE" ]; then
    echo ">> Downloading dataset_2_2.hdf5 from Zenodo (DOI 10.5281/zenodo.6366270) ..."
    curl -L -o "$REF_FILE" \
        "https://zenodo.org/api/records/6366271/files/dataset_2_2.hdf5/content"
fi
REF_SIZE=$(stat -c%s "$REF_FILE")
if [ "$REF_SIZE" -ne 1366708391 ]; then
    echo "ERROR: $REF_FILE is $REF_SIZE bytes, expected 1366708391 -- looks truncated/corrupt. Aborting."
    exit 1
fi
echo "Reference file OK ($REF_SIZE bytes)."

# ---- 2. Regenerate the reconstruction file --------------------------------
# ALWAYS regenerate -- never reuse an existing ae_recon_ds2_full.hdf5. That's
# the actual bug we hit: an old export sat on disk from a previous version of
# this script and got fed into evaluate.py for days, producing FPD/KPD/AUC
# numbers ~1000x worse than reality. generate_caloeval_hdf5.py itself needed
# no code change -- it already does the right MeV/GeV handling -- the fix is
# procedural: regenerate before every evaluation, and gate on the sanity
# check below so a bad file can never again be evaluated silently.
echo ">> Regenerating $GEN_FILE ..."
python "$REPO_DIR/generate_caloeval_hdf5.py" \
    --weights "$WEIGHTS" \
    --data    "$DATA_PT" \
    --out     "$GEN_FILE" \
    --split   all

python - "$GEN_FILE" "$REF_FILE" <<'PYEOF'
import sys, h5py
gen_f, ref_f = sys.argv[1], sys.argv[2]
with h5py.File(gen_f) as f:
    gen = f['showers'][:5000].sum(axis=1) / f['incident_energies'][:5000].squeeze()
with h5py.File(ref_f) as f:
    ref = f['showers'][:5000].sum(axis=1) / f['incident_energies'][:5000].squeeze()
g, r = float(gen.mean()), float(ref.mean())
print(f"Sanity check: generated E_dep/E_inc={g:.4f}, reference={r:.4f}, ratio={g/r:.3f}")
assert 0.5 < (g / r) < 2.0, (
    f"Generated/reference E_dep/E_inc ratio is {g/r:.3f}, expected close to 1.0 "
    f"-- this looks like the same 1000x units bug. Aborting rather than running "
    f"a multi-hour evaluation against bad data."
)
print("Sanity check passed.")
PYEOF

# ---- 3. Full evaluation ----------------------------------------------------
mkdir -p "$OUT_DIR"
cd "$CODE_DIR"

for MODE in all cls-high cls-resnet; do
    echo ">> evaluate.py --mode $MODE ..."
    python evaluate.py \
        --input_file "$GEN_FILE" \
        --reference_file "$REF_FILE" \
        --dataset 2 \
        --mode "$MODE" \
        --save_mem \
        --output_dir "$OUT_DIR"
done

# ---- 4. Comparison plots ----------------------------------------------------
cd "$REPO_DIR"
mkdir -p "$COMPARISON_OUTDIR"
python build_comparison_plots.py \
    --eval-dir "$OUT_DIR" \
    --official CaloChallenge/final_results/CaloChallenge_results_ds2.hdf \
    --outdir "$COMPARISON_OUTDIR" \
    --our-name "Our AE (FactorizedCaloAE)"

echo "Done."
echo "  Evaluation results : $OUT_DIR/"
echo "  Comparison plots   : $COMPARISON_OUTDIR/"
