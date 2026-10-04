#!/bin/bash
#SBATCH -A m4392_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 08:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -J v17_diag
#SBATCH -o logs/diag-%j.out
#SBATCH -e logs/diag-%j.err

# ═══════════════════════════════════════════════════════════════════════
# DIAGNOSTICS — why reach doesn't help.  No autoencoder is retrained.
#
#   sbatch v17_final/slurm_diagnostic.sh     (from the folder containing v17_final/)
#
#   1  rescore   every model re-scored with log D2 (suite.json rewritten,
#                the old one kept as suite_pre_d2fix.json)
#   2  factor    Test A1: ECF(2, beta=1) from per-particle sums only
#      learned   Test A2: reach 0 / 2 / 10 encoders trained to predict it
#   3  probe     Test B:  what the frozen latent holds vs what comes back
#   4  count     Experiment 1: StatsPool vs Set2Set on count-correct jets
#
# ~3-5 h in total.  Every step caches its output, so if the job hits the
# time limit just submit it again -- it carries on where it stopped.
# Afterwards it rebuilds results.md with the fixed D2.
# ═══════════════════════════════════════════════════════════════════════

module load pytorch/2.6.0-1
export PYTHONUSERBASE=$SCRATCH/nersc_pytorch_extras
export JETNET_DATA_DIR=$SCRATCH/jetnet_data
# graph caches (train + test, directed + undirected) already live here
export ABLATION_SAVE_ROOT=$SCRATCH/gsoc_rushil/v17_final/results

ROUND1=$SCRATCH/gsoc_rushil/v16_ablation/results
SCOUT=$SCRATCH/gsoc_rushil/v16_ablation/results_scout
REACH=$SCRATCH/gsoc_rushil/v17_final/results
OUT=$SCRATCH/gsoc_rushil/v17_final/diagnostics

cd "$SLURM_SUBMIT_DIR"
if [ ! -f v17_final/diagnostic.py ]; then
  echo "Submit from the directory that CONTAINS v17_final/."; exit 1
fi
mkdir -p logs "$OUT"
for d in "$ROUND1" "$SCOUT" "$REACH"; do
  [ -d "$d" ] || { echo "MISSING: $d -- edit the paths above"; exit 1; }
done

echo "started : $(date)"
srun python -u v17_final/diagnostic.py \
     --pooling "$ROUND1" --compression "$SCOUT" --reach "$REACH" \
     --out "$OUT"

echo "=== results.md with the fixed D2 ==="
srun python -u v17_final/aggregate.py --pooling "$ROUND1" \
     --compression "$SCOUT" --reach "$REACH" \
     --out $SCRATCH/gsoc_rushil/v17_final/paper
echo "finished: $(date)"
