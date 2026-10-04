#!/bin/bash
#SBATCH -A m4392_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 36:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -J v16_local
#SBATCH -o logs/local-%j.out
#SBATCH -e logs/local-%j.err

# ═══════════════════════════════════════════════════════════════════════
# ARM 1 — LOCAL (StatsPool, no attention).  RUN THIS ONE FIRST.
#
# It builds the shared kNN graph cache (~3 GB, ~30-40 min).  The global
# arm reuses that cache, so starting both at once would make them build
# the same file simultaneously and clobber each other.
#
#   sbatch slurm_local.sh
#   # wait for "cache saved" in the log, THEN:
#   sbatch slurm_global.sh
#
# If this job hits the 36 h wall, just resubmit it.  Every run
# checkpoints every 10 epochs and resumes; runs that already finished
# skip straight to evaluation.
# ═══════════════════════════════════════════════════════════════════════

module load pytorch/2.6.0-1
export PYTHONUSERBASE=$SCRATCH/nersc_pytorch_extras

# Both arms MUST agree on these two, or the six runs land in different
# places and aggregate_results.py will only ever see three of them.
export JETNET_DATA_DIR=$SCRATCH/jetnet_data
export ABLATION_SAVE_ROOT=$SCRATCH/gsoc_rushil/v16_ablation/results

cd $SLURM_SUBMIT_DIR
mkdir -p logs "$ABLATION_SAVE_ROOT"

echo "host        : $(hostname)"
echo "job         : $SLURM_JOB_ID"
echo "data dir    : $JETNET_DATA_DIR"
echo "save root   : $ABLATION_SAVE_ROOT"
echo "started     : $(date)"

# Fail fast on a missing package instead of 40 minutes into graph building.
python -u -c "
import torch, torch_geometric, geomloss, jetnet, sklearn, scipy, networkx
print('torch          ', torch.__version__, '| cuda', torch.cuda.is_available())
print('torch_geometric', torch_geometric.__version__)
print('geomloss/jetnet  ok')
" || { echo 'IMPORT CHECK FAILED — fix the environment before queueing'; exit 1; }

# -u keeps the log streaming so you can watch progress on a 36 h job.
srun python -u v16_ablation/run_ablation_local.py

echo "finished    : $(date)"
