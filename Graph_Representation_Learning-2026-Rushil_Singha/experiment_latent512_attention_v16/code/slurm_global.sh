#!/bin/bash
#SBATCH -A m4392_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 36:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -J v16_global
#SBATCH -o logs/global-%j.out
#SBATCH -e logs/global-%j.err

# ═══════════════════════════════════════════════════════════════════════
# ARM 2 — GLOBAL (Set2Set pooling; decoder identical to arm 1).
#
# SUBMIT THIS ONLY AFTER slurm_local.sh HAS BUILT THE GRAPH CACHE.
# Look for "cache saved" in the local job's log, or check that
#   $ABLATION_SAVE_ROOT/_graph_cache/train_np150_knn6_mjall.pkl
# exists and has stopped growing.  Then this job loads it in seconds.
#
# Starting both arms at once makes them build the same ~3 GB pickle
# simultaneously — corrupt file or 40 minutes of duplicated work.
# ═══════════════════════════════════════════════════════════════════════

module load pytorch/2.6.0-1
export PYTHONUSERBASE=$SCRATCH/nersc_pytorch_extras

# MUST be identical to slurm_local.sh.
export JETNET_DATA_DIR=$SCRATCH/jetnet_data
export ABLATION_SAVE_ROOT=$SCRATCH/gsoc_rushil/v16_ablation/results

cd $SLURM_SUBMIT_DIR
mkdir -p logs "$ABLATION_SAVE_ROOT"

echo "host        : $(hostname)"
echo "job         : $SLURM_JOB_ID"
echo "save root   : $ABLATION_SAVE_ROOT"
echo "started     : $(date)"

CACHE="$ABLATION_SAVE_ROOT/_graph_cache/train_np150_knn6_mjall.pkl"
if [ -f "$CACHE" ]; then
  echo "graph cache : found ($(du -h "$CACHE" | cut -f1)) — will load, not rebuild"
else
  echo "graph cache : NOT FOUND — this job will build it from scratch."
  echo "              If slurm_local.sh is running right now, CANCEL THIS JOB;"
  echo "              both would write the same file and corrupt it."
fi

python -u -c "
import torch, torch_geometric, geomloss, jetnet, sklearn, scipy, networkx
print('torch          ', torch.__version__, '| cuda', torch.cuda.is_available())
print('torch_geometric', torch_geometric.__version__)
print('geomloss/jetnet  ok')
" || { echo 'IMPORT CHECK FAILED — fix the environment before queueing'; exit 1; }

srun python -u v16_ablation/run_ablation_global.py

echo "finished    : $(date)"
