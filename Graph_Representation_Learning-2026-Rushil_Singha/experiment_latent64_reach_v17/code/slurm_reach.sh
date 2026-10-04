#!/bin/bash
#SBATCH -A m4392_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 24:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -J v17_reach
#SBATCH -o logs/reach-%A_%a.out
#SBATCH -e logs/reach-%A_%a.err
#SBATCH --array=0-8

# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — REACH.  9 runs: 3 encoders x 3 seeds, latent 64.
#
# SUBMIT FROM THE FOLDER THAT CONTAINS v17_final/, in two steps:
#
#   sbatch --array=0 v17_final/slurm_reach.sh      # builds the graph cache
#   # wait for "cache saved" in logs/reach-*_0.out (~30-40 min), then:
#   sbatch --array=1-8 v17_final/slurm_reach.sh
#
# All nine read the same ~3 GB graph cache.  Launched together they would
# each try to write it at once and corrupt it.
#
#   index   encoder       reach     seed
#   0 1 2   GraphSAGE      2 hops   0 1 2     the local baseline
#   3 4 5   ChebNet K=2    2 hops   0 1 2     operator control: same reach
#   6 7 8   ChebNet K=6   10 hops   0 1 2     the reach arm
#
# Everything else is identical: StatsPool, no decoder attention, undirected
# kNN(6) graph, 2 layers, 1/sqrt(K) normalisation, 200 epochs on the FULL
# JetNet-150 train split -- the same budget as the latent-512 pooling runs.
#
# ~12-16 h per run.  Each checkpoints every 10 epochs; if one times out,
# resubmit just that index:   sbatch --array=5 v17_final/slurm_reach.sh
# ═══════════════════════════════════════════════════════════════════════

module load pytorch/2.6.0-1
export PYTHONUSERBASE=$SCRATCH/nersc_pytorch_extras
export JETNET_DATA_DIR=$SCRATCH/jetnet_data
export ABLATION_SAVE_ROOT=$SCRATCH/gsoc_rushil/v17_final/results

cd "$SLURM_SUBMIT_DIR"
if [ ! -f v17_final/ae_core.py ]; then
  echo "Submit from the directory that CONTAINS v17_final/ (not from inside it)."
  exit 1
fi
mkdir -p logs "$ABLATION_SAVE_ROOT"

I=$SLURM_ARRAY_TASK_ID
SEED=$(( I % 3 ))
case $(( I / 3 )) in
  0) NAME=sage;   ARGS="--conv sage --K 1" ;;
  1) NAME=chebK2; ARGS="--conv cheb --K 2" ;;
  2) NAME=chebK6; ARGS="--conv cheb --K 6" ;;
esac
TAG="reach_lat64_${NAME}_s${SEED}"

echo "host      : $(hostname)"
echo "array idx : $I   ->   $TAG"
echo "started   : $(date)"

python -u -c "
import torch, torch_geometric, geomloss, jetnet, sklearn, scipy, networkx
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())
" || { echo 'IMPORT CHECK FAILED'; exit 1; }

srun python -u v17_final/ae_core.py --tag "$TAG" --seed "$SEED" $ARGS \
     --latent-dim 64 --pool stats --use-attn false \
     --knn 6 --graph-symmetric true --layers 2 --term-norm sqrt \
     --epochs 200 --batch-size 512 --total-particles 150

echo "finished  : $(date)"
