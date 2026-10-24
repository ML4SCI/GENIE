#!/bin/bash
#SBATCH -A m4392_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 24:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -J v18_layers
#SBATCH --array=0-2
#SBATCH -o logs/v18-%A_%a.out
#SBATCH -e logs/v18-%A_%a.err

# ═══════════════════════════════════════════════════════════════════════
# Graph-layer comparison (latent 128, graph decoder with a count head).
# One array task per model, each on its own GPU.  Everything is identical
# across tasks; only the graph layer changes, in the encoder AND the decoder.
#
#   task  model         graph layer                   layers  reach   tag
#   0     ChebNet K=2   spectral, 1 hop per layer     2       2 hops  v19_chebK2_lat128_s0
#   1     GraphSAGE     averages direct neighbours    2       2 hops  v19_sage2_lat128_s0
#   2     DeepSets      none (each particle alone)    2       0 hops  v19_deepsets_lat128_s0
#
# Submit from this code/ folder:
#   mkdir -p logs
#   sbatch --array=0 slurm_v18.sh      # first: builds the JetNet graph caches
#   sbatch --array=1-2 slurm_v18.sh    # then the other two, in parallel
#
# Each run writes $V19_SAVE_ROOT/<tag>/: suite.json, log.txt, timing.json,
# loss_terms.png, recon_pairs.png, dist_comparison.png, observables.png.
# If a task hits the 24 h limit, resubmit that index: it resumes from the
# last 10-epoch checkpoint.
#
# --skip-ceiling: the original runs also scored a JetNet W1 "ceiling" against
# EPiC-GAN with a separate script (ceiling_eval.py) that is not in this
# repository; it does not change training or suite.json.
# ═══════════════════════════════════════════════════════════════════════

TAGS=(v19_chebK2_lat128_s0  v19_sage2_lat128_s0  v19_deepsets_lat128_s0)
LAYER_ARGS=("--conv cheb --K 2 --layers 2"
            "--conv sage --K 1 --layers 2"
            "--conv deepsets --K 1 --layers 2")

i=${SLURM_ARRAY_TASK_ID:-0}
TAG=${TAGS[$i]}
LAYERS=${LAYER_ARGS[$i]}
[ -n "$TAG" ] || { echo "no model for array index $i (use 0-2)"; exit 1; }

# ---- paths: edit for your system ---------------------------------------
cd "$SLURM_SUBMIT_DIR"
export V17_DIR=${V17_DIR:-$(cd ../../experiment_latent64_reach_v17/code && pwd)}  # ae_core.py, suite.py
export V19_SAVE_ROOT=${V19_SAVE_ROOT:-$SCRATCH/gsoc_rushil/v18/results}          # run folders
export ABLATION_SAVE_ROOT=${ABLATION_SAVE_ROOT:-$SCRATCH/gsoc_rushil/v17_final/results}  # graph caches
export JETNET_DATA_DIR=${JETNET_DATA_DIR:-$SCRATCH/jetnet_data}

# NERSC Perlmutter environment
module load pytorch/2.6.0-1
export PYTHONUSERBASE=$SCRATCH/nersc_pytorch_extras

[ -f ae_v19.py ] || { echo "Submit from the code/ folder that holds ae_v19.py."; exit 1; }
for f in "$V17_DIR/ae_core.py" "$V17_DIR/suite.py"; do
  [ -f "$f" ] || { echo "MISSING: $f"; exit 1; }
done
mkdir -p logs "$V19_SAVE_ROOT"

python -c "import torch, torch_geometric, geomloss, jetnet, energyflow; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())" \
  || { echo "IMPORT CHECK FAILED"; exit 1; }

echo "task $i: $TAG  ($LAYERS)"
echo "started : $(date)"
srun python -u ae_v19.py --tag "$TAG" --seed 0 \
     $LAYERS --knn 6 --latent-dim 128 \
     --epochs 200 --batch-size 512 --lr 1e-3 \
     --pt-scale 3 --blur 0.02 --lam-emd 1 --emd-blur 0.01 --lam-count 0.1 \
     --skip-ceiling --efp-jobs "${SLURM_CPUS_PER_TASK:-32}"
echo "finished: $(date)"
