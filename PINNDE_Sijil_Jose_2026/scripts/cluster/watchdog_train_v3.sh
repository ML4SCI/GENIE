#!/bin/bash
# Self-healing training loop for v3: units fix + hit-logit-scale fix + the
# new calibrated rel_loss term (direct, differentiable relative-error
# objective, rel_weight=0.05). Fresh start (not resumed from v2's
# checkpoint) since the loss landscape genuinely changed with rel_loss
# added -- v2 converged to a bad plateau under the old objective and isn't
# a meaningful warm start for this one.
#
# Same rationale as watchdog_train_v2.sh: this session's execution sandbox
# has been observed to silently kill long-running background jobs. --resume
# makes retrying safe (train_autoencoder's epoch loop is `for epoch in
# range(start_epoch, epochs)`, start_epoch read from the checkpoint), and is
# a no-op on the very first launch since no checkpoint exists yet.

# Repo root, derived from this script's location rather than hardcoded --
# the old absolute path pointed into CaloChallenge/, which the restructure
# emptied, so this script could not start at all.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1

CHECKPOINT="best_calo_ae_rel_fix_v3_TEST.pt"
TARGET_EPOCHS=40
LOG_DIR="logs"
PYTHON="${PYTHON:-python3}"          # override for a specific conda env

attempt=0
while true; do
  attempt=$((attempt + 1))
  ts=$(date +%Y%m%d_%H%M%S)
  logfile="$LOG_DIR/train_rel_fix_v3_watchdog_attempt${attempt}_${ts}.log"
  echo "[$(date)] watchdog attempt $attempt -- launching (resume if checkpoint exists) -> $logfile"

  CALO_BATCH_SIZE=128 "$PYTHON" models/1_latent_diffusion/train_autoencoder.py \
    --epochs "$TARGET_EPOCHS" \
    --checkpoint-path "$CHECKPOINT" \
    --resume \
    > "$logfile" 2>&1
  exit_code=$?

  echo "[$(date)] watchdog attempt $attempt exited with code $exit_code"

  reached=$("$PYTHON" -c "
import torch
try:
    ck = torch.load('$CHECKPOINT', map_location='cpu', weights_only=False)
    print(ck.get('epoch', 0))
except Exception:
    print(0)
" 2>/dev/null)

  echo "[$(date)] checkpoint reports epoch=$reached / target=$TARGET_EPOCHS"

  if [ "$reached" -ge "$TARGET_EPOCHS" ] 2>/dev/null; then
    echo "[$(date)] reached target epoch count -- watchdog done."
    break
  fi

  echo "[$(date)] not yet at target -- retrying in 10s ..."
  sleep 10
done
