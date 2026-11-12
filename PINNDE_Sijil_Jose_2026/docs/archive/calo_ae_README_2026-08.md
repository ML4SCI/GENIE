> **Archived.** This was the README of the flat `CaloChallenge/` working
> directory, before the repository was restructured. **Every path in it is
> stale** — the scripts it names now live under `scripts/`, `sweeps/` and
> `models/1_latent_diffusion/`, and `train_auto_encoder_improved.py` is now
> `models/1_latent_diffusion/train_autoencoder.py`.
>
> It is kept because its "Known gotchas" section is the contemporaneous record
> of the 1000x units bug, the resume bug and the training plateau, written
> while they were being diagnosed. The current, corrected treatment of all
> three is in [../REPLICATION.md](../REPLICATION.md) §15 and
> [../MODEL_EVOLUTION.md](../MODEL_EVOLUTION.md).

---

# Tooling and remaining scripts

The autoencoder model, losses and training loop now live in `src/calods2/`;
the entrypoint is `models/1_latent_diffusion/train_autoencoder.py`. What
remains here is tooling.

`train_auto_encoder_improved.py` was removed in the untangling commit. It
loaded Dataset 2 and built DataLoaders at **import** time, so merely importing
the autoencoder class pulled 2.5 GB off disk — which is why `train_latent.py`
had to set `sys.argv` and `CALO_BATCH_SIZE` before importing it. Recover it
from history if needed:

```bash
git log --diff-filter=D -- CaloChallenge/train_auto_encoder_improved.py
```

## The duplicate models, and what removing them uncovered

`generate_caloeval_hdf5.py` and `diagnose_training_plateau.py` each carried
their own copy of `FactorizedCaloAE` (175 and 114 lines). Both now import it
from `calods2.models.autoencoder`, verified **bit-identical** (max absolute
difference 0.000e+00 on the same weights and inputs).

Deduplicating them turned up a real bug, in the main script rather than the
copies. The transverse encoder downsamples (16, 9) → (8, 5) → (4, 3), so a
mirrored decoder must go (4, 3) → **(8, 5)** → (16, 9). The training script had
been corrected to that — but **every checkpoint in this project was trained
with (8, 4)**, and both copies still used it.

Convolution weights do not encode spatial size, so an (8, 4)-trained
checkpoint loads into an (8, 5) model with **zero missing or unexpected keys**
and silently reconstructs ~10% worse. Measured across all six local
checkpoints, (8, 4) wins every time:

| checkpoint | at (8, 5) | at (8, 4) |
|---|---:|---:|
| `best_calo_ae_cond_improved.pt` | 0.4392 | **0.3980** |
| latent sweep, 8 → 128 | 0.6067 … 0.3976 | **0.5981 … 0.3487** |

So the copies matched the models and the main script had drifted away from
them. `mid_transverse` is now an explicit constructor argument, recorded in
every new checkpoint; `from_checkpoint()` defaults a file without that field
to `LEGACY_MID_TRANSVERSE = (8, 4)` and says so. New models get the correct
mirror by default.

Full write-up: `docs/MODEL_EVOLUTION.md` §9, item 5.

## Removed in the cleanup

- `check.py` -- nine lines printing shower sums and energy ranges for two
  HDF5 files. Subsumed by `compare_to_reference.py` and
  `calods2.metrics.quick_metrics`.
- `../reserve_files/run_latent_sweep_resume.sbatch` -- byte-identical to
  `Hyper_paramater_tuning/run_latent_sweep_resume.sbatch`; the directory held
  nothing else.

Both are in git history.

## Scripts that were rewired

`ae_baselines.py` and `ae_sweep.py` imported the old module for its global
`showers`, `energies`, `train_loader`, … Each now carries a small shim that
builds the same handful of attributes explicitly from `calods2.data`. Behaviour
is unchanged.

`ae_baselines.py` additionally gained a **units check**. A units mismatch and a
broken objective produce the same symptom in its table — a huge relative error
— and its verdict ("the objective is broken, not the capacity") would blame the
objective either way. It now warns when a checkpoint lacks the
`shower_units_gev` marker, because that checkpoint learned
`log1p(E_MeV / E_GeV)` and this script feeds it `log1p(E_GeV / E_GeV)`.

## Inventory

| file | role |
|---|---|
| `run_full_evaluation.py` | driver for the official `evaluate.py` |
| `generate_caloeval_hdf5.py` | autoencoder reconstructions to CaloChallenge HDF5 |
| `ae_baselines.py` | autoencoder vs zeros / mean shower / PCA-k, with a verdict |
| `ae_sweep.py` | latent-size and capacity sweeps |
| `compare_to_reference.py` | generated vs the evaluation's own reference; units check |
| `diagnose_training_plateau.py` | the v3 post-mortem that found the dead objective |
| `build_comparison_plots.py` | figures vs the published submissions → `docs/figures/` |
| `visualize_calo_ae.py`, `visualize_latent_structure.py`, `visualize_correlations.py` | diagnostics |
| `quick_residual_check.py` | **standalone inference + residual plots from a checkpoint.** This is the working, independent version of the entrypoint's end-of-run plotting block, which no test exercises -- prefer this one |
| `subsample_hdf5.py` | shrink an HDF5 to match event counts. **Not usable for dataset-2 classifier modes** -- ttv_split hard-codes 60k/20k/20k, so both sides need exactly 100k; see its docstring |
| `PlotDataset2.ipynb` | dataset exploration. Outputs are stripped before commit (955 KB -> 10 KB); re-run locally to view |
| `CaloChallenge/` | clone of the upstream evaluation repo (gitignored) |
| `data/` | `Dataset2.pt` and the challenge HDF5 files (gitignored) |

---

*The sections below are preserved from the previous revision of this file.
They predate the restructure, so paths and pipeline commands may refer to the
old layout -- the data sources, checkpoint inventory and gotchas are still
accurate and are the reason they are kept.*

## Data

- `data/Dataset2.pt` — training data as a torch tensor dict: `energies` and `showers`, both in **MeV**.
- `dataset_2_2.hdf5` — official Geant4 reference file for evaluation. From Zenodo,
  [DOI 10.5281/zenodo.6366270](https://doi.org/10.5281/zenodo.6366270)
  (direct file: `https://zenodo.org/api/records/6366271/files/dataset_2_2.hdf5/content`, 1,366,708,391 bytes).
  Dataset 2 ships as two files of 100k showers each — one for training, one held out as the
  evaluation reference; `dataset_2_2.hdf5` is the reference half.
- `dataset_2_2.pkl`, `dataset_2_2_10k.pkl` — cached `HighLevelFeatures` objects (derived
  per-layer statistics for plotting only). **These do not contain raw showers** and cannot
  substitute for the `.hdf5` reference file.

## Checkpoints

Only one checkpoint matches the current model code — the others are kept for history
and are **not** loadable by the `FactorizedCaloAE` class defined in the scripts above
(different submodule names from earlier architecture iterations).

| File | Compatible with current code? | Notes |
|---|---|---|
| `best_calo_ae_cond_improved.pt` | ✅ yes — **the live checkpoint** | Full dict: `epoch=85`, `val_loss=0.236`, energy-norm stats, geometry. Every script defaults to this file. |
| `best_calo_ae_cond.pt` | ❌ no | Bare state_dict, `enc2d.0.block.*` naming — earlier architecture. |
| `best_calo_ae_2.pt` | ❌ no | Same earlier architecture as above. |
| `best_calo_ae.pt` | ❌ no | Same earlier architecture as above. |
| `auto_encoder_model_l1.pth` | ❌ no | Earliest experiment — uses BatchNorm (`enc2d.net.*`, `running_mean`), predates the GroupNorm rewrite. |

## Outputs

- `evaluation_results_ds2/` — output of the vendored `evaluate.py`: classifier AUC (`classifier_*.txt`),
  FPD/KPD (`fpd_kpd_2.txt`), PRDC (`prdc_2.txt`), histogram chi² (`histogram_chi2_2.txt`), and per-layer/average-shower plots.
- `viz_out/`, `latent_out/` — outputs of the two `visualize_*.py` scripts.
- `logs/` — stdout/stderr from cluster SLURM runs.

## Nested `CaloChallenge/` directory

A vendored clone of the official
[CaloChallenge/homepage](https://github.com/CaloChallenge/homepage) repo (has its own `.git` —
treat it as third-party, don't restructure it). Provides:

- `code/evaluate.py`, `code/HighLevelFeatures.py`, `code/resnet.py`, `code/XMLHandler.py` — the official evaluation suite.
- `final_results/CaloChallenge_results_ds2.hdf` — the official numeric results table (pandas-readable,
  `pd.read_hdf(..., key='ds2')`) for all 18 published Dataset-2 submissions plus a GEANT4 self-comparison
  row: KPD, FPD, precision/recall/density/coverage, and classifier AUC (low/high/CNN) for each. This is
  the ground truth used to build the comparison plots — not something to hand-copy from the paper.

One local-only change vs. upstream: `code/evaluate.py` has `usetex=False` (no LaTeX install on this machine).

## Known gotchas

- **Regenerate `ae_recon_ds2_full.hdf5` before trusting any evaluation off it.** The version dated
  2026-08-11 09:59 was stale: it read `Output E_dep/E_inc ≈ 787` instead of the physically correct
  ≈0.79 (a ~1000x error) — confirmed by loading the checkpoint and running inference manually, which
  reproduces the correct ratio. The model itself is fine; only that specific export was bad, most likely
  produced by an older version of `generate_caloeval_hdf5.py` before its unit-conversion fix, and never
  regenerated afterward. Every classifier AUC / FPD / KPD / PRDC number computed against that file
  (anything in `evaluation_results_ds2/` from before 2026-08-17) is meaningless — it was scoring the
  stale artifact, not the model. The bad file is kept as `ae_recon_ds2_full.STALE_1000x_bug.hdf5` for
  reference; regenerate with `generate_caloeval_hdf5.py --split all` and check the printed
  `Output E_dep/E_inc` line is close to the reference's true ratio (compute it directly from
  `dataset_2_2.hdf5` if in doubt) before running `evaluate.py` again.
- **The `~0.033` sampling-fraction comment in `generate_caloeval_hdf5.py` is wrong for this dataset** —
  don't use it as the sanity threshold. The true Dataset-2 reference file has a mean `E_dep/E_inc` of
  ≈0.785, not ≈0.033; the script's own `WARNING: E_dep/E_inc > 0.5` will fire on every correct run.
  Compare against the reference file's own ratio directly instead of that hardcoded expectation.
- **Training loss looked suspiciously flat — FIXED 2026-08-17.** `logs/output_training_no_noise_improved.out`
  showed train/val loss barely moving across a full 100-epoch run (0.2353 → 0.2313), which lined up with
  the checkpoint-resume bug: the run was unconditionally reloading `best_calo_ae_cond_improved.pt` before
  training started ([train_auto_encoder_improved.py:959](train_auto_encoder_improved.py:959), old lines),
  with `best_val_loss` reset to `inf` on every run — so a fresh environment with no checkpoint yet would
  crash, and every subsequent run silently resumed from an already-settled state with a fresh optimizer
  and no LR decay (the cosine warmup scheduler was commented out), risking overwriting a better checkpoint
  with a worse one after just one epoch. Fixed: resume is now opt-in via `RESUME_TRAINING` at the top of
  `if __name__ == "__main__":`, and when enabled it properly restores optimizer state, scheduler state,
  `start_epoch`, and `best_val_loss` from the checkpoint instead of just the model weights. The cosine
  warmup scheduler is now actually wired in. Existing checkpoint (`best_calo_ae_cond_improved.pt`,
  relative error plateaued ~39%) was produced under the old buggy behavior — a genuine from-scratch run
  with these fixes hasn't been done yet and is worth doing to see if it converges further.
- **Units.** `Dataset2.pt` stores showers and energies in MeV. The training step converts energies to
  GeV but leaves showers in MeV, so the model is trained on `log1p(shower_MeV / energy_GeV)` — a
  deliberate mixed-unit convention, not a bug. Read the comment block at the top of
  `generate_caloeval_hdf5.py` in full before touching any unit conversion anywhere in this pipeline.
- **Training resume.** `train_auto_encoder_improved.py`'s `__main__` block unconditionally loads
  `best_calo_ae_cond_improved.pt` before training starts, and `best_val_loss` is not restored from the
  checkpoint on resume. A fresh environment with no checkpoint yet will crash with `FileNotFoundError`;
  a resume can silently overwrite a better saved model with a worse one after just one epoch. Worth
  fixing (guard the load, seed `best_val_loss` from `checkpoint["val_loss"]`) before the next resume run.
- **Long unattended cluster jobs.** On hosts with a `sec=krb5p` NFS home (e.g. `dagda.hep.fsu.edu`), a
  plain `kinit` ticket dies at its 24h `ticket_lifetime` regardless of whether a `tmux` session is still
  running, taking any file I/O down with it. Use a renewable ticket in the persistent keyring
  (`KRB5CCNAME=KEYRING:persistent:$(id -u); kinit -r 7d`) plus an hourly `kinit -R` loop started inside
  the same `tmux` session as the job.

## Reproducing the evaluation + comparison

```bash
# 1. Reference file (see Data section above for the Zenodo source)
curl -L -o dataset_2_2.hdf5 \
  "https://zenodo.org/api/records/6366271/files/dataset_2_2.hdf5/content"

# 2. Full evaluation (mode=all covers avg/avg-E/hist/cls-low/fpd/kpd/prdc;
#    cls-high and cls-resnet are separate classifier variants not included in "all")
python CaloChallenge/code/evaluate.py \
  --input_file ae_recon_ds2_full.hdf5 --reference_file dataset_2_2.hdf5 \
  --dataset 2 --mode all --output_dir evaluation_results_ds2

python CaloChallenge/code/evaluate.py \
  --input_file ae_recon_ds2_full.hdf5 --reference_file dataset_2_2.hdf5 \
  --dataset 2 --mode cls-high --output_dir evaluation_results_ds2

python CaloChallenge/code/evaluate.py \
  --input_file ae_recon_ds2_full.hdf5 --reference_file dataset_2_2.hdf5 \
  --dataset 2 --mode cls-resnet --output_dir evaluation_results_ds2

# 3. Comparison plots against the 18 official submissions -> ../comparison_plots/
#    (reads evaluation_results_ds2/ + CaloChallenge/final_results/CaloChallenge_results_ds2.hdf)
```

`evaluate.py` needs `numpy`, `h5py`, `torch`, `scikit-learn`, `scipy`, `matplotlib`, `pandas`,
and (for the `fpd`/`kpd`/`prdc` parts of `mode=all`) `jetnet` and `prdc`.
