# 1 — latent diffusion over the autoencoder's 96-number latent

**cls-low AUC 0.991 — not competitive, and the reason is instructive.**

Trains the same `VelocityNet` in the autoencoder's 96-dimensional latent, then
decodes through the frozen decoder:

```
showers --encode--> z (96) --diffusion--> z (96) --decode--> showers
```

Two of the voxel path's hard problems disappear by construction: latents are
dense so the hit head is switched off, and the decoder already knows the
geometry. The formalism was validated at 1–3 dimensions and fails at 6480, so
96 is far closer to where it works.

## Why it did not work, and how we know

**It reached its own ceiling.** The autoencoder's *reconstruction* of real
showers — no diffusion model involved — scores cls-low **0.989**. The latent
sampler scores **0.991**. The sampler is not the problem; the decoder is.

Always measure that ceiling first:

```bash
python models/1_latent_diffusion/train_latent.py --ae <checkpoint> --ceiling \
    --export generated/latent_ceiling.hdf5 --n-sample 100000
```

**What carries the ceiling** is one feature: the autoencoder invents **8.3×
too many sub-MeV voxels** (941 against a real 114), and that single feature
alone has AUC 0.966 against the full classifier's 0.989. Total energy and
peak energy are indistinguishable from real. So the ceiling is not diffuse
reconstruction error — reconstruction is a respectable 0.43 — but one specific
defect: dim-energy smearing into empty voxels.

**Independent corroboration:** CaloLatent, the one published Dataset-2
submission that also does latent-space diffusion, placed 15th of 17 (0.983).
And CaloDREAM's own latent variant attributes its sparsity failure explicitly
to the autoencoder reconstruction rather than the generator.

## Status: open, not closed

The ceiling was measured on a **pre-v4** autoencoder with no hit head. The v4
model reports a lit fraction of 0.256 against a real 0.250, so it may already
have largely fixed the defect — and its ceiling has never been measured. A
focal hurdle loss (`--loss v5`, see `docs/NOVA_TRANSFER.md`) is the principled
training-time fix.

One caveat that may put a second ceiling underneath the first: the latent
model's worst observable is the azimuthal energy centroid (EC phi chi² ≈ 0.445),
a *positional* defect of the 96-number bottleneck that occupancy modelling will
not touch.

See `docs/EXPERIMENTS.md` experiments 0b and 3.

## The autoencoder itself

`train_autoencoder.py` in this directory, a thin entrypoint over
`calods2.training.ae_trainer` and `calods2.models.autoencoder`.

This was previously `CaloChallenge/train_auto_encoder_improved.py`, which ran
`torch.load('data/Dataset2.pt')` at **module scope** — importing the model
class pulled 2.5 GB off disk, which is why the latent trainer had to set
`sys.argv` and `CALO_BATCH_SIZE` before importing it. That is untangled: the
library now loads nothing on import, and the standardization constants are
non-persistent buffers on the model rather than module globals.

Results: `results/latent_diffusion/`, `results/autoencoder_prev4_ep85/`,
`results/autoencoder_sweep_d96_ep460/`
