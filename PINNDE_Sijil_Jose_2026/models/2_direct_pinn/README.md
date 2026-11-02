# 2 — direct physics-informed sampler over all 6480 voxels

**cls-low AUC 0.683**, mean chi² 0.0082. Mid-table against the published
submissions and ahead of every flow-based one on low-level observables.
Superseded by `3_two_stage/`, but the simpler model: one network, one pass.

## The formalism

```
x(t) = (1-t)·x0 + sigma(t)·eps      sigma(t) = sigma0 + (1-sigma0)·t,  sigma0 = 1e-2
dx/dt = G(t,x,E) = [(1-sigma0)·x - q(t,x,E)] / sigma(t)      t: 1 -> 0
q(t,x,E) = E[x0 | x_t, E]
```

G is **not learned** — the schedule and the algebra are exact, and the network
supplies only `q`, pinned by a plain regression whose minimizer is exactly the
conditional mean. That is where the "physics-informed" content lives: not in a
residual penalty (which was tried and measurably hurt, see
`0_original_flowmap/`), but in the parameterization.

## What each component was added to fix

| component | defect it fixed | measured |
|---|---|---|
| local `q` instead of a flow map | cannot generalize at 6480-D | MSE ~ N^-0.29 → generalizes |
| global standardization of x0 | noise 1450× larger than signal | ‖G‖ was 79.6 at every t |
| clean-x0 target, not Monte-Carlo | the MC q-field memorizes | NN-to-train 7–9 → 32 (real 47) |
| hit head | 77% of voxels lit vs a real 25% | occupancy gating |
| FiLM conditioning | per-energy response wrong | divergences 236 → 1 |
| 3D conv U-Net trunk | no spatial structure | profiles 3–5× better at 20× fewer params |

The conv trunk is the decisive one: a width sweep across 4.6× parameters left
the per-voxel defect **flat**, so the flat MLP's limit was its lack of
neighbourhood structure, not its size.

## Run

```bash
python models/2_direct_pinn/train.py --data data/Dataset2.pt --cond conv \
    --checkpoint checkpoints/velocity_conv.pt \
    --steps 16000 --batch 64 --hidden 1024 \
    --export generated/conv_recon_ds2.hdf5 --n-sample 100000 --sample-steps 100

# re-sample or re-gate an existing checkpoint, no retraining
python models/2_direct_pinn/sample.py --checkpoint checkpoints/velocity_conv.pt \
    --data data/Dataset2.pt --export generated/gen.hdf5 --n-sample 100000 --gate topk
```

**Do not simply raise `--steps`.** With the plain unpreconditioned net samples
got measurably *worse* past ~8k steps at batch 32, and blew up to inf by 16k at
constant LR. The cosine decay to zero is what makes long runs stable, and
`--steps` therefore sets the schedule, not just a stopping point.

Operates on the `fraction` representation, so it inherits that
representation's faint-end blindness — which is most of the gap to model 3.

Results: `results/conv_unet_runA/`, `results/conv_unet_runB/` (the same file
evaluated twice: the project's only error bar, ~0.013 AUC),
`results/film_mlp_100k/` (the superseded flat MLP).
