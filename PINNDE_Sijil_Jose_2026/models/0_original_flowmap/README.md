# 0 — the original flow-map PINNDE (abandoned)

The formulation this project started from: the network learns the **trajectory**
`x(t; z)` as a function of its initial noise — the whole flow map — with a
physics-residual penalty.

**Kept because the measurements that killed it are the reason every later model
looks the way it does.** It is not a working model.

## Why it fails at 6480 dimensions

Three measurements, all from this code:

- **It cannot generalize.** 512 → 4096 anchor trajectories (8× the data) cut
  held-out MSE only 1.84×, i.e. **MSE ~ N^-0.29**. Reaching usable error would
  need ~10^6× more data.
- **It cannot train from a cold start.** The residual is evaluated at the
  model's own prediction, which early in training is off-manifold, where the
  q-kernel degenerates and `G ~ x/sigma_t` becomes radially repulsive. The
  residual **rose** during training, 6862 → 22002.
- **Samples came out worse than noise**: E_dep/E_inc ≈ 3.4 against a true 0.78.

It was validated at 1–3 dimensions, where it works. The failure is purely one
of scale.

## What replaced it

Parameterizing the **local field** `q(t,x,E) = E[x0 | x_t, E]` and recovering
the velocity analytically — see `2_direct_pinn/`. Same interpolant, same ODE,
same schedule; only what the network learns changed. Measured E_dep/E_inc went
from 3.4 to 0.92.

The physics therefore lives in the *parameterization*, not in a penalty term.

## Files

| file | role |
|---|---|
| `networks_calo.py` | the PINN residual network |
| `generate_anchor_trajectories.py` | the anchor pipeline the later models do not need |
| `train_calo_pinnde.py` | trainer |
| `sample_and_export.py` | sampler |

These import `flow_de` and `gendata` by their old relative paths and **are not
wired to `src/calods2/`**. They are preserved as-is for the record; fix the
imports if you want to re-run them.
