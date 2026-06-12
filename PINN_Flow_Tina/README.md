# PINN_Flow_Tina

Tina's track of the **PINNDE** effort (GSoC 2026, ML4SCI / GENIE):
physics-informed neural networks that solve generative ODEs for fast calorimeter
shower simulation. This folder is the **flow-matching** track — sibling to the
score-based track in `../Physics_Informed_Neural_Network_Diffusion_Equation_Sijil_Jose/`.

The folder is **self-contained**: it does not import from Sijil's folder. Anything
needed from there (e.g. GMM toy generators) is mirrored locally.

## Contents

- **`pinnde_eval/`** — shared quantitative evaluation module (the first agreed
  deliverable). One entry point `evaluate(real, gen, tier=...)` plus direct
  Tier-3 monitors `mmd`/`swd`. Three tiers:
  - Tier 1 — classifier two-sample AUC + per-feature reduced χ² (CaloChallenge-standard)
  - Tier 2 — FPD / KPD (jetnet) + per-feature Wasserstein-1 (headline numbers)
  - Tier 3 — RBF-MMD + sliced-Wasserstein (cheap pure-torch training-loop monitors)
  - See `pinnde_eval/README.md` for the API and `pinnde_eval/DEVLOG.md` for the
    full validation/calibration record.

## Quick start (Colab or local)

```python
import sys; sys.path.insert(0, "PINN_Flow_Tina")   # so `import pinnde_eval` works
import numpy as np
import pinnde_eval as pe

real = np.random.default_rng(1).normal(size=(5000, 3))
gen  = np.random.default_rng(2).normal(size=(5000, 3))

print(pe.mmd(real, gen), pe.swd(real, gen))          # cheap monitors
res = pe.evaluate(real, gen, tier="full"); pe.report(res)
```

Validate the metrics on toys, and run the tests:

```bash
cd PINN_Flow_Tina
python -m pinnde_eval.validate_toys     # null + sensitivity + speed checks
python -m pytest pinnde_eval/tests -q   # 14 edge-case tests
```

## Status

- Done: evaluation module + toy validation (this folder).
- Deferred (waiting on dataset/latent decisions): calorimeter wiring
  (CaloChallenge dataset, high-level shower features, incident-energy
  conditioning) and a wrapper over the official CaloChallenge `evaluate.py`.
  These slot in via `evaluate()`'s `features_fn` hook — no API change.
- Next (flow-matching track proper): architecture upgrades (Fourier features,
  GELU, adaptive collocation) and a flow-matching training objective replacing
  the O(NM) Monte-Carlo score estimate.
