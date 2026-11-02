# 3 — two-stage factorized sampler (current best)

**cls-low AUC 0.578**, mean histogram chi² 0.0017. Best or tied-best in six of
seven observable classes. For context: CaloDREAM 0.531, CaloDiffusion 0.577.

Factorizes the joint exactly, via the chain rule, into two independent
generative problems:

```
p(shower | E_inc) = p(u | E_inc) · p(w | E_inc, u)
                     └ stage A ┘   └──  stage B  ──┘

u  (45)    total energy + 44 stick-breaking ratios along z
w  (6480)  each voxel's share of its OWN layer; every layer sums to 1
```

Both stages use the *same* `VelocityNet` and the *same* Heun sampler as
`2_direct_pinn/`. Only the variable differs.

## Why this beats the single-stage model

1. **Energy conservation becomes an algebraic identity.** The decode map
   telescopes, so `Σ L_i = T` for *any* u in the cube, and
   `∂(ΣE)/∂w = 0` — the shape network structurally cannot move the total.
   χ²(E_tot/E_inc): 0.127 → **0.0020**, a factor of 63.
2. **The faint end becomes representable.** Once layer totals live in u, w can
   go through a logit, and the empty-to-lit gap goes from 5.5e-4 standardized
   units to **2.28** — a factor of 4000. Voxel-distribution χ²: 0.024 → 0.0025.
3. Each network solves an easier problem: a 45-dim density, and a normalized
   field with no scale freedom.

## Run

```bash
pip install -e .            # or: export PYTHONPATH=src
```

The entrypoint is flag-driven; `configs/*.yaml` **record** the flags each
committed result was produced with, they are not yet consumed by the script.
Wiring a `--config` loader is a small, worthwhile next task.

```bash
python models/3_two_stage/train.py --data data/Dataset2.pt \
    --checkpoint-energy checkpoints/ts_energy.pt \
    --checkpoint-shape  checkpoints/ts_shape.pt \
    --steps-energy 20000 --steps-shape 100000 --cond conv --batch 64 \
    --export generated/two_stage_ds2_100k.hdf5 --n-sample 100000 --sample-steps 50
```

`--resume` continues an interrupted run (keep `--steps-*` identical, or the
cosine schedule lands on a different curve). `--skip-energy` reuses a trained
stage A — safe because stage B is conditioned on the *true* u and never sees
stage A's output, so the two trainings are fully decoupled.

**Generate exactly 100,000 showers.** `evaluate.py` hard-codes a 60k/20k/20k
split for dataset 2, so classifier modes fail on anything else.

## Known weakness

cls-high 0.695 and cls-resnet 0.804 lag cls-low 0.578 by 0.10–0.12, ~8× the
run-to-run scatter. Per-voxel marginals are close to right; voxel-to-voxel
**correlations** are not. `teacher_forced.py` measures whether the stage A → B
handoff contributes (see `docs/EXPERIMENTS.md` experiment 1).

Results: `results/two_stage_100k/`
