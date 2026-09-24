"""Conditional demo: learn p(shower observables | energy) on a calo-like toy.

Run from the ``Tina`` folder (or paste into Colab):

    python -m flow_matching.demo_conditional

The generative target of the calorimeter task is the *conditional* density
p(shower | E_inc), not a marginal. This demo trains a conditional velocity
field v_theta(x, t, c) on the 3-observable shower toy (sampling fraction,
depth, width vs. normalized log-energy c), then scores it three ways:

1. pooled over all energies (the number a marginal model could also fake),
2. per energy bin with ``evaluate_by_condition`` (a bad bin cannot hide),
3. at three *fixed* conditions c in {0.1, 0.5, 0.9} against fresh truth draws
   at exactly those c -- the interpolation test a lookup table would fail.

Also saves ``figures/fm_conditional.png`` (conditional means vs c + per-bin
histograms at the energy extremes).
"""

import os
import sys

# Make pinnde_eval and flow_matching importable when run as a script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from pinnde_eval import evaluate, evaluate_by_condition, report, swd
from pinnde_eval._utils import to_numpy
from pinnde_eval.data import sample_shower_toy

from .core import sample
from .train import train_flow_matching

FEATURES = ["sampling fraction", "depth", "width"]
BIN_EDGES = (0.25, 0.5, 0.75)
BIN_NAMES = ("c 0.00-0.25", "c 0.25-0.50", "c 0.50-0.75", "c 0.75-1.00")


def energy_bin_labels(c):
    return np.array(BIN_NAMES)[np.digitize(np.asarray(c).ravel(), BIN_EDGES)]


def main(n_data=40000, n_eval=8000, n_steps=6000, seed=0, plot_path=None):
    x, c = sample_shower_toy(n_data, seed=seed + 1)
    x_eval, c_eval = sample_shower_toy(n_eval, seed=seed + 3)

    print(f"training conditional flow matching: d=3, cond=1, "
          f"n_data={n_data}, steps={n_steps}")
    model, history = train_flow_matching(
        x, dim=3, cond=c, n_steps=n_steps, seed=seed,
        monitor_every=max(1, n_steps // 8),
        monitor_real=x_eval, monitor_cond=c_eval,
    )
    print("\nstep |      loss |        mmd |        swd")
    for step, loss, m, s in history:
        print(f"{step:5d} | {loss:9.4f} | {m:10.4e} | {s:10.4f}")

    # matched conditions: gen[i] is drawn from p(x | c_eval[i])
    gen = sample(model, n_eval, 3, cond=c_eval, steps=50, seed=seed)

    # 1. pooled over all energies
    print()
    report(evaluate(x_eval, gen, tier="full", seed=seed),
           title="conditional FM vs truth (all energies pooled)")

    # 2. per energy bin -- a bad bin cannot hide behind the pooled number
    labels = energy_bin_labels(to_numpy(c_eval))
    by_bin = evaluate_by_condition(x_eval, gen, labels, tier="full", seed=seed)
    print(f"\n{'energy bin':>14} | {'n':>5} | {'swd':>7} | {'auc':>13} | {'w1_mean':>8}")
    for name in BIN_NAMES:
        r = by_bin[name]
        print(f"{name:>14} | {r['n_real']:5d} | {r['swd']:7.4f} | "
              f"{r['auc'][0]:5.3f} +/- {r['auc'][1]:5.3f} | {r['w1_mean']:8.4f}")

    # 3. fixed-condition interpolation: generate at c*, compare to fresh truth
    #    at exactly c* (conditions never seen paired with these noise draws)
    print(f"\n{'c*':>5} | {'swd':>7} | truth mean -> generated mean (per feature)")
    for c_star in (0.1, 0.5, 0.9):
        truth, _ = sample_shower_toy(4000, cond=c_star, seed=seed + 7)
        g_star = sample(model, 4000, 3, cond=float(c_star), steps=50, seed=seed + 8)
        pair = ", ".join(
            f"{t:.3f}->{gm:.3f}"
            for t, gm in zip(to_numpy(truth).mean(0), to_numpy(g_star).mean(0)))
        print(f"{c_star:5.2f} | {swd(truth, g_star, seed=seed):7.4f} | {pair}")

    if plot_path:
        _plot(x_eval, c_eval, gen, plot_path)
    return model, history


def _plot(x_eval, c_eval, gen, path):
    """Conditional means vs c (top) and energy-extreme histograms (bottom)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xr, cr, xg = to_numpy(x_eval), to_numpy(c_eval).ravel(), to_numpy(gen)
    edges = np.linspace(0, 1, 11)
    mids = 0.5 * (edges[:-1] + edges[1:])
    lo_mask, hi_mask = cr < 0.25, cr > 0.75

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for j, name in enumerate(FEATURES):
        ax = axes[0, j]
        for arr, label, color in ((xr, "truth", "tab:blue"),
                                  (xg, "generated", "tab:orange")):
            mean = [arr[(cr >= a) & (cr < b), j].mean() for a, b in zip(edges, edges[1:])]
            std = [arr[(cr >= a) & (cr < b), j].std() for a, b in zip(edges, edges[1:])]
            mean, std = np.array(mean), np.array(std)
            ax.plot(mids, mean, marker="o", ms=4, color=color, label=label)
            ax.fill_between(mids, mean - std, mean + std, color=color, alpha=0.2)
        ax.set_xlabel("condition c (normalized log energy)")
        ax.set_title(f"{name}: conditional mean ± std")
        if j == 0:
            ax.legend()

        ax = axes[1, j]
        for mask, style, tag in ((lo_mask, "-", "low E"), (hi_mask, "--", "high E")):
            lo = min(xr[mask, j].min(), xg[mask, j].min())
            hi = max(xr[mask, j].max(), xg[mask, j].max())
            b = np.linspace(lo, hi, 41)
            ax.hist(xr[mask, j], bins=b, density=True, histtype="step",
                    linestyle=style, color="tab:blue", label=f"truth {tag}")
            ax.hist(xg[mask, j], bins=b, density=True, histtype="step",
                    linestyle=style, color="tab:orange", label=f"gen {tag}")
        ax.set_xlabel(name)
        if j == 0:
            ax.legend(fontsize=8)
    fig.suptitle("Conditional flow matching learns p(observables | energy)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
    main(plot_path=os.path.join(out, "fm_conditional.png")
         if os.path.isdir(out) else None)
