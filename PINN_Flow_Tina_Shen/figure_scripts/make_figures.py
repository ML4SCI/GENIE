"""Generate presentation figures into Tina/figures/.

    python make_figures.py

Produces:
  1. sensitivity.png   -- pinnde_eval metrics grow with perturbation size (the
                          module is calibrated).
  2. fm_convergence.png -- MMD/SWD vs training step for the flow-matching model.
  3. fm_scatter.png    -- real vs generated samples (2-D GMM).
  4. fm_histograms.png -- per-feature real vs generated histograms.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pinnde_eval._utils import to_numpy
from pinnde_eval.data import gmm_params, sample_gmm
from pinnde_eval.validate_toys import sensitivity_test
from flow_matching import sample, train_flow_matching

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUT, exist_ok=True)
SEED = 0


def fig_sensitivity():
    # Reuse the validation routine; it saves the metric-vs-eps grid.
    sensitivity_test(d=3, n=8000, seed=SEED,
                     plot_path=os.path.join(OUT, "sensitivity.png"))


def fig_flow_matching():
    d, k = 2, 6
    params = gmm_params(d=d, k=k, seed=SEED)
    data = sample_gmm(params, 20000, seed=SEED + 1)
    real = sample_gmm(params, 5000, seed=SEED + 3)

    model, history = train_flow_matching(
        data, dim=d, n_steps=4000, seed=SEED,
        monitor_every=200, monitor_real=real,
    )
    gen = sample(model, 5000, d, steps=50, seed=SEED)

    # --- convergence curve ---
    steps = [h[0] for h in history]
    mmd = [abs(h[2]) for h in history]
    swd = [h[3] for h in history]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, swd, marker="o", label="SWD")
    ax.plot(steps, mmd, marker="s", label="|MMD|")
    ax.axhline(0.049, ls="--", color="gray", label="SWD null floor")
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("distance to truth (log)")
    ax.set_title("Flow matching converges to the null floor")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fm_convergence.png"), dpi=130)
    plt.close(fig)

    # --- real vs generated scatter ---
    r, gg = to_numpy(real), to_numpy(gen)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(r[:, 0], r[:, 1], s=4, alpha=0.3, label="real")
    ax.scatter(gg[:, 0], gg[:, 1], s=4, alpha=0.3, label="generated")
    ax.set_title("Real vs generated (2-D GMM)")
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.legend(markerscale=3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fm_scatter.png"), dpi=130)
    plt.close(fig)

    # --- per-feature histograms ---
    from pinnde_eval import plot_histograms
    fig = plot_histograms(real, gen, bins=50, labels=("real", "generated"))
    fig.savefig(os.path.join(OUT, "fm_histograms.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    print("generating sensitivity figure ...")
    fig_sensitivity()
    print("generating flow-matching figures ...")
    fig_flow_matching()
    print(f"done -> {OUT}")
