"""Pedagogical figure: the learned velocity field ("wind map") and the
noise -> data trajectories of the toy flow-matching model.

    python make_flow_figure.py    ->  figures/fm_flow.png
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pinnde_eval._utils import to_numpy
from pinnde_eval.data import gmm_params, sample_gmm
from flow_matching import train_flow_matching

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
SEED = 0


@torch.no_grad()
def trajectories(model, n, dim, steps=50, seed=0):
    """Heun integration that records the full path: returns (steps+1, n, dim)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, dim, generator=g)
    ts = torch.linspace(0.0, 1.0, steps + 1)
    path = [x.clone()]
    for i in range(steps):
        dt = ts[i + 1] - ts[i]
        v0 = model(x, ts[i].expand(n, 1))
        v1 = model(x + dt * v0, ts[i + 1].expand(n, 1))
        x = x + dt * 0.5 * (v0 + v1)
        path.append(x.clone())
    return torch.stack(path).numpy()


def main():
    params = gmm_params(d=2, k=6, seed=SEED)
    data = sample_gmm(params, 20000, seed=SEED + 1)
    real = to_numpy(sample_gmm(params, 4000, seed=SEED + 3))

    model, _ = train_flow_matching(data, dim=2, n_steps=4000, seed=SEED)
    model.eval()

    paths = trajectories(model, 60, 2, steps=50, seed=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    # --- panel 1: the wind map (streamplot of the field at t=0.5) ---
    ax = axes[0]
    lo, hi = real.min(0) - 1.0, real.max(0) + 1.0
    xs = np.linspace(lo[0], hi[0], 32)
    ys = np.linspace(lo[1], hi[1], 32)
    XX, YY = np.meshgrid(xs, ys)
    grid = torch.tensor(np.stack([XX.ravel(), YY.ravel()], 1), dtype=torch.float32)
    with torch.no_grad():
        V = model(grid, torch.full((grid.shape[0], 1), 0.5)).numpy()
    U = V[:, 0].reshape(XX.shape)
    W = V[:, 1].reshape(XX.shape)
    speed = np.sqrt(U ** 2 + W ** 2)
    ax.scatter(real[:, 0], real[:, 1], s=3, color="0.8", alpha=0.30)
    ax.streamplot(XX, YY, U, W, color=speed, cmap="viridis", density=1.1, linewidth=0.9)
    ax.set_title("The learned wind map  (velocity field at t = 0.5)")
    ax.set_xlabel("x0"); ax.set_ylabel("x1")

    # --- panel 2: noise -> data trajectories ---
    ax = axes[1]
    ax.scatter(real[:, 0], real[:, 1], s=3, color="0.8", alpha=0.25, label="data (target)")
    cmap = plt.cm.plasma
    n_traj = paths.shape[1]
    for j in range(n_traj):
        ax.plot(paths[:, j, 0], paths[:, j, 1], color=cmap(j / n_traj), alpha=0.5, lw=0.8)
    ax.scatter(paths[0, :, 0], paths[0, :, 1], s=20, color="navy", zorder=3,
               label="start: noise (t=0)")
    ax.scatter(paths[-1, :, 0], paths[-1, :, 1], s=20, color="crimson", zorder=3,
               label="end: data (t=1)")
    ax.set_title("Noise → data: points ride the flow")
    ax.set_xlabel("x0"); ax.set_ylabel("x1")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    path = os.path.join(OUT, "fm_flow.png")
    fig.savefig(path, dpi=140)
    print("saved", path)


if __name__ == "__main__":
    main()
