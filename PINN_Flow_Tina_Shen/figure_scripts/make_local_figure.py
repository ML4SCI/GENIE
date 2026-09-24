"""Demo figure for the local discrepancy maps (pinnde_eval.local).

    python make_local_figure.py       ->  figures/local_discrepancy.png

Builds a 2-D toy where the "generator" fails in two *localized* ways -- it
drops one GMM mode entirely and shifts another -- and shows that all three
local maps light up exactly at the failing modes while the global numbers
stay deceptively moderate. Left to right:

  1. real vs generated scatter (ground truth of what is wrong),
  2. MMD witness function on a grid (red = real over-dense / missed mode,
     blue = generated over-dense / hallucinated mode),
  3. per-bin normalized histogram residuals (same story, per bin, ~N(0,1)
     under the null so |z| > 3 is a genuine local failure),
  4. generated samples colored by the classifier's out-of-fold P(real | x)
     (blue = confidently fake -> the hallucinated mode).
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pinnde_eval import evaluate, classifier_discrepancy, mmd_witness
from pinnde_eval._utils import seed_all, to_numpy
from pinnde_eval.data import gmm_params, perturb_params, sample_gmm
from pinnde_eval.local import binned_residual_map

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUT, exist_ok=True)
SEED = 0
N = 6000


def broken_generator_params(params):
    """Drop mode 0 and shift mode 1: two localized failures, rest untouched."""
    p = perturb_params(params, "drop", 1.0)      # remove component 0 entirely
    p["mu"] = p["mu"].copy()
    p["mu"][1] = p["mu"][1] + 1.5                # move component 1
    return p


def main():
    seed_all(SEED)
    params = gmm_params(d=2, k=6, seed=SEED)
    real = to_numpy(sample_gmm(params, N, seed=SEED + 1))
    gen = to_numpy(sample_gmm(broken_generator_params(params), N, seed=SEED + 2))

    res = evaluate(real, gen, tier="full", seed=SEED)
    print(f"global numbers despite two broken modes: swd={res['swd']:.3f}  "
          f"mmd={res['mmd']:.2e}  auc={res['auc'][0]:.3f}")

    lo = np.floor(np.minimum(real.min(0), gen.min(0))) - 1
    hi = np.ceil(np.maximum(real.max(0), gen.max(0))) + 1
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))

    # 1. ground truth
    ax = axes[0]
    ax.scatter(real[:, 0], real[:, 1], s=3, alpha=0.3, label="real")
    ax.scatter(gen[:, 0], gen[:, 1], s=3, alpha=0.3, label="generated")
    ax.scatter(*params["mu"][0], marker="x", s=90, color="black", zorder=3)
    ax.annotate("dropped\nmode", params["mu"][0], textcoords="offset points",
                xytext=(8, 8), fontsize=9)
    ax.scatter(*params["mu"][1], marker="+", s=90, color="black", zorder=3)
    ax.annotate("shifted\nmode", params["mu"][1], textcoords="offset points",
                xytext=(8, 8), fontsize=9)
    ax.set_title("what is actually wrong")
    ax.legend(markerscale=4, fontsize=9)

    # 2. MMD witness on a grid
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 90),
                         np.linspace(lo[1], hi[1], 90))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    w = mmd_witness(real, gen, points=grid, seed=SEED).reshape(gx.shape)
    vmax = np.abs(w).max()
    ax = axes[1]
    im = ax.pcolormesh(gx, gy, w, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, label="witness(z)")
    ax.set_title("MMD witness\n(red = missed, blue = hallucinated)")

    # 3. per-bin normalized residuals
    edges = [np.linspace(lo[0], hi[0], 25), np.linspace(lo[1], hi[1], 25)]
    z, _ = binned_residual_map(real, gen, features=(0, 1), edges=edges)
    zmax = np.abs(z).max()
    ax = axes[2]
    im = ax.pcolormesh(edges[0], edges[1], z.T, cmap="RdBu_r",
                       vmin=-zmax, vmax=zmax)
    fig.colorbar(im, ax=ax, label="per-bin z")
    ax.set_title("histogram residuals per bin\n(|z| > 3 = genuine disagreement)")

    # 4. classifier P(real | x) on the generated sample
    _, p_gen, auc = classifier_discrepancy(real, gen, seed=SEED)
    ax = axes[3]
    sc = ax.scatter(gen[:, 0], gen[:, 1], s=4, c=p_gen, cmap="RdBu_r",
                    vmin=0.0, vmax=1.0)
    fig.colorbar(sc, ax=ax, label="P(real | x)")
    ax.set_title(f"generated colored by P(real | x)\n"
                 f"(blue = confidently fake; oof AUC {auc:.2f})")

    for ax in axes:
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal")

    fig.suptitle("Local discrepancy maps localize the failing modes", y=1.0)
    fig.tight_layout()
    path = os.path.join(OUT, "local_discrepancy.png")
    fig.savefig(path, dpi=130)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
