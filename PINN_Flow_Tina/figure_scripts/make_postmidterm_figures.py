"""Reproduce the post-midterm result figures (DEVLOG sections 11-16).

Run from the Tina folder with both ds2 files present:

    OPENBLAS_NUM_THREADS=1 python make_postmidterm_figures.py

Five figures, each one measurement from the write-up:

  fig_voxel_order.png    the flatten-order discovery (section 11)
  fig_scale_bug.png      why SWD/W1 need standardization (section 11)
  fig_sep_floor.png      separation power floor law S ~ n_bins/(2N) (section 12)
  fig_energy_bins.png    the low-energy failure and its fix (sections 13-14)
  fig_zero_inflation.png the per-layer representational wall (sections 15-16)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pinnde_eval as pe
from pinnde_eval.observables import observables_from_file
from pinnde_eval.tier2 import wasserstein_per_feature

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
DS1 = "dataset_2_1.hdf5"
DS2 = "dataset_2_2.hdf5"
GEOM = pe.GEOMETRIES["ds2"]
N = 20000        # matches validate_calo, so the floor table repeats line up

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12,
                     "axes.labelsize": 11, "figure.dpi": 130})

GOOD, BAD, NEUTRAL = "#1F8A5B", "#C2413A", "#41566B"


def fig_voxel_order():
    """6480 = 45 x 16 x 9. Only one reshape is physical."""
    showers, _ = pe.load_calochallenge(DS1, n=2000)
    mean = showers.mean(axis=0)

    wrong = mean.reshape(45, 9, 16)     # (layer, radial?, angular?) -- wrong
    right = mean.reshape(45, 16, 9)     # (layer, angular,  radial)  -- correct

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    a = wrong.sum(axis=(0, 2)); a = a / a.max()
    b = right.sum(axis=(0, 1)); b = b / b.max()
    ax.semilogy(np.arange(9), a, "o-", color=BAD, lw=2,
                label="reshape(45, 9, 16): sawtooth, unphysical")
    ax.semilogy(np.arange(9), b, "s-", color=GOOD, lw=2,
                label="reshape(45, 16, 9): ~50x monotonic falloff = radius")
    ax.set_xlabel("index along the 9-bin axis")
    ax.set_ylabel("mean energy (normalized, log)")
    ax.set_title("Which 9-bin axis is the radius?")
    ax.legend(fontsize=8.5, loc="lower left")

    # Positive evidence for the chosen order: the (angle, radius) map must be
    # uniform around the axis and steeply graded outward from it.
    ax = axes[1]
    grid = right.sum(axis=0)                      # (16 angular, 9 radial)
    im = ax.imshow(np.log10(grid / grid.max() + 1e-6), aspect="auto",
                   origin="lower", cmap="viridis")
    ax.set_xlabel("radial bin  (9)")
    ax.set_ylabel("angular bin  (16)")
    ax.set_title("reshape(45, 16, 9): uniform in angle, graded in radius")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("log$_{10}$ mean energy (normalized)", fontsize=9)

    fig.suptitle("Voxel flatten order determined from physics, not from the "
                 "dataset description", y=1.0)
    fig.tight_layout()
    out = os.path.join(FIG, "fig_voxel_order.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def fig_scale_bug(a, b, names):
    """W1 per feature, raw vs standardized: E_tot swamps everything."""
    raw = wasserstein_per_feature(a[:8000], b[:8000])
    mean, std = a[:8000].mean(0), a[:8000].std(0)
    std = np.where(std > 0, std, 1.0)
    stz = wasserstein_per_feature((a[:8000] - mean) / std,
                                  (b[:8000] - mean) / std)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(names))
    for ax, vals, tag, col in ((axes[0], raw, "raw physical units", BAD),
                               (axes[1], stz, "standardized", GOOD)):
        ax.bar(x, vals, color=col, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=9)
        ax.set_ylabel("Wasserstein-1 per observable")
        ax.set_title(f"{tag}   (mean {vals.mean():.4g})")
        ax.set_yscale("log")
    axes[0].annotate(f"E_tot = {raw[0]:.0f}\n(MeV scale)",
                     xy=(0, raw[0]), xytext=(1.4, raw[0] * 0.35), fontsize=9,
                     arrowprops=dict(arrowstyle="->", color=NEUTRAL))
    fig.suptitle("Geant4 vs Geant4 null: SWD/W1 are scale-dependent, so raw "
                 "units measure E_tot and nothing else", y=1.02)
    fig.tight_layout()
    out = os.path.join(FIG, "fig_scale_bug.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}  (raw w1_mean {raw.mean():.1f} -> "
          f"standardized {stz.mean():.4f})")
    plt.close(fig)


def fig_sep_floor(a, b):
    """Separation power null floor: ~1/N, and linear in the binning.

    Averaged over disjoint repeats, matching ``validate_calo`` so the figure
    and the DEVLOG section 12 table carry the same numbers.
    """
    n_grid = (500, 1000, 2000, 4000, 8000, 16000)
    bins_fixed = 50
    floor_n = []
    for n in n_grid:
        vals = [pe.separation_power(a[r * n:(r + 1) * n],
                                    b[r * n:(r + 1) * n],
                                    bins=bins_fixed).mean()
                for r in range(4) if (r + 1) * n <= len(a)]
        floor_n.append(float(np.mean(vals)))

    bin_grid = (25, 50, 100, 200)
    n_fixed = 4000
    floor_b = [pe.separation_power(a[:n_fixed], b[:n_fixed], bins=nb).mean()
               for nb in bin_grid]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.loglog(n_grid, floor_n, "o-", color=NEUTRAL, lw=2, ms=7,
              label="measured (Geant4 vs Geant4)")
    ax.loglog(n_grid, [bins_fixed / (2 * n) for n in n_grid], "--",
              color=BAD, lw=2, label=r"$n_{bins}\,/\,2N$")
    ax.set_xlabel("samples N")
    ax.set_ylabel("separation power")
    ax.set_title(f"Floor falls as 1/N  ({bins_fixed} bins)")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(bin_grid, floor_b, "o-", color=NEUTRAL, lw=2, ms=7,
            label="measured")
    ax.plot(bin_grid, [nb / (2 * n_fixed) for nb in bin_grid], "--",
            color=BAD, lw=2, label=r"$n_{bins}\,/\,2N$")
    ax.set_xlabel("number of histogram bins")
    ax.set_ylabel("separation power")
    ax.set_title(f"Floor is linear in binning  (N = {n_fixed})")
    ax.legend(fontsize=9)

    fig.suptitle("A perfect generator does not score zero: the separation "
                 "power floor", y=1.02)
    fig.tight_layout()
    out = os.path.join(FIG, "fig_sep_floor.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")
    print("  N grid :", [f"{v:.5f}" for v in floor_n])
    print("  bins   :", [f"{v:.5f}" for v in floor_b])
    plt.close(fig)


def fig_energy_bins():
    """Pooled AUC hid a failed quartile; capacity fixed it. Measured values."""
    bins = ["E 0-25%", "E 25-50%", "E 50-75%", "E 75-100%"]
    before = [0.787, 0.573, 0.498, 0.527]
    before_e = [0.010, 0.016, 0.015, 0.018]
    after = [0.509, 0.510, 0.487, 0.496]
    after_e = [0.020, 0.011, 0.020, 0.009]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2),
                            gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    x = np.arange(4)
    w = 0.36
    ax.bar(x - w / 2, before, w, yerr=before_e, capsize=4, color=BAD,
           alpha=0.9, label="h128 d3 12k steps")
    ax.bar(x + w / 2, after, w, yerr=after_e, capsize=4, color=GOOD,
           alpha=0.9, label="h384 d5 30k steps, 200 ODE steps")
    ax.axhline(0.5, ls="--", color=NEUTRAL, lw=1.5,
               label="0.5 = indistinguishable from Geant4")
    ax.axhline(0.5048, ls=":", color="black", lw=1.5,
               label="pooled AUC of the failing model = 0.5048")
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_ylabel("classifier two-sample AUC")
    ax.set_ylim(0.45, 0.85)
    ax.set_title("Per energy quartile: the pooled score hid a failed slice")
    ax.legend(fontsize=8.5, loc="upper right")

    ax = axes[1]
    labels = ["+16x ODE\nsteps", "dequantize\nsparsity", "capacity\n(h384 d5 30k)"]
    deltas = [0.7881 - 0.7832, 0.787 - 0.785, 0.7881 - 0.5068]
    cols = [NEUTRAL, NEUTRAL, GOOD]
    ax.barh(np.arange(3), deltas, color=cols, alpha=0.9)
    ax.set_yticks(np.arange(3))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("reduction in low-energy AUC")
    ax.set_title("What actually moved the number")
    for i, d in enumerate(deltas):
        ax.text(d + 0.006, i, f"{d:+.4f}", va="center", fontsize=9)
    ax.set_xlim(0, 0.34)

    fig.suptitle("The low-energy failure: found per-condition, fixed by "
                 "capacity", y=1.02)
    fig.tight_layout()
    out = os.path.join(FIG, "fig_energy_bins.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def fig_zero_inflation():
    """Per-layer observables carry a point mass at a physical boundary."""
    a, names, _ = observables_from_file(DS1, 8000, per_layer=True)
    idx = {n: i for i, n in enumerate(names)}
    # E_layer=0, r_mean=0 and sparsity=1 are the *same* event -- an empty
    # layer -- so those three curves coincide exactly. Drawn with decreasing
    # width so the coincidence is visible rather than looking like one line.
    groups = [("E_layer", 0.0, "layer energy = 0", 6.0, "-"),
              ("r_mean_layer", 0.0, "radial centre = 0", 3.2, "-"),
              ("sparsity_layer", 1.0, "sparsity = 1", 1.4, "-"),
              ("r_width_layer", 0.0, "radial width = 0", 2.2, "--")]
    cols = ["#B8C4CE", "#0E7C86", "#12181D", BAD]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    for (g, atom, lab, lw, ls), c in zip(groups, cols):
        frac = [(a[:, idx[f"{g}_{k}"]] == atom).mean() for k in range(45)]
        ax.plot(range(45), frac, lw=lw, ls=ls, color=c, label=lab)
    ax.set_xlabel("calorimeter layer (depth)")
    ax.set_ylabel("fraction of Geant4 showers exactly on the atom")
    ax.set_title("Over half the 187 columns carry a point mass")
    ax.legend(fontsize=8.5, loc="upper left", title="first three coincide:\n"
              "all three mean 'layer empty'", title_fontsize=8)
    ax.set_ylim(0, 0.7)

    # Recorded pairs from the d=187 run (DEVLOG section 16).
    ax = axes[1]
    atom = np.array([10.6, 17.0, 40.2, 59.7])
    oor = np.array([9.70, 8.74, 25.01, 31.61])
    ax.scatter(atom, oor, s=80, color=BAD, zorder=3,
               label="measured layers 1, 19, 31, 43")
    fit = np.polyfit(atom, oor, 1)
    xs = np.linspace(5, 65, 20)
    ax.plot(xs, np.polyval(fit, xs), "--", color=NEUTRAL, lw=1.8,
            label="corr over all 44 layers = +0.989")
    for x_, y_, k in zip(atom, oor, (1, 19, 31, 43)):
        ax.annotate(f"layer {k}", (x_, y_), textcoords="offset points",
                    xytext=(10, 6), fontsize=8.5, color=NEUTRAL)
    ax.set_xlabel("% of Geant4 showers on the atom")
    ax.set_ylabel("% of generated values physically impossible")
    ax.set_title("Each layer's failure rate is set by its atom mass")
    ax.legend(fontsize=8.5, loc="upper left")

    fig.suptitle("Why the per-layer model fails: a continuous flow cannot put "
                 "mass on a point", y=1.02)
    fig.tight_layout()
    out = os.path.join(FIG, "fig_zero_inflation.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def main():
    for p in (DS1, DS2):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} -- download ds2 from "
                             f"https://zenodo.org/records/6366271 into Tina/")
    print(f"extracting core observables from {N} showers in each file...")
    a, names, _ = observables_from_file(DS1, N)
    b, _, _ = observables_from_file(DS2, N)

    fig_voxel_order()
    fig_scale_bug(a, b, names)
    fig_sep_floor(a, b)
    fig_energy_bins()
    fig_zero_inflation()
    print("\nall post-midterm figures regenerated.")


if __name__ == "__main__":
    main()
