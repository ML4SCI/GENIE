"""Figures for the classical two-sample tests (see pinnde_eval/validate_classical.py).

    OPENBLAS_NUM_THREADS=1 python make_classical_figures.py

  fig_calibration.png   KS p-values under an exact (permutation) null, compared
                        with the same test run across the two Geant4 files
  fig_power.png         classical tests detect a shift that separation power
                        cannot see at any sample size
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pinnde_eval.classical import (combine_pvalues, pvalue_uniformity,
                                   two_sample_tests)
from pinnde_eval.observables import observables_from_file
from pinnde_eval.tier1 import separation_power

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
N_LOAD, CHUNK = 100000, 1000
GOOD, BAD, NEUTRAL = "#1F8A5B", "#C2413A", "#41566B"

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12,
                     "axes.labelsize": 11, "figure.dpi": 130})


def fig_calibration(a, b, names):
    """Left: p-value histograms. Right: per-observable uniformity."""
    rng = np.random.default_rng(0)
    pool = np.vstack([a, b])

    # Disjoint on both sides, so the two histograms are built the same way.
    perm_order = rng.permutation(len(pool))
    perm, across = [], []
    for i in range(30):
        blk = perm_order[i * 2 * CHUNK:(i + 1) * 2 * CHUNK]
        perm.append(two_sample_tests(pool[blk[:CHUNK]], pool[blk[CHUNK:]],
                                     tests=("ks",))["ks"]["pvalue"])
        across.append(two_sample_tests(a[i * CHUNK:(i + 1) * CHUNK],
                                       b[i * CHUNK:(i + 1) * CHUNK],
                                       tests=("ks",))["ks"]["pvalue"])
    perm, across = np.concatenate(perm), np.concatenate(across)

    # Fresh generator, matching validate_classical's check_calibration, so the
    # figure and the module report the same numbers.
    # DISJOINT splits. Drawing each split independently reuses showers across
    # splits, correlating the p-values and manufacturing apparent differences
    # between tests that more sampling does not reproduce.
    rng2 = np.random.default_rng(0)
    order = rng2.permutation(len(pool))
    n_split = len(pool) // (2 * CHUNK)
    ks_u, es_u = [], []
    for i in range(n_split):
        blk = order[i * 2 * CHUNK:(i + 1) * 2 * CHUNK]
        r = two_sample_tests(pool[blk[:CHUNK]], pool[blk[CHUNK:]],
                             tests=("ks", "ad"))
        ks_u.append(r["ks"]["pvalue"])
        es_u.append(r["ad"]["pvalue"])
    ks_u, es_u = np.array(ks_u), np.array(es_u)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    ax = axes[0]
    bins = np.linspace(0, 1, 11)
    ax.hist(perm, bins=bins, alpha=0.75, color=GOOD, label="permutation null (exact)")
    ax.hist(across, bins=bins, histtype="step", lw=2.2, color=BAD,
            label="Geant4 file 1 vs file 2")
    ax.axhline(len(perm) / 10, ls="--", color=NEUTRAL, lw=1.6,
               label="uniform expectation")
    ax.set_xlabel("KS p-value")
    ax.set_ylabel("count")
    ax.set_title("A correct test gives uniform p-values under a true null")
    ax.legend(fontsize=8.5, loc="upper center")
    _, up, fp = pvalue_uniformity(perm)
    _, ua, fa = pvalue_uniformity(across)
    ax.text(0.02, 0.02,
            f"permutation : uniform p = {up:.3f},  frac<0.05 = {fp:.3f}\n"
            f"across files: uniform p = {ua:.4f},  frac<0.05 = {fa:.3f}",
            transform=ax.transAxes, fontsize=8.5, va="bottom", family="monospace")

    ax = axes[1]
    y = np.arange(len(names))
    ks_p = [pvalue_uniformity(ks_u[:, j])[2] for j in range(len(names))]
    es_p = [pvalue_uniformity(es_u[:, j])[2] for j in range(len(names))]
    ax.barh(y - 0.2, ks_p, 0.4, color=NEUTRAL, label="Kolmogorov–Smirnov")
    ax.barh(y + 0.2, es_p, 0.4, color="#0E7C86", label="Anderson–Darling")
    ax.axvline(0.05, ls="--", color=BAD, lw=1.8,
               label="0.05 = the target rate")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("false-positive rate on a true null   (target 0.05)")
    ax.set_title(f"False-positive rate per observable ({n_split} disjoint splits)")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.set_xlim(0, max(max(ks_p), max(es_p)) * 1.3 + 0.01)

    fig.suptitle("Calibration: the classical tests are correct, and our "
                 "file-vs-file null is not a true null", y=1.02)
    fig.tight_layout()
    out = os.path.join(FIG, "fig_calibration.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def fig_power(a, log_e, n_grid=(250, 500, 1000, 2000), shift=0.05, n_repeats=20):
    """Detection rate vs N, against separation power and its floor."""
    lo, hi = np.quantile(log_e, [0.40, 0.60])
    band = a[(log_e >= lo) & (log_e < hi)]
    scale = band.std(axis=0)
    rng = np.random.default_rng(0)

    ks_pw, cvm_pw, seps, floors = [], [], [], []
    for n in n_grid:
        k = c = 0
        sp = []
        for _ in range(n_repeats):
            idx = rng.choice(len(band), 2 * n, replace=False)
            real, gen = band[idx[:n]], band[idx[n:]] + shift * scale
            r = two_sample_tests(real, gen, tests=("ks", "cvm"))
            k += combine_pvalues(r["ks"]["pvalue"]) < 0.05
            c += combine_pvalues(r["cvm"]["pvalue"]) < 0.05
            sp.append(separation_power(real, gen).mean())
        ks_pw.append(k / n_repeats)
        cvm_pw.append(c / n_repeats)
        seps.append(np.mean(sp))
        floors.append(50 / (2 * n))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    ax = axes[0]
    ax.plot(n_grid, ks_pw, "o-", lw=2, ms=7, color=NEUTRAL,
            label="Kolmogorov–Smirnov")
    ax.plot(n_grid, cvm_pw, "s-", lw=2, ms=7, color="#0E7C86",
            label="Cramér–von Mises")
    ax.axhline(0.05, ls="--", color=BAD, lw=1.6, label="chance (5%)")
    ax.set_xscale("log")
    ax.set_xticks(n_grid)
    ax.set_xticklabels([str(n) for n in n_grid])
    ax.minorticks_off()
    ax.set_xlabel("samples N")
    ax.set_ylabel("detection rate at the 5% level")
    ax.set_ylim(0, 1)
    ax.set_title("Classical tests detect the shift")
    ax.legend(fontsize=9, loc="upper left")

    ax = axes[1]
    ax.plot(n_grid, seps, "o-", lw=2, ms=7, color=NEUTRAL,
            label="separation power, perturbed data")
    ax.plot(n_grid, floors, "--", lw=2, color=BAD,
            label=r"its own null floor, $n_{bins}/2N$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(n_grid)
    ax.set_xticklabels([str(n) for n in n_grid])
    ax.minorticks_off()
    ax.set_xlabel("samples N")
    ax.set_ylabel("separation power")
    ax.set_title("Separation power sits on its floor: it sees nothing")
    ax.legend(fontsize=9, loc="upper right")

    fig.suptitle(f"The same {int(shift*100)}% shift, measured two ways "
                 f"(shift defined at fixed incident energy)", y=1.02)
    fig.tight_layout()
    out = os.path.join(FIG, "fig_power.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")
    print(f"  KS power  : {ks_pw}")
    print(f"  CvM power : {cvm_pw}")
    print(f"  sep       : {[round(v,5) for v in seps]}")
    print(f"  sep floor : {[round(v,5) for v in floors]}")
    plt.close(fig)


def main():
    for f in ("dataset_2_1.hdf5", "dataset_2_2.hdf5"):
        if not os.path.exists(f):
            raise SystemExit(f"missing {f} -- download ds2 from "
                             f"https://zenodo.org/records/6366271")
    print(f"extracting observables from {N_LOAD} showers in each file...")
    a, names, e_a = observables_from_file("dataset_2_1.hdf5", N_LOAD)
    b, _, _ = observables_from_file("dataset_2_2.hdf5", N_LOAD)
    fig_calibration(a, b, names)
    fig_power(a, np.log(e_a))
    print("\nclassical-test figures regenerated.")


if __name__ == "__main__":
    main()
