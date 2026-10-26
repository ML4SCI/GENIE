"""Sample-size stability study: how do the metrics behave as N changes?

Answers the June 26 meeting question "investigate the stability of the metrics
comparing two distributions with respect to the number of samples". For every
sample size N in a grid, the study repeatedly draws

* a **null** pair  -- two independent samples from the *same* GMM, and
* a **signal** pair -- the same real sample vs. a draw from a *perturbed* GMM,

and records each metric. The repeats give the finite-N sampling distribution of
the metric, from which three practical questions are answered:

1. Where is the **null floor** (the value the metric shows for a perfect
   generator at that N) and how fast does it shrink with N?
2. How large is the **spread** (error bar) of a single measurement at that N?
3. At which N does a fixed perturbation become **resolvable**, i.e. the signal
   separates from the null floor by ``z_min`` combined standard deviations?

Runnable:  ``python -m pinnde_eval.stability``  (prints the table per metric and
saves ``figures/stability.png`` when run from the ``Tina/`` folder).
"""

import numpy as np

from ._utils import seed_all
from .data import gmm_params, perturb_params, sample_gmm
from .tier1 import classifier_two_sample_test, histogram_chi2
from .tier2 import wasserstein_per_feature
from .tier3 import mmd, swd

# The metric's ideal "no difference" value; spreads and z-scores are measured
# around these, and the plots draw them as reference lines.
IDEAL = {"mmd": 0.0, "swd": 0.0, "w1_mean": 0.0, "chi2_mean": 1.0, "auc": 0.5}


def _metric_row(real, gen, seed, include_auc, auc_k):
    """All tracked metrics for one (real, gen) pair -> dict of floats."""
    row = {
        "mmd": mmd(real, gen, seed=seed),
        "swd": swd(real, gen, seed=seed),
        "w1_mean": float(wasserstein_per_feature(real, gen).mean()),
        "chi2_mean": float(np.nanmean(histogram_chi2(real, gen))),
    }
    if include_auc:
        row["auc"] = classifier_two_sample_test(real, gen, k=auc_k, seed=seed)[0]
    return row


def stability_study(d=3, k=8, n_grid=(250, 500, 1000, 2000, 4000, 8000),
                    n_repeats=6, kind="mean", eps=0.2, seed=0,
                    include_auc=True, auc_k=2, verbose=False):
    """Measure every metric's null floor and signal response as N grows.

    Returns a dict with ``n_grid``, ``kind``, ``eps`` and, per metric name, a
    dict of two ``(len(n_grid), n_repeats)`` arrays: ``"null"`` (real vs.
    independent same-GMM draw) and ``"signal"`` (real vs. draw from the GMM
    perturbed by (kind, eps)). Every draw uses a fresh deterministic seed, so
    the repeat axis is the metric's true finite-N sampling distribution.
    """
    seed_all(seed)
    params = gmm_params(d=d, k=k, seed=seed)
    pert = perturb_params(params, kind, eps)

    names = ["mmd", "swd", "w1_mean", "chi2_mean"] + (["auc"] if include_auc else [])
    out = {name: {"null": np.empty((len(n_grid), n_repeats)),
                  "signal": np.empty((len(n_grid), n_repeats))} for name in names}

    draw_seed = seed + 1   # incremented per draw -> all samples independent
    for ni, n in enumerate(n_grid):
        for r in range(n_repeats):
            real = sample_gmm(params, n, seed=draw_seed)
            gen_null = sample_gmm(params, n, seed=draw_seed + 1)
            gen_sig = sample_gmm(pert, n, seed=draw_seed + 2)
            draw_seed += 3

            null_row = _metric_row(real, gen_null, seed, include_auc, auc_k)
            sig_row = _metric_row(real, gen_sig, seed, include_auc, auc_k)
            for name in names:
                out[name]["null"][ni, r] = null_row[name]
                out[name]["signal"][ni, r] = sig_row[name]
        if verbose:
            print(f"  n={n}: done ({n_repeats} repeats)")

    out["n_grid"] = np.array(n_grid)
    out["kind"] = kind
    out["eps"] = eps
    return out


def _metric_names(study):
    return [k for k in study if isinstance(study[k], dict)]


def separation_z(study):
    """Per metric: z(N) = (signal mean - null mean) / combined std.

    z >= 2 means a single measurement at that N separates the perturbed
    generator from a perfect one by two error bars. MMD/χ² z-scores use the
    absolute difference (their signal can only move one way from the ideal).
    """
    zs = {}
    for name in _metric_names(study):
        null, sig = study[name]["null"], study[name]["signal"]
        gap = np.abs(sig.mean(axis=1) - null.mean(axis=1))
        spread = np.sqrt(null.std(axis=1) ** 2 + sig.std(axis=1) ** 2)
        zs[name] = gap / np.clip(spread, 1e-12, None)
    return zs


def min_resolvable_n(study, z_min=2.0):
    """Smallest N in the grid where each metric resolves the perturbation.

    Requires z >= z_min at that N *and every larger N* (a single lucky
    fluctuation at small N does not count). Returns ``{metric: N or None}``.
    """
    zs = separation_z(study)
    out = {}
    for name, z in zs.items():
        ok = z >= z_min
        resolved = None
        for i in range(len(ok)):
            if ok[i:].all():
                resolved = int(study["n_grid"][i])
                break
        out[name] = resolved
    return out


def report_stability(study, z_min=2.0):
    """Print null floor, spread, signal, and separation z for every metric and N."""
    zs = separation_z(study)
    resolved = min_resolvable_n(study, z_min=z_min)
    print(f"perturbation: kind={study['kind']!r}, eps={study['eps']}, "
          f"repeats={study[_metric_names(study)[0]]['null'].shape[1]}")
    for name in _metric_names(study):
        null, sig = study[name]["null"], study[name]["signal"]
        print(f"\n-- {name} (ideal {IDEAL[name]}) --")
        print(f"{'N':>6} | {'null mean':>11} | {'null std':>10} | "
              f"{'signal mean':>11} | {'z':>6}")
        for i, n in enumerate(study["n_grid"]):
            print(f"{n:6d} | {null[i].mean():11.4g} | {null[i].std():10.3g} | "
                  f"{sig[i].mean():11.4g} | {zs[name][i]:6.1f}")
        n_res = resolved[name]
        print(f"   resolves the perturbation (z>={z_min:g}) from N = "
              f"{n_res if n_res is not None else '> grid max'}")


def plot_stability(study, path, z_min=2.0):
    """Two-row figure: metric vs N (null and signal bands), and z vs N."""
    import matplotlib.pyplot as plt

    names = _metric_names(study)
    n = study["n_grid"]
    zs = separation_z(study)
    fig, axes = plt.subplots(2, len(names), figsize=(3.6 * len(names), 6.4),
                             squeeze=False)

    def _log_x(ax):
        ax.set_xscale("log")
        ax.set_xticks(n)
        ax.set_xticklabels([f"{v // 1000}k" if v >= 1000 else str(v) for v in n])
        ax.minorticks_off()

    for j, name in enumerate(names):
        null, sig = study[name]["null"], study[name]["signal"]
        ax = axes[0, j]
        for arr, label, color in ((null, "null (same GMM)", "tab:blue"),
                                  (sig, f"{study['kind']} eps={study['eps']}", "tab:red")):
            m, s = arr.mean(axis=1), arr.std(axis=1)
            ax.plot(n, m, marker="o", color=color, label=label)
            ax.fill_between(n, m - s, m + s, color=color, alpha=0.2)
        ax.axhline(IDEAL[name], ls=":", color="gray", lw=1)
        _log_x(ax)
        ax.set_title(name)
        if j == 0:
            ax.set_ylabel("metric value (mean ± std)")
            ax.legend(fontsize=8)

        ax = axes[1, j]
        ax.plot(n, zs[name], marker="o", color="black")
        ax.axhline(z_min, ls="--", color="tab:red", lw=1, label=f"z = {z_min:g}")
        _log_x(ax)
        ax.set_xlabel("samples N")
        if j == 0:
            ax.set_ylabel("separation z(N)")
            ax.legend(fontsize=8)
    fig.suptitle("Metric stability vs sample size: null floor and resolvability",
                 y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"saved stability plot to {path}")
    return fig


def main():
    import os

    print("=== METRIC STABILITY vs SAMPLE SIZE ===")
    print("(null floor + spread + resolvability of a mean-shift eps=0.2; "
          "takes a few minutes because of the classifier)\n")
    study = stability_study(verbose=True)
    report_stability(study)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
    if os.path.isdir(out):
        plot_stability(study, os.path.join(out, "stability.png"))


if __name__ == "__main__":
    main()
