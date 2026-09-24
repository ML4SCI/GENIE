"""Calibration checks for the metric module on Gaussian-mixture toys.

Runnable top-to-bottom (``python -m pinnde_eval.validate_toys`` or pasted into a
Colab cell). The script runs three asserted checks:

1. Null test     -- two independent draws from the same 3D GMM. AUC ~ 0.5,
                    reduced chi^2 ~ 1, Tier-3 distances ~ 0.
2. Sensitivity   -- perturb the GMM (mean shift, variance scale, drop a
                    component) and show every metric grows with the perturbation.
3. Speed check   -- time the Tier-3 monitors on 5k samples (target << 1 s).

FPD/KPD are reported when jetnet is installed and skipped otherwise; the null
and sensitivity logic does not depend on them.
"""

import time

import numpy as np
from scipy.stats import spearmanr

from ._utils import seed_all
from .data import gmm_params, perturb_params, sample_gmm
from .evaluate import evaluate, report
from .tier3 import mmd, swd


def null_test(d=3, n=8000, seed=0):
    """Two independent samples from the same GMM should look identical."""
    print("\n=== 1. NULL TEST (same distribution, independent draws) ===")
    seed_all(seed)
    params = gmm_params(d=d, k=8, seed=seed)
    real = sample_gmm(params, n, seed=seed + 1)
    gen = sample_gmm(params, n, seed=seed + 2)

    results = evaluate(real, gen, tier="full", seed=seed)
    report(results, title="null test (real vs independent draw)")

    auc_mean, auc_std = results["auc"]
    assert abs(auc_mean - 0.5) < 0.05, f"AUC not ~0.5: {auc_mean:.3f}"
    assert results["chi2_mean"] < 2.0, f"reduced chi2 too high: {results['chi2_mean']:.3f}"
    assert results["swd"] < 0.1, f"SWD not ~0: {results['swd']:.4f}"
    assert abs(results["mmd"]) < 1e-2, f"MMD not ~0: {results['mmd']:.4e}"
    if results["fpd"] is not None:
        fpd_val, fpd_err = results["fpd"]
        assert fpd_val < 5.0 * fpd_err + 1e-3, f"FPD not consistent with 0: {results['fpd']}"
    print("null test passed: metrics consistent with 'no difference'.")
    return results


def sensitivity_test(d=3, n=8000, seed=0, eps_grid=(0.0, 0.05, 0.1, 0.2, 0.4),
                     plot_path=None):
    """Each metric should grow with the perturbation size eps.

    Monotonicity is checked with the Spearman rank correlation between eps and
    the metric (>= 0.8): this captures "grows with eps" while tolerating the
    sub-noise-floor wiggle that any distance shows when the perturbation is
    smaller than the sampling noise.
    """
    print("\n=== 2. SENSITIVITY TEST (metric vs perturbation size) ===")
    seed_all(seed)
    params = gmm_params(d=d, k=8, seed=seed)
    real = sample_gmm(params, n, seed=seed + 1)

    eps_arr = np.array(eps_grid)
    tables = {}
    for kind in ("mean", "var", "drop"):
        print(f"\n-- perturbation: {kind} --")
        print(f"{'eps':>6} | {'mmd':>10} | {'swd':>10} | {'w1_mean':>10} | {'auc':>10}")
        rows = []
        for eps in eps_grid:
            gen = sample_gmm(perturb_params(params, kind, eps), n, seed=seed + 2)
            res = evaluate(real, gen, tier="full", seed=seed)
            auc = res["auc"][0]
            rows.append((eps, res["mmd"], res["swd"], res["w1_mean"], auc))
            print(f"{eps:6.2f} | {res['mmd']:10.4e} | {res['swd']:10.4f} | "
                  f"{res['w1_mean']:10.4f} | {auc:10.4f}")
        tables[kind] = np.array(rows)

        # Every metric should rank-correlate strongly with eps.
        for col, name in ((1, "mmd"), (2, "swd"), (3, "w1_mean"), (4, "auc")):
            series = tables[kind][:, col]
            rho = spearmanr(eps_arr, series).correlation
            assert rho >= 0.8, f"{name} not monotone for {kind}: rho={rho:.2f}, {series}"
        # The strong distances should at least double over their eps=0 noise floor.
        for col, name in ((2, "swd"), (3, "w1_mean")):
            series = tables[kind][:, col]
            assert series[-1] > 2 * series[0], f"{name} did not grow for {kind}: {series}"
        # The largest perturbation must lift AUC clearly above the null floor.
        auc = tables[kind][:, 4]
        assert auc[-1] > auc[0] + 0.01, f"AUC did not rise for {kind}: {auc}"

    if plot_path:
        _plot_sensitivity(tables, plot_path)

    print("\nsensitivity test passed: distances rank-correlate with eps (rho>=0.8); "
          "AUC separates at large eps.")
    return tables


def _plot_sensitivity(tables, path):
    """Save a metric-vs-eps grid for validation records."""
    import matplotlib.pyplot as plt

    metrics = [(1, "mmd"), (2, "swd"), (3, "w1_mean"), (4, "auc")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.2), squeeze=False)
    for ax, (col, name) in zip(axes[0], metrics):
        for kind, tab in tables.items():
            ax.plot(tab[:, 0], tab[:, col], marker="o", label=kind)
        ax.set_xlabel("eps")
        ax.set_title(name)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"saved sensitivity plot to {path}")


def speed_check(d=3, n=5000, seed=0):
    """Tier-3 monitors must be fast enough to call inside a training loop."""
    print("\n=== 3. SPEED CHECK (Tier-3 monitors on 5k samples) ===")
    seed_all(seed)
    params = gmm_params(d=d, k=8, seed=seed)
    real = sample_gmm(params, n, seed=seed + 1)
    gen = sample_gmm(params, n, seed=seed + 2)

    t0 = time.perf_counter()
    m = mmd(real, gen, seed=seed)
    t_mmd = time.perf_counter() - t0

    t0 = time.perf_counter()
    s = swd(real, gen, seed=seed)
    t_swd = time.perf_counter() - t0

    print(f"mmd: {m:.4e}  ({t_mmd * 1e3:.1f} ms)")
    print(f"swd: {s:.4f}  ({t_swd * 1e3:.1f} ms)")
    assert t_mmd + t_swd < 2.0, "Tier-3 monitors slower than expected on 5k samples"
    print("speed check passed.")
    return t_mmd, t_swd


def main():
    null_test()
    sensitivity_test()
    speed_check()
    print("\nAll validation checks passed.")


if __name__ == "__main__":
    main()
