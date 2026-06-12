"""Single entry point that ties the three tiers together.

    results = evaluate(real, gen, tier="full")     # Tier 1 + 2 + 3
    results = evaluate(real, gen, tier="monitor")   # Tier 3 only (fast)
    report(results)

``real`` and ``gen`` are torch tensors or numpy arrays of shape (N, d); they are
converted internally. The optional ``features_fn`` maps raw samples to
high-level features before any metric runs -- the hook through which calorimeter
observables (or the official CaloChallenge wrapper) plug in later without
changing this API.
"""

import numpy as np

from ._utils import check_pair
from .tier1 import classifier_two_sample_test, histogram_chi2
from .tier2 import fpd, kpd, wasserstein_per_feature
from .tier3 import mmd, swd


def evaluate(real, gen, tier="full", features_fn=None,
             bins=50, n_classifier=5, n_projections=128,
             device="cpu", seed=0, fpd_kwargs=None, kpd_kwargs=None):
    """Run the metric suite and return a flat results dict.

    Keys: ``mmd``, ``swd`` (Tier 3, always); plus for ``tier="full"`` ``auc``
    (mean, std), ``chi2_per_feature``, ``chi2_mean``, ``w1_per_feature``,
    ``w1_mean``, ``fpd`` (value, error), ``kpd`` (value, error). FPD/KPD are
    ``None`` if jetnet is not installed.
    """
    if tier not in ("full", "monitor"):
        raise ValueError(f"tier must be 'full' or 'monitor', got {tier!r}")

    real, gen = check_pair(real, gen)
    if features_fn is not None:
        real, gen = check_pair(features_fn(real), features_fn(gen))

    results = {}
    # Tier 3 -- always (it is the whole of "monitor" and part of "full").
    results["mmd"] = mmd(real, gen, device=device, seed=seed)
    results["swd"] = swd(real, gen, n_projections=n_projections, device=device, seed=seed)
    if tier == "monitor":
        return results

    # Tier 1
    results["auc"] = classifier_two_sample_test(real, gen, k=n_classifier, seed=seed)
    chi2 = histogram_chi2(real, gen, bins=bins)
    results["chi2_per_feature"] = chi2
    results["chi2_mean"] = float(np.nanmean(chi2))

    # Tier 2
    w1 = wasserstein_per_feature(real, gen)
    results["w1_per_feature"] = w1
    results["w1_mean"] = float(np.mean(w1))
    try:
        results["fpd"] = fpd(real, gen, seed=seed, **(fpd_kwargs or {}))
        results["kpd"] = kpd(real, gen, seed=seed, **(kpd_kwargs or {}))
    except ImportError as e:
        print(f"[pinnde_eval] {e} -- skipping FPD/KPD")
        results["fpd"] = None
        results["kpd"] = None

    return results


def report(results, title="pinnde_eval"):
    """Print a results dict as a clean aligned table."""
    lines = []
    for key, val in results.items():
        if val is None:
            body = "n/a"
        elif isinstance(val, tuple):
            body = f"{val[0]:.4g} +/- {val[1]:.4g}"
        elif isinstance(val, np.ndarray):
            body = "[" + ", ".join(f"{x:.4g}" for x in np.atleast_1d(val)) + "]"
        else:
            body = f"{val:.4g}"
        lines.append(f"{key:>16s} : {body}")

    width = max(len(l) for l in lines)
    print(title)
    print("-" * width)
    print("\n".join(lines))


def plot_histograms(real, gen, path=None, bins=50, labels=("real", "gen")):
    """Overlay per-feature histograms (nice-to-have). Saves to ``path`` if given."""
    import matplotlib.pyplot as plt

    real, gen = check_pair(real, gen)
    d = real.shape[1]
    fig, axes = plt.subplots(1, d, figsize=(4 * d, 3), squeeze=False)
    for j in range(d):
        lo = min(real[:, j].min(), gen[:, j].min())
        hi = max(real[:, j].max(), gen[:, j].max())
        edges = np.linspace(lo, hi, bins + 1)
        ax = axes[0, j]
        ax.hist(real[:, j], bins=edges, density=True, alpha=0.5, label=labels[0])
        ax.hist(gen[:, j], bins=edges, density=True, alpha=0.5, label=labels[1])
        ax.set_title(f"feature {j}")
        ax.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    return fig
