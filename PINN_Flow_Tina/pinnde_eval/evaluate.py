"""Single entry point for the three evaluation tiers.

    results = evaluate(real, gen, tier="full")     # Tier 1 + 2 + 3
    results = evaluate(real, gen, tier="monitor")   # Tier 3 only (fast)
    report(results)

``real`` and ``gen`` are torch tensors or numpy arrays of shape (N, d); they are
converted internally. The optional ``features_fn`` maps raw samples to
high-level features before any metric runs. Calorimeter observables or an
official CaloChallenge wrapper can plug in through this hook without
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
    # Tier 3 runs for both monitor and full evaluations.
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
        print(f"[pinnde_eval] {e} -- FPD/KPD not evaluated")
        results["fpd"] = None
        results["kpd"] = None

    return results


def evaluate_by_condition(real, gen, real_condition, gen_condition=None,
                          min_count=2, **kwargs):
    """Run ``evaluate`` separately for each shared condition/bin label.

    ``real_condition`` and ``gen_condition`` are 1D arrays such as class labels,
    incident-energy bins, or detector regions. If ``gen_condition`` is omitted,
    ``real_condition`` is used for both samples.
    """
    real, gen = check_pair(real, gen)
    real_condition = np.asarray(real_condition)
    if gen_condition is None:
        gen_condition = real_condition
    gen_condition = np.asarray(gen_condition)

    if len(real_condition) != len(real):
        raise ValueError("real_condition length must match real samples")
    if len(gen_condition) != len(gen):
        raise ValueError("gen_condition length must match gen samples")

    labels = np.intersect1d(np.unique(real_condition), np.unique(gen_condition))
    out = {}
    for label in labels:
        r_mask = real_condition == label
        g_mask = gen_condition == label
        if r_mask.sum() < min_count or g_mask.sum() < min_count:
            continue
        res = evaluate(real[r_mask], gen[g_mask], **kwargs)
        res["n_real"] = int(r_mask.sum())
        res["n_gen"] = int(g_mask.sum())
        out[label] = res
    return out


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
    """Overlay per-feature histograms and save to ``path`` if given."""
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
