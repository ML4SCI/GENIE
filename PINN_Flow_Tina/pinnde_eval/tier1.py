"""Tier 1 -- CaloChallenge-standard metrics.

* ``classifier_two_sample_test`` -- train a small MLP to separate real from
  generated samples on a held-out split; report test AUC as mean +/- std over k
  retrainings. AUC ~ 0.5 means the two samples are indistinguishable.
* ``histogram_chi2`` -- per-feature 1D histograms on common binning and the
  reduced chi-squared between the real and generated histograms. ~1 means the
  histograms agree within statistical fluctuations.

For toys the "features" are the raw coordinates; for the calorimeter the same
path takes high-level shower observables (passed in via ``evaluate``'s
``features_fn``).
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from ._utils import check_pair


def classifier_two_sample_test(real, gen, k=5, hidden_layer_sizes=(64, 64),
                               test_size=0.3, max_iter=300, seed=0):
    """Classifier two-sample test. Returns ``(mean_auc, std_auc)`` over k retrainings.

    Each retraining uses a different seed, so it gets a different train/test split
    and a different weight init -- the spread across runs is the uncertainty.
    """
    real, gen = check_pair(real, gen)
    X = np.vstack([real, gen])
    y = np.concatenate([np.zeros(len(real)), np.ones(len(gen))])

    aucs = []
    for i in range(k):
        rs = seed + i
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=rs, stratify=y
        )
        scaler = StandardScaler().fit(X_tr)
        clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=rs
        )
        clf.fit(scaler.transform(X_tr), y_tr)
        proba = clf.predict_proba(scaler.transform(X_te))[:, 1]
        aucs.append(roc_auc_score(y_te, proba))

    aucs = np.array(aucs)
    return float(aucs.mean()), float(aucs.std())


def _two_sample_chi2(a, b):
    """Reduced chi-squared between two histograms with counts a, b.

    Uses the scaled two-sample statistic that is correct even when the totals
    differ; empty bins are dropped and the result is divided by the degrees of
    freedom, so independent samples from the same distribution give ~1.
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    total_a, total_b = a.sum(), b.sum()
    if total_a == 0 or total_b == 0:
        return float("nan")

    mask = (a + b) > 0
    a, b = a[mask], b[mask]
    num = (np.sqrt(total_b / total_a) * a - np.sqrt(total_a / total_b) * b) ** 2
    chi2 = (num / (a + b)).sum()
    ndof = max(int(mask.sum()) - 1, 1)
    return float(chi2 / ndof)


def histogram_chi2(real, gen, bins=50):
    """Reduced chi-squared per feature on common binning. Returns a vector of length d."""
    real, gen = check_pair(real, gen)
    d = real.shape[1]
    out = np.empty(d)
    for j in range(d):
        lo = min(real[:, j].min(), gen[:, j].min())
        hi = max(real[:, j].max(), gen[:, j].max())
        if hi <= lo:
            hi = lo + 1.0
        edges = np.linspace(lo, hi, bins + 1)
        a, _ = np.histogram(real[:, j], bins=edges)
        b, _ = np.histogram(gen[:, j], bins=edges)
        out[j] = _two_sample_chi2(a, b)
    return out
