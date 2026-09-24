"""Local discrepancy maps: *where* do two distributions disagree?

Answers the June 26 meeting question "extend the metric to be sensitive to
local variations in the distributions rather than global differences" -- i.e.
metrics that can tell in which regions of the showers the reconstruction works
or fails. A global score (one number) can hide a localized failure; these three
complementary tools localize it:

* ``mmd_witness``            -- the MMD witness function evaluated at query
  points: > 0 where *real* is over-dense (generator misses mass), < 0 where
  *generated* is over-dense (generator hallucinates mass). Uses the same RBF
  kernel/bandwidth as ``tier3.mmd``, so it literally decomposes the global MMD
  monitor in space.
* ``classifier_discrepancy`` -- out-of-fold P(real | x) per sample from the
  Tier-1 two-sample classifier. Samples scoring far from 0.5 live in regions
  the classifier can tell apart; sorting by it ranks the worst regions.
* ``binned_residual_map``    -- per-bin normalized residuals of the 1D/2D
  histogram comparison: each bin is ~N(0,1) under the null, so |r| > 3 flags a
  region where the histograms genuinely disagree. This is the Tier-1 two-sample
  chi^2 decomposed bin by bin (same statistic, same sign convention:
  positive = real excess).

All three run in feature space, so for the calorimeter they apply per shower
observable (or per layer/voxel block) through the same ``features_fn`` hook as
``evaluate``.
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from ._utils import check_pair, to_torch
from .tier3 import _pairwise_sq_dists, median_bandwidth


def mmd_witness(real, gen, points=None, bandwidth=None, device="cpu", seed=0):
    """Evaluate the MMD witness function at ``points``.

    witness(z) = mean_i k(x_i, z) - mean_j k(y_j, z), with the same Gaussian
    RBF kernel and median-heuristic bandwidth as ``tier3.mmd``. Positive values
    mark regions where real is over-dense relative to generated; negative
    values mark generated excess; ~0 everywhere means the samples agree.

    ``points`` defaults to the pooled (real + generated) sample, which puts
    query points exactly where there is probability mass to inspect. Returns a
    numpy array of shape (len(points),).
    """
    real = to_torch(real, device)
    gen = to_torch(gen, device)
    if real.shape[1] != gen.shape[1]:
        raise ValueError(
            f"real and gen must share feature dimension, got {real.shape[1]} vs {gen.shape[1]}"
        )
    if points is None:
        points = torch.cat([real, gen], dim=0)
    else:
        points = to_torch(points, device)
        if points.shape[1] != real.shape[1]:
            raise ValueError(
                f"points must share feature dimension, got {points.shape[1]} vs {real.shape[1]}"
            )

    if bandwidth is None:
        sigma = median_bandwidth(real, gen, seed=seed)
    else:
        sigma = torch.as_tensor(float(bandwidth), device=real.device)
    gamma = 1.0 / (2.0 * sigma * sigma)

    k_real = torch.exp(-gamma * _pairwise_sq_dists(points, real)).mean(dim=1)
    k_gen = torch.exp(-gamma * _pairwise_sq_dists(points, gen)).mean(dim=1)
    return (k_real - k_gen).cpu().numpy()


def classifier_discrepancy(real, gen, n_folds=5, hidden_layer_sizes=(64, 64),
                           max_iter=300, seed=0):
    """Per-sample out-of-fold P(real | x) from the two-sample classifier.

    Trains the Tier-1 MLP on stratified folds and scores every sample with the
    fold that did *not* train on it, so the probabilities are honest. Returns
    ``(p_real, p_gen, auc)``: the P(real) arrays aligned with the input rows,
    and the out-of-fold AUC (one number, comparable to Tier 1).

    Interpretation: where the generator is fine, both arrays sit near 0.5.
    Real samples with p_real -> 1 live in regions the generator under-covers;
    generated samples with p_gen -> 0 are confidently fake (hallucinated
    regions). Sorting ``gen`` by ``p_gen`` ascending ranks the worst samples.
    """
    real, gen = check_pair(real, gen)
    X = np.vstack([real, gen])
    y = np.concatenate([np.ones(len(real)), np.zeros(len(gen))])

    proba = np.empty(len(X))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for i, (tr, te) in enumerate(skf.split(X, y)):
        scaler = StandardScaler().fit(X[tr])
        clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                            max_iter=max_iter, random_state=seed + i)
        clf.fit(scaler.transform(X[tr]), y[tr])
        proba[te] = clf.predict_proba(scaler.transform(X[te]))[:, 1]

    auc = float(roc_auc_score(y, proba))
    return proba[: len(real)], proba[len(real):], auc


def binned_residual_map(real, gen, features=(0, 1), bins=20, edges=None):
    """Signed per-bin residuals of the histogram comparison on 1 or 2 features.

    Histograms both samples on a common grid over the selected feature axes and
    returns ``(residuals, edges)`` where each bin holds

        r = (sqrt(Tg/Tr) * n_real - sqrt(Tr/Tg) * n_gen) / sqrt(n_real + n_gen)

    -- the signed square root of that bin's contribution to the Tier-1
    two-sample chi^2 (Tr, Tg are the total counts). Under the null each bin is
    ~N(0,1), so ``|r| > 3`` localizes a genuine disagreement; positive = real
    excess (generator misses this region), negative = generated excess. Empty
    bins give 0. ``features`` picks one axis (1D: returns shape ``(bins,)`` and
    one edge array) or two axes (2D: shape ``(bins, bins)`` and two edge
    arrays, indexed [i, j] = [feature-0 bin, feature-1 bin]).
    """
    real, gen = check_pair(real, gen)
    feats = (features,) if np.isscalar(features) else tuple(features)
    if len(feats) not in (1, 2):
        raise ValueError(f"features must select 1 or 2 axes, got {len(feats)}")
    for f in feats:
        if not 0 <= f < real.shape[1]:
            raise ValueError(f"feature index {f} out of range for d={real.shape[1]}")

    r = real[:, list(feats)]
    g = gen[:, list(feats)]
    if edges is None:
        edges = []
        for j in range(len(feats)):
            lo = min(r[:, j].min(), g[:, j].min())
            hi = max(r[:, j].max(), g[:, j].max())
            if hi <= lo:
                hi = lo + 1.0
            edges.append(np.linspace(lo, hi, bins + 1))
    else:
        edges = [np.asarray(e) for e in (edges if isinstance(edges, (list, tuple))
                                         else [edges])]

    if len(feats) == 1:
        n_real, _ = np.histogram(r[:, 0], bins=edges[0])
        n_gen, _ = np.histogram(g[:, 0], bins=edges[0])
        edges_out = edges[0]
    else:
        n_real, _, _ = np.histogram2d(r[:, 0], r[:, 1], bins=edges)
        n_gen, _, _ = np.histogram2d(g[:, 0], g[:, 1], bins=edges)
        edges_out = tuple(edges)

    n_real = n_real.astype(np.float64)
    n_gen = n_gen.astype(np.float64)
    t_r, t_g = n_real.sum(), n_gen.sum()
    if t_r == 0 or t_g == 0:
        raise ValueError("binned_residual_map needs non-empty real and gen samples")

    tot = n_real + n_gen
    num = np.sqrt(t_g / t_r) * n_real - np.sqrt(t_r / t_g) * n_gen
    with np.errstate(divide="ignore", invalid="ignore"):
        res = np.where(tot > 0, num / np.sqrt(np.clip(tot, 1, None)), 0.0)
    return res, edges_out
