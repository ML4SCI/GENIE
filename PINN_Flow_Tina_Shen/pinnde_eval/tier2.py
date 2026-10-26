"""Tier 2 -- statistically grounded distances (the headline numbers).

* ``fpd`` -- Frechet physics distance: 2-Wasserstein between Gaussian fits, with
  jetnet's infinite-sample extrapolation giving a central value and error bar.
* ``kpd`` -- kernel physics distance: polynomial-kernel MMD with batched
  uncertainty.
* ``wasserstein_per_feature`` -- 1D Wasserstein-1 distance per coordinate.

FPD and KPD are thin wrappers over ``jetnet.evaluation`` (a pip-installable,
peer-reviewed reference implementation). jetnet is imported lazily so the rest
of the package works without it.
"""

import numpy as np
from scipy.stats import wasserstein_distance

from ._utils import check_pair


def wasserstein_per_feature(real, gen):
    """Per-coordinate Wasserstein-1 distance. Returns a vector of length d."""
    real, gen = check_pair(real, gen)
    d = real.shape[1]
    return np.array([wasserstein_distance(real[:, j], gen[:, j]) for j in range(d)])


def _require_jetnet():
    try:
        from jetnet import evaluation
    except ImportError as e:  # pragma: no cover - exercised only without jetnet
        raise ImportError(
            "FPD/KPD require the 'jetnet' package: pip install jetnet"
        ) from e
    return evaluation


def fpd(real, gen, min_samples=None, max_samples=None,
        num_batches=20, num_points=10, seed=0):
    """Frechet physics distance with infinite-sample extrapolation.

    Returns ``(value, error)``. ``min_samples``/``max_samples`` bracket the batch
    sizes used for the extrapolation; if left as None they are derived from the
    sample count so the call works on small toy sets as well as large ones.
    """
    evaluation = _require_jetnet()
    real, gen = check_pair(real, gen)
    n = min(len(real), len(gen))
    if max_samples is None:
        max_samples = min(n, 50000)
    if min_samples is None:
        min_samples = max(1000, max_samples // 5)
    min_samples = min(min_samples, max_samples - 1)

    value, error = evaluation.fpd(
        real, gen,
        min_samples=min_samples, max_samples=max_samples,
        num_batches=num_batches, num_points=num_points, seed=seed,
    )
    return float(value), float(error)


def kpd(real, gen, num_batches=10, batch_size=None, seed=0):
    """Kernel physics distance (polynomial-kernel MMD) with batched uncertainty.

    Returns ``(median, error)``. ``batch_size`` defaults to min(5000, N).
    """
    evaluation = _require_jetnet()
    real, gen = check_pair(real, gen)
    n = min(len(real), len(gen))
    if batch_size is None:
        batch_size = min(5000, n)

    median, error = evaluation.kpd(
        real, gen, num_batches=num_batches, batch_size=batch_size, seed=seed,
    )
    return float(median), float(error)
