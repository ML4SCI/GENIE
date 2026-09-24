"""Tier 3 -- cheap, pure-torch training-loop monitors.

Fast metrics meant to be called every few hundred steps on ~1-5k samples:

* ``mmd`` -- squared Maximum Mean Discrepancy with a Gaussian RBF kernel and a
  median-heuristic bandwidth (optionally a small bandwidth mixture).
* ``swd`` -- Sliced Wasserstein distance via random 1D projections.

Both accept torch tensors or numpy arrays of shape (N, d) and return a python
float, so they drop straight into a training loop.
"""

import torch

from ._utils import to_torch


def _pairwise_sq_dists(x, y):
    """Squared Euclidean distances between rows of x (n,d) and y (m,d) -> (n,m)."""
    x2 = (x * x).sum(1, keepdim=True)        # (n, 1)
    y2 = (y * y).sum(1, keepdim=True).t()    # (1, m)
    d2 = x2 + y2 - 2.0 * (x @ y.t())
    return d2.clamp_min(0.0)


def median_bandwidth(real, gen, max_points=2000, seed=0):
    """Median pairwise distance over the pooled sample (the median heuristic)."""
    z = torch.cat([real, gen], dim=0)
    n = z.shape[0]
    if n > max_points:
        g = torch.Generator(device=z.device).manual_seed(seed)
        idx = torch.randperm(n, generator=g, device=z.device)[:max_points]
        z = z[idx]
    d2 = _pairwise_sq_dists(z, z)
    iu = torch.triu_indices(z.shape[0], z.shape[0], offset=1, device=z.device)
    dists = d2[iu[0], iu[1]].clamp_min(0.0).sqrt()
    med = torch.median(dists)
    return med.clamp_min(1e-12)


def mmd(real, gen, bandwidth=None, n_bandwidths=1, device="cpu", seed=0):
    """Unbiased squared MMD with a Gaussian RBF kernel.

    ``bandwidth`` is the RBF sigma; if None it is set by the median heuristic on
    the pooled sample. With ``n_bandwidths > 1`` the estimate is averaged over a
    geometric mixture of bandwidths spanning [sigma/2, 2*sigma], which makes the
    monitor less sensitive to a single scale choice.

    Returns the unbiased MMD^2 estimate. For two samples from the same
    distribution it fluctuates around 0 and may be slightly negative -- that is
    expected and is evidence the estimator is unbiased.
    """
    real = to_torch(real, device)
    gen = to_torch(gen, device)
    if real.shape[1] != gen.shape[1]:
        raise ValueError(
            f"real and gen must share feature dimension, got {real.shape[1]} vs {gen.shape[1]}"
        )
    n, m = real.shape[0], gen.shape[0]
    if n < 2 or m < 2:
        raise ValueError("mmd needs at least 2 samples in each of real and gen")

    if bandwidth is None:
        sigma = median_bandwidth(real, gen, seed=seed)
    else:
        sigma = torch.as_tensor(float(bandwidth), device=real.device)

    if n_bandwidths > 1:
        factors = torch.logspace(-1.0, 1.0, n_bandwidths, base=2.0, device=real.device)
        sigmas = sigma * factors
    else:
        sigmas = sigma.reshape(1)

    dxx = _pairwise_sq_dists(real, real)
    dyy = _pairwise_sq_dists(gen, gen)
    dxy = _pairwise_sq_dists(real, gen)

    total = torch.zeros((), device=real.device)
    for s in sigmas:
        gamma = 1.0 / (2.0 * s * s)
        kxx = torch.exp(-gamma * dxx)
        kyy = torch.exp(-gamma * dyy)
        kxy = torch.exp(-gamma * dxy)
        # unbiased: drop the self-similarity on the diagonal
        sum_xx = (kxx.sum() - torch.diagonal(kxx).sum()) / (n * (n - 1))
        sum_yy = (kyy.sum() - torch.diagonal(kyy).sum()) / (m * (m - 1))
        sum_xy = kxy.mean()
        total = total + sum_xx + sum_yy - 2.0 * sum_xy

    return float(total / len(sigmas))


def swd(real, gen, n_projections=128, device="cpu", seed=0):
    """Sliced Wasserstein distance.

    Projects both samples onto ``n_projections`` random unit directions, computes
    the exact 1D Wasserstein-1 distance per slice (mean abs difference of sorted
    projections), and averages. Samples are subsampled to the smaller size so the
    sorted arrays align; the subsample is seeded for reproducibility.
    """
    real = to_torch(real, device)
    gen = to_torch(gen, device)
    if real.shape[1] != gen.shape[1]:
        raise ValueError(
            f"real and gen must share feature dimension, got {real.shape[1]} vs {gen.shape[1]}"
        )
    d = real.shape[1]
    n = min(real.shape[0], gen.shape[0])
    if n < 1:
        raise ValueError("swd needs at least 1 sample in each of real and gen")

    g = torch.Generator(device=real.device).manual_seed(seed)
    proj = torch.randn(d, n_projections, generator=g, device=real.device)
    proj = proj / proj.norm(dim=0, keepdim=True).clamp_min(1e-12)

    idx_r = torch.randperm(real.shape[0], generator=g, device=real.device)[:n]
    idx_g = torch.randperm(gen.shape[0], generator=g, device=real.device)[:n]

    pr = (real[idx_r] @ proj).sort(dim=0).values   # (n, P)
    pg = (gen[idx_g] @ proj).sort(dim=0).values     # (n, P)
    return float((pr - pg).abs().mean())


def _subsample(x, max_points, seed):
    """Seeded row subsample, so the estimate is reproducible."""
    if max_points is None or len(x) <= max_points:
        return x
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(len(x), generator=g)[:max_points].to(x.device)
    return x[idx]


def _sinkhorn_cost(x, y, eps, n_iter):
    """Entropic optimal transport cost between two uniform point clouds.

    Log-domain Sinkhorn iterations on the dual potentials, which is the
    numerically stable formulation -- the naive kernel version underflows as
    soon as ``eps`` is small relative to the costs.
    """
    n, m = len(x), len(y)
    cost = _pairwise_sq_dists(x, y)
    log_a = -torch.log(torch.tensor(float(n), device=x.device))
    log_b = -torch.log(torch.tensor(float(m), device=x.device))

    f = torch.zeros(n, device=x.device)
    g = torch.zeros(m, device=x.device)
    for _ in range(n_iter):
        f = -eps * torch.logsumexp((g.unsqueeze(0) - cost) / eps + log_b, dim=1)
        g = -eps * torch.logsumexp((f.unsqueeze(1) - cost) / eps + log_a, dim=0)

    plan = torch.exp((f.unsqueeze(1) + g.unsqueeze(0) - cost) / eps
                     + log_a + log_b)
    return (plan * cost).sum()


def sinkhorn(real, gen, eps=None, n_iter=100, max_points=2000, debias=True,
             device="cpu", seed=0):
    """Sinkhorn divergence: entropically regularised optimal transport.

    An *unbinned* transport distance, in the same family as the sliced
    Wasserstein distance but solving the transport problem in the full feature
    space rather than on 1-D projections. Answers "how far must one cloud of
    points be moved to become the other", with an entropy term that makes the
    problem solvable in O(n*m) per iteration instead of by a linear program.

    ``debias=True`` returns the **Sinkhorn divergence**

        S(x, y) = OT(x, y) - 0.5*OT(x, x) - 0.5*OT(y, y)

    rather than the raw entropic cost. This matters here: the raw cost
    ``OT(x, x)`` is *not* zero -- the entropy term biases it -- so an
    unmodified Sinkhorn distance would carry a large offset that behaves like
    yet another floor to measure. The debiased form is constructed to vanish
    when the two samples come from the same distribution, so it is the version
    worth reporting. Set ``debias=False`` to see the raw cost.

    ``eps`` (the entropic blur) defaults to the median pairwise squared
    distance divided by 20, so it adapts to the scale of the data the way
    ``mmd``'s bandwidth does. Smaller ``eps`` approaches exact optimal
    transport but converges more slowly.

    **Scale-dependent**, like ``swd`` and the per-feature Wasserstein: the cost
    is squared Euclidean distance in feature space, so a single large-scale
    observable dominates. Pass ``evaluate(..., standardize=True)`` when the
    features carry different units (DEVLOG section 11).

    ``max_points`` caps both samples (seeded) because the cost matrix is
    n x m; 2000 points is a 4M-entry matrix, 8000 would be 64M.
    """
    real = _subsample(to_torch(real, device), max_points, seed)
    # Same seed for both: identical inputs must subsample identically, so
    # that S(x, x) is exactly 0 rather than a spurious finite-sample offset.
    gen = _subsample(to_torch(gen, device), max_points, seed)
    if real.shape[1] != gen.shape[1]:
        raise ValueError(f"real and gen must share feature dimension, got "
                         f"{real.shape[1]} vs {gen.shape[1]}")

    if eps is None:
        with torch.no_grad():
            med = _pairwise_sq_dists(real, gen).median()
        eps = float(med) / 20.0
    eps = max(float(eps), 1e-12)

    with torch.no_grad():
        value = _sinkhorn_cost(real, gen, eps, n_iter)
        if debias:
            value = value - 0.5 * _sinkhorn_cost(real, real, eps, n_iter) \
                          - 0.5 * _sinkhorn_cost(gen, gen, eps, n_iter)
    return float(value)
