"""Seeded Gaussian-mixture toys for metric validation.

These mirror the GMMs in ``flow_de/gendata.py`` (same shapes and ranges) but are
fully seeded and split into parameter generation and sampling. That separation
lets the validation draw two independent samples from the same distribution for
the null test, then apply controlled perturbations for the sensitivity test.

Both PINNDE tracks can validate against the same toys by using these generators.
"""

import numpy as np
import torch


def gmm_params(d, k=8, seed=0, spread=4.0, base_sigma=0.4):
    """Build mixture parameters for a d-dimensional GMM. Returns a params dict."""
    rng = np.random.default_rng(seed)
    prob = rng.uniform(0.2, 1.0, k)
    prob /= prob.sum()
    mu = rng.uniform(-spread, spread, (k, d))
    covs = []
    for _ in range(k):
        a = rng.uniform(-0.5, 0.5, (d, d))
        covs.append(a @ a.T + np.eye(d) * base_sigma)
    return {"d": d, "k": k, "prob": prob, "mu": mu, "cov": np.array(covs)}


def sample_gmm(params, n, seed=0, device="cpu"):
    """Draw n samples from a GMM described by ``params``. Returns a (n, d) tensor."""
    rng = np.random.default_rng(seed)
    d, k = params["d"], params["k"]
    comp = rng.choice(k, size=n, p=params["prob"])
    out = np.empty((n, d))
    for i in range(k):
        m = comp == i
        c = int(m.sum())
        if c:
            out[m] = rng.multivariate_normal(params["mu"][i], params["cov"][i], size=c)
    return torch.tensor(out, dtype=torch.float32, device=device)


def perturb_params(params, kind, eps):
    """Return a copy of ``params`` perturbed by amount ``eps``.

    kind:
      * "mean" -- shift every component mean by +eps in every coordinate.
      * "var"  -- scale every covariance by (1+eps)^2.
      * "drop" -- down-weight component 0 by a factor (1-eps); eps=1 drops it.
    eps=0 returns an identical distribution for every kind.
    """
    p = {
        "d": params["d"], "k": params["k"],
        "prob": params["prob"].copy(),
        "mu": params["mu"].copy(),
        "cov": params["cov"].copy(),
    }
    if kind == "mean":
        p["mu"] = p["mu"] + eps
    elif kind == "var":
        p["cov"] = p["cov"] * (1.0 + eps) ** 2
    elif kind == "drop":
        p["prob"][0] *= (1.0 - eps)
        p["prob"] /= p["prob"].sum()
    else:
        raise ValueError(f"unknown perturbation kind {kind!r}")
    return p
