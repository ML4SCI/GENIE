"""Shared helpers: array conversion and seeding.

Every public metric accepts torch tensors or numpy arrays of shape (N, d) and
converts internally, so the rest of the package can assume a single
representation.
"""

import numpy as np
import torch


def to_numpy(x):
    """Return x as a numpy array (detached, on CPU)."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def to_torch(x, device="cpu", dtype=torch.float32):
    """Return x as a 2D torch tensor of shape (N, d) on the given device."""
    if isinstance(x, torch.Tensor):
        t = x.to(device=device, dtype=dtype)
    else:
        t = torch.as_tensor(np.asarray(x), dtype=dtype, device=device)
    if t.ndim == 1:
        t = t.view(-1, 1)
    return t


def as_2d(x):
    """Coerce a numpy array to shape (N, d); a 1D array becomes (N, 1)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x


def check_pair(real, gen):
    """Convert a (real, gen) pair to 2D numpy arrays and check dimensions match."""
    real = as_2d(to_numpy(real))
    gen = as_2d(to_numpy(gen))
    if real.shape[1] != gen.shape[1]:
        raise ValueError(
            f"real and gen must share feature dimension, got {real.shape[1]} vs {gen.shape[1]}"
        )
    return real, gen


def seed_all(seed):
    """Seed numpy and torch for reproducible metrics."""
    np.random.seed(seed)
    torch.manual_seed(seed)
