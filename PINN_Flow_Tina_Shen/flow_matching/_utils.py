"""Tiny shared helper for the flow_matching package."""

import numpy as np
import torch


def seed_all(seed):
    """Seed numpy and torch for reproducible training/sampling."""
    np.random.seed(seed)
    torch.manual_seed(seed)
