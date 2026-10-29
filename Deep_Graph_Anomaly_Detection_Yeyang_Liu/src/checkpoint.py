"""Load training checkpoints and save complete epochs atomically."""

from pathlib import Path

import torch


def load_checkpoint(path):
    """Load the current checkpoint format onto CPU."""
    return torch.load(path, map_location='cpu', weights_only=True)


def save_checkpoint(path, checkpoint):
    """Keep last.pt intact until the next complete epoch has been written."""
    path = Path(path)
    temporary = path.with_suffix('.pt.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)
