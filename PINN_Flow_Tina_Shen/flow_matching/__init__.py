"""flow_matching -- conditional flow-matching generator for the PINNDE flow track.

A simulation-free generative model: learn a velocity field v_theta(x, t) by
regressing it onto the straight-line interpolant target (x1 - x0), then generate
by integrating dx/dt = v_theta(x, t) from noise to data.

    from flow_matching import VelocityField, train_flow_matching, sample
    model, history = train_flow_matching(data, dim=2)
    gen = sample(model, n=5000, dim=2)

Quality is measured with the ``pinnde_eval`` module, so results here are
directly comparable with anything else scored the same way.
"""

from .core import fm_loss, sample
from .model import FourierFeatures, FourierTimeEmbedding, VelocityField
from .train import train_flow_matching

__all__ = [
    "fm_loss",
    "sample",
    "FourierFeatures",
    "FourierTimeEmbedding",
    "VelocityField",
    "train_flow_matching",
]
