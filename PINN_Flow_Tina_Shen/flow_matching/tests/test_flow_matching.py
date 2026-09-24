"""Lightweight edge-case tests for the flow_matching package.

Run from the project folder:  pytest flow_matching/tests -q
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest

from flow_matching import VelocityField, fm_loss, sample, train_flow_matching


# ---------- model + objective shapes ----------

def test_velocity_output_shape():
    model = VelocityField(dim=3, seed=0)
    x = torch.randn(16, 3)
    t = torch.rand(16)
    assert model(x, t).shape == (16, 3)


def test_fm_loss_is_finite_scalar():
    model = VelocityField(dim=2, seed=0)
    g = torch.Generator().manual_seed(0)
    loss = fm_loss(model, torch.randn(32, 2), generator=g)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_sample_shape_and_finite():
    model = VelocityField(dim=2, seed=0)
    g = sample(model, n=50, dim=2, steps=10, seed=0)
    assert g.shape == (50, 2) and torch.isfinite(g).all()


def test_sampling_is_reproducible():
    model = VelocityField(dim=2, seed=0)
    a = sample(model, n=40, dim=2, steps=10, seed=1)
    b = sample(model, n=40, dim=2, steps=10, seed=1)
    assert torch.allclose(a, b)


def test_euler_and_heun_both_run():
    model = VelocityField(dim=2, seed=0)
    e = sample(model, n=30, dim=2, steps=10, method="euler", seed=0)
    h = sample(model, n=30, dim=2, steps=10, method="heun", seed=0)
    assert torch.isfinite(e).all() and torch.isfinite(h).all()


# ---------- it actually learns ----------

def test_training_reduces_loss():
    rng = np.random.default_rng(0)
    data = torch.tensor(rng.normal(size=(4000, 2)) * 0.5 + 2.0, dtype=torch.float32)
    _, history = train_flow_matching(
        data, dim=2, n_steps=600, monitor_every=50, seed=0,
    )
    losses = [loss for _, loss, _, _ in history]
    assert np.mean(losses[-3:]) < np.mean(losses[:3])


def test_gradient_clipping_bounds_the_step_and_still_learns():
    rng = np.random.default_rng(0)
    data = torch.tensor(rng.normal(size=(4000, 2)) * 0.5 + 2.0, dtype=torch.float32)
    _, history = train_flow_matching(
        data, dim=2, n_steps=600, monitor_every=50, seed=0, clip_grad=1.0,
    )
    losses = [loss for _, loss, _, _ in history]
    assert np.mean(losses[-3:]) < np.mean(losses[:3])

    # the cap actually binds: an unclipped step on this data is larger
    from flow_matching.model import VelocityField
    from flow_matching.core import fm_loss
    model = VelocityField(2, seed=0)
    fm_loss(model, data[:256]).backward()
    before = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    after = torch.sqrt(sum(p.grad.pow(2).sum() for p in model.parameters()))
    assert float(before) > 1.0 and float(after) <= 1.0 + 1e-5


def test_training_moves_samples_toward_data():
    # A shifted Gaussian: a trained model should match it far better than the
    # untrained (random-init) model does.
    rng = np.random.default_rng(1)
    real = torch.tensor(rng.normal(size=(3000, 2)) * 0.3 + 3.0, dtype=torch.float32)

    from pinnde_eval import swd
    untrained = VelocityField(dim=2, seed=0)
    swd_before = swd(real, sample(untrained, 3000, 2, steps=50, seed=0))

    model, _ = train_flow_matching(real, dim=2, n_steps=1500, seed=0)
    swd_after = swd(real, sample(model, 3000, 2, steps=50, seed=0))

    assert swd_after < 0.3 * swd_before


# ---------- conditional path ----------

def test_conditional_velocity_shape_and_checks():
    model = VelocityField(dim=3, cond_dim=1, seed=0)
    x = torch.randn(16, 3)
    t = torch.rand(16)
    cond = torch.rand(16, 1)
    assert model(x, t, cond).shape == (16, 3)
    with pytest.raises(ValueError):
        model(x, t)                       # conditional model needs cond
    with pytest.raises(ValueError):
        VelocityField(dim=3, seed=0)(x, t, cond)   # unconditional got cond


def test_fm_loss_and_sample_with_cond():
    model = VelocityField(dim=2, cond_dim=1, seed=0)
    g = torch.Generator().manual_seed(0)
    loss = fm_loss(model, torch.randn(32, 2), cond=torch.rand(32, 1), generator=g)
    assert loss.ndim == 0 and torch.isfinite(loss)

    # scalar condition broadcasts; per-sample (n,) and (n,1) both accepted
    a = sample(model, n=40, dim=2, cond=0.3, steps=8, seed=1)
    b = sample(model, n=40, dim=2, cond=torch.full((40,), 0.3), steps=8, seed=1)
    c = sample(model, n=40, dim=2, cond=torch.full((40, 1), 0.3), steps=8, seed=1)
    assert a.shape == (40, 2)
    assert torch.allclose(a, b) and torch.allclose(a, c)


def test_conditional_training_learns_the_condition():
    # x | c ~ N(4c - 2, 0.3): the generated mean must track the condition.
    rng = np.random.default_rng(2)
    c = rng.uniform(0.0, 1.0, size=(6000, 1))
    x = 4.0 * c - 2.0 + 0.3 * rng.normal(size=(6000, 1))
    model, _ = train_flow_matching(
        torch.tensor(x, dtype=torch.float32), dim=1,
        cond=torch.tensor(c, dtype=torch.float32), n_steps=1500, seed=0,
    )
    lo = sample(model, 2000, 1, cond=0.05, steps=25, seed=0).mean().item()
    hi = sample(model, 2000, 1, cond=0.95, steps=25, seed=0).mean().item()
    assert abs(lo - (-1.8)) < 0.5, lo
    assert abs(hi - 1.8) < 0.5, hi


def test_train_rejects_misaligned_cond():
    data = torch.randn(100, 2)
    with pytest.raises(ValueError):
        train_flow_matching(data, dim=2, cond=torch.rand(50, 1), n_steps=1)


def test_shower_toy_shapes_trends_and_determinism():
    from pinnde_eval.data import sample_shower_toy

    x, c = sample_shower_toy(3000, seed=0)
    assert x.shape == (3000, 3) and c.shape == (3000, 1)
    x2, _ = sample_shower_toy(3000, seed=0)
    assert torch.equal(x, x2)

    lo, _ = sample_shower_toy(4000, cond=0.1, seed=1)
    hi, _ = sample_shower_toy(4000, cond=0.9, seed=2)
    # physics-flavored trends: at high energy the sampling fraction rises,
    # the shower is deeper, narrower, and fluctuates less.
    assert hi[:, 0].mean() > lo[:, 0].mean() + 0.05
    assert hi[:, 1].mean() > lo[:, 1].mean() + 0.5
    assert hi[:, 2].mean() < lo[:, 2].mean() - 0.3
    assert hi[:, 1].std() < lo[:, 1].std()
