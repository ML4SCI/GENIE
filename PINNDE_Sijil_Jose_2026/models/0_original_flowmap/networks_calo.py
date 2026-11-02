"""
networks_calo.py -- the PINN itself: PhysicsNN, the hard-constraint ansatz,
the ODE residual, and training loops.

Direct generalization of the reference repo's networks_1d.py / networks_3d.py
to (a) 6480 shower dimensions instead of 1-3, and (b) energy conditioning,
which the toy GMM cases don't need. Every piece of the PINN formalism itself
-- the ansatz that hard-enforces the t=1 boundary condition, the autograd-
based ODE residual, the t-weighted residual, the two-term loss, and the
ramp-step training curriculum -- is unchanged in structure from the
reference; only tensor shapes, network width, and the extra energy argument
threaded through every call are new.
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from . import flow_de


class PhysicsNN(nn.Module):
    """
    Same hard-constraint construction as the reference PhysicsNN: `net`
    itself is NOT the trajectory -- x_function() below combines it with z
    so that x(t=1, z, E) = z exactly, by construction, for any weights.

    Widened relative to the reference's 3-hidden-layer, 32-unit MLP because
    the input/output here is 6480-dimensional rather than 1-3-dimensional;
    a network that size would have nowhere near enough capacity to
    represent a calorimeter shower. Still a plain MLP, matching the
    reference's architecture choice (a PINN's collocation-based training
    doesn't need a convolutional inductive bias the way the autoencoder
    does -- every "pixel" is just another output coordinate here).
    """

    def __init__(self, x0, e0, shower_dim=6480, hidden=512, sigma0=1e-2,
                 energy_kernel_scale=5.0, device="cpu"):
        super().__init__()
        self.q = flow_de.qVectorField(x0, e0, sigma0=sigma0,
                                       energy_kernel_scale=energy_kernel_scale).to(device)
        self.shower_dim = shower_dim

        in_dim = 1 + shower_dim + 1   # t, z, energy
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, shower_dim),
        )
        # x_function multiplies this network's raw output by (1-t), so at
        # t far from 1 there's nothing else keeping x(t,z,E) close to z --
        # a generic Linear init gives net(t,z,E) an arbitrary, uncontrolled
        # magnitude, which showed up directly as a ~1000x inflation of the
        # physics-residual loss/gradient once the cancellation-prone
        # subtractive ansatz was replaced. Near-zero-init the final layer so
        # the network starts as a small perturbation for every t, not just
        # t=1 -- standard practice for a residual/correction term like this.
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, t: torch.Tensor, z: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
        x = torch.cat([t, z, energy], dim=1)
        return self.net(x)


def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def x_function(model: PhysicsNN, t: torch.Tensor, z: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
    """
    x(t, z, E) = z + (1-t) * net(t, z, E) -- hard-constrains x(1,z,E)=z via a
    multiplicative factor that vanishes at t=1, not a subtracted second
    network evaluation.

    Was z + net(t,z,E) - net(1,z,E) (the reference repo's original form).
    That works in exact arithmetic, but net(t,·) and net(1,·) are the SAME
    shared-weight network evaluated at nearby inputs -- at/near initialization
    (and evidently well into training, per direct reports of MSE_u not
    moving) their outputs are highly correlated, so the gradient of their
    DIFFERENCE partially cancels. Measured directly: MSE_u's gradient norm
    was ~2800x smaller than MSE_f's on a random anchor batch, even before
    any training. Multiplying by (1-t) instead of subtracting a second
    evaluation hard-constrains the same boundary condition without ever
    computing two correlated evaluations to begin with -- one forward pass
    instead of two (cheaper), and the gradient at time t is just
    (1-t) * d(net(t,z,E))/d(weights), with no cancellation term at all.
    """
    out = model(t, z, energy)
    if not torch.isfinite(out).all():
        raise ValueError("x_function: model output is NaN/Inf")
    return z + (1.0 - t) * out


def derivative(model: PhysicsNN, t_f: torch.Tensor, z: torch.Tensor, energy: torch.Tensor,
                order: int = 1) -> torch.Tensor:
    """dx/dt via autograd through x_function -- the "physics-informed" part."""
    t_f = t_f.clone().requires_grad_(True)
    y = x_function(model, t_f, z, energy)
    for _ in range(order):
        y = torch.autograd.grad(
            y, t_f, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True
        )[0]
    return y


def f_residual(model: PhysicsNN, t_f: torch.Tensor, z: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
    """
    ODE residual at collocation points: dx/dt = G(t,x,E), rearranged and
    scaled by t (matching the reference repo's f = t*dxdt - t*q_val exactly
    -- the t-weighting softens the residual near t=0, where sigma(t) -> sigma0
    and the vector field is most sensitive).
    """
    x = x_function(model, t_f, z, energy)
    dxdt = derivative(model, t_f, z, energy, order=1)
    q_val = model.q.G(t_f, x, energy)
    if not (torch.isfinite(dxdt).all() and torch.isfinite(q_val).all()):
        raise ValueError("f_residual: NaN/Inf in dx/dt or q.G")
    return t_f * dxdt - t_f * q_val


def loss_function(model, t_f, z_f, e_f, t_u, z_u, e_u, x_u,
                   physics_weight=10.0, data_weight=1.0):
    """
    data_weight * MSE_u (fit to exact anchor trajectories)
      + physics_weight * MSE_f (ODE residual on the collocation grid).

    The reference repo's version has MSE_u at an implicit, hardcoded weight
    of 1.0. data_weight is exposed here so the data term can be turned OFF
    entirely (data_weight=0) for pure-PINN training. That mode is
    well-posed on its own: the ODE plus the t=1 boundary condition -- which
    x_function hard-enforces exactly, for any weights -- already determines
    the solution uniquely. The anchor data is a convenience, not a
    requirement, and at 6480 dimensions it is a poor one: an anchor set of
    N trajectories supplies only N distinct z vectors, mutually ~sqrt(2d)
    apart, while the physics term samples fresh z every step. Those isolated
    points are measure-zero against the collocation sampling and lose the
    tug-of-war regardless of weighting.

    When data_weight == 0, MSE_u is still computed (under no_grad) so it
    stays visible as a diagnostic -- seeing whether physics-only training
    incidentally improves the data fit is exactly the interesting question.
    """
    mse_f = f_residual(model, t_f, z_f, e_f).pow(2).mean()

    if data_weight == 0.0:
        with torch.no_grad():
            mse_u = (x_function(model, t_u, z_u, e_u) - x_u).pow(2).mean()
        return physics_weight * mse_f, mse_u.item(), mse_f.item()

    mse_u = (x_function(model, t_u, z_u, e_u) - x_u).pow(2).mean()
    return (data_weight * mse_u + physics_weight * mse_f,
            mse_u.item(), mse_f.item())


def _ramp_schedule(total, ramp_steps, initial_frac, lr_target, lr_warmup_factor):
    sizes = [int(initial_frac * total + (i / max(ramp_steps - 1, 1)) * (total - initial_frac * total))
             for i in range(ramp_steps)]
    lrs = [(lr_target + 1e-5) - lr_target * (lr_warmup_factor +
           (i / max(ramp_steps - 1, 1)) * (1 - lr_warmup_factor))
           for i in range(ramp_steps)]
    return sizes, lrs


def train_adam(model: PhysicsNN, anchors, n_colloc=20000, n_epochs=200,
               ramp_steps=4, initial_frac=0.25, lr_target=1e-3, lr_warmup_factor=0.3,
               colloc_batch=2048, physics_weight=10.0, data_weight=1.0,
               device="cpu", checkpoint_path=None, log_every=10):
    """
    Adam-based training loop -- the practical alternative to the reference
    repo's LBFGS closure-based `train()`, which assumes it's cheap to do a
    full-batch forward/backward inside a single `optimizer.step(closure)`
    call. At 6480 dimensions and a physics collocation grid of thousands of
    points, that assumption doesn't hold on modest hardware; the reference
    repo's own train_adam() variant is the better starting point here, kept
    with the same ramp-step curriculum (train on progressively larger
    slices of anchor data at a cooling learning rate).

    anchors: dict with t_u, z_u, e_u, x_u tensors (see
             generate_anchor_trajectories.py), already flattened to
             (n_anchor_points, ...).
    """
    t_u = anchors["t_u"].to(device)
    z_u = anchors["z_u"].to(device)
    e_u = anchors["e_u"].to(device)
    x_u = anchors["x_u"].to(device)
    n_anchor = t_u.shape[0]

    sizes, lrs = _ramp_schedule(n_anchor, ramp_steps, initial_frac, lr_target, lr_warmup_factor)
    history = []

    for stage, (size, lr) in enumerate(zip(sizes, lrs)):
        print(f"\nRamp stage {stage+1}/{ramp_steps} -- anchor subset size {size}, lr {lr:.2e}")
        perm = torch.randperm(n_anchor)[:size]
        t_u_s, z_u_s, e_u_s, x_u_s = t_u[perm], z_u[perm], e_u[perm], x_u[perm]

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(n_epochs):
            # Fresh collocation points every epoch: random t in [0,1], random
            # z ~ N(0,I) (the reverse process's t=1 state), and energies drawn
            # from the same conditioning distribution as the anchors so the
            # physics residual gets enforced across the energies we actually
            # care about generating.
            t_f = torch.rand(n_colloc, 1, device=device)
            z_f = torch.randn(n_colloc, model.shower_dim, device=device)
            e_idx = torch.randint(0, n_anchor, (n_colloc,))
            e_f = e_u[e_idx]

            optimizer.zero_grad()
            loss, mse_u, mse_f = loss_function(
                model, t_f, z_f, e_f, t_u_s, z_u_s, e_u_s, x_u_s,
                physics_weight, data_weight
            )
            loss.backward()
            optimizer.step()

            if epoch % log_every == 0:
                print(f"  epoch {epoch:4d}  loss={loss.item():.5f}  "
                      f"MSE_u={mse_u:.5f}  MSE_f={mse_f:.5f}")
                history.append((stage, epoch, loss.item(), mse_u, mse_f))

        if checkpoint_path:
            torch.save({"model_state": model.state_dict(),
                        "shower_dim": model.shower_dim}, checkpoint_path)
            print(f"  Saved checkpoint -> {checkpoint_path}")

    return model, history


def train_lbfgs(model: PhysicsNN, anchors, n_colloc=4096, ramp_steps=4,
                initial_frac=0.25, lr_target=1.0, lr_warmup_factor=0.1,
                physics_weight=10.0, data_weight=1.0, device="cpu",
                checkpoint_path=None, max_iter=200):
    """
    LBFGS variant, matching the reference repo's primary `train()`. Kept
    for faithfulness to the original formalism, but max_iter is far lower
    than the reference's 10000 -- a full-batch closure over 6480-dim
    collocation points is much more expensive per step than the reference's
    1-3 dimensional case, so this is realistically a fine-tuning pass after
    train_adam, not the primary optimizer at this scale.
    """
    t_u = anchors["t_u"].to(device)
    z_u = anchors["z_u"].to(device)
    e_u = anchors["e_u"].to(device)
    x_u = anchors["x_u"].to(device)
    n_anchor = t_u.shape[0]

    sizes, lrs = _ramp_schedule(n_anchor, ramp_steps, initial_frac, lr_target, lr_warmup_factor)

    for stage, (size, lr) in enumerate(zip(sizes, lrs)):
        print(f"\n[LBFGS] Ramp stage {stage+1}/{ramp_steps} -- anchor subset {size}, lr {lr:.4f}")
        perm = torch.randperm(n_anchor)[:size]
        t_u_s, z_u_s, e_u_s, x_u_s = t_u[perm], z_u[perm], e_u[perm], x_u[perm]

        t_f = torch.rand(n_colloc, 1, device=device)
        z_f = torch.randn(n_colloc, model.shower_dim, device=device)
        e_f = e_u[torch.randint(0, n_anchor, (n_colloc,))]

        optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=max_iter,
                                       history_size=100, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            loss, mse_u, mse_f = loss_function(
                model, t_f, z_f, e_f, t_u_s, z_u_s, e_u_s, x_u_s,
                physics_weight, data_weight
            )
            loss.backward()
            return loss

        optimizer.step(closure)
        if checkpoint_path:
            torch.save({"model_state": model.state_dict(),
                        "shower_dim": model.shower_dim}, checkpoint_path)

    return model
