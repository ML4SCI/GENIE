"""Training loop for the flow-matching velocity field.

Plain Adam on the conditional flow-matching loss. Optionally logs the Tier-3
``pinnde_eval`` monitors (MMD/SWD vs. a held-out real sample) every
``monitor_every`` steps, so convergence can be watched with the same numbers
used to compare the two tracks.
"""

import torch

from ._utils import seed_all
from .core import fm_loss, sample
from .model import VelocityField


def train_flow_matching(data, dim, cond=None, n_steps=4000, batch_size=256,
                        lr=1e-3, hidden=128, depth=3, time_dim=64, device="cpu",
                        seed=0, monitor_every=0, monitor_real=None,
                        monitor_cond=None, monitor_n=2000, sample_steps=50,
                        clip_grad=None):
    """Train a ``VelocityField`` on ``data`` (N, d). Returns (model, history).

    ``clip_grad`` caps the gradient norm of every step. A deep velocity field
    can be knocked into a bad region by one unusually large step and never
    recover: at d=361 the loss jumped from 1.19 to 1.46 mid-training and two
    generated showers later diverged to NaN. Clipping bounds that step.

    ``cond`` (N, c), aligned row-by-row with ``data``, makes the model
    conditional: it learns v_theta(x, t, c) and ``sample`` then needs a
    condition. ``history`` is a list of (step, loss, mmd, swd) rows; the
    mmd/swd columns are filled only when ``monitor_every > 0`` and
    ``monitor_real`` is given (for a conditional model also pass
    ``monitor_cond``, one condition per monitor_real row, so the monitor
    compares matched conditions).
    """
    seed_all(seed)
    if cond is not None:
        cond = torch.as_tensor(cond, dtype=torch.float32, device=device)
        if cond.ndim == 1:
            cond = cond.reshape(-1, 1)
        if len(cond) != len(data):
            raise ValueError("cond must have one row per data row")
    cond_dim = 0 if cond is None else cond.shape[1]
    model = VelocityField(dim, hidden=hidden, depth=depth, time_dim=time_dim,
                          cond_dim=cond_dim, seed=seed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # Cosine decay to ~0: stabilizes the late training so the final samples
    # don't drift back off the data manifold.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
    data = data.to(device)
    g = torch.Generator(device=device).manual_seed(seed)

    history = []
    for step in range(n_steps):
        idx = torch.randint(0, len(data), (batch_size,), generator=g, device=device)
        loss = fm_loss(model, data[idx], cond=None if cond is None else cond[idx],
                       generator=g)
        opt.zero_grad()
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        opt.step()
        sched.step()

        if monitor_every and (step % monitor_every == 0 or step == n_steps - 1):
            row = [step, float(loss.detach()), None, None]
            if monitor_real is not None:
                from pinnde_eval import mmd, swd
                n_mon = min(monitor_n, len(monitor_real))
                mc = None
                if cond is not None:
                    if monitor_cond is None:
                        raise ValueError("conditional model: pass monitor_cond "
                                         "alongside monitor_real")
                    mc = torch.as_tensor(monitor_cond, dtype=torch.float32,
                                         device=device)[:n_mon]
                gen = sample(model, n_mon, dim, cond=mc, steps=sample_steps,
                             device=device, seed=seed)
                real = monitor_real[:n_mon]
                row[2] = mmd(real, gen, seed=seed)
                row[3] = swd(real, gen, seed=seed)
            history.append(tuple(row))

    return model, history
