"""
flow_de.py -- energy-conditioned reverse-time diffusion vector field.

This ports `qVectorField` / `FlowDE` from Sijil Jose's GSoC 2025 PINNDE
project (ML4SCI/GENIE, Physics_Informed_Neural_Network_Diffusion_Equation_
Sijil_Jose/flow_de/flow_de.py) with a single mathematical generalization:
the reference implementation solves the reverse-time diffusion ODE for an
*unconditional* target distribution (1/2/3-D toy GMMs); calorimeter showers
need the same ODE solved *conditional on incident energy*, since shower
shape depends strongly on E_inc.

Unchanged from the reference formalism
---------------------------------------
Given a stochastic-interpolant path

    x(t) = alpha(t) * x0 + sigma(t) * eps,      alpha(t) = 1 - t,
                                                 sigma(t) = sigma0 + (1-sigma0)*t

(so x(0) = x0 is the data, x(1) = eps is standard Gaussian noise up to the
small floor sigma0), the reverse-time ODE is

    dx/dt = G(t, x) = [(1 - sigma0)*x - q(t, x)] / sigma(t)

where q(t, x) = E[x0 | x(t) = x] is the exact posterior mean, approximated
by Monte Carlo importance weighting over a fixed reference ensemble x0 of
size M (qVectorField below) -- exactly the reference repo's construction.
FlowDE integrates this ODE backwards from t=1 to t=0 with the same
2nd-order Runge-Kutta (midpoint-style) scheme as the reference `FlowDE.
forward`.

Generalization made here: energy conditioning
-----------------------------------------------
Each reference-ensemble shower x0_i carries an incident energy e0_i. The
Gaussian kernel that defines the MC weights in `qVectorField.forward` gets
an extra term in its exponent penalizing energy mismatch between the query
point's conditioning energy and each reference shower's energy, with its
own kernel width `energy_kernel_scale`. This is the standard way to turn a
Parzen-window estimator into a *conditional* one: extend the kernel's
distance metric to include the conditioning variable. The energy channel
is never part of what gets diffused/generated -- only the shower part of
each x0_i is summed into the returned q(t, x), matching the fact that
E_inc is a given input to the sampler, not a random variable being
generated.
"""

import torch
import torch.nn as nn
from tqdm import tqdm


class qVectorField(nn.Module):
    """
    Monte Carlo approximation of the energy-conditioned q vector field,

        dx/dt = [(1-s0)*xt - q(t, xt, E)] / [s0 + (1-s0)*t]

    q(t, xt, E) = E[x0 | x(t)=xt, energy=E] is estimated by importance-
    weighting a fixed reference ensemble (x0, e0) of M showers: the weight
    of reference shower i is proportional to
        exp( -1/2 * [ ||(xt - alpha_t*x0_i)/sigma_t||^2
                      + (energy_kernel_scale * (E - e0_i))^2 ] ),
    i.e. the same Gaussian kernel as the unconditional case, with an extra
    energy-proximity factor. Reference showers whose incident energy is far
    from the query's are down-weighted continuously rather than hard-binned.

    Example
    -------
    q = qVectorField(x0, e0)
    qt = q(t, xt, energy)
    """

    def __init__(self, x0, e0, sigma0=1e-2, energy_kernel_scale=5.0, debug=False):
        super().__init__()
        assert x0.ndim == 2          # (M, d)
        assert e0.ndim == 2 and e0.shape[1] == 1   # (M, 1)
        assert x0.shape[0] == e0.shape[0]

        # (M, d) -> (1, M, d) / (M, 1) -> (1, M, 1) so broadcasting works
        # against a (N, 1, .) query batch.
        self.register_buffer("x0", x0.unsqueeze(0))
        self.register_buffer("e0", e0.unsqueeze(0))
        self.sigma0 = sigma0
        self.energy_kernel_scale = energy_kernel_scale
        self.debug = debug

    def set_debug(self, debug=True):
        self.debug = debug

    def forward(self, t, xt, energy):
        assert xt.ndim == 2          # (N, d)
        assert energy.ndim == 2 and energy.shape[1] == 1   # (N, 1)

        if torch.is_tensor(t) and t.ndim == 2:
            t = t.unsqueeze(1)       # (N,1) -> (N,1,1)

        x0, e0 = self.x0, self.e0
        sigma0 = self.sigma0

        xt = xt.unsqueeze(1)         # (N, d) -> (N, 1, d)
        energy = energy.unsqueeze(1) # (N, 1) -> (N, 1, 1)

        alphat = 1 - t
        sigmat = sigma0 + (1 - sigma0) * t
        vt = (xt - alphat * x0) / sigmat        # (N, M, d)
        if torch.isnan(vt).any():
            raise ValueError("qVectorField: vt contains NaN")

        de = self.energy_kernel_scale * (energy - e0)   # (N, M, 1)

        v2 = (vt * vt).sum(dim=-1)              # (N, M)
        e2 = (de * de).sum(dim=-1)              # (N, M)

        log_weights = -0.5 * (v2 + e2)
        log_weights = log_weights - torch.logsumexp(log_weights, dim=-1, keepdim=True)
        wt = torch.exp(log_weights)             # (N, M), rows sum to 1
        if torch.isnan(wt).any():
            raise ValueError("qVectorField: wt contains NaN")

        qt = (x0 * wt.unsqueeze(-1)).sum(dim=1)  # (N, d) -- shower-space only
        if torch.isnan(qt).any():
            raise ValueError("qVectorField: qt contains NaN")
        return qt

    def G(self, t, xt, energy):
        sigma0 = self.sigma0
        sigmat = sigma0 + (1 - sigma0) * t
        qt = self(t, xt, energy)
        return ((1 - sigma0) * xt - qt) / sigmat


def get_normal_sample(shape, device="cpu"):
    """Standard-normal sample of the given (N, d) shape -- the x(t=1) draws."""
    return torch.randn(shape, device=device)


def get_target_sample(x0_all, e0_all, M):
    """Random subsample of M (shower, energy) pairs from the full dataset,
    used as the fixed reference ensemble inside qVectorField."""
    idx = torch.randint(0, x0_all.shape[0], (M,))
    return x0_all[idx], e0_all[idx]


class FlowDE(nn.Module):
    """
    Numerically solves the energy-conditioned reverse-time ODE with the
    same 2nd-order Runge-Kutta scheme as the reference `FlowDE.forward`.
    Used only to generate a modest number of *exact* anchor trajectories
    for the PINN's supervised data loss (see generate_anchor_trajectories.
    py) -- this is the expensive, non-amortized solver; the whole point of
    training the PINN is to replace it with one network forward pass.
    """

    def __init__(self, x0, e0, sigma0=1e-2, T=250, savepath=False, debug=False):
        super().__init__()
        assert x0.ndim == 2
        self.q = qVectorField(x0, e0, sigma0, debug=debug)
        self.T = max(T, 4)
        self.h = 1.0 / self.T
        self.savepath = savepath
        self.debug = debug

    def G(self, t, xt, energy):
        return self.q.G(t, xt, energy)

    def forward(self, z, energy):
        assert z.ndim == 2
        T, h = self.T, self.h
        t = 1.0
        xt = z

        G1 = self.G(t, xt, energy)
        times, path = [], []
        for _ in tqdm(range(T), desc="FlowDE RK2 solve", disable=not self.debug):
            t -= h
            if t < 0:
                break
            G2 = self.G(t, xt - G1 * h, energy)
            xt = xt - (G1 + G2) * h / 2
            G1 = G2.detach().clone()
            if self.savepath:
                path.append(xt.detach().cpu())
                times.append(t)

        if self.savepath:
            return path, times
        return xt
