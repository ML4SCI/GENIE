"""
velocity_net.py -- the parameterization that actually scales to 6480 dimensions.

Why this exists (measured, not assumed)
----------------------------------------
The reference repo's PINNDE ansatz has the network learn the TRAJECTORY as a
function of its initial noise, x(t; z) -- the full flow map. At d=1-3 that is
fine. At d=6480 it is not, and the failure is quantitative:

  * Held-out generalization to unseen z barely improves with more data:
    512 -> 4096 anchor trajectories (8x) cut held-out MSE only 1.84x, i.e.
    MSE ~ N^-0.29. Reaching a usable error would need ~10^6x more data.
  * The physics residual cannot train from cold start: it is evaluated at the
    model's own prediction, which early in training is off-manifold, where the
    q-kernel degenerates and G ~ x/sigma_t becomes radially repulsive. MSE_f
    then rises during training (measured: 6862 -> 22002) -- it cannot even
    minimize its own objective.
  * Consequence: samples came out worse than noise (E_dep/E_inc ~ 3.4 against
    a true 0.78).

This module keeps the formalism -- same stochastic interpolant, same
reverse-time ODE, same Monte-Carlo q-function from flow_de -- and changes only
WHAT the network parameterizes: the local field q(t, x, E) = E[x0 | x_t],
rather than the global flow map. That is the same choice standard diffusion
models make, and it fixes the three problems above at once:

  * q is a LOCAL function of the current state, smooth and low-complexity, so
    it generalizes instead of memorizing.
  * Training data is unlimited and always on-manifold: take any real x0, any
    t, any eps, form x_t = alpha_t*x0 + sigma_t*eps, and the MC q-field gives
    an exact target. No trajectory generation, no anchor set, no collocation
    points off the manifold.
  * q is bounded (it is a convex combination of real showers, ||q|| ~ 68),
    unlike G which ranges 90-600 across t. Predicting q and recovering
    G = [(1-sigma0)*x - q]/sigma_t analytically is far better conditioned --
    the same reason diffusion models predict x0/eps rather than the raw score.

Measured on a 3.6GB GPU, n_ref=128, 6000 steps: E_dep/E_inc = 0.92 against a
true 0.78, versus 3.4 for the trajectory ansatz.

Sampling still integrates the reverse-time ODE, but with the learned field
instead of the O(M) Monte-Carlo sum -- which is the speedup the whole project
is after.
"""

import copy
import math
import os
import shutil

import torch
import torch.nn as nn

from . import flow_de


# A conv trunk holds ~16x the activations per sample that the MLP does: at
# batch 2048 one intermediate is 2048 x 32 x 6480 x 4 B = 1.7 GB, and several
# are live at once, which OOMs even a 32 GB card. The safe batch therefore
# depends on the card, so it is derived from free memory rather than fixed --
# a hard 256 (chosen on a 4 GB card) makes sampling needlessly slow on a large
# one. An explicitly requested batch is always honoured.
CONV_SAMPLE_BATCH = 256          # fallback when free memory cannot be queried


def max_sample_batch(model, requested, explicit=False, device=None):
    """Largest safe per-call sampling batch for this model.

    explicit=True means the caller asked for this batch size on purpose, so it
    is returned unchanged (and may OOM -- that is the caller's choice).
    """
    if getattr(model, "cond", "") != "conv" or explicit:
        return requested
    try:
        dev = device if device is not None else next(model.parameters()).device
        if dev.type != "cuda":
            return min(requested, CONV_SAMPLE_BATCH)
        free, _ = torch.cuda.mem_get_info(dev)
        # ~24 live intermediates of (B, c1, 6480) float32 through the U-Net
        c1 = model.enc0.conv1.out_channels
        per_sample = 24 * c1 * model.shower_dim * 4
        cap = int(0.5 * free / max(per_sample, 1))
        return max(32, min(requested, cap))
    except Exception:
        return min(requested, CONV_SAMPLE_BATCH)


def _pad_phi_circular(x):
    """phi (dim -2) wraps around the calorimeter; z and r do not."""
    return nn.functional.pad(x, (0, 0, 1, 1, 0, 0), mode="circular")


class _ConvBlock3D(nn.Module):
    """Conv -> GroupNorm -> FiLM -> SiLU, twice, with a residual shortcut.
    Padding is circular in phi and zero in z and r."""

    def __init__(self, in_ch, out_ch, stride=1, cond_dim=128):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=(1, 0, 1))
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=(1, 0, 1))
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        # per-channel scale and shift from (t, E); zero-init so the block starts
        # as the identity modulation and learns the conditioning
        self.film = nn.Linear(cond_dim, 2 * out_ch)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.skip = (nn.Conv3d(in_ch, out_ch, 1, stride=stride)
                     if (in_ch != out_ch or stride != 1) else nn.Identity())

    def forward(self, x, c):
        h = self.norm1(self.conv1(_pad_phi_circular(x)))
        scale, shift = self.film(c).chunk(2, dim=1)
        h = nn.functional.silu(h * (1 + scale[..., None, None, None]) + shift[..., None, None, None])
        h = self.norm2(self.conv2(_pad_phi_circular(h)))
        return nn.functional.silu(h + self.skip(x))


class VelocityNet(nn.Module):
    """Predicts q(t, x, E) = E[x0 | x_t]; G is recovered analytically."""

    def __init__(self, shower_dim=6480, hidden=1024, sigma0=1e-2, precond=False,
                 hit_head=True, empty_value=0.0, cond="concat", n_freq=16,
                 lit_head=False, cond_extra=0, pos_channels=True):
        super().__init__()
        self.shower_dim = shower_dim
        self.hidden = hidden
        self.sigma0 = sigma0
        self.precond = precond
        # cond="concat": t and E are two extra inputs alongside the 6480 voxels.
        # cond="film":   sinusoidal features of (t, log E) -> MLP -> per-layer
        #   scale and shift applied to every hidden activation (FiLM).
        # Why film exists: with concat, the overall mean E_dep/E_inc comes out
        # right (0.79 vs a real 0.78) while the per-energy response does not
        # (1-10 GeV 15% high and over-dispersed, 10-100 GeV 8% low, 100+ GeV
        # 3% low). Getting the average right but the dependence wrong is what
        # weak conditioning looks like, and two scalars competing with 6480
        # inputs at the first layer is a weak way to condition.
        self.cond = cond
        self.n_freq = n_freq
        # cond_extra: extra conditioning numbers carried in the `energy`
        # argument beyond the incident energy itself -- the two-stage shape
        # model passes the 45 layer energies this way, so every existing call
        # signature (sample, train, loss) is unchanged.
        self.cond_extra = cond_extra
        # hit_head: a second block of shower_dim outputs holding a per-voxel
        # "is this voxel lit in x0" logit. It does NOT enter the ODE -- G is
        # built from q alone -- it only gates the final read-out at t=0, where
        # the sample is handed back as a shower.
        #
        # Why: the q-regression is a conditional mean, so a voxel that is
        # uncertain between empty and lit comes out at an intermediate value,
        # which in any continuous representation is a small spurious deposit.
        # Measured consequence: 77% of generated voxels landed above the
        # 15.15 keV threshold against 25% for real showers, carrying the
        # excess energy with them. Thresholding the output cannot fix it
        # without also destroying real small deposits (measured). This is the
        # same defect, with the same fix, as the autoencoder's hit head.
        self.hit_head = hit_head
        # value an empty voxel takes in the (standardized) representation;
        # gated voxels are set to exactly this, so they invert to exactly 0 MeV.
        self.register_buffer("empty_value", torch.tensor(float(empty_value)))
        # lit_head: a THIRD block holding E[value | voxel is lit], trained only
        # on voxels that really are lit. The ODE still runs on q, which must stay
        # trained everywhere; this block exists purely for the read-out.
        # Why: q is a conditional mean over both outcomes, so on a truly-lit
        # voxel it carries only p(lit) of the energy -- measured at 0.85 on real
        # showers, which is most of the sampler's energy deficit. The labels say
        # which voxels are lit, so that expectation can be learned directly.
        self.lit_head = lit_head and hit_head
        n_blocks = 1 + (1 if hit_head else 0) + (1 if self.lit_head else 0)
        out_dim = shower_dim * n_blocks
        if cond == "concat":
            self.net = nn.Sequential(
                nn.Linear(1 + shower_dim + 1 + cond_extra, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, out_dim),
            )
        elif cond == "conv":
            # Geometry-aware trunk: a 3D U-Net over (z, phi, r) with FiLM
            # conditioning, replacing the flat MLP entirely.
            #
            # Why: a width sweep at fixed budget (512 -> 1024 -> 2048 hidden,
            # 10.9M -> 49.8M parameters) left the per-voxel defect completely
            # flat -- dim-voxel fraction 0.0050 / 0.0065 / 0.0064 against a real
            # 0.072, and 5th-percentile voxel energy 3.55 / 3.36 / 3.42 MeV
            # against a real 0.63. Profile errors improved from 512 to 1024 and
            # then stopped. So the flat MLP's limit is its lack of spatial
            # structure, not its size: it has no notion of a voxel's
            # neighbourhood, which is exactly what the dim edge voxels need.
            # All three leading CaloChallenge submissions use geometry-aware
            # convolutions.
            #
            # phi is periodic, so it is padded circularly; z and r are padded
            # with zeros (the calorimeter ends there).
            self.geom = (45, 16, 9)
            assert shower_dim == self.geom[0] * self.geom[1] * self.geom[2], (
                f"conv trunk expects a {self.geom} shower, got shower_dim={shower_dim}")
            self.cond_mlp = nn.Sequential(
                nn.Linear(4 * n_freq + 2 + cond_extra, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU())
            base = max(8, hidden // 32)        # hidden reused as a width knob
            c1, c2, c3 = base, base * 2, base * 4
            # Position channels: the normalized layer index and radial index,
            # concatenated to every input. The convolution is translation
            # invariant by construction, but a calorimeter is not -- depth and
            # radius both matter, and the dim voxels we under-produce live at
            # the edges. CaloDiffusion (arXiv:2308.03876) feeds exactly these.
            #
            # pos_channels=False rebuilds the 1-input-channel trunk used before
            # these were added, so conv checkpoints trained earlier still load.
            # sample_velocity.load_model reads the flag from the checkpoint and
            # defaults to False, which is what an older file implies.
            self.pos_channels = pos_channels
            if pos_channels:
                zz = torch.linspace(0, 1, self.geom[0]).view(1, 1, -1, 1, 1)
                rr = torch.linspace(0, 1, self.geom[2]).view(1, 1, 1, 1, -1)
                self.register_buffer("pos_z", zz.expand(1, 1, *self.geom).clone())
                self.register_buffer("pos_r", rr.expand(1, 1, *self.geom).clone())
            self.enc0 = _ConvBlock3D(3 if pos_channels else 1, c1)
            self.enc1 = _ConvBlock3D(c1, c2, stride=2)
            self.enc2 = _ConvBlock3D(c2, c3, stride=2)
            self.mid = _ConvBlock3D(c3, c3)
            self.dec1 = _ConvBlock3D(c3 + c2, c2)
            self.dec0 = _ConvBlock3D(c2 + c1, c1)
            self.out_conv = nn.Conv3d(c1, n_blocks, 3, padding=(1, 0, 1))
            nn.init.zeros_(self.out_conv.weight)
            nn.init.zeros_(self.out_conv.bias)
            self.n_blocks = n_blocks
            self.register_buffer(
                "freqs", 2.0 ** torch.arange(n_freq, dtype=torch.float32))
        elif cond == "film":
            # same depth/width as concat, so an A/B isolates the conditioning
            self.layers = nn.ModuleList([
                nn.Linear(shower_dim, hidden),
                nn.Linear(hidden, hidden),
                nn.Linear(hidden, hidden),
            ])
            self.out = nn.Linear(hidden, out_dim)
            self.cond_mlp = nn.Sequential(
                nn.Linear(4 * n_freq + 2 + cond_extra, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU())
            # one (scale, shift) pair per hidden layer; zero-init so the model
            # starts at scale 1 / shift 0 and learns the modulation
            self.film = nn.ModuleList([nn.Linear(128, 2 * hidden) for _ in self.layers])
            for f in self.film:
                nn.init.zeros_(f.weight)
                nn.init.zeros_(f.bias)
            self.register_buffer(
                "freqs", 2.0 ** torch.arange(n_freq, dtype=torch.float32))
        else:
            raise ValueError(f"unknown cond {cond!r}")

    def _cond_features(self, t, energy):
        """Sinusoidal features of t and the (already standardized) log-energy,
        plus any extra conditioning columns carried in `energy`."""
        e, extra = energy[:, :1], energy[:, 1:]
        a = t * self.freqs * math.pi
        b = e * self.freqs * math.pi
        return torch.cat([t, e, a.sin(), a.cos(), b.sin(), b.cos(), extra], dim=1)

    def _conv_trunk(self, t, x, energy):
        """3D U-Net over (z, phi, r). Returns (B, n_blocks * shower_dim)."""
        c = self.cond_mlp(self._cond_features(t, energy))
        Nz, Np, Nr = self.geom
        v = x.reshape(-1, 1, Nz, Np, Nr)
        if self.pos_channels:
            n = v.shape[0]
            v = torch.cat([v, self.pos_z.expand(n, -1, -1, -1, -1),
                           self.pos_r.expand(n, -1, -1, -1, -1)], dim=1)
        h0 = self.enc0(v, c)                                   # (B, c1, 45, 16, 9)
        h1 = self.enc1(h0, c)                                   # (B, c2, 23, 8, 5)
        h2 = self.mid(self.enc2(h1, c), c)                      # (B, c3, 12, 4, 3)
        u1 = nn.functional.interpolate(h2, size=h1.shape[2:], mode="nearest")
        u1 = self.dec1(torch.cat([u1, h1], dim=1), c)
        u0 = nn.functional.interpolate(u1, size=h0.shape[2:], mode="nearest")
        u0 = self.dec0(torch.cat([u0, h0], dim=1), c)
        out = self.out_conv(_pad_phi_circular(u0))              # (B, n_blocks, 45, 16, 9)
        return out.reshape(out.shape[0], self.n_blocks * self.shower_dim)

    def _trunk(self, t, x, energy):
        """FiLM trunk: x through the layers, each activation modulated by
        (scale, shift) derived from (t, E)."""
        c = self.cond_mlp(self._cond_features(t, energy))
        h = x
        for layer, film in zip(self.layers, self.film):
            h = layer(h)
            scale, shift = film(c).chunk(2, dim=1)
            h = nn.functional.silu(h * (1 + scale) + shift)
        return self.out(h)

    def coeffs(self, t):
        """
        EDM preconditioning (Karras et al. 2022) for x_t = a*x0 + s*eps with
        a = 1-t, s = sigma0 + (1-sigma0)*t and unit data std (x0 is
        standardized):  q_hat = c_skip*x_t + c_out*F(c_in*x_t, t, E).

        Why: without it the MLP must pass the identity through its
        1024-wide hidden layers -- at t~0 the ideal output is ~the input
        itself -- and it cannot. Measured: the plain net recovers only
        64-79% of the RMS structure of real showers at t=0.
        """
        a, s = 1 - t, self.sigma0 + (1 - self.sigma0) * t
        D2 = s * s + a * a
        return a / D2, s / D2.sqrt(), 1.0 / D2.sqrt()

    def _raw(self, t, x, energy):
        """Network output, split into (value block, hit-logit block)."""
        xin = (self.coeffs(t)[2] * x) if self.precond else x
        if self.cond == "conv":
            out = self._conv_trunk(t, xin, energy)
        elif self.cond == "film":
            out = self._trunk(t, xin, energy)
        else:
            out = self.net(torch.cat([t, xin, energy], dim=1))
        if not self.hit_head:
            return out, None, None
        d = self.shower_dim
        if not self.lit_head:
            return out[:, :d], out[:, d:2 * d], None
        return out[:, :d], out[:, d:2 * d], out[:, 2 * d:3 * d]

    def q_hat(self, t, x, energy):
        v = self._raw(t, x, energy)[0]
        if not self.precond:
            return v
        c_skip, c_out, _ = self.coeffs(t)
        return c_skip * x + c_out * v

    def hit_logit(self, t, x, energy):
        if not self.hit_head:
            raise ValueError("this model was built with hit_head=False")
        return self._raw(t, x, energy)[1]

    def lit_value(self, t, x, energy):
        """E[value | lit], from the third head (lit_head=True only)."""
        u = self._raw(t, x, energy)[2]
        if u is None:
            raise ValueError("this model was built with lit_head=False")
        return u

    def denoise(self, t, x, energy, gate="topk", value_correct=False, p_floor=0.1):
        """
        Final read-out: the posterior-mean shower, with voxels the hit head
        calls empty set to exactly empty_value. Used at t=0 by sample().

        gate="topk" (default): keep the k most probable voxels of each shower,
            with k = round(sum of the head's own probabilities) -- its expected
            hit count. Matches the hit count like "sample" while keeping the
            spatial coherence of "threshold".
        gate="threshold": keep where sigmoid(h) > 0.5, i.e. the most likely
            occupancy per voxel. Spatially coherent but biased sparse --
            it drops every voxel the model is genuinely unsure about.
        gate="sample": keep a voxel with probability sigmoid(h), independently.
            Statistically right per voxel, but real occupancy is spatially
            CORRELATED, so independent draws add salt-and-pepper noise.

        Measured on the 16k checkpoint (real: ~1520 hits/shower, radial
        profile error and NN distance vs a real-to-real ~120):

            gate         hits/shower   radial err   NN distance
            threshold       1359          0.321         138
            sample          1502          0.283         161
            topk            1543          0.300         151

        topk also matched the mean E_dep/E_inc best (0.780 against a real
        0.784, vs 0.718 and 0.711), which is why it is the default. No option
        wins everywhere: threshold stays closest to the data manifold.
        """
        q = self.q_hat(t, x, energy)
        if not self.hit_head:
            return q
        h = self.hit_logit(t, x, energy)
        if value_correct:
            # q is a conditional MEAN over both outcomes of the voxel being lit:
            #     q - empty = p(lit) * ( E[value | lit] - empty )
            # so a voxel we are about to declare lit should carry q / p, not q.
            # Without this the kept voxels are diluted by the probability of
            # being empty: measured on real showers, the field recovers only
            # 0.79 of the energy in truly-lit voxels while inventing 0.26 in
            # truly-empty ones, and the aggregate only looks right because the
            # two errors partly cancel. Exact for the near-linear 'fraction'
            # representation, where energy is affine in the standardized value.
            #
            # OFF BY DEFAULT: it measured WORSE on the one model it has been
            # tried on (16k concat, topk gate) -- E_dep/E_inc 0.780 -> 0.963
            # against a real 0.784, longitudinal error 0.139 -> 0.257, radial
            # 0.302 -> 0.379. The reasoning above is sound but incomplete: that
            # model's aggregate was right only through cancellation, so
            # undiluting the kept voxels also inflates the spurious energy that
            # survived the gate, and never recovers the true energy the gate
            # discarded. It is also too aggressive for voxels topk keeps on
            # marginal confidence (p = 0.3 scales a voxel by 3.3). Worth
            # retrying only on a model whose --check-denoiser decomposition
            # shows a genuine deficit on truly-lit voxels.
            p = torch.sigmoid(h.float()).clamp(min=p_floor)
            q = self.empty_value + (q - self.empty_value) / p
        lit = self.gate_mask(h, gate)
        return torch.where(lit, q, self.empty_value.expand_as(q))

    def gate_mask(self, h, gate="topk"):
        """Which voxels the hit head declares lit. See denoise() for the
        measured comparison of the three rules."""
        if gate == "threshold":
            lit = h > 0
        elif gate == "sample":
            lit = torch.rand_like(h) < torch.sigmoid(h)
        elif gate == "gumbel":
            # Sampling WITHOUT replacement proportional to p (Gumbel top-k):
            # exactly k voxels like "topk", chosen stochastically like "sample".
            # Why it exists: "topk" ranks by probability, and probability tracks
            # brightness, so it systematically drops the DIM voxels that real
            # showers carry at the shower edges -- measured 0.6% of hits below
            # 1 MeV against a real 7.2%. "sample" restores them but scatters
            # them independently, which costs manifold distance (NN 1.36 vs
            # 1.26 real-to-real).
            p = torch.nan_to_num(torch.sigmoid(h.float()), nan=0.0, posinf=1.0, neginf=0.0)
            k = p.sum(dim=1).round().long().clamp(min=1, max=p.shape[1])
            # Parenthesize the inner negation: -torch.log(u).clamp_min(eps)
            # binds as -(log(u).clamp_min(eps)), and log(u) < 0, so the clamp
            # flattens it to eps and the outer log of a negative value is NaN
            # -- which silently gated every voxel off (NaN compares False).
            u = torch.rand_like(p).clamp_min(1e-20)
            g = -torch.log((-torch.log(u)).clamp_min(1e-20))
            key = torch.log(p.clamp_min(1e-20)) + g
            srt = key.sort(dim=1, descending=True).values
            cut = srt.gather(1, (k - 1).unsqueeze(1))
            lit = (key >= cut) & (p > 0)
        elif gate == "topk":
            p = torch.sigmoid(h.float())
            # A diverged shower (see export.diverged_mask) can leave h
            # non-finite. NaN survives clamp() and NaN.long() is undefined, so
            # the row count below became an out-of-bounds gather index and
            # killed the whole sampling pass with a CUDA device-side assert.
            # Treat non-finite as "not lit" and keep going: the divergence is
            # then caught and re-sampled downstream, where it belongs.
            p = torch.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)
            k = p.sum(dim=1).round().long().clamp(min=1, max=p.shape[1])
            srt = p.sort(dim=1, descending=True).values
            # the k-th largest probability of each row is the cut
            cut = srt.gather(1, (k - 1).unsqueeze(1))
            # p > 0 excludes the zeroed-out non-finite case, where cut would
            # also be 0 and every voxel would come out "lit"
            lit = (p >= cut) & (p > 0)
        else:
            raise ValueError(f"unknown gate {gate!r}")
        return lit

    def loss(self, t, xt, energy, y, hit_target=None, hit_weight=0.1):
        """
        MSE to target y (+ hit-head BCE when hit_target is given).
        With preconditioning, the value block regresses F on its own
        unit-variance target (EDM loss weight 1/c_out^2).
        """
        v, h, u = self._raw(t, xt, energy)
        if self.precond:
            c_skip, c_out, _ = self.coeffs(t)
            mse = (v - (y - c_skip * xt) / c_out).pow(2).mean()
        else:
            mse = (v - y).pow(2).mean()
        if h is None or hit_target is None:
            return mse
        total = mse + hit_weight * nn.functional.binary_cross_entropy_with_logits(
            h, hit_target)
        if u is not None:
            # value-given-lit head: supervised only where the voxel is lit
            n = hit_target.sum().clamp(min=1.0)
            total = total + ((u - y).pow(2) * hit_target).sum() / n
        return total

    @torch.no_grad()
    def fixed_eval(self, x0, energy, n_t=8, seed=1234):
        """
        Deterministic validation loss: FIXED showers, FIXED noise, FIXED t grid.

        The per-step training loss is a single-batch estimate over randomly
        drawn t, and the objective's value depends strongly on t -- near the
        data variance at t = 1, near 0 at t = 0. That makes consecutive printed
        losses differ by 3x (measured: 0.07 to 0.27 within one run) for reasons
        that have nothing to do with learning, hiding the trend entirely.

        This holds everything but the weights fixed, so successive calls are
        comparable and a flat curve really means "not improving". Returns the
        mean over the t grid and the per-t breakdown, which is the more useful
        read: a model usually stops improving at high t long before low t.
        """
        was_training = self.training
        self.eval()
        g = torch.Generator(device="cpu").manual_seed(seed)
        eps = torch.randn(x0.shape, generator=g).to(x0.device)
        per_t = []
        for i in range(n_t):
            t = torch.full((x0.shape[0], 1), (i + 0.5) / n_t, device=x0.device)
            xt = (1 - t) * x0 + (self.sigma0 + (1 - self.sigma0) * t) * eps
            per_t.append(((self.q_hat(t, xt, energy) - x0).pow(2).mean().item(),
                          float(t[0, 0])))
        if was_training:
            self.train()
        return sum(v for v, _ in per_t) / len(per_t), per_t

    def G_hat(self, t, x, energy):
        """Same algebra as flow_de.qVectorField.G, with the learned q."""
        sigma_t = self.sigma0 + (1 - self.sigma0) * t
        return ((1 - self.sigma0) * x - self.q_hat(t, x, energy)) / sigma_t


def train(model, x0_all, e_all, q_field=None, steps=16000, batch=32, lr=2e-4,
          device="cpu", log_every=1000, checkpoint_path=None, stats=None,
          target="clean", warmup=500, ema_decay=0.999, hit_weight=0.1, resume=False):
    """
    Regress q(t, x, E) = E[x0 | x_t, E]. Every batch is freshly sampled --
    random showers, random t, random noise -- so every point lies on the
    interpolant manifold.

    target="clean" (default): the regression target is the clean shower x0
        that generated x_t. The MSE minimizer is exactly E[x0 | x_t, E] over
        the training data, so no reference ensemble is needed at all.
    target="mc": the target is the Monte-Carlo posterior mean from q_field.

    Measured head-to-head (8k steps, same net/batch/seed/noise, held-out real
    showers as baseline): clean beat mc on every diversity metric -- spread
    across showers 0.276 vs 0.149-0.176 (real 0.408), pairwise distance 58 vs
    24-27 (real 86), distance to nearest training shower 32 vs 7-9 (real 47)
    -- and trained 2-4x faster. The mc target collapses samples toward the
    mean, and a larger n_ref does not fix it (128 -> 512 moved spread only
    0.149 -> 0.176 at 2.3x the cost). The underlying reason: the exact
    finite-reference q-field is a memorizing sampler -- run as an ODE it
    lands within 1.76 of a training shower, vs 47 for a genuinely new one --
    so it is a poor thing to distill.
    """
    if target == "mc" and q_field is None:
        raise ValueError("target='mc' needs a q_field (see build_q_field)")
    history = []
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    # Resume: the checkpoint carries the raw weights, the EMA weights, the
    # optimizer moments and the step reached, so a killed run continues on the
    # same cosine schedule instead of restarting. Without this a multi-hour run
    # lost to a crash (or a driver upgrade) has to be redone from scratch.
    start_step, best_val = 0, float("inf")
    if resume and checkpoint_path and os.path.isfile(checkpoint_path):
        ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ck.get("raw_state", ck["model_state"]))
        ema.load_state_dict(ck["model_state"])
        if ck.get("optim_state") is not None:
            opt.load_state_dict(ck["optim_state"])
        else:
            print("  resume: no optimizer state in the checkpoint (written by an older "
                  "version); Adam moments restart, which costs a few hundred steps")
        hist = ck.get("history") or []
        start_step = hist[-1][0] if hist else 0
        best_val = min((h[2] for h in hist if len(h) > 2), default=float("inf"))
        history[:] = hist
        print(f"  resumed from {checkpoint_path} at step {start_step}/{steps} "
              f"(best val so far {best_val:.6f})")
    elif resume:
        print(f"  --resume given but {checkpoint_path} does not exist yet -- starting fresh")

    def lr_at(step):     # linear warmup, then cosine decay to 0 at `steps`
        return lr * min(1.0, step / warmup) * 0.5 * (1 + math.cos(math.pi * step / steps))

    s0 = model.sigma0
    eval_x0 = eval_e = None          # fixed held-out batch for fixed_eval
    for step in range(start_step + 1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        idx = torch.randint(0, x0_all.shape[0], (batch,))
        x0b = x0_all[idx].to(device)
        eb = e_all[idx].to(device)
        t = torch.rand(batch, 1, device=device)
        xt = (1 - t) * x0b + (s0 + (1 - s0) * t) * torch.randn_like(x0b)

        if target == "clean":
            y = x0b
        else:
            with torch.no_grad():
                y = q_field(t, xt, eb)

        # hit label: in the standardized representation an empty voxel sits at
        # exactly empty_value, so "lit in x0" is x0 above it.
        hit_target = ((x0b > model.empty_value + 1e-6).float()
                      if model.hit_head else None)

        opt.zero_grad(set_to_none=True)
        loss = model.loss(t, xt, eb, y, hit_target=hit_target,
                          hit_weight=hit_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(ema_decay).add_(pm, alpha=1 - ema_decay)

        if step % log_every == 0 or step == steps:
            # The training loss alone is uninformative (see fixed_eval): print a
            # deterministic validation loss next to it, on a held-out batch that
            # never changes, so the trend is readable.
            if eval_x0 is None:
                eval_x0 = x0_all[-256:].to(device)
                eval_e = e_all[-256:].to(device)
            val, _ = ema.fixed_eval(eval_x0, eval_e)
            print(f"  step {step:6d}  loss = {loss.item():.6f}  val = {val:.6f}"
                  f"  lr = {lr_at(step):.2e}")
            history.append((step, loss.item(), val))
            if checkpoint_path:
                torch.save({"model_state": ema.state_dict(),      # EMA: use for sampling
                             "raw_state": model.state_dict(),
                             "optim_state": opt.state_dict(),
                             "step": step,
                             "shower_dim": model.shower_dim,
                             "hidden": model.hidden,
                             "sigma0": model.sigma0,
                             "precond": model.precond,
                             "hit_head": model.hit_head,
                             "lit_head": model.lit_head,
                             "cond": model.cond,
                             "cond_extra": model.cond_extra,
                             "pos_channels": getattr(model, "pos_channels", False),
                             "empty_value": float(model.empty_value),
                             "stats": stats, "history": history},
                            checkpoint_path)
                # A separate best-by-val copy. Note this is a WEAK selector:
                # sample quality and the denoising loss were measured to come
                # apart (the pre-hit-head MLP degraded while its loss was flat),
                # so the latest EMA remains what sampling uses by default.
                if val < best_val:
                    best_val = val
                    shutil.copyfile(checkpoint_path, checkpoint_path + ".best")
    return ema, history


@torch.no_grad()
def sample(model, energy_cond, n_steps=100, device="cpu", z=None, gate="topk",
           value_correct=False, readout="denoise"):
    """
    Integrate the reverse-time ODE from t=1 (noise) to t=0 using the LEARNED
    field -- Heun's 2nd-order scheme, each step one pair of network calls
    instead of an O(n_ref * 6480) Monte-Carlo sum.

    Time grid t_k = (1 - k/n)^2 concentrates steps near t=0, where
    G ~ x/sigma_t is stiffest (|lambda| ~ 1/sigma0 = 100).

    Returns the DENOISED, hit-gated read-out rather than the raw ODE end
    point. At t=0 the state still carries sigma0 = 0.01 of noise in every
    voxel; the denoiser removes it. Measured with the exact MC field: the
    denoising step alone brings the hit fraction from 0.58 to 0.25
    (real: 0.25). The gate (see VelocityNet.denoise) is what keeps a LEARNED
    q from filling empty voxels with small spurious deposits.
    """
    n = energy_cond.shape[0]
    x = torch.randn(n, model.shower_dim, device=device) if z is None else z.clone()
    grid = ((1 - torch.arange(n_steps + 1, dtype=torch.float64) / n_steps) ** 2).tolist()

    def tt(v):
        return torch.full((n, 1), float(v), device=device)

    for k in range(n_steps):
        t0, t1 = grid[k], grid[k + 1]
        h = t0 - t1
        G1 = model.G_hat(tt(t0), x, energy_cond)
        G2 = model.G_hat(tt(t1), x - G1 * h, energy_cond)
        x = x - (G1 + G2) * h / 2
    if readout == "denoise":
        return model.denoise(tt(0.0), x, energy_cond, gate=gate,
                             value_correct=value_correct)
    if readout == "lit":
        logit = model.hit_logit(tt(0.0), x, energy_cond)
        lit = model.gate_mask(logit, gate)
        u = model.lit_value(tt(0.0), x, energy_cond)
        return torch.where(lit, u, model.empty_value.expand_as(u))
    if readout != "raw":
        raise ValueError(f"unknown readout {readout!r}")
    # raw: keep the ODE end point's own values and use the hit head only to
    # decide occupancy.
    #
    # x(0) is already a SAMPLE from the model's distribution; q_hat(0, x) is its
    # conditional MEAN, which is diluted by the probability that each voxel is
    # empty (measured on real showers: the mean carries only 0.85 of the energy
    # that truly-lit voxels hold, which is most of the sampler's energy
    # deficit). The denoised read-out existed to remove the sigma0 = 0.01 of
    # noise still present at t = 0, and the gate now does that for the empty
    # voxels where it mattered.
    if not model.hit_head:
        return x
    logit = model.hit_logit(tt(0.0), x, energy_cond)
    lit = model.gate_mask(logit, gate)
    return torch.where(lit, x, model.empty_value.expand_as(x))


def build_q_field(x0_all, e_all, n_ref, device="cpu", sigma0=1e-2,
                   energy_kernel_scale=5.0):
    """
    Reference ensemble for the Monte-Carlo target (only used by
    train(..., target="mc")). Not recommended: see train()'s docstring.
    Raising n_ref does not rescue it -- 128 -> 512 barely changed sample
    diversity while costing 2.3x the training time, since cost scales with
    one (batch x n_ref x 6480) tensor per step.
    """
    x0_ref, e0_ref = flow_de.get_target_sample(x0_all, e_all, n_ref)
    return flow_de.qVectorField(x0_ref.to(device), e0_ref.to(device),
                                 sigma0=sigma0,
                                 energy_kernel_scale=energy_kernel_scale).to(device)


def from_checkpoint(checkpoint, device="cpu", verbose=True):
    """Rebuild a VelocityNet from a checkpoint written by train().

    Reads every architecture flag from the file rather than from CLI defaults,
    so a checkpoint always reconstructs the model it was trained as. Note the
    `pos_channels` default of False: conv checkpoints predating the position
    channels have a 1-input-channel trunk, and defaulting to True would fail
    to load them.

    Loads `model_state`, which holds the EMA weights -- what sampling uses.
    """
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    model = VelocityNet(
        shower_dim=ck["shower_dim"], hidden=ck["hidden"], sigma0=ck["sigma0"],
        precond=ck.get("precond", False), hit_head=ck.get("hit_head", False),
        empty_value=ck.get("empty_value", 0.0),
        cond=ck.get("cond", "concat"), lit_head=ck.get("lit_head", False),
        cond_extra=ck.get("cond_extra", 0),
        pos_channels=ck.get("pos_channels", False)).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    if verbose:
        hist = ck.get("history") or []
        print(f"loaded {checkpoint}: hidden={ck['hidden']}, hit_head={model.hit_head}, "
              f"precond={model.precond}, steps trained={hist[-1][0] if hist else '?'}, "
              f"representation={ck.get('stats', {}).get('representation', '?')}")
    return model, ck
