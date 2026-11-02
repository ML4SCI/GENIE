"""
autoencoder.py -- FactorizedCaloAE and its building blocks.

Extracted from train_auto_encoder_improved.py. One behavioural change: the
log10(E) standardization constants were module-level globals computed at
import time from the dataset, and `energy_to_cond` read them from module
scope while being called from inside the model's forward pass. The model was
therefore not constructible without loading 2.5 GB of data first.

They are now buffers on the model, defaulting to the measured Dataset-2
values, and `energy_to_cond` is a method. Checkpoints already recorded
log_e_mean / log_e_std, so existing files load unchanged.

Architecture, and the idea that makes 96 numbers enough: treat the 45
longitudinal layers as 45 separate 16x9 images, encode them with a SHARED 2D
stack, then model the longitudinal direction with 1D convolutions.

    [B, 45, 16, 9]
      -> per-layer 2D encoder (shared): (1,16,9) -> (32,8,5) -> (64,4,3)
      -> 768 features per layer, reshape to [B, 768, 45]
      -> 1D encoder along z: 768 -> 256 -> 128
      -> flatten [B, 5760] + energy embedding [B, 64] -> Linear -> [B, 8, 12]
      -> decoder: mirror, with the energy embedding injected AGAIN

The latent is STRUCTURED (8 channels x 12 longitudinal positions), not a flat
96-vector, so a downstream diffusion model can use convolutions on it. The
decoder also receives the energy embedding, which is what lets a latent
generator model only the shape distribution.

3,110,882 parameters at width_mult=1.0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.calo_dataset import DS2_LOG_E_MEAN, DS2_LOG_E_STD

# What every pre-existing checkpoint was trained with. Not the encoder's mirror
# -- see the note in FactorizedCaloAE.__init__ -- but loading one of those
# checkpoints into the geometrically correct model costs ~10% reconstruction
# error with no error message.
LEGACY_MID_TRANSVERSE = (8, 4)


def pad_phi(x, phi_circular):
    """Transverse tensors are [B*Nz, C, Nphi, Nr]. phi wraps around the
    calorimeter, r does not. Zero-padding phi (the original behaviour) tells
    the network there is a wall at phi = 0, which there is not."""
    if not phi_circular:
        return F.pad(x, (1, 1, 1, 1))
    return F.pad(F.pad(x, (0, 0, 1, 1), mode="circular"), (1, 1, 0, 0))


class ResConvBlock2D(nn.Module):
    """
    Conv block with residual connection.

    The residual path is essential when the network has no U-Net skip
    connections (which we can't have, because the decoder must be usable
    standalone after diffusion). Residuals dramatically improve trainability
    and gradient flow.

    When in_ch != out_ch, the shortcut uses a 1x1 conv.
    """

    def __init__(self, in_ch, out_ch, phi_circular=False):
        super().__init__()
        self.phi_circular = phi_circular

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=0)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=0)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)

        # 1x1 conv shortcut if channel count changes; identity otherwise.
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = F.gelu(self.norm1(self.conv1(pad_phi(x, self.phi_circular))))
        h = self.norm2(self.conv2(pad_phi(h, self.phi_circular)))
        return F.gelu(h + self.shortcut(x))


class StridedDown2D(nn.Module):
    """
    Learnable downsampling via stride-2 conv. Replaces MaxPool2d.

    MaxPool throws away information non-recoverably; a strided conv lets the
    network learn which features to preserve. Output spatial size is computed
    to match the target (rather than just halving), so we can hit exactly
    (16, 9) -> (8, 4) -> (4, 2).
    """

    def __init__(self, in_ch, out_ch, phi_circular=False):
        super().__init__()
        self.phi_circular = phi_circular
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=0)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)

    def forward(self, x):
        return F.gelu(self.norm(self.conv(pad_phi(x, self.phi_circular))))


class UpBlock2D(nn.Module):
    """Bilinear upsample + conv. Same as original, just using ResConvBlock."""

    def __init__(self, in_ch, out_ch, size, phi_circular=False):
        super().__init__()
        self.size = size
        self.phi_circular = phi_circular
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=0)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)

    def forward(self, x):
        x = F.interpolate(x, size=self.size, mode='bilinear', align_corners=False)
        return F.gelu(self.norm(self.conv(pad_phi(x, self.phi_circular))))


class EnergyEmbed(nn.Module):
    """
    Small MLP that lifts the scalar log-energy into a richer embedding.

    A 1-dim conditioning signal forces every downstream linear layer to learn
    its own scale for the energy term. A learned embedding gives the model
    much more flexibility and works better in practice.
    """

    def __init__(self, embed_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )

    def forward(self, e_cond):
        # e_cond: [B, 1] standardized log10(E)
        return self.mlp(e_cond)  # [B, embed_dim]


class FactorizedCaloAE(nn.Module):

    def __init__(
        self,
        Nz=45,
        Nphi=16,
        Nr=9,
        latent_channels=8,
        latent_length=12,
        energy_embed_dim=64,
        two_head=True,
        x_scale=1.0,
        phi_circular=False,
        width_mult=1.0,
        log_e_mean=DS2_LOG_E_MEAN,
        log_e_std=DS2_LOG_E_STD,
        mid_transverse=None,
    ):
        super().__init__()
        # Intermediate transverse resolution in the DECODER.
        #
        # The encoder goes (16, 9) -> (8, 5) -> (4, 3), so a mirrored decoder
        # must go (4, 3) -> (8, 5) -> (16, 9). None means "derive it", which
        # gives that correct mirror.
        #
        # But EVERY checkpoint in this project was trained with (8, 4), which
        # is not the encoder's mirror. The training script was corrected to
        # (8, 5) at some point after all of those models were produced, and the
        # correction went unnoticed because conv weights do not encode spatial
        # size: a (8, 4)-trained checkpoint loads into an (8, 5) model with
        # zero missing or unexpected keys and silently reconstructs ~10% worse.
        #
        # from_checkpoint() therefore defaults a checkpoint with no
        # `mid_transverse` field to LEGACY_MID_TRANSVERSE, and new checkpoints
        # record what they used.
        self.mid_transverse = tuple(mid_transverse) if mid_transverse else None
        # log10(E) standardization, formerly module-level globals set at import
        # time from whatever data file happened to be loaded. They are buffers
        # now, so a model carries the constants it was trained with and can be
        # constructed without touching the dataset at all. Checkpoints already
        # stored log_e_mean / log_e_std, so this is backward compatible --
        # from_checkpoint() reads them back.
        # persistent=False: these are metadata, carried in the checkpoint dict
        # under "log_e_mean"/"log_e_std", NOT in the state_dict. Making them
        # persistent would add keys that every pre-existing checkpoint lacks,
        # breaking strict loading of every model trained so far.
        self.register_buffer("log_e_mean", torch.tensor(float(log_e_mean)),
                             persistent=False)
        self.register_buffer("log_e_std", torch.tensor(float(log_e_std)),
                             persistent=False)
        # phi wraps around the calorimeter; zero-padding it (the original
        # behaviour) asserts a wall at phi = 0. width_mult scales every conv
        # channel count, so capacity is sweepable independently of the latent.
        self.phi_circular = phi_circular
        self.width_mult = width_mult
        w = lambda c: max(4, int(round(c * width_mult)))
        self._w = w

        self.Nz = Nz
        self.Nphi = Nphi
        self.Nr = Nr

        # v4 (two_head=True): the decoder outputs TWO channels per voxel --
        #   v: the value, in SCALED log space (x_log / x_scale, O(1))
        #   h: a hit/no-hit logit
        # and the public decode()/forward() return x_hat_log = x_scale * v
        # where h > 0, else exactly 0 -- same interface as before.
        # Why (measured on the v3 checkpoint, see the v4 note on
        # calorimeter_loss_v4): with one shared output the hit BCE needed
        # logits of +-3 from a number that must also equal ~4e-4, which pinned
        # empty voxels at -0.058, halved lit voxels, and left the model WORSE
        # than predicting the energy-binned mean shower (rel 0.688 vs 0.638).
        # two_head=False rebuilds the old single-output model so pre-v4
        # checkpoints still load.
        self.two_head = two_head
        if two_head:
            self.register_buffer("x_scale", torch.tensor(float(x_scale)))
        else:
            self.x_scale = 1.0

        # Structured latent dimensions
        self.latent_channels = latent_channels
        self.latent_length = latent_length
        self.latent_dim = latent_channels * latent_length

        self.energy_embed_dim = energy_embed_dim
        self.energy_embed = EnergyEmbed(energy_embed_dim)

        # Learnable temperature for the hit/no-hit BCE logit (see
        # calorimeter_loss). x_hat_log itself is reused as that logit, but
        # under the GeV/GeV units convention its natural scale is tiny
        # (mean nonzero ~0.001) -- sigmoid(x_hat_log) sits at ~0.5
        # regardless of class, which floors hit_loss at ln(2) no matter how
        # good the fit is (diagnosed directly: see the Training Plateau
        # Postmortem). Scaling the logit before the sigmoid decouples the
        # BCE's usable dynamic range from the reconstruction target's own
        # scale. Initialized to 10 (roughly restoring the old MeV-scale
        # convention's typical logit magnitude) and left learnable so
        # training can adjust it further.
        if not two_head:
            self.hit_logit_scale = nn.Parameter(torch.tensor(10.0))

        # --------------------------------------------------------------------
        # Transverse encoder: processes each z-layer as an independent 16x9 img
        # --------------------------------------------------------------------
        pc = phi_circular
        self.enc2d = nn.Sequential(
            ResConvBlock2D(1, w(16), pc),           # (1, 16, 9) -> (16, 16, 9)
            StridedDown2D(w(16), w(32), pc),        # -> (32, 8, 5)
            ResConvBlock2D(w(32), w(32), pc),
            StridedDown2D(w(32), w(64), pc),        # -> (64, 4, 3)
            ResConvBlock2D(w(64), w(64), pc),
        )

        # After two stride-2 convs with input (16, 9), output is (4, 3) using
        # standard PyTorch conv arithmetic: ceil(16/2)=8, ceil(8/2)=4 and
        # ceil(9/2)=5, ceil(5/2)=3. Let's verify with a dummy tensor below.
        # We'll set Henc/Wenc dynamically in the first forward pass.
        self._enc_shape_inferred = False
        self.Henc = None
        self.Wenc = None
        self.transverse_features = None

        # Placeholder modules - will be built lazily once we know Henc, Wenc.
        # This avoids hardcoding shapes that depend on input geometry.
        self.enc1d = None
        self.fc_enc = None
        self.fc_dec = None
        self.dec1d = None
        self.dec2d = None

        # We need to know transverse output shape to build the rest.
        # Run a dummy forward through enc2d once to capture it.
        self._build_lazy_modules()

    # ----------------------------------------------------------------------
    # Lazy module construction
    # ----------------------------------------------------------------------
    # The output shape of enc2d depends on the input transverse size and the
    # exact stride/padding choices. Rather than hardcoding (and risking a
    # silent shape mismatch later), we infer it from a dummy forward pass.
    # ----------------------------------------------------------------------

    def _mid_size(self):
        """The decoder's intermediate transverse size.

        Derived from the encoder when not set explicitly: one stride-2 step up
        from (Henc, Wenc) using the same ceil arithmetic the encoder used
        coming down, which is exactly the encoder's own intermediate shape.
        """
        if self.mid_transverse is not None:
            return self.mid_transverse
        return ((self.Nphi + 1) // 2, (self.Nr + 1) // 2)

    def _build_lazy_modules(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.Nphi, self.Nr)
            out = self.enc2d(dummy)

        _, C, H, W = out.shape
        self.Henc = H
        self.Wenc = W
        self.transverse_features = C * H * W
        self._enc_shape_inferred = True

        print(
            f"[AE] transverse encoder output: "
            f"C={C}, H={H}, W={W}  -> features={self.transverse_features}"
        )

        # 1D longitudinal encoder
        w, pc = self._w, self.phi_circular
        self.c1d_mid, self.c1d_out = w(256), w(128)
        self.enc1d = nn.Sequential(
            nn.Conv1d(self.transverse_features, self.c1d_mid, 3, padding=1),
            nn.GroupNorm(min(8, self.c1d_mid), self.c1d_mid),
            nn.GELU(),
            nn.Conv1d(self.c1d_mid, self.c1d_out, 3, padding=1),
            nn.GroupNorm(min(8, self.c1d_out), self.c1d_out),
            nn.GELU(),
        )

        # Bottleneck
        # +energy_embed_dim because we concat the energy embedding
        self.fc_enc = nn.Linear(
            self.c1d_out * self.Nz + self.energy_embed_dim,
            self.latent_dim,
        )

        # Decoder mirror: latent + energy_embed -> 128*Nz
        self.fc_dec = nn.Linear(
            self.latent_dim + self.energy_embed_dim,
            self.c1d_out * self.Nz,
        )

        # 1D longitudinal decoder
        self.dec1d = nn.Sequential(
            nn.Conv1d(self.c1d_out, self.c1d_mid, 3, padding=1),
            nn.GroupNorm(min(8, self.c1d_mid), self.c1d_mid),
            nn.GELU(),
            nn.Conv1d(self.c1d_mid, self.transverse_features, 3, padding=1),
            nn.GroupNorm(min(8, self.transverse_features), self.transverse_features),
            nn.GELU(),
        )

        # 2D transverse decoder: mirror of encoder.
        # NOTE: NO Softplus at the end. We use a linear output. In log-space
        # the target for empty voxels is exactly 0, and Softplus has a floor
        # near 0.69 that would prevent the model from ever predicting 0.
        self.dec2d = nn.Sequential(
            ResConvBlock2D(w(64), w(64), pc),
            UpBlock2D(w(64), w(32), size=self._mid_size(), phi_circular=pc),
            ResConvBlock2D(w(32), w(32), pc),
            UpBlock2D(w(32), w(16), size=(self.Nphi, self.Nr), phi_circular=pc),
            ResConvBlock2D(w(16), w(16), pc),
            # LINEAR output: 2 channels (value, hit logit) for v4, 1 for pre-v4
            nn.Conv2d(w(16), 2 if self.two_head else 1, kernel_size=3, padding=1),
        )


    def energy_to_cond(self, energy):
        """Incident energy -> standardized log10(E), shape [B, 1].

        Uses this model's own constants, not module globals.
        """
        energy = energy.reshape(-1)
        log_e = torch.log10(energy + 1e-8)
        out = ((log_e - self.log_e_mean) / self.log_e_std).unsqueeze(1)
        assert out.dim() == 2 and out.shape[1] == 1, (
            f"energy_to_cond: expected output [B, 1], got {tuple(out.shape)}")
        return out

    # ----------------------------------------------------------------------
    # Encode
    # ----------------------------------------------------------------------

    def encode(self, x_log, energy):
        """
        x_log:  [B, Nz, Nphi, Nr]  log-normalized shower
        energy: [B]                 incident energy (raw GeV)
        returns latent: [B, latent_channels, latent_length]
        """
        # ---- input contract ----
        assert x_log.dim() == 4, (
            f"encode: expected x_log [B, Nz, Nphi, Nr], "
            f"got {tuple(x_log.shape)}"
        )
        B, Nz_in, Nphi_in, Nr_in = x_log.shape
        assert (Nz_in, Nphi_in, Nr_in) == (self.Nz, self.Nphi, self.Nr), (
            f"encode: input geometry {(Nz_in, Nphi_in, Nr_in)} != "
            f"model geometry {(self.Nz, self.Nphi, self.Nr)}"
        )
        assert energy.numel() == B, (
            f"encode: energy has {energy.numel()} elements, expected {B}"
        )

        # Per-layer 2D encoding (on O(1) scaled input for v4; x_scale = 1 pre-v4)
        x = (x_log / self.x_scale).reshape(B * self.Nz, 1, self.Nphi, self.Nr)
        x = self.enc2d(x)
        assert x.shape == (
            B * self.Nz, self._w(64), self.Henc, self.Wenc
        ), (
            f"encode: enc2d output {tuple(x.shape)} != expected "
            f"{(B * self.Nz, self._w(64), self.Henc, self.Wenc)}"
        )
        x = x.reshape(B, self.Nz, self.transverse_features)

        # Longitudinal encoding: features become channels, z becomes length
        x = x.permute(0, 2, 1)                # [B, F, Nz]
        x = self.enc1d(x)                     # [B, 128, Nz]
        assert x.shape == (B, self.c1d_out, self.Nz), (
            f"encode: enc1d output {tuple(x.shape)} != expected "
            f"{(B, self.c1d_out, self.Nz)}"
        )
        x = x.reshape(B, -1)                  # [B, 128*Nz]
        assert x.dim() == 2, (
            f"encode: flattened x should be 2D, got {tuple(x.shape)}"
        )

        # Energy conditioning
        e_emb = self.energy_embed(self.energy_to_cond(energy))  # [B, embed]
        assert e_emb.shape == (B, self.energy_embed_dim), (
            f"encode: energy embedding {tuple(e_emb.shape)} != "
            f"{(B, self.energy_embed_dim)}"
        )
        x = torch.cat([x, e_emb], dim=1)

        # Bottleneck -> structured latent
        z = self.fc_enc(x)
        z = z.reshape(B, self.latent_channels, self.latent_length)
        assert z.shape == (B, self.latent_channels, self.latent_length), (
            f"encode: latent shape {tuple(z.shape)} != expected "
            f"{(B, self.latent_channels, self.latent_length)}"
        )
        return z

    # ----------------------------------------------------------------------
    # Decode
    # ----------------------------------------------------------------------

    def decode(self, z, energy):
        """
        z:      [B, latent_channels, latent_length]
        energy: [B]
        returns x_log: [B, Nz, Nphi, Nr] in log-normalized space
        (v4: value head gated by the hit head; empty voxels exactly 0)
        """
        if not self.two_head:
            return self._decode_raw(z, energy)[:, :, 0]
        v, h = self.decode_heads(z, energy)
        return torch.where(h > 0, (v.float() * self.x_scale).clamp(min=0.0),
                           torch.zeros_like(v, dtype=torch.float32))

    def decode_heads(self, z, energy):
        """v4 only: (v, h) = (value in scaled log space, hit logit)."""
        x = self._decode_raw(z, energy)
        return x[:, :, 0], x[:, :, 1]

    def _decode_raw(self, z, energy):
        """returns [B, Nz, C_out, Nphi, Nr], C_out = 2 (v4) or 1 (pre-v4)"""
        # ---- input contract ----
        assert z.dim() == 3, (
            f"decode: expected z [B, C_lat, L_lat], got {tuple(z.shape)}"
        )
        B, C_lat, L_lat = z.shape
        assert (C_lat, L_lat) == (self.latent_channels, self.latent_length), (
            f"decode: latent geometry {(C_lat, L_lat)} != "
            f"{(self.latent_channels, self.latent_length)}"
        )
        assert energy.numel() == B, (
            f"decode: energy has {energy.numel()} elements, expected {B}"
        )

        # Flatten latent and concatenate energy embedding for the decoder too.
        # This is the key change: the decoder now KNOWS the target energy,
        # so the diffusion model only needs to learn the shape distribution.
        z_flat = z.reshape(B, self.latent_dim)
        e_emb = self.energy_embed(self.energy_to_cond(energy))
        assert e_emb.shape == (B, self.energy_embed_dim), (
            f"decode: energy embedding {tuple(e_emb.shape)} != "
            f"{(B, self.energy_embed_dim)}"
        )
        zc = torch.cat([z_flat, e_emb], dim=1)

        # Latent -> longitudinal features
        x = self.fc_dec(zc)
        x = x.reshape(B, self.c1d_out, self.Nz)

        x = self.dec1d(x)                       # [B, F, Nz]
        assert x.shape == (B, self.transverse_features, self.Nz), (
            f"decode: dec1d output {tuple(x.shape)} != expected "
            f"{(B, self.transverse_features, self.Nz)}"
        )
        x = x.permute(0, 2, 1)                  # [B, Nz, F]

        # Per-layer 2D decoding
        x = x.reshape(B * self.Nz, self._w(64), self.Henc, self.Wenc)
        x = self.dec2d(x)                       # [B*Nz, C_out, Nphi, Nr]
        C_out = 2 if self.two_head else 1
        assert x.shape == (B * self.Nz, C_out, self.Nphi, self.Nr), (
            f"decode: dec2d output {tuple(x.shape)} != expected "
            f"{(B * self.Nz, C_out, self.Nphi, self.Nr)}"
        )

        return x.reshape(B, self.Nz, C_out, self.Nphi, self.Nr)

    def forward_heads(self, x_log, energy):
        """v4 training path: returns (v, h, z)."""
        z = self.encode(x_log, energy)
        v, h = self.decode_heads(z, energy)
        return v, h, z

    def forward(self, x_log, energy):
        z = self.encode(x_log, energy)
        x_hat_log = self.decode(z, energy)
        assert x_hat_log.shape == x_log.shape, (
            f"forward: output {tuple(x_hat_log.shape)} != "
            f"input {tuple(x_log.shape)}"
        )
        return x_hat_log, z


# Parameters that some historical checkpoints legitimately lack. hit_logit_scale
# was added partway through the pre-v4 era and is used ONLY by the v3 loss
# (calorimeter_loss), never in forward() or decode(), so a checkpoint without it
# reconstructs identically. Anything missing beyond this set is a real mismatch.
_BENIGN_MISSING = {"hit_logit_scale"}


def from_checkpoint(path, device="cpu", strict=True, verbose=True):
    """Rebuild a FactorizedCaloAE from a checkpoint.

    Reads the architecture and the standardization constants from the file, so
    a checkpoint reconstructs the model it was trained as. Checkpoints predating
    the v4 two-head decoder have no `two_head` key and default to False;
    those predating the units fix have no `shower_units_gev` and were trained on
    log1p(E_MeV / E_GeV), which is reported but not corrected -- evaluate such a
    model under the convention it was trained with.
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    model = FactorizedCaloAE(
        Nz=ck.get("Nz", 45), Nphi=ck.get("Nphi", 16), Nr=ck.get("Nr", 9),
        latent_channels=ck.get("latent_channels", 8),
        latent_length=ck.get("latent_length", 12),
        energy_embed_dim=ck.get("energy_embed_dim", 64),
        two_head=ck.get("two_head", False),
        x_scale=ck.get("x_scale", 1.0),
        phi_circular=ck.get("phi_circular", False),
        width_mult=ck.get("width_mult", 1.0),
        log_e_mean=ck.get("log_e_mean", DS2_LOG_E_MEAN),
        log_e_std=ck.get("log_e_std", DS2_LOG_E_STD),
        mid_transverse=ck.get("mid_transverse", LEGACY_MID_TRANSVERSE),
    ).to(device)
    state = ck.get("model_state", ck)
    res = model.load_state_dict(state, strict=False)
    missing = [k for k in res.missing_keys if k not in _BENIGN_MISSING]
    benign = [k for k in res.missing_keys if k in _BENIGN_MISSING]
    if strict and (missing or res.unexpected_keys):
        raise RuntimeError(
            f"{path} does not match FactorizedCaloAE as constructed from its own "
            f"metadata: missing={missing} unexpected={list(res.unexpected_keys)}. "
            f"Pass strict=False to load anyway.")
    model.eval()
    if verbose:
        print(f"loaded {path}: epoch={ck.get('epoch')}, "
              f"two_head={ck.get('two_head')}, "
              f"units_gev={ck.get('shower_units_gev')}, "
              f"latent={model.latent_channels}x{model.latent_length}")
    if "mid_transverse" not in ck:
        print(f"  note: no `mid_transverse` field; assuming the legacy "
              f"{LEGACY_MID_TRANSVERSE} decoder resolution this checkpoint was "
              f"trained with (the encoder's mirror is "
              f"{((model.Nphi + 1) // 2, (model.Nr + 1) // 2)})")
    if benign:
        print(f"  note: {benign} absent (pre-dates that parameter; it is used only "
              f"by the v3 loss, so reconstruction is unaffected)")
    if missing or res.unexpected_keys:
        print(f"  WARNING missing={missing} unexpected={list(res.unexpected_keys)}")
    return model, ck
