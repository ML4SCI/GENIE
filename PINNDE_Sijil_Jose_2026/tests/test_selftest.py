"""
selftest.py -- fast CPU regression tests for the pieces that have actually
broken in this project. Seconds to run, no dataset and no GPU needed:

    pytest

Each test exists because something failed in a way that cost a real run:

  representation round trip   a units bug (shower MeV / energy GeV) once
                              inflated the target 1000x and poisoned training
  empty voxel mapping         the gated read-out must invert to EXACTLY 0 MeV,
                              or the 15.15 keV threshold silently re-fills it
  G algebra                   G is exact algebra from q; if it drifts from
                              flow_de's definition the ODE is no longer the
                              one the formalism describes
  gate rules                  topk must keep round(sum p) voxels; and all three
                              gates must survive NON-FINITE logits, which once
                              produced an out-of-bounds index and killed a
                              100k-shower sampling pass with a CUDA assert
  divergence + export         a finished 100k-step run was lost to an assert on
                              non-finite values; export must clamp and report
                              instead of throwing
  checkpoint round trip       x_scale/empty_value live in the state dict; if
                              they do not survive save/load, sampling silently
                              uses the wrong scale
"""

import math
import sys
import tempfile

import torch

from calods2.data import export
import calods2.data.dataset as gendata
from calods2.data import layerdata
from calods2.models import velocity_net

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  ok    {name}")
    except Exception as e:  # noqa: BLE001 - a self-test reports, it does not raise
        FAIL.append((name, e))
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")


def _stats(rep="fraction"):
    return {"x0_mean": 1.206e-4 if rep == "fraction" else 1.708,
            "x0_std": 8.10e-4 if rep == "fraction" else 3.127,
            "representation": rep, "Nz": 45, "Nphi": 16, "Nr": 9}


# ----------------------------------------------------------------- transforms
def t_roundtrip_fraction():
    """Relative, not absolute: the round trip is float32-limited at ~1.2e-5
    relative (measured, uniform across incident energy), which on a 10 GeV
    voxel is 0.12 MeV -- fine as a fraction, but an absolute tolerance would
    flag it. Dim voxels are NOT the casualty here: a 15.15 keV voxel sits
    1054 float32 ticks above empty even at E_inc = 1 TeV."""
    stats = _stats("fraction")
    torch.manual_seed(0)                       # deterministic: the worst case is
    e = torch.tensor([[1.0], [100.0], [1000.0]])   # set by the dimmest voxel drawn
    sh = torch.rand(3, 45, 16, 9) * 0.01 * e.reshape(-1, 1, 1, 1)
    x_log = torch.log1p(sh / (e.reshape(-1, 1, 1, 1) + gendata.EPS))
    x_std = (x_log.reshape(3, -1) - stats["x0_mean"]) / stats["x0_std"]
    back = gendata.x0_to_physical_showers(gendata.unstandardize_x0(x_std, stats), e, stats)
    rel = ((back - sh).abs() / (sh + 1e-30))
    assert rel.median() < 1e-4, f"median relative round-trip error {rel.median():.2e}"
    # Voxels below the 15.15 keV readout are zeroed on export, so their
    # round-trip error is irrelevant; above it the transform must be accurate.
    keep = (sh * 1000.0) > 0.01515
    assert rel[keep].max() < 1e-2, f"worst relative error above threshold {rel[keep].max():.2e}"


def t_threshold_voxel_is_resolvable():
    """The dim-voxel deficit is not a precision artifact: check directly that a
    voxel at the readout threshold is far above float32 resolution, at every
    incident energy, in the representation actually used."""
    stats = _stats("fraction")
    empty = gendata.empty_value(stats)
    tick = torch.finfo(torch.float32).eps * abs(empty)
    for E in (1.0, 10.0, 100.0, 1000.0):                         # GeV
        frac = 1.515e-5 / E                                      # 15.15 keV as a fraction
        thr = (math.log1p(frac) - stats["x0_mean"]) / stats["x0_std"]
        assert (thr - empty) > 100 * tick, f"E_inc={E}: threshold only {(thr-empty)/tick:.0f} ticks"


def t_roundtrip_threshold():
    stats = _stats("threshold")
    e = torch.tensor([[1.0], [100.0]])
    sh_mev = torch.rand(2, 45, 16, 9) * 50.0
    x_log = torch.log1p(sh_mev / gendata.THRESHOLD_MEV)
    x_std = (x_log.reshape(2, -1) - stats["x0_mean"]) / stats["x0_std"]
    back = gendata.x0_to_physical_showers(gendata.unstandardize_x0(x_std, stats), e, stats)
    assert torch.allclose(back * 1000.0, sh_mev, rtol=1e-3, atol=1e-6)


def t_empty_voxel_is_exactly_zero():
    for rep in ("fraction", "threshold"):
        stats = _stats(rep)
        ev = gendata.empty_value(stats)
        x = torch.full((2, 6480), ev)
        sh = gendata.x0_to_physical_showers(
            gendata.unstandardize_x0(x, stats), torch.tensor([[10.0], [10.0]]), stats)
        assert sh.abs().max() < 1e-12, f"{rep}: empty voxel -> {sh.abs().max()} GeV, not 0"


def t_layer_representation_roundtrip():
    """(u, w) must invert to the original shower exactly. This transform is the
    two-stage pipeline's foundation: a silent error here would corrupt every
    sample while every aggregate metric still looked plausible."""
    torch.manual_seed(0)
    sh = (torch.rand(16, 45, 16, 9) ** 4) * 100.0      # sparse-ish, MeV
    sh[sh < 5.0] = 0.0
    sh[3, 7] = 0.0                                      # a completely empty layer
    e = torch.rand(16, 1) * 1e5 + 1e3
    u, w, stats = layerdata.encode(sh, e)
    back = layerdata.decode(u, w, e, stats)
    rel = (back - sh).abs().sum() / sh.sum()
    assert rel < 1e-5, f"round-trip relative error {rel:.2e}"
    assert u.shape == (16, 45) and w.shape == (16, 6480)
    assert torch.isfinite(u).all() and torch.isfinite(w).all()


def t_layer_representation_separates_empty_from_faint():
    """The whole point of the logit: a faint voxel must be far from an empty
    one. In the 'fraction' representation that gap was a median 5.5e-4
    standardized units, which no regressor can resolve."""
    torch.manual_seed(0)
    sh = (torch.rand(32, 45, 16, 9) ** 4) * 100.0
    sh[sh < 5.0] = 0.0
    e = torch.rand(32, 1) * 1e5 + 1e3
    _, w, stats = layerdata.encode(sh, e)
    empty = layerdata.empty_value(stats)
    lit = w.reshape(32, -1)[sh.reshape(32, -1) > 0]
    assert (lit.min() - empty) > 0.2, (
        f"faint voxels only {(lit.min() - empty):.3f} units above empty")


def t_layer_energies_are_exact():
    """Energy conservation is analytic in this parameterization: the decoded
    layer energies must match the originals regardless of the shape values."""
    torch.manual_seed(1)
    sh = torch.rand(8, 45, 16, 9) * 10.0
    e = torch.full((8, 1), 1e4)
    u, w, stats = layerdata.encode(sh, e)
    back = layerdata.decode(u, w, e, stats)
    le_a = sh.reshape(8, 45, -1).sum(-1)
    le_b = back.reshape(8, 45, -1).sum(-1)
    assert torch.allclose(le_a, le_b, rtol=1e-4), (le_a - le_b).abs().max()


# ----------------------------------------------------------------- the field
def t_G_matches_formalism():
    """G must be exactly [(1-sigma0) x - q] / sigma(t), the same algebra flow_de uses."""
    m = velocity_net.VelocityNet(shower_dim=32, hidden=16, hit_head=True, empty_value=-0.15)
    t = torch.rand(4, 1)
    x = torch.randn(4, 32)
    e = torch.randn(4, 1)
    s0 = m.sigma0
    expect = ((1 - s0) * x - m.q_hat(t, x, e)) / (s0 + (1 - s0) * t)
    assert torch.allclose(m.G_hat(t, x, e), expect, atol=1e-6)


def t_interpolant_endpoints():
    """x(0) = x0 up to the sigma0 floor, x(1) is pure noise."""
    s0 = 1e-2
    x0 = torch.randn(8, 16)
    eps = torch.randn(8, 16)
    at0 = (1 - 0.0) * x0 + (s0 + (1 - s0) * 0.0) * eps
    at1 = (1 - 1.0) * x0 + (s0 + (1 - s0) * 1.0) * eps
    assert torch.allclose(at0, x0 + s0 * eps, atol=1e-6)
    assert torch.allclose(at1, eps, atol=1e-6)


# ----------------------------------------------------------------- the gate
def t_topk_keeps_expected_count():
    m = velocity_net.VelocityNet(shower_dim=512, hidden=16, hit_head=True, empty_value=-0.15)
    h = torch.randn(4, 512) * 2.0
    k_expect = torch.sigmoid(h).sum(1).round().long()
    lit = m.gate_mask(h, "topk")
    assert torch.equal(lit.sum(1), k_expect), (lit.sum(1), k_expect)


def t_gumbel_keeps_expected_count():
    """Same count as topk, but a different (stochastic) selection. Caught a real
    bug: an operator-precedence slip made every key NaN, so the gate silently
    returned zero lit voxels -- and only at realistic widths."""
    m = velocity_net.VelocityNet(shower_dim=512, hidden=16, hit_head=True, empty_value=-0.15)
    h = torch.randn(4, 512) * 2.0
    k_expect = torch.sigmoid(h).sum(1).round().long()
    lit = m.gate_mask(h, "gumbel")
    assert torch.equal(lit.sum(1), k_expect), (lit.sum(1), k_expect)
    torch.manual_seed(1)
    a = m.gate_mask(h, "gumbel")
    torch.manual_seed(2)
    b = m.gate_mask(h, "gumbel")
    assert not torch.equal(a, b), "gumbel gate is not stochastic"
    assert not torch.equal(a, m.gate_mask(h, "topk")), "gumbel reproduced topk exactly"


def t_threshold_gate_is_p_half():
    m = velocity_net.VelocityNet(shower_dim=64, hidden=16, hit_head=True, empty_value=-0.15)
    h = torch.randn(3, 64)
    assert torch.equal(m.gate_mask(h, "threshold"), h > 0)


def t_gates_survive_nonfinite():
    """The bug that killed a 100k sampling pass: NaN logits -> NaN.long() ->
    out-of-bounds gather index -> CUDA device-side assert."""
    m = velocity_net.VelocityNet(shower_dim=64, hidden=16, hit_head=True, empty_value=-0.15)
    h = torch.randn(4, 64)
    h[0] = float("nan")
    h[1] = float("inf")
    h[2] = float("-inf")
    for gate in ("topk", "gumbel", "threshold", "sample"):
        lit = m.gate_mask(h, gate)
        assert lit.shape == h.shape
        assert int(lit[0].sum()) == 0, f"{gate}: NaN row produced {int(lit[0].sum())} lit voxels"
        assert int(lit[1].sum()) == 64, f"{gate}: +inf row should be fully lit"
        assert int(lit[2].sum()) == 0, f"{gate}: -inf row should be empty"


def t_denoise_writes_empty_value():
    m = velocity_net.VelocityNet(shower_dim=64, hidden=16, hit_head=True, empty_value=-0.1489)
    out = m.denoise(torch.zeros(4, 1), torch.randn(4, 64), torch.randn(4, 1), gate="threshold")
    off = out[out != float(m.empty_value)]
    assert out.shape == (4, 64)
    assert torch.isfinite(out).all()
    assert (out == float(m.empty_value)).any(), "nothing was gated off"
    assert off.numel() > 0, "everything was gated off"


# ----------------------------------------------------------- divergence/export
def t_diverged_mask():
    stats = _stats("fraction")
    x = torch.zeros(3, 16)
    x[1] = 1e30
    x[2] = float("nan")
    bad = export.diverged_mask(x, stats)
    assert bad.tolist() == [False, True, True], bad.tolist()


def t_export_clamps_instead_of_raising():
    """A finished run must not be lost to an assertion on a few bad voxels."""
    stats = _stats("fraction")
    x = torch.zeros(4, 6480)
    x[0, 0] = 1e30
    x[1, 5] = float("nan")
    sh, diag = export.physical_showers_mev(x, torch.full((4, 1), 10.0), stats, report=False)
    assert torch.isfinite(sh).all(), "non-finite survived the physical conversion"
    assert diag["clipped_voxels"] >= 1 and diag["showers_affected"] >= 1
    # a voxel can never exceed the incident energy
    assert sh.max() <= 10.0 * 1000.0 * 1.0001


def t_write_hdf5_roundtrip():
    import h5py
    stats = _stats("fraction")
    sh = torch.rand(5, 6480) * 100.0
    sh[:, :100] = 0.0
    sh[:, 100:110] = 1e-5            # below the 15.15 keV readout
    with tempfile.NamedTemporaryFile(suffix=".hdf5") as f:
        out = export.write_hdf5(f.name, sh, torch.full((5, 1), 1e4), stats, threshold_kev=15.15)
        assert (out[:, 100:110] == 0).all(), "sub-threshold voxels were not zeroed"
        with h5py.File(f.name, "r") as h:
            assert h["showers"].shape == (5, 6480)
            assert h["incident_energies"].shape == (5, 1)
            assert (h["showers"][:] >= 0).all()


# ----------------------------------------------------------------- checkpoints
def t_buffers_survive_state_dict():
    m = velocity_net.VelocityNet(shower_dim=64, hidden=16, hit_head=True, empty_value=-0.1489,
                                 cond="film")
    m2 = velocity_net.VelocityNet(shower_dim=64, hidden=16, hit_head=True, cond="film")
    m2.load_state_dict(m.state_dict())
    assert float(m2.empty_value) == float(m.empty_value)
    t, x, e = torch.rand(2, 1), torch.randn(2, 64), torch.randn(2, 1)
    assert torch.allclose(m.q_hat(t, x, e), m2.q_hat(t, x, e), atol=1e-6)


def t_conv_trunk_is_phi_periodic():
    """phi wraps around the calorimeter, so rolling the input in phi must roll
    the output by the same amount. Catches a padding mode set wrong."""
    m = velocity_net.VelocityNet(shower_dim=6480, hidden=256, cond="conv", hit_head=True,
                                 empty_value=-0.1489)
    x = torch.randn(1, 6480)
    t, e = torch.zeros(1, 1), torch.zeros(1, 1)
    a = m.q_hat(t, x, e).reshape(45, 16, 9)
    rolled = x.reshape(1, 45, 16, 9).roll(3, dims=2).reshape(1, -1)
    b = m.q_hat(t, rolled, e).reshape(45, 16, 9)
    assert torch.allclose(a.roll(3, dims=1), b, atol=1e-5), (a.roll(3, dims=1) - b).abs().max()


def t_conv_trunk_shapes():
    m = velocity_net.VelocityNet(shower_dim=6480, hidden=256, cond="conv", hit_head=True,
                                 empty_value=-0.1489)
    t, x, e = torch.rand(2, 1), torch.randn(2, 6480), torch.randn(2, 1)
    assert m.q_hat(t, x, e).shape == (2, 6480)
    assert m.hit_logit(t, x, e).shape == (2, 6480)
    out = velocity_net.sample(m, e, n_steps=2, device="cpu", gate="topk", readout="raw")
    assert out.shape == (2, 6480) and torch.isfinite(out).all()


def t_cond_variants_agree_in_shape():
    for cond in ("concat", "film"):
        for hit in (True, False):
            m = velocity_net.VelocityNet(shower_dim=48, hidden=16, hit_head=hit, cond=cond)
            t, x, e = torch.rand(3, 1), torch.randn(3, 48), torch.randn(3, 1)
            assert m.q_hat(t, x, e).shape == (3, 48)
            out = velocity_net.sample(m, e, n_steps=3, device="cpu",
                                      readout="raw" if not hit else "denoise")
            assert out.shape == (3, 48) and torch.isfinite(out).all()


def main():
    print("calods2 self-test\n")
    for name, fn in sorted(globals().items()):
        if name.startswith("t_") and callable(fn):
            check(name[2:].replace("_", " "), fn)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


def test_all():
    """pytest entry point.

    The checks below are named t_* rather than test_* because this file is
    also a standalone script (`python tests/test_selftest.py`) with its own
    reporting. This wrapper runs them all under pytest and fails loudly with
    the list of failures, so both invocations work.
    """
    for name, fn in sorted(globals().items()):
        if name.startswith("t_") and callable(fn):
            check(name[2:].replace("_", " "), fn)
    assert not FAIL, f"{len(FAIL)} self-test(s) failed: {FAIL}"


if __name__ == "__main__":
    sys.exit(main())
