"""Conditional flow matching on the raw ds2 voxels.

``demo_calo`` generates the 362 summary features directly. This generates what
a shower actually is -- the **6480 voxel energies** (45 layers x 16 angular x 9
radial) -- and then computes the features from those voxels with the
CaloChallenge's own code, so the numbers land in exactly the same space, against
exactly the same measured floors.

This is the harder problem and the one that matters: an analysis never uses a
shower directly, it uses quantities derived from one, so a generator that gets
the voxels right is known to be right for anything downstream. Generating the
summaries only shows the summaries are right.

Two things follow. Every physical rule that ties features together (E_tot is the
sum of the layers, an empty layer has no centre, sparsity sits on the 1/144
lattice) holds by construction, because the features are computed from a voxel
grid rather than predicted as separate numbers. Conservation does not: see
DEVLOG section 26. And the average shower shape can be drawn with their own
``DrawAverageShower``.

The voxel grid is mostly empty: 75.1% of all voxels are exactly zero (97.2% in
the lowest energy quartile, 39.4% in the highest), so the point-mass problem
from DEVLOG section 24 is the whole problem here, 6480 columns of it. The same
treatment applies: a voxel's zero is spread over the empty band during training
and snapped back when sampling.

Run from the ``Tina`` folder with both ds2 files present:

    python -m flow_matching.demo_voxels --device cuda
    python -m flow_matching.demo_voxels --device cuda --n-train 100000 --steps 200000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from pinnde_eval import evaluate, evaluate_by_condition, report
from pinnde_eval.calochallenge import official_features

from .core import sample
from .demo_calo import (BIN_NAMES, NULL_FLOORS, FeatureTransform,
                        drop_non_finite, energy_bin_labels,
                        normalized_log_energy)
from .train import train_flow_matching

N_VOXELS = 6480
EPS = 1e-8          # the same epsilon their features use inside log10


def load_voxels(path, n, start=0):
    """Return ``(showers, e_inc)`` in MeV: (n, 6480) and (n,)."""
    import h5py
    with h5py.File(path, "r") as f:
        showers = f["showers"][start:start + n].astype(np.float64)
        e_inc = f["incident_energies"][start:start + n].astype(np.float64).ravel()
    return showers, e_inc


def to_log_voxels(showers):
    """MeV -> log10(E + 1e-8). An empty voxel becomes exactly -8."""
    return np.log10(np.asarray(showers, dtype=np.float64) + EPS)


def from_log_voxels(x):
    """log10 back to MeV, with no voxel allowed to hold negative energy."""
    return np.clip(10.0 ** np.asarray(x, dtype=np.float64) - EPS, 0.0, None)


def draw_average_showers(real, gen, e_inc, out_dir, geometry="ds2"):
    """Their own average-shower plot, once for Geant4 and once for the model."""
    from pinnde_eval.calochallenge import _high_level_features
    paths = []
    for name, data in (("geant4", real), ("flow_matching", gen)):
        hlf = _high_level_features(geometry)     # a fresh one: Draw* caches state
        path = os.path.join(out_dir, f"voxels_average_{name}.png")
        hlf.DrawAverageShower(np.asarray(data, dtype=np.float64), filename=path,
                              title=f"average shower, {name}")
        paths.append(path)
    return paths


def main(n_train=50000, n_eval=8000, n_steps=100000, hidden=1024, depth=8,
         ode_steps=200, seed=0, data_dir=None, device="cpu", lr=1e-3,
         clip_grad=1.0, atom_snap=True, save_samples=None, plot_dir=None,
         batch_size=256, atom_band=None):
    """Train and score p(6480 voxels | E_inc) on real ds2."""
    data_dir = data_dir or os.path.abspath(
        os.path.join(os.path.dirname(__file__), ".."))
    train_path = os.path.join(data_dir, "dataset_2_1.hdf5")
    eval_path = os.path.join(data_dir, "dataset_2_2.hdf5")
    for p in (train_path, eval_path):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")

    print(f"loading voxels: {n_train} train / {n_eval} eval showers")
    x_train, e_train = load_voxels(train_path, n_train)
    x_eval, e_eval = load_voxels(eval_path, n_eval)
    print(f"  {x_train.shape[1]} voxels per shower, "
          f"{100 * np.mean(x_train == 0):.1f}% of them exactly zero")

    names = [f"voxel_{i}" for i in range(x_train.shape[1])]
    tf = FeatureTransform(to_log_voxels(x_train), names, seed=seed, spacings={},
                          atoms=atom_snap, layer_consistent=False,
                          atom_band=atom_band)
    if atom_snap:
        width = ("the whole gap, about 6 log units" if not atom_band
                 else f"{atom_band} log units just below the first real value")
        print(f"  {len(tf.atoms)} voxels carry an empty-voxel atom at log10 E = -8; "
              f"in training it is spread over {width}, and snapped back when "
              f"sampling")
    z_train = tf.forward(to_log_voxels(x_train))
    c_train, lo, hi = normalized_log_energy(e_train)
    c_eval, _, _ = normalized_log_energy(e_eval, lo, hi)

    print(f"\ntraining conditional flow matching: d={z_train.shape[1]}, cond=1, "
          f"n={len(z_train)}, steps={n_steps}, hidden={hidden}, depth={depth}, "
          f"lr={lr}, clip_grad={clip_grad}")
    model, history = train_flow_matching(
        torch.tensor(z_train, dtype=torch.float32), dim=z_train.shape[1],
        cond=torch.tensor(c_train, dtype=torch.float32).reshape(-1, 1),
        n_steps=n_steps, hidden=hidden, depth=depth, lr=lr, seed=seed,
        device=device, monitor_every=max(1, n_steps // 8), clip_grad=clip_grad,
        batch_size=batch_size,
    )
    print("\nstep |      loss")
    for step, loss, _, _ in history:
        print(f"{step:5d} | {loss:9.4f}")

    z_gen = sample(model, len(x_eval), z_train.shape[1],
                   cond=torch.tensor(c_eval, dtype=torch.float32).reshape(-1, 1),
                   steps=ode_steps, device=device, seed=seed)
    gen = from_log_voxels(tf.inverse(z_gen.detach().cpu().numpy()))
    print(f"\ngenerated {len(gen)} showers; "
          f"{100 * np.mean(gen == 0):.1f}% of voxels exactly zero "
          f"(Geant4: {100 * np.mean(x_eval == 0):.1f}%)")

    # Their code turns voxels into the 362 features that everything is scored in.
    print("\ncomputing their 362 features from the generated voxels")
    f_real, feat_names = official_features(x_eval, e_eval)
    f_gen, _ = official_features(gen, e_eval)
    f_real, f_gen, _, _, dropped = drop_non_finite(f_real, f_gen, c_eval, e_eval)
    if dropped:
        print(f"WARNING: {dropped} generated showers gave non-finite features "
              f"and were dropped")

    if save_samples:
        os.makedirs(os.path.dirname(os.path.abspath(save_samples)), exist_ok=True)
        np.savez(save_samples, real=f_real, generated=f_gen,
                 e_inc=np.asarray(e_eval).ravel()[:len(f_gen)],
                 names=np.array(feat_names))
        print(f"saved features of {len(f_gen)} showers to {save_samples}")

    res = evaluate(f_real, f_gen, tier="full", seed=seed, standardize=True)
    report(res, title="voxel flow matching vs Geant4, scored in their 362 features")
    floor = NULL_FLOORS[len(feat_names)]
    print("\nagainst the measured perfect-generator floor")
    print(f"    {'metric':>9} | {'model':>10} | {'null floor':>10} | {'vs floor':>9}")
    for key in ("auc", "chi2_mean", "swd", "w1_mean", "sep_mean"):
        value, ref = float(np.ravel(res[key])[0]), floor[key]
        rel = f"{value - ref:+.4f}" if key == "auc" else f"{value / ref:.1f}x"
        print(f"    {key:>9} | {value:10.4f} | {ref:10.4f} | {rel:>9}")

    by_bin = evaluate_by_condition(f_real, f_gen, energy_bin_labels(c_eval[:len(f_gen)]),
                                   tier="full", seed=seed, standardize=True)
    print(f"\n  {'energy bin':>11} | {'n':>5} | {'auc':>15}")
    for name in BIN_NAMES:
        row = by_bin.get(name)
        if row:
            auc, err = np.ravel(row["auc"])[0], np.ravel(row["auc"])[1]
            print(f"  {name:>11} | {row['n_real']:5d} | {auc:.3f} +/- {err:.3f}")

    if plot_dir and os.path.isdir(plot_dir):
        paths = draw_average_showers(x_eval, gen, e_eval, plot_dir)
        print("\nsaved " + ", ".join(os.path.basename(p) for p in paths))
    return model, res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu, cuda or mps")
    parser.add_argument("--n-train", type=int, default=50000)
    parser.add_argument("--n-eval", type=int, default=8000)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ode-steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-atom-snap", action="store_true",
                        help="ablation: leave the 75%% of exactly-zero voxels "
                             "as an unreachable point mass")
    parser.add_argument("--atom-band", type=float, default=None,
                        help="width, in log10 units, of the band the empty-voxel "
                             "point mass is spread over during training "
                             "(default: the whole gap, ~6 units, which injects "
                             "noise worth half the variance of a sparsely lit "
                             "voxel)")
    parser.add_argument("--save-samples", default=None, metavar="PATH")
    args = parser.parse_args()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
    main(n_train=args.n_train, n_eval=args.n_eval, n_steps=args.steps,
         hidden=args.hidden, depth=args.depth, ode_steps=args.ode_steps,
         lr=args.lr, clip_grad=args.clip_grad, seed=args.seed,
         device=args.device, atom_snap=not args.no_atom_snap,
         batch_size=args.batch_size, save_samples=args.save_samples,
         atom_band=args.atom_band,
         plot_dir=out if os.path.isdir(out) else None)
