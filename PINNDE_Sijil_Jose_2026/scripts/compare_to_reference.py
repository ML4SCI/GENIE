"""
compare_to_reference.py -- compare a generated HDF5 against the exact reference
file the official evaluation uses, on the quantities a classifier can exploit.

Motivation: every physics check in this project compared generated showers with
Dataset2.pt, while evaluate.py compares them with dataset_2_2.hdf5. If the two
files differ in units, in the incident-energy distribution, or in how empty
voxels are represented, a classifier reaches AUC ~ 1 on that alone, with no
statement about shower quality at all.

This does not train anything. It prints, for both files side by side, the
things a near-perfect AUC would most likely be exploiting:

  incident energy     range and quantiles; a different conditioning
                      distribution is directly visible to the classifier
  E_dep / E_inc       the ratio that any units error would destroy
  voxel values        quantiles of the NON-ZERO voxels, and the smallest
                      non-zero value -- a readout-threshold or rounding
                      mismatch shows up here and nowhere else
  sparsity            exact zeros per shower; generated showers whose empty
                      voxels are 1e-12 rather than 0.0 are trivially separable
  dynamic range       max voxel energy

    python compare_to_reference.py --gen generated/latent_recon_ds2.hdf5 \
        --ref dataset_2_2.hdf5
"""

import argparse

import h5py
import numpy as np


def load(path, n):
    with h5py.File(path, "r") as f:
        sh = np.array(f["showers"][:n], dtype=np.float64)
        e = np.array(f["incident_energies"][:n], dtype=np.float64).reshape(-1)
    return sh.reshape(sh.shape[0], -1), e


def describe(tag, sh, e):
    tot = sh.sum(1)
    ratio = tot / e
    nz = sh[sh > 0]
    exact_zero = (sh == 0).mean()
    tiny = ((sh > 0) & (sh < 0.01515)).mean()      # below the 15.15 keV readout
    q = [5, 50, 95]
    return {
        "showers": sh.shape[0],
        "E_inc MeV min/med/max": f"{e.min():.1f} / {np.median(e):.1f} / {e.max():.1f}",
        "E_inc log10 mean/sd": f"{np.log10(e).mean():.4f} / {np.log10(e).std():.4f}",
        "E_dep/E_inc mean/sd": f"{ratio.mean():.4f} / {ratio.std():.4f}",
        "E_dep/E_inc 5/50/95%": " / ".join(f"{v:.3f}" for v in np.percentile(ratio, q)),
        "exact zeros (fraction)": f"{exact_zero:.4f}",
        "0 < E < 15.15 keV": f"{tiny:.6f}",
        "nonzero voxels/shower": f"{(sh > 0).sum(1).mean():.0f}",
        "nonzero MeV 5/50/95%": " / ".join(f"{v:.4g}" for v in np.percentile(nz, q)),
        "smallest nonzero MeV": f"{nz.min():.6g}",
        "max voxel MeV": f"{sh.max():.4g}",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen", required=True)
    p.add_argument("--ref", required=True, help="the SAME file evaluate.py used as reference")
    p.add_argument("--n", type=int, default=20000)
    args = p.parse_args()

    g_sh, g_e = load(args.gen, args.n)
    r_sh, r_e = load(args.ref, args.n)
    g, r = describe("gen", g_sh, g_e), describe("ref", r_sh, r_e)

    print(f"\n{'':26s} {'generated':>28s} {'reference':>28s}")
    print("-" * 86)
    for k in g:
        flag = ""
        if k == "E_dep/E_inc mean/sd":
            gm = float(g[k].split(" / ")[0]); rm = float(r[k].split(" / ")[0])
            if not 0.8 < gm / rm < 1.25:
                flag = "   <== SCALE MISMATCH"
        if k == "exact zeros (fraction)" and abs(float(g[k]) - float(r[k])) > 0.05:
            flag = "   <== sparsity differs"
        if k == "smallest nonzero MeV":
            gv, rv = float(g[k]), float(r[k])
            if gv < 0.5 * rv or gv > 2.0 * rv:
                flag = "   <== threshold/rounding differs"
        print(f"{k:26s} {str(g[k]):>28s} {str(r[k]):>28s}{flag}")

    print("\nA classifier reaching AUC ~1 needs only ONE systematic difference. Rows flagged")
    print("above are trivially exploitable and would make the AUC meaningless as a measure")
    print("of shower quality. If nothing is flagged, the separation is genuine physics.")


if __name__ == "__main__":
    main()
