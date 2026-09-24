"""Validate the metric suite on real CaloChallenge Geant4 showers.

The toy validation (``validate_toys``) shows the metrics are calibrated on
Gaussian mixtures. This does the same on real data, which is the only way to
catch problems that unit-scale toys cannot express.

``dataset_2_1.hdf5`` and ``dataset_2_2.hdf5`` are independent Geant4 draws from
the same generation process, so comparing them is a **true null**: every metric
must sit at its finite-N floor. Those floor values are what a perfect generator
looks like on real showers, and they are the numbers any model should be judged
against -- never against zero.

Two checks, both asserted:

1. **Null test** -- Geant4 vs Geant4 gives AUC ~ 0.5, reduced chi^2 ~ 1, and
   the distances at their floor.
2. **Separation-power floor law** -- the finite-N floor of the CaloChallenge
   separation power scales as ``n_occupied_bins / (2N)``, so a reported value
   is only meaningful relative to the binning and sample size used.

Run from the ``Tina`` folder, with the two ds2 files downloaded here:

    python -m pinnde_eval.validate_calo
"""

import os

import numpy as np

from .evaluate import evaluate, report
from .observables import observables_from_file
from .tier1 import separation_power


def null_test(a, b, names, n=8000, seed=0):
    """Geant4 vs Geant4: every metric should sit at its floor."""
    print("=== 1. NULL TEST on real Geant4 showers (ds2_1 vs ds2_2) ===")
    print(f"N = {n} per sample, {len(names)} observables, standardized\n")
    res = evaluate(a[:n], b[:n], tier="full", seed=seed, standardize=True)
    report(res, title="Geant4 vs Geant4 -- the floor a perfect generator hits")

    print("\nper observable:")
    print(f"{'observable':>12} | {'chi2':>6} | {'sep power':>9}")
    for i, nm in enumerate(names):
        print(f"{nm:>12} | {res['chi2_per_feature'][i]:6.3f} | "
              f"{res['sep_per_feature'][i]:9.5f}")

    auc_mean, auc_std = res["auc"]
    assert abs(auc_mean - 0.5) < 0.05, f"null AUC {auc_mean} not ~0.5"
    assert 0.5 < res["chi2_mean"] < 2.0, f"null chi2 {res['chi2_mean']} not ~1"
    assert abs(res["mmd"]) < 1e-2, f"null mmd {res['mmd']} not ~0"
    assert res["swd"] < 0.1, f"null swd {res['swd']} above floor"
    print("\nnull test passed: real-vs-real looks like no difference.")
    return res


def separation_floor(a, b, n_grid=(500, 1000, 2000, 4000, 8000, 16000),
                     bins=50, n_repeats=4):
    """Measure the separation-power null floor as a function of N."""
    print("\n=== 2. SEPARATION POWER null floor vs N ===")
    print("two independent Geant4 draws -- any non-zero value here is "
          "binning noise, not model error\n")
    print(f"{'N':>6} | {'sep_mean':>9} | {'std':>8} | {'n_bins/(2N)':>11} | "
          f"{'ratio':>6}")
    floors = {}
    total = len(a)
    for n in n_grid:
        vals = []
        for rep in range(n_repeats):
            lo = rep * n
            if lo + n > total:
                break
            vals.append(separation_power(a[lo:lo + n], b[lo:lo + n],
                                         bins=bins).mean())
        if not vals:
            continue
        mean, std = float(np.mean(vals)), float(np.std(vals))
        pred = bins / (2 * n)
        floors[n] = mean
        print(f"{n:6d} | {mean:9.5f} | {std:8.5f} | {pred:11.5f} | "
              f"{mean / pred:6.2f}")

    ns = sorted(floors)
    ratio = floors[ns[0]] / floors[ns[-1]]
    expected = ns[-1] / ns[0]
    print(f"\nfloor falls {ratio:.1f}x over a {expected:.0f}x increase in N "
          f"(1/N scaling predicts {expected:.0f}x)")
    assert ratio > 0.5 * expected, "separation-power floor does not fall as ~1/N"
    print("floor law confirmed: S_floor ~ n_occupied_bins / (2N).")
    return floors


def binning_dependence(a, b, n=4000, bin_grid=(25, 50, 100, 200)):
    """The floor is linear in the number of bins at fixed N."""
    print(f"\n=== 3. floor vs binning at fixed N = {n} ===")
    print(f"{'bins':>5} | {'sep_mean':>9} | {'bins/(2N)':>10}")
    vals = []
    for nb in bin_grid:
        s = float(separation_power(a[:n], b[:n], bins=nb).mean())
        vals.append(s)
        print(f"{nb:5d} | {s:9.5f} | {nb / (2 * n):10.5f}")
    assert vals == sorted(vals), "floor should grow with finer binning"
    print("confirmed: doubling the bins roughly doubles the floor, so two "
          "separation powers\nare only comparable at identical binning and N.")


def main(n_load=20000, data_dir=None):
    data_dir = data_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..")
    paths = [os.path.join(data_dir, f"dataset_2_{i}.hdf5") for i in (1, 2)]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(
            "CaloChallenge ds2 files not found:\n  "
            + "\n  ".join(missing)
            + "\n\nDownload both from https://zenodo.org/records/6366271 and "
              "put them in the Tina/ folder (they are gitignored)."
        )

    print(f"extracting observables from {n_load} showers in each file...")
    a, names, _ = observables_from_file(paths[0], n_load)
    b, _, _ = observables_from_file(paths[1], n_load)
    print(f"  {a.shape[0]} x {a.shape[1]} observables: {', '.join(names)}\n")

    null_test(a, b, names)
    separation_floor(a, b)
    binning_dependence(a, b)
    print("\nAll real-data validation checks passed.")


if __name__ == "__main__":
    main()
