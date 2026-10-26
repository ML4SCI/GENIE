"""Are the classical two-sample tests calibrated, and is our null a real null?

The CaloChallenge metrics need a measured floor: separation power reads
``n_bins/(2N)`` for a perfect generator (DEVLOG section 12). Classical tests
promise to avoid that -- their null distributions are known, so a p-value is
Uniform(0, 1) whenever the two samples match, at any N and with no calibration.

Two questions, and it matters that they are separate:

1. **Are the tests calibrated?** Checked with a permutation null: pool the two
   samples and split them at random. That is an exact null by construction, so
   any deviation from uniform is a fault in the test, not in the data.

2. **Is Geant4-vs-Geant4 exactly a null?** Not by construction: the two ds2
   files drew their incident energies independently, so their energy
   distributions differ by sampling noise, and observables that depend on
   energy inherit that. On all 100,000 showers per file the difference in mean
   log E_inc is about 2 sigma (KS on E_inc itself p ~ 0.10), which is within
   what two independent draws produce by chance. A file-vs-file null floor is
   therefore at most slightly generous; worth knowing before quoting one.

Run from the ``Tina`` folder with both ds2 files present:

    python -m pinnde_eval.validate_classical
"""

import os

import numpy as np
from scipy import stats

from .classical import pvalue_uniformity, two_sample_tests
from .observables import observables_from_file

TESTS = ("ks", "cvm", "ad")


def permutation_null(x, y, n_splits=40, tests=TESTS, seed=0):
    """Pool both samples and re-split at random: an exact null.

    Returns ``{test: pooled p-values}``. Under a correct test these are
    Uniform(0, 1) no matter what the underlying data looks like.
    """
    rng = np.random.default_rng(seed)
    pooled = np.vstack([x, y])
    half = len(pooled) // 2
    out = {t: [] for t in tests}
    for _ in range(n_splits):
        idx = rng.permutation(len(pooled))
        a, b = pooled[idx[:half]], pooled[idx[half:2 * half]]
        res = two_sample_tests(a, b, tests=tests)
        for t in tests:
            out[t].append(res[t]["pvalue"])
    return {t: np.concatenate(v) for t, v in out.items()}


def check_calibration(x, y, names, n_splits=100, chunk=1000, tests=TESTS,
                      seed=0):
    """Permutation null: the tests must produce uniform p-values.

    Checked **per observable**, not pooled. Pooling is tempting -- 60 splits x
    7 observables gives 420 p-values instead of 60 -- but the seven observables
    are computed from the same showers and are strongly correlated (E_tot and
    sparsity at -0.93 as observables, and their p-values at +0.94 -- measured
    in check_pvalue_independence), so those 420 values are far from independent.
    The uniformity test would then be over-confident and reject a perfectly
    good test. There are only ~60 independent groups, and each observable is
    used on its own anyway, so per observable is both the honest check and the
    one that matches how the tests are applied.
    """
    print("=== 1. ARE THE TESTS CALIBRATED? (permutation null) ===")
    print(f"pool both samples and cut them into disjoint {chunk}+{chunk}")
    print("splits. An exact null by construction, so the rejection rate here")
    print("is the test's true false-positive rate. It should sit near 0.05.\n")

    # Splits must be DISJOINT. Drawing each split independently from the pool
    # reuses showers across splits, so the p-values become correlated and the
    # uniformity check -- which assumes independence -- starts inventing
    # differences between tests that more sampling does not reproduce.
    rng = np.random.default_rng(seed)
    pool = np.vstack([x, y])
    order = rng.permutation(len(pool))
    n_splits = min(n_splits, len(pool) // (2 * chunk))
    collected = {t: [] for t in tests}
    for i in range(n_splits):
        blk = order[i * 2 * chunk:(i + 1) * 2 * chunk]
        res = two_sample_tests(pool[blk[:chunk]], pool[blk[chunk:]], tests=tests)
        for t in tests:
            collected[t].append(res[t]["pvalue"])
    collected = {t: np.array(v) for t, v in collected.items()}
    print(f"({n_splits} disjoint splits, using {n_splits * 2 * chunk} "
          f"of {len(pool)} showers)\n")

    header = f"{'observable':>12} | " + " | ".join(
        f"{t + ' frac p<.05':>15}" for t in tests)
    print(header + f"   (target {0.05})")
    for j, nm in enumerate(names):
        cells = " | ".join(f"{pvalue_uniformity(collected[t][:, j])[2]:15.3f}"
                           for t in tests)
        print(f"{nm:>12} | {cells}")

    # Assert on the false-positive RATE, which is the operationally meaningful
    # quantity and far less noisy than a uniformity p-value computed from only
    # ~100 samples. A uniformity p-value is itself a random draw; running 7
    # observables x len(tests) of them guarantees a low one now and then.
    for t in tests:
        for j, nm in enumerate(names):
            frac = pvalue_uniformity(collected[t][:, j])[2]
            assert frac < 0.15, f"{t} rejects {frac:.1%} of true nulls on {nm}"
    print("\ncalibrated: every test holds a ~5% false-positive rate on every")
    print("observable, including the discrete one (sparsity).")
    return collected


def check_files_are_not_a_perfect_null(a, b, names, e_a, e_b):
    """The two ds2 files differ, because their incident energies differ."""
    print("\n=== 2. IS GEANT4-vs-GEANT4 A PERFECT NULL? (no) ===")
    la, lb = np.log(e_a), np.log(e_b)
    se = np.sqrt(la.var() / len(la) + lb.var() / len(lb))
    n_sigma = (lb.mean() - la.mean()) / se
    print(f"mean log E_inc differs by {n_sigma:+.2f} sigma between the files "
          f"({la.mean():.5f} vs {lb.mean():.5f}).")
    print("Each file drew its own incident energies, so this is ordinary")
    print("sampling noise -- but the energy-dependent observables inherit it.\n")

    # Compare all-energy against a narrow matched energy band.
    lo, hi = np.quantile(np.r_[la, lb], [0.45, 0.55])
    ma, mb = (la >= lo) & (la < hi), (lb >= lo) & (lb < hi)
    print(f"{'observable':>12} | {'corr w/ logE':>12} | {'KS p, all E':>11} | "
          f"{'KS p, matched E':>15}")
    for j, nm in enumerate(names):
        corr = np.corrcoef(la, a[:, j])[0, 1]
        full = stats.ks_2samp(a[:, j], b[:, j]).pvalue
        band = stats.ks_2samp(a[ma, j], b[mb, j]).pvalue
        print(f"{nm:>12} | {corr:+12.3f} | {full:11.4g} | {band:15.4g}")

    print("\nA ~2 sigma difference between two independent random draws is")
    print("expected about 1 time in 20, so this is consistent with sampling")
    print("noise. The matched-band column uses only ~10% of the showers, so it")
    print("is noisier than the all-E column; read it as a rough check, not proof.")
    print("Consequence: a file-vs-file null floor carries this extra E_inc")
    print("variance, so it is at most slightly generous to a model that")
    print("generates at the evaluation set's own energies.")


def check_power(a, log_e, n_grid=(250, 500, 1000, 2000), shift=0.05,
                n_repeats=50, seed=0):
    """A controlled shift must be detected, and more strongly as N grows.

    The perturbation is applied **inside a narrow incident-energy band**, and
    scaled by each observable's spread *within that band*. That is the only
    scale with a physical meaning here: E_tot's marginal spread is set by the
    1 GeV - 1 TeV energy range rather than by shower physics, so a fraction of
    its marginal std or IQR is a huge perturbation dressed up as a small one
    (its IQR is 5.6x its own median). At fixed energy each observable is
    unimodal and "5% of the shower-to-shower spread" means what it sounds like.
    """
    from .classical import combine_pvalues
    from .tier1 import separation_power

    lo, hi = np.quantile(log_e, [0.40, 0.60])
    band = a[(log_e >= lo) & (log_e < hi)]
    scale = band.std(axis=0)
    print(f"\n=== 3. POWER: shift by {shift} of the spread at fixed energy ===")
    print(f"    inside a narrow E_inc band ({len(band)} showers), where the")
    print(f"    observables are unimodal and a fractional shift is meaningful\n")
    # A single comparison gives one random p-value, so the curve would wobble.
    # Power is the *rate* of detection: repeat and count rejections at 5%.
    rng = np.random.default_rng(seed)
    print(f"{'N':>6} | {'KS power':>9} | {'CvM power':>9} | {'AD power':>9} | "
          f"{'sep power':>10} | {'sep floor':>9}")
    for n in n_grid:
        if 2 * n > len(band):
            break
        ks_hits, cvm_hits, ad_hits, seps = 0, 0, 0, []
        for _ in range(n_repeats):
            idx = rng.choice(len(band), 2 * n, replace=False)
            real, gen = band[idx[:n]], band[idx[n:]] + shift * scale
            res = two_sample_tests(real, gen, tests=("ks", "cvm", "ad"))
            ks_hits += combine_pvalues(res["ks"]["pvalue"]) < 0.05
            cvm_hits += combine_pvalues(res["cvm"]["pvalue"]) < 0.05
            ad_hits += combine_pvalues(res["ad"]["pvalue"]) < 0.05
            seps.append(separation_power(real, gen).mean())
        print(f"{n:6d} | {ks_hits / n_repeats:9.2f} | "
              f"{cvm_hits / n_repeats:9.2f} | {ad_hits / n_repeats:9.2f} | "
              f"{np.mean(seps):10.5f} | {50 / (2 * n):9.5f}")
    print(f"\n'power' = fraction of {n_repeats} repeats detecting the shift at the")
    print(f"5% level, after Bonferroni across the 7 observables. Standard error on")
    print(f"each rate is at most {0.5 / np.sqrt(n_repeats):.2f}.")
    print("The classical tests give a detection rate directly. Separation power")
    print("gives a number that only means something next to its own floor --")
    print("the last two columns are the same order of magnitude throughout,")
    print("which is why an uncalibrated separation power is hard to act on.")


def check_pvalue_independence(x, y, names, chunk=1000, seed=0):
    """Are the per-observable p-values independent? (No.)

    This is the assumption Fisher's method makes when it pools p-values, so it
    is worth measuring rather than asserting. Two things get measured:

    1. correlation between the **observables** themselves, and
    2. correlation between their **p-values** across many independent nulls --
       which is the quantity Fisher actually needs to be zero.

    They are related but not the same: two observables locked together produce
    p-values that move together regardless of the sign of the correlation.
    """
    print("\n=== 4. ARE THE PER-OBSERVABLE P-VALUES INDEPENDENT? ===")
    pool = np.vstack([x, y])
    d = len(names)
    iu = np.triu_indices(d, 1)

    obs_c = np.corrcoef(pool, rowvar=False)
    k = int(np.argmax(np.abs(obs_c[iu])))
    print(f"\ncorrelation between observables (pooled):")
    print(f"  strongest pair : {names[iu[0][k]]} / {names[iu[1][k]]} = "
          f"{obs_c[iu][k]:+.3f}")
    print(f"  mean |corr| over the {len(iu[0])} pairs = "
          f"{np.abs(obs_c[iu]).mean():.3f}")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pool))
    n_split = len(pool) // (2 * chunk)
    P = []
    for i in range(n_split):
        blk = order[i * 2 * chunk:(i + 1) * 2 * chunk]
        P.append(two_sample_tests(pool[blk[:chunk]], pool[blk[chunk:]],
                                  tests=("ks",))["ks"]["pvalue"])
    P = np.array(P)
    p_c = np.corrcoef(P, rowvar=False)
    band = 1.96 / np.sqrt(n_split)
    n_out = int((np.abs(p_c[iu]) > band).sum())
    kp = int(np.argmax(np.abs(p_c[iu])))

    print(f"\ncorrelation between the p-values ({n_split} disjoint nulls):")
    print(f"  strongest pair : {names[iu[0][kp]]} / {names[iu[1][kp]]} = "
          f"{p_c[iu][kp]:+.3f}")
    print(f"  mean |corr|    : {np.abs(p_c[iu]).mean():.3f}")
    print(f"  if independent, 95% of pairs should fall inside +/-{band:.3f}")
    print(f"  pairs outside that band: {n_out} of {len(iu[0])}")

    assert n_out > len(iu[0]) // 4, "expected the p-values to be correlated"
    print("\nThe p-values are clearly not independent, so Fisher's method does")
    print("not apply. combine_pvalues defaults to Bonferroni, which holds under")
    print("arbitrary dependence.")
    return obs_c, p_c


def main(n_load=100000, data_dir=None):
    data_dir = data_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..")
    paths = [os.path.join(data_dir, f"dataset_2_{i}.hdf5") for i in (1, 2)]
    for p in paths:
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}\nDownload ds2 from "
                             f"https://zenodo.org/records/6366271")

    print(f"extracting observables from {n_load} showers in each file "
          f"(a few minutes)...\n")
    a, names, e_a = observables_from_file(paths[0], n_load)
    b, _, e_b = observables_from_file(paths[1], n_load)

    check_calibration(a, b, names)
    check_files_are_not_a_perfect_null(a, b, names, e_a, e_b)
    check_power(a, np.log(e_a))
    check_pvalue_independence(a, b, names)
    print("\nAll classical-test validation checks passed.")


if __name__ == "__main__":
    main()
