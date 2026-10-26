"""Classical two-sample tests -- the statistics literature's answer.

The CaloChallenge metrics (``tier1.histogram_chi2``, ``tier1.separation_power``)
are histogram statistics invented for this problem. They work, but they share an
awkward property: their value under the null is not known in advance, so every
number has to be read against an empirically measured floor that depends on the
sample size and the binning (DEVLOG sections 8 and 12).

The classical two-sample tests do not have that problem. Their null
distributions have been derived and studied for decades, so each one returns a
**p-value that is Uniform(0, 1) when the two samples come from the same
distribution** -- calibrated by construction, with no floor to measure. That
makes them a useful independent check on conclusions drawn from the
CaloChallenge metrics, and it is why they are worth reporting alongside.

All of these are one-dimensional, so they are applied per observable.

* ``ks``  -- Kolmogorov-Smirnov: max gap between the two empirical CDFs.
  The standard general-purpose test; most sensitive near the centre of a
  distribution and comparatively weak in the tails.
* ``cvm`` -- Cramer-von Mises: integrates the squared CDF gap instead of taking
  its maximum, so it uses the whole distribution rather than one point.
* ``ad``  -- Anderson-Darling: like Cramer-von Mises but weighted to emphasise
  the tails, which is usually where a generative model fails first.
Anderson-Darling came out slightly ahead of the other two at every sample
size in our power check (e.g. detection rate 0.62 vs 0.50 for KS at N=2000;
see validate_classical.check_power), at the cost
that scipy clips its p-value to [0.001, 0.25] -- it can say "p <= 0.001"
but cannot quote a precise significance.

Reading them: a **small p-value means the samples differ**. Under the null the
p-values should be uniform, so roughly 5% of observables land below 0.05 by
chance -- which is why ``combine_pvalues`` is provided rather than eyeballing
the smallest one.
"""

import numpy as np
from scipy import stats

from ._utils import check_pair

# scipy's Anderson-Darling k-sample p-value is interpolated from a table and
# clipped to this range; values at either end mean "at least this extreme".
AD_PVALUE_FLOOR, AD_PVALUE_CEIL = 0.001, 0.25

TEST_NAMES = ("ks", "cvm", "ad")


def _ks(a, b):
    r = stats.ks_2samp(a, b)
    return float(r.statistic), float(r.pvalue)


def _cvm(a, b):
    r = stats.cramervonmises_2samp(a, b)
    return float(r.statistic), float(r.pvalue)


def _ad(a, b):
    # anderson_ksamp warns when the p-value is clipped; that is expected and the
    # clipping is documented in AD_PVALUE_FLOOR/CEIL, so silence just this warn.
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = stats.anderson_ksamp([a, b])
    return float(r.statistic), float(r.pvalue)


_TESTS = {"ks": _ks, "cvm": _cvm, "ad": _ad}


def two_sample_tests(real, gen, tests=("ks", "cvm", "ad"), max_samples=None,
                     seed=0):
    """Run classical two-sample tests per feature.

    Returns ``{test_name: {"stat": (d,), "pvalue": (d,)}}``.

    ``max_samples`` subsamples both inputs (seeded) before testing. These tests
    get *more* sensitive as N grows, so on 100k showers they will reject on
    physically negligible differences; capping N is how you ask "is the
    difference larger than what matters at this sample size".
    """
    real, gen = check_pair(real, gen)
    unknown = set(tests) - set(_TESTS)
    if unknown:
        raise ValueError(f"unknown tests {sorted(unknown)}; "
                         f"choose from {sorted(_TESTS)}")

    if max_samples is not None and max_samples < min(len(real), len(gen)):
        rng = np.random.default_rng(seed)
        real = real[rng.choice(len(real), max_samples, replace=False)]
        gen = gen[rng.choice(len(gen), max_samples, replace=False)]

    d = real.shape[1]
    out = {}
    for name in tests:
        fn = _TESTS[name]
        stat = np.empty(d)
        pval = np.empty(d)
        for j in range(d):
            stat[j], pval[j] = fn(real[:, j], gen[:, j])
        out[name] = {"stat": stat, "pvalue": pval}
    return out


def combine_pvalues(pvalues, method="bonferroni"):
    """Combine per-feature p-values into a single number.

    With d observables you get d p-values, and about 5% of them fall below 0.05
    under the null purely by chance. Reporting the smallest one without
    correction manufactures a false positive on most runs.

    * ``"bonferroni"`` (default) -- min(p) * d, clipped at 1. **The only one of
      these that is valid here.** It holds under arbitrary dependence between
      the features, which matters because shower observables are strongly
      correlated (E_tot/sparsity observables -0.93, and their p-values +0.94
      across independent nulls -- see validate_classical). Measured false-positive
      rate on an exact null: 0.04-0.05 against a 0.05 target.
    * ``"fisher"``     -- Fisher's method, -2 sum(log p). More powerful *if the
      p-values are independent*, which ours are not. Measured false-positive
      rate on an exact null: **0.14 to 0.29**, i.e. it rejects true nulls three
      to six times too often. Available for independent features; do not use it
      on correlated observables.
    """
    p = np.asarray(pvalues, dtype=np.float64)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return float("nan")
    if method == "fisher":
        return float(stats.combine_pvalues(np.clip(p, 1e-300, 1.0),
                                           method="fisher").pvalue)
    if method == "bonferroni":
        return float(min(p.min() * p.size, 1.0))
    raise ValueError(f"unknown method {method!r}; use bonferroni or fisher")


def pvalue_uniformity(pvalues):
    """Is this collection of p-values Uniform(0, 1), as the null predicts?

    The whole reason to use classical tests is that their null distribution is
    known: repeat a true null many times and the p-values must be uniform. This
    checks that with a one-sample KS test against U(0,1), so a *large* returned
    p-value means the tests are behaving as advertised.

    Returns ``(ks_statistic, pvalue, fraction_below_0.05)``. The last should sit
    near 0.05 under the null -- that is the false-positive rate you are buying.
    """
    p = np.asarray(pvalues, dtype=np.float64)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return float("nan"), float("nan"), float("nan")
    r = stats.kstest(p, "uniform")
    return float(r.statistic), float(r.pvalue), float((p < 0.05).mean())


def report_tests(results, names=None, title="classical two-sample tests"):
    """Print a per-observable table of statistics and p-values."""
    tests = list(results)
    d = len(results[tests[0]]["pvalue"])
    names = list(names) if names is not None else [f"feature {i}" for i in range(d)]

    width = max(len(n) for n in names)
    header = f"{'observable':>{width}} | " + " | ".join(
        f"{t + ' stat':>9} {t + ' p':>8}" for t in tests)
    print(title)
    print("-" * len(header))
    print(header)
    for j in range(d):
        row = f"{names[j]:>{width}} | " + " | ".join(
            f"{results[t]['stat'][j]:9.4g} {results[t]['pvalue'][j]:8.4f}"
            for t in tests)
        print(row)
    print()
    for t in tests:
        p = results[t]["pvalue"]
        print(f"{t:>4}: Fisher combined p = {combine_pvalues(p):.4g}   "
              f"Bonferroni = {combine_pvalues(p, 'bonferroni'):.4g}")
