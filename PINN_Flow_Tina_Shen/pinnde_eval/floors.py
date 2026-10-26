"""Floors measured over disjoint repeats, in whatever feature space is given.

A metric value means nothing on its own: two independent Geant4 samples do not
score AUC exactly 0.5, and every distance sits above zero at finite sample size.
The floor is what a *perfect* generator scores, so it is the only thing a model
can be read against.

Two kinds of pair are worth measuring, and the difference between them is
informative:

* **inside one file** -- both samples come from the same generator, so they
  differ only by sampling. This is an exact null.
* **across the two files** -- the floor a model is scored against, which also
  carries anything that genuinely differs between the files.

Every floor quoted before 2026-09-16 came from a *single* comparison, with the
AUC error bar coming from retraining the classifier on that one pair. That
measures classifier noise; the spread across disjoint pairs, which is what these
helpers produce, measures sampling noise, and the two are not interchangeable.

Disjoint matters: reusing showers across repeats correlates the numbers and
makes the spread look smaller than it is.
"""

import time

import numpy as np

from .evaluate import evaluate

METRICS = ("auc", "chi2_mean", "swd", "w1_mean", "sep_mean", "sinkhorn",
           "mmd", "fpd", "kpd")


def pairs_across_files(a, b, n, repeats):
    """``repeats`` disjoint pairs, one sample from each file."""
    if repeats * n > min(len(a), len(b)):
        raise ValueError(f"{repeats} x {n} needs more showers than the "
                         f"{min(len(a), len(b))} available per file")
    return [(a[r * n:(r + 1) * n], b[r * n:(r + 1) * n]) for r in range(repeats)]


def pairs_within_file(a, b, n, repeats):
    """``repeats`` disjoint pairs, both samples from the same file."""
    out = []
    for source in (a, b):
        for k in range((repeats + 1) // 2):
            lo = 2 * k * n
            if lo + 2 * n <= len(source):
                out.append((source[lo:lo + n], source[lo + n:lo + 2 * n]))
    if len(out) < repeats:
        raise ValueError(f"{repeats} same-file pairs of {n} need more showers "
                         f"than the {min(len(a), len(b))} available per file")
    return out[:repeats]


def score_pairs(pairs, tier="full", seed=0, label="", n_classifier=5,
                quiet=False):
    """Run ``evaluate`` on each pair. Returns ``{metric: one value per pair}``."""
    collected = {m: [] for m in METRICS}
    for r, (real, other) in enumerate(pairs):
        start = time.time()
        res = evaluate(real, other, tier=tier, seed=seed + r, standardize=True,
                       n_classifier=n_classifier)
        for m in METRICS:
            value = res.get(m)
            if isinstance(value, tuple):          # auc, fpd, kpd carry an error
                value = value[0]
            collected[m].append(np.nan if value is None else float(value))
        if not quiet:
            print(f"  {label}{r + 1}/{len(pairs)}: auc {collected['auc'][-1]:.4f}, "
                  f"swd {collected['swd'][-1]:.4f} ({time.time() - start:.0f} s)")
    return {m: np.array(v) for m, v in collected.items()}


def mean_and_spread(values):
    """``(mean, standard deviation)`` ignoring metrics that were unavailable."""
    if np.all(np.isnan(values)):
        return float("nan"), float("nan")
    return float(np.nanmean(values)), float(np.nanstd(values))


def match_by_energy(target_log_e, pool_log_e, order=None, taken=None):
    """Indices into the pool whose incident energies match the targets.

    Greedy nearest match without reuse: each target takes the closest pool
    shower still free. Returns ``(indices, worst |difference| in log E)``.

    ``order`` and ``taken`` let several calls share one pool without reuse
    across them, which is what disjoint repeats need.

    Why this exists: a conditional model generates at the evaluation set's own
    incident energies, so its sample and the real one share energies exactly. Two
    independent Geant4 samples do not -- they differ by the energy draw as well,
    which is variance the model never pays. Scoring a model against an unmatched
    floor is therefore mildly generous to the model, and at a floor this tight
    the effect is not negligible: it is what lets a model score "better than
    Geant4". Matching the energies removes it.
    """
    pool_log_e = np.asarray(pool_log_e).ravel()
    order = np.argsort(pool_log_e) if order is None else order
    ordered = pool_log_e[order]
    taken = np.zeros(len(ordered), dtype=bool) if taken is None else taken
    picks, worst = np.empty(len(target_log_e), dtype=int), 0.0
    for i, value in enumerate(np.asarray(target_log_e).ravel()):
        start = int(np.searchsorted(ordered, value))
        lo, hi, best = start - 1, start, -1
        while best < 0:
            options = [k for k in (hi, lo)
                       if 0 <= k < len(ordered) and not taken[k]]
            if options:
                best = min(options, key=lambda k: abs(ordered[k] - value))
            elif lo < 0 and hi >= len(ordered):
                raise ValueError("pool exhausted while matching energies")
            else:
                lo, hi = lo - 1, hi + 1
        taken[best] = True
        picks[i] = order[best]
        worst = max(worst, abs(ordered[best] - value))
    return picks, worst


def pairs_matched_energy(target, target_log_e, pool, pool_log_e, n, repeats):
    """Disjoint pairs whose incident energies agree shower by shower.

    One side is taken in disjoint blocks from ``target``; its partner is drawn
    from ``pool`` by nearest energy, without reuse inside a pair.
    """
    pool_log_e = np.asarray(pool_log_e).ravel()
    order = np.argsort(pool_log_e)                 # shared, so no shower is
    taken = np.zeros(len(pool_log_e), dtype=bool)  # used by two repeats
    out, worst = [], 0.0
    for r in range(repeats):
        block = slice(r * n, (r + 1) * n)
        picks, far = match_by_energy(np.asarray(target_log_e).ravel()[block],
                                     pool_log_e, order=order, taken=taken)
        out.append((target[block], pool[picks]))
        worst = max(worst, far)
    print(f"  energies matched to within {worst:.2e} in log E_inc "
          f"(the two files' own difference is 1.8e-02)")
    return out


def save(path, within, across, n, repeats, space):
    """Store the per-repeat numbers, so a table can be redone without rescoring.

    Scoring ten pairs costs minutes to half an hour; rereading a table should
    not. It also means a quoted floor can be traced back to the runs behind it.
    """
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, n=n, repeats=repeats, space=space,
             **{f"within_{m}": within[m] for m in METRICS},
             **{f"across_{m}": across[m] for m in METRICS})
    print(f"\nper-repeat numbers saved to {path}")


def gap_in_errors(within_values, across_values):
    """``(difference, standard error)`` between the two floors for one metric."""
    in_mean, in_spread = mean_and_spread(within_values)
    ac_mean, ac_spread = mean_and_spread(across_values)
    error = np.sqrt(in_spread ** 2 / len(within_values)
                    + ac_spread ** 2 / len(across_values))
    return ac_mean - in_mean, error


def report(within, across, n, repeats, space=""):
    """Both floors side by side, with the spread over repeats and their gap.

    The gap column is given for **every** metric on purpose. Reading only the
    one that suits the conclusion is how a difference gets talked out of
    existence; and with nine metrics, one of them landing two standard errors
    from zero is ordinary.
    """
    print(f"\nfloors over {repeats} disjoint pairs of {n} showers each"
          + (f", {space}" if space else "") + ", standardized")
    print("same file = sampling noise only; different files = what a model is "
          "scored against")
    print("gap = (different files) - (same file), in standard errors of that "
          "difference\n")
    print(f"{'metric':>10} | {'same file':>19} | {'different files':>19} | {'gap':>8}")
    gaps = {}
    for metric in METRICS:
        cells = []
        for values in (within[metric], across[metric]):
            mean, spread = mean_and_spread(values)
            cells.append(f"{'n/a':>19}" if np.isnan(mean)
                         else f"{mean:10.5f} +/-{spread:7.5f}")
        gap, error = gap_in_errors(within[metric], across[metric])
        gaps[metric] = (gap, error)
        shown = "     n/a" if (np.isnan(gap) or error == 0) else f"{gap / error:+8.1f}"
        print(f"{metric:>10} | {cells[0]} | {cells[1]} | {shown}")

    worst = max((m for m in METRICS if np.isfinite(gaps[m][1]) and gaps[m][1] > 0),
                key=lambda m: abs(gaps[m][0] / gaps[m][1]), default=None)
    if worst:
        ratio = gaps[worst][0] / gaps[worst][1]
        print(f"\nLargest gap: {worst} at {ratio:+.1f} standard errors. Above "
              f"about 3 that is a real\ndifference between the files in this "
              f"space; below it, with nine metrics looked at,\nit is what "
              f"chance produces.")
    return gaps
