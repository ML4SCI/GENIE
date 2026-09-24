"""pinnde_eval -- shared quantitative evaluation for the PINNDE tracks.

Three tiers of metrics behind one entry point:

* Tier 1 (CaloChallenge-standard): classifier two-sample AUC, per-feature
  histogram reduced chi-squared.
* Tier 2 (headline distances): FPD, KPD, per-feature Wasserstein-1.
* Tier 3 (cheap training-loop monitors): RBF-MMD, sliced Wasserstein.

    from pinnde_eval import evaluate, evaluate_by_condition, report
    results = evaluate(real, gen, tier="full")
    report(results)

Tier 3 metrics are also exposed directly (``mmd``, ``swd``) for use inside a
training loop.

Two study/diagnostic layers sit on top of the metrics:

* ``stability`` -- null floor, spread, and resolvability of every metric as a
  function of the sample size N (``python -m pinnde_eval.stability``).
* ``local`` -- local discrepancy maps (``mmd_witness``,
  ``classifier_discrepancy``, ``binned_residual_map``) that localize *where*
  two distributions disagree instead of returning one global number.
"""

from .calochallenge import official_features, official_features_from_file
from .classical import (combine_pvalues, pvalue_uniformity, report_tests,
                        two_sample_tests)
from .evaluate import evaluate, evaluate_by_condition, report, plot_histograms
from .floors import (match_by_energy, pairs_across_files, pairs_matched_energy,
                     pairs_within_file, score_pairs)
from .local import binned_residual_map, classifier_discrepancy, mmd_witness
from .observables import (Geometry, GEOMETRIES, READOUT_THRESHOLD_MEV,
                          discrete_observables, layer_energies,
                          load_calochallenge, observables_from_file,
                          per_layer_observables, radial_profile,
                          shower_features_fn, shower_observables,
                          voxel_energy_spectrum)
from .stability import min_resolvable_n, separation_z, stability_study
from .tier1 import classifier_two_sample_test, histogram_chi2, separation_power
from .tier2 import fpd, kpd, wasserstein_per_feature
from .tier3 import mmd, swd, median_bandwidth, sinkhorn

__all__ = [
    "evaluate",
    "evaluate_by_condition",
    "report",
    "plot_histograms",
    "classifier_two_sample_test",
    "histogram_chi2",
    "separation_power",
    "fpd",
    "kpd",
    "wasserstein_per_feature",
    "mmd",
    "swd",
    "sinkhorn",
    "median_bandwidth",
    "official_features",
    "official_features_from_file",
    "pairs_across_files",
    "pairs_within_file",
    "pairs_matched_energy",
    "match_by_energy",
    "score_pairs",
    "two_sample_tests",
    "combine_pvalues",
    "pvalue_uniformity",
    "report_tests",
    "stability_study",
    "separation_z",
    "min_resolvable_n",
    "mmd_witness",
    "classifier_discrepancy",
    "binned_residual_map",
    "Geometry",
    "GEOMETRIES",
    "READOUT_THRESHOLD_MEV",
    "shower_observables",
    "per_layer_observables",
    "discrete_observables",
    "observables_from_file",
    "shower_features_fn",
    "layer_energies",
    "radial_profile",
    "voxel_energy_spectrum",
    "load_calochallenge",
]
