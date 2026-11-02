"""Event-level AUC and SIC with baseline and KS efficiency cuts."""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


SIC_MIN_BACKGROUND_EFFICIENCY = 0.001
# KS plot_utils.py: 20% background statistical error, published count 339,995.
KS_MIN_BACKGROUND_EFFICIENCY = 1 / (339_995 * 0.2**2)


def score_metrics(scores, truth):
    """Higher scores select anomalies; ties share a threshold, truth=1 is signal.

    max_sic uses epsilon_B >= 0.001; max_sic_ks uses the KS strict cut.
    A single-class sample has undefined metrics (None), not a zero AUC/SIC.
    """
    scores, truth = np.asarray(scores), np.asarray(truth)
    if scores.ndim != 1 or scores.shape != truth.shape:
        raise ValueError('Scores and truth must be matching one-dimensional arrays')
    if not np.isfinite(scores).all() or not np.isin(truth, [0, 1]).all():
        raise ValueError('Scores must be finite and truth must be 0 or 1')
    result = dict(auc=None, max_sic=None, max_sic_threshold=None,
                  max_sic_ks=None, max_sic_ks_threshold=None,
                  n_background=int((truth == 0).sum()), n_signal=int((truth == 1).sum()),
                  sic_min_background_efficiency=SIC_MIN_BACKGROUND_EFFICIENCY,
                  ks_min_background_efficiency=KS_MIN_BACKGROUND_EFFICIENCY)
    if not result['n_background'] or not result['n_signal']:
        return result

    result['auc'] = float(roc_auc_score(truth, scores))
    background_efficiency, signal_efficiency, thresholds = roc_curve(
        truth, scores, drop_intermediate=False,
    )
    selections = {
        'max_sic': background_efficiency >= SIC_MIN_BACKGROUND_EFFICIENCY,
        'max_sic_ks': background_efficiency > KS_MIN_BACKGROUND_EFFICIENCY,
    }
    for name, selected in selections.items():
        sic = signal_efficiency[selected] / np.sqrt(background_efficiency[selected])
        best = int(np.argmax(sic))
        result[name] = float(sic[best])
        result[name+'_threshold'] = float(thresholds[selected][best])
    return result
