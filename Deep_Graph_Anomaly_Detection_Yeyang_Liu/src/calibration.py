"""Validation-fitted percentile calibration and event-wise structure gating."""

import numpy as np
import torch


def _sum_by_jet(values, index, count):
    return values.new_zeros(count).index_add_(0, index, values)


@torch.no_grad()
def jet_structure_features(batch):
    """Normalized participation ratio and pT-weighted RMS radius per jet.

    phi is circularly centered in graph.pos; radius uses the pT-weighted
    variance of rapidity.
    """
    index, count = batch.batch, batch.num_graphs
    pt = batch.pt.reshape(-1).clamp_min(0.0)
    fraction = pt / _sum_by_jet(pt, index, count).clamp_min(1e-12)[index]
    nodes = _sum_by_jet(torch.ones_like(pt), index, count)
    sum_squared = _sum_by_jet(fraction.square(), index, count)
    pr = 1.0 / (nodes.clamp_min(1.0) * sum_squared.clamp_min(1e-12))
    center = _sum_by_jet(fraction * batch.pos[:, 0], index, count)
    radius_squared = (batch.pos[:, 0] - center[index]).square() + batch.pos[:, 1].square()
    radius = _sum_by_jet(fraction * radius_squared, index, count).clamp_min(0.0).sqrt()
    return pr, radius


def fit_calibration(scores):
    """Fit pooled-jet median/IQR and event-score CDFs on validation only."""
    structure = dict(tau=0.5, temperature=0.25)
    for name in ('pr', 'radius'):
        q25, center, q75 = np.quantile(np.asarray(scores[name], dtype=np.float64), [.25, .5, .75])
        if q75 <= q25:
            raise ValueError(f'{name} calibration IQR must be positive')
        structure[name+'_center'] = float(center)
        structure[name+'_scale'] = float(q75-q25)
    return {
        'structure': structure,
        'baseline_reference': torch.from_numpy(np.sort(np.asarray(scores['baseline'], dtype=np.float64))),
        'regularized_reference': torch.from_numpy(np.sort(np.asarray(scores['regularized'], dtype=np.float64))),
    }


def apply_calibration(scores, calibration):
    """Blend event-score percentiles with the two jets' mean gate weight."""
    # Use the same scalar precision as the original gate implementation.
    parameters = {name: torch.as_tensor(value) for name, value in calibration['structure'].items()}
    pr = torch.as_tensor(scores['pr'], dtype=torch.float64)
    radius = torch.as_tensor(scores['radius'], dtype=torch.float64)
    standardized_pr = (pr - parameters['pr_center']) / parameters['pr_scale']
    standardized_radius = (radius - parameters['radius_center']) / parameters['radius_scale']
    chi = 0.5 * (standardized_pr + standardized_radius)
    jet_gate = torch.sigmoid((parameters['tau'] - chi) / parameters['temperature'])
    event_gate = jet_gate.mean(dim=1).numpy()

    percentiles = {}
    for name in ('baseline', 'regularized'):
        reference = calibration[name+'_reference'].numpy()
        percentiles[name] = np.searchsorted(reference, scores[name], side='right') / len(reference)
    return {
        'total': event_gate * percentiles['regularized'] + (1 - event_gate) * percentiles['baseline'],
        'gate': event_gate,
        'baseline_percentile': percentiles['baseline'],
        'regularized_percentile': percentiles['regularized'],
    }
