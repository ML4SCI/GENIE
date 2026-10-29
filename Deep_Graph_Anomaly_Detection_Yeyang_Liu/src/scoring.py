"""Jet anomaly scores combined by sum, or geometric scoring for max/bilinear NetGe."""

import torch

from .calibration import jet_structure_features
from .dataset import combine_jets
from .losses import cycle_errors, netge_errors, reconstruction_errors
from .models import GLADC, NetGe, NetGeJet, NetGeMaxBilinear, SDMATII, SDMEnergy, SDMATI, SDMNAT


def scoring_options(model_name, *, variant='original'):
    """Default score and available alternatives for each model."""
    if model_name == 'sdm_ati':
        if variant == 'original':
            return 'critic', ('total', 'critic')
        return 'energy', ('total', 'energy', 'geometric')
    if model_name == 'sdm_nat':
        if variant == 'control':
            return 'reconstruction', ('total', 'reconstruction')
        default = 'full_objective' if variant == 'paired' else 'classifier'
        return default, ('total', 'classifier', 'full_objective')
    if model_name == 'sdm_atii':
        return 'critic', ('total', 'critic')
    if model_name == 'sdm_energy':
        return 'generator_objective', ('total', 'energy', 'generator', 'generator_objective')
    if model_name == 'edgeconv':
        return 'total', ('total', 'node', 'edge')
    if model_name == 'netge' and variant == 'max_bilinear':
        return 'geometric', ('total', 'node', 'edge', 'reconstruction', 'geometric')
    if model_name == 'netge' and variant in ('jet', 'fraction'):
        names = ('total', 'node', 'edge', 'reconstruction')
        if variant == 'jet':
            names += ('cycle', 'node_cycle', 'graph_cycle', 'reconstruction_cycle')
        return 'reconstruction', names
    names = ('total', 'node', 'cycle', 'node_cycle', 'graph_cycle')
    if model_name == 'netge':
        names += ('attribute_l2', 'structure')
    if model_name == 'gladc':
        names += ('edge', 'node_edge', 'node_edge_cycle')
    return ('cycle' if model_name == 'netge' else 'node_edge'), names


def _jet_scores(model, jets, edge_weight, score=None):
    if isinstance(model, SDMATI):
        name = 'critic' if model.variant == 'original' else 'energy'
        value = model.score(jets)
        return {name: value, 'total': value}
    if isinstance(model, SDMEnergy):
        if score in (None, 'total'):
            score = 'generator_objective'
        if score in ('generator', 'generator_objective'):
            generated_score = model.generator_score(jets, full_objective=score == 'generator_objective')
            return {score: generated_score, 'total': generated_score}
        energy = model.critic_score(jets)
        return dict(energy=energy, total=energy)
    if isinstance(model, SDMATII):
        # Real graphs are trained toward higher critic outputs; no sigmoid.
        score = model.score(jets)
        return dict(critic=score, total=score)
    if isinstance(model, NetGeJet):
        errors = model.scores(model.reconstruct(jets))
        return dict(errors, total=errors['reconstruction'])
    if isinstance(model, NetGe):
        output = model.reconstruct(jets)
        terms = netge_errors(output)
        x, _, _, mask = output['inputs']
        node = ((output['node'] - x).square().mean(-1) * mask).sum(1) / mask.sum(1)
        attribute_l2 = terms.pop('attribute')
        errors = dict(node=node, attribute_l2=attribute_l2, **terms)
    else:
        if isinstance(model, GLADC):
            jets = model.prepare(jets)
        output = model(jets.x, jets.edge_index, jets.edge_attr)
        errors = reconstruction_errors(output, jets, edge_weight=edge_weight)
        if not isinstance(model, GLADC):
            return errors
        errors.update(cycle_errors(model, output, jets))
    errors['cycle'] = errors['node_cycle'] + errors['graph_cycle']
    if 'edge' in errors:
        errors['node_edge'] = errors['node'] + edge_weight * errors['edge']
        errors['node_edge_cycle'] = errors['node_edge'] + errors['cycle']
    errors['total'] = errors['cycle'] if isinstance(model, NetGe) else errors['node_edge']
    return errors


@torch.no_grad()
def score_events(model, batch, *, edge_weight=1.0, score=None):
    """Return event scores; total is the selected anomaly score.

    It follows the model default when score is None, not the training objective.
    Call model.eval() first. Each score vector follows the event order in
    batch['jets'], matching batch['source'] and batch['row'].
    """
    if isinstance(model, SDMNAT):
        result = model.event_scores(combine_jets(batch['jets']))
        default = scoring_options('sdm_nat', variant=model.variant)[0]
        result['total'] = result[default if score in (None, 'total') else score]
        return result
    errors = []
    for jets in batch['jets']:
        errors.append(_jet_scores(model, jets, edge_weight, score))
    leading, subleading = errors
    # Match the baseline's float64 event-sum accumulation.
    result = {name: leading[name].double() + subleading[name].double() for name in leading}
    if isinstance(model, SDMATI) and model.variant != 'original':
        result['geometric'] = 2 * leading['energy'].double().sqrt() * subleading['energy'].double().sqrt()
    if isinstance(model, NetGeMaxBilinear):
        first, second = leading['reconstruction'].double(), subleading['reconstruction'].double()
        result['jet_reconstruction'] = torch.stack([first, second], dim=1)
        result['geometric'] = 2 * first.sqrt() * second.sqrt()
        result['total'] = result['geometric']
    if score is not None:
        result['total'] = result[score]
    return result


def score_loader(model, loader, device, *, edge_weight=1.0, score=None):
    """Return NumPy scores and event metadata in loader order, using eval mode."""
    model.eval()
    metadata = ('source', 'row', 'mjj', 'truth', 'weak_label')
    columns = {}
    for batch in loader:
        for jets in batch['jets']:
            jets.to(device)
        values = {name: batch[name] for name in metadata}
        values.update(score_events(model, batch, edge_weight=edge_weight, score=score))
        for name, value in values.items():
            columns.setdefault(name, []).append(value.cpu())
    return {name: torch.cat(values).numpy() for name, values in columns.items()}


@torch.no_grad()
def score_pair_loader(baseline, regularized, loader, device, *, baseline_edge_weight=1.0,
                      reg_edge_weight=1.0, baseline_score=None, reg_score=None, reg_loader=None):
    """Score two gate branches, allowing different architectures and scoring rules.

    A separate reg_loader supplies the same events at a different graph resolution.
    Gating features come from the baseline graphs.
    """
    baseline.eval()
    regularized.eval()
    metadata = ('source', 'row', 'mjj', 'truth', 'weak_label')
    columns = {name: [] for name in (*metadata, 'baseline', 'regularized', 'pr', 'radius')}
    batches = ((batch, batch) for batch in loader) if reg_loader is None else zip(loader, reg_loader, strict=True)
    for batch, reg_batch in batches:
        if reg_loader is not None:
            if any(not torch.equal(batch[key], reg_batch[key]) for key in ('source', 'row')):
                raise ValueError('Calibration branches must contain the same events in the same order')
            for jets in reg_batch['jets']:
                jets.to(device)
        for jets in batch['jets']:
            jets.to(device)
        values = {name: batch[name] for name in metadata}
        values['baseline'] = score_events(baseline, batch, edge_weight=baseline_edge_weight,
                                          score=baseline_score)['total']
        values['regularized'] = score_events(regularized, reg_batch, edge_weight=reg_edge_weight,
                                             score=reg_score)['total']
        features = [jet_structure_features(jets) for jets in batch['jets']]
        values['pr'] = torch.stack([part[0] for part in features], dim=1)
        values['radius'] = torch.stack([part[1] for part in features], dim=1)
        for name, value in values.items():
            columns[name].append(value.cpu())
    return {name: torch.cat(values).numpy() for name, values in columns.items()}
