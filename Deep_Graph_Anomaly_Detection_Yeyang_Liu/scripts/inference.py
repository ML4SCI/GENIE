"""Score a split with a model checkpoint or a frozen calibrated model pair."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.calibration import apply_calibration
from src.checkpoint import load_checkpoint
from src.dataset import EventDataset
from src.models import create_model
from src.metrics import score_metrics
from src.scoring import score_loader, score_pair_loader, scoring_options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', type=Path)
    source.add_argument('--calibration', type=Path, help='frozen model pair from scripts.calibrate')
    parser.add_argument('--split-file', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'validation', 'test'], default='test')
    parser.add_argument('--graph-dir', type=Path, help='defaults to the checkpoint graph directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=128, help='events per batch')
    parser.add_argument('--device', default='cpu', help='cpu or cuda:0, cuda:1, ...')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--score', help='score to store as total; defaults to the checkpoint setting')
    args = parser.parse_args()

    checkpoint = (load_checkpoint(args.checkpoint) if args.checkpoint else
                  torch.load(args.calibration, map_location='cpu', weights_only=True))
    settings = checkpoint['settings']
    graph_settings = settings if args.checkpoint else checkpoint['baseline_settings']
    graph_dir = args.graph_dir or Path(graph_settings['graph_dir'])
    dataset = EventDataset(args.split_file, split=args.split, graph_dir=graph_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    variant_scores = ()
    if args.checkpoint:
        model_name = settings['model']
        variant = settings.get('netge_variant', settings.get('sdm_variant', 'original'))
        _, available = scoring_options(model_name, variant=variant)
        score = settings['score'] if args.score in (None, 'total') else args.score
        if score not in available:
            parser.error(f'--score for {model_name} must be one of: {", ".join(available)}')
        model = create_model(settings)
        model.load_state_dict(checkpoint['model'])
        model.to(args.device)
        edge_weight = checkpoint['settings']['edge_weight']
        epoch = checkpoint['epoch']
        print(f'Scoring {len(dataset):,} {args.split} events at epoch {epoch}', flush=True)
        result = score_loader(model, loader, args.device, edge_weight=edge_weight, score=score)
        if model_name != 'edgeconv':
            variant_scores = available
        provenance = dict(checkpoint=str(args.checkpoint.resolve()), edge_weight=edge_weight, epoch=epoch)
        if model_name in ('sdm_ati', 'sdm_nat'):
            provenance.update(model=model_name, variant=variant, max_subjets=model.max_subjets,
                              training_sample=settings['sample'], score_definition=score)
        if model_name == 'netge' and settings.get('netge_variant') == 'max_bilinear':
            provenance.update(model='netge', netge_variant='max_bilinear',
                              per_jet_score='standardized node MSE + physical-edge MSE',
                              geometric_score='2 * sqrt(jet0_reconstruction) * sqrt(jet1_reconstruction)')
        if model_name == 'sdm_atii':
            provenance.update(model='sdm_atii', score_definition='-D(jet0) - D(jet1)')
        if model_name == 'sdm_energy':
            provenance.update(model='sdm_energy', backbone=settings['sdm_backbone'],
                              max_subjets=model.max_subjets,
                              critic_context=settings.get('sdm_critic_context', False),
                              score_definition='E(jet0) + E(jet1); standardized node and edge MSE')
            if score in ('generator', 'generator_objective'):
                provenance.update(
                    score_definition='sum over jets of lambda * posterior-mean node/edge MSE + gamma * KL',
                    discrepancy_weight=model.discrepancy_weight, kl_weight=model.kl_weight)
                if score == 'generator_objective':
                    provenance['score_definition'] = (
                        'sum over jets of lambda * posterior-mean node/edge MSE + gamma * KL + E(posterior-mean reconstructed graph)')
    else:
        if args.score not in (None, 'total'):
            parser.error('Calibrated gated scoring uses --score total')
        score = 'total'
        bundle = checkpoint
        models = {}
        for name in ('baseline', 'regularized'):
            model = create_model(bundle[name+'_settings'])
            model.load_state_dict(bundle[name+'_model'])
            models[name] = model.to(args.device)
        settings = bundle['settings']
        reg_loader = None
        reg_graph_dir = args.graph_dir or Path(bundle['regularized_settings']['graph_dir'])
        if reg_graph_dir.resolve() != graph_dir.resolve():
            reg_dataset = EventDataset(args.split_file, split=args.split, graph_dir=reg_graph_dir)
            reg_loader = DataLoader(reg_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers)
        print(f'Gated scoring: {len(dataset):,} {args.split} events', flush=True)
        result = score_pair_loader(models['baseline'], models['regularized'], loader, args.device,
                                   baseline_edge_weight=settings['baseline_edge_weight'],
                                   reg_edge_weight=settings['reg_edge_weight'],
                                   baseline_score=settings['baseline_score'],
                                   reg_score=settings['reg_score'], reg_loader=reg_loader)
        result.update(apply_calibration(result, bundle['calibration']))
        provenance = dict(calibration_file=str(args.calibration.resolve()))
        for name in ('baseline_epoch', 'reg_epoch'):
            provenance[name] = settings[name]
    result.update(provenance)
    metrics = score_metrics(result['total'], result['truth'])
    score_variants = {name: score_metrics(result[name], result['truth'])
                      for name in variant_scores if name in result}
    with np.load(args.split_file) as split:
        result.update(source_files=split['source_files'], mass_unit=split['mass_unit'])
    result.update(split_file=str(args.split_file.resolve()), split=args.split, score=score)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output.with_suffix('.json')
    if args.output.exists() or metrics_path.exists():
        raise FileExistsError(f'Score output already exists: {args.output} or {metrics_path}')
    with args.output.open('xb') as file:
        np.savez_compressed(file, **result)
    with metrics_path.open('x') as file:
        json.dump(dict(split=args.split, split_file=str(args.split_file.resolve()),
                       score=score, **provenance, metrics=metrics, score_variants=score_variants),
                  file, indent=2, allow_nan=False)
    print(f'Saved {len(result["row"]):,} event scores to {args.output}', flush=True)
    print(f'AUC={metrics["auc"]}  maxSIC(KS)={metrics["max_sic_ks"]}', flush=True)


if __name__ == '__main__':
    main()
