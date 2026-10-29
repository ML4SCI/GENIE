"""Fit validation-based gated scoring for two checkpoints; save the frozen model pair."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from src.calibration import fit_calibration
from src.checkpoint import load_checkpoint
from src.dataset import EventDataset
from src.models import create_model
from src.scoring import score_pair_loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-checkpoint', type=Path, required=True, help='baseline gate branch')
    parser.add_argument('--reg-checkpoint', type=Path, required=True, help='second gate branch; may use a different architecture')
    parser.add_argument('--split-file', type=Path, required=True)
    parser.add_argument('--sample', choices=['SR', 'reference'], default='reference',
                        help='validation subset; reference (default) = CWoLa SB or IAD extra QCD')
    parser.add_argument('--graph-dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=128, help='events per batch')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--num-workers', type=int, default=0)
    args = parser.parse_args()

    models, checkpoints, model_scores = {}, {}, {}
    for name, path in [('baseline', args.baseline_checkpoint), ('reg', args.reg_checkpoint)]:
        checkpoint = load_checkpoint(path)
        model_settings = checkpoint['settings']
        model_settings['graph_dir'] = str(Path(args.graph_dir or model_settings['graph_dir']).resolve())
        model = create_model(model_settings)
        model.load_state_dict(checkpoint['model'])
        models[name], checkpoints[name] = model.to(args.device), checkpoint
        model_scores[name] = model_settings['score']
    args.graph_dir = args.graph_dir or Path(checkpoints['baseline']['settings']['graph_dir'])
    dataset = EventDataset(args.split_file, split='validation', graph_dir=args.graph_dir)
    indices = np.flatnonzero(dataset.events['weak_label'] == {'SR': 1, 'reference': 0}[args.sample])
    if len(indices) == 0:
        parser.error(f'No {args.sample} events in validation')
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    reg_loader = None
    reg_graph_dir = Path(checkpoints['reg']['settings']['graph_dir'])
    if reg_graph_dir != args.graph_dir.resolve():
        reg_dataset = EventDataset(args.split_file, split='validation', graph_dir=reg_graph_dir)
        reg_loader = DataLoader(Subset(reg_dataset, indices), batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers)
    baseline_edge_weight = checkpoints['baseline']['settings']['edge_weight']
    reg_edge_weight = checkpoints['reg']['settings']['edge_weight']
    print(f'Calibrating on {len(indices):,} validation {args.sample} events', flush=True)
    scores = score_pair_loader(models['baseline'], models['reg'], loader, args.device,
                               baseline_edge_weight=baseline_edge_weight, reg_edge_weight=reg_edge_weight,
                               baseline_score=model_scores['baseline'], reg_score=model_scores['reg'],
                               reg_loader=reg_loader)
    calibration = fit_calibration(scores)
    settings = {key: str(value.resolve()) if isinstance(value, Path) else value
                for key, value in vars(args).items()}
    settings.update(partition='validation', n_calibration=len(indices),
                    baseline_epoch=checkpoints['baseline']['epoch'], reg_epoch=checkpoints['reg']['epoch'],
                    baseline_edge_weight=baseline_edge_weight, reg_edge_weight=reg_edge_weight,
                    baseline_score=model_scores['baseline'], reg_score=model_scores['reg'])
    with np.load(args.split_file) as split:
        source_files = split['source_files'].tolist()
    bundle = {
        'baseline_model': checkpoints['baseline']['model'],
        'regularized_model': checkpoints['reg']['model'],
        'baseline_settings': checkpoints['baseline']['settings'],
        'regularized_settings': checkpoints['reg']['settings'],
        'calibration': calibration,
        'settings': settings,
        'source_files': source_files,
        'source': torch.from_numpy(scores['source']),
        'row': torch.from_numpy(scores['row']),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as file:
        torch.save(bundle, file)
    print(f'Saved model pair and calibration to {args.output}', flush=True)


if __name__ == '__main__':
    main()
