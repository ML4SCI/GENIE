"""SB-trained dual AE: 50-epoch warmups, 50-epoch control/InfoNCE, then a KS50 readout.

Uses data/packed/<signal> from scripts.physics_ae pack and the prepared KS feature tables.
Nsig=1000, 30 subjets. Each signal is trained separately; only the AE seed varies.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.physics_ae import BATCH, EPOCHS, OPTIMIZER, SCHEDULE, graph_paths, load_packed, split_path
from src.dataset import jet_batch
from src.kitchensink import SIGNALS, pool_order, prepare_splits
from src.metrics import score_metrics
from src.models.dual_ae import CONTRAST_WEIGHT, TEMPERATURE, DualAE
from src.physics import event_energy_flow, input_normalization, mass_features, standardization, subjettiness
from src.readout import ensemble_scores
from src.softdrop import event_softdrop


PACKED = Path('data/packed')
PREPARED = Path('data/dual_ae')
OUTPUT = Path('runs/dual_ae')
KS_SPLITS = Path('data/splits/kitchensink')
VARIANTS = ('control', 'contrast')


def inputs(data, rows, device):
    positions = torch.cat([data['positions'][rows, 0], data['positions'][rows, 1]]).to(device)
    return (*jet_batch(data, rows, device), positions)


def target_values(data, events, rows, signal, device):
    selected = events[rows]
    physics = np.empty((len(rows), 56), dtype=np.float64)
    for source in np.unique(selected['source']):
        at = selected['source'] == source
        table = {0: 'LHCO', 2: signal}[source]
        physics[at] = subjettiness(selected[at], table=table)
    graphs = {name: data[name][rows] for name in ('pt', 'positions', 'mask')}
    return dict(physics=physics, efp=event_energy_flow(graphs, device), grooming=event_softdrop(graphs))


def prepare(signal, device):
    folder = PREPARED/signal
    folder.mkdir(parents=True, exist_ok=True)
    for part in ('train', 'validation'):
        data, events = load_packed(PACKED/signal, part)
        rows = np.flatnonzero(events['weak_label'] == 0)
        values = target_values(data, events, rows, signal, device)
        if part == 'train':
            statistics = {name: value.numpy() for name, value in input_normalization(data, rows).items()}
            for name, value in values.items():
                statistics[name+'_center'], statistics[name+'_scale'] = standardization(value, np.arange(len(rows)))
        targets = np.concatenate([
            (value.astype(np.float64)-statistics[name+'_center'])/statistics[name+'_scale']
            for name, value in values.items()
        ], axis=1).astype(np.float32)
        np.save(folder/f'{part}_targets.npy', targets)
        print('prepared', signal, part, len(rows), 'SB events', flush=True)
    torch.save({name: torch.as_tensor(value) for name, value in statistics.items()}, folder/'statistics.pt')
    prepare_splits(Path('data/features/kitchensink'), KS_SPLITS, [signal], [1000])


def training_data(signal):
    parts = {}
    for part in ('train', 'validation'):
        data, events = load_packed(PACKED/signal, part)
        rows = np.flatnonzero(events['weak_label'] == 0)
        targets = np.load(PREPARED/signal/f'{part}_targets.npy')
        parts[part] = data, rows, targets
    return parts


def run_epoch(models, part, phase, device, seed, epoch, optimizers=(), schedulers=()):
    data, rows, targets = part
    training = bool(optimizers)
    models.train(training)
    order = np.random.RandomState(seed+epoch).permutation(len(rows)) if training else np.arange(len(rows))
    branches = list(models.values()) if phase in VARIANTS else [models[phase]]
    totals = {}
    for start in range(0, len(rows), BATCH):
        selected = order[start:start+BATCH]
        batch = inputs(data, rows[selected], device)
        target = torch.as_tensor(targets[selected], device=device)
        with torch.set_grad_enabled(training):
            if phase in VARIANTS:
                losses = models.loss(batch, target, contrast=phase == 'contrast')
            else:
                losses = models[phase].loss(batch, target)[0]
            if training:
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                losses['objective'].backward()
                for branch, optimizer, scheduler in zip(branches, optimizers, schedulers):
                    torch.nn.utils.clip_grad_norm_(branch.parameters(), 1.)
                    optimizer.step()
                    scheduler.step()
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.) + float(value.detach())*len(selected)
    return {name: value/len(rows) for name, value in totals.items()}


def train_phase(models, parts, statistics, signal, seed, phase, device):
    joint = phase in VARIANTS
    branches = dict(models.items()) if joint else {phase: models[phase]}
    steps = EPOCHS*((len(parts['train'][1])+BATCH-1)//BATCH)
    optimizers = [torch.optim.AdamW(model.parameters(), lr=3e-4 if joint else OPTIMIZER['lr'],
                                   weight_decay=OPTIMIZER['weight_decay']) for model in branches.values()]
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=3e-5) if joint
                  else torch.optim.lr_scheduler.OneCycleLR(opt, total_steps=steps, **SCHEDULE)
                  for opt in optimizers]
    settings = dict(signal=signal, seed=seed, variant=phase, nsig=1000, subjets=30, train_region='SB',
                    epochs=EPOCHS, total_epochs_per_branch=EPOCHS*(2 if joint else 1), batch_events=BATCH,
                    contrastive_weight=CONTRAST_WEIGHT if phase == 'contrast' else 0., temperature=TEMPERATURE)
    folder = OUTPUT/signal/f'seed{seed}'/phase
    folder.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, EPOCHS+1):
        started = time.monotonic()
        trained = run_epoch(models, parts['train'], phase, device, seed, epoch, optimizers, schedulers)
        validated = run_epoch(models, parts['validation'], phase, device, seed, epoch)
        record = dict(epoch=epoch, seconds=time.monotonic()-started,
                      **{'train_'+key: value for key, value in trained.items()},
                      **{'validation_'+key: value for key, value in validated.items()})
        history.append(record)
        torch.save(dict(model={name: model.state_dict() for name, model in branches.items()},
                        epoch=epoch, settings=settings, statistics=statistics, history=history), folder/'last.pt')
        (folder/'history.json').write_text(json.dumps(history, indent=2)+'\n')
        print(signal, seed, phase, epoch, 'objective', trained['objective'], validated['objective'], flush=True)
    (folder/'last.pt').replace(folder/'ae.pt')


def train(signal, seed, device):
    statistics = torch.load(PREPARED/signal/'statistics.pt', weights_only=True)
    parts = training_data(signal)
    models = DualAE(seed)
    models.set_normalization(statistics)
    models.to(device)
    for name in models:
        train_phase(models, parts, statistics, signal, seed, name, device)
    warmstart = {name: {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                 for name, model in models.items()}
    for variant in VARIANTS:
        for name, model in models.items():
            model.load_state_dict(warmstart[name])
        train_phase(models, parts, statistics, signal, seed, variant, device)


def load_model(checkpoint, device):
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    models = DualAE(state['settings']['seed'])
    for name, model in models.items():
        model.load_state_dict(state['model'][name])
    return models.to(device).eval()


@torch.no_grad()
def readout_inputs(models, data, events, paths, device):
    chunks = []
    for start in range(0, len(events), 512):
        rows = np.arange(start, min(start+512, len(events)))
        chunks.append(models.features(inputs(data, rows, device)).cpu().numpy())
    return np.c_[mass_features(events, paths), np.concatenate(chunks)]


def evaluate(signal, seed, device):
    paths = graph_paths(split_path(signal))
    results = {}
    for variant in VARIANTS:
        folder = OUTPUT/signal/f'seed{seed}'/variant
        models = load_model(folder/'ae.pt', device)
        features, metadata = {}, {}
        for part in ('train', 'validation', 'test'):
            data, events = load_packed(PACKED/signal, part)
            features[part] = readout_inputs(models, data, events, paths, device)
            metadata[part] = events
        pool_events = np.r_[metadata['train'], metadata['validation']]
        order = pool_order(signal, pool_events, KS_SPLITS)
        pool = np.r_[features['train'], features['validation']][order]
        scores = ensemble_scores(pool, pool_events['weak_label'][order], features['test'],
                                 workers=6, model_dir=folder/'readout')
        metrics = score_metrics(scores, metadata['test']['truth'])
        np.savez(folder/'test_scores.npz', score=scores, events=metadata['test'])
        (folder/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
        print(signal, seed, variant, metrics, flush=True)
        results[variant] = metrics
    difference = {name: results['contrast'][name]-results['control'][name]
                  for name in ('auc', 'max_sic', 'max_sic_ks')}
    (OUTPUT/signal/f'seed{seed}'/'comparison.json').write_text(json.dumps(difference, indent=2)+'\n')
    print('contrast - control', difference, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=('prepare', 'train', 'evaluate'))
    parser.add_argument('--signal', choices=SIGNALS, default='LHCO')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.step == 'prepare':
        prepare(args.signal, args.device)
    elif args.step == 'train':
        train(args.signal, args.seed, args.device)
    else:
        evaluate(args.signal, args.seed, args.device)


if __name__ == '__main__':
    main()
