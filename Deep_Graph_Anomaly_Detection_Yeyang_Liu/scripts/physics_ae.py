"""Physics-informed AE features with a CWoLa tree readout.

Protocol: the AE trains on the adjacent LHCO sidebands (mJJ 3.1–3.3 and 3.7–3.9 TeV),
then is frozen. For each signal, the 50-member tree readout trains on its SR/SB pool.
Default readout inputs per event (80): graph jet masses (2), AE-predicted tau (54), EFP (16), log
reconstruction errors (2) and tau ratios (6).

Steps (run from the repository root):
    python -m scripts.physics_ae pack --signal LHCO
    python -m scripts.physics_ae train --seed 42 --device cuda:0
    python -m scripts.physics_ae evaluate --signal LHCO --seed 42 --device cuda:0
    python -m scripts.physics_ae train --variant predicted --device cuda:0
    python -m scripts.physics_ae evaluate --variant predicted --device cuda:0
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from src.dataset import jet_batch, pack_events
from src.kitchensink import pool_order, prepare_splits
from src.metrics import score_metrics
from src.models.physics_ae import PhysicsAE
from src.physics import (event_energy_flow, input_normalization, mass_features, standardization,
                         subjettiness, tau_ratios)
from src.readout import ensemble_scores


SIGNALS = ('LHCO', 'XtoWRto3W', 'XtoYYprime', 'ZpToTpTp', 'YtoHHto4T')
VARIANTS = ('predicted', 'graph_mass', 'errors', 'ratios')
EPOCHS, BATCH = 50, 256
OPTIMIZER = dict(lr=.003, weight_decay=.01)
SCHEDULE = dict(max_lr=.003, pct_start=.02, anneal_strategy='linear', div_factor=5., final_div_factor=3.)
KS_SPLITS = Path('data/splits/kitchensink')


def split_path(signal):
    return Path(f'data/splits/CWoLa_{signal}_nsig1000.npz')


def graph_paths(split_file, graph_dir='data/graphs'):
    with np.load(split_file) as split:
        return [Path(graph_dir)/name for name in split['source_files']]


def pack(split_file, out_dir, parts=('train', 'validation', 'test'), workers=8):
    out_dir.mkdir(parents=True, exist_ok=True)
    for part in parts:
        data, events = pack_events(split_file, part, workers=workers)
        torch.save(data, out_dir/f'{part}.pt')
        np.save(out_dir/f'{part}_events.npy', events)
        print('packed', split_file, part, len(events), flush=True)


def load_packed(directory, part):
    return torch.load(Path(directory)/f'{part}.pt', weights_only=True, mmap=True), np.load(Path(directory)/f'{part}_events.npy')


def statistics(data, events, rows, device):
    """Input normalization and target standardization, fitted on the given training rows."""
    raw = subjettiness(events[rows])
    efp = event_energy_flow({key: data[key][rows] for key in ('pt', 'positions', 'mask')}, device)
    every = np.arange(len(rows))
    stats = {key: value.numpy() for key, value in input_normalization(data, rows).items()}
    for name, values in (('physics', raw), ('efp', efp), ('ratio', tau_ratios(raw))):
        stats[name+'_center'], stats[name+'_scale'] = standardization(values, every)
    return stats


def targets(data, events, stats, device):
    """Standardized subjettiness (56) and EFP (16) targets, and raw tau ratios (6), for every event."""
    raw = subjettiness(events)
    efp = event_energy_flow(data, device)
    target = np.c_[(raw - stats['physics_center'])/stats['physics_scale'],
                   (efp.astype(np.float64) - stats['efp_center'])/stats['efp_scale']]
    return target.astype(np.float32), tau_ratios(raw).astype(np.float32)


def run_epoch(model, data, rows, target, ratio, device, *, seed=None, number=None, optimizer=None,
              scheduler=None, sort_batches=False):
    training = optimizer is not None
    model.train(training)
    if training:
        rows = rows[np.random.RandomState(seed + number).permutation(len(rows))]
    totals = {}
    for start in range(0, len(rows), BATCH):
        selected = rows[start:start+BATCH]
        if sort_batches:
            selected = np.sort(selected)
        inputs = jet_batch(data, selected, device)
        features = torch.as_tensor(target[selected], device=device)
        with torch.set_grad_enabled(training):
            losses = model.loss(inputs, features[:, :56], features[:, 56:], torch.as_tensor(ratio[selected], device=device))
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses['objective'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.) + float(value.detach())*len(selected)
    return {name: value/len(rows) for name, value in totals.items()}


def train(data, events, train_rows, validation_rows, out_dir, *, seed=42, device='cpu', epochs=EPOCHS,
          stats=None, sort_batches=False, ratios=True):
    """Train on sideband train_rows; log sideband validation_rows each epoch.

    stats defaults to statistics fitted on train_rows. ae.pt is written only after the last epoch;
    last.pt holds the most recent epoch while training.
    """
    stats = statistics(data, events, train_rows, device) if stats is None else stats
    target, ratio = targets(data, events, stats, device)
    model = PhysicsAE(seed, ratios=ratios)
    model.set_normalization(**{key: stats[key] for key in ('node_center', 'node_scale', 'edge_center',
                                                           'edge_scale', 'ratio_center', 'ratio_scale')})
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), **OPTIMIZER)
    updates = (len(train_rows) + BATCH - 1) // BATCH
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=epochs*updates, **SCHEDULE)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for number in range(1, epochs + 1):
        started = time.monotonic()
        trained = run_epoch(model, data, train_rows, target, ratio, device, seed=seed, number=number,
                            optimizer=optimizer, scheduler=scheduler, sort_batches=sort_batches)
        validated = run_epoch(model, data, validation_rows, target, ratio, device, sort_batches=sort_batches)
        history.append(dict(epoch=number, **{'train_'+k: v for k, v in trained.items()},
                            **{'validation_'+k: v for k, v in validated.items()}, seconds=time.monotonic()-started))
        state = dict(model=model.state_dict(), seed=seed, ratios=ratios, epoch=number, epochs=epochs,
                     train_region='SB', history=history,
                     statistics={k: np.asarray(v).tolist() for k, v in stats.items()})
        torch.save(state, out_dir/'last.pt')
        print('epoch', number, json.dumps(history[-1]), flush=True)
    (out_dir/'last.pt').replace(out_dir/'ae.pt')
    return model


def lhco_training(packed_dir=Path('data/packed')):
    """Load the LHCO pool and select adjacent-SB rows for AE training and validation."""
    train_data, train_events = load_packed(packed_dir/'LHCO', 'train')
    validation_data, validation_events = load_packed(packed_dir/'LHCO', 'validation')
    data = {key: torch.cat([train_data[key], validation_data[key]]) for key in train_data}
    events = np.concatenate([train_events, validation_events])
    sideband = events['weak_label'] == 0
    train_rows = np.flatnonzero(sideband[:len(train_events)])
    validation_rows = len(train_events) + np.flatnonzero(sideband[len(train_events):])
    return data, events, train_rows, validation_rows


def load_model(checkpoint, device, *, ratios=True):
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = PhysicsAE(state.get('seed', 42), ratios=ratios)
    model.load_state_dict(state['model'])
    return model.to(device).eval()


@torch.no_grad()
def ae_features(model, data, rows, device, chunk=512):
    return np.concatenate([model.features(*jet_batch(data, rows[start:start+chunk], device)).cpu().numpy()
                           for start in range(0, len(rows), chunk)])


def readout_inputs(model, data, events, paths, device, *, variant='ratios'):
    learned = ae_features(model, data, np.arange(len(events)), device)
    if variant == 'predicted':
        return learned[:, :72]
    masses = mass_features(events, paths)
    if variant == 'graph_mass':
        learned = learned[:, 2:72]
    elif variant == 'errors':
        learned = learned[:, 2:]
    return np.c_[masses, learned]


def evaluate(model, packed_dir, split_file, out_dir, device, *, run=0, variant='ratios'):
    """Fit and evaluate one KS50 readout using the selected feature variant."""
    paths = graph_paths(split_file)
    with np.load(split_file) as split:
        signal = str(split['signal'])
    prepare_splits(Path('data/features/kitchensink'), KS_SPLITS, [signal], [1000])
    features, metadata = {}, {}
    for part in ('train', 'validation', 'test'):
        data, events = load_packed(packed_dir, part)
        features[part] = readout_inputs(model, data, events, paths, device, variant=variant)
        metadata[part] = events
    pool_events = np.r_[metadata['train'], metadata['validation']]
    order = pool_order(signal, pool_events, KS_SPLITS)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir/'events.npz', pool=pool_events[order], test=metadata['test'])
    pool = np.r_[features['train'], features['validation']][order]
    scores = ensemble_scores(pool, pool_events['weak_label'][order], features['test'], run=run,
                             workers=6, model_dir=out_dir/'readout')
    metrics = score_metrics(scores, metadata['test']['truth'])
    np.save(out_dir/'test_scores.npy', scores)
    (out_dir/'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(out_dir, 'AUC %.4f maxSIC %.2f maxSIC(KS) %.2f' %
          (metrics['auc'], metrics['max_sic'], metrics['max_sic_ks']), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('step', choices=('pack', 'train', 'evaluate'))
    parser.add_argument('--variant', choices=VARIANTS, default='ratios')
    parser.add_argument('--signal', choices=SIGNALS, default='LHCO')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--packed-dir', type=Path, default=Path('data/packed'))
    parser.add_argument('--run-dir', type=Path, default=Path('runs/physics_ae/sb'))
    parser.add_argument('--checkpoint', type=Path, help='AE checkpoint for evaluate (default: trained seed)')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    ratios = args.variant == 'ratios'
    folder = args.run_dir/f'seed{args.seed}'
    if not ratios:
        folder = folder/'physics'
    if args.step == 'pack':
        pack(split_path(args.signal), args.packed_dir/args.signal, workers=args.workers)
    elif args.step == 'train':
        data, events, train_rows, validation_rows = lhco_training(args.packed_dir)
        train(data, events, train_rows, validation_rows, folder, seed=args.seed, device=args.device, ratios=ratios)
    else:
        checkpoint = args.checkpoint or folder/'ae.pt'
        output = folder/args.signal if ratios else folder/args.signal/args.variant
        evaluate(load_model(checkpoint, args.device, ratios=ratios), args.packed_dir/args.signal,
                 split_path(args.signal), output, args.device, variant=args.variant)


if __name__ == '__main__':
    main()
