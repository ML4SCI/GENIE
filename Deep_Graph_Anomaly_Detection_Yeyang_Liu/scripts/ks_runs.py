"""Physics-informed AE on repeated Kitchen Sink CWoLa splits (LHCO, Nsig=1000).

Run r rebuilds the Kitchen Sink pool and test signals with its seeds (signal shuffle 1934 + r;
mixture shuffle s_0 = 1001, s_r = s_(r-1) + r + 1000), retrains the AE on the adjacent-SB events of a
stratified half of that pool, and trains the readout with the run-r member seeds. The 340k test
background events are the same for every run. Every run reuses the input normalization and target
standardization fitted on the LHCO training SB events, and each
training batch is processed in sorted event order.

    python -m scripts.ks_runs --runs 4 --device cuda:0

--runs is the number of runs, starting at run 0; default: 1.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from scripts.physics_ae import lhco_training, load_packed, pack, readout_inputs, split_path, statistics, train
from src.kitchensink import feature_events
from src.metrics import score_metrics
from src.readout import ensemble_scores
from src.splits import make_samples, mixture_seed


def samples(run, n_signal=1000, feature_dir=Path('data/features/kitchensink')):
    """CWoLa pool (Kitchen Sink order) and test signals of run r, as LHCO table events."""
    banks = feature_events(feature_dir, ('LHCO', 'extra_qcd'))
    events = banks['LHCO']
    pool, test = make_samples(events[events['truth'] == 0], events[events['truth'] == 1],
                              banks['extra_qcd'], n_signal=n_signal, run=run)
    return pool, test[test['truth'] == 1]


def prepare(run, split_dir, packed_dir):
    """Write the run split (shared test background from the LHCO split) and pack its graphs."""
    pool, test_signal = samples(run)
    with np.load(split_path('LHCO')) as split:
        background = split['test'][split['test']['source'] == 1]
        files = split['source_files'].copy()
    split_file = split_dir/f'run{run}.npz'
    split_dir.mkdir(parents=True, exist_ok=True)
    np.savez(split_file, pool=pool, test=np.concatenate([background, test_signal]), source_files=files)
    pack(split_file, packed_dir/f'run{run}', parts=('pool', 'test'))
    return split_file


def run_one(run, *, stats, device, seed, split_dir, packed_dir, run_dir):
    split_file = split_dir/f'run{run}.npz'
    if not (packed_dir/f'run{run}'/'test.pt').exists():
        split_file = prepare(run, split_dir, packed_dir)
    data, events = load_packed(packed_dir/f'run{run}', 'pool')
    half, other = train_test_split(np.arange(len(events)), test_size=.5, stratify=events['weak_label'],
                                   random_state=mixture_seed(run))
    sideband = events['weak_label'] == 0
    model = train(data, events, np.sort(half[sideband[half]]), np.sort(other[sideband[other]]), run_dir/f'run{run}',
                  seed=seed, device=device, stats=stats, sort_batches=True).eval()
    with np.load(split_file) as split:
        paths = [Path('data/graphs')/name for name in split['source_files']]
    test_data, test_events = load_packed(packed_dir/f'run{run}', 'test')
    scores = ensemble_scores(readout_inputs(model, data, events, paths, device), events['weak_label'],
                             readout_inputs(model, test_data, test_events, paths, device), run=run,
                             model_dir=run_dir/f'run{run}'/'readout')
    metrics = score_metrics(scores, test_events['truth'])
    np.save(run_dir/f'run{run}'/'test_scores.npy', scores)
    (run_dir/f'run{run}'/'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print('run', run, 'AUC %.4f maxSIC %.2f maxSIC(KS) %.2f' % (metrics['auc'], metrics['max_sic'], metrics['max_sic_ks']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', type=int, default=1, help='number of runs, starting at run 0 (default: 1)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--split-dir', type=Path, default=Path('data/splits/ks_runs'))
    parser.add_argument('--packed-dir', type=Path, default=Path('data/packed/ks_runs'))
    parser.add_argument('--run-dir', type=Path, default=Path('runs/physics_ae/sb/ks_runs'))
    parser.add_argument('--lhco-packed-dir', type=Path, default=Path('data/packed'))
    args = parser.parse_args()
    data, events, train_rows, _ = lhco_training(args.lhco_packed_dir)
    stats = statistics(data, events, train_rows, args.device)
    del data
    for run in range(args.runs):
        run_one(run, stats=stats, device=args.device, seed=args.seed, split_dir=args.split_dir,
                packed_dir=args.packed_dir, run_dir=args.run_dir)


if __name__ == '__main__':
    main()
