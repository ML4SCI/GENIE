"""Train a single KS classifier or repeated 50-model ensembles on native-table features."""

import argparse
import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from src.metrics import score_metrics
from src.kitchensink import FEATURE_SETS, SIGNALS, SOURCES, feature_events
from src.readout import KS_CLASSIFIER, MEMBERS, ensemble_scores
from src.splits import SIGNAL_COUNTS, make_samples, mixture_seed


SEED = 42


def load_features(feature_dir, feature, sources=SOURCES):
    """Return columns and source -> (matrix, column indices); keep EFP/Combined memory-mapped."""
    if feature in ('baseline', 'subjettiness'):
        frame = pd.read_hdf(feature_dir / 'LHCO' / f'{feature}.h5', start=0, stop=1)
        columns = list(frame.columns[1:-1])
    else:
        columns = json.loads((feature_dir / 'LHCO/combined.json').read_text())['columns']
        if feature == 'efp':
            columns = [name for name in columns if name in ('mj1', 'mdiff') or name.startswith('EFP')]

    banks = {}
    for source in sources:
        folder = feature_dir / source
        if feature in ('baseline', 'subjettiness'):
            frame = pd.read_hdf(folder / f'{feature}.h5')
            matrix = frame[columns].to_numpy(dtype=np.float32)
            indices = np.arange(len(columns))
        else:
            metadata = json.loads((folder / 'combined.json').read_text())
            matrix = np.load(folder / 'combined.npy', mmap_mode='r')
            indices = [metadata['columns'].index(name) for name in columns]
        banks[source] = matrix, indices
    return columns, banks


def assemble(events, banks):
    """Join feature rows in the exact order of the selected event identities."""
    _, columns = next(iter(banks.values()))
    matrix = np.empty((len(events), len(columns)), dtype=np.float32)
    for code, source in enumerate(SOURCES):
        selected = events['source'] == code
        if selected.any():
            values, indices = banks[source]
            rows = events['row'][selected]
            matrix[selected] = values[np.ix_(rows, indices)]
    return matrix


def train_model(folder, feature, signal, n_signal, columns, banks, events):
    folder.mkdir(parents=True)
    pool = events['pool']
    matrix = assemble(pool, banks)
    target = pool['weak_label']
    positive_weight = len(target) / target.sum() - 1
    weight = np.where(target == 1, positive_weight, 1.)

    # Let sklearn split the full pool internally, matching the KS run.
    model = HistGradientBoostingClassifier(
        max_iter=200, max_leaf_nodes=31, validation_fraction=.5, random_state=SEED,
    )
    start = time.monotonic()
    model.fit(matrix, target, sample_weight=weight)
    training_seconds = time.monotonic() - start
    joblib.dump(model, folder / 'classifier.joblib')
    np.savez_compressed(folder / 'loss.npz',
                        training=-model.train_score_, validation=-model.validation_score_)
    config = dict(
        mode='CWoLa', seed=SEED, signal='background_only' if n_signal == 0 else signal,
        n_signal=n_signal, feature_set=feature, columns=columns,
        classifier=model.get_params(), sklearn=sklearn.__version__,
        boosting_iterations=model.n_iter_, training_seconds=training_seconds,
        train_events=len(events['train']), validation_events=len(events['validation']),
        positive_class_weight=float(positive_weight),
        target='SR=1, SB=0; truth is never supplied to fit.',
        validation='Internal stratified 50/50 split; early stopping on weighted weak-label log loss.',
        score='predict_proba(X)[:,1]',
        differences_from_paper='Single model at seed42; no ensembles or repeated runs.',
    )
    (folder / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    print(feature, signal, 'Nsig', n_signal, 'trained', model.n_iter_, 'rounds', flush=True)
    return model, config


def evaluate(folder, checkpoint, events_file, model, config, banks, events):
    folder.mkdir(parents=True, exist_ok=True)
    test = events['test']
    scores = model.predict_proba(assemble(test, banks))[:, 1]
    with (folder / 'scores.npz').open('xb') as file:
        np.savez_compressed(file, score=scores)
    result = score_metrics(scores, test['truth'])
    result.update(
        signal=events_file.parent.parent.name, nsig_injected=config['n_signal'],
        feature_set=config['feature_set'], seed=SEED,
        boosting_iterations=config['boosting_iterations'], training_seconds=config['training_seconds'],
        checkpoint=str(checkpoint.resolve()), events=str(events_file.resolve()),
        source_names=SOURCES, mass_unit='TeV', score_order='Exactly events["test"] order',
    )
    with (folder / 'metrics.json').open('x') as file:
        json.dump(result, file, indent=2, allow_nan=False)
    print(config['feature_set'], result['signal'], 'Nsig', config['n_signal'],
          f'AUC {result["auc"]:.5f} maxSIC(KS) {result["max_sic_ks"]:.3f}', flush=True)
    return result


def train_ensemble(folder, signal, n_signal, banks, event_banks, *, runs, workers):
    """Rebuild each KS run's mixture, average 50 probabilities, then report median metrics."""
    background = event_banks['LHCO']
    background = background[background['truth'] == 0]
    signal_events = event_banks[signal]
    signal_events = signal_events[signal_events['truth'] == 1]
    results = []
    for run in range(runs):
        pool, test = make_samples(background, signal_events, event_banks['extra_qcd'],
                                  n_signal=n_signal, run=run)
        output = folder/f'run{run}'
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output/'events.npz', pool=pool, test=test)
        scores = ensemble_scores(assemble(pool, banks), pool['weak_label'], assemble(test, banks),
                                 run=run, n_signal=n_signal, workers=workers, model_dir=output/'readout')
        np.savez_compressed(output/'scores.npz', score=scores)
        metrics = dict(score_metrics(scores, test['truth']), run=run,
                       mixture_seed=mixture_seed(run, n_signal), signal_seed=934+n_signal+run)
        (output/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
        results.append(metrics)
        print(signal, 'run', run, f'AUC {metrics["auc"]:.5f} maxSIC(KS) {metrics["max_sic_ks"]:.3f}', flush=True)
    median = {name: float(np.median([result[name] for result in results]))
              for name in ('auc', 'max_sic', 'max_sic_ks')}
    summary = dict(signal=signal, n_signal=n_signal, runs=runs, members=MEMBERS,
                   classifier=KS_CLASSIFIER, median=median, per_run=results)
    (folder/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(signal, 'median', median, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-sets', nargs='+', choices=FEATURE_SETS, default=list(FEATURE_SETS))
    parser.add_argument('--signals', nargs='+', choices=SIGNALS, default=list(SIGNALS))
    parser.add_argument('--n-signal', nargs='+', type=int, choices=SIGNAL_COUNTS, default=[1000])
    parser.add_argument('--feature-dir', type=Path, default=Path('data/features/kitchensink'))
    parser.add_argument('--split-dir', type=Path,
                        default=Path('data/splits/kitchensink'))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--ensemble', action='store_true', help='average 50 classifiers per run')
    parser.add_argument('--runs', type=int, default=1, help='ensemble repetitions starting at run 0')
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = Path('runs/kitchensink_cwola')/('ensemble' if args.ensemble else 'seed42')
    sources = tuple(dict.fromkeys(('LHCO', 'extra_qcd', *args.signals)))
    event_banks = feature_events(args.feature_dir, sources) if args.ensemble else None

    with threadpool_limits(limits=args.threads):
        for feature in args.feature_sets:
            columns, banks = load_features(args.feature_dir, feature, sources)
            zero_model = None
            for n_signal in args.n_signal:
                for signal in args.signals:
                    if args.ensemble:
                        folder = args.output_dir/feature/signal/f'nsig{n_signal:04d}'
                        train_ensemble(folder, signal, n_signal, banks, event_banks,
                                       runs=args.runs, workers=max(1, args.threads//2))
                        continue
                    events_file = args.split_dir / signal / f'nsig{n_signal:04d}/events.npz'
                    with np.load(events_file) as loaded:
                        events = dict(loaded)
                    model_name = 'background_only' if n_signal == 0 else signal
                    model_dir = args.output_dir / feature / model_name / f'nsig{n_signal:04d}'
                    result_dir = args.output_dir / feature / signal / f'nsig{n_signal:04d}'

                    # At Nsig=0 every signal uses the same background-only pool.
                    if n_signal == 0 and zero_model is not None:
                        model, config = zero_model, zero_config
                    else:
                        model, config = train_model(model_dir, feature, signal, n_signal,
                                                    columns, banks, events)
                        if n_signal == 0:
                            zero_model, zero_config = model, config
                    evaluate(result_dir, model_dir / 'classifier.joblib', events_file,
                             model, config, banks, events)
                    del events, model
            del banks, zero_model
            gc.collect()


if __name__ == '__main__':
    main()
