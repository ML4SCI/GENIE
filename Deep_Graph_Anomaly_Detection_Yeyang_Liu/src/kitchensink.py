"""KS tree-classifier features and splits in published feature-table row order.

Feature definitions: https://github.com/lang-lukas/KitchenSink
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .splits import EVENT_DTYPE, dijet_mass, make_samples


SIGNALS = ('LHCO', 'XtoWRto3W', 'XtoYYprime', 'ZpToTpTp', 'YtoHHto4T')
SOURCES = ('LHCO', 'extra_qcd', *SIGNALS[1:])
FEATURE_SETS = ('baseline', 'subjettiness', 'efp', 'combined')
TAU_TABLES = {
    'LHCO': 'lhco/lhco_subjettiness.h5',
    'extra_qcd': 'lhco/extra_qcd_subjettiness.h5',
    **{name: f'kitchensink/{name}/{name}_subjettinesses.h5' for name in SIGNALS[1:]},
}
EFP_TABLES = {
    'LHCO': 'lhco/lhco_efps.h5',
    'extra_qcd': 'lhco/extra_qcd_efps.h5',
    **{name: f'kitchensink/{name}/{name}_efps.h5' for name in SIGNALS[1:]},
}


def pool_order(signal, events, split_dir=Path('data/splits/kitchensink')):
    """Index graph events in KS run-0 pool order before the trees split them internally."""
    with np.load(split_dir/signal/'nsig1000/events.npz') as split:
        expected = split['pool'].copy()
    expected['source'][expected['source'] >= 2] = 2
    keys = events[['source', 'row']]
    order = np.argsort(keys, order=('source', 'row'))
    return order[np.searchsorted(keys[order], expected[['source', 'row']])]


def ordered_features(raw, *, background=False):
    """Put the lower-mass jet first; express masses in TeV."""
    frame = raw.astype(np.float64).copy()
    mjj = dijet_mass(frame)
    swap = frame['mj1'] > frame['mj2']
    first = [name for name in frame if 'j1' in name]
    second = [name.replace('j1', 'j2') for name in first]
    left, right = frame.loc[swap, first].to_numpy(), frame.loc[swap, second].to_numpy()
    frame.loc[swap, first], frame.loc[swap, second] = right, left
    frame['mjj'] = mjj
    frame['mdiff'] = (frame['mj2'] - frame['mj1']) * 1e-3
    frame[['mj1', 'mj2']] *= 1e-3
    if background:
        frame['label'] = 0
    return frame


def prepare_features(data_dir, feature_dir, sources, feature_sets):
    """Write shared feature banks once; build Combined in chunks to limit memory."""
    want_combined = bool({'efp', 'combined'} & set(feature_sets))
    tau_columns = list(pd.read_hdf(data_dir / TAU_TABLES['LHCO'], start=0, stop=1).columns)
    combined_tau = [name for name in tau_columns if name.startswith('tau')]
    efp_columns = [f'EFP{i}j{jet}' for jet in (1, 2) for i in range(1, 490)]
    combined_columns = ['mj1', 'mdiff', *combined_tau, *efp_columns]
    for source in sources:
        folder = feature_dir / source
        folder.mkdir(parents=True, exist_ok=True)
        needed = [name for name in ('baseline', 'subjettiness')
                  if (name == 'baseline' or name in feature_sets) and not (folder / f'{name}.h5').exists()]
        build_combined = want_combined and not all((folder / name).exists()
                                                   for name in ('combined.npy', 'combined.json'))
        if not needed and not build_combined:
            continue
        raw = pd.read_hdf(data_dir / TAU_TABLES[source]).astype(np.float64)
        frame = ordered_features(raw, background=source == 'extra_qcd')
        for name in needed:
            if name == 'baseline':
                columns = []
                for jet in (1, 2):
                    numerator, denominator = f'tau2j{jet}_1', f'tau1j{jet}_1'
                    ratio = f'{numerator}/{denominator}'
                    frame[ratio] = np.nan_to_num(frame[numerator].to_numpy() / frame[denominator].to_numpy())
                    columns.append(ratio)
            else:
                columns = [f'tau{i}j{jet}_{beta}' for jet in (1, 2) for beta in (5, 1, 2)
                           for i in range(1, 10)]
            selected = frame[['mjj', 'mj1', 'mdiff', *columns, 'label']].astype(np.float32)
            selected.to_hdf(folder / f'{name}.h5', key='df', mode='w')

        if build_combined:
            matrix = np.lib.format.open_memmap(folder / 'combined.npy', mode='w+', dtype=np.float32,
                                               shape=(len(raw), len(combined_columns)))
            swap = (raw['mj1'] > raw['mj2']).to_numpy()
            for start in range(0, len(raw), 10_000):
                stop = min(start + 10_000, len(raw))
                efp = pd.read_hdf(data_dir / EFP_TABLES[source], start=start, stop=stop).to_numpy(dtype=np.float64, copy=True)
                selected = swap[start:stop]
                efp[selected] = np.concatenate([efp[selected, 490:], efp[selected, :490]], axis=1)
                tau = frame.iloc[start:stop][['mj1', 'mdiff', *combined_tau]].to_numpy()
                matrix[start:stop] = np.concatenate([tau, efp[:, 1:490], efp[:, 491:]], axis=1)
            matrix.flush()
            del matrix
            (folder / 'combined.json').write_text(json.dumps(dict(columns=combined_columns), indent=2) + '\n')
        print(f'{source}: features ready', flush=True)


def feature_events(feature_dir, sources):
    """Event identities, masses and labels in each prepared feature table's row order."""
    banks = {}
    for source in sources:
        frame = pd.read_hdf(feature_dir / source / 'baseline.h5')
        events = np.zeros(len(frame), dtype=EVENT_DTYPE)
        events['source'], events['row'] = SOURCES.index(source), np.arange(len(frame))
        events['mjj'], events['truth'], events['weak_label'] = frame.mjj, frame.label, -1
        banks[source] = events
    return banks


def prepare_splits(feature_dir, split_dir, signals, n_signals):
    """Keep native row identities; match the seed-42 classifier's internal split."""
    sources = tuple(dict.fromkeys(('LHCO', 'extra_qcd', *signals)))
    banks = feature_events(feature_dir, sources)
    background = banks['LHCO'][banks['LHCO']['truth'] == 0]
    validation_seed = np.random.RandomState(42).randint(np.iinfo(np.uint32).max, dtype='u8')
    for n_signal in n_signals:
        for signal in signals:
            path = split_dir / signal / f'nsig{n_signal:04d}/events.npz'
            if path.exists():
                continue
            signal_events = banks[signal][banks[signal]['truth'] == 1]
            pool, test = make_samples(background, signal_events, banks['extra_qcd'], n_signal=n_signal)
            train, validation = train_test_split(
                np.arange(len(pool)), test_size=.5, stratify=pool['weak_label'], random_state=validation_seed,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, pool=pool, train=train, validation=validation, test=test)
            print(f'{signal} Nsig={n_signal}: {len(pool):,} train/validation, {len(test):,} test', flush=True)
