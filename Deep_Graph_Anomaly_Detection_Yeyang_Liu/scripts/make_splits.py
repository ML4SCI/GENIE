"""Save KS training, validation and test indices for one signal."""

import argparse
from pathlib import Path

import numpy as np

from src.splits import SIGNAL_COUNTS, make_splits, read_events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/raw'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/splits'))
    parser.add_argument('--mode', choices=['CWoLa', 'IAD'], default='CWoLa')
    parser.add_argument('--signal', choices=['LHCO', 'XtoWRto3W', 'XtoYYprime', 'ZpToTpTp', 'YtoHHto4T'], default='LHCO')
    parser.add_argument('--n-signal', type=int, choices=SIGNAL_COUNTS, default=1000)
    args = parser.parse_args()

    source_files = ['lhco/events_anomalydetection.h5', 'lhco/extra_qcd_cartesian.h5']
    feature_files = ['lhco/lhco_subjettiness.h5', 'lhco/extra_qcd_features.h5']
    lhco = read_events(args.data_dir / feature_files[0], source=0)
    # This table follows the extra-QCD particle file's row order.
    extra = read_events(args.data_dir / feature_files[1], source=1, background=True)
    signal = lhco[lhco['truth'] == 1]
    if args.signal != 'LHCO':
        source_files.append(f'kitchensink/{args.signal}/events.h5')
        feature_files.append(f'kitchensink/{args.signal}/{args.signal}_subjettinesses.h5')
        signal = read_events(args.data_dir / feature_files[2], source=2)
        signal = signal[signal['truth'] == 1]

    splits = make_splits(
        lhco[lhco['truth'] == 0], signal, extra, mode=args.mode, n_signal=args.n_signal,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f'{args.mode}_{args.signal}_nsig{args.n_signal}.npz'
    np.savez_compressed(
        output, **splits, source_files=source_files, feature_files=feature_files,
        mode=args.mode, signal=args.signal, n_signal=args.n_signal,
        run=0, member=0, mass_unit='TeV',
    )
    print('split       | events  | background | signal | SR (weak=1) | reference (weak=0)')
    for name, rows in splits.items():
        print(f'{name:11} | {len(rows):7,d} | {(rows["truth"] == 0).sum():10,d} | '
              f'{rows["truth"].sum():6,d} | {(rows["weak_label"] == 1).sum():11,d} | '
              f'{(rows["weak_label"] == 0).sum():18,d}')
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
