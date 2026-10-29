"""Save BB1 full-range and signal-region evaluation indices from processed jets."""

import argparse
from pathlib import Path

import numpy as np
import tables

from src.splits import EVENT_DTYPE, signal_region


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jet-file', type=Path,
                        default=Path('data/processed/bb1/events_LHCO2020_BlackBox1.h5'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/splits'))
    args = parser.parse_args()

    with tables.open_file(args.jet_file) as file:
        if not file.root.valid[:].all():
            raise ValueError('BB1 preprocessing is incomplete: some events have no valid jet pair')
        events = np.zeros(len(file.root.row), dtype=EVENT_DTYPE)
        for name in ('row', 'mjj', 'truth'):
            events[name] = file.get_node('/' + name)[:]
    events['weak_label'] = -1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, selected in [('full', events), ('SR', events[signal_region(events)])]:
        path = args.output_dir / f'BB1_{name}.npz'
        np.savez_compressed(path, test=selected, mass_unit='TeV',
                            source_files=['bb1/events_LHCO2020_BlackBox1.h5'])
        print(f'{path}: {len(selected):,} events, {selected["truth"].sum():,} signal')


if __name__ == '__main__':
    main()
