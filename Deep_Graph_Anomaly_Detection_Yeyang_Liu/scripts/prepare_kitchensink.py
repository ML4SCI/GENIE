"""Prepare native-table features and CWoLa splits for the KS tree classifier."""

import argparse
from pathlib import Path
import subprocess

from src.kitchensink import EFP_TABLES, FEATURE_SETS, SIGNALS, prepare_features, prepare_splits
from src.splits import SIGNAL_COUNTS


def download_efps(data_dir, sources):
    """Fetch the optional EFP tables; signal tables come from the existing archives."""
    release = 'https://zenodo.org/records/18983506/files'
    names = {'LHCO': 'LHCO_2-prong_efps.h5', 'extra_qcd': 'LHCO_extra_bkg_efps.h5'}
    for source in sources:
        path = data_dir / EFP_TABLES[source]
        if path.exists():
            continue
        if source in names:
            temporary = path.with_suffix('.h5.part')
            subprocess.run(['curl', '-fL', f'{release}/{names[source]}', '-o', str(temporary)], check=True)
            temporary.rename(path)
        else:
            archive = data_dir / 'kitchensink' / f'{source}.tar.gz'
            subprocess.run(['tar', '-xzf', str(archive), '-C', str(path.parent), f'./{path.name}'], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/raw'))
    parser.add_argument('--feature-dir', type=Path, default=Path('data/features/kitchensink'))
    parser.add_argument('--split-dir', type=Path, default=Path('data/splits/kitchensink'))
    parser.add_argument('--signals', nargs='+', choices=SIGNALS, default=list(SIGNALS))
    parser.add_argument('--n-signal', nargs='+', type=int, choices=SIGNAL_COUNTS, default=[1000])
    parser.add_argument('--feature-sets', nargs='+', choices=FEATURE_SETS, default=list(FEATURE_SETS))
    args = parser.parse_args()
    sources = tuple(dict.fromkeys(('LHCO', 'extra_qcd', *args.signals)))
    if {'efp', 'combined'} & set(args.feature_sets):
        download_efps(args.data_dir, sources)
    prepare_features(args.data_dir, args.feature_dir, sources, args.feature_sets)
    prepare_splits(args.feature_dir, args.split_dir, args.signals, args.n_signal)


if __name__ == '__main__':
    main()
