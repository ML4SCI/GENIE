"""Prepare the two leading jets and their constituents for each raw dataset."""

import argparse
from pathlib import Path

from src.preprocessing import RAW_FILES, preprocess_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/raw'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/processed'))
    parser.add_argument('--datasets', nargs='+', choices=RAW_FILES, default=list(RAW_FILES))
    parser.add_argument('--batch-size', type=int, default=1000)
    args = parser.parse_args()
    for dataset in args.datasets:
        preprocess_dataset(args.data_dir, args.output_dir, dataset, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
