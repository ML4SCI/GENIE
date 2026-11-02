"""Build baseline subjet graphs from processed jets."""

import argparse
from pathlib import Path

from src.graph import build_graph_file
from src.preprocessing import RAW_FILES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/processed'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/graphs'))
    parser.add_argument('--datasets', nargs='+', choices=RAW_FILES, default=list(RAW_FILES))
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--max-subjets', type=int, choices=[30, 60], default=30)
    args = parser.parse_args()
    for dataset in args.datasets:
        relative = RAW_FILES[dataset]
        build_graph_file(args.data_dir / relative, args.output_dir / relative, batch_size=args.batch_size,
                         max_subjets=args.max_subjets)


if __name__ == '__main__':
    main()
