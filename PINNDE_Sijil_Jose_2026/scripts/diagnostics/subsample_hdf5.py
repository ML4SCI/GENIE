"""
subsample_hdf5.py
=================
Subsamples a CaloChallenge-format HDF5 file to a target number of events.

Used to make the reference (Geant4) file match the event count of the input
(generated) file, which the classifier metrics in evaluate.py require:
    assert len(data1) == len(data2)   # ttv_split

DOES NOT WORK FOR DATASET 2's CLASSIFIER MODES. Satisfying that assert is
necessary but not sufficient: just below it, ttv_split hard-codes the split
sizes per dataset --

    '2': np.array([60000, 20000, 20000])

-- so dataset 2 needs exactly 100,000 showers on BOTH sides. Subsample both to
10k and the assert passes while np.split hands the classifier mostly-empty
arrays. The fix is to GENERATE 100k showers, not to shrink the reference.

Still useful for the histogram/chi-squared modes, which do not go through
ttv_split, and for quick eyeball checks on a smaller file.

Usage:
    python subsample_hdf5.py \
        --in   dataset_2_2.hdf5 \
        --out  dataset_2_2_10k.hdf5 \
        --n    10000 \
        --seed 42
"""

import argparse
import numpy as np
import h5py


def main():
    p = argparse.ArgumentParser(description="Subsample a CaloChallenge HDF5.")
    p.add_argument('--in', dest='infile', required=True,
                   help='Input HDF5 (incident_energies + showers).')
    p.add_argument('--out', dest='outfile', required=True,
                   help='Output HDF5 path.')
    p.add_argument('--n', type=int, required=True,
                   help='Number of events to keep.')
    p.add_argument('--seed', type=int, default=42,
                   help='RNG seed for the random subset (default 42).')
    args = p.parse_args()

    with h5py.File(args.infile, 'r') as f:
        einc = f['incident_energies'][:]
        showers = f['showers'][:]

    n_total = einc.shape[0]
    print(f'Input: {n_total} events, showers shape {showers.shape}')

    if args.n > n_total:
        raise ValueError(f'Requested {args.n} events but file only has {n_total}.')
    if args.n == n_total:
        print('Requested count equals file size -- copying as-is.')
        idx = np.arange(n_total)
    else:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(n_total, size=args.n, replace=False)
        idx.sort()  # keep file order stable
        print(f'Selected {args.n} events at random (seed {args.seed}).')

    einc_sub = einc[idx]
    showers_sub = showers[idx]

    with h5py.File(args.outfile, 'w') as f:
        f.create_dataset('incident_energies', data=einc_sub, compression='gzip')
        f.create_dataset('showers', data=showers_sub, compression='gzip')

    print(f'Wrote {args.outfile}: {einc_sub.shape[0]} events, '
          f'showers shape {showers_sub.shape}')


if __name__ == '__main__':
    main()
