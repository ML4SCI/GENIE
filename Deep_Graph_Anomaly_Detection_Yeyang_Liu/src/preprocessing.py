"""Cluster published events into their two leading anti-kt R=1 jets.

Output rows retain the raw event order. For event r, jet j, its constituents
are constituents[offsets[2*r+j]:offsets[2*r+j+1]]. Four-vectors are
(px, py, pz, E) in GeV; mjj is in TeV. Missing jets are NaN with valid=False.

"""

from pathlib import Path

import awkward as ak
import fastjet
import numpy as np
import tables


RAW_FILES = {
    'LHCO': 'lhco/events_anomalydetection.h5',
    'extra_qcd': 'lhco/extra_qcd_cartesian.h5',
    'BB1': 'bb1/events_LHCO2020_BlackBox1.h5',
    'XtoWRto3W': 'kitchensink/XtoWRto3W/events.h5',
    'XtoYYprime': 'kitchensink/XtoYYprime/events.h5',
    'ZpToTpTp': 'kitchensink/ZpToTpTp/events.h5',
    'YtoHHto4T': 'kitchensink/YtoHHto4T/events.h5',
}
FOUR_VECTOR_COLUMNS = ('px', 'py', 'pz', 'E')


def four_vectors(particles):
    """Convert massless (pT, eta, phi) particles to (px, py, pz, E) in GeV."""
    pt, eta, phi = np.moveaxis(np.asarray(particles, dtype=np.float64), -1, 0)
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    energy = pt * np.cosh(eta)
    return np.stack([px, py, pz, energy], axis=-1)


def iter_event_batches(data_dir, dataset, *, batch_size=1000):
    """Yield (raw row positions, four-vectors, truth) without loading a full file."""
    path = Path(data_dir) / RAW_FILES[dataset]
    if dataset == 'BB1':
        labels = np.loadtxt(path.with_suffix('.masterkey'), dtype=np.float32).astype(np.int8)

    with tables.open_file(path) as file:
        if dataset in ('LHCO', 'extra_qcd', 'BB1'):
            events = file.root.df.block0_values
        else:
            events = file.root.Particles.table

        for start in range(0, events.nrows, batch_size):
            stop = min(start + batch_size, events.nrows)
            chunk = events.read(start, stop)
            if dataset == 'LHCO':
                particles, truth = chunk[:, :2100], chunk[:, 2100]
            elif dataset == 'extra_qcd':
                # Particle values are pT/eta/phi despite the release's px/py/pz names.
                # The trailing 14 columns are jet features, not particles.
                particles, truth = chunk[:, :2100], np.zeros(stop - start)
            elif dataset == 'BB1':
                particles, truth = chunk, labels[start:stop]
            else:
                particles = chunk['values_block_0']
                truth = chunk['values_block_1'][:, 0]
            vectors = four_vectors(particles.reshape(-1, 700, 3))
            yield np.arange(start, stop), vectors, truth.astype(np.int8)


def cluster_jets(vectors):
    """Cluster a batch; retain event positions and pack variable-length constituents."""
    nonzero = vectors[..., 3] > 0
    flat_vectors = vectors[nonzero]
    particles = ak.zip({name: flat_vectors[:, i] for i, name in enumerate(FOUR_VECTOR_COLUMNS)})
    particles = ak.unflatten(particles, nonzero.sum(axis=1))
    sequence = fastjet.ClusterSequence(particles, fastjet.JetDefinition(fastjet.antikt_algorithm, 1.0))
    jets = sequence.inclusive_jets()
    order = ak.argsort(jets.px**2 + jets.py**2, ascending=False)[:, :2]
    jets = jets[order]
    constituents = sequence.constituents()[order]
    constituents = constituents[ak.argsort(constituents.px**2 + constituents.py**2, ascending=False)]

    valid = ak.to_numpy(ak.num(jets, axis=1) == 2)
    padded_jets = ak.pad_none(jets, 2, axis=1, clip=True)
    jet_vectors = np.stack([
        ak.to_numpy(ak.fill_none(padded_jets[name], np.nan)) for name in FOUR_VECTOR_COLUMNS
    ], axis=-1)
    counts = ak.pad_none(ak.num(constituents, axis=2), 2, axis=1, clip=True)
    counts = ak.to_numpy(ak.fill_none(counts, 0)).reshape(-1)
    flat = ak.flatten(ak.flatten(constituents, axis=1), axis=1)
    constituent_vectors = np.stack([ak.to_numpy(flat[name]) for name in FOUR_VECTOR_COLUMNS], axis=-1)
    total = jet_vectors.sum(axis=1)
    mass_squared = total[:, 3]**2 - np.sum(total[:, :3]**2, axis=1)

    return {
        'jets': jet_vectors,
        'constituents': constituent_vectors,
        'offsets': np.r_[0, np.cumsum(counts)],
        'mjj': np.sqrt(np.maximum(mass_squared, 0)) / 1000,
        'valid': valid,
    }


def preprocess_dataset(data_dir, output_dir, dataset, *, batch_size=1000):
    """Write one compressed, batch-readable HDF5 file per raw source, not per split."""
    output = Path(output_dir) / RAW_FILES[dataset]
    if output.resolve() == (Path(data_dir) / RAW_FILES[dataset]).resolve():
        raise ValueError('Output must not overwrite the raw input file')
    output.parent.mkdir(parents=True, exist_ok=True)
    schema = {
        'row': ('int64', ()), 'truth': ('int8', ()), 'valid': ('bool', ()),
        'mjj': ('float32', ()), 'jets': ('float32', (2, 4)),
        'offsets': ('int64', ()), 'constituents': ('float32', (4,)),
    }
    with tables.open_file(output, mode='w') as file:
        attrs = file.root._v_attrs
        attrs.source_file = RAW_FILES[dataset]
        attrs.four_vector_order = 'px,py,pz,E'
        attrs.momentum_unit = 'GeV'
        attrs.mass_unit = 'TeV'
        attrs.jet_algorithm = 'antikt'
        attrs.jet_radius = 1.0
        arrays = {
            name: file.create_earray(
                '/', name, atom=tables.Atom.from_dtype(np.dtype(dtype)), shape=(0, *shape),
                filters=tables.Filters(complevel=3, complib='blosc:zstd'),
            )
            for name, (dtype, shape) in schema.items()
        }
        arrays['offsets'].append(np.array([0], dtype=np.int64))
        missing = 0
        for rows, vectors, truth in iter_event_batches(data_dir, dataset, batch_size=batch_size):
            result = cluster_jets(vectors)
            result['offsets'] = result['offsets'][1:] + arrays['constituents'].nrows
            result.update(row=rows, truth=truth)
            for name, values in result.items():
                arrays[name].append(values.astype(arrays[name].atom.dtype, copy=False))
            missing += np.count_nonzero(~result['valid'])
        print(f'{dataset}: {arrays["row"].nrows:,} events, {missing:,} without two jets -> {output}')
    return output
