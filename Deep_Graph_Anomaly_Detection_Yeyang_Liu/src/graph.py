"""Exclusive-kT subjet graphs with Unique-6 edges; default maximum 30 subjets.

For event r and jet j, graph index g = 2*r+j. Slice nodes with
node_offsets[g:g+2] and edges with edge_offsets[g:g+2]. Stored edge_index
has shape (edges, 2) and uses graph-local indices; transpose it for PyG.
"""

from pathlib import Path

import awkward as ak
import fastjet
import numpy as np
import tables


N_SUBJETS = 30
N_NEIGHBOURS = 6
FOUR_VECTOR_COLUMNS = ('px', 'py', 'pz', 'E')


def recluster_subjets(constituents, offsets, *, max_subjets=N_SUBJETS):
    """Batch exclusive-kT R=1 reclustering; smaller jets remain unpadded."""
    momentum = np.asarray(constituents[:, :3], dtype=np.float64)
    # Restore massless energies after float32 storage of constituent four-vectors.
    energy = np.linalg.norm(momentum, axis=1)
    particles = ak.zip(
        {'px': momentum[:, 0], 'py': momentum[:, 1], 'pz': momentum[:, 2], 'E': energy},
        with_name='Momentum4D',
    )
    constituent_counts = np.diff(offsets)
    particles_by_jet = ak.unflatten(particles, constituent_counts)
    needs_clustering = constituent_counts > max_subjets
    subjets = particles_by_jet
    if needs_clustering.any():
        definition = fastjet.JetDefinition(fastjet.kt_algorithm, 1.0)
        sequence = fastjet.ClusterSequence(particles_by_jet[needs_clustering], definition)
        clustered_subjets = sequence.exclusive_jets(n_jets=max_subjets)
        grouped_subjets = ak.concatenate([particles_by_jet[~needs_clustering], clustered_subjets])

        # Concatenation puts unclustered jets first; restore the original jet order.
        grouped_jet_indices = np.concatenate([
            np.flatnonzero(~needs_clustering),
            np.flatnonzero(needs_clustering),
        ])
        restore_order = np.argsort(grouped_jet_indices)
        subjets = grouped_subjets[restore_order]

    pt_squared = subjets.px**2 + subjets.py**2
    return subjets[ak.argsort(pt_squared, ascending=False)]


def build_edges(pos):
    """Seed a clique of seven hardest nodes; connect each later node to six prior neighbours."""
    n = len(pos)
    if n <= N_NEIGHBOURS + 1:
        return np.stack(np.nonzero(~np.eye(n, dtype=bool)))

    source, target = np.triu_indices(N_NEIGHBOURS + 1, 1)
    source = source.tolist()
    target = target.tolist()
    for node in range(N_NEIGHBOURS + 1, n):
        distance_squared = np.sum((pos[:node] - pos[node])**2, axis=1)
        neighbours = np.argsort(distance_squared, kind='stable')[:N_NEIGHBOURS]
        source.extend([node] * N_NEIGHBOURS)
        target.extend(neighbours.tolist())
    return np.array([source + target, target + source], dtype=np.int64)


def edge_features(pos, pt, edge_index):
    """Return (ln theta, ln kT, ln z), with pT normalized within each jet."""
    pos = np.asarray(pos, dtype=np.float64)
    pt = np.asarray(pt, dtype=np.float64)
    q = pt / (pt.sum() + 1e-12)
    source, target = edge_index
    theta = np.linalg.norm(pos[source] - pos[target], axis=1)
    pt_min = np.minimum(q[source], q[target])
    kt = pt_min * theta
    z = pt_min / np.maximum(q[source] + q[target], 1e-12)
    values = np.column_stack([theta, kt, z])
    return np.log(np.maximum(values, [1e-6, 1e-6, 1e-12])).astype(np.float32)


def build_graph(nodes):
    """Build one graph from pT-sorted subjet four-vectors; x is only ln(pT)."""
    if len(nodes) == 0:
        return {
            'x': np.empty((0, 1), dtype=np.float32),
            'pos': np.empty((0, 2), dtype=np.float32),
            'pt': np.empty(0, dtype=np.float32),
            'edge_index': np.empty((2, 0), dtype=np.int64),
            'edge_attr': np.empty((0, 3), dtype=np.float32),
        }

    px, py, pz, energy = np.asarray(nodes, dtype=np.float64).T
    pt = np.hypot(px, py)
    # Merged subjets are massive: use rapidity, not pseudorapidity.
    rapidity = 0.5 * np.log((energy + pz) / (energy - pz))
    rapidity -= np.sum(pt * rapidity) / (pt.sum() + 1e-12)
    phi = np.arctan2(py, px)
    phi_center = np.arctan2(np.sum(pt * np.sin(phi)), np.sum(pt * np.cos(phi)))
    pos = np.column_stack([
        rapidity,
        (phi - phi_center + np.pi) % (2*np.pi) - np.pi,
    ]).astype(np.float32)
    x = np.log(pt[:, None] + 1e-10).astype(np.float32)
    pt = pt.astype(np.float32)
    edges = build_edges(pos)
    return {
        'x': x,
        'pos': pos,
        'pt': pt,
        'edge_index': edges,
        'edge_attr': edge_features(pos, pt, edges),
    }


def build_graph_file(input_path, output_path, *, batch_size=1000,
                     max_subjets=N_SUBJETS):
    """Stream a processed jet file into one packed graph file shared by all splits."""
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError('Output must not overwrite the processed input file')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema = {
        'row': ('int64', ()), 'truth': ('int8', ()), 'valid': ('bool', ()),
        'mjj': ('float32', ()), 'x': ('float32', (1,)), 'pos': ('float32', (2,)),
        'pt': ('float32', ()), 'edge_index': ('int64', (2,)), 'edge_attr': ('float32', (3,)),
        'node_offsets': ('int64', ()), 'edge_offsets': ('int64', ()),
    }
    with tables.open_file(input_path) as source, tables.open_file(output_path, mode='w') as output:
        attrs = output.root._v_attrs
        attrs.source_file = source.root._v_attrs.source_file
        attrs.mass_unit = source.root._v_attrs.mass_unit
        attrs.momentum_unit = 'GeV'
        attrs.max_subjets = max_subjets
        attrs.subjet_algorithm = 'exclusive_kt'
        attrs.subjet_radius = 1.0
        attrs.node_features = 'ln_pt'
        attrs.position = 'relative_rapidity,relative_phi'
        attrs.neighbours = N_NEIGHBOURS
        attrs.edge_features = 'ln_theta,ln_kt,ln_z'
        attrs.edge_pt_scale = 'normalized'
        arrays = {
            name: output.create_earray(
                '/', name, atom=tables.Atom.from_dtype(np.dtype(dtype)), shape=(0, *shape),
                filters=tables.Filters(complevel=3, complib='blosc:zstd'),
            )
            for name, (dtype, shape) in schema.items()
        }
        for name in ('node_offsets', 'edge_offsets'):
            arrays[name].append(np.array([0], dtype=np.int64))

        for start in range(0, source.root.row.nrows, batch_size):
            stop = min(start + batch_size, source.root.row.nrows)
            offsets = source.root.offsets[2*start:2*stop+1]
            constituents = source.root.constituents[offsets[0]:offsets[-1]]
            subjets = recluster_subjets(constituents, offsets - offsets[0], max_subjets=max_subjets)
            counts = ak.to_numpy(ak.num(subjets, axis=1))
            bounds = np.r_[0, np.cumsum(counts)]
            flat = ak.flatten(subjets, axis=1)
            vectors = np.column_stack([ak.to_numpy(flat[name]) for name in FOUR_VECTOR_COLUMNS])
            graphs = [
                build_graph(vectors[node_start:node_stop])
                for node_start, node_stop in zip(bounds[:-1], bounds[1:])
            ]

            arrays['node_offsets'].append(bounds[1:] + arrays['x'].nrows)
            edge_counts = [len(graph['edge_attr']) for graph in graphs]
            arrays['edge_offsets'].append(np.cumsum(edge_counts) + arrays['edge_attr'].nrows)
            for name in ('x', 'pos', 'pt', 'edge_attr'):
                arrays[name].append(np.concatenate([graph[name] for graph in graphs]))
            arrays['edge_index'].append(np.concatenate([graph['edge_index'].T for graph in graphs]))
            for name in ('row', 'truth', 'valid', 'mjj'):
                arrays[name].append(getattr(source.root, name)[start:stop])

        print(f'{arrays["row"].nrows:,} events, {arrays["x"].nrows:,} nodes -> {output_path}')
    return output_path
