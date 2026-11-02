"""Read split events as pairs of jet graphs for torch_geometric.loader.DataLoader.

Each item has two ``jets`` plus event-level ``source``, ``row``, ``truth``,
``weak_label`` and ``mjj`` (TeV, from the split). A batch contains two PyG
Batch objects: leading jets and subleading jets in the same event order.
"""

from pathlib import Path

import numpy as np
import tables
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch


PACKED_NODES, PACKED_EDGES = 30, 318


def combine_jets(jets, *, positions=False):
    """One batch with all leading jets followed by matching subleading jets."""
    leading, subleading = jets
    combined = Batch(
        x=torch.cat([leading.x, subleading.x]),
        edge_index=torch.cat([leading.edge_index, subleading.edge_index + leading.num_nodes], dim=1),
        edge_attr=torch.cat([leading.edge_attr, subleading.edge_attr]),
        batch=torch.cat([leading.batch, subleading.batch + leading.num_graphs]),
        ptr=torch.cat([leading.ptr, subleading.ptr[1:] + leading.num_nodes]),
    )
    if positions:
        combined.pos = torch.cat([leading.pos, subleading.pos])
    return combined


def _read_jet(root, index):
    start, stop = root.node_offsets[index:index+2]
    edge_start, edge_stop = root.edge_offsets[index:index+2]
    return Data(
        x=torch.from_numpy(root.x[start:stop]),
        pos=torch.from_numpy(root.pos[start:stop]),
        pt=torch.from_numpy(root.pt[start:stop]),
        edge_index=torch.from_numpy(root.edge_index[edge_start:edge_stop]).t().contiguous(),
        edge_attr=torch.from_numpy(root.edge_attr[edge_start:edge_stop]),
    )


class EventDataset(Dataset):
    """Load only split metadata up front; read graph slices as batches are requested."""

    def __init__(self, split_path, *, split='train', graph_dir='data/graphs'):
        with np.load(split_path) as file:
            self.events = file[split]
            self.graph_paths = [Path(graph_dir) / name for name in file['source_files']]

    def __len__(self):
        return len(self.events)

    def __getitem__(self, index):
        return self.__getitems__([index])[0]

    def __getitems__(self, indices):
        """Open each source once per batch; keep the requested order and duplicate indices.

        PyTorch calls this for batched loading. Handles stay local to the call,
        so DataLoader workers never share or pickle open HDF5 files.
        """
        events = self.events[indices]
        items = [None] * len(events)
        for source in np.unique(events['source']):
            path = self.graph_paths[source]
            with tables.open_file(path) as file:
                for index in np.flatnonzero(events['source'] == source):
                    event = events[index]
                    row = int(event['row'])
                    if not file.root.valid[row]:
                        raise ValueError(f'{path}: row {row} has fewer than two jets')
                    items[index] = {
                        'jets': (_read_jet(file.root, 2*row), _read_jet(file.root, 2*row+1)),
                        'source': int(source),
                        'row': row,
                        'truth': int(event['truth']),
                        'weak_label': int(event['weak_label']),
                        'mjj': float(event['mjj']),
                    }
        return items


def _pack_batch(batch):
    """Two PyG jet batches -> fixed-width tensors indexed [event, jet, ...]; edges stay sparse."""
    count = len(batch['row'])
    data = dict(x=torch.zeros(count, 2, PACKED_NODES, 1),
                mask=torch.zeros(count, 2, PACKED_NODES, dtype=torch.bool),
                edge_index=torch.zeros(count, 2, 2, PACKED_EDGES, dtype=torch.int16),
                edge_attr=torch.zeros(count, 2, PACKED_EDGES, 3),
                edge_mask=torch.zeros(count, 2, PACKED_EDGES, dtype=torch.bool),
                positions=torch.zeros(count, 2, PACKED_NODES, 2),
                pt=torch.zeros(count, 2, PACKED_NODES))
    for slot, jets in enumerate(batch['jets']):
        if int(torch.bincount(jets.batch).max()) > PACKED_NODES:
            raise ValueError(f'jets must have at most {PACKED_NODES} subjets')
        data['x'][:, slot], data['mask'][:, slot] = to_dense_batch(jets.x, jets.batch, max_num_nodes=PACKED_NODES)
        data['positions'][:, slot] = to_dense_batch(jets.pos, jets.batch, max_num_nodes=PACKED_NODES)[0]
        data['pt'][:, slot] = to_dense_batch(jets.pt, jets.batch, max_num_nodes=PACKED_NODES)[0]
        graph = jets.batch[jets.edge_index[0]]
        counts = torch.bincount(graph, minlength=count)
        offsets = counts.cumsum(0) - counts
        position = torch.arange(len(graph)) - torch.repeat_interleave(offsets, counts)
        local = jets.edge_index - jets.ptr[graph]
        data['edge_index'][:, slot].permute(0, 2, 1)[graph, position] = local.T.short()
        data['edge_attr'][:, slot][graph, position] = jets.edge_attr
        data['edge_mask'][:, slot][graph, position] = True
    return data


def pack_events(split_path, split, *, graph_dir='data/graphs', workers=8, batch_size=512):
    """Read every event of one split into packed tensors, in split order (see jet_batch)."""
    dataset = EventDataset(split_path, split=split, graph_dir=graph_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    data, offset = None, 0
    for batch in loader:
        values = _pack_batch(batch)
        if data is None:
            data = {key: torch.empty((len(dataset), *value.shape[1:]), dtype=value.dtype)
                    for key, value in values.items()}
        count = len(batch['row'])
        if not np.array_equal(batch['row'].numpy(), dataset.events['row'][offset:offset+count]):
            raise RuntimeError('packed events out of split order')
        for key, value in values.items():
            data[key][offset:offset+count] = value
        offset += count
    return data, dataset.events


def jet_batch(data, index, device):
    """Packed events -> dense per-jet inputs (x, adjacency, mask, edges), first jets then second jets."""
    values = {name: torch.cat([data[name][index, 0], data[name][index, 1]]).to(device)
              for name in ('x', 'mask', 'edge_index', 'edge_attr', 'edge_mask')}
    x, mask = values['x'], values['mask']
    size, nodes = mask.shape
    graph, edge_number = values['edge_mask'].nonzero(as_tuple=True)
    source = values['edge_index'][graph, 0, edge_number].long()
    target = values['edge_index'][graph, 1, edge_number].long()
    adjacency = torch.zeros(size, nodes, nodes, device=device)
    edges = torch.zeros(size, nodes, nodes, 3, device=device)
    adjacency[graph, source, target] = 1.
    edges[graph, source, target] = values['edge_attr'][graph, edge_number]
    return x, adjacency, mask, edges
