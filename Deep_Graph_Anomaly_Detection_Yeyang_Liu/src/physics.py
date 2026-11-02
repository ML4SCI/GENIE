"""Jet observables for the physics-informed AE: graph kinematics, energy-flow polynomials and targets.

Masses and energy-flow polynomials are computed from the AE input graphs (30 massless subjets with
pT and centred rapidity/phi). Subjettiness targets come from the published Kitchen Sink tables.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tables
import torch

from .models.netge import log_fractions


SUBJETTINESS_COLUMNS = ['mj1', 'mdiff'] + [f'tau{n}j{jet}_{beta}' for jet in (1, 2) for beta in (5, 1, 2)
                                           for n in range(1, 10)]
# Column index of tau_N (beta=1) for the lighter (1) and heavier (2) jet.
TAU_BETA1 = {jet: {n: (11 if jet == 1 else 38) + n - 1 for n in range(1, 10)} for jet in (1, 2)}


def graph_jet_masses(path):
    """Native rows and both jet masses [TeV] (stored jet order) from one graph file."""
    with tables.open_file(path) as file:
        pt = file.root.pt[:].astype(np.float64)
        pos = file.root.pos[:].astype(np.float64)
        offsets = file.root.node_offsets[:]
        rows = file.root.row[:]
    rapidity, phi = pos[:, 0], pos[:, 1]
    momenta = np.stack([pt*np.cosh(rapidity), pt*np.cos(phi), pt*np.sin(phi), pt*np.sinh(rapidity)], 1)
    sums = np.add.reduceat(momenta, offsets[:-1], axis=0)
    sums[offsets[1:] == offsets[:-1]] = 0
    mass2 = sums[:, 0]**2 - (sums[:, 1:]**2).sum(1)
    return rows, (np.sqrt(np.clip(mass2, 0, None)) * 1e-3).reshape(-1, 2)


def event_masses(events, graph_paths):
    """Per event (lighter, heavier) jet mass [TeV]; events carry 'source' and 'row'."""
    out = np.full((len(events), 2), np.nan)
    for source, path in enumerate(graph_paths):
        selected = events['source'] == source
        if not selected.any():
            continue
        rows, mass = graph_jet_masses(path)
        order = np.argsort(rows)
        index = order[np.searchsorted(rows[order], events['row'][selected])]
        if not np.array_equal(rows[index], events['row'][selected]):
            raise ValueError(f'{path}: events missing from the graph file')
        out[selected] = np.sort(mass[index], 1)
    return out


def mass_features(events, graph_paths):
    """Readout mass inputs as in Kitchen Sink: lighter jet mass and mass difference [TeV]."""
    mass = event_masses(events, graph_paths)
    return np.c_[mass[:, 0], mass[:, 1] - mass[:, 0]].astype(np.float32)


def energy_flow_polynomials(pt, pos):
    """Eight connected EFPs (degree <= 3, beta = kappa = 1). pt[B,N], pos[B,N,2]; zero pT is padding."""
    pt, pos = pt.double(), pos.double()
    q = pt / pt.sum(-1, keepdim=True)
    delta = pos[:, :, None] - pos[:, None, :]
    dphi = (delta[..., 1] + torch.pi) % (2*torch.pi) - torch.pi
    theta = torch.sqrt(delta[..., 0].square() + dphi.square())
    a1 = (theta*q[:, None]).sum(-1)
    a2 = (theta.square()*q[:, None]).sum(-1)
    a3 = (theta.pow(3)*q[:, None]).sum(-1)
    weighted = theta*q[:, None]
    triangle = ((weighted @ weighted)*weighted.transpose(-1, -2)).sum((-1, -2))
    chain = (q*a1*(theta*(q*a1)[:, None]).sum(-1)).sum(-1)
    values = ((q*a1).sum(-1), (q*a2).sum(-1), (q*a1.square()).sum(-1), (q*a3).sum(-1),
              (q*a1*a2).sum(-1), triangle, (q*a1.pow(3)).sum(-1), chain)
    return torch.stack(values, -1)


def event_energy_flow(data, device='cpu', chunk=4096):
    """Per event 16 coordinates: log EFPs of both jets combined as sum and |difference|."""
    out = []
    for start in range(0, len(data['pt']), chunk):
        mask = data['mask'][start:start+chunk]
        pt = (data['pt'][start:start+chunk] * mask).to(device)
        pos = (data['positions'][start:start+chunk] * mask[..., None]).to(device)
        values = torch.stack([energy_flow_polynomials(pt[:, j], pos[:, j]) for j in (0, 1)], 1).cpu().numpy()
        logs = np.log(np.maximum(values.astype(np.float64), 1e-12))
        out.append(np.concatenate([logs[:, 0] + logs[:, 1], abs(logs[:, 0] - logs[:, 1])], -1).astype(np.float32))
    return np.concatenate(out)


def subjettiness(events, feature_dir='data/features/kitchensink', table='LHCO'):
    """Raw Kitchen Sink subjettiness observables (float64) for events of one table, by native row."""
    frame = pd.read_hdf(Path(feature_dir)/table/'subjettiness.h5')
    return frame[SUBJETTINESS_COLUMNS].to_numpy(np.float64)[events['row']]


def tau_ratios(raw):
    """log tau21, tau32, tau43 (beta=1) of the lighter then heavier jet."""
    columns = []
    for jet in (1, 2):
        for n in (2, 3, 4):
            numerator, denominator = raw[:, TAU_BETA1[jet][n]], raw[:, TAU_BETA1[jet][n-1]]
            columns.append(np.log(np.clip(numerator, 1e-6, None)/np.clip(denominator, 1e-6, None)))
    return np.stack(columns, 1)


def standardization(values, rows):
    selected = values[rows].astype(np.float64)
    return selected.mean(0), selected.std(0)


def input_normalization(data, rows, chunk=1024):
    """Node (log pT fraction) and edge statistics over the real nodes/edges of the given events."""
    totals = {}
    for name in ('node', 'edge'):
        width = 1 if name == 'node' else 3
        total, squares, count = torch.zeros(width, dtype=torch.float64), torch.zeros(width, dtype=torch.float64), 0
        for start in range(0, len(rows), chunk):
            index = rows[start:start+chunk]
            if name == 'node':
                mask = data['mask'][index]
                values = log_fractions(data['x'][index], mask)[mask].double()
            else:
                values = data['edge_attr'][index][data['edge_mask'][index]].double()
            total += values.sum(0)
            squares += values.square().sum(0)
            count += len(values)
        center = total/count
        totals[name+'_center'] = center.float()
        totals[name+'_scale'] = (squares/count - center.square()).sqrt().float()
    return totals
