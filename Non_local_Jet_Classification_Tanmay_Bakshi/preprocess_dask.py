import dask as da
import numpy as np
import pandas as pd
import numba as nb
import vector
import awkward as ak
import os
import numpy as np
import argparse
import pprint

'''
Datasets introduction:
https://energyflow.network/docs/datasets/#quark-and-gluon-jets

Download:
- Pythia8 Quark and Gluon Jets for Energy Flow:
  - https://zenodo.org/record/3164691

- Herwig7.1 Quark and Gluon Jets:
  - https://zenodo.org/record/3066475

Versions:
 - awkward==2.6.4
 - vector==1.4.0
'''

def _extract_coords(df, start=0, stop=None):
    from collections import OrderedDict
    v = OrderedDict()

    def _col_list(prefix, max_particles=200):
        return ['%s_%d' % (prefix, i) for i in range(max_particles)]

    df = df.iloc[start:stop]
    # We take the values in the dataframe for all particles of a single event in each row
    # px, py, pz, e are in separate arrays

    _px = df[_col_list(prefix='PX')].values
    _py = df[_col_list(prefix='PY')].values
    _pz = df[_col_list(prefix='PZ')].values
    _e = df[_col_list(prefix='E')].values
    mask = _e > 0
    n_particles = np.sum(mask, axis=1)

    px = ak.Array(_px[mask])
    py = ak.Array(_py[mask])
    pz = ak.Array(_pz[mask])
    energy = ak.Array(_e[mask])
    del _px, _py, _pz, _e

    # px = ak.sum(px, axis=0)
    # py = ak.sum(py, axis=0)
    # pz = ak.sum(pz, axis=0)
    # energy = ak.sum(energy, axis=0)
    p4 = _p4_from_pxpypze(px, py, pz, energy)
    v['x'] = p4.x
    # print("x : ", type(v['x']))
    # print("\n")
    v['y'] = p4.y
    # print("y : ", v['y'])
    # print("\n")
    v['z'] = p4.z
    # print("z : ", v['z'])
    # print("\n")
    # input()
    v['t'] = p4.t
    v['phi'] = p4.phi
    v['rho'] = p4.rho
    v['theta'] = p4.theta
    v['eta'] = p4.eta
    v['jet_pt'] = p4.pt
    v['jet_nparticles'] = n_particles

    _label = df['is_signal_new'].values
    v['label'] = np.stack((_label, 1-_label), axis=-1)
    v['train_val_test'] = df['ttv'].values
    return v

def _p4_from_pxpypze(px, py, pz, e):
    import vector
    vector.register_awkward()
    return vector.zip({'px': px, 'py': py, 'z': pz, 'E': e})







def _transform(df, start=0, stop=None):
    from collections import OrderedDict
    v = OrderedDict()

    # generate the column list to be extracted
    def _col_list(prefix, max_particles=200):
        return ['%s_%d' % (prefix, i) for i in range(max_particles)]

    df = df.iloc[start:stop]
    # We take the values in the dataframe for all particles of a single event in each row
    # px, py, pz, e are in separate arrays

    _px = df[_col_list(prefix='PX')].values
    _py = df[_col_list(prefix='PY')].values
    _pz = df[_col_list(prefix='PZ')].values
    _e = df[_col_list(prefix='E')].values
    mask = _e > 0
    n_particles = np.sum(mask, axis=1)

    px = ak.Array(_px[mask])
    py = ak.Array(_py[mask])
    pz = ak.Array(_pz[mask])
    energy = ak.Array(_e[mask])
    del _px, _py, _pz, _e

    # px = ak.sum(px, axis=0)
    # py = ak.sum(py, axis=0)
    # pz = ak.sum(pz, axis=0)
    # energy = ak.sum(energy, axis=0)
    p4 = _p4_from_pxpypze(px, py, pz, energy)

    # Boolean indexing flattens the constituents. Restore the per-jet structure
    # before computing jet four-momenta and particle-to-jet angular offsets.
    particles_by_jet = ak.unflatten(p4, n_particles)
    jet_p4 = ak.sum(particles_by_jet, axis=1)

    v['jet_pt'] = p4.pt
    v['part_eta'] = p4.eta
    v['part_phi'] = p4.phi
    v['jet_eta'] = jet_p4.eta
    v['jet_phi'] = jet_p4.phi
    v['jet_energy'] = jet_p4.energy
    v['jet_mass'] = jet_p4.mass
    v['jet_nparticles'] = n_particles

    v['part_px'] = px
    v['part_py'] = py
    v['part_pz'] = pz
    v['part_energy'] = energy

    _jet_etasign = ak.to_numpy(np.sign(v['jet_eta']))
    _jet_etasign[_jet_etasign == 0] = 1
    # Broadcast each jet's direction to its constituents, then flatten back
    # to the original per-particle layout used by the other output fields.
    jet_eta = jet_p4.eta[:, np.newaxis]
    jet_axis = jet_p4[:, np.newaxis]
    v['part_deta'] = ak.flatten(
        (particles_by_jet.eta - jet_eta) * _jet_etasign[:, np.newaxis]
    )
    v['part_dphi'] = ak.flatten(particles_by_jet.deltaphi(jet_axis))

    _label = df['is_signal_new'].values
    v['label'] = np.stack((_label, 1-_label), axis=-1)
    v['train_val_test'] = df['ttv'].values


    del px, py, pz, energy, p4, _jet_etasign, _label
    return v



def natural_sort(l):
    import re
    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

#
# if __name__ == '__main__':
#
#     parser = argparse.ArgumentParser('Convert qg benchmark datasets')
#     parser.add_argument('-i', '--inputdir', required=True, help='Directory of input numpy files.')
#     parser.add_argument('-o', '--outputdir', required=True, help='Output directory.')
#     parser.add_argument('--train-test-split', default=0.9, help='Training / testing split fraction.')
#     args = parser.parse_args()
#
#     import glob
#     sources = natural_sort(glob.glob(os.path.join(args.inputdir, 'QG_jets*.npz')))
#     n_train = int(args.train_test_split * len(sources))
#     train_sources = sources[:n_train]
#     test_sources = sources[n_train:]
#
#     convert(train_sources, destdir=args.outputdir, basename='train_file')
#     convert(test_sources, destdir=args.outputdir, basename='test_file')
