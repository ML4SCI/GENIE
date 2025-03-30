import dask as da
import numpy as np
import pandas as pd
import numba as nb
import vector
import awkward as ak
import os
import argparse
import pprint
from typing import Dict, List, Optional
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@nb.jit(nopython=True)
def _create_mask(e: np.ndarray) -> np.ndarray:
    """Create mask for valid particles using Numba for speed."""
    return e > 0

@nb.jit(nopython=True)
def _calculate_n_particles(mask: np.ndarray) -> np.ndarray:
    """Calculate number of particles per event using Numba."""
    return np.sum(mask, axis=1)

def _col_list(prefix: str, max_particles: int = 200) -> List[str]:
    """Generate list of column names for particle data."""
    return [f'{prefix}_{i}' for i in range(max_particles)]

def _extract_coords(df: pd.DataFrame, start: int = 0, stop: int = -1) -> Dict:
    """
    Extract coordinates and features from DataFrame efficiently.
    
    Args:
        df: Input DataFrame
        start: Starting index
        stop: Ending index
        
    Returns:
        Dictionary containing extracted features
    """
    try:
        from collections import OrderedDict
        v = OrderedDict()

        # Slice DataFrame
        df = df.iloc[start:stop]
        
        # Extract particle data efficiently
        px = df[_col_list(prefix='PX')].values
        py = df[_col_list(prefix='PY')].values
        pz = df[_col_list(prefix='PZ')].values
        e = df[_col_list(prefix='E')].values
        
        # Create mask and calculate particles using Numba
        mask = _create_mask(e)
        n_particles = _calculate_n_particles(mask)
        
        # Convert to awkward arrays efficiently
        px_ak = ak.Array(px[mask])
        py_ak = ak.Array(py[mask])
        pz_ak = ak.Array(pz[mask])
        energy_ak = ak.Array(e[mask])
        
        # Calculate four-momentum
        p4 = _p4_from_pxpypze(px_ak, py_ak, pz_ak, energy_ak)
        
        # Calculate features in vectorized manner
        v.update({
            'x': p4.x,
            'y': p4.y,
            'z': p4.z,
            't': p4.t,
            'phi': p4.phi,
            'rho': p4.rho,
            'theta': p4.theta,
            'eta': p4.eta,
            'jet_pt': p4.pt,
            'jet_nparticles': n_particles,
            'label': np.stack((df['is_signal_new'].values, 1-df['is_signal_new'].values), axis=-1),
            'train_val_test': df['ttv'].values
        })
        
        return v
        
    except Exception as e:
        logger.error(f"Error in _extract_coords: {str(e)}")
        raise

def _p4_from_pxpypze(px: ak.Array, py: ak.Array, pz: ak.Array, e: ak.Array) -> vector.LorentzVector:
    """Calculate four-momentum from components efficiently."""
    vector.register_awkward()
    return vector.zip({'px': px, 'py': py, 'z': pz, 'E': e})

def _transform(df: pd.DataFrame, start: int = 0, stop: int = -1) -> Dict:
    """
    Transform particle data into jet features efficiently.
    
    Args:
        df: Input DataFrame
        start: Starting index
        stop: Ending index
        
    Returns:
        Dictionary containing transformed features
    """
    try:
        from collections import OrderedDict
        v = OrderedDict()

        # Slice DataFrame
        df = df.iloc[start:stop]
        
        # Extract particle data efficiently
        px = df[_col_list(prefix='PX')].values
        py = df[_col_list(prefix='PY')].values
        pz = df[_col_list(prefix='PZ')].values
        e = df[_col_list(prefix='E')].values
        
        # Create mask and calculate particles using Numba
        mask = _create_mask(e)
        n_particles = _calculate_n_particles(mask)
        
        # Convert to awkward arrays efficiently
        px_ak = ak.Array(px[mask])
        py_ak = ak.Array(py[mask])
        pz_ak = ak.Array(pz[mask])
        energy_ak = ak.Array(e[mask])
        
        # Calculate four-momentum
        p4 = _p4_from_pxpypze(px_ak, py_ak, pz_ak, energy_ak)
        
        # Calculate jet features
        jet_p4 = ak.sum(p4, axis=1)
        
        # Calculate features in vectorized manner
        v.update({
            'jet_pt': p4.pt,
            'part_eta': p4.eta,
            'part_phi': p4.phi,
            'jet_eta': jet_p4.eta,
            'jet_phi': jet_p4.phi,
            'jet_energy': jet_p4.energy,
            'jet_mass': jet_p4.mass,
            'jet_nparticles': n_particles,
            'part_px': px_ak,
            'part_py': py_ak,
            'part_pz': pz_ak,
            'part_energy': energy_ak,
            'part_deta': (p4.eta - jet_p4.eta) * np.sign(jet_p4.eta),
            'part_dphi': p4.deltaphi(p4),
            'label': np.stack((df['is_signal_new'].values, 1-df['is_signal_new'].values), axis=-1),
            'train_val_test': df['ttv'].values
        })
        
        return v
        
    except Exception as e:
        logger.error(f"Error in _transform: {str(e)}")
        raise

def natural_sort(l: List[str]) -> List[str]:
    """Sort strings with numbers naturally."""
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