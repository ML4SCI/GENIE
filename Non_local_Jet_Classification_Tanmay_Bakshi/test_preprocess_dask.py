"""Small regression tests for jet constituent preprocessing."""

import unittest

import awkward as ak
import numpy as np
import pandas as pd

from preprocess_dask import _extract_coords, _transform


def _two_jet_dataframe():
    """Two jets with unequal multiplicities and padded constituent slots."""
    fields = {}
    components = {
        "PX": [(2.0, 0.0), (0.0, 0.0)],
        "PY": [(0.0, 1.0), (2.0, 0.0)],
        "PZ": [(1.0, -0.25), (1.0, 0.0)],
        "E": [(4.0, 3.0), (3.0, 0.0)],
    }
    for component, jets in components.items():
        for particle_index in range(200):
            fields[f"{component}_{particle_index}"] = [
                jet[particle_index] if particle_index < len(jet) else 0.0
                for jet in jets
            ]
    fields["is_signal_new"] = [1, 0]
    fields["ttv"] = [0, 1]
    return pd.DataFrame(fields)


def _dataframe_from_jets(jets):
    """Build zero-padded constituent data for arbitrary jet multiplicities."""
    fields = {}
    for component_index, component in enumerate(("PX", "PY", "PZ", "E")):
        for particle_index in range(200):
            fields[f"{component}_{particle_index}"] = [
                float(jet[particle_index][component_index])
                if particle_index < len(jet) else 0.0
                for jet in jets
            ]
    fields["is_signal_new"] = [index % 2 for index in range(len(jets))]
    fields["ttv"] = [0] * len(jets)
    return pd.DataFrame(fields)


class TestJetPreprocessing(unittest.TestCase):
    def setUp(self):
        self.frame = _two_jet_dataframe()

    def test_default_slice_keeps_last_jet(self):
        coords = _extract_coords(self.frame)
        self.assertEqual(coords["jet_nparticles"].tolist(), [2, 1])
        self.assertEqual(coords["label"].shape[0], 2)

    def test_transform_computes_jet_relative_angles(self):
        result = _transform(self.frame)

        self.assertEqual(result["jet_nparticles"].tolist(), [2, 1])
        self.assertEqual(len(result["jet_mass"]), 2)
        self.assertEqual(len(result["part_phi"]), 3)

        particle_phi = ak.to_numpy(result["part_phi"])
        jet_phi = ak.to_numpy(result["jet_phi"])
        expected_dphi = np.arctan2(
            np.sin(particle_phi - np.repeat(jet_phi, [2, 1])),
            np.cos(particle_phi - np.repeat(jet_phi, [2, 1])),
        )
        np.testing.assert_allclose(ak.to_numpy(result["part_dphi"]), expected_dphi)
        self.assertGreater(abs(expected_dphi[0]), 0.1)

        particle_eta = ak.to_numpy(result["part_eta"])
        jet_eta = ak.to_numpy(result["jet_eta"])
        eta_sign = np.where(np.sign(jet_eta) == 0, 1, np.sign(jet_eta))
        expected_deta = (particle_eta - np.repeat(jet_eta, [2, 1])) * np.repeat(eta_sign, [2, 1])
        np.testing.assert_allclose(ak.to_numpy(result["part_deta"]), expected_deta)

    def test_jet_pt_single_jet(self):
        frame = _dataframe_from_jets([[(2.0, 0.0, 1.0, 4.0), (0.0, 1.0, -0.25, 3.0)]])
        result = _transform(frame)
        self.assertEqual(len(result["jet_pt"]), 1)
        np.testing.assert_allclose(ak.to_numpy(result["jet_pt"]), [np.sqrt(5.0)])
        np.testing.assert_allclose(ak.to_numpy(result["part_pt"]), [2.0, 1.0])

    def test_jet_pt_unequal_multiplicity_multi_jet(self):
        jets = [
            [(3.0, 0.0, 0.0, 5.0), (-1.0, 0.0, 0.0, 3.0), (0.0, 2.0, 0.0, 4.0)],
            [(0.0, 3.0, 0.0, 4.0)],
            [(1.0, 0.0, 0.0, 3.0), (0.0, -2.0, 0.0, 4.0)],
        ]
        result = _transform(_dataframe_from_jets(jets))
        self.assertEqual(len(result["jet_pt"]), 3)
        self.assertEqual(len(result["part_pt"]), 6)
        self.assertEqual(result["jet_nparticles"].tolist(), [3, 1, 2])
        expected_jet_pt = [np.sqrt(8.0), 3.0, np.sqrt(5.0)]
        np.testing.assert_allclose(ak.to_numpy(result["jet_pt"]), expected_jet_pt)
        np.testing.assert_allclose(
            ak.to_numpy(result["part_pt"]), [3.0, 1.0, 2.0, 3.0, 1.0, 2.0]
        )

    def test_extract_coords_jet_pt_stays_constituent_level(self):
        coords = _extract_coords(self.frame)
        self.assertEqual(len(coords["jet_pt"]), 3)
        np.testing.assert_allclose(ak.to_numpy(coords["jet_pt"]), [2.0, 1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
