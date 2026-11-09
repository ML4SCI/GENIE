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


def _massless(pt, eta, phi):
    """Massless constituents as (px, py, pz, E) rows."""
    pt, eta, phi = (np.asarray(values, dtype=float) for values in (pt, eta, phi))
    momentum = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)], axis=-1)
    return np.column_stack([momentum, np.linalg.norm(momentum, axis=-1)])


def _wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


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

    def test_relative_angles_are_invariant_under_azimuthal_rotation(self):
        rng = np.random.default_rng(7)
        jets = [
            _massless(rng.uniform(1.0, 50.0, n), eta + rng.uniform(-0.5, 0.5, n),
                      phi + rng.uniform(-0.4, 0.4, n))
            for n, eta, phi in ((4, 0.8, 0.3), (1, -1.2, 2.5), (6, 0.4, -2.9))
        ]
        angle = 1.234
        rotation = np.eye(4)
        rotation[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        base = _transform(_dataframe_from_jets(jets))
        rotated = _transform(_dataframe_from_jets([jet @ rotation.T for jet in jets]))

        for field in ("part_deta", "part_dphi", "jet_pt", "jet_eta", "jet_mass"):
            np.testing.assert_allclose(ak.to_numpy(rotated[field]), ak.to_numpy(base[field]),
                                       atol=1e-9, err_msg=field)
        jet_phi_shift = ak.to_numpy(rotated["jet_phi"]) - ak.to_numpy(base["jet_phi"])
        np.testing.assert_allclose(_wrap(jet_phi_shift - angle), 0.0, atol=1e-12)

    def test_dphi_wraps_across_the_phi_branch_cut(self):
        # The constituents straddle phi = +-pi, so the jet axis sits next to the cut.
        phi = np.array([np.pi - 0.05, -np.pi + 0.05, np.pi - 0.01])
        jet = _massless([50.0, 40.0, 30.0], [0.1, -0.1, 0.0], phi)
        result = _transform(_dataframe_from_jets([jet]))

        jet_phi = np.arctan2(jet[:, 1].sum(), jet[:, 0].sum())
        self.assertGreater(np.max(np.abs(phi - jet_phi)), np.pi)  # naive difference
        dphi = ak.to_numpy(result["part_dphi"])
        np.testing.assert_allclose(dphi, _wrap(phi - jet_phi), atol=1e-12)
        self.assertTrue(np.all(np.abs(dphi) < 0.1))

    def test_empty_jet_keeps_other_jets_aligned(self):
        first = [(3.0, 0.0, 0.0, 5.0), (0.0, 2.0, 1.0, 4.0)]
        last = [(1.0, 1.0, 0.5, 3.0)]
        reference = _transform(_dataframe_from_jets([first, last]))
        result = _transform(_dataframe_from_jets([first, [], last]))

        self.assertEqual(result["jet_nparticles"].tolist(), [2, 0, 1])
        self.assertEqual(len(result["part_dphi"]), 3)
        np.testing.assert_array_equal(ak.to_numpy(result["jet_pt"])[1], 0.0)
        np.testing.assert_array_equal(ak.to_numpy(result["jet_energy"])[1], 0.0)
        for field in ("jet_pt", "jet_eta", "jet_phi", "jet_mass"):
            np.testing.assert_allclose(np.delete(ak.to_numpy(result[field]), 1),
                                       ak.to_numpy(reference[field]), err_msg=field)
        for field in ("part_pt", "part_deta", "part_dphi"):
            np.testing.assert_allclose(ak.to_numpy(result[field]),
                                       ak.to_numpy(reference[field]), err_msg=field)

        all_empty = _transform(_dataframe_from_jets([[], []]))
        self.assertEqual(all_empty["jet_nparticles"].tolist(), [0, 0])
        self.assertEqual(len(all_empty["part_dphi"]), 0)
        np.testing.assert_array_equal(ak.to_numpy(all_empty["jet_pt"]), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
