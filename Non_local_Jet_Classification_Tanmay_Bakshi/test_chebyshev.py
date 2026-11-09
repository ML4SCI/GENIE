"""Regression tests for degenerate and integer-valued graph Laplacians."""

import importlib.util
import itertools
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from scnn import chebyshev
from scnn.build_matrices import build_boundaries, build_laplacians
from scnn.chebyshev import normalize

HAS_TORCH = importlib.util.find_spec("torch") is not None


def complex_laplacians(n_vertices, edges, triangles=()):
    """Laplacians of a simplicial complex, built exactly as the pipeline does."""
    simplices = [
        {frozenset([v]): v for v in range(n_vertices)},
        {frozenset(e): k for k, e in enumerate(edges)},
    ]
    if triangles:
        simplices.append({frozenset(t): k for k, t in enumerate(triangles)})
    return build_laplacians(build_boundaries(simplices))


def random_complex(rng, max_vertices=30, max_triangles=40):
    n = int(rng.integers(2, max_vertices))
    p = rng.uniform(0.05, 0.5)
    edges = [e for e in itertools.combinations(range(n), 2) if rng.random() < p]
    edge_set = set(edges)
    triangles = [t for t in itertools.combinations(range(n), 3)
                 if all(e in edge_set for e in itertools.combinations(t, 2))]
    return n, edges, triangles[:max_triangles]


class TestChebyshevNormalize(unittest.TestCase):
    def test_single_vertex_laplacian(self):
        matrix = sp.coo_matrix([[2.0]], dtype=np.float32)
        for half_interval in (False, True):
            with self.subTest(half_interval=half_interval):
                scaled = normalize(matrix, half_interval=half_interval)
                np.testing.assert_allclose(scaled.toarray(), [[1.0]])

    def test_pipeline_singleton_laplacians(self):
        # A single edge gives L1 = [[2]]; a filled triangle gives L2 = [[3]].
        single_edge = complex_laplacians(2, [(0, 1)])[1]
        filled_triangle = complex_laplacians(3, [(0, 1), (1, 2), (0, 2)], [(0, 1, 2)])[2]
        for matrix in (single_edge, filled_triangle):
            self.assertTrue(sp.isspmatrix_coo(matrix))
            self.assertEqual((matrix.dtype, matrix.shape), (np.float32, (1, 1)))
            for half_interval in (False, True):
                with self.subTest(value=matrix.data[0], half_interval=half_interval):
                    scaled = normalize(matrix, half_interval=half_interval)
                    self.assertTrue(sp.isspmatrix_coo(scaled))
                    np.testing.assert_allclose(scaled.toarray(), [[1.0]])

    def test_sparse_format_and_dtype_are_kept(self):
        laplacian = np.array([[1.0, -1.0], [-1.0, 1.0]])
        cases = [
            (sp.coo_matrix(laplacian, dtype=np.float32), "coo", np.float32),
            (sp.csr_matrix(laplacian), "csr", np.float64),
            (sp.csc_matrix(laplacian), "csc", np.float64),
            (sp.coo_matrix(laplacian.astype(int)), "coo", np.float64),
        ]
        for matrix, fmt, dtype in cases:
            for half_interval in (False, True):
                with self.subTest(fmt=fmt, dtype=matrix.dtype, half_interval=half_interval):
                    scaled = normalize(matrix, half_interval=half_interval)
                    self.assertEqual(scaled.format, fmt)
                    self.assertEqual(scaled.dtype, dtype)

    def test_empty_laplacian(self):
        for half_interval in (False, True):
            with self.subTest(half_interval=half_interval):
                scaled = normalize(sp.coo_matrix((0, 0), dtype=np.float32),
                                   half_interval=half_interval)
                self.assertEqual(scaled.shape, (0, 0))
                self.assertTrue(sp.isspmatrix_coo(scaled))

    def test_zero_laplacian_maps_to_chebyshev_limit(self):
        # Every eigenvalue is 0, which maps to -1 (full) or 0 (half interval).
        for size in (1, 2, 3, 5, 200):
            explicit_zeros = sp.coo_matrix(
                (np.zeros(size, np.float32), (np.arange(size), np.arange(size))),
                shape=(size, size))
            for matrix in (sp.coo_matrix((size, size), dtype=np.float32), explicit_zeros):
                with self.subTest(size=size, stored=matrix.nnz):
                    full = normalize(matrix)
                    half = normalize(matrix, half_interval=True)
                    self.assertTrue(sp.isspmatrix_coo(full) and sp.isspmatrix_coo(half))
                    np.testing.assert_array_equal(full.toarray(), -np.eye(size))
                    np.testing.assert_array_equal(half.toarray(), np.zeros((size, size)))

    def test_zero_laplacian_matches_isolated_vertices(self):
        # Isolated vertices are the zero block of a non-zero Laplacian, so an
        # all-zero Laplacian must be scaled the same way.
        with_edge = normalize(complex_laplacians(4, [(0, 1)])[0]).toarray()
        edgeless = normalize(sp.coo_matrix((2, 2), dtype=np.float32)).toarray()
        np.testing.assert_array_equal(with_edge[2:, 2:], edgeless)

    def test_duplicate_coo_entries_are_summed(self):
        # [[1, -1], [-1, 1]] split across duplicate entries, plus a cancelling pair.
        rows = [0, 0, 0, 1, 1, 1, 1]
        cols = [0, 0, 1, 0, 1, 1, 1]
        vals = [0.5, 0.5, -1.0, -1.0, 2.0, -1.0, 0.0]
        matrix = sp.coo_matrix((np.array(vals, np.float32), (rows, cols)), shape=(2, 2))
        np.testing.assert_allclose(normalize(matrix).toarray(), [[0.0, -1.0], [-1.0, 0.0]])
        cancelling = sp.coo_matrix((np.array([1.0, -1.0], np.float32), ([0, 0], [0, 0])),
                                   shape=(2, 2))
        np.testing.assert_array_equal(normalize(cancelling).toarray(), -np.eye(2))

    def test_integer_laplacian_is_rescaled_without_mutation(self):
        matrix = sp.csr_matrix([[1, -1], [-1, 1]], dtype=int)
        full = normalize(matrix)
        half = normalize(matrix, half_interval=True)

        np.testing.assert_allclose(full.toarray(), [[0.0, -1.0], [-1.0, 0.0]])
        np.testing.assert_allclose(half.toarray(), [[0.5, -0.5], [-0.5, 0.5]])
        np.testing.assert_array_equal(matrix.toarray(), [[1, -1], [-1, 1]])

    def test_input_is_not_mutated(self):
        rows = np.array([1, 0, 0, 1, 0])
        cols = np.array([1, 1, 0, 0, 0])
        vals = np.array([1.0, -1.0, 0.5, -1.0, 0.5], np.float32)
        for matrix in (sp.coo_matrix((vals, (rows, cols)), shape=(2, 2)),
                       sp.csr_matrix([[1.0, -1.0], [-1.0, 1.0]])):
            names = [n for n in ("row", "col", "indices", "indptr", "data") if hasattr(matrix, n)]
            before = {n: getattr(matrix, n).copy() for n in names}
            for half_interval in (False, True):
                scaled = normalize(matrix, half_interval=half_interval)
                for n in names:
                    np.testing.assert_array_equal(getattr(matrix, n), before[n])
                self.assertFalse(np.shares_memory(scaled.data, matrix.data))

    def test_rejects_non_finite_entries(self):
        for bad in (np.nan, np.inf, -np.inf):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    normalize(sp.coo_matrix(np.array([[1.0, bad], [bad, 1.0]])))

    def test_rejects_negative_and_non_psd_spectra(self):
        for matrix in ([[-2.0]], -np.eye(3), np.diag([-1.0, 0.0]),
                       [[-1.0, 1.0], [1.0, -1.0]], np.diag([-3.0, 1.0, 0.5])):
            with self.subTest(matrix=matrix):
                with self.assertRaises(ValueError):
                    normalize(sp.csr_matrix(np.asarray(matrix, dtype=float)))

    def test_symmetry(self):
        with self.assertRaises(ValueError):
            normalize(sp.csr_matrix([[1.0, 5.0], [0.0, 1.0]]))
        with self.assertRaises(ValueError):
            normalize(sp.csr_matrix([[2.0, -1.0], [-1.000001, 2.0]]))
        # float32 rounding-level asymmetry is accepted.
        nudged = np.array([[2.0, -1.0], [np.nextafter(np.float32(-1.0), 0), 2.0]], np.float32)
        np.testing.assert_allclose(normalize(sp.coo_matrix(nudged)).toarray(),
                                   [[1.0 / 3.0, -2.0 / 3.0], [-2.0 / 3.0, 1.0 / 3.0]], rtol=1e-6)

    def test_scale_invariance(self):
        laplacian = np.array([[1.0, -1.0], [-1.0, 1.0]])
        for scale in (1e-300, 1e-17, 1.0, 1e17, 1e300):
            with self.subTest(scale=scale):
                np.testing.assert_allclose(normalize(sp.csr_matrix(laplacian * scale)).toarray(),
                                           [[0.0, -1.0], [-1.0, 0.0]], atol=1e-12)

    def test_random_complexes_have_chebyshev_spectrum(self):
        rng = np.random.default_rng(3)
        checked = 0
        for _ in range(150):
            n, edges, triangles = random_complex(rng)
            if not edges:
                continue
            for laplacian in complex_laplacians(n, edges, triangles):
                # Pipeline Laplacians are float32, so allow float32 rounding.
                full = np.linalg.eigvalsh(normalize(laplacian).toarray().astype(np.float64))
                half = np.linalg.eigvalsh(
                    normalize(laplacian, half_interval=True).toarray().astype(np.float64))
                self.assertGreaterEqual(full.min(), -1 - 1e-6)
                self.assertGreaterEqual(half.min(), -1e-6)
                self.assertAlmostEqual(full.max(), 1.0, delta=1e-6)
                self.assertAlmostEqual(half.max(), 1.0, delta=1e-6)
                checked += 1
        self.assertGreater(checked, 250)

    def test_sparse_eigensolver_matches_dense(self):
        rng = np.random.default_rng(4)
        n = 300
        edges = [e for e in itertools.combinations(range(n), 2) if rng.random() < 0.02]
        laplacian = complex_laplacians(n, edges)[0].astype(np.float64)
        self.assertGreater(laplacian.shape[0], chebyshev.DENSE_EIGEN_MAX_SIZE)
        sparse_result = normalize(laplacian).toarray()
        with mock.patch.object(chebyshev, "DENSE_EIGEN_MAX_SIZE", 10 ** 6):
            dense_result = normalize(laplacian).toarray()
        np.testing.assert_allclose(sparse_result, dense_result, atol=1e-12)

    def test_matches_original_formula_on_nondegenerate_input(self):
        rng = np.random.default_rng(5)
        for _ in range(50):
            boundary = sp.random(8, 12, density=0.3, random_state=int(rng.integers(10 ** 9)))
            laplacian = (boundary @ boundary.T).tocoo()
            topeig = spl.eigsh(laplacian, k=1, which="LM", return_eigenvectors=False)[0]
            expected = 2.0 * laplacian.toarray() / topeig - np.eye(8)
            np.testing.assert_allclose(normalize(laplacian).toarray(), expected, atol=1e-10)

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_coo2tensor_accepts_pipeline_output(self):
        import torch
        from scnn.scnn import coo2tensor

        laplacians = complex_laplacians(3, [(0, 1), (1, 2), (0, 2)], [(0, 1, 2)])
        laplacians.append(sp.coo_matrix((2, 2), dtype=np.float32))
        for laplacian in laplacians:
            for half_interval in (False, True):
                scaled = normalize(laplacian, half_interval=half_interval)
                tensor = coo2tensor(scaled)
                self.assertEqual(tuple(tensor.shape), laplacian.shape)
                np.testing.assert_allclose(tensor.to_dense().numpy(), scaled.toarray(),
                                           rtol=1e-6)
                self.assertEqual(tensor.dtype, torch.float32)

    @unittest.expectedFailure
    def test_known_limitation_indefinite_input(self):
        # Documented precondition, not validated: lambda_max >= max |L_ij| > 0
        # but lambda_min < 0, so the scaled spectrum leaves [-1, 1].
        scaled = normalize(sp.csr_matrix(np.diag([-0.5, 1.0, 0.5])))
        self.assertGreaterEqual(np.linalg.eigvalsh(scaled.toarray()).min(), -1 - 1e-12)


if __name__ == "__main__":
    unittest.main()
