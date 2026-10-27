"""Regression tests for degenerate and integer-valued graph Laplacians."""

import unittest

import numpy as np
import scipy.sparse as sp

from scnn.chebyshev import normalize


class TestChebyshevNormalize(unittest.TestCase):
    def test_single_vertex_laplacian(self):
        matrix = sp.csr_matrix([[2.0]])
        for half_interval in (False, True):
            with self.subTest(half_interval=half_interval):
                scaled = normalize(matrix, half_interval=half_interval)
                np.testing.assert_allclose(scaled.toarray(), [[1.0]])

    def test_zero_and_empty_laplacians(self):
        for size in (0, 1, 3):
            for half_interval in (False, True):
                with self.subTest(size=size, half_interval=half_interval):
                    matrix = sp.csr_matrix((size, size), dtype=float)
                    scaled = normalize(matrix, half_interval=half_interval)
                    self.assertEqual(scaled.shape, (size, size))
                    self.assertTrue(np.isfinite(scaled.data).all())
                    self.assertTrue(scaled.nnz == 0 or np.all(scaled.data == 0))

    def test_integer_laplacian_is_rescaled_without_mutation(self):
        matrix = sp.csr_matrix([[1, -1], [-1, 1]], dtype=int)
        full = normalize(matrix)
        half = normalize(matrix, half_interval=True)

        np.testing.assert_allclose(full.toarray(), [[0.0, -1.0], [-1.0, 0.0]])
        np.testing.assert_allclose(half.toarray(), [[0.5, -0.5], [-0.5, 0.5]])
        np.testing.assert_array_equal(matrix.toarray(), [[1, -1], [-1, 1]])


if __name__ == "__main__":
    unittest.main()
