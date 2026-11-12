import scipy.sparse as sp
import scipy.sparse.linalg as spl
import numpy as np

# Up to this size a dense eigensolver is as fast as ARPACK, exact and
# deterministic, and it also covers M == 1 (eigsh needs k < M).
DENSE_EIGEN_MAX_SIZE = 128


def normalize(L, half_interval=False):
    """Rescale a Laplacian for Chebyshev filtering.

    Maps the spectrum [0, lambda_max] to [-1, 1] (``2 L / lambda_max - I``) or,
    with ``half_interval``, to [0, 1] (``L / lambda_max``). A zero Laplacian
    has every eigenvalue at 0, so it maps to ``-I`` or ``0`` respectively.

    Precondition: ``L`` is real, symmetric and positive semi-definite, as
    Hodge Laplacians ``B B^T + B^T B`` are by construction. Non-finite
    entries, asymmetry and ``lambda_max < max |L_ij|`` (impossible for PSD
    matrices) raise ``ValueError``; other indefinite inputs are not detected.

    ``L`` is not modified. The result keeps the sparse format of ``L`` (COO
    stays COO for ``coo2tensor``) and its floating dtype (float64 otherwise).
    """
    assert (sp.isspmatrix(L))
    M = L.shape[0]
    assert (M == L.shape[1])
    dtype = L.dtype if np.issubdtype(L.dtype, np.floating) else np.float64
    # CSR conversion sums duplicate COO entries; work on a float64 copy.
    A = sp.csr_matrix(L, dtype=np.float64, copy=True)
    A.eliminate_zeros()
    if not np.isfinite(A.data).all():
        raise ValueError("Laplacian has non-finite entries")

    if A.nnz == 0:
        # Any positive scale gives the zero Laplacian's limit.
        topeig = 1.0
    else:
        scale = np.abs(A.data).max()
        # Allow asymmetry from rounding at the input's own precision.
        asym = abs(A - A.T)
        if asym.nnz and asym.max() > 100 * np.finfo(dtype).eps * scale:
            raise ValueError("Laplacian must be symmetric")
        if M <= DENSE_EIGEN_MAX_SIZE:
            topeig = np.linalg.eigvalsh(A.toarray())[-1]
        else:
            topeig = spl.eigsh(A, k=1, which="LA", return_eigenvectors=False)[0]
        # lambda_max >= max |L_ij| holds for every PSD matrix.
        if not (np.isfinite(topeig) and topeig >= (1 - 1e-6) * scale):
            raise ValueError("Laplacian must be positive semi-definite")

    A.data /= topeig
    if not half_interval:
        A = 2.0 * A - sp.identity(M, format="csr")

    return A.astype(dtype, copy=False).asformat(L.format)


def assemble(K, L, x):
    import torch

    (B, C_in, M) = x.shape
    assert (L.shape[0] == M)
    assert (L.shape[0] == L.shape[1])
    assert (K > 0)

    X = []
    for b in range(0, B):
        X123 = []
        for c_in in range(0, C_in):
            X23 = []
            X23.append(x[b, c_in, :].unsqueeze(1))  # Constant, k = 0 term.

            if K > 1:
                X23.append(L.mm(X23[0]))
            for k in range(2, K):
                X23.append(2 * (L.mm(X23[k - 1])) - X23[k - 2])

            X23 = torch.cat(X23, 1)
            assert (X23.shape == (M, K))
            X123.append(X23.unsqueeze(0))

        X123 = torch.cat(X123, 0)
        assert (X123.shape == (C_in, M, K))
        X.append(X123.unsqueeze(0))

    X = torch.cat(X, 0)
    assert (X.shape == (B, C_in, M, K))

    return X
