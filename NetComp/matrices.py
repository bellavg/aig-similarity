from scipy import sparse as sps
import numpy as np
from scipy.sparse import issparse


def _flat(D):
    """Flatten column or row matrices, as well as arrays."""
    if issparse(D):
        raise ValueError('Cannot flatten sparse matrix.')
    d_flat = np.array(D).flatten()
    return d_flat


def _pad(A, N):
    """
    Pad the sparse adjacency matrix A to size (N, N).

    Parameters:
    -----------
    A : scipy sparse matrix (csr_matrix or similar)
        The sparse adjacency matrix to be padded.
    N : int
        The target size of the padded matrix.

    Returns:
    --------
    A_pad : scipy sparse matrix
        The padded sparse adjacency matrix.
    """
    n, m = A.shape

    # If the matrix is already the correct size or larger, return it as is
    if n >= N and m >= N:
        return A

    # Padding dimensions
    pad_bottom = N - n if n < N else 0
    pad_right = N - m if m < N else 0

    # Pad bottom (add rows) if needed
    if pad_bottom > 0:
        A = sps.vstack([A, sps.csr_matrix((pad_bottom, m))])

    # Pad right (add columns) if needed
    if pad_right > 0:
        A = sps.hstack([A, sps.csr_matrix((N, pad_right))])

    return A


def laplacian_matrix(A, normalized=False):
    """Diagonal degree matrix of graph with adjacency matrix A

    Parameters
    ----------
    A : matrix
        Adjacency matrix
    normalized : Bool, optional (default=False)
        If true, then normalized laplacian is returned.

    Returns
    -------
    L : SciPy sparse matrix
        Combinatorial laplacian matrix.
    """
    n, m = A.shape
    D = degree_matrix(A)
    L = D - A
    if normalized:
        degs = _flat(A.sum(axis=1))
        rootD = sps.spdiags(np.power(degs, -1 / 2), [0], n, n, format='csr')
        L = rootD * L * rootD
    return L


def degree_matrix(A):
    """Diagonal degree matrix of graph with adjacency matrix A

    Parameters
    ----------
    A : matrix
        Adjacency matrix

    Returns
    -------
    D : SciPy sparse matrix
        Diagonal matrix of degrees.
    """
    n, m = A.shape
    degs = _flat(A.sum(axis=1))
    D = sps.spdiags(degs, [0], n, n, format='csr')
    return D
