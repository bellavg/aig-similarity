import numpy as np
import scipy.sparse as sps

from .fast_bp import fast_bp
from .matrices import _pad_sparse


def deltacon0(A1, A2, eps=None):
    """
    DeltaCon0 distance between two graphs. The distance is the Frobenius norm
    of the element-wise square root of the fast belief propagation matrix.

    Parameters
    ----------
    A1, A2 : Scipy sparse matrices
        Adjacency matrices of graphs to be compared.

    eps : float, optional (default=None)
        Small parameter used in calculation of the fast belief propagation matrix.

    Returns
    -------
    dist : float
        DeltaCon0 distance between graphs.
    """
    # Pad smaller adjacency matrices so they're the same size
    n1, n2 = A1.shape[0], A2.shape[0]
    N = max(n1, n2)
    A1, A2 = [_pad_sparse(A, N) for A in [A1, A2]]

    # Compute fast belief propagation matrices S1 and S2
    S1, S2 = [fast_bp(A, eps=eps) for A in [A1, A2]]

    # Convert S1 and S2 to dense format if they are sparse
    if sps.issparse(S1):
        S1 = S1.toarray()
    if sps.issparse(S2):
        S2 = S2.toarray()

    # Compute the element-wise square root of the matrices
    S1_sqrt = np.sqrt(S1)
    S2_sqrt = np.sqrt(S2)

    # Compute the difference between the two matrices
    diff = S1_sqrt - S2_sqrt

    # If diff is a 2D matrix, use Frobenius norm; otherwise, use 2-norm for vectors
    if diff.ndim == 2:
        dist = np.linalg.norm(diff, ord='fro')  # Frobenius norm for matrices
    else:
        dist = np.linalg.norm(diff, ord=2)  # 2-norm for vectors (just in case)

    return dist