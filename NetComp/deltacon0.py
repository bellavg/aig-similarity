import numpy as np

from .fast_bp import fast_bp
from .matrices import _pad


def deltacon0(A1, A2, eps=None):
    """DeltaCon0 distance between two graphs. The distance is the Frobenius norm
    of the element-wise square root of the fast belief propogation matrix.

    Parameters
    ----------
    A1, A2 : NumPy Matrices
        Adjacency matrices of graphs to be compared.

    Returns
    -------
    dist : float
        DeltaCon0 distance between graphs.

    References
    ----------

    See Also
    --------
    fast_bp
    """
    # pad smaller adj. mat. so they're the same size
    n1, n2 = [A.shape[0] for A in [A1, A2]]
    N = max(n1, n2)
    A1, A2 = [_pad(A, N) for A in [A1, A2]]
    S1, S2 = [fast_bp(A, eps=eps) for A in [A1, A2]]
    dist = np.abs(np.sqrt(S1) - np.sqrt(S2)).sum()
    return dist
