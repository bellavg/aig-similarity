from scipy import sparse as sps
import numpy as np
from numpy import linalg as la
from scipy.sparse import issparse


# TODO: check deltacon implementation

def fast_bp(A, eps=None):
    """Return the fast belief propogation matrix of graph associated with A.

    Parameters
    ----------
    A : NumPy matrix or Scipy sparse matrix
        Adjacency matrix of a graph. If sparse, can be any format; CSC or CSR
        recommended.

    eps : float, optional (default=None)
        Small parameter used in calculation of matrix. If not provided, it is
        set to 1/(1+d_max) where d_max is the maximum degree.

    Returns
    -------
    S : NumPy matrix or Scipy sparse matrix
        The fast belief propogation matrix. If input is sparse, will be returned
        as (sparse) CSC matrix.

    Notes
    -----

    References
    ----------

    """
    n, m = A.shape
    # Create a binary adjacency matrix where edge presence is indicated by 1
    A_binary = (A != 0).astype(int)

    # Compute out-degrees (sum over rows) based on binary matrix
    degs = np.array(A_binary.sum(axis=1)).flatten()

    # Default epsilon based on max out-degree
    if eps is None:
        eps = 1 / (1 + max(degs))

    I = sps.identity(n)  # Identity matrix
    D = sps.dia_matrix((degs, [0]), shape=(n, n))  # Diagonal matrix of out-degrees

    # Form the inverse of the fast belief propagation matrix
    Sinv = I + eps ** 2 * D - eps * A  # A is directed adjacency matrix

    try:
        S = la.inv(Sinv)  # Invert the matrix
    except:
        Sinv = sps.csc_matrix(Sinv)
        S = sps.linalg.inv(Sinv)

    return S


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
    # Calculate fast belief propagation matrices for both graphs
    S1, S2 = [fast_bp(A, eps=eps) for A in [A1, A2]]

    # Compute the DeltaCon0 distance using the element-wise difference in square roots
    dist = np.abs(np.sqrt(np.abs(S1)) - np.sqrt(np.abs(S2))).sum() # added abs bc negative weights

    return dist