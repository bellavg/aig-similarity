"""
***********************
Fast Belief Propagation
***********************

The fast approximation of the Belief Propogation matrix.
"""

import numpy as np
from scipy import sparse as sps


def fast_bp(A, eps=None):
    """
    Return the fast belief propagation matrix of the graph associated with A.

    Parameters
    ----------
    A : Scipy sparse matrix (CSC or CSR recommended)
        Adjacency matrix of a graph.

    eps : float, optional (default=None)
        Small parameter used in calculation of the fast belief propagation matrix.
        If not provided, it is set to 1/(1 + d_max), where d_max is the maximum degree.

    Returns
    -------
    S : Scipy sparse matrix
        The fast belief propagation matrix in sparse format.
    """
    n = A.shape[0]

    # Compute the degree vector (sum of rows of A)
    degs = np.array(A.sum(axis=1)).flatten()

    # If eps is not provided, calculate it as 1 / (1 + d_max)
    if eps is None:
        eps = 1 / (1 + max(degs))

    # Create the identity matrix I (sparse)
    I = sps.identity(n)

    # Create the diagonal degree matrix D
    D = sps.diags(degs, format='dia')

    # Compute the inverse of the fast belief propagation matrix: Sinv = I + eps^2 * D - eps * A
    Sinv = I + eps ** 2 * D - eps * A

    # Invert the matrix to get the belief propagation matrix S
    S = sps.linalg.inv(Sinv.tocsc())

    return S