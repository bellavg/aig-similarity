import networkx as nx
import numpy as np

from utils import get_graph


def get_lap_spectral_distance(aig1, aig2, k=None):
    #TODO run unittests
    """
    Compute the spectral distance between two graphs based on their Laplacian spectra.

    Parameters
    ----------
    G1, G2 : networkx.Graph
        The graphs to compare.

    k : int, optional
        Number of eigenvalues to consider. If None, uses all eigenvalues.

    Returns
    -------
    dist : float
        The spectral distance between the graphs.
    """
    # Compute Laplacian matrices
    G1, G2 = get_graph(aig1, aig2, directed=False)  # resistance distance is not for directed graphs
    L1 = nx.laplacian_matrix(G1).todense()
    L2 = nx.laplacian_matrix(G2).todense()

    # Compute eigenvalues
    eigs1 = np.sort(np.linalg.eigvalsh(L1))
    eigs2 = np.sort(np.linalg.eigvalsh(L2))

    # Use the smallest size
    min_size = min(len(eigs1), len(eigs2))
    if k is not None:
        min_size = min(min_size, k)
    eigs1 = eigs1[:min_size]
    eigs2 = eigs2[:min_size]

    # Compute the distance
    dist = np.linalg.norm(eigs1 - eigs2)
    return dist
