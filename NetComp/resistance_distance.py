from .matrices import _pad
import numpy as np
import networkx as nx
from scipy.linalg import pinv, block_diag as dense_block_diag
from scipy.sparse import issparse


def resistance_matrix(G):
    """Return the resistance matrix of G.

    Parameters
    ----------
    G : networkx.Graph

    Returns
    -------
    R : NumPy array
        Matrix of pairwise resistances between nodes.

    Notes
    -----
    For disconnected graphs, the resistance between nodes in different components
    is set to infinity.
    """
    n = G.number_of_nodes()
    L = nx.laplacian_matrix(G).todense()
    M = pinv(L)
    d = np.diag(M).reshape(n, 1)
    ones = np.ones((n, 1))
    R = np.dot(d, ones.T) + np.dot(ones, d.T) - M - M.T

    # Identify connected components
    components = list(nx.connected_components(G))
    node_to_component = {}
    for idx, comp in enumerate(components):
        for node in comp:
            node_to_component[node] = idx
    nodes = list(G.nodes())
    for i in range(n):
        for j in range(n):
            if node_to_component[nodes[i]] != node_to_component[nodes[j]]:
                R[i, j] = np.inf
    return R


def renormalized_res_mat(A, beta=1):
    """Return the renormalized resistance matrix of the graph associated with A."""
    if issparse(A):
        G = nx.from_scipy_sparse_array(A)
    else:
        G = nx.from_numpy_array(A)

    n = G.number_of_nodes()
    R = np.full((n, n), np.inf)

    components = list(nx.connected_components(G))
    node_to_index = {node: idx for idx, node in enumerate(G.nodes())}
    for component in components:
        subgraph = G.subgraph(component)
        nodes_in_comp = list(subgraph.nodes())
        indices = [node_to_index[node] for node in nodes_in_comp]
        r_sub = resistance_matrix(subgraph)
        for i, u in enumerate(indices):
            for j, v in enumerate(indices):
                R[u, v] = r_sub[i, j]

    # Renormalize resistances
    with np.errstate(divide='ignore', invalid='ignore'):
        R = R / (R + beta)
    R = np.nan_to_num(R, nan=1.0, posinf=1.0)

    # Set the diagonal to 0
    np.fill_diagonal(R, 0)

    return R


def resistance_distance(A1, A2, p=2, beta=1):
    """Compare two graphs using resistance distance (possibly renormalized)."""
    n1, n2 = A1.shape[0], A2.shape[0]
    N = max(n1, n2)
    A1, A2 = [_pad(A, N) for A in [A1, A2]]
    R1, R2 = [renormalized_res_mat(A, beta=beta) for A in [A1, A2]]
    # Compute the element-wise differences, handling infinite values
    diff = np.abs(R1 - R2) ** p
    distance = np.nansum(diff)
    return distance ** (1 / p)
