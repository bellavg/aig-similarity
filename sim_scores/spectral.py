from graph_utils import get_graph
import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh

def spectral_distance(graph1, graph2, matrix_type='adjacency', k=100):
    """
    Compute the spectral distance between two graphs based on their adjacency, Laplacian,
    or normalized Laplacian matrices using the top k or n-1 eigenvalues of the smallest graph.

    Parameters:
    graph1, graph2 : networkx.Graph
        Input graphs for which the spectral distance will be computed.
    matrix_type : str
        Type of matrix to use for the spectral distance calculation.
        Options: 'adjacency', 'laplacian', 'normalized_laplacian'.
    k : int
        Number of top eigenvalues to compute. Defaults to 5, but will use min(k, n-1) of the smallest graph.

    Returns:
    float
        The spectral distance between the two graphs.
    """
    # Select the matrix type
    if matrix_type == 'adjacency':
        matrix1 = nx.adjacency_matrix(graph1).astype(np.float64)
        matrix2 = nx.adjacency_matrix(graph2).astype(np.float64)
    elif matrix_type == 'laplacian':
        matrix1 = nx.laplacian_matrix(graph1).astype(np.float64)
        matrix2 = nx.laplacian_matrix(graph2).astype(np.float64)
    elif matrix_type == 'normalized_laplacian':
        matrix1 = nx.normalized_laplacian_matrix(graph1).astype(np.float64)
        matrix2 = nx.normalized_laplacian_matrix(graph2).astype(np.float64)
    else:
        raise ValueError("matrix_type must be 'adjacency', 'laplacian', or 'normalized_laplacian'")

    # Get the sizes of the graphs
    n1 = matrix1.shape[0]
    n2 = matrix2.shape[0]

    # Use min(k, n-1) of the smallest graph
    k = min(k, min(n1, n2) - 1)

    def compute_eigenvalues(matrix, k):
        n = matrix.shape[0]
        if k >= n:
            # If k >= n, use dense computation
            return eigh(matrix.toarray(), eigvals_only=True)
        else:
            # Use eigsh for sparse matrices
            return eigsh(matrix, k=k, which='LA', return_eigenvectors=False)

    # Compute eigenvalues
    eigenvalues1 = compute_eigenvalues(matrix1, k)
    eigenvalues2 = compute_eigenvalues(matrix2, k)

    # Sort the eigenvalues
    eigenvalues1_sorted = np.sort(eigenvalues1)
    eigenvalues2_sorted = np.sort(eigenvalues2)

    # Pad the shorter list of eigenvalues with zeros
    len_diff = len(eigenvalues1_sorted) - len(eigenvalues2_sorted)
    if len_diff > 0:
        eigenvalues2_sorted = np.pad(eigenvalues2_sorted, (0, len_diff), 'constant')
    elif len_diff < 0:
        eigenvalues1_sorted = np.pad(eigenvalues1_sorted, (0, -len_diff), 'constant')

    # Compute the spectral distance using the padded eigenvalues
    spectral_distance_value = np.sqrt(np.sum((eigenvalues1_sorted - eigenvalues2_sorted) ** 2))

    return spectral_distance_value


def get_lap_spectral_dist(aig1, aig2):
    G1, G2 = get_graph(aig1, aig2, directed=False)
    return spectral_distance(G1, G2, matrix_type='laplacian')


def get_adj_spectral_dist(aig1, aig2):
    G1, G2 = get_graph(aig1, aig2, directed=False)
    return spectral_distance(G1, G2, matrix_type='adjacency')


def get_dir_adj_sd_inverted(aig1, aig2):
    G1, G2 = get_graph(aig1, aig2, directed=True)
    return spectral_distance(G1, G2, matrix_type='adjacency')

def get_dir_adj_sd_uninverted(aig1, aig2):
    G1, G2 = get_graph(aig1, aig2, directed=True, weights = (1,1))
    return spectral_distance(G1, G2, matrix_type='adjacency')

def get_dir_adj_sd_uninverted_weighted(aig1, aig2):
    G1, G2 = get_graph(aig1, aig2, directed=True, weighted=True)
    return spectral_distance(G1, G2, matrix_type='adjacency')


