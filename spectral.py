import networkx as nx
import numpy as np
import scipy.sparse as sps
from utils import get_graph


def spectral_distance(graph1, graph2, matrix_type='adjacency'):
    """
    Compute the spectral distance between two graphs based on their adjacency, Laplacian, or normalized Laplacian matrices.

    Parameters:
    graph1, graph2 : networkx.Graph
        Input graphs for which the spectral distance will be computed.
    matrix_type : str
        Type of matrix to use for the spectral distance calculation.
        Options: 'adjacency', 'laplacian', 'normalized_laplacian'

    Returns:
    float
        The spectral distance between the two graphs.
    """
    # Select the matrix type
    if matrix_type == 'adjacency':
        # Adjacency matrix
        matrix1 = nx.adjacency_matrix(graph1).todense()
        matrix2 = nx.adjacency_matrix(graph2).todense()
    elif matrix_type == 'laplacian':
        # Laplacian matrix
        matrix1 = nx.laplacian_matrix(graph1).todense()
        matrix2 = nx.laplacian_matrix(graph2).todense()
    elif matrix_type == 'normalized_laplacian':
        # Normalized Laplacian matrix
        matrix1 = nx.normalized_laplacian_matrix(graph1).todense()
        matrix2 = nx.normalized_laplacian_matrix(graph2).todense()
    else:
        raise ValueError("matrix_type must be 'adjacency', 'laplacian', or 'normalized_laplacian'")

    # Compute eigenvalues of the matrices
    eigenvalues1 = np.linalg.eigvals(matrix1)
    eigenvalues2 = np.linalg.eigvals(matrix2)

    # Sort eigenvalues based on the matrix type
    if matrix_type == 'adjacency':
        # Sort in descending order for adjacency matrix
        eigenvalues1_sorted = np.sort(eigenvalues1)[::-1]
        eigenvalues2_sorted = np.sort(eigenvalues2)[::-1]
    else:
        # Sort in ascending order for Laplacian or normalized Laplacian
        eigenvalues1_sorted = np.sort(eigenvalues1)
        eigenvalues2_sorted = np.sort(eigenvalues2)

    # Pad the shorter list of eigenvalues with zeros, if necessary
    len_diff = len(eigenvalues1_sorted) - len(eigenvalues2_sorted)
    if len_diff > 0:
        eigenvalues2_sorted = np.pad(eigenvalues2_sorted, (0, len_diff), 'constant')
    elif len_diff < 0:
        eigenvalues1_sorted = np.pad(eigenvalues1_sorted, (0, -len_diff), 'constant')

    # Compute the adjacency spectral distance using the provided formula
    spectral_distance_value = np.sqrt(np.sum((eigenvalues1_sorted - eigenvalues2_sorted) ** 2))

    return spectral_distance_value


def get_lap_spectral_dist(aig1, aig2):
    G1, G2 = get_graph(aig1, aig2, directed=False)
    return spectral_distance(G1, G2, matrix_type='laplacian')


def get_adj_spectral_dist(aig1, aig2):
    G1, G2 = get_graph(aig1, aig2, directed=False)
    return spectral_distance(G1, G2, matrix_type='adjacency')
