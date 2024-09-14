import networkx as nx
from NetComp.deltacon0 import deltacon0
from NetComp.netsimile import netsimile
from  graph_utils import get_graph
import numpy as np


def get_sparse_adjacency_matrix(graph, size):
    """w
    Returns the adjacency matrix of the graph as a sparse matrix (lil_matrix).
    Dynamically grows the matrix to match 'size' without explicit padding.
    Size should be the size of the biggest graph in the comparison you will make
    """
    # Get the adjacency matrix of the graph in sparse format (lil_matrix allows dynamic growth)
    adj_matrix = nx.adjacency_matrix(graph, nodelist=graph.nodes(), dtype=np.int32, weight='weight').tolil()

    # Dynamically grow the matrix if needed
    current_size = adj_matrix.shape[0]

    if current_size < size:
        adj_matrix.resize((size, size))  # This dynamically adjusts the size without explicit padding

    return adj_matrix.tocsr()


def get_deltacon0(aig1, aig2):
    G1, G2 = get_graph(aig1, aig2, directed=True, weights=(1,1))
    # Get the sizes of both graphs (number of nodes)
    size1 = G1.number_of_nodes()
    size2 = G2.number_of_nodes()

    max_size = max(size1, size2)
    A1 = get_sparse_adjacency_matrix(G1, max_size)
    A2 = get_sparse_adjacency_matrix(G2, max_size)

    return deltacon0(A1, A2, eps=1e-8)


def get_net_simile(aig1,aig2):
    G1, G2 = get_graph(aig1, aig2, directed=False)  # resistance distance is not for directed graphs

    return netsimile(G1, G2)


def get_ns_dir_inverted(aig1, aig2):
    G1, G2 = get_graph(aig1, aig2, directed=True)
    return netsimile(G1, G2)

def get_ns_dir_uninverted(aig1, aig2):
    G1, G2 = get_graph(aig1, aig2, directed=True, weights=(1,1))
    return netsimile(G1, G2)

