import networkx as nx
import numpy as np
from utils import get_graph
from NetComp.resistance_distance import resistance_distance


def get_ged(G1, G2):
    return nx.graph_edit_distance(G1, G2, timeout=10)


def get_res_dist(aig1, aig2):
    G1, G2 = get_graph(aig1, aig2, directed=False)  # resistance distance is not for directed graphs

    # Check if either graph is empty (handled separately)
    if G1.number_of_nodes() == 0 or G2.number_of_nodes() == 0:
        raise ValueError("Resistance distance is undefined for empty graphs.")

    # Check if both graphs are disconnected
    if not nx.is_connected(G1) or not nx.is_connected(G2):
        print("Warning: One or both graphs are disconnected. Resistance distance may not behave as expected.")

    A1 = nx.adjacency_matrix(G1)
    A2 = nx.adjacency_matrix(G2)

    return resistance_distance(A1, A2)
