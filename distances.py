import networkx as nx

from NetComp.deltacon0 import deltacon0
from NetComp.netsimile import netsimile
from NetComp.resistance_distance import resistance_distance
from NetComp.vertex_edge import vertex_edge_overlap
from utils import get_graph


def get_ged(G1, G2):
    return nx.graph_edit_distance(G1, G2, timeout=30)


def get_res_dist(aig1, aig2):
    # TODO: change for no known node correspondence
    G1, G2 = get_graph(aig1, aig2, directed=False)  # resistance distance is not for directed graphs

    # Check if both graphs are disconnected
    if not nx.is_connected(G1) or not nx.is_connected(G2):
        print("Warning: One or both graphs are disconnected. Resistance distance may not behave as expected.")

    A1 = nx.adjacency_matrix(G1)
    A2 = nx.adjacency_matrix(G2)

    return resistance_distance(A1, A2)


def get_deltacon0(aig1, aig2):
    G1, G2 = get_graph(aig1, aig2, directed=False)  # resistance distance is not for directed graphs

    A1 = nx.adjacency_matrix(G1)
    A2 = nx.adjacency_matrix(G2)

    return deltacon0(A1, A2)


def get_net_simile(aig1,aig2):
    #TODO do unittests
    G1, G2 = get_graph(aig1, aig2, directed=False)  # resistance distance is not for directed graphs

    A1 = nx.adjacency_matrix(G1)
    A2 = nx.adjacency_matrix(G2)

    return netsimile(A1, A2)


def get_veo(aig1, aig2):
    # TODO do unittests
    G1, G2 = get_graph(aig1, aig2, directed=False)  # resistance distance is not for directed graphs

    return vertex_edge_overlap(G1,G2)


def get_ved(aig1, aig2):
    # TODO do unittests
    G1, G2 = get_graph(aig1, aig2, directed=False)  # resistance distance is not for directed graphs

    return vertex_edge_overlap(G1,G2)

def get_lambda_dist(aig1, aig2):
