import networkx as nx

from NetComp.deltacon0 import deltacon0
from NetComp.netsimile import netsimile
from NetComp.vertex_edge import vertex_edge_overlap
from utils import get_graph


def get_deltacon0(aig1, aig2):
    G1, G2 = get_graph(aig1, aig2, directed=False)  # resistance distance is not for directed graphs

    A1 = nx.adjacency_matrix(G1)
    A2 = nx.adjacency_matrix(G2)

    return deltacon0(A1, A2)


def get_net_simile(aig1,aig2):
    G1, G2 = get_graph(aig1, aig2, directed=False)  # resistance distance is not for directed graphs

    return netsimile(G1, G2)



