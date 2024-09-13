from utils import get_graph


def vertex_edge_overlap(G, G_prime):
    """
    Compute the Vertex Edge Overlap (VEO) between two graphs G and G_prime.

    Parameters:
    G, G_prime : networkx.Graph
        The two input graphs for which VEO will be computed.

    Returns:
    float
        The VEO similarity score between 0 and 1.
    """
    # Vertices and edges of both graphs
    V_G = set(G.nodes())
    V_G_prime = set(G_prime.nodes())
    E_G = set(G.edges())
    E_G_prime = set(G_prime.edges())

    # Common vertices and edges
    common_vertices = V_G.intersection(V_G_prime)
    common_edges = E_G.intersection(E_G_prime)

    # Number of vertices and edges in both graphs
    num_V_G = len(V_G)
    num_V_G_prime = len(V_G_prime)
    num_E_G = len(E_G)
    num_E_G_prime = len(E_G_prime)

    # Number of common vertices and edges
    num_common_vertices = len(common_vertices)
    num_common_edges = len(common_edges)

    # VEO formula
    numerator = num_common_vertices + num_common_edges
    denominator = num_V_G + num_V_G_prime + num_E_G + num_E_G_prime

    if denominator == 0:
        return 1.0  # If both graphs have no vertices and edges, consider them identical

    VEO_value = 2 * (numerator / denominator)

    return VEO_value


def get_veo(aig1, aig2):
    G1, G2 = get_graph(aig1, aig2, directed=False)
    return vertex_edge_overlap(G1, G2)
