import networkx as nx
from aigverse import to_edge_list


def get_graph(aig1, aig2, directed=False):
    edges1 = to_edge_list(aig1, inverted_weight=-1, regular_weight=1)
    edges2 = to_edge_list(aig2, inverted_weight=-1, regular_weight=1)

    # Convert to list of tuples (source, target, weight)
    edges1 = [(e.source, e.target, e.weight) for e in edges1]
    edges2 = [(e.source, e.target, e.weight) for e in edges2]

    # Apply transformation directly on the edge lists
    transformed_edges1 = transform_edge_list(edges1)
    transformed_edges2 = transform_edge_list(edges2)
    if directed:
        G1 = nx.DiGraph()
        G2 = nx.DiGraph()
    else:
        G1 = nx.Graph()
        G2 = nx.Graph()
    G1.add_edges_from(transformed_edges1)
    G2.add_edges_from(transformed_edges2)

    # Check if either graph is empty (handled separately)
    if G1.number_of_nodes() == 0 or G2.number_of_nodes() == 0:
        raise ValueError("Resistance distance is undefined for empty graphs.")

    return G1, G2




def transform_edge_list(edges):
    """
    For all edges in the edge list with weight -1, remove those edges
    and add a new edge in the reverse direction with weight 1.

    Parameters:
    -----------
    edges : list of tuples (u, v, weight)
        A list of directed edges with signed weights.

    Returns:
    --------
    transformed_edges : list of tuples (u, v, weight)
        The transformed edge list.
    """
    transformed_edges = []

    # Traverse all edges in the list
    for u, v, weight in edges:
        if weight == -1:
            # Add reversed edge with weight 1
            transformed_edges.append((v, u))
        else:
            # Keep the original edge
            transformed_edges.append((u, v))

    return transformed_edges


