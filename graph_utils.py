import networkx as nx
from aigverse import to_edge_list


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


def get_graph(aig1, aig2, directed=False, weighted=False, weights=(-1,1)):
    # Convert AIG to edge list with weight information
    edges1 = to_edge_list(aig1, inverted_weight=weights[0], regular_weight=weights[1])
    edges2 = to_edge_list(aig2, inverted_weight=weights[0], regular_weight=weights[1])


    # If unweighted, strip the weights, if also undirected as no inversion
    if not weighted and not directed:
        edges1 = [(e.source, e.target) for e in edges1]
        edges2 = [(e.source, e.target) for e in edges2]
    else:
        edges1 = [(e.source, e.target, e.weight) for e in edges1]
        edges2 = [(e.source, e.target, e.weight) for e in edges2]

    if directed:
        G1 = nx.DiGraph()
        G2 = nx.DiGraph()
    else:
        G1 = nx.Graph()
        G2 = nx.Graph()

    # Create graphs based on the directed and weighted options
    if weighted:
        G1.add_weighted_edges_from(edges1)
        G2.add_weighted_edges_from(edges2)
    elif directed and not weighted: #invert negative edges to keep inversion direction
        G1.add_edges_from(transform_edge_list(edges1))
        G2.add_edges_from(transform_edge_list(edges2))
    else:  # Undirected and unweighted
        G1.add_edges_from(edges1)
        G2.add_edges_from(edges2)

    # Check if either graph is empty (handled separately)
    if G1.number_of_nodes() == 0 or G2.number_of_nodes() == 0:
        raise ValueError("Resistance distance is undefined for empty graphs.")

    return G1, G2
