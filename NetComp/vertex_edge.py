def vertex_edge_overlap(G1, G2):
    """
    Vertex-edge overlap. The VEO similarity is defined as:

        VEO(G1, G2) = (|V1 ∩ V2| + |E1 ∩ E2|) / (|V1 ∪ V2| + |E1 ∪ E2|)

    where |S| is the size of a set, ∩ represents the intersection of sets,
    and ∪ represents the union of sets.

    Parameters
    ----------
    G1, G2 : networkx.Graph
        The graphs to compare.

    Returns
    -------
    sim : float
        The similarity between the two graphs.
    """
    V1, V2 = [set(G.nodes()) for G in [G1, G2]]
    E1, E2 = [set(G.edges()) for G in [G1, G2]]
    V_overlap = len(V1 & V2)  # Corrected to intersection
    E_overlap = len(E1 & E2)
    V_union = len(V1 | V2)
    E_union = len(E1 | E2)
    sim = (V_overlap + E_overlap) / (V_union + E_union)
    return sim


def vertex_edge_distance(G1, G2):
    """
    Vertex-edge overlap transformed into a distance via:

        D = (1 - VEO) / VEO

    Parameters
    ----------
    G1, G2 : networkx.Graph
        The graphs to compare.

    Returns
    -------
    dist : float
        The distance between the two graphs.
    """
    sim = vertex_edge_overlap(G1, G2)
    if sim == 0:
        return float('inf')  # Handle case where similarity is 0 to avoid division by zero
    dist = (1 - sim) / sim
    return dist
