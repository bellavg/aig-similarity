def vertex_edge_overlap(G1, G2):
    """Vertex-edge overlap. Basically a souped-up edit distance, but in similarity
    form. The VEO similarity is defined as

        VEO(G1,G2) = (|V1&V2| + |E1&E2|) / (|V1|+|V2|+|E1|+|E2|)

    where |S| is the size of a set S and U&T is the union of U and T.

    Parameters
    ----------
    G1, G2 : Networkx Graphs

    Returns
    -------
    sim : float
        The similarity between the two graphs


    References
    ----------

    """
    V1, V2 = [set(G.nodes()) for G in [G1, G2]]
    E1, E2 = [set(G.edges()) for G in [G1, G2]]
    V_overlap = len(V1 | V2)  # set union
    E_overlap = len(E1 | E2)
    sim = (V_overlap + E_overlap) / (len(V1) + len(V2) + len(E1) + len(E2))
    return sim


def vertex_edge_distance(A1, A2):
    """Vertex-edge overlap transformed into a distance via

        D = (1-VEO)/VEO

    which is the inversion of the common distance-to-similarity function

        sim = 1/(1+D).

    Parameters
    ----------
    A1, A2 : NumPy matrices
        Adjacency matrices of graphs to be compared

    Returns
    -------
    dist : float
        The distance between the two graphs
    """
    sim = vertex_edge_overlap(A1, A2)
    dist = (1 - sim) / sim
    return dist
