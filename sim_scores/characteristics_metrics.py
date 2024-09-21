from aigverse import Aig, to_edge_list

from sim_scores.cosine_similarity_metric import cosine_similarity_metric
from sim_scores.euclidean_similarity_metric import normalized_euclidean_distance_metric


def absolute_gate_count_metric(aig1: Aig, aig2: Aig) -> int:
    """
    Compute the absolute gate count metric for two AIGs. The absolute gate count metric is defined as the absolute
    difference in the number of gates between the two AIGs.

    Parameters:
    aig1, aig2 (Aig): The input AIGs to compare.

    Returns:
    int: The absolute gate count metric between the two AIGs.
    """
    return abs(aig1.num_gates() - aig2.num_gates())


def relative_gate_count_metric(aig1: Aig, aig2: Aig) -> float:
    """
    Compute the relative gate count metric for two AIGs. The relative gate count metric is defined as the relative
    difference in the number of gates between the two AIGs.

    Parameters:
    aig1, aig2 (Aig): The input AIGs to compare.

    Returns:
    float: The relative gate count metric between the two AIGs.
    """
    total_gates = aig1.num_gates() + aig2.num_gates()

    if total_gates == 0:
        return 0.0

    return abs(aig1.num_gates() - aig2.num_gates()) / total_gates


def absolute_edge_count_metric(aig1: Aig, aig2: Aig) -> int:
    """
    Compute the absolute edge count metric for two AIGs. The absolute edge count metric is defined as the absolute
    difference in the number of edges between the two AIGs.

    Parameters:
    aig1, aig2 (Aig): The input AIGs to compare.

    Returns:
    int: The absolute edge count metric between the two AIGs.
    """
    return abs(len(to_edge_list(aig1)) - len(to_edge_list(aig2)))


def relative_edge_count_metric(aig1: Aig, aig2: Aig) -> float:
    """
    Compute the relative edge count metric for two AIGs. The relative edge count metric is defined as the relative
    difference in the number of edges between the two AIGs.

    Parameters:
    aig1, aig2 (Aig): The input AIGs to compare.

    Returns:
    float: The relative edge count metric between the two AIGs.
    """
    edge_list1 = to_edge_list(aig1)
    edge_list2 = to_edge_list(aig2)

    total_edges = len(edge_list1) + len(edge_list2)

    if total_edges == 0:
        return 0.0

    return abs(len(edge_list1) - len(edge_list2)) / total_edges


def absolute_level_count_metric(aig1: Aig, aig2: Aig) -> int:
    """
    Compute the absolute level count metric for two AIGs. The absolute level count metric is defined as the absolute
    difference in the number of levels between the two AIGs.

    Parameters:
    aig1, aig2 (Aig): The input AIGs to compare.

    Returns:
    int: The absolute level count metric between the two AIGs.
    """
    return abs(aig1.num_levels() - aig2.num_levels())


def relative_level_count_metric(aig1: Aig, aig2: Aig) -> float:
    """
    Compute the relative level count metric for two AIGs. The relative level count metric is defined as the relative
    difference in the number of levels between the two AIGs.

    Parameters:
    aig1, aig2 (Aig): The input AIGs to compare.

    Returns:
    float: The relative level count metric between the two AIGs.
    """
    total_levels = aig1.num_levels() + aig2.num_levels()

    if total_levels == 0:
        return 0.0

    return abs(aig1.num_levels() - aig2.num_levels()) / total_levels


def gate_level_normalized_euclidean_similarity_metric(aig1: Aig, aig2: Aig) -> float:
    """
    Compute the normalized Euclidean similarity metric for two AIGs. The normalized Euclidean similarity metric is
    defined as the Euclidean distance between the normalized sim_scores of the two AIGs, where the sim_scores are the
    number of gates, number of edges, and number of levels. The similarity score is then calculated as 1 minus the
    normalized distance divided by the maximum possible distance in the normalized space.

    Parameters:
    aig1, aig2 (Aig): The input AIGs to compare.

    Returns:
    float: The normalized Euclidean similarity metric between the two AIGs.
    """
    # Extract sim_scores for AIG1 and AIG2
    m1 = [aig1.num_gates(), aig1.num_levels()]
    m2 = [aig2.num_gates(), aig2.num_levels()]

    return normalized_euclidean_distance_metric(m1, m2)


def gate_level_cosine_similarity_metric(aig1: Aig, aig2: Aig) -> float:
    """
    Computes the normalized cosine similarity between two AIGs based on their number of gates and number of levels.

    Parameters:
    aig1, aig2 (Aig): The input AIGs to compare.

    Returns:
    float: The cosine similarity between the two AIGs, ranging from -1 to 1.
    """
    m1 = [aig1.num_gates(), aig1.num_levels()]
    m2 = [aig2.num_gates(), aig2.num_levels()]

    return cosine_similarity_metric(m1, m2)
