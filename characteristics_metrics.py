from aigverse import Aig, to_edge_list

import math
import numpy as np


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


def normalized_euclidean_similarity_metric(aig1: Aig, aig2: Aig) -> float:
    """
    Compute the normalized Euclidean similarity metric for two AIGs. The normalized Euclidean similarity metric is
    defined as the Euclidean distance between the normalized metrics of the two AIGs, where the metrics are the number
    of gates, number of edges, and number of levels. The similarity score is then calculated as 1 minus the normalized
    distance divided by the maximum possible distance in the normalized space.

    Parameters:
    aig1, aig2 (Aig): The input AIGs to compare.

    Returns:
    float: The normalized Euclidean similarity metric between the two AIGs.
    """
    edge_list1 = to_edge_list(aig1)
    edge_list2 = to_edge_list(aig2)

    # Extract metrics for AIG1 and AIG2
    g1, e1, l1 = aig1.num_gates(), len(edge_list1), aig1.num_levels()
    g2, e2, l2 = aig2.num_gates(), len(edge_list2), aig2.num_levels()

    # Define normalization factors (could be based on maximum values in the dataset or a fixed range)
    max_gates = max(g1, g2)
    max_edges = max(e1, e2)
    max_levels = max(l1, l2)

    # Avoid division by zero in case max values are zero
    max_gates = max(max_gates, 1)
    max_edges = max(max_edges, 1)
    max_levels = max(max_levels, 1)

    # Normalize metrics to [0, 1] range
    g1_norm, e1_norm, l1_norm = g1 / max_gates, e1 / max_edges, l1 / max_levels
    g2_norm, e2_norm, l2_norm = g2 / max_gates, e2 / max_edges, l2 / max_levels

    # Compute the Euclidean distance in the normalized space
    distance = np.sqrt((g1_norm - g2_norm) ** 2 + (e1_norm - e2_norm) ** 2 + (l1_norm - l2_norm) ** 2)

    # The maximum possible Euclidean distance in the normalized space is sqrt(3)
    max_distance = np.sqrt(3)

    # Compute the similarity score (1 - normalized distance)
    similarity_score = 1 - (distance / max_distance)

    return float(similarity_score)


def normalized_cosine_similarity_score(aig1: Aig, aig2: Aig) -> float:
    """
    Computes the normalized cosine similarity between two AIGs based on their number of gates, number of edges, and
    number of levels.

    Parameters:
        aig1, aig2 (Aig): The input AIGs to compare.

    Returns:
        float: The cosine similarity between the two AIGs, ranging from -1 to 1.
    """
    edge_list1 = to_edge_list(aig1)
    edge_list2 = to_edge_list(aig2)

    m1 = [aig1.num_gates(), len(edge_list1), aig1.num_levels()]
    m2 = [aig2.num_gates(), len(edge_list2), aig2.num_levels()]

    # Compute dot product of the two metric vectors
    dot_product = sum(x1 * x2 for x1, x2 in zip(m1, m2))

    # Compute the magnitudes (Euclidean norms) of the metric vectors
    magnitude1 = math.sqrt(sum(x1 ** 2 for x1 in m1))
    magnitude2 = math.sqrt(sum(x2 ** 2 for x2 in m2))

    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    # Compute cosine similarity
    cosine_similarity = dot_product / (magnitude1 * magnitude2)

    return cosine_similarity
