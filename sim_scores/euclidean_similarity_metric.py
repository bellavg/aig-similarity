import numpy as np


def euclidean_distance_metric(attrs1: list[float], attrs2: list[float]) -> float:
    """
    Compute the Euclidean distance between two lists of attributes. The Euclidean distance is defined as the square
    root of the sum of the squared differences between corresponding attributes.

    Parameters:
    attrs1 (list[float]): The first list of attributes.
    attrs2 (list[float]): The second list of attributes.

    Returns:
    float: The Euclidean distance between the two attribute lists.
    """
    if len(attrs1) != len(attrs2):
        raise ValueError("Both attribute lists must have the same length.")

    # Compute the sum of squared differences
    distance_squared = sum((a1 - a2) ** 2 for a1, a2 in zip(attrs1, attrs2))

    # Take the square root to get the Euclidean distance
    return float(distance_squared ** 0.5)


def normalized_euclidean_distance_metric(attrs1: list[float], attrs2: list[float]) -> float:
    """
    Compute the normalized Euclidean similarity metric for two lists of attributes. The normalized Euclidean similarity
    metric is defined as the Euclidean distance between the normalized attributes, where the similarity score is then
    calculated as 1 minus the normalized distance divided by the maximum possible distance in the normalized space.

    Parameters:
    attrs1 (list[float]): The first list of attributes.
    attrs2 (list[float]): The second list of attributes.

    Returns:
    float: The normalized Euclidean similarity metric between the two attribute lists.
    """
    if len(attrs1) != len(attrs2):
        raise ValueError("Both attribute lists must have the same length.")

    # Define normalization factors for each attribute
    max_values = [max(a1, a2) for a1, a2 in zip(attrs1, attrs2)]

    # Avoid division by zero by ensuring all max values are at least 1
    max_values = [max(mv, 1) for mv in max_values]

    # Normalize both attribute lists
    attrs1_norm = [a1 / mv for a1, mv in zip(attrs1, max_values)]
    attrs2_norm = [a2 / mv for a2, mv in zip(attrs2, max_values)]

    # Compute the Euclidean distance in the normalized space
    distance = np.sqrt(sum((a1 - a2) ** 2 for a1, a2 in zip(attrs1_norm, attrs2_norm)))

    # The maximum possible Euclidean distance is sqrt of the number of attributes (n-dimensional space)
    max_distance = np.sqrt(len(attrs1))

    # Compute the similarity score (1 - normalized distance)
    similarity_score = 1 - (distance / max_distance)

    return float(similarity_score)
