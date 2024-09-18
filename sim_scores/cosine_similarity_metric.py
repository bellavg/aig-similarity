import math


def cosine_similarity_metric(attrs1: list[float], attrs2: list[float]) -> float:
    """
    Computes the cosine similarity between two lists of attributes. The cosine similarity is the dot product of
    the two normalized attribute vectors divided by the product of their magnitudes.

    Parameters:
    attrs1 (list[float]): The first list of attributes.
    attrs2 (list[float]): The second list of attributes.

    Returns:
    float: The cosine similarity between the two attribute lists, ranging from -1 to 1.
    """
    if len(attrs1) != len(attrs2):
        raise ValueError("Both attribute lists must have the same length.")

    # Compute dot product of the two attribute vectors
    dot_product = sum(x1 * x2 for x1, x2 in zip(attrs1, attrs2))

    # Compute the magnitudes (Euclidean norms) of the attribute vectors
    magnitude1 = math.sqrt(sum(x1 ** 2 for x1 in attrs1))
    magnitude2 = math.sqrt(sum(x2 ** 2 for x2 in attrs2))

    # Handle the case where both vectors are zero vectors
    if magnitude1 == 0 and magnitude2 == 0:
        return 0.0

    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    # Compute cosine similarity
    cosine_similarity = dot_product / (magnitude1 * magnitude2)

    return cosine_similarity
