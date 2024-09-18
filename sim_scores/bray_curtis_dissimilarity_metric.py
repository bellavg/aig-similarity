def bray_curtis_dissimilarity_metric(attrs1: list[float], attrs2: list[float]) -> float:
    """
    Compute the Bray-Curtis dissimilarity metric for two lists of attributes. The Bray-Curtis dissimilarity is defined
    as the sum of the absolute differences between corresponding attributes, normalized by the sum of the absolute values
    of both attributes.

    Parameters:
    attrs1 (list[float]): The first list of attributes.
    attrs2 (list[float]): The second list of attributes.

    Returns:
    float: The Bray-Curtis dissimilarity metric between the two attribute lists, ranging from 0 to 1.
    """
    if len(attrs1) != len(attrs2):
        raise ValueError("Both attribute lists must have the same length.")

    # Compute the Bray-Curtis dissimilarity
    numerator = sum(abs(a1 - a2) for a1, a2 in zip(attrs1, attrs2))
    denominator = sum(abs(a1) + abs(a2) for a1, a2 in zip(attrs1, attrs2))

    # Avoid division by zero in case the denominator is zero (i.e., both vectors are all zeros)
    if denominator == 0:
        return 0.0

    # Compute the Bray-Curtis dissimilarity
    dissimilarity = numerator / denominator

    return float(dissimilarity)
