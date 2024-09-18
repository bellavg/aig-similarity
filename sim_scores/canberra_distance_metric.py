def canberra_distance_metric(attrs1: list[float], attrs2: list[float]) -> float:
    """
    Compute the Canberra distance metric for two lists of attributes. The Canberra distance is defined as the sum of
    the absolute differences between corresponding attributes, each normalized by the sum of the attributes' magnitudes.

    Parameters:
    attrs1 (list[float]): The first list of attributes.
    attrs2 (list[float]): The second list of attributes.

    Returns:
    float: The Canberra distance metric between the two attribute lists.
    """
    if len(attrs1) != len(attrs2):
        raise ValueError("Both attribute lists must have the same length.")

    # Compute the Canberra distance
    distance = sum(
        abs(a1 - a2) / (abs(a1) + abs(a2)) if (a1 != 0 or a2 != 0) else 0
        for a1, a2 in zip(attrs1, attrs2)
    )

    # The Canberra distance is the sum of these normalized differences, there is no need to normalize the final value
    return float(distance)
