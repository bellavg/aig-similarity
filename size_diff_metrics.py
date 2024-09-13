from aigverse import Aig



def absolute_size_diff_metric(aig1: Aig, aig2: Aig) -> int:
    """
    Compute the absolute size difference of two AIGs.

    Parameters:
    aig1, aig2 (Aig): The input AIGs to compare.

    Returns:
    int: The absolute size difference between the two AIGs.
    """
    # Get the sizes of the AIGs
    size_1 = aig1.num_gates()
    size_2 = aig2.num_gates()

    # Return the absolute difference between the two sizes
    return abs(size_1 - size_2)


def relative_size_diff_metric(aig1: Aig, aig2: Aig) -> float:
    """
    Compute the relative size difference of two AIGs.

    Parameters:
    aig1, aig2 (Aig): The input AIGs to compare.

    Returns:
    float: The relative size difference between the two AIGs.
    """
    # Get the sizes of the AIGs
    size_1 = aig1.num_gates()
    size_2 = aig2.num_gates()

    # Special case for when both sizes are 0
    if size_1 == 0 and size_2 == 0:
        return 0.0

    # Return the relative difference between the two sizes
    return abs(size_1 - size_2) / max(size_1, size_2)
