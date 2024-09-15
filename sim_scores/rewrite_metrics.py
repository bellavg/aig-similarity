from aigverse import Aig, aig_cut_rewriting


def absolute_rewrite_metric(aig1: Aig, aig2: Aig) -> int:
    """
    Compute the absolute rewrite metric for two AIGs. The absolute rewrite metric is defined as the
    absolute difference in size between the original and optimized AIGs after performing the cut rewriting optimization.

    Parameters:
    aig1, aig2 (Aig): The input AIGs to compare.

    Returns:
    int: The absolute rewrite metric between the two AIGs.
    """
    # Clone the AIGs to avoid modifying the original ones
    aig1 = aig1.clone()
    aig2 = aig2.clone()

    # Get the original sizes of the AIGs
    original_size_1 = aig1.num_gates()
    original_size_2 = aig2.num_gates()

    # Perform cut rewriting on both AIGs
    aig_cut_rewriting(aig1)
    aig_cut_rewriting(aig2)

    # Get the optimized sizes of the AIGs
    optimized_size_1 = aig1.num_gates()
    optimized_size_2 = aig2.num_gates()

    # Return the absolute difference between the original and optimized sizes
    return abs((optimized_size_1 - original_size_1) - (optimized_size_2 - original_size_2))


def relative_rewrite_metric(aig1: Aig, aig2: Aig) -> float:
    """
    Compute the relative rewrite metric for two AIGs. The relative rewrite metric is defined as the
    relative difference in size between the original and optimized AIGs after performing the cut rewriting optimization.

    Parameters:
    aig1, aig2 (Aig): The input AIGs to compare.

    Returns:
    float: The relative rewrite metric between the two AIGs.
    """
    # Clone the AIGs to avoid modifying the original ones
    aig1 = aig1.clone()
    aig2 = aig2.clone()

    # Get the original sizes of the AIGs
    original_size_1 = aig1.num_gates()
    original_size_2 = aig2.num_gates()

    if original_size_1 == 0 and original_size_2 == 0:
        return 0.0

    # Perform cut rewriting on both AIGs
    aig_cut_rewriting(aig1)
    aig_cut_rewriting(aig2)

    # Get the optimized sizes of the AIGs
    optimized_size_1 = aig1.num_gates()
    optimized_size_2 = aig2.num_gates()

    relative_improvement_1 = (original_size_1 - optimized_size_1) / original_size_1
    relative_improvement_2 = (original_size_2 - optimized_size_2) / original_size_2

    # Return the relative difference between the original and optimized sizes
    return abs(relative_improvement_1 - relative_improvement_2)
