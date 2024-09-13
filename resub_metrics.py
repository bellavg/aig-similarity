from aigverse import Aig, aig_resubstitution


def absolute_resub_metric(aig1: Aig, aig2: Aig) -> int:
    """
    Compute the absolute resubstitution metric for two AIGs. The absolute resubstitution metric is defined as the
    absolute difference in size between the original and optimized AIGs after performing resubstitution optimization.

    Parameters:
    aig1, aig2 (Aig): The input AIGs to compare.

    Returns:
    int: The absolute resubstitution metric between the two AIGs.
    """
    # Clone the AIGs to avoid modifying the original ones
    aig1 = aig1.clone()
    aig2 = aig2.clone()

    # Get the original sizes of the AIGs
    original_size_1 = aig1.num_gates()
    original_size_2 = aig2.num_gates()

    # Perform resubstitution optimization on both AIGs
    aig_resubstitution(aig1)
    aig_resubstitution(aig2)

    # Get the optimized sizes of the AIGs
    optimized_size_1 = aig1.num_gates()
    optimized_size_2 = aig2.num_gates()

    # Return the absolute difference between the original and optimized sizes
    return abs((optimized_size_1 - original_size_1) - (optimized_size_2 - original_size_2))


def relative_resub_metric(aig1: Aig, aig2: Aig) -> float:
    """
    Compute the relative resubstitution metric for two AIGs. The relative resubstitution metric is defined as the
    relative difference in size between the original and optimized AIGs after performing resubstitution optimization.

    Parameters:
    aig1, aig2 (Aig): The input AIGs to compare.

    Returns:
    float: The relative resubstitution metric between the two AIGs.
    """
    # Clone the AIGs to avoid modifying the original ones
    aig1 = aig1.clone()
    aig2 = aig2.clone()

    # Get the original sizes of the AIGs
    original_size_1 = aig1.num_gates()
    original_size_2 = aig2.num_gates()

    # Perform resubstitution optimization on both AIGs
    aig_resubstitution(aig1)
    aig_resubstitution(aig2)

    # Get the optimized sizes of the AIGs
    optimized_size_1 = aig1.num_gates()
    optimized_size_2 = aig2.num_gates()

    relative_improvement_1 = (original_size_1 - optimized_size_1) / original_size_1
    relative_improvement_2 = (original_size_2 - optimized_size_2) / original_size_2

    # Return the relative difference between the original and optimized sizes
    return abs(relative_improvement_1 - relative_improvement_2)
