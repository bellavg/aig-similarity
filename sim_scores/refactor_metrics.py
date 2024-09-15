from aigverse import Aig, sop_refactoring


def absolute_refactor_metric(aig1: Aig, aig2: Aig) -> int:
    """
    Compute the absolute refactor metric for two AIGs. The absolute refactor metric is defined as the
    absolute difference in size between the original and optimized AIGs after performing the SOP refactoring
    optimization.

    Parameters:
    aig1, aig2 (Aig): The input AIGs to compare.

    Returns:
    int: The absolute refactor metric between the two AIGs.
    """
    # Clone the AIGs to avoid modifying the original ones
    aig1 = aig1.clone()
    aig2 = aig2.clone()

    # Get the original sizes of the AIGs
    original_size_1 = aig1.num_gates()
    original_size_2 = aig2.num_gates()

    # Perform SOP refactoring on both AIGs
    sop_refactoring(aig1)
    sop_refactoring(aig2)

    # Get the optimized sizes of the AIGs
    optimized_size_1 = aig1.num_gates()
    optimized_size_2 = aig2.num_gates()

    # Return the absolute difference between the original and optimized sizes
    return abs((optimized_size_1 - original_size_1) - (optimized_size_2 - original_size_2))


def relative_refactor_metric(aig1: Aig, aig2: Aig) -> float:
    """
    Compute the relative refactor metric for two AIGs. The relative refactor metric is defined as the
    relative difference in size between the original and optimized AIGs after performing the SOP refactoring
    optimization.

    Parameters:
    aig1, aig2 (Aig): The input AIGs to compare.

    Returns:
    float: The relative refactor metric between the two AIGs.
    """
    # Clone the AIGs to avoid modifying the original ones
    aig1 = aig1.clone()
    aig2 = aig2.clone()

    # Get the original sizes of the AIGs
    original_size_1 = aig1.num_gates()
    original_size_2 = aig2.num_gates()

    if original_size_1 == 0 and original_size_2 == 0:
        return 0.0

    # Perform SOP refactoring on both AIGs
    sop_refactoring(aig1)
    sop_refactoring(aig2)

    # Get the optimized sizes of the AIGs
    optimized_size_1 = aig1.num_gates()
    optimized_size_2 = aig2.num_gates()

    relative_improvement_1 = (original_size_1 - optimized_size_1) / original_size_1
    relative_improvement_2 = (original_size_2 - optimized_size_2) / original_size_2

    # Return the relative difference between the original and optimized sizes
    return abs(relative_improvement_1 - relative_improvement_2)
