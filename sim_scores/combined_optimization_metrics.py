from aigverse import Aig, aig_resubstitution, sop_refactoring, aig_cut_rewriting

from sim_scores.euclidean_similarity_metric import euclidean_distance_metric
from sim_scores.cosine_similarity_metric import cosine_similarity_metric
from sim_scores.canberra_distance_metric import canberra_distance_metric
from sim_scores.bray_curtis_dissimilarity_metric import bray_curtis_dissimilarity_metric


def relative_rrr(aig1: Aig, aig2: Aig) -> (list[float], list[float]):
    """
    Compute relative optimizability via rewrite, refactor, and resub for two AIGs.

    Parameters:
    aig1, aig2 (Aig): The input AIGs to compare.

    Returns:
    float: The RRR between the two AIGs.
    """
    # Get the original sizes of the AIGs
    original_size_1 = aig1.num_gates()
    original_size_2 = aig2.num_gates()

    if original_size_1 == 0 and original_size_2 == 0:
        return 0.0

    # Clone the AIGs to avoid modifying the original ones
    rw_aig1, rf_aig1, rs_aig1 = aig1.clone(), aig1.clone(), aig1.clone()
    rw_aig2, rf_aig2, rs_aig2 = aig2.clone(), aig2.clone(), aig2.clone()

    # Perform rewriting optimization on both AIGs
    aig_cut_rewriting(rw_aig1)
    aig_cut_rewriting(rw_aig2)

    # Get the optimized sizes of the AIGs
    rw_size_1 = rw_aig1.num_gates()
    rw_size_2 = rw_aig2.num_gates()

    rw_improvement_1 = (original_size_1 - rw_size_1) / original_size_1
    rw_improvement_2 = (original_size_2 - rw_size_2) / original_size_2

    # Perform refactoring optimization on both AIGs
    sop_refactoring(rf_aig1)
    sop_refactoring(rf_aig2)

    # Get the optimized sizes of the AIGs
    rf_size_1 = rf_aig1.num_gates()
    rf_size_2 = rf_aig2.num_gates()

    rf_improvement_1 = (original_size_1 - rf_size_1) / original_size_1
    rf_improvement_2 = (original_size_2 - rf_size_2) / original_size_2

    # Perform resubstitution optimization on both AIGs
    aig_resubstitution(rs_aig1)
    aig_resubstitution(rs_aig2)

    # Get the optimized sizes of the AIGs
    rs_size_1 = rs_aig1.num_gates()
    rs_size_2 = rs_aig2.num_gates()

    rs_improvement_1 = (original_size_1 - rs_size_1) / original_size_1
    rs_improvement_2 = (original_size_2 - rs_size_2) / original_size_2

    return ([rw_improvement_1, rf_improvement_1, rs_improvement_1],
            [rw_improvement_2, rf_improvement_2, rs_improvement_2])


def relative_rrr_euclidean_metric(aig1: Aig, aig2: Aig) -> float:
    """
    Compute the Euclidean metric for two AIGs based on their relative optimizability via rewrite, refactor,
    and resub.

    Parameters:
    aig1, aig2 (Aig): The input AIGs to compare.

    Returns:
    float: The relative RRR Euclidean metric between the two AIGs.
    """
    rrr = relative_rrr(aig1, aig2)

    return euclidean_distance_metric(rrr[0], rrr[1])


def relative_rrr_cosine_metric(aig1: Aig, aig2: Aig) -> float:
    """
    Compute the Cosine metric for two AIGs based on their relative optimizability via rewrite, refactor, and resub.

    Parameters:
    aig1, aig2 (Aig): The input AIGs to compare.

    Returns:
    float: The RRR Cosine metric between the two AIGs.
    """
    rrr = relative_rrr(aig1, aig2)

    return cosine_similarity_metric(rrr[0], rrr[1])


def relative_rrr_canberra_metric(aig1: Aig, aig2: Aig) -> float:
    """
    Compute the Canberra metric for two AIGs based on their relative optimizability via rewrite, refactor, and resub.

    Parameters:
    aig1, aig2 (Aig): The input AIGs to compare.

    Returns:
    float: The RRR Canberra metric between the two AIGs.
    """
    rrr = relative_rrr(aig1, aig2)

    return canberra_distance_metric(rrr[0], rrr[1])


def relative_rrr_bray_curtis_metric(aig1: Aig, aig2: Aig) -> float:
    """
    Compute the Bray-Curtis metric for two AIGs based on their relative optimizability via rewrite, refactor, and resub.

    Parameters:
    aig1, aig2 (Aig): The input AIGs to compare.

    Returns:
    float: The RRR Bray-Curtis metric between the two AIGs.
    """
    rrr = relative_rrr(aig1, aig2)

    return bray_curtis_dissimilarity_metric(rrr[0], rrr[1])
