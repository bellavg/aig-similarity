from sim_scores.spectral import get_lap_spectral_dist, get_adj_spectral_dist, get_directed_adj_sd
from sim_scores.netcomp_distances import get_net_simile, get_deltacon0, get_ns_dir_inverted, get_ns_dir_uninverted
from sim_scores.kernel_sim import get_kernel_sim
from sim_scores.veo import get_veo, get_directed_veo, get_directed_uninverted
from sim_scores.resub_metrics import absolute_resub_metric, relative_resub_metric
from sim_scores.size_diff_metrics import absolute_size_diff_metric, relative_size_diff_metric
from sim_scores.characteristics_metrics import absolute_gate_count_metric, relative_gate_count_metric, \
    absolute_edge_count_metric, relative_edge_count_metric, absolute_level_count_metric, relative_level_count_metric, \
    normalized_euclidean_similarity_metric, normalized_cosine_similarity_score




# Map function names to actual function calls
FUNCTION_MAP = {
    "deltacon0": get_deltacon0, # takes forever, not for no known node correspondence

    "netsimile": get_net_simile,
    "ns_inv": get_ns_dir_inverted,
    "ns_dir_uninverted": get_ns_dir_uninverted,

    "lap_sd": get_lap_spectral_dist,
    "adj_sd": get_adj_spectral_dist,
    "dir_edj_sd": get_directed_adj_sd,

    "veo": get_veo,
    "veo_dir": get_directed_veo,
    "veo_dir_uninverted":get_directed_uninverted,

    "kernel_sim": get_kernel_sim,

    "rel_resub": relative_resub_metric,
    "abs_resub": absolute_resub_metric,

    "abs_size_diff": absolute_size_diff_metric,
    "rel_size_diff": relative_size_diff_metric,

    "abs_gate_count": absolute_gate_count_metric,
    "rel_gate_count": relative_gate_count_metric,

    "abs_edge_count": absolute_edge_count_metric,
    "rel_edge_count": relative_edge_count_metric,

    "abs_level_count": absolute_level_count_metric,
    "rel_level_count": relative_level_count_metric,

    "euclidean": normalized_euclidean_similarity_metric,
    "cosine": normalized_cosine_similarity_score
}




