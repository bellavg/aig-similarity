"""
********
Features
********

Calculation of features for NetSimile algorithm.
"""

import networkx as nx
import numpy as np
from scipy import stats

_eps = 1e-10


def get_features(G):
    """Extract features for NetSimile algorithm."""
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return np.empty((0, 7))

    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    # degrees
    d_vec = np.array([G.degree(node) for node in nodes], dtype=float)

    # clustering coefficient
    clust_dict = nx.clustering(G)
    clust_vec = np.array([clust_dict[node] for node in nodes], dtype=float)

    # neighbors
    neighbors = [list(G.neighbors(node)) for node in nodes]

    # average degree of neighbors
    neighbor_deg = []
    for i in range(n):
        neighbor_nodes = neighbors[i]
        if neighbor_nodes:
            neighbor_indices = [node_to_idx[neighbor] for neighbor in neighbor_nodes]
            deg_sum = d_vec[neighbor_indices].sum()
            avg_deg = deg_sum / len(neighbor_nodes)
        else:
            avg_deg = 0
        neighbor_deg.append(avg_deg)

    # average clustering coefficient of neighbors
    neighbor_clust = []
    for i in range(n):
        neighbor_nodes = neighbors[i]
        if neighbor_nodes:
            neighbor_indices = [node_to_idx[neighbor] for neighbor in neighbor_nodes]
            clust_sum = clust_vec[neighbor_indices].sum()
            avg_clust = clust_sum / len(neighbor_nodes)
        else:
            avg_clust = 0
        neighbor_clust.append(avg_clust)

    # egonets
    egonets = [nx.ego_graph(G, node) for node in nodes]

    # number of edges in egonet
    ego_size = [egonet.number_of_edges() for egonet in egonets]

    # number of neighbors of egonet
    ego_neighbors = []
    for i in range(n):
        ego_nodes = set(egonets[i].nodes())
        all_neighbors = set()
        for node in ego_nodes:
            all_neighbors.update(G.neighbors(node))
        ego_neighbor_nodes = all_neighbors - ego_nodes
        ego_neighbors.append(len(ego_neighbor_nodes))

    # number of edges outgoing from egonet
    outgoing_edges = []
    for i in range(n):
        ego_nodes = set(egonets[i].nodes())
        count = 0
        for node in ego_nodes:
            for neighbor in G.neighbors(node):
                if neighbor not in ego_nodes:
                    count += 1
        outgoing_edges.append(count)

    # assemble feature matrix
    feature_mat = np.array([
        d_vec,
        clust_vec,
        neighbor_deg,
        neighbor_clust,
        ego_size,
        ego_neighbors,
        outgoing_edges
    ], dtype=float).T

    # Replace any NaN values with zeros
    feature_mat = np.nan_to_num(feature_mat, nan=0.0)

    return feature_mat


def aggregate_features(feature_mat, row_var=False, as_matrix=False):
    """Returns column-wise descriptive statistics of a feature matrix.

    Parameters
    ----------
    feature_mat : NumPy array
        Matrix on which statistics are to be calculated. Assumed to be formatted
        so each row is an observation (a node, in the case of NetSimile).

    row_var : bool, optional (default=False)
        If True, then each variable has its own row, and statistics are
        computed along rows rather than columns.

    as_matrix : bool, optional (default=False)
        If True, then description is returned as matrix. Otherwise, it is
        flattened into a vector.

    Returns
    -------
    description : NumPy array
        Descriptive statistics of feature_mat
    """
    axis = int(row_var)  # 0 if column-oriented, 1 if not

    # Use nan-aware functions to ignore NaN values
    mean = np.nanmean(feature_mat, axis=axis)
    median = np.nanmedian(feature_mat, axis=axis)
    std = np.nanstd(feature_mat, axis=axis)

    # Initialize skewness and kurtosis
    if axis == 0:
        num_features = feature_mat.shape[1]
    else:
        num_features = feature_mat.shape[0]
    skewness = np.zeros(num_features)
    kurtosis = np.zeros(num_features)

    # Threshold for determining if data are nearly identical
    delta = 1e-8

    # For each feature, compute skewness and kurtosis if data varies enough
    for i in range(num_features):
        # Get the data for this feature
        if axis == 0:
            data = feature_mat[:, i]
        else:
            data = feature_mat[i, :]

        # Remove NaN values
        data = data[~np.isnan(data)]

        if len(data) == 0:
            # Can't compute statistics on empty data
            skewness[i] = 0.0
            kurtosis[i] = 0.0
            continue

        data_range = np.max(data) - np.min(data)
        if data_range < delta:
            # Data are nearly identical; set skewness and kurtosis to zero
            skewness[i] = 0.0
            kurtosis[i] = 0.0
        else:
            # Compute skewness and kurtosis
            skewness[i] = stats.skew(data, bias=False)
            kurtosis[i] = stats.kurtosis(data, bias=False)

    description = np.array([mean, median, std, skewness, kurtosis])

    if not as_matrix:
        description = description.flatten()
    return description
