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
    """Returns column-wise descriptive statistics of a feature matrix."""
    axis = int(row_var)  # 0 if column-oriented, 1 if not

    # Use nan-aware functions to ignore NaN values
    mean = np.nanmean(feature_mat, axis=axis)
    median = np.nanmedian(feature_mat, axis=axis)
    std = np.nanstd(feature_mat, axis=axis)

    # Handle skewness and kurtosis with try-except blocks
    try:
        skewness = stats.skew(feature_mat, axis=axis, bias=False, nan_policy='omit')
        skewness = np.nan_to_num(skewness, nan=0.0)
    except:
        skewness = np.zeros(feature_mat.shape[1 - axis])

    try:
        kurtosis = stats.kurtosis(feature_mat, axis=axis, bias=False, nan_policy='omit')
        kurtosis = np.nan_to_num(kurtosis, nan=0.0)
    except:
        kurtosis = np.zeros(feature_mat.shape[1 - axis])

    description = np.array([mean, median, std, skewness, kurtosis])

    if not as_matrix:
        description = description.flatten()
    return description