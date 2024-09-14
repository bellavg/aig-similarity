import networkx as nx
from  graph_utils import get_graph


def compute_graph_kernel(G1, G2, kernel_type='weisfeiler_lehman'):
    from grakel import Graph
    from grakel.kernels import WeisfeilerLehman
    import numpy as np

    def nx_to_grakel(G):
        if G.number_of_nodes() == 0:
            return Graph()
        else:
            adj_dict = {n: list(G.neighbors(n)) for n in G.nodes()}
            node_labels = nx.get_node_attributes(G, 'label')
            if node_labels:
                formatted_node_labels = {node: str(node_labels[node]) for node in G.nodes()}
            else:
                # Assign unique labels if none are provided
                formatted_node_labels = {node: str(node) for node in G.nodes()}
            return Graph(adj_dict, node_labels=formatted_node_labels)

    if G1.number_of_nodes() == 0 or G2.number_of_nodes() == 0:
        return 0.0

    if kernel_type == 'weisfeiler_lehman':
        gk = WeisfeilerLehman(n_iter=5)
    else:
        raise ValueError("Unsupported kernel type.")

    Gs = [nx_to_grakel(G1), nx_to_grakel(G2)]
    K = gk.fit_transform(Gs)

    # Normalize the kernel matrix properly
    K_normalized = K / np.sqrt(np.outer(np.diag(K), np.diag(K)))

    similarity = K_normalized[0, 1]
    return similarity

def get_kernel_sim(aig1, aig2):
    G1, G2 = get_graph(aig1, aig2, directed=False)
    return compute_graph_kernel(G1, G2, kernel_type='weisfeiler_lehman')
