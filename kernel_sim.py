import networkx as nx
from utils import get_graph


def compute_graph_kernel(G1, G2, kernel_type='weisfeiler_lehman'):
    from grakel import Graph
    from grakel.kernels import WeisfeilerLehman, VertexHistogram, ShortestPath
    from sklearn.preprocessing import normalize

    def nx_to_grakel(G):
        if G.number_of_nodes() == 0:
            return Graph({}, node_labels={})
        else:
            adj_dict = nx.to_dict_of_lists(G)
            node_labels = nx.get_node_attributes(G, 'label')
            if node_labels:
                formatted_node_labels = {node: str(node_labels.get(node, '0')) for node in G.nodes()}
                return Graph(adj_dict, node_labels=formatted_node_labels)
            else:
                default_labels = {node: '0' for node in G.nodes()}
                return Graph(adj_dict, node_labels=default_labels)

    if G1.number_of_nodes() == 0 or G2.number_of_nodes() == 0:
        return 0.0

    if kernel_type == 'weisfeiler_lehman':
        gk = WeisfeilerLehman(n_iter=5)
    else:
        raise ValueError("Unsupported kernel type.")

    Gs = [nx_to_grakel(G1), nx_to_grakel(G2)]
    K = gk.fit_transform(Gs)

    # Normalize the kernel matrix
    K = normalize(K, norm='l2')

    similarity = K[0, 1]
    return similarity


def get_kernel_sim(aig1, aig2):
    G1, G2 = get_graph(aig1, aig2, directed=False)
    return compute_graph_kernel(G1, G2, kernel_type='weisfeiler_lehman')
