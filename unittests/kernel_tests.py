import unittest
import networkx as nx
from kernel_sim import compute_graph_kernel

class TestComputeGraphKernel(unittest.TestCase):

    def setUp(self):
        # Create graphs for testing
        # In setUp, define new graphs for this test
        self.graph1 = nx.path_graph(5)  # Path graph
        self.graph2 = nx.star_graph(4)  # Star graph, structurally different from path graph
        self.graph3 = nx.complete_graph(5)  # Complete graph
        self.graph4 = nx.empty_graph(5)  # Empty graph (no edges)
        self.graph5 = nx.Graph()  # Empty graph (no nodes)
        self.graph6 = nx.path_graph(5)
        self.graph7 = nx.path_graph(6)  # Different size graph

        # Add node labels to some graphs
        for node in self.graph1.nodes():
            self.graph1.nodes[node]['label'] = 'A'
        for node in self.graph2.nodes():
            self.graph2.nodes[node]['label'] = 'A'
        for node in self.graph3.nodes():
            self.graph3.nodes[node]['label'] = 'B'
        for node in self.graph7.nodes():
            self.graph7.nodes[node]['label'] = 'A'

    def test_identical_graphs_wl_kernel(self):
        """Test similarity of identical graphs using Weisfeiler-Lehman kernel."""
        similarity = compute_graph_kernel(self.graph1, self.graph1, kernel_type='weisfeiler_lehman')
        self.assertGreaterEqual(similarity, 0,
                                msg="Similarity score should be non-negative.")
        self.assertIsInstance(similarity, (float, int),
                              msg="Similarity score should be a number.")

    def test_different_graphs_wl_kernel(self):
        """Test similarity of different graphs using Weisfeiler-Lehman kernel."""
        similarity = compute_graph_kernel(self.graph1, self.graph2, kernel_type='weisfeiler_lehman')
        self.assertGreaterEqual(similarity, 0,
                                msg="Similarity score should be non-negative.")
        self.assertLess(similarity, compute_graph_kernel(self.graph1, self.graph1),
                        msg="Similarity between different graphs should be less than similarity between identical graphs.")

    def test_graphs_with_labels_wl_kernel(self):
        """Test similarity of graphs with different labels using Weisfeiler-Lehman kernel."""
        similarity = compute_graph_kernel(self.graph1, self.graph3, kernel_type='weisfeiler_lehman')
        self.assertGreaterEqual(similarity, 0,
                                msg="Similarity score should be non-negative.")

    def test_graphs_without_labels_wl_kernel(self):
        """Test similarity of graphs without labels using Weisfeiler-Lehman kernel."""
        # Remove labels from graph1
        for node in self.graph1.nodes():
            del self.graph1.nodes[node]['label']

        similarity = compute_graph_kernel(self.graph1, self.graph2, kernel_type='weisfeiler_lehman')
        self.assertGreaterEqual(similarity, 0,
                                msg="Similarity score should be non-negative.")

    def test_unsupported_kernel(self):
        """Test that an unsupported kernel type raises a ValueError."""
        with self.assertRaises(ValueError):
            compute_graph_kernel(self.graph1, self.graph2, kernel_type='unsupported_kernel')

    def test_graphs_of_different_sizes(self):
        """Test similarity of graphs with different number of nodes."""
        similarity = compute_graph_kernel(self.graph1, self.graph7, kernel_type='weisfeiler_lehman')
        self.assertGreaterEqual(similarity, 0,
                                msg="Similarity score should be non-negative.")
        self.assertLess(similarity, compute_graph_kernel(self.graph1, self.graph1),
                        msg="Similarity between different graphs should be less than similarity between identical graphs.")

    def test_graph_with_self_loops(self):
        """Test similarity of graphs with self-loops."""
        # Add self-loops to graph1
        self.graph1_with_loops = self.graph1.copy()
        self.graph1_with_loops.add_edge(0, 0)
        self.graph1_with_loops.add_edge(1, 1)

        similarity = compute_graph_kernel(self.graph1, self.graph1_with_loops, kernel_type='weisfeiler_lehman')
        self.assertGreaterEqual(similarity, 0,
                                msg="Similarity score should be non-negative.")
        self.assertLess(similarity, compute_graph_kernel(self.graph1, self.graph1),
                        msg="Similarity between graph and its self-looped version should be less than similarity between identical graphs.")

if __name__ == '__main__':
    unittest.main()
