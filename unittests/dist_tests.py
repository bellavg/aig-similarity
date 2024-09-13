from distances import get_res_dist

import unittest
import networkx as nx
import numpy as np
from aigverse import read_aiger_into_aig
from NetComp.resistance_distance import resistance_distance


class TestGetResDist(unittest.TestCase):

    def setUp(self):
        # This setup creates a series of graphs for testing

        # Graph 1: Simple path graph with 4 nodes
        self.G1 = nx.path_graph(4)

        # Graph 2: Identical to G1 (same as path graph with 4 nodes)
        self.G2 = nx.path_graph(4)

        # Graph 3: A cycle graph with 4 nodes (same number of nodes, different structure)
        self.G3 = nx.cycle_graph(4)

        # Graph 4: Path graph with 5 nodes (different size)
        self.G4 = nx.path_graph(5)

        # Graph 5: Main connected component (path) and a small disconnected subcomponent (single node)
        self.G5 = nx.Graph()
        self.G5.add_edges_from([(0, 1), (1, 2)])  # Main component: 0 -- 1 -- 2
        self.G5.add_node(3)  # Disconnected subcomponent: Node 3

        # Graph 6: Main connected component (triangle) and a small disconnected subcomponent (single node)
        self.G6 = nx.Graph()
        self.G6.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4)])  # A triangle extended with additional nodes
        # Second subcomponent (a connected path or cycle, no edges connecting to the first subcomponent)
        self.G6.add_edges_from([(5, 6), (6, 7), (7, 8), (8, 5)])  # A cycle of nodes 5-8

        # Graph 7: Path graph with 3 nodes (smaller than others)
        self.G7 = nx.path_graph(3)

        # Graph 8: Empty
        self.G8 = nx.empty_graph(4)

    def get_res_dist(self, G1, G2):
        # Helper method to wrap the adjacency matrices into resistance distance function
        A1 = nx.adjacency_matrix(G1)
        A2 = nx.adjacency_matrix(G2)
        return resistance_distance(A1, A2)

    def test_identical_graphs(self):
        # Test that the resistance distance between identical graphs is zero (or near zero)
        dist = self.get_res_dist(self.G1, self.G2)
        self.assertAlmostEqual(dist, 0, places=5, msg="Resistance distance between identical graphs should be zero.")

    def test_different_topology_same_size(self):
        # Test that the resistance distance between different topology graphs with the same number of nodes is non-zero
        dist = self.get_res_dist(self.G1, self.G3)  # Path vs cycle
        self.assertGreater(dist, 0,
                           "Resistance distance between graphs with different topology should be greater than zero.")

    def test_different_size_graphs(self):
        # Test the resistance distance between graphs of different sizes
        dist = self.get_res_dist(self.G1, self.G4)  # Path graph with 4 nodes vs Path graph with 5 nodes
        self.assertGreater(dist, 0,
                           "Resistance distance between graphs of different sizes should be greater than zero.")

    def test_empty_graph_vs_nonempty_graph(self):
        # Test the resistance distance between an empty graph and a non-empty graph
        dist = self.get_res_dist(self.G8, self.G1)  # Empty graph vs Path graph
        self.assertGreater(dist, 0,
                           "Resistance distance between an empty and a non-empty graph should be greater than zero.")

    def test_disconnected_graphs(self):
        # Test the resistance distance between two disconnected graphs (G5 and G6)
        self.assertFalse(nx.is_connected(self.G5))
        self.assertFalse(nx.is_connected(self.G6))
        dist = self.get_res_dist(self.G5, self.G6)  # Both graphs are disconnected

        # The distance should now be greater than zero
        self.assertGreater(dist, 0,
                           "Resistance distance between different disconnected graphs should be greater than zero after renormalization.")

    def test_same_size_disconnected_vs_connected(self):
        # Test the resistance distance between a disconnected and connected graph with the same size
        dist = self.get_res_dist(self.G5, self.G1)  # Disconnected graph (G5) vs Connected graph (G1)
        self.assertGreater(dist, 0,
                           "Resistance distance between a disconnected and a connected graph should be greater than zero.")

    def test_smaller_vs_larger_graph(self):
        # Test the resistance distance between a smaller and larger graph
        dist = self.get_res_dist(self.G7, self.G1)  # Path graph with 3 nodes vs Path graph with 4 nodes
        self.assertGreater(dist, 0,
                           "Resistance distance between smaller and larger graphs should be greater than zero.")

    def test_self_comparison(self):
        # Test that comparing a graph to itself gives a resistance distance of zero
        dist = self.get_res_dist(self.G1, self.G1)
        self.assertAlmostEqual(dist, 0, places=5,
                               msg="Resistance distance of a graph compared to itself should be zero.")
        dist = self.get_res_dist(self.G2, self.G2)
        self.assertAlmostEqual(dist, 0, places=5,
                               msg="Resistance distance of a graph compared to itself should be zero.")
        dist = self.get_res_dist(self.G7, self.G7)
        self.assertAlmostEqual(dist, 0, places=5,
                               msg="Resistance distance of a graph compared to itself should be zero.")

    def test_different_beta_values(self):
        beta_values = [0.5, 1, 2, 10]
        previous_dist = None
        for beta in beta_values:
            dist = resistance_distance(nx.adjacency_matrix(self.G1),
                                       nx.adjacency_matrix(self.G3), beta=beta)
            if previous_dist is not None:
                self.assertNotEqual(dist, previous_dist,
                                    f"Resistance distance should vary with different beta values (beta={beta}).")
            previous_dist = dist

    def test_graphs_differing_by_one_edge(self):
        # Graph 13: Path graph with an extra edge
        self.G13 = self.G1.copy()
        self.G13.add_edge(0, 3)  # Add an edge to make it a cycle

        dist = self.get_res_dist(self.G1, self.G13)
        self.assertGreater(dist, 0,
                           "Resistance distance should reflect small structural changes (one edge difference).")


    def test_with_aigs_identical(self):
        aig1 = read_aiger_into_aig("/Users/bellavg/AIG_SIM/data/aigs/espresso/ex03.aig")
        aig2 = read_aiger_into_aig("/Users/bellavg/AIG_SIM/data/aigs/espresso/ex03.aig")
        dist = get_res_dist(aig1, aig2)
        self.assertAlmostEqual(dist, 0, places=5,
                               msg="Resistance distance of a graph compared to itself should be zero.")


    def test_with_aigs_diff(self):
        aig1 = read_aiger_into_aig("/Users/bellavg/AIG_SIM/data/aigs/espresso/ex03.aig")
        aig2 = read_aiger_into_aig("/Users/bellavg/AIG_SIM/data/aigs/espresso/ex19.aig")
        dist = get_res_dist(aig1, aig2)
        self.assertGreater(dist, 0,
                               msg="Resistance distance of a graph compared to itself should be zero.")

if __name__ == '__main__':
    unittest.main()
