import unittest

import networkx as nx
from aigverse import read_aiger_into_aig

from NetComp.deltacon0 import deltacon0
from sim_scores.netcomp_distances import get_deltacon0


class TestDeltaCon(unittest.TestCase):

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

        # Graph 5: Disconnected graph with two components
        self.G5 = nx.Graph()
        self.G5.add_edges_from([(0, 1), (1, 2)])  # Main component: 0 -- 1 -- 2
        self.G5.add_node(3)  # Disconnected subcomponent: Node 3

        # Graph 6: Two large disconnected components
        self.G6 = nx.Graph()
        self.G6.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4)])  # Subcomponent 1: 0 -- 1 -- 2 -- 3 -- 4
        self.G6.add_edges_from([(5, 6), (6, 7), (7, 8), (8, 5)])  # Subcomponent 2: 5 -- 6 -- 7 -- 8

        # Graph 7: Empty graph with no nodes
        self.G7 = nx.empty_graph(0)

        # Graph 8: Single-node graph
        self.G8 = nx.Graph()
        self.G8.add_node(0)  # One node, no edges

    def get_deltacon0(self, G1, G2):
        # Helper method to wrap the adjacency matrices into the deltacon function
        A1 = nx.adjacency_matrix(G1).tocsc()
        A2 = nx.adjacency_matrix(G2).tocsc()
        return deltacon0(A1, A2)

    def test_identical_graphs(self):
        # Test that the deltacon0 distance between identical graphs is very small or zero
        dist = self.get_deltacon0(self.G1, self.G2)
        print(f"Distance between identical graphs: {dist}")
        self.assertAlmostEqual(dist, 0, places=5,
                               msg="DeltaCon0 distance between identical graphs should be zero or near zero.")

    def test_completely_different_graphs(self):
        # Test that the deltacon0 distance between completely different graphs is large
        dist = self.get_deltacon0(self.G1, self.G6)  # Path graph vs disconnected graph with large components
        print(f"Distance between completely different graphs: {dist}")
        self.assertGreater(dist, 0,
                           msg="DeltaCon0 distance between completely different graphs should be greater than zero.")

    def test_different_size_graphs(self):
        # Test deltacon0 with graphs of different sizes
        dist = self.get_deltacon0(self.G1, self.G4)  # Path graph with 4 nodes vs path graph with 5 nodes
        print(f"Distance between different size graphs: {dist}")
        self.assertGreater(dist, 0,
                           msg="DeltaCon0 distance between graphs of different sizes should be greater than zero.")

    def test_disconnected_components(self):
        # Test deltacon0 with graphs that have disconnected components
        dist = self.get_deltacon0(self.G5, self.G6)  # Disconnected graph vs disconnected graph
        print(f"Distance between disconnected graphs: {dist}")
        self.assertGreater(dist, 0,
                           msg="DeltaCon0 distance between graphs with disconnected components should be greater than zero.")

    def test_single_node_graph(self):
        # Test deltacon0 with single-node graphs
        dist = self.get_deltacon0(self.G8, self.G8)  # Single-node graph compared to itself
        print(f"Distance between single-node graphs: {dist}")
        self.assertAlmostEqual(dist, 0, places=5,
                               msg="DeltaCon0 distance between identical single-node graphs should be zero.")

    def test_with_aigs_identical(self):
        aig1 = read_aiger_into_aig("/Users/bellavg/AIG_SIM/data/aigs/espresso/ex03.aig")
        aig2 = read_aiger_into_aig("/Users/bellavg/AIG_SIM/data/aigs/espresso/ex03.aig")
        dist = get_deltacon0(aig1, aig2)
        self.assertAlmostEqual(dist, 0, places=5,
                               msg="Resistance distance of a graph compared to itself should be zero.")

    def test_with_aigs_diff(self):
        aig1 = read_aiger_into_aig("/Users/bellavg/AIG_SIM/data/aigs/espresso/ex03.aig")
        aig2 = read_aiger_into_aig("/Users/bellavg/AIG_SIM/data/aigs/espresso/ex19.aig")
        dist = get_deltacon0(aig1, aig2)
        self.assertGreater(dist, 0,
                           msg="Resistance distance of a graph compared to itself should be zero.")




if __name__ == '__main__':
    unittest.main()
