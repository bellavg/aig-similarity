import unittest
import networkx as nx
from NetComp.vertex_edge import vertex_edge_overlap,vertex_edge_distance
from distances import get_veo, get_ved
from aigverse import read_aiger_into_aig


class TestVertexEdgeOverlapDistance(unittest.TestCase):

    def setUp(self):
        # Set up some example graphs for testing

        # Graph 1: Path graph with 4 nodes
        self.G1 = nx.path_graph(4)

        # Graph 2: Identical to G1 (path graph with 4 nodes)
        self.G2 = nx.path_graph(4)

        # Graph 3: Cycle graph with 4 nodes
        self.G3 = nx.cycle_graph(4)

        # Graph 4: Path graph with 5 nodes (different size)
        self.G4 = nx.path_graph(5)

        # Graph 5: Disconnected graph with 3 nodes
        self.G5 = nx.Graph()
        self.G5.add_nodes_from([0, 1, 2])

        # Graph 6: Completely different graph with 4 nodes and a different structure
        self.G6 = nx.Graph()
        self.G6.add_edges_from([(0, 1), (1, 2), (2, 3)])

    def test_identical_graphs(self):
        # Test vertex-edge overlap for identical graphs
        overlap = vertex_edge_overlap(self.G1, self.G2)
        print(f"Vertex-edge overlap for identical graphs: {overlap}")
        self.assertAlmostEqual(overlap, 1.0, places=5, msg="Overlap of identical graphs should be 1.")

        # Test vertex-edge distance for identical graphs
        distance = vertex_edge_distance(self.G1, self.G2)
        print(f"Vertex-edge distance for identical graphs: {distance}")
        self.assertAlmostEqual(distance, 0.0, places=5, msg="Distance of identical graphs should be 0.")

    def test_completely_different_graphs(self):
        # Test vertex-edge overlap for completely different graphs
        overlap = vertex_edge_overlap(self.G1, self.G6)  # Path graph vs different structure
        print(f"Vertex-edge overlap for different graphs: {overlap}")
        self.assertGreater(overlap, 0, msg="Overlap of completely different graphs should be greater than 0.")
        self.assertLess(overlap, 1, msg="Overlap of completely different graphs should be less than 1.")

        # Test vertex-edge distance for completely different graphs
        distance = vertex_edge_distance(self.G1, self.G6)
        print(f"Vertex-edge distance for different graphs: {distance}")
        self.assertGreater(distance, 0, msg="Distance of completely different graphs should be greater than 0.")

    def test_different_size_graphs(self):
        # Test vertex-edge overlap for graphs of different sizes
        overlap = vertex_edge_overlap(self.G1, self.G4)  # Path graph with 4 nodes vs path graph with 5 nodes
        print(f"Vertex-edge overlap for different size graphs: {overlap}")
        self.assertGreater(overlap, 0, msg="Overlap of different size graphs should be greater than 0.")
        self.assertLess(overlap, 1, msg="Overlap of different size graphs should be less than 1.")

        # Test vertex-edge distance for graphs of different sizes
        distance = vertex_edge_distance(self.G1, self.G4)
        print(f"Vertex-edge distance for different size graphs: {distance}")
        self.assertGreater(distance, 0, msg="Distance of different size graphs should be greater than 0.")

    def test_disconnected_graph(self):
        # Test vertex-edge overlap for a disconnected graph and a connected graph
        overlap = vertex_edge_overlap(self.G1, self.G5)  # Path graph vs disconnected graph
        print(f"Vertex-edge overlap for connected vs disconnected graph: {overlap}")
        self.assertGreater(overlap, 0, msg="Overlap of connected vs disconnected graph should be greater than 0.")
        self.assertLess(overlap, 1, msg="Overlap of connected vs disconnected graph should be less than 1.")

        # Test vertex-edge distance for disconnected graphs
        distance = vertex_edge_distance(self.G1, self.G5)
        print(f"Vertex-edge distance for connected vs disconnected graph: {distance}")
        self.assertGreater(distance, 0, msg="Distance of connected vs disconnected graph should be greater than 0.")


    def test_single_node_graphs(self):
        # Test vertex-edge overlap for two single-node graphs
        G_single1 = nx.empty_graph(1)
        G_single2 = nx.empty_graph(1)
        overlap = vertex_edge_overlap(G_single1, G_single2)
        print(f"Vertex-edge overlap for two single-node graphs: {overlap}")
        self.assertAlmostEqual(overlap, 1.0, places=5, msg="Overlap of two single-node graphs should be 1.")

        # Test vertex-edge distance for two single-node graphs
        distance = vertex_edge_distance(G_single1, G_single2)
        print(f"Vertex-edge distance for two single-node graphs: {distance}")
        self.assertAlmostEqual(distance, 0.0, places=5, msg="Distance of two single-node graphs should be 0.")

    def test_with_aigs_identical(self):
        aig1 = read_aiger_into_aig("/Users/bellavg/AIG_SIM/data/aigs/espresso/ex03.aig")
        aig2 = read_aiger_into_aig("/Users/bellavg/AIG_SIM/data/aigs/espresso/ex03.aig")
        dist = get_veo(aig1, aig2)
        self.assertAlmostEqual(dist, 1.0, places=5,
                               msg="Resistance distance of a graph compared to itself should be zero.")

        dist2 = get_ved(aig1, aig2)
        self.assertAlmostEqual(dist2, 0, places=5,
                               msg="Resistance distance of a graph compared to itself should be zero.")

    def test_with_aigs_different(self):
        aig1 = read_aiger_into_aig("/Users/bellavg/AIG_SIM/data/aigs/espresso/ex03.aig")
        aig2 = read_aiger_into_aig("/Users/bellavg/AIG_SIM/data/aigs/espresso/ex19.aig")
        dist = get_veo(aig1, aig2)
        self.assertGreater(dist, 0, msg="Overlap of connected vs disconnected graph should be greater than 0.")
        dist = get_ved(aig1, aig2)
        self.assertGreater(dist, 0, msg="Distance of connected vs disconnected graph should be greater than 0.")



if __name__ == '__main__':
    unittest.main()
