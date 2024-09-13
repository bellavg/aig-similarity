import unittest
import networkx as nx
from veo import vertex_edge_overlap
from veo import get_veo
from aigverse import read_aiger_into_aig

import unittest
import networkx as nx



class TestVertexEdgeOverlap(unittest.TestCase):

    def setUp(self):
        # Set up sample graphs for testing
        self.graph1 = nx.erdos_renyi_graph(5, 0.5, seed=42)
        self.graph2 = nx.erdos_renyi_graph(5, 0.5, seed=24)
        self.graph3 = nx.complete_graph(5)
        self.graph4 = nx.empty_graph(5)

        # Create graphs of different sizes for testing edge cases
        self.graph5 = nx.erdos_renyi_graph(4, 0.5, seed=42)
        self.graph6 = nx.erdos_renyi_graph(6, 0.5, seed=24)

    def test_vertex_edge_overlap_identical_graphs(self):
        """Test VEO for identical graphs returns 1.0."""
        veo_value = vertex_edge_overlap(self.graph1, self.graph1)
        self.assertAlmostEqual(veo_value, 1.0)

    def test_vertex_edge_overlap_different_graphs(self):
        """Test VEO for different graphs returns a value between 0 and 1."""
        veo_value = vertex_edge_overlap(self.graph1, self.graph2)
        self.assertGreaterEqual(veo_value, 0.0)
        self.assertLessEqual(veo_value, 1.0)

    def test_vertex_edge_overlap_same_vertices_no_edges(self):
        """Test VEO for graphs with same vertices but no edges."""
        G1 = nx.empty_graph(5)
        G2 = nx.empty_graph(5)
        veo_value = vertex_edge_overlap(G1, G2)
        self.assertEqual(veo_value, 1.0)  # Since both graphs have no edges but have identical vertices

    def test_vertex_edge_overlap_different_sizes(self):
        """Test VEO for graphs with different numbers of vertices."""
        veo_value = vertex_edge_overlap(self.graph1, self.graph5)
        self.assertGreaterEqual(veo_value, 0.0)
        self.assertLessEqual(veo_value, 1.0)

    def test_vertex_edge_overlap_known_case(self):
        """Test VEO for a known case of two small graphs."""
        G1 = nx.path_graph(4)
        G2 = nx.star_graph(3)

        veo_value = vertex_edge_overlap(G1, G2)
        # Manually compute expected VEO value
        expected_value = 2 * (len(set(G1.nodes()).intersection(G2.nodes())) +
                              len(set(G1.edges()).intersection(G2.edges()))) / \
                         (len(G1.nodes()) + len(G2.nodes()) + len(G1.edges()) + len(G2.edges()))
        self.assertAlmostEqual(veo_value, expected_value)

    def test_vertex_edge_overlap_no_vertices_no_edges(self):
        """Test VEO for graphs with no vertices and no edges (empty graphs)."""
        G1 = nx.Graph()
        G2 = nx.Graph()
        veo_value = vertex_edge_overlap(G1, G2)
        self.assertEqual(veo_value, 1.0)  # Two empty graphs should be considered identical


    def test_with_aigs_identical(self):
        aig1 = read_aiger_into_aig("/Users/bellavg/AIG_SIM/data/aigs/espresso/ex03.aig")
        aig2 = read_aiger_into_aig("/Users/bellavg/AIG_SIM/data/aigs/espresso/ex03.aig")
        dist = get_veo(aig1, aig2)
        self.assertAlmostEqual(dist, 1.0, places=5,
                               msg="Should be  1 for identical")

    def test_with_aigs_diff(self):
        aig1 = read_aiger_into_aig("/Users/bellavg/AIG_SIM/data/aigs/espresso/ex03.aig")
        aig2 = read_aiger_into_aig("/Users/bellavg/AIG_SIM/data/aigs/espresso/ex19.aig")
        dist = get_veo(aig1, aig2)
        self.assertGreater(dist, 0, msg="Should be greater than 0")
        self.assertLess(dist, 1.0, msg="Should be less than 1.0")


if __name__ == '__main__':
    unittest.main()
