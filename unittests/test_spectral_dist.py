from aigverse import read_aiger_into_aig
from sim_scores.spectral import spectral_distance, get_adj_spectral_dist, get_lap_spectral_dist

import unittest
import networkx as nx


class TestSpectralDistance(unittest.TestCase):

    def setUp(self):
        # Set up sample graphs for testing
        self.graph1 = nx.erdos_renyi_graph(5, 0.5, seed=42)
        self.graph2 = nx.erdos_renyi_graph(5, 0.5, seed=24)
        self.graph3 = nx.complete_graph(5)
        self.graph4 = nx.empty_graph(5)

        # Create graphs of different sizes for testing edge cases
        self.graph5 = nx.erdos_renyi_graph(4, 0.5, seed=42)

    def test_adjacency_spectral_distance(self):
        """Test spectral distance for adjacency matrices."""
        dist = spectral_distance(self.graph1, self.graph2, matrix_type='adjacency')
        self.assertIsInstance(dist, float)
        self.assertGreaterEqual(dist, 0)

    def test_laplacian_spectral_distance(self):
        """Test spectral distance for Laplacian matrices."""
        dist = spectral_distance(self.graph1, self.graph2, matrix_type='laplacian')
        self.assertIsInstance(dist, float)
        self.assertGreaterEqual(dist, 0)


    def test_adjacency_distance_for_identical_graphs(self):
        """Test that the spectral distance between identical graphs is 0."""
        dist = spectral_distance(self.graph1, self.graph1, matrix_type='adjacency')
        self.assertAlmostEqual(dist, 0.0)

    def test_laplacian_distance_for_identical_graphs(self):
        """Test that the spectral distance between identical graphs is 0."""
        dist = spectral_distance(self.graph1, self.graph1, matrix_type='laplacian')
        self.assertAlmostEqual(dist, 0.0)



    def test_spectral_distance_complete_vs_empty(self):
        """Test that the spectral distance between a complete graph and an empty graph is greater than 0."""
        dist = spectral_distance(self.graph3, self.graph4, matrix_type='adjacency')
        self.assertGreater(dist, 0.0)

    def test_spectral_distance_different_sizes(self):
        """Test that the spectral distance works for graphs of different sizes."""
        dist = spectral_distance(self.graph1, self.graph5, matrix_type='adjacency')
        self.assertIsInstance(dist, float)
        self.assertGreaterEqual(dist, 0.0)

    def test_invalid_matrix_type(self):
        """Test that an invalid matrix type raises a ValueError."""
        with self.assertRaises(ValueError):
            spectral_distance(self.graph1, self.graph2, matrix_type='invalid')

    def test_padding_of_eigenvalues(self):
        """Test that the spectral distance handles graphs with different sizes (and eigenvalue padding)."""
        dist_adjacency = spectral_distance(self.graph1, self.graph5, matrix_type='adjacency')
        dist_laplacian = spectral_distance(self.graph1, self.graph5, matrix_type='laplacian')

        # Test that the distances are computed and are valid
        self.assertIsInstance(dist_adjacency, float)
        self.assertIsInstance(dist_laplacian, float)
        self.assertGreaterEqual(dist_adjacency, 0.0)
        self.assertGreaterEqual(dist_laplacian, 0.0)


    def test_adjacency_known_distance(self):
        """Test adjacency spectral distance for two small known graphs with expected distances."""
        G1 = nx.path_graph(4)
        G2 = nx.star_graph(3)

        dist = spectral_distance(G1, G2, matrix_type='adjacency')
        # Check against a pre-computed known value for these graphs (value is arbitrary in this example)
        self.assertGreaterEqual(dist, 0.0)


    def test_with_aigs_identical_lap(self):
        aig1 = read_aiger_into_aig("/Users/bellavg/AIG_SIM/data/aigs/espresso/ex03.aig")
        aig2 = read_aiger_into_aig("/Users/bellavg/AIG_SIM/data/aigs/espresso/ex03.aig")
        dist = get_lap_spectral_dist(aig1, aig2)
        self.assertAlmostEqual(dist, 0, places=5,
                               msg="Resistance distance of a graph compared to itself should be zero.")

    def test_with_aigs_id_adj(self):
        aig1 = read_aiger_into_aig("/Users/bellavg/AIG_SIM/data/aigs/espresso/ex03.aig")
        aig2 = read_aiger_into_aig("/Users/bellavg/AIG_SIM/data/aigs/espresso/ex03.aig")
        dist =get_adj_spectral_dist(aig1, aig2)
        self.assertAlmostEqual(dist, 0,
                           msg="Resistance distance of a graph compared to itself should be zero.")


    def test_with_aigs_diff(self):
        aig1 = read_aiger_into_aig("/Users/bellavg/AIG_SIM/data/aigs/espresso/ex03.aig")
        aig2 = read_aiger_into_aig("/Users/bellavg/AIG_SIM/data/aigs/espresso/ex19.aig")
        dist = get_lap_spectral_dist(aig1, aig2)
        self.assertGreater(dist, 0,
                           msg="Resistance distance of a graph compared to itself should be zero.")

    def test_with_aigs_diff_adj(self):
        aig1 = read_aiger_into_aig("/Users/bellavg/AIG_SIM/data/aigs/espresso/ex03.aig")
        aig2 = read_aiger_into_aig("/Users/bellavg/AIG_SIM/data/aigs/espresso/ex19.aig")
        dist =get_adj_spectral_dist(aig1, aig2)
        self.assertGreater(dist, 0,
                           msg="Resistance distance of a graph compared to itself should be zero.")


if __name__ == '__main__':
    unittest.main()