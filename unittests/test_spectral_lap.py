import unittest

import networkx as nx
import numpy as np
from aigverse import read_aiger_into_aig

from spectral import get_lap_spectral_distance


def lap_spectral_distance(G1, G2, k=None):
    """
    Compute the spectral distance between two graphs based on their Laplacian spectra.

    Parameters
    ----------
    G1, G2 : networkx.Graph
        The graphs to compare.

    k : int, optional
        Number of eigenvalues to consider. If None, uses all eigenvalues.

    Returns
    -------
    dist : float
        The spectral distance between the graphs.
    """
    # Compute Laplacian matrices

    L1 = nx.laplacian_matrix(G1).todense()
    L2 = nx.laplacian_matrix(G2).todense()

    # Compute eigenvalues
    eigs1 = np.sort(np.linalg.eigvalsh(L1))
    eigs2 = np.sort(np.linalg.eigvalsh(L2))

    # Use the smallest size
    min_size = min(len(eigs1), len(eigs2))
    if k is not None:
        min_size = min(min_size, k)
    eigs1 = eigs1[:min_size]
    eigs2 = eigs2[:min_size]

    # Compute the distance
    dist = np.linalg.norm(eigs1 - eigs2)
    return dist

class TestSpectralDistance(unittest.TestCase):

    def setUp(self):
        # Set up some example graphs

        # Graph 1: Path graph with 4 nodes
        self.G1 = nx.path_graph(4)

        # Graph 2: Identical to G1 (same as path graph with 4 nodes)
        self.G2 = nx.path_graph(4)

        # Graph 3: Cycle graph with 4 nodes
        self.G3 = nx.cycle_graph(4)

        # Graph 4: Path graph with 5 nodes (different size)
        self.G4 = nx.path_graph(5)

        # Graph 5: Disconnected graph with 2 components
        self.G5 = nx.Graph()
        self.G5.add_edges_from([(0, 1), (1, 2)])  # Component 1: 0 -- 1 -- 2
        self.G5.add_node(3)  # Disconnected node (Component 2)


    def test_identical_graphs(self):
        # Test spectral distance between two identical graphs (should be zero)
        dist = lap_spectral_distance(self.G1, self.G2)
        print(f"Spectral distance between identical graphs: {dist}")
        self.assertAlmostEqual(dist, 0, places=5, msg="Spectral distance between identical graphs should be zero or near zero.")

    def test_completely_different_graphs(self):
        # Test spectral distance between two completely different graphs
        dist = lap_spectral_distance(self.G1, self.G3)  # Path vs cycle graph
        print(f"Spectral distance between path and cycle graphs: {dist}")
        self.assertGreater(dist, 0, msg="Spectral distance between different graphs should be greater than zero.")

    def test_different_size_graphs(self):
        # Test spectral distance between graphs of different sizes
        dist = lap_spectral_distance(self.G1, self.G4)  # Path graph with 4 nodes vs 5 nodes
        print(f"Spectral distance between graphs of different sizes: {dist}")
        self.assertGreater(dist, 0, msg="Spectral distance between graphs of different sizes should be greater than zero.")

    def test_disconnected_graphs(self):
        # Test spectral distance between disconnected and connected graphs
        dist = lap_spectral_distance(self.G1, self.G5)  # Path graph vs disconnected graph
        print(f"Spectral distance between connected and disconnected graphs: {dist}")
        self.assertGreater(dist, 0, msg="Spectral distance between connected and disconnected graphs should be greater than zero.")

    def test_k_eigenvalues(self):
        # Test spectral distance with a limited number of eigenvalues (k=2)
        dist = lap_spectral_distance(self.G1, self.G3, k=2)  # Path vs cycle graph, only 2 eigenvalues
        print(f"Spectral distance using only 2 eigenvalues: {dist}")
        self.assertGreater(dist, 0, msg="Spectral distance with k eigenvalues should be greater than zero.")

    def test_empty_graphs(self):
        # Test spectral distance between two empty graphs
        G_empty1 = nx.empty_graph(0)
        G_empty2 = nx.empty_graph(0)
        dist = lap_spectral_distance(G_empty1, G_empty2)
        print(f"Spectral distance between two empty graphs: {dist}")
        self.assertAlmostEqual(dist, 0, places=5, msg="Spectral distance between two empty graphs should be zero.")

    def test_single_node_graphs(self):
        # Test spectral distance between two single-node graphs
        G_single1 = nx.empty_graph(1)
        G_single2 = nx.empty_graph(1)
        dist = lap_spectral_distance(G_single1, G_single2)
        print(f"Spectral distance between two single-node graphs: {dist}")
        self.assertAlmostEqual(dist, 0, places=5, msg="Spectral distance between two single-node graphs should be zero.")


    def test_with_aigs_identical(self):
        aig1 = read_aiger_into_aig("/Users/bellavg/AIG_SIM/data/aigs/espresso/ex03.aig")
        aig2 = read_aiger_into_aig("/Users/bellavg/AIG_SIM/data/aigs/espresso/ex03.aig")
        dist =get_lap_spectral_distance(aig1, aig2)
        self.assertAlmostEqual(dist, 0, places=5,
                               msg="Resistance distance of a graph compared to itself should be zero.")

    def test_with_aigs_diff(self):
        aig1 = read_aiger_into_aig("/Users/bellavg/AIG_SIM/data/aigs/espresso/ex03.aig")
        aig2 = read_aiger_into_aig("/Users/bellavg/AIG_SIM/data/aigs/espresso/ex19.aig")
        dist = get_lap_spectral_distance(aig1, aig2)
        self.assertGreater(dist, 0,
                           msg="Resistance distance of a graph compared to itself should be zero.")

if __name__ == '__main__':
    unittest.main()
