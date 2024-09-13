from utils import get_graph, transform_edge_list
import unittest


class TestTransformEdgeList(unittest.TestCase):

    def test_no_negative_edges(self):
        edges = [(1, 2, 1), (2, 3, 1), (3, 4, 1)]
        expected = [(1, 2), (2, 3), (3, 4)]
        self.assertEqual(transform_edge_list(edges), expected)
        # Test the same input again
        self.assertEqual(transform_edge_list(edges), expected)

    def test_all_negative_edges(self):
        edges = [(1, 2, -1), (3, 4, -1), (5, 6, -1)]
        expected = [(2, 1), (4, 3), (6, 5)]
        self.assertEqual(transform_edge_list(edges), expected)
        # Test the same input again
        self.assertEqual(transform_edge_list(edges), expected)

    def test_mixed_edges(self):
        edges = [(1, 2, -1), (2, 3, 1), (3, 4, -1)]
        expected = [(2, 1), (2, 3), (4, 3)]
        self.assertEqual(transform_edge_list(edges), expected)
        # Test the same input again
        self.assertEqual(transform_edge_list(edges), expected)

    def test_empty_list(self):
        edges = []
        expected = []
        self.assertEqual(transform_edge_list(edges), expected)
        # Test the same input again
        self.assertEqual(transform_edge_list(edges), expected)

    def test_single_negative_edge(self):
        edges = [(1, 2, -1)]
        expected = [(2, 1)]
        self.assertEqual(transform_edge_list(edges), expected)
        # Test the same input again
        self.assertEqual(transform_edge_list(edges), expected)

    def test_single_positive_edge(self):
        edges = [(1, 2, 1)]
        expected = [(1, 2)]
        self.assertEqual(transform_edge_list(edges), expected)
        # Test the same input again
        self.assertEqual(transform_edge_list(edges), expected)

    def test_repeat_same_input_multiple_times(self):
        # Mixed edge list with both positive and negative weights
        edges = [(1, 2, -1), (2, 3, 1), (4, 5, -1), (6, 7, 1)]
        expected = [(2, 1), (2, 3), (5, 4), (6, 7)]

        # Test the function once
        self.assertEqual(transform_edge_list(edges), expected)

        # Test the same input again
        self.assertEqual(transform_edge_list(edges), expected)

        # Test the same input a third time
        self.assertEqual(transform_edge_list(edges), expected)

    def test_identical_edge_list(self):
        edges = [(1, 2, -1), (2, 3, 1), (4, 5, -1), (6, 7, 1)]
        e1 = transform_edge_list(edges)
        e2 = transform_edge_list(edges)

        self.assertEqual(e1, e2)



import unittest
import networkx as nx

class TestGetGraph(unittest.TestCase):

    def test_graph_equivalence(self):
        # Manually set the edge lists
        edges1 = [(1, 2, 1), (2, 3, -1), (3, 4, 1)]
        edges2 = [(1, 2, 1), (2, 3, -1), (3, 4, 1)]

        # Transform the edges
        transformed_edges1 = transform_edge_list(edges1)
        transformed_edges2 = transform_edge_list(edges2)

        # Check the transformed edges
        expected_edges = [(1, 2), (3, 2), (3, 4)]
        self.assertEqual(transformed_edges1, expected_edges, "Edges should be transformed correctly for edges1")
        self.assertEqual(transformed_edges2, expected_edges, "Edges should be transformed correctly for edges2")

        # Create the graphs
        G1 = nx.Graph()
        G2 = nx.Graph()
        G1.add_edges_from(transformed_edges1)
        G2.add_edges_from(transformed_edges2)

        # Assert the graphs are isomorphic (identical structures)
        self.assertTrue(nx.is_isomorphic(G1, G2), "Graphs should be identical for identical edges")

    def test_graph_different(self):
        # Manually set the edge lists for two different graphs
        edges1 = [(1, 2, 1), (2, 3, -1), (3, 4, 1)]
        edges2 = [(1, 3, 1), (3, 4, -1), (4, 5, 1)]

        # Transform the edges
        transformed_edges1 = transform_edge_list(edges1)
        transformed_edges2 = transform_edge_list(edges2)

        # Check the transformed edges
        expected_edges1 = [(1, 2), (3, 2), (3, 4)]
        expected_edges2 = [(1, 3), (4, 3), (4, 5)]
        self.assertEqual(transformed_edges1, expected_edges1, "Edges should be transformed correctly for edges1")
        self.assertEqual(transformed_edges2, expected_edges2, "Edges should be transformed correctly for edges2")

        # Create the graphs
        G1 = nx.Graph()
        G2 = nx.Graph()
        G1.add_edges_from(transformed_edges1)
        G2.add_edges_from(transformed_edges2)

        # Compare the edge sets directly instead of using is_isomorphic
        self.assertNotEqual(set(G1.edges()), set(G2.edges()),
                            "Graphs should have different edge sets for different inputs")

    def test_unweighted_graphs(self):
        # Manually set the edge lists
        edges1 = [(1, 2, 1), (2, 3, -1), (3, 4, 1)]
        edges2 = [(1, 2, 1), (2, 3, -1), (3, 4, 1)]

        # Transform the edges
        transformed_edges1 = transform_edge_list(edges1)
        transformed_edges2 = transform_edge_list(edges2)

        # Create the graphs
        G1 = nx.Graph()
        G2 = nx.Graph()
        G1.add_edges_from(transformed_edges1)
        G2.add_edges_from(transformed_edges2)

        # Check that the graphs are unweighted
        for u, v, data in G1.edges(data=True):
            self.assertNotIn('weight', data, "G1 should have unweighted edges")
        for u, v, data in G2.edges(data=True):
            self.assertNotIn('weight', data, "G2 should have unweighted edges")

    def test_single_edge_graph(self):
        # Manually set the edge list with a single edge
        edges1 = [(1, 2, 1)]
        edges2 = [(1, 2, 1)]

        # Transform the edges
        transformed_edges1 = transform_edge_list(edges1)
        transformed_edges2 = transform_edge_list(edges2)

        # Check the transformed edges
        expected_edges = [(1, 2)]
        self.assertEqual(transformed_edges1, expected_edges, "Single edge should not be transformed")
        self.assertEqual(transformed_edges2, expected_edges, "Single edge should not be transformed")

        # Create the graphs
        G1 = nx.Graph()
        G2 = nx.Graph()
        G1.add_edges_from(transformed_edges1)
        G2.add_edges_from(transformed_edges2)

        # Check that the graph has exactly one edge
        self.assertEqual(len(G1.edges()), 1, "G1 should have exactly one edge")
        self.assertEqual(len(G2.edges()), 1, "G2 should have exactly one edge")

    def test_empty_aig(self):
        # Manually set an empty edge list
        edges1 = []
        edges2 = []

        # Transform the edges
        transformed_edges1 = transform_edge_list(edges1)
        transformed_edges2 = transform_edge_list(edges2)

        # Check the transformed edges
        self.assertEqual(transformed_edges1, [], "Empty edge list should remain empty")
        self.assertEqual(transformed_edges2, [], "Empty edge list should remain empty")

        # Create the graphs
        G1 = nx.Graph()
        G2 = nx.Graph()
        G1.add_edges_from(transformed_edges1)
        G2.add_edges_from(transformed_edges2)

        # Check that the graphs are empty
        self.assertEqual(len(G1.nodes()), 0, "G1 should have no nodes")
        self.assertEqual(len(G2.nodes()), 0, "G2 should have no nodes")
        self.assertEqual(len(G1.edges()), 0, "G1 should have no edges")
        self.assertEqual(len(G2.edges()), 0, "G2 should have no edges")

    def test_inverted_edges(self):
        # Manually set the edge list with inverted edges (weight = -1)
        edges1 = [(1, 2, -1), (2, 3, -1), (3, 4, -1)]
        edges2 = [(1, 2, -1), (2, 3, -1), (3, 4, -1)]

        # Transform the edges (reverse edges with weight -1)
        transformed_edges1 = transform_edge_list(edges1)
        transformed_edges2 = transform_edge_list(edges2)

        # Check the transformed edges
        expected_edges = [(2, 1), (3, 2), (4, 3)]  # All edges should be reversed
        self.assertEqual(transformed_edges1, expected_edges, "Edges should be reversed for inverted weights")
        self.assertEqual(transformed_edges2, expected_edges, "Edges should be reversed for inverted weights")

        # Create the graphs
        G1 = nx.Graph()
        G2 = nx.Graph()
        G1.add_edges_from(transformed_edges1)
        G2.add_edges_from(transformed_edges2)

        # Check if graphs are isomorphic (should be the same after transformation)
        self.assertTrue(nx.is_isomorphic(G1, G2), "Graphs should be identical for identical inverted edges")


