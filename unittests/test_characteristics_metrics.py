import unittest
from aigverse import Aig

from sim_scores.characteristics_metrics import absolute_gate_count_metric, relative_gate_count_metric, absolute_edge_count_metric, \
    relative_edge_count_metric, absolute_level_count_metric, relative_level_count_metric, \
    normalized_euclidean_similarity_metric, \
    normalized_cosine_similarity_score


class TestCharacteristicsMetrics(unittest.TestCase):
    def test_empty_aigs(self):
        # Create two empty AIGs
        aig1 = Aig()
        aig2 = Aig()

        # Ensure the absolute gate count metric is 0 in both directions
        self.assertEqual(absolute_gate_count_metric(aig1, aig2), 0)
        self.assertEqual(absolute_gate_count_metric(aig2, aig1), 0)

        # Ensure the relative gate count metric is 0.0 in both directions
        self.assertEqual(relative_gate_count_metric(aig1, aig2), 0.0)
        self.assertEqual(relative_gate_count_metric(aig2, aig1), 0.0)

        # Ensure the absolute edge count metric is 0 in both directions
        self.assertEqual(absolute_edge_count_metric(aig1, aig2), 0)
        self.assertEqual(absolute_edge_count_metric(aig2, aig1), 0)

        # Ensure the relative edge count metric is 0.0 in both directions
        self.assertEqual(relative_edge_count_metric(aig1, aig2), 0.0)
        self.assertEqual(relative_edge_count_metric(aig2, aig1), 0.0)

        # Ensure the absolute level count metric is 0 in both directions
        self.assertEqual(absolute_level_count_metric(aig1, aig2), 0)
        self.assertEqual(absolute_level_count_metric(aig2, aig1), 0)

        # Ensure the relative level count metric is 0.0 in both directions
        self.assertEqual(relative_level_count_metric(aig1, aig2), 0.0)
        self.assertEqual(relative_level_count_metric(aig2, aig1), 0.0)

        # Ensure the normalized Euclidean similarity metric is 1.0 in both directions
        self.assertEqual(normalized_euclidean_similarity_metric(aig1, aig2), 1.0)
        self.assertEqual(normalized_euclidean_similarity_metric(aig2, aig1), 1.0)

        # Ensure the normalized cosine similarity score is 0.0 in both directions
        self.assertEqual(normalized_cosine_similarity_score(aig1, aig2), 0.0)
        self.assertEqual(normalized_cosine_similarity_score(aig2, aig1), 0.0)

    def test_trivial_aigs(self):
        # Create a trivial AIG
        aig1 = Aig()
        a1 = aig1.create_pi()
        a2 = aig1.create_pi()
        a3 = aig1.create_and(a1, a2)
        aig1.create_po(a3)

        aig2 = aig1.clone()

        # Ensure the absolute gate count metric is 0 in both directions
        self.assertEqual(absolute_gate_count_metric(aig1, aig2), 0)
        self.assertEqual(absolute_gate_count_metric(aig2, aig1), 0)

        # Ensure the relative gate count metric is 0.0 in both directions
        self.assertEqual(relative_gate_count_metric(aig1, aig2), 0.0)
        self.assertEqual(relative_gate_count_metric(aig2, aig1), 0.0)

        # Ensure the absolute edge count metric is 0 in both directions
        self.assertEqual(absolute_edge_count_metric(aig1, aig2), 0)
        self.assertEqual(absolute_edge_count_metric(aig2, aig1), 0)

        # Ensure the relative edge count metric is 0.0 in both directions
        self.assertEqual(relative_edge_count_metric(aig1, aig2), 0.0)
        self.assertEqual(relative_edge_count_metric(aig2, aig1), 0.0)

        # Ensure the absolute level count metric is 0 in both directions
        self.assertEqual(absolute_level_count_metric(aig1, aig2), 0)
        self.assertEqual(absolute_level_count_metric(aig2, aig1), 0)

        # Ensure the relative level count metric is 0.0 in both directions
        self.assertEqual(relative_level_count_metric(aig1, aig2), 0.0)
        self.assertEqual(relative_level_count_metric(aig2, aig1), 0.0)

        # Ensure the normalized Euclidean similarity metric is 1.0 in both directions
        self.assertEqual(normalized_euclidean_similarity_metric(aig1, aig2), 1.0)
        self.assertEqual(normalized_euclidean_similarity_metric(aig2, aig1), 1.0)

        # Ensure the normalized cosine similarity score is 1.0 in both directions
        self.assertAlmostEqual(normalized_cosine_similarity_score(aig1, aig2), 1.0, places=5)
        self.assertAlmostEqual(normalized_cosine_similarity_score(aig2, aig1), 1.0, places=5)

    def test_medium_simple_aigs(self):
        aig1 = Aig()
        x0 = aig1.create_pi()
        x1 = aig1.create_pi()
        x2 = aig1.create_pi()
        n0 = aig1.create_and(x0, ~x2)
        n1 = aig1.create_and(~x1, ~x2)
        n2 = aig1.create_and(~n0, n1)
        aig1.create_po(n2)

        aig2 = Aig()
        x0 = aig2.create_pi()
        x1 = aig2.create_pi()
        x2 = aig2.create_pi()
        x3 = aig2.create_pi()
        n0 = aig2.create_and(~x2, x3)
        n1 = aig2.create_and(~x2, n0)
        n2 = aig2.create_and(x3, ~n1)
        n3 = aig2.create_and(x0, ~x1)
        n4 = aig2.create_and(~n2, n3)
        n5 = aig2.create_and(x1, ~n2)
        n6 = aig2.create_and(~n4, ~n5)
        n7 = aig2.create_and(n1, n3)
        aig2.create_po(n6)
        aig2.create_po(n7)

        # Ensure the absolute gate count metric is 5 in both directions
        self.assertEqual(absolute_gate_count_metric(aig1, aig2), 5)
        self.assertEqual(absolute_gate_count_metric(aig2, aig1), 5)

        # Ensure the relative gate count metric is 0.454545 in both directions
        self.assertAlmostEqual(relative_gate_count_metric(aig1, aig2), 0.454545, places=6)
        self.assertAlmostEqual(relative_gate_count_metric(aig2, aig1), 0.454545, places=6)

        # Ensure the absolute edge count metric is 10 in both directions
        self.assertEqual(absolute_edge_count_metric(aig1, aig2), 10)
        self.assertEqual(absolute_edge_count_metric(aig2, aig1), 10)

        # Ensure the relative edge count metric is 0.454545 in both directions
        self.assertAlmostEqual(relative_edge_count_metric(aig1, aig2), 0.454545, places=6)
        self.assertAlmostEqual(relative_edge_count_metric(aig2, aig1), 0.454545, places=6)

        # Ensure the absolute level count metric is 3 in both directions
        self.assertEqual(absolute_level_count_metric(aig1, aig2), 3)
        self.assertEqual(absolute_level_count_metric(aig2, aig1), 3)

        # Ensure the relative level count metric is 0.42857 in both directions
        self.assertAlmostEqual(relative_level_count_metric(aig1, aig2), 0.42857, places=5)
        self.assertAlmostEqual(relative_level_count_metric(aig2, aig1), 0.42857, places=5)

        # Ensure the normalized Euclidean similarity metric is 0.38322 in both directions
        self.assertAlmostEqual(normalized_euclidean_similarity_metric(aig1, aig2), 0.38322, places=5)
        self.assertAlmostEqual(normalized_euclidean_similarity_metric(aig2, aig1), 0.38322, places=5)

        # Ensure the normalized cosine similarity score is 0.999852
        self.assertAlmostEqual(normalized_cosine_similarity_score(aig1, aig2), 0.999852, places=6)
        self.assertAlmostEqual(normalized_cosine_similarity_score(aig2, aig1), 0.999852, places=6)


if __name__ == '__main__':
    unittest.main()
