import unittest
from aigverse import Aig

from resub_metrics import absolute_resub_metric, relative_resub_metric


class TestResubMetrics(unittest.TestCase):
    def test_empty_aigs(self):
        # Create two empty AIGs
        aig1 = Aig()
        aig2 = Aig()

        # Ensure the absolute resubstitution metric is 0
        self.assertEqual(absolute_resub_metric(aig1, aig2), 0)

        # Ensure the relative resubstitution metric is 0
        self.assertEqual(relative_resub_metric(aig1, aig2), 0)

    def test_trivial_aigs(self):
        # Create a trivial AIG
        aig1 = Aig()
        a1 = aig1.create_pi()
        a2 = aig1.create_pi()
        a3 = aig1.create_and(a1, a2)
        aig1.create_po(a3)

        aig2 = aig1.clone()

        # Ensure the absolute resubstitution metric is 0
        self.assertEqual(absolute_resub_metric(aig1, aig2), 0)

        # Ensure the relative resubstitution metric is 0
        self.assertEqual(relative_resub_metric(aig1, aig2), 0)

    def test_simple_aigs(self):
        # x0 * !(!x0 * !x1) == > x0 (reduction of 2 nodes)
        aig1 = Aig()
        x0 = aig1.create_pi()
        x1 = aig1.create_pi()
        n0 = aig1.create_and(~x0, ~x1)
        n1 = aig1.create_and(x0, ~n0)
        aig1.create_po(n1)

        # x1 * ( x0 * x1 ) ==> x0 * x1 (reduction of 1 node)
        aig2 = Aig()
        x0 = aig2.create_pi()
        x1 = aig2.create_pi()
        n0 = aig2.create_and(x0, x1)
        n1 = aig2.create_and(x1, n0)
        aig2.create_po(n1)

        # Ensure the absolute resubstitution metric is 1 in both directions
        self.assertEqual(absolute_resub_metric(aig1, aig2), 1)
        self.assertEqual(absolute_resub_metric(aig2, aig1), 1)

        # Ensure the relative resubstitution metric is 0.5 in both directions
        self.assertAlmostEqual(relative_resub_metric(aig1, aig2), 0.5)
        self.assertAlmostEqual(relative_resub_metric(aig2, aig1), 0.5)

    def test_medium_simple_aigs(self):
        # !( x0 * !x2 ) * ( !x1 * !x2 ) ==> ( !x0 * !x1 ) * !x2 (reduction of 1 node)
        aig1 = Aig()
        x0 = aig1.create_pi()
        x1 = aig1.create_pi()
        x2 = aig1.create_pi()
        n0 = aig1.create_and(x0, ~x2)
        n1 = aig1.create_and(~x1, ~x2)
        n2 = aig1.create_and(~n0, n1)
        aig1.create_po(n2)

        # reduction of 1 node
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

        # Ensure the absolute resubstitution metric is 1 in both directions
        self.assertEqual(absolute_resub_metric(aig1, aig2), 0)
        self.assertEqual(absolute_resub_metric(aig2, aig1), 0)

        # Ensure the relative resubstitution metric is 0.20833 in both directions
        self.assertAlmostEqual(relative_resub_metric(aig1, aig2), 0.20833, places=3)
        self.assertAlmostEqual(relative_resub_metric(aig2, aig1), 0.20833, places=3)


if __name__ == '__main__':
    unittest.main()
