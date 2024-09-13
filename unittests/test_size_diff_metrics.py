import unittest
from aigverse import Aig

from size_diff_metrics import absolute_size_diff_metric, relative_size_diff_metric


class TestSizeDiffMetrics(unittest.TestCase):
    def test_empty_aigs(self):
        # Create two empty AIGs
        aig1 = Aig()
        aig2 = Aig()

        # Ensure the absolute size diff metric is 0
        self.assertEqual(absolute_size_diff_metric(aig1, aig2), 0)

        # Ensure the relative size diff metric is 0
        self.assertEqual(relative_size_diff_metric(aig1, aig2), 0)

    def test_trivial_aigs(self):
        # Create a trivial AIG
        aig1 = Aig()
        a1 = aig1.create_pi()
        a2 = aig1.create_pi()
        a3 = aig1.create_and(a1, a2)
        aig1.create_po(a3)

        aig2 = aig1.clone()

        # Ensure the absolute size diff metric is 0
        self.assertEqual(absolute_size_diff_metric(aig1, aig2), 0)

        # Ensure the relative size diff metric is 0
        self.assertEqual(relative_size_diff_metric(aig1, aig2), 0)

    def test_simple_aigs(self):
        aig1 = Aig()
        x0 = aig1.create_pi()
        x1 = aig1.create_pi()
        n0 = aig1.create_and(~x0, ~x1)
        n1 = aig1.create_and(x0, ~n0)
        aig1.create_po(n1)

        aig2 = Aig()
        x0 = aig2.create_pi()
        x1 = aig2.create_pi()
        n0 = aig2.create_and(x0, x1)
        n1 = aig2.create_and(x1, n0)
        aig2.create_po(n1)

        # Ensure the absolute size diff metric is 0 in both directions
        self.assertEqual(absolute_size_diff_metric(aig1, aig2), 0)
        self.assertEqual(absolute_size_diff_metric(aig2, aig1), 0)

        # Ensure the relative size diff metric is 0.0 in both directions
        self.assertAlmostEqual(relative_size_diff_metric(aig1, aig2), 0.0)
        self.assertAlmostEqual(relative_size_diff_metric(aig2, aig1), 0.0)

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

        # Ensure the absolute size diff metric is 5 in both directions
        self.assertEqual(absolute_size_diff_metric(aig1, aig2), 5)
        self.assertEqual(absolute_size_diff_metric(aig2, aig1), 5)

        # Ensure the relative size diff metric is 1 in both directions
        self.assertAlmostEqual(relative_size_diff_metric(aig1, aig2), 0.625, places=3)
        self.assertAlmostEqual(relative_size_diff_metric(aig2, aig1), 0.625, places=3)


def test_incremental_aigs(self):
    aig1 = Aig()
    aig2 = Aig()

    # Add PI and PO pairs incrementally to aig1
    for i in range(5):
        pi = aig1.create_pi()
        aig1.create_po(pi)

        # For aig2, add pairs twice as fast
        for j in range(2):
            pi = aig2.create_pi()
            aig2.create_po(pi)

        # Ensure the absolute size diff metric is correct
        if i == 0:
            self.assertEqual(absolute_size_diff_metric(aig1, aig2), 0)
        else:
            self.assertEqual(absolute_size_diff_metric(aig1, aig2), abs((i + 1) - 2 * (i + 1)))

        # Ensure the relative size diff metric is correct
        if i == 0:
            self.assertEqual(relative_size_diff_metric(aig1, aig2), 0.0)
        else:
            self.assertEqual(relative_size_diff_metric(aig1, aig2),
                             abs((i + 1) - 2 * (i + 1)) / max(i + 1, 2 * (i + 1)))


if __name__ == '__main__':
    unittest.main()
