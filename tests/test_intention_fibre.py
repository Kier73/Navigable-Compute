"""NCR-010: exhaustive small oracles, coupled large grounds and tampering."""
import random
import unittest
from dataclasses import replace

from navigable_compute.intention_fibre import (
    FibreBudget, Goal, construct_fibre, construct_temporal_fibre)
from navigable_compute.intention_fibre_verify import verify_fibre, verify_temporal_fibre
from nonlinear_koopman_demo import nonlinear_family


def direct(rows, state):
    return all(sum((state & m) == m for m in p) % 2 == 0 for p in rows)


class FibreTests(unittest.TestCase):
    def test_random_exhaustive_rank_and_count(self):
        rng = random.Random(1010)
        for _ in range(100):
            width = rng.randint(1, 7)
            rows = tuple(tuple(sorted(rng.sample(range(2**width),
                                  rng.randint(0, min(8, 2**width)))))
                         for _ in range(rng.randint(0, 5)))
            c = construct_fibre(width, rows)
            expected = [x for x in range(2**width) if direct(rows, x)]
            self.assertTrue(verify_fibre(width, rows, c))
            self.assertEqual(c.count, len(expected))
            self.assertEqual([c.unrank(i) for i in range(c.count)], expected)
            self.assertEqual([c.rank(x) for x in expected], list(range(c.count)))
            for x in set(range(2**width)) - set(expected):
                with self.assertRaises(ValueError):
                    c.rank(x)

    def test_connected_128_bit_path(self):
        # One connected nonlinear incidence component: adjacent ones forbidden.
        width = 128
        rows = tuple(((3 << i),) for i in range(width - 1))
        c = construct_fibre(width, rows)
        a, b = 0, 1
        for _ in range(width + 2):
            a, b = b, a + b
        self.assertEqual(c.count, a)
        self.assertLess(len(c.nodes), 3 * width)
        self.assertTrue(verify_fibre(width, rows, c))
        for rank in (0, 1, 10**20, c.count // 2, c.count - 1):
            x = c.unrank(rank)
            self.assertEqual(x & (x >> 1), 0)
            self.assertEqual(c.rank(x), rank)

    def test_free_unsat_and_budgets(self):
        c = construct_fibre(128, ())
        self.assertEqual(c.count, 2**128)
        self.assertEqual(c.unrank(2**128 - 1), 2**128 - 1)
        self.assertTrue(verify_fibre(128, (), c))
        c = construct_fibre(3, ((0,),))
        self.assertEqual(c.count, 0)
        self.assertTrue(verify_fibre(3, ((0,),), c))
        with self.assertRaises(ValueError):
            c.unrank(0)
        with self.assertRaises(FibreBudget):
            construct_fibre(128, (), max_nodes=10)
        with self.assertRaises(FibreBudget):
            construct_fibre(128, (), max_work=10)

    def test_forged_count_edge_and_equation(self):
        rows = ((3,),)
        c = construct_fibre(3, rows)
        for n in (replace(c.nodes[-1], count=c.count + 1),
                  replace(c.nodes[-1], high=0),
                  replace(c.nodes[-1], equations=())):
            self.assertFalse(verify_fibre(3, rows, replace(c, nodes=c.nodes[:-1] + (n,))))
        self.assertFalse(verify_fibre(3, rows, replace(c, work=c.work + 1)))
        self.assertFalse(verify_fibre(3, ((1,),), c))

    def test_native_temporal_inverse(self):
        f = nonlinear_family()
        goals = (Goal((0,), 1, 3, 3), Goal((0,), 10**100+173, 3, 2))
        c = construct_temporal_fibre(f, goals)
        self.assertTrue(verify_temporal_fibre(f, goals, c))
        self.assertEqual(c.fibre.count, 2**125)
        for rank in (0, 10**30, c.fibre.count - 1):
            x = c.fibre.unrank(rank)
            self.assertEqual(c.fibre.rank(x), rank)
            self.assertEqual(f.project_observer(f.operators[0].apply(x)), 3)
            y = x
            for _ in range(f.description.drain_horizon):
                y = f.operators[0].apply(y)
            self.assertEqual(f.project_observer(y), 2)
        self.assertFalse(verify_temporal_fibre(f, goals, replace(c, source_digest="0"*64)))
        conflict = goals + (Goal((0,), 1, 1, 0),)
        z = construct_temporal_fibre(f, conflict)
        self.assertEqual(z.fibre.count, 0)
        self.assertTrue(verify_temporal_fibre(f, conflict, z))


if __name__ == "__main__":
    unittest.main()
