"""Exact coordinate transport and persistent intersection controls."""
import random
import unittest
from dataclasses import replace

from navigable_compute.intention_fibre import construct_fibre, FibreBudget
from navigable_compute.fibre_order import (
    choose_ordered, construct_ordered, verify_ordered, frontier_order, rename_rows)
from navigable_compute.fibre_refinement import refine, observation_rows
from navigable_compute.fibre_refinement_verify import verify_refinement


def valid(rows, x):
    return all(sum((x & m) == m for m in p) % 2 == 0 for p in rows)


def shuffled_path(width):
    # Source labels are split into even path positions and then odd positions.
    order = tuple(range(width // 2)) + tuple(range(width // 2, width))
    path = tuple(v for pair in zip(order[:width//2], order[width//2:]) for v in pair)
    return tuple(( (1 << a) | (1 << b), ) for a,b in zip(path, path[1:]))


class AdaptiveFibreTests(unittest.TestCase):
    def test_shuffled_path_budget(self):
        rows = shuffled_path(128)
        with self.assertRaises(FibreBudget):
            construct_fibre(128, rows, max_nodes=400, max_work=200_000)
        c, attempts, plan_ns = choose_ordered(128, rows, max_nodes=400, max_work=200_000)
        self.assertTrue(verify_ordered(128, rows, c))
        self.assertEqual(len(c.fibre.nodes), 255)
        self.assertEqual(c.count, 659034621587630041982498215)
        self.assertEqual({a.status for a in attempts}, {"QUALIFIED", "UNKNOWN"})
        self.assertGreaterEqual(plan_ns, 0)
        for i in (0, 10**20, c.count-1):
            x = c.unrank(i)
            self.assertTrue(valid(rows, x))
            self.assertEqual(c.rank(x), i)

    def test_random_order_and_intersection_oracles(self):
        rng = random.Random(1101)
        for _ in range(120):
            n = rng.randint(1, 7)
            def rows():
                return tuple(tuple(sorted(rng.sample(range(2**n), rng.randint(0, min(5, 2**n)))))
                             for _ in range(rng.randint(0, 4)))
            old, extra = rows(), rows()
            order = list(range(n))
            rng.shuffle(order)
            c = construct_ordered(n, old, tuple(order))
            self.assertTrue(verify_ordered(n, old, c))
            expected = {x for x in range(2**n) if valid(old, x)}
            self.assertEqual({c.unrank(i) for i in range(c.count)}, expected)
            new_rows = rename_rows(extra, c.order)
            f = refine(c.fibre, new_rows)
            self.assertTrue(verify_refinement(c.fibre, new_rows, f))
            addresses = [i for i in range(c.count) if valid(extra, c.unrank(i))]
            self.assertEqual(f.count, len(addresses))
            self.assertEqual([f.parent_address(i) for i in range(f.count)], addresses)
            for i, j in enumerate(addresses):
                self.assertEqual(f.refined_address(j), i)
                self.assertEqual(f.rank(f.unrank(i)), i)
                self.assertEqual(f.unrank(i), c.fibre.unrank(j))
            for j in set(range(c.count))-set(addresses):
                with self.assertRaises(ValueError):
                    f.refined_address(j)

    def test_reuse_and_refusal(self):
        c = construct_fibre(128, tuple((3 << i,) for i in range(127)))
        same = refine(c, (), max_nodes=0, max_work=0)
        self.assertEqual(same.root, -c.root)
        self.assertEqual(same.count, c.count)
        self.assertTrue(verify_refinement(c, (), same))
        rows = observation_rows(128, 1 << 127, 1 << 127)
        f = refine(c, rows, max_nodes=1)
        self.assertEqual(len(f.nodes), 1)
        self.assertEqual(f.count, c.count_at(c.nodes[c.root-2].high))
        self.assertTrue(verify_refinement(c, rows, f))
        self.assertTrue(f.unrank(f.count-1) >> 127)
        with self.assertRaises(FibreBudget):
            refine(c, rows, max_nodes=0)
        z = refine(c, ((0,),))
        self.assertEqual(z.count, 0)
        self.assertTrue(verify_refinement(c, ((0,),), z))
        self.assertEqual(c.count, 659034621587630041982498215)

    def test_tamper(self):
        old = ((3,),)
        c = construct_ordered(4, old, (2, 0, 1, 3))
        self.assertFalse(verify_ordered(4, old, replace(c, order=(0, 1, 2, 3))))
        self.assertFalse(verify_ordered(4, old, replace(c, order=(0, 0, 2, 3))))
        rows = observation_rows(4, 8, 8)
        f = refine(c.fibre, rows)
        n = f.nodes[-1]
        for bad in (replace(n, count=n.count + 1), replace(n, high=-1),
                    replace(n, parent_id=1), replace(n, equations=())):
            self.assertFalse(verify_refinement(c.fibre, rows, replace(f, nodes=f.nodes[:-1]+(bad,))))
        self.assertFalse(verify_refinement(c.fibre, rows, replace(f, work=f.work+1)))
        self.assertFalse(verify_refinement(c.fibre, (), f))
        other = construct_fibre(4, ())
        self.assertFalse(verify_refinement(other, rows, f))


if __name__ == "__main__":
    unittest.main()
