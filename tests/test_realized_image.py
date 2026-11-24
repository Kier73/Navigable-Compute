"""NCR-008 image cardinality, compression, and native operator controls."""
from dataclasses import replace
import random
import unittest

from nonlinear_koopman_demo import nonlinear_family
from navigable_compute.nonlinear_koopman import construct_koopman_ground, eval_observers
from navigable_compute.nonlinear_koopman_verify import verify_polynomial_laws, verify_koopman_ground
from navigable_compute.realized_image import (
    construct_image, verify_image, pack, unpack, packed_step, ImageBudget,
)


class RealizedImageTests(unittest.TestCase):
    def test_independent_nonlinear_lanes(self):
        # x, y, xy are linearly independent functions but realize four values.
        basis = tuple(p for i in range(64) for p in (
            (1 << (2*i),), (1 << (2*i+1),), (3 << (2*i),)))
        cert = construct_image(basis, 128)
        self.assertTrue(verify_image(basis, 128, cert))
        self.assertEqual(cert.cardinality, 1 << 128)
        self.assertEqual(cert.bits, 128)
        self.assertEqual(len(basis), 192)
        self.assertEqual(cert.cells, 1536)
        for x in (0, 1, 3, (1 << 128)-1, 0xDEADBEEF):
            z = eval_observers(basis, x)
            self.assertEqual(unpack(cert, pack(cert, z)), z)
        with self.assertRaises(ValueError):
            pack(cert, 4)  # xy=1 but x=y=0
        with self.assertRaises(ValueError):
            unpack(cert, 1 << 128)

    def test_small_random_against_full_source_enumeration(self):
        rng = random.Random(801)
        for width in range(1, 7):
            for _ in range(25):
                basis = tuple(tuple(sorted(rng.sample(range(1 << width),
                              rng.randrange(min(5, 1 << width)+1))))
                              for _ in range(rng.randrange(7)))
                cert = construct_image(basis, width)
                self.assertTrue(verify_image(basis, width, cert))
                expected = {eval_observers(basis, x) for x in range(1 << width)}
                self.assertEqual(cert.cardinality, len(expected))
                self.assertEqual({unpack(cert, c) for c in range(cert.cardinality)}, expected)
                for z in expected:
                    self.assertEqual(unpack(cert, pack(cert, z)), z)

    def test_native_koopman_and_operator_commutation(self):
        family = nonlinear_family()
        self.assertTrue(verify_polynomial_laws(family))
        ground = construct_koopman_ground(family)
        self.assertTrue(verify_koopman_ground(family, ground))
        cert = construct_image(ground.basis, family.description.width)
        self.assertTrue(verify_image(ground.basis, family.description.width, cert))
        self.assertEqual(ground.dimension, 10)
        self.assertEqual(cert.cardinality, 192)
        self.assertEqual(cert.bits, 8)
        rng = random.Random(8)
        for _ in range(64):
            x = rng.getrandbits(128)
            code = pack(cert, ground.project(x))
            for action, op in enumerate(family.operators):
                after = packed_step(ground, cert, code, action)
                self.assertEqual(unpack(cert, after), ground.project(op.apply(x)))
        # Every packed coordinate must remain in the exact image under each action.
        self.assertLessEqual(cert.cardinality, 1 << 16)
        for code in range(cert.cardinality):
            for a in range(len(family.operators)):
                self.assertLess(packed_step(ground, cert, code, a), cert.cardinality)

    def test_original_fixture_independent_truth_table_oracle(self):
        # Relabel protected source bit 64 as local bit 8. The selected
        # observer cone is confined to lane zero and that protected bit.
        a = ((6,), (8,), (16,), (64,), (128,), (), (), (), (256,))
        b = ((2,), (16,), (8,), (128,), (64,), (192,), (), (), (256,))
        count = 512
        def apply(rows, x):
            return sum((sum((x & m) == m for m in p) % 2) << i
                       for i, p in enumerate(rows))
        actions = [[apply(rows, x) for x in range(count)] for rows in (a,b)]
        features, pivots = [], {}
        def insert(v):
            for p in sorted(pivots, reverse=True):
                if v >> p & 1:
                    v ^= pivots[p]
            if v:
                pivots[v.bit_length()-1] = v
                features.append(v)
        for bit in (0,8):
            insert(sum(((x >> bit) & 1) << x for x in range(count)))
        i = 0
        while i < len(features):
            for action in actions:
                insert(sum(((features[i] >> action[x]) & 1) << x
                           for x in range(count)))
            i += 1
        image = {sum(((f >> x) & 1) << j for j, f in enumerate(features))
                 for x in range(count)}
        self.assertEqual((len(features), len(image)), (10,192))

    def test_budget_and_forgery(self):
        basis = ((1,), (2,), (3,))
        cert = construct_image(basis, 2)
        self.assertTrue(verify_image(basis, 2, cert))
        for altered in (
            replace(cert, cells=cert.cells-1),
            replace(cert, factors=(replace(cert.factors[0], values=(0,1,2,3)),)),
            replace(cert, factors=()),
            replace(cert, factors=(replace(cert.factors[0], source_mask=1),)),
        ):
            self.assertFalse(verify_image(basis, 2, altered))
        with self.assertRaises(ImageBudget):
            construct_image(((2**128-1,),), 128)
        self.assertFalse(verify_image(basis, 2, cert, max_cells=1))

    def test_constant_and_empty_features(self):
        for basis, expected in (((), 0), (((), (0,)), 2)):
            cert = construct_image(basis, 0)
            self.assertTrue(verify_image(basis, 0, cert))
            self.assertEqual(cert.cardinality, 1)
            self.assertEqual(cert.bits, 0)
            self.assertEqual(unpack(cert, 0), expected)
