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
        self.assertLessEqual(cert.bits, ground.dimension)
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
