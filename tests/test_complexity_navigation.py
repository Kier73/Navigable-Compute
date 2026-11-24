"""NCR-009 exact portfolio, observation support, continuation and cost controls."""
from dataclasses import replace
import random
import unittest

from nonlinear_koopman_demo import nonlinear_family
from navigable_compute.complexity_navigation import Portfolio, Profile, choose
from navigable_compute.nonlinear_koopman import eval_observers


class ComplexityNavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.family = nonlinear_family()

    def test_native_all_modes_and_continuation(self):
        rng = random.Random(901)
        for word, repetitions in (((), 10**100), ((0,), 0), ((0,), 1),
                                   ((0,1), 1), ((1,0), 10**100+173)):
            p = Portfolio(self.family, word, repetitions)
            self.assertEqual(set(p.candidates), {"source", "terminal", "feature", "packed"})
            self.assertEqual(p.candidates["feature"].retained_bits, 10)
            self.assertEqual(p.candidates["packed"].retained_bits, 8)
            for _ in range(8):
                x = rng.getrandbits(128)
                y, full = p.realize("source", x)
                for mode in p.candidates:
                    got, z = p.realize(mode, x)
                    self.assertEqual(got, y)
                    if mode != "terminal":
                        ref = full
                        for a in (1,0,1):
                            got, z = p.continue_from(mode, z, a)
                            ref = self.family.operators[a].apply(ref)
                            self.assertEqual(got, self.family.project_observer(ref))
                    else:
                        with self.assertRaises(ValueError):
                            p.continue_from(mode, z, 0)

    def test_terminal_support_is_essential_and_sufficient(self):
        p = Portfolio(self.family, (0,), 1)
        c = p.candidates["terminal"]
        self.assertEqual(c.read_mask, (1<<1)|(1<<2)|(1<<64))
        self.assertEqual(c.read_mask.bit_count(), 3)
        variables = (1,2,64)
        states = [sum(((v>>i)&1)<<b for i,b in enumerate(variables)) for v in range(8)]
        for b in variables:
            self.assertTrue(any(eval_observers(c.polys, x) !=
                                eval_observers(c.polys, x^(1<<b)) for x in states))
        for x in states:
            self.assertEqual(p.realize("terminal", x),
                             p.realize("terminal", x|(((1<<128)-1)^c.read_mask)))
        far = Portfolio(self.family, (0,1), 10**100+173)
        self.assertEqual(len(far.prefix), 4)
        self.assertEqual(far.candidates["terminal"].read_mask, 1<<64)
        self.assertEqual(far.candidates["feature"].read_mask, 1<<64)
        self.assertEqual(far.candidates["source"].read_mask.bit_count(), 64)

    def test_cost_policy_exact_counterfactual_and_eligibility(self):
        p = Portfolio(self.family, (0,), 1)
        # Deliberate deterministic cost profile, distinct from observed timings.
        profiles = tuple(Profile(replace(p.candidates[m], setup_ns=s), w, 0)
                         for m,s,w in (("source",0,10),("terminal",100,1),
                                       ("feature",500,2),("packed",700,4)))
        self.assertEqual(choose(profiles, 1, cold=True).candidate.mode, "source")
        self.assertEqual(choose(profiles, 100, cold=True).candidate.mode, "terminal")
        self.assertEqual(choose(profiles, 1, continuation=True).candidate.mode, "feature")
        self.assertEqual(choose(profiles, 1, continuation=True, max_bits=8).candidate.mode, "packed")
        self.assertEqual(choose(profiles, 1, continuation=True, bit_price=2).candidate.mode, "packed")
        with self.assertRaises(ValueError):
            choose(profiles, 1, continuation=True, max_bits=7)
        with self.assertRaises(ValueError):
            choose(profiles, 0)
        self.assertGreaterEqual(p.preparation_ns, p.source_qualification_ns)

    def test_real_calibration_and_invalid_requests(self):
        p = Portfolio(self.family, (0,), 2)
        profiles = p.calibrate((0, 1, 3, 1<<64))
        self.assertTrue(all(x.warm_ns >= 0 and x.calibration_ns >= x.warm_ns for x in profiles))
        selected = choose(profiles, 4, continuation=True)
        self.assertIn(selected.candidate.mode, ("source","feature","packed"))
        self.assertEqual(p.realize(selected.candidate.mode, 0xDEADBEEF)[0],
                         p.realize("source", 0xDEADBEEF)[0])
        with self.assertRaises(ValueError):
            p.calibrate(())
        with self.assertRaises(ValueError):
            p.realize("source", -1)
        with self.assertRaises(ValueError):
            Portfolio(self.family, (99,), 1)
        with self.assertRaises(ValueError):
            Portfolio(self.family, (), -1)
