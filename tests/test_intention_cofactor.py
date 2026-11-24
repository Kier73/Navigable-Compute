"""Direct signed-cube intentions, including exponential vocabulary avoidance."""
from dataclasses import replace
import random
import unittest
from navigable_compute.intention_fibre import FibreBudget,construct_fibre
from navigable_compute.fibre_refinement import refine,observation_rows
from navigable_compute.fibre_refinement_verify import verify_refinement
from navigable_compute.procedural_vocabulary import Vocabulary
from navigable_compute.intention_cofactor import prove_cube
from navigable_compute.intention_cofactor_verify import verify_cube
from navigable_compute.induced_transversal import activate
from test_induced_transversal import make,trained


def cold_count(v,ones,zeros):
    rows=v.observations+observation_rows(v.parent.width,ones|zeros,ones)
    f=refine(v.parent,rows)
    if not verify_refinement(v.parent,rows,f):
        raise AssertionError("cold proof")
    return f.count


class CubeTests(unittest.TestCase):
    def test_exponential_vocabulary_never_formed(self):
        skill=trained()
        source=(0,)+tuple(3<<(2*i) for i in range(48))
        v=make(source)
        self.assertEqual(v.overlay.count,((4**48-2**48)//2)*2**32)
        self.assertEqual(v.proposals(),())
        with self.assertRaises(FibreBudget):
            activate(skill,v,0,max_arity=11)
        self.assertEqual(v.learned,())
        zeros=sum(1<<(2*i) for i in range(48))
        # Disable the ordinary target refinement method. This signed factored
        # intention cannot even be encoded as a bounded expanded ANF.
        exact=Vocabulary.exact
        def forbidden(*a,**k):
            raise AssertionError("expanded cold target attempted")
        try:
            p=prove_cube(skill,v,1<<127,zeros)
        finally:
            Vocabulary.exact=exact
        self.assertIsNotNone(p)
        self.assertTrue(verify_cube(v,1<<127,zeros,p))
        self.assertEqual(p.inspected_terms,49)
        self.assertLess(len(p.packet()),400)
        self.assertEqual(cold_count(v,1<<127,zeros),0)
        self.assertEqual(v.learned,())
        partial=zeros^(1<<94)
        self.assertIsNone(prove_cube(skill,v,1<<127,partial))
        self.assertGreater(cold_count(v,1<<127,partial),0)
        self.assertFalse(verify_cube(v,1<<127,partial,p))
        self.assertFalse(verify_cube(v,1<<127,zeros,replace(p,rule_digest="0"*64)))
        other=make(source,context="other")
        self.assertFalse(verify_cube(other,1<<127,zeros,p))
        with self.assertRaises(FibreBudget):
            prove_cube(skill,v,0,zeros,max_work=48)

    def test_signed_generalization(self):
        skill=trained()
        # xy XOR z=0. Setting x=y=1, z=0 is impossible even though
        # the source has no constant term and no positive-transversal rule.
        v=make((3,4),width=5)
        p=prove_cube(skill,v,3,4)
        self.assertTrue(verify_cube(v,3,4,p))
        self.assertEqual(cold_count(v,3,4),0)
        self.assertIsNone(prove_cube(skill,v,3,0))
        self.assertGreater(cold_count(v,3,0),0)
        with self.assertRaises(ValueError):
            prove_cube(skill,v,3,1)

    def test_exhaustive_oracle_for_signed_cubes(self):
        skill=trained()
        rng=random.Random(18018)
        for _ in range(160):
            n=rng.randint(2,8)
            r=tuple(sorted(rng.sample(range(2**n),rng.randint(1,min(10,2**n)))))
            xs=[x for x in range(2**n) if sum((x&m)==m for m in r)%2==0]
            if not xs:
                continue
            v=make(r,width=n)
            ones=zeros=0
            for i in range(n):
                trit=rng.randrange(3)
                if trit==1: ones|=1<<i
                if trit==2: zeros|=1<<i
            expected=[x for x in xs if x&ones==ones and not x&zeros]
            p=prove_cube(skill,v,ones,zeros)
            self.assertEqual(p is not None,not expected)
            if p is not None:
                self.assertTrue(verify_cube(v,ones,zeros,p))
            self.assertEqual(cold_count(v,ones,zeros),len(expected))

    def test_joint_entailment_unknown(self):
        # Neither single equation excludes the cube, but together they do.
        skill=trained()
        f=construct_fibre(4,())
        obs=((1,2),(2,4))
        v=Vocabulary(f,obs,refine(f,obs),context="joint")
        self.assertIsNone(prove_cube(skill,v,1,4))
        self.assertEqual(cold_count(v,1,4),0)


if __name__=="__main__":
    unittest.main()
