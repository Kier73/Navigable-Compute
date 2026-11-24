"""NCR-019 portfolio: composition, witnesses, independent exhaustive controls."""
import random
import unittest
from dataclasses import replace
from navigable_compute.intention_fibre import construct_fibre
from navigable_compute.fibre_refinement import refine,observation_rows
from navigable_compute.procedural_vocabulary import Vocabulary
from navigable_compute.intention_interpreter import (
    Interpreter,Truth,Cube,Limits,SpanProof,combine_cofactors)
from navigable_compute.intention_interpreter_verify import verify_result
from test_induced_transversal import trained


def make(rows,width=128,context="portfolio"):
    f=construct_fibre(width,())
    return Vocabulary(f,rows,refine(f,rows),context=context)


def bit(p,x):
    return sum((m&x)==m for m in p)%2


class InterpreterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skill=trained()

    def test_direct_and_joint_exclusion(self):
        v=make(((1,2),(2,4)))
        i=Interpreter(v,self.skill)
        q=Cube(1,2)
        r=i.realize(q)
        self.assertEqual((r.status,r.mechanism),("EXCLUDED","cofactor"))
        self.assertTrue(verify_result(v,q,r))
        q=Cube(1,4)
        r=i.realize(q,limits=Limits(residual_nodes=0,residual_work=0))
        self.assertEqual((r.status,r.mechanism),("EXCLUDED","cofactor-span"))
        self.assertEqual(r.certificate.indices,(0,1))
        self.assertTrue(verify_result(v,q,r))
        self.assertFalse(verify_result(v,q,replace(r,certificate=SpanProof((0,)))))
        self.assertFalse(verify_result(v,Cube(1,2),r))
        self.assertFalse(verify_result(make(v.observations,context="other"),q,r))

    def test_rewrite_composition_and_navigation(self):
        v=make(((1,2),(0,12,20,24)))
        i=Interpreter(v,self.skill)
        k=1<<120
        q=Truth((4,8,12,k|1,k|2))
        r=i.realize(q)
        self.assertEqual((r.status,r.mechanism),("ONE","rewrite+residual"))
        self.assertEqual(r.certificate.rewrite.residual,(4,8,12))
        self.assertTrue(verify_result(v,q,r))
        self.assertLess(r.certificate.refinement.work,v.exact(q.target)[1].work)
        self.assertFalse(verify_result(v,q,replace(r,status="ZERO")))
        q=Truth((k|1,k|2))
        self.assertEqual(i.realize(q).mechanism,"rewrite")
        q=Truth((1<<127,))
        r=i.realize(q)
        self.assertEqual((r.status,r.mechanism),("MIXED","address-witness"))
        self.assertFalse(verify_result(v,q,replace(r,certificate=replace(
            r.certificate,addresses=tuple(reversed(r.certificate.addresses))))))
        q=Cube(1<<127,0)
        self.assertEqual(i.realize(q).mechanism,"address-witness")

    def test_exact_fallback_and_budget(self):
        v=make((),width=4)
        i=Interpreter(v,self.skill)
        q=Cube(2,1)
        r=i.realize(q,limits=Limits(witness_samples=0))
        self.assertEqual((r.status,r.mechanism),("POSSIBLE","residual"))
        self.assertTrue(verify_result(v,q,r))
        r=i.realize(q,limits=Limits(witness_samples=0,residual_nodes=0,residual_work=0))
        self.assertEqual(r.status,"UNKNOWN")
        self.assertFalse(verify_result(v,q,r))
        self.assertEqual(r.attempts[-1].outcome,"BUDGET")
        q=Truth((1,2))
        r=i.realize(q,limits=Limits(witness_samples=0))
        self.assertEqual((r.status,r.mechanism),("MIXED","residual"))
        with self.assertRaises(ValueError):
            i.realize(Cube(1,1))
        # Nonlinear joint constraint requires more than an XOR span.
        v=make(((1,2),(3,4)),width=4)
        i=Interpreter(v,self.skill)
        q=Cube(1,4)
        # x=1 forces y=1; xy=z gives z=1; here linear combination
        # happens to suffice after substitution. Use a different cube instead:
        # x=y and xy=0 forces x=y=0 but with no selected coords the cube is possible.
        # A genuinely unresolved span with x=y, yz=0, x=z=1 becomes
        # y+1 and y, so choose chain requiring multiplication:
        # x=y, y*z=1 forces both y,z=1; x=0 restricts x=y to y,
        # while yz+1 still contains z. No linear combination is constant.
        v=make(((1,2),(0,6)),width=4)
        i=Interpreter(v,self.skill)
        q=Cube(0,1)
        self.assertIsNone(combine_cofactors(v,q))
        r=i.realize(q)
        self.assertEqual((r.status,r.mechanism),("EXCLUDED","residual"))

    def test_small_oracle(self):
        rng=random.Random(19019)
        for case in range(120):
            width=rng.randint(2,7)
            rows=tuple(tuple(sorted(rng.sample(range(2**width),rng.randint(0,min(6,2**width)))))
                       for _ in range(rng.randint(0,4)))
            xs=[x for x in range(2**width) if not any(bit(p,x) for p in rows)]
            if not xs:
                continue
            v=make(rows,width,context=str(case))
            i=Interpreter(v,self.skill)
            q=Truth(tuple(sorted(rng.sample(range(2**width),rng.randint(0,min(8,2**width))))))
            r=i.realize(q)
            ys={bit(q.target,x) for x in xs}
            self.assertEqual(r.status,"MIXED" if len(ys)==2 else "ONE" if 1 in ys else "ZERO")
            self.assertTrue(verify_result(v,q,r))
            ones=zeros=0
            for j in range(width):
                trit=rng.randrange(3)
                if trit==1: ones|=1<<j
                if trit==2: zeros|=1<<j
            q=Cube(ones,zeros)
            r=i.realize(q)
            self.assertEqual(r.status,"POSSIBLE" if any(
                x&ones==ones and not x&zeros for x in xs) else "EXCLUDED")
            self.assertTrue(verify_result(v,q,r))


if __name__=="__main__":
    unittest.main()
