"""Learn a proof-operation on earlier traces, activate on unseen hypergraphs."""
import random
import unittest
from dataclasses import replace
from navigable_compute.intention_fibre import construct_fibre,FibreBudget
from navigable_compute.fibre_refinement import refine
from navigable_compute.procedural_vocabulary import Vocabulary,_canonical
from navigable_compute.procedural_vocabulary_verify import verify_reduction
from navigable_compute.induced_transversal import induce,activate,transversals,Skill,Example
from navigable_compute.induced_transversal_verify import verify_skill,verify_example


def make(source,width=128,context="fixture"):
    f=construct_fibre(width,())
    rows=(source,)
    return Vocabulary(f,rows,refine(f,rows),context=context)


def trained():
    # Actual earlier induction, on disjoint variable alphabets.
    a=make((0,3,5,6),context="learn-a")
    b=make((0,24,40,48),context="learn-b")
    assert len(a.train_symbolic()[0])==2
    assert len(b.train_symbolic()[0])==2
    return induce((a,b))


def multiply(p,m):
    out=set()
    for t in p:
        out^={t|m}
    return out


class InducedSkillTests(unittest.TestCase):
    def test_transfer_to_new_arity_and_support(self):
        s=trained()
        self.assertTrue(verify_skill(s))
        self.assertEqual(Skill.restore(s.packet()),s)
        # 9 source variables, three disjoint triple monomials.
        # The old grammar returns no proposals because support exceeds eight.
        source=(0,7,56,448)
        v=make(source,context="unseen")
        self.assertEqual(v.proposals(),())
        self.assertEqual(v.train_symbolic()[0],())
        exact=Vocabulary.exact
        def forbidden(*a,**k):
            raise AssertionError("cold query during activation")
        try:
            Vocabulary.exact=forbidden
            receipt=activate(s,v,0,max_arity=3)
        finally:
            Vocabulary.exact=exact
        self.assertEqual(len(receipt.masks),27)
        self.assertEqual(len(v.learned),27)
        for i,l in enumerate(v.learned):
            self.assertEqual(l.bit,1)
            self.assertTrue(verify_reduction(v,l.proof.target,l.proof))
            self.assertEqual(v.exact(l.target)[0],1)
            k=(1<<(30+i))|(1<<120)
            target=tuple(sorted(multiply(l.proof.target,k)^{0} if i%2 else multiply(l.proof.target,k)))
            self.assertEqual(v.reduce(target).value,i%2)
            self.assertEqual(v.exact(target)[0],i%2)
            shifted=tuple(sorted(set(target)^{1<<127}))
            self.assertIsNone(v.reduce(shifted).value)
        self.assertEqual(v.overlay.count, (1+3*7**2)*2**119)

    def test_large_support_and_negative_controls(self):
        skill=trained()
        # 48 variables; all nonconstant terms share the lowest bit.
        source=(0,)+tuple(1|(1<<i) for i in range(1,48))
        v=make(source)
        self.assertEqual(v.proposals(),())
        a=activate(skill,v,0,max_arity=3)
        self.assertEqual(a.masks,(1,))
        self.assertEqual(v.reduce((1,)).value,1)
        self.assertEqual(v.exact((1,))[0],1)
        opposite=make(source[1:])
        with self.assertRaises(ValueError):
            activate(skill,opposite,0,max_arity=3)
        self.assertEqual(opposite.learned,())
        small=make((0,7,56,448))
        with self.assertRaises(FibreBudget):
            activate(skill,small,0,max_work=0)
        with self.assertRaises(FibreBudget):
            activate(skill,small,0,max_product_terms=1)
        self.assertEqual(small.learned,())
        self.assertEqual(transversals((0,7,56,448),max_arity=2)[0],())

    def test_independent_complete_small_mask_oracle(self):
        rng=random.Random(17017)
        for _ in range(160):
            n=rng.randint(2,8)
            edges=tuple(sorted(rng.sample(range(1,2**n),rng.randint(1,min(8,2**n-1)))))
            k=rng.randint(1,n)
            actual,_=transversals((0,)+edges,max_arity=k)
            expected=[]
            for m in range(1,2**n):
                if m.bit_count()>k or not all(m&e for e in edges):
                    continue
                if any(all((m^(1<<i))&e for e in edges) for i in range(n) if m>>i&1):
                    continue
                expected.append(m)
            self.assertEqual(actual,tuple(sorted(expected,key=lambda m:(m.bit_count(),m))))

    def test_trace_diversity_and_tamper(self):
        v=make((0,3,5,6))
        v.train_symbolic()
        with self.assertRaises(ValueError):
            induce((v,v))
        s=trained()
        e=s.examples[0]
        self.assertFalse(verify_example(Example(e.source,(0,1))))
        bad=replace(s,examples=(replace(e,factor=(0,1)),)+s.examples[1:])
        self.assertFalse(verify_skill(bad))
        with self.assertRaises(ValueError):
            activate(bad,make((0,7,56,448)),0)
        with self.assertRaises(ValueError):
            Skill.restore(_canonical(dict(schema="ncr017.transversal-skill.v1",
                                        examples=[(e.source,(0,1)),(s.examples[1].source,s.examples[1].factor)])))


if __name__=="__main__":
    unittest.main()
