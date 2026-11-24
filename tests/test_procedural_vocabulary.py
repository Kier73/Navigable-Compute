"""NCR-015 exact procedural reuse and negative transfer controls."""
from dataclasses import replace
import json
import random
import unittest

from navigable_compute.intention_fibre import construct_fibre,FibreBudget
from navigable_compute.fibre_refinement import refine
from navigable_compute.procedural_vocabulary import Vocabulary,_canonical
from navigable_compute.procedural_vocabulary_verify import verify_reduction

# Observed parity x0=x1 and observed majority(x2,x3,x4)=1.
OBS=((1,2),(0,12,20,24))
PAIR_ORS=((4,8,12),(4,16,20),(8,16,24))


def fixture(width=128,context="held-out",observations=OBS):
    parent=construct_fibre(width,())
    overlay=refine(parent,observations)
    return Vocabulary(parent,observations,overlay,context=context)


def multiply(p,m):
    terms=set()
    for x in p:
        y=x|m
        if y in terms: terms.remove(y)
        else: terms.add(y)
    return terms


def held_out(width=128,n=48):
    rng=random.Random(15015)
    targets=[]
    for i in range(n):
        terms={0} if i%2 else set()
        for _ in range(1+i%4):
            mask=sum(1<<j for j in rng.sample(range(5,width),1+i%3))
            # Pair OR is one, so multiply (pair OR XOR 1) by an unseen mask.
            relation=(0,)+PAIR_ORS[i%3]
            terms^=multiply(relation,mask)
        # These are unseen combinations, with no all-constant target terms
        # except the deliberately added answer bit.
        targets.append(tuple(sorted(terms)))
    return tuple(targets)


def bit(p,x):
    return sum((x&m)==m for m in p)%2


class VocabularyTests(unittest.TestCase):
    def test_induction_unseen_coordinates(self):
        v=fixture()
        base=[v.reduce(p).value for p in held_out()]
        learned=v.train()
        self.assertTrue(all(p in learned for p in PAIR_ORS))
        self.assertGreater(sum(x is None for x in base),0)
        for i,p in enumerate(held_out()):
            result=v.reduce(p)
            self.assertEqual(result.value,i%2)
            self.assertTrue(verify_reduction(v,p,result))
            # Unobserved bit must not become a fabricated constant.
            self.assertIsNone(v.reduce(tuple(sorted(set(p)^{1<<127}))).value)
        for m in (1<<100,(1<<100)|(1<<110)):
            p=tuple(sorted((1|m,2|m)))
            self.assertEqual(v.reduce(p).value,0)
        self.assertEqual(v.overlay.count,2**126)

    def test_exact_small_oracle_and_random_unknowns(self):
        v=fixture(8)
        v.train()
        xs=[x for x in range(256) if not any(bit(p,x) for p in OBS)]
        self.assertEqual(len(xs),v.overlay.count)
        rng=random.Random(8)
        for p in held_out(8,24)+tuple(tuple(sorted(rng.sample(range(256),rng.randint(0,8)))) for _ in range(120)):
            result=v.reduce(p)
            expected={bit(p,x) for x in xs}
            if result.value is not None:
                self.assertEqual(expected,{result.value})
            self.assertTrue(verify_reduction(v,p,result))
            self.assertEqual(v.exact(p)[0],next(iter(expected)) if len(expected)==1 else None)

    def test_restore_and_tamper(self):
        v=fixture()
        v.train()
        p=held_out()[0]
        a=v.reduce(p)
        self.assertFalse(verify_reduction(v,p,replace(a,residual=(0,) if a.value==0 else ())))
        self.assertFalse(verify_reduction(v,p,replace(a,steps=a.steps[:-1])))
        with self.assertRaises(FibreBudget):
            v.reduce(p,max_steps=0)
        recovered=Vocabulary.restore(v.parent,v.observations,v.overlay,
                                      context=v.context,payload=v.packet())
        self.assertEqual(recovered.reduce(p),a)
        b=json.loads(v.packet())
        b["learned"][0][1]^=1
        with self.assertRaises(ValueError):
            Vocabulary.restore(v.parent,v.observations,v.overlay,
                               context=v.context,payload=_canonical(b))
        with self.assertRaises(ValueError):
            Vocabulary.restore(v.parent,v.observations,v.overlay,
                               context="different",payload=v.packet())
        # Majority zero does NOT permit transfer of the majority-one rules.
        opposite=fixture(observations=((1,2),(12,20,24)))
        with self.assertRaises(ValueError):
            Vocabulary.restore(opposite.parent,opposite.observations,opposite.overlay,
                               context=v.context,payload=v.packet())


if __name__=="__main__":
    unittest.main()
