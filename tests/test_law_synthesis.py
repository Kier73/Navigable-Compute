"""NCL-001 tests: inverse construction of parametric programs from future goals."""
import random
import unittest
from dataclasses import replace
from navigable_compute.machine import execute, Instruction, Program
from navigable_compute.observability import ObservationAt
from navigable_compute.synthesis import Unrealizable
from navigable_compute.law_synthesis import synthesize_additive_law,verify_additive_law


class LawSynthesisTests(unittest.TestCase):
    def test_construction_without_supplied_program(self):
        huge=10**80+211
        seed=37
        delta=13
        qs=tuple(ObservationAt(t,0,255,(seed+delta*t)%256) for t in
                 (huge,huge+1,huge+3,huge+512))
        r=synthesize_additive_law(1,qs)
        self.assertEqual(r.program.instructions,(Instruction('add_const',0,13),))
        self.assertEqual(r.initial,bytes((37,)))
        self.assertEqual(r.law_solution_count,1)
        self.assertTrue(verify_additive_law(1,qs,r))
        for t in range(100):
            self.assertEqual(execute(r.program,r.initial,t)[0],(seed+t*delta)%256)

    def test_underspecification_is_not_silently_hidden(self):
        r=synthesize_additive_law(3,(ObservationAt(10,0,255,97),))
        self.assertEqual(r.law_solution_count,256*65536*65536)
        self.assertEqual(r.delta_residue_moduli,((0,1),(0,1),(0,1)))
        self.assertTrue(verify_additive_law(3,(ObservationAt(10,0,255,97),),r))

    def test_partial_information_increases_constraint_modulus(self):
        base=10**43+255
        start=201
        delta=187
        qs=tuple(ObservationAt(base+offset,0,255,(start+delta*(base+offset))%256)
                 for offset in (0,64,128))
        r=synthesize_additive_law(1,qs)
        self.assertEqual(r.law_solution_count,64)
        self.assertEqual(r.delta_residue_moduli[0],(delta%4,4))
        self.assertTrue(verify_additive_law(1,qs,r))
        self.assertEqual(r.program.instructions[0].b,delta%4)
        # Additional observation can distinguish among 64 candidate laws.
        q2=qs+(ObservationAt(base+1,0,255,(start+delta*(base+1))%256),)
        rr=synthesize_additive_law(1,q2)
        self.assertEqual(rr.law_solution_count,1)
        self.assertEqual(rr.program.instructions[0].b,delta)

    def test_impossible_goals_and_invalid_observations(self):
        with self.assertRaises(Unrealizable):
            synthesize_additive_law(1,(ObservationAt(0,0,255,6),ObservationAt(256,0,255,7)))
        with self.assertRaises(ValueError):
            synthesize_additive_law(1,(ObservationAt(0,0,1,1),))
        with self.assertRaises(ValueError):
            synthesize_additive_law(1,())

    def test_random_parametric_law_recovery(self):
        rng=random.Random(3351)
        for _ in range(120):
            n=rng.randrange(1,6)
            original=[rng.randrange(256) for i in range(n)]
            deltas=[rng.randrange(256) for i in range(n)]
            epoch=10**30+rng.randrange(2000)
            queries=tuple(ObservationAt(epoch+j,a,255,(original[a]+deltas[a]*(epoch+j))%256)
                          for a in range(n) for j in range(2))
            r=synthesize_additive_law(n,queries)
            self.assertEqual(r.initial,bytes(original))
            self.assertEqual(tuple(i.b for i in r.program.instructions),tuple(deltas))
            self.assertTrue(verify_additive_law(n,queries,r))
            self.assertFalse(verify_additive_law(n,queries,replace(r,witness_digest='0'*64)))

if __name__=='__main__': unittest.main()
