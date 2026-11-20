"""NCS-001: exact self-describing finite CPU, refusal and forged-capsule tests."""
import json
import random
import unittest
from dataclasses import replace
from hashlib import sha256

from invariant_demo import grid
from navigable_compute.invariant_space import canonical
from navigable_compute.self_space import (
    build_endogenous_processor, AlgebraBudgetExceeded)
from navigable_compute.self_verify import verify_endogenous_processor
from navigable_compute.self_wire import encode_processor,decode_processor
from navigable_compute.wire import _wrap


class EndogenousProcessorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.desc=grid(5)
        cls.machine=build_endogenous_processor(cls.desc)

    def test_generates_isa_from_invariants_not_from_external_opcode_list(self):
        m=self.machine
        self.assertEqual(m.source.law_count,16)
        self.assertEqual(len(m.generator_coordinates),16)
        self.assertEqual(m.generator_coordinates,tuple(range(16)))
        self.assertEqual(m.instruction_count,29)
        self.assertTrue(verify_endogenous_processor(self.desc,m))

    def test_noncommutative_program_order_genuinely_matters(self):
        m=self.machine
        a,b=m.generator_opcodes[:2]
        self.assertNotEqual(m.multiply[a][b],m.multiply[b][a])
        self.assertTrue(any(m.execute((0,1),s)!=m.execute((1,0),s) for s in m.states))

    def test_all_derived_opcodes_have_working_internal_programs(self):
        m=self.machine
        for opcode in range(m.instruction_count):
            for state in m.states:
                self.assertEqual(m.interpret_own_opcode(opcode,state),m.execute_opcode(opcode,state))

    def test_interpreter_matches_raw_concrete_transition_steps(self):
        m=self.machine
        rng=random.Random(17)
        for _ in range(100):
            s=rng.choice(m.states)
            word=tuple(rng.randrange(len(m.generator_opcodes)) for _ in range(rng.randrange(18)))
            expected=s
            for symbol in word:
                # Directly use the source law's *semantic table*, not the monoid.
                from navigable_compute.invariant_space import navigate_invariant_instance
                expected=navigate_invariant_instance(m.source,symbol).transition[expected]
            self.assertEqual(m.execute(word,s),expected)

    def test_large_coordinate_and_small_reference_iteration(self):
        m=self.machine
        w=(0,1,1,0)
        for state in m.states:
            direct=state
            for repetitions in range(35):
                result,comps=m.repeat_program(w,repetitions,state)
                self.assertEqual(result,direct)
                self.assertLessEqual(comps,2*max(1,repetitions.bit_length()))
                direct=m.execute(w,direct)
        far=10**100+173
        outcome,ops=m.repeat_program(w,far,m.states[0])
        self.assertIn(outcome,self.desc.goal)
        self.assertLess(ops,1000)

    def test_capsule_round_trip_reconstructs_source_and_semantics(self):
        m=self.machine
        raw=encode_processor(self.desc,m)
        other=decode_processor(raw,self.desc)
        self.assertEqual(other,m)
        for state in m.states:
            for k in range(m.instruction_count):
                self.assertEqual(other.execute(other.normal_forms[k],state),m.execute_opcode(k,state))

    def test_rejects_wrong_description(self):
        raw=encode_processor(self.desc,self.machine)
        with self.assertRaises(ValueError):
            decode_processor(raw,grid(3))

    def test_rejects_noncanonical_and_corrupt_circuit(self):
        m=self.machine
        raw=encode_processor(self.desc,m)
        with self.assertRaises(ValueError):
            decode_processor(raw+b' ',self.desc)
        p=m.packet()
        p['multiplication'][1][1]=0
        # Even an attacker recomputing BOTH digests cannot forge semantics.
        p['sha256']=sha256(canonical({k:v for k,v in p.items() if k!='sha256'})).hexdigest()
        hostile=_wrap('task-derived-operator-algebra',p)
        with self.assertRaises(ValueError):
            decode_processor(hostile,self.desc)

    def test_tampered_normal_form_and_wrong_generator_rejected(self):
        m=self.machine
        bad=replace(m,normal_forms=((),)*len(m.normal_forms))
        bad=replace(bad,digest=sha256(canonical(bad.encode())).hexdigest())
        self.assertFalse(verify_endogenous_processor(self.desc,bad))
        bad=replace(m,generator_opcodes=tuple(0 for _ in m.generator_opcodes))
        bad=replace(bad,digest=sha256(canonical(bad.encode())).hexdigest())
        self.assertFalse(verify_endogenous_processor(self.desc,bad))

    def test_budget_refuses_partial_closure(self):
        with self.assertRaises(AlgebraBudgetExceeded):
            build_endogenous_processor(self.desc,max_operators=2)

    def test_invalid_program_state_and_time_refuse(self):
        m=self.machine
        for word in ((True,),(-1,),(17,),[0],tuple([0]*65537)):
            with self.assertRaises((ValueError,TypeError)):
                m.execute(word,0)
        for state in (-1,9.0,True):
            with self.assertRaises((ValueError,TypeError)):
                m.execute((),state)
        for k in (-1,1.5,True,1<<4097):
            with self.assertRaises((ValueError,TypeError)):
                m.repeat_program((0,),k,0)

    def test_all_goal_space_has_identity_only(self):
        from navigable_compute.invariant_space import InvariantDescription
        b=grid(3)
        d=InvariantDescription(b.coordinates,b.symmetry,b.labels,
            b.initial,tuple(range(9)),(),1)
        m=build_endogenous_processor(d)
        self.assertEqual(m.instruction_count,1)
        self.assertEqual(m.execute((0,0,0),3),3)
        self.assertTrue(verify_endogenous_processor(d,m))

    def test_small_orbit_exhaustive_independent_semantics(self):
        d=grid(3)
        m=build_endogenous_processor(d)
        self.assertEqual(m.instruction_count,4)
        self.assertTrue(verify_endogenous_processor(d,m))
        # Enumerate all words up to length 7, independent of the monoid indices.
        from itertools import product
        from navigable_compute.invariant_space import navigate_invariant_instance
        gens=[navigate_invariant_instance(m.source,c).transition for c in m.generator_coordinates]
        for length in range(8):
            for word in product(range(len(gens)),repeat=length):
                for state in d.initial:
                    target=state
                    for op in word:target=gens[op][target]
                    self.assertEqual(m.execute(word,state),target)


if __name__=='__main__': unittest.main()
