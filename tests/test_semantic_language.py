"""NCR-003: exact contextual language construction and independent verification."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import unittest

from invariant_demo import grid
from navigable_compute.self_space import build_endogenous_processor
from navigable_compute.semantic_language import (ObservationContract,
                                                  derive_semantic_language)
from navigable_compute.semantic_language_verify import verify_semantic_language
from navigable_compute.semantic_language_wire import (encode_semantic_language,
                                                      decode_semantic_language)
from navigable_compute.wire import _wrap
from navigable_compute.reflective_architecture import (construct_reflective_architecture,
                                                       architecture_run)


def rehash(obj):
    return replace(obj, digest=sha256(json.dumps(obj.encode(),sort_keys=True,
                   separators=(',', ':'),ensure_ascii=True).encode('ascii')).hexdigest())


def goal_contract(machine):
    return ObservationContract(machine.states, tuple(
        'goal' if machine.source.ranks[s] == 0 else 'not-goal' for s in machine.states))


class ObserverDerivedLanguageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.small = build_endogenous_processor(grid(3))
        cls.large = build_endogenous_processor(grid(5))
        cls.goal = derive_semantic_language(cls.large,goal_contract(cls.large))

    def test_goal_observer_derives_five_instructions_from_29(self):
        m = self.large
        q = self.goal
        self.assertEqual(len(m.operators),29)
        self.assertEqual(len(q.classes),5)
        self.assertEqual(tuple(map(len,q.classes)),(1,16,9,2,1))
        self.assertTrue(verify_semantic_language(m,q))
        self.assertEqual(q.normal_forms[0],())
        self.assertEqual(tuple(len(w) for w in q.normal_forms), tuple(range(5)))

    def test_exact_state_observer_preserves_29_and_constant_collapses_one(self):
        m=self.large
        exact=derive_semantic_language(m,ObservationContract(
            m.states,tuple(f'state-{s}' for s in m.states)))
        empty=derive_semantic_language(m,ObservationContract(m.states,('same',)*len(m.states)))
        self.assertEqual(len(exact.classes),29)
        self.assertEqual(len(empty.classes),1)
        self.assertTrue(verify_semantic_language(m,exact))
        self.assertTrue(verify_semantic_language(m,empty))
        self.assertEqual(empty.apply_code_to_code(0,0),0)

    def test_two_sided_context_changes_the_equivalence(self):
        m=self.large
        observer=ObservationContract((m.states[0],),tuple(
            'visible' if i in (0,2) else 'hidden' for i in range(len(m.states))))
        lang=derive_semantic_language(m,observer)
        self.assertEqual(lang.instruction_count,6)
        self.assertTrue(verify_semantic_language(m,lang))
        # One-sided contexts distinguish only four; omitting either context
        # would incorrectly identify different program behavior.
        signatures=[]
        for op in range(len(m.operators)):
            left=tuple(observer.outputs[m.operators[a][m.operators[op][0]]]
                       for a in range(len(m.operators)))
            right=tuple(observer.outputs[m.operators[op][m.operators[b][0]]]
                        for b in range(len(m.operators)))
            signatures.append((left,right))
        self.assertEqual(len({x[0] for x in signatures}),4)
        self.assertEqual(len({x[1] for x in signatures}),4)
        self.assertGreater(lang.refinements,0)

    def test_execution_preserves_observations_not_internal_states(self):
        m=self.large
        q=self.goal
        for state in m.states:
            for word in ((),(0,), (0,1), (2,4,7), (15,14,13,12)):
                expected=m.execute(word,state)
                observer=goal_contract(m)
                label=observer.outputs[m.states.index(expected)]
                self.assertEqual(q.observe(m,word,state),label)
        self.assertRaises(ValueError,q.observe,m,(0,),999)
        partial=derive_semantic_language(m,ObservationContract((m.states[0],),q.contract.outputs))
        self.assertRaises(ValueError,partial.observe,m,(0,),m.states[1])

    def test_language_executes_its_own_codes_faithfully(self):
        q=self.goal
        for a in range(q.instruction_count):
            for b in range(q.instruction_count):
                self.assertEqual(q.apply_code_to_code(a,b),q.multiplication[a][b])
            self.assertEqual(q.apply_code_to_code(a,q.identity),a)
        # Self-application is multiplication of one code by itself, not a
        # claim to run an encoded general-purpose evaluator AST.
        self.assertEqual(q.apply_code_to_code(1,1),2)
        self.assertEqual(q.apply_code_to_code(4,4),4)

    def test_full_future_coordinate_without_trajectory_replay(self):
        m=self.large
        q=self.goal
        n=10**100+173
        code, products=q.power_code((0,2,4),n)
        self.assertEqual(code,q.class_of[m.program_opcode((0,2,4)*4)])
        self.assertLess(products,2*n.bit_length()+1)
        self.assertEqual(q.power_code((0,),0),(q.identity,0))
        self.assertEqual(q.power_code((0,),1)[0],q.generator_codes[0])
        for bad in (-1,True,1<<4100):
            self.assertRaises(ValueError,q.power_code,(0,),bad)

    def test_second_ground_reconstructs_its_own_language(self):
        m=self.small
        q=derive_semantic_language(m,goal_contract(m))
        self.assertEqual(len(m.operators),4)
        self.assertEqual(q.instruction_count,3)
        self.assertTrue(verify_semantic_language(m,q))
        self.assertNotEqual(q.source_sha256,self.goal.source_sha256)

    def test_constructed_reflective_evaluator_agrees_at_observer_level(self):
        m=self.small
        q=derive_semantic_language(m,goal_contract(m))
        architecture=construct_reflective_architecture(m)
        receipt=architecture_run(m,architecture,m.generator_opcodes[:2],m.states[0],depth=2)
        expected=q.contract.outputs[m.states.index(receipt['result_state'])]
        self.assertEqual(q.observe(m,(0,1),m.states[0]),expected)

    def test_canonical_wire_and_tampering(self):
        m=self.large
        encoded=encode_semantic_language(m,self.goal)
        self.assertEqual(decode_semantic_language(encoded,m),self.goal)
        self.assertEqual(encode_semantic_language(m,decode_semantic_language(encoded,m)),encoded)
        for mutated in (
            rehash(replace(self.goal,classes=self.goal.classes[:-1])),
            rehash(replace(self.goal,multiplication=((0,)*5,)*5)),
            rehash(replace(self.goal,regular_actions=((0,)*5,)*5)),
            rehash(replace(self.goal,class_of=(0,)*29)),
            rehash(replace(self.goal,normal_forms=((),)*5)),
        ):
            self.assertFalse(verify_semantic_language(m,mutated))
        self.assertRaises(ValueError,decode_semantic_language,encoded+b' ',m)
        self.assertRaises(ValueError,decode_semantic_language,encoded,self.small)
        payload=self.goal.packet()
        self.assertRaises(ValueError,decode_semantic_language,_wrap(
            'observer-derived-language',{**payload,'unknown':1}),m)

    def test_changed_refinement_counter_is_not_a_valid_receipt(self):
        m=self.large
        contract=ObservationContract((m.states[0],),tuple(
            'visible' if i in (0,2) else 'hidden' for i in range(len(m.states))))
        cert=derive_semantic_language(m,contract)
        forged=rehash(replace(cert,refinements=cert.refinements+1))
        self.assertFalse(verify_semantic_language(m,forged))

    def test_seven_by_seven_observer_relative_language(self):
        m=build_endogenous_processor(grid(7))
        q=derive_semantic_language(m,goal_contract(m))
        self.assertEqual(len(m.operators),38)
        self.assertEqual(q.instruction_count,7)
        self.assertTrue(verify_semantic_language(m,q))

    def test_semantic_partition_adversarial_label_profiles(self):
        m=self.large
        for modulus in (2,3,5,7):
            labels=tuple(f'bucket-{(i*i+3*i+7)%modulus}' for i in range(len(m.states)))
            contract=ObservationContract((m.states[0],m.states[6]),labels)
            lang=derive_semantic_language(m,contract)
            self.assertTrue(verify_semantic_language(m,lang))

    def test_invalid_contracts_and_budget_refuse(self):
        m=self.large
        for contract in (
             ObservationContract((),('a',)*25),
             ObservationContract((0,0),('a',)*25),
             ObservationContract(m.states,('a',)*24),
             ObservationContract((999,),('a',)*25),
             ObservationContract((m.states[1],m.states[0]),('a',)*25),
             ObservationContract((m.states[0],),('a',)*24+(3,))):
            self.assertRaises(ValueError,derive_semantic_language,m,contract)
        self.assertRaises(ValueError,derive_semantic_language,m,goal_contract(m),max_source_ops=10)


if __name__ == '__main__':
    unittest.main()
