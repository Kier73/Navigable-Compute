"""NCR-002 constructed self-interpretation: positive and adversarial controls."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import unittest

from invariant_demo import grid
from navigable_compute.self_space import build_endogenous_processor
from navigable_compute.reflective_architecture import (
    OPERATIONS,FORMS,ReflectiveArchitecture,_select_basis,_codebook,
    construct_reflective_architecture,architecture_run,encode_source)
from navigable_compute.reflective_architecture_verify import (
    verify_architecture,verify_architecture_execution)
from navigable_compute.reflective_architecture_wire import (
    encode_architecture,decode_architecture,encode_execution,decode_execution)
from navigable_compute.reflective_ground import lit,arg,primitive,host_run
from navigable_compute.reflective_verify import reference_run
from navigable_compute.wire import _wrap


def rehash(design):
    return replace(design,digest=sha256(json.dumps(design.encode(),sort_keys=True,
          separators=(',',':'),ensure_ascii=True).encode('ascii')).hexdigest())


class ConstructedReflectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.small=build_endogenous_processor(grid(3))
        cls.large=build_endogenous_processor(grid(5))
        cls.arch=construct_reflective_architecture(cls.large)

    def test_constructor_selects_four_seeds_not_seven(self):
        self.assertEqual(self.arch.primitive_seed,('at','cons','eq','rest'))
        self.assertEqual(len(self.arch.recipes),7)
        self.assertIn(('task','nested-index'),self.arch.recipes)
        self.assertIn(('first','index-zero'),self.arch.recipes)
        self.assertIn(('empty','compare-empty'),self.arch.recipes)
        self.assertTrue(verify_architecture(self.large,self.arch))

    def test_minimum_is_relative_to_declared_macro_grammar(self):
        basis,_=_select_basis(OPERATIONS)
        self.assertEqual(len(basis),4)
        with self.assertRaises(ValueError):
            _select_basis(tuple(op for op in OPERATIONS if op!='at'))
        with self.assertRaises(ValueError):
            construct_reflective_architecture(self.large,allowed_primitives=('eq','at','cons'))
        for malformed in (('at','at'),('arbitrary',),['at'],(1,)):
            with self.assertRaises(ValueError):_select_basis(malformed)

    def test_entire_finite_representation_space_is_addressable(self):
        codes=tuple(_codebook(self.large.digest,n) for n in range(120))
        self.assertEqual(len(set(codes)),120)
        with self.assertRaises(ValueError):_codebook(self.large.digest,120)
        with self.assertRaises(ValueError):_codebook(self.large.digest,-1)
        with self.assertRaises(ValueError):_codebook(self.large.digest,True)

    def test_different_representation_coordinates_have_same_semantics(self):
        hashes=set()
        for coordinate in (0,17,83,119):
            design=construct_reflective_architecture(self.large,
                                  representation_coordinate=coordinate)
            self.assertTrue(verify_architecture(self.large,design))
            hashes.add(design.digest)
            rec=architecture_run(self.large,design,(0,1,2),self.large.states[0],depth=2)
            self.assertTrue(verify_architecture_execution(self.large,design,rec))
            expected=self.large.states.index(self.large.states[0])
            for opcode in (0,1,2):
                expected=self.large.operators[opcode][expected]
            self.assertEqual(rec['result_state'],self.large.states[expected])
        self.assertEqual(len(hashes),4)

    def test_self_interpretation_on_two_task_generated_algebras(self):
        for machine in (self.small,self.large):
            design=construct_reflective_architecture(machine)
            results=[architecture_run(machine,design,(0,1),machine.states[0],depth=d)
                     for d in range(3)]
            self.assertEqual(len({r['result_state'] for r in results}),1)
            self.assertTrue(all(verify_architecture_execution(machine,design,r)
                                for r in results))
            self.assertLess(results[0]['expression_steps'],results[1]['expression_steps'])
            self.assertLess(results[1]['expression_steps'],results[2]['expression_steps'])

    def test_encoded_interpreter_is_data_not_eval_opcode(self):
        code=encode_source(self.arch.interpreter,self.arch.syntax_codebook)
        self.assertNotEqual(code,self.arch.interpreter)
        self.assertEqual(tuple(name for name,_ in code),tuple(name for name,_ in self.arch.interpreter))
        self.assertEqual(reference_run((('main',lit(code)),),'main',(),())[0],code)
        with self.assertRaises(ValueError):
            encode_source((('main',('magic',0)),),self.arch.syntax_codebook)

    def test_constructed_host_rejects_missing_primitive(self):
        unsafe=(('main',primitive('task',lit(0),lit(0))),)
        with self.assertRaises(ValueError):
            host_run(unsafe,'main',(),self.large.operators,
                     seed_primitives=self.arch.primitive_seed)
        with self.assertRaises(ValueError):
            reference_run(unsafe,'main',(),self.large.operators,
                          seed_primitives=self.arch.primitive_seed)

    def test_false_self_description_rejected_even_after_digest_rewrite(self):
        changed=[]
        for name,expr in self.arch.interpreter:
            changed.append((name,lit(0) if name=='run' else expr))
        forged=rehash(replace(self.arch,interpreter=tuple(changed)))
        self.assertFalse(verify_architecture(self.large,forged))
        tampered=rehash(replace(self.arch,primitive_seed=('eq','rest','cons','at','task')))
        self.assertFalse(verify_architecture(self.large,tampered))
        altered=rehash(replace(self.arch,representation_coordinate=119))
        self.assertFalse(verify_architecture(self.large,altered))
        wrong_ground=rehash(replace(self.arch,ground_sha256=self.small.digest))
        self.assertFalse(verify_architecture(self.large,wrong_ground))

    def test_invalid_or_fuel_exhausted_construction_refuses(self):
        with self.assertRaises(ValueError):
            architecture_run(self.large,self.arch,(0,1),0,depth=2,fuel=100)
        for word in ((True,),(-1,),(999,),[0,1]):
            with self.assertRaises(ValueError):
                architecture_run(self.large,self.arch,word,0)
        with self.assertRaises(ValueError):
            architecture_run(self.large,self.arch,(0,),0,depth=3)

    def test_source_bound_architecture_and_execution_wire(self):
        design=self.arch
        encoded=encode_architecture(self.large,design)
        self.assertEqual(decode_architecture(encoded,self.large),design)
        self.assertEqual(encode_architecture(self.large,decode_architecture(encoded,self.large)),encoded)
        rec=architecture_run(self.large,design,(0,1),0,depth=2)
        wire=encode_execution(self.large,design,rec)
        self.assertEqual(decode_execution(wire,self.large,design),rec)
        with self.assertRaises(ValueError):decode_architecture(encoded+b' ',self.large)
        with self.assertRaises(ValueError):decode_architecture(encoded,self.small)
        with self.assertRaises(ValueError):decode_execution(wire,self.small,design)
        for field,invalid in [('result_state',999),('expression_steps',1),('architecture_sha256','0'*64)]:
            with self.assertRaises(ValueError):
                decode_execution(_wrap('constructed-reflective-execution',{**rec,field:invalid}),self.large,design)

    def test_encoded_source_varies_with_ground_even_at_same_coordinate(self):
        a=construct_reflective_architecture(self.small)
        b=self.arch
        self.assertNotEqual(a.syntax_codebook,b.syntax_codebook)
        self.assertNotEqual(a.digest,b.digest)
        self.assertTrue(verify_architecture(self.small,a))
        self.assertFalse(verify_architecture(self.large,a))


if __name__=='__main__':unittest.main()
