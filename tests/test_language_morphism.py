"""NCR-003: observer revisions preserve semantics or yield an obstruction."""
from dataclasses import replace
from hashlib import sha256
import json
import unittest

from invariant_demo import grid
from navigable_compute.self_space import build_endogenous_processor
from navigable_compute.semantic_language import ObservationContract, derive_semantic_language
from navigable_compute.language_morphism import (LanguageProjection,NonRefinementWitness,
                                                 assess_language_projection)
from navigable_compute.language_morphism_verify import verify_language_relation


def rehash(obj):
    return replace(obj,digest=sha256(json.dumps(obj.encode(),sort_keys=True,
                                    separators=(',',':'),ensure_ascii=True).encode('ascii')).hexdigest())


class LanguageMorphismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m=build_endogenous_processor(grid(5))
        cls.full=derive_semantic_language(cls.m,ObservationContract(
            cls.m.states,tuple(str(s) for s in cls.m.states)))
        cls.goal=derive_semantic_language(cls.m,ObservationContract(
            cls.m.states,tuple('goal' if cls.m.source.ranks[s]==0 else 'not-goal' for s in cls.m.states)))
        cls.constant=derive_semantic_language(cls.m,ObservationContract(cls.m.states,('same',)*25))

    def test_construct_homomorphic_language_revision(self):
        m=self.m
        for old,new in ((self.full,self.goal),(self.goal,self.constant),
                        (self.full,self.constant),(self.goal,self.goal)):
            projection=assess_language_projection(m,old,new)
            self.assertIsInstance(projection,LanguageProjection)
            self.assertTrue(verify_language_relation(m,old,new,projection))
            self.assertEqual(len(projection.mapping),old.instruction_count)
            for a in range(old.instruction_count):
                for b in range(old.instruction_count):
                    self.assertEqual(projection.translate(old.multiplication[a][b]),
                        new.multiplication[projection.translate(a)][projection.translate(b)])

    def test_reverse_projection_produces_finite_information_loss_witness(self):
        result=assess_language_projection(self.m,self.goal,self.full)
        self.assertIsInstance(result,NonRefinementWitness)
        self.assertTrue(verify_language_relation(self.m,self.goal,self.full,result))
        self.assertFalse(self.goal.class_of[result.first_original_opcode]!=
                         self.goal.class_of[result.second_original_opcode])
        self.assertNotEqual(self.full.class_of[result.first_original_opcode],
                            self.full.class_of[result.second_original_opcode])
        self.assertIsInstance(assess_language_projection(self.m,self.constant,self.goal),NonRefinementWitness)

    def test_tampered_projection_and_obstruction_rejected(self):
        projection=assess_language_projection(self.m,self.full,self.goal)
        broken=rehash(replace(projection,mapping=(0,)*self.full.instruction_count))
        self.assertFalse(verify_language_relation(self.m,self.full,self.goal,broken))
        obstruction=assess_language_projection(self.m,self.goal,self.full)
        wrong=rehash(replace(obstruction,second_original_opcode=obstruction.first_original_opcode))
        self.assertFalse(verify_language_relation(self.m,self.goal,self.full,wrong))

    def test_code_translation_and_projection_type_refusals(self):
        projection=assess_language_projection(self.m,self.full,self.goal)
        for invalid in (-1,True,29):
            self.assertRaises(ValueError,projection.translate,invalid)
        self.assertFalse(verify_language_relation(self.m,self.full,self.goal,{'fake':1}))


if __name__=='__main__':unittest.main()
