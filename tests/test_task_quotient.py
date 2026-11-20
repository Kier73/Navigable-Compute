import random
import unittest
from dataclasses import replace
from navigable_compute.machine import Program,Instruction as I,execute
from navigable_compute.observability import ObservationAt
from navigable_compute.task_quotient import synthesize_task_trace
from navigable_compute.task_verify import verify_task_trace
from navigable_compute.quotient import build_component_quotient,QuotientBudgetExceeded
from navigable_compute.synthesis import Unrealizable

class TaskQuotientTests(unittest.TestCase):
    def test_forgotten_byte_collapses_at_future(self):
        p=Program(1,(I('set',0,0),))
        all_future=build_component_quotient(p,(0,),((0,255),))
        q=(ObservationAt(10**66+1,0,255,0),)
        r=synthesize_task_trace(p,q)
        self.assertEqual(all_future.classes,256)
        self.assertEqual(r.certificates[0].classes,1)
        self.assertEqual(r.candidate_count,256)
        self.assertTrue(verify_task_trace(p,q,r))
        at_zero=synthesize_task_trace(p,(ObservationAt(0,0,255,73),))
        self.assertEqual(at_zero.certificates[0].classes,256)
        self.assertEqual(at_zero.initial,bytes((73,)))
        self.assertTrue(verify_task_trace(p,(ObservationAt(0,0,255,73),),at_zero))

    def test_high_bit_becomes_two_task_classes(self):
        p=Program(1,(I('add_const',0,1),))
        q=(ObservationAt(10**37+12,0,128,0),)
        a=synthesize_task_trace(p,q)
        self.assertEqual(a.certificates[0].classes,2)
        self.assertEqual(a.candidate_count,128)
        self.assertTrue(verify_task_trace(p,q,a))
        self.assertEqual(build_component_quotient(p,(0,),((0,128),)).classes,256)

    def test_multiple_times_derive_task_specific_distinctions(self):
        p=Program(1,(I('add_const',0,1),))
        for coords in [(0,),(0,1),(0,1,2),(0,64,128)]:
            q=tuple(ObservationAt(t,0,128,((43+t)&255)&128) for t in coords)
            r=synthesize_task_trace(p,q)
            signatures={tuple(((x+t)&255)&128 for t in coords) for x in range(256)}
            self.assertEqual(r.certificates[0].classes,len(signatures))
            self.assertTrue(verify_task_trace(p,q,r))
            for item in q:
                self.assertEqual(execute(p,r.initial,item.cycle)[0]&item.mask,item.value)

    def test_independent_global_and_refusal(self):
        p=Program(6,tuple(I('add_const',i,i+1) for i in range(6)))
        epoch=10**90+7
        qs=(ObservationAt(epoch,1,128,0),ObservationAt(epoch+1,1,128,0),
            ObservationAt(epoch,4,1,0))
        r=synthesize_task_trace(p,qs)
        self.assertEqual(r.global_bits,48)
        self.assertEqual(r.examined_states,512)
        self.assertTrue(verify_task_trace(p,qs,r))
        self.assertFalse(verify_task_trace(p,qs,replace(r,candidate_count=r.candidate_count+1)))
        c=r.certificates[0]
        wrong=replace(c,class_of=(1,)+c.class_of[1:])
        self.assertFalse(verify_task_trace(p,qs,replace(r,certificates=(wrong,)+r.certificates[1:])))
        coupled=Program(2,(I('add_const',0,1),I('xor',1,0)))
        with self.assertRaises(QuotientBudgetExceeded):
            synthesize_task_trace(coupled,(ObservationAt(10,1,1,0),))

    def test_contradiction(self):
        p=Program(1,(I('set',0,0),))
        with self.assertRaises(Unrealizable):
            synthesize_task_trace(p,(ObservationAt(1,0,255,1),))
        with self.assertRaises(Unrealizable):
            synthesize_task_trace(p,(ObservationAt(1,0,255,0),ObservationAt(1,0,255,1)))

    def test_random_against_reference(self):
        rng=random.Random(1097)
        for _ in range(35):
            p=Program(1,tuple(I(rng.choice(('add_const','xor_const','rotl','set','and_const')),
                                 0,rng.randrange(256)) for k in range(3)))
            start=rng.randrange(256)
            coords=[rng.randrange(0,64) for k in range(3)]
            masks=[rng.choice((1,3,15,128,255)) for k in range(3)]
            q=tuple(ObservationAt(t,0,m,execute(p,bytes((start,)),t)[0]&m)
                    for t,m in zip(coords,masks))
            r=synthesize_task_trace(p,q)
            self.assertTrue(verify_task_trace(p,q,r))
            for item in q:
                self.assertEqual(execute(p,r.initial,item.cycle)[0]&item.mask,item.value)

if __name__=='__main__': unittest.main()
