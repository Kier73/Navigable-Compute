"""NCQ-001 regression: nonlinear state, observer quotient, reverse intention,
and an independent concrete-transition/cycle-detection receipt validator."""
import random
import unittest
from dataclasses import replace
from navigable_compute import Instruction as I, Program, execute
from navigable_compute.synthesis import Constraint, Unrealizable
from navigable_compute.observability import ObservationAt
from navigable_compute.quotient import (byte_components, build_component_quotient,
    power_successor, synthesize_quotient_trace, QuotientBudgetExceeded)
from navigable_compute.quotient_verify import verify_quotient_trace


class QuotientTests(unittest.TestCase):
    def test_parity_is_two_state_processor_for_nonlinear_addition(self):
        p=Program(1,(I('add_const',0,1),))
        ground=build_component_quotient(p,(0,),((0,1),))
        self.assertEqual(ground.classes,2)
        self.assertEqual(ground.state_count,256)
        self.assertEqual(ground.refinement_rounds,0)
        for start in range(256):
            for steps in (0,1,2,3,11,253,511):
                powered,_=power_successor(ground.successor,steps)
                got=ground.labels[powered[ground.class_of[start]]][0]
                self.assertEqual(got,execute(p,bytes((start,)),steps)[0]&1)

    def test_observer_changes_required_virtual_machine(self):
        p=Program(1,(I('add_const',0,1),))
        parity=build_component_quotient(p,(0,),((0,1),))
        high=build_component_quotient(p,(0,),((0,128),))
        full=build_component_quotient(p,(0,),((0,255),))
        self.assertEqual(parity.classes,2)
        self.assertEqual(high.classes,256)
        self.assertEqual(full.classes,256)
        self.assertNotEqual(parity.transition_sha256,'')

    def test_large_global_state_stays_factored(self):
        n=10**33+123
        p=Program(8,tuple(I('add_const',i,2*i+1) for i in range(8)))
        self.assertEqual(byte_components(p),tuple((i,) for i in range(8)))
        start=bytes((0x2a,0x4d,0x66,0x87,0,0,0,0))
        t1=n+1
        q=(ObservationAt(n,0,1,(start[0]+n)&1),
           ObservationAt(n,1,128,((start[1]+3*n)&255)&128),
           ObservationAt(n,2,15,((start[2]+5*n)&255)&15),
           ObservationAt(t1,0,1,(start[0]+t1)&1))
        r=synthesize_quotient_trace(p,q)
        self.assertEqual(r.global_state_bits,64)
        self.assertEqual(r.examined_states,3*256)
        self.assertEqual(tuple(cert.classes for cert in r.certificates),(2,256,16))
        self.assertEqual(r.observed,tuple(x.value for x in q))
        self.assertLess(r.products,700)
        self.assertTrue(verify_quotient_trace(p,q,r))
        for query in q:
            i=query.address
            self.assertEqual((r.initial[i]+(2*i+1)*query.cycle)&query.mask,query.value)
        self.assertFalse(verify_quotient_trace(p,q,replace(r,products=r.products+1)))

    def test_nonlinear_synthesis_multipart_goals_and_fixed_bits(self):
        p=Program(3,(I('add_const',0,7),I('rotl',1,1),I('and_const',2,3)))
        original=bytes((28,175,225))
        queries=(ObservationAt(123,0,255,(original[0]+7*123)&255),
                 ObservationAt(124,0,255,(original[0]+7*124)&255),
                 ObservationAt(16,1,15,original[1]&15),
                 ObservationAt(12,2,3,1))
        # On a repeated AND mask, the only possible output 12 steps later is
        # the initial byte's two low bits (1 is deliberately realizable).
        fixed=(Constraint(2,3,1),)
        result=synthesize_quotient_trace(p,queries,fixed=fixed)
        self.assertTrue(verify_quotient_trace(p,queries,result,fixed=fixed))
        for q in queries:
            self.assertEqual(execute(p,result.initial,q.cycle)[q.address]&q.mask,q.value)
        self.assertFalse(verify_quotient_trace(p,queries,result,fixed=()))

    def test_unrealizable_goal_and_bad_fixed_overlap(self):
        p=Program(2,(I('add_const',0,1),I('add_const',1,2)))
        with self.assertRaises(Unrealizable):
            synthesize_quotient_trace(p,(ObservationAt(0,0,1,0),ObservationAt(1,0,1,0)))
        with self.assertRaises(Unrealizable):
            synthesize_quotient_trace(p,(ObservationAt(1,0,1,1),),fixed=(Constraint(1,1,0),Constraint(1,1,1)))

    def test_refuse_large_coupled_nonlinearity(self):
        p=Program(2,(I('add_const',0,1),I('xor',1,0)))
        with self.assertRaises(QuotientBudgetExceeded):
            synthesize_quotient_trace(p,(ObservationAt(10,1,1,0),),max_states=4096)
        with self.assertRaises(QuotientBudgetExceeded):
            synthesize_quotient_trace(Program(1,(I('add_const',0,1),)),
                          (ObservationAt(1<<5000,0,1,0),))

    def test_constant_future_does_not_infer_unknown_initial_data(self):
        p=Program(2,(I('set',0,0),I('add_const',1,1)))
        q=(ObservationAt(10**60,0,255,0),)
        result=synthesize_quotient_trace(p,q)
        self.assertEqual(result.candidate_count,1<<16)
        self.assertEqual(result.initial,b'\x00\x00')
        self.assertTrue(verify_quotient_trace(p,q,result))
        with self.assertRaises(Unrealizable):
            synthesize_quotient_trace(p,(ObservationAt(1,0,255,7),))

    def test_tampered_certificates_rejected(self):
        p=Program(2,(I('add_const',0,1),I('xor_const',1,1)))
        q=(ObservationAt(10**27+7,0,1,0),)
        r=synthesize_quotient_trace(p,q)
        c=r.certificates[0]
        wrong=replace(c,successor=tuple(0 for _ in c.successor))
        self.assertFalse(verify_quotient_trace(p,q,replace(r,certificates=(wrong,))))
        wrong2=replace(c,transition_sha256='0'*64)
        self.assertFalse(verify_quotient_trace(p,q,replace(r,certificates=(wrong2,))))
        wrong3=replace(c,class_of=(c.class_of[1],)+c.class_of[1:])
        self.assertFalse(verify_quotient_trace(p,q,replace(r,certificates=(wrong3,))))
        self.assertFalse(verify_quotient_trace(p,q,replace(r,observed=(1,))))
        self.assertFalse(verify_quotient_trace(p,q,replace(r,witness_digest='0'*64)))

    def test_random_nonlinear_small_programs_vs_independent_interpreter(self):
        rng=random.Random(731209)
        ops=['add_const','xor_const','rotl','and_const','set']
        for _ in range(35):
            p=Program(1,tuple(I(rng.choice(ops),0,rng.randrange(256)) for __ in range(3)))
            mask=rng.choice([1,3,15,128,255])
            initial=rng.randrange(256)
            steps=rng.randrange(0,80)
            target=execute(p,bytes((initial,)),steps)[0]&mask
            q=(ObservationAt(steps,0,mask,target),)
            r=synthesize_quotient_trace(p,q)
            self.assertTrue(verify_quotient_trace(p,q,r),str((p,steps,mask)))
            self.assertEqual(execute(p,r.initial,steps)[0]&mask,target)
            cert=r.certificates[0]
            self.assertLessEqual(cert.classes,256)


if __name__=='__main__': unittest.main()
