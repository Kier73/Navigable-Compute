"""Strict receipt wire round-trip and hostile payload checks."""
import json
import unittest
from navigable_compute.machine import Instruction,Program
from navigable_compute.observability import ObservationAt
from navigable_compute.quotient import synthesize_quotient_trace
from navigable_compute.law_synthesis import synthesize_additive_law
from navigable_compute.wire import (encode_quotient,decode_quotient,encode_law,decode_law,
    encode_task,decode_task)
from navigable_compute.task_quotient import synthesize_task_trace

class WireTests(unittest.TestCase):
    def test_quotient_cross_process_wire_contract(self):
        p=Program(2,(Instruction('add_const',0,7),Instruction('add_const',1,2)))
        q=(ObservationAt(10**50+3,0,1,0),)
        r=synthesize_quotient_trace(p,q)
        encoded=encode_quotient(p,q,r)
        qq,ff,rr=decode_quotient(encoded,p)
        self.assertEqual(q,qq)
        self.assertEqual((),ff)
        self.assertEqual(r,rr)
        with self.assertRaises(ValueError):
            decode_quotient(encoded,Program(2,(Instruction('add_const',0,5),)))
        with self.assertRaises(ValueError):
            decode_quotient(encoded+b' ',p)
        with self.assertRaises(ValueError):
            decode_law(encoded)
        changed=encoded.replace(b'"quotient_products":',b'"quotient_products":99999,"extra":')
        with self.assertRaises(ValueError):
            decode_quotient(changed,p)

    def test_task_wire_cannot_masquerade_as_future_stable(self):
        p=Program(1,(Instruction('set',0,0),))
        q=(ObservationAt(10**90+1,0,255,0),)
        r=synthesize_task_trace(p,q)
        packet=encode_task(p,q,r)
        qq,ff,rr=decode_task(packet,p)
        self.assertEqual((qq,ff,rr),(q,(),r))
        with self.assertRaises(ValueError):
            decode_quotient(packet,p)
        with self.assertRaises(ValueError):
            decode_task(packet,Program(1,(Instruction('set',0,1),)))
        with self.assertRaises(ValueError):
            decode_task(packet+b' ',p)

    def test_law_wire_contract(self):
        q=(ObservationAt(10**40+5,0,255,41),ObservationAt(10**40+6,0,255,48))
        l=synthesize_additive_law(1,q)
        encoded=encode_law(1,q,l)
        qs,out=decode_law(encoded)
        self.assertEqual(q,qs)
        self.assertEqual(l,out)
        with self.assertRaises(ValueError):
            decode_law(encoded.replace(b'"construct-additive-law"',b'"finite-observer-quotient"'))
        with self.assertRaises(ValueError):
            decode_law(encoded+b'\n')
        with self.assertRaises(ValueError):
            decode_quotient(encoded,l.program)
        with self.assertRaises(ValueError):
            decode_law(b'{"a":1,"a":2}')

if __name__=='__main__': unittest.main()
