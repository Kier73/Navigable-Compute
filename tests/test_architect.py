import unittest
from navigable_compute.machine import Program,Instruction
from navigable_compute.observability import ObservationAt
from navigable_compute.synthesis import Constraint
from navigable_compute.architect import Intention,realize

class ArchitectTests(unittest.TestCase):
    def test_affine_lane(self):
        p=Program(1,(Instruction('rotl',0,1),))
        r=realize(Intention(1,(ObservationAt(1024,0,1,1),),p))
        self.assertEqual(r.lane,'exact-gf2-affine')
        self.assertEqual(r.initial,b'\x01')
        self.assertFalse(r.independent_checker)

    def test_nonlinear_quotient_lane(self):
        p=Program(1,(Instruction('add_const',0,1),))
        r=realize(Intention(1,(ObservationAt(10**45+7,0,1,0),),p))
        self.assertEqual(r.lane,'finite-observer-quotient')
        self.assertTrue(r.independent_checker)
        self.assertEqual(r.certificate.certificates[0].classes,2)

    def test_construct_program_from_goals(self):
        q=(ObservationAt(10**60+1,0,255,8),ObservationAt(10**60+2,0,255,21))
        r=realize(Intention(1,q))
        self.assertEqual(r.lane,'construct-additive-law')
        self.assertEqual(r.program.instructions[0].b,13)
        self.assertTrue(r.independent_checker)

    def test_refuses_unsupported_fixed_input_with_missing_program(self):
        with self.assertRaises(ValueError):
            realize(Intention(1,(ObservationAt(0,0,255,12),),fixed_input=(Constraint(0,255,2),)))

if __name__=='__main__': unittest.main()
