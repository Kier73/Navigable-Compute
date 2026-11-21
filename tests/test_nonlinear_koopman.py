"""NCR-007 adversarial and exact reference tests."""
from dataclasses import replace
from hashlib import sha256
import json
import random
import unittest

from nonlinear_koopman_demo import nonlinear_family, long_shift_family
from navigable_compute.nonlinear_koopman import (
    PolynomialInvariants, construct_polynomial_laws, construct_koopman_ground,
    navigate_nonlinear, PolynomialBudget, canonical, eval_observers)
from navigable_compute.nonlinear_koopman_verify import (
    verify_polynomial_laws, verify_koopman_ground, verify_nonlinear_result)
from navigable_compute.nonlinear_koopman_wire import (
    encode_nonlinear, decode_nonlinear)
from navigable_compute.symbolic_quotient import (
    AffineMap,SymbolicSource,construct_symbolic_quotient)
from navigable_compute.symbolic_quotient_verify import verify_symbolic_quotient


def small_description():
    return PolynomialInvariants(4,(0,1,1,2),(0,1,2,3),
        ((1,2),(3,),(3,),()),2,((1,),))


def small_family(first,second):
    return construct_polynomial_laws(small_description(),(first,second))


def independent_moore_pair_oracle(family):
    """Exhaustive explicit-state oracle, deliberately only for n<=6."""
    n=family.description.width
    assert n<=6
    count=1<<n
    source_outputs=[family.project_observer(i) for i in range(count)]
    transitions=[tuple(op.apply(i) for i in range(count))
                 for op in family.operators]
    different={(i,j) for i in range(count) for j in range(count)
               if source_outputs[i]!=source_outputs[j]}
    while True:
        additional={(i,j) for i in range(count) for j in range(count)
                    if (i,j) not in different and
                    any((table[i],table[j]) in different for table in transitions)}
        if not additional:
            return different
        different.update(additional)


class NonlinearKoopmanTests(unittest.TestCase):
    def _build(self,family,**kwargs):
        self.assertTrue(verify_polynomial_laws(family))
        model=construct_koopman_ground(family,**kwargs)
        self.assertTrue(verify_koopman_ground(family,model))
        return model

    def test_genuinely_nonlinear_128_bit_invariant_generated_processor(self):
        family=nonlinear_family()
        model=self._build(family)
        self.assertEqual(family.description.width,128)
        self.assertEqual(family.description.drain_horizon,4)
        self.assertGreater(len(family.coefficient_orbits),64)
        self.assertGreaterEqual(model.dimension,7)
        self.assertLessEqual(model.dimension,16)
        op=family.operators[0]
        x,y=(1<<1),(1<<2)
        self.assertEqual(op.apply(x)&1,0)
        self.assertEqual(op.apply(y)&1,0)
        self.assertEqual(op.apply(x|y)&1,1)
        initial=sum(1<<i for i in (1,2,3,4,6,7,64))
        observations=[]
        for t in (0,1,2,3,4,10**100+173):
            result=navigate_nonlinear(family,model,initial,(0,),t)
            self.assertTrue(verify_nonlinear_result(family,model,result))
            observations.append(int(result.observed_decimal))
            self.assertLessEqual(result.operator_products,2*t.bit_length())
        self.assertEqual(observations,[2,3,3,3,2,2])
        self.assertEqual((len(family.operators), len(family.description.observers)),(2,2))

    def test_exact_future_equivalence_exhaustive_moore_oracle(self):
        desc=small_description()
        proto=construct_polynomial_laws(desc,(0,))
        self.assertEqual(len(proto.coefficient_orbits),5)
        for first,second in ((0,0),(1,2),(4,8),(7,9),(13,31),(31,17)):
            family=small_family(first,second)
            ground=self._build(family,max_features=16)
            distinct=independent_moore_pair_oracle(family)
            for x in range(16):
                for y in range(16):
                    self.assertEqual((x,y) not in distinct,
                                     ground.project(x)==ground.project(y),
                                     (first,second,x,y))
            for state in (0,7,15):
                for word in ((),(0,),(1,),(0,1),(1,0)):
                    for time in (0,1,2,3,10**22+5):
                        result=navigate_nonlinear(family,ground,state,word,time)
                        self.assertTrue(verify_nonlinear_result(family,ground,result))

    def test_symmetry_equivariance_and_nontrivial_orbit(self):
        desc=PolynomialInvariants(8,(0,1,2,3,0,1,2,3),
            (4,5,6,7,0,1,2,3),
            ((1,),(2,),(3,),(),(5,),(6,),(7,),()),1,
            ((1,), (1<<4,)))
        proto=construct_polynomial_laws(desc,(0,))
        self.assertEqual(len(proto.coefficient_orbits),3)
        family=construct_polynomial_laws(desc,((1<<3)-1,))
        ground=self._build(family)
        for source_state in range(1<<8):
            permuted=0
            for i in range(8):
                if source_state & (1<<i):
                    permuted |= 1<<desc.symmetry[i]
            left=family.operators[0].apply(permuted)
            right=0
            for i in range(8):
                if family.operators[0].apply(source_state)&(1<<i):
                    right |= 1<<desc.symmetry[i]
            self.assertEqual(left,right)
            self.assertEqual(family.operators[0].apply(source_state)&(1<<3),0)
        self.assertTrue(verify_koopman_ground(family,ground))

    def test_refuse_all_hidden_future_features_are_needed(self):
        family=long_shift_family(14)
        with self.assertRaises(PolynomialBudget):
            construct_koopman_ground(family,max_features=8)
        model=self._build(family)
        self.assertEqual(model.dimension,14)
        for t in (0,1,7,14,10**40+13):
            result=navigate_nonlinear(family,model,1<<13,(0,),t)
            self.assertTrue(verify_nonlinear_result(family,model,result))

    def test_affine_subfamily_agrees_with_ncr006(self):
        desc=PolynomialInvariants(5,(0,1,2,3,4),(0,1,2,3,4),
             ((1,),(2,),(3,),(4,),()),1,((1,),))
        family=construct_polynomial_laws(desc,((1<<4)-1,0))
        model=self._build(family)
        affine_source=SymbolicSource(5,tuple(AffineMap(
            tuple(sum(poly) for poly in op.rows),0) for op in family.operators),
            (1,),'NCR-007-derived-affine-control')
        ref=construct_symbolic_quotient(affine_source)
        self.assertTrue(verify_symbolic_quotient(affine_source,ref))
        self.assertEqual(model.dimension,ref.dimension)
        for x in range(32):
            for y in range(32):
                self.assertEqual(model.project(x)==model.project(y),
                                 ref.project(x)==ref.project(y))

    def test_zero_dimensional_constant_observer(self):
        desc=PolynomialInvariants(4,(0,1,1,2),tuple(range(4)),
             ((1,2),(3,),(3,),()),2,((),))
        family=construct_polynomial_laws(desc,(31,))
        ground=self._build(family,max_features=0)
        self.assertEqual(ground.dimension,0)
        result=navigate_nonlinear(family,ground,7,(0,),10**40+13)
        self.assertTrue(verify_nonlinear_result(family,ground,result))
        self.assertEqual(result.observed_decimal,'0')

    def test_source_binding_and_rehashed_forgery(self):
        family=small_family(31,17)
        ground=self._build(family)
        other=small_family(17,31)
        self.assertFalse(verify_koopman_ground(other,ground))
        with self.assertRaisesRegex(ValueError,'source'):
            navigate_nonlinear(other,ground,7,(0,),1)
        matrices=list(ground.operator_rows)
        row=list(matrices[0])
        row[0] ^= 1
        matrices[0]=tuple(row)
        fake=replace(ground,operator_rows=tuple(matrices),digest='')
        fake=replace(fake,digest=sha256(canonical(fake.encode())).hexdigest())
        self.assertFalse(verify_koopman_ground(family,fake))
        receipt=navigate_nonlinear(family,ground,7,(0,1),10**30+9)
        fake_result=replace(receipt,observed_decimal='7',digest='')
        fake_result=replace(fake_result,digest=sha256(canonical(fake_result.encode())).hexdigest())
        self.assertFalse(verify_nonlinear_result(family,ground,fake_result))

    def test_canonical_packet_rejects_rogue_and_rehashed_semantics(self):
        family=small_family(31,17)
        ground=self._build(family)
        receipt=navigate_nonlinear(family,ground,7,(0,1),10**40+17)
        raw=encode_nonlinear(family,ground,receipt)
        self.assertEqual(decode_nonlinear(raw),(family,ground,receipt))
        for bad in (raw+b' ',raw.replace(b'"schema":',b'"extra":0,"schema":',1),
                    raw.replace(b'"family":',b'"family":{},"family":',1)):
            self.assertRaises(ValueError,decode_nonlinear,bad)
        doc=json.loads(raw)
        doc['data']['ground']['operator_rows'][0][0]='0'
        body=doc['data']['ground']
        body['sha256']=sha256(canonical({k:v for k,v in body.items() if k!='sha256'})).hexdigest()
        self.assertRaises(ValueError,decode_nonlinear,canonical(doc))

    def test_invalid_invariants_architecture_and_budget(self):
        base=small_description()
        for changes in ({'tiers':(0,0,1,2)},
                        {'tiers':(0,1,3,4)},
                        {'allowed_inputs':((1,2),(1,),(3,),())},
                        {'symmetry':(1,0,2,3)},
                        {'degree':4},
                        {'observers':((True,),)}):
            params={**base.__dict__,**changes}
            self.assertRaises(ValueError,PolynomialInvariants,**params)
        for bad in ((-1,), (1 << 5,), (True,), (), (0,)*9):
            self.assertRaises(ValueError,construct_polynomial_laws,base,bad)
        family=small_family(31,17)
        self.assertRaises(PolynomialBudget,construct_koopman_ground,family,
                          max_features=1)
        self.assertRaises(ValueError,construct_koopman_ground,family,
                          max_features=True)

    def test_bounded_word_initial_and_logical_coordinate(self):
        family=small_family(31,17)
        ground=self._build(family)
        for x,word,reps in ((16,(0,),1),(True,(0,),1),(1,(2,),1),
                            (1,(True,),1),(1,(0,),True),(1,(0,),-1),
                            (1,(0,),1<<4096)):
            self.assertRaises(ValueError,navigate_nonlinear,
                              family,ground,x,word,reps)


if __name__=='__main__':
    unittest.main()
