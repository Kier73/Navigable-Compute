"""NCR-006 tests: symbolic quotient proof, direct small-state oracles, refusal."""
from dataclasses import replace
from hashlib import sha256
import json
import random
import unittest

from navigable_compute.symbolic_quotient import (
    AffineMap, SymbolicSource, construct_symbolic_quotient,
    navigate_symbolic, canonical)
from navigable_compute.symbolic_quotient_verify import (
    verify_symbolic_quotient, verify_symbolic_result)
from navigable_compute.symbolic_quotient_wire import encode_symbolic, decode_symbolic
from navigable_compute.preclosure_language import (
    GeneratorObservation, construct_preclosure_language)
from navigable_compute.preclosure_verify import verify_preclosure_language
from navigable_compute.observer_native import construct_native_algebra
from navigable_compute.observer_native_verify import verify_native_algebra


def large_hidden_source(width=128, observed=3):
    """Arbitrary hidden states and nontrivial hidden evolution, no backflow."""
    maps = []
    hidden = width - observed
    for g in range(observed):
        shift = [1, 7, 13][g] % hidden
        rows = [1 << i for i in range(width)]
        for i in range(observed, width):
            rows[i] = 1 << (observed + ((i - observed + shift) % hidden))
        rows[observed] ^= 1 << ((g + 1) % observed)
        maps.append(AffineMap(tuple(rows), (1 << g) | (1 << (observed + shift))))
    return SymbolicSource(width, tuple(maps),
                          tuple(1 << i for i in range(observed)),
                          'synthetic-hidden-coupled-no-backflow')


def shift_source(width):
    """A single visible bit eventually reveals all bits under repeated shifts."""
    rows = tuple((1 << (i + 1)) if i + 1 < width else 0
                 for i in range(width))
    return SymbolicSource(width, (AffineMap(rows, 0),), (1,),
                          'shift-reveals-the-whole-state')


def direct_pair_oracle(source):
    """Independent exhaustive small-state Moore pair distinguishability."""
    count = 1 << source.width
    outputs = [source.observe(x) for x in range(count)]
    transitions = [tuple(g.apply(x) for x in range(count))
                   for g in source.generators]
    distinct = {(a, b) for a in range(count) for b in range(count)
                if outputs[a] != outputs[b]}
    while True:
        additions = {(a, b) for a in range(count) for b in range(count)
                     if (a,b) not in distinct and
                     any((table[a], table[b]) in distinct for table in transitions)}
        if not additions:
            return distinct
        distinct.update(additions)


class SymbolicQuotientTests(unittest.TestCase):
    def _build(self, source, **kw):
        result = construct_symbolic_quotient(source,**kw)
        self.assertTrue(verify_symbolic_quotient(source,result))
        return result

    def test_128_bit_implicit_state_space_three_bit_future(self):
        source = large_hidden_source()
        quotient = self._build(source)
        self.assertEqual((source.width, quotient.dimension), (128,3))
        self.assertEqual(quotient.basis_rows, (4,2,1))
        self.assertEqual(tuple(quotient.observer_decoder), (4,2,1))
        x = (1 << 120) | 4
        word = (0,1,2,0,2)  # net visible toggle is bit 1
        for reps in (0,1,2,3,10**100+173):
            result = navigate_symbolic(source,quotient,x,word,reps)
            self.assertTrue(verify_symbolic_result(source,quotient,result))
            self.assertEqual(int(result.observed_decimal),
                             4 ^ (2 if reps % 2 else 0))
            self.assertEqual(result.source_projection_decimal,str(quotient.project(x)))
            self.assertLessEqual(result.operator_products,
                                 2*reps.bit_length())
        self.assertEqual(len(source.generators),3)

    def test_all_hidden_bits_relevant_refuse_compression(self):
        source = shift_source(48)
        with self.assertRaisesRegex(ValueError,'dimension budget'):
            construct_symbolic_quotient(source,max_dimension=8)
        quotient = self._build(source)
        self.assertEqual(quotient.dimension,48)
        for t in (0,1,9,48,10**30+9):
            r = navigate_symbolic(source,quotient,1<<35,(0,),t)
            self.assertTrue(verify_symbolic_result(source,quotient,r))

    def test_constant_observer_zero_dimensional_ground(self):
        source = SymbolicSource(12,(AffineMap(tuple(1 << i for i in range(12)),27),),
                                (0,),'constant-output-ground')
        quotient = self._build(source,max_dimension=0)
        self.assertEqual(quotient.dimension,0)
        self.assertEqual(quotient.basis_rows,())
        for t in (0,1,10**100+173):
            result = navigate_symbolic(source,quotient,332,(0,),t)
            self.assertTrue(verify_symbolic_result(source,quotient,result))
            self.assertEqual(result.observed_decimal,'0')

    def test_noncommuting_affine_quotient_and_ncr004_bridge(self):
        n = 4
        swap = AffineMap((2,1,4,8),0)
        toggle = AffineMap(tuple(1 << i for i in range(n)),1)
        source = SymbolicSource(n,(swap,toggle),(1,2),'noncommuting-affine')
        quotient = self._build(source)
        self.assertEqual(quotient.dimension,2)
        self.assertNotEqual(quotient.operators[0].apply(
            quotient.operators[1].apply(0)),
            quotient.operators[1].apply(quotient.operators[0].apply(0)))
        for word in ((0,1),(1,0),(0,1,0,1)):
            r = navigate_symbolic(source,quotient,12,word,10**60+17)
            self.assertTrue(verify_symbolic_result(source,quotient,r))
        states = tuple(range(1 << n))
        explicit = GeneratorObservation(
            states,tuple(tuple(g.apply(x) for x in states) for g in source.generators),
            states,tuple(f'observation-{source.observe(x)}' for x in states),
            'exact-source-enumeration-control-only')
        lang = construct_preclosure_language(explicit,max_quotient_ops=256)
        self.assertTrue(verify_preclosure_language(explicit,lang))
        self.assertEqual(len(lang.state_classes),1 << quotient.dimension)
        representation = construct_native_algebra(explicit,lang)
        self.assertEqual(representation.kind,'onehot')
        self.assertTrue(verify_native_algebra(explicit,lang,representation))

    def test_exhaustive_pair_oracle_random_affine_families(self):
        rng = random.Random(250604)
        for width in (2,3,4,5):
            for case in range(5):
                matrix = []
                for _ in range(2):
                    rows = tuple(rng.randrange(1 << width) for _ in range(width))
                    matrix.append(AffineMap(rows,rng.randrange(1 << width)))
                observations = tuple(rng.randrange(1 << width) for _ in range(2))
                source = SymbolicSource(width,tuple(matrix),observations,
                                        f'exhaustive-{width}-{case}')
                quotient = self._build(source)
                distinct = direct_pair_oracle(source)
                for a in range(1 << width):
                    for b in range(1 << width):
                        self.assertEqual((a,b) not in distinct,
                                         quotient.project(a)==quotient.project(b))

    def test_source_binding_and_semantic_forgery(self):
        source = large_hidden_source(14)
        q = self._build(source)
        changed = replace(source,observer_rows=(1,2,8))
        self.assertFalse(verify_symbolic_quotient(changed,q))
        with self.assertRaisesRegex(ValueError,'source'):
            navigate_symbolic(changed,q,0,(0,),1)
        badops = list(q.operators)
        badops[0] = AffineMap(badops[0].rows,badops[0].bias^1)
        forged = replace(q,operators=tuple(badops),digest='')
        forged = replace(forged,digest=sha256(canonical(forged.encode())).hexdigest())
        self.assertFalse(verify_symbolic_quotient(source,forged))
        result = navigate_symbolic(source,q,0,(0,1),2)
        forged_result = replace(result,observed_decimal='7',digest='')
        forged_result = replace(forged_result,digest=sha256(canonical(forged_result.encode())).hexdigest())
        self.assertFalse(verify_symbolic_result(source,q,forged_result))

    def test_strict_portable_wire_and_rehashed_tampering(self):
        source = large_hidden_source(18)
        quotient = self._build(source)
        result = navigate_symbolic(source,quotient,(1<<17)|4,(0,2),10**25+1)
        encoded = encode_symbolic(source,quotient,result)
        self.assertEqual(decode_symbolic(encoded),(source,quotient,result))
        for wrong in (encoded+b' ',encoded.replace(b'"schema":',b'"schema":"x","schema":',1),
                      encoded.replace(b'"source":',b'"source":{},"source":',1)):
            self.assertRaises(ValueError,decode_symbolic,wrong)
        doc = json.loads(encoded)
        doc['data']['quotient']['observer_decoder'][0]='0'
        qpacket = doc['data']['quotient']
        qpacket['sha256'] = sha256(canonical({k:v for k,v in qpacket.items() if k!='sha256'})).hexdigest()
        self.assertRaises(ValueError,decode_symbolic,canonical(doc))

    def test_invalid_binary_source_and_navigation(self):
        n=4
        identity=AffineMap(tuple(1<<i for i in range(n)),0)
        for width, generators, outputs in (
            (0,(identity,),(1,)),
            (4,(AffineMap((1,2,4,16),0),),(1,)),
            (4,(identity,),(16,)),
            (4,(identity,),(True,))):
            self.assertRaises(ValueError,SymbolicSource,width,generators,outputs)
        source=SymbolicSource(4,(identity,),(1,))
        q=self._build(source)
        for x,word,t in ((True,(0,),1),(16,(0,),1),(0,(2,),1),
                         (0,(True,),1),(0,(0,),-1),(0,(0,),True),
                         (0,(0,),1<<4096)):
            self.assertRaises(ValueError,navigate_symbolic,source,q,x,word,t)


if __name__=='__main__':
    unittest.main()
