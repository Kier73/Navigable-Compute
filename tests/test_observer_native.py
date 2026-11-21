"""NCR-005 finite exact algebra and independent adversarial verification."""
from dataclasses import replace
from hashlib import sha256
import json
import unittest

from invariant_demo import grid
from preclosure_demo import cyclic_system
from navigable_compute.invariant_space import canonical
from navigable_compute.preclosure_language import (GeneratorObservation,
    construct_preclosure_language, generators_from_invariants)
from navigable_compute.observer_native import (construct_native_algebra, navigate_native)
from navigable_compute.observer_native_verify import verify_native_algebra, verify_native_result
from navigable_compute.observer_native_wire import encode_native, decode_native


def translation_group(bits=2):
    q = 1 << bits
    states = tuple(range(q))
    generators = tuple(tuple(i ^ (1 << b) for i in states) for b in range(bits))
    return GeneratorObservation(states, generators, (0,), tuple(f'label-{i}' for i in states),
                                'declared-elementary-abelian-fixture')


class ObserverNativeTests(unittest.TestCase):
    def _build(self, src):
        lang = construct_preclosure_language(src)
        ground = construct_native_algebra(src, lang)
        self.assertTrue(verify_native_algebra(src, lang, ground))
        return lang, ground

    def test_hidden_s8_shortcut_is_character_dimension_one(self):
        src = cyclic_system(symmetry_group=True, output_mode='visible')
        lang, alg = self._build(src)
        self.assertEqual((len(src.states), len(lang.state_classes),
                          len(lang.operators), alg.kind, alg.dimension),
                         (16, 2, 2, 'character', 1))
        for time in (0, 1, 2, 97, 10**100+173):
            event = navigate_native(src, lang, alg, 0, (0,1,2), time)
            self.assertTrue(verify_native_result(src, lang, alg, event))
            self.assertLess(event.operator_products, 2*max(1, time.bit_length()))

    def test_character_basis_and_exact_walsh_outputs(self):
        src = translation_group(3)
        lang, alg = self._build(src)
        self.assertEqual(alg.kind, 'character')
        self.assertEqual(alg.dimension, 3)
        self.assertEqual(alg.basis_generators, (0,1,2))
        self.assertEqual(len(alg.readout), 8)
        for op in range(len(lang.operators)):
            for state in range(8):
                self.assertEqual(alg.act(op, alg.state_vectors[state]),
                                 alg.state_vectors[lang.operators[op][state]])
        for i, vec in enumerate(alg.state_vectors):
            self.assertEqual(alg.read(vec), f'label-{i}')
        for i in range(3):
            event = navigate_native(src, lang, alg, 0, (i,), 10**120+3)
            self.assertTrue(verify_native_result(src, lang, alg, event))

    def test_invariant_generated_noninvertible_fallback(self):
        desc = grid(5)
        source = generators_from_invariants(desc, tuple('goal' if i in desc.goal
                             else 'not-goal' for i in range(len(desc.coordinates))))
        lang, alg = self._build(source)
        self.assertEqual(alg.kind, 'onehot')
        self.assertEqual(alg.dimension, len(lang.state_classes))
        self.assertLess(alg.dimension, len(source.states))
        for n in (0, 1, 2, 10**100+173):
            event = navigate_native(source, lang, alg, source.initial[0], (0,), n)
            self.assertTrue(verify_native_result(source, lang, alg, event))

    def test_noncommuting_and_nonbijective_fallback(self):
        for gens in (((1,2,0), (1,0,2)), ((0,0,2), (0,1,1))):
            source = GeneratorObservation((0,1,2), gens, (0,1,2),
                                          ('a','b','c'))
            lang, alg = self._build(source)
            self.assertEqual(alg.kind, 'onehot')
            for n in (0, 1, 77):
                event = navigate_native(source, lang, alg, 0, (0,1), n)
                self.assertTrue(verify_native_result(source, lang, alg, event))

    def test_rehashed_semantic_forgery_is_refused(self):
        source = translation_group(2)
        lang, alg = self._build(source)
        vectors = list(alg.state_vectors)
        vectors[1] = vectors[0]
        fake = replace(alg, state_vectors=tuple(vectors), digest='')
        fake = replace(fake, digest=sha256(canonical(fake.encode())).hexdigest())
        self.assertFalse(verify_native_algebra(source, lang, fake))
        readout = list(alg.readout)
        readout[0] = (readout[0]+1) % alg.prime
        fake = replace(alg, readout=tuple(readout), digest='')
        fake = replace(fake, digest=sha256(canonical(fake.encode())).hexdigest())
        self.assertFalse(verify_native_algebra(source, lang, fake))
        output = navigate_native(source, lang, alg, 0, (0,1), 10**22+1)
        fake_result = replace(output, output='fabricated', digest='')
        fake_result = replace(fake_result, digest=sha256(canonical(fake_result.encode())).hexdigest())
        self.assertFalse(verify_native_result(source, lang, alg, fake_result))

    def test_identity_scope_and_budget_refusal(self):
        source = translation_group(2)
        lang, alg = self._build(source)
        self.assertRaises(ValueError, construct_native_algebra, source, lang, max_dimension=1)
        self.assertRaises(ValueError, navigate_native, source, lang, alg, 3, (0,), 4)
        self.assertRaises(ValueError, navigate_native, source, lang, alg, 0, (0,), True)
        self.assertRaises(ValueError, navigate_native, source, lang, alg, 0, (0,), -1)
        self.assertRaises(ValueError, navigate_native, source, lang, alg, 0, (0,), 1<<4096)
        changed = replace(source, outputs=('same',)*4)
        self.assertFalse(verify_native_algebra(changed, lang, alg))
        self.assertRaises(ValueError, navigate_native, changed, lang, alg, 0, (0,), 1)

    def test_canonical_end_to_end_wire_and_rehashed_forgery(self):
        source = translation_group(2)
        lang, alg = self._build(source)
        result = navigate_native(source, lang, alg, 0, (0,1), 10**60+17)
        raw = encode_native(source, lang, alg, result)
        self.assertEqual(decode_native(raw), (source,lang,alg,result))
        for altered in (raw+b' ',
                        raw.replace(b'"schema":', b'"surplus":0,"schema":',1),
                        raw.replace(b'"data":', b'"data":{},"data":',1)):
            self.assertRaises(ValueError, decode_native, altered)
        tampered = json.loads(raw)
        tampered['data']['algebra']['readout'][0] = 99
        a = tampered['data']['algebra']
        a['sha256'] = sha256(canonical({k:v for k,v in a.items() if k != 'sha256'})).hexdigest()
        self.assertRaises(ValueError, decode_native, canonical(tampered))

    def test_adversarial_random_distinct_state_actions(self):
        # Complete checked finite data, including mixed extra generators.
        for bits in (1,2,3,4):
            src = translation_group(bits)
            lang, alg = self._build(src)
            for n in range(13):
                event = navigate_native(src, lang, alg, 0,
                                        tuple(range(bits)), n)
                self.assertTrue(verify_native_result(src, lang, alg, event))


if __name__ == '__main__':
    unittest.main()
