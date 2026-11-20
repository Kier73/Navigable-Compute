"""NCR-004 finite exact preclosure language & temporal refinement regressions."""
from dataclasses import replace
from hashlib import sha256
import json
import unittest

from invariant_demo import grid
from navigable_compute.invariant_space import canonical
from navigable_compute.self_space import build_endogenous_processor, AlgebraBudgetExceeded
from navigable_compute.semantic_language import ObservationContract, derive_semantic_language
from navigable_compute.preclosure_language import (
    GeneratorObservation, PreclosureLanguage, LanguageRevision, QuotientExplosion,
    generators_from_invariants, construct_preclosure_language, derive_language_revision)
from navigable_compute.preclosure_verify import verify_preclosure_language, verify_language_revision
from navigable_compute.preclosure_wire import (encode_preclosure,decode_preclosure,
    encode_revision,decode_revision)


def synthetic(hidden=8, *, permute=False, labels='visible'):
    states=tuple(range(2*hidden))
    toggle=tuple((1-v)*hidden+i for v in range(2) for i in range(hidden))
    cycle=tuple(v*hidden+(i+1)%hidden for v in range(2) for i in range(hidden))
    maps=(toggle,cycle)
    if permute:
        swap=tuple(v*hidden+(1-i if i in (0,1) else i)
                   for v in range(2) for i in range(hidden))
        maps+=(swap,)
    def label(i):
        v,h=divmod(i,hidden)
        if labels=='visible':return str(v)
        if labels=='parity':return f'{v}:{h%2}'
        if labels=='exact':return f'{v}:{h}'
        if labels=='constant':return 'constant'
        raise ValueError(labels)
    return GeneratorObservation(states,maps,(0,),tuple(label(i) for i in states),
                                'synthetic-independent-hidden-permutation')


def with_digest(language, **kwargs):
    x=replace(language,**kwargs,digest='')
    return replace(x,digest=sha256(canonical(x.encode())).hexdigest())


def revision_digest(rev, **kwargs):
    x=replace(rev,**kwargs,digest='')
    return replace(x,digest=sha256(canonical(x.encode())).hexdigest())


class PreclosureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.five=build_endogenous_processor(grid(5))
        cls.goal=tuple('goal' if cls.five.source.ranks[s]==0 else 'not-goal'
                       for s in cls.five.states)
        cls.base=generators_from_invariants(grid(5),cls.goal)
        cls.derived=construct_preclosure_language(cls.base)

    def test_nci_generated_without_source_closure(self):
        self.assertEqual(len(self.derived.operators),5)
        self.assertEqual(len(self.derived.state_classes),5)
        self.assertEqual(len(self.five.operators),29)
        self.assertTrue(verify_preclosure_language(self.base,self.derived))
        self.assertEqual(self.derived.dynamics_sha256,self.base.dynamics_digest)

    def test_all_three_sizes_against_existing_monoid_construction(self):
        for side in (3,5,7):
            source=build_endogenous_processor(grid(side))
            for name in ('goal','exact','constant'):
                labels=(tuple('goal' if source.source.ranks[s]==0 else 'not-goal'
                        for s in source.states) if name=='goal' else
                        tuple(str(s) for s in source.states) if name=='exact' else
                        ('same',)*len(source.states))
                system=generators_from_invariants(grid(side),labels)
                new=construct_preclosure_language(system)
                old=derive_semantic_language(source,ObservationContract(source.states,labels))
                self.assertEqual(len(new.operators),len(old.classes))
                self.assertTrue(verify_preclosure_language(system,new))
                for i in range(len(source.generator_opcodes)):
                    for j in range(len(source.generator_opcodes)):
                        word=(i,j)
                        for state in source.states:
                            self.assertEqual(new.observe(system,word,state),old.observe(source,word,state))

    def test_partial_context_source_monoid_avoided(self):
        src=self.five
        labels=tuple('visible' if i in (0,2) else 'hidden' for i in range(len(src.states)))
        system=generators_from_invariants(grid(5),labels,initial=(src.states[0],))
        design=construct_preclosure_language(system)
        old=derive_semantic_language(src,ObservationContract((src.states[0],),labels))
        self.assertEqual(len(design.operators),6)
        self.assertEqual(len(old.classes),6)
        self.assertTrue(verify_preclosure_language(system,design))

    def test_hidden_symmetric_group_80640_operations_never_enumerated(self):
        # 8-cycle + (0 1) generate S_8 (8!); independent toggle C2
        # creates the *source* algebra S8 x C2 = 80640 elements.
        source=synthetic(8,permute=True)
        design=construct_preclosure_language(source,max_quotient_ops=2)
        self.assertEqual(len(design.state_classes),2)
        self.assertEqual(len(design.operators),2)
        self.assertTrue(verify_preclosure_language(source,design))
        self.assertEqual(design.observe(source,(2,1,2,0),0),'1')
        self.assertRaises(QuotientExplosion,construct_preclosure_language,source,
                          max_source_states=8)
        self.assertEqual(design.power_code((0,1,2),10**100+173)[0],1)

    def test_independent_full_s8_group_census(self):
        # This costly control is only an oracle in the test. Production
        # construction never expands these source operators.
        source=synthetic(8,permute=True)
        identity=tuple(range(len(source.states)))
        states={identity}; queue=[identity]
        for op in queue:
            for gen in source.generators:
                next_op=tuple(gen[op[i]] for i in identity)
                if next_op not in states:
                    states.add(next_op);queue.append(next_op)
        self.assertEqual(len(states),80640)
        del states,queue
        small=construct_preclosure_language(source)
        self.assertEqual(len(small.operators),2)

    def test_randomized_exhaustive_source_context_oracle(self):
        # Compare directly with *all* two-sided contexts in small random
        # source monoids, without invoking NCR-003 or the NCR-004 verifier.
        from random import Random
        rng=Random(11092026)
        tested=0
        for _ in range(70):
            n=4
            gens=tuple(tuple(rng.randrange(n) for _ in range(n)) for _ in range(2))
            initial=tuple(i for i in range(n) if rng.randrange(2)) or (0,)
            labels=tuple('c'+str(rng.randrange(3)) for _ in range(n))
            system=GeneratorObservation(tuple(range(n)),gens,initial,labels,'random-finite-oracle')
            identity=tuple(range(n)); source_ops=[identity]; words=[()];seen={identity}
            for idx,op in enumerate(source_ops):
                for g_idx,g in enumerate(gens):
                    after=tuple(g[op[i]] for i in range(n))
                    if after not in seen:
                        seen.add(after);source_ops.append(after);words.append(words[idx]+(g_idx,))
                    if len(source_ops)>40:
                        break
                if len(source_ops)>40:
                    break
            if len(source_ops)>40:
                continue
            language=construct_preclosure_language(system)
            signatures=[]
            for middle in source_ops:
                sig=[]
                for prefix in source_ops:
                    for suffix in source_ops:
                        for start in initial:
                            sig.append(labels[prefix[middle[suffix[start]]]])
                signatures.append(tuple(sig))
            codes=tuple(language.word_code(w) for w in words)
            for a in range(len(source_ops)):
                for b in range(len(source_ops)):
                    self.assertEqual(signatures[a]==signatures[b],codes[a]==codes[b])
            self.assertEqual(len(set(signatures)),len(language.operators))
            self.assertTrue(verify_preclosure_language(system,language))
            tested+=1
        self.assertGreaterEqual(tested,30)

    def test_mathematical_revisions_without_restarting_ground_identity(self):
        a=synthetic(labels='visible')
        b=synthetic(labels='parity')
        c=synthetic(labels='exact')
        old=construct_preclosure_language(a)
        middle=construct_preclosure_language(b)
        new=construct_preclosure_language(c)
        self.assertEqual([x.instruction_count for x in (old,middle,new)],[2,4,16])
        for left,first,right,second in ((a,old,b,middle),(b,middle,c,new)):
            relation=derive_language_revision(left,first,right,second)
            self.assertTrue(verify_language_revision(left,first,right,second,relation))
            self.assertEqual(len(relation.new_to_old_opcodes),second.instruction_count)
            self.assertIsNotNone(relation.lost_reverse_pair)
            self.assertEqual(second.dynamics_sha256,first.dynamics_sha256)
            self.assertEqual(decode_revision(encode_revision(left,first,right,second,relation))[-1],relation)

    def test_changed_law_requires_new_ground_epoch(self):
        old=synthetic()
        a=construct_preclosure_language(old)
        shifted=replace(old,generators=old.generators[::-1])
        b=construct_preclosure_language(shifted)
        self.assertRaises(ValueError,derive_language_revision,old,a,shifted,b)

    def test_unqualified_downgrade_does_not_preserve_information(self):
        fine=synthetic(labels='exact')
        coarse=synthetic(labels='visible')
        a=construct_preclosure_language(fine)
        b=construct_preclosure_language(coarse)
        self.assertRaises(ValueError,derive_language_revision,fine,a,coarse,b)

    def test_observation_change_that_merges_prior_labels_refused(self):
        old=synthetic(labels='visible')
        new=replace(old,outputs=tuple(str(i%2) for i in old.states))
        a=construct_preclosure_language(old)
        b=construct_preclosure_language(new)
        self.assertRaises(ValueError,derive_language_revision,old,a,new,b)

    def test_contexts_over_all_reachable_before_and_after(self):
        sys=synthetic(4,permute=True,labels='parity')
        l=construct_preclosure_language(sys)
        # Exhaustive *source* group for the small 4-state hidden domain,
        # independent of the quotient builder. Group order 2*4! = 48.
        identity=tuple(range(len(sys.states)))
        source_ops={identity}
        queue=[identity]
        for trans in queue:
            for gen in sys.generators:
                other=tuple(gen[trans[i]] for i in identity)
                if other not in source_ops:
                    source_ops.add(other);queue.append(other)
        self.assertEqual(len(source_ops),48)
        for left in source_ops:
            for right in source_ops:
                for a,b in zip(sys.states,sys.states[1:]):
                    i=sys.states.index(a);j=sys.states.index(b)
                    if l.state_class_of[i]==l.state_class_of[j]:
                        self.assertEqual(sys.outputs[left[right[i]]],
                                         sys.outputs[left[right[j]]])
        self.assertTrue(verify_preclosure_language(sys,l))

    def test_independent_verifier_rejects_rehashed_false_maps(self):
        q=self.derived
        corruptions=(
            with_digest(q,operators=q.operators[:-1]),
            with_digest(q,state_class_of=(0,)*len(q.state_class_of)),
            with_digest(q,quotient_generators=((0,1,2,3,4),)+q.quotient_generators[1:]),
            with_digest(q,multiplication=((0,)*5,)*5),
            with_digest(q,normal_forms=((),)*5),
            with_digest(q,state_refinements=q.state_refinements+1),
            with_digest(q,dynamics_sha256='0'*64),
        )
        for forged in corruptions:
            with self.subTest(kind=forged):
                self.assertFalse(verify_preclosure_language(self.base,forged))

    def test_revision_rehashed_false_mapping_refused(self):
        a=synthetic(labels='visible'); b=synthetic(labels='parity')
        l=construct_preclosure_language(a); m=construct_preclosure_language(b)
        r=derive_language_revision(a,l,b,m)
        for forged in (
            revision_digest(r,new_to_old_opcodes=(0,)*len(r.new_to_old_opcodes)),
            revision_digest(r,new_to_old_states=(0,)*len(r.new_to_old_states)),
            revision_digest(r,lost_reverse_pair=None),
        ):
            self.assertFalse(verify_language_revision(a,l,b,m,forged))

    def test_canonical_wire_rejects_duplicate_and_unknown_fields(self):
        source=synthetic(labels='visible')
        l=construct_preclosure_language(source)
        wire=encode_preclosure(source,l)
        self.assertEqual(decode_preclosure(wire),(source,l))
        for raw in (wire+b' ',wire.replace(b'"schema":',b'"extra":1,"schema":'),
                    wire.replace(b'"source":',b'"source":{},"source":')):
            self.assertRaises(ValueError,decode_preclosure,raw)
        doc=json.loads(wire)
        doc['data']['language']['operators'][0][0]=1
        doc['data']['language']['sha256']=sha256(canonical({k:v for k,v in doc['data']['language'].items() if k!='sha256'})).hexdigest()
        self.assertRaises(ValueError,decode_preclosure,canonical(doc))

    def test_source_changes_are_not_silently_accepted(self):
        src=synthetic()
        l=construct_preclosure_language(src)
        other=replace(src,outputs=('same',)*len(src.states))
        self.assertRaises(ValueError,l.observe,other,(0,),0)
        self.assertRaises(ValueError,l.observe,src,(0,),12)

    def test_canonical_invalid_input_and_source_budget(self):
        src=synthetic()
        for val in (True,0,-1,4097):
            self.assertRaises(ValueError,construct_preclosure_language,src,max_quotient_ops=val)
        for malformed in (
            {'states': tuple(reversed(src.states))},
            {'initial': (0,0)},
            {'generators': ((True,)*len(src.states),)},
            {'outputs': ('same',)},
        ):
            with self.subTest(malformed=malformed):
                params={**src.__dict__,**malformed}
                self.assertRaises(ValueError,GeneratorObservation,**params)

    def test_time_coordinate_log_products(self):
        src=synthetic(labels='visible');lang=construct_preclosure_language(src)
        far=10**150+7
        code,cost=lang.power_code((0,1),far)
        self.assertEqual(code,1)
        self.assertLess(cost,2*far.bit_length())
        for wrong in (True,-1,1<<4096):
            self.assertRaises(ValueError,lang.power_code,(0,),wrong)
        for wrong in ((-1,),(len(src.generators),),(True,)):
            self.assertRaises(ValueError,lang.word_code,wrong)

if __name__=='__main__':
    unittest.main()
