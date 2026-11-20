"""Adversarial and exhaustive finite tests for intention-first world construction."""
import itertools
import random
import unittest
from dataclasses import replace

from navigable_compute.possibility import (PossibilityAction,FactorContract,
    construct_world,certify_nonrealizable,NotRealizable,navigate_instance)
from navigable_compute.possibility_verify import verify_world,verify_obstruction
from possibility_demo import fixture,fixture_brief,run
from navigable_compute.brief_verify import verify_brief_world
from navigable_compute.brief import (EffectCase,ActionSketch,GroundBrief,compile_brief,construct_from_briefs)


def small_contract(table_a,table_b):
    return FactorContract(3,(0,1,2),(0,),(),(
        PossibilityAction('a',tuple((x,) for x in table_a)),
        PossibilityAction('b',tuple((x,) for x in table_b))))


def independent_policy_oracle(contract):
    """Actually enumerate every stationary program, not the attractor.

    Classify a policy by following its chosen transitions from every state,
    refusing cycles outside the terminal goal. Intended only for n<=3.
    """
    nonterminal=[s for s in range(contract.states) if s not in contract.goal]
    per_state=[[] for _ in range(contract.states)]
    for chosen in itertools.product(range(len(contract.actions)),repeat=len(nonterminal)):
        moves=dict(zip(nonterminal,chosen))
        for start in range(contract.states):
            seen=set();s=start;distance=0
            while s not in contract.goal and s not in seen:
                seen.add(s)
                ss=contract.actions[moves[s]].outcomes[s]
                if len(ss)!=1: raise AssertionError('oracle supports deterministic only')
                s=ss[0];distance+=1
            if s in contract.goal:
                per_state[start].append(distance)
    return tuple(min(vals) if vals else -1 for vals in per_state)


class PossibilityTests(unittest.TestCase):
    def test_brief_constructs_possibility_space_not_preexisting_program(self):
        spec=fixture_brief()
        self.assertEqual(compile_brief(spec),fixture())
        constructed=construct_from_briefs((spec,))
        self.assertEqual(constructed.digest,construct_world((fixture(),)).digest)
        self.assertTrue(verify_brief_world((spec,),constructed))
        altered=replace(spec,goal=(1,))
        self.assertFalse(verify_brief_world((altered,),constructed))
        with self.assertRaises(ValueError):
            ActionSketch('bad',(EffectCase(0,3,deltas=(-1,)),EffectCase(2,4,deltas=(1,))))
        with self.assertRaises(ValueError):
            compile_brief(replace(spec,actions=(ActionSketch('bad',(EffectCase(0,17,deltas=(-1,)),)),)))
        with self.assertRaises(ValueError):
            compile_brief(replace(spec,actions=(ActionSketch('bad',(EffectCase(0,1,absolute=(16,)),)),)))

    def test_global_product_without_enumeration(self):
        contracts=(fixture(),)*32
        instance=construct_world(contracts)
        self.assertTrue(verify_world(contracts,instance))
        self.assertTrue(verify_brief_world((fixture_brief(),)*32,instance))
        self.assertEqual(instance.global_states,2**128)
        self.assertEqual(instance.initial_states,15**32)
        self.assertEqual(instance.convergence_bound,12)
        self.assertGreater(instance.factored_optimal_laws,2**128)
        self.assertEqual(instance.observations_at(0),(tuple(range(15)),)*32)
        self.assertEqual(instance.observations_at(10**100+173),((0,),)*32)
        self.assertEqual(instance.observations_at(12),((0,),)*32)
        self.assertEqual(instance.digest,run()['certificate_sha256'])

    def test_nondeterministic_existence_not_guarantee(self):
        contract=FactorContract(3,(1,),(0,),(),(
            PossibilityAction('risky',((0,),(0,2),(2,))),
            PossibilityAction('reset',((0,),(0,),(2,))),))
        world=construct_world((contract,))
        self.assertTrue(verify_world((contract,),world))
        self.assertEqual(world.factors[0].action[1],1)
        self.assertEqual(world.observations_at(1),((0,),))

    def test_certified_obstruction_with_a_possible_but_not_guaranteed_path(self):
        spec=FactorContract(3,(1,),(0,),(),(
            PossibilityAction('risk',((0,),(0,2),(2,))),
            PossibilityAction('hold',((0,),(1,),(2,))),))
        with self.assertRaises(NotRealizable): construct_world((spec,))
        cert=certify_nonrealizable((spec,))
        self.assertTrue(verify_obstruction((spec,),cert))
        self.assertEqual(cert.trap,(1,2))
        self.assertFalse(verify_obstruction((spec,),replace(cert,trap=(2,))))
        self.assertFalse(verify_obstruction((spec,),replace(cert,witness_sha256='0'*64)))

    def test_refuse_unqualified_or_unrealizable(self):
        with self.assertRaises(ValueError):
            FactorContract(3,(1,),(0,),(),(PossibilityAction('invalid',((2,),(3,),())),))
        with self.assertRaises(ValueError):
            FactorContract(3,(1,),(0,),(),(PossibilityAction('invalid',((),(2,2),())),))
        with self.assertRaises(ValueError): certify_nonrealizable((fixture(),))
        with self.assertRaises(ValueError): construct_world((fixture(),),max_total_local_states=3)
        with self.assertRaises(ValueError): construct_world(())
        world=construct_world((fixture(),))
        with self.assertRaises(ValueError): world.observations_at(-1)
        with self.assertRaises(ValueError): world.observations_at(1<<5000)

    def test_navigate_qualified_processor_instances(self):
        specs=(fixture(),fixture())
        world=construct_world(specs)
        selected=[]
        for coordinate in (0,1,2,9,1003,world.factored_optimal_laws-1):
            r=navigate_instance(world,coordinate)
            self.assertTrue(verify_world(specs,r))
            self.assertEqual(r.instance_coordinate,coordinate)
            self.assertEqual(r.observations_at(10**40),((0,),(0,)))
            selected.append(tuple(x.action for x in r.factors))
        self.assertEqual(len(set(selected)),len(selected))
        with self.assertRaises(ValueError):navigate_instance(world,-1)
        with self.assertRaises(ValueError):navigate_instance(world,world.factored_optimal_laws)
        with self.assertRaises(ValueError):navigate_instance(replace(world,digest='0'*64),9)
        # Even a recomputed hash cannot conceal a wrong coordinate/policy relation.
        from hashlib import sha256
        from navigable_compute.possibility import _canonical
        bad=replace(world,instance_coordinate=1,digest='')
        bad=replace(bad,digest=sha256(_canonical(bad.encode())).hexdigest())
        self.assertFalse(verify_world(specs,bad))

    def test_multi_goal_only_reachable_subset_reported_at_large_time(self):
        spec=FactorContract(4,(2,),(0,1),(),(
            PossibilityAction('realize',((0,),(1,),(0,),(1,))),))
        w=construct_world((spec,))
        self.assertEqual(w.observations_at(10**100),((0,),))
        self.assertTrue(verify_world((spec,),w))

    def test_certificate_tampering_even_with_recomputed_hash(self):
        from hashlib import sha256
        from navigable_compute.possibility import _canonical
        cs=(fixture(),)
        r=construct_world(cs)
        bad_factor=replace(r.factors[0],rank=tuple([0]*16))
        bad=replace(r,factors=(bad_factor,),digest='')
        bad=replace(bad,digest=sha256(_canonical(bad.encode())).hexdigest())
        self.assertFalse(verify_world(cs,bad))
        self.assertFalse(verify_world(cs,replace(r,global_states=100)))
        self.assertFalse(verify_world(cs,replace(r,digest='0'*64)))

    def test_all_729_deterministic_three_state_worlds(self):
        total_ok=0;total_no=0
        for a in itertools.product(range(3),repeat=3):
            for b in itertools.product(range(3),repeat=3):
                contract=small_contract(a,b)
                oracle=independent_policy_oracle(contract)
                if all(r>=0 for r in oracle):
                    world=construct_world((contract,))
                    self.assertEqual(world.factors[0].rank,oracle)
                    self.assertTrue(verify_world((contract,),world))
                    total_ok+=1
                else:
                    with self.assertRaises(NotRealizable):construct_world((contract,))
                    obstruction=certify_nonrealizable((contract,))
                    self.assertTrue(verify_obstruction((contract,),obstruction))
                    total_no+=1
        self.assertEqual(total_ok+total_no,729)
        self.assertGreater(total_ok,0)
        self.assertGreater(total_no,0)


if __name__=='__main__': unittest.main()
