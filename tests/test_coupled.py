"""NCP-002 regression: construct safe joint processors from coupled intentions."""
from dataclasses import replace
from hashlib import sha256
import json
import unittest

from navigable_compute.brief import (
    ActionSketch,EffectCase,GroundBrief,construct_from_briefs,
)
from navigable_compute.coupled import (
    CoupledBrief,SharedResource,CoupledNotRealizable,
    construct_coupled,navigate_coupled,certify_coupled_obstruction,
)
from navigable_compute.coupled_verify import (
    verify_coupled,verify_coupled_obstruction,
)


def local_brief(*,nondeterministic=False):
    """Describe a machine that can advance or hold; neither program is supplied."""
    return GroundBrief(
        3,(2,),(0,),(),(
            ActionSketch('advance',(
                EffectCase(1,3,deltas=(-1,0) if nondeterministic else (-1,)),
            )),
            ActionSketch('hold',(EffectCase(0,3,deltas=(0,)),)),
        ),
    )


def coupling(*,capacity=1,nondeterministic=False,unsafe_joint=()):
    """A and B share one actuating resource, rather than independent processors."""
    a=local_brief(nondeterministic=nondeterministic)
    return CoupledBrief(
        (a,a),(SharedResource('shared-actuator',capacity,
                            ((0,'advance'),(1,'advance'))),),
        unsafe_joint,
    )


class CoupledPossibilityTests(unittest.TestCase):
    def test_shared_capacity_changes_the_constructed_processor(self):
        """Two local 2-step plans require four steps with one shared actuator."""
        source=coupling()
        alone=construct_from_briefs(source.factors)
        self.assertEqual(alone.convergence_bound,2)
        candidate=construct_coupled(source)
        self.assertTrue(verify_coupled(source,candidate))
        self.assertEqual(len(candidate.states),9)
        self.assertEqual(candidate.worst_steps,4)
        self.assertGreater(candidate.optimal_laws,1)
        self.assertEqual(candidate.observations_at(0),((2,2),))
        self.assertEqual(candidate.observations_at(1),((1,2),))
        self.assertEqual(candidate.observations_at(10**100+173),((0,0),))
        self.assertEqual(candidate.packet()['sha256'],candidate.digest)
        self.assertNotEqual(candidate.digest,alone.digest)

    def test_navigate_distinct_joint_machine_without_policy_enumeration(self):
        """Mixed-radix coordinates select concrete different safe action tables."""
        source=coupling()
        canonical=construct_coupled(source)
        selected=navigate_coupled(canonical,1)
        self.assertTrue(verify_coupled(source,selected))
        self.assertNotEqual(canonical.policy,selected.policy)
        self.assertNotEqual(canonical.digest,selected.digest)
        self.assertEqual(selected.observations_at(10**80+1),((0,0),))
        self.assertEqual(selected.instance_coordinate,1)
        with self.assertRaises(ValueError):
            navigate_coupled(canonical,canonical.optimal_laws)
        with self.assertRaises(ValueError):
            selected.observations_at(-1)

    def test_joint_unsafe_state_requires_a_qualifying_global_path(self):
        """Independent local goals do not license crossing forbidden joint states."""
        source=coupling(unsafe_joint=((1,1),))
        candidate=construct_coupled(source)
        self.assertTrue(verify_coupled(source,candidate))
        self.assertEqual(candidate.observations_at(10**20),((0,0),))
        for time in range(8):
            self.assertNotIn((1,1),candidate.observations_at(time))

    def test_shared_capacity_zero_yields_verified_adversarial_trap(self):
        """Existential unqualified paths are not a winning controller."""
        source=coupling(capacity=0)
        with self.assertRaises(CoupledNotRealizable):
            construct_coupled(source)
        witness=certify_coupled_obstruction(source)
        self.assertTrue(verify_coupled_obstruction(source,witness))
        self.assertIn((2,2),witness.trapped_initial)
        self.assertNotIn((0,0),witness.trap)
        self.assertNotEqual(witness.digest,witness.source_digest)
        with self.assertRaises(ValueError):
            certify_coupled_obstruction(coupling())

    def test_nondeterminism_is_adversarial_not_cooperative(self):
        """An advance operation that may remain in place does not ensure arrival."""
        source=coupling(nondeterministic=True)
        with self.assertRaises(CoupledNotRealizable):
            construct_coupled(source)
        self.assertTrue(verify_coupled_obstruction(
            source,certify_coupled_obstruction(source),
        ))

    def test_checker_refuses_source_and_witness_tampering_even_rehashed(self):
        """A recomputed digest is not a source-bound mathematical proof."""
        source=coupling()
        world=construct_coupled(source)
        mutated_rank=list(world.rank)
        index=world.states.index((2,2))
        mutated_rank[index]=3
        altered=replace(world,rank=tuple(mutated_rank),digest='')
        altered=replace(altered,digest=sha256(
            json.dumps(altered.encode(),sort_keys=True,
                       separators=(',',':'),ensure_ascii=True).encode(),
        ).hexdigest())
        self.assertFalse(verify_coupled(source,altered))
        with self.assertRaises(ValueError):
            navigate_coupled(altered,0)
        changed_resource=replace(source,resources=(
            SharedResource('shared-actuator',2,((0,'advance'),(1,'advance'))),
        ))
        self.assertFalse(verify_coupled(changed_resource,world))
        failed=coupling(capacity=0)
        witness=certify_coupled_obstruction(failed)
        invalid=replace(witness,trap=((0,0),))
        invalid=replace(invalid,digest=sha256(
            json.dumps(invalid.encode(),sort_keys=True,
                       separators=(',',':'),ensure_ascii=True).encode(),
        ).hexdigest())
        self.assertFalse(verify_coupled_obstruction(failed,invalid))

    def test_refuse_ambiguous_resource_and_unsound_source(self):
        """Missing action identifiers, excluded origins and excess product size fail."""
        with self.assertRaises(ValueError):
            CoupledBrief((local_brief(),local_brief()),
                         (SharedResource('shared',1,((0,'unknown'),)),))
        with self.assertRaises(ValueError):
            coupling(unsafe_joint=((2,2),))
        with self.assertRaises(ValueError):
            SharedResource('invalid',True,((0,'advance'),))
        with self.assertRaises(ValueError):
            SharedResource('invalid',1,((1,'advance'),(0,'advance')))
        source=coupling()
        with self.assertRaises(ValueError):
            construct_coupled(source,max_states=4)
        with self.assertRaises(ValueError):
            construct_coupled(source,max_actions=2)


if __name__ == '__main__':
    unittest.main()
