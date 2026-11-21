"""Tests run using actual Navigable-Compute NCR-004 and independent checker."""
from dataclasses import replace
import unittest

from navigable_compute.observed_ground import (
    TransitionObservation, ObservedEpoch, IncompleteSource,
    admit_observed_epoch,navigate_observed)
from navigable_compute.preclosure_verify import verify_preclosure_language


def demo():
    # No transition or operator table is supplied to NCR-004.
    labels=lambda s: 'active' if s%2 else 'quiet'
    raw=[]
    for s in range(6):
        raw.append((s,'pulse',(s+1)%6))
        raw.append((s,'reset',s%2))
    observations=tuple(TransitionObservation(i,s,a,t,labels(s),labels(t))
                       for i,(s,a,t) in enumerate(raw))
    return ObservedEpoch('observed-source-A','stationary-epoch-1',
                         tuple(range(6)),('pulse','reset'),(0,),observations)


class ObservedGroundTests(unittest.TestCase):
    def test_source_to_native_language_and_far_time(self):
        e=demo()
        a=admit_observed_epoch(e,expected_head=e.source_head)
        self.assertTrue(verify_preclosure_language(a.source,a.language))
        self.assertEqual(len(a.source.states),6)
        self.assertEqual(len(a.language.state_classes),2)
        self.assertEqual(navigate_observed(a,expected_head=e.source_head,
            initial=0,word=(0,),repetitions=10**80+19),'active')
        # Ground scope is unchanged when queried through a different task.
        self.assertEqual(navigate_observed(a,expected_head=e.source_head,
            initial=0,word=(1,),repetitions=10**80+19),'quiet')

    def test_unknown_requires_actual_source_observation(self):
        e=demo()
        partial=replace(e,witnesses=e.witnesses[:-1])
        with self.assertRaises(IncompleteSource) as ctx:
            admit_observed_epoch(partial,expected_head=partial.source_head)
        self.assertEqual(ctx.exception.missing,((5,'reset'),))

    def test_changed_source_revision_invalidates_pin(self):
        e=demo()
        revised=replace(e,context='stationary-epoch-2')
        self.assertNotEqual(revised.source_head,e.source_head)
        with self.assertRaisesRegex(ValueError,'pinned'):
            admit_observed_epoch(revised,expected_head=e.source_head)

    def test_conflicting_evidence_is_not_a_new_machine_law(self):
        e=demo()
        bad=TransitionObservation(len(e.witnesses),0,'pulse',3,'quiet','active')
        with self.assertRaisesRegex(ValueError,'contradictory'):
            replace(e,witnesses=e.witnesses+(bad,))

    def test_tampered_portable_native_receipt_rejected(self):
        e=demo();a=admit_observed_epoch(e,expected_head=e.source_head)
        tampered=replace(a,wire=a.wire+b' ')
        with self.assertRaises(ValueError):
            navigate_observed(tampered,expected_head=e.source_head,
                              initial=0,word=(0,),repetitions=19)

    def test_unknown_is_not_global_state_space_completeness(self):
        e=demo()
        extra=replace(e,states=e.states+(6,))
        # Unreachable state is explicitly outside the witnessed initial closure.
        a=admit_observed_epoch(extra,expected_head=extra.source_head)
        self.assertEqual(len(a.source.states),6)

if __name__=='__main__':unittest.main()
