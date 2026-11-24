"""Actual pinned VGI channel/encounter/quorum integration (separate CI)."""
import unittest
from dataclasses import replace
from navigable_compute.intention_fibre import construct_fibre
from navigable_compute.fibre_attention import Offer,Prices,plan_attention
from navigable_compute.adaptive_attention import compile_policy,Runtime
from navigable_compute.attention_evidence import (
    evidence_policy,advance_vgi,verify_evidence_step,STATE)
from vgi.channels import ChannelContract,ChannelExchangeLedger,ChannelMessage
from vgi.encounters import EncounterProvenance
from vgi.evidence import TransitionEvidenceLedger,TransportFrameReceipt


def observed(runtime,session,policy,bit,*,finalize=True,missing=False,dissent=False):
    name=runtime.view(session).decision.next_offer
    ledger=TransitionEvidenceLedger(policy)
    contract=ChannelContract("adaptive.bus","adaptive.frame/1",
                             observation_schema="vgi.transition-observation/1")
    exchange=ChannelExchangeLedger()
    messages=[]
    for i,source in enumerate(policy.allowed_sources):
        if missing and i:
            continue
        message=ChannelMessage(f"obs-{i}","adaptive.bus","adaptive.frame/1","observation",
            dict(status="known",source_state=STATE,action=name,target=STATE,
                 output=[bit ^ int(dissent and i==1)]),
            EncounterProvenance(source,f"capture://{source}/{i}",collector="fixture"))
        messages.append(message)
        exchange=exchange.record_observation(contract,message)
    encounters=exchange.to_encounter_ledger()
    for i,message in enumerate(messages,1):
        frame=TransportFrameReceipt.capture(message,transport_id="ncr013-fixture",sequence=i)
        ledger=ledger.ingest_channel_message(exchange,message.message_id,frame,encounters)
    return ledger.finalize(STATE,name) if finalize else ledger


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        f=construct_fibre(8,())
        c=plan_attention(f,tuple(Offer(n,(m,),1) for n,m in zip("abc",(1,2,4))),
                         (3,4,5),Prices(10),max_packet=2)
        self.r=Runtime(c,compile_policy(c))
        self.s=self.r.begin("source-epoch","context-1")

    def policy(self,s):
        return evidence_policy(self.r,s,("sensor-A","sensor-B"),2)

    def test_two_contexts_and_branch_specific_acquisition(self):
        r,s=self.r,self.s
        for a in (0,1):
            p=self.policy(s)
            ledger=observed(r,s,p,a)
            first=advance_vgi(r,s,p,ledger)
            self.assertTrue(verify_evidence_step(r,s,p,ledger,first))
            next_session=first.after
            self.assertEqual(r.view(next_session).decision.next_offer,"b" if a else "c")
            p2=self.policy(next_session)
            self.assertNotEqual(p.context_digest,p2.context_digest)
            with self.assertRaises(ValueError):
                advance_vgi(r,next_session,p,ledger)
            with self.assertRaises(ValueError):
                advance_vgi(r,r.begin("source-epoch","context-2"),p,ledger)
            for value in (0,1):
                l2=observed(r,next_session,p2,value)
                receipt=advance_vgi(r,next_session,p2,l2)
                self.assertTrue(verify_evidence_step(r,next_session,p2,l2,receipt))
                self.assertEqual(r.view(receipt.after).inferred,value)
                self.assertFalse(verify_evidence_step(r,next_session,p2,l2,
                    replace(receipt,decision_digest="0"*64)))

    def test_pending_conflict_missing_and_wrong_authority(self):
        r,s=self.r,self.s
        p=self.policy(s)
        for kwargs in (dict(finalize=False),dict(missing=True),dict(dissent=True)):
            ledger=observed(r,s,p,0,**kwargs)
            with self.assertRaises(ValueError):
                advance_vgi(r,s,p,ledger)
        other=evidence_policy(r,s,("sensor-A",),1)
        ledger=observed(r,s,other,0)
        with self.assertRaises(ValueError):
            advance_vgi(r,s,p,ledger)


if __name__=="__main__":
    unittest.main()
