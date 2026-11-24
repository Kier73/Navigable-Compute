"""Use actual NCR-014 VGI-finalized memory as the learning substrate."""
import unittest
from hashlib import sha256
from navigable_compute.intention_fibre import construct_fibre
from navigable_compute.fibre_attention import Offer,Prices,plan_attention
from navigable_compute.adaptive_attention import compile_policy,Runtime
from navigable_compute.observation_memory import Memory,restore
from navigable_compute.procedural_vocabulary import Vocabulary
from test_attention_evidence import observed


class VocabularyIntegration(unittest.TestCase):
    def test_vgi_memory_and_vocabulary_recovery(self):
        parent=construct_fibre(128,())
        offers=(Offer("parity",(1,2),1),Offer("majority",(12,20,24),1))
        c=plan_attention(parent,offers,(1,2,12,20,24),Prices(10),max_packet=2)
        runtime=Runtime(c,compile_policy(c))
        m=Memory(runtime,"fixed-epoch","member-1",("sensor-A","sensor-B"),2)
        while not m.view.terminal:
            name=m.view.decision.next_offer
            policy=m.expected_policy()
            m.append(observed(runtime,m.session,policy,int(name=="majority")))
        self.assertEqual(len(m.session.trace),2)
        v=Vocabulary.from_memory(m)
        self.assertEqual(v.overlay.count,2**126)
        self.assertIn((4,8,12),v.train())
        k=1<<90
        target=(k,k|4,k|8,k|12)
        self.assertEqual(v.reduce(target).value,0)
        self.assertEqual(m.read(target).value,0)
        payload=m.snapshot()
        recovered=restore(runtime,"fixed-epoch","member-1",("sensor-A","sensor-B"),2,payload)
        w=Vocabulary.from_memory(recovered)
        restored=Vocabulary.restore(w.parent,w.observations,w.overlay,
                                     context=sha256(payload).hexdigest(),payload=v.packet())
        self.assertEqual(restored.reduce(target).value,0)


if __name__=="__main__":
    unittest.main()
