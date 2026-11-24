"""Independent intent succession, evidence replay and native E-CPU recovery."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from navigable_compute.intention_fibre import construct_fibre
from navigable_compute.fibre_attention import Offer,Prices,plan_attention
from navigable_compute.adaptive_attention import compile_policy,Runtime
from navigable_compute.observation_memory import Memory,restore,canonical
from navigable_compute.observation_memory_native import seal,recover,CUT
from mailcpu.federated_ground_continuity import AtomicGroundEvidenceStore
from mailcpu.recovery_lineage import (RecoveryLineageAuthority,RecoveryLineageCoordinator,
                                      RecoveryStateManifest)
from mailcpu.seed_commit_quorum import SeedCommitMembership
from test_attention_evidence import observed

EPOCH,CONTEXT="source-law-v1","one-hidden-member"
SOURCES=("sensor-A","sensor-B")


def runtime():
    # a XOR b must be learned; afterward a OR b, a AND b and arbitrary
    # other polynomial intentions can reuse the evidence.
    f=construct_fibre(128,())
    c=plan_attention(f,(Offer("a",(1,),1),Offer("b",(2,),1)),(1,2),Prices(10),max_packet=2)
    return Runtime(c,compile_policy(c))


def authority(root):
    root=Path(root)
    membership=SeedCommitMembership("ncr014-machine",1,("owner","anchor-b","anchor-c"),
                                     "owner",1,"f"*64)
    journals={n:RecoveryLineageAuthority(membership,n,root/(n+".jsonl"))
              for n in membership.member_ids}
    return RecoveryLineageCoordinator(membership,journals),AtomicGroundEvidenceStore(root/"states")


def initial(r):
    return Memory(r,EPOCH,CONTEXT,SOURCES,2)


def append(m,b):
    p=m.expected_policy()
    m.append(observed(m.runtime,m.session,p,b))


def reconstruct(r,raw):
    return restore(r,EPOCH,CONTEXT,SOURCES,2,raw)


def load(r,coord,store,m):
    return recover(r,EPOCH,CONTEXT,SOURCES,2,coord,store,m)


def worker(root):
    # A disposable OS process makes one new observation, seals a second
    # generation, then dies before replying. The parent knows the prepared
    # manifest path and independently verifies the live quorum on recovery.
    r=runtime()
    coord,store=authority(root)
    first=RecoveryStateManifest.from_snapshot(json.loads((Path(root)/"first.json").read_bytes()))
    m=load(r,coord,store,first)
    append(m,0)
    second,_=seal(m,coord,store,previous=first)
    # Retain only the locator; no semantic success reply.
    (Path(root)/"second.json").write_bytes(canonical(second.snapshot()))
    import os
    os._exit(73)


class MemoryTests(unittest.TestCase):
    def test_cross_intention_and_corruption(self):
        r=runtime()
        m=initial(r)
        append(m,1)
        self.assertEqual(m.read((1,)).value,1)
        self.assertIsNone(m.read((2,)).value)
        self.assertEqual(m.read((1,2,3)).value,1)
        self.assertFalse(m.read((1,2,3)).reused)
        append(m,0)
        self.assertEqual(m.read((1,2)).value,1)
        self.assertTrue(m.read((1,2)).reused)
        self.assertEqual(m.read((3,)).value,0)
        self.assertIsNone(m.read((4,)).value)
        self.assertEqual(m.read((4,)).yes,2**125)
        rebuilt=reconstruct(runtime(),m.snapshot())
        self.assertEqual(rebuilt.snapshot(),m.snapshot())
        o=rebuilt.view.outcome.overlay
        j=10**30
        self.assertEqual(o.refined_address(o.parent_address(j)),j)
        self.assertEqual(r.parent.unrank(o.parent_address(j))&3,1)
        for field,value in (("epoch","changed"),("trace",[["a",0],["b",0]]),
                            ("quorum",1),("records",[])):
            b=json.loads(m.snapshot())
            b[field]=value
            with self.assertRaises(ValueError):
                reconstruct(r,canonical(b))
        b=json.loads(m.snapshot())
        b["records"][0]["decision"]="0"*64
        with self.assertRaises(ValueError):
            reconstruct(r,canonical(b))
        with self.assertRaises(ValueError):
            restore(r,EPOCH,"other-member",SOURCES,2,m.snapshot())

    def test_native_quorum_process_cut_and_stale_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            r=runtime()
            coord,store=authority(root)
            m=initial(r)
            append(m,1)
            first,proof=seal(m,coord,store)
            self.assertGreaterEqual(len(proof.decisions),coord.membership.quorum)
            (root/"first.json").write_bytes(canonical(first.snapshot()))
            process=subprocess.run([sys.executable,str(Path(__file__).resolve()),"worker",tmp],
                                   capture_output=True,text=True,timeout=90)
            self.assertEqual(process.returncode,73,process.stdout+process.stderr)
            # Reopen authority journals and reconstruct on an independent runtime.
            coord,store=authority(root)
            second=RecoveryStateManifest.from_snapshot(json.loads((root/"second.json").read_bytes()))
            self.assertEqual(coord.high_water(),(2,second.digest))
            self.assertEqual(second.predecessor_digest,first.digest)
            restored=load(runtime(),coord,store,second)
            self.assertEqual(restored.session.trace,(("a",1),("b",0)))
            self.assertEqual(restored.read((1,2,3)).value,1)
            self.assertEqual(restored.read((3,)).value,0)
            with self.assertRaises(ValueError):
                load(r,coord,store,first)
            path=store.directory/(second.state_digest+".json")
            path.write_bytes(b"{}")
            with self.assertRaises(ValueError):
                load(r,coord,store,second)

    def test_prefix_fence_and_minority(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=runtime()
            coord,store=authority(tmp)
            m=initial(r)
            append(m,1)
            minority=RecoveryLineageCoordinator(coord.membership,{"owner":coord.authorities["owner"]})
            with self.assertRaises(ValueError):
                seal(m,minority,store)
            first,_=seal(m,coord,store)
            different=initial(r)
            append(different,0)
            append(different,1)
            with self.assertRaises(ValueError):
                seal(different,coord,store,previous=first)
            append(m,0)
            second,_=seal(m,coord,store,previous=first)
            self.assertEqual(load(r,coord,store,second).snapshot(),m.snapshot())


if __name__=="__main__":
    if len(sys.argv)==3 and sys.argv[1]=="worker":
        worker(sys.argv[2])
    else:
        unittest.main()
