"""Separate hosted jobs contribute, then an independent job coalesces artifacts."""
import argparse
import json
from pathlib import Path
from navigable_compute.coalescent_fleet import Replica
from navigable_compute.fleet_worker import fixture,intention,vocabulary,Task,run_task
from navigable_compute.procedural_vocabulary import _canonical
from navigable_compute.procedural_vocabulary_verify import verify_reduction


def main():
    p=argparse.ArgumentParser()
    p.add_argument("role",choices=("produce","merge"))
    p.add_argument("--index",type=int,default=0)
    p.add_argument("--directory",default="fleet-artifacts")
    a=p.parse_args()
    s=fixture(4)
    root=Path(a.directory)
    if a.role=="produce":
        t=Task("job-"+str(a.index),"part-"+str(a.index),
               "transversal" if a.index%2==0 else "product",a.index)
        packets,pid,elapsed=run_task(s,t)
        root.mkdir(parents=True,exist_ok=True)
        (root/("part-"+str(a.index)+".json")).write_bytes(_canonical(
            [json.loads(p) for p in packets]))
        print("NCR022 producer",a.index,"packets",len(packets),"pid",pid,"ns",elapsed)
    else:
        base=vocabulary(s)
        q=intention(4)
        paths=sorted(root.rglob("part-*.json"))
        assert len(paths)==4
        sets=[tuple(_canonical(x) for x in json.loads(p.read_bytes())) for p in paths]
        for packets in sets:
            r=Replica(base)
            r.ingest(packets)
            assert r.materialize().reduce(q).value is None
        a,b=Replica(base),Replica(vocabulary(s))
        for packets in sets:
            a.ingest(packets)
        for packets in reversed(sets):
            b.ingest(tuple(reversed(packets)))
        assert a.snapshot()==b.snapshot()
        v=a.materialize()
        proof=v.reduce(q)
        assert proof.value==0 and verify_reduction(v,q,proof)
        assert base.exact(q)[0]==0
        (root/"coalesced.json").write_bytes(a.snapshot())
        print("NCR022 hosted coalescence",dict(jobs=len(paths),packets=len(a.inventory),
              rules=len(v.learned),individual_resolved=0,collective_resolved=1,
              snapshot_bytes=len(a.snapshot())))


if __name__=="__main__":
    main()
