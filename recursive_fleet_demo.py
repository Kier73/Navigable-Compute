"""Five-minute bounded continuous recursive fleet stress campaign."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import json
from pathlib import Path
import random
from time import perf_counter,perf_counter_ns

from navigable_compute.fleet_worker import vocabulary
from navigable_compute.recursive_fleet import Graph,digest
from navigable_compute.recursive_fleet_worker import source,init_worker,run
from navigable_compute.procedural_vocabulary import Reduction
from navigable_compute.procedural_vocabulary_verify import verify_reduction


def round_trip(pool,base,n,number,*,cold=False):
    rng=random.Random(23023+number)
    order=list(range(n))
    rng.shuffle(order)
    slots=list(range(96,96+n))
    rng.shuffle(slots)
    prefix=str(number)+":"
    tasks=[("leaf","observer-"+str(j%8),prefix+"leaf-"+str(i),(i,1<<slots[i]))
           for j,i in enumerate(order)]
    g=Graph(base)
    pids=set()
    worker_ns=0
    packets=[]
    wire=0
    transfer=0

    def collect(tasks):
        nonlocal worker_ns,wire
        rs=tuple(pool.map(run,tasks))
        for raw,pid,t in rs:
            pids.add(pid)
            worker_ns+=t
            wire+=len(raw)
            packets.append(raw)
        ps=tuple(raw for raw,_,_ in rs)
        g.ingest(ps)
        assert all(g.status(digest(p))=="VERIFIED" for p in ps)
        return tuple(digest(p) for p in ps)

    heads=collect(tasks)
    leaf_heads=heads
    while len(heads)>1:
        tasks=[]
        for j in range(0,len(heads),2):
            parents=heads[j:j+2]
            closure=g.ancestors(parents)
            transfer+=sum(map(len,closure))
            tasks.append(("join","observer-"+str(j%8),prefix+"join-"+str(len(heads))+"-"+str(j),
                          (parents,closure)))
        heads=collect(tasks)
    root=heads[0]
    assert len(packets)==2*n-1
    assert g.depth==n.bit_length()
    # Late root, then reverse arrival, duplicate replays and full recovery.
    late=Graph(base)
    late.ingest((packets[-1],))
    assert late.status(root)=="PENDING"
    late.ingest(tuple(reversed(packets[:-1])))
    assert late.status(root)=="VERIFIED"
    assert late.ingest(tuple(packets))==0
    assert late.snapshot()==g.snapshot()
    restored=Graph.restore(base,late.snapshot())
    v,refs=restored.materialize()
    # A new intention not present in the packet inventory: multiply root by
    # a spectator coordinate. The recursive proof needs one admitted rule.
    k=1<<127
    q=set()
    for m in restored.relation(root):
        q^={m|k}
    target=tuple(sorted(q))
    assert target
    proof=Reduction(target,(),((refs[root],k),),len(restored.relation(root)))
    assert verify_reduction(v,target,proof)
    # Independently flatten to base certificates, without trusting internal
    # join nodes. Compare the same target, not an artificially slow baseline.
    flat=[]
    for i in range(n):
        r=base.rules[i]
        a=r[1]&-r[1]
        b=r[1]^a
        flat.extend((i,m|(1<<slots[i])|k) for m in (0,a,b,a|b))
    flat=Reduction(target,(),tuple(flat),sum(len(base.rules[i]) for i,_ in flat))
    assert verify_reduction(base,target,flat)
    if cold:
        assert base.exact(target)[0]==0
    # Negative claim changes an independent term: proof must refuse.
    bad=tuple(sorted(set(target)^{0}))
    assert not verify_reduction(v,bad,Reduction(bad,(),proof.steps,proof.work))
    return dict(packets=len(packets),depth=g.depth,worker_ns=worker_ns,
                contribution_bytes=wire,dependency_bytes=transfer,
                snapshot_bytes=len(g.snapshot()),audit_terms=g.work,
                target_terms=len(target),root_steps=len(proof.steps),
                flat_steps=len(flat.steps),pids=tuple(sorted(pids)))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seconds",type=float,default=0)
    p.add_argument("--leaves",type=int,default=32)
    p.add_argument("--workers",type=int,default=8)
    p.add_argument("--output",default="recursive-fleet-report.json")
    a=p.parse_args()
    if not 0<=a.seconds<=300 or not 1<=a.workers<=16:
        raise ValueError("stress envelope")
    s=source(a.leaves)
    base=vocabulary(s)
    counts=Counter()
    pids=set()
    max_snapshot=0
    max_depth=0
    rounds=0
    durations=[]
    start=perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=get_context("spawn"),
                             initializer=init_worker,initargs=(s,)) as pool:
        while True:
            t=perf_counter()
            report=round_trip(pool,base,a.leaves,rounds,cold=a.leaves<=4)
            durations.append(perf_counter()-t)
            rounds+=1
            pids.update(report.pop("pids"))
            max_snapshot=max(max_snapshot,report["snapshot_bytes"])
            max_depth=max(max_depth,report["depth"])
            counts.update({k:v for k,v in report.items() if k not in ("depth","snapshot_bytes")})
            if rounds%25==0:
                print("NCR023 progress",rounds,"rounds",round(perf_counter()-start,3),
                      "seconds",counts["packets"],"proof packets",flush=True)
            if perf_counter()-start>=a.seconds:
                break
    elapsed=perf_counter()-start
    durations.sort()
    result=dict(rounds=rounds,leaves=a.leaves,workers_requested=a.workers,
        distinct_worker_processes=len(pids),elapsed_seconds=elapsed,max_depth=max_depth,
        max_snapshot_bytes=max_snapshot,round_p50_seconds=durations[len(durations)//2],
        round_p95_seconds=durations[min(len(durations)-1,int(len(durations)*.95))],
        totals=dict(counts),cold_controls=a.leaves<=4,
        note="bounded independent shards; no unbounded cross-round memory accumulation")
    Path(a.output).write_text(json.dumps(result,sort_keys=True,indent=2),encoding="utf-8")
    print("NCR023",json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
