"""Measure fleet process work, transfer, convergence and new collective intention."""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from time import perf_counter_ns
from navigable_compute.coalescent_fleet import Replica
from navigable_compute.fleet_worker import fixture,intention,vocabulary,Task,run_task,skill
from navigable_compute.intention_interpreter import Interpreter,Truth,Limits
from navigable_compute.intention_interpreter_verify import verify_result


def main():
    s=fixture(8)
    # 8 different primary tasks + 24 redundant independent tasks.
    tasks=tuple(Task("observer-"+str(i),"part-"+str(i%8),
                     "transversal" if i%8%2==0 else "product",i%8) for i in range(32))
    start=perf_counter_ns()
    base=vocabulary(s)
    admission_ns=perf_counter_ns()-start
    start=perf_counter_ns()
    with ProcessPoolExecutor(max_workers=8,mp_context=get_context("spawn")) as pool:
        results=tuple(pool.map(run_task,(s,)*len(tasks),tasks))
    fleet_ns=perf_counter_ns()-start
    q=Truth(intention(8))
    start=perf_counter_ns()
    singles=[]
    for packets,_,_ in results[:8]:
        r=Replica(base)
        r.ingest(packets)
        singles.append(r.materialize().reduce(q.target).value)
    individual_ns=perf_counter_ns()-start
    assert singles==[None]*8
    a,b=Replica(base),Replica(vocabulary(s))
    packets=tuple(p for ps,_,_ in results for p in ps)
    start=perf_counter_ns()
    a.ingest(packets[::2])
    b.ingest(tuple(reversed(packets[1::2])))
    a.ingest(b.missing(a.inventory))
    b.ingest(a.missing(b.inventory))
    assert a.snapshot()==b.snapshot()
    assert a.ingest(packets)==0
    merged=Replica.restore(base,a.snapshot()).materialize()
    coalesce_ns=perf_counter_ns()-start
    i=Interpreter(merged,skill())
    start=perf_counter_ns()
    result=i.realize(q,limits=Limits(residual_nodes=0,residual_work=0))
    assert result.status=="ZERO" and result.mechanism=="rewrite"
    assert verify_result(merged,q,result)
    warm_ns=perf_counter_ns()-start
    start=perf_counter_ns()
    cold,proof=base.exact(q.target)
    cold_ns=perf_counter_ns()-start
    assert cold==0
    print("NCR022",dict(tasks=len(tasks),processes=len({pid for _,pid,_ in results}),
        primary_tasks=8,individual_resolved=sum(x is not None for x in singles),
        contributions=len(packets),unique_packets=len(a.inventory),
        unique_rules=len(merged.learned),wire_bytes=sum(map(len,packets)),
        snapshot_bytes=len(a.snapshot()),source_count=base.overlay.count,
        admission_ns=admission_ns,fleet_wall_ns=fleet_ns,
        worker_total_ns=sum(t for _,_,t in results),individual_check_ns=individual_ns,
        coalesce_ns=coalesce_ns,warm_ns=warm_ns,cold_ns=cold_ns))


if __name__=="__main__":
    main()
