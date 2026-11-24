"""Standalone sparse CRT observation network, no external research dependency."""
import json,time
from navigable_compute.ground_network import View,Network,discover,compose,select
from navigable_compute.ground_network_verify import verify_realization

BASE=(2,3,5,7,11,13,17,19,23,29,31,37,41,43,47)

def marker(name,shift,*,extra=False,cutoff=0,prefix=0):
    base=BASE+((53,) if extra else ());nodes=[];root=1
    for j in reversed(range(len(BASE))):
        nodes.append((j,tuple(root if a==(-shift)%base[j] else 0 for a in range(base[j]))));root=len(nodes)+1
    return View(name,base,tuple(nodes),root,cutoff,prefix)


def run():
    a=marker('original',0);b=marker('different-basis',1,extra=True);c=marker('changed-prefix',3,cutoff=1,prefix=1)
    graph=Network();[graph.register(v) for v in (a,b,c)]
    t=time.perf_counter();ab,wa=discover(a,b,[(0,0),(1,0)],cost=3);bc,wb=discover(b,c,[(2,0),(2,1)],cost=4)
    assert ab and bc and graph.admit(ab) and graph.admit(bc)
    bridge_ms=(time.perf_counter()-t)*1000;path=(bc.key,ab.key)
    k=10**900+173;offer=lambda key,ordinal:0 if key==a.key else None
    route,_=graph.plan(c.key,k,offer);assert route.edges==path
    t=time.perf_counter();shortcut=compose(graph,path,cost=1);assert graph.admit_composite(shortcut,path);compose_ms=(time.perf_counter()-t)*1000
    route,_=graph.plan(c.key,k,offer);assert route.edges==(shortcut.key,)
    t=time.perf_counter();result=select(a,route.source_ordinal);assert result['status']=='FOUND';dest=verify_realization(graph,c.key,k,route,result['time']);query_ms=(time.perf_counter()-t)*1000
    assert dest==(k-1)*a.period-3
    assert graph.plan(c.key,1,offer)[0] is None # earlier prefix event is not covered
    return dict(period=str(a.period),intermediate_period=str(b.period),diagram_nodes=sum(len(v.nodes) for v in (a,b,c)),discovery_work=wa+wb,bridge_ms=bridge_ms,composite_ms=compose_ms,query_ms=query_ms,shift=shortcut.shift,cutoff=shortcut.start,rank_offset=shortcut.offset,ordinal=str(k),destination=str(dest),source_rank_work=result['work'])

if __name__=='__main__':print(json.dumps(run(),indent=2))
