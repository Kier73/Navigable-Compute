"""Coalescence is semantic union, not a vote or an arrival-order cache."""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from dataclasses import replace
import json
import random
import unittest

from navigable_compute.coalescent_fleet import Replica
from navigable_compute.fleet_worker import (fixture,intention,vocabulary,Task,
                                            run_task)
from navigable_compute.procedural_vocabulary import _canonical
from navigable_compute.procedural_vocabulary_verify import verify_reduction


class FleetTests(unittest.TestCase):
    def test_complementary_processes_and_convergence(self):
        s=fixture(4)
        tasks=tuple(Task("observer-"+str(i),"part-"+str(i),
                         "transversal" if i%2==0 else "product",i) for i in range(4))
        # Independent spawned processes, no common mutable Vocabulary.
        with ProcessPoolExecutor(max_workers=4,mp_context=get_context("spawn")) as pool:
            results=tuple(pool.map(run_task,(s,)*4,tasks))
        packets=tuple(p for ps,_,_ in results for p in ps)
        q=intention(4)
        base=vocabulary(s)
        self.assertIsNone(base.reduce(q).value)
        for ps,_,_ in results:
            r=Replica(base)
            r.ingest(ps)
            self.assertIsNone(r.materialize().reduce(q).value)
        r=Replica(base)
        r.ingest(packets[:len(packets)//2])
        r2=Replica(vocabulary(s))
        r2.ingest(tuple(reversed(packets[len(packets)//2:])))
        # Complementary deliveries, duplicate replays, arbitrary arrival.
        r.ingest(r2.missing(r.inventory))
        r2.ingest(r.missing(r2.inventory))
        self.assertEqual(r.snapshot(),r2.snapshot())
        self.assertEqual(r.ingest(packets),0)
        rebuilt=Replica.restore(vocabulary(s),r.snapshot())
        v=rebuilt.materialize()
        p=v.reduce(q)
        self.assertEqual(p.value,0)
        self.assertTrue(verify_reduction(v,q,p))
        self.assertEqual(base.exact(q)[0],0)
        self.assertIsNone(v.reduce(tuple(sorted(set(q)^{1<<127}))).value)
        rng=random.Random(22022)
        for _ in range(24):
            shuffled=list(packets)
            rng.shuffle(shuffled)
            r3=Replica(base)
            for item in shuffled:
                r3.ingest((item,item))
            self.assertEqual(r3.snapshot(),r.snapshot())
            self.assertEqual(r3.materialize().rules,v.rules)

    def test_redundant_workers_and_bad_batch(self):
        s=fixture(2)
        base=vocabulary(s)
        a,_,_=run_task(s,Task("a","same","product",0))
        b,_,_=run_task(s,Task("b","same","product",0))
        r=Replica(base)
        self.assertEqual(r.ingest(a+b),2)
        self.assertEqual(len(r.materialize().learned),1)
        old=r.snapshot()
        c,_,_=run_task(s,Task("c","other","product",1))
        forged=json.loads(c[0])
        forged["relation"]=[1<<127]
        with self.assertRaises(ValueError):
            r.ingest(c+(_canonical(forged),))
        self.assertEqual(old,r.snapshot())
        wrong=vocabulary(replace(s,context="other"))
        with self.assertRaises(ValueError):
            Replica(wrong).ingest(a)
        with self.assertRaises(ValueError):
            Replica.restore(wrong,old)
        self.assertEqual(r.materialize().reduce(intention(2)).value,None)
        r.ingest(c)
        self.assertEqual(r.materialize().reduce(intention(2)).value,0)


if __name__=="__main__":
    unittest.main()
