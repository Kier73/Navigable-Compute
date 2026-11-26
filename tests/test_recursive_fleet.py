"""Recursive proof dependencies, partial arrival, tamper and native replay."""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import json
import unittest

from navigable_compute.fleet_worker import vocabulary
from navigable_compute.recursive_fleet_worker import source,leaf,join,init_worker,run
from navigable_compute.recursive_fleet import Graph,describe,digest
from navigable_compute.procedural_vocabulary import _canonical
from recursive_fleet_demo import round_trip


class RecursiveFleetTests(unittest.TestCase):
    def test_recursive_late_and_invalid_dependencies(self):
        v=vocabulary(source(2))
        a=leaf(v,"a","a",0,1<<96)
        b=leaf(v,"b","b",1,1<<97)
        h1,h2=digest(a),digest(b)
        c=join(v,"c","c",(h1,h2),(a,b))
        root=digest(c)
        g=Graph(v)
        g.ingest((c,))
        self.assertEqual(g.status(root),"PENDING")
        with self.assertRaises(ValueError):
            g.relation(root)
        g.ingest((a,))
        self.assertEqual(g.status(root),"PENDING")
        g.ingest((b,b))
        self.assertEqual(g.status(root),"VERIFIED")
        self.assertEqual(g.depth,2)
        restored=Graph.restore(v,g.snapshot())
        self.assertEqual(restored.snapshot(),g.snapshot())
        native,refs=restored.materialize()
        self.assertEqual(native.rules[refs[root]],g.relation(root))
        self.assertEqual(v.exact(g.relation(root))[0],0)
        bad=json.loads(a)
        bad["relation"]=[1<<127]
        bad=_canonical(bad)
        child=describe(v,"d","d",(1<<127,),(("dependency",digest(bad),0),))
        g.ingest((child,bad))
        self.assertEqual(g.status(digest(bad)),"INVALID")
        self.assertEqual(g.status(digest(child)),"INVALID")
        self.assertEqual(g.status(root),"VERIFIED")
        old=g.snapshot()
        with self.assertRaises(ValueError):
            g.ingest((b,b"{}"))
        self.assertEqual(g.snapshot(),old)

    def test_spawned_recursive_and_flat_control(self):
        s=source(4)
        v=vocabulary(s)
        with ProcessPoolExecutor(max_workers=4,mp_context=get_context("spawn"),
                                 initializer=init_worker,initargs=(s,)) as pool:
            report=round_trip(pool,v,4,1,cold=True)
        self.assertEqual(report["packets"],7)
        self.assertEqual(report["depth"],3)
        self.assertEqual(report["root_steps"],1)
        self.assertEqual(report["flat_steps"],16)


if __name__=="__main__":
    unittest.main()
