"""Policy choices remain advisory, including adversarial and shifted policies."""
from dataclasses import replace
import json
import unittest

from navigable_compute.intention_interpreter import Interpreter,Truth,Cube,Limits
from navigable_compute.intention_interpreter_verify import verify_result
from navigable_compute.route_policy import Trainer,Policy,Stat,features,offered,realize
from navigable_compute.procedural_vocabulary import _canonical
from test_intention_interpreter import make
from test_induced_transversal import trained


class RoutePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skill=trained()

    def test_paired_training_and_restore(self):
        v=make(((1,2),(2,4)),width=8)
        i=Interpreter(v,self.skill)
        t=Trainer("local","one")
        for k in (8,16,32):
            self.assertTrue(t.observe(i,Cube(k|1,4)))
        p=t.freeze()
        self.assertEqual(len(p.stats),4)
        self.assertTrue(all(s.n==s.resolved==3 for s in p.stats))
        q=Cube(65,4)
        self.assertEqual(p,Policy.restore(p.packet(),actor="local",segment="one"))
        choice,r=realize(p,i,q,actor="local",segment="one")
        self.assertIn(choice,offered(q))
        self.assertEqual(r.status,"EXCLUDED")
        self.assertTrue(verify_result(v,q,r))
        self.assertEqual(realize(p,i,q,actor="local",segment="new")[0],"baseline")
        self.assertEqual(p.choose(v,Truth((64,))),"baseline")
        self.assertEqual(p.choose(v,q,Limits(span_work=0)),"baseline")
        with self.assertRaises(ValueError):
            Policy.restore(p.packet(),actor="other",segment="one")
        bad=json.loads(p.packet())
        bad["stats"].pop()
        with self.assertRaises(ValueError):
            Policy.restore(_canonical(bad),actor="local",segment="one")

    def test_shift_and_unknown_are_not_authority(self):
        v=make((),width=8)
        i=Interpreter(v,self.skill)
        q=Cube(1,2)
        c=features(v,q)
        # Deliberately biased advice: schedule exact-first, even though a
        # witness may be much cheaper. Semantic verification is unchanged.
        p=Policy("local","one",tuple(sorted(
            (Stat(c,s,3,3,1 if s=="exact-first" else 1000) for s in offered(q)),
            key=lambda s:(s.context,s.strategy))))
        p=Policy.restore(p.packet(),actor="local",segment="one")
        choice,r=realize(p,i,q,actor="local",segment="one")
        self.assertEqual(choice,"exact-first")
        self.assertEqual(r.status,"POSSIBLE")
        self.assertTrue(verify_result(v,q,r))
        l=Limits(witness_samples=0,residual_nodes=0,residual_work=0)
        t=Trainer("local","one",min_samples=1)
        t.observe(i,q,limits=l)
        u=t.freeze()
        self.assertEqual(u.choose(v,q,l),"baseline")
        self.assertTrue(all(s.resolved==0 for s in u.stats))
        self.assertEqual(realize(u,i,q,actor="local",segment="one",limits=l)[1].status,"UNKNOWN")

    def test_all_schedules_semantic_agreement(self):
        v=make(((1,2),(0,6)),width=8)
        i=Interpreter(v,self.skill)
        for q in (Cube(0,1),Cube(1,0),Truth((1,)),Truth((8,)),Truth((3,4))):
            rs=[i.realize(q,strategy=s) for s in offered(q)]
            self.assertEqual(len({r.status for r in rs}),1)
            self.assertTrue(all(verify_result(v,q,r) for r in rs))
        with self.assertRaises(ValueError):
            i.realize(Truth((1,)),strategy="span-first")


if __name__=="__main__":
    unittest.main()
