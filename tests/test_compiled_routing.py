"""Exact policy projection, stale catalogue refusal, no cached answers."""
import unittest
from navigable_compute.intention_interpreter import Cube,Truth,Limits,Interpreter
from navigable_compute.intention_interpreter_verify import verify_result
from navigable_compute.route_policy import Policy,Stat,offered,features
from navigable_compute.compiled_routing import CompiledPolicy
from test_induced_transversal import trained
from test_intention_interpreter import make


class CompiledTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skill=trained()

    def test_exact_policy_projection_and_catalogue_change(self):
        v=make(((1,2),(0,12,20,24)),width=8)
        i=Interpreter(v,self.skill)
        q=Cube(128,0)
        c=features(v,q)
        rows=tuple(sorted((Stat(c,s,3,3,1 if s=="witness-first" else 1000)
                           for s in offered(q)),key=lambda s:(s.context,s.strategy)))
        p=Policy("actor","segment",rows)
        compiled=CompiledPolicy(p)
        bound=compiled.bind(i)
        for request in (q,Cube(64,0),Cube(64,1),Truth((128,)),Truth((1,2))):
            for l in (Limits(),Limits(span_work=0)):
                expected=p.choose(v,request,l)
                self.assertEqual(compiled.choose(v,request,l),expected)
                self.assertEqual(bound.choose(request,actor="actor",segment="segment",limits=l),expected)
                s,r=bound.realize(request,actor="actor",segment="segment",limits=l)
                self.assertEqual(s,expected)
                self.assertTrue(verify_result(v,request,r))
        self.assertEqual(bound.choose(q,actor="actor",segment="other"),"baseline")
        self.assertEqual(bound.choose(q,actor="actor",segment="segment"),"witness-first")
        self.assertTrue(v.learn((4,8,12)))
        self.assertFalse(bound.current())
        self.assertEqual(bound.choose(q,actor="actor",segment="segment"),"baseline")
        self.assertEqual(compiled.bind(i).choose(q,actor="actor",segment="segment"),p.choose(v,q))
        with self.assertRaises(ValueError):
            bound.choose(Cube(1,1),actor="actor",segment="segment")

    def test_unresolved_training_and_table_immutability(self):
        v=make((),width=8)
        q=Truth((1,))
        c=features(v,q)
        rows=tuple(sorted((Stat(c,s,3,2,0) for s in offered(q)),
                          key=lambda s:(s.context,s.strategy)))
        p=Policy("actor","segment",rows)
        compiled=CompiledPolicy(p)
        self.assertEqual(compiled.choose(v,q),"baseline")
        with self.assertRaises(TypeError):
            compiled.table[c]="exact-first"


if __name__=="__main__":
    unittest.main()
