"""Adaptive attention: early stopping, branch-specific probes, exact oracles."""
import random
import unittest
from dataclasses import replace
from navigable_compute.intention_fibre import construct_fibre
from navigable_compute.fibre_attention import Offer,Prices,plan_attention
from navigable_compute.adaptive_attention import compile_policy,Runtime
from navigable_compute.adaptive_attention_verify import verify_policy


def bit(p,x):
    return sum((x&m)==m for m in p)%2


def fixture(width,target,offers,prices=Prices(10),max_packet=2):
    f=construct_fibre(width,())
    c=plan_attention(f,offers,target,prices,max_packet=max_packet)
    p=compile_policy(c)
    return c,p


class AdaptiveTests(unittest.TestCase):
    def test_or_early_stop_and_reuse(self):
        offers=(Offer("a",(1,),1),Offer("b",(2,),1))
        c,p=fixture(128,(1,2,3),offers)
        self.assertTrue(verify_policy(c,p))
        self.assertEqual(2*p.numerator,3*c.denominator)
        self.assertEqual(min(q.score_numerator for q in c.candidates),2*c.denominator)
        runtime=Runtime(c,p)
        s=runtime.begin("source-v1","first")
        a=runtime.advance(s,"a",1)
        self.assertTrue(runtime.view(a).terminal)
        self.assertEqual(runtime.view(a).inferred,1)
        b=runtime.advance(s,"a",0)
        self.assertEqual(runtime.view(b).decision.next_offer,"b")
        self.assertEqual(runtime.view(runtime.advance(b,"b",0)).inferred,0)
        self.assertNotEqual(runtime.challenge(s),runtime.challenge(runtime.begin("source-v1","second")))
        self.assertNotEqual(runtime.challenge(s),runtime.challenge(b))
        with self.assertRaises(ValueError):
            runtime.advance(a,"b",0)
        with self.assertRaises(ValueError):
            runtime.advance(s,"b",1)
        with self.assertRaises(ValueError):
            runtime.advance(s,"a",True)

    def test_multiplexer_chooses_different_second_probe(self):
        offers=tuple(Offer(n,(m,),1) for n,m in zip("abc",(1,2,4)))
        # if a then b else c: ab XOR c XOR ac
        c,p=fixture(128,(3,4,5),offers)
        self.assertTrue(verify_policy(c,p))
        self.assertEqual(p.numerator,2*c.denominator)
        self.assertEqual(min(q.score_numerator for q in c.candidates),7*c.denominator)
        r=Runtime(c,p)
        s=r.begin("source-v1","mux")
        self.assertEqual(r.view(s).decision.next_offer,"a")
        for a in (0,1):
            v=r.advance(s,"a",a)
            name="b" if a else "c"
            self.assertEqual(r.view(v).decision.next_offer,name)
            for b in (0,1):
                w=r.view(r.advance(v,name,b))
                self.assertEqual(w.inferred,b)
                self.assertTrue(w.terminal)
                x=w.outcome.overlay.unrank(10**20)
                self.assertEqual(bit(c.target,x),b)
                self.assertEqual(c.parent.unrank(w.outcome.overlay.parent_address(10**20)),x)

    def test_oracle_random_sources_and_policies(self):
        rng=random.Random(13013)
        for _ in range(45):
            n=rng.randint(2,5)
            def poly():
                return tuple(sorted(rng.sample(range(2**n),rng.randint(0,4))))
            rows=tuple(poly() for _ in range(rng.randint(0,3)))
            states=[x for x in range(2**n) if not any(bit(q,x) for q in rows)]
            if not states:
                continue
            f=construct_fibre(n,rows)
            offers=tuple(Offer(k,poly(),rng.randint(1,4)) for k in "abc")
            target=poly()
            prices=Prices(rng.randint(0,12))
            c=plan_attention(f,offers,target,prices,max_packet=2)
            p=compile_policy(c)
            self.assertTrue(verify_policy(c,p))
            # Exhaustively enumerate every deterministic two-stage strategy
            # on the actual states, independently of residual graphs.
            def stop(xs):
                return len(xs)*prices.fallback if len({bit(target,x) for x in xs})==2 else 0
            best=stop(states)
            for first in offers:
                branches=[[x for x in states if bit(first.polynomial,x)==b] for b in (0,1)]
                alternatives=[None]+[o for o in offers if o!=first]
                for left in alternatives:
                    for right in alternatives:
                        score=len(states)*first.cost
                        for xs,q in zip(branches,(left,right)):
                            if q is None:
                                score+=stop(xs)
                            else:
                                score+=len(xs)*q.cost
                                score+=sum(stop([x for x in xs if bit(q.polynomial,x)==b]) for b in (0,1))
                        best=min(best,score)
            self.assertEqual(p.numerator,best)
            r=Runtime(c,p)
            observed_cost=0
            by_name={o.name:o for o in offers}
            for x in states:
                s=r.begin("random",str(x))
                while not r.view(s).terminal:
                    q=by_name[r.view(s).decision.next_offer]
                    observed_cost+=q.cost
                    s=r.advance(s,q.name,bit(q.polynomial,x))
                v=r.view(s)
                if v.fallback_required:
                    observed_cost+=prices.fallback
                else:
                    self.assertEqual(v.inferred,bit(target,x))
                self.assertEqual(v.outcome.overlay.unrank(v.outcome.overlay.rank(x)),x)
            self.assertEqual(observed_cost,best)

    def test_geometry_and_tamper(self):
        hi,lo=1<<15,1
        c,p=fixture(16,(lo,hi,lo|hi),
                    (Offer("a-low",(lo,),1),Offer("z-high",(hi,),1)),
                    Prices(20,1,1,1),max_packet=1)
        self.assertEqual(next(d for d in p.decisions if not d.history).next_offer,"z-high")
        self.assertTrue(verify_policy(c,p))
        d=p.decisions[0]
        for change in (replace(d,numerator=d.numerator+1),replace(d,next_offer="a-low"),
                       replace(d,count=d.count+1)):
            self.assertFalse(verify_policy(c,replace(p,decisions=(change,)+p.decisions[1:])))
        self.assertFalse(verify_policy(c,replace(p,decisions=p.decisions[:-1])))
        self.assertFalse(verify_policy(replace(c,selected=()),p))
        # Impossible observations remain refused even if a mathematical caller asserts them.
        f=construct_fibre(3,((3,),))
        c=plan_attention(f,(Offer("a",(1,),1),Offer("b",(2,),1)),(1,2),Prices(10),max_packet=2)
        p=compile_policy(c)
        r=Runtime(c,p)
        s=r.begin("epoch","impossible")
        a=r.advance(s,"a",0)
        self.assertEqual(r.view(a).decision.next_offer,"b")
        # a=1 implies b=0, so this branch stops without acquiring b.
        b=r.advance(s,"a",1)
        self.assertTrue(r.view(b).terminal)
        with self.assertRaises(ValueError):
            r.advance(b,"b",1)

    def test_zero_horizon(self):
        c,p=fixture(3,(1,),(Offer("a",(1,),1),),max_packet=0)
        self.assertTrue(verify_policy(c,p))
        r=Runtime(c,p)
        self.assertTrue(r.view(r.begin("epoch","zero")).fallback_required)


if __name__=="__main__":
    unittest.main()
