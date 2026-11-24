"""NCR-012 task-versus-information and geometry-versus-information controls."""
import random
import unittest
from dataclasses import replace
from navigable_compute.intention_fibre import construct_fibre, FibreBudget
from navigable_compute.fibre_attention import Offer, Prices, plan_attention, select_outcome
from navigable_compute.fibre_attention_verify import verify_attention


def value(poly, x):
    return sum((x & m) == m for m in poly) % 2


def eligible(rows, x):
    return not any(value(p,x) for p in rows)


def candidate(plan, *names):
    return next(c for c in plan.candidates if c.names == names)


class AttentionTests(unittest.TestCase):
    def test_complementary_packet(self):
        f = construct_fibre(8, ())
        offers = (Offer("a", (1,), 1), Offer("b", (2,), 1))
        target = (1, 2)
        prices = Prices(10)
        single = plan_attention(f, offers, target, prices)
        self.assertEqual(single.selected, ())
        p = plan_attention(f, offers, target, prices, max_packet=2)
        self.assertEqual(p.selected, ("a","b"))
        self.assertTrue(verify_attention(f,offers,target,prices,p,max_packet=2))
        self.assertEqual(candidate(p,"a").fallback_mass, f.count)
        self.assertEqual(candidate(p,"a","b").target_pairs, 0)
        for a in (0,1):
            for b in (0,1):
                self.assertEqual(select_outcome(p,(("a",a),("b",b))).inferred,a^b)

    def test_information_is_not_task_value(self):
        f = construct_fibre(16, ())
        target = (3 << 14,)
        offers = (Offer("balanced", (1,), 1), Offer("gate", target, 1))
        prices = Prices(20)
        p = plan_attention(f,offers,target,prices)
        self.assertTrue(verify_attention(f,offers,target,prices,p))
        a,b = candidate(p,"balanced"),candidate(p,"gate")
        self.assertLess(a.unresolved_pairs,b.unresolved_pairs)
        self.assertEqual(p.selected,("gate",))
        self.assertEqual(a.fallback_mass,f.count)
        self.assertEqual(b.fallback_mass,0)
        self.assertEqual(select_outcome(p,(("gate",1),)).count,2**14)

    def test_geometry_breaks_information_tie(self):
        f = construct_fibre(16, ())
        hi,lo = 1<<15,1
        target = (lo,hi,lo|hi)
        offers = (Offer("a-low",(lo,),1),Offer("z-high",(hi,),1))
        baseline = plan_attention(f,offers,target,Prices(20))
        self.assertEqual(baseline.selected,("a-low",))
        prices = Prices(20,node=1,address_visit=1,address_queries=1)
        p = plan_attention(f,offers,target,prices)
        self.assertTrue(verify_attention(f,offers,target,prices,p))
        a,b = candidate(p,"a-low"),candidate(p,"z-high")
        self.assertEqual(a.unresolved_pairs,b.unresolved_pairs)
        self.assertEqual(a.target_pairs,b.target_pairs)
        self.assertEqual(a.fallback_mass,b.fallback_mass)
        self.assertEqual(p.selected,("z-high",))
        self.assertEqual((a.node_mass//f.count,b.node_mass//f.count),(16,1))
        self.assertEqual((a.visit_mass//f.count,b.visit_mass//f.count),(16,1))
        for o in b.outcomes:
            self.assertEqual(o.overlay.parent_address(0),o.bits[0]*2**15)

    def test_exhaustive_small_oracle(self):
        rng = random.Random(12012)
        for _ in range(40):
            n = rng.randint(2,6)
            def polynomial():
                return tuple(sorted(rng.sample(range(2**n),rng.randint(0,4))))
            rows = tuple(polynomial() for _ in range(rng.randint(0,3)))
            states = [x for x in range(2**n) if eligible(rows,x)]
            if not states:
                continue
            f = construct_fibre(n,rows)
            offers = tuple(Offer(chr(97+i),polynomial(),rng.randint(1,4)) for i in range(3))
            target,prices = polynomial(),Prices(9,1,1,2)
            p = plan_attention(f,offers,target,prices,max_packet=2)
            self.assertTrue(verify_attention(f,offers,target,prices,p,max_packet=2))
            for c in p.candidates:
                packet = [o for o in offers if o.name in c.names]
                for o in c.outcomes:
                    expected = [x for x in states if tuple(value(q.polynomial,x) for q in packet)==o.bits]
                    self.assertEqual(o.count,len(expected))
                    self.assertEqual(o.yes,sum(value(target,x) for x in expected))
                    self.assertEqual([o.overlay.unrank(i) for i in range(o.count)],expected)
                    v = 0
                    for i in range(o.count):
                        ref,j = o.overlay.root,i
                        while ref > 0:
                            v += 1
                            node = o.overlay.nodes[ref-1]
                            left = o.overlay.count_at(node.low)
                            if j < left:
                                ref = node.low
                            else:
                                j -= left
                                ref = node.high
                    self.assertEqual(o.visits,v)

    def test_tamper_impossible_and_budgets(self):
        f = construct_fibre(4, ((1,),))
        offers = (Offer("fixed",(1,),1),Offer("other",(2,),1))
        target,prices = (2,),Prices(5)
        p = plan_attention(f,offers,target,prices,max_packet=2)
        self.assertTrue(verify_attention(f,offers,target,prices,p,max_packet=2))
        self.assertFalse(verify_attention(f,offers,target,prices,replace(p,selected=()),max_packet=2))
        self.assertFalse(verify_attention(f,offers,target,prices,replace(p,candidates=p.candidates[:-1]),max_packet=2))
        c = p.candidates[-1]
        for bad in (replace(c,score_numerator=c.score_numerator+1),
                    replace(c,outcomes=(replace(c.outcomes[0],visits=99),)+c.outcomes[1:])):
            self.assertFalse(verify_attention(f,offers,target,prices,
                replace(p,candidates=p.candidates[:-1]+(bad,)),max_packet=2))
        # A deliberately expensive fallback forces a jointly needed packet
        # only in the complementary test; here other alone is optimal.
        with self.assertRaises(ValueError):
            select_outcome(p,(("other",True),))
        # Directly test a zero-mass observed branch on a valid chosen packet:
        # the second probe is nonconstant, but its joint outcome with the
        # first is impossible under the constraint a*b=0.
        g = construct_fibre(3,((3,),))
        q = plan_attention(g,(Offer("a",(1,),1),Offer("b",(2,),1)),(1,2),Prices(10),max_packet=2)
        self.assertEqual(q.selected,("a","b"))
        with self.assertRaises(ValueError):
            select_outcome(q,(("a",1),("b",1)))
        with self.assertRaises(FibreBudget):
            plan_attention(f,offers,target,prices,max_nodes=0)
        with self.assertRaises(ValueError):
            plan_attention(construct_fibre(3,((0,),)),offers,target,prices)


if __name__ == "__main__":
    unittest.main()
