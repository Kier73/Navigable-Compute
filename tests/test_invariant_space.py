"""NCI-001: independent verification, exhaustive small-world oracle and refusal."""
import itertools
import json
import unittest
from dataclasses import replace
from hashlib import sha256

from navigable_compute.invariant_space import (
    InvariantDescription, NoInvariantRealization, construct_invariant_space,
    navigate_invariant_instance, certify_invariant_obstruction,
    construct_invariant_product, canonical)
from navigable_compute.invariant_verify import (
    verify_invariant_space, verify_invariant_obstruction, verify_invariant_product)
from navigable_compute.invariant_wire import (
    encode_invariant, decode_invariant, encode_invariant_obstruction,
    decode_invariant_obstruction)


def grid(side):
    points = tuple((x,y) for y in range(side) for x in range(side))
    symmetry = tuple(x*side + side-1-y for x,y in points)
    middle = (side-1)/2
    goals = tuple(i for i,(x,y) in enumerate(points)
                  if abs(x-middle)<1 and abs(y-middle)<1)
    return InvariantDescription(points,symmetry,('payload',)*len(points),
                                tuple(range(len(points))),goals,(),1)


def diamond():
    return InvariantDescription(((0,0),(1,0),(0,1),(1,1)),(3,2,1,0),
                                ('p',)*4,(1,2),(0,3),(),1)


def impossible():
    s=grid(3)
    return InvariantDescription(s.coordinates,s.symmetry,s.labels,(4,),
                                (0,2,6,8),(),2)


class InvariantSpaceTests(unittest.TestCase):
    def test_rotation_constructs_no_transition_table(self):
        d=grid(7)
        o=construct_invariant_space(d)
        self.assertEqual(o.law_count,512)
        self.assertEqual(o.convergence_bound,6)
        self.assertTrue(verify_invariant_space(d,o))
        self.assertEqual(o.observe(10**100+173),(24,))

    def test_exhaustive_four_state_oracle(self):
        d=diamond()
        o=construct_invariant_space(d)
        self.assertEqual(o.law_count,2)
        laws=[]
        # Independent reference: enumerate all 4**4 maps, with no use of
        # constructive orbit decomposition or its target list.
        for table in itertools.product(range(4),repeat=4):
            if any(table[g]!=g for g in d.goal):
                continue
            if any(table[d.symmetry[s]]!=d.symmetry[table[s]] for s in range(4)):
                continue
            if any(not 0<abs(d.coordinates[s][0]-d.coordinates[table[s]][0])+
                       abs(d.coordinates[s][1]-d.coordinates[table[s]][1])<=1
                   for s in d.initial):
                continue
            if any(table[s] not in d.goal for s in d.initial):
                continue
            laws.append(table)
        self.assertEqual(len(laws),2)
        found=[navigate_invariant_instance(o,i) for i in range(o.law_count)]
        self.assertEqual({f.transition for f in found},set(laws))
        self.assertTrue(all(verify_invariant_space(d,f) for f in found))

    def test_many_solutions_have_non_aliasing_architecture_coordinates(self):
        d=grid(7)
        base=construct_invariant_space(d)
        a=navigate_invariant_instance(base,17)
        b=navigate_invariant_instance(base,499)
        self.assertNotEqual(a.transition,b.transition)
        self.assertTrue(verify_invariant_space(d,a))
        self.assertTrue(verify_invariant_space(d,b))
        self.assertEqual(a.observe(10**100),(24,))
        self.assertEqual(b.observe(10**100),(24,))
        with self.assertRaises(ValueError):
            navigate_invariant_instance(base,512)
        with self.assertRaises(ValueError):
            navigate_invariant_instance(base,-1)

    def test_32_factor_enormous_implicit_space(self):
        ds=(grid(7),)*32
        result=construct_invariant_product(ds,coordinate=10**50+17)
        self.assertEqual(result.global_state_count,49**32)
        self.assertEqual(result.law_count,512**32)
        self.assertEqual(result.horizon,6)
        self.assertEqual(sum(len(f.description.coordinates) for f in result.factors),1568)
        self.assertTrue(verify_invariant_product(ds,result))
        self.assertEqual(result.observe(10**100+173),((24,),)*32)
        # Try a forged global coordinate and an apparently fresh self-digest.
        fake=replace(result,coordinate=1)
        fake=replace(fake,digest=sha256(canonical(fake.encode())).hexdigest())
        self.assertFalse(verify_invariant_product(ds,fake))

    def test_geometrically_reachable_but_symmetry_obstructed(self):
        d=impossible()
        # Center is distance 2 from the four corners, yet a fixed point
        # under rotation cannot map equivariantly to a four-cycle.
        with self.assertRaises(NoInvariantRealization):
            construct_invariant_space(d)
        w=certify_invariant_obstruction(d)
        self.assertTrue(verify_invariant_obstruction(d,w))
        self.assertIn(4,w.trapped_initial)
        p=encode_invariant_obstruction(d,w)
        self.assertEqual(decode_invariant_obstruction(p,d),w)

    def test_incompatibility_of_preserved_labels(self):
        d=diamond()
        new=InvariantDescription(d.coordinates,d.symmetry,
           ('goal','origin','origin','goal'),d.initial,d.goal,(),1)
        with self.assertRaises(NoInvariantRealization):
            construct_invariant_space(new)
        self.assertTrue(verify_invariant_obstruction(new,certify_invariant_obstruction(new)))

    def test_refuse_false_geometry_and_noninvariant_symmetry(self):
        d=diamond()
        with self.assertRaises(ValueError):
            replace(d,symmetry=(1,2,3,0))
        with self.assertRaises(ValueError):
            replace(d,goal=(0,))
        with self.assertRaises(ValueError):
            replace(d,coordinates=((0,0),)*4)
        with self.assertRaises(ValueError):
            replace(d,forbidden=(1,))

    def test_empty_sink_does_not_become_progress(self):
        d=InvariantDescription(((0,0),(2,0)),(0,1),('x','x'),
                               (1,),(0,),(),1)
        with self.assertRaises(NoInvariantRealization):
            construct_invariant_space(d)

    def test_unsafe_state_is_never_mapped_into(self):
        d=InvariantDescription(((0,0),(1,0),(2,0)),(0,1,2),
                               ('x',)*3,(2,),(0,),(1,),3)
        # A direct 2->0 jump is valid; unsafe intermediate never visited.
        obj=construct_invariant_space(d)
        self.assertEqual(obj.transition, (0,-1,0))
        self.assertTrue(verify_invariant_space(d,obj))

    def test_schema_roundtrip_and_independent_receiver(self):
        d=grid(5)
        q=navigate_invariant_instance(construct_invariant_space(d),13)
        data=encode_invariant(d,q)
        self.assertEqual(decode_invariant(data,d),q)
        with self.assertRaises(ValueError):
            decode_invariant(data,grid(7))
        with self.assertRaises(ValueError):
            decode_invariant(data.replace(b'"instance_coordinate_decimal":"13"',
                                               b'"instance_coordinate_decimal":"14"'),d)

    def test_forged_self_consistent_digest_is_still_rejected(self):
        d=grid(5)
        obj=construct_invariant_space(d)
        bad=replace(obj,ranks=tuple(0 for _ in obj.ranks),digest='')
        bad=replace(bad,digest=sha256(canonical(bad.encode())).hexdigest())
        self.assertFalse(verify_invariant_space(d,bad))
        with self.assertRaises(ValueError):
            encode_invariant(d,bad)

    def test_forged_transformation_with_unchanged_digest_is_rejected(self):
        d=grid(5)
        obj=construct_invariant_space(d)
        altered=list(obj.transition); altered[0]=24
        self.assertFalse(verify_invariant_space(d,replace(obj,transition=tuple(altered))))

    def test_noncanonical_and_duplicate_key_wire_rejected(self):
        d=grid(5)
        obj=construct_invariant_space(d)
        receipt=encode_invariant(d,obj)
        with self.assertRaises(ValueError):
            decode_invariant(b'{"x":0,"x":1}',d)
        with self.assertRaises(ValueError):
            decode_invariant(b'\n'+receipt,d)
        with self.assertRaises(ValueError):
            decode_invariant(receipt+b' ',d)

    def test_one_step_and_zero_step_observations(self):
        d=grid(5)
        obj=construct_invariant_space(d)
        self.assertEqual(obj.observe(0),d.initial)
        self.assertTrue(all(0<=s<25 for s in obj.observe(1)))
        self.assertEqual(obj.observe(10**90),(12,))
        with self.assertRaises(ValueError):
            obj.observe(-4)


if __name__ == '__main__':
    unittest.main()
