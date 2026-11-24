"""NCR-016: exact symbolic induction, independent semantic controls."""
import unittest
from navigable_compute.procedural_vocabulary import Vocabulary, Reduction
from navigable_compute.procedural_vocabulary_verify import verify_reduction
from navigable_compute.intention_fibre import FibreBudget
from test_procedural_vocabulary import fixture,held_out,PAIR_ORS


class SymbolicInductionTests(unittest.TestCase):
    def test_symbolic_induction_without_residual_formation(self):
        v=fixture()
        # Test the production method with cold qualification disabled.
        original=Vocabulary.exact
        def forbidden(*args,**kwargs):
            raise AssertionError("symbolic induction attempted cold query")
        try:
            Vocabulary.exact=forbidden
            added,trials,work=v.train_symbolic()
        finally:
            Vocabulary.exact=original
        self.assertEqual(added,PAIR_ORS[:2])
        self.assertGreater(trials,0)
        self.assertGreater(work,0)
        for i,learned in enumerate(v.learned):
            self.assertIs(type(learned.proof),Reduction)
            self.assertEqual(learned.proof.value,0)
            self.assertTrue(verify_reduction(v,learned.proof.target,learned.proof))
            self.assertEqual(v.exact(learned.target)[0],learned.bit)
        for i,q in enumerate(held_out()):
            self.assertEqual(v.reduce(q).value,i%2)
            self.assertIsNone(v.reduce(tuple(sorted(set(q)^{1<<127}))).value)
        self.assertEqual(v.reduce(PAIR_ORS[2]).value,1)
        restored=Vocabulary.restore(v.parent,v.observations,v.overlay,
                                     context=v.context,payload=v.packet())
        self.assertEqual(restored.rules,v.rules)

    def test_opposite_ground_and_budget(self):
        v=fixture(observations=((1,2),(12,20,24)))
        added,_,_=v.train_symbolic()
        self.assertNotIn(PAIR_ORS[0],added)
        self.assertIsNone(v.reduce(PAIR_ORS[0]).value)
        for learned in v.learned:
            self.assertEqual(v.exact(learned.target)[0],learned.bit)
        with self.assertRaises(FibreBudget):
            fixture().train_symbolic(max_trials=0)


if __name__=="__main__":
    unittest.main()
