"""Executable contracts: exact Ground versus independently coded byte interpreter."""
import random
import unittest
from dataclasses import replace
from navigable_compute import (
    Instruction as I, Program, Constraint, execute, compile_ground, synthesize,
    verify, BudgetExceeded, UnsupportedGround, Unrealizable)
from navigable_compute.synthesis import solve_gf2

class MachineTests(unittest.TestCase):
    def test_all_256_single_byte_states(self):
        instructions = (I('xor',0,0), I('xor_const',0,175), I('rotl',0,3),
                        I('and_const',0,61), I('set',0,231))
        for ins in instructions:
            program = Program(1,(ins,))
            ground = compile_ground(program)
            for state in range(256):
                data=bytes([state])
                self.assertEqual(ground.apply(data),execute(program,data,1), str(ins))
                for cycles in (0,2,3,11):
                    self.assertEqual(ground.power(cycles).ground.apply(data),
                                     execute(program,data,cycles), str((ins,cycles,state)))

    def test_random_programs_multibyte(self):
        rng = random.Random(20261120)
        ops = ['xor', 'xor_const', 'rotl', 'swap', 'copy', 'and_const', 'set']
        for _ in range(220):
            size = rng.randint(2,6)
            instructions = tuple(I(op,rng.randrange(size),rng.randrange(size) if op in ('xor','swap','copy')
                                   else rng.randrange(256)) for op in (rng.choice(ops) for _ in range(rng.randint(1,15))))
            program = Program(size,instructions)
            data = bytes(rng.randrange(256) for _ in range(size))
            for cycles in (0,1,2,3,7,31):
                g=compile_ground(program).power(cycles).ground
                actual=execute(program,data,cycles)
                self.assertEqual(actual,g.apply(data),str((program,cycles,data.hex())))
                addresses=(0,size-1)
                self.assertEqual(bytes(actual[a] for a in addresses),g.observe(data,addresses))

    def test_power_multiplicative_law(self):
        p=Program(2,(I('xor',0,1),I('rotl',1,5),I('xor_const',0,31)))
        g=compile_ground(p)
        for m,n in ((0,0),(0,100),(32,45),(1234,7892)):
            self.assertEqual(g.power(m+n).ground, g.power(m).ground.after(g.power(n).ground))

    def test_far_coordinate_budget(self):
        p=Program(4,(I('xor',1,0),I('rotl',0,3),I('swap',2,3)))
        cycles = 10**30+123
        power=compile_ground(p).power(cycles)
        self.assertLess(power.products,2*cycles.bit_length())
        self.assertEqual(power.ground.apply(bytes([42,0,3,7])),
                         power.ground.apply(bytes([42,0,3,7])))
        with self.assertRaises(BudgetExceeded): execute(p,b'\x00'*4,cycles)

    def test_nonaffine_refusal_and_reference_interpreter(self):
        p=Program(1,(I('add_const',0,1),))
        self.assertEqual(execute(p,b'\xff',1),b'\x00')
        with self.assertRaises(UnsupportedGround): compile_ground(p)

    def test_bad_operands_and_inputs(self):
        with self.assertRaises(ValueError): Program(1,(I('swap',0,1),))
        with self.assertRaises(ValueError): execute(Program(1,(I('set',0,0),)),b'\x00',-1)
        with self.assertRaises(ValueError): Constraint(0,0,1)

class SynthesisTests(unittest.TestCase):
    def test_goal_instead_of_iteration(self):
        p=Program(4,(I('xor',1,0), I('rotl',0,3), I('xor_const',2,19),
                     I('swap',2,3),I('copy',1,0),I('xor',3,2)))
        goal=(Constraint(0,255,0xa5), Constraint(3,240,0xb0))
        result=synthesize(p,10**12+123,goal)
        self.assertTrue(verify(p,result,goal))
        self.assertLess(result.products,100)
        self.assertEqual(result.observed[0],0xa5)
        self.assertEqual(result.observed[1]&240,0xb0)
        self.assertGreater(result.free_bits,0)
        self.assertFalse(verify(p,replace(result,initial=b'\xff'*4),goal))
        self.assertFalse(verify(p,replace(result,witness_digest='0'*64),goal))
        self.assertFalse(verify(Program(4,(I('set',0,10),)),result,goal))

    def test_small_goal_synthesis_vs_sequential_reference(self):
        rng=random.Random(313)
        ops=['xor','xor_const','rotl','swap','copy','and_const','set']
        for _ in range(150):
            size=3
            instructions=tuple(I(op,rng.randrange(size),rng.randrange(size) if op in ('xor','swap','copy')
                                 else rng.randrange(256)) for op in (rng.choice(ops) for _ in range(7)))
            p=Program(size,instructions)
            cycles=rng.randrange(1,20)
            seed=bytes(rng.randrange(256) for _ in range(size))
            final=execute(p,seed,cycles)
            a=rng.randrange(size)
            goal=(Constraint(a,255,final[a]),)
            result=synthesize(p,cycles,goal)
            self.assertEqual(execute(p,result.initial,cycles)[a],final[a])
            self.assertTrue(verify(p,result,goal))

    def test_conflicting_and_unrealizable(self):
        p=Program(2,(I('set',0,0),))
        with self.assertRaises(Unrealizable): synthesize(p,1,(Constraint(0,255,123),))
        q=Program(2,(I('xor_const',0,1),))
        with self.assertRaises(Unrealizable):
            synthesize(q,5,(Constraint(1,255,42), Constraint(1,255,43)))

    def test_fixed_input_qualifiers(self):
        p=Program(2,(I('xor',1,0),))
        goal=(Constraint(1,255,0xab),)
        fixed=(Constraint(0,255,0x19),)
        s=synthesize(p,7,goal,fixed=fixed)
        self.assertEqual(s.initial[0],0x19)
        self.assertTrue(verify(p,s,goal,fixed=fixed))
        self.assertEqual(execute(p,s.initial,7)[1],0xab)
        with self.assertRaises(Unrealizable):
            synthesize(Program(1,(I('set',0,0),)),1,(Constraint(0,255,1),),fixed=(Constraint(0,255,0),))

    def test_solver_matches_bruteforce_small_systems(self):
        rng=random.Random(779)
        for width in range(1,9):
            for _ in range(40):
                equations=[(rng.randrange(1<<width),rng.randrange(2)) for __ in range(9)]
                brute=[x for x in range(1<<width) if all((row & x).bit_count()%2==rhs for row,rhs in equations)]
                if not brute:
                    with self.assertRaises(Unrealizable): solve_gf2(equations,width)
                else:
                    actual,rank=solve_gf2(equations,width)
                    self.assertIn(actual,brute)
                    self.assertLessEqual(rank,width)

if __name__=='__main__': unittest.main()

class TemporalObservabilityTests(unittest.TestCase):
    def test_rank_grows_on_rotating_observation(self):
        from navigable_compute import ObservationAt, synthesize_trace, verify_trace, observability_profile
        program=Program(1,(I('rotl',0,1),))
        target=0b10110101
        query=tuple(ObservationAt(t,0,1, (target >> ((-t)%8)) & 1) for t in range(8))
        self.assertEqual(observability_profile(program,query),tuple(range(1,9)))
        result=synthesize_trace(program,query)
        self.assertEqual(result.initial,bytes([target]))
        self.assertEqual(result.rank,8)
        self.assertEqual(result.free_bits,0)
        self.assertTrue(verify_trace(program,query,result))
        for q in query:
            self.assertEqual(execute(program,result.initial,q.cycle)[0] & q.mask,q.value)
        self.assertFalse(verify_trace(program,query,replace(result,rank=7)))

    def test_rank_does_not_necessarily_grow(self):
        from navigable_compute import ObservationAt, synthesize_trace, observability_profile
        program=Program(1,(I('set',0,0),))
        query=tuple(ObservationAt(t+1,0,255,0) for t in range(15))
        self.assertEqual(observability_profile(program,query),(0,)*15)
        result=synthesize_trace(program,query)
        self.assertEqual(result.rank,0)
        self.assertEqual(result.free_bits,8)

    def test_incompatible_temporal_observations(self):
        from navigable_compute import ObservationAt, synthesize_trace
        program=Program(1,(I('set',0,0),))
        with self.assertRaises(Unrealizable):
            synthesize_trace(program,(ObservationAt(1,0,1,1),))

