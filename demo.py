"""Demonstrate intention -> exact state Ground -> remote coordinate -> witness."""
from __future__ import annotations
from dataclasses import asdict
import json
from navigable_compute import Instruction as I, Program, Constraint, execute, compile_ground, synthesize, verify, ObservationAt, synthesize_trace, observability_profile

program = Program(4, (I('xor',1,0), I('rotl',0,3), I('xor_const',2,19),
                      I('swap',2,3), I('copy',1,0), I('xor',3,2)))
cycles=10**12+123
requests=(Constraint(0,255,0xA5), Constraint(3,0xF0,0xB0))
result=synthesize(program,cycles,requests)
print(json.dumps({
    'program_digest':program.digest,
    'goal': [[x.index,x.mask,x.value] for x in requests],
    'coordinate':str(cycles),
    'synthesized_initial_hex':result.initial.hex(),
    'observed_addresses':result.addresses,
    'observed_bytes_hex':result.observed.hex(),
    'program_bound_algebraic_recheck': verify(program,result,requests),
    'ground_composition_products':result.products,
    'baseline_instructions_if_replayed':cycles*len(program.instructions),
    'free_input_bits':result.free_bits,
    'receipt_sha256':result.witness_digest,
    'claim':'Exact for this byte-program family only; not RH/theta lift or physical speedup',
}, indent=2))

# Temporal observability: 8 one-bit observations reveal an entire hidden byte.
rotator=Program(1,(I('rotl',0,1),))
secret=0b10110101
trace=tuple(ObservationAt(t,0,1,(secret>>((-t)%8))&1) for t in range(8))
recovered=synthesize_trace(rotator,trace)
print(json.dumps({
  'experiment':'temporal-observability',
  'observer_mask':1,
  'observation_cycles':list(range(8)),
  'independent_constraint_rank_by_step':observability_profile(rotator,trace),
  'synthesized_hidden_byte':recovered.initial.hex(),
  'remaining_free_input_bits':recovered.free_bits,
  'claim':'A chain of observations accumulated rank in this known system; not a theta-lift theorem',
},indent=2))

