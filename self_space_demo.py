"""NCS-001 deterministic source-to-self-contained-algebra demonstration."""
from __future__ import annotations
import json
from hashlib import sha256
from invariant_demo import grid
from navigable_compute.self_space import build_endogenous_processor
from navigable_compute.self_verify import verify_endogenous_processor


def run():
    desc = grid(5)
    machine = build_endogenous_processor(desc)
    assert verify_endogenous_processor(desc,machine)
    a, b = machine.generator_opcodes[:2]
    p_ab = machine.multiply[b][a] # execute a then b
    p_ba = machine.multiply[a][b] # execute b then a
    assert p_ab != p_ba
    w = (0, 1, 0, 1, 1, 0)
    far = 10**100 + 173
    start = desc.initial[0]
    value, comps = machine.repeat_program(w, far, start)
    assert machine.execute(w,start) == machine.interpret_own_opcode(machine.program_opcode(w),start)
    assert all(machine.interpret_own_opcode(i,s) == machine.execute_opcode(i,s)
               for i in range(machine.instruction_count) for s in machine.states)
    packet = machine.packet()
    return {
        'schema':'navigable-compute.ncs001.evidence.v0.1',
        'claim':'finite-invariant-derived-execution-algebra;not-a-universal-self-interpreter',
        'input_has_supplied_isa':False,
        'state_count':len(machine.states),
        'invariant_generated_laws':machine.source.law_count,
        'selected_generator_laws':len(machine.generator_coordinates),
        'distinct_semantic_opcodes':machine.instruction_count,
        'noncommutative':p_ab != p_ba,
        'different_order_program_opcodes':[p_ab,p_ba],
        'all_own_opcodes_have_internal_programs':True,
        'source_description_sha256':sha256(json.dumps(desc.encode(),sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        'witness_sha256':machine.digest,
        'far_repetition_coordinate_decimal':str(far),
        'far_output_state':value,
        'far_monoid_multiplications':comps,
        'independent_semantic_verification':True,
        'scope':{'operators':'exact finite transformation-monoid closure of 16 canonical laws',
                 'external_host':'still required; no substrate-free execution',
                 'complexity':'enumeration is explicitly bounded; operator algebra is task-and-invariant-relative'}
    }


if __name__ == '__main__':
    print(json.dumps(run(),indent=2,sort_keys=True))
