"""NCR-001: source-ground reflection, independent operational audit, and cost."""
from __future__ import annotations

import json
from hashlib import sha256
from invariant_demo import grid
from navigable_compute.self_space import build_endogenous_processor
from navigable_compute.reflective_ground import reflect, compile_reflector
from navigable_compute.reflective_verify import verify_reflection
from navigable_compute.reflective_wire import encode_reflection,decode_reflection


def run():
    large=build_endogenous_processor(grid(5))
    small=build_endogenous_processor(grid(3))
    generator_word=(0,1,2)
    word=tuple(large.generator_opcodes[k] for k in generator_word)
    initial=large.states[0]
    results=[reflect(large,word,initial,depth=d) for d in range(3)]
    assert all(verify_reflection(large,v) for v in results)
    assert len(set(r['result_state'] for r in results))==1
    encoded=encode_reflection(large,results[2])
    assert decode_reflection(encoded,large)==results[2]
    smaller=reflect(small,(0,1,2),small.states[0],depth=2)
    assert verify_reflection(small,smaller)
    assert smaller['interpreter_program_sha256']==results[2]['interpreter_program_sha256']
    direct,once,twice=results
    opcode=large.program_opcode(generator_word)
    assert large.execute_opcode(opcode,initial)==twice['result_state']
    far=10**100+173
    far_state,products=large.repeat_program(generator_word,far,initial)
    return {
        'schema':'navigable-compute.ncr001.evidence.v0.1',
        'claim':'finite-same-language-self-interpretation;task-data-semantics;not-universal-self-hosted-hardware',
        'task_source':'5x5 NCI invariant geometry -> 16 laws -> 29 operator algebra',
        'source_ground_sha256':large.digest,
        'interpreted_operator_count':large.instruction_count,
        'source_word':list(word),
        'source_initial_state':initial,
        'output_at_all_reflection_depths':[r['result_state'] for r in results],
        'host_ast_steps_by_depth':[r['host_evaluation_steps'] for r in results],
        'program_sha256':twice['task_program_sha256'],
        'self_interpreter_sha256':twice['interpreter_program_sha256'],
        'interpreted_interpreter_is_encoded_in_its_own_language':True,
        'second_operational_checker_reconstructed_result_and_cost':True,
        'portable_receipt_bytes':len(encoded),
        'portable_receipt_sha256':sha256(encoded).hexdigest(),
        'comparison_world':{
            'task_source':'3x3 NCI invariant geometry',
            'operators':small.instruction_count,
            'different_ground_digest':small.digest!=large.digest,
            'same_reflective_language_and_interpreter_code':smaller['interpreter_program_sha256']==twice['interpreter_program_sha256']
        },
        'separately_specialized_navigation':{
            'macro_opcode':opcode,
            'logical_coordinate_decimal':str(far),
            'far_result':far_state,
            'monoid_products_after_construction':products,
            'not_reflective_host_steps':True
        },
        'limitations':[
            'finite NCI semantic algebra and explicitly declared tuple/call/conditional reflection seed',
            'host Python executes the reflective interpreter AST',
            'two-level self-interpretation is not universal self-hosting',
            'fuel and recursion bounds can cause legitimate programs to be refused',
            'different task programs may not share a convergent executable evaluator in general',
            'no RH theorem, physical host authorization, or complexity-class implication'
        ]
    }


if __name__=='__main__':
    print(json.dumps(run(),indent=2,sort_keys=True))
