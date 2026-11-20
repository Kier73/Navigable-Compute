"""NCR-002 reproducible constructive self-interpretation experiment."""
from __future__ import annotations

from hashlib import sha256
import json

from invariant_demo import grid
from navigable_compute.self_space import build_endogenous_processor
from navigable_compute.reflective_architecture import (
    construct_reflective_architecture,architecture_run)
from navigable_compute.reflective_architecture_verify import (
    verify_architecture,verify_architecture_execution)
from navigable_compute.reflective_architecture_wire import (
    encode_architecture,decode_architecture,encode_execution,decode_execution)


def run():
    comparison=[]
    for geometry,coordinate in ((3,0),(5,17),(5,119)):
        machine=build_endogenous_processor(grid(geometry))
        design=construct_reflective_architecture(machine,
                   representation_coordinate=coordinate)
        assert verify_architecture(machine,design)
        architecture_wire=encode_architecture(machine,design)
        assert decode_architecture(architecture_wire,machine)==design
        word=(0,1,2) if geometry==5 else (0,1)
        results=[]
        for depth in (0,1,2):
            record=architecture_run(machine,design,word,machine.states[0],depth=depth)
            assert verify_architecture_execution(machine,design,record)
            output_wire=encode_execution(machine,design,record)
            assert decode_execution(output_wire,machine,design)==record
            results.append(record)
        assert len({r['result_state'] for r in results})==1
        comparison.append({
            'geometry':f'{geometry}x{geometry}',
            'qualified_semantic_operators':machine.instruction_count,
            'source_ground_sha256':machine.digest,
            'architecture_coordinate':coordinate,
            'architecture_sha256':design.digest,
            'source_codebook':dict(design.syntax_codebook),
            'primitive_seed':list(design.primitive_seed),
            'derived_operations':{op:recipe for op,recipe in design.recipes
                                  if not recipe.startswith('seed-')},
            'source_word':list(word),
            'same_result_at_depth_0_1_2':[r['result_state'] for r in results],
            'expression_steps_at_depth_0_1_2':[r['expression_steps'] for r in results],
            'architecture_packet_sha256':sha256(architecture_wire).hexdigest(),
            'self_interpretation_receipt_sha256':sha256(output_wire).hexdigest(),
        })
    return {
        'schema':'navigable-compute.ncr002.evidence.v0.1',
        'intention':'synthesize an encoding, primitive basis and self-interpreter from a declared finite semantic contract',
        'declared_semantic_operations':7,
        'physical_seed_operations':4,
        'represented_source_forms':5,
        'reachable_encoding_coordinates':120,
        'minimality_claim':'minimum seed cardinality within explicitly declared finite recipe grammar, not among all possible interpreters',
        'independent_verification':'alternate stack evaluator;finite semantic samples;macro prerequisite enumeration;source-bound receipts',
        'results':comparison,
        'limitations':[
           'five-form AST grammar and seven target operation semantics remain external declarations',
           'synthetic source encoding changes representation, not the underlying task semantics or computational complexity',
           'finite checks do not establish universal same-language reflection or physical self-hosting',
           'semantic macro identities require the declared well-typed operands',
           'fully qualified processor Ground is required; no physical effect is authenticated',
           'the self-interpreter runs on a separately implemented physical host with bounded evaluation fuel',
           'the finite recipe search does not solve unrestricted program synthesis, RH or P versus NP',
        ],
    }

if __name__=='__main__':
    print(json.dumps(run(),indent=2,sort_keys=True))
