"""NCR-003 deterministic evidence: intention yields an exact language quotient."""
from __future__ import annotations

import json

from invariant_demo import grid
from navigable_compute.self_space import build_endogenous_processor
from navigable_compute.semantic_language import ObservationContract, derive_semantic_language
from navigable_compute.semantic_language_verify import verify_semantic_language
from navigable_compute.semantic_language_wire import encode_semantic_language,decode_semantic_language
from navigable_compute.language_morphism import (LanguageProjection,NonRefinementWitness,
                                                assess_language_projection)
from navigable_compute.language_morphism_verify import verify_language_relation


def observations(machine,kind):
    if kind=='goal':
        outputs=tuple('goal' if machine.source.ranks[s]==0 else 'not-goal'
                      for s in machine.states)
    elif kind=='exact':
        outputs=tuple(str(s) for s in machine.states)
    elif kind=='constant':
        outputs=('same',)*len(machine.states)
    else:raise ValueError('unknown kind')
    return ObservationContract(machine.states,outputs)


def run():
    small=build_endogenous_processor(grid(3))
    large=build_endogenous_processor(grid(5))
    records=[]
    languages={}
    for size,machine in ((3,small),(5,large)):
        source_count=len(machine.operators)
        for observer in ('exact','goal','constant'):
            contract=observations(machine,observer)
            design=derive_semantic_language(machine,contract)
            assert verify_semantic_language(machine,design)
            wire=encode_semantic_language(machine,design)
            assert decode_semantic_language(wire,machine)==design
            languages[(size,observer)] = design
            records.append({'side':size,'observer':observer,
                'source_opcodes':source_count,'derived_opcodes':design.instruction_count,
                'class_sizes':[len(c) for c in design.classes],
                'language_digest':design.digest,
                'wire_bytes':len(wire),
                'regular_interpreter_faithful':all(
                    design.apply_code_to_code(i,design.identity)==i
                    for i in range(design.instruction_count))})
    exact=languages[(5,'exact')]
    goal_lang=languages[(5,'goal')]
    constant=languages[(5,'constant')]
    to_goal=assess_language_projection(large,exact,goal_lang)
    to_constant=assess_language_projection(large,goal_lang,constant)
    impossible_reverse=assess_language_projection(large,goal_lang,exact)
    assert isinstance(to_goal,LanguageProjection)
    assert isinstance(to_constant,LanguageProjection)
    assert isinstance(impossible_reverse,NonRefinementWitness)
    assert verify_language_relation(large,exact,goal_lang,to_goal)
    assert verify_language_relation(large,goal_lang,constant,to_constant)
    assert verify_language_relation(large,goal_lang,exact,impossible_reverse)
    partial=ObservationContract((large.states[0],),tuple(
        'visible' if i in (0,2) else 'hidden' for i in range(len(large.states))))
    contextual=derive_semantic_language(large,partial)
    assert contextual.instruction_count==6
    assert verify_semantic_language(large,contextual)
    time=10**100+173
    goal=derive_semantic_language(large,observations(large,'goal'))
    far,products=goal.power_code((0,2,4),time)
    assert far==goal.instruction_count-1
    return {'schema':'navigable-compute.ncr003.observer-generated-language.evidence.v0.1',
            'qualification':'finite-source-algebra-and-exhaustive-context-only',
            'result':'language-algebra-generated-from-observation-invariants',
            'profiles':records,
            'language_revision':{'full_to_goal_mapping':list(to_goal.mapping),
                'goal_to_constant_mapping':list(to_constant.mapping),
                'reverse_nonrefinement_witness':[impossible_reverse.first_original_opcode,
                                                 impossible_reverse.second_original_opcode],
                'mathematical_only':True},
            'adversarial_context':{'initial':[large.states[0]],
                'one_sided_contexts_needed':True,'both_sides_class_count':6,
                'refinement_passes':contextual.refinements,
                'language_digest':contextual.digest},
            'distant_coordinate':{'logical_coordinate_decimal':str(time),
                'program':[0,2,4],'result_code':far,
                'composition_products':products,
                'explicit_trajectory_replayed':False},
            'limitations':['original-operator-monoid-must-first-be-constructed',
                'full finite context verification not general formal proof',
                'regular-action interpreter is not a universal AST self-interpreter',
                'source semantics and actual physical hosting remain distinct']}

if __name__=='__main__':
    print(json.dumps(run(),sort_keys=True,separators=(',',':')))
