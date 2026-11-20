"""NCR-004 deterministic research: construction without source-operator closure.

A finite prototype for environment-conditioned languages and monotone revision,
NOT an external observer, live VL-I process, authenticated receipt or trained AI.
"""
from __future__ import annotations

from dataclasses import replace
from math import factorial
import json

from invariant_demo import grid
from navigable_compute.preclosure_language import (
    GeneratorObservation, generators_from_invariants,
    construct_preclosure_language, derive_language_revision)
from navigable_compute.preclosure_verify import verify_preclosure_language, verify_language_revision
from navigable_compute.preclosure_wire import (encode_preclosure,decode_preclosure,
                                                 encode_revision,decode_revision)


def cyclic_system(*, symmetry_group: bool, output_mode: str) -> GeneratorObservation:
    hidden=8
    states=tuple(range(2*hidden))
    toggle=tuple((1-v)*hidden+i for v in range(2) for i in range(hidden))
    cycle=tuple(v*hidden+(i+1)%hidden for v in range(2) for i in range(hidden))
    source=(toggle,cycle)
    if symmetry_group:
        flip=tuple(v*hidden+(1-i if i in (0,1) else i)
                   for v in range(2) for i in range(hidden))
        source+=(flip,)
    def observe(s):
        v,i=divmod(s,hidden)
        if output_mode=='visible':return f'visible-{v}'
        if output_mode=='parity':return f'visible-{v}-parity-{i%2}'
        if output_mode=='exact':return f'visible-{v}-hidden-{i}'
        raise ValueError('unknown demonstration observer')
    return GeneratorObservation(states,source,(0,),tuple(observe(s) for s in states),
                                'synthetic-declared-generator-group')


def independent_group_size(ground: GeneratorObservation) -> int:
    """Independent expensive oracle used only by demonstration, not builder."""
    ident=tuple(range(len(ground.states)))
    discovered={ident};worklist=[ident]
    for source in worklist:
        for g in ground.generators:
            next_map=tuple(g[source[i]] for i in ident)
            if next_map not in discovered:
                discovered.add(next_map);worklist.append(next_map)
    return len(discovered)


def run() -> dict:
    large=cyclic_system(symmetry_group=True,output_mode='visible')
    reduced=construct_preclosure_language(large,max_quotient_ops=2)
    assert verify_preclosure_language(large,reduced)
    assert decode_preclosure(encode_preclosure(large,reduced))==(large,reduced)
    original_size=independent_group_size(large)
    assert original_size==2*factorial(8)==80640
    fixed=cyclic_system(symmetry_group=False,output_mode='visible')
    middle=replace(fixed,outputs=cyclic_system(symmetry_group=False,output_mode='parity').outputs)
    detailed=replace(fixed,outputs=cyclic_system(symmetry_group=False,output_mode='exact').outputs)
    versions=[]
    for s in (fixed,middle,detailed):
        language=construct_preclosure_language(s)
        assert verify_preclosure_language(s,language)
        versions.append(language)
    revisions=[]
    for i,(before,after) in enumerate(((fixed,middle),(middle,detailed))):
        parent=versions[i];child=versions[i+1]
        revision=derive_language_revision(before,parent,after,child)
        assert verify_language_revision(before,parent,after,child,revision)
        assert decode_revision(encode_revision(before,parent,after,child,revision))[-1]==revision
        revisions.append({'from':parent.instruction_count,'to':child.instruction_count,
                          'new_to_old_opcode_map':list(revision.new_to_old_opcodes),
                          'reverse_information_loss':list(revision.lost_reverse_pair),
                          'digest':revision.digest})
    nci_cases=[]
    for width in (3,5,7):
        desc=grid(width)
        # Rank-free goal IDs are directly from input invariants, not the full
        # source monoid or a constructed instruction table.
        labels=tuple('goal' if idx in desc.goal else 'not-goal'
                     for idx in range(len(desc.coordinates)))
        ground=generators_from_invariants(desc,labels)
        machine=construct_preclosure_language(ground)
        assert verify_preclosure_language(ground,machine)
        nci_cases.append({'grid_width':width,'ground_states':len(ground.states),
                          'source_generators':len(ground.generators),
                          'observer_state_classes':len(machine.state_classes),
                          'derived_language_operations':machine.instruction_count,
                          'language_sha256':machine.digest})
    huge_time=10**100+173
    code,cost=reduced.power_code((0,1,2),huge_time)
    wrong=replace(fixed,generators=tuple(reversed(fixed.generators)))
    wrong_lang=construct_preclosure_language(wrong)
    refused=False
    try:
        derive_language_revision(fixed,versions[0],wrong,wrong_lang)
    except ValueError:
        refused=True
    assert refused
    return {
        'schema':'navigable-compute.ncr004.preclosure-environment-language.evidence.v0.1',
        'result':'finite-goal-relative-language-before-source-operator-closure',
        'hidden_symmetric_group':{
            'source_states':len(large.states),
            'source_transformations_independently_enumerated_by_oracle':original_size,
            'source_monoid_constructed_by_language_algorithm':False,
            'observational_quotient_states':len(reduced.state_classes),
            'derived_instructions':reduced.instruction_count,
            'source_generator_count':len(large.generators),
            'source_digest':large.dynamics_digest,
            'language_digest':reduced.digest,
            'portable_packet_bytes':len(encode_preclosure(large,reduced))},
        'invariant_generated_grounds':nci_cases,
        'information_refinement':{
            'language_sizes':[x.instruction_count for x in versions],
            'transfers':revisions,
            'same_dynamics_digest':len({x.dynamics_sha256 for x in versions})==1,
            'changed_transition_law_refused':refused,
            'continuity_means_mathematical_projection_not_authenticated_lineage':True},
        'logical_time':{'coordinate_decimal':str(huge_time),
                        'result_opcode':code,'multiplication_products':cost,
                        'sequential_source_steps_replayed':False},
        'scope':['finite deterministic stationary source generators',
                 'source transformations must be independently admitted',
                 'observer labels are stipulated, not independently measured',
                 'the independent full-source group enumeration is control-only',
                 'language refinement is not physical temporal law extrapolation',
                 'no live VL-I process or E-CPU authority is claimed',
                 'no universal acceleration or neural intelligence claim']}

if __name__=='__main__':
    print(json.dumps(run(),sort_keys=True,separators=(',',':')))
