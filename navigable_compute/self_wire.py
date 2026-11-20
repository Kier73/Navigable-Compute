"""Portable NCS-001 processor capsule: authenticated *meaning* requires a trusted Ground.

The capsule is closed over its finite operator vocabulary and semantics, but
remains mathematical data interpreted on a physical host. The packet contains
no executable Python; it is rechecked from the caller's expected description.
"""
from __future__ import annotations

from .wire import _loads, _wrap, _decimal, _uint
from .invariant_space import InvariantDescription, InvariantTransformationSpace
from .invariant_verify import verify_invariant_space
from .self_space import EndogenousProcessor
from .self_verify import verify_endogenous_processor

KIND = 'task-derived-operator-algebra'


def encode_processor(desc: InvariantDescription, machine: EndogenousProcessor) -> bytes:
    if not verify_endogenous_processor(desc, machine):
        raise ValueError('cannot encode unverified mathematical processor')
    return _wrap(KIND, machine.packet())


def decode_processor(data: bytes, desc: InvariantDescription) -> EndogenousProcessor:
    doc = _loads(data)
    if doc['kind'] != KIND:
        raise ValueError('wrong processor capsule kind')
    p = doc['payload']
    fields = {'schema','source','generator_limit','states',
              'generator_coordinates_decimal','generator_opcodes','operators',
              'normal_forms','multiplication','claim','sha256'}
    if set(p) != fields or type(p['source']) is not dict:
        raise ValueError('unrecognized processor capsule structure')
    c = p['source']
    source_fields = {'schema','description','orbits','ranks','choices','law_count_decimal',
                     'convergence_bound','instance_coordinate_decimal','transition',
                     'claim','sha256'}
    if set(c) != source_fields or c['description'] != desc.encode():
        raise ValueError('source mathematical Ground differs from trusted description')
    try:
        source = InvariantTransformationSpace(desc,
            tuple(tuple(r) for r in c['orbits']),tuple(c['ranks']),
            tuple(tuple(r) for r in c['choices']),
            _decimal(c['law_count_decimal']), _uint(c['convergence_bound'],max_bits=16),
            _decimal(c['instance_coordinate_decimal']),tuple(c['transition']),c['sha256'])
        machine = EndogenousProcessor(source,_uint(p['generator_limit'],max_bits=8),
            tuple(p['states']),tuple(_decimal(v) for v in p['generator_coordinates_decimal']),
            tuple(p['generator_opcodes']),tuple(tuple(row) for row in p['operators']),
            tuple(tuple(row) for row in p['normal_forms']),
            tuple(tuple(row) for row in p['multiplication']),p['sha256'])
    except (TypeError,ValueError,IndexError,KeyError,OverflowError) as e:
        raise ValueError('malformed processor capsule') from e
    if (source.packet() != c or not verify_invariant_space(desc,source) or
            machine.packet() != p or not verify_endogenous_processor(desc,machine)):
        raise ValueError('independent processor reconstruction failed')
    return machine
