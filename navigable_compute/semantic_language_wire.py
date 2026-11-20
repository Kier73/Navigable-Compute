"""Canonical, source-bound mathematical language receipts (no network authority)."""
from __future__ import annotations

from .semantic_language import SemanticLanguage, ObservationContract
from .semantic_language_verify import verify_semantic_language
from .wire import _wrap, _loads

KIND = 'observer-derived-language'


def encode_semantic_language(machine, language):
    if not verify_semantic_language(machine, language):
        raise ValueError('unqualified semantic-language certificate')
    return _wrap(KIND, language.packet())


def decode_semantic_language(payload, machine):
    doc = _loads(payload)
    if doc['kind'] != KIND or type(doc['payload']) is not dict:
        raise ValueError('incorrect language envelope')
    data = doc['payload']
    required = {'schema', 'source_sha256', 'observation', 'classes', 'class_of',
                'multiplication', 'generator_codes', 'normal_forms',
                'regular_actions', 'refinements', 'claim', 'sha256'}
    if set(data) != required:
        raise ValueError('malformed language packet')
    observation = data['observation']
    if type(observation) is not dict or set(observation) != {'initial','outputs'}:
        raise ValueError('malformed observer contract')
    fields = ('initial','outputs')
    if any(type(observation[k]) is not list for k in fields):
        raise ValueError('wrong observation encoding')
    matrix_fields = ('classes','multiplication','normal_forms','regular_actions')
    if (any(type(data[k]) is not list or len(data[k]) > 128 or
            any(type(v) is not list for v in data[k]) for k in matrix_fields) or
            type(data['class_of']) is not list or type(data['generator_codes']) is not list):
        raise ValueError('wrong algebra encoding')
    contract = ObservationContract(tuple(observation['initial']), tuple(observation['outputs']))
    obj = SemanticLanguage(
        data['source_sha256'], contract,
        tuple(tuple(x) for x in data['classes']), tuple(data['class_of']),
        tuple(tuple(x) for x in data['multiplication']),tuple(data['generator_codes']),
        tuple(tuple(x) for x in data['normal_forms']),
        tuple(tuple(x) for x in data['regular_actions']),
        data['refinements'], data['sha256']
    )
    if not verify_semantic_language(machine, obj):
        raise ValueError('semantic language rejected by independent context audit')
    return obj
