"""NCR-004 canonical portable mathematics packets; never authenticated telemetry.

Receivers reconstruct source semantics, verify the complete finite algebra and
reject noncanonical/extra/malformed fields. The digest is integrity-only.
"""
from __future__ import annotations

import json

from .invariant_space import canonical
from .preclosure_language import GeneratorObservation, PreclosureLanguage, LanguageRevision
from .preclosure_verify import verify_preclosure_language, verify_language_revision

MAX_PACKET = 2_000_000
LANGUAGE_WIRE_SCHEMA = 'navigable-compute.preclosure-math-wire.v0.1'
REVISION_WIRE_SCHEMA = 'navigable-compute.preclosure-revision-wire.v0.1'


def _reject_duplicate_keys(pairs):
    obj = {}
    for k, v in pairs:
        if k in obj:
            raise ValueError('duplicate key is not canonical')
        obj[k] = v
    return obj


def _parse(raw: bytes, schema: str, body_key: str):
    if type(raw) is not bytes or not 0 < len(raw) <= MAX_PACKET:
        raise ValueError('invalid bounded packet')
    try:
        body = json.loads(raw.decode('ascii'), object_pairs_hook=_reject_duplicate_keys,
                          parse_constant=lambda s: (_ for _ in ()).throw(ValueError('invalid constant')))
    except (UnicodeError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError('invalid packet JSON') from exc
    if (type(body) is not dict or set(body) != {'schema', body_key} or
        body['schema'] != schema or canonical(body) != raw):
        raise ValueError('noncanonical/unknown packet content')
    return body[body_key]


def _keys(x, names):
    if type(x) is not dict or set(x) != set(names):
        raise ValueError('missing/unknown fields')


def _decode_system(body):
    _keys(body, ('states', 'generators', 'initial', 'outputs', 'origin'))
    for key in ('states','initial','outputs','generators'):
        if type(body[key]) is not list:
            raise ValueError('noncanonical source array')
    if any(type(x) is not list for x in body['generators']):
        raise ValueError('noncanonical generator arrays')
    source = GeneratorObservation(tuple(body['states']),tuple(tuple(g) for g in body['generators']),
                                  tuple(body['initial']),tuple(body['outputs']),body['origin'])
    if source.encode() != body:
        raise ValueError('noncanonical Ground semantics')
    return source


def _decode_language(body):
    names=('schema', 'dynamics_sha256', 'observation_sha256','reachable',
           'state_classes','state_class_of','quotient_generators','operators',
           'normal_forms','generator_opcodes','multiplication','state_refinements',
           'claim','sha256')
    _keys(body,names)
    for key in ('reachable','state_classes','state_class_of','quotient_generators',
                'operators','normal_forms','generator_opcodes','multiplication'):
        if type(body[key]) is not list:
            raise ValueError('noncanonical language arrays')
    for key in ('state_classes','quotient_generators','operators','normal_forms','multiplication'):
        if any(type(x) is not list for x in body[key]):
            raise ValueError('noncanonical nested arrays')
    from .preclosure_language import SCHEMA
    if (body['schema'] != SCHEMA or
        body['claim'] != 'reachable-Moore-quotient-before-operator-closure;finite-stationary-only'):
        raise ValueError('wrong declared proof scope')
    obj=PreclosureLanguage(body['dynamics_sha256'],body['observation_sha256'],
        tuple(body['reachable']),tuple(tuple(x) for x in body['state_classes']),
        tuple(body['state_class_of']),tuple(tuple(x) for x in body['quotient_generators']),
        tuple(tuple(x) for x in body['operators']),tuple(tuple(x) for x in body['normal_forms']),
        tuple(body['generator_opcodes']),tuple(tuple(x) for x in body['multiplication']),
        body['state_refinements'],body['sha256'])
    if obj.packet() != body:
        raise ValueError('noncanonical language record')
    return obj


def encode_preclosure(system: GeneratorObservation, language: PreclosureLanguage) -> bytes:
    if not verify_preclosure_language(system, language):
        raise ValueError('unverified mathematical language')
    raw=canonical({'schema':LANGUAGE_WIRE_SCHEMA,
                   'data':{'source':system.encode(),'language':language.packet()}})
    if len(raw)>MAX_PACKET:
        raise ValueError('packet transport size exceeded')
    return raw


def decode_preclosure(raw: bytes) -> tuple[GeneratorObservation, PreclosureLanguage]:
    record=_parse(raw,LANGUAGE_WIRE_SCHEMA,'data')
    _keys(record,('source','language'))
    source=_decode_system(record['source'])
    language=_decode_language(record['language'])
    if not verify_preclosure_language(source,language):
        raise ValueError('independent source-anchored semantic verification failed')
    return source,language


def encode_revision(old_source: GeneratorObservation, old: PreclosureLanguage,
                    new_source: GeneratorObservation, new: PreclosureLanguage,
                    revision: LanguageRevision) -> bytes:
    if not verify_language_revision(old_source,old,new_source,new,revision):
        raise ValueError('unverified language revision')
    raw=canonical({'schema':REVISION_WIRE_SCHEMA,'data':{
        'prior_source':old_source.encode(),'prior_language':old.packet(),
        'new_source':new_source.encode(),'new_language':new.packet(),
        'revision':{**revision.encode(),'sha256':revision.digest}}})
    if len(raw)>MAX_PACKET:
        raise ValueError('packet transport size exceeded')
    return raw


def decode_revision(raw: bytes) -> tuple[GeneratorObservation, PreclosureLanguage,
                                         GeneratorObservation, PreclosureLanguage, LanguageRevision]:
    record=_parse(raw,REVISION_WIRE_SCHEMA,'data')
    _keys(record,('prior_source','prior_language','new_source','new_language','revision'))
    a=_decode_system(record['prior_source'])
    b=_decode_language(record['prior_language'])
    c=_decode_system(record['new_source'])
    d=_decode_language(record['new_language'])
    body=record['revision']
    _keys(body,('schema','prior_language_sha256','new_language_sha256',
                'dynamics_sha256','new_to_old_opcodes','new_to_old_states',
                'lost_reverse_pair','claim','sha256'))
    if body['schema'] != 'navigable-compute.preclosure-language-revision.v0.1' or \
       body['claim'] != 'monotone-observation-refinement;mathematical-only':
        raise ValueError('revision schema/claim mismatch')
    for name in ('new_to_old_opcodes','new_to_old_states'):
        if type(body[name]) is not list:
            raise ValueError('noncanonical revision array')
    if body['lost_reverse_pair'] is not None and (type(body['lost_reverse_pair']) is not list or len(body['lost_reverse_pair']) != 2):
        raise ValueError('invalid irreversibility witness')
    x=LanguageRevision(body['prior_language_sha256'],body['new_language_sha256'],
          body['dynamics_sha256'],tuple(body['new_to_old_opcodes']),
          tuple(body['new_to_old_states']),
          tuple(body['lost_reverse_pair']) if body['lost_reverse_pair'] is not None else None,
          body['sha256'])
    if {**x.encode(),'sha256':x.digest} != body or not verify_language_revision(a,b,c,d,x):
        raise ValueError('independent language revision verification failed')
    return a,b,c,d,x
