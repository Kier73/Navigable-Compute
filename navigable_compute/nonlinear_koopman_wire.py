"""Strict canonical NCR-007 source/Koopman/result packets and full math audit.

Integrity-only portable records. No external evidence, signature, VGI
admission, E-CPU operator authorization or physical host is implied.
"""
from __future__ import annotations

import json

from .nonlinear_koopman import (
    canonical, PolynomialInvariants, BooleanOperator, PolynomialLawFamily,
    NonlinearKoopmanGround, NonlinearResult, SCHEMA, KOOPMAN_SCHEMA, RESULT_SCHEMA)
from .nonlinear_koopman_verify import (
    verify_polynomial_laws,verify_koopman_ground,verify_nonlinear_result)

WIRE='navigable-compute.nonlinear-observer-ground-wire.v0.1'
MAX_PACKET=2_000_000


def _unique(pairs):
    result={}
    for key,value in pairs:
        if key in result:
            raise ValueError('duplicate JSON field')
        result[key]=value
    return result


def _fields(obj,expected):
    if type(obj) is not dict or set(obj)!=set(expected):
        raise ValueError('unknown or missing packet fields')


def _decimal(value, max_bits: int):
    if (type(value) is not str or not value or not value.isascii() or
        not value.isdecimal() or len(value) > max_bits*30103//100000+3):
        raise ValueError('invalid bounded polynomial decimal')
    v=int(value)
    if str(v)!=value or v.bit_length()>max_bits:
        raise ValueError('invalid canonical decimal')
    return v


def _poly(row,width:int):
    if type(row) is not list:
        raise ValueError('polynomial monomials must be an ordered array')
    return tuple(_decimal(mon,width) for mon in row)


def _description(data):
    _fields(data,('width','tiers','symmetry','allowed_inputs','degree','observers','claim'))
    if (data['claim']!='rank-decreasing-topology-symmetry-conservation-invariants' or
        type(data['width']) is not int or
        type(data['tiers']) is not list or
        type(data['symmetry']) is not list or
        type(data['allowed_inputs']) is not list or
        any(type(row) is not list for row in data['allowed_inputs']) or
        type(data['observers']) is not list):
        raise ValueError('invalid invariant description')
    obj=PolynomialInvariants(data['width'],tuple(data['tiers']),
        tuple(data['symmetry']),tuple(tuple(x) for x in data['allowed_inputs']),
        data['degree'],tuple(_poly(row,data['width']) for row in data['observers']))
    if obj.encode()!=data:
        raise ValueError('noncanonical source invariants')
    return obj


def _family(doc):
    _fields(doc,('schema','description','coefficient_orbits','coordinates',
                 'operators','law_count_decimal','drain_horizon','claim','sha256'))
    if (doc['schema']!=SCHEMA or
        doc['claim']!='symbolic-nonlinear-equivariant-draining-law-family' or
        type(doc['coefficient_orbits']) is not list or
        type(doc['coordinates']) is not list or
        type(doc['operators']) is not list):
        raise ValueError('invalid law-family schema')
    desc=_description(doc['description'])
    if (any(type(orbit) is not list for orbit in doc['coefficient_orbits']) or
        any(type(pair) is not list or len(pair)!=2
            for orbit in doc['coefficient_orbits'] for pair in orbit)):
        raise ValueError('invalid template symmetry orbits')
    orbits=tuple(tuple((pair[0],_decimal(pair[1],desc.width)) for pair in orbit)
                 for orbit in doc['coefficient_orbits'])
    if any(type(row) is not list for op in doc['operators'] for row in op):
        raise ValueError('invalid Boolean-law polynomial')
    operators=tuple(BooleanOperator(tuple(_poly(row,desc.width) for row in op))
                    for op in doc['operators'])
    obj=PolynomialLawFamily(desc,orbits,
        tuple(_decimal(c,4096) for c in doc['coordinates']),
        operators,doc['law_count_decimal'],doc['sha256'])
    if obj.packet()!=doc:
        raise ValueError('noncanonical mathematical law family')
    return obj


def _ground(doc, width):
    _fields(doc,('schema','family_sha256','basis','operator_rows',
                 'observer_decoder','claim','sha256'))
    if (doc['schema'] != KOOPMAN_SCHEMA or
        doc['claim']!='least-pullback-invariant-observer-feature-linear-ground' or
        type(doc['basis']) is not list or
        type(doc['operator_rows']) is not list or
        type(doc['observer_decoder']) is not list or
        any(type(row) is not list for row in doc['basis']) or
        any(type(row) is not list for row in doc['operator_rows'])):
        raise ValueError('invalid feature algebra packet')
    obj=NonlinearKoopmanGround(doc['family_sha256'],
        tuple(_poly(row,width) for row in doc['basis']),
        tuple(tuple(_decimal(v,256) for v in row) for row in doc['operator_rows']),
        tuple(_decimal(v,256) for v in doc['observer_decoder']),doc['sha256'])
    if obj.packet()!=doc:
        raise ValueError('noncanonical feature algebra encoding')
    return obj


def _result(doc):
    _fields(doc,('schema','family_sha256','ground_sha256','initial_decimal',
                 'word','repetitions_decimal','initial_features_decimal',
                 'result_features_decimal','observed_decimal',
                 'operator_products','claim','sha256'))
    if (doc['schema']!=RESULT_SCHEMA or
        doc['claim']!='exact-nonlinear-source-observer-navigation;mathematical-only' or
        type(doc['word']) is not list):
        raise ValueError('invalid observer result')
    # Scalar decimal scope is verified against source width in the subsequent
    # independently reconstructed result audit.
    obj=NonlinearResult(doc['family_sha256'],doc['ground_sha256'],
        doc['initial_decimal'],tuple(doc['word']),doc['repetitions_decimal'],
        doc['initial_features_decimal'],doc['result_features_decimal'],
        doc['observed_decimal'],doc['operator_products'],doc['sha256'])
    if obj.packet()!=doc:
        raise ValueError('noncanonical result record')
    return obj


def encode_nonlinear(family: PolynomialLawFamily,ground: NonlinearKoopmanGround,
                     result: NonlinearResult | None=None) -> bytes:
    if not verify_koopman_ground(family,ground):
        raise ValueError('refused unverified observer feature algebra')
    if result is not None and not verify_nonlinear_result(family,ground,result):
        raise ValueError('refused unverified nonlinear source navigation')
    raw=canonical({'schema':WIRE,'data':{
        'family':family.packet(),'ground':ground.packet(),
        'result':result.packet() if result is not None else None}})
    if len(raw)>MAX_PACKET:
        raise ValueError('packet exceeds portable mathematics budget')
    return raw


def decode_nonlinear(raw:bytes):
    if type(raw) is not bytes or not 0<len(raw)<=MAX_PACKET:
        raise ValueError('invalid bounded byte packet')
    try:
        doc=json.loads(raw.decode('ascii'),object_pairs_hook=_unique,
                       parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite constant')))
    except (UnicodeError,TypeError,json.JSONDecodeError) as exc:
        raise ValueError('noncanonical serialized mathematics') from exc
    _fields(doc,('schema','data'))
    if doc['schema']!=WIRE or canonical(doc)!=raw:
        raise ValueError('noncanonical/wrong source packet')
    _fields(doc['data'],('family','ground','result'))
    family=_family(doc['data']['family'])
    ground=_ground(doc['data']['ground'],family.description.width)
    result=_result(doc['data']['result']) if doc['data']['result'] is not None else None
    if not verify_koopman_ground(family,ground):
        raise ValueError('independent polynomial law/feature verification failed')
    if result is not None and not verify_nonlinear_result(family,ground,result):
        raise ValueError('independent nonlinear observation reconstruction failed')
    return family,ground,result
