"""Canonical portable NCR-006 packets, with independent mathematical audit.

Large source-state bitmasks use canonical decimal strings to preserve exactness
across languages. A digest binds bytes, not external observation or authority.
"""
from __future__ import annotations

import json

from .symbolic_quotient import (canonical, AffineMap, SymbolicSource,
                                SymbolicQuotient, SymbolicResult)
from .symbolic_quotient_verify import verify_symbolic_quotient, verify_symbolic_result

WIRE_SCHEMA = 'navigable-compute.symbolic-quotient-wire.v0.1'
MAX_BYTES = 2_000_000


def _pairs(values):
    doc = {}
    for k, v in values:
        if k in doc:
            raise ValueError('noncanonical duplicate JSON field')
        doc[k] = v
    return doc


def _fields(doc, names):
    if type(doc) is not dict or set(doc) != set(names):
        raise ValueError('unknown or missing mathematical packet fields')


def _decimal(value, limit=256):
    if type(value) is not str or not value or not value.isascii() or not value.isdecimal():
        raise ValueError('invalid decimal bitmask')
    if len(value) > limit * 30103 // 100000 + 3:
        raise ValueError('oversized decimal bitmask')
    x = int(value)
    if str(x) != value or x.bit_length() > limit:
        raise ValueError('noncanonical decimal bitmask')
    return x


def _map(doc, width):
    _fields(doc, ('rows', 'bias_decimal'))
    if type(doc['rows']) is not list:
        raise ValueError('missing operator rows')
    return AffineMap(tuple(_decimal(x,width) for x in doc['rows']),
                     _decimal(doc['bias_decimal'],width))


def _source(doc):
    _fields(doc, ('schema','width','generators','observer_rows','origin','claim'))
    if (doc['schema'] != 'navigable-compute.symbolic-affine-source.v0.1' or
        doc['claim'] != 'fully-declared-stationary-affine-source;not-external-observation' or
        type(doc['width']) is not int or not 1 <= doc['width'] <= 256 or
        type(doc['generators']) is not list or
        type(doc['observer_rows']) is not list):
        raise ValueError('invalid source schema')
    width = doc['width']
    source = SymbolicSource(width,tuple(_map(x,width) for x in doc['generators']),
                            tuple(_decimal(x,width) for x in doc['observer_rows']),
                            doc['origin'])
    if source.encode() != doc:
        raise ValueError('noncanonical source')
    return source


def _quotient(doc):
    _fields(doc,('schema','source_sha256','dimension','basis_rows',
                 'operators','observer_decoder','closure_rounds','claim','sha256'))
    if (doc['schema'] != 'navigable-compute.symbolic-observer-quotient.v0.1' or
        doc['claim'] != 'minimal-future-stable-linear-quotient-all-source-states' or
        type(doc['dimension']) is not int or not 0 <= doc['dimension'] <= 256 or
        type(doc['basis_rows']) is not list or
        type(doc['operators']) is not list or
        type(doc['observer_decoder']) is not list):
        raise ValueError('invalid symbolic quotient schema')
    d = doc['dimension']
    quotient = SymbolicQuotient(
        doc['source_sha256'],d,
        tuple(_decimal(v,256) for v in doc['basis_rows']),
        tuple(_map(x,d) for x in doc['operators']),
        tuple(_decimal(x,d) for x in doc['observer_decoder']),
        doc['closure_rounds'],doc['sha256'])
    if quotient.packet() != doc:
        raise ValueError('noncanonical quotient')
    return quotient


def _result(doc):
    _fields(doc,('schema','source_sha256','quotient_sha256','initial_decimal',
                 'word','repetitions_decimal','source_projection_decimal',
                 'final_projection_decimal','observed_decimal',
                 'operator_products','claim','sha256'))
    if (doc['schema'] != 'navigable-compute.symbolic-observer-result.v0.1' or
        doc['claim'] != 'exact-symbolic-law-navigation;not-external-future-evidence' or
        type(doc['word']) is not list):
        raise ValueError('invalid result schema')
    r = SymbolicResult(doc['source_sha256'],doc['quotient_sha256'],
                       doc['initial_decimal'],tuple(doc['word']),
                       doc['repetitions_decimal'],doc['source_projection_decimal'],
                       doc['final_projection_decimal'],doc['observed_decimal'],
                       doc['operator_products'],doc['sha256'])
    if r.packet() != doc:
        raise ValueError('noncanonical result')
    return r


def encode_symbolic(source: SymbolicSource, quotient: SymbolicQuotient,
                    result: SymbolicResult | None = None) -> bytes:
    if not verify_symbolic_quotient(source, quotient):
        raise ValueError('independent symbolic quotient audit refused')
    if result is not None and not verify_symbolic_result(source, quotient, result):
        raise ValueError('independent distant source observation audit refused')
    raw = canonical({'schema': WIRE_SCHEMA,'data':{
        'source':source.encode(), 'quotient':quotient.packet(),
        'result':result.packet() if result else None}})
    if len(raw) > MAX_BYTES:
        raise ValueError('portable mathematics packet exceeds budget')
    return raw


def decode_symbolic(raw: bytes):
    if type(raw) is not bytes or not 0 < len(raw) <= MAX_BYTES:
        raise ValueError('invalid bounded portable mathematical packet')
    try:
        data = json.loads(raw.decode('ascii'),object_pairs_hook=_pairs,
                          parse_constant=lambda x: (_ for _ in ()).throw(ValueError('noncanonical constant')))
    except (UnicodeError,TypeError,json.JSONDecodeError) as exc:
        raise ValueError('malformed mathematical packet') from exc
    _fields(data,('schema','data'))
    if data['schema'] != WIRE_SCHEMA or canonical(data) != raw:
        raise ValueError('wrong schema or noncanonical packet bytes')
    _fields(data['data'],('source','quotient','result'))
    source = _source(data['data']['source'])
    quotient = _quotient(data['data']['quotient'])
    result = _result(data['data']['result']) if data['data']['result'] is not None else None
    if not verify_symbolic_quotient(source,quotient):
        raise ValueError('source-anchored symbolic quotient proof refused')
    if result is not None and not verify_symbolic_result(source,quotient,result):
        raise ValueError('independent logical-time observation proof refused')
    return source,quotient,result
