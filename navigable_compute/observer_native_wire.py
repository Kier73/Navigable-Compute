"""Canonical NCR-005 mathematics packets built on NCR-004's source decoder.

No packet authenticates an external observer or grants E-CPU host authority.
"""
from __future__ import annotations

from .invariant_space import canonical
from .preclosure_language import GeneratorObservation, PreclosureLanguage
from .preclosure_wire import (_parse, _keys, _decode_system, _decode_language,
                              MAX_PACKET)
from .observer_native import NativeAlgebra, NativeResult, SCHEMA, RESULT_SCHEMA
from .observer_native_verify import verify_native_algebra, verify_native_result

WIRE = 'navigable-compute.observer-native-wire.v0.1'


def _parse_algebra(record: object) -> NativeAlgebra:
    names = ('schema','dynamics_sha256','observation_sha256','language_sha256',
             'kind','prime','dimension','basis_generators','state_vectors',
             'opcode_actions','output_labels','readout','claim','sha256')
    _keys(record,names)
    for key in ('basis_generators','state_vectors','opcode_actions',
                'output_labels','readout'):
        if type(record[key]) is not list:
            raise ValueError('noncanonical algebra array')
    if (any(type(row) is not list for row in record['state_vectors']) or
        any(type(row) is not list for row in record['opcode_actions']) or
        record['schema'] != SCHEMA or
        record['claim'] != 'finite-exact-operator-intertwining;not-physical-admission'):
        raise ValueError('invalid mathematical representation')
    result = NativeAlgebra(record['dynamics_sha256'],record['observation_sha256'],
        record['language_sha256'],record['kind'],record['prime'],record['dimension'],
        tuple(record['basis_generators']),
        tuple(tuple(x) for x in record['state_vectors']),
        tuple(tuple(x) for x in record['opcode_actions']),
        tuple(record['output_labels']),tuple(record['readout']),record['sha256'])
    if result.packet() != record:
        raise ValueError('noncanonical algebra encoding')
    return result


def _parse_result(record: object) -> NativeResult:
    _keys(record,('schema','language_sha256','algebra_sha256','initial','word',
                  'repetitions_decimal','source_class','result_class','output',
                  'operator_products','claim','sha256'))
    if (record['schema'] != RESULT_SCHEMA or
        record['claim'] != 'exact-model-observation;not-external-future-evidence' or
        type(record['word']) is not list):
        raise ValueError('invalid observation evidence')
    result = NativeResult(record['language_sha256'],record['algebra_sha256'],
        record['initial'],tuple(record['word']),record['repetitions_decimal'],
        record['source_class'],record['result_class'],record['output'],
        record['operator_products'],record['sha256'])
    if result.packet() != record:
        raise ValueError('noncanonical observation encoding')
    return result


def encode_native(source: GeneratorObservation, language: PreclosureLanguage,
                  algebra: NativeAlgebra, result: NativeResult | None = None) -> bytes:
    if not verify_native_algebra(source, language, algebra):
        raise ValueError('independent algebra verification refused')
    if result is not None and not verify_native_result(source, language, algebra, result):
        raise ValueError('independent result verification refused')
    data = {'source':source.encode(), 'language':language.packet(),
            'algebra':algebra.packet(), 'result':result.packet() if result else None}
    raw = canonical({'schema':WIRE,'data':data})
    if len(raw) > MAX_PACKET:
        raise ValueError('portable mathematical packet budget exceeded')
    return raw


def decode_native(raw: bytes) -> tuple[GeneratorObservation,PreclosureLanguage,NativeAlgebra,NativeResult | None]:
    data = _parse(raw,WIRE,'data')
    _keys(data,('source','language','algebra','result'))
    source = _decode_system(data['source'])
    lang = _decode_language(data['language'])
    algebra = _parse_algebra(data['algebra'])
    result = _parse_result(data['result']) if data['result'] is not None else None
    if not verify_native_algebra(source,lang,algebra):
        raise ValueError('independent source/language/algebra audit refused')
    if result is not None and not verify_native_result(source,lang,algebra,result):
        raise ValueError('independent observation audit refused')
    return source,lang,algebra,result
