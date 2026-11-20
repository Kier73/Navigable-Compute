"""NCW-001: strict, portable, canonical JSON mathematical receipt transport.

Transport profile ONLY. The wire digest protects accidental alterations but is
not an identity signature, a quorum vote, or evidence of outside observations.
A receiver supplies the trusted Program identity and rechecks every relation.
"""
from __future__ import annotations
from hashlib import sha256
import json

from .machine import Program, Instruction
from .observability import ObservationAt
from .synthesis import Constraint
from .quotient import QuotientTraceReceipt, ComponentCertificate, canonical_bytes
from .quotient_verify import verify_quotient_trace
from .law_synthesis import LawCertificate, verify_additive_law

WIRE_SCHEMA='navigable-compute.math-receipt-envelope.v0.1'
MAX_WIRE_BYTES=2_000_000


def _pairs_unique(items):
    d={}
    for k,v in items:
        if k in d:
            raise ValueError('duplicate JSON object key')
        d[k]=v
    return d


def _loads(encoded: bytes) -> dict:
    if not isinstance(encoded,bytes) or len(encoded)>MAX_WIRE_BYTES:
        raise ValueError('not bounded receipt bytes')
    try:
        doc=json.loads(encoded.decode('ascii'),object_pairs_hook=_pairs_unique,
                       parse_constant=lambda v: (_ for _ in ()).throw(ValueError('invalid floating token')))
    except (UnicodeError,TypeError,json.JSONDecodeError) as exc:
        raise ValueError('invalid JSON receipt') from exc
    if not isinstance(doc,dict) or canonical_bytes(doc)!=encoded:
        raise ValueError('noncanonical receipt')
    if set(doc)!= {'schema','kind','payload','payload_sha256'} or doc['schema']!=WIRE_SCHEMA:
        raise ValueError('unknown wire receipt structure')
    if not isinstance(doc['payload'],dict):
        raise ValueError('invalid payload type')
    if sha256(canonical_bytes(doc['payload'])).hexdigest()!=doc['payload_sha256']:
        raise ValueError('payload digest mismatch')
    return doc


def _wrap(kind:str, payload:dict)->bytes:
    raw=canonical_bytes(payload)
    result=canonical_bytes({'schema':WIRE_SCHEMA,'kind':kind,'payload':payload,
                            'payload_sha256':sha256(raw).hexdigest()})
    if len(result)>MAX_WIRE_BYTES:
        raise ValueError('receipt exceeds wire budget')
    return result


def _uint(v, *, max_bits=4096):
    if isinstance(v,bool) or not isinstance(v,int) or v<0 or v.bit_length()>max_bits:
        raise ValueError('invalid unsigned integer field')
    return v


def _decimal(v, *, max_bits=4096):
    if not isinstance(v,str) or not v or (len(v)>1 and v[0]=='0') or not v.isascii() or not v.isdecimal():
        raise ValueError('noncanonical decimal coordinate')
    n=int(v)
    return _uint(n,max_bits=max_bits)


def _parse_queries(payload:dict):
    q=tuple(ObservationAt(_decimal(t),_uint(a,max_bits=16),_uint(mask,max_bits=8),_uint(val,max_bits=8))
            for t,a,mask,val in payload['queries'])
    return q


def _parse_fixed(payload:dict):
    return tuple(Constraint(_uint(a,max_bits=16),_uint(mask,max_bits=8),_uint(val,max_bits=8))
                 for a,mask,val in payload['fixed'])


def encode_quotient(program: Program, queries:tuple[ObservationAt,...],
                    receipt:QuotientTraceReceipt, *, fixed:tuple[Constraint,...]=())->bytes:
    if not verify_quotient_trace(program,queries,receipt,fixed=fixed):
        raise ValueError('producer attempted to export an invalid mathematical receipt')
    return _wrap('finite-observer-quotient',receipt.packet(queries,fixed))


def decode_quotient(encoded:bytes, program:Program):
    """Return (queries, fixed, checked receipt) or refuse. Never trusts self-hash."""
    doc=_loads(encoded)
    if doc['kind']!='finite-observer-quotient':
        raise ValueError('wrong receipt kind')
    p=doc['payload']
    if set(p)!={'schema','program_sha256','queries','fixed','initial_hex','rendered_masked',
               'candidate_count_decimal','global_state_bits','examined_states','quotient_products',
               'factors','certificates'} or p['schema']!='navigable-compute.quotient-trace.v0.2':
        raise ValueError('invalid quotient packet shape')
    try:
        q=_parse_queries(p)
        fixed=_parse_fixed(p)
        certs=tuple(ComponentCertificate(tuple(c['indices']),tuple(tuple(x) for x in c['observed_masks']),
                                         tuple(c['class_of']),tuple(tuple(x) for x in c['labels']),
                                         tuple(c['successor']),c['transition_sha256'],c['refinement_rounds'])
                    for c in p['certificates'])
        r=QuotientTraceReceipt(bytes.fromhex(p['initial_hex']),tuple(p['rendered_masked']),
                  _decimal(p['candidate_count_decimal'],max_bits=2048),
                  _uint(p['global_state_bits'],max_bits=16),
                  _uint(p['examined_states'],max_bits=24),
                  _uint(p['quotient_products'],max_bits=32),
                  tuple(tuple(x) for x in p['factors']),certs,p['program_sha256'],
                  sha256(canonical_bytes(p)).hexdigest())
    except (TypeError,ValueError,IndexError,KeyError,OverflowError) as exc:
        raise ValueError('malformed quotient fields') from exc
    if r.packet(q,fixed)!=p or not verify_quotient_trace(program,q,r,fixed=fixed):
        raise ValueError('invalid quotient mathematical receipt')
    return q,fixed,r


def encode_law(size:int, queries:tuple[ObservationAt,...], receipt:LawCertificate)->bytes:
    if not verify_additive_law(size,queries,receipt):
        raise ValueError('invalid constructed-law receipt')
    return _wrap('construct-additive-law',receipt.packet(queries))


def decode_law(encoded:bytes):
    """Return (queries, independently verified law) or refuse."""
    doc=_loads(encoded)
    if doc['kind']!='construct-additive-law':
        raise ValueError('wrong receipt kind')
    p=doc['payload']
    if set(p)!={'schema','grammar','program','program_sha256','initial_hex','queries','observed',
               'solutions_decimal','delta_congruences'}:
        raise ValueError('wrong law packet fields')
    if p['schema']!='navigable-compute.law-construction.v0.1' or p['grammar']!='independent-byte-additive-mod-256':
        raise ValueError('unsupported law profile')
    try:
        q=_parse_queries(p)
        raw=p['program']
        if set(raw)!={'size','instructions'}:
            raise ValueError('noncanonical program profile')
        program=Program.decode(raw)
        cert=LawCertificate(program,bytes.fromhex(p['initial_hex']),
                            _decimal(p['solutions_decimal'],max_bits=2048),
                            tuple(tuple(x) for x in p['delta_congruences']),
                            tuple(p['observed']),sha256(canonical_bytes(p)).hexdigest())
    except (TypeError,ValueError,KeyError,IndexError,OverflowError) as exc:
        raise ValueError('malformed law fields') from exc
    if cert.packet(q)!=p or not verify_additive_law(program.size,q,cert):
        raise ValueError('invalid law mathematical receipt')
    return q,cert

# NCQ-002 is intentionally a distinct profile: a finite target-time quotient
# is not transition-stable and cannot be silently interpreted as NCQ-001.
from .task_quotient import TaskComponentCertificate,TaskTraceReceipt
from .task_verify import verify_task_trace


def encode_task(program:Program, queries:tuple[ObservationAt,...],
                receipt:TaskTraceReceipt, *, fixed:tuple[Constraint,...]=())->bytes:
    if not verify_task_trace(program,queries,receipt,fixed=fixed):
        raise ValueError('invalid task-scoped receipt')
    return _wrap('task-scoped-quotient',receipt.packet(queries,fixed))


def decode_task(encoded:bytes, trusted_program:Program):
    doc=_loads(encoded)
    if doc['kind']!='task-scoped-quotient':
        raise ValueError('not a task-scoped receipt')
    p=doc['payload']
    if set(p)!={'schema','scope','program_sha256','queries','fixed','initial_hex',
               'observed_masked','candidate_count_decimal','global_bits','examined_states',
               'power_products','factors','certificates'}:
        raise ValueError('task receipt shape mismatch')
    if (p['schema']!='navigable-compute.task-quotient.v0.1'
        or p['scope']!='only-the-declared-temporal-observations'):
        raise ValueError('unsupported task scope')
    try:
        queries=_parse_queries(p)
        fixed=_parse_fixed(p)
        certs=tuple(TaskComponentCertificate(tuple(c['indices']),tuple(c['class_of']),
                             tuple(tuple(v) for v in c['labels']),c['transition_sha256'])
                    for c in p['certificates'])
        r=TaskTraceReceipt(bytes.fromhex(p['initial_hex']),tuple(p['observed_masked']),
                           _decimal(p['candidate_count_decimal'],max_bits=2048),
                           _uint(p['global_bits'],max_bits=16),
                           _uint(p['examined_states'],max_bits=24),
                           _uint(p['power_products'],max_bits=32),
                           tuple(tuple(c) for c in p['factors']),certs,
                           p['program_sha256'],sha256(canonical_bytes(p)).hexdigest())
    except (TypeError,ValueError,KeyError,IndexError,OverflowError) as exc:
        raise ValueError('invalid task receipt fields') from exc
    if r.packet(queries,fixed)!=p or not verify_task_trace(trusted_program,queries,r,fixed=fixed):
        raise ValueError('independent task receipt check failed')
    return queries,fixed,r
