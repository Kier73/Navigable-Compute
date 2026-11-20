"""NCR-002 portable transport for generated interpreter designs and receipts."""
from __future__ import annotations

from .reflective_architecture import ReflectiveArchitecture
from .reflective_architecture_verify import verify_architecture,verify_architecture_execution
from .wire import _wrap,_loads

ARCH_KIND='constructed-reflective-architecture'
EXEC_KIND='constructed-reflective-execution'


def _tuplefy(obj, *, depth=0):
    if depth>120:raise ValueError('excessively nested reflective AST')
    if type(obj) is list:return tuple(_tuplefy(x,depth=depth+1) for x in obj)
    if type(obj) in (str,int,bool) or obj is None:return obj
    raise ValueError('invalid reflective wire datum')


def encode_architecture(machine,architecture):
    if not verify_architecture(machine,architecture):
        raise ValueError('unqualified reflective architecture')
    return _wrap(ARCH_KIND,architecture.packet())


def decode_architecture(payload,machine):
    doc=_loads(payload)
    if doc['kind']!=ARCH_KIND or type(doc['payload']) is not dict:
        raise ValueError('wrong architecture envelope')
    body=doc['payload']
    expected={'schema','ground_sha256','primitive_seed','recipes','syntax_codebook',
              'interpreter_source','representation_coordinate','representation_count','claim','sha256'}
    if set(body)!=expected:raise ValueError('malformed architecture packet')
    if (body['representation_count']!=120 or type(body['primitive_seed']) is not list or
        type(body['recipes']) is not list or type(body['syntax_codebook']) is not list):
        raise ValueError('invalid architecture encoding')
    architecture=ReflectiveArchitecture(body['ground_sha256'],
          tuple(body['primitive_seed']), _tuplefy(body['recipes']),
          _tuplefy(body['syntax_codebook']),_tuplefy(body['interpreter_source']),
          body['representation_coordinate'],body['sha256'])
    if not verify_architecture(machine,architecture):
        raise ValueError('architecture rejected by independent semantic checker')
    return architecture


def encode_execution(machine,architecture,receipt):
    if not verify_architecture_execution(machine,architecture,receipt):
        raise ValueError('unqualified constructed execution')
    return _wrap(EXEC_KIND,receipt)


def decode_execution(payload,machine,architecture):
    doc=_loads(payload)
    if doc['kind']!=EXEC_KIND or type(doc['payload']) is not dict:
        raise ValueError('wrong execution envelope')
    if not verify_architecture_execution(machine,architecture,doc['payload']):
        raise ValueError('execution rejected by independent semantic checker')
    return doc['payload']
