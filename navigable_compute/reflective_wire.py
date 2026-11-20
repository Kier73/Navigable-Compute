"""Portable, source-bound NCR-001 reflection receipts, mathematical only."""
from __future__ import annotations

from .reflective_verify import verify_reflection
from .self_space import EndogenousProcessor
from .wire import _loads, _wrap

KIND = 'finite-self-interpreting-task-ground'


def encode_reflection(machine:EndogenousProcessor, receipt:dict) -> bytes:
    if not verify_reflection(machine,receipt):
        raise ValueError('cannot publish an unverified reflective execution')
    return _wrap(KIND,receipt)


def decode_reflection(data:bytes, trusted_machine:EndogenousProcessor) -> dict:
    doc=_loads(data)
    if doc['kind']!=KIND or not isinstance(doc['payload'],dict):
        raise ValueError('wrong or missing reflective receipt')
    receipt=doc['payload']
    if not verify_reflection(trusted_machine,receipt):
        raise ValueError('reflected program failed independent semantic audit')
    return receipt
