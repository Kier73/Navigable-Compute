"""NCI-001 mathematical certificate transport (not physical-source authority)."""
from __future__ import annotations

from hashlib import sha256
from .invariant_space import InvariantDescription,InvariantTransformationSpace,InvariantObstruction,canonical
from .invariant_verify import verify_invariant_space, verify_invariant_obstruction
from .wire import _wrap,_loads,_decimal,_uint


def encode_invariant(desc: InvariantDescription, cert: InvariantTransformationSpace) -> bytes:
    if not verify_invariant_space(desc,cert):
        raise ValueError('cannot transmit an unverified invariant space')
    return _wrap('constructed-invariant-space',cert.packet())


def decode_invariant(encoded:bytes, trusted_description:InvariantDescription)->InvariantTransformationSpace:
    doc = _loads(encoded)
    if doc['kind']!='constructed-invariant-space':
        raise ValueError('wrong invariant receipt kind')
    p=doc['payload']
    required={'schema','description','orbits','ranks','choices','law_count_decimal',
              'convergence_bound','instance_coordinate_decimal','transition','claim','sha256'}
    if set(p)!=required or p['description']!=trusted_description.encode():
        raise ValueError('unrecognized or mismatched source invariant description')
    try:
        cert=InvariantTransformationSpace(trusted_description,
            tuple(tuple(x) for x in p['orbits']),tuple(p['ranks']),
            tuple(tuple(x) for x in p['choices']),
            _decimal(p['law_count_decimal']),_uint(p['convergence_bound'],max_bits=16),
            _decimal(p['instance_coordinate_decimal']),tuple(p['transition']),p['sha256'])
    except (TypeError,ValueError,KeyError,IndexError,OverflowError) as e:
        raise ValueError('invalid invariant receipt fields') from e
    if cert.packet()!=p or not verify_invariant_space(trusted_description,cert):
        raise ValueError('independent invariant receipt check failed')
    return cert


def encode_invariant_obstruction(desc:InvariantDescription, witness:InvariantObstruction)->bytes:
    if not verify_invariant_obstruction(desc,witness):
        raise ValueError('invalid obstruction')
    return _wrap('constructed-invariant-obstruction',
                 {**witness.encode(),'sha256':witness.digest})


def decode_invariant_obstruction(encoded:bytes, trusted_description:InvariantDescription)->InvariantObstruction:
    doc=_loads(encoded)
    if doc['kind']!='constructed-invariant-obstruction':
        raise ValueError('wrong obstruction receipt kind')
    p=doc['payload']
    required={'schema','description','trapped_orbits','trapped_initial','claim','sha256'}
    if set(p)!=required or p['description']!=trusted_description.encode():
        raise ValueError('unrecognized obstruction profile')
    try:
        witness=InvariantObstruction(trusted_description,
                      tuple(tuple(x) for x in p['trapped_orbits']),
                      tuple(p['trapped_initial']),p['sha256'])
    except (TypeError,ValueError,KeyError,IndexError) as e:
        raise ValueError('invalid obstruction fields') from e
    if {**witness.encode(),'sha256':witness.digest}!=p or not verify_invariant_obstruction(trusted_description,witness):
        raise ValueError('independent obstruction check failed')
    return witness
