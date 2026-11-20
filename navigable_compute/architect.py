"""NCA-001: a small typed entrypoint for description-to-realization.

The architect does not turn arbitrary requests into truthful physics. It
selects an exact qualified Ground lane and refuses unsupported requests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from .machine import Program
from .ground import compile_ground, UnsupportedGround
from .synthesis import Constraint
from .observability import ObservationAt, synthesize_trace, verify_trace, TraceResult
from .quotient import synthesize_quotient_trace, QuotientTraceReceipt
from .quotient_verify import verify_quotient_trace
from .law_synthesis import synthesize_additive_law, verify_additive_law, LawCertificate


@dataclass(frozen=True)
class Intention:
    size: int
    observations: tuple[ObservationAt, ...]
    program: Optional[Program] = None
    fixed_input: tuple[Constraint, ...] = ()


@dataclass(frozen=True)
class Realization:
    lane: str
    program: Program
    initial: bytes
    observations: tuple[int, ...]
    certificate: TraceResult | QuotientTraceReceipt | LawCertificate
    independent_checker: bool


def realize(intention: Intention, *, max_states: int=4096,
            max_coordinate_bits: int=4096) -> Realization:
    """Choose an exact construction family, without a claimed general solver.

    The boolean ``independent_checker`` describes whether a distinct finite
    interpreter/cycle finder or modular enumeration checked the result; the
    affine lane's built-in verifier is deterministic recomputation only.
    """
    if not isinstance(intention, Intention):
        raise TypeError('typed intention required')
    if not 1<=intention.size<=128 or not intention.observations:
        raise ValueError('invalid machine shape or missing observations')
    if intention.program is None:
        if intention.fixed_input:
            raise ValueError('law construction with fixed initial bytes is not yet admitted')
        cert=synthesize_additive_law(intention.size,intention.observations)
        if not verify_additive_law(intention.size,intention.observations,cert):
            raise AssertionError('independent law checker disagreed')
        return Realization('construct-additive-law',cert.program,cert.initial,cert.observed,cert,True)
    program=intention.program
    if program.size!=intention.size:
        raise ValueError('declared machine size disagrees with program')
    try:
        compile_ground(program)
    except UnsupportedGround:
        cert=synthesize_quotient_trace(program,intention.observations,fixed=intention.fixed_input,
                                      max_states=max_states,max_coordinate_bits=max_coordinate_bits)
        if not verify_quotient_trace(program,intention.observations,cert,fixed=intention.fixed_input,
                                     max_states=max_states,max_coordinate_bits=max_coordinate_bits):
            raise AssertionError('independent quotient checker disagreed')
        return Realization('finite-observer-quotient',program,cert.initial,cert.observed,cert,True)
    cert=synthesize_trace(program,intention.observations,fixed=intention.fixed_input)
    if not verify_trace(program,intention.observations,cert,fixed=intention.fixed_input):
        raise AssertionError('affine Ground checker disagreed')
    return Realization('exact-gf2-affine',program,cert.initial,cert.rendered,cert,False)
