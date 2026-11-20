"""Navigable-Compute: exact executable byte-state Grounds and intention synthesis."""
from .machine import Instruction, Program, execute, BudgetExceeded
from .ground import Ground, PowerResult, UnsupportedGround, compile_ground
from .synthesis import Constraint, SynthesisResult, Unrealizable, synthesize, verify
from .observability import ObservationAt, TraceResult, synthesize_trace, verify_trace, observability_profile

__all__ = ['Instruction','Program','execute','BudgetExceeded','Ground','PowerResult',
           'UnsupportedGround','compile_ground','Constraint','SynthesisResult',
           'Unrealizable','synthesize','verify','ObservationAt','TraceResult','synthesize_trace','verify_trace','observability_profile']

# Extended research lanes: observer-specific nonlinear quotients and
# constructive law inference. Explicitly bounded; no generic acceleration claim.
from .quotient import (QuotientBudgetExceeded, ComponentCertificate,
                       QuotientTraceReceipt, byte_components, build_component_quotient,
                       synthesize_quotient_trace, power_successor)
from .quotient_verify import verify_quotient_trace
from .law_synthesis import LawCertificate, synthesize_additive_law, verify_additive_law
from .architect import Intention, Realization, realize
__all__ += ['QuotientBudgetExceeded','ComponentCertificate','QuotientTraceReceipt',
            'byte_components','build_component_quotient','synthesize_quotient_trace',
            'power_successor','verify_quotient_trace','LawCertificate',
            'synthesize_additive_law','verify_additive_law','Intention','Realization','realize']

from .wire import encode_quotient,decode_quotient,encode_law,decode_law
__all__ += ['encode_quotient','decode_quotient','encode_law','decode_law']
