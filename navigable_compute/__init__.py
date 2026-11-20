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

from .task_quotient import TaskComponentCertificate,TaskTraceReceipt,synthesize_task_trace
from .task_verify import verify_task_trace
from .wire import encode_task,decode_task
__all__ += ['TaskComponentCertificate','TaskTraceReceipt','synthesize_task_trace',
            'verify_task_trace','encode_task','decode_task']


# Constructive possibility-space research (NCP/NCB-001).
from .possibility import (PossibilityAction,FactorContract,ConstructedFactor,
    ConstructedGround,ObstructionCertificate,NotRealizable,construct_world,
    navigate_instance,certify_nonrealizable)
from .possibility_verify import verify_world,verify_obstruction
from .brief import EffectCase,ActionSketch,GroundBrief,compile_brief,construct_from_briefs
from .brief_verify import verify_brief_world
__all__ += ['PossibilityAction','FactorContract','ConstructedFactor','ConstructedGround',
            'ObstructionCertificate','NotRealizable','construct_world',
            'navigate_instance','certify_nonrealizable','verify_world',
            'verify_obstruction','EffectCase','ActionSketch','GroundBrief',
            'compile_brief','construct_from_briefs','verify_brief_world']

# NCI-001: derive an entire admissible transformation family from invariants.
from .invariant_space import (InvariantDescription, InvariantTransformationSpace,
    InvariantObstruction, InvariantProduct, NoInvariantRealization,
    construct_invariant_space, navigate_invariant_instance,
    certify_invariant_obstruction, construct_invariant_product)
from .invariant_verify import (verify_invariant_space, verify_invariant_obstruction,
                               verify_invariant_product)
from .invariant_wire import (encode_invariant, decode_invariant,
                             encode_invariant_obstruction, decode_invariant_obstruction)
__all__ += ['InvariantDescription','InvariantTransformationSpace',
            'InvariantObstruction','InvariantProduct','NoInvariantRealization',
            'construct_invariant_space','navigate_invariant_instance',
            'certify_invariant_obstruction','construct_invariant_product',
            'verify_invariant_space','verify_invariant_obstruction','verify_invariant_product',
            'encode_invariant','decode_invariant',
            'encode_invariant_obstruction','decode_invariant_obstruction']

# NCS-001: task-generated finite operator vocabulary and executable algebra.
from .self_space import EndogenousProcessor, AlgebraBudgetExceeded, build_endogenous_processor
from .self_verify import verify_endogenous_processor
from .self_wire import encode_processor, decode_processor
__all__ += ['EndogenousProcessor','AlgebraBudgetExceeded','build_endogenous_processor',
            'verify_endogenous_processor','encode_processor','decode_processor']
