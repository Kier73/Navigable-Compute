"""NCR-005: source invariants -> quotient language -> exact native algebra."""
from __future__ import annotations
import json
from invariant_demo import grid
from preclosure_demo import cyclic_system
from navigable_compute.preclosure_language import (generators_from_invariants,
                                                     construct_preclosure_language)
from navigable_compute.observer_native import construct_native_algebra, navigate_native
from navigable_compute.observer_native_verify import (verify_native_algebra,
                                                      verify_native_result)


def run() -> dict:
    results = []
    huge = 10**100 + 173
    for label, source in (
        ('NCI-001-grid-5-goal', generators_from_invariants(
            grid(5), tuple('goal' if s in grid(5).goal else 'not-goal'
                           for s in range(25)))),
        ('NCR-004-hidden-S8', cyclic_system(symmetry_group=True, output_mode='visible')),
    ):
        language = construct_preclosure_language(source)
        algebra = construct_native_algebra(source, language)
        assert verify_native_algebra(source, language, algebra)
        initial = source.initial[0]
        word = (0,)
        observed = navigate_native(source, language, algebra, initial, word, huge)
        assert verify_native_result(source, language, algebra, observed)
        results.append({'source': label, 'states': len(source.states),
                        'quotient_states': len(language.state_classes),
                        'source_generator_count': len(source.generators),
                        'observer_language_operations': len(language.operators),
                        'coordinate_algebra': algebra.kind,
                        'coordinate_dimension': algebra.dimension,
                        'algebra_sha256': algebra.digest,
                        'coordinate': observed.repetitions_decimal,
                        'output': observed.output,
                        'operator_products': observed.operator_products,
                        'source_steps_replayed': 0})
    return {'schema': 'navigable-compute.ncr005-evidence.v0.1',
            'results': results,
            'scope': 'exact finite model only; no external observation or physical authority'}


if __name__ == '__main__':
    print(json.dumps(run(), sort_keys=True, separators=(',', ':')))
