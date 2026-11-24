"""Measure construction, independent qualification and warm observer encoding."""
import json
from statistics import median
from time import perf_counter_ns

from nonlinear_koopman_demo import nonlinear_family
from navigable_compute.nonlinear_koopman import (
    construct_koopman_ground, eval_observers, navigate_nonlinear)
from navigable_compute.nonlinear_koopman_verify import (
    verify_polynomial_laws, verify_koopman_ground, verify_nonlinear_result)
from navigable_compute.realized_image import construct_image, verify_image, pack, unpack


def timed(call, trials=3):
    """Return final value and median measured nanoseconds."""
    spans = []
    for _ in range(trials):
        start = perf_counter_ns()
        result = call()
        spans.append(perf_counter_ns()-start)
    return result, median(spans)


def row(basis, width, *, explicit):
    """Compare exact image construction against optional full enumeration."""
    cert, build = timed(lambda: construct_image(basis, width))
    ok, check = timed(lambda: verify_image(basis, width, cert))
    assert ok
    x = ((1 << width)-1) // 3
    z = eval_observers(basis, x)
    restored, warm = timed(lambda: unpack(cert, pack(cert, z)), 21)
    assert restored == z
    baseline = None
    if explicit:
        image, baseline = timed(lambda: {eval_observers(basis, v) for v in range(1 << width)})
        assert len(image) == cert.cardinality
    return dict(width=width, feature_rank_or_count=len(basis),
        cardinality=str(cert.cardinality), packed_bits=cert.bits,
        factors=len(cert.factors), charged_local_cells=cert.cells,
        full_source_states=str(1 << width), build_ns=build,
        independent_check_ns=check, warm_roundtrip_ns=warm,
        full_enumeration_ns=baseline)


def run():
    """Run analytical scaling fixture and original NCR-007 native source."""
    rows = []
    for lanes in (2, 4, 6, 8, 16, 32, 64):
        basis = tuple(p for i in range(lanes) for p in
                      ((1 << (2*i),), (1 << (2*i+1),), (3 << (2*i),)))
        rows.append(row(basis, 2*lanes, explicit=lanes<=8))
    family = nonlinear_family()
    assert verify_polynomial_laws(family)
    ground, compile_ns = timed(lambda: construct_koopman_ground(family))
    ok, ground_check_ns = timed(lambda: verify_koopman_ground(family, ground))
    assert ok
    native = row(ground.basis, family.description.width, explicit=False)
    cert = construct_image(ground.basis, family.description.width)
    receipt = navigate_nonlinear(family, ground, 0xDEADBEEF, (0,1), 10**100+173)
    assert verify_nonlinear_result(family, ground, receipt)
    z = int(receipt.result_features_decimal)
    assert ground.read(unpack(cert, pack(cert, z))) == int(receipt.observed_decimal)
    native.update(ground_compile_ns=compile_ns, ground_check_ns=ground_check_ns,
                  distant_coordinate=receipt.repetitions_decimal,
                  distant_observation=receipt.observed_decimal)
    return {"schema":"ncr008.realized-image.v1", "independent_lanes":rows,
            "native_ncr007":native,
            "scope":"synthetic ANF image certification; no universal speedup claim"}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
