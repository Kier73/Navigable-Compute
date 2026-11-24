"""Reproduce NCR-010 cold formation, independent checking and warm navigation."""
from time import perf_counter_ns
from navigable_compute.intention_fibre import construct_fibre, construct_temporal_fibre, Goal
from navigable_compute.intention_fibre_verify import verify_fibre, verify_temporal_fibre
from nonlinear_koopman_demo import nonlinear_family


def measured(fn):
    start = perf_counter_ns()
    result = fn()
    return result, perf_counter_ns() - start


def main():
    rows = tuple((3 << i,) for i in range(127))
    c, formation = measured(lambda: construct_fibre(128, rows))
    ok, audit = measured(lambda: verify_fibre(128, rows, c))
    assert ok
    print("connected path", dict(count=c.count, residuals=len(c.nodes),
          work=c.work, formation_ns=formation, audit_ns=audit))
    x, navigation = measured(lambda: c.unrank(10**20))
    assert c.rank(x) == 10**20 and x & (x >> 1) == 0
    print("path address", dict(address=10**20, state_hex=hex(x), navigation_ns=navigation))
    f = nonlinear_family()
    goals = (Goal((0,), 1, 3, 3), Goal((0,), 10**100 + 173, 3, 2))
    t, formation = measured(lambda: construct_temporal_fibre(f, goals))
    ok, audit = measured(lambda: verify_temporal_fibre(f, goals, t))
    assert ok and t.fibre.count == 2**125
    print("native temporal fibre", dict(count=t.fibre.count, residuals=len(t.fibre.nodes),
          work=t.fibre.work, formation_ns=formation, audit_ns=audit))


if __name__ == "__main__":
    main()
