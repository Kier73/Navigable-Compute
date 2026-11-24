"""Report paid coordinate planning/attempts and persistent refinement costs."""
from time import perf_counter_ns
from navigable_compute.fibre_order import choose_ordered, rename_rows
from navigable_compute.fibre_refinement import refine, observation_rows
from navigable_compute.fibre_refinement_verify import verify_refinement


def main():
    path = tuple(v for pair in zip(range(64), range(64,128)) for v in pair)
    rows = tuple(((1 << a) | (1 << b),) for a,b in zip(path,path[1:]))
    c, attempts, planning_ns = choose_ordered(128, rows, max_nodes=400, max_work=200_000)
    print("selection", dict(planning_ns=planning_ns, attempts=attempts,
          count=c.count, nodes=len(c.fibre.nodes)))
    # Observe the first branching bit in the selected source geometry.
    bit = 1 << c.order[-1]
    added = rename_rows(observation_rows(128, bit, bit), c.order)
    start = perf_counter_ns()
    r = refine(c.fibre, added)
    formation = perf_counter_ns()-start
    start = perf_counter_ns()
    assert verify_refinement(c.fibre, added, r)
    audit = perf_counter_ns()-start
    start = perf_counter_ns()
    j = r.parent_address(10**20)
    assert r.refined_address(j) == 10**20
    mapping = perf_counter_ns()-start
    assert c.unrank(j) & bit
    print("persistent refinement", dict(count=r.count, new_nodes=len(r.nodes),
          old_nodes=len(c.fibre.nodes), work=r.work, formation_ns=formation,
          audit_ns=audit, address_roundtrip_ns=mapping, old_address=j))


if __name__ == "__main__":
    main()
