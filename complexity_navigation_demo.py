"""NCR-009: costed representation experiments on disjoint source states."""
import json
import random
from time import perf_counter_ns
from nonlinear_koopman_demo import nonlinear_family
from navigable_compute.complexity_navigation import Portfolio, choose


def run():
    family = nonlinear_family()
    rng = random.Random(9009)
    training = tuple(rng.getrandbits(128) for _ in range(9))
    heldout = tuple(rng.getrandbits(128) for _ in range(12))
    assert not set(training).intersection(heldout)
    rows = []
    for word, repetitions in (((0,),0), ((0,),1), ((0,1),1),
                               ((0,1),10**100+173)):
        start = perf_counter_ns()
        portfolio = Portfolio(family, word, repetitions)
        profiles = portfolio.calibrate(training)
        exploration_ns = perf_counter_ns()-start
        cases = []
        for label, constraints in (
                ("warm_output", {}),
                ("warm_output_priced_reads", {"read_price":1000}),
                ("warm_continuation", {"continuation":True}),
                ("warm_continuation_priced_state", {"continuation":True,"bit_price":10000}),
                ("eight_bit_continuation", {"continuation":True,"max_bits":8}),
                ("cold_counterfactual", {"cold":True})):
            start = perf_counter_ns()
            selected = choose(profiles, len(heldout), **constraints)
            selection_ns = perf_counter_ns()-start
            start = perf_counter_ns()
            result = tuple(portfolio.realize(selected.candidate.mode,x)[0] for x in heldout)
            elapsed = perf_counter_ns()-start
            # Measure alternative paths separately, after the selected execution;
            # counterfactual baseline effort is not hidden in selected execution.
            reference = tuple(portfolio.realize("source",x)[0] for x in heldout)
            assert result == reference
            cases.append(dict(case=label, selected=selected.candidate.mode,
                selection_ns=selection_ns, heldout_realization_ns=elapsed,
                exploration_plus_selection_plus_realization_ns=exploration_ns+selection_ns+elapsed,
                predicted_candidate_score=selected.score(len(heldout),
                    bit_price=constraints.get("bit_price",0),
                    read_price=constraints.get("read_price",0),
                    cold=constraints.get("cold",False))))
        rows.append(dict(word=word,repetitions=str(repetitions),
            admitted_prefix=len(portfolio.prefix), source_qualification_ns=portfolio.source_qualification_ns,
            preparation_ns=portfolio.preparation_ns, exploration_wall_ns=exploration_ns,
            profiles=[dict(mode=p.candidate.mode, setup_ns=p.candidate.setup_ns,
                warm_verified_median_ns=p.warm_ns, calibration_ns=p.calibration_ns,
                producer_read_bits=p.candidate.read_mask.bit_count(),
                retained_bits=p.candidate.retained_bits) for p in profiles],
            cases=cases, failures=portfolio.failures))
    return dict(schema="ncr009.complexity-navigation.v1", rows=rows,
                training_count=len(training), heldout_count=len(heldout),
                scope="local exact Boolean fixture; priced bits hypothetical; full source available to verifier")


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
