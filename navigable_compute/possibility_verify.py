"""Independent NCP-001 checker: certificate ranking and adversarial trap audit.

Does not call the constructive AND/OR attractor algorithm. It checks a local
Bellman optimality/minimality certificate, plus a closed losing-state trap.
Every check uses the original declared contract, not a self-authored digest.
"""
from __future__ import annotations

from hashlib import sha256
from math import prod
from .possibility import ConstructedGround,FactorContract,_canonical


def verify_world(contracts: tuple[FactorContract,...],ground:ConstructedGround)->bool:
    try:
        if not isinstance(contracts,tuple) or not contracts or not isinstance(ground,ConstructedGround):
            return False
        if len(contracts)!=len(ground.factors) or ground.contracts!=contracts:
            return False
        if (type(ground.instance_coordinate) is not int or
            not 0<=ground.instance_coordinate<ground.factored_optimal_laws or
            ground.instance_coordinate.bit_length()>4096):
            return False
        remainder=ground.instance_coordinate
        for spec,receipt in zip(contracts,ground.factors):
            n=spec.states
            if len(receipt.rank)!=n or len(receipt.action)!=n or len(receipt.optimal)!=n:
                return False
            if any(type(v) is not int or not -1<=v<=n for v in receipt.rank):
                return False
            safe=set(range(n))-set(spec.unsafe)
            if any((receipt.rank[s]==0)!=(s in spec.goal) for s in range(n)):
                return False
            if any(receipt.rank[s]!=-1 for s in spec.unsafe):
                return False
            if any(receipt.rank[s]<0 for s in spec.initial):
                return False
            law_count=1
            for state in range(n):
                depth=receipt.rank[state]
                if depth<=0:
                    if receipt.action[state]!=-1 or receipt.optimal[state]:
                        return False
                    if depth<0 and state in safe:
                        # No admissible action can escape the losing trap.
                        for a in spec.actions:
                            ss=a.outcomes[state]
                            if ss and set(ss)<=safe and all(receipt.rank[t]>=0 for t in ss):
                                return False
                    continue
                achievable=[]
                lower_depth=depth-1
                for i,a in enumerate(spec.actions):
                    ss=a.outcomes[state]
                    if not ss or any(t not in safe for t in ss):
                        continue
                    if all(receipt.rank[t]>=0 for t in ss):
                        max_rank=max(receipt.rank[t] for t in ss)
                        if max_rank<lower_depth: # A faster law exists: claimed rank not minimal.
                            return False
                        if max_rank==lower_depth:
                            achievable.append(i)
                    # Otherwise the action is not a winning candidate.
                if not achievable or tuple(achievable)!=receipt.optimal[state]:
                    return False
                expected=achievable[remainder%len(achievable)]
                remainder//=len(achievable)
                if receipt.action[state]!=expected:
                    return False
                law_count*=len(achievable)
            if receipt.optimal_laws!=law_count:
                return False
            if receipt.worst_steps!=max(receipt.rank[s] for s in spec.initial):
                return False
        if remainder!=0:return False
        if ground.global_states!=prod(c.states for c in contracts):return False
        if ground.initial_states!=prod(len(c.initial) for c in contracts):return False
        if ground.factored_optimal_laws!=prod(x.optimal_laws for x in ground.factors):return False
        if ground.convergence_bound!=max(x.worst_steps for x in ground.factors):return False
        return sha256(_canonical(ground.encode())).hexdigest()==ground.digest
    except (ValueError,IndexError,TypeError,AttributeError,KeyError):
        return False


def verify_obstruction(contracts, obstruction)->bool:
    """Prove a losing trap using only the supplied possibility relation.

    Unlike construction, verification needs no attractor iteration. A trap
    containing an admitted origin, disjoint from goals, where no action has
    *all* its outcomes outside the trap and inside the safe region, proves
    there can be no universally winning stationary or history-based policy.
    """
    from .possibility import ObstructionCertificate
    try:
        if (not isinstance(obstruction,ObstructionCertificate) or
            not isinstance(contracts,tuple) or
            not 0<=obstruction.factor_index<len(contracts)):
            return False
        spec=contracts[obstruction.factor_index]
        trap=obstruction.trap
        if (not trap or tuple(sorted(set(trap)))!=trap or
            any(type(s) is not int or not 0<=s<spec.states for s in trap)):
            return False
        bad=set(trap)
        if bad & set(spec.goal) or bad & set(spec.unsafe):
            return False
        if (obstruction.trapped_initial!=tuple(s for s in spec.initial if s in bad) or
            not obstruction.trapped_initial):
            return False
        for state in trap:
            for action in spec.actions:
                outcomes=action.outcomes[state]
                # A safe, enabled action whose *all* outcomes escape the trap
                # destroys this alleged impossibility proof.
                if outcomes and all(t not in spec.unsafe and t not in bad for t in outcomes):
                    return False
        if obstruction.source_sha256!=sha256(_canonical(spec.encode())).hexdigest():
            return False
        return obstruction.witness_sha256==sha256(_canonical(obstruction.encode())).hexdigest()
    except (ValueError,TypeError,IndexError,AttributeError):
        return False
