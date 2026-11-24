"""Optional real VGI finalization adapter for one adaptive observation step.

Receiver owns the expected policy and ledger provenance. Local hashes are
context identities, not signatures or proof of a physical sensor reading.
"""
from dataclasses import dataclass

from .adaptive_attention import Runtime, Session

GROUND_ID = "ncr013-adaptive-attention"
STATE = "observe"


@dataclass(frozen=True)
class EvidenceStep:
    before: Session
    after: Session
    context: str
    ledger_digest: str
    policy_digest: str
    decision_digest: str


def evidence_policy(runtime,session,allowed_sources,quorum):
    from vgi.evidence import TransitionEvidencePolicy
    return TransitionEvidencePolicy(GROUND_ID,runtime.challenge(session),
                                    allowed_sources,quorum,True)


def advance_vgi(runtime,session,expected_policy,ledger):
    """Read only finalized authorized observations under receiver policy."""
    from vgi.evidence import TransitionEvidencePolicy, TransitionEvidenceLedger
    if type(runtime) is not Runtime or type(expected_policy) is not TransitionEvidencePolicy:
        raise ValueError("receiver policy required")
    context = runtime.challenge(session)
    if (expected_policy.ground_id != GROUND_ID or expected_policy.context_digest != context
            or expected_policy.strict is not True or type(ledger) is not TransitionEvidenceLedger
            or ledger.policy != expected_policy):
        raise ValueError("observation context or authority mismatch")
    checked = TransitionEvidenceLedger.from_dict(ledger.to_dict())
    name = runtime.view(session).decision.next_offer
    decision = checked.decision(STATE,name)
    if decision.status != "AUTHORIZED":
        raise ValueError("observation not finalized AUTHORIZED")
    transition = checked.transition_observation(STATE,name)
    if len(transition.outcomes)!=1:
        raise ValueError("expected one bit outcome")
    outcome = transition.outcomes[0]
    if (outcome.target != STATE or len(outcome.output)!=1
            or type(outcome.output[0]) is not int or outcome.output[0] not in (0,1)):
        raise ValueError("invalid observation outcome")
    after = runtime.advance(session,name,outcome.output[0])
    return EvidenceStep(session,after,context,checked.fingerprint,
                        expected_policy.fingerprint,decision.digest)


def verify_evidence_step(runtime,session,expected_policy,ledger,receipt):
    """Reconstruct the VGI finalization and exact resulting session; no sensing."""
    try:
        return type(receipt) is EvidenceStep and receipt == advance_vgi(
            runtime,session,expected_policy,ledger)
    except (ValueError,TypeError,AttributeError,KeyError,IndexError):
        return False
