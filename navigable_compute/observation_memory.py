"""NCR-014: observation memory independent of the next requested intention.

Persist the evidence, not a guessed target value or a host placement. A
receiver provides the admitted original runtime, epoch, context and authority.
"""
from dataclasses import dataclass
import json

from .adaptive_attention import Runtime, Session
from .attention_evidence import advance_vgi, evidence_policy
from .fibre_attention import _equal
from .intention_fibre import _input, MAX_NODES, MAX_WORK
from .fibre_refinement import refine
from .fibre_refinement_verify import verify_refinement

SCHEMA = "ncr014.observation-memory.v1"
MAX_BYTES = 2_000_000


def canonical(body):
    return json.dumps(body,sort_keys=True,separators=(",",":"),
                      ensure_ascii=True,allow_nan=False).encode("ascii")


@dataclass(frozen=True)
class Reading:
    count: int
    yes: int
    proof: object | None
    reused: bool

    @property
    def value(self):
        if self.yes == 0:
            return 0
        if self.yes == self.count:
            return 1
        return None


class Memory:
    """Receiver-local evidence history with portable canonical reconstruction.

    Admission is a local object discipline, not a sandbox for untrusted Python.
    A unique context names ONE hidden initial member at a fixed source epoch.
    """
    def __init__(self,runtime,epoch,context,allowed_sources,quorum):
        if type(runtime) is not Runtime:
            raise ValueError("admitted runtime required")
        self.runtime = runtime
        self._session = runtime.begin(epoch,context)
        from vgi.evidence import TransitionEvidencePolicy
        # Validate authority even for an already-terminal policy.
        p = TransitionEvidencePolicy("ncr014-memory-authority","0"*64,
                                     allowed_sources,quorum,True)
        self._sources, self._quorum = p.allowed_sources,p.quorum
        self._records = ()

    @property
    def session(self):
        return self._session

    @property
    def view(self):
        return self.runtime.view(self._session)

    def expected_policy(self):
        return evidence_policy(self.runtime,self._session,self._sources,self._quorum)

    def append(self,ledger):
        """Only a finalized VGI outcome for the exact next step may be appended."""
        p = self.expected_policy()
        step = advance_vgi(self.runtime,self._session,p,ledger)
        record = dict(ledger=ledger.to_dict(),context=step.context,
                      policy=step.policy_digest,decision=step.decision_digest,
                      ledger_digest=step.ledger_digest)
        # Immutable canonical bytes, so caller mutation cannot change history.
        encoded = canonical(record)
        if sum(map(len,self._records))+len(encoded)>MAX_BYTES:
            raise ValueError("memory byte budget exceeded")
        self._records += (encoded,)
        self._session = step.after
        return step

    def snapshot(self):
        s = self._session
        return canonical(dict(schema=SCHEMA,runtime=self.runtime.identity,
             epoch=s.epoch,context=s.session_id,allowed_sources=self._sources,
             quorum=self._quorum,records=[json.loads(x) for x in self._records],
             trace=s.trace))

    def read(self,target,*,max_nodes=MAX_NODES,max_work=MAX_WORK):
        """Resolve another intention against the SAME hidden member.

        Known offer equations and the original target reuse existing proofs.
        Other polynomials get a fresh independently checked intersection of
        the original parent, conditioned on the entire retained evidence.
        No external measurement or parent-coordinate renumbering occurs.
        """
        parent = self.runtime.parent
        _input(parent.width,(target,))
        v = self.view
        for name,bit in self._session.trace:
            if self.runtime.polynomial(name)==target:
                return Reading(v.outcome.count,v.outcome.count*bit,None,True)
        if target == self.runtime.target:
            return Reading(v.outcome.count,v.outcome.yes,v.outcome.target_one,True)
        rows = self.runtime.observation_equations(self._session)+(_equal(target,1),)
        proof = refine(parent,rows,max_nodes=max_nodes,max_work=max_work)
        if not verify_refinement(parent,rows,proof) or proof.count>v.outcome.count:
            raise ValueError("unqualified new intention")
        return Reading(v.outcome.count,proof.count,proof,False)


def restore(runtime,epoch,context,allowed_sources,quorum,payload):
    """Replay admitted evidence, never the physical observation operation.

    Caller pins source/context and authority independently of stored bytes.
    Canonical equality checks every derived record and chronological trace.
    """
    from vgi.evidence import TransitionEvidenceLedger
    if type(payload) is not bytes or not 0<len(payload)<=MAX_BYTES:
        raise ValueError("invalid memory payload size")
    b = json.loads(payload)
    m = Memory(runtime,epoch,context,allowed_sources,quorum)
    if (type(b) is not dict or set(b)!={"schema","runtime","epoch","context",
            "allowed_sources","quorum","records","trace"} or b["schema"]!=SCHEMA
            or b["runtime"]!=runtime.identity or b["epoch"]!=epoch or b["context"]!=context
            or b["allowed_sources"]!=list(m._sources) or type(b["quorum"]) is not int
            or b["quorum"]!=quorum or type(b["records"]) is not list
            or len(b["records"])>2):
        raise ValueError("memory identity or schema mismatch")
    for record in b["records"]:
        if type(record) is not dict or "ledger" not in record:
            raise ValueError("invalid evidence record")
        m.append(TransitionEvidenceLedger.from_dict(record["ledger"]))
    if m.snapshot()!=payload:
        raise ValueError("memory reconstruction disagrees")
    return m
