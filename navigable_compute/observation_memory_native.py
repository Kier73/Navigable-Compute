"""NCR-014: seal/recover observation memory via native E-CPU recovery lineage.

Uses real AtomicGroundEvidenceStore and RecoveryStateManifest, not duplicate
quorum logic. Receiver passes independently pinned context and membership.
"""
from .observation_memory import Memory, restore

CUT = "ncr014.observation-memory"


def _native():
    from mailcpu.federated_ground_continuity import AtomicGroundEvidenceStore
    from mailcpu.recovery_lineage import RecoveryLineageCoordinator, RecoveryStateManifest
    return AtomicGroundEvidenceStore,RecoveryLineageCoordinator,RecoveryStateManifest


def seal(memory,coordinator,storage,*,previous=None):
    Store,Coordinator,Manifest = _native()
    if type(memory) is not Memory or not isinstance(coordinator,Coordinator) or not isinstance(storage,Store):
        raise ValueError("native memory, coordinator and storage required")
    payload = memory.snapshot()
    # Audit the serialized evidence before granting it a durable root.
    checked = restore(memory.runtime,memory.session.epoch,memory.session.session_id,
                      memory._sources,memory._quorum,payload)
    members = coordinator.membership
    if previous is None:
        generation,predecessor=1,members.genesis_state_digest
        if len(checked.session.trace)!=1:
            raise ValueError("first cut must contain exactly one admitted observation")
    else:
        old = recover(memory.runtime,memory.session.epoch,memory.session.session_id,
                      memory._sources,memory._quorum,coordinator,storage,previous)
        if (len(checked._records)!=len(old._records)+1
                or checked._records[:-1]!=old._records):
            raise ValueError("new cut must extend the exact committed evidence by one step")
        generation,predecessor=previous.generation+1,previous.digest
    root=storage.put(payload)
    manifest=Manifest(machine_id=members.system_id,incarnation=members.incarnation,
        owner_id=members.owner_id,owner_term=members.owner_term,
        membership_digest=members.digest,generation=generation,
        predecessor_digest=predecessor,state_digest=root,capture_cut=CUT)
    manifest.verify_payload(storage.get(root))
    proof=coordinator.seal(manifest)
    if coordinator.verify_state(manifest).snapshot()!=proof.snapshot():
        raise ValueError("quorum changed accepted root")
    return manifest,proof


def recover(runtime,epoch,context,allowed_sources,quorum,coordinator,storage,manifest):
    """Read live high-water authority, content bytes and replay evidence only."""
    Store,Coordinator,Manifest = _native()
    if (not isinstance(coordinator,Coordinator) or not isinstance(storage,Store)
            or type(manifest) is not Manifest or manifest.capture_cut!=CUT):
        raise ValueError("native recovery context required")
    coordinator.verify_state(manifest)
    payload=storage.get(manifest.state_digest)
    manifest.verify_payload(payload)
    m=restore(runtime,epoch,context,allowed_sources,quorum,payload)
    if len(m.session.trace)!=manifest.generation:
        raise ValueError("observation generation mismatch")
    return m
