# Architecture — observable state before execution

**2026-11-20 — architectural hypothesis and executable v0.1.**

## What the researcher asks for

A computation is treated as a relation among a byte-level program, an initial state, a temporal coordinate, and an observer's requested output. The requested observation need not include the full internal state.

For a deterministic transition function `T` on `n` input bits, and a Boolean projection `C`, the goal is to make two queries constructive:

- **Forward navigation:** `y = C T^N(x)` without forcing `T` to execute N times.
- **Reverse intention:** find `x` such that `C T^N(x) = y` for specified bits of `y`.

With an affine transition `T(x) = Ax ⊕ b` over GF(2), composition is closed: the pair `(A,b)` can be squared and multiplied exactly. The machine Ground is not a neural guess or a result from RH. It is constructed symbolically from byte instructions.

The processor interprets `x` as **arbitrary bytes**, not an application-specific number. Specific instructions define transformations of bit-level byte state. `C` can select any memory byte or any mask of a byte. A full result is an optional materialization, not the core addressable object.

## Why the Stepanov / theta-lift news is useful but not a theorem here

The November 20, 2026 claimed RH/GRH proof proposes iterated theta lifts as a replacement for multiplicity in a Stepanov-style argument. The research idea that survives without importing any unlicensed source code or unverified theorem is that a **structured chain of representations can reveal otherwise unavailable constraints**.

The first exact CPU analogue is the observability matrix. For a linear machine `T(x)=Ax`, repeated observations produce:

    O_k = [ C ; C A ; C A^2 ; ... ; C A^(k-1) ]

For an affine machine, the corresponding offsets are known and can be subtracted from each observed result. The rank of `O_k` tells us how many independent facts the observer's measurements constrain about the initial state. This is established classical linear-system observability, *not* the automorphic theta lift or a new consequence of RH.

**Multiplicities can be useful, but not automatically.** Eight one-bit observations of a rotating byte at eight time coordinates independently determine the eight initial bits. Their rank sequence is `1,2,3,4,5,6,7,8`. Conversely, repeatedly observing a byte which the program sets to zero yields rank `0` for every later observation, no matter how many lifts or measurements are attempted. The chain cannot create missing information by itself.

This is a useful design gate: an intelligent navigator should request the **next coordinate that increases observation rank**, not just perform another lift. That is an active-attention problem and can connect to VGI observation selection.

## Implementation

- `machine.py` — independent imperative byte interpreter with an operation budget (not a shortcut).
- `ground.py` — exact symbolic GF(2) affine compiler; composition; far-coordinate navigation; selected-byte rendering.
- `synthesis.py` — exact reverse construction from masked output constraints, plus optional fixed input conditions; deterministic receipts.
- `observability.py` — cross-coordinate constraint accumulation, rank profiling, trace synthesis, deterministic recheck.
- `tests/test_machine.py` — exhaustive single-byte oracle checks, randomized multibyte conformance, distant-state composition, contradictory goals, negative controls, and rank experiments.

## Important distinction: cost and representation size

A matrix on `n` bits may have `n²` bits and an affine offset of `n` bits; this is not a universal compression of program state. The individual Ground product costs more than an instruction and may involve up to quadratic bitset work. Repeated squaring uses `O(log N)` products, making it beneficial for sufficiently large `N` and structured `T`, but no generic algorithm can promise this for arbitrary byte-programs.

Some programs have exact algebraic regularity; others require richer switching, decision diagrams, constraints, or ordinary execution. A good CPU should pick among those lanes by **measured cost and verified semantic contract**, not by assuming everything admits a small spectrum.

## Proposed functional interface beyond v0.1

    Request = (description, observer, output_contract, coordinate)
    Plan = qualify_ground(Request) | refuse_and_execute(Request)
    Result = render(Plan, requested_bytes)
    Certificate = verify(description, Plan, Result, evidence)

An observer may specify a partial output in terms of bytes, higher-level decoded objects, physical observables, or cryptographic evidence; the kernel should preserve the typed relation between the abstract request and the exact byte observations. Avoid inventing a law from a tiny sample and treating it as universally valid.

## Tests that could disprove an overly broad claim

1. Nonlinear add-with-carry: `add_const` must refuse the GF(2) Ground, even though the reference interpreter handles it.
2. Non-injective transitions: `set`/`copy` destroy input information. Multiple observations may be redundant.
3. Inconsistent intent: if two observations logically conflict, no starting state exists; solver refuses.
4. Large output: rendering 10^9 bytes still requires significant work even if the coordinate jump is cheap.
5. External physical systems: a synthetic law with fitting observations is not proof of the physical system's behavior beyond its qualified horizon.

## Research frontier

- Generalize Ground families to semirings, modular arithmetic with carry, finite-state transducers, reversible gates, and sparse Clifford/metaplectic transformations.
- Make the Ground synthesizer search *program structures* satisfying an intended observation, not only initial states.
- Learn exact linear/affine candidates from finite traces, then qualify them against independent witness sets; explicitly separate empirical fit from universal semantic proof.
- Create an observer-aware planner that chooses new observations by independent-rank gain per unit measured resource cost.
- Produce a portable Ground receipt whose semantics VGI can independently reconstruct and E-CPU can transport across authenticated observer-hosts.
- Measure total cost (Ground admission, memory, requested output, certificate verification, host communication) against ordinary sequential execution for both favorable and adversarial programs.

