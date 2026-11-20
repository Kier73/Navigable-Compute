# Navigable-Compute

**An intention-first synthetic byte processor.** Version 0.1 — November 20, 2026.

The purpose is not to emulate physical instructions more quickly. It is to construct an *exact computational Ground* from a program, navigate to a requested future coordinate without replaying every cycle, and synthesize an initial byte state capable of satisfying an observer's intended output.

## First running experiment

Given a deterministic byte-machine program `P`, a remote cycle coordinate `N`, and a desired masked observation of memory at that coordinate, construct a specific initial memory `x` such that:

`observe(P^N(x)) == intended_observation`.

This is both a forward state navigator and a reverse constructor. An exact GF(2) affine Ground is compiled **symbolically from the byte instructions**. Repeated composition uses exponentiation by squaring, so its composition count grows logarithmically in `N`. The inverse query is solved by exact bitset Gaussian elimination, without enumerating candidate initial states. The observed bytes are rendered directly without materializing the full destination memory. Every complete result carries a deterministic SHA-256 evidence digest. Small-coordinate answers can be compared with a separately implemented sequential byte interpreter; long-coordinate answers are rechecked algebraically, not independently replayed.

No third-party Python packages are needed.

```bash
python -m unittest discover -s tests -v
python demo.py
```

The demo requests a partial four-byte state at cycle `10^12+123`, synthesizes a suitable starting state and reports the compact Ground composition count. It deliberately does not pretend to replay those trillion cycles.

## Scope, interpretation, and safety

- Supported exact Ground opcodes: `xor`, `xor_const`, `rotl`, `swap`, `copy`, `and_const`, `set`. They are byte-addressed and well-defined on arbitrary byte data. The reference interpreter also implements `add_const` (modulo 256), for which Ground admission **refuses** because carrying makes it non-GF(2)-affine. More nonaffine opcodes may be added with different qualified Ground families.
- `compile_ground()` proves its own algebraic translation only to the extent tested; it is not a Lean kernel proof or independent proof of semantic equivalence. Small-cycle exhaustive/randomized conformance compares it with the separately written interpreter.
- SHA-256 receipts bind the exact program, input, requested coordinates, goal and output. They are *not* signatures, external observations or attestations. `verify` reconstructs the mathematical claim from the program.
- A valid output is an exact mathematical result for the declared machine, not proof that an outside physical process behaved the same way.
- Ground construction and reverse constraint solving have costs in the number of state bits, and this method is beneficial only for a suitable structural family. `O(log N)` Ground products does **not** imply `O(log N)` total work, nor universal speedup. Materializing all bytes may itself dominate.
- Incompatible target constraints are rejected rather than hallucinating a machine state. Ambiguous input bits are explicitly counted.
- The observer requests bytes/masks; this is not yet a complete instruction set, ISA, OS, I/O scheduler, network-host protocol or hardware substitute.

## Why this relates to the November 20 RH/GRH claim

**Research premise, not dependency:** Levent Alpöge reported (with Claude) an RH/GRH argument using a Stepanov-style multiplicity surrogate through iterated theta lifts (roughly `sqrt(X)` lifts for an arithmetic sum of length `X`). This project *assumes the announcement true for exploration* but imports no theorem/code from the release and makes no inference from RH to accelerated arbitrary computing. Its first architectural analogy is **compose representations until a difficult direct observation becomes an efficiently addressable constraint**. The actual GF(2) affine transformations here are standard, established algebra and **are not automorphic theta lifts**.

Public sources:
- https://alpo.ge/remark.pdf (author's note)
- https://github.com/CocoSgt/riemann-hypothesis-lean (unmodified public mirror; original archive has no stated license, so **no code copied**)

Nearby prior art, not copied or imported:
- https://github.com/Kier73/In-Formation-Space/tree/main/experiments/spectral-ground-navigation
- https://github.com/Kier73/In-Formation-Space/tree/main/experiments/theta_cayley_navigation
- https://github.com/Kier73/In-Formation-Space/tree/main/experiments/quadratic_transfer_grounds
- https://github.com/Kier73/In-Formation-Space/tree/main/bridges/observation-equivalence
- VGI supplies formal Ground/witness vocabulary; E-CPU supplies authenticated, distributed observer-host execution; VL-64 and VL-I supply compact/general-purpose interpreter research.

## Observability chain — a computational analogue of multiplicity

`observability.py` adds *observations at multiple time coordinates* as independent equations about one starting state. It measures the exact constraint rank after each observation, refuses contradictory observations, and synthesizes an initial state satisfying the whole trace. Observing one bit of a rotating byte at eight different coordinates recovers all eight input bits, with rank increasing from 1 to 8. Observing a byte that is forcibly reset to zero gives rank 0 regardless of how often it is observed. This is **classical linear observability**, not Alpöge's theta lift; it illustrates why the mathematical quality of additional observations matters more than their count.

See [the architecture note](docs/architecture.md) and [reproduction record](docs/reproducibility.md).

## Current architecture

1. **Description:** typed finite byte-program and desired partial observation.
2. **Ground:** compile byte instructions to exactly defined GF(2) affine relations. Unsupported semantics fail closed.
3. **Lift:** compose those relations by binary exponentiation to address distant cycle counts.
4. **Reverse construction:** solve the intended output constraints for an initial state using exact field arithmetic.
5. **Rendering:** reconstruct only requested output bytes; optional full-state evaluation.
6. **Temporal observability:** combine constraints from many coordinates, measure rank, and recover otherwise hidden byte state.
7. **Evidence:** produce a deterministic program-bound digest, algebraic witness and small-cycle oracle conformance.

## Frontier

1. Extend to modular addition, carry, branching and memory access through **typed switching Grounds**, benchmarking against sequential emulation and symbolic execution.
2. Make partial observation *structurally cheaper* than full-map compilation, by backward slicing a requested output under composed transformations.
3. Add source-supported proof certificates with an independent checker, optionally Lean; distinguish logical equivalence from finite testing.
4. Integrate finite-field metaplectic/Weil transfers as an **additional admissible Ground family**, with direct output rendering from basis/sparse states and empirical rank-growth guardrails.
5. Establish a generic byte-observation protocol and connect authoritative receipts to VGI/E-CPU transport; preserve observer provenance and host identity.
6. Synthesize the **program itself** from intentional constraints, not just its initial state; compare with SMT / superoptimization / program synthesis baselines.
7. Investigate whether composition depth and independent observable constraints can play a computational role analogous to *multiplicity* in the Stepanov/theta-lift idea without claiming they are the same mathematical construction.

## Honest speed comparison

A CPU still requires physical substrate to evaluate the Ground. The claim is **fewer necessary operations than direct emulation for certain programs**, not that virtual computation transcends hardware, all tasks can be fast-forwarded, or RH implies P=NP. For an arbitrary program, `add_const` demonstrates an immediate family-admission refusal; for arbitrary nonlinear programs, the representation may grow as expensive as iteration.