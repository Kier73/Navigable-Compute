# Navigable-Compute

**An intention-first synthetic byte processor.** Version 0.8 — November 20, 2026.

The purpose is not to emulate physical instructions more quickly. It is to construct an *exact computational Ground* from a program, navigate to a requested future coordinate without replaying every cycle, and synthesize an initial byte state capable of satisfying an observer's intended output.




## v0.8 — Construct an interpreter as a semantic invariant (NCR-002)

**The reflective interpreter is now a constructed research output**, not a
source fixture chosen in advance. A compiler begins with NCS-001's qualified
endogenous processor and a **declared** five-form language and seven-operation
semantic contract. It searches a finite set of recipe bases, selects the
minimum supported **four primitive seed operations** (`at`, `cons`, `eq`,
`rest`), constructs the other three operation implementations, derives a
Ground-specific instruction encoding and builds a same-language interpreter.
The constructor refuses unless a separate stack-based evaluator can verify
that the resulting interpreter **interprets its own encoded source** while
realizing a task word. Different architecture coordinates choose among all
120 distinct source-tag encodings (representational variations, not 120 new
computing laws).

The remaining source grammar, recipe library, finite semantic requirements and
physical host are **explicitly supplied**, not magically generated. Minimality
is within the admitted recipe library only; bounded self-application tests do
not imply a universal self-interpreter theorem. The mathematical receipts
remain separate from E-CPU/VGI host authority.

```bash
python -m unittest discover -s tests -v
python reflective_architecture_demo.py
```

See [NCR-002](docs/NCR-002.md) for constructive contracts, independent
verification, costs, controls and the next research frontier.

## v0.6 — A task can construct its own instruction algebra (NCS-001)

**New executable frontier.** Starting only from an NCI-001 invariant description, [NCS-001](docs/NCS-001.md) derives canonical processor laws, closes them under finite transformation composition, and thereby constructs a **semantic instruction vocabulary and interpreter algebra** that did not exist in the input. Each derived opcode is internally representable by a program in the generated vocabulary. Programs compose and execute within that algebra.

A 5×5 symmetry-constrained task yields **16** canonical processor-law generators and **29** distinct semantic operations. Execution order is noncommutative. Repeated execution at logical coordinate `10^100+173` uses **442** exact monoid multiplications *after algebra construction*. An independent checker validates closure, every multiplication cell, each opcode program and the bounded source-bound capsule. A forged multiplication table fails even with recomputed hashes.

This demonstrates **self-contained mathematical execution semantics**, not yet a universal self-interpreter or substrate-free CPU. The next milestone is to express the interpreter *itself* within a constructed reflective Ground.

```bash
python -m unittest discover -s tests -v
python self_space_demo.py
```

## v0.4 — Construct the possibility space, then navigate to a processor

**[NCP-001 / NCB-001: Intention-first possibility construction](docs/NCP-001.md)** begins from a **declarative requirement** rather than a supplied program or observed trajectory. A brief describes allowed effects, unsafe outcomes, intended goal states and admissible initial uncertainty. The compiler constructs a mathematical possibility relation and synthesizes a safe state-observed processor law, robust to **all** declared adversarial successor choices. Its rank certificate proves a finite worst-case convergence bound. A separate checker audits the source-to-relation translation, Bellman minimality, and a tamper-resistant-by-recomputation mathematical witness. An impossible brief can produce an independently verifiable adversarial trap rather than a fabricated realization.

**New capability: navigation among architectures.** The constructor retains a factored space of all rank-optimal controller tables. An integer coordinate selects an exact alternative processor without enumerating candidate processors. In the deterministic test, 32 independent 16-state components represent a global **2^128-state** universe, but construction examines **512 local states**, converges from every admitted initial state within **12** logical steps, and gives a unique terminal observation at time `10^100+173`. A second valid processor is constructed at instance coordinate `10^30+17`. This result relies critically on the stated independence, complete transition possibilities, full local observability and terminal goal assumption; it is not a universal computation shortcut or proof of available physical actuators.

```bash
python -m unittest discover -s tests -v
python possibility_demo.py
```

**52 local regression tests** (v0.1–v0.4), including independent enumeration of all 729 deterministic three-state/two-action tables. See [NCP-001 detailed obligations](docs/NCP-001.md) for costs, independent verification, failure witnesses, and connections to E-CPU and IFS reactive planning research. This is a constructive mathematical processor, **not yet a bound E-CPU observer-host**.

## v0.3 — The temporal coordinate changes what machine is necessary

**[NCQ-002: Task-scoped temporal Ground construction](docs/NCQ-002.md)** adds a stricter form of intention-relative computation. NCQ-001 builds the smallest machine that preserves **every future observation** in an observer contract. NCQ-002 builds the smallest exact representation preserving **only the explicitly requested coordinate/output pairs**. A machine that overwrites one byte every cycle needs 256 classes to preserve the full time-zero byte but **one task class** to answer only its post-overwrite byte; observing the high bit of an incrementing byte at one distant time needs only 2 task classes, versus 256 classes to preserve its entire future observation sequence.

The task quotient is **not transition-stable** and cannot be used to answer unrequested times. Receipts carry the full task scope; an independent checker uses finite concrete transitions and cycle decomposition. The portable wire format distinguishes `task-scoped-quotient` from `finite-observer-quotient` and rejects reinterpretation across these scopes. Call `realize(Intention(..., observation_scope='task-only'))` to select it; the default remains `all-future`.

**43 local regression tests** cover the v0.1–v0.3 research lanes. Run `python -m unittest discover -s tests -v` and `python deep_demo.py`.

## v0.2 — Construct the processor from the observation

The original exact GF(2) affine byte Ground remains available. New research lanes are:

- **NCQ-001 — Observer-minimal nonlinear Grounds.** Analyze independent read/write byte factors exhaustively, construct the coarsest future-stable Moore quotient for the *requested* byte masks, navigate very large coordinates by exact class-map squaring, and synthesize an initial state matching a multi-time observation contract. A separate checker recomputes every finite transition and verifies quotient minimality and the result via cycle decomposition. Some queries on an 8-byte/64-bit machine qualify by examining only three 256-state factors; globally coupled factors beyond the state budget are refused. [Proof and experiments](docs/NCQ-001.md).
- **NCL-001 — Construct a processor law from outputs.** When no program is supplied, synthesize a machine in a declared independent-modular-addition grammar directly from complete-byte observations at chosen times. Intersection of modular congruences recovers a canonical executable program, initial state and exact number of alternative constructions; a distinct per-byte enumerative checker validates it. [Constructed-law experiment](docs/NCL-001.md).
- **NCA-001 — Typed intention router.** `realize(Intention(...))` chooses GF(2) affine navigation, finite nonlinear quotient navigation, or exact additive-law construction. It does not invent a Ground for unqualified semantics.
- **NCW-001 — Portable mathematical receipts.** Strict canonical JSON envelopes carry source-bound quotient certificates or newly constructed law claims and support independent receiver-side mathematical verification. These are not authenticated E-CPU host receipts. [Wire contract](docs/NCW-001.md).

Run all **34** local regression tests and the deterministic deep experiment:

```bash
python -m unittest discover -s tests -v
python deep_demo.py
```

The [research frontier](docs/RESEARCH_FRONTIER.md) connects E-CPU, E-CPU Virtual Processor, VGI, IFS quadratic/metaplectic studies, and the next experiments in state-space quotienting, symbolic coupling, automatic law synthesis and distributed qualification. The claimed RH proof remains a research premise, **not a theorem used by the executable code**.


## NCR-001 — A reflective Ground interpreting its own interpreter (v0.7)

**The processor's task-derived algebra now has a same-language interpreter
expressible as ordinary data in its reflective computational space.** Given
invariants, NCI-001 derives valid laws, NCS-001 closes them into an endogenous
29-operation semantic processor (5×5 fixture), and NCR-001 constructs a small
program `E` that interprets programs of its own expression language. The
experiment evaluates a task program directly, through `E`, and by asking `E`
to interpret **its own source** while it interprets the task. A separately
implemented, stack-based evaluator rechecks the exact outputs and work.

This is finite, bounded same-language reflection, **not** a general or
substrate-free self-hosting claim. The tuple/conditional/call language and
seven primitive operations remain an explicitly supplied reflection seed; the
task's instruction semantics are generated from the invariant Ground. The
source is not executed as arbitrary Python and there is no magic `eval` opcode.
Portable receipts are audited against the independently verified source Ground.

```bash
python -m unittest discover -s tests -v
python reflective_demo.py
```

Read [NCR-001](docs/NCR-001.md) for the commuting semantic identity,
independent continuation-machine checker, measured reflection costs, and the
remaining path toward generating the reflective language itself.

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

## NCI-001 — Construct transformation spaces directly from invariants (v0.5)

**New direction: the transition relation itself is an output.** NCP-001 previously
required a declaration of possible action effects. NCI-001 requires only a
finite *mathematical geometry and its invariants*: safe/goal states, a spatial
metric and locality bound, a symmetry, preserved observable labels and allowed
initial states. It derives the full space of **rank-optimal, symmetry-equivariant
partial processor laws** consistent with those invariants, proves goal
convergence, and navigates directly to a selected law by integer coordinate.

A 7×7 geometric fixture has **512** admissible shortest-progress laws, without
providing a program, effect clauses or a ranking. An independent 32-factor
construction represents **49^32** possible global states and **2^288** such
processor laws while examining only **1,568** local states (declared
independence is essential). A rotation-symmetry obstruction demonstrates that
nearby geometric states need not admit a symmetry-compatible transition;
independently checkable closed-trap certificates refuse the impossible brief.

The implementation is finite and exact, and is not a claim of universal
optimization, externally witnessed behavior, or physical host availability.

```bash
python -m unittest discover -s tests -v
python invariant_demo.py
```

Read [the construction and proof contract](docs/NCI-001.md). This lane has
its own independent semantic checker and strict mathematical-receipt wire
profile, and leaves authority and real-world effect execution with E-CPU/VGI.
