# Temporal Ground network — proof-qualified relationships as computational resources

Consolidated native utility from IFS CIG-033A/034A. It has no IFS, VGI, E-CPU or third-party runtime dependency. The identical core is also exercised through the IFS source-qualified Seed adapter. Existing NCR-007 nonlinear observer Grounds and NCR-008 realized-image encodings can supply future adapters; this increment does not change their contracts or claim that every NCR trajectory admits a compact CRT representation.

## Mathematical contract

`View(source, base, nodes, root, cutoff, prefix)` is an immutable, validated Boolean temporal language on nonnegative integer time. A finite prefix precedes a periodic function described by an ordered reduced multi-valued decision diagram (MDD) on pairwise coprime prime-power coordinates. For radix b, the selected arc is t mod b. The product of the radices is a common period, not necessarily a minimal one. Terminals 0 and 1 are fixed. The view key binds the source namespace, basis, graph, root, cutoff and prefix. A namespace is not external proof of source behavior; source-specific adapters must establish that before registering a view. Logical integers are at most 4096 bits, radices at most 4096, cutoff at most 4096, at most 256 coordinates and 2,000,000 diagram arcs.

A directed `Edge(source,target,shift,start,source_before,target_before,cost)` certifies

```
target(t) = source(t + shift)           for t >= start
start >= 0 and start + shift >= 0
source_before = R_source(start + shift)
target_before = R_target(start)
R_target(t) = R_source(t + shift) - offset   for t >= start
offset = source_before - target_before
```

Rank is exclusive, `R(t) = sum_{0 <= x < t} I(x)`. A target ordinal k is eligible only for `k > target_before`. Its translated ordinal is `k + offset`; a qualified source destination s gives target destination `s - shift`. The signed shift supports delay as well as advance. Initial observations outside the suffix are not silently identified. No uniform state-map or physical continuation authority follows from this trajectory-language contract.

`discover` tries a declared finite palette of (shift,start) candidates using a synchronized CRT product traversal. A finite unmatched-prefix region is checked explicitly; tail equality is checked over the common prime-power domain without enumerating the full common period. Exact offset ranks use full-period arithmetic and a budgeted CRT cover. `verify_edge` independently checks the relation by recursive prime-coordinate traversal and recomputes the counts with a separate breadth-first CRT algorithm. Failure, including budget exhaustion, does not prove that no mathematical relationship exists.

## Composition and retained shortcuts

Paths run from a requested target toward reusable source knowledge. Let edge i have shift d_i, cutoff s_i and offset h_i. Define cumulative values before edge i as D_i=sum_{j<i}d_j and H_i=sum_{j<i}h_j. Then

```
D = sum_i d_i
H = sum_i h_i
S = max_i(s_i - D_i)
a = any i attaining that maximum
R_target(S) = target_before_a - H_a
R_source(S+D) = R_target(S)+H
```

The new edge is valid on t >= S. The cutoff's prefix rank is already available in the active edge, so composition requires no new rank computation or diagram traversal. `compose` proposes this summary; `verify_composite` separately replays pairwise rank and cutoff composition against previously admitted edges. `Network.admit_composite` records the constituent edge IDs as provenance. The implementation is O(path length) in arithmetic operations plus registry access and big-integer arithmetic, not independent of integer size. A local registry stores immutable objects and does not implement durable external admission or authentication.

## Native usage

```python
from navigable_compute.ground_network import View, Network, discover, compose, select
from navigable_compute.ground_network_verify import verify_realization

# Alternating event streams, with a larger common period on B.
a = View('A', (2,), ((0, (1, 0)),), 2)
b = View('B', (4,), ((0, (0, 1, 0, 1)),), 2)
c = View('C', (2,), ((0, (1, 0)),), 2, 1, 0)
g = Network()
for view in (a,b,c):
    g.register(view)
ab,_ = discover(a,b,[(1,0)],cost=3)
bc,_ = discover(b,c,[(1,1)],cost=4)
assert g.admit(ab) and g.admit(bc)
shortcut = compose(g,(bc.key,ab.key),cost=1)
assert g.admit_composite(shortcut,(bc.key,ab.key))
k = 10**900+17
route,_ = g.plan(c.key,k,lambda node,q: 0 if node==a.key else None)
source = select(a,route.source_ordinal)
assert source['status']=='FOUND'
t = verify_realization(g,c.key,k,route,source['time'])
assert t == 2*k
```

`Network.plan` searches states (node, translated ordinal, accumulated displacement, hop count), using nonnegative edge and stable endpoint cost estimates. If a route is returned before the declared search budgets are exhausted, it minimizes the offered cost among bounded walks. Cycles may translate an ordinal and are not automatically pruned as identical nodes. `None` is UNKNOWN (no offered route within the declared search), not proof of unreachability. Costs are hypothetical caller-supplied units, not measured physical resource prices. `replay` independently reconstructs each eligibility check, translated ordinal, and displacement. `verify_realization` checks the native source rank and transfers the destination. `lift_time` is a lower-level adapter seam: call only **after** independently qualifying the source ordinal; membership alone is not ordinal proof.

`select` is the standalone exact native ordinal service: skip complete periods, use bounded residual CRT rank and binary selection. It can return UNREACHABLE for a finite exhausted language or UNKNOWN under budget. Native `verify_realization` verifies FOUND results; source adapters may separately qualify finite exhaustion. Dense partial-period counting can exceed the CRT traversal budget; IFS uses its existing signed rank/partial rank/retained anchor routes rather than duplicating those systems in the native graph kernel.

## Usage evidence and reproduction

```sh
python -m unittest discover -s tests -p 'test_ground_network.py' -v
python ground_network_demo.py
```

Seven native test methods: 240 random language pairs, 720 signed-suffix decisions against exhaustive common-period trajectories, exact small ranks, 160 compatible three-edge chains with independently rechecked composite summaries, 60 costed graph controls against exhaustive bounded-walk minima, malformed and forged inputs, cyclic ordinal translation, finite transients and a large sparse CRT example. The standalone demo uses 45 MDD nodes across three Grounds, a 614,889,782,588,491,410-period source and an intermediate common period 32,589,158,477,190,044,730. An ordinal 10^900+173 is checked against a closed-form destination, and the target's extra prefix event is refused by the suffix route. Timings in the JSON ledger are local empirical measurements, not performance promises. IFS CIG-034A separately exercises actual proof-derived CNF decoders, nonlinear transport, and retained Seed anchors, through 1,023 original observation bits.

## To Do

Durable graph admission and source-specific adapters to native NCR observer operators; evidence-aware discovery of offered edge families; cost calibration for actual local/remote work; integration with E-CPU's authenticated recovery and publication (mathematical identity remains separate from execution authority).
