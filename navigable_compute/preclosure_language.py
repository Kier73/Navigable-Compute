"""NCR-004: derive the observer language before closing the source operator algebra.

Given only finite *generator maps*, reachable initial states, and observer
labels, derive the largest Moore state congruence and then close generators
on that quotient. This avoids enumerating the original source monoid.

Correctness is scoped to the declared finite, deterministic and stationary
source transformations. It is not a general compressed automaton oracle,
external observation admission, signed evidence, or physical execution.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Any

from .invariant_space import InvariantDescription, construct_invariant_space, navigate_invariant_instance, canonical
from .invariant_verify import verify_invariant_space

SCHEMA = 'navigable-compute.preclosure-observer-language.v0.1'


class QuotientExplosion(ValueError):
    """Observer-preserving state or operator representation exceeds budget."""


@dataclass(frozen=True)
class GeneratorObservation:
    """Finite source-generators, not a preclosed monoid.

    Generator tuples use dense state positions, while `states` and `initial`
    are stable external IDs. The `origin` links the source to a separately
    qualified Ground; no caller-supplied digest authenticates it.
    """
    states: tuple[int, ...]
    generators: tuple[tuple[int, ...], ...]
    initial: tuple[int, ...]
    outputs: tuple[str, ...]
    origin: str = 'declared-finite-generator-hypothesis'

    def __post_init__(self) -> None:
        n = len(self.states)
        if (type(self.states) is not tuple or not 1 <= n <= 256 or
            any(type(x) is not int for x in self.states) or
            tuple(sorted(set(self.states))) != self.states):
            raise ValueError('require 1..256 canonical source state IDs')
        if (type(self.generators) is not tuple or not 1 <= len(self.generators) <= 32 or
            any(type(g) is not tuple or len(g) != n or
                any(type(v) is not int or not 0 <= v < n for v in g)
                for g in self.generators)):
            raise ValueError('finite total generator functions required')
        if (type(self.initial) is not tuple or not self.initial or
            tuple(sorted(set(self.initial))) != self.initial or
            any(type(v) is not int or v not in self.states for v in self.initial)):
            raise ValueError('initial states must be a nonempty canonical subset')
        if (type(self.outputs) is not tuple or len(self.outputs) != n or
            any(type(x) is not str or not x or len(x) > 128 or not x.isascii()
                for x in self.outputs)):
            raise ValueError('one bounded source observation per state required')
        if (type(self.origin) is not str or not 1 <= len(self.origin) <= 180 or
            not self.origin.isascii()):
            raise ValueError('invalid source Ground description')

    def encode(self, *, include_outputs: bool = True) -> dict[str, Any]:
        result = {'states': list(self.states),
                  'generators': [list(g) for g in self.generators],
                  'initial': list(self.initial), 'origin': self.origin}
        if include_outputs:
            result['outputs'] = list(self.outputs)
        return result

    @property
    def dynamics_digest(self) -> str:
        return sha256(canonical(self.encode(include_outputs=False))).hexdigest()

    @property
    def observation_digest(self) -> str:
        return sha256(canonical(self.encode())).hexdigest()


def generators_from_invariants(description: InvariantDescription, outputs: tuple[str, ...], *,
                               generator_limit: int = 16,
                               initial: tuple[int, ...] | None = None) -> GeneratorObservation:
    """Obtain generator laws directly from NCI, without NCS operator closure."""
    if type(generator_limit) is not int or not 1 <= generator_limit <= 32:
        raise ValueError('invalid law generator limit')
    source = construct_invariant_space(description)
    if not verify_invariant_space(description, source):
        raise ValueError('invariant Ground failed independent verification')
    states = tuple(s for s, rank in enumerate(source.ranks) if rank >= 0)
    if len(outputs) != len(states):
        raise ValueError('outputs must match qualified invariant states')
    ids = {s: i for i, s in enumerate(states)}
    selected = tuple(range(min(source.law_count, generator_limit)))
    generators = tuple(tuple(ids[navigate_invariant_instance(source, k).transition[s]]
                             for s in states) for k in selected)
    if initial is None:
        initial = states
    return GeneratorObservation(states, generators, initial, outputs,
                                f'NCI-001:{sha256(canonical(description.encode())).hexdigest()}:{len(selected)}')


@dataclass(frozen=True)
class PreclosureLanguage:
    dynamics_sha256: str
    observation_sha256: str
    reachable: tuple[int, ...]                 # actual source IDs
    state_classes: tuple[tuple[int, ...], ...] # observer Moore equivalence
    state_class_of: tuple[int, ...]            # dense source ID, -1 unreachable
    quotient_generators: tuple[tuple[int, ...], ...]
    operators: tuple[tuple[int, ...], ...]
    normal_forms: tuple[tuple[int, ...], ...]
    generator_opcodes: tuple[int, ...]
    multiplication: tuple[tuple[int, ...], ...]
    state_refinements: int
    digest: str

    @property
    def instruction_count(self) -> int:
        return len(self.operators)

    def encode(self) -> dict:
        return {'schema': SCHEMA, 'dynamics_sha256': self.dynamics_sha256,
                'observation_sha256': self.observation_sha256,
                'reachable': list(self.reachable),
                'state_classes': [list(b) for b in self.state_classes],
                'state_class_of': list(self.state_class_of),
                'quotient_generators': [list(x) for x in self.quotient_generators],
                'operators': [list(x) for x in self.operators],
                'normal_forms': [list(x) for x in self.normal_forms],
                'generator_opcodes': list(self.generator_opcodes),
                'multiplication': [list(x) for x in self.multiplication],
                'state_refinements': self.state_refinements,
                'claim': 'reachable-Moore-quotient-before-operator-closure;finite-stationary-only'}

    def packet(self) -> dict:
        return {**self.encode(), 'sha256': self.digest}

    def word_code(self, word: tuple[int, ...]) -> int:
        if type(word) is not tuple or len(word) > 65536:
            raise ValueError('bounded tuple of generator IDs required')
        code = 0
        for g in word:
            if type(g) is not int or not 0 <= g < len(self.generator_opcodes):
                raise ValueError('unknown generator')
            code = self.multiplication[self.generator_opcodes[g]][code]
        return code

    def observe(self, system: GeneratorObservation, word: tuple[int, ...], initial: int) -> str:
        if system.dynamics_digest != self.dynamics_sha256 or system.observation_digest != self.observation_sha256:
            raise ValueError('changed source semantics or evidence scope')
        if type(initial) is not int or initial not in system.initial:
            raise ValueError('unadmitted initial source state')
        state_class = self.state_class_of[system.states.index(initial)]
        after_class = self.operators[self.word_code(word)][state_class]
        sample = self.state_classes[after_class][0]
        return system.outputs[system.states.index(sample)]

    def power_code(self, word: tuple[int, ...], repetitions: int) -> tuple[int, int]:
        if type(repetitions) is not int or repetitions < 0 or repetitions.bit_length() > 4096:
            raise ValueError('invalid logical time coordinate')
        result = 0
        power = self.word_code(word)
        cost = 0
        while repetitions:
            if repetitions & 1:
                result = self.multiplication[power][result]
                cost += 1
            repetitions >>= 1
            if repetitions:
                power = self.multiplication[power][power]
                cost += 1
        return result, cost


def _partition(keys: tuple) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    signatures: dict[tuple, int] = {}
    groups: list[list[int]] = []
    positions = []
    for i, key in enumerate(keys):
        if key not in signatures:
            signatures[key] = len(groups)
            groups.append([])
        j = signatures[key]
        groups[j].append(i)
        positions.append(j)
    return tuple(tuple(x) for x in groups), tuple(positions)


def construct_preclosure_language(system: GeneratorObservation, *,
                                  max_quotient_ops: int = 1024,
                                  max_source_states: int = 256) -> PreclosureLanguage:
    """Minimize reachable source states BEFORE generating the source monoid.

    Source closure is never performed, even for an intractably large original
    operator algebra. Discovery work depends on finite states/generators and
    on the admitted quotient's actual operator closure.
    """
    if not isinstance(system, GeneratorObservation):
        raise ValueError('invalid generator Ground')
    if (type(max_source_states) is not int or type(max_quotient_ops) is not int or
        not 1 <= max_source_states <= 256 or
        not 1 <= max_quotient_ops <= 4096):
        raise ValueError('invalid computation budgets')
    if len(system.states) > max_source_states:
        raise QuotientExplosion('source state budget exceeded')
    n = len(system.states)
    index = {s: i for i, s in enumerate(system.states)}
    start = [index[s] for s in system.initial]
    reached = set(start)
    frontier = list(start)
    for s in frontier:
        for g in system.generators:
            if g[s] not in reached:
                reached.add(g[s])
                frontier.append(g[s])
    reachable_indices = tuple(sorted(reached))
    keys = tuple((system.outputs[s],) for s in reachable_indices)
    _, class_of_reachable = _partition(keys)
    source_pos = {s: i for i, s in enumerate(reachable_indices)}
    refinements = 0
    while True:
        keys = tuple((system.outputs[s],
            tuple(class_of_reachable[source_pos[g[s]]] for g in system.generators))
            for s in reachable_indices)
        classes, upgraded = _partition(keys)
        if upgraded == class_of_reachable:
            break
        if len(classes) <= len(set(class_of_reachable)) or refinements >= len(reachable_indices):
            raise AssertionError('invalid monotone state partition')
        class_of_reachable = upgraded
        refinements += 1
    classes_indices, _ = _partition(keys)
    state_class = [-1] * n
    for position, s in enumerate(reachable_indices):
        state_class[s] = class_of_reachable[position]
    quotient_generators = tuple(tuple(state_class[g[reachable_indices[block[0]]]]
                                      for block in classes_indices)
                                for g in system.generators)
    ident = tuple(range(len(classes_indices)))
    operators = [ident]
    words = [()]
    idx = {ident: 0}
    for pos, old in enumerate(operators):
        for g_id, g in enumerate(quotient_generators):
            after = tuple(g[old[s]] for s in range(len(old)))
            if after not in idx:
                if len(operators) >= max_quotient_ops:
                    raise QuotientExplosion('observer quotient operator budget exceeded')
                idx[after] = len(operators)
                operators.append(after)
                words.append(words[pos] + (g_id,))
    mul = tuple(tuple(idx[tuple(left[right[s]] for s in range(len(ident)))]
                      for right in operators) for left in operators)
    generator_ids = tuple(idx[g] for g in quotient_generators)
    classes = tuple(tuple(system.states[reachable_indices[i]] for i in block)
                    for block in classes_indices)
    result = PreclosureLanguage(system.dynamics_digest, system.observation_digest,
              tuple(system.states[s] for s in reachable_indices), classes,
              tuple(state_class), quotient_generators, tuple(operators),
              tuple(words), generator_ids, mul, refinements, '')
    return replace(result, digest=sha256(canonical(result.encode())).hexdigest())


@dataclass(frozen=True)
class LanguageRevision:
    """A new observation refines the old, preserving causal identity."""
    prior_language_sha256: str
    new_language_sha256: str
    dynamics_sha256: str
    new_to_old_opcodes: tuple[int, ...]
    new_to_old_states: tuple[int, ...]
    lost_reverse_pair: tuple[int, int] | None
    digest: str

    def encode(self) -> dict:
        return {'schema': 'navigable-compute.preclosure-language-revision.v0.1',
                'prior_language_sha256': self.prior_language_sha256,
                'new_language_sha256': self.new_language_sha256,
                'dynamics_sha256': self.dynamics_sha256,
                'new_to_old_opcodes': list(self.new_to_old_opcodes),
                'new_to_old_states': list(self.new_to_old_states),
                'lost_reverse_pair': list(self.lost_reverse_pair) if self.lost_reverse_pair is not None else None,
                'claim': 'monotone-observation-refinement;mathematical-only'}


def derive_language_revision(old_system: GeneratorObservation, old: PreclosureLanguage,
                             new_system: GeneratorObservation, new: PreclosureLanguage) -> LanguageRevision:
    """Map a richer language to its predecessor, or refuse non-refinement.

    Changed transitions, changed initial support, source identity changes, or
    changed observational information incompatible with the predecessor fail.
    Observed environment *law changes* require a new Ground epoch, not this map.
    """
    if (old_system.dynamics_digest != new_system.dynamics_digest or
        old.dynamics_sha256 != old_system.dynamics_digest or
        new.dynamics_sha256 != old_system.dynamics_digest or
        old.observation_sha256 != old_system.observation_digest or
        new.observation_sha256 != new_system.observation_digest):
        raise ValueError('changed source identity or unchecked observation views')
    old_labels = old_system.outputs
    new_labels = new_system.outputs
    if any(new_labels[a] == new_labels[b] and old_labels[a] != old_labels[b]
           for a in range(len(old_labels)) for b in range(len(old_labels))):
        raise ValueError('new observations do not preserve prior information')
    if old.reachable != new.reachable:
        raise ValueError('changed reachable support')
    state_map = []
    for group in new.state_classes:
        targets = {old.state_class_of[old_system.states.index(s)] for s in group}
        if len(targets) != 1:
            raise ValueError('new state class merges previously distinguished states')
        state_map.append(next(iter(targets)))
    opcode_map = tuple(old.word_code(word) for word in new.normal_forms)
    for a, row in enumerate(new.multiplication):
        for b, k in enumerate(row):
            if opcode_map[k] != old.multiplication[opcode_map[a]][opcode_map[b]]:
                raise ValueError('language revision does not preserve composition')
    if set(opcode_map) != set(range(old.instruction_count)):
        raise ValueError('revision no longer covers all old language codes')
    lost = next(((a, b) for a in range(len(opcode_map)) for b in range(a + 1, len(opcode_map))
                if opcode_map[a] == opcode_map[b]), None)
    result = LanguageRevision(old.digest, new.digest, old.dynamics_sha256,
                              opcode_map, tuple(state_map), lost, '')
    return replace(result, digest=sha256(canonical(result.encode())).hexdigest())
