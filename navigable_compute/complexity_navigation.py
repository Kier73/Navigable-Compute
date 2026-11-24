"""NCR-009: costed choice between qualified source and observer representations.

All inputs are declared NCR-007 Boolean sources. Timings are empirical,
and prices on bits are explicit scenario parameters, not measured networks.
"""
from dataclasses import dataclass
from statistics import median
from time import perf_counter_ns

from .nonlinear_koopman import (
    PolynomialBudget, construct_koopman_ground, eval_observers, pullback)
from .nonlinear_koopman_verify import (
    _source_substitute, verify_koopman_ground, verify_polynomial_laws)
from .realized_image import (
    ImageBudget, construct_image, verify_image, pack, unpack)


def _measure(fn):
    start = perf_counter_ns()
    value = fn()
    return value, perf_counter_ns() - start


def _support(polys):
    mask = 0
    for poly in polys:
        for monomial in poly:
            mask |= monomial
    return mask


def _prefix(family, word, repetitions):
    """Bound execution by the independently admitted causal drain horizon."""
    if (type(word) is not tuple or len(word) > 65536 or
        any(type(a) is not int or not 0 <= a < len(family.operators) for a in word)
        or type(repetitions) is not int or repetitions < 0
        or repetitions.bit_length() > 4096):
        raise ValueError("invalid action word or logical horizon")
    n = min(len(word)*repetitions, family.description.drain_horizon)
    return tuple(word[i % len(word)] for i in range(n))


def _protected(family):
    return sum(1 << i for i, tier in enumerate(family.description.tiers) if tier == -1)


def _terminal(family, prefix, *, independent=False):
    """Compile only the requested terminal observation, not future behavior."""
    polys = family.description.observers
    if len(prefix) == family.description.drain_horizon:
        preserved = _protected(family)
        return tuple(tuple(m for m in p if m & preserved == m) for p in polys)
    for action in reversed(prefix):
        op = family.operators[action]
        if independent:
            polys = tuple(_source_substitute(p, op, 1024, 16384) for p in polys)
        else:
            polys = tuple(pullback(p, op, term_limit=1024, product_limit=16384)
                          for p in polys)
    return polys


def _independent_eval(polys, state):
    """Separate set-based ANF evaluation for terminal and retained-state audit."""
    active = {i for i in range(state.bit_length()) if state >> i & 1}
    out = 0
    for j, poly in enumerate(polys):
        parity = 0
        for m in poly:
            if all(i in active for i in range(m.bit_length()) if m >> i & 1):
                parity ^= 1
        out |= parity << j
    return out


def _linear(rows, state):
    return sum(((row & state).bit_count() & 1) << i for i, row in enumerate(rows))


@dataclass(frozen=True)
class Candidate:
    """Qualified realization mode and its explicit preparation expense."""
    mode: str
    read_mask: int
    retained_bits: int
    setup_ns: int
    polys: tuple = ()


@dataclass(frozen=True)
class Profile:
    """Calibration medians include independent per-query mathematical checks."""
    candidate: Candidate
    warm_ns: int
    calibration_ns: int

    def score(self, count, *, bit_price=0, read_price=0, cold=False):
        """Score in declared ns-equivalent cost units, not a latency promise."""
        if (type(count) is not int or count < 1 or
            any(type(v) is not int or v < 0 for v in (bit_price, read_price))
            or type(cold) is not bool):
            raise ValueError("invalid campaign prices")
        c = self.candidate
        return ((c.setup_ns if cold else 0) + count *
                (self.warm_ns + bit_price*c.retained_bits +
                 read_price*c.read_mask.bit_count()))


class Portfolio:
    """Admit alternatives, calibrate on source states, select and execute.

    The terminal candidate cannot be used as a future-complete state.
    A full initial state is available to the independent local verifier.
    """
    def __init__(self, family, word, repetitions):
        admitted, self.source_qualification_ns = _measure(
            lambda: verify_polynomial_laws(family))
        if not admitted:
            raise ValueError("unqualified source")
        self.family = family
        self.prefix = _prefix(family, word, repetitions)
        self.word, self.repetitions = word, repetitions
        self.ground = self.image = None
        self.failures = {}
        self.failed_setup_ns = 0
        self.candidates = {}
        full = (1 << family.description.width)-1
        self.candidates["source"] = Candidate("source", full,
                                               family.description.width, 0)
        start = perf_counter_ns()
        try:
            polys = _terminal(family, self.prefix)
            if polys != _terminal(family, self.prefix, independent=True):
                raise ValueError("independent terminal compiler disagrees")
            self.candidates["terminal"] = Candidate(
                "terminal", _support(polys), len(polys), perf_counter_ns()-start, polys)
        except PolynomialBudget as exc:
            self.failures["terminal"] = str(exc)
            self.failed_setup_ns += perf_counter_ns()-start
        start = perf_counter_ns()
        try:
            self.ground = construct_koopman_ground(family)
            if not verify_koopman_ground(family, self.ground):
                raise ValueError("unqualified feature Ground")
        except PolynomialBudget as exc:
            self.failures["feature"] = str(exc)
            self.failed_setup_ns += perf_counter_ns()-start
            self.ground = None
        if self.ground is not None:
            g = self.ground
            feature_ns = perf_counter_ns()-start
            mask = _support(g.basis)
            self.candidates["feature"] = Candidate("feature", mask, g.dimension, feature_ns)
            start = perf_counter_ns()
            try:
                self.image = construct_image(g.basis, family.description.width)
                if not verify_image(g.basis, family.description.width, self.image):
                    raise ValueError("unqualified packed image")
                self.candidates["packed"] = Candidate(
                    "packed", mask, self.image.bits, feature_ns+perf_counter_ns()-start)
            except ImageBudget as exc:
                self.failures["packed"] = str(exc)
                self.failed_setup_ns += perf_counter_ns()-start
                self.image = None
        self.preparation_ns = (
            self.source_qualification_ns + self.failed_setup_ns +
            max((c.setup_ns for c in self.candidates.values()
                 if c.mode in ("feature", "packed")), default=0) +
            self.candidates.get("terminal", Candidate("terminal", 0, 0, 0)).setup_ns)

    def _validate_state(self, state):
        if type(state) is not int or not 0 <= state < 1 << self.family.description.width:
            raise ValueError("invalid source state")

    def _reference(self, state):
        # Independent source replay, not the producer's terminal shortcut.
        for a in self.prefix:
            state = _independent_eval(self.family.operators[a].rows, state)
        return state

    def realize(self, mode, state):
        """Return independently checked (observation, retained coordinate)."""
        self._validate_state(state)
        candidate = self.candidates[mode]
        initial = state & candidate.read_mask
        if mode == "source":
            z = initial
            if len(self.prefix) == self.family.description.drain_horizon:
                z &= _protected(self.family)
            else:
                for a in self.prefix:
                    z = self.family.operators[a].apply(z)
            observation = self.family.project_observer(z)
        elif mode == "terminal":
            z = observation = eval_observers(candidate.polys, initial)
        else:
            g = self.ground
            z = g.project(initial)
            for a in self.prefix:
                z = _linear(g.operator_rows[a], z)
            observation = g.read(z)
            if mode == "packed":
                z = pack(self.image, z)
        expected = self._reference(state)
        if observation != _independent_eval(self.family.description.observers, expected):
            raise ValueError("independent observer check failed")
        if mode == "source" and z != expected:
            raise ValueError("source state check failed")
        if mode in ("feature", "packed"):
            decoded = unpack(self.image, z) if mode == "packed" else z
            if decoded != _independent_eval(self.ground.basis, expected):
                raise ValueError("independent future state check failed")
        return observation, z

    def continue_from(self, mode, coordinate, action):
        """Apply another source action using only a future-complete coordinate."""
        if mode == "terminal" or mode not in self.candidates:
            raise ValueError("no future-complete coordinate")
        if type(action) is not int or not 0 <= action < len(self.family.operators):
            raise ValueError("invalid action")
        if mode == "source":
            self._validate_state(coordinate)
            z = self.family.operators[action].apply(coordinate)
            return self.family.project_observer(z), z
        g = self.ground
        if mode == "packed":
            coordinate = unpack(self.image, coordinate)
        if type(coordinate) is not int or not 0 <= coordinate < 1 << g.dimension:
            raise ValueError("invalid feature coordinate")
        z = _linear(g.operator_rows[action], coordinate)
        return g.read(z), pack(self.image, z) if mode == "packed" else z

    def calibrate(self, states):
        """Measure all candidates on an explicit calibration corpus."""
        states = tuple(states)
        if not states or len(states) > 256:
            raise ValueError("invalid calibration corpus")
        profiles = []
        for candidate in self.candidates.values():
            samples = []
            for x in states:
                _, elapsed = _measure(lambda: self.realize(candidate.mode, x))
                samples.append(elapsed)
            profiles.append(Profile(candidate, int(median(samples)), sum(samples)))
        return tuple(profiles)


def choose(profiles, count, *, continuation=False, max_bits=None,
           bit_price=0, read_price=0, cold=False):
    """Choose lowest declared score among semantically eligible candidates.

    Calibration and exploration are sunk for a warm choice. Cold=True is a
    counterfactual standalone plan estimate and not a free discovery policy.
    """
    if max_bits is not None and (type(max_bits) is not int or max_bits < 0):
        raise ValueError("invalid retained-state ceiling")
    eligible = [p for p in profiles
                if (not continuation or p.candidate.mode != "terminal")
                and (max_bits is None or p.candidate.retained_bits <= max_bits)]
    if not eligible:
        raise ValueError("no eligible representation")
    return min(eligible, key=lambda p: (
        p.score(count, bit_price=bit_price, read_price=read_price, cold=cold),
        p.candidate.mode))
