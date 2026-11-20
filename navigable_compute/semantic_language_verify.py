"""NCR-003 independent exhaustive-context checker.

Unlike the partition-refinement constructor, the verifier enumerates every
left AND right context and every admitted starting state. It reconstructs the
unique observational congruence from semantic behavior, so a forged partition
or multiplication table cannot be accepted by recomputing a hash.
"""
from __future__ import annotations

from hashlib import sha256

from .invariant_space import canonical
from .self_space import EndogenousProcessor
from .self_verify import verify_endogenous_processor
from .semantic_language import SemanticLanguage, ObservationContract


def _context_signature(machine: EndogenousProcessor, contract: ObservationContract, opcode: int,
                       idxs: tuple[int, ...]) -> tuple[str, ...]:
    result = []
    # Compose the original source operators by direct source-state arrays;
    # the certificate's derived quotient/multiplication is never used here.
    maps = machine.operators
    for before in maps:
        for after in maps:
            for state in idxs:
                result.append(contract.outputs[after[maps[opcode][before[state]]]])
    return tuple(result)


def verify_semantic_language(machine: EndogenousProcessor, cert: SemanticLanguage,
                             *, max_source_ops: int = 64) -> bool:
    try:
        if (not isinstance(machine, EndogenousProcessor) or
                not verify_endogenous_processor(machine.source.description, machine) or
                not isinstance(cert, SemanticLanguage) or
                cert.source_sha256 != machine.digest):
            return False
        n = len(machine.operators)
        if (type(max_source_ops) is not int or not 1 <= max_source_ops <= 128 or
                n > max_source_ops or not isinstance(cert.contract, ObservationContract)):
            return False
        cert.contract.check(machine)
        dense_initial = tuple(machine.states.index(s) for s in cert.contract.initial)
        signatures = [_context_signature(machine, cert.contract, opcode, dense_initial)
                      for opcode in range(n)]
        expected_by_sig = {}
        blocks = []
        for i, sig in enumerate(signatures):
            if sig not in expected_by_sig:
                expected_by_sig[sig] = len(blocks)
                blocks.append([])
            blocks[expected_by_sig[sig]].append(i)
        canonical_blocks = tuple(tuple(block) for block in blocks)
        if cert.classes != canonical_blocks or type(cert.class_of) is not tuple:
            return False
        expected_class = [0]*n
        for idx, group in enumerate(canonical_blocks):
            for element in group:
                expected_class[element] = idx
        if cert.class_of != tuple(expected_class):
            return False
        q = len(blocks)
        if not (len(cert.multiplication) == len(cert.normal_forms) == len(cert.regular_actions) == q):
            return False
        # Directly recompose source maps to reconstruct the quotient table.
        maps = machine.operators
        for i, block in enumerate(blocks):
            row = cert.multiplication[i]
            if type(row) is not tuple or len(row) != q:
                return False
            for j, other in enumerate(blocks):
                target = tuple(maps[block[0]][maps[other[0]][s]] for s in range(len(machine.states)))
                try:
                    op = maps.index(target)
                except ValueError:
                    return False
                if type(row[j]) is not int or row[j] != expected_class[op]:
                    return False
        # Congruence must be independent of the representatives, not just
        # correct on one chosen representative per class.
        for x in range(n):
            for y in range(n):
                if cert.multiplication[expected_class[x]][expected_class[y]] != expected_class[machine.multiply[x][y]]:
                    return False
        generators = tuple(expected_class[x] for x in machine.generator_opcodes)
        if cert.generator_codes != generators:
            return False
        # Independently compute shortest-lex generator words on the certified
        # quotient, to rule out supplied long or incorrect instruction forms.
        unit = expected_class[0]
        shortest = {unit: ()}
        queue = [unit]
        for code in queue:
            for symbol, g in enumerate(generators):
                successor = cert.multiplication[g][code]
                if successor not in shortest:
                    shortest[successor] = shortest[code] + (symbol,)
                    queue.append(successor)
        if len(queue) != q or cert.normal_forms != tuple(shortest[i] for i in range(q)):
            return False
        # The generated Cayley action must be faithful: every code is uniquely
        # recovered as what it does to the identity code.
        if cert.regular_actions != cert.multiplication or any(row[unit] != i for i,row in enumerate(cert.regular_actions)):
            return False
        # A derivation-cost field is also independently checked. Compute
        # successive separations without invoking the producer partitioner.
        initial_signatures = [tuple(cert.contract.outputs[maps[j][s]]
                                    for s in dense_initial) for j in range(n)]
        def indexed(values):
            ids = {}
            return tuple(ids.setdefault(value, len(ids)) for value in values)
        old = indexed(initial_signatures)
        refinement_passes = 0
        while True:
            next_labels = indexed(tuple((initial_signatures[j],
                    tuple(old[machine.multiply[x][j]] for x in range(n)),
                    tuple(old[machine.multiply[j][x]] for x in range(n)))
                    for j in range(n)))
            if next_labels == old:
                break
            if refinement_passes >= n:
                return False
            old = next_labels
            refinement_passes += 1
        if (type(cert.refinements) is not int or cert.refinements != refinement_passes or
                sha256(canonical(cert.encode())).hexdigest() != cert.digest):
            return False
        return True
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError, MemoryError):
        return False
