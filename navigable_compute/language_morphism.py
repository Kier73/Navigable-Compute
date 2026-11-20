"""NCR-003: certified one-way projection between observer-derived languages.

An intention change can discard distinctions only if the old source-program
classes refine the new ones. A quotient homomorphism then carries every
program code and its composition exactly. Reversing such a map is generally
impossible: this module emits a concrete pair of indistinguishable old codes
that new observations distinguish, rather than inventing lost information.
"""
from __future__ import annotations

from dataclasses import dataclass,replace
from hashlib import sha256

from .invariant_space import canonical
from .semantic_language import SemanticLanguage
from .semantic_language_verify import verify_semantic_language

MORPH_SCHEMA='navigable-compute.language-projection.v0.1'
OBSTRUCTION_SCHEMA='navigable-compute.language-nonrefinement.v0.1'


@dataclass(frozen=True)
class LanguageProjection:
    source_sha256: str
    from_language_sha256: str
    to_language_sha256: str
    mapping: tuple[int,...]
    digest: str

    def encode(self):
        return {'schema':MORPH_SCHEMA,'source_sha256':self.source_sha256,
                'from_language_sha256':self.from_language_sha256,
                'to_language_sha256':self.to_language_sha256,
                'mapping':list(self.mapping),
                'claim':'total-surjective-monoid-homomorphism;no-information-reconstruction'}

    def translate(self,code:int)->int:
        if type(code) is not int or not 0<=code<len(self.mapping):
            raise ValueError('source language code is not admitted')
        return self.mapping[code]


@dataclass(frozen=True)
class NonRefinementWitness:
    source_sha256: str
    from_language_sha256: str
    to_language_sha256: str
    first_original_opcode: int
    second_original_opcode: int
    digest: str

    def encode(self):
        return {'schema':OBSTRUCTION_SCHEMA,'source_sha256':self.source_sha256,
                'from_language_sha256':self.from_language_sha256,
                'to_language_sha256':self.to_language_sha256,
                'first_original_opcode':self.first_original_opcode,
                'second_original_opcode':self.second_original_opcode,
                'claim':'source-code-equivalent-but-target-code-distinct;projection-impossible'}


def assess_language_projection(machine,source:SemanticLanguage,target:SemanticLanguage):
    """Either construct an exact quotient projection, or exhibit a blocker."""
    if (not verify_semantic_language(machine,source) or
            not verify_semantic_language(machine,target) or
            source.source_sha256!=target.source_sha256):
        raise ValueError('unqualified or unrelated language pair')
    seen={}
    mapping={}
    for opcode in range(len(machine.operators)):
        old=source.class_of[opcode]
        new=target.class_of[opcode]
        if old in mapping and mapping[old] != new:
            first=seen[old]
            witness=NonRefinementWitness(machine.digest,source.digest,target.digest,first,opcode,'')
            return replace(witness,digest=sha256(canonical(witness.encode())).hexdigest())
        mapping[old]=new
        seen.setdefault(old,opcode)
    codes=tuple(mapping[i] for i in range(source.instruction_count))
    if set(codes)!=set(range(target.instruction_count)):
        raise AssertionError('source monoid projection unexpectedly non-surjective')
    for i in range(source.instruction_count):
        for j in range(source.instruction_count):
            if codes[source.multiplication[i][j]]!=target.multiplication[codes[i]][codes[j]]:
                raise AssertionError('candidate projection is not a homomorphism')
    result=LanguageProjection(machine.digest,source.digest,target.digest,codes,'')
    return replace(result,digest=sha256(canonical(result.encode())).hexdigest())
