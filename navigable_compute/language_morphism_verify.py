"""Independent finite-identity checks for language projections and obstructions."""
from __future__ import annotations

from hashlib import sha256

from .invariant_space import canonical
from .language_morphism import LanguageProjection,NonRefinementWitness
from .semantic_language_verify import verify_semantic_language


def verify_language_relation(machine,source,target,result)->bool:
    try:
        if (not verify_semantic_language(machine,source) or
                not verify_semantic_language(machine,target) or
                source.source_sha256 != target.source_sha256):
            return False
        if isinstance(result,LanguageProjection):
            if (result.source_sha256 != machine.digest or
                    result.from_language_sha256 != source.digest or
                    result.to_language_sha256 != target.digest or
                    type(result.mapping) is not tuple or
                    len(result.mapping)!=source.instruction_count or
                    any(type(v) is not int or not 0<=v<target.instruction_count
                        for v in result.mapping)):
                return False
            if set(result.mapping)!=set(range(target.instruction_count)):
                return False
            # Check every original operation rather than the producer's
            # selected representatives; this is the exact factorization rule.
            for op in range(len(machine.operators)):
                if result.mapping[source.class_of[op]]!=target.class_of[op]:
                    return False
            for i in range(source.instruction_count):
                for j in range(source.instruction_count):
                    if result.mapping[source.multiplication[i][j]] != \
                            target.multiplication[result.mapping[i]][result.mapping[j]]:
                        return False
        elif isinstance(result,NonRefinementWitness):
            if (result.source_sha256!=machine.digest or
                    result.from_language_sha256!=source.digest or
                    result.to_language_sha256!=target.digest):
                return False
            a,b=result.first_original_opcode,result.second_original_opcode
            if (type(a) is not int or type(b) is not int or not
                    0<=a<len(machine.operators) or not 0<=b<len(machine.operators) or a>=b):
                return False
            # The obstruction is semantic: one source-language code needs
            # two different target-language images.
            if source.class_of[a]!=source.class_of[b] or target.class_of[a]==target.class_of[b]:
                return False
        else:
            return False
        return sha256(canonical(result.encode())).hexdigest()==result.digest
    except (ValueError,TypeError,KeyError,IndexError,AttributeError,OverflowError,MemoryError):
        return False
