"""CGR-002: construct an NCR-004 processor directly from source-reported transitions.

Evidence is source-reported and content-pinned, not independently authenticated.
Missing or conflicting transitions refuse rather than becoming inferred laws.
The actual quotient/operator construction and independent proof live in NCR-004.
No physical resource or E-CPU authority is created here.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from hashlib import sha256
import json

from .preclosure_language import (GeneratorObservation, PreclosureLanguage,
                                  construct_preclosure_language)
from .preclosure_verify import verify_preclosure_language
from .preclosure_wire import encode_preclosure, decode_preclosure


def _canonical(value: dict) -> bytes:
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode('ascii')


@dataclass(frozen=True)
class TransitionObservation:
    sequence: int
    before: int
    action: str
    after: int
    before_output: str
    after_output: str

    def packet(self) -> dict:
        return {key: getattr(self,key) for key in (
            'sequence','before','action','after','before_output','after_output')}


@dataclass(frozen=True)
class ObservedEpoch:
    """Source identity and declared finite vocabulary, WITHOUT a supplied law."""
    source: str
    context: str
    states: tuple[int,...]
    actions: tuple[str,...]
    initial: tuple[int,...]
    witnesses: tuple[TransitionObservation,...]

    def __post_init__(self):
        if (type(self.source) is not str or not self.source or len(self.source)>100 or
            type(self.context) is not str or not self.context or len(self.context)>100 or
            not self.source.isascii() or not self.context.isascii()):
            raise ValueError('invalid bounded source and context')
        if (type(self.states) is not tuple or not 1<=len(self.states)<=256 or
            any(type(s) is not int or s<0 or s>65535 for s in self.states) or
            tuple(sorted(set(self.states)))!=self.states):
            raise ValueError('invalid canonical state IDs')
        if (type(self.actions) is not tuple or not 1<=len(self.actions)<=32 or
            any(type(a) is not str or not a or len(a)>32 or not a.isascii()
                for a in self.actions) or tuple(sorted(set(self.actions)))!=self.actions):
            raise ValueError('invalid canonical action vocabulary')
        if (type(self.initial) is not tuple or not self.initial or
            tuple(sorted(set(self.initial)))!=self.initial or
            any(s not in self.states for s in self.initial)):
            raise ValueError('invalid initial state support')
        if type(self.witnesses) is not tuple or len(self.witnesses)>16384:
            raise ValueError('invalid bounded witness collection')
        labels={}; transitions={}
        for i, event in enumerate(self.witnesses):
            if (type(event) is not TransitionObservation or
                type(event.sequence) is not int or event.sequence!=i or
                event.before not in self.states or event.after not in self.states or
                event.action not in self.actions or
                any(type(v) is not str or not v or len(v)>128 or not v.isascii()
                    for v in (event.before_output,event.after_output))):
                raise ValueError('source witness outside declared epoch')
            for state,label in ((event.before,event.before_output),
                                (event.after,event.after_output)):
                if state in labels and labels[state]!=label:
                    raise ValueError('contradictory source readout requires new epoch')
                labels[state]=label
            pair=(event.before,event.action)
            if pair in transitions and transitions[pair]!=event.after:
                raise ValueError('contradictory source transition requires new epoch')
            transitions[pair]=event.after

    def packet(self)->dict:
        return {'schema':'navigable-compute.observed-source-epoch.v0.1',
                'source':self.source,'context':self.context,
                'states':list(self.states),'actions':list(self.actions),
                'initial':list(self.initial),
                'witnesses':[w.packet() for w in self.witnesses]}

    @property
    def source_head(self)->str:
        return sha256(_canonical(self.packet())).hexdigest()


class IncompleteSource(ValueError):
    """A required state-action observation is unknown, not an implicit self-loop."""
    def __init__(self,missing: tuple[tuple[int,str], ...],source_head: str):
        self.missing=missing
        self.source_head=source_head
        super().__init__(f'{len(missing)} reachable source transitions remain unknown')


@dataclass(frozen=True)
class SourceAdmission:
    source_head: str
    source: GeneratorObservation
    language: PreclosureLanguage
    wire: bytes


def admit_observed_epoch(epoch:ObservedEpoch,*,expected_head:str,
                         max_quotient_ops:int=1024)->SourceAdmission:
    """Qualify source-reported law; then delegate algebra/proof to NCR-004."""
    if not isinstance(epoch,ObservedEpoch) or type(expected_head) is not str or \
       epoch.source_head!=expected_head:
        raise ValueError('source epoch not pinned by external receiver')
    labels={}; edges={}
    for event in epoch.witnesses:
        labels[event.before]=event.before_output
        labels[event.after]=event.after_output
        edges[event.before,event.action]=event.after
    reached=set(epoch.initial)
    queue=deque(epoch.initial)
    missing=[]
    while queue:
        current=queue.popleft()
        if current not in labels:
            raise IncompleteSource(tuple((current,action) for action in epoch.actions),epoch.source_head)
        for action in epoch.actions:
            successor=edges.get((current,action))
            if successor is None:
                missing.append((current,action))
            elif successor not in reached:
                reached.add(successor)
                queue.append(successor)
    if missing:
        raise IncompleteSource(tuple(sorted(missing)),epoch.source_head)
    states=tuple(sorted(reached))
    position={state:i for i,state in enumerate(states)}
    source=GeneratorObservation(
        states,
        tuple(tuple(position[edges[state,action]] for state in states)
              for action in epoch.actions),
        epoch.initial,tuple(labels[state] for state in states),
        'source-witness:'+epoch.source_head,
    )
    language=construct_preclosure_language(source,max_quotient_ops=max_quotient_ops)
    if not verify_preclosure_language(source,language):
        raise ValueError('native independent verifier refused observed Ground')
    wire=encode_preclosure(source,language)
    if decode_preclosure(wire)!=(source,language):
        raise ValueError('portable native Ground roundtrip failed')
    return SourceAdmission(epoch.source_head,source,language,wire)


def navigate_observed(admission:SourceAdmission, *, expected_head:str,
                      initial:int, word:tuple[int,...], repetitions:int)->str:
    """Use NCR-004 exact logarithmic-time quotient map powering."""
    if (not isinstance(admission,SourceAdmission) or
        admission.source_head!=expected_head or
        not verify_preclosure_language(admission.source,admission.language) or
        decode_preclosure(admission.wire)!=(admission.source,admission.language) or
        type(initial) is not int or initial not in admission.source.initial or
        type(word) is not tuple or not word or len(word)>1024 or
        type(repetitions) is not int or repetitions<0 or repetitions.bit_length()>4096):
        raise ValueError('unqualified mathematical navigation')
    code,_=admission.language.power_code(word,repetitions)
    state_class=admission.language.state_class_of[admission.source.states.index(initial)]
    result=admission.language.operators[code][state_class]
    representative=admission.language.state_classes[result][0]
    return admission.source.outputs[admission.source.states.index(representative)]
