"""Independent verification of declared effects against a compiled possibility Ground.

No use of compile_brief: reconstruct every action/state outcome directly
from original interval semantics, then check the chosen law with the separate
local Bellman and obstruction verifier. This is still one Python trust domain.
"""
from __future__ import annotations

from .brief import GroundBrief
from .possibility import ConstructedGround
from .possibility_verify import verify_world


def verify_brief_world(briefs:tuple[GroundBrief,...],ground:ConstructedGround)->bool:
    try:
        if (not isinstance(briefs,tuple) or len(briefs)!=len(ground.contracts) or
            any(not isinstance(b,GroundBrief) for b in briefs)):
            return False
        for brief,compiled in zip(briefs,ground.contracts):
            if (brief.states!=compiled.states or brief.initial!=compiled.initial or
                brief.goal!=compiled.goal or brief.unsafe!=compiled.unsafe or
                len(brief.actions)!=len(compiled.actions)):
                return False
            for sketch,action in zip(brief.actions,compiled.actions):
                if sketch.name!=action.name or len(action.outcomes)!=brief.states:
                    return False
                for state in range(brief.states):
                    matched=[case for case in sketch.cases if case.first<=state<case.stop]
                    if len(matched)>1:
                        return False
                    if not matched:
                        required=()
                    else:
                        case=matched[0]
                        if case.stop>brief.states:
                            return False
                        destinations=set(case.absolute)
                        for change in case.deltas:
                            destinations.add(max(0,min(brief.states-1,state+change)))
                        required=tuple(sorted(destinations))
                    if action.outcomes[state]!=required:
                        return False
        return verify_world(ground.contracts,ground)
    except (TypeError,ValueError,IndexError,AttributeError):
        return False
