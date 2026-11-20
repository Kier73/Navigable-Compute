"""NCR-002: synthesize a reflective architecture from a semantic contract.

The contract supplies a finite data/AST grammar and the required semantics of
seven operations; *not* a finished interpreter.  The constructor selects a
minimum primitive basis in a declared finite template space, synthesizes the
remaining operations, chooses a Ground-dependent source encoding and emits a
self-applicable interpreter AST.  Every cost, template and remaining external
seed is explicit.  This is not unrestricted program synthesis or hardware.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
from math import factorial
import json

from .self_space import EndogenousProcessor
from .self_verify import verify_endogenous_processor
from .reflective_ground import (arg, lit, primitive, choice, call,
                                compile_task_program, host_run)

SCHEMA = 'navigable-compute.reflective-architecture.v0.1'
FORMS = ('lit', 'arg', 'if', 'prim', 'call')
OPERATIONS = ('at', 'cons', 'empty', 'eq', 'first', 'rest', 'task')

# Exactly this search grammar is part of the contract.  These equivalences
# hold for admissible typed operands; invalid-input behavior is NOT promised
# identical for macros on malformed source programs.
RECIPES = {
    'at': (('seed-at', ('at',)),),
    'cons': (('seed-cons', ('cons',)),),
    'eq': (('seed-eq', ('eq',)),),
    'rest': (('seed-rest', ('rest',)),),
    'first': (('seed-first', ('first',)), ('index-zero', ('at',))),
    'empty': (('seed-empty', ('empty',)), ('compare-empty', ('eq',))),
    'task': (('seed-task', ('task',)), ('nested-index', ('at',))),
}


def _jsonbytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      separators=(',', ':')).encode('ascii')


def _hash(value):
    return sha256(_jsonbytes(value)).hexdigest()


def _select_basis(allowed: tuple[str, ...]):
    """Enumerate all bases and recipes in a finite, declared template family."""
    if (type(allowed) is not tuple or len(set(allowed)) != len(allowed)
            or any(type(x) is not str or x not in OPERATIONS for x in allowed)):
        raise ValueError('invalid permitted primitive seed')
    allowed = tuple(sorted(allowed))
    candidates = []
    for width in range(len(allowed)+1):
        for subset in combinations(allowed, width):
            available = set(subset)
            selected = []
            for operation in OPERATIONS:
                choices = [name for name, needed in RECIPES[operation]
                           if set(needed) <= available]
                if not choices:
                    break
                # The first recipe is the raw primitive when available;
                # otherwise take the first admissible derived construction.
                selected.append((operation, choices[0]))
            if len(selected) == len(OPERATIONS):
                candidates.append((subset, tuple(selected)))
        if candidates:
            break
    if not candidates:
        raise ValueError('no fully interpreting architecture in admitted template space')
    return min(candidates, key=lambda x:x[0])


def _codebook(ground_sha, coordinate):
    """Unrank one of 5! representation gauges without enumerating them."""
    if type(coordinate) is not int or not 0 <= coordinate < factorial(len(FORMS)):
        raise ValueError('representation coordinate outside finite design space')
    # The canonical lexicographic gauge originates from the Ground fingerprint.
    available = sorted(FORMS, key=lambda tag:sha256((ground_sha+':'+tag).encode()).hexdigest())
    index = coordinate
    order = []
    for count in range(len(available),0,-1):
        divisor = factorial(count-1)
        digit,index = divmod(index,divisor)
        order.append(available.pop(digit))
    return tuple((tag,position) for position,tag in enumerate(order))


def encode_source(program, codebook):
    """Reify a source program under its constructed instruction representation."""
    mapping = dict(codebook)
    if set(mapping) != set(FORMS) or set(mapping.values()) != set(range(5)):
        raise ValueError('invalid reflective source alphabet')
    if type(program) is not tuple:
        raise ValueError('source functions must be a tuple')

    def expression(node, depth=0):
        if depth > 128 or type(node) is not tuple or not node:
            raise ValueError('invalid source expression')
        tag = node[0]
        if tag == 'lit' and len(node)==2:
            return (mapping[tag],node[1])  # Quoted data is never reinterpreted.
        if tag == 'arg' and len(node)==2 and type(node[1]) is int:
            return (mapping[tag],node[1])
        if tag == 'if' and len(node)==4:
            return (mapping[tag],*(expression(v,depth+1) for v in node[1:]))
        if tag in ('prim','call') and len(node)==3 and type(node[1]) is str and type(node[2]) is tuple:
            return (mapping[tag],node[1],tuple(expression(v,depth+1) for v in node[2]))
        raise ValueError('unqualified source syntax')

    return tuple((name,expression(expr)) for name,expr in program)


def _macro(name,values,table,recipe):
    """Construct each semantic operation from a *selected* recipe."""
    v0 = primitive('at',values,lit(0))
    v1 = primitive('at',values,lit(1))
    if recipe.startswith('seed-'):
        return primitive(name,*((v0,v1) if name in ('eq','cons','at','task') else (v0,)))
    if name == 'first' and recipe == 'index-zero':
        return primitive('at',v0,lit(0))
    if name == 'empty' and recipe == 'compare-empty':
        return primitive('eq',v0,lit(()))
    if name == 'task' and recipe == 'nested-index':
        return primitive('at',primitive('at',table,v0),v1)
    raise AssertionError('unrecognized semantic recipe')


def _build_program(codebook, recipes):
    """Generate the interpreter syntax tree from semantic rules and recipes."""
    codes = dict(codebook)
    base = dict(recipes)
    # A tiny protocol-generated helper vocabulary, not user-supplied bytecode.
    def m(op,*args):
        # Use the same *derived* semantic macro when building control logic,
        # not privileged host calls to removed primitives.
        values = arg(0)  # unused: actual expression in args below
        if op in ('first','empty','task') and not base[op].startswith('seed-'):
            if op=='first': return primitive('at',args[0],lit(0))
            if op=='empty': return primitive('eq',args[0],lit(()))
            return primitive('at',primitive('at',args[2],args[0]),args[1])
        return primitive(op,*args)

    z=lit(())
    functions=arg(0)
    name=arg(1)
    lookup=choice(m('eq',m('first',m('first',functions)),name),
                  m('at',m('first',functions),lit(1)),
                  call('lookup',m('rest',functions),name))

    p_name,p_values,p_table=arg(0),arg(1),arg(2)
    # Unknown names deliberately produce a failing expression.
    dispatch=m('at',z,lit(0))
    for op in reversed(OPERATIONS):
        dispatch=choice(m('eq',p_name,lit(op)),
                        _macro(op,p_values,p_table,base[op]),dispatch)

    node,frame,program,table=arg(0),arg(1),arg(2),arg(3)
    tag=m('first',node)
    # Encoded syntax tags are Ground-dependent integers, not fixed strings.
    branches={
        'lit':m('at',node,lit(1)),
        'arg':m('at',frame,m('at',node,lit(1))),
        'if':choice(call('evaluate',m('at',node,lit(1)),frame,program,table),
                    call('evaluate',m('at',node,lit(2)),frame,program,table),
                    call('evaluate',m('at',node,lit(3)),frame,program,table)),
        'prim':call('perform',m('at',node,lit(1)),
                    call('evaluate_list',m('at',node,lit(2)),frame,program,table),table),
        'call':call('invoke',program,m('at',node,lit(1)),
                    call('evaluate_list',m('at',node,lit(2)),frame,program,table),table),
    }
    evaluator=m('at',z,lit(0))
    for form in reversed(FORMS):
        evaluator=choice(m('eq',tag,lit(codes[form])),branches[form],evaluator)

    nodes,env,prog,ground=arg(0),arg(1),arg(2),arg(3)
    eval_list=choice(m('empty',nodes),z,
        m('cons',call('evaluate',m('first',nodes),env,prog,ground),
          call('evaluate_list',m('rest',nodes),env,prog,ground)))
    return (
       ('lookup',lookup),
       ('evaluate_list',eval_list),
       ('perform',dispatch),
       ('evaluate',evaluator),
       ('invoke',call('evaluate',call('lookup',arg(0),arg(1)),arg(2),arg(0),arg(3))),
       ('run',call('invoke',arg(0),arg(1),arg(2),arg(3))),
    )


@dataclass(frozen=True)
class ReflectiveArchitecture:
    ground_sha256: str
    primitive_seed: tuple[str,...]
    recipes: tuple[tuple[str,str],...]
    syntax_codebook: tuple[tuple[str,int],...]
    interpreter: tuple
    representation_coordinate: int
    digest: str

    def encode(self):
        return {'schema':SCHEMA,
                'ground_sha256':self.ground_sha256,
                'primitive_seed':list(self.primitive_seed),
                'recipes':[list(row) for row in self.recipes],
                'syntax_codebook':[list(row) for row in self.syntax_codebook],
                'interpreter_source':self.interpreter,
                'representation_coordinate':self.representation_coordinate,
                'representation_count':factorial(len(FORMS)),
                'claim':'finite-contract-synthesis;encoded-same-language-self-interpretation;seeded-grammar'}

    def packet(self):
        return {**self.encode(),'sha256':self.digest}


def construct_reflective_architecture(machine:EndogenousProcessor,*,
                                      allowed_primitives=OPERATIONS,
                                      representation_coordinate:int=0)->ReflectiveArchitecture:
    if not isinstance(machine,EndogenousProcessor) or not verify_endogenous_processor(machine.source.description,machine):
        raise ValueError('requires a qualified endogenous processor Ground')
    basis,recipes=_select_basis(allowed_primitives)
    codebook=_codebook(machine.digest,representation_coordinate)
    program=_build_program(codebook,recipes)
    # A constructed interpreter must itself be expressible as source input.
    encode_source(program,codebook)
    design=ReflectiveArchitecture(machine.digest,basis,recipes,codebook,program,representation_coordinate,'')
    from dataclasses import replace
    design=replace(design,digest=_hash(design.encode()))
    from .reflective_architecture_verify import verify_architecture
    if not verify_architecture(machine,design):
        raise ValueError('self-interpretation invariant could not be independently qualified')
    return design


def architecture_run(machine:EndogenousProcessor, design:ReflectiveArchitecture,
                     word:tuple[int,...], state:int, *, depth=2, fuel=20000000):
    """Run the constructed interpreter on an encoded program and its own code."""
    if type(depth) is not int or depth not in (0,1,2):
        raise ValueError('reflection depth must be 0,1,2')
    if (type(word) is not tuple or len(word)>24 or
            any(type(x) is not int or not 0<=x<machine.instruction_count for x in word) or
            type(state) is not int or state not in machine.states):
        raise ValueError('unqualified task intention')
    from .reflective_architecture_verify import verify_architecture
    if not verify_architecture(machine,design):
        raise ValueError('unqualified reflective architecture')
    program=compile_task_program(machine)
    args=(word,machine.states.index(state))
    ground=machine.operators
    baseline,base_steps=host_run(program,'main',args,ground,fuel=fuel)
    if depth==0:
        result,cost=baseline,base_steps
    else:
        encoded_task=encode_source(program,design.syntax_codebook)
        inner=(encoded_task,'main',args,ground)
        if depth==2:
            inner=(encode_source(design.interpreter,design.syntax_codebook),'run',inner,ground)
        result,cost=host_run(design.interpreter,'run',inner,ground,fuel=fuel,
                             seed_primitives=design.primitive_seed)
    expected=args[1]
    for op in word:
        expected=ground[op][expected]
    if result!=baseline or result!=expected:
        raise AssertionError('constructed interpreter did not realize intention')
    return {'schema':'navigable-compute.ncr002.execution.v0.1',
            'ground_sha256':machine.digest,'architecture_sha256':design.digest,
            'reflection_depth':depth,'word':list(word),
            'initial_state':state,'result_state':machine.states[result],
            'expression_steps':cost,'direct_expression_steps':base_steps,
            'claim':'bounded-generated-interpreter;explicit-external-seed;task-ground-verified'}
