"""NCR-001: a bounded, genuinely self-interpretable *task-derived* program.

The data semantics (finite transformation algebra) are constructed from the
intention's invariant Ground by NCI/NCS.  A small, explicitly declared host
language provides tuples, conditional choice and call frames.  A source-level
interpreter FOR THAT VERY LANGUAGE is encoded as an ordinary program in it.
There is no host 'eval', dynamic Python execution, or interpreter primitive.

This is bounded reflective interpretation, not universal computation, and the
host language / syntax remain an external, honestly accounted-for seed.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

from .self_space import EndogenousProcessor

SCHEMA = 'navigable-compute.reflective-ground.v0.1'
PRIMITIVES = frozenset({'eq', 'first', 'rest', 'cons', 'empty', 'at', 'task'})


def lit(value): return ('lit', value)
def arg(index): return ('arg', index)
def choice(test, yes, no): return ('if', test, yes, no)
def primitive(name, *inputs): return ('prim', name, tuple(inputs))
def call(name, *inputs): return ('call', name, tuple(inputs))


def _tuple_count(x, cap=300000):
    """Inspect syntax/quoted data without following cyclic object graphs."""
    count = 0
    stack = [x]
    # Flat bounds elsewhere preclude creating recursive tuples without unsafe
    # Python object manipulation; this defensive walk has a hard overall cap.
    while stack:
        obj = stack.pop()
        count += 1
        if count > cap:
            raise ValueError('reflective source exceeds bounded size')
        if type(obj) is tuple:
            if len(obj) > 8192:
                raise ValueError('tuple length exceeds source contract')
            stack.extend(obj)
        elif type(obj) not in (int, str, bool):
            raise ValueError('unsupported literal or instruction datum')
    return count


def _program_map(program):
    if type(program) is not tuple or not 1 <= len(program) <= 48:
        raise ValueError('program must be a bounded function tuple')
    functions = {}
    for fn in program:
        if type(fn) is not tuple or len(fn) != 2 or type(fn[0]) is not str:
            raise ValueError('malformed function description')
        if fn[0] in functions or not fn[0].isidentifier():
            raise ValueError('duplicate or invalid function')
        functions[fn[0]] = fn[1]
    return functions


def _primitive(name, values, table):
    if name == 'eq' and len(values) == 2:
        return type(values[0]) is type(values[1]) and values[0] == values[1]
    if name == 'empty' and len(values) == 1 and type(values[0]) is tuple:
        return not values[0]
    if name == 'first' and len(values) == 1 and type(values[0]) is tuple and values[0]:
        return values[0][0]
    if name == 'rest' and len(values) == 1 and type(values[0]) is tuple:
        return values[0][1:]
    if name == 'cons' and len(values) == 2 and type(values[1]) is tuple:
        return (values[0],) + values[1]
    if name == 'at' and len(values) == 2 and type(values[0]) is tuple and type(values[1]) is int and 0<=values[1]<len(values[0]):
        return values[0][values[1]]
    if name == 'task' and len(values) == 2 and all(type(x) is int for x in values) and 0<=values[0]<len(table) and 0<=values[1]<len(table[values[0]]):
        # No independent instruction semantics are postulated here.  This
        # table is read from the verified NCS-001 task-generated Ground.
        return table[values[0]][values[1]]
    raise ValueError('unqualified primitive call')


@dataclass
class StepBudget:
    limit: int
    used: int = 0

    def tick(self):
        self.used += 1
        if self.used > self.limit:
            raise ValueError('bounded interpretation fuel exhausted')


def host_run(program, entry, argv, table, *, fuel=5000000, seed_primitives=None):
    """Reference host semantics, deliberately lacking an 'eval' primitive."""
    if type(entry) is not str or type(argv) is not tuple or type(table) is not tuple:
        raise ValueError('invalid reflective input types')
    if type(fuel) is not int or not 1 <= fuel <= 20000000:
        raise ValueError('invalid reflection fuel')
    if seed_primitives is not None and (type(seed_primitives) is not tuple or any(x not in PRIMITIVES for x in seed_primitives)):
        raise ValueError('invalid physical primitive seed')
    defs = _program_map(program)
    budget = StepBudget(fuel)
    def evaluate(node, frame):
        budget.tick()
        if type(node) is not tuple or not node:
            raise ValueError('malformed expression')
        tag = node[0]
        if tag == 'lit' and len(node) == 2:
            return node[1]
        if tag == 'arg' and len(node) == 2 and type(node[1]) is int and 0<=node[1]<len(frame):
            return frame[node[1]]
        if tag == 'if' and len(node) == 4:
            cond = evaluate(node[1], frame)
            if type(cond) is not bool:
                raise ValueError('condition not Boolean')
            return evaluate(node[2] if cond else node[3],frame)
        if tag in ('prim', 'call') and len(node) == 3 and type(node[1]) is str and type(node[2]) is tuple:
            values = tuple(evaluate(v, frame) for v in node[2])
            if tag == 'prim':
                if seed_primitives is not None and node[1] not in seed_primitives:
                    raise ValueError('primitive absent from constructed host seed')
                return _primitive(node[1], values, table)
            if node[1] not in defs:
                raise ValueError('unknown function')
            return evaluate(defs[node[1]], values)
        raise ValueError('unqualified expression')
    if entry not in defs:
        raise ValueError('unknown entrypoint')
    try:
        result = evaluate(defs[entry],argv)
    except (IndexError,RecursionError) as exc:
        raise ValueError('invalid or nonterminating bounded source program') from exc
    return result, budget.used


def compile_task_program(machine: EndogenousProcessor):
    """Derive a task-specific program from its verified semantic operator table."""
    if not isinstance(machine, EndogenousProcessor):
        raise ValueError('requires an admitted finite semantic machine')
    # Executes a word of NCS *semantic opcode indices*; no x86-style ISA.
    # Input: (word, dense_state).  The task semantics are provided separately
    # as data at the Ground boundary and do not need to be embedded in code.
    w, state = arg(0), arg(1)
    recursive = choice(primitive('empty',w), state,
                       call('fold', primitive('rest',w),
                            primitive('task',primitive('first',w),state)))
    return (('fold',recursive), ('main',call('fold',w,state)))


def compile_reflector():
    """Construct a meta-interpreter written solely in its own interpreted AST.

    'run(program,entry,args)' interprets ANY well-formed program expressible
    by the declared grammar, including this very program.  Task semantics are
    passed as an immutable table, not captured as an external Python function.
    The interpreter contains no 'eval' opcode or escape to host Python.
    """
    z = lit(())
    # Program = tuple((function_name, expression), ...)
    defs, name = arg(0),arg(1)
    lookup = choice(primitive('eq',primitive('first',primitive('first',defs)),name),
                    primitive('at',primitive('first',defs),lit(1)),
                    call('lookup',primitive('rest',defs),name))
    # Arguments to a source function are recursively evaluated, in order.
    nodes, frame, program = arg(0),arg(1),arg(2)
    # Each source primitive has its semantics *implemented by instructions of
    # the same language*, not delegated to an eval/apply primitive.
    pname, values = arg(0),arg(1)
    def v(i):return primitive('at',values,lit(i))
    dispatch = primitive('at',z,lit(0))  # Fail closed on any unknown operation.
    operations = [
        ('task',primitive('at',primitive('at',arg(2),v(0)),v(1))),
        ('at',primitive('at',v(0),v(1))),
        ('cons',primitive('cons',v(0),v(1))),
        ('rest',primitive('rest',v(0))),
        ('first',primitive('first',v(0))),
        ('empty',primitive('empty',v(0))),
        ('eq',primitive('eq',v(0),v(1))),
    ]
    for op,implementation in operations:
        dispatch=choice(primitive('eq',pname,lit(op)),implementation,dispatch)
    # Interpreter for the five source syntactic forms.
    node, env, prog=arg(0),arg(1),arg(2)
    tag=primitive('first',node)
    evaluation=primitive('at',z,lit(0))
    variants=[
        ('call',call('invoke',prog,primitive('at',node,lit(1)),
                     call('evaluate_list',primitive('at',node,lit(2)),env,prog,arg(3)),arg(3))),
        ('prim',call('perform',primitive('at',node,lit(1)),
                     call('evaluate_list',primitive('at',node,lit(2)),env,prog,arg(3)),arg(3))),
        ('if',choice(call('evaluate',primitive('at',node,lit(1)),env,prog,arg(3)),
                    call('evaluate',primitive('at',node,lit(2)),env,prog,arg(3)),
                    call('evaluate',primitive('at',node,lit(3)),env,prog,arg(3)))),
        ('arg',primitive('at',env,primitive('at',node,lit(1)))),
        ('lit',primitive('at',node,lit(1))),
    ]
    for syntax,implementation in variants:
        evaluation=choice(primitive('eq',tag,lit(syntax)),implementation,evaluation)
    # Correctly thread one Ground table into every recursive eval call.
    eval_list = choice(primitive('empty',nodes),z,
                       primitive('cons',call('evaluate',primitive('first',nodes),frame,program,arg(3)),
                                 call('evaluate_list',primitive('rest',nodes),frame,program,arg(3))))
    return (
        ('lookup',lookup),
        ('evaluate_list',eval_list),
        ('perform',dispatch),
        ('evaluate',evaluation),
        ('invoke',call('evaluate',call('lookup',arg(0),arg(1)),arg(2),arg(0),arg(3))),
        ('run',call('invoke',arg(0),arg(1),arg(2),arg(3))),
    )


def task_table(machine: EndogenousProcessor) -> tuple[tuple[int,...],...]:
    return machine.operators


def reflect(machine: EndogenousProcessor, word: tuple[int,...], state: int,
            *, depth: int = 2, fuel: int = 10000000) -> dict:
    """Compare native task execution, one reflection and reflection of itself.

    depth=2: host(interpreter, interpreter_source, task_program(...))
    NOT host evaluator re-entering itself or a hardcoded EVAL opcode.
    """
    if not isinstance(machine,EndogenousProcessor) or type(depth) is not int or depth not in (0,1,2) or type(word) is not tuple:
        raise ValueError('reflection depth must be 0, 1 or 2')
    if len(word)>32 or any(type(op) is not int or not 0 <= op < machine.instruction_count for op in word):
        raise ValueError('invalid or oversized reflective source word')
    if state not in machine.states or type(state) is not int:
        raise ValueError('unqualified source state')
    from .self_verify import verify_endogenous_processor
    if not verify_endogenous_processor(machine.source.description,machine):
        raise ValueError('unverified task-generated processor')
    table=task_table(machine)
    start=machine.states.index(state)
    program=compile_task_program(machine)
    reflector=compile_reflector()
    _tuple_count(reflector)
    frame=(word,start)
    direct, direct_cost=host_run(program,'main',frame,table,fuel=fuel)
    if depth==0:
        result,cost=direct,direct_cost
    else:
        inner=(program,'main',frame,table)
        if depth==1:
            result,cost=host_run(reflector,'run',inner,table,fuel=fuel)
        else:
            result,cost=host_run(reflector,'run',
                  (reflector,'run',inner,table),table,fuel=fuel)
    if direct!=result:
        raise AssertionError('reflexive evaluation disagrees with direct task program')
    native=start
    for op in word:
        native=machine.operators[op][native]
    if native!=result:
        raise AssertionError('reflexive evaluation disagrees with source algebra')
    return {
        'schema':SCHEMA,
        'reflection_depth':depth,
        'source_operator_count':machine.instruction_count,
        'source_state_count':len(machine.states),
        'task_program_sha256':sha256(json.dumps(program,separators=(',',':')).encode()).hexdigest(),
        'interpreter_program_sha256':sha256(json.dumps(reflector,separators=(',',':')).encode()).hexdigest(),
        'source_ground_sha256':machine.digest,
        'word':list(word),
        'initial_state':state,
        'result_state':machine.states[result],
        'host_evaluation_steps':cost,
        'direct_host_steps':direct_cost,
        'claim':'same-language-finite-self-interpretation;finite-fuel;declared-host-seed'
    }
