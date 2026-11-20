"""NCR-002 independent receiver: audit synthesized reflective architectures.

A separate iterative evaluator executes the generated interpreter; a second
enumerator checks minimal seed cardinality within the *declared* finite macro
search space.  Behavioral checks are bounded, not a universal kernel theorem.
"""
from __future__ import annotations
from hashlib import sha256
from itertools import combinations
from math import factorial
import json

from .reflective_verify import reference_run
from .reflective_ground import (arg, lit, primitive, call, choice,
                                compile_task_program)
from .self_verify import verify_endogenous_processor
from .self_space import EndogenousProcessor
from .reflective_architecture import (ReflectiveArchitecture, FORMS,
                                      OPERATIONS, RECIPES, encode_source)


def _canonical(obj):
    return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode('ascii')


def _independent_minimal_seed():
    # This verifier does NOT call the producer's _select_basis.  Each declared
    # recipe lists its physical prerequisites and only those are considered.
    primitive_names=tuple(sorted(OPERATIONS))
    for width in range(8):
        solutions=[]
        for basis in combinations(primitive_names,width):
            if all(any(set(needs)<=set(basis) for _,needs in RECIPES[op]) for op in OPERATIONS):
                solutions.append(basis)
        if solutions:
            return min(solutions)
    raise AssertionError('semantics grammar has no solution')


def _host_primitive_names(program):
    used=[]
    def visit(node,depth=0):
        if depth>128 or type(node) is not tuple or not node:
            raise ValueError('malformed source expression')
        if node[0]=='lit' and len(node)==2: return
        if node[0]=='arg' and len(node)==2 and type(node[1]) is int: return
        if node[0]=='if' and len(node)==4:
            for child in node[1:]:visit(child,depth+1)
            return
        if node[0] in ('prim','call') and len(node)==3 and type(node[1]) is str and type(node[2]) is tuple:
            if node[0]=='prim':used.append(node[1])
            for child in node[2]:visit(child,depth+1)
            return
        raise ValueError('bad generated AST')
    for name,expr in program:
        if type(name) is not str:raise ValueError('bad function name')
        visit(expr)
    return frozenset(used)


def verify_architecture(machine:EndogenousProcessor, design:ReflectiveArchitecture)->bool:
    """Check the delivered architecture without calling its constructor."""
    try:
        if not isinstance(machine,EndogenousProcessor) or not isinstance(design,ReflectiveArchitecture):
            return False
        if not verify_endogenous_processor(machine.source.description,machine):
            return False
        if design.ground_sha256!=machine.digest or design.primitive_seed!=_independent_minimal_seed():
            return False
        if type(design.digest) is not str or len(design.digest)!=64 or design.digest!=sha256(_canonical(design.encode())).hexdigest():
            return False
        if type(design.representation_coordinate) is not int or not 0<=design.representation_coordinate<factorial(len(FORMS)):
            return False
        available=sorted(FORMS,key=lambda tag:sha256((machine.digest+':'+tag).encode()).hexdigest())
        position=design.representation_coordinate
        observed_order=[]
        for count in range(len(available),0,-1):
            digit,position=divmod(position,factorial(count-1))
            observed_order.append(available.pop(digit))
        if design.syntax_codebook!=tuple((tag,i) for i,tag in enumerate(observed_order)):
            return False
        if tuple(op for op,_ in design.recipes)!=OPERATIONS:
            return False
        for op,selected in design.recipes:
            eligible=[name for name,requires in RECIPES[op] if set(requires)<=set(design.primitive_seed)]
            if selected!=eligible[0]: return False
        if (type(design.interpreter) is not tuple or
            tuple(name for name,_ in design.interpreter)!=('lookup','evaluate_list','perform','evaluate','invoke','run')):
            return False
        if not _host_primitive_names(design.interpreter)<=set(design.primitive_seed):
            return False
        encoded_self=encode_source(design.interpreter,design.syntax_codebook)
        table=machine.operators

        # Exact, independently implemented source primitive semantics on the
        # provided typed *samples*.  Invalid-input equivalence isn't asserted.
        cases=[
            ('eq',(1,1)),('eq',(1,2)),('eq',(True,1)),('eq',((1,2),(1,2))),
            ('cons',('a',())),('cons',(7,(8,9))),
            ('empty',((),)),('empty',((1,2),)),
            ('first',((3,4),)),('first',(('abc',),)),
            ('rest',((3,4),)),('rest',((1,),)),('rest',((),)),
            ('at',((4,5,6),0)),('at',((4,5,6),2)),
            ('task',(0,0)),('task',(len(table)-1,len(table[0])-1)),
        ]
        for opname,operands in cases:
            expected=reference_run((('main',primitive(opname,*(lit(x) for x in operands))),),'main',(),table)[0]
            got=reference_run(design.interpreter,'perform',(opname,operands,table),table,
                              seed_primitives=design.primitive_seed)[0]
            if got!=expected or type(got)!=type(expected): return False
        # Test AST semantics, not only the primitive dispatch.  The programs
        # exercise literal, address, conditional, calls, lists and tasks.
        checks=[
            ((('main',lit(('literal',2))),),'main',()),
            ((('id',arg(0)),('main',call('id',arg(0)))),'main',(47,)),
            ((('main',choice(primitive('eq',arg(0),lit(2)),lit('yes'),lit('no'))),),'main',(2,)),
            ((('main',choice(primitive('eq',arg(0),lit(2)),lit('yes'),lit('no'))),),'main',(3,)),
            ((('main',primitive('cons',lit(1),lit((2,3)))),),'main',()),
            ((('main',primitive('task',lit(0),lit(0))),),'main',()),
            ((('main',primitive('empty',lit(()))),),'main',()),
        ]
        for program,entry,argv in checks:
            original=reference_run(program,entry,argv,table)[0]
            interpreted=reference_run(design.interpreter,'run',
                 (encode_source(program,design.syntax_codebook),entry,argv,table),table,
                 seed_primitives=design.primitive_seed)[0]
            if original!=interpreted or type(original)!=type(interpreted): return False
        # Test genuine *second-level* interpretation of the generated source.
        task=compile_task_program(machine)
        word=(0,1) if machine.instruction_count>1 else (0,)
        initial=(word,0)
        inner=(encode_source(task,design.syntax_codebook),'main',initial,table)
        self_interpreted=reference_run(design.interpreter,'run',
                    (encoded_self,'run',inner,table),table,
                    fuel=20000000,seed_primitives=design.primitive_seed)[0]
        direct=reference_run(task,'main',initial,table)[0]
        if direct!=self_interpreted: return False
        return True
    except (ValueError,TypeError,KeyError,IndexError,RecursionError,
            MemoryError,OverflowError,AttributeError,AssertionError):
        return False


def verify_architecture_execution(machine,design,record,*,fuel=20000000):
    """Replay a transmitted execution receipt using the independent evaluator."""
    try:
        if type(record) is not dict or not verify_architecture(machine,design):return False
        expected={'schema','ground_sha256','architecture_sha256','reflection_depth',
                  'word','initial_state','result_state','expression_steps',
                  'direct_expression_steps','claim'}
        if set(record)!=expected or record['schema']!='navigable-compute.ncr002.execution.v0.1':return False
        if record['ground_sha256']!=machine.digest or record['architecture_sha256']!=design.digest:return False
        if record['claim']!='bounded-generated-interpreter;explicit-external-seed;task-ground-verified':return False
        depth=record['reflection_depth'];word=record['word'];state=record['initial_state']
        if type(depth) is not int or depth not in (0,1,2) or type(word) is not list or len(word)>24:return False
        if any(type(x) is not int or not 0<=x<machine.instruction_count for x in word):return False
        if type(state) is not int or state not in machine.states:return False
        if type(record['expression_steps']) is not int or type(record['direct_expression_steps']) is not int:return False
        table=machine.operators
        task=compile_task_program(machine)
        args=(tuple(word),machine.states.index(state))
        direct,direct_steps=reference_run(task,'main',args,table,fuel=fuel)
        if depth==0:got,steps=direct,direct_steps
        else:
            inputs=(encode_source(task,design.syntax_codebook),'main',args,table)
            if depth==2:inputs=(encode_source(design.interpreter,design.syntax_codebook),'run',inputs,table)
            got,steps=reference_run(design.interpreter,'run',inputs,table,fuel=fuel,
                                    seed_primitives=design.primitive_seed)
        native=args[1]
        for op in word:native=table[op][native]
        return (got==direct==native and machine.states[got]==record['result_state']
                and steps==record['expression_steps'] and direct_steps==record['direct_expression_steps'])
    except (ValueError,TypeError,KeyError,IndexError,RecursionError,MemoryError,OverflowError,AttributeError):
        return False
