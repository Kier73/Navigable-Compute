"""NCR-001 independent *iterative* evaluator and receipt auditor.

This is deliberately not an alias of reflective_ground.host_run(): expression
reduction uses an explicit continuation machine, rather than Python recursion.
Every 'eval' step and every host primitive has its own executable definition.
"""
from __future__ import annotations

from hashlib import sha256
import json

from .self_space import EndogenousProcessor
from .self_verify import verify_endogenous_processor
from .reflective_ground import (compile_reflector, compile_task_program,
                                SCHEMA)


def reference_run(program, entry, argv, table, *, fuel=10000000):
    if type(program) is not tuple or not 1 <= len(program) <= 48 or type(entry) is not str:
        raise ValueError('invalid program')
    if type(argv) is not tuple or type(table) is not tuple:
        raise ValueError('bad machine values')
    if type(fuel) is not int or not 1 <= fuel <= 20000000:
        raise ValueError('bad fuel')
    functions = {}
    for row in program:
        if type(row) is not tuple or len(row) != 2 or type(row[0]) is not str or row[0] in functions:
            raise ValueError('bad function table')
        functions[row[0]]=row[1]
    if entry not in functions:
        raise ValueError('missing entry')
    # Evaluation work, results are separate stacks. A continuation is either
    # another expression, a lazy conditional, or a finished call/primitive.
    stack=[('expression',functions[entry],argv)]
    results=[]
    count=0
    while stack:
        work=stack.pop()
        if work[0]=='expression':
            count+=1
            if count>fuel:
                raise ValueError('fuel exhausted')
            node,frame=work[1],work[2]
            if type(node) is not tuple or not node:
                raise ValueError('invalid expression')
            if node[0]=='lit' and len(node)==2:
                results.append(node[1]); continue
            if node[0]=='arg' and len(node)==2 and type(node[1]) is int:
                if not 0 <= node[1] < len(frame):
                    raise ValueError('bad argument address')
                results.append(frame[node[1]]);continue
            if node[0]=='if' and len(node)==4:
                stack.append(('conditional',node[2],node[3],frame))
                stack.append(('expression',node[1],frame))
                continue
            if node[0] in ('prim','call') and len(node)==3 and type(node[1]) is str and type(node[2]) is tuple:
                stack.append(('application',node[0],node[1],len(node[2])))
                for subexpression in reversed(node[2]):
                    stack.append(('expression',subexpression,frame))
                continue
            raise ValueError('malformed instruction')
        elif work[0]=='conditional':
            verdict=results.pop()
            if type(verdict) is not bool:
                raise ValueError('non-Boolean condition')
            stack.append(('expression',work[1] if verdict else work[2],work[3]))
        elif work[0]=='application':
            kind,name,n=work[1],work[2],work[3]
            if n>len(results):
                raise ValueError('stack underflow')
            args=tuple(results[-n:]) if n else ()
            if n:del results[-n:]
            if kind=='call':
                if name not in functions:
                    raise ValueError('unknown called function')
                stack.append(('expression',functions[name],args))
                continue
            if name=='eq' and len(args)==2:
                value=type(args[0]) is type(args[1]) and args[0]==args[1]
            elif name=='empty' and len(args)==1 and type(args[0]) is tuple:
                value=len(args[0])==0
            elif name=='first' and len(args)==1 and type(args[0]) is tuple and args[0]:
                value=args[0][0]
            elif name=='rest' and len(args)==1 and type(args[0]) is tuple:
                value=args[0][1:]
            elif name=='cons' and len(args)==2 and type(args[1]) is tuple:
                value=(args[0],)+args[1]
            elif name=='at' and len(args)==2 and type(args[0]) is tuple and type(args[1]) is int and 0<=args[1]<len(args[0]):
                value=args[0][args[1]]
            elif name=='task' and len(args)==2 and all(type(x) is int for x in args) and 0<=args[0]<len(table) and 0<=args[1]<len(table[args[0]]):
                value=table[args[0]][args[1]]
            else:
                raise ValueError('unqualified primitive instruction')
            results.append(value)
        else:
            raise AssertionError('bad internal continuation')
    if len(results)!=1:
        raise ValueError('incomplete expression reduction')
    return results[0],count


def _hash(value):
    return sha256(json.dumps(value,separators=(',',':'),ensure_ascii=True).encode('ascii')).hexdigest()


def verify_reflection(machine: EndogenousProcessor, receipt:dict, *, fuel=10000000)->bool:
    """Re-derive both interpreter levels and compare with source semantics.

    A successful audit is evidence for THIS finite program and inputs, not a
    kernel-level proof that every well-formed program self-interprets.
    """
    try:
        if not isinstance(machine,EndogenousProcessor) or type(receipt) is not dict:
            return False
        if not verify_endogenous_processor(machine.source.description,machine):
            return False
        expected={'schema','reflection_depth','source_operator_count','source_state_count',
                  'task_program_sha256','interpreter_program_sha256','source_ground_sha256',
                  'word','initial_state','result_state','host_evaluation_steps',
                  'direct_host_steps','claim'}
        if set(receipt)!=expected or receipt['schema']!=SCHEMA or receipt['source_ground_sha256']!=machine.digest:
            return False
        depth=receipt['reflection_depth']
        if type(depth) is not int or depth not in (0,1,2):
            return False
        word=receipt['word']
        state=receipt['initial_state']
        if (type(word) is not list or len(word)>32 or any(type(x) is not int or not 0<=x<machine.instruction_count for x in word)
                or type(state) is not int or state not in machine.states):
            return False
        if (type(receipt['host_evaluation_steps']) is not int or
              type(receipt['direct_host_steps']) is not int):
            return False
        program=compile_task_program(machine)
        interpreter=compile_reflector()
        if (receipt['task_program_sha256']!=_hash(program) or
                receipt['interpreter_program_sha256']!=_hash(interpreter) or
                receipt['source_operator_count']!=machine.instruction_count or
                receipt['source_state_count']!=len(machine.states) or
                receipt['claim']!='same-language-finite-self-interpretation;finite-fuel;declared-host-seed'):
            return False
        table=machine.operators
        args=(tuple(word),machine.states.index(state))
        baseline,base_steps=reference_run(program,'main',args,table,fuel=fuel)
        if base_steps!=receipt['direct_host_steps']:
            return False
        if depth==0:
            result,used=baseline,base_steps
        elif depth==1:
            result,used=reference_run(interpreter,'run',
                      (program,'main',args,table),table,fuel=fuel)
        else:
            result,used=reference_run(interpreter,'run',
                      (interpreter,'run',(program,'main',args,table),table),table,fuel=fuel)
        source_value=args[1]
        for opcode in word:
            source_value=machine.operators[opcode][source_value]
        return (result==baseline==source_value and receipt['result_state']==machine.states[result]
                and receipt['host_evaluation_steps']==used)
    except (ValueError,TypeError,IndexError,KeyError,RecursionError,OverflowError,MemoryError):
        return False
