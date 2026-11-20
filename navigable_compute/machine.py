"""Reference byte-machine execution, deliberately separate from Ground algebra."""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import json

OPS = {'xor', 'xor_const', 'rotl', 'swap', 'copy', 'and_const', 'set', 'add_const'}
BINARY = {'xor', 'swap', 'copy'}
AFFINE = OPS - {'add_const'}

class BudgetExceeded(ValueError):
    pass

@dataclass(frozen=True)
class Instruction:
    op: str
    a: int
    b: int = 0

    def __post_init__(self):
        if self.op not in OPS: raise ValueError(f'unknown opcode: {self.op}')
        if not isinstance(self.a, int) or not isinstance(self.b, int): raise TypeError('operands must be integers')

@dataclass(frozen=True)
class Program:
    size: int
    instructions: tuple[Instruction, ...]

    def __post_init__(self):
        if not isinstance(self.size, int) or not 1 <= self.size <= 128:
            raise ValueError('memory size must be 1..128 bytes')
        if not isinstance(self.instructions, tuple) or not self.instructions:
            raise ValueError('at least one instruction required')
        for instruction in self.instructions:
            if not isinstance(instruction, Instruction): raise TypeError('invalid instruction')
            if not 0 <= instruction.a < self.size: raise ValueError('destination out of range')
            bound = self.size if instruction.op in BINARY else 256
            if not 0 <= instruction.b < bound: raise ValueError('operand out of range')

    def encode(self):
        return {'size': self.size, 'instructions': [[i.op, i.a, i.b] for i in self.instructions]}

    @property
    def digest(self):
        return sha256(json.dumps(self.encode(), sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    @classmethod
    def decode(cls, obj):
        return cls(int(obj['size']), tuple(Instruction(*ins) for ins in obj['instructions']))


def execute(program: Program, initial: bytes, cycles: int, *, budget: int = 1_000_000) -> bytes:
    if len(initial) != program.size: raise ValueError('initial memory width mismatch')
    if not isinstance(cycles, int) or cycles < 0: raise ValueError('negative or noninteger cycle count')
    if cycles * len(program.instructions) > budget:
        raise BudgetExceeded('sequential instruction budget exceeded; not a fast-forward')
    state = bytearray(initial)
    for _ in range(cycles):
        for ins in program.instructions:
            a, b = ins.a, ins.b
            if ins.op == 'xor': state[a] ^= state[b]
            elif ins.op == 'xor_const': state[a] ^= b
            elif ins.op == 'rotl':
                k = b % 8
                state[a] = ((state[a] << k) | (state[a] >> ((8-k) % 8))) & 255
            elif ins.op == 'swap': state[a], state[b] = state[b], state[a]
            elif ins.op == 'copy': state[a] = state[b]
            elif ins.op == 'and_const': state[a] &= b
            elif ins.op == 'set': state[a] = b
            elif ins.op == 'add_const': state[a] = (state[a] + b) & 255
            else: raise AssertionError('unreachable opcode')
    return bytes(state)

