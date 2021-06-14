from st.expr import Expr
from typing import List, Tuple
from st.grid import Grid


class Buffer:
    iteration: List[Tuple[int, int]]
    rhs: Expr
    name: str
    grid: Grid

    def __init__(self, rhs):
        self.rhs = rhs
        self.depends = list()
        self.iteration = list()
        self.name = None
        self.grid = None

    def ref_name(self):
        return self.name
    
    def is_complex(self) -> bool:
        if self.grid is not None:
            if self.rhs.is_complex() and not self.grid.is_complex():
                raise ValueError("Cannot assign complex rhs to real-valued grid")
            return self.grid.is_complex()
        return self.rhs.is_complex()


class BufferRead(Expr):
    buf: Buffer

    def __init__(self, buf):
        super().__init__()
        self.buf = buf


class Shift(Expr):
    _children = ['subexpr']
    subexpr: Expr
    shifts: List[int]

    def __init__(self, shifts, *pargs, **kwargs):
        super().__init__(*pargs, **kwargs)
        self.shifts = shifts[:]
