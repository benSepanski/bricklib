"""Here are the AST nodes contained for a simple stencil language"""

from st.expr_meta import ExprMeta
import st.alop
from typing import List, Dict, Callable


def conv_expr(input):
    if isinstance(input, int):
        return IntLiteral(input)
    if isinstance(input, float):
        return FloatLiteral(input)
    if isinstance(input, complex):
        return ComplexLiteral(input)
    if isinstance(input, str):
        return ConstRef(input)
    if isinstance(input, Expr):
        return input
    raise ValueError(
        "Cannot convert to expression, {}".format(repr(input)))


class Expr(object, metaclass=ExprMeta):
    """Generic AST node

    Contains a list of children and forms a multiway tree.

    Attributes:
        children List[Node]: The list of children.
        scope (optional): The current scope additions.
        attr: record information attached to the node, can be initialized for all classes using _attr
    """
    _children = []
    _arg_sig = None
    _attr = dict()
    attr: Dict
    _COMPLEX_FLAG = "complex"  # key for complex flag in _attr

    def __init__(self, *args, **kwargs):
        bound = self._arg_sig.bind(*args, **kwargs)
        self.children = [None] * len(self._children)
        for name, val in bound.arguments.items():
            setattr(self, name, val)
        self.attr = dict(self._attr)
        self.parent = None

    def visit(self, init, func):
        """Preorder traversal"""
        init, recurse = func(init, self)
        if recurse:
            for child in self.children:
                init = child.visit(init, func)
        return init
    
    def is_complex(self) -> bool:
        """
        Return true if this expression may evaluate to a complex-typed
        expression
        """
        return self.get_attr(Expr._COMPLEX_FLAG) == True

    def mk_child(self, child):
        """ Make one node a child
            This does not append the child but rather fixes the parent-child
            relationship
        """
        child.parent = self

    def str_attr(self):
        """ Extra attributes that should be printed in str representation """
        return ""

    def get_attr(self, attr):
        if attr in self.attr:
            return self.attr[attr]
        return None

    def __str__(self):
        ret = "({} [{}] ".format(self.__class__.__name__, self.str_attr())
        for child in self.children:
            ret += str(child)
        ret += ")"
        return ret
    
    # Arithmetic operators
    def __add__(self, other):
        return BinOp(st.alop.BinaryOperators.Add,
                     self, conv_expr(other))

    def __radd__(self, other):
        return conv_expr(other).__add__(self)

    def __sub__(self, other):
        return BinOp(st.alop.BinaryOperators.Sub,
                     self, conv_expr(other))

    def __rsub__(self, other):
        return conv_expr(other).__sub__(self)

    def __mul__(self, other):
        return BinOp(st.alop.BinaryOperators.Mul,
                     self, conv_expr(other))

    def __rmul__(self, other):
        return conv_expr(other).__mul__(self)

    def __truediv__(self, other):
        return BinOp(st.alop.BinaryOperators.Div,
                     self, conv_expr(other))

    def __rtruediv__(self, other):
        return conv_expr(other).__truediv__(self)

    def __mod__(self, other):
        return BinOp(st.alop.BinaryOperators.Mod,
                     self, conv_expr(other))

    def __rmod__(self, other):
        return conv_expr(other).__mod__(self)

    def __and__(self, other):
        return BinOp(st.alop.BinaryOperators.BitAnd,
                     self, conv_expr(other))

    def __rand__(self, other):
        return conv_expr(other).__and__(self)

    def __xor__(self, other):
        return BinOp(st.alop.BinaryOperators.BitXor,
                     self, conv_expr(other))

    def __rxor__(self, other):
        return conv_expr(other).__xor__(self)

    def __or__(self, other):
        return BinOp(st.alop.BinaryOperators.BitOr,
                     self, conv_expr(other))

    def __ror__(self, other):
        return conv_expr(other).__or__(self)

    def __lshift__(self, other):
        return BinOp(st.alop.BinaryOperators.BitSHL,
                     self, conv_expr(other))

    def __rlshift__(self, other):
        return conv_expr(other).__lshift__(self)

    def __rshift__(self, other):
        return BinOp(st.alop.BinaryOperators.BitSHR,
                     self, conv_expr(other))

    def __rrshift__(self, other):
        return conv_expr(other).__rshift__(self)

    def __neg__(self):
        return UnOp(st.alop.UnaryOperators.Neg, self)

    # Comparison operators
    def __lt__(self, other):
        return BinOp(st.alop.BinaryOperators.Lt,
                     self, conv_expr(other))

    def __le__(self, other):
        return BinOp(st.alop.BinaryOperators.Leq,
                     self, conv_expr(other))

    def __eq__(self, other):
        return BinOp(st.alop.BinaryOperators.Eq,
                     self, conv_expr(other))

    def __ne__(self, other):
        return BinOp(st.alop.BinaryOperators.Neq,
                     self, conv_expr(other))

    def __gt__(self, other):
        return BinOp(st.alop.BinaryOperators.Gt,
                     self, conv_expr(other))

    def __ge__(self, other):
        return BinOp(st.alop.BinaryOperators.Geq,
                     self, conv_expr(other))

    # Logical operators
    def logical_and(self, other):
        return BinOp(st.alop.BinaryOperators.And,
                     self, conv_expr(other))

    def logical_or(self, other):
        return BinOp(st.alop.BinaryOperators.Or,
                     self, conv_expr(other))

    def logical_not(self):
        return UnOp(st.alop.UnaryOperators.Not, self)

    def __hash__(self):
        return id(self)


class Index(Expr):
    def __init__(self, n: int):
        super().__init__()
        self.n = n
        self._attr[Expr._COMPLEX_FLAG] = False

    def genericName(self):
        return "axis{}".format(self.n)

    def str_attr(self):
        """ Extra attributes that should be printed in str representation """
        return str(self.n)


class ReductionOp(Expr):
    def __init__(self, operator: st.alop.BinaryOperators, terms: List[Expr]):
        super().__init__()
        self.children = terms[:]
        self.op = operator
        assert all(isinstance(child, Expr) for child in self.children)
        self._attr[Expr._COMPLEX_FLAG] = any(child.is_complex() for child in self.children)


class If(Expr):
    _children = ['cnd', 'thn', 'els']
    cnd: Expr
    thn: Expr
    els: Expr

    def __init__(self, *pargs, **kwargs):
        super().__init__(*pargs, **kwargs)
        self._attr[Expr._COMPLEX_FLAG] = self.thn.is_complex() or self.els.is_complex()


class BinOp(Expr):
    _children = ['lhs', 'rhs']
    lhs: Expr
    rhs: Expr

    def __init__(self, operator: st.alop.BinaryOperators = None,
                 *pargs, **kwargs):
        super().__init__(*pargs, **kwargs)
        self.operator = operator
        # Handle complex types
        has_complex_operand = self.lhs.is_complex() or self.rhs.is_complex()
        if has_complex_operand and operator not in (st.alop.BinaryOperators.Add,
                                                    st.alop.BinaryOperators.Sub,
                                                    st.alop.BinaryOperators.Mul,
                                                    st.alop.BinaryOperators.Div,
                                                    st.alop.BinaryOperators.Assign,
                                                    st.alop.BinaryOperators.Eq,
                                                    st.alop.BinaryOperators.Neq,
                                                    ):
            raise ValueError(f"Cannot perform operator {operator} on complex types.")
        if has_complex_operand and operator == st.alop.BinaryOperators.Assign:
            if not self.lhs.is_complex():
                raise ValueError(f"Cannot assign complex-type to real-typed value {self.lhs}")
        
        is_complex = has_complex_operand and operator not in \
            (st.alop.BinaryOperators.Eq, st.alop.BinaryOperators.Neq)
        self._attr[Expr._COMPLEX_FLAG] = is_complex

    def str_attr(self):
        """ Extra attributes that should be printed in str representation """
        return str(self.operator)


class UnOp(Expr):
    _children = ['subexpr']
    subexpr: Expr

    def __init__(self, operator: st.alop.UnaryOperators = None,
                 *pargs, **kwargs):
        super().__init__(*pargs, **kwargs)
        self.operator = operator
        if self.subexpr.is_complex() and operator not in \
                (st.alop.UnaryOperators.Neg,
                 st.alop.UnaryOperators.Pos,
                 st.alop.UnaryOperators.Inc,
                 st.alop.UnaryOperators.Dec,
                 ):
            raise ValueError(f"Cannot apply operator {operator} to complex types")
        self._attr[Expr._COMPLEX_FLAG] = self.subexpr.is_complex()

    def str_attr(self):
        """ Extra attributes that should be printed in str representation """
        return str(self.operator)


class FunctionOfLocalVectorIndex(Expr):
    def __init__(self, operator: Callable[..., str], *args, **kwargs):
        """
        User-defined functions which take as input
        string-expressions of the indices currently being read from
        and output a string

        For example, if a brick is shaped KxJxI = 2x2x2 with vectors
        of shape KxJxI = 1x2x2,
        a function(i, j, k) would be invoked with values i, j, k = ("0", "0", "0")
        and ("0", "0", "1")

        The strings may contain variables in the case of loops

        :param operator: A function to apply local vector indices
        :param args: the indices in question
        :param kwargs:
            :kwarg: complex_valued either true or false
        """
        super().__init__()
        terms = []
        if len(args) > 0:
            terms += args[0]
        if not all([isinstance(term, st.expr.Index) for term in terms]):
            bad_types = set(filter(lambda x: not isinstance(x, st.expr.Index), terms))
            raise TypeError(f"All parameters to local index function must be of type Index, not {bad_types}")
        self.children = terms[:]
        self.op = operator
        complex_valued = kwargs.get("complex_valued", False)
        self._attr[Expr._COMPLEX_FLAG] = complex_valued
        self.tile_dims = None
        self.fold = None

    def record_tile_dims(self, tile_dims):
        self.tile_dims = tile_dims

    def record_fold(self, fold):
        self.fold = fold


class IntLiteral(Expr):
    _attr = {'num_literal': True,
             'num_const': True,
             'atomic': True,
             Expr._COMPLEX_FLAG: False,
             }

    def __init__(self, v: int):
        super().__init__()
        self.val = v

    def __int__(self):
        return self.val

    def __float__(self):
        return float(int(self))
    
    def __complex__(self):
        return complex(int(self))


class FloatLiteral(Expr):
    _attr = {'num_literal': True,
             'num_const': True,
             'atomic': True,
             Expr._COMPLEX_FLAG: False,
             }

    def __init__(self, v: float):
        super().__init__()
        self.val = v

    def __float__(self):
        return self.val
    
    def __complex__(self):
        return complex(float(self))

class ComplexLiteral(Expr):
    _attr = {'num_literal': True,
             'num_const': True,
             'atomic': True,
             Expr._COMPLEX_FLAG: True,
             }

    def __init__(self, v: complex):
        super().__init__()
        self.val = v
    
    def __complex__(self):
        return self.val

class ConstRef(Expr):
    _attr = {'num_const': True,
             'atomic': True,
             }

    def __init__(self, v: str, complex_valued: bool = False):
        super().__init__()
        self.val = v
        self._attr[Expr._COMPLEX_FLAG] = complex_valued

    def str_attr(self):
        return self.val
