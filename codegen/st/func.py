from st.expr import Expr, conv_expr


class Func:
    def __init__(self, name: str, arity: int, complex_valued: bool = False):
        self.name = name
        self.arity = arity
        self.complex_valued = complex_valued

    def __call__(self, *args, **kwargs):
        if len(args) != self.arity:
            raise ValueError("Func {} passed wrong number of arguments".format(self.name))
        return CallExpr(self, *args)


class CallExpr(Expr):
    def __init__(self, func: Func, *args):
        super().__init__()
        self.callee = func
        self.children = [conv_expr(a) for a in args]
        self._attr[self._COMPLEX_FLAG] = func.complex_valued

    def str_attr(self):
        return self.callee.name