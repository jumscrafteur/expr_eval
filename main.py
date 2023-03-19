import math
from dataclasses import dataclass
from numbers import Number
from typing import Union

import matplotlib.pyplot as plt


@dataclass
class BinExpr:
    a: object
    b: object


Expr = Union[BinExpr, int, float, str]


class Call(BinExpr):
    a: str
    funcMap = {"ln": math.log}

    def sem(self, a: str, b: Number) -> Number:
        try:
            if a in dir(math):
                return getattr(math, a)(b)
            elif a in self.funcMap.keys():
                return self.funcMap[a](b)
            else:
                raise ValueError(a)
        except ValueError:
            return math.nan

    pass


class BinOp(BinExpr):
    pass


class Add(BinOp):
    sym = "+"

    def sem(self, a: Number, b: Number) -> Number:
        return a + b


class Sub(BinOp):
    sym = "-"

    def sem(self, a: Number, b: Number) -> Number:
        return a - b


class Mul(BinOp):
    sym = "*"

    def sem(self, a: Number, b: Number) -> Number:
        return a * b


class Div(BinOp):
    sym = "/"

    def sem(self, a: Number, b: Number) -> Number:
        return float("inf") if b == 0 else a / b


class Pow(BinOp):
    sym = "^"

    def sem(self, a: Number, b: Number) -> Number:
        return a**b


@dataclass
class UnOp:
    a: object


def is_expr(el):
    return type(el) in [Call, Add, Sub, Mul, Div, Pow]


def estr(expr: Expr):
    match expr:
        case str() | Number():
            return expr
        case BinOp(a, b):
            return f"({estr(expr.a)}{expr.sym}{estr(expr.b)})"
        case Call(a, b):
            return f"{expr.a}({estr(expr.b)})"
        case _:
            raise ValueError(expr)


assert estr(Call("ln", "x")) == "ln(x)", f'{estr(Call("ln", "x"))} != "ln(x)"'
assert estr(Add("x", "y")) == "(x+y)", f'{estr(Add("x", "y"))} != "(x+y)"'
assert estr(Sub("x", "y")) == "(x-y)", f'{estr(Sub("x", "y"))} != "(x-y)"'
assert estr(Mul("x", "y")) == "(x*y)", f'{estr(Mul("x", "y"))} != "(x*y)"'
assert estr(Div("x", "y")) == "(x/y)", f'{estr(Div("x", "y"))} != "(x/y)"'
assert estr(Pow("x", "y")) == "(x^y)", f'{estr(Pow("x", "y"))} != "(x^y)"'


def eval(expr: Expr, var: str = "x", val: Number = 0):
    match expr:
        case Number():
            return expr
        case Call(a, b):
            return expr.sem(a, eval(b, var, val))
        case BinExpr(a, b):
            return expr.sem(eval(a, var, val), eval(b, var, val))
        case str():
            if expr == var:
                return val
            else:
                raise ValueError(expr)
        case _:
            raise ValueError(expr)


assert eval(Call("ln", "x"), "x", 1) == 0
assert eval(Add("x", 0), "x", 1) == 1
assert eval(Sub("x", 0), "x", 1) == 1
assert eval(Sub("x", 1), "x", 0) == -1
assert eval(Mul("x", 0), "x", 1) == 0
assert eval(Div("x", 0), "x", 1) == float("inf")
assert eval(Div("x", 2), "x", 1) == 0.5
assert eval(Pow("x", 2), "x", 3) == 9


def Rep(expr1, var, expr2):

    match expr1:
        case str():
            return expr2 if expr1 == var else expr1
        case Number():
            return expr1
        case BinExpr(a, b):
            expr1.a = Rep(a, var, expr2)
            expr1.b = Rep(b, var, expr2)
            return expr1


def D(expr: Expr, var: str = "x"):
    match expr:
        case Number():
            return 0
        case str():
            return 1 if expr == var else 0
        case Sub(a, b):
            return Sub(D(a, var), D(b, var))
        case Add(a, b):
            return Add(D(a, var), D(b, var))
        case Mul(a, b):
            return Add(Mul(D(a, var), b), Mul(a, D(b, var)))
        case Div(a, b):
            return Div(Sub(Mul(D(a, var), b), Mul(a, D(b, var))), Pow(b, 2))
        case Pow(a, b):
            return Mul(b, Mul(D(a, var), Pow(a, b - 1)))
        case Call(a, b):
            match b:
                case str():
                    if b != var:
                        raise ValueError(b)
                    match a:
                        case "ln":
                            return Div(1, var)
                        case "cos":
                            return Mul(-1, Call("sin", var))
                        case "sin":
                            return Call("cos", var)
                case BinExpr():
                    return Mul(Rep(D(Call(a, "x")), "x", b), D(b, var))


# Check Const
assert eval(D(6)) == 0

# Check
assert eval(D("x")) == 1

# Check f+g
assert eval(D(Add("x", 2))) == 1

# Check f-g
assert eval(D(Sub("x", 2))) == 1

# Check f*g
assert eval(D(Mul("x", Sub("x", 3)))) == -3

# Check f*g
assert eval(D(Div("x", Sub("x", 3)))) == -1 / 3

# Check x^n et u^n
assert eval(D(Pow("x", 2)), val=3) == 6
assert eval(D(Pow(Sub("x", Pow("x", 2)), 2)), val=1) == 0

# Check ln(x)
assert eval(D(Call("ln", "x")), val=2) == 0.5


assert eval(D(Call("ln", Pow("x", 2))), val=2) == 1

assert eval(D(Call("cos", Pow("x", 2)))) == 0


def fixpoint(f, e):
    if f(e) == e:
        return e
    else:
        return fixpoint(f, f(e))


def simp(e: Expr) -> Expr:
    def z(e):
        match e:
            case BinExpr(int(), int()):
                return eval(e)
            case Add(0, e) | Add(e, 0):
                return e
            case Add(Mul() as a, Mul() as b) if a.a == b.a:
                return Mul(simp(Add(a.b, b.b)), a.a)
            case Add(Mul() as a, Mul() as b) if a.b == b.a:
                return Mul(simp(Add(a.a, b.b)), a.b)
            case Add(Mul() as a, Mul() as b) if a.a == b.b:
                return Mul(simp(Add(a.b, b.a)), a.a)
            case Add(Mul() as a, Mul() as b) if a.b == b.b:
                return Mul(simp(Add(a.a, b.a)), a.b)
            case Add(a, b) if a == b:
                return Mul(2, simp(a))
            # case Sub(a, b): return Add(a, Mul(-1, b))
            case Sub(0, a):
                return Mul(-1, b)
            case Sub(a, -0):
                return a
            case Mul(0, e) | Mul(e, 0):
                return 0
            case Mul(1, e) | Mul(e, 1):
                return e
            case Mul(int() as a, Add() as b):
                return Add(simp(Mul(a, b.a)), simp(Mul(a, b.b)))
            case Mul(Add() as a, int() as b):
                return Add(simp(Mul(b, a.a)), simp(Mul(b, a.b)))
            case Mul(int() as a, Mul(int(), _) as b):
                return Mul(simp(Mul(a, b.a)), simp(b.b))
            case Mul(int() as a, Mul(_, int()) as b):
                return Mul(simp(Mul(a, b.a)), simp(b.b))
            case Mul(Mul(int(), _) as a, int() as b):
                return Mul(simp(Mul(b, a.a)), simp(a.b))
            case Mul(Mul(_, int()) as a, int() as b):
                return Mul(simp(Mul(b, a.a)), simp(a.b))
            case Mul(a, b) if a == b:
                return Pow(simp(a), 2)
            case Pow(e, 1):
                return e
            case Pow(0, _):
                return 1
            case BinExpr(a, b):
                return e.__class__(a=simp(a), b=simp(b))
            case _:
                return e

    return fixpoint(z, e)


# e + 0 == 0
assert estr(simp(Add("x", 0))) == "x", f'{estr(simp(Add("x", 0)))} != "x"'

# e+e = 2e
assert estr(simp(Add("x", "x"))) == "(2*x)", f'{estr(simp(Add("x", "x")))} != "(2*x)"'

# 0*e = 0
assert estr(simp(Mul("x", 0))) == 0, f'{estr(simp(Mul("x", 0)))} != 0'

# 1*e = e
assert estr(simp(Mul("x", 1))) == "x", f'{estr(simp(Mul("x", 1)))} != "x"'

# factorisation
assert (
    estr(simp(Add(Mul(3, "x"), Mul(4, "x")))) == "(7*x)"
), f'{estr(simp(Add(Mul(3, "x"), Mul(4, "x"))))} != "(7*x)"'

# développement
assert (
    estr(simp(Mul(3, Add(4, "x")))) == "(12+(3*x))"
), f'{estr(simp(Mul(3, Add(4, "x"))))} != "(12+(3*x))"'
assert (
    estr(simp(Mul(3, Mul(4, "x")))) == "(12*x)"
), f'{estr(simp(Mul(3, Mul(4, "x"))))} != "(12*x)"'
assert (
    estr(simp(Mul(3, Mul("x", 4)))) == "(12*x)"
), f'{estr(simp(Mul(3, Mul("x", 4))))} != "(12*x)"'
assert (
    estr(simp(Mul(Mul("x", 4), 3))) == "(12*x)"
), f'{estr(simp(Mul(Mul("x", 4), 3)))} != "(12*x)"'

assert (
    estr(simp(D(Mul(Call("ln", "x"), Add(Mul(3, Pow("x", 2)), 1)))))
    == "(((1/x)*((3*(x^2))+1))+(ln(x)*(6*x)))"
), f'{estr(simp(D(Mul(Call("ln", "x"), Add(Mul(3, Pow("x", 2)), 1)))))} != (((1/x)*((3*(x^2))+1))+(ln(x)*(6*x)))'


def dis(op, s, o):
    match o:
        case Number():
            return F(op(s.expr, o))
        case F():
            return F(op(s.expr, o.expr))
        case _:
            raise ValueError(o)


def rdis(op, s, o):
    match o:
        case Number():
            return F(op(o, s.expr))
        case F():
            return F(op(o.expr, s.expr))
        case _:
            raise ValueError(o)


class F:
    def __init__(self, expr):
        self.expr = simp(expr)

    def __sub__(self, other_expr):
        return dis(Sub, self, other_expr)

    def __add__(self, other_expr):
        return dis(Add, self, other_expr)

    def __mul__(self, other_expr):
        return dis(Mul, self, other_expr)

    def __truediv__(self, other_expr):
        return dis(Div, self, other_expr)

    def __pow__(self, other_expr):
        return dis(Pow, self, other_expr)

    def __rsub__(self, other_expr):
        return rdis(Sub, self, other_expr)

    def __radd__(self, other_expr):
        return rdis(Add, self, other_expr)

    def __rmul__(self, other_expr):
        return rdis(Mul, self, other_expr)

    def __rtruediv__(self, other_expr):
        return rdis(Div, self, other_expr)

    def __rpow__(self, other_expr):
        return rdis(Pow, self, other_expr)

    def __repr__(self) -> str:
        return str(estr(simp(self.expr)))

    def __str__(self) -> str:
        return str(estr(simp(self.expr)))

    def __call__(self, var, val):
        return eval(self.expr, var.expr, val)

    def D(self, var):
        print(var.expr)
        print(self.expr)
        return F(D(simp(self.expr), var.expr))


F1 = F(Add(1, 1))


# add
assert str(F1 + 10) == "12", f'{str(F1+10)} != "12"'
assert str(10 + F1) == "12", f'{str(10+F1)} != "12"'

# sub
assert str(F1 - 10) == "-8", f'{str(F1-10)} != "-8"'
assert str(10 - F1) == "8", f'{str(10-F1)} != "8"'

# mul
assert str(F1 * 10) == "20", f'{str(F1*10)} != "20"'
assert str(10 * F1) == "20", f'{str(10*F1)} != "20"'

# div
assert str(F1 / 10) == "0.2", f'{str(F1/10)} != "0.2"'
assert str(10 / F1) == "5.0", f'{str(10/F1)} != "5.0"'

# pow
assert str(F1**10) == "1024", f'{str(F1**10)} != "1024"'
assert str(10**F1) == "100", f'{str(10**F1)} != "100"'


X = F("x")  # declare a symbolic variable


def ln(x):
    return F(Call("ln", x.expr))  # declare a symbolic function


FF1 = 1 + 2 * ln(X**2 - 1)

assert str(FF1) == "(1+(2*ln(((x^2)-1))))", f'{str(FF1)} != "(1+(2*ln(((x^2)-1))))"'


assert FF1(X, 2) == 3.1972245773362196


PLOT = True
SELECT = 10

if PLOT:
    x = "x"
    f1 = Add(1, Mul(2, Call("ln", Sub(Pow(x, 2), 1))))
    f2 = Mul(Call("ln", x), Add(Mul(3, Pow(x, 2)), 1))

    df1 = simp(D(f1))
    df2 = simp(D(f2))

    plt.figure(figsize=(12, 8))
    plt.rcParams.update({"font.size": 18})

    X = [-2 + i / 100 for i in range(500)]

    plt.ylim([-5, 10])  # limit the y axis

    if SELECT & 8:
        Yf1 = [eval(f1, x, X) for X in X]
        plt.plot(X, Yf1, color="blue", label="f1 : " + estr(simp(f1)), linewidth=2)
    if SELECT & 4:
        Yf2 = [eval(f2, x, X) for X in X]
        plt.plot(X, Yf2, color="red", label="f2 : " + estr(simp(f2)), linewidth=2)
    if SELECT & 2:
        Ydf1 = [eval(df1, x, X) for X in X]
        plt.plot(X, Ydf1, color="purple", label="df1 : " + estr(simp(df1)), linewidth=2)
    if SELECT & 1:
        Ydf2 = [eval(df2, x, X) for X in X]
        plt.plot(X, Ydf2, color="orange", label="df2 : " + estr(simp(df2)), linewidth=2)

    plt.legend(loc="best")
    plt.axvline(0)

    plt.savefig("excasf1f2.pdf", transparent=True)
    plt.show()
