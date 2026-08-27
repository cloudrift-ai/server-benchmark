from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.sigma import Sigma


def _deep_expr(depth: int) -> BinaryExpr:
    expr = Var("i")
    for _ in range(depth):
        expr = expr * 2 + Var("j")
    return BinaryExpr("^", expr, Literal(1, "int"))


def test_restrict_reuses_parent_canonical_entries(monkeypatch):
    target = _deep_expr(128)
    sigma = Sigma({"b": Var("b"), "a": target})
    root_walks = 0
    original = BinaryExpr.pretty

    def counted_pretty(expr):
        nonlocal root_walks
        if expr.op == "^":
            root_walks += 1
        return original(expr)

    monkeypatch.setattr(BinaryExpr, "pretty", counted_pretty)
    restricted = [sigma.restrict({"a"}) for _ in range(1_000)]

    assert root_walks == 0
    assert all(item == restricted[0] and hash(item) == hash(restricted[0]) for item in restricted)
    assert all(item._key[0][1] is sigma._key[0][1] for item in restricted)


def test_extend_reuses_unchanged_canonical_entries(monkeypatch):
    target = _deep_expr(128)
    sigma = Sigma({"a": target})
    root_walks = 0
    original = BinaryExpr.pretty

    def counted_pretty(expr):
        nonlocal root_walks
        if expr.op == "^":
            root_walks += 1
        return original(expr)

    monkeypatch.setattr(BinaryExpr, "pretty", counted_pretty)
    extended = sigma.extend("b", Var("j"))

    assert root_walks == 0
    assert extended == Sigma({"a": target, "b": Var("j")})
    assert hash(extended) == hash(Sigma({"a": target, "b": Var("j")}))
    assert extended._key[0][1] is sigma._key[0][1]
