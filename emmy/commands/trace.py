"""Trace a transformer layer (or an inline torch module) to our Graph IR."""

import ast
import json
import logging
import sys
from pathlib import Path

from emmy.compiler.pipeline.search.working_golden import preflight_trace_inventory, write_trace_inventory

logger = logging.getLogger(__name__)


def register_trace_command(subparsers):
    from emmy.commands.compile import add_input_args  # noqa: PLC0415

    parser = subparsers.add_parser(
        "trace",
        help="Trace a model, pre-traced IR, or inline torch module to Graph IR",
    )
    add_input_args(parser, include_dump_dir=False)
    parser.add_argument("--output", "-o", help="Output JSON path (default: auto-generated)")
    parser.add_argument(
        "--golden-output",
        metavar="PATH",
        help=(
            "Write a working golden YAML skeleton plus one relative .torch.json reproducer per "
            "distinct post-fusion kernel. The skeleton contains no knobs or measurements and "
            "refuses to replace an existing YAML or sidecar directory."
        ),
    )
    parser.set_defaults(func=handle_trace)


def handle_trace(args):
    if args.code and args.input:
        logger.error("--code and positional input are mutually exclusive")
        sys.exit(2)
    if not args.code and not args.input:
        logger.error("either a positional model ID / IR file or --code is required")
        sys.exit(2)
    if args.golden_output:
        try:
            preflight_trace_inventory(args.golden_output, graph_output=args.output)
        except FileExistsError as e:
            logger.error(str(e))
            sys.exit(2)

    from emmy.commands.compile import load_or_trace  # noqa: PLC0415
    from emmy.compiler.target import apply_target_arg  # noqa: PLC0415

    apply_target_arg(args)
    graph, basename, _ = load_or_trace(args)
    _save(graph, args, default_basename=basename)
    if args.golden_output:
        input_path = Path(args.input) if args.input else None
        model = args.input if input_path is not None and not (input_path.suffix == ".json" and input_path.exists()) else None
        result = write_trace_inventory(graph, args.golden_output, graph_output=args.output, model=model)
        logger.info("Saved working golden: %s (%d distinct kernel(s))", result.path, result.target_count)


def graph_from_code(code: str, dynamic_shapes: dict | None = None):
    """Trace an inline torch expression and return ``(graph, slug, bundle)`` where
    ``bundle = (module, args, kwargs)`` is the runnable torch module + example inputs
    (used by ``tune --bench`` / ``run --bench`` to time eager / ``torch.compile``
    against the lowered graph).

    Shared by ``emmy trace --code`` and ``emmy compile --code``.
    ``dynamic_shapes`` flows through to ``torch.export.export`` so the resulting
    graph can carry symbolic dims from the SymInt pass.
    """
    info = trace_inline_code(code, dynamic_shapes=dynamic_shapes)
    return info["graph"], info["slug"], (info["module"], info["args"], info["kwargs"])


def trace_inline_code(code: str, dynamic_shapes: dict | None = None) -> dict:
    """Trace an inline torch expression and return graph + the runnable module.

    Returns a dict with ``graph``, ``slug``, ``module``, ``args``, ``kwargs``,
    and ``const_targets`` (placeholder→attribute path for parameters/buffers).
    Used by ``emmy run --code`` to compile, execute, and benchmark
    against the original PyTorch module.
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        logger.error("torch is required: pip install torch")
        sys.exit(1)

    from emmy.compiler.trace.torch import trace_module_with_constants

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        logger.error("--code failed to parse: %s", e)
        sys.exit(2)

    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        logger.error('--code must end with an expression (e.g. "torch.exp(torch.neg(x))")')
        sys.exit(2)

    final_expr = tree.body[-1].value
    scope = {"torch": torch, "nn": torch.nn, "F": F}
    preamble = ast.Module(body=tree.body[:-1], type_ignores=[])
    exec(compile(preamble, "<--code>", "exec"), scope)  # noqa: S102 — local CLI

    # Fast path: direct call to an nn.Module — trace it straight, preserving
    # the module's parameter capture (weights land as Graph constants).
    if isinstance(final_expr, ast.Call):
        try:
            maybe_mod = eval(compile(ast.Expression(final_expr.func), "<--code>", "eval"), scope)  # noqa: S307
        except Exception:  # noqa: BLE001
            maybe_mod = None
        if isinstance(maybe_mod, torch.nn.Module):
            args = tuple(eval(compile(ast.Expression(a), "<--code>", "eval"), scope) for a in final_expr.args)  # noqa: S307
            kws = {
                kw.arg: eval(compile(ast.Expression(kw.value), "<--code>", "eval"), scope)  # noqa: S307
                for kw in final_expr.keywords
                if kw.arg
            }
            logger.info("Tracing inline module: %s", ast.unparse(final_expr.func))
            graph, const_targets = trace_module_with_constants(maybe_mod, args, kwargs=kws or None, dynamic_shapes=dynamic_shapes)
            return {
                "graph": graph,
                "slug": _slugify(ast.unparse(final_expr.func)),
                "module": maybe_mod,
                "args": args,
                "kwargs": kws,
                "const_targets": const_targets,
            }

    # General path: treat the final expression as a function body. Inputs
    # come from two sources: (1) bare Name references to tensors in scope
    # (set up via a preamble like ``x = torch.randn(8)``), and (2) inline
    # tensor-constructor calls (``torch.randn(...)``, etc.) which get
    # eagerly evaluated and bound to fresh placeholder names. Everything
    # else (torch, F, nn, helper modules, scalars) stays in the closure.
    rewritten, tensor_params = _lift_tensor_inputs(final_expr, scope)
    if not tensor_params:
        logger.error("--code expression has no tensor inputs to trace")
        sys.exit(2)

    # Polish synthesized ``_x<N>`` placeholder names: use ``x`` when there's
    # exactly one synthesized input, or ``x0``/``x1``/... for multiple.
    # Names brought in from the preamble (``x = torch.randn(8); ...``) are
    # left alone.
    synth = [k for k in tensor_params if k.startswith("_x")]
    if synth:
        rename = {synth[0]: "x"} if len(synth) == 1 else {old: f"x{i}" for i, old in enumerate(synth)}
        tensor_params = {rename.get(k, k): v for k, v in tensor_params.items()}
        for node in ast.walk(rewritten):
            if isinstance(node, ast.Name) and node.id in rename:
                node.id = rename[node.id]

    expr_src = ast.unparse(rewritten)
    forward_sig = ", ".join(["self", *tensor_params.keys()])
    wrapper_src = f"class _Wrapper(torch.nn.Module):\n    def forward({forward_sig}):\n        return {expr_src}\n"
    exec(wrapper_src, scope)  # noqa: S102 — local CLI
    module = scope["_Wrapper"]()
    example_inputs = tuple(tensor_params.values())
    logger.info("Tracing inline expression: %s", ast.unparse(final_expr))
    graph, const_targets = trace_module_with_constants(module, example_inputs, dynamic_shapes=dynamic_shapes)
    return {
        "graph": graph,
        "slug": _slugify(ast.unparse(final_expr)),
        "module": module,
        "args": example_inputs,
        "kwargs": {},
        "const_targets": const_targets,
    }


_TENSOR_CTOR_NAMES = frozenset({"randn", "rand", "zeros", "ones", "empty", "full", "arange", "linspace", "tensor", "randint", "eye"})


def _lift_tensor_inputs(expr: "ast.expr", scope: dict) -> tuple["ast.expr", dict]:
    """Rewrite ``expr`` so every tensor input becomes a named placeholder.

    Two kinds of input subtrees are lifted to the returned ``tensor_params``
    dict (preserving order for function-parameter generation):

    * Bare ``Name`` references that resolve to a tensor in ``scope`` — the
      original name is preserved as a parameter.
    * ``Call`` nodes to known tensor constructors (``torch.randn``, etc.)
      with no free tensor refs below them — eagerly evaluated and replaced
      with a fresh ``_x<N>`` placeholder.

    Everything else (non-constructor calls, attribute chains, operators) is
    left intact so torch.export still traces it.
    """
    import copy

    import torch

    tensor_params: dict[str, torch.Tensor] = {}

    def is_constructor_call(node: ast.AST) -> bool:
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        # Accept torch.<ctor> and nn.<ctor>-shaped attribute chains.
        while isinstance(func, ast.Attribute):
            if func.attr in _TENSOR_CTOR_NAMES:
                return True
            func = func.value
        return False

    def fresh_placeholder() -> str:
        i = 0
        while True:
            name = f"_x{i}"
            if name not in tensor_params and name not in scope:
                return name
            i += 1

    def visit(node: ast.AST) -> ast.AST:
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            val = scope.get(node.id)
            if isinstance(val, torch.Tensor):
                tensor_params.setdefault(node.id, val)
            return node
        if is_constructor_call(node):
            try:
                val = eval(compile(ast.Expression(node), "<--code>", "eval"), scope)  # noqa: S307
            except Exception:  # noqa: BLE001
                val = None
            if isinstance(val, torch.Tensor):
                name = fresh_placeholder()
                tensor_params[name] = val
                return ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node)
        # Otherwise recurse into every child field.
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                setattr(node, field, [visit(v) if isinstance(v, ast.AST) else v for v in value])
            elif isinstance(value, ast.AST):
                setattr(node, field, visit(value))
        return node

    rewritten = visit(copy.deepcopy(expr))
    ast.fix_missing_locations(rewritten)
    return rewritten, tensor_params


def _slugify(src: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in src).strip("_").lower() or "inline"


def _save(graph, args, default_basename: str) -> None:
    ops_count: dict[str, int] = {}
    for n in graph.nodes.values():
        name = type(n.op).__name__
        ops_count[name] = ops_count.get(name, 0) + 1

    logger.info(
        "Traced: %d nodes (%s)",
        len(graph.nodes),
        ", ".join(f"{v} {k}" for k, v in sorted(ops_count.items())),
    )

    # A working-golden-only invocation should not leave a second, auto-named
    # Graph artifact. An explicit --output still requests both artifacts.
    if args.output is None and args.golden_output:
        return

    output_path = args.output or f"{default_basename}.json"
    with open(output_path, "w") as f:
        json.dump(graph.to_dict(), f, indent=2)

    logger.info("Saved: %s", output_path)
