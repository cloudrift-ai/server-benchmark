"""Trace a PyTorch module and convert to Torch IR (faithful capture).

The tracer creates one graph node per FX op, using PyTorch's exact shapes.
No decomposition, no skipping, no shape overrides.  Decomposition into
primitive Emmy IR ops happens in separate rewriter passes.

Requires PyTorch (optional dependency). All torch imports are guarded.
"""

from __future__ import annotations

import logging
import operator
from typing import TYPE_CHECKING, Any

from emmy.compiler.dtype import get as resolve_dtype
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, placeholder
from emmy.compiler.ir.frontend.ir import (
    CatOp,
    LayerNormOp,
    LinearOp,
    MatmulOp,
    MeanOp,
    ReshapeOp,
    RmsNormOp,
    SdpaOp,
    SliceOp,
    SoftmaxOp,
    TransposeOp,
    UnsqueezeOp,
)
from emmy.compiler.ir.tensor.ir import ElementwiseOp, GatherOp, IndexMapOp, IndexSource, RangeOp, ReduceOp

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

logger = logging.getLogger(__name__)

NodeRef = str | tuple[str, ...]


def has_torch() -> bool:
    """Check if PyTorch is available."""
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def trace_module(
    module: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    kwargs: dict[str, Any] | None = None,
    *,
    dynamic_shapes: dict | None = None,
) -> Graph:
    """Trace a PyTorch module and convert the FX graph to our IR.

    ``dynamic_shapes`` flows straight through to ``torch.export.export``.
    Pass a nested dict like ``{"input": {1: torch.export.Dim("seq_len")}}``
    to mark axis 1 of the ``input`` argument as dynamic — the resulting
    graph carries ``Dim('seq_len')`` at every position where torch's
    SymInt propagated, no post-trace shape rewrite needed.
    """
    graph, _ = trace_module_with_constants(module, example_inputs, kwargs=kwargs, dynamic_shapes=dynamic_shapes)
    return graph


def trace_module_with_constants(
    module: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    kwargs: dict[str, Any] | None = None,
    *,
    dynamic_shapes: dict | None = None,
) -> tuple[Graph, dict[str, str]]:
    """Trace a module and return the IR graph plus a placeholder→attribute map.

    The second return value maps each graph-level constant name (``p_*`` /
    ``b_*``) to the dotted attribute path on ``module`` where the tensor
    lives (e.g. ``self_attn.q_proj.weight``). ``torch.export`` sometimes
    strips prefixes like ``self_`` from the placeholder name, so this map
    is needed to feed constants at runtime.

    ``dynamic_shapes`` is forwarded to ``torch.export.export``. When set,
    the FX nodes' ``meta["val"]`` shapes contain ``SymInt`` entries for
    every dynamic axis; the FX→IR walker converts those to symbolic
    ``Dim`` instances. Internal names torch assigns (``s0``, ``s1``,
    …) are renamed back to the user-supplied ``Dim`` names from
    ``dynamic_shapes`` so the resulting IR reads as ``Dim('seq_len')``.
    """
    import time

    import torch

    t0 = time.monotonic()
    logger.info("torch.export.export() starting...")
    expanded_dynamic = _expand_dynamic_shapes(module, example_inputs, kwargs or {}, dynamic_shapes) if dynamic_shapes else None
    # ``prefer_deferred_runtime_asserts_over_guards`` lets torch.export accept
    # internal model guards (HF attention's ``min(head_dim*S, max_pos*S)`` etc.)
    # by deferring them to runtime asserts rather than failing the trace.
    # Only meaningful when there are dynamic shapes — keeps the static path
    # behavior identical.
    export_kwargs: dict[str, object] = {"kwargs": kwargs or {}, "dynamic_shapes": expanded_dynamic}
    if expanded_dynamic is not None:
        export_kwargs["prefer_deferred_runtime_asserts_over_guards"] = True
    # Emmy compiles inference graphs. Export under the matching grad mode so helpers decorated
    # with ``torch.no_grad`` (notably advanced RoPE modules) inline as ordinary ATen operations
    # instead of tuple-valued ``wrap_with_set_grad_enabled`` higher-order nodes.
    with torch.no_grad():
        exported = torch.export.export(module, example_inputs, **export_kwargs)
    gm = exported.graph_module
    t1 = time.monotonic()
    fx_nodes = list(gm.graph.nodes)
    live_fx_nodes = _output_live_fx_nodes(fx_nodes)
    # Runtime inputs are the exported call signature, even when this implementation does not
    # read one of them. Dynamic-shape declarations and callers still bind them by name.
    retained_fx_nodes = live_fx_nodes | {node for node in fx_nodes if node.op == "placeholder"}
    dead_count = len(fx_nodes) - len(retained_fx_nodes)
    logger.info(
        "torch.export.export() done in %.1fs (%d retained FX nodes%s)",
        t1 - t0,
        len(retained_fx_nodes),
        f", pruned {dead_count} dead" if dead_count else "",
    )

    sym_rename = _sym_rename_map(exported, dynamic_shapes) if dynamic_shapes else {}

    g = Graph()
    node_map: dict[str, NodeRef] = {}

    sig = exported.graph_signature
    const_targets: dict[str, str] = {}
    const_targets.update(sig.inputs_to_parameters)
    const_targets.update(sig.inputs_to_buffers)

    for fx_node in fx_nodes:
        if fx_node not in retained_fx_nodes:
            continue
        if fx_node.op == "placeholder":
            _handle_placeholder(g, fx_node, node_map, module, const_targets, sym_rename=sym_rename)
        elif fx_node.op == "call_function":
            _handle_call_function(g, fx_node, node_map, sym_rename=sym_rename)
        elif fx_node.op == "output":
            _handle_output(g, fx_node, node_map)
        else:
            logger.debug("Skipping FX node: %s (op=%s)", fx_node.name, fx_node.op)
    logger.info("FX→Graph IR walk done in %.1fs (%d IR nodes)", time.monotonic() - t1, len(g.nodes))

    return g, const_targets


def _output_live_fx_nodes(nodes: list[Any]) -> set[Any]:
    """Return the FX nodes observable through the exported value outputs.

    ``torch.export`` can retain a dead local mutation because FX's stock dead-code pass treats
    every mutating ATen schema as impure. Reverse reachability removes that branch, but must also
    retain a mutation through a view of a returned tensor. ATen schema aliases identify the
    storage roots that a write can affect; unsupported observable writes therefore still fail in
    the walker instead of being silently pruned.
    """
    outputs = [node for node in nodes if node.op == "output"]
    if len(outputs) != 1:
        raise ValueError(f"torch export graph must contain exactly one output node, got {len(outputs)}")

    alias_roots = _fx_alias_roots(nodes)

    live: set[Any] = set()

    def add_ancestors(seeds) -> None:
        stack = list(seeds)
        while stack:
            node = stack.pop()
            if node in live:
                continue
            live.add(node)
            stack.extend(node.all_input_nodes)

    add_ancestors(outputs)
    while True:
        live_roots = set().union(*(alias_roots[node] for node in live))
        writes = [
            node
            for node in nodes
            if node not in live and any(alias_roots.get(target, {target}) & live_roots for target in _written_fx_nodes(node))
        ]
        if not writes:
            break
        add_ancestors(writes)
    return live


def _fx_alias_roots(nodes: list[Any]) -> dict[Any, set[Any]]:
    """Return each FX value's storage roots from ATen schema alias labels."""
    alias_roots: dict[Any, set[Any]] = {}
    for node in nodes:
        roots = {node}
        schema = getattr(node.target, "_schema", None) if node.op == "call_function" else None
        if schema is not None:
            argument_roots: dict[str, set[Any]] = {}
            for index, argument in enumerate(schema.arguments):
                alias = argument.alias_info
                if alias is None:
                    continue
                value = node.kwargs.get(argument.name, node.args[index] if index < len(node.args) else None)
                for source in _fx_nodes_in_arg(value):
                    for label in alias.before_set | alias.after_set:
                        argument_roots.setdefault(label, set()).update(alias_roots.get(source, {source}))
            returned_roots: set[Any] = set()
            for result in schema.returns:
                alias = result.alias_info
                if alias is None:
                    continue
                for label in alias.before_set | alias.after_set:
                    returned_roots.update(argument_roots.get(label, ()))
            if returned_roots:
                roots = returned_roots
        alias_roots[node] = roots
    return alias_roots


def _observable_alias_read_after_write(node) -> tuple[Any, Any] | None:
    """Find a live later read of written storage that bypasses ``node``'s returned value."""
    nodes = list(node.graph.nodes)
    aliases = _fx_alias_roots(nodes)
    written = _written_fx_nodes(node)
    if not written:
        return None
    written_roots = set().union(*(aliases.get(target, {target}) for target in written))

    returned_descendants = {node}
    stack = list(node.users)
    while stack:
        descendant = stack.pop()
        if descendant in returned_descendants:
            continue
        returned_descendants.add(descendant)
        stack.extend(descendant.users)

    live = _output_live_fx_nodes(nodes)
    position = nodes.index(node)
    for later in nodes[position + 1 :]:
        if later not in live:
            continue
        for source in later.all_input_nodes:
            if source in returned_descendants:
                continue
            if aliases.get(source, {source}) & written_roots:
                return later, source
    return None


def _static_local_slice_destination(node) -> tuple[Any, tuple[int, ...], tuple[int, ...]] | None:
    """Describe a unit-step slice destination rooted at a local tensor constructor.

    Returns ``(root, offsets, extents)`` in root coordinates. This deliberately accepts
    only the allocation + static ``aten.slice`` form whose functional update is exactly
    representable by a two-source ``IndexMapOp``. Other aliases stay fail-closed.
    """
    destination = _static_affine_view_destination(node)
    if destination is None:
        return None
    root, offsets, extents, view_axes = destination
    if any(axis is None for axis in view_axes):
        return None
    if root.op != "call_function" or _op_name(root.target) not in ("new_zeros", "new_full"):
        return None
    return root, offsets, extents


def _static_affine_view_destination(
    node,
) -> tuple[Any, tuple[int, ...], tuple[int, ...], tuple[int | None, ...]] | None:
    """Describe a static unit-step slice/select view in its root coordinates.

    ``view_axes[root_axis]`` names the corresponding destination axis, or ``None``
    when ``aten.select`` fixed that root coordinate. The descriptor is intentionally
    rectangular; dynamic, strided, and other indexed views remain unsupported.
    """
    destination = node.args[0] if node.args else None
    if destination is None or not hasattr(destination, "op"):
        return None

    chain = []
    current = destination
    while current.op == "call_function" and _op_name(current.target) in ("slice", "select"):
        chain.append(current)
        current = current.args[0] if current.args else None
        if current is None or not hasattr(current, "op"):
            return None

    root_shape = _static_fx_shape(current)
    if root_shape is None:
        return None
    offsets = [0] * len(root_shape)
    extents = list(root_shape)
    view_axes: list[int | None] = list(range(len(root_shape)))

    for view in reversed(chain):
        args = view.args
        op_name = _op_name(view.target)
        dim = args[1] if len(args) > 1 else 0
        if not isinstance(dim, int) or isinstance(dim, bool):
            return None
        rank = sum(axis is not None for axis in view_axes)
        norm_dim = dim if dim >= 0 else rank + dim
        if not 0 <= norm_dim < rank:
            return None
        root_axis = next((axis for axis, view_axis in enumerate(view_axes) if view_axis == norm_dim), None)
        if root_axis is None:
            return None

        if op_name == "slice":
            start = args[2] if len(args) > 2 else None
            end = args[3] if len(args) > 3 else None
            step = args[4] if len(args) > 4 else 1
            if any(value is not None and (not isinstance(value, int) or isinstance(value, bool)) for value in (start, end)):
                return None
            if step != 1:
                return None
            normalized_start, normalized_end, _ = slice(start, end, 1).indices(extents[root_axis])
            offsets[root_axis] += normalized_start
            extents[root_axis] = normalized_end - normalized_start
        else:
            index = args[2] if len(args) > 2 else 0
            if not isinstance(index, int) or isinstance(index, bool):
                return None
            normalized_index = index if index >= 0 else extents[root_axis] + index
            if not 0 <= normalized_index < extents[root_axis]:
                return None
            offsets[root_axis] += normalized_index
            extents[root_axis] = 1
            view_axes[root_axis] = None
            view_axes = [axis - 1 if axis is not None and axis > norm_dim else axis for axis in view_axes]

        expected_shape = [0] * sum(axis is not None for axis in view_axes)
        for axis, view_axis in enumerate(view_axes):
            if view_axis is not None:
                expected_shape[view_axis] = extents[axis]
        if _static_fx_shape(view) != tuple(expected_shape):
            return None

    return current, tuple(offsets), tuple(extents), tuple(view_axes)


def _static_fx_shape(node) -> tuple[int, ...] | None:
    val = node.meta.get("val") if isinstance(getattr(node, "meta", None), dict) else None
    shape = getattr(val, "shape", None)
    if shape is None or any(not isinstance(extent, int) or isinstance(extent, bool) for extent in shape):
        return None
    return tuple(shape)


def _reassemblable_local_slice_copy(node) -> tuple[Any, tuple[int, ...], tuple[int, ...]] | None:
    """Return a local slice-update description when every alias can be versioned safely."""
    # A used ``copy_`` return is itself an alias. Supporting both it and an updated base would
    # require versioning that returned view across later writes, outside this narrow local form.
    if node.users:
        return None
    destination = _static_local_slice_destination(node)
    if destination is None:
        return None
    root, _, _ = destination
    if not _local_alias_version_is_rebindable(node, root):
        return None
    return destination


def _local_alias_version_is_rebindable(node, root) -> bool:
    """Whether a local root can be rebound without leaving a stale live alias."""
    nodes = list(node.graph.nodes)
    aliases = _fx_alias_roots(nodes)
    if aliases.get(root, {root}) != {root}:
        return False

    written_roots = set().union(*(aliases.get(target, {target}) for target in _written_fx_nodes(node)))
    live = _output_live_fx_nodes(nodes)
    position = nodes.index(node)
    positions = {candidate: index for index, candidate in enumerate(nodes)}

    # The root name is rebound to the functional update during the walk, so later aliases
    # constructed from it see the new version. An alias constructed before this write would
    # still name the old Graph node and therefore cannot be updated safely here.
    for later in nodes[position + 1 :]:
        if later not in live:
            continue
        for source in later.all_input_nodes:
            if not (aliases.get(source, {source}) & written_roots):
                continue
            if source is root or positions[source] > position:
                continue
            return False
    return True


def _reassemblable_local_affine_fill(
    node,
) -> tuple[Any, tuple[int, ...], tuple[int, ...], tuple[int | None, ...]] | None:
    """Return a local affine-view update when every observable alias can be versioned."""
    if node.users:
        return None
    destination = _static_affine_view_destination(node)
    if destination is None:
        return None
    root, _, _, _ = destination
    if root.op != "call_function":
        return None
    if not _local_alias_version_is_rebindable(node, root):
        return None
    return destination


def _fx_nodes_in_arg(value) -> list[Any]:
    if hasattr(value, "all_input_nodes") and hasattr(value, "op"):
        return [value]
    if isinstance(value, (tuple, list)):
        return [node for item in value for node in _fx_nodes_in_arg(item)]
    if isinstance(value, dict):
        return [node for item in value.values() for node in _fx_nodes_in_arg(item)]
    return []


def _written_fx_nodes(node) -> list[Any]:
    if node.op != "call_function":
        return []
    schema = getattr(node.target, "_schema", None)
    if schema is None:
        return []
    written: list[Any] = []
    for index, argument in enumerate(schema.arguments):
        alias = argument.alias_info
        if alias is None or not alias.is_write:
            continue
        value = node.kwargs.get(argument.name, node.args[index] if index < len(node.args) else None)
        written.extend(_fx_nodes_in_arg(value))
    return written


def _expand_dynamic_shapes(module, example_inputs: tuple, kwargs: dict, user_dynamic: dict) -> dict:
    """Auto-fill ``None`` for any forward-arg the user didn't mark dynamic.

    ``torch.export.export(dynamic_shapes=...)`` requires the top-level
    dict to match EXACTLY the args / kwargs actually passed (not the
    forward signature). Container-typed args (tuple of tensors for HF's
    ``position_embeddings`` etc.) need a structurally-matching spec,
    not a bare ``None``, or torch raises on the type mismatch."""
    import inspect

    sig = inspect.signature(module.forward)
    sig_params = [n for n in sig.parameters if n != "self"]
    positional_names = sig_params[: len(example_inputs)]
    expected = list(zip(positional_names, example_inputs, strict=True)) + list(kwargs.items())
    out: dict[str, object] = {}
    for name, value in expected:
        out[name] = user_dynamic.get(name) if name in user_dynamic else _static_spec_for(value)
    # Pass through any user keys that weren't expected so torch's error
    # message stays informative (e.g. typo'd input names).
    for name, spec in user_dynamic.items():
        out.setdefault(name, spec)
    return out


def _static_spec_for(value) -> object:
    """Build a structurally-matching ``None`` spec for an input value.

    Bare tensor → ``None``. Tuple / list of tensors → matching tuple /
    list of ``None``. Anything else → ``None`` (torch accepts that for
    non-tensor inputs)."""
    if isinstance(value, tuple):
        return tuple(_static_spec_for(v) for v in value)
    if isinstance(value, list):
        return [_static_spec_for(v) for v in value]
    return None


def _sym_rename_map(exported, dynamic_shapes: dict) -> dict[str, str]:
    """Build a ``{torch-internal-symbol-name: user-Dim-name}`` mapping.

    ``torch.export`` invents internal names (``s0``, ``s1``, …) for each
    dynamic dim, but the user passed in ``torch.export.Dim('seq_len')``
    and expects ``Dim('seq_len')`` to show up in the resulting IR.
    Match by walking ``exported.graph_signature.input_specs`` against the
    ``dynamic_shapes`` keys, then inspecting the placeholder FX node's
    ``meta["val"].shape[axis]`` to find the SymInt torch assigned at
    that position.
    """
    rename: dict[str, str] = {}
    gm = exported.graph_module
    # Map arg-name → placeholder FX node. ``torch.export`` may rename
    # placeholders (strip ``self_`` etc.), but the signature carries
    # the original user-arg-name → placeholder-name mapping.
    sig = exported.graph_signature
    user_to_placeholder: dict[str, str] = {}
    for spec in sig.input_specs:
        # ``user_inputs`` specs have ``arg.name == placeholder.name``;
        # parameter / buffer specs aren't user-facing inputs.
        if getattr(spec, "kind", None) is not None and spec.kind.name == "USER_INPUT":
            user_to_placeholder[spec.arg.name] = spec.arg.name
    placeholder_by_name: dict[str, Any] = {n.name: n for n in gm.graph.nodes if n.op == "placeholder"}

    for arg_name, axis_map in dynamic_shapes.items():
        ph_name = user_to_placeholder.get(arg_name, arg_name)
        ph = placeholder_by_name.get(ph_name)
        if ph is None:
            continue
        val = ph.meta.get("val")
        if val is None or not hasattr(val, "shape"):
            continue
        for axis, user_dim in axis_map.items():
            if axis >= len(val.shape):
                continue
            sym_int = val.shape[axis]
            # ``SymInt`` carries the symbol via ``.node.expr`` (sympy
            # ``Symbol``); ``str(symbol)`` is the internal name like ``s0``.
            sym_name = _symint_name(sym_int)
            if sym_name is None:
                continue
            rename[sym_name] = user_dim.__name__ if hasattr(user_dim, "__name__") else str(user_dim)
    return rename


def _symint_name(value) -> str | None:
    """Return the underlying sympy symbol name for a ``SymInt`` placeholder,
    or ``None`` if ``value`` is a plain int (the static-axis case)."""
    if isinstance(value, int):
        return None
    node = getattr(value, "node", None)
    if node is None:
        return None
    expr = getattr(node, "expr", None) or node
    # Sympy Symbols stringify to their name (``s0``); compound expressions
    # would stringify to e.g. ``2*s0`` — we only support plain symbols today.
    return str(expr)


def _wrap_shape(raw_shape, sym_rename: dict[str, str] | None = None):
    """Convert a torch ``Size`` (possibly containing ``SymInt``) to the
    tuple form our IR expects.

    Plain ints pass through. ``SymInt`` placeholders become ``Dim(name)``
    with ``name`` resolved through ``sym_rename`` (so torch's ``s0``
    becomes the user's ``seq_len``); unrenamed symbols keep their
    torch-internal name. The symbolic ``Dim`` carries the default expected
    size (``DEFAULT_SEQ_HINT``) so the planner / tuner can size tiles for it.
    The static-shape case (no SymInt anywhere) returns a ``tuple[int, ...]``
    — the existing IR construction path coerces those to ``Dim(int)`` via
    ``Tensor.__post_init__``.
    """
    from emmy.compiler.dim import Dim

    if sym_rename is None:
        sym_rename = {}
    out = []
    for d in raw_shape:
        if isinstance(d, int):
            out.append(d)
        else:
            sym_name = _symint_name(d)
            if sym_name is None:
                # Compound expression we can't represent — fall back to a stringified placeholder.
                out.append(Dim(str(d)))
            else:
                out.append(Dim(sym_rename.get(sym_name, sym_name)))
    return tuple(out)


def _get_shape(fx_node: Any, sym_rename: dict[str, str] | None = None) -> tuple:
    meta = fx_node.meta.get("val", None)
    if meta is not None and hasattr(meta, "shape"):
        return _wrap_shape(meta.shape, sym_rename)
    return ()


def _op_shape(raw_shape, sym_rename: dict[str, str] | None = None):
    """Convert an FX-arg shape (``view``/``reshape``'s second arg, etc.) to
    the ``tuple[int | str, ...]`` form ``ReshapeOp.shape`` /
    ``SliceOp.shape`` expect.

    Ints pass through. ``-1`` (numpy/torch infer-this-dim sentinel) passes
    through. FX node references to ``aten.sym_size.int`` outputs (which
    appear in reshape arg lists as ``[1, sym_size_int_1, 4, -1]``)
    resolve through ``meta["val"]`` to the underlying ``SymInt``.
    ``SymInt`` becomes the renamed symbolic name (``s0`` → ``seq_len``).
    Unknown values get stringified as a fallback so we don't silently
    lose information.
    """
    sym_rename = sym_rename or {}
    out = []
    for d in raw_shape:
        if isinstance(d, int):
            out.append(d)
            continue
        # FX node reference: resolve to ``meta["val"]`` (a SymInt scalar).
        meta = getattr(d, "meta", None)
        if isinstance(meta, dict):
            val = meta.get("val")
            if val is not None:
                d = val
        sym = _symint_name(d)
        if sym is not None:
            out.append(sym_rename.get(sym, sym))
        else:
            out.append(str(d))
    return tuple(out)


def _dim_tuple_to_op_shape(shape):
    """Convert a ``tuple[Dim | int, ...]`` (from ``_wrap_shape`` output) back
    to the ``tuple[int | str, ...]`` form ``ReshapeOp.shape`` /
    ``SliceOp.shape`` carry. Atomic ``Dim`` wrappers unwrap via ``.value``
    (Literal → int, Var → name); composite Dims aren't emitted by the
    tracer here so the int|str form is sufficient."""
    from emmy.compiler.dim import Dim

    return tuple(d.value if isinstance(d, Dim) else d for d in shape)


def _get_dtype(fx_node: Any) -> str:
    meta = getattr(fx_node, "meta", None)
    val = meta.get("val", None) if isinstance(meta, dict) else None
    if val is not None and hasattr(val, "dtype"):
        return str(val.dtype).replace("torch.", "")
    return "f32"


def _resolve_inputs(fx_node: Any, node_map: dict[str, NodeRef], g: Graph | None = None) -> list[str]:
    """Resolve FX node args to our node IDs. Scalars become ConstantOp nodes."""
    result = []
    for a in fx_node.args:
        if hasattr(a, "name") and a.name in node_map:
            ref = node_map[a.name]
            if isinstance(ref, str):
                result.append(ref)
        elif isinstance(a, (list, tuple)):
            for item in a:
                if hasattr(item, "name") and item.name in node_map:
                    ref = node_map[item.name]
                    if isinstance(ref, str):
                        result.append(ref)
        elif isinstance(a, (int, float)) and not isinstance(a, bool) and g is not None:
            const_name = f"{fx_node.name}_c{len(result)}"
            # Inherit dtype from the consuming op's output — scalar literals
            # in mixed-dtype graphs (fp16 * 0.5, etc.) must stay in the
            # consumer's dtype to avoid widening every elementwise step. A
            # bool-output consumer (a mask-construction comparison like
            # ``x > 0.5``) compares in the operand's domain, not bool — the
            # literal falls back to f32 there.
            dtype = _get_dtype(fx_node)
            const_id = g.add_node(
                op=ConstantOp(name=const_name, value=float(a)),
                inputs=[],
                output=Tensor(const_name, (1,), "f32" if dtype == "bool" else dtype),
                node_id=const_name,
            )
            node_map[const_name] = const_id
            result.append(const_id)
    return result


def _handle_placeholder(
    g: Graph,
    fx_node: Any,
    node_map: dict[str, NodeRef],
    module: nn.Module,
    const_targets: dict[str, str],
    *,
    sym_rename: dict[str, str] | None = None,
) -> None:
    """Handle placeholder nodes (inputs, parameters, and buffers).

    ``torch.export`` prefixes parameter placeholders with ``p_`` and buffer
    placeholders with ``b_``. Both are bake-in constants from the compiler's
    perspective — only actual user-supplied activations (no prefix) are
    graph inputs.

    Parameter / buffer ``ConstantOp``s carry the source attribute path
    (``source_path``) and pre-chain layout (``source_shape`` / ``source_dtype``)
    so the loader can read them directly from safetensors without a
    side-channel ``const_targets`` dict.
    """
    name = fx_node.name
    shape = _get_shape(fx_node, sym_rename)
    dtype = _get_dtype(fx_node)
    # The prefix alone is not enough: a USER forward arg may legitimately start with
    # ``b_`` (the MoE expert wrapper's ``b_gate_up`` bias input) — only placeholders the
    # export signature actually maps to a parameter/buffer are constants.
    is_const = name.startswith(("p_", "b_")) and name in const_targets

    if is_const:
        op = ConstantOp(
            name=name,
            source_path=const_targets.get(name),
            source_shape=tuple(shape) if shape else None,
            source_dtype=dtype,
        )
        nid = g.add_node(op=op, inputs=[], output=Tensor(name, shape, dtype), node_id=name)
    else:
        nid = g.add_node(op=InputOp(), inputs=[], output=Tensor(name, shape, dtype), node_id=name)
        g.inputs.append(nid)
    node_map[name] = nid


def _handle_output(g: Graph, fx_node: Any, node_map: dict[str, NodeRef]) -> None:
    """Handle output nodes."""
    for arg in fx_node.args[0] if isinstance(fx_node.args[0], (tuple, list)) else [fx_node.args[0]]:
        if hasattr(arg, "name") and arg.name in node_map:
            ref = node_map[arg.name]
            if isinstance(ref, str):
                g.outputs.append(ref)
            else:
                g.outputs.extend(ref)


def _op_name(target: Any) -> str | None:
    """Extract a short op name from an ATen target."""
    s = str(target)
    if "aten." in s:
        parts = s.split(".")
        for i, p in enumerate(parts):
            if p == "aten" and i + 1 < len(parts):
                return parts[i + 1]
    return None


def _get_reduce_axis(fx_node: Any) -> int | str:
    """Extract the reduction axis from an FX node."""
    if len(fx_node.args) > 1:
        axis = fx_node.args[1]
        if isinstance(axis, (list, tuple)):
            return axis[0] if len(axis) == 1 else axis[0]
        if isinstance(axis, int):
            return axis
    return -1


def _keepdim_shape(input_shape: tuple, axis: int | str) -> tuple:
    """Return the keepdim output shape for a reduction over ``axis``."""
    if not isinstance(axis, int) or not input_shape:
        return tuple(input_shape)
    a = axis if axis >= 0 else len(input_shape) + axis
    if a < 0 or a >= len(input_shape):
        return tuple(input_shape)
    return tuple(input_shape[:a]) + (1,) + tuple(input_shape[a + 1 :])


def _squeeze_indexmap(in_shape: tuple, out_shape: tuple, axis: int | str) -> IndexMapOp:
    """Build an IndexMapOp that drops a single size-1 axis.

    ``in_shape`` is the keepdim-reduce output (rank N, dim ``axis`` = 1).
    ``out_shape`` is the non-keepdim shape (rank N-1, ``axis`` removed).
    """
    if not isinstance(axis, int):
        # Symbolic axis fallback — identity map (no safety net).
        coord_map = tuple(placeholder(d) for d in range(len(out_shape)))
        return IndexMapOp(out_shape=tuple(out_shape), sources=(IndexSource(input_idx=0, coord_map=coord_map),))
    a = axis if axis >= 0 else len(in_shape) + axis
    coord_map = []
    out_d = 0
    for in_d in range(len(in_shape)):
        if in_d == a:
            coord_map.append(Literal(0, "int"))
        else:
            coord_map.append(placeholder(out_d))
            out_d += 1
    return IndexMapOp(out_shape=tuple(out_shape), sources=(IndexSource(input_idx=0, coord_map=tuple(coord_map)),))


def _handle_getitem(fx_node: Any, node_map: dict[str, NodeRef]) -> None:
    """Resolve a tuple-producing ``aten.chunk`` result to one materialized slice."""
    if len(fx_node.args) < 2:
        raise ValueError("operator.getitem on an aten.chunk result requires a tuple index")
    source, raw_index = fx_node.args[:2]
    source_name = getattr(source, "name", None)
    ref = node_map.get(source_name)
    if not isinstance(ref, tuple):
        if isinstance(ref, str):
            node_map[fx_node.name] = ref
        return
    if not isinstance(raw_index, int) or isinstance(raw_index, bool):
        raise NotImplementedError("operator.getitem on an aten.chunk result requires a constant integer tuple index")
    index = raw_index if raw_index >= 0 else len(ref) + raw_index
    if not 0 <= index < len(ref):
        if hasattr(fx_node, "users") and not fx_node.users:
            return  # torch.export may leave a dead getitem for an unused tuple result
        raise IndexError(f"aten.chunk tuple index {raw_index} is out of range for {len(ref)} outputs")
    node_map[fx_node.name] = ref[index]


def _handle_chunk(
    g: Graph,
    fx_node: Any,
    node_map: dict[str, NodeRef],
    *,
    sym_rename: dict[str, str] | None = None,
) -> None:
    """Materialize a static ``aten.chunk`` tuple as cumulative ``SliceOp`` nodes."""
    if len(fx_node.args) < 2:
        raise ValueError("aten.chunk requires a chunk count")
    source = fx_node.args[0]
    source_ref = node_map.get(getattr(source, "name", None))
    if not isinstance(source_ref, str):
        raise ValueError("aten.chunk input did not resolve to a tensor")

    raw_chunks = fx_node.args[1]
    if not isinstance(raw_chunks, int) or isinstance(raw_chunks, bool):
        raise NotImplementedError("aten.chunk requires a static integer chunk count")
    if raw_chunks <= 0:
        raise ValueError("aten.chunk chunk count must be greater than zero")

    raw_dim = fx_node.args[2] if len(fx_node.args) > 2 else 0
    if not isinstance(raw_dim, int) or isinstance(raw_dim, bool):
        raise NotImplementedError("aten.chunk requires a constant integer dimension")

    source_node = g.nodes[source_ref]
    rank = len(source_node.output.shape)
    dim = raw_dim if raw_dim >= 0 else rank + raw_dim
    if not 0 <= dim < rank:
        raise ValueError(f"aten.chunk dimension {raw_dim} is out of range for rank-{rank} input")

    values = fx_node.meta.get("val")
    if not isinstance(values, (list, tuple)):
        raise NotImplementedError("aten.chunk requires FX-provided static tuple output metadata")

    output_ids: list[str] = []
    offset = 0
    for index, value in enumerate(values):
        if not hasattr(value, "shape"):
            raise NotImplementedError("aten.chunk output metadata must contain tensor shapes")
        raw_shape = tuple(value.shape)
        chunk_extent = raw_shape[dim]
        if not isinstance(chunk_extent, int):
            raise NotImplementedError("aten.chunk does not support a dynamic chunked dimension")
        shape = _wrap_shape(raw_shape, sym_rename)
        dtype = str(value.dtype).replace("torch.", "")
        output_name = f"{fx_node.name}_{index}"
        output_id = g.add_node(
            op=SliceOp(shape=_dim_tuple_to_op_shape(shape), dim=dim, start=offset),
            inputs=[source_ref],
            output=Tensor(output_name, shape, dtype),
            node_id=output_name,
        )
        output_ids.append(output_id)
        offset += chunk_extent

    if len(output_ids) > raw_chunks:
        raise ValueError(f"aten.chunk produced {len(output_ids)} outputs for chunk count {raw_chunks}")
    input_extent = source_node.output.shape[dim]
    if getattr(input_extent, "is_static", False) and offset != input_extent.as_static():
        raise ValueError(f"aten.chunk output extents cover {offset} elements but input dimension has {input_extent.as_static()}")
    node_map[fx_node.name] = tuple(output_ids)


def _handle_split_with_sizes(
    g: Graph,
    fx_node: Any,
    node_map: dict[str, NodeRef],
    *,
    sym_rename: dict[str, str] | None = None,
) -> None:
    """Materialize a static ``aten.split_with_sizes`` tuple as slices."""
    if len(fx_node.args) < 2:
        raise ValueError("aten.split_with_sizes requires a size list")
    source = fx_node.args[0]
    source_ref = node_map.get(getattr(source, "name", None))
    if not isinstance(source_ref, str):
        raise ValueError("aten.split_with_sizes input did not resolve to a tensor")

    raw_sizes = fx_node.args[1]
    if not isinstance(raw_sizes, (list, tuple)) or any(
        not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in raw_sizes
    ):
        raise NotImplementedError("aten.split_with_sizes requires a static list of non-negative integer sizes")

    raw_dim = fx_node.args[2] if len(fx_node.args) > 2 else 0
    if not isinstance(raw_dim, int) or isinstance(raw_dim, bool):
        raise NotImplementedError("aten.split_with_sizes requires a constant integer dimension")
    source_node = g.nodes[source_ref]
    rank = len(source_node.output.shape)
    dim = raw_dim if raw_dim >= 0 else rank + raw_dim
    if not 0 <= dim < rank:
        raise ValueError(f"aten.split_with_sizes dimension {raw_dim} is out of range for rank-{rank} input")

    values = fx_node.meta.get("val")
    if not isinstance(values, (list, tuple)) or len(values) != len(raw_sizes):
        raise NotImplementedError("aten.split_with_sizes requires matching FX tuple output metadata")

    output_ids: list[str] = []
    offset = 0
    for index, (size, value) in enumerate(zip(raw_sizes, values, strict=True)):
        if not hasattr(value, "shape") or value.shape[dim] != size:
            raise ValueError(f"aten.split_with_sizes output {index} metadata does not match declared size {size}")
        shape = _wrap_shape(tuple(value.shape), sym_rename)
        dtype = str(value.dtype).replace("torch.", "")
        output_name = f"{fx_node.name}_{index}"
        output_id = g.add_node(
            op=SliceOp(shape=_dim_tuple_to_op_shape(shape), dim=dim, start=offset),
            inputs=[source_ref],
            output=Tensor(output_name, shape, dtype),
            node_id=output_name,
        )
        output_ids.append(output_id)
        offset += size

    input_extent = source_node.output.shape[dim]
    if getattr(input_extent, "is_static", False) and offset != input_extent.as_static():
        raise ValueError(f"aten.split_with_sizes sizes cover {offset} elements but input dimension has {input_extent.as_static()}")
    node_map[fx_node.name] = tuple(output_ids)


def _handle_unbind(
    g: Graph,
    fx_node: Any,
    node_map: dict[str, NodeRef],
    *,
    sym_rename: dict[str, str] | None = None,
) -> None:
    """Materialize static ``aten.unbind`` outputs as axis-fixing index maps."""
    if not fx_node.args:
        raise ValueError("aten.unbind requires an input tensor")
    source = fx_node.args[0]
    source_ref = node_map.get(getattr(source, "name", None))
    if not isinstance(source_ref, str):
        raise ValueError("aten.unbind input did not resolve to a tensor")

    raw_dim = fx_node.args[1] if len(fx_node.args) > 1 else 0
    if not isinstance(raw_dim, int) or isinstance(raw_dim, bool):
        raise NotImplementedError("aten.unbind requires a constant integer dimension")
    source_node = g.nodes[source_ref]
    rank = len(source_node.output.shape)
    dim = raw_dim if raw_dim >= 0 else rank + raw_dim
    if not 0 <= dim < rank:
        raise ValueError(f"aten.unbind dimension {raw_dim} is out of range for rank-{rank} input")

    values = fx_node.meta.get("val")
    if not isinstance(values, (list, tuple)):
        raise NotImplementedError("aten.unbind requires FX-provided static tuple output metadata")
    input_extent = source_node.output.shape[dim]
    if getattr(input_extent, "is_static", False) and len(values) != input_extent.as_static():
        raise ValueError(f"aten.unbind produced {len(values)} outputs but input dimension has {input_extent.as_static()} elements")

    output_ids: list[str] = []
    for index, value in enumerate(values):
        if not hasattr(value, "shape") or len(value.shape) != rank - 1:
            raise NotImplementedError("aten.unbind output metadata must contain rank-reduced tensor shapes")
        shape = _wrap_shape(tuple(value.shape), sym_rename)
        dtype = str(value.dtype).replace("torch.", "")
        coord_map = []
        output_dim = 0
        for input_dim in range(rank):
            if input_dim == dim:
                coord_map.append(Literal(index, "int"))
            else:
                coord_map.append(placeholder(output_dim))
                output_dim += 1
        output_name = f"{fx_node.name}_{index}"
        output_id = g.add_node(
            op=IndexMapOp(out_shape=shape, sources=(IndexSource(input_idx=0, coord_map=tuple(coord_map)),)),
            inputs=[source_ref],
            output=Tensor(output_name, shape, dtype),
            node_id=output_name,
        )
        output_ids.append(output_id)
    node_map[fx_node.name] = tuple(output_ids)


def _handle_max_dim_values(
    g: Graph,
    fx_node: Any,
    node_map: dict[str, NodeRef],
    *,
    sym_rename: dict[str, str] | None = None,
) -> None:
    """Lower ``aten.max.dim`` when the graph consumes only its values tuple item."""
    for user in fx_node.users:
        uses_values = (
            user.target is operator.getitem
            and len(user.args) >= 2
            and isinstance(user.args[1], int)
            and (user.args[1] == 0 or (user.args[1] == 1 and not user.users))
        )
        if not uses_values:
            raise NotImplementedError("aten.max.dim argmax indices are not supported by Torch IR")

    source = fx_node.args[0] if fx_node.args else None
    source_ref = node_map.get(getattr(source, "name", None))
    if not isinstance(source_ref, str):
        raise ValueError("aten.max.dim input did not resolve to a tensor")
    raw_dim = fx_node.args[1] if len(fx_node.args) > 1 else -1
    if not isinstance(raw_dim, int) or isinstance(raw_dim, bool):
        raise NotImplementedError("aten.max.dim requires a constant integer dimension")
    raw_keepdim = fx_node.args[2] if len(fx_node.args) > 2 else False
    if not isinstance(raw_keepdim, bool):
        raise NotImplementedError("aten.max.dim requires a constant keepdim flag")

    values = fx_node.meta.get("val")
    if not isinstance(values, (list, tuple)) or len(values) != 2 or not hasattr(values[0], "shape"):
        raise NotImplementedError("aten.max.dim requires FX-provided values/indices output metadata")
    value = values[0]
    shape = _wrap_shape(tuple(value.shape), sym_rename)
    dtype = str(value.dtype).replace("torch.", "")
    input_shape = tuple(g.nodes[source_ref].output.shape)
    axis = raw_dim if raw_dim >= 0 else len(input_shape) + raw_dim
    keepdim_shape = _keepdim_shape(input_shape, axis)
    reduce_op = ReduceOp(op=ElementwiseImpl("amax"), axis=axis)
    value_name = f"{fx_node.name}_values"
    if raw_keepdim:
        value_id = g.add_node(
            op=reduce_op,
            inputs=[source_ref],
            output=Tensor(value_name, shape, dtype),
            node_id=value_name,
        )
    else:
        reduce_id = g.add_node(
            op=reduce_op,
            inputs=[source_ref],
            output=Tensor(f"{value_name}_keepdim", keepdim_shape, dtype),
        )
        value_id = g.add_node(
            op=_squeeze_indexmap(keepdim_shape, shape, axis),
            inputs=[reduce_id],
            output=Tensor(value_name, shape, dtype),
            node_id=value_name,
        )
    # Only tuple item zero is permitted above, so a one-element reference tuple
    # faithfully services every downstream getitem without inventing argmax IR.
    node_map[fx_node.name] = (value_id,)


def _handle_call_function(g: Graph, fx_node: Any, node_map: dict[str, NodeRef], *, sym_rename: dict[str, str] | None = None) -> None:
    """Handle call_function nodes — faithful 1:1 capture of FX ops."""
    if fx_node.target is operator.getitem:
        _handle_getitem(fx_node, node_map)
        return

    op_name = _op_name(fx_node.target)
    if op_name == "chunk":
        _handle_chunk(g, fx_node, node_map, sym_rename=sym_rename)
        return
    if op_name == "split_with_sizes":
        _handle_split_with_sizes(g, fx_node, node_map, sym_rename=sym_rename)
        return
    if op_name == "unbind":
        _handle_unbind(g, fx_node, node_map, sym_rename=sym_rename)
        return
    if op_name == "max" and isinstance(fx_node.meta.get("val"), (list, tuple)):
        _handle_max_dim_values(g, fx_node, node_map, sym_rename=sym_rename)
        return

    # ``aten.sym_size.int`` and similar shape-metadata ops return a
    # scalar ``SymInt`` (no tensor shape). They're consumed inline by
    # ``_op_shape`` when a downstream reshape / view references them via
    # ``args``; making them graph nodes would only confuse the matcher.
    val = fx_node.meta.get("val")
    if isinstance(val, (list, tuple)):
        raise NotImplementedError(f"no tracer mapping for multi-output op aten.{op_name or fx_node.target} ({len(val)} outputs)")
    if val is not None and not hasattr(val, "shape"):
        return
    name = fx_node.name
    shape = _get_shape(fx_node, sym_rename)
    dtype = _get_dtype(fx_node)
    input_ids = _resolve_inputs(fx_node, node_map, g)

    if op_name is None:
        if input_ids:
            node_map[name] = input_ids[0]
        return

    # --- Tensor constructors ---
    if op_name == "arange":
        args = fx_node.args
        if len(args) == 1:
            raw_start, raw_stop, raw_step = 0, args[0], 1
        elif len(args) == 2:
            raw_start, raw_stop, raw_step = args[0], args[1], 1
        elif len(args) == 3:
            raw_start, raw_stop, raw_step = args
        else:
            raise NotImplementedError(f"aten.arange requires one to three static integer arguments, got {len(args)}")
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in (raw_start, raw_stop, raw_step)):
            raise NotImplementedError("aten.arange requires static integer start, stop, and step")
        if raw_step == 0:
            raise ValueError("aten.arange step must be non-zero")
        if resolve_dtype(dtype).np.kind not in ("i", "u"):
            raise NotImplementedError(f"aten.arange requires an integer output dtype, got {dtype}")
        expected_shape = (len(range(raw_start, raw_stop, raw_step)),)
        if _static_fx_shape(fx_node) != expected_shape:
            raise ValueError(f"aten.arange output shape mismatch: expected {expected_shape}, got {_static_fx_shape(fx_node)}")
        node_map[name] = g.add_node(
            op=RangeOp(start=raw_start, stop=raw_stop, step=raw_step, dtype=dtype),
            inputs=[],
            output=Tensor(name, shape, dtype),
            node_id=name,
        )
        return

    # ``Tensor.new_zeros(shape)`` / ``new_full(shape, fill)`` use their receiver only as a
    # dtype/device template; receiver values do not participate in the result. Represent the
    # constructed tensor as a scalar plus an explicit broadcast so its FX metadata shape is
    # preserved even when the receiver has an unrelated shape.
    if op_name in ("new_zeros", "new_full"):
        from emmy.compiler.pipeline.passes.frontend.decomposition._broadcast import broadcast_to

        if op_name == "new_full":
            fill = fx_node.args[2] if len(fx_node.args) > 2 else (fx_node.kwargs or {}).get("fill_value")
            if not isinstance(fill, (int, float, bool)):
                raise NotImplementedError(f"aten.new_full requires a static scalar fill value, got {fill!r}")
            scalar_id = input_ids[-1] if input_ids and isinstance(g.nodes[input_ids[-1]].op, ConstantOp) else None
        else:
            fill = 0.0
            scalar_id = None
        if scalar_id is None:
            scalar_name = f"{name}_scalar"
            scalar_id = g.add_node(
                op=ConstantOp(name=scalar_name, value=fill),
                inputs=[],
                output=Tensor(scalar_name, (1,), dtype),
                node_id=scalar_name,
            )
        node_map[name] = broadcast_to(g, scalar_id, shape).id
        return

    # Static ``roll`` and ``select`` are coordinate maps. Keeping them in the tensor
    # dialect preserves their layout meaning and avoids inventing elementwise fallbacks.
    if op_name == "roll":
        if not input_ids:
            raise ValueError("aten.roll requires one resolved tensor input")
        source = fx_node.args[0] if fx_node.args else None
        source_shape = _static_fx_shape(source)
        if source_shape is None:
            raise NotImplementedError("aten.roll requires a static input shape")
        raw_shifts = fx_node.args[1] if len(fx_node.args) > 1 else None
        raw_dims = fx_node.args[2] if len(fx_node.args) > 2 else None
        shifts = (raw_shifts,) if isinstance(raw_shifts, int) and not isinstance(raw_shifts, bool) else raw_shifts
        dims = (raw_dims,) if isinstance(raw_dims, int) and not isinstance(raw_dims, bool) else raw_dims
        if not isinstance(shifts, (list, tuple)) or not all(isinstance(value, int) and not isinstance(value, bool) for value in shifts):
            raise NotImplementedError("aten.roll requires static integer shifts")
        if not isinstance(dims, (list, tuple)) or not all(isinstance(value, int) and not isinstance(value, bool) for value in dims):
            raise NotImplementedError("aten.roll requires static integer dimensions")
        if len(shifts) != len(dims):
            raise ValueError(f"aten.roll shifts/dimensions length mismatch: {len(shifts)} != {len(dims)}")
        if len(dims) != 1:
            raise NotImplementedError("aten.roll affine lowering requires exactly one dimension")

        dim = dims[0] if dims[0] >= 0 else len(source_shape) + dims[0]
        if not 0 <= dim < len(source_shape):
            raise ValueError(f"aten.roll dimension is out of range for rank {len(source_shape)}: {dims[0]}")
        extent = source_shape[dim]
        if extent <= 0:
            raise NotImplementedError("aten.roll requires a positive static input extent")
        shift = shifts[0] % extent
        identity = tuple(placeholder(axis) for axis in range(len(source_shape)))
        if shift == 0:
            sources = (IndexSource(input_idx=0, coord_map=identity),)
        else:
            tail_coords = list(identity)
            tail_coords[dim] = placeholder(dim) + Literal(extent - shift, "int")
            head_coords = list(identity)
            head_coords[dim] = placeholder(dim) - Literal(shift, "int")
            sources = (
                IndexSource(
                    input_idx=0,
                    coord_map=tuple(tail_coords),
                    select=placeholder(dim).lt(Literal(shift, "int")),
                ),
                IndexSource(input_idx=0, coord_map=tuple(head_coords)),
            )
        node_map[name] = g.add_node(
            op=IndexMapOp(out_shape=shape, sources=sources),
            inputs=input_ids[:1],
            output=Tensor(name, shape, dtype),
            node_id=name,
        )
        return

    if op_name == "select":
        if not input_ids:
            raise ValueError("aten.select requires one resolved tensor input")
        source = fx_node.args[0] if fx_node.args else None
        source_shape = _static_fx_shape(source)
        raw_dim = fx_node.args[1] if len(fx_node.args) > 1 else None
        raw_index = fx_node.args[2] if len(fx_node.args) > 2 else None
        if source_shape is None:
            raise NotImplementedError("aten.select requires a static input shape")
        if not isinstance(raw_dim, int) or isinstance(raw_dim, bool):
            raise NotImplementedError("aten.select requires a static integer dimension")
        if not isinstance(raw_index, int) or isinstance(raw_index, bool):
            raise NotImplementedError("aten.select requires a static integer index")
        dim = raw_dim if raw_dim >= 0 else len(source_shape) + raw_dim
        if not 0 <= dim < len(source_shape):
            raise ValueError(f"aten.select dimension {raw_dim} is out of range for rank {len(source_shape)}")
        index = raw_index if raw_index >= 0 else source_shape[dim] + raw_index
        if not 0 <= index < source_shape[dim]:
            raise ValueError(f"aten.select index {raw_index} is out of range for extent {source_shape[dim]}")
        coord_map = tuple(
            Literal(index, "int") if axis == dim else placeholder(axis if axis < dim else axis - 1) for axis in range(len(source_shape))
        )
        node_map[name] = g.add_node(
            op=IndexMapOp(out_shape=shape, sources=(IndexSource(input_idx=0, coord_map=coord_map),)),
            inputs=input_ids[:1],
            output=Tensor(name, shape, dtype),
            node_id=name,
        )
        return

    # ``fill_`` returns destination-shaped scalar values. When a later live read observes
    # the written storage, a static slice/select view of a local value can additionally
    # version its base through a bounded two-source IndexMap. All other alias forms fail closed.
    if op_name == "fill_":
        from emmy.compiler.pipeline.passes.frontend.decomposition._broadcast import broadcast_to

        if len(input_ids) < 2:
            raise ValueError("aten.fill_ requires resolved destination and fill tensors")
        filled_id = broadcast_to(g, input_ids[1], shape).id
        node_map[name] = filled_id

        observable = _observable_alias_read_after_write(fx_node)
        if observable is None:
            return
        reassembly = _reassemblable_local_affine_fill(fx_node)
        if reassembly is None:
            later, source = observable
            raise NotImplementedError(
                "aten.fill_ observable alias mutation is unsupported: "
                f"later live node {later.name!r} reads original destination alias {source.name!r}; "
                "functional fill_ is supported only through its returned value"
            )

        root, offsets, extents, view_axes = reassembly
        previous_base = node_map.get(root.name)
        if not isinstance(previous_base, str):
            raise ValueError(f"aten.fill_ local destination root {root.name!r} did not resolve to a tensor")
        base = g.nodes[previous_base].output
        base_shape = tuple(base.shape)
        view_shape = [0] * sum(axis is not None for axis in view_axes)
        for root_axis, view_axis in enumerate(view_axes):
            if view_axis is not None:
                view_shape[view_axis] = extents[root_axis]
        if len(base_shape) != len(offsets) or tuple(shape) != tuple(view_shape):
            raise ValueError(
                f"aten.fill_ local view shape mismatch: base={base_shape}, destination={tuple(view_shape)}, fill={tuple(shape)}"
            )
        if any(extent == 0 for extent in extents):
            return

        select = Literal(1, "int")
        for axis, (offset, extent) in enumerate(zip(offsets, extents, strict=True)):
            coord = placeholder(axis)
            in_axis = BinaryExpr(">=", coord, Literal(offset, "int"))
            in_axis = BinaryExpr("&&", in_axis, coord.lt(Literal(offset + extent, "int")))
            select = BinaryExpr("&&", select, in_axis)
        source_coords: list[Any] = [Literal(0, "int")] * len(view_shape)
        for root_axis, view_axis in enumerate(view_axes):
            if view_axis is None:
                continue
            coord = placeholder(root_axis)
            offset = offsets[root_axis]
            source_coord = coord - Literal(offset, "int") if offset else coord
            source_coords[view_axis] = TernaryExpr(select, source_coord, Literal(0, "int"))

        identity = tuple(placeholder(axis) for axis in range(len(base_shape)))
        update_name = f"{name}_base"
        node_map[root.name] = g.add_node(
            op=IndexMapOp(
                out_shape=base_shape,
                sources=(
                    IndexSource(input_idx=0, coord_map=tuple(source_coords), select=select),
                    IndexSource(input_idx=1, coord_map=identity),
                ),
            ),
            inputs=[filled_id, previous_base],
            output=Tensor(update_name, base_shape, base.dtype),
            node_id=update_name,
        )
        return

    # ``copy_(dest, src)`` returns destination-shaped source values. A static slice of a local
    # constructor can also update its base functionally: one IndexMap source supplies the written
    # region and the previous base supplies the rest. Broader alias mutation remains fail-closed.
    if op_name == "copy_":
        from emmy.compiler.pipeline.passes.frontend.decomposition._broadcast import broadcast_to

        observable = _observable_alias_read_after_write(fx_node)
        reassembly = _reassemblable_local_slice_copy(fx_node) if observable is not None else None
        if observable is not None and reassembly is None:
            later, source = observable
            raise NotImplementedError(
                "aten.copy_ observable alias mutation is unsupported: "
                f"later live node {later.name!r} reads original destination alias {source.name!r}; "
                "functional copy_ is supported only through its returned value"
            )
        if len(input_ids) < 2:
            raise ValueError("aten.copy_ requires resolved destination and source tensors")
        copied = broadcast_to(g, input_ids[1], shape)
        if copied.output.dtype == dtype:
            copied_id = copied.id
        else:
            coord_map = tuple(placeholder(axis) for axis in range(len(shape)))
            copied_id = g.add_node(
                op=IndexMapOp(out_shape=shape, sources=(IndexSource(input_idx=0, coord_map=coord_map),)),
                inputs=[copied],
                output=Tensor(name, shape, dtype),
                node_id=name,
            )
        node_map[name] = copied_id

        if reassembly is not None:
            root, offsets, extents = reassembly
            previous_base = node_map.get(root.name)
            if not isinstance(previous_base, str):
                raise ValueError(f"aten.copy_ local destination root {root.name!r} did not resolve to a tensor")
            if any(extent == 0 for extent in extents):
                # An empty destination is a no-op on the base. Keep ``node_map[name]``
                # above for the copy_ return, but do not emit an IndexMap whose eager
                # source loads would index the zero-sized copied tensor.
                return
            base = g.nodes[previous_base].output
            base_shape = tuple(base.shape)
            if len(base_shape) != len(offsets) or tuple(shape) != tuple(extents):
                raise ValueError(
                    f"aten.copy_ local slice shape mismatch: base={base_shape}, destination={tuple(extents)}, copy={tuple(shape)}"
                )

            select = Literal(1, "int")
            source_coords = []
            for axis, (offset, extent) in enumerate(zip(offsets, extents, strict=True)):
                coord = placeholder(axis)
                in_axis = BinaryExpr(">=", coord, Literal(offset, "int"))
                in_axis = BinaryExpr("&&", in_axis, coord.lt(Literal(offset + extent, "int")))
                select = BinaryExpr("&&", select, in_axis)
                source_coord = coord - Literal(offset, "int") if offset else coord
                source_coords.append(TernaryExpr(select, source_coord, Literal(0, "int")))

            identity = tuple(placeholder(axis) for axis in range(len(base_shape)))
            update_name = f"{name}_base"
            updated_base = g.add_node(
                op=IndexMapOp(
                    out_shape=base_shape,
                    sources=(
                        IndexSource(input_idx=0, coord_map=tuple(source_coords), select=select),
                        IndexSource(input_idx=1, coord_map=identity),
                    ),
                ),
                inputs=[copied_id, previous_base],
                output=Tensor(update_name, base_shape, base.dtype),
                node_id=update_name,
            )
            node_map[root.name] = updated_base
        return

    # Data-dependent selection is a ternary elementwise op. ``masked_fill(self, mask, fill)``
    # is exactly ``where(mask, fill, self)``; spelling it as arithmetic would turn unselected
    # infinities into NaNs through ``0 * inf``.
    if op_name in ("where", "masked_fill"):
        from emmy.compiler.pipeline.passes.frontend.decomposition._broadcast import broadcast_to

        if op_name == "masked_fill":
            if len(input_ids) < 3:
                raise ValueError("aten.masked_fill requires resolved self, mask, and fill inputs")
            select_ids = [input_ids[1], input_ids[2], input_ids[0]]
        else:
            if len(input_ids) < 3:
                raise ValueError("aten.where requires resolved condition, true, and false inputs")
            select_ids = input_ids[:3]
        nid = g.add_node(
            op=ElementwiseOp(op="where"),
            inputs=[broadcast_to(g, inp, shape) for inp in select_ids],
            output=Tensor(name, shape, dtype),
            node_id=name,
        )
        node_map[name] = nid
        return

    # --- Elementwise ops ---
    # Torch's aten-level short names (``sub`` / ``mul`` / ``div`` / ``neg``)
    # get translated to numpy-style long names here so the rest of the
    # pipeline can read our ``ElementwiseOp.op.name`` as a numpy attribute
    # (``np.subtract`` / ``np.multiply`` / …) without further aliasing.
    # Names that already match numpy (``add`` / ``mod`` / ``pow`` / ``exp`` /
    # ``tanh`` / ``abs`` / ``sqrt`` / ``reciprocal`` / …) pass through.
    _ATEN_TO_NUMPY = {
        "sub": "subtract",
        "mul": "multiply",
        "div": "divide",
        "neg": "negative",
        # In-place variants (``*=`` / ``+=`` / … — e.g. Gemma's
        # ``hidden_states *= self.layer_scalar``). The trace is functional, so
        # they lower to the same out-of-place numpy op.
        "mul_": "multiply",
        "add_": "add",
        "sub_": "subtract",
        "div_": "divide",
        # Comparisons and bool combines — the bool-output mask construction in
        # whole-model traces (the explicit attention-mask subgraph). aten spells
        # the operator combines as dunders (``mask | other`` → ``aten.__or__``).
        "gt": "greater",
        "lt": "less",
        "ge": "greater_equal",
        "le": "less_equal",
        "eq": "equal",
        "ne": "not_equal",
        "__or__": "bitwise_or",
        "__and__": "bitwise_and",
    }
    _ELEMENTWISE_SOURCES = frozenset(_ATEN_TO_NUMPY) | {
        "add",
        "exp",
        "rsqrt",
        "reciprocal",
        "pow",
        "silu",
        "softplus",
        "relu",
        "tanh",
        "abs",
        "sigmoid",
        "gelu",
        "erf",
        # Binary elementwise min/max (``torch.maximum`` / ``torch.minimum``). NOT the
        # reductions: ``amax`` (below) reduces an axis; ``maximum`` compares two operands
        # elementwise — the ``clamp`` decomposition emits these too.
        "maximum",
        "minimum",
    }
    # --- Clamp ---
    # ``aten.clamp(x, min, max)`` / ``clamp_min`` / ``clamp_max`` decompose to the
    # elementwise ``maximum`` (lower bound) / ``minimum`` (upper bound) chain with the
    # bound constants; a ``None`` bound skips that side (gpt-oss clamps ``gate`` with
    # ``max=`` only). Scalar bounds only — a tensor bound raises rather than mis-clamping.
    if op_name in ("clamp", "clamp_min", "clamp_max"):
        from emmy.compiler.pipeline.passes.frontend.decomposition._broadcast import broadcast_to

        raw = list(fx_node.args[1:]) + [None, None]
        kw = fx_node.kwargs or {}
        if op_name == "clamp":
            lo, hi = kw.get("min", raw[0]), kw.get("max", raw[1])
        elif op_name == "clamp_min":
            lo, hi = kw.get("min", raw[0]), None
        else:
            lo, hi = None, kw.get("max", raw[0])
        for bound in (lo, hi):
            if bound is not None and not isinstance(bound, (int, float)):
                raise NotImplementedError(f"aten.{op_name}: only scalar (or None) bounds are supported, got {bound!r}")
        cur = input_ids[0] if input_ids else None
        if cur is None:
            return
        sides = [(b, ew) for b, ew in ((lo, "maximum"), (hi, "minimum")) if b is not None]
        if not sides:
            node_map[name] = cur  # clamp(None, None) is the identity
            return
        for j, (bound, ew) in enumerate(sides):
            const_name = f"{name}_bound{j}"
            const_id = g.add_node(
                op=ConstantOp(name=const_name, value=float(bound)),
                inputs=[],
                output=Tensor(const_name, (1,), dtype),
                node_id=const_name,
            )
            last = j == len(sides) - 1
            out_name = name if last else f"{name}_lo"
            cur = g.add_node(
                op=ElementwiseOp(op=ew),
                inputs=[broadcast_to(g, cur, shape), broadcast_to(g, const_id, shape)],
                output=Tensor(out_name, shape, dtype),
                node_id=out_name,
            )
        node_map[name] = cur
        return

    if op_name in _ELEMENTWISE_SOURCES:
        from emmy.compiler.pipeline.passes.frontend.decomposition._broadcast import broadcast_to

        if op_name == "softplus":
            beta = fx_node.args[1] if len(fx_node.args) > 1 else (fx_node.kwargs or {}).get("beta", 1)
            threshold = fx_node.args[2] if len(fx_node.args) > 2 else (fx_node.kwargs or {}).get("threshold", 20)
            if float(beta) != 1.0 or float(threshold) != 20.0:
                raise NotImplementedError("aten.softplus currently supports only beta=1 and threshold=20")
            input_ids = input_ids[:1]
        canonical = _ATEN_TO_NUMPY.get(op_name, op_name)
        # Disambiguate gelu's tanh approximation from the default erf form
        # — the FX node carries ``kwargs={'approximate': 'tanh'}`` only in
        # that case, and the decomposition rule keys on op name.
        if canonical == "gelu" and (fx_node.kwargs or {}).get("approximate") == "tanh":
            canonical = "gelu_tanh"
        bc_ids = [broadcast_to(g, inp, shape) for inp in input_ids[:2]]
        nid = g.add_node(
            op=ElementwiseOp(op=canonical),
            inputs=bc_ids,
            output=Tensor(name, shape, dtype),
            node_id=name,
        )
        node_map[name] = nid
        return

    # --- Linear ---
    if op_name == "linear":
        has_bias = len(input_ids) > 2 and input_ids[2] in g.nodes
        nid = g.add_node(
            op=LinearOp(has_bias=has_bias),
            inputs=input_ids[:3] if has_bias else input_ids[:2],
            output=Tensor(name, shape, dtype),
            node_id=name,
        )
        node_map[name] = nid
        return

    # --- Matmul ---
    if op_name in ("mm", "matmul", "bmm"):
        nid = g.add_node(
            op=MatmulOp(),
            inputs=input_ids[:2],
            output=Tensor(name, shape, dtype),
            node_id=name,
        )
        node_map[name] = nid
        return

    if op_name == "addmm":
        nid = g.add_node(
            op=MatmulOp(has_bias=True),
            inputs=[input_ids[1], input_ids[2], input_ids[0]] if len(input_ids) >= 3 else input_ids,
            output=Tensor(name, shape, dtype),
            node_id=name,
        )
        node_map[name] = nid
        return

    # --- SDPA ---
    if op_name == "scaled_dot_product_attention":
        # args: (Q, K, V, attn_mask, dropout_p, is_causal, scale). ``attn_mask`` is an
        # additive float bias added to the QK^T scores — HF passes its causal
        # mask this way (a precomputed (1,1,S,S) tensor) rather than via the
        # ``is_causal`` flag. Thread it through as a 4th SdpaOp input so the
        # decomposition can fold it into the scores; dropping it (the old
        # ``input_ids[:3]``) silently turned masked attention into full
        # bidirectional attention.
        is_causal = False
        for a in (*fx_node.args[3:], *(fx_node.kwargs or {}).values()):
            if isinstance(a, bool):
                is_causal = a
                break
        # ``scale``: positional slot 6 (after dropout_p at 4 — never captured, so a
        # bare float at 4 is dropout, at 6 is scale) or the kwarg. ``None`` = torch's
        # ``1/sqrt(head_dim)`` default. Dropping an explicit scale (Gemma-nano passes
        # ``scale=1.0``) silently re-scales the attention logits.
        scale = (fx_node.kwargs or {}).get("scale")
        if scale is None and len(fx_node.args) > 6 and isinstance(fx_node.args[6], (int, float)) and not isinstance(fx_node.args[6], bool):
            scale = fx_node.args[6]
        sdpa_inputs = list(input_ids[:3])
        mask_arg = fx_node.args[3] if len(fx_node.args) > 3 else (fx_node.kwargs or {}).get("attn_mask")
        if mask_arg is not None and hasattr(mask_arg, "name") and mask_arg.name in node_map:
            sdpa_inputs.append(node_map[mask_arg.name])
        nid = g.add_node(
            op=SdpaOp(is_causal=is_causal, scale=None if scale is None else float(scale)),
            inputs=sdpa_inputs,
            output=Tensor(name, shape, dtype),
            node_id=name,
        )
        node_map[name] = nid
        return

    # --- Reductions ---
    # Reductions are always emitted as keepdim (rank-preserving). If the
    # traced op was non-keepdim, a squeeze IndexMapOp is inserted afterwards
    # so the downstream graph sees the correct (dropped-axis) shape while the
    # ReduceOp itself stays rank-preserving.
    # Torch reduction names pass through to ``ReduceOp`` as-is; ``mean`` is
    # the exception that lands as ``MeanOp`` (decomposition splits it into
    # sum + div). Keeping torch's spelling — ``amax`` stays ``amax``, not
    # mapped to ``max`` — avoids a needless name table. (``maximum`` is NOT a
    # reduction — the binary elementwise max is handled above.)
    if op_name in ("sum", "amax", "mean"):
        axis = _get_reduce_axis(fx_node)
        x_shape = tuple(g.nodes[input_ids[0]].output.shape) if input_ids else ()
        keepdim_shape = _keepdim_shape(x_shape, axis)
        if op_name == "mean":
            red_node_op = MeanOp(axis=axis)
        else:
            red_node_op = ReduceOp(op=ElementwiseImpl(op_name), axis=axis)

        if tuple(shape) == keepdim_shape:
            nid = g.add_node(op=red_node_op, inputs=input_ids[:1], output=Tensor(name, shape, dtype), node_id=name)
        else:
            # Emit keepdim reduce + squeeze IndexMapOp.
            red_id = g.add_node(op=red_node_op, inputs=input_ids[:1], output=Tensor(f"{name}_keepdim", keepdim_shape, dtype))
            nid = g.add_node(
                op=_squeeze_indexmap(keepdim_shape, shape, axis),
                inputs=[red_id],
                output=Tensor(name, shape, dtype),
                node_id=name,
            )
        node_map[name] = nid
        return

    # --- Transpose ---
    if op_name == "transpose":
        dim0 = fx_node.args[1] if len(fx_node.args) > 1 else 0
        dim1 = fx_node.args[2] if len(fx_node.args) > 2 else 1
        nid = g.add_node(
            op=TransposeOp(axes=(dim0, dim1)),
            inputs=input_ids[:1],
            output=Tensor(name, shape, dtype),
            node_id=name,
        )
        node_map[name] = nid
        return

    if op_name == "t":
        nid = g.add_node(
            op=TransposeOp(axes=(1, 0)),
            inputs=input_ids[:1],
            output=Tensor(name, shape, dtype),
            node_id=name,
        )
        node_map[name] = nid
        return

    # --- Reshape / view ---
    if op_name in ("view", "reshape", "_unsafe_view", "flatten"):
        if op_name == "flatten":
            # aten.flatten carries start/end axes rather than the resulting
            # dimensions. FakeTensor metadata has already calculated the exact
            # output shape, including any symbolic dimensions.
            new_shape = _dim_tuple_to_op_shape(shape)
        else:
            new_shape = fx_node.args[1] if len(fx_node.args) > 1 else shape
            if isinstance(new_shape, (list, tuple)):
                new_shape = _op_shape(new_shape, sym_rename)
        nid = g.add_node(
            op=ReshapeOp(shape=new_shape),
            inputs=input_ids[:1],
            output=Tensor(name, shape, dtype),
            node_id=name,
        )
        node_map[name] = nid
        return

    # --- Unsqueeze ---
    if op_name == "unsqueeze":
        dim = fx_node.args[1] if len(fx_node.args) > 1 and isinstance(fx_node.args[1], int) else 0
        nid = g.add_node(
            op=UnsqueezeOp(dim=dim),
            inputs=input_ids[:1],
            output=Tensor(name, shape, dtype),
            node_id=name,
        )
        node_map[name] = nid
        return

    # --- Expand ---
    if op_name == "expand":
        # ``expand`` *broadcasts* size-1 dims (the element count changes), so it
        # is a broadcast — NOT a reshape. Routing it through ``ReshapeOp`` makes
        # the decomposition treat the broadcast dim as a real flat-offset stride
        # (repeat_kv's ``(kv,1,...)->(kv,n_rep,...)`` then reshape gives a wrong
        # ``q_head % kv`` index instead of ``q_head // n_rep``). Emit a broadcast
        # IndexMapOp so each broadcast dim reads coord 0.
        from emmy.compiler.pipeline.passes.frontend.decomposition._broadcast import broadcast_to

        node_map[name] = broadcast_to(g, input_ids[0], shape).id
        return

    if op_name == "repeat_interleave":
        if not input_ids:
            raise ValueError("aten.repeat_interleave input did not resolve to a tensor")
        kw = fx_node.kwargs or {}
        raw_repeats = kw.get("repeats", fx_node.args[1] if len(fx_node.args) > 1 else None)
        raw_dim = kw.get("dim", fx_node.args[2] if len(fx_node.args) > 2 else None)
        if not isinstance(raw_repeats, int) or isinstance(raw_repeats, bool) or raw_repeats <= 0:
            raise NotImplementedError("aten.repeat_interleave requires a positive static integer repeat count")
        if not isinstance(raw_dim, int) or isinstance(raw_dim, bool):
            raise NotImplementedError("aten.repeat_interleave requires a constant integer dimension")
        source_shape = tuple(g.nodes[input_ids[0]].output.shape)
        rank = len(source_shape)
        dim = raw_dim if raw_dim >= 0 else rank + raw_dim
        if not 0 <= dim < rank:
            raise ValueError(f"aten.repeat_interleave dimension {raw_dim} is out of range for rank-{rank} input")
        coord_map = tuple(placeholder(axis) / Literal(raw_repeats, "int") if axis == dim else placeholder(axis) for axis in range(rank))
        nid = g.add_node(
            op=IndexMapOp(out_shape=shape, sources=(IndexSource(input_idx=0, coord_map=coord_map),)),
            inputs=input_ids[:1],
            output=Tensor(name, shape, dtype),
            node_id=name,
        )
        node_map[name] = nid
        return

    if op_name == "stack":
        raw_tensors = fx_node.args[0] if fx_node.args else ()
        if not isinstance(raw_tensors, (list, tuple)) or not raw_tensors:
            raise ValueError("aten.stack requires a non-empty static tensor list")
        tensor_ids = []
        for tensor_arg in raw_tensors:
            ref = node_map.get(getattr(tensor_arg, "name", None))
            if not isinstance(ref, str):
                raise ValueError("aten.stack input did not resolve to a tensor")
            tensor_ids.append(ref)
        kw = fx_node.kwargs or {}
        raw_dim = kw.get("dim", fx_node.args[1] if len(fx_node.args) > 1 else 0)
        if not isinstance(raw_dim, int) or isinstance(raw_dim, bool):
            raise NotImplementedError("aten.stack requires a constant integer dimension")
        output_rank = len(shape)
        dim = raw_dim if raw_dim >= 0 else output_rank + raw_dim
        if not 0 <= dim < output_rank:
            raise ValueError(f"aten.stack dimension {raw_dim} is out of range for rank-{output_rank} output")
        expected_input_shape = tuple(shape[:dim]) + tuple(shape[dim + 1 :])
        for tensor_id in tensor_ids:
            if tuple(g.nodes[tensor_id].output.shape) != expected_input_shape:
                raise ValueError("aten.stack inputs must have the output shape with the stack axis removed")
        sources = []
        coord_map = tuple(placeholder(axis if axis < dim else axis + 1) for axis in range(output_rank - 1))
        for index in range(len(tensor_ids)):
            select = BinaryExpr("==", placeholder(dim), Literal(index, "int")) if index < len(tensor_ids) - 1 else None
            sources.append(IndexSource(input_idx=index, coord_map=coord_map, select=select))
        nid = g.add_node(
            op=IndexMapOp(out_shape=shape, sources=tuple(sources)),
            inputs=tensor_ids,
            output=Tensor(name, shape, dtype),
            node_id=name,
        )
        node_map[name] = nid
        return

    # --- Squeeze / permute ---
    if op_name in ("squeeze", "permute"):
        # ``ReshapeOp.shape`` is ``tuple[int | str, ...]`` — unwrap Dim wrappers
        # so the op-level field carries the raw int / symbolic-name form.
        op_shape = _dim_tuple_to_op_shape(shape)
        nid = g.add_node(
            op=ReshapeOp(shape=op_shape),
            inputs=input_ids[:1],
            output=Tensor(name, shape, dtype),
            node_id=name,
        )
        node_map[name] = nid
        return

    # --- Pass-through ---
    # ``contiguous`` / ``clone`` / ``detach`` / ``alias`` are pure aliases. ``to`` and
    # ``type_as`` are aliases ONLY when they do not actually change dtype — when they do,
    # aliasing silently drops a real narrowing and the wider dtype propagates forward.
    #
    # That is not hypothetical: Gemma's RMSNorm is
    # ``self._norm(x.float()) * w.float()`` then ``.type_as(x)``. The f32 the statistic is
    # computed in reaches the graph anyway (the traced scalar constants are f32, so
    # ``f16 ** f32 -> f32`` promotes the whole chain), but dropping the closing ``type_as``
    # left the norm OUTPUT f32 too. Every consumer then became a mixed f32xf16 contraction,
    # which is not offered the staged (``d2/tma``) transports — the gemma-4 prefill
    # norm->qkv projections deployed at 147-174 us each on a spilling ``w1x1`` tile instead
    # of the ~31-51 us their pure-f16 goldens record.
    #
    # Emit the narrowing as an identity ``IndexMapOp`` (read each element, write it) whose
    # output Tensor carries the target dtype — the same mechanism broadcast already uses,
    # and the conversion falls out of the typed store. Keeping the statistic itself in f32
    # is deliberate and must not change: squaring gemma activations in f16 overflows above
    # |x| = 256 (measured: max|err| 60.7 at peak 300 vs HF), so only the OUTPUT narrows.
    if op_name in ("to", "contiguous", "_assert_tensor_metadata", "clone", "detach", "alias", "type_as"):
        if input_ids:
            src = g.nodes.get(input_ids[0])
            if op_name in ("to", "type_as") and src is not None and dtype != src.output.dtype:
                coord_map = tuple(placeholder(d) for d in range(len(shape)))
                nid = g.add_node(
                    op=IndexMapOp(out_shape=tuple(shape), sources=(IndexSource(input_idx=0, coord_map=coord_map),)),
                    inputs=input_ids[:1],
                    output=Tensor(name, shape, dtype),
                    node_id=name,
                )
                node_map[name] = nid
                return
            node_map[name] = input_ids[0]
        return

    # --- Slice ---
    if op_name == "slice":
        if input_ids:
            # Record dim/start from the raw FX args: ``aten.slice.Tensor(self,
            # dim, start, end)`` may carry ``start=None`` (``x[:, :s]``) or a
            # SymInt ``end`` — ``_resolve_inputs`` drops both, leaving the
            # surviving ConstantOp inputs positionally ambiguous. A non-int
            # dim/start (a SymInt slice origin) stays ``None`` so the
            # decomposition rule fails loudly instead of mis-slicing.
            args = fx_node.args
            dim_raw = args[1] if len(args) > 1 else 0
            start_raw = args[2] if len(args) > 2 else None
            dim = dim_raw if isinstance(dim_raw, int) and not isinstance(dim_raw, bool) else None
            start = 0 if start_raw is None else (start_raw if isinstance(start_raw, int) and not isinstance(start_raw, bool) else None)
            nid = g.add_node(
                op=SliceOp(shape=_dim_tuple_to_op_shape(shape), dim=dim, start=start),
                inputs=input_ids,
                output=Tensor(name, shape, dtype),
                node_id=name,
            )
            node_map[name] = nid
        return

    # --- Cat ---
    if op_name == "cat":
        nid = g.add_node(
            op=CatOp(),
            inputs=input_ids,
            output=Tensor(name, shape, dtype),
            node_id=name,
        )
        node_map[name] = nid
        return

    # --- Gather ---
    if op_name in ("index_select", "gather", "embedding", "index"):
        axis = fx_node.args[1] if len(fx_node.args) > 1 and isinstance(fx_node.args[1], int) else 0
        # ``index_select(input, dim, index)`` / ``gather(input, dim, index)``
        # pass ``dim`` as a Python int in args[1] — ``_resolve_inputs`` has
        # captured it as a ConstantOp at input_ids[1]. ``GatherOp.axis``
        # already carries the dim value, so drop the spurious input or the
        # lift sees a 3-input gather (and picks the wrong node as idx).
        # ``embedding`` / simple ``index`` keep the original args (args[1]
        # contains the index tensor). Advanced multi-axis indexing is rejected.
        if op_name in ("index_select", "gather") and len(input_ids) >= 3 and isinstance(fx_node.args[1], int):
            input_ids = [input_ids[0], *input_ids[2:]]
        if op_name == "index":
            raw_indices = fx_node.args[1] if len(fx_node.args) > 1 else ()
            if not isinstance(raw_indices, (list, tuple)) or len(raw_indices) != 1 or raw_indices[0] is None:
                raise NotImplementedError("aten.index currently supports one tensor index on axis 0")
        nid = g.add_node(
            op=GatherOp(axis=axis),
            inputs=input_ids,
            output=Tensor(name, shape, dtype),
            node_id=name,
        )
        node_map[name] = nid
        return

    # --- Fused frontend ops (decomposed in later passes) ---
    if op_name == "rms_norm":
        # aten.rms_norm: (x, normalized_shape, weight [, eps]). The tracer
        # drops normalized_shape (a list literal), leaving (x, weight) plus
        # an optional eps ConstantOp. eps is the op's own field, not a graph
        # input, so we peel it off here.
        eps_value = 1e-6
        if len(input_ids) >= 3:
            eps_node = g.nodes.get(input_ids[2])
            if eps_node and isinstance(eps_node.op, ConstantOp) and isinstance(eps_node.op.value, (int, float)):
                eps_value = float(eps_node.op.value)
                input_ids = input_ids[:2]
        nid = g.add_node(op=RmsNormOp(eps=eps_value), inputs=input_ids, output=Tensor(name, shape, dtype), node_id=name)
        node_map[name] = nid
        return

    if op_name == "layer_norm":
        # aten.layer_norm: (x, normalized_shape, weight?, bias?, eps, cudnn_enable).
        # The tracer drops normalized_shape (a list literal), None affine
        # params, and the bool flag, leaving (x [, weight [, bias]]) plus a
        # trailing eps ConstantOp. eps is the op's own field, not a graph
        # input, so we peel it off here. The affine params are real
        # parameter ConstantOps (value=None, source_path set), so the
        # scalar-value check can't mistake one for eps.
        eps_value = 1e-5
        if len(input_ids) >= 2:
            eps_node = g.nodes.get(input_ids[-1])
            if eps_node and isinstance(eps_node.op, ConstantOp) and isinstance(eps_node.op.value, (int, float)):
                eps_value = float(eps_node.op.value)
                input_ids = input_ids[:-1]
        nid = g.add_node(op=LayerNormOp(eps=eps_value), inputs=input_ids, output=Tensor(name, shape, dtype), node_id=name)
        node_map[name] = nid
        return

    if op_name == "softmax":
        # aten.softmax.int: (x, dim_const). dim becomes the op's field.
        axis: int = -1
        if len(input_ids) >= 2:
            dim_node = g.nodes.get(input_ids[1])
            if dim_node and isinstance(dim_node.op, ConstantOp) and isinstance(dim_node.op.value, (int, float)):
                axis = int(dim_node.op.value)
                input_ids = input_ids[:1]
        nid = g.add_node(op=SoftmaxOp(axis=axis), inputs=input_ids, output=Tensor(name, shape, dtype), node_id=name)
        node_map[name] = nid
        return

    if op_name == "dropout":
        # Inference-time dropout is the identity. Drop the p / training args and
        # lower it as a `copy` passthrough so it becomes a no-op in the graph.
        input_ids = input_ids[:1]
        op_name = "copy"

    # --- Fallback: unknown op becomes ElementwiseOp by torch-aten name ---
    logger.debug("Fallback elementwise for %s (%s)", op_name, fx_node.target)
    if input_ids:
        from emmy.compiler.pipeline.passes.frontend.decomposition._broadcast import broadcast_to

        try:
            input_ids = [broadcast_to(g, inp, shape) for inp in input_ids]
        except ValueError as exc:
            raise ValueError(f"cannot map fallback aten.{op_name} as elementwise for output shape {shape}") from exc
        nid = g.add_node(
            op=ElementwiseOp(op=op_name),
            inputs=input_ids,
            output=Tensor(name, shape, dtype),
            node_id=name,
        )
        node_map[name] = nid
