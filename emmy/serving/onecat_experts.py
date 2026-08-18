"""Guarded Emmy adapter for pinned 1Cat routing and retained compact experts."""

from __future__ import annotations

import importlib
import inspect
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from math import prod
from types import ModuleType
from typing import Any

from emmy.compiler.loader import onecat_sm70_experts as expert_loader
from emmy.serving.deepseek_experts import EXPERTS, HIDDEN, TOP_K, VOCAB

logger = logging.getLogger(__name__)

PROFILE_ROWS = (1, 2, 4, 8, 16, 128, 1024, 4096)
_CAPACITY = PROFILE_ROWS[-1]
# Route IDs remain exact. The released binary and Emmy use different fast
# exp/log implementations, so payloads admit a one-ULP-class FP32 delta.
_ROUTE_ATOL = 1e-6
_EXPERT_TOL = 5e-2


@dataclass
class _Program:
    runtime: Any
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    verified: bool = False
    symbolic: bool = False


def _build_route(rows: int, kind: str) -> _Program:
    from emmy.serving.deepseek_experts import trace_deepseek_route
    from emmy.serving.external import load_external_program

    symbolic = rows not in PROFILE_ROWS
    runtime, plan = load_external_program(
        trace_deepseek_route(rows=128 if symbolic else rows, kind=kind, symbolic=symbolic),
        symbolic_values={"num_tokens": _CAPACITY} if symbolic else None,
    )
    expected = ("router_logits", "bias") if kind == "learned" else ("router_logits", "table", "input_ids")
    if tuple(plan.inputs) != expected or tuple(plan.outputs) != ("weights", "ids"):
        raise RuntimeError(f"1Cat {kind} route ABI changed: {plan.inputs!r} -> {plan.outputs!r}")
    return _Program(runtime, tuple(plan.inputs), tuple(plan.outputs), symbolic=symbolic)


def _build_experts(rows: int) -> _Program:
    from emmy.serving.deepseek_experts import trace_deepseek_experts
    from emmy.serving.external import load_external_program

    del rows
    runtime, plan = load_external_program(
        trace_deepseek_experts(rows=128, symbolic=True),
        symbolic_values={"num_tokens": PROFILE_ROWS[-1]},
    )
    expected = ("x", "route_weights", "route_ids", "w13", "w2", "w13_scale", "w2_scale")
    if tuple(plan.inputs) != expected or len(plan.outputs) != 1:
        raise RuntimeError(f"1Cat retained expert ABI changed: {plan.inputs!r} -> {plan.outputs!r}")
    decoded = [buffer.name for buffer in plan.buffers if "decoded" in buffer.name]
    full_weight_scratch = []
    for buffer in plan.buffers:
        if buffer.role != "scratch":
            continue
        try:
            numel = prod(buffer.resolve_shape(plan.symbolic_hints))
        except (KeyError, TypeError, ValueError):
            numel = EXPERTS * HIDDEN * expert_loader.INTERMEDIATE
        if numel >= EXPERTS * HIDDEN * expert_loader.INTERMEDIATE:
            full_weight_scratch.append(buffer.name)
    if decoded or full_weight_scratch or len(plan.launches) != 4:
        raise RuntimeError(
            "1Cat retained expert plan violated the compact-storage contract: "
            f"decoded={decoded!r}, full_weight_scratch={full_weight_scratch!r}, launches={len(plan.launches)}"
        )
    return _Program(runtime, tuple(plan.inputs), tuple(plan.outputs), symbolic=True)


def _run(program: _Program, tensors: dict[str, Any], device: Any) -> None:
    import cupy as cp
    import torch

    from emmy.compiler.backend.gpu_lock import gpu_lock

    stream = torch.cuda.current_stream(device)
    with gpu_lock(), cp.cuda.Stream.from_external(stream):
        # CuPy 14 must create DLPack views inside the external-stream context;
        # creating them earlier routes capture through the legacy stream.
        bindings = {name: cp.from_dlpack(tensor) for name, tensor in tensors.items()}
        if program.symbolic:
            rows = tensors.get("x", tensors.get("router_logits"))
            program.runtime.set_sym_values({"num_tokens": int(rows.shape[0])})
        program.runtime.run_once_external(bindings)


def _is_capturing() -> bool:
    import torch

    return torch.cuda.is_current_stream_capturing()


def _is_sm70(*tensors: Any) -> bool:
    import torch

    if not tensors or not all(tensor.is_cuda for tensor in tensors):
        return False
    device = tensors[0].device
    return bool(all(tensor.device == device for tensor in tensors) and torch.cuda.get_device_capability(device) == (7, 0))


class _Adapter:
    def __init__(
        self,
        original_route: Callable,
        original_experts: Callable,
        *,
        build_route: Callable[[int, str], _Program] = _build_route,
        build_experts: Callable[[int], _Program] = _build_experts,
        run: Callable[[_Program, dict[str, Any], Any], None] = _run,
        is_capturing: Callable[[], bool] = _is_capturing,
        platform_supported: Callable[..., bool] = _is_sm70,
    ) -> None:
        self.original_route = original_route
        self.original_experts = original_experts
        self._build_route = build_route
        self._build_experts = build_experts
        self._run = run
        self._is_capturing = is_capturing
        self._platform_supported = platform_supported
        self.routes: dict[tuple[int, str], _Program] = {}
        self.experts: dict[int, _Program] = {}
        self.disabled_routes: set[tuple[int, str]] = set()
        self.disabled_experts: set[int] = set()
        self.lock = threading.RLock()

    def _route_contract(self, router, hidden_states, router_logits, indices_type, input_ids):
        import torch

        rows = int(router_logits.shape[0]) if router_logits.ndim == 2 else -1
        table = getattr(router, "_hash_indices_table", None)
        kind = "hash" if table is not None and input_ids is not None else "learned"
        bias = getattr(router, "e_score_correction_bias", None)
        valid = (
            0 < rows <= _CAPACITY
            and self._platform_supported(hidden_states, router_logits)
            and hidden_states.dtype == torch.float16
            and router_logits.dtype == torch.float32
            and tuple(hidden_states.shape) == (rows, HIDDEN)
            and tuple(router_logits.shape) == (rows, EXPERTS)
            and hidden_states.is_contiguous()
            and router_logits.is_contiguous()
            and getattr(router, "top_k", None) == TOP_K
            and getattr(router, "global_num_experts", None) == EXPERTS
            and getattr(router, "renormalize", None) is True
            and getattr(router, "scoring_func", None) == "sqrtsoftplus"
            and float(getattr(router, "routed_scaling_factor", 0.0)) == 1.5
            and indices_type in {None, torch.int32}
        )
        if kind == "learned":
            valid = (
                valid
                and table is None
                and bias is not None
                and bias.dtype == torch.float32
                and tuple(bias.shape) == (EXPERTS,)
                and bias.device == router_logits.device
                and bias.is_contiguous()
                and self._platform_supported(bias)
            )
        else:
            valid = (
                valid
                and bias is None
                and table.dtype == torch.int32
                and tuple(table.shape) == (VOCAB, TOP_K)
                and input_ids.dtype == torch.int32
                and tuple(input_ids.shape) == (rows,)
                and table.device == input_ids.device == router_logits.device
                and table.is_contiguous()
                and input_ids.is_contiguous()
                and self._platform_supported(table, input_ids)
            )
        if not valid:
            return None
        return rows, kind, bias.data if bias is not None else None, table

    def _expert_contract(self, layer, x, weights, ids) -> expert_loader.ExpertBinding | None:
        return expert_loader.bind_experts(layer, x, weights, ids, self._platform_supported)

    def _route_program(self, rows: int, kind: str, *, capturing: bool) -> _Program | None:
        key = (rows, kind)
        if key in self.disabled_routes:
            return None
        program = self.routes.get(key)
        if program is not None or capturing:
            return program
        try:
            existing = next(
                (
                    candidate
                    for (seen_rows, seen_kind), candidate in self.routes.items()
                    if rows not in PROFILE_ROWS and seen_rows not in PROFILE_ROWS and seen_kind == kind
                ),
                None,
            )
            program = (
                _Program(existing.runtime, existing.inputs, existing.outputs, symbolic=existing.symbolic)
                if existing is not None
                else self._build_route(rows, kind)
            )
        except Exception:  # noqa: BLE001 -- pack/ABI incompatibility permanently falls back
            self.disabled_routes.add(key)
            logger.exception("1Cat %s route M=%d: Emmy pack load failed; retaining 1Cat", kind, rows)
            return None
        self.routes[key] = program
        return program

    def _expert_program(self, rows: int, *, capturing: bool) -> _Program | None:
        if rows in self.disabled_experts:
            return None
        program = self.experts.get(rows)
        if program is not None or capturing:
            return program
        try:
            existing = next(iter(self.experts.values()), None)
            program = (
                _Program(existing.runtime, existing.inputs, existing.outputs, symbolic=existing.symbolic)
                if existing is not None
                else self._build_experts(rows)
            )
        except Exception:  # noqa: BLE001 -- pack/ABI incompatibility permanently falls back
            self.disabled_experts.add(rows)
            logger.exception("1Cat retained experts M=%d: Emmy pack load failed; retaining 1Cat", rows)
            return None
        self.experts[rows] = program
        return program

    def execute_route(self, program: _Program, kind: str, router_logits, bias, table, input_ids):
        import torch

        rows = router_logits.shape[0]
        weights = torch.empty((rows, TOP_K), dtype=torch.float32, device=router_logits.device)
        ids = torch.empty((rows, TOP_K), dtype=torch.int32, device=router_logits.device)
        tensors = {"router_logits": router_logits, "weights": weights, "ids": ids}
        if kind == "learned":
            tensors["bias"] = bias
        else:
            tensors.update(table=table, input_ids=input_ids)
        self._run(program, tensors, router_logits.device)
        return weights, ids

    def execute_experts(self, program: _Program, binding: expert_loader.ExpertBinding):
        import torch

        output = torch.empty((binding.rows, HIDDEN), dtype=torch.float16, device=binding.x.device)
        tensors = {
            "x": binding.x,
            "route_weights": binding.weights,
            "route_ids": binding.ids,
            program.outputs[0]: output,
        }
        tensors.update(binding.carriers)
        self._run(program, tensors, binding.x.device)
        return output

    def dispatch_route(self, router, hidden_states, router_logits, indices_type, input_ids, op):
        contract = self._route_contract(router, hidden_states, router_logits, indices_type, input_ids)
        if contract is None:
            return self.original_route(router, hidden_states, router_logits, indices_type, input_ids=input_ids)
        rows, kind, bias, table = contract
        key = (rows, kind)
        capturing = self._is_capturing()
        with self.lock:
            program = self._route_program(rows, kind, capturing=capturing)
            if program is None or (capturing and not program.verified):
                return self.original_route(router, hidden_states, router_logits, indices_type, input_ids=input_ids)
            if program.verified:
                return op(router_logits, bias, table, input_ids, kind == "hash")
            reference = self.original_route(router, hidden_states, router_logits, indices_type, input_ids=input_ids)
            try:
                output = self.execute_route(program, kind, router_logits, bias, table, input_ids)
            except Exception:  # noqa: BLE001 -- launch incompatibility permanently falls back
                self.disabled_routes.add(key)
                self.routes.pop(key, None)
                logger.exception("1Cat %s route M=%d: Emmy launch failed; retaining 1Cat", kind, rows)
                return reference
            import torch

            ids_equal = torch.equal(output[1], reference[1])
            weights_close = torch.allclose(output[0], reference[0], rtol=0.0, atol=_ROUTE_ATOL)
            if not ids_equal or not weights_close:
                self.disabled_routes.add(key)
                self.routes.pop(key, None)
                max_abs = float((output[0] - reference[0]).abs().max().item())
                logger.error(
                    "1Cat %s route M=%d: Emmy first-use parity failed (ids_equal=%s, weight_max_abs=%g); retaining 1Cat",
                    kind,
                    rows,
                    ids_equal,
                    max_abs,
                )
                return reference
            program.verified = True
            return output

    def dispatch_experts(self, method, layer, x, weights, ids, shared_experts, shared_input, op):
        binding = self._expert_contract(layer, x, weights, ids)
        if binding is None:
            return self.original_experts(method, layer, x, weights, ids, shared_experts, shared_input)
        rows = binding.rows
        capturing = self._is_capturing()
        with self.lock:
            program = self._expert_program(rows, capturing=capturing)
            if program is None or (capturing and not program.verified):
                return self.original_experts(method, layer, x, weights, ids, shared_experts, shared_input)
            if program.verified:
                return op(x, weights, ids, *(tensor for _name, tensor in binding.carriers))
            reference = self.original_experts(method, layer, x, weights, ids, shared_experts, shared_input)
            try:
                output = self.execute_experts(program, binding)
            except Exception:  # noqa: BLE001 -- launch incompatibility permanently falls back
                self.disabled_experts.add(rows)
                self.experts.pop(rows, None)
                logger.exception("1Cat retained experts M=%d: Emmy launch failed; retaining 1Cat", rows)
                return reference
            import torch

            if not torch.allclose(output, reference, rtol=_EXPERT_TOL, atol=_EXPERT_TOL):
                self.disabled_experts.add(rows)
                self.experts.pop(rows, None)
                max_abs = float((output - reference).abs().max().item())
                logger.error(
                    "1Cat retained experts M=%d: Emmy first-use parity failed (max_abs=%g); retaining 1Cat",
                    rows,
                    max_abs,
                )
                return reference
            program.verified = True
            return output


_ACTIVE: _Adapter | None = None
_ROUTE_OP: Any | None = None
_EXPERT_OP: Any | None = None
_INSTALL_LOCK = threading.Lock()


def _custom_ops():
    global _EXPERT_OP, _ROUTE_OP
    if _ROUTE_OP is not None and _EXPERT_OP is not None:
        return _ROUTE_OP, _EXPERT_OP
    import torch

    @torch.library.custom_op(
        "emmy::onecat_dsv4_route",
        mutates_args=(),
        schema="(Tensor router_logits, Tensor? bias, Tensor? table, Tensor? input_ids, bool use_hash) -> (Tensor, Tensor)",
    )
    def route_op(router_logits, bias, table, input_ids, use_hash):
        if _ACTIVE is None:
            raise RuntimeError("1Cat route custom op called before installation")
        rows = int(router_logits.shape[0])
        program = _ACTIVE.routes[(rows, "hash" if use_hash else "learned")]
        return _ACTIVE.execute_route(program, "hash" if use_hash else "learned", router_logits, bias, table, input_ids)

    @route_op.register_fake
    def route_fake(router_logits, bias, table, input_ids, use_hash):
        del bias, table, input_ids, use_hash
        rows = router_logits.shape[0]
        return (
            router_logits.new_empty((rows, TOP_K), dtype=torch.float32),
            router_logits.new_empty((rows, TOP_K), dtype=torch.int32),
        )

    @torch.library.custom_op(
        "emmy::onecat_dsv4_experts",
        mutates_args=(),
        schema="(Tensor x, Tensor weights, Tensor ids, Tensor w13, Tensor s13, Tensor w2, Tensor s2) -> Tensor",
    )
    def expert_op(x, weights, ids, w13, s13, w2, s2):
        if _ACTIVE is None:
            raise RuntimeError("1Cat expert custom op called before installation")
        rows = int(x.shape[0])
        binding = expert_loader.ExpertBinding(
            rows,
            x,
            weights,
            ids,
            (("w13", w13), ("w13_scale", s13), ("w2", w2), ("w2_scale", s2)),
        )
        return _ACTIVE.execute_experts(_ACTIVE.experts[rows], binding)

    @expert_op.register_fake
    def expert_fake(x, weights, ids, w13, s13, w2, s2):
        del weights, ids, w13, s13, w2, s2
        return x.new_empty((x.shape[0], HIDDEN))

    _ROUTE_OP, _EXPERT_OP = route_op, expert_op
    return route_op, expert_op


def _route_wrapper(adapter: _Adapter, op: Callable) -> Callable:
    def _compute_routing(self, hidden_states, router_logits, indices_type, *, input_ids=None):
        return adapter.dispatch_route(self, hidden_states, router_logits, indices_type, input_ids, op)

    _compute_routing._emmy_onecat_experts = True  # type: ignore[attr-defined]
    _compute_routing._emmy_adapter = adapter  # type: ignore[attr-defined]
    return _compute_routing


def _expert_wrapper(adapter: _Adapter, op: Callable) -> Callable:
    def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        return adapter.dispatch_experts(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input, op)

    apply._emmy_onecat_experts = True  # type: ignore[attr-defined]
    apply._emmy_adapter = adapter  # type: ignore[attr-defined]
    return apply


def _signature(function: Callable) -> tuple[str, ...] | None:
    try:
        return tuple(inspect.signature(function).parameters)
    except (TypeError, ValueError):
        return None


def register_onecat_expert_kernels(
    router_module: ModuleType | None = None,
    expert_module: ModuleType | None = None,
) -> bool:
    """Atomically install the exact router and retained-expert adapters."""
    global _ACTIVE

    try:
        router_module = router_module or importlib.import_module("vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router")
        expert_cls = expert_loader.expert_method_class(expert_module)
    except ImportError:
        logger.warning("1Cat routed experts requested, but compatible modules are unavailable")
        return False
    router_cls = getattr(router_module, "FusedTopKBiasRouter", None)
    if router_cls is None or expert_cls is None:
        return False
    with _INSTALL_LOCK:
        original_route = getattr(router_cls, "_compute_routing", None)
        original_experts = getattr(expert_cls, "apply", None)
        installed = [bool(getattr(function, "_emmy_onecat_experts", False)) for function in (original_route, original_experts)]
        if all(installed):
            first = original_route._emmy_adapter  # type: ignore[attr-defined]
            if first is original_experts._emmy_adapter:  # type: ignore[attr-defined]
                _ACTIVE = first
                return True
            return False
        if any(installed) or _signature(original_route) != ("self", "hidden_states", "router_logits", "indices_type", "input_ids"):
            return False
        if _signature(original_experts) != (
            "self",
            "layer",
            "x",
            "topk_weights",
            "topk_ids",
            "shared_experts",
            "shared_experts_input",
        ):
            return False
        adapter = _Adapter(original_route, original_experts)
        route_op, expert_op = _custom_ops()
        previous = _ACTIVE
        try:
            router_cls._compute_routing = _route_wrapper(adapter, route_op)
            expert_cls.apply = _expert_wrapper(adapter, expert_op)
            _ACTIVE = adapter
        except Exception:  # noqa: BLE001 -- restore all-or-none installation
            router_cls._compute_routing = original_route
            expert_cls.apply = original_experts
            _ACTIVE = previous
            logger.exception("1Cat routed expert installation failed; restored both originals")
            return False
    logger.info("1Cat routed experts: installed guarded Emmy routing and retained compact expert tier")
    return True
