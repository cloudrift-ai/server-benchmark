# Style Guide

## Python Code Style

### Naming

- `snake_case` for functions, variables, and modules.
- `PascalCase` for classes.
- `UPPER_SNAKE_CASE` for module-level constants.
- Prefix private/internal helpers with underscore (e.g., `_ssh_base_args`).

### Logging

All output goes through Python's `logging` module — never use `print()`.

Each module gets a module-level logger:

```python
import logging

logger = logging.getLogger(__name__)
```

Level mapping:

| Pattern | Level |
|---|---|
| Normal output | `logger.info(...)` |
| Warnings | `logger.warning(...)` |
| Errors | `logger.error(...)` |

Two logging configurations:

- **Standalone CLI** (`setup_cli_logging()` in `logging_setup.py`): `%(message)s` format — output identical to `print()`.
- **Bench** (`setup_logging()` in `bench_logging.py`): `[%(name)s] %(message)s` — prefixed output with module/group names.

The bench formatter shows short module names for library loggers (`emmy.deploy.orchestrate` → `[orchestrate]`) and split group loggers (`rtx5090_x_1.ModelName` → `[rtx5090_x_1] [ModelName]`).

### Error Handling

Log errors and return failure for operational errors:

```python
logger.error("Failed to pull images")
return False
```

Raise exceptions for programming errors and invalid internal state:

```python
raise ValueError(f"Model config must have 'name' field: {model}")
```

### Docstrings

Use triple-quote docstrings for modules and public functions. Keep them
to one line when the purpose is obvious:

```python
def load_recipe(recipe_dir):
    """Load recipe.yaml and return base Recipe (no matrix expansion)."""
```

Use `Args:` / `Returns:` sections only when parameters or return values
are non-obvious.

### Module Structure

- `__init__.py` files contain only re-exports. No classes, functions, interfaces, or business logic.
- ABCs and interfaces go in explicitly named files (e.g., `backend/base.py`, not `backend/__init__.py`).
- Business logic goes in named modules (e.g., `recipe.py`, `compose.py`).
- `commands/` layer: CLI code only (argparse registration + `handle_*` handlers). Reusable logic lives in top-level domain packages (`emmy/deploy/`, `emmy/provisioning/`, `emmy/benchmark/`).

### Imports

Group imports in this order, separated by blank lines:

1. Standard library (`os`, `sys`, `subprocess`, etc.)
2. Third-party packages (`yaml`, `pandas`, `pytest`)
3. Local imports (`from emmy.deploy import ...`)

### Formatting

- 4 spaces for indentation.
- Keep lines under ~140 characters (enforced by Ruff).
- Double quotes for strings.

### Tooling

Style rules are enforced by [Ruff](https://docs.astral.sh/ruff/), configured in `pyproject.toml`. Run `make lint` to check and `make format` to auto-fix. Enabled rule sets: `E` (pycodestyle), `F` (pyflakes), `W` (warnings), `I` (isort), `UP` (pyupgrade), `B` (bugbear).

### IR statements must be frozen dataclasses

Every concrete `Stmt` subclass — Loop-IR (`Loop`, `StridedLoop`, `Cond`, leaves), Tile-IR (`GridTile`, `ThreadTile`,
`RegisterTile`, `SerialTile`, `StridedTile`, `Stage`, `StageBundle`, `AsyncWait`), Kernel-IR (`Smem`, `Sync`,
`CpAsyncCopy`, `TmaDescriptor`, `TmaLoad`, `MbarrierInit`, …) — must be declared `@dataclass(frozen=True)`. `Body` is
already a `tuple[Stmt, ...]` subclass, so freezing every Stmt makes the entire body tree hashable end-to-end.

Why: structural caches (`Body.structural_key()` and any future bodies-as-cache-keys work) traverse the body and hash
every Stmt. A single mutable Stmt anywhere in the tree poisons every cache that keys on the surrounding Body — and
the surrounding code can't degrade gracefully without losing the optimization.

If you need to "edit" a frozen Stmt, return a new instance via `dataclasses.replace(stmt, field=value)`. If a
`__post_init__` needs to coerce a field (e.g. `tuple → Body`), use `object.__setattr__(self, "field", coerced)` —
that's the standard pattern for frozen dataclasses that still need light normalization at construction time. Don't
add `try/except TypeError` fallbacks around structural caches to tolerate unhashable stmts; fix the unhashable stmt
instead.

Every `Op` dataclass is `frozen=True` as well (an architecture test ratchets it), and its maps (`knobs`,
`inputs`, `outputs`, a `TileOp`'s `schedule`) land as `frozendict` via `Op.__post_init__` — a rewrite is always a
`dataclasses.replace` + graph-node rebind, never an edit. The one sanctioned bypass is the spelling-preserving
clone in `LoopOp.rename_buffers` (field-level `object.__setattr__`, no `__init__`): a buffer rename must not
renormalize, since commutative-arg order sorts by buffer name. Ops stay UNHASHABLE (`__hash__ = None`) — semantic
comparison is `identity_key`, never `hash`. Also make sure no Op ends up as a *field value* of a Stmt —
`Assign.op` / `Accum.op` / `Select.op` take an `ElementwiseImpl` (the lightweight value object, already hashable),
never an `ElementwiseOp` wrapper.

### Immutable mappings are `frozendict`

When a mapping must not be mutable in place (the `Op.inputs` / `Op.outputs` io maps), use `frozendict` — a builtin
in Python 3.15, provided by the `frozendict` package on our 3.12+ floor (`from frozendict import frozendict`; on
CPython the package's type is a `dict` subclass, so pickling, deepcopy, `isinstance` and serializers behave like the
dict it replaces — note the 3.15 builtin is NOT a `dict` subclass, which the eventual stdlib switch must audit).
Don't hand-roll immutable dict subclasses, and don't use `types.MappingProxyType` — it can't be pickled or
deep-copied and is not a `dict`.

### Value types are `frozen=True`; simple ones are `slots=True` too

Default to `@dataclass(frozen=True)` for anything that models a value — an IR term, a schedule choice, a domain, a
composition prefix. Frozen is what lets one object be shared by many wrappers with no defensive copy, lets a derived
read cache on the term at all, and makes `dataclasses.replace` the only way a "change" happens.

A SIMPLE value type — a small object that is its fields plus the behavior reading them, with no derived state of its
own — takes `slots=True` as well. It drops the per-instance `__dict__`, which cuts memory and speeds attribute
access, and that is worth having on the leaf choice objects an enumeration builds in the millions (`Tile`, `Work`,
`Reduce`, `Stage`, `Raster`).

A BIGGER type — one that owns derived reads the rest of the compiler depends on (`Fold`, `TileOp`, `ClassicDomains`,
`ClassicScheduleContext`) — stays unslotted and declares those reads with `cached_property` or `cached_method`, per
the next section. The two are mutually exclusive: both cache through the very `__dict__` that `slots=True` removes,
so a slotted class raises `TypeError: No '__dict__' attribute on 'X' instance to cache 'y' property` on the first
read. Never add `__dict__` back to `__slots__` to have both — a type wanting caches is telling you which side it is
on.

One cost worth knowing either way: `@dataclass(slots=True)` builds a NEW class object and regenerates every method,
so it makes import-time class construction more expensive, not less.

### Derived values are first-class members, not `__dict__` stashes

A derived value another module reads (a lowered body, an identity digest, a canonical rendering) is part of the
owning type's interface: declare it as a `functools.cached_property` on the class that owns it, or compute it in
`__post_init__`. This works on a frozen dataclass too — `cached_property` writes through `__dict__` directly, and a
`__getstate__` that pickles only declared fields strips it.

A derived read that is a FAMILY keyed by an argument — one value per site, per operand, per family name — takes
`utils.cached_method`, which is `cached_property`'s storage under a private slot while the attribute stays a call. It
is a declared descriptor on the owning class, so the family is as visible as any other member, and it keeps the
per-key laziness a whole-family `cached_property` would give up. Do not reach for a `functools.cache` keyed on `self`:
it hashes the whole term per call and pins every term ever built for the life of the process.

Never stash a derived value in a memo table keyed off some other object — `obj.__dict__.get(...)`,
`object.__setattr__(obj, "_my_cache", ...)`, or a named memo-slot helper. An undeclared attribute is invisible to
readers of the class, dodges pickling rules, grows a second spelling of the same fact, and smears one type's
functionality across the modules that cache for it.

**Pick the owner by what the value is a property OF.** A fact about a term's own value (its lowering, its free axes)
belongs on the term. A fact about a node's POSITION in a tree — its parent, its segment path, whether it is derived
evaluation — belongs on the object that owns that tree, never on the node: this IR shares subterms, so one Fold
reached down two paths has two parents, and caching one of them on the value is a correctness hazard, not just a
layering one. `TileOp` owns its kernel's site table for exactly this reason.

If several wrappers over one value each re-derive the same fact, the fix is to stop re-deriving — not to hide a
shared cache under the value they have in common.

### Dependency Injection for Testability

Shared logic accepts callable parameters (`run_cmd`, `write_file`) so
that local and SSH targets can provide their own implementations and
tests can use dry-run or mock versions:

```python
def run_deploy(run_cmd, write_file, recipe, model_dir, ...):
```

### Concurrency

The codebase is fully async. All subprocess and network I/O uses native
`asyncio` APIs (`asyncio.create_subprocess_exec/shell`, `httpx.AsyncClient`).
Use `asyncio.Semaphore` to limit concurrency. CLI entry points use
`asyncio.run()`:

```python
def handle_foo(args):
    asyncio.run(_handle_foo(args))


async def _handle_foo(args):
    await ...
```

All `run_cmd` callables are `async def` with a `timeout` parameter.
Timeouts use `asyncio.wait_for()` around `proc.communicate()`:

```python
async def run_cmd(command, stream=True, timeout=600):
    proc = await asyncio.create_subprocess_shell(command, ...)
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
```

On `TimeoutError`: kill the process, await termination, log, return `(1, "", "")`.

## Commit Messages

- Keep the subject line short (under ~72 characters).
- Use imperative mood: "Add feature", not "Added feature".
- No multi-line descriptions unless truly necessary — if the change needs
  explanation, put it in the PR description.

Good:
```
Add CLI deploy tool with local and SSH targets
Fix variant resolution for single-GPU models
Remove unused benchmark script
```

Bad:
```
This commit adds a new CLI deploy tool that supports both local and SSH
deployment targets with dry-run mode and variant resolution for different
GPU configurations.
```
