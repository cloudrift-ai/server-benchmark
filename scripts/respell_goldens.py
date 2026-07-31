"""The step-7 value-grammar re-speller — the ONE mechanical migration of the golden corpus.

Re-spells every live ``knobs: {...}`` row in the per-GPU golden YAMLs from the legacy
embedded-worker value grammar to the site-local end-state codec: the ``TILE`` worker tokens
(``a:<atom>/w<M>x<N>/…``, ``n<N>x<M>/…``), the ``REDUCE`` coop width (``b<n>[t]``) and the
``WSPEC`` producer band (``p<n>``) all fold into ONE ``WORK`` entry (``w<M>x<N>[+p<np>]`` /
``t<N>[x<M>]``), spelled first in every dict; the per-site values keep only site-local facts.
Semantics-preserving BY CONSTRUCTION: each value routes through the live codec
(``TilePlan.parse(...).spell_site()`` / ``ReducePlan``), and a worker-geometry conflict inside
one row is a hard error, never a silent pick. Comment lines — including the retired PLACE-era
blocks — are untouched; the file is edited textually, never YAML-dumped (the corpus is
hand-curated flow style).

Usage: venv/bin/python scripts/respell_goldens.py [--check] [file ...]
    (default: every emmy/compiler/pipeline/search/goldens/*.yaml; --check reports without writing)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

from emmy.compiler.ir.schedule import ReducePlan, TilePlan, WarpSpec, Workers, is_warp_codec, plan_workers

_GOLDENS = Path(__file__).resolve().parents[1] / "emmy" / "compiler" / "pipeline" / "search" / "goldens"
_KNOBS_RE = re.compile(r"^(?P<indent>\s*)knobs:\s*\{(?P<body>.*)\}(?P<tail>\s*(?:#.*)?)$")


def _fmt(key: str, value) -> str:
    k = f"'{key}'" if "@" in key else key
    if isinstance(value, bool):
        return f"{k}: {str(value).lower()}"
    return f"{k}: '{value}'"


def _respell_row(knobs: dict, where: str) -> dict:
    """One knob dict → the site-local spelling + the merged ``WORK`` entry (first)."""
    work: Workers | None = None
    src: str | None = None

    def merge(w: Workers | None, origin: str) -> None:
        nonlocal work, src
        if w is None:
            return
        if work is not None and (work.kind, work.units) != (w.kind, w.units):
            raise ValueError(f"{where}: {origin} implies WORK {w.spell()} but {src} implied {work.spell()} — one kernel, one inventory")
        if work is None:
            work, src = w, origin

    out: dict = {}
    producer = 0
    for key, value in knobs.items():
        fam = str(key).split("@", 1)[0]
        if fam == "WORK":  # already re-spelled — idempotence
            merge(Workers.parse(str(value)), key)
            continue
        if fam == "WSPEC":
            if value:
                producer = WarpSpec.parse(str(value)).aux_warps
            continue  # the family is retired — the band rides WORK's +p
        if fam == "TILE" and isinstance(value, str) and value:
            if is_warp_codec(value) or re.match(r"^n\d", value.strip()) or value.strip().startswith("a:"):
                plan = TilePlan.parse(value)
                merge(plan_workers(plan), key)
                out[key] = plan.spell_site()
                continue
        if fam == "REDUCE" and isinstance(value, str) and re.search(r"(^|/)b\d+t?($|/)", value):
            plan = ReducePlan.parse(value)
            if plan.coop > 1:
                merge(Workers(kind="thread", units=(plan.coop, 1)), key)
            out[key] = plan.spell_site()
            continue
        out[key] = value
    if producer:
        if work is None or work.kind != "warp":
            raise ValueError(f"{where}: a WSPEC producer band needs a warp inventory, got {work.spell() if work else None}")
        work = Workers(kind=work.kind, units=work.units, producer=producer)
    return {"WORK": work.spell() if work is not None else "", **out}


def respell_file(path: Path, check: bool = False) -> int:
    lines = path.read_text().splitlines(keepends=True)
    changed = 0
    for i, line in enumerate(lines):
        m = _KNOBS_RE.match(line.rstrip("\n"))
        if m is None:
            continue
        knobs = yaml.safe_load("{" + m.group("body") + "}")
        new = _respell_row(knobs, f"{path.name}:{i + 1}")
        rendered = f"{m.group('indent')}knobs: {{{', '.join(_fmt(k, v) for k, v in new.items())}}}{m.group('tail')}\n"
        if rendered != line:
            lines[i] = rendered
            changed += 1
    if changed and not check:
        path.write_text("".join(lines))
    print(f"{path.name}: {changed} row(s) {'would be ' if check else ''}re-spelled")
    return changed


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--check"]
    check = "--check" in sys.argv[1:]
    files = [Path(a) for a in args] if args else sorted(_GOLDENS.glob("*.yaml"))
    total = sum(respell_file(f, check=check) for f in files)
    print(f"total: {total} row(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
