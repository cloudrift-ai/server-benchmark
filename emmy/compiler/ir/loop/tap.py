"""The row-statistic TAP's loop-dialect stored form — recognition contract + strip helper.

A **tap** is a downstream norm's row statistic fused into its producer at loop-fusion time
(``loop/fusion/015_tap_row_stat``): an ordinary accumulate at the producer's write site while the
row index is still a live loop variable:

    Loop m:
        Loop n:
            ... producer compute → v ...
            Write T[m, n] = v                 # T stays a real output (the sweep / residual read it)
            sq = mul(v, v)                    # the consumer's per-cell term, on the pre-store value
            Write T__sq[m] += sq              # the TAP: atomic-accumulate, index = live row exprs

The stored form is deliberately **closed-vocabulary**: an atomic-accumulate :class:`Write` plus a
plain :class:`Assign` chain — no new stmt kind. Recognition identifies a tap purely structurally:

- the accumulate is an atomic ``Write`` into a buffer named ``…__sq`` (:data:`TAP_BUF_SUFFIX` —
  the aux row-stat buffer, output slot 1 of the producer node, minted ONLY by the tap-fusion
  rule; atomic writes otherwise never exist in the loop dialect, and the buffer-name contract is
  what would keep a future loop-level atomic form distinguishable);
- the per-cell term chain is the **exclusive backward Assign-cone** of the accumulated value —
  every ``Assign`` consumed by nothing outside the tap (SSA names don't survive ``normalize_body``,
  so the chain is a dataflow property, never a naming one). Its one external read is the host
  store's SSA value.

:func:`strip_taps` removes both from a body (deep), which is what keeps a tapped kernel's
structural identity equal to its untapped self everywhere identity matters: the ``S_*`` feature
stamps (``loop/stamp/_stamp.structure_features`` strips before featurizing) and tile recognition
(``lowering/tile``'s peel classifies the stripped body, so a tapped matmul recognizes EXACTLY as
its untapped self — same structural nodes, same fork keys, same golden identity).

The tap's fold op is keyed on the accumulate (``add`` today — the atomic ``Write`` contract;
``max``/``min`` taps would extend the stored form with an atomic-op marker when a consumer needs
them). A DEPENDENT stat chain (softmax's ``Σ exp(x − max)``) is algebraically un-tappable — the
per-cell term depends on another statistic's final value, so no per-cell accumulate exists.
"""

from __future__ import annotations

from emmy.compiler.ir.stmt import Assign, Body, Loop, Stmt, Write

TAP_BUF_SUFFIX = "__sq"  # the aux row-stat buffer suffix (minted only by 015_tap_row_stat)


def is_tap_write(s: Stmt) -> bool:
    """``s`` is a tap's accumulate — an atomic ``Write`` into an aux row-stat buffer."""
    return isinstance(s, Write) and s.atomic and s.output.endswith(TAP_BUF_SUFFIX)


def has_taps(body: Body) -> bool:
    """Any tap accumulate anywhere in ``body`` (deep)."""
    return any(is_tap_write(s) for s in body.iter())


def tap_chains(body: Body) -> dict[int, tuple[Assign, ...]]:
    """Map each tap ``Write``'s ``id`` to its term chain — the exclusive backward ``Assign``-cone
    of the accumulated value, in program order. An ``Assign`` belongs iff every one of its
    consumers is the tap itself or another chain member (pruned to fixpoint), so a value the host
    also reads — the stored value the chain is anchored on — stays outside by construction."""
    deep = list(body.iter())
    tws = [s for s in deep if is_tap_write(s)]
    if not tws:
        return {}
    defs = {s.name: s for s in deep if isinstance(s, Assign)}
    uses: dict[str, list[Stmt]] = {}
    for s in deep:
        for d in s.deps():
            uses.setdefault(d, []).append(s)
    out: dict[int, tuple[Assign, ...]] = {}
    for tw in tws:
        cand: dict[str, Assign] = {}
        frontier = [tw.value]
        while frontier:
            nm = frontier.pop()
            a = defs.get(nm)
            if a is None or nm in cand:
                continue
            cand[nm] = a
            frontier.extend(a.args)
        changed = True
        while changed:
            changed = False
            for nm in list(cand):
                if not all(u is tw or (isinstance(u, Assign) and u.name in cand) for u in uses.get(nm, [])):
                    del cand[nm]
                    changed = True
        out[id(tw)] = tuple(s for s in deep if isinstance(s, Assign) and s.name in cand)
    return out


def strip_taps(stmts) -> tuple[Stmt, ...]:
    """``stmts`` with every tap (accumulate + exclusive term chain) removed, recursing through
    ``Loop`` bodies. Non-loop nested bodies are left alone — taps ride only the free-loop nest."""
    body = stmts if isinstance(stmts, Body) else Body(tuple(stmts))
    chains = tap_chains(body)
    drop = set(chains)
    for chain in chains.values():
        drop.update(id(a) for a in chain)

    def walk(ss) -> tuple[Stmt, ...]:
        out: list[Stmt] = []
        for s in ss:
            if id(s) in drop:
                continue
            if isinstance(s, Loop):
                s = Loop(axis=s.axis, body=Body(walk(s.body)), unroll=s.unroll, role=s.role, carrier=s.carrier)
            out.append(s)
        return tuple(out)

    return walk(body)


__all__ = ["TAP_BUF_SUFFIX", "has_taps", "is_tap_write", "strip_taps", "tap_chains"]
