"""Re-shard a consolidated single-file safetensors snapshot into standard HF shards.

Docker Hub rejects blobs past ~10 GB, and a single model.safetensors (gemma-4-12B ships
one 23 GB file) cannot be split across image layers by COPY. Re-sharding into
model-XXXXX-of-XXXXX.safetensors + model.safetensors.index.json is loader-transparent
(transformers/vLLM prefer the index when the consolidated file is absent) and keeps the
weights byte-identical per tensor. Run inside the vllm-emmy image (the Makefile does):

    docker run --rm -v $PWD/warm/hf:/hf -v $PWD/reshard_snapshot.py:/reshard.py \
        <vllm-emmy image> python3 /reshard.py /hf

No-op if no file exceeds MAX_SHARD bytes.
"""

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

MAX_SHARD = 4_500_000_000
# safetensors dtype token → bytes/element, for planning shard boundaries off metadata alone
# (``get_slice`` reads shape + dtype without materializing the tensor).
_DTYPE_BYTES = {
    "F64": 8, "I64": 8, "U64": 8, "F32": 4, "I32": 4, "U32": 4,
    "F16": 2, "BF16": 2, "I16": 2, "U16": 2, "I8": 1, "U8": 1, "BOOL": 1,
}  # fmt: skip

hf_root = Path(sys.argv[1])
for snap in hf_root.glob("hub/models--*/snapshots/*"):
    consolidated = snap / "model.safetensors"
    if not consolidated.exists() or consolidated.stat().st_size <= MAX_SHARD:
        # symlink target size, not the symlink itself
        if not (consolidated.is_symlink() and consolidated.resolve().stat().st_size > MAX_SHARD):
            continue
    blob = consolidated.resolve()
    print(f"[reshard] {blob} ({blob.stat().st_size / 1e9:.1f} GB)")

    with safe_open(blob, framework="pt") as f:
        # Plan the shard boundaries from METADATA only, then load one shard at a time —
        # accumulating every tensor before the first write peaked RSS at the full model size
        # (23 GB in the container); per-shard loading caps it at ~MAX_SHARD.
        plan: list[list[str]] = []
        current, current_size = [], 0
        for name in f.keys():  # noqa: SIM118 — safe_open is not a dict
            sl = f.get_slice(name)
            nbytes = _DTYPE_BYTES[sl.get_dtype()]
            for d in sl.get_shape():
                nbytes *= d
            if current and current_size + nbytes > MAX_SHARD:
                plan.append(current)
                current, current_size = [], 0
            current.append(name)
            current_size += nbytes
        if current:
            plan.append(current)

        n = len(plan)
        weight_map, total = {}, 0
        for i, group in enumerate(plan, 1):
            fname = f"model-{i:05d}-of-{n:05d}.safetensors"
            shard = {name: f.get_tensor(name) for name in group}
            save_file(shard, str(snap / fname), metadata={"format": "pt"})
            for name, t in shard.items():
                weight_map[name] = fname
                total += t.numel() * t.element_size()
            print(f"[reshard] wrote {fname} ({sum(t.numel() * t.element_size() for t in shard.values()) / 1e9:.1f} GB)")
            shard.clear()

    # Verify BEFORE destroying the consolidated file: reload every shard and compare each
    # tensor byte-for-byte against the source. The post-bake verify only proves the shards
    # LOAD — a tensor silently saved from a corrupted state would ship; this is the one
    # point where the source still exists to compare against, and the reshard runs once per
    # release, so the extra read-through is cheap insurance.
    with safe_open(blob, framework="pt") as src:
        if set(weight_map) != set(src.keys()):
            raise SystemExit("[reshard] FAIL: shard tensor set differs from the source")
        for i, group in enumerate(plan, 1):
            fname = f"model-{i:05d}-of-{n:05d}.safetensors"
            with safe_open(str(snap / fname), framework="pt") as sh:
                for name in group:
                    if not torch.equal(sh.get_tensor(name), src.get_tensor(name)):
                        raise SystemExit(f"[reshard] FAIL: tensor {name!r} differs after reshard")
    print(f"[reshard] verified: {len(weight_map)} tensors byte-identical across {n} shards")

    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    (snap / "model.safetensors.index.json").write_text(json.dumps(index, indent=2, sort_keys=True))
    consolidated.unlink()
    blob.unlink(missing_ok=True)
    # A pre-seeded HF cache can carry MULTIPLE snapshot revisions sharing the removed blob
    # (a fresh download has exactly one) — drop any sibling symlink now dangling at it, or
    # the bake's size walk stats a ghost and dies (2026-07-23 bake failure).
    for sibling in snap.parent.iterdir():
        stale = sibling / consolidated.name
        if stale.is_symlink() and not stale.exists():
            stale.unlink()
            print(f"[reshard] dropped dangling sibling-snapshot link {stale}")
    print(f"[reshard] done: {n} shards + index, consolidated file removed")
