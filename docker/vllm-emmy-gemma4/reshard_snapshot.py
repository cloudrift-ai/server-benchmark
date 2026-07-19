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

import torch  # noqa: F401  — safetensors framework="pt" needs it
from safetensors import safe_open
from safetensors.torch import save_file

MAX_SHARD = 4_500_000_000

hf_root = Path(sys.argv[1])
for snap in hf_root.glob("hub/models--*/snapshots/*"):
    consolidated = snap / "model.safetensors"
    if not consolidated.exists() or consolidated.stat().st_size <= MAX_SHARD:
        # symlink target size, not the symlink itself
        if not (consolidated.is_symlink() and consolidated.resolve().stat().st_size > MAX_SHARD):
            continue
    blob = consolidated.resolve()
    print(f"[reshard] {blob} ({blob.stat().st_size / 1e9:.1f} GB)")

    shards, current, current_size = [], {}, 0
    with safe_open(blob, framework="pt") as f:
        names = list(f.keys())
        for name in names:
            t = f.get_tensor(name)
            nbytes = t.numel() * t.element_size()
            if current and current_size + nbytes > MAX_SHARD:
                shards.append(current)
                current, current_size = {}, 0
            current[name] = t
            current_size += nbytes
    if current:
        shards.append(current)

    n = len(shards)
    weight_map, total = {}, 0
    for i, shard in enumerate(shards, 1):
        fname = f"model-{i:05d}-of-{n:05d}.safetensors"
        save_file(shard, str(snap / fname), metadata={"format": "pt"})
        for name, t in shard.items():
            weight_map[name] = fname
            total += t.numel() * t.element_size()
        print(f"[reshard] wrote {fname} ({sum(t.numel() * t.element_size() for t in shard.values()) / 1e9:.1f} GB)")
        shard.clear()

    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    (snap / "model.safetensors.index.json").write_text(json.dumps(index, indent=2, sort_keys=True))
    consolidated.unlink()
    blob.unlink(missing_ok=True)
    print(f"[reshard] done: {n} shards + index, consolidated file removed")
