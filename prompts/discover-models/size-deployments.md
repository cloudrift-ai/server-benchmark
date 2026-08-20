# Size One Candidate Onto GPU Deployments

Size exactly the one Hugging Face model ID supplied by the parent, following the attached `model-fit.md` contract. Do
no other work: no heat scoring, no lifecycle states, no alternative model proposals.

Read that exact repository's `config.json` and model card. Never substitute a sibling, quantized, or same-family
repository — the deployment is for the ID you were given. Read `emmy/gpu.py` for canonical GPU names and their
`vram_mib`; those spellings are the only accepted values and the registry is the authority on capacity.

Return one to three admissible deployments, smallest platform first, each holding `min-to-serve` for the checkpoint as
it is actually published. Prefer platforms the fleet can plausibly supply over the theoretical minimum.

Return an empty `deployments` array when the checkpoint cannot be sized — the repository is gated or unreadable, the
parameter count cannot be established, or no fleet platform holds the weights. An empty array is a correct, useful
answer: it drops a new candidate that cannot be served and leaves an existing shell's current matrix alone. Never
guess a platform to avoid returning nothing, and never infer size from the model ID.

Return exactly one JSON object without prose or a Markdown fence:

```json
{
  "model_id": "exact/supplied-id",
  "total_params_b": 109.0,
  "bytes_per_param": 2,
  "min_to_serve_gb": 283.4,
  "deployments": [{"deploy.gpu": "NVIDIA H200 141GB", "deploy.gpu_count": 4}],
  "note": "109B total / 17B active MoE; only a BF16 repository is published."
}
```

Copy `model_id` letter for letter from the parent. `total_params_b`, `bytes_per_param`, and `min_to_serve_gb` are the
arithmetic behind the answer and must be present whenever `deployments` is non-empty; when it is empty, `note` must
say why. Keep `note` under 30 words.
