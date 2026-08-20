# Size One Candidate Onto GPU Deployments

Size exactly the one Hugging Face model ID supplied by the parent, following the attached `model-fit.md` contract. Do
no other work: no heat scoring, no lifecycle states, no alternative model proposals.

Read that exact repository's `config.json` and model card. Never substitute a sibling, quantized, or same-family
repository — the deployment is for the ID you were given. Read `emmy/gpu.py` for canonical GPU names and their
`vram_mib`; those spellings are the only accepted values and the registry is the authority on capacity.

`bytes_per_param` describes the repository you were given, not a variant of it. A BF16 repository is sized at 2 bytes
even when an FP8 sibling exists under another ID; sizing the repository you wish you had is how an unservable platform
reaches rented hardware.

## Choose a platform the fleet actually rents

`deploy.gpu_count` is 1, 2, 4, 8, or 16 — the node shapes the provider offers. It is not the result of dividing the
footprint by a card's capacity: 5 GPUs and 12 GPUs are arithmetic, not platforms, and nothing can deploy them. Round
up to the next shape in that list and confirm the total still holds `min-to-serve`. A multi-GPU count must also divide
the checkpoint's attention head count, so prefer 2, 4, and 8 over 16.

Return one to three admissible deployments, smallest platform first, each holding `min-to-serve` for the checkpoint as
it is actually published.

Return an empty `deployments` array when the checkpoint cannot be sized — the repository is gated or unreadable, the
parameter count cannot be established, or no allowed platform holds the weights. A checkpoint needing more than the
largest listed shape of the largest fleet GPU has no deployment; say so rather than naming a count nobody can rent. An
empty array is a correct, useful answer: it drops a new candidate that cannot be served and leaves an existing shell's
current matrix alone. Never guess a platform to avoid returning nothing, and never infer size from the model ID.

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
