# Score One Recipe Batch

Score exactly the supplied recipe batch using the shared source evidence and the inventory fields. Do not perform
additional research, choose lifecycle states, propose new models, or change an exact model ID.

Heat is current onboarding priority, not measured model quality:

- `90-100` — breakout attention across independent current sources;
- `70-89` — strong current attention;
- `40-69` — moderate or established interest;
- `20-39` — niche or cooling interest;
- `0-19` — little current evidence.

Weight recent community attention and Hugging Face momentum most heavily. OpenRouter availability, arena evidence,
technical novelty, serving value, and the recipe's useful hardware coverage are supporting signals. When current
evidence says little about a recipe, use its existing heat and rationale as context rather than inventing demand.

Return exactly one JSON array without prose or a Markdown fence. Preserve batch order and include every supplied row
once:

```json
[
  {"model_id": "exact/existing-id", "rationale": "Evidence-backed rationale of at most 20 words.", "heat": 75}
]
```
