---
name: discover-models
description: >-
  Use this skill when the user asks what new models to try or benchmark, wants newly released open models discovered,
  wants trending models mapped to suitable GPU hardware, or wants the maintained recipe set refreshed. It produces a
  ranked shortlist or lifecycle selection ready for the `onboard-model` skill, using keyless discovery data, web
  search, and a VRAM fit calculation.
---

# Discover Models to Explore

Turn "what models are worth our GPU hours?" into either a concrete shortlist or a refresh of Emmy's recipe
lifecycle. New open-weight models are filtered to the ones with real demand, then mapped to the GPU configurations
they fit. Every recipe receives a 0-100 heat score for current discovery priority. Existing recipes are ranked by
current community and serving value so the small maintained set stays focused, useful lower-priority recipes stay
available on a best-effort basis, and only technically superseded or unusable models become obsolete.

Everything here is **keyless and read-only**: `scripts/new_models.py` hits public OpenRouter + HuggingFace
endpoints, and the rest is web search. No servers are touched. In automated lifecycle mode the skill returns a compact
selection; repository-owned workflow code restores existing onboarding data, derives the complete manifest, validates
it, and writes recipe tags and onboarding shells.

## Automated rolling PR prerequisite

The discovery agent remains read-only in lifecycle mode. When an existing rolling discovery PR or unpaired discovery
branch is present, workflow orchestration must complete these steps before it builds the recipe inventory or starts
this skill:

1. Fetch the latest `main` and the exact current remote head of the rolling branch.
2. Fail if the checked-out head no longer matches that remote head.
3. Rebase the rolling branch onto `main`; fail without applying lifecycle updates if the rebase conflicts.
4. Push a changed rebase with an exact `--force-with-lease` expectation for the original remote head.

The workflow, not the discovery agent, owns this Git mutation. A lease failure means another writer advanced the
branch, so the run must stop rather than overwrite it. Research and lifecycle classification begin only from the
successfully rebased checkout.

## Pipeline

```
HF/OpenRouter data ─┐
Reddit discussions ─┼→ reconcile exact models → score heat → VRAM fit calc → hardware → model matrix
OpenRouter/Arena ───┘       (parent agent)          (0-100)     (params×quant)    (the deliverable)
```

## Step 0 — Choose the output mode

Use **survey mode** for an interactive shortlist or hardware matrix. For automated **lifecycle mode**, read
`prompts/discover-models/lifecycle.md` and `prompts/discover-models/score-recipes.md` from the repository root
completely before research. Those files are the canonical automation prompts and define the task payload, delegation,
and output contract used by GitHub Actions. Do not ask follow-up questions in lifecycle mode. Keep research bounded by
the lifecycle prompt and return the required selection as soon as the evidence supports it.

For survey mode, ask only if the user has not already implied it:

- **Time window** — default the script's ~90 days (`--since` default). Widen with `--since 2026-01-01` if they
  want "this year".
- **Modality** — default include multimodal (shown with a `modality` column); add `--text-only` if they only
  serve text.
- **Target hardware** — default the full fleet (B200, H200, H100, Pro6000, RTX5090, RTX4090). If they name a
  subset, only bucket for those.
- **How many finalists** — default ~5–8, spread across the hardware tiers.

## Step 1 — Inventory recipes and candidates

In interactive lifecycle work, inventory every `recipes/*/recipe.yaml` first with one compact query that returns only
the recipe path, model ID, tags, task, deployment matrix, and existing rationale. In automated lifecycle mode, use the
attached task's deterministic recipe batches instead; never rebuild that inventory. Read a complete recipe only when
a specific obsolete comparison needs closer inspection. Treat the top-level tags as follows:

- `maintained` — a tested recipe selected for periodic testing and optimization;
- `best-effort` — a useful runnable recipe that is not selected for periodic testing and optimization;
- `obsolete` — retained for history but disabled because an all-around better model for the same task is available at
  a comparable or lower practical VRAM footprint;
- `onboarding` + `untested` — a recipe shell that is not eligible for the maintained set yet.

Low demand, age, and exclusion from the maintained set are not reasons to mark a recipe obsolete. An obsolete recipe
can return to the maintained or best-effort set when the evidence changes. Never classify an onboarding/untested
shell. Untagged complete recipes are eligible and must be classified on the first lifecycle run.

Treat obsolete as a conservative, tradeoff-free decision. When a successor exists, compare the qualified targets in
the recipe YAML: the replacement's smallest deployment must use no more total physical GPU memory than the old
recipe's smallest deployment. The old model must retain no material advantage in quality, context, supported
hardware, latency or throughput, operating cost, modality, or licensing. A smaller or quantized recipe is not obsolete
merely because a larger or unquantized recipe exists. An obsolete decision without a successor needs a concrete
technical reason the recipe should no longer be used. Prefer best-effort whenever evidence is ambiguous or the model
has a useful advantage.

In survey mode or a local interactive run, run the discovery script with arena enrichment, capturing JSON for parsing
and the table for a human view. Use `--workers 4` to stay gentle on the HF metadata endpoint (it rate-limits bursts;
don't re-run in a loop):

```bash
./venv/bin/python scripts/new_models.py --arena --workers 4 --json > /tmp/new_models.json
./venv/bin/python scripts/new_models.py --arena --workers 4          # readable table for the user
```

In automated lifecycle mode, the read-only discovery agent cannot run repository scripts. Use the Hugging Face and
OpenRouter source investigators from Step 2 for the equivalent public catalog evidence instead; do not attempt a
denied shell command.

The script supplies one quantitative source: it lists open-weight models OpenRouter hosts (catalog entries with a
`hugging_face_id`), **excludes active families already in `recipes/`**, drops anything older than `--since`, verifies
each on HuggingFace, and ranks by HF momentum. Each JSON row in `models[]` carries:

| Field | Meaning | Use |
|---|---|---|
| `hf_id` | HuggingFace repo id | the model identity; feeds `onboard-model` |
| `created_at` | HF release date | recency |
| `downloads` | HF 30-day pulls | adoption (lagging, size-biased toward small models) |
| `likes` | cumulative HF likes | reputation |
| `trending` | HF trendingScore | **momentum / "hot right now"** (best single demand signal) |
| `elo` / `arena_rank` | LMArena Elo + rank (blank if unrated) | **quality**; blank usually = too new to be rated, not bad |
| `modality` | `text->text`, `text+image->text`, … | multimodal flag |

The table footer also flags stale OpenRouter→HF mappings ("NOT ON HF") and likely arena fuzzy-match misses —
skim those; a miss can mean a model you'd otherwise drop actually has a strong Elo under a slightly different name.

Obsolete recipes are deliberately not excluded, which lets a renewed model resurface as a reactivation candidate.
Take the top ~8–12 by `trending` (tie-break `elo`, then `downloads`) into Step 2. The script's full flag list is
documented in `AGENTS.md` (scripts section).

## Step 2 — Investigate independent demand sources (web search required)

Do not use Reddit only to validate candidates already returned by the script. Inspect recent high-engagement Reddit
threads as an independent source, then merge those model names with the Hugging Face and OpenRouter/Arena candidate
lists. Resolve an exact open-weight Hugging Face ID before accepting a Reddit-only discovery.

In automated lifecycle mode, follow `prompts/discover-models/lifecycle.md`: invoke the three named source investigators
once and in parallel, then invoke `discover-scorer` once per deterministic recipe batch and in parallel. Each source
investigator stays read-only, uses at most three public-web calls, and returns a compact source-specific candidate
list. Each scorer follows `prompts/discover-models/score-recipes.md` without doing more research or choosing lifecycle
state. The parent alone reconciles identities, compares scores globally, selects the maintained set, and proposes new
onboarding models.

Layer quantitative demand with qualitative mindshare — what people are actually saying. For each top candidate,
search for:

- `"<model name>" release` / `"<model name>" benchmark` — official announcement, benchmark claims (MMLU, GPQA,
  LiveCodeBench, SWE-bench, AIME).
- `"<model name>" vs` — head-to-head comparisons (a sign people care).
- Buzz on Reddit (r/LocalLLaMA), Hacker News, X — is it being discussed, or did it land silently?
- The releasing **lab's reputation** (DeepSeek, Qwen, MiniMax, Moonshot/Kimi, Mistral, NVIDIA, Liquid, IBM
  Granite, Allen AI/OLMo) — established labs draw adoption faster.

Distill each into a one-line **demand read**: *strong* (benchmark wins + active discussion + reputable lab),
*moderate*, or *niche/quiet*. Cross-check against the script signals — a model high on HF trend **and** loud
online is a strong pick; high downloads but silent is often a small fine-tuning base, not a flagship.

Assign an integer **heat score** from 0 through 100 to every existing recipe and every selected new model. Heat is
current onboarding priority, not measured model quality:

- `90-100` — breakout attention across independent current sources;
- `70-89` — strong current attention;
- `40-69` — moderate or established interest;
- `20-39` — niche or cooling interest;
- `0-19` — little current evidence.

Weight recent community attention and Hugging Face momentum most heavily. OpenRouter availability, arena evidence,
technical novelty, and serving value are supporting signals. Compare all scores within the run before returning them;
do not give an unsupported model a high score merely because it is new.

## Step 3 — Classify recipes and select onboarding models

Combine signals into a shortlist. A model is **promising** when it scores on several of:

- High HF `trending` (real, current momentum) — weighted highest for "what's hot".
- High arena `elo` / low `arena_rank` (proven quality) — when present.
- Strong Step-2 hype (mindshare, benchmark wins, reputable lab).
- **Novelty for emmy** — a new architecture / quant / size we haven't benchmarked teaches us more than yet
  another sibling of an existing recipe.

Drop: tiny fine-tuning bases riding download counts, models with no engine support yet (note it, revisit later),
and anything the user explicitly doesn't care about. Aim for a spread of **sizes** so the next step can fill
several hardware tiers (don't pick five 400B MoEs).

In automated lifecycle mode, `prompts/discover-models/lifecycle.md` is the complete selection contract. Score every
existing recipe, select the requested number of complete recipes as maintained, make only conservative obsolete
proposals, and return new onboarding candidates. Repository code keeps every existing onboarding shell in that
lifecycle and derives every unselected, non-obsolete complete recipe as best-effort. The agent never reproduces those
mechanical lists.

## Step 4 — Hardware requirements per finalist

For each finalist, follow [`prompts/model-fit.md`](../../../prompts/model-fit.md) — the shared VRAM fit contract
that `onboard-model` follows too, so a proposed platform and a measured one mean the same thing. It defines how to
read total parameters from `config.json`, the footprint arithmetic, the Mixture of Experts and quantization traps,
tensor-parallel sizing, and the requirement to state the numbers behind every fit claim.

Read canonical GPU names and their `vram_mib` from `emmy/gpu.py`; that registry is the authority on fleet capacity.

## Step 5 — Hardware → model matrix (the deliverable)

Present a table mapping each target GPU config to the promising model(s) that fit, with the quant and a one-line
why. Cover the spectrum — small flagships on consumer cards, mid-size on Pro6000/H100, large MoE on H200/B200
(single or tensor-parallel). Example shape:

| Hardware | gpu_count | Recommended model (quant) | Why it's promising | Fit note |
|---|---|---|---|---|
| RTX 4090 / 5090 | 1 | `<8B model>` (AWQ/FP8) | strong small-model Elo, hot on HF | ~Xgb, fits one card |
| RTX PRO 6000 | 1 | `<30–70B>` (FP8) | benchmark wins, reputable lab | ~Xgb of 96 |
| H100 80GB | 1–2 | `<70–120B MoE>` (FP8) | top arena Elo, high HF trend | TP2 for context headroom |
| H200 141GB | 1 | `<120B>` (FP8) | flagship, loud online | fits 1×, long context |
| B200 | 1–8 | `<235–400B+ MoE>` (FP8/NVFP4) | SOTA open, high demand | TPn across the node |

Flag any model with **no engine support yet** or **no suitable quant** as "watch, revisit" rather than slotting it.

## Step 6 — Return the lifecycle result

In automated lifecycle mode, return exactly the compact selection in `prompts/discover-models/lifecycle.md` with no
prose or Markdown fence. Do not edit recipe files. The workflow requires one score for every exact existing recipe,
reconstructs existing onboarding entries from the task, derives best-effort entries, and validates the complete
manifest before applying it. It demotes a superseded obsolete decision to best-effort unless the replacement is active,
serves the same task, and its smallest known qualified deployment uses no more total physical GPU memory than the old
recipe's smallest deployment.

## Step 7 — Hand off in survey mode

For each (model, hardware) pair the user wants to pursue, offer to invoke **`onboard-model`** — pass the
`hf_id` and the chosen GPU + `gpu_count`. That skill does the real work (engine/image research, recipe, validate,
benchmark within time caps). Don't reimplement any of it here.

If the user just wanted the survey, stop at the matrix.

## Common mistakes to avoid

- **Don't rank by downloads alone** — it's lagging and size-biased (tiny fine-tuning bases dominate). `trending`
  + arena `elo` + Step-2 hype together are the demand signal.
- **Don't assign a platform from the model name.** `prompts/model-fit.md` is the contract: total parameters read
  from `config.json`, a quantization whose repository exists, and the stated arithmetic behind the claim.
- **Don't treat a blank arena Elo as "bad".** It usually means the model is too new for the last weekly arena
  snapshot — lean on HF `trending` + news there.
- **Don't spam the script.** HF rate-limits bursts; use `--workers 4` and re-run sparingly (transient failures
  land in the script's "COULD NOT VERIFY" bucket — wait and re-run, don't hammer).
- **Don't deploy or edit recipes in this skill.** The automated workflow applies a validated lifecycle manifest;
  `onboard-model` owns real deployment and qualification.
