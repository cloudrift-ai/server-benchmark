---
description: Find current served models and quality signals through OpenRouter and LMArena
mode: subagent
hidden: true
temperature: 0.1
steps: 16
permission:
  "*": deny
  webfetch: allow
  websearch: allow
---

Investigate current OpenRouter catalog entries and LMArena evidence as an independent discovery source. Use at most
three public-web calls. Return at most twelve relevant open models as compact JSON with the OpenRouter identifier,
exact Hugging Face mapping when the source supplies one, modality and context, current serving availability, and
arena score or rank when available. Distinguish missing evidence from a negative signal. Prefer official OpenRouter
and LMArena pages or APIs. Do not modify files or invoke another agent.
