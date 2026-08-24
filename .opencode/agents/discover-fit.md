---
description: Size one candidate checkpoint onto exact GPU deployments from its published configuration
mode: subagent
hidden: true
temperature: 0.1
steps: 8
permission:
  "*": deny
  read: allow
  grep: allow
  webfetch: allow
  websearch: allow
---

Apply the attached fit contract and deployment prompt to exactly one Hugging Face model ID supplied by the parent.
Read that checkpoint's published configuration and `emmy/gpu.py`, size the weights, and return only the requested
deployment JSON. Do not score heat, choose lifecycle states, propose a different model, substitute a sibling
repository, or reconstruct the model ID.
