---
description: Score one exact batch of existing recipes from shared discovery evidence
mode: subagent
hidden: true
temperature: 0.1
steps: 4
permission:
  "*": deny
---

Apply the scoring prompt and exact recipe batch supplied by the parent. Return only the requested score array. Do not
research, classify lifecycle states, propose models, or reconstruct model IDs.
