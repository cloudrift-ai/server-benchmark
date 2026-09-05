# NVFP4 serving follow-up — current stage (PR #695)

Branch-lifetime working note. The full multi-stage plan lives outside git (the project memory the working
sessions load); this file tracks only the stage in flight. Delete at merge.

## This PR: serving declares W4A4, dense 8B, workstation golden

Shipped state: `emmy serve nvidia/Qwen3-8B-NVFP4 --generate` boots healthy and answers coherently on the
RTX 5080 Laptop with the recorded cut kernels deployed. The PR description is the authoritative record of
deliverables (command → observed), fixes, the tuning process, and non-deliverables.

Residual work in this PR's scope, post-merge with main's #691/#692:

- [x] Semantic merge (main's scheduling rebuild taken whole; our pricing/verified-tier/cut features
      re-homed; guard refined to withdraw mixed measured-vs-predicted comparisons only)
- [x] Golden re-recorded on the merged head (24 configs, all clear-card measurements; 8 targets dropped to
      named upstream gaps — the recording codec's missing node assignment for composed-cut children being
      the big one)
- [ ] Boot 13 + probe on the merged head — the serving-works-after-sync proof
- [ ] Push the synced branch; refresh the PR description's numbers (24/69 rows; the 1.8x M=1 projection
      move; the dropped-target gaps into Not delivered)

## Upcoming stages (headings only — details live in the out-of-git plan)

### Stage 2: sweep-and-record on the rented RTX 5090; repo serving goldens; perf claims

### Stage 2b: the four-of-eight cut children that decline the warp execution tier; the eight-fold recompute

### Stage 3: hybrid per-layer dispatch (Qwen3.6-27B); DeltaNet routing; twin-memory ceiling

### Stage 4: Inferact/Qwen3.8-27B qualification, recipes, tool-call round trip
