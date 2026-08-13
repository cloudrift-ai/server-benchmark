# Provisioning Layer

This package owns *all* VM lifecycle logic for the CLI:

* `emmy deploy cloud` → `provision_cloud_vm()`
* `emmy bench` → `provision_cloud_vm()` (via `benchmark/execution.py`)
* `emmy vm create gpu` → `provision_cloud_vm()`
* `emmy vm create gcp / cloudrift` → provider `create_instance()` directly (single-shot manual)

`deploy ssh` and fixed-host `bench` go through `host.py` / `remote.py` and don't touch this orchestrator.

## Layout

```
provisioning/
  cloud.py        # orchestrator: provision_cloud_vm + provider dispatch
  lease.py        # durable allocation ownership, deletion, and audit
  candidates.py   # iter_candidates: ordered list of allocation attempts
  errors.py       # CapacityExhausted, TerminalProvisionError
  cloudrift.py    # CloudRift API wrapper (create/delete/wait)
  gcp.py          # gcloud-compute wrapper
  ssh.py          # generic wait_for_ssh
  host.py         # RemoteHost abstraction over an existing SSH target
  remote.py       # bare-VM bootstrap (driver/CUDA install)
  staging.py      # tar-and-scp helpers used by the deploy layer
  shell.py        # async shell-out helper
  types.py        # VMConnectionInfo dataclass
```

## Allocation model

`provision_cloud_vm()` enumerates *candidates* via `iter_candidates()`. A candidate is one concrete `(provider, instance_type, zone?)` tuple. Order:

1. Filter `hardware.GPU_INSTANCE_TYPES[gpu_name]` by `--provider` if set; else use the full preference-ordered list.
2. For each `(provider, base_type)` entry:
   * **CloudRift** → one candidate per base type.
   * **GCP** → one candidate per zone in `hardware.GPU_GCP_ZONES[gpu_name]` (falls back to `DEFAULT_GCP_ZONE`); all zones for the current base type before advancing to the next entry.

The orchestrator tries candidates in this order until one succeeds or all are exhausted. Without a provider filter,
fallback follows the hardware table across providers. An explicit `--provider` restricts the entire candidate list,
so callers that request CloudRift are never silently relocated to GCP.

For each candidate, the orchestrator makes up to `SAME_CANDIDATE_RETRIES` (= 2) attempts on transient errors. On the contracted exceptions it short-circuits:

| Provider raises | Orchestrator does |
|---|---|
| `CapacityExhausted` | Advance to next candidate immediately. |
| `TerminalProvisionError` | Abort. Propagate to caller. |
| Any other `Exception` | Treat as transient: retry same candidate up to `SAME_CANDIDATE_RETRIES`, then advance. |
| Returns `None` | Treat as soft capacity exhaustion: advance. |

When every candidate is exhausted, the orchestrator returns `None` and logs the last error.

`vm create gpu --exact-gpu-count` filters out provider shapes that would allocate more GPUs than requested. The normal
deploy and benchmark paths retain the hardware table's smallest-fitting GCP behavior.

## Durable allocation lease

Automation can pass a `VmLeaseObserver` to `provision_cloud_vm()`. Before every attempt, the observer refuses to
overwrite an active handle. Each provider then records its deletion handle immediately after allocation and before
readiness or SSH polling; once ready, the orchestrator adds connection details and marks the lease active. This
explicit lifecycle replaces interception of provider internals and preserves the handle if the process is interrupted
after allocation.

`emmy vm create gpu --lease PATH --owner ID` enables the observer. `emmy vm delete lease` validates the exact owner,
deletes only the recorded handle, retries and polls provider state, then marks the lease deleted. `emmy vm audit
lease` independently fails while that handle remains active. A missing lease is an idempotent no-op; an owner mismatch
is always a hard refusal.

**Timing:** `bench` (`benchmark/execution.py`) wraps `provision_cloud_vm()` → `vm_provision` and `provision_remote()`
→ `remote_provision` in a timer. These run once per `ExecutionGroup` (shared VM) but are seeded into each task's timer,
so every task's result reflects its host's stand-up cost. `vm_provision` is omitted for pre-allocated/fixed/local hosts
(no VM created). See `emmy/commands/ARCHITECTURE.md` → Timing metrics.

## Command source staging

Command recipes stage only the Git-visible files under their declared paths: tracked and untracked files are included,
while ignored files are excluded. Staging returns a manifest containing the Git revision, selected paths, per-file
SHA-256 digests, working-tree status, and a content-derived source ID. `command.strict` rejects dirty selected paths
before any transfer. The same manifest is retained in the command result, binding a measurement to the bytes sent to
the remote host without requiring a `.git` directory there.

## Error contract

`errors.py` defines two exceptions that providers raise to communicate intent:

* **`CapacityExhausted`** — the current candidate has no capacity. Raised for CloudRift HTTP 503/429 on rent, CloudRift `Inactive` / `Failed` terminal status / readiness timeout, GCP `ZONE_RESOURCE_POOL_EXHAUSTED` / `QUOTA_EXCEEDED` / `STOCKOUT` in `gcloud create` stderr, and GCP RUNNING-status timeout. Recoverable by trying a different candidate.
* **`TerminalProvisionError`** — auth, malformed request, anything that will recur on any candidate. Raised for CloudRift HTTP 4xx (other than 429) and unrecognized non-zero `gcloud create` exit. Not recoverable; propagated to caller.

Anything else a provider raises is treated as transient by the orchestrator.

## Orphan cleanup invariant

**A provider's `create_instance` must never leave a VM behind that the caller can't see.** If `create_instance` returns
a `VMConnectionInfo`, the caller owns the VM and is responsible for delete. If it raises, the provider attempts to
terminate any partially-provisioned instance before re-raising. When an allocation observer is present, the durable
lease remains the final cleanup authority even if provider cleanup or the caller is interrupted.

This invariant lets the orchestrator iterate candidates without leaking orphans:

* `cloudrift.create_instance` wraps `wait_for_status` in try/except and calls `_terminate_instance` on any exception or `None` return.
* `gcp.create_instance` wraps the post-`gcloud create` flow (`wait_for_status`, external-IP fetch, SSH wait) and runs `gcloud delete` on any exception. The pre-create classification branch (`_classify_create_failure`) only fires when nothing was provisioned, so no cleanup is needed there.

Both providers swallow termination errors and log them — the original failure is what the orchestrator needs to classify the next move, not a cascade from cleanup.

## CloudRift image selection

`cloudrift.select_image_url` maps the resolved instance type to one of three VM images:

| Instance prefix       | Image                                                  | Why                                                              |
|-----------------------|--------------------------------------------------------|------------------------------------------------------------------|
| `mi*` (AMD Instinct)  | `DEFAULT_IMAGE_URL_AMD` (Ubuntu + ROCm)                | ROCm kernel module                                               |
| `v100*`, `p100*`      | `DEFAULT_IMAGE_URL_NVIDIA_PROPRIETARY` (R580 proprietary) | Pre-Turing NVIDIA GPUs have no GSP and can't load the open kernel modules baked into the default image |
| anything else (NVIDIA)| `DEFAULT_IMAGE_URL_NVIDIA` (R580 open)                 | Standard path for Turing-and-newer                               |

Mismatches leave the GPU unusable because the wrong kernel-module flavor is on disk. The proprietary-driver image
mirrors the recipe CloudRift's `rift-console` surfaces only for hosts whose `brand_short` matches `/\bV100|P100\b/`.

## CloudRift API protocol version

Every CloudRift request carries an envelope `{"version": API_VERSION, "data": {...}}`. The server versions its public
types by calendar date and decodes each request against the newest declared schema whose date is `<= API_VERSION` (an
unknown in-between date silently resolves *down* to the nearest older schema). `API_VERSION` (`cloudrift.py`) is pinned
to `2026-08-05`, the current public generation for the instance endpoints used here. Pin to a date rather than
`~upcoming` (CloudRift's own client default) so a future server
release can't change request/response shapes under us.

Two v059-era behaviours the client relies on:

* **`instances/list` mask.** v058 added a `mask` (`with_connection_info` / `with_hardware_info` / `with_usage_info`,
  all default-false) and honours it for v058+ callers — older callers were force-fed `ALL`. `_get_instance_info` sends
  `{"with_connection_info": True}` only: `host_address` / `port_mappings` / login info are gated behind that flag, while
  `status` and `virtual_machines` (the `ready` flag + credentials) are ungated. We read no hardware/usage fields, so
  leaving those flags off lets the server skip those lookups. **Dropping the mask would null out host/port and break
  SSH.**
* **`Failed` status.** v059 reports a first-class `Failed` status plus a `failure` object (`cause`, `user_message`).
  `wait_for_status` treats `Failed` as terminal regardless of the caller's `fail_statuses` and logs `user_message`, so a
  failed rental fails fast with a reason instead of polling until timeout (the `None` return then flows through the
  orphan-cleanup path to `CapacityExhausted`).

## Adding a new provider

See `commands/ARCHITECTURE.md` § *Adding a New VM Provider*. The provider's `create_instance` must:

1. Raise `CapacityExhausted` on no-capacity / no-stock signals.
2. Raise `TerminalProvisionError` on credentials / bad-request errors.
3. Terminate any orphan it created before re-raising or returning `None`.
4. Return `VMConnectionInfo` (with `delete_info` set) on success.

Then add it to `_provision_candidate` in `cloud.py` and to `iter_candidates` if it has zone- or region-style fan-out.
