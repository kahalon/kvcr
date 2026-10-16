# Composable KV Hints: Per-Action Dispatch Infrastructure

Design for handling multi-action KV hint envelopes in KVCR. The first action
built on this infrastructure is a `kv.fetch` whose destination is KVCR local
DRAM (a prefetch from a remote node's DRAM into this node's KVCR-owned DRAM).

## Summary

KVCR treats a KV hint envelope as a batch of independent actions rather than
as a single fetch hint. Each action type has its own parser and handler, its
own lifecycle, its own completion, and runs under shared admission budgets so
that no action class can slow the engine-driven `fetch`/`deliver` path. Future
actions such as invalidate, evict, priority or retention declarations, and
move mode are added by registering a parser and a handler, without touching
the dispatch path or the engine path again. KVCR core stays framework-neutral;
vLLM is the current reference integration, and SGLang and TensorRT-LLM
integrations consume the same contract.

## Context

KVCR receives KV hints from a KV router through a framework integration layer
that calls `KVCR.submit_hint(envelope, request_id)`. The router protocol is
posted: no acknowledgement flows back. The envelope format is shared across
the framework RFCs (TensorRT-LLM #18153, vLLM #59708, SGLang #36224): several
actions per envelope, each with a source and a destination, per-action failure
isolation, and the rule that submitted does not mean ready.

Before this design KVCR:

- parsed only the first `kv.fetch` action and ignored the rest;
- stored the hint as per-request metadata in `_RemoteFWDram._request_hints`
  and moved no data on receipt; the engine later called `query`, `fetch` or
  `deliver`, which started asynchronous pulls on the single progress thread,
  and the engine collected results from `poll_completed`;
- allowed one source per request; a second source poisoned the request, and
  any failure poisoned the whole request;
- deleted the per-request hint in `discard_hint` when the request ended.

Pulls for different requests already ran concurrently on the progress thread.
The one-at-a-time limit was not a lock; it was the first-action-only parser,
the single-source rule, and the per-request failure model.

## Goals, non-goals, anti-goals

**Goals.** Multi-action parsing with per-action isolation. An action registry
and a deferred-admission lane that future actions reuse unchanged.
CPU-destination fetch end to end as the first deferred action. Per-action
completion visible in telemetry and key history. Configurable eviction for
prefetch fills. No measurable change to the engine path.

**Non-goals for this iteration** (documented, not built): the additional
action types themselves (`kv.invalidate`, `kv.evict`, priority or retention,
`mode=move`, `no_retain`); coalescing queued prefetches by block hash across
actions; priority scheduling of engine pulls over prefetch pulls on the
progress thread; multiple engine-destination sources per request; the
side-band listener itself.

**Anti-goals.** Adding lock acquisition or data movement to the
engine-destination `submit_hint` path. Adding worker threads that touch NIXL
or ZMQ; that state is single-threaded by design.

## Design

### Action registry and two execution lanes

- **One envelope, N actions.** Each is dispatched and completed independently.
  An invalid or unsupported action never affects its siblings.
- **Action registry.** The parser holds a table from `action_type` to a
  payload parser; core holds a table from `action_type` to a handler. A
  handler declares how to execute its actions inline at submit time and, when
  it defers work, how to admit it later under budget.
- **Inline lane.** Cheap, lock-free metadata updates on the caller's thread.
  The engine-destination fetch hint lives here.
- **Deferred lane.** Anything touching block records or starting transfers is
  queued and admitted in `poll_completed` under the state lock, subject to the
  budgets below. CPU-destination fetch lives here; so would invalidate or
  evict.
- **Side-band readiness.** A single internal entry point, `submit_actions`,
  takes a parsed action list with an optional request id. `submit_hint` is a
  thin adapter onto it, and a future side-band listener calls the same entry
  point.

### `kv.fetch` with two destinations

- **Engine destination** (unchanged behavior): request-scoped and advisory.
  The engine reserves memory and drives `fetch` and `deliver`.
- **KVCR DRAM destination** (new): KVCR itself starts a pull into local DRAM.
  The block becomes READY in `LOCAL_G2` and is owned by eviction policy from
  then on. It is not stored in the per-request hint, is unaffected by
  `discard_hint`, and its failure never poisons a request.

### Protocol shape: reuse the existing medium vocabulary

The `kv.fetch` payload KVCR parsed before this design carried no destination
field (source endpoint, block hashes, mode, no_retain). The hint ecosystem
already has a medium and locality vocabulary, and the design reuses it rather
than adding new tier names:

- per-request tier matchers in `kv_transfer_params` are dicts with `medium`
  (`CPU`, `STORAGE`) and `locality` (`LOCAL`, `REMOTE`) keys;
- the RFC's `src` and `dst` tuples name a worker and a medium (HBM, DRAM);
- KVCR's `CacheTier` values are `HBM`, `FW_DRAM`, `DRAM`, `REMOTE_G2`, `G3`.

The payload gains an optional `destination` object with the same keys as a
tier matcher:

```json
{"action_type": "kv.fetch", "action_version": "1.0",
 "payload": {"source_control_endpoint": "...", "block_hashes": [...],
             "mode": "copy", "destination": {"medium": "CPU"}}}
```

- absent, or `medium: "GPU"`: engine destination (the engine owns GPU memory
  and drives `fetch` and `deliver`);
- `medium: "CPU"`: KVCR local DRAM (prefetch; maps to `CacheTier.LOCAL_G2`);
- `medium: "STORAGE"`: KVCR G3; reserved and rejected until a storage-fill
  action exists;
- `locality` is accepted and must be `LOCAL` or absent for a destination. It
  stays in the schema so a later action can name a remote destination.

Medium values match case-insensitively, as the vLLM matcher parser does.
Unknown media or localities reject that action only. Actions with unknown
`action_type` are skipped. Internally the parsed destination is a `CacheTier`
(`FW_G1`, `LOCAL_G2`, `G3`), so no new enum is introduced.

### Key resolution: router hash to `BlockKey`

A router hint carries 64-bit block hashes. KVCR keys are framework-defined
`BlockKey`s, and the mapping from a hash to the keys that store it is owned by
each framework's `KeyAdapter`. In general it is one-to-many (a framework may
keep several KV entries per hash, for example per cache group), and `encode`
is not an inverse of a router hash. For engine-destination hints this never
matters because the engine supplies the keys and KVCR only compares
`decode(key)` against the hint.

Prefetch has no engine call, so KVCR must obtain keys itself. The
`KeyAdapter` protocol gains an optional method:

```python
def keys_for_block_hash(self, block_hash: int) -> Sequence[BlockKey]: ...
```

It returns every `BlockKey` that stores that hash. CPU-destination actions
are rejected at dispatch, with a once-logged warning, when the adapter does
not implement it. Each framework integration implements it in its own
repository; KVCR unit tests use a test adapter. The source side needs no
change: it is addressed by exact `BlockKey`.

**Lookup contract.** A prefetched block is only useful if the engine's later
`query`, `fetch` or `deliver` finds it, and those look up the block record map
by the exact `BlockKey` the engine computes. The contract on
`keys_for_block_hash` is therefore: for every hash, the returned keys are
byte-identical to the keys the framework will later pass for that hash, and
`decode(key)` equals the hash for each of them. KVCR enforces what it can:

- at admission, each resolved key is round-tripped through `decode`. A
  mismatch drops the key, counts it with reason `key_mismatch`, and logs once
  per adapter;
- unit tests: after a prefetch completes, `query` with no request id returns
  `HIT` in `LOCAL_G2`, and a subsequent `fetch` for the same keys returns a
  local claim without starting a remote pull;
- integration tests, per framework and outside this repository: prefetch via
  a real envelope, then run the framework's own lookup path for the same
  hashes and assert a local hit.

**Inventory publication to the router.** No new code is needed. A prefetch
completes through the same fill path as an engine fetch. On success
`_apply_fill_result` flips the residency from FILLING to READY, calls the
policy's `on_ingest` hook, and publishes an `InventoryEvent` for `LOCAL_G2` to
the integration's `inventory_sink`. The integration forwards it to the router
as it does for any other local DRAM admission; the vLLM adapter maps it to an
offloading event with medium CPU and ownership `kvcr`. Eviction of a
prefetched block publishes the matching removal event through the existing
paths.

### Concurrency model

No new threads. The progress thread already runs any number of target pull
operations concurrently; prefetch pulls are more of them, with no request id
and `local_fill` set. Engine-destination hints keep the lock-free store.
Prefetch admission touches block records, so it runs under the state lock on
the main thread. To keep `submit_hint` cheap and to give a future side-band
thread a safe hand-off, prefetch actions are enqueued and admitted in
`poll_completed`, which already holds the state lock.

This makes two integration contracts explicit: every integration must call
`poll_completed` regularly even when it has no operations outstanding, and
`poll_completed` may run on a latency-sensitive path such as a scheduler
loop, so admission work per call is budgeted:

- **Ingress bound.** The queue counts queued keys, not actions. An action
  whose keys would push the count past `max_queued_prefetch_blocks` is dropped
  at submit time (reason `queue_full`). Dropping is constant-time.
- **Stale expiry.** Each queued action carries its submission time. At
  admission, an action older than `prefetch_timeout_ms` is dropped (reason
  `stale`) without touching block records.
- **Per-poll budget.** Admission resolves and reserves at most
  `prefetch_admit_blocks_per_poll` keys per `poll_completed` call. A partially
  admitted action stays at the head of the queue with its remaining hashes, so
  work is resumable and each poll's time under the lock is bounded.
- **In-flight budget per pool.** Prefetch reservations are capped at
  `prefetch_reserve_limit_percent` of each pool's slots, counting keys in
  prefetch fills not yet READY. This bounds progress-thread work and
  guarantees foreground headroom: a stalled prefetch slot is FILLING and not an
  eviction candidate, so without this budget a burst could fill the pool and
  make a disjoint demand fetch fail outright. Keys over budget are dropped
  (reason `reserve_limit`).

Priority of foreground submissions over prefetch on the progress thread stays
a documented follow-up; the budgets above bound how much prefetch work can be
in flight at once.

### Deadlines at both peers

The target's deadline is `prefetch_timeout_ms`, but the source clamps the
transmitted remaining timeout to its own `operation_timeout_ms`, so source
pinning that takes longer than one second would still fail the prefetch. The
`start_write` message gains an optional `op_class` field with value
`prefetch`. A source that understands it clamps to its `prefetch_timeout_ms`
instead. Sources that predate the field ignore it and keep the
`operation_timeout_ms` clamp.

### Eviction

`reserve_fill` evicts READY blocks when free slots run out. A new flag,
`prefetch_may_evict` (default true), is threaded into `reserve_fill` so that,
when false, a prefetch only fills free capacity and otherwise drops the key
with no capacity waiter. This flag is independent of the in-flight budget:
eviction controls whether prefetch may displace READY blocks, the budget
controls how many slots prefetch may hold in FILLING.

## Design decisions

| Decision | Choice | Reason |
| --- | --- | --- |
| Payload field | `destination: {medium, locality?}` | Reuses the tier-matcher vocabulary already in the ecosystem instead of new tier names. |
| Prefetch layout | From config, `prefetch_layout`; required for CPU-destination actions, no inference | A residency's layout is fixed at reservation and a later engine fetch with a different layout fails that key. A single configured pool does not determine the layout (named pools and repeated-pool layouts exist). |
| Engine fetch on an in-flight prefetch | Keep existing semantics: the engine fetch waits on the prefetch fill and fails with it | Matches existing concurrent-fetch behavior. Fallback to the request's own source is a follow-up. |
| Two engine-destination sources in one envelope | Unchanged: the second source poisons the request | Multi-source per request is separate work. |
| Same-source engine actions | Aggregate per source (union of hashes) before storing | Storing per action would replace the hash set and lose earlier actions when opportunistic query is off. |
| Prefetch deadline and granularity | `prefetch_timeout_ms` (default 5000 ms); each action split into pulls of at most `prefetch_blocks_per_op` keys | A local fill that misses its deadline loses every key, and router hints can carry whole request paths. |
| Asynchronous eviction (G3 configured) | Keys waiting on an in-flight G3 spill are dropped and counted (reason `eviction_pending`) | The existing capacity waiter needs a pending residency op and a per-request hint to resume. Documented limitation: under G3 pressure prefetch admits little. |
| Bounds | Four finite budgets (queue, stale, per-poll, per-pool reserve) | A single in-flight cap did not bound queued memory, per-poll work under the lock, or FILLING slots that starve demand fetches. Coalescing by block hash remains a follow-up. |
| Policy visibility | Prefetch admission reaches `decide_ingest` as source `REMOTE_G2` with `required_local=False` | Unique among call sites, so a policy can treat prefetches differently without an API change. |

## Configuration (`RemoteFWDramOptions`)

| Field | Default | Purpose |
| --- | --- | --- |
| `prefetch_may_evict` | `True` | Allow CPU-destination fills to evict READY blocks. |
| `prefetch_timeout_ms` | `5000` | Deadline per prefetch pull, applied at both peers via `op_class`. |
| `prefetch_blocks_per_op` | `64` | Chunk size per target pull operation. |
| `prefetch_layout` | `None` | Pool layout for prefetched residencies; required for CPU-destination actions. |
| `max_queued_prefetch_blocks` | `4096` | Ingress bound on queued keys; drop newest. |
| `prefetch_admit_blocks_per_poll` | `256` | Admission work budget per `poll_completed` call. |
| `prefetch_reserve_limit_percent` | `25` | Share of each pool that prefetch may hold in FILLING. |

## Telemetry

- `kvcr_hint_actions` counter, labels `(action_type, destination, outcome)`,
  outcomes `accepted`, `rejected`.
- `kvcr_prefetch_blocks` counter, label `(outcome,)`, outcomes `admitted`,
  `skipped_resident`, `dropped_queue_full`, `dropped_stale`,
  `dropped_reserve_limit`, `dropped_eviction_pending`, `dropped_no_capacity`,
  `dropped_key_mismatch`.
- `kvcr_prefetch_queued_blocks` gauge and `kvcr_prefetch_reserved_blocks`
  gauge with label `(pool,)`.
- Transfer scope `remote_prefetch` on the existing duration and transfer
  counters; key-history event `remote_prefetch_completed`;
  `KVCR_EVENT prefetch_admitted` debug log.

## Integration contract (every framework, outside this repository)

To enable CPU-destination actions, an integration must:

1. implement `KeyAdapter.keys_for_block_hash`, returning keys byte-identical
   to those its own lookup path will later use for the same hash;
2. configure `prefetch_layout` to match the layout it passes as
   `expected_layout` on `fetch`;
3. call `poll_completed` while idle.

Without the first two, CPU-destination actions are rejected and the engine
path is unaffected.

### vLLM (current reference)

- The vLLM key adapter's `encode` accepts only bytes offload keys, and
  `decode` strips a cache-group suffix, so `keys_for_block_hash` must produce
  one offload key per cache group.
- The adapter already polls while idle.
- Config plumbing for the new options goes through the KVCR secondary tier
  manager's extra config.

### SGLang and TensorRT-LLM (future)

Same three items. The hash-to-key mapping and layout are determined by each
framework's KV layout. Nothing in KVCR core assumes a particular framework.

## Adding a future action

For `kv.invalidate`, `kv.evict`, or a priority declaration, the work is: a
payload parser registered in `hint_parser.py`, a handler in `hint_actions.py`
implementing `submit` (inline) and, when it defers work, `admit` (under the
state lock with budget), telemetry labels, and tests. Envelope handling,
per-action isolation, the deferred admission step, and the side-band entry
point are shared and unchanged.

## Follow-ups

- Next action types: `kv.invalidate`, `kv.evict`, priority or retention,
  `mode=move`, `no_retain`.
- Coalescing queued prefetches by block hash across actions.
- Priority of engine pulls over prefetch pulls on the progress thread.
- Fallback for engine fetches that waited on a failed prefetch.
- Prefetch-aware capacity waiter for asynchronous (G3) eviction.
- Side-band ingress (for example a ZMQ listener) feeding `submit_actions`.
- Multiple engine-destination sources per request.
