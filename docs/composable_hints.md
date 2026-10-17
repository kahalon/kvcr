# Composable KV Hint Handling

Design for handling multi-hint envelopes in KVCR. This document
covers the hint-agnostic infrastructure only: parsing every hint, routing
each to a handler, running deferred work asynchronously under budgets, and
reporting a result per hint.

## Summary

KVCR treats a KV hint envelope as a batch of independent hints. Each hint
type has its own parser and handler. A handler runs cheap work inline at
submit time, or defers work that must run under KVCR's state lock to the next
`poll_completed`, where the deferred lane admits it under a per-poll budget
with stale expiry. Every hint produces a result (accepted, rejected,
completed, failed, or dropped, with a reason) that the integration drains
as telemetry and, when retention is enabled, as drainable results.

## Context

KVCR receives KV hints from a KV router through a framework integration layer
that calls `KVCR.submit_hint(envelope, request_id)`. The router protocol is
posted: no acknowledgement flows back to the router. The envelope format is
shared across the framework RFCs (TensorRT-LLM #18153, vLLM #59708, SGLang
#36224): several actions per envelope, per-hint failure isolation, and the
rule that submitted does not mean ready.

## Design

### Hint model

Each parsed hint is a frozen `msgspec.Struct` with at least `action_id:
str | None` and a class-level `ACTION_TYPE: str`. Two registries live in the
`kvcr.hints` package and are filled by the hint modules themselves at import
time: `parser.register_hint_parser` maps `action_type` to
`(supported_version, parse_payload)`, and `dispatcher.register_hint_handler`
maps `action_type` to a handler factory called as `factory(core, dispatcher)`.
A `HintDispatcher` instantiates every registered handler for its core; a
handler uses `core` for backend access and the dispatcher to `report`
results and `defer` hints. Core holds one dispatcher and never names a
handler.

### Results

```python
class HintStatus(Enum):
    ACCEPTED = "ACCEPTED"    # inline work done, or deferred work queued
    REJECTED = "REJECTED"    # never queued: invalid, unsupported, handler_error
    COMPLETED = "COMPLETED"  # deferred work finished successfully
    FAILED = "FAILED"        # deferred work finished unsuccessfully
    DROPPED = "DROPPED"      # queued but never run: queue_full, stale, closed

class HintResult(msgspec.Struct, frozen=True):
    action_id: str | None
    action_type: str
    status: HintStatus
    reason: str | None = None
    request_id: str | None = None
    message_id: str | None = None
```

Every result is recorded in the `kvcr_hints` counter. Results can
also be retained in a bounded queue (`max_hint_results`; when full the
oldest is discarded and counted) and drained internally through
`HintDispatcher.drain_results()`. The default is `0`: nothing is retained,
because no integration consumes results yet. A public drain on `KVCR` is
added when the first consumer needs it. A hint whose
`action_type` cannot be read (the hint is not a mapping, or its type is not
a string) is reported with the sentinel `action_type="unknown"`. An inline hint reaches a terminal status
(`ACCEPTED` or `REJECTED`) at submit time; an inline handler that raises has
its hints reported `REJECTED` with reason `handler_error`. A deferred hint
reports `ACCEPTED` when queued and later exactly one of `COMPLETED`, `FAILED`,
or `DROPPED`. A fatal KVCR failure (the progress thread failed) is not an
hint result: `submit_hint` and `submit_hints` raise it before dispatching,
as `poll_completed` and `get_stats` do, and report nothing.

### Parsing

`parse_hints(envelope) -> (hints, rejections)`. A structurally
invalid envelope (not a mapping or struct, `actions` not a list) raises
`ValueError`, as today. Everything else is per hint and never raises: a
non-mapping hint, an unknown `action_type`, a non-mapping payload, or a
payload the type's parser refuses produces a `HintResult(REJECTED,
reason)` and the loop continues. Protocol and hint version mismatches warn
once per issue. An envelope whose hints are all rejected returns an empty
hint list and the rejections; it does not raise.

### Two lanes

- **Inline lane.** `handler.submit(hint, request_id, message_id)` runs on
  the caller's thread, without the state lock, once per hint in envelope
  order. It must only touch state that is safe there and must report exactly
  one result for the hint, either directly or by deferring. A handler
  exception is caught, logged once, and reported `REJECTED` with reason
  `handler_error` for that hint.
- **Deferred lane.** A handler defers a hint by calling
  `dispatcher.defer(handler, hint, request_id, message_id)`. The lane is
  owned by the dispatcher and is generic:
  - a bounded queue (`max_queued_hints`, counted in hints; when full
    the new hint is reported `DROPPED` with reason `queue_full`);
  - admission inside `poll_completed`, under `_state_lock`, before the
    backends' `poll_main`;
  - stale expiry: an item older than `hint_timeout_ms` at admission is
    reported `DROPPED` with reason `stale` without calling the handler;
  - a per-poll budget of `hints_per_poll` admitted hints, so a
    latency-sensitive `poll_completed` caller sees bounded work;
  - `handler.admit(hint, request_id, message_id, now) -> bool` is called
    per item; `True` means the item is consumed (the handler owns its later
    `COMPLETED`/`FAILED` report); `False` means the handler could not take it
    now, the item stays at the head, and admission stops for this poll;
  - a handler exception during `admit` is caught, logged once, and reported
    as `FAILED` with reason `handler_error`; admission continues;
  - `close()` drains the queue, reporting `DROPPED` with reason `closed`.

The lane holds its own lock for the queue only and never calls a handler
while holding it. No new threads are added.

### Integration contracts

Every integration must call `poll_completed` regularly even when it has no
operations outstanding, because deferred hints are admitted there. The
call stays cheap when nothing is queued.

### `kv.fetch`

Module `hints/fetch.py`, the only registered hint type today; it stores
the per-request remote hint that the engine later turns into `fetch` and
`deliver` calls.

The `kv.fetch` payload parser keeps its current rules (source endpoint
required, `mode` must be `copy`, `no_retain` refused, 64-bit hashes). A
`destination` object, if present, must name the engine (`medium` GPU);
anything else is rejected as unsupported, which leaves room for a later hint
to claim other destinations without changing the infrastructure.

Each `kv.fetch` hint is one call to the existing per-request hint store,
reported `ACCEPTED`. The store's semantics are unchanged: a later hint from
the same source replaces the stored hashes, and a hint naming a different
source marks the request's stored hint failed, so the request is no longer
served remotely. Both apply within one envelope exactly as across calls,
and the store does not report either case, so one `kv.fetch` per envelope
is the supported shape. The engine path takes no lock and starts no
transfer beyond the pre-existing eager control connect. Reporting
`REJECTED` when the store refuses a conflicting source would need the store
to report acceptance; that is left for later.

### Configuration (`KVCRConfig`)

| Field | Default | Purpose |
| --- | --- | --- |
| `max_queued_hints` | `1024` | Bound on deferred hints awaiting admission; newest dropped as `queue_full`. |
| `hints_per_poll` | `64` | Deferred hints admitted per `poll_completed`. |
| `hint_timeout_ms` | `5000` | Age at which a queued hint is dropped as `stale`. |
| `max_hint_results` | `0` | Retained hint results, drained internally; `0` retains none (telemetry only). Oldest discarded when full. |

### Telemetry

- `kvcr_hints` counter, labels `(action_type, status)` with the
  `HintStatus` value, incremented on every reported result. An
  `action_type` with no registered handler is labelled `unknown` to keep the
  label set bounded; the `HintResult` keeps the original string.
- `kvcr_state{resource="hints_queued"}` gauge: deferred hints awaiting
  admission, sampled in every `get_stats()` snapshot alongside the other
  state gauges.
- `kvcr_hint_results_discarded` counter: results lost to the bound.

## Adding a hint type

A new hint type is one module in `kvcr/hints/`: the
parsed struct with its `ACTION_TYPE`, the payload parser registered with
`parser.register_hint_parser`, a handler class implementing `submit` and, if
it defers, `admit` plus its own completion reports, registered with
`dispatcher.register_hint_handler`, telemetry labels, tests, and a config
block of its own if it needs one. The module is imported at the bottom of
`dispatcher.py` so that importing the dispatcher registers it. Envelope
handling, isolation, the deferred lane, results, and the side-band entry
point are shared and unchanged, and `core.py` is not touched.

## Potential future work

- Per-type grouping of an envelope's hints, so a handler sees all of its
  hints from one envelope at once.
- Prefetch into KVCR local DRAM as a deferred hint (a `destination` payload
  field or a dedicated action type): a key-adapter
  method to resolve router hashes to keys, prefetch layout, eviction flag,
  per-pool reservation budget, explicit-source remote pulls with a prefetch
  timeout at both peers. A reference implementation exists on branch
  `composable-hints-prefetch-wip`.
- `kv.invalidate`, `kv.evict`, priority or retention, `mode=move`,
  `no_retain`.
- Side-band ingress (for example a ZMQ listener) feeding `submit_hints`.
- Priority between deferred hints and over engine pulls on the progress
  thread.
