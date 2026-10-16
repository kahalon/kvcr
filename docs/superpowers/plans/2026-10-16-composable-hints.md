# Composable KV Hints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parse every action in a KV hint envelope, dispatch each through a per-action handler registry, and ship the first deferred action: a `kv.fetch` with `destination.medium == "CPU"` that prefetches blocks from a peer KVCR into KVCR-owned local DRAM under admission budgets.

**Architecture:** `hint_parser.py` turns an envelope into typed actions through a parser registry. A new `hint_actions.py` holds the handler protocol and the fetch handler: engine-destination actions take the existing lock-free per-request store; CPU-destination actions go into a bounded deferred queue that `_KVCRCore.poll_completed` admits under `_state_lock` with per-poll, stale, and per-pool budgets. Admission reserves local DRAM through `reserve_fill` and starts `_TargetPullOp`s via a new `_RemoteFWDram.prefetch` that names its source explicitly and tags `start_write` with `op_class="prefetch"` so the source applies the prefetch timeout. No new threads.

**Tech Stack:** Python 3.10+, msgspec, pytest. Tests run with `.venv/bin/python -m pytest` in this worktree (a `nixl` stub is installed in the venv; Linux-only G3, guard, service and journal tests fail on macOS at baseline and are out of scope).

**Spec:** `docs/composable_hints_design.md`

## Global Constraints

- Engine-destination `submit_hint` path stays lock-free and starts no transfers (spec: anti-goals).
- No new threads; NIXL and ZMQ state is touched only on the progress thread (spec: concurrency model).
- All new `RemoteFWDramOptions` fields and defaults exactly as the spec's configuration table: `prefetch_may_evict=True`, `prefetch_timeout_ms=5000`, `prefetch_blocks_per_op=64`, `prefetch_layout=None`, `max_queued_prefetch_blocks=4096`, `prefetch_admit_blocks_per_poll=256`, `prefetch_reserve_limit_percent=25`.
- Destination media `GPU` (default), `CPU`, `STORAGE` (rejected), matched case-insensitively; `locality` must be `LOCAL` or absent.
- Telemetry names exactly as the spec's telemetry section.
- Existing tests in `tests/unit/test_hint_parser.py` and `tests/unit/test_kvcr_remote_target.py` keep passing unless a step says otherwise.
- Every source file keeps the SPDX header used across `src/kvcr/`.
- `ruff check` and `ruff format --check` pass on changed files (`.venv/bin/ruff`).

## Review Focus

1. Envelope with a valid CPU action and an invalid GPU action: the CPU action is still queued (per-action isolation). Pinned in Task 2 and Task 5.
2. Prefetch for a hash that resolves to a key already FILLING from an engine fetch: skipped, not reserved twice, no reserve count change. Pinned in Task 5.
3. `discard_hint` or `close` while a prefetch is queued but not admitted: queue drained on close without touching block records; nothing raised. Pinned in Task 5.
4. Prefetch pull that returns partial success: completed keys go READY, missing keys free their slots and reserve counts, no request hint poisoned. Pinned in Task 5.
5. `prefetch_layout` naming a pool that does not exist: rejected at construction, not at first hint. Pinned in Task 1.

---

### Task 1: Configuration and key-adapter contract

**Files:**
- Modify: `src/kvcr/config.py` (`RemoteFWDramOptions`, `KeyAdapter`)
- Modify: `src/kvcr/remote_fw_dram.py` (`_RemoteFWDram.__init__` validation, around line 504)
- Modify: `src/kvcr/core.py` (`_KVCRCore.__init__`, after `_validate_pool_layouts`)
- Test: `tests/unit/test_kvcr.py`

**Interfaces:**
- Produces: `RemoteFWDramOptions` fields `prefetch_may_evict: bool = True`, `prefetch_timeout_ms: int = 5000`, `prefetch_blocks_per_op: int = 64`, `prefetch_layout: tuple[str, ...] | None = None`, `max_queued_prefetch_blocks: int = 4096`, `prefetch_admit_blocks_per_poll: int = 256`, `prefetch_reserve_limit_percent: float = 25`.
- Produces: `KeyAdapter.keys_for_block_hash(self, block_hash: int) -> Sequence[BlockKey]` documented as optional (callers use `getattr(adapter, "keys_for_block_hash", None)`); the Protocol itself gains the method with a docstring saying integrations may omit it.
- Produces: `_KVCRCore._prefetch_layout: list[str] | None` (validated copy of the option).

- [ ] **Step 1: Write failing tests**

In `tests/unit/test_kvcr.py` add:

```python
def test_remote_options_prefetch_defaults():
    options = RemoteFWDramOptions()
    assert (options.prefetch_may_evict, options.prefetch_timeout_ms) == (True, 5000)
    assert (options.prefetch_blocks_per_op, options.prefetch_layout) == (64, None)
    assert options.max_queued_prefetch_blocks == 4096
    assert options.prefetch_admit_blocks_per_poll == 256
    assert options.prefetch_reserve_limit_percent == 25


@pytest.mark.parametrize(
    "field, value",
    [
        ("prefetch_timeout_ms", 0),
        ("prefetch_blocks_per_op", 0),
        ("max_queued_prefetch_blocks", -1),
        ("prefetch_admit_blocks_per_poll", 0),
        ("prefetch_reserve_limit_percent", 101),
        ("prefetch_layout", ("missing-pool",)),
    ],
)
def test_remote_options_prefetch_validation(field, value):
    with pytest.raises(ValueError):
        _new_kvcr(
            FakeNixlAgent(), FakePrimaryPinning(), FakeBytesControl(),
            remote_options=RemoteFWDramOptions(**{field: value}),
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_kvcr.py -k prefetch -q`
Expected: FAIL (unexpected keyword / no ValueError).

- [ ] **Step 3: Add the fields and validation**

Add the seven fields to `RemoteFWDramOptions` in `src/kvcr/config.py`. Add `keys_for_block_hash` to `KeyAdapter` with a docstring. In `_RemoteFWDram.__init__` raise `ValueError` when `prefetch_timeout_ms <= 0`, `prefetch_blocks_per_op <= 0`, `max_queued_prefetch_blocks < 0`, `prefetch_admit_blocks_per_poll <= 0`, or `not 0 <= prefetch_reserve_limit_percent <= 100`. In `_KVCRCore.__init__`, after `_RemoteFWDram` is constructed, set `self._prefetch_layout = list(layout)` and call `self._validate_block_layout(self._prefetch_layout, "prefetch_layout must use configured pools")` when the option is not `None`; else `None`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_kvcr.py -k prefetch -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/kvcr/config.py src/kvcr/remote_fw_dram.py src/kvcr/core.py tests/unit/test_kvcr.py
git commit -m "feat(hints): add prefetch options and keys_for_block_hash contract"
```

---

### Task 2: Parser registry and destination parsing

**Files:**
- Modify: `src/kvcr/hint_parser.py`
- Test: `tests/unit/test_hint_parser.py`

**Interfaces:**
- Produces: `class _KVFetchAction(msgspec.Struct, frozen=True)` with `action_id: str | None`, `source_endpoint: str`, `block_hashes: frozenset[int]`, `destination: CacheTier` and `ACTION_TYPE: ClassVar[str] = "kv.fetch"`.
- Produces: `_HintAction = _KVFetchAction` (type alias; widen to a Union when more actions exist).
- Produces: `parse_hint_actions(hint: Mapping[str, object] | msgspec.Struct) -> list[_HintAction]`.
- Produces: `register_action_parser(action_type: str, action_version: str, parse: Callable[[str | None, Mapping[str, object]], _HintAction]) -> None` and the module dict `_ACTION_PARSERS`. The parse callable receives `(action_id, payload)` and raises `ValueError` to reject that action.
- Produces: `_parse_destination(payload: Mapping[str, object]) -> CacheTier` mapping `GPU`→`CacheTier.FW_G1`, `CPU`→`CacheTier.LOCAL_G2`, `STORAGE`→`ValueError("STORAGE destination is not currently supported")`, missing→`FW_G1`.
- Keeps: `_KVFetchHint.from_hint` returning the first `FW_G1` fetch action as today (implemented on top of `parse_hint_actions`).

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/test_hint_parser.py`:

```python
from kvcr.hint_parser import (
    _ACTION_PARSERS, _KVFetchAction, parse_hint_actions, register_action_parser,
)
from kvcr.types import CacheTier


def _fetch_action(action_id, source, hashes, **extra):
    payload = {"source_control_endpoint": source, "block_hashes": list(hashes), **extra}
    return _HintAction(action_id, "kv.fetch", "1.0", payload)


def test_parse_hint_actions_returns_every_valid_fetch_action():
    hint = _HintEnvelope("0.1", "m", [
        _fetch_action("a", "tcp://a:1", (1, 2)),
        _HintAction("x", "kv.future_action", "1.0", {}),
        _fetch_action("bad", "", (3,)),
        _fetch_action("b", "tcp://b:1", (4,), destination={"medium": "cpu"}),
    ])
    assert parse_hint_actions(hint) == [
        _KVFetchAction("a", "tcp://a:1", frozenset({1, 2}), CacheTier.FW_G1),
        _KVFetchAction("b", "tcp://b:1", frozenset({4}), CacheTier.LOCAL_G2),
    ]


@pytest.mark.parametrize(
    "destination, expected",
    [
        (None, CacheTier.FW_G1),
        ({"medium": "GPU"}, CacheTier.FW_G1),
        ({"medium": "Cpu", "locality": "local"}, CacheTier.LOCAL_G2),
    ],
)
def test_parse_destination_media(destination, expected):
    extra = {} if destination is None else {"destination": destination}
    hint = _HintEnvelope("0.1", "m", [_fetch_action("a", "tcp://a:1", (1,), **extra)])
    assert parse_hint_actions(hint)[0].destination is expected


@pytest.mark.parametrize(
    "destination",
    [{"medium": "STORAGE"}, {"medium": "TAPE"}, {"medium": "CPU", "locality": "REMOTE"}, "CPU", {}],
)
def test_parse_destination_rejects_action_only(destination):
    hint = _HintEnvelope("0.1", "m", [
        _fetch_action("bad", "tcp://a:1", (1,), destination=destination),
        _fetch_action("ok", "tcp://a:1", (2,)),
    ])
    assert [a.action_id for a in parse_hint_actions(hint)] == ["ok"]


def test_parse_hint_actions_raises_when_nothing_valid():
    hint = _HintEnvelope("0.1", "m", [_fetch_action("bad", "", (1,))])
    with pytest.raises(ValueError):
        parse_hint_actions(hint)


def test_registered_action_type_is_parsed(monkeypatch):
    class _Stub(msgspec.Struct, frozen=True):
        action_id: str | None
        ACTION_TYPE = "kv.test"
    monkeypatch.setitem(_ACTION_PARSERS, "kv.test", ("1.0", lambda i, p: _Stub(i)))
    hint = _HintEnvelope("0.1", "m", [_HintAction("s", "kv.test", "1.0", {})])
    assert parse_hint_actions(hint) == [_Stub("s")]
```

(`{}` as a destination means `medium` missing inside a present object: reject. `"CPU"` as a bare string: reject.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_hint_parser.py -q`
Expected: new tests FAIL with ImportError; existing tests PASS.

- [ ] **Step 3: Implement the registry**

In `src/kvcr/hint_parser.py`: add `_KVFetchAction`; move the per-payload checks from `from_hint` into `_parse_kv_fetch(action_id, payload) -> _KVFetchAction` (keep the `no_retain`, `mode`, source, and hash rules verbatim; add `_parse_destination`); define `_ACTION_PARSERS = {"kv.fetch": ("1.0", _parse_kv_fetch)}` and `register_action_parser`; implement `parse_hint_actions` (envelope checks as today, per action: skip non-mapping/struct, skip unknown type, version warning via `_warn_version_mismatch` with issue `f"{action_type}-version-mismatch"`, call parser, on `ValueError` log once per `(action_type, str(error))` via `_LOGGED_HINT_ISSUES` and continue; raise `ValueError("invalid router hint")` when the result is empty). Reimplement `_KVFetchHint.from_hint` as the first `FW_G1` action from `parse_hint_actions`, raising `ValueError` when none. Import `CacheTier` from `.types`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_hint_parser.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/kvcr/hint_parser.py tests/unit/test_hint_parser.py
git commit -m "feat(hints): parse every envelope action through a parser registry"
```

---

### Task 3: Local DRAM `allow_evict` flag

**Files:**
- Modify: `src/kvcr/local_dram.py` (`reserve_fill` ~line 857, `_allocate_slots` ~line 1204)
- Test: `tests/unit/test_kvcr_local_dram.py`

**Interfaces:**
- Produces: `reserve_fill(..., layout: list[str], allow_evict: bool = True)`; `_allocate_slots(layout, protected, deadline, *, allow_evict: bool = True)`. With `allow_evict=False` and insufficient free slots, `_allocate_slots` returns `(None, [], False)` before the candidate search, so `reserve_fill` leaves the key out of both `destinations` and `eviction_pending`.

- [ ] **Step 1: Write failing test**

In `tests/unit/test_kvcr_local_dram.py`, using the existing `_new_local_kvcr` pattern with a 2-slot pool, deposit two blocks so the pool is full and READY, then call `kvcr._core._local_dram.reserve_fill((BlockKey(b"new"),), sources={BlockKey(b"new"): CacheTier.REMOTE_G2}, required_local=False, deadline=kvcr._core._operation_deadline(), layout=[""], allow_evict=False)` inside `kvcr._core._state_lock` after creating the record with `kvcr._core._block_record(key)`. Assert it returns `({}, set())`, both deposited keys are still READY, and the policy recorded no `decide_eviction` call. Then repeat with `allow_evict=True` and assert one key was evicted and `destinations` has the new key.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_kvcr_local_dram.py -k allow_evict -q`
Expected: FAIL with TypeError (unexpected keyword).

- [ ] **Step 3: Implement the flag**

Thread `allow_evict` through `reserve_fill` into `_allocate_slots`; early-return `(None, [], False)` when `not allow_evict` and the free-slot check fails.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/test_kvcr_local_dram.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/kvcr/local_dram.py tests/unit/test_kvcr_local_dram.py
git commit -m "feat(local-dram): allow fills that must not evict"
```

---

### Task 4: Remote backend: explicit-source prefetch pull and `op_class`

**Files:**
- Modify: `src/kvcr/remote_fw_dram.py` (`_TargetPullOp` ~line 92, `_start_target_pull` ~line 593, `_finish_target_pull` ~line 794, `_handle_start_write` ~line 1084)
- Test: `tests/unit/test_kvcr_remote_target.py`, `tests/unit/test_kvcr_remote_source.py`

**Interfaces:**
- Produces: `_TargetPullOp.op_class: str | None = None`; scope is `"remote_prefetch"` when `op_class == "prefetch"`, else as today.
- Produces: `_RemoteFWDram.prefetch(self, source: str, blocks: Mapping[BlockKey, list[_TransferRef]], deadline: float, *, op_handle: OpHandle) -> bool`. Submits a `_TargetPullOp(local_fill=True, request_id=None, op_class="prefetch", remote_ctrl_ep=source)` with `kvcr._add_block_dependencies(op, new_operation=True)`, logs `KVCR_EVENT target_transfer_started scope=remote_prefetch ...` at DEBUG, returns `True`; returns `False` when `blocks` is empty.
- Produces: `_start_target_pull` split into `_resolve_hint_source(request_id, blocks, scope, started_at) -> tuple[str, tuple[BlockKey, ...]] | None` (the current lines 605-620) and `_submit_target_pull(source, keys, blocks, request_id, deadline, op_handle, *, local_fill, op_class)`; `deliver`/`fetch` behave exactly as before.
- Produces: `start_write` payload includes `"op_class": self.op_class` only when not `None`. Source side: `op_class = payload.get("op_class")`; clamp to `self._options.prefetch_timeout_ms` when `op_class == "prefetch"`, else `self._kvcr.config.operation_timeout_ms`; any non-string `op_class` is treated as `None`.
- Produces: key-history event `remote_prefetch_completed` in `_finish_target_pull` for `op_class == "prefetch"` (local fills otherwise keep `remote_fetch_completed`).

- [ ] **Step 1: Write failing tests**

In `test_kvcr_remote_target.py`:

```python
def test_prefetch_pull_names_source_and_op_class(monkeypatch, caplog):
    monkeypatch.setenv("KVCR_KEY_HISTORY", "1")
    caplog.set_level(logging.DEBUG, logger="kvcr.core")
    local = ctypes.create_string_buffer(16 * 2)
    agent = FakeNixlAgent(metadata=b"target-md")
    control = FakeBytesControl("tcp://target:1")
    kvcr = _new_kvcr(
        agent, FakePrimaryPinning(), control,
        local_dram=LocalDramOptions([("", ctypes.addressof(local), len(local))]),
        remote_options=RemoteFWDramOptions(eager_ctrl_connect=False, prefetch_timeout_ms=5000),
    )
    core = kvcr._core
    key = BlockKey(b"p0")
    with core._state_lock:
        core._block_record(key)
        destinations, pending = core._local_dram.reserve_fill(
            (key,), sources={key: CacheTier.REMOTE_G2}, required_local=False,
            deadline=core._clock() + 5, layout=[""],
        )
        assert not pending
        handle = core._next_fill_handle
        core._next_fill_handle -= 1
        assert core._remote_fw_dram.prefetch("tcp://source:1", destinations, core._clock() + 5, op_handle=handle)
    _wait_until(lambda: bool(control.sent))
    endpoint, message = control.sent[0]
    decoded = _decode_control_message(message)
    assert endpoint == "tcp://source:1"
    assert decoded["op_class"] == "prefetch"
    assert decoded["allow_layout_subset"] is False
    assert 4000 < decoded["remaining_timeout_ms"] <= 5000
    agent.notifs["source"] = [_write_done_notification(handle)]
    _poll_until(kvcr, lambda _: not _has_outstanding_operations(kvcr))
    assert kvcr.query((key,)) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]
    assert any(r["event"] == "remote_prefetch_completed" for r in _key_history(caplog))
```

In `test_kvcr_remote_source.py`, find the existing test that drives `_handle_start_write` through `source_control.incoming` and add a parametrized variant: with `op_class="prefetch"` in the message and `remaining_timeout_ms=4000`, the created `_SourcePinOp.deadline - received_at` is about `4.0` when the source has `RemoteFWDramOptions(prefetch_timeout_ms=5000)` and about `1.0` without `op_class` (the `operation_timeout_ms` default). Use `source._core._clock = lambda: 0.0` so deadlines are exact.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_kvcr_remote_target.py -k prefetch_pull tests/unit/test_kvcr_remote_source.py -k op_class -q`
Expected: FAIL (no attribute `prefetch`; KeyError `op_class`).

- [ ] **Step 3: Implement**

Per the Interfaces block. Keep `deliver` and `fetch` call paths byte-for-byte equivalent in behavior (the existing `test_kvcr_deliver_fails_closed_*` and `test_remote_framework_dram_transfers_available_keys` tests pin this).

- [ ] **Step 4: Run the remote test files**

Run: `.venv/bin/python -m pytest tests/unit/test_kvcr_remote_target.py tests/unit/test_kvcr_remote_source.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/kvcr/remote_fw_dram.py tests/unit/test_kvcr_remote_target.py tests/unit/test_kvcr_remote_source.py
git commit -m "feat(remote): explicit-source prefetch pulls with op_class timeout"
```

---

### Task 5: Handler registry, deferred admission, budgets, telemetry

**Files:**
- Create: `src/kvcr/hint_actions.py`
- Modify: `src/kvcr/core.py` (`submit_hint` ~line 422, `poll_completed` ~line 657, `_complete_local_dram_fill` ~line 1059, `close`, constructor)
- Modify: `src/kvcr/api.py` (export metric names), `src/kvcr/__init__.py`
- Test: `tests/unit/test_hint_actions.py` (new), `tests/unit/test_kvcr_remote_target.py`

**Interfaces:**
- Consumes: `parse_hint_actions`, `_KVFetchAction` (Task 2); `reserve_fill(allow_evict=...)` (Task 3); `_RemoteFWDram.prefetch` (Task 4); options and `_prefetch_layout` (Task 1).
- Produces in `hint_actions.py`:
  - `HINT_ACTIONS_METRIC = "kvcr_hint_actions"`, `PREFETCH_BLOCKS_METRIC = "kvcr_prefetch_blocks"`, `PREFETCH_QUEUED_METRIC = "kvcr_prefetch_queued_blocks"`, `PREFETCH_RESERVED_METRIC = "kvcr_prefetch_reserved_blocks"`.
  - `class _ActionHandler(Protocol)`: `action_type: str`; `def submit(self, actions: Sequence[_HintAction], request_id: str | None) -> None`; `def admit(self, budget: int, now: float) -> int` (returns keys consumed from the budget; called under `_state_lock`); `def close(self) -> None`.
  - `@dataclass class _QueuedPrefetch`: `source: str`, `hashes: deque[int]`, `submitted_at: float`.
  - `class _FetchActionHandler(_ActionHandler)` constructed with `(core: _KVCRCore)`; owns `_queue: deque[_QueuedPrefetch]`, `_queue_lock: threading.Lock`, `_queued_blocks: int`, `_reserved: Counter[str]` (pool → keys in prefetch FILLING), `_reserve_limits: dict[str, int]` (`floor(slots * percent / 100)` per pool), `_prefetch_fill_keys: set[BlockKey]`.
    - `submit`: group `FW_G1` actions by `source_endpoint`, union hashes, call `core._remote_fw_dram.submit_hint(src, union, request_id)` once per source (in first-seen order). For `LOCAL_G2` actions: reject with a once-logged warning and `HINT_ACTIONS_METRIC (kv.fetch, CPU, rejected)` when `core._local_dram is None`, `core._prefetch_layout is None`, or the adapter lacks `keys_for_block_hash`; else under `_queue_lock` drop when `_queued_blocks + len(hashes) > max_queued_prefetch_blocks` (`PREFETCH_BLOCKS_METRIC (dropped_queue_full,) += len(hashes)`), else append and `_queued_blocks += len(hashes)`; then `core._progress.submit(_TargetMetadataRequest(src))` when `eager_ctrl_connect`. `G3` actions: rejected, counted. Count `HINT_ACTIONS_METRIC (kv.fetch, <GPU|CPU|STORAGE>, accepted|rejected)` per action.
    - `admit(budget, now)`: loop while budget > 0 and queue non-empty: head = queue[0]; if `now - head.submitted_at > prefetch_timeout_ms / 1000`, pop it, `_queued_blocks -= len(hashes)`, count `dropped_stale`, continue. Take up to `budget` hashes from `head.hashes` (popleft), decrement `_queued_blocks` by that many, pop the entry when empty. For each hash: `keys = adapter.keys_for_block_hash(hash)`; for each key: if `adapter.decode(key) != hash` → `dropped_key_mismatch` (log once); elif `core._is_local_resident(key)` → `skipped_resident`; else collect. Enforce `_reserved[pool] + needed <= _reserve_limits[pool]` for every pool in the layout (pool counts from `Counter(label.partition(":")[0] for label in layout)`); keys over the limit → `dropped_reserve_limit`. Create records with `core._block_record(key)`; call `core._local_dram.reserve_fill(keys, sources={k: CacheTier.REMOTE_G2 for k in keys}, required_local=False, deadline=now + prefetch_timeout_ms/1000, layout=core._prefetch_layout, allow_evict=prefetch_may_evict)`; keys in `eviction_pending` → `dropped_eviction_pending`; keys in neither → `dropped_no_capacity`; prune records for all dropped keys with `core._prune_block_record`. For reserved keys: add to `_prefetch_fill_keys`, bump `_reserved` by layout pool counts, count `admitted`; chunk `destinations` into `prefetch_blocks_per_op` and for each chunk take `handle = core._next_fill_handle; core._next_fill_handle -= 1` and call `core._remote_fw_dram.prefetch(head.source, chunk, deadline, op_handle=handle)`; if it returns `False`, call `core._complete_local_dram_fill(chunk, success=False)`. Log `KVCR_EVENT prefetch_admitted source=%s blocks=%d` at DEBUG. Set gauges `PREFETCH_QUEUED_METRIC` and `PREFETCH_RESERVED_METRIC (pool,)` when telemetry is enabled. Return budget consumed (hashes taken).
    - `on_fill_settled(keys)`: for each key in `_prefetch_fill_keys`, remove it and decrement `_reserved` by its layout pool counts. Called by `core._complete_local_dram_fill` before delegating.
    - `close()`: clear the queue and counters.
- Produces in `core.py`: `self._action_handlers: dict[str, _ActionHandler] = {"kv.fetch": _FetchActionHandler(self)}`; `submit_hint` → `self.submit_actions(parse_hint_actions(hints), request_id)`; `def submit_actions(self, actions: Sequence[_HintAction], request_id: str | None = None) -> None` grouping by `type(action).ACTION_TYPE` and calling each handler's `submit`; `def _admit_deferred_actions(self) -> None` called inside `poll_completed`'s `with self._state_lock` block before backend `poll_main`s: `budget = prefetch_admit_blocks_per_poll; now = self._clock(); for handler in handlers: budget -= handler.admit(budget, now); if budget <= 0: break`; `close()` calls each handler's `close()`.
- Produces: `KVCR.submit_actions` is **not** added to the public API in this task (side-band entry stays internal).

- [ ] **Step 1: Write failing tests (new file `tests/unit/test_hint_actions.py`)**

Use `_new_kvcr` with `local_dram=LocalDramOptions([("", addr, 16 * N)])`, `remote_options=RemoteFWDramOptions(eager_ctrl_connect=False, prefetch_layout=("",), ...)`, telemetry on, and this adapter:

```python
class _HashKeyAdapter:
    def __init__(self, groups=1): self.groups = groups
    def encode(self, k): return BlockKey(k)
    def decode(self, key): return int.from_bytes(key[:8], "big")
    def keys_for_block_hash(self, h):
        return [BlockKey(h.to_bytes(8, "big") + g.to_bytes(4, "big")) for g in range(self.groups)]
```

Tests (one function each, names as given):

1. `test_gpu_only_envelope_takes_inline_path`: wrap `kvcr._core._state_lock` with a Mock-counting proxy (or patch `acquire`), submit `_router_hint("tcp://a:1", (1,))`; assert acquire count is 0 and `kvcr._core._action_handlers["kv.fetch"]._queued_blocks == 0`; `query` for a matching key with the request id returns FETCHABLE/REMOTE_G2.
2. `test_same_source_gpu_actions_union_hashes`: envelope with `A:{1}` then `A:{2}`, `opportunistic_query=False`; both hashes FETCHABLE (adapter decode returns the hash of the key passed).
3. `test_cpu_action_is_queued_and_admitted_on_poll`: envelope with GPU action from A (hash 1) and CPU action from B (hash 2); after submit, `control.sent == []` and request hint source is A; after one `poll_completed`, `control.sent[0]` goes to B with `op_class == "prefetch"`; complete with `_write_done_notification`; `query((key2,))` without request id → HIT LOCAL_G2; `discard_hint("req")` beforehand does not cancel it; `_reserved[""] == 0` after completion; one `InventoryEvent(LOCAL_G2, removed=False)` reached the sink.
4. `test_cpu_action_failure_does_not_poison_request`: as 3 but `_write_done_notification(success=False)`; `query((key1,), "req")` still FETCHABLE; prefetched key MISS; no inventory event.
5. `test_resident_and_filling_keys_are_skipped`: deposit key2 first; submit CPU action for hash 2 → poll → `control.sent == []`, counter `skipped_resident == 1`.
6. `test_queue_full_drops_at_submit`: `max_queued_prefetch_blocks=3`; submit CPU action with 4 hashes → nothing queued, `dropped_queue_full == 4`; submit one with 3 → queued.
7. `test_per_poll_budget_resumes_partial_action`: `prefetch_admit_blocks_per_poll=2`, action with 5 hashes, `prefetch_blocks_per_op=1`; after poll 1 `len(control.sent) == 2`, after poll 3 `== 5`, sources in hash order.
8. `test_stale_action_dropped_without_records`: `prefetch_timeout_ms=1000`; submit, then `kvcr._core._clock = lambda: t0 + 2`; poll → `control.sent == []`, `_block_record_map` has no new keys, `dropped_stale == n`.
9. `test_reserve_limit_leaves_headroom_for_demand_fetch`: pool of 4 slots, `prefetch_reserve_limit_percent=50`; CPU action with 4 hashes → only 2 pulls sent, `dropped_reserve_limit == 2`; then `fetch([other_key], "req")` with a GPU hint for it succeeds in reserving (a `start_write` for `other_key` is sent) while the 2 prefetches are still FILLING.
10. `test_prefetch_may_evict_false_drops_when_full`: 2-slot pool, deposit 2 READY blocks, `prefetch_may_evict=False` → `dropped_no_capacity == 1`, both deposits still READY; with `True` one is evicted and a pull is sent.
11. `test_key_mismatch_is_dropped`: adapter whose `decode` returns 0 → `dropped_key_mismatch`, no records, one warning.
12. `test_multi_group_hash_reserves_every_key`: `groups=2` → one pull with 2 keys; both HIT after completion.
13. `test_cpu_action_rejected_without_layout_or_adapter_method`: `prefetch_layout=None`, or adapter without `keys_for_block_hash` → nothing queued, `HINT_ACTIONS_METRIC (kv.fetch, CPU, rejected)`, exactly one warning across two submissions.
14. `test_stub_deferred_handler_receives_budget`: register a stub handler object in `kvcr._core._action_handlers["kv.test"]` whose `admit` records `(budget, now)` and returns 1; after `poll_completed`, recorded budget equals `prefetch_admit_blocks_per_poll` minus whatever the fetch handler consumed (0 here).
15. `test_close_with_queued_prefetch`: submit CPU action, call `kvcr.close()` without polling → no exception, queue empty.
16. `test_partial_prefetch_completion`: 1 action with 2 hashes, `_write_done_notification(handle, completed_indices=(0,))` → key0 HIT, key1 MISS, `_reserved[""] == 0`, one stored inventory event with only key0.

Also add to `test_kvcr_remote_target.py` the layout test: `pool_layouts=[("", 16)]` with `prefetch_layout=("", "")` and a 2-element local DRAM pool: a prefetch reserves both elements and the `dst_descriptors` in `start_write` has 2 entries.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_hint_actions.py -q`
Expected: FAIL (ImportError / attribute errors).

- [ ] **Step 3: Implement `hint_actions.py` and the core wiring**

Per the Interfaces block. Import `_TargetMetadataRequest` from `.remote_fw_dram` lazily inside the handler to avoid import cycles (core already imports `remote_fw_dram` lazily). Export the four metric names from `kvcr.api` and `kvcr.__init__`.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/test_hint_actions.py tests/unit/test_hint_parser.py tests/unit/test_kvcr_remote_target.py tests/unit/test_kvcr_remote_source.py tests/unit/test_kvcr_local_dram.py tests/unit/test_kvcr.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/kvcr/hint_actions.py src/kvcr/core.py src/kvcr/api.py src/kvcr/__init__.py tests/unit/test_hint_actions.py tests/unit/test_kvcr_remote_target.py
git commit -m "feat(hints): action handler registry with budgeted prefetch admission"
```

---

### Task 6: Documentation

**Files:**
- Modify: `docs/design_overview.md` ("Request-scoped hints" section, ~line 254)
- Modify: `docs/dev-guide.md` (vLLM configuration table, ~line 540; add the prefetch options with one line each)
- Modify: `README.md` ("KV Hint Protocol" section, ~line 67)

- [ ] **Step 1: Edit docs**

In `design_overview.md`, keep the first two paragraphs of "Request-scoped hints", replace the third paragraph's claim that `submit_hint` never starts data movement with: engine-destination hints behave as before; a `kv.fetch` with `destination.medium == "CPU"` is admitted from `poll_completed` and starts a prefetch into KVCR local DRAM, see `composable_hints_design.md`. Add the two integration contracts (idle polling; `poll_completed` may run on a latency-sensitive path). In `dev-guide.md` add rows for the seven options. In `README.md` add one paragraph on the `destination` payload field linking to the design doc.

- [ ] **Step 2: Verify**

Run: `.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests`
Expected: no findings. Open the three docs and confirm the links resolve (`ls docs/composable_hints_design.md`).

- [ ] **Step 3: Commit**

```bash
git add docs/design_overview.md docs/dev-guide.md README.md
git commit -m "docs: composable hints design, prefetch options, integration contract"
```
