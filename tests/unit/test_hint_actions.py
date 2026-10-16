# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Hint action handlers: inline engine hints and budgeted prefetch admission."""

import ctypes
import logging
from collections import Counter
from types import SimpleNamespace

import msgspec
import pytest
from _kvcr_test_utils import (
    FakeBytesControl,
    FakeNixlAgent,
    FakePrimaryPinning,
    _buffer_region,
    _decode_control_message,
    _has_outstanding_operations,
    _HintAction,
    _HintEnvelope,
    _mem_descriptor,
    _new_kvcr,
    _poll_until,
    _router_hint,
    _wait_until,
    _write_done_notification,
)

from kvcr import (
    HINT_ACTIONS_METRIC,
    PREFETCH_BLOCKS_METRIC,
    PREFETCH_QUEUED_METRIC,
    PREFETCH_RESERVED_METRIC,
)
from kvcr import hint_actions as hint_actions_module
from kvcr.config import KVCRConfig, LocalDramOptions, RemoteFWDramOptions
from kvcr.types import BlockKey, CacheTier, InventoryEvent, QueryStatus

SOURCE_A = "tcp://a:1"
SOURCE_B = "tcp://b:1"
BLOCK_SIZE = 16


class _HashKeyAdapter:
    def __init__(self, groups=1):
        self.groups = groups

    def encode(self, k):
        return BlockKey(k)

    def decode(self, key):
        return int.from_bytes(key[:8], "big")

    def keys_for_block_hash(self, h):
        return [
            BlockKey(h.to_bytes(8, "big") + g.to_bytes(4, "big"))
            for g in range(self.groups)
        ]


class _MismatchAdapter(_HashKeyAdapter):
    def decode(self, key):
        return 0


class _NoReverseAdapter:
    def decode(self, key):
        return int.from_bytes(key[:8], "big")


class _CountingLock:
    def __init__(self, lock):
        self.lock = lock
        self.count = 0

    def __enter__(self):
        self.count += 1
        return self.lock.__enter__()

    def __exit__(self, *args):
        return self.lock.__exit__(*args)

    def acquire(self, *args, **kwargs):
        self.count += 1
        return self.lock.acquire(*args, **kwargs)

    def release(self):
        return self.lock.release()


def _key(block_hash: int, group: int = 0) -> BlockKey:
    return BlockKey(block_hash.to_bytes(8, "big") + group.to_bytes(4, "big"))


def _fetch_action(source, hashes, medium=None, action_id="a"):
    payload = {"source_control_endpoint": source, "block_hashes": list(hashes)}
    if medium is not None:
        payload["destination"] = {"medium": medium}
    return _HintAction(action_id, "kv.fetch", "1.0", payload)


def _envelope(*actions):
    return _HintEnvelope("0.1", "m", list(actions))


def _cpu_hint(source, hashes):
    return _envelope(_fetch_action(source, hashes, "CPU"))


def _setup(
    *,
    slots=8,
    adapter=None,
    local=True,
    framework_slots=0,
    sink=None,
    config=None,
    on_resilience_event=None,
    **options,
):
    local_buffer = ctypes.create_string_buffer(BLOCK_SIZE * slots)
    primary = (
        ctypes.create_string_buffer(BLOCK_SIZE * framework_slots)
        if framework_slots
        else None
    )
    agent = FakeNixlAgent(metadata=b"target-md")
    control = FakeBytesControl("tcp://target:1")
    remote = {
        "eager_ctrl_connect": False,
        "prefetch_layout": ("",),
        "prefetch_reserve_limit_percent": 100,
        **options,
    }
    kvcr = _new_kvcr(
        agent,
        FakePrimaryPinning(),
        control,
        KVCRConfig(
            nixl_agent_name="target",
            pool_layouts=[("", BLOCK_SIZE)],
            enable_telemetry=True,
            **(config or {}),
        ),
        key_adapter=_HashKeyAdapter() if adapter is None else adapter,
        remote_options=RemoteFWDramOptions(**remote),
        local_dram=(
            LocalDramOptions([("", ctypes.addressof(local_buffer), len(local_buffer))])
            if local
            else None
        ),
        framework_regions=(
            [_buffer_region(primary, BLOCK_SIZE)] if primary is not None else None
        ),
        inventory_sink=sink,
        on_resilience_event=on_resilience_event,
    )
    return SimpleNamespace(
        kvcr=kvcr,
        agent=agent,
        control=control,
        local=local_buffer,
        primary=primary,
        handler=kvcr._core._action_handlers["kv.fetch"],
    )


def _start_writes(control, count):
    def writes():
        return [
            (endpoint, decoded)
            for endpoint, message in control.sent
            if (decoded := _decode_control_message(message))["type"] == "start_write"
        ]

    _wait_until(lambda: len(writes()) >= count)
    return writes()


def _complete(env, decoded, **kwargs):
    env.agent.notifs["source"] = [
        _write_done_notification(decoded["op_handle"], **kwargs)
    ]
    _poll_until(env.kvcr, lambda _: not _has_outstanding_operations(env.kvcr))


def _counters(kvcr, name):
    totals: Counter = Counter()
    for kind, metric, value, labels in kvcr.get_stats().records:
        if kind == "counter" and metric == name:
            totals[labels] += value
    return totals


def _deposit_ready(env, keys):
    env.agent.state = "PROC"
    transfers = len(env.agent.transfers)
    env.kvcr.deposit({key: [_mem_descriptor(index)] for index, key in enumerate(keys)})
    _wait_until(lambda: len(env.agent.transfers) > transfers)
    env.agent.state = "DONE"
    _poll_until(env.kvcr, lambda results: bool(results))
    env.agent.state = "PROC"


def _probe_ack(handle, source, **fields):
    return msgspec.msgpack.encode(
        {
            "type": "write_probe_ack",
            "sender_control_endpoint": source,
            "sender_incarnation": "source",
            "op_handle": handle,
            "terminal": False,
            **fields,
        }
    )


def _is_filling(env, key):
    record = env.kvcr._core._block_record_map.get(key)
    return (
        record is not None
        and record.local_dram is not None
        and record.local_dram.state.name == "FILLING"
    )


@pytest.fixture(autouse=True)
def _fresh_warning_set(monkeypatch):
    monkeypatch.setattr(hint_actions_module, "_LOGGED_ACTION_ISSUES", set())


def test_metric_names_are_exported():
    assert HINT_ACTIONS_METRIC == "kvcr_hint_actions"
    assert PREFETCH_BLOCKS_METRIC == "kvcr_prefetch_blocks"
    assert PREFETCH_QUEUED_METRIC == "kvcr_prefetch_queued_blocks"
    assert PREFETCH_RESERVED_METRIC == "kvcr_prefetch_reserved_blocks"


def test_gpu_only_envelope_takes_inline_path():
    env = _setup()
    core = env.kvcr._core
    original = core._state_lock
    counting = core._state_lock = _CountingLock(original)
    try:
        env.kvcr.submit_hint(_router_hint(SOURCE_A, (1,)), "req")
    finally:
        core._state_lock = original
    assert counting.count == 0
    assert env.handler._queued_blocks == 0
    assert env.kvcr.query((_key(1),), "req") == [
        (QueryStatus.FETCHABLE, CacheTier.REMOTE_G2)
    ]
    assert _counters(env.kvcr, HINT_ACTIONS_METRIC) == {
        ("kv.fetch", "GPU", "accepted"): 1
    }


def test_same_source_gpu_actions_union_hashes():
    env = _setup(opportunistic_query=False)
    env.kvcr.submit_hint(
        _envelope(
            _fetch_action(SOURCE_A, (1,), action_id="a"),
            _fetch_action(SOURCE_A, (2,), action_id="b"),
        ),
        "req",
    )
    assert env.kvcr.query((_key(1), _key(2)), "req") == [
        (QueryStatus.FETCHABLE, CacheTier.REMOTE_G2),
        (QueryStatus.FETCHABLE, CacheTier.REMOTE_G2),
    ]


def test_first_gpu_source_wins_and_other_sources_are_rejected(caplog):
    env = _setup(opportunistic_query=False)
    with caplog.at_level(logging.WARNING, logger="kvcr.hint_actions"):
        env.kvcr.submit_hint(
            _envelope(
                _fetch_action(SOURCE_A, (1,), action_id="a"),
                _fetch_action(SOURCE_B, (2,), action_id="b"),
                _fetch_action(SOURCE_B, (3,), action_id="c"),
            ),
            "req",
        )
    hint = env.kvcr._core._remote_fw_dram._request_hints["req"]
    assert hint.source == SOURCE_A
    assert not hint.failed
    assert env.kvcr.query((_key(1),), "req") == [
        (QueryStatus.FETCHABLE, CacheTier.REMOTE_G2)
    ]
    assert env.kvcr.query((_key(2),), "req") == [(QueryStatus.MISS, None)]
    assert _counters(env.kvcr, HINT_ACTIONS_METRIC) == {
        ("kv.fetch", "GPU", "accepted"): 1,
        ("kv.fetch", "GPU", "rejected"): 2,
    }
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert [r.getMessage() for r in warnings] == [
        "Rejected kv.fetch action: multiple engine-destination sources in one envelope"
    ]


def test_cpu_action_is_queued_and_admitted_on_poll():
    events: list[InventoryEvent] = []
    env = _setup(sink=events.append)
    env.kvcr.submit_hint(
        _envelope(
            _fetch_action(SOURCE_A, (1,), action_id="gpu"),
            _fetch_action(SOURCE_B, (2,), "CPU", action_id="cpu"),
        ),
        "req",
    )
    assert env.control.sent == []
    assert env.kvcr._core._remote_fw_dram._request_hints["req"].source == SOURCE_A
    assert env.handler._queued_blocks == 1
    env.kvcr.discard_hint("req")

    list(env.kvcr.poll_completed())
    [(endpoint, decoded)] = _start_writes(env.control, 1)
    assert endpoint == SOURCE_B
    assert decoded["op_class"] == "prefetch"
    assert decoded["keys"] == [_key(2)]
    assert env.kvcr.query((_key(2),)) == [(QueryStatus.FETCHING, CacheTier.LOCAL_G2)]
    assert env.handler._reserved[""] == 1

    _complete(env, decoded)
    assert env.kvcr.query((_key(2),)) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]
    assert env.handler._reserved[""] == 0
    assert not env.handler._prefetch_fill_keys
    assert events == [InventoryEvent((_key(2),), CacheTier.LOCAL_G2, False)]

    # A later engine fetch is served locally without another pull.
    fetch = env.kvcr.fetch((_key(2),))
    results = dict(_poll_until(env.kvcr, bool))
    assert results[fetch][_key(2)].success
    assert len(env.control.sent) == 1

    assert _counters(env.kvcr, HINT_ACTIONS_METRIC) == {
        ("kv.fetch", "GPU", "accepted"): 1,
        ("kv.fetch", "CPU", "accepted"): 1,
    }


def test_cpu_action_failure_does_not_poison_request():
    events: list[InventoryEvent] = []
    env = _setup(sink=events.append)
    env.kvcr.submit_hint(
        _envelope(
            _fetch_action(SOURCE_A, (1,), action_id="gpu"),
            _fetch_action(SOURCE_B, (2,), "CPU", action_id="cpu"),
        ),
        "req",
    )
    list(env.kvcr.poll_completed())
    [(_, decoded)] = _start_writes(env.control, 1)
    _complete(env, decoded, success=False)

    assert env.kvcr.query((_key(1),), "req") == [
        (QueryStatus.FETCHABLE, CacheTier.REMOTE_G2)
    ]
    assert env.kvcr.query((_key(2),)) == [(QueryStatus.MISS, None)]
    assert _key(2) not in env.kvcr._core._block_record_map
    assert env.handler._reserved[""] == 0
    assert events == []


def test_resident_and_filling_keys_are_skipped():
    env = _setup(framework_slots=2)
    _deposit_ready(env, (_key(2),))
    # A second deposit left FILLING.
    env.kvcr.deposit({_key(3): [_mem_descriptor(1)]})
    _wait_until(lambda: len(env.agent.transfers) == 2)

    reserved = Counter(env.handler._reserved)
    env.kvcr.submit_hint(_cpu_hint(SOURCE_B, (2, 3)))
    list(env.kvcr.poll_completed())
    assert env.control.sent == []
    assert env.handler._queued_blocks == 0
    assert _counters(env.kvcr, PREFETCH_BLOCKS_METRIC) == {("skipped_resident",): 2}
    # Skipped keys take no prefetch reservation.
    assert env.handler._reserved == reserved
    assert env.handler._reserved[""] == 0
    assert not env.handler._prefetch_fill_keys
    env.agent.state = "DONE"
    _poll_until(env.kvcr, bool)


def test_queue_full_drops_at_submit():
    env = _setup(max_queued_prefetch_blocks=3)
    env.kvcr.submit_hint(_cpu_hint(SOURCE_B, (1, 2, 3, 4)))
    assert env.handler._queued_blocks == 0
    assert not env.handler._queue
    env.kvcr.submit_hint(_cpu_hint(SOURCE_B, (5, 6, 7)))
    assert env.handler._queued_blocks == 3
    assert len(env.handler._queue) == 1
    assert _counters(env.kvcr, PREFETCH_BLOCKS_METRIC) == {("dropped_queue_full",): 4}


def test_per_poll_budget_resumes_partial_action():
    # Large, non-monotone hashes: admission must follow the router's order.
    hashes = (
        0xF000_0000_0000_0005,
        0x1,
        0xA000_0000_0000_0003,
        0x7FFF_FFFF_FFFF_FFFE,
        0x2,
    )
    env = _setup(prefetch_admit_blocks_per_poll=2, prefetch_blocks_per_op=1)
    env.kvcr.submit_hint(_cpu_hint(SOURCE_B, hashes))
    expected_counts = (2, 4, 5)
    for count in expected_counts:
        list(env.kvcr.poll_completed())
        writes = _start_writes(env.control, count)
        assert len(writes) == count
        assert env.handler._queued_blocks == 5 - count
        assert [decoded["keys"][0] for _, decoded in writes] == [
            _key(h) for h in hashes[:count]
        ]
    assert [endpoint for endpoint, _ in writes] == [SOURCE_B] * 5
    env.agent.notifs["source"] = [
        _write_done_notification(decoded["op_handle"]) for _, decoded in writes
    ]
    _poll_until(env.kvcr, lambda _: not _has_outstanding_operations(env.kvcr))
    assert env.handler._reserved[""] == 0


def test_stale_action_dropped_without_records():
    env = _setup(prefetch_timeout_ms=1000)
    core = env.kvcr._core
    t0 = core._clock()
    env.kvcr.submit_hint(_cpu_hint(SOURCE_B, (1, 2, 3)))
    core._clock = lambda: t0 + 2
    list(env.kvcr.poll_completed())
    assert env.control.sent == []
    assert core._block_record_map == {}
    assert env.handler._queued_blocks == 0
    assert _counters(env.kvcr, PREFETCH_BLOCKS_METRIC) == {("dropped_stale",): 3}


def test_reserve_limit_leaves_headroom_for_demand_fetch():
    env = _setup(slots=4, prefetch_reserve_limit_percent=50, prefetch_blocks_per_op=1)
    hashes = (0xA000_0000_0000_0003, 0x3, 0xF000_0000_0000_0001, 0x1)
    env.kvcr.submit_hint(_cpu_hint(SOURCE_B, hashes))
    list(env.kvcr.poll_completed())
    prefetches = _start_writes(env.control, 2)
    assert len(prefetches) == 2
    assert env.handler._reserved[""] == 2
    # The prefix chain's head is admitted; the tail over the limit is dropped.
    assert [d["keys"][0] for _, d in prefetches] == [_key(h) for h in hashes[:2]]
    assert all(_key(h) not in env.kvcr._core._block_record_map for h in hashes[2:])

    env.kvcr.submit_hint(_router_hint(SOURCE_A, (9,)), "req")
    fetch = env.kvcr.fetch((_key(9),), "req")
    writes = _start_writes(env.control, 3)
    endpoint, demand = writes[2]
    assert endpoint == SOURCE_A
    assert demand["keys"] == [_key(9)]

    env.agent.notifs["source"] = [
        _write_done_notification(decoded["op_handle"]) for _, decoded in writes
    ]
    results = dict(
        _poll_until(
            env.kvcr,
            lambda done: (
                any(handle == fetch for handle, _ in done)
                and not _has_outstanding_operations(env.kvcr)
            ),
        )
    )
    assert results[fetch][_key(9)].success
    assert env.handler._reserved[""] == 0
    counters = _counters(env.kvcr, PREFETCH_BLOCKS_METRIC)
    assert counters[("dropped_reserve_limit",)] == 2
    assert counters[("admitted",)] == 2


@pytest.mark.parametrize("may_evict", [False, True])
def test_prefetch_may_evict_false_drops_when_full(may_evict):
    env = _setup(slots=2, framework_slots=2, prefetch_may_evict=may_evict)
    resident = (_key(100), _key(101))
    _deposit_ready(env, resident)

    env.kvcr.submit_hint(_cpu_hint(SOURCE_B, (5,)))
    list(env.kvcr.poll_completed())
    statuses = env.kvcr.query(resident)
    if not may_evict:
        assert env.control.sent == []
        assert statuses == [(QueryStatus.HIT, CacheTier.LOCAL_G2)] * 2
        assert _key(5) not in env.kvcr._core._block_record_map
        assert _counters(env.kvcr, PREFETCH_BLOCKS_METRIC) == {
            ("dropped_no_capacity",): 1
        }
    else:
        [(endpoint, decoded)] = _start_writes(env.control, 1)
        assert decoded["keys"] == [_key(5)]
        assert sorted(statuses, key=str) == sorted(
            [(QueryStatus.HIT, CacheTier.LOCAL_G2), (QueryStatus.MISS, None)],
            key=str,
        )
        _complete(env, decoded)
        assert env.kvcr.query((_key(5),)) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]


def test_key_mismatch_is_dropped(caplog):
    env = _setup(adapter=_MismatchAdapter())
    with caplog.at_level(logging.WARNING, logger="kvcr.hint_actions"):
        env.kvcr.submit_hint(_cpu_hint(SOURCE_B, (1, 2)))
        list(env.kvcr.poll_completed())
    assert env.control.sent == []
    assert env.kvcr._core._block_record_map == {}
    assert _counters(env.kvcr, PREFETCH_BLOCKS_METRIC) == {("dropped_key_mismatch",): 2}
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1


def test_multi_group_hash_reserves_every_key():
    env = _setup(adapter=_HashKeyAdapter(groups=2))
    env.kvcr.submit_hint(_cpu_hint(SOURCE_B, (1,)))
    list(env.kvcr.poll_completed())
    [(_, decoded)] = _start_writes(env.control, 1)
    assert decoded["keys"] == [_key(1, 0), _key(1, 1)]
    _complete(env, decoded, completed_indices=(0, 1))
    assert (
        env.kvcr.query((_key(1, 0), _key(1, 1)))
        == [(QueryStatus.HIT, CacheTier.LOCAL_G2)] * 2
    )
    assert env.handler._reserved[""] == 0


@pytest.mark.parametrize(
    "setup",
    [
        {"prefetch_layout": None},
        {"adapter": _NoReverseAdapter()},
        {"local": False},
    ],
    ids=["no-layout", "no-reverse-lookup", "no-local-dram"],
)
def test_cpu_action_rejected_without_layout_or_adapter_method(setup, caplog):
    env = _setup(**setup)
    with caplog.at_level(logging.WARNING, logger="kvcr.hint_actions"):
        env.kvcr.submit_hint(_cpu_hint(SOURCE_B, (1,)))
        env.kvcr.submit_hint(_cpu_hint(SOURCE_B, (2,)))
        list(env.kvcr.poll_completed())
    assert env.handler._queued_blocks == 0
    assert not env.handler._queue
    assert env.control.sent == []
    assert _counters(env.kvcr, HINT_ACTIONS_METRIC) == {
        ("kv.fetch", "CPU", "rejected"): 2
    }
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1


def test_stub_deferred_handler_receives_budget():
    env = _setup(prefetch_admit_blocks_per_poll=7)

    class _Stub:
        action_type = "kv.test"

        def __init__(self):
            self.admits = []
            self.closed = False

        def submit(self, actions, request_id):
            pass

        def admit(self, budget, now):
            self.admits.append((budget, now))
            return 1

        def close(self):
            self.closed = True

    stub = _Stub()
    env.kvcr._core._action_handlers["kv.test"] = stub
    core = env.kvcr._core
    core._clock = lambda: 42.0
    list(env.kvcr.poll_completed())
    assert stub.admits == [(7, 42.0)]
    env.kvcr.close()
    assert stub.closed


def test_close_with_queued_prefetch():
    env = _setup()
    env.kvcr.submit_hint(_cpu_hint(SOURCE_B, (1, 2)))
    assert env.handler._queued_blocks == 2
    env.kvcr.close()
    assert not env.handler._queue
    assert env.handler._queued_blocks == 0
    assert env.control.sent == []


def test_partial_prefetch_completion():
    events: list[InventoryEvent] = []
    env = _setup(sink=events.append)
    env.kvcr.submit_hint(_cpu_hint(SOURCE_B, (1, 2)))
    list(env.kvcr.poll_completed())
    [(_, decoded)] = _start_writes(env.control, 1)
    first, second = map(BlockKey, decoded["keys"])
    assert {first, second} == {_key(1), _key(2)}
    _complete(env, decoded, completed_indices=(0,))
    assert env.kvcr.query((first, second)) == [
        (QueryStatus.HIT, CacheTier.LOCAL_G2),
        (QueryStatus.MISS, None),
    ]
    assert env.handler._reserved[""] == 0
    assert not env.handler._prefetch_fill_keys
    assert events == [InventoryEvent((first,), CacheTier.LOCAL_G2, False)]


def test_prefetch_gauges_track_queue_and_reservations():
    env = _setup()
    env.kvcr.submit_hint(_cpu_hint(SOURCE_B, (1,)))
    list(env.kvcr.poll_completed())
    [(_, decoded)] = _start_writes(env.control, 1)
    gauges = [
        (name, value, labels)
        for kind, name, value, labels in env.kvcr.get_stats().records
        if kind == "gauge" and name != "kvcr_state"
    ]
    assert (PREFETCH_QUEUED_METRIC, 0, ()) in gauges
    assert (PREFETCH_RESERVED_METRIC, 1, ("",)) in gauges
    _complete(env, decoded)


class _RaisingAdapter(_HashKeyAdapter):
    def keys_for_block_hash(self, h):
        raise RuntimeError("adapter bug")


def test_adapter_failure_is_isolated_from_poll(caplog):
    env = _setup(adapter=_RaisingAdapter())
    with caplog.at_level(logging.WARNING, logger="kvcr.hint_actions"):
        env.kvcr.submit_hint(_cpu_hint(SOURCE_B, (1, 2)))
        assert list(env.kvcr.poll_completed()) == []
    assert env.control.sent == []
    assert env.kvcr._core._block_record_map == {}
    assert env.handler._reserved[""] == 0
    assert env.handler._queued_blocks == 0
    assert _counters(env.kvcr, PREFETCH_BLOCKS_METRIC) == {("dropped_key_mismatch",): 2}
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert warnings[0].exc_info is not None


def test_prefetch_timeout_releases_reservation_once_terminal():
    events: list[InventoryEvent] = []
    errors = []
    now = 0.0
    env = _setup(
        sink=events.append,
        prefetch_timeout_ms=1000,
        config={"operation_timeout_ms": 1000, "abandon_timeout_ms": 2000},
        on_resilience_event=errors.append,
    )
    core = env.kvcr._core
    core._clock = lambda: now
    env.kvcr.submit_hint(_cpu_hint(SOURCE_B, (1,)))
    list(env.kvcr.poll_completed())
    [(_, decoded)] = _start_writes(env.control, 1)
    handle = decoded["op_handle"]
    env.control.incoming.append(
        msgspec.msgpack.encode(
            {
                "type": "target_metadata_ack",
                "sender_control_endpoint": SOURCE_B,
                "sender_incarnation": "source",
                "op_handle": handle,
            }
        )
    )
    operation = core._progress._in_flight_ops[("target", handle)]
    _wait_until(lambda: operation.source_incarnation == "source")
    assert env.handler._reserved[""] == 1

    # Past prefetch_timeout_ms: the fill is discarded but keeps its slot.
    now = 1.5
    _poll_until(env.kvcr, lambda _: not _is_filling(env, _key(1)))
    assert env.kvcr.query((_key(1),)) == [(QueryStatus.MISS, None)]
    assert env.handler._reserved[""] == 1

    # Past abandon_timeout_ms: quarantined until the source proves quiescence.
    now = 2.5
    _poll_until(env.kvcr, lambda _: bool(errors))
    assert [error.state for error in errors] == ["uncertain"]
    assert env.handler._reserved[""] == 1

    env.control.incoming.append(_probe_ack(handle, SOURCE_B, terminal=True))
    _poll_until(env.kvcr, lambda _: not _has_outstanding_operations(env.kvcr))
    assert [error.state for error in errors] == ["uncertain", "quiesced"]
    assert env.handler._reserved[""] == 0
    assert not env.handler._prefetch_fill_keys
    assert env.kvcr.query((_key(1),)) == [(QueryStatus.MISS, None)]
    assert _key(1) not in core._block_record_map
    assert events == []


def test_engine_fetch_waits_on_in_flight_prefetch_and_fails_with_it():
    env = _setup(opportunistic_query=False)
    env.kvcr.submit_hint(_cpu_hint(SOURCE_B, (2,)))
    list(env.kvcr.poll_completed())
    [(_, prefetch)] = _start_writes(env.control, 1)

    env.kvcr.submit_hint(_router_hint(SOURCE_A, (2, 3)), "req")
    assert env.kvcr.query((_key(2),), "req") == [
        (QueryStatus.FETCHING, CacheTier.LOCAL_G2)
    ]
    fetch = env.kvcr.fetch((_key(2),), "req")
    env.agent.notifs["source"] = [
        _write_done_notification(prefetch["op_handle"], success=False)
    ]
    results = dict(_poll_until(env.kvcr, bool))
    assert not results[fetch][_key(2)].success
    # Only the prefetch pull was sent: the engine fetch joined its fill.
    assert len(env.control.sent) == 1
    # A failed prefetch never poisons the request's hint.
    assert not env.kvcr._core._remote_fw_dram._request_hints["req"].failed
    assert env.kvcr.query((_key(3),), "req") == [
        (QueryStatus.FETCHABLE, CacheTier.REMOTE_G2)
    ]
    assert env.handler._reserved[""] == 0
