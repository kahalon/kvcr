# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU-destination ``kv.fetch`` (prefetch into KVCR local DRAM), end to end.

Two KVCRs in one process: a source holding blocks in its local DRAM and a
target that receives hint envelopes. Control messages and write
notifications are shuttled by hand, so every step is observable.
"""

import ctypes
import logging
import time

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
    _key_history,
    _mem_descriptor,
    _new_kvcr,
    _op_entries,
    _poll_until,
    _record_hint_results,
    _wait_until,
)
from test_kvcr_remote_target import _CopyingWriteAgent

from kvcr.config import KVCRConfig, LocalDramOptions, RemoteFWDramOptions
from kvcr.hints.dispatcher import _LOGGED_HANDLER_ERRORS
from kvcr.hints.parser import _LOGGED_HINT_ISSUES
from kvcr.hints.results import HintResult, HintStatus
from kvcr.policy import FIFOPolicy
from kvcr.types import (
    BlockKey,
    CacheTier,
    InventoryEvent,
    MemoryRef,
    PlacementAction,
    QueryStatus,
)

BLOCK = 16
SOURCE = "tcp://source:1"
_HIT = (QueryStatus.HIT, CacheTier.LOCAL_G2)
_MISS = (QueryStatus.MISS, None)


@pytest.fixture(autouse=True)
def _reset_once_logs():
    _LOGGED_HANDLER_ERRORS.clear()
    _LOGGED_HINT_ISSUES.clear()


class _HashKeyAdapter:
    """The router hash is the first 8 bytes of the key, as vLLM's int mode is."""

    def decode(self, key):
        return int.from_bytes(key[:8], "big")


def _key(block_hash: int, group: int = 0) -> BlockKey:
    return BlockKey(block_hash.to_bytes(8, "big") + group.to_bytes(4, "big"))


def _content(block_hash: int) -> bytes:
    return bytes([96 + block_hash]) * BLOCK  # hash 1 -> b"a" * 16, 2 -> b"b", ...


def _fetch(action_id, hashes, **extra) -> _HintAction:
    payload = {"source_control_endpoint": SOURCE, "block_hashes": list(hashes)}
    payload.update(extra)
    return _HintAction(action_id, "kv.fetch", "1.0", payload)


def _prefetch(action_id, hashes) -> _HintAction:
    return _fetch(action_id, hashes, destination={"medium": "CPU"})


def _envelope(*actions, message_id="m1") -> _HintEnvelope:
    return _HintEnvelope("0.1", message_id, list(actions))


def _take(results):
    taken = [(r.action_id, r.status, r.reason) for r in results]
    results.clear()
    return taken


class _Pair:
    """A source KVCR and a target KVCR with hand-driven transport between them."""

    def __init__(
        self,
        monkeypatch,
        *,
        source_slots: int = 4,
        target_slots: int = 4,
        target_policy=None,
        target_options: RemoteFWDramOptions | None = None,
        target_config: dict | None = None,
        clock=None,
        parts=False,
    ):
        self.source_primary = ctypes.create_string_buffer(
            b"".join(_content(i + 1) for i in range(source_slots))
        )
        self.source_local = ctypes.create_string_buffer(BLOCK * source_slots)
        self.source_agent = _CopyingWriteAgent(metadata=b"source-md")
        self.source_control = FakeBytesControl(SOURCE)
        self.source = _new_kvcr(
            self.source_agent,
            FakePrimaryPinning(),
            self.source_control,
            KVCRConfig(nixl_agent_name="source", pool_layouts=[("", BLOCK)]),
            name="source",
            key_adapter=_HashKeyAdapter(),
            local_dram=LocalDramOptions(
                [("", ctypes.addressof(self.source_local), len(self.source_local))]
            ),
            framework_regions=[
                _buffer_region(self.source_primary, BLOCK, ":*" if parts else "")
            ],
        )
        self.source_agent.state = "DONE"

        self.target_primary = ctypes.create_string_buffer(BLOCK * 2)
        self.target_local = ctypes.create_string_buffer(BLOCK * target_slots)
        self.target_agent = _CopyingWriteAgent(metadata=b"target-md")
        self.target_control = FakeBytesControl("tcp://target:1")
        self.events: list[InventoryEvent] = []
        self.target = _new_kvcr(
            self.target_agent,
            FakePrimaryPinning(),
            self.target_control,
            KVCRConfig(
                nixl_agent_name="target",
                pool_layouts=[("", BLOCK)],
                **(target_config or {}),
            ),
            name="target",
            key_adapter=_HashKeyAdapter(),
            remote_options=target_options
            or RemoteFWDramOptions(eager_ctrl_connect=False),
            local_dram=LocalDramOptions(
                [("", ctypes.addressof(self.target_local), len(self.target_local))]
            ),
            framework_regions=[
                _buffer_region(self.target_primary, BLOCK, ":*" if parts else "")
            ],
            inventory_sink=self.events.append,
            policy=target_policy,
        )
        if clock is not None:
            self.target._core._clock = clock
        self.results = _record_hint_results(monkeypatch, self.target)
        self.to_source: list[dict] = []  # every control message the target sent
        self.target_completed: list = []
        self._notified = 0
        self._sent_notifs = 0

    def seed(self, *hashes: int) -> None:
        """Make blocks READY in the source's local DRAM, by hash."""
        for block_hash in hashes:
            key = _key(block_hash)
            deposit = self.source.deposit(
                {key: [_mem_descriptor(block_hash - 1, end_point_name="source")]}
            )
            assert _poll_until(self.source, bool) == [
                (deposit, _op_entries({key: True}))
            ]

    def shuttle(self) -> None:
        """Move control messages both ways and deliver write notifications."""
        for _, raw in self.target_control.sent:
            self.to_source.append(_decode_control_message(raw))
            self.source_control.incoming.append(raw)
        self.target_control.sent = []
        for _, raw in self.source_control.sent:
            self.target_control.incoming.append(raw)
        self.source_control.sent = []
        notifs = self.target_agent.notifs.setdefault("source", [])
        # NIXL delivers a transfer's notification when the transfer completes,
        # which the source observes by releasing its handle; mirror that.
        released = self.source_agent.released_xfers
        while self._notified < len(released):
            notif = self.source_agent.xfers[released[self._notified] - 1][5]
            self._notified += 1
            if notif:
                notifs.append(notif)
        while self._sent_notifs < len(self.source_agent.sent_notifs):
            notifs.append(self.source_agent.sent_notifs[self._sent_notifs][1])
            self._sent_notifs += 1

    def pump(self, until, *, timeout: float = 2.0) -> None:
        """Shuttle and poll both cores until ``until()`` holds."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.shuttle()
            self.source.poll_completed()
            self.target_completed.extend(self.target.poll_completed())
            if until():
                return
            time.sleep(0.001)
        raise AssertionError("prefetch condition was not reached")

    def terminal(self, action_id: str):
        return [
            (r.status, r.reason)
            for r in self.results
            if r.action_id == action_id and r.status is not HintStatus.QUEUED
        ]

    def start_writes(self) -> list[dict]:
        return [m for m in self.to_source if m["type"] == "start_write"]

    def local_contents(self) -> set[bytes]:
        raw = self.target_local.raw
        return {raw[i : i + BLOCK] for i in range(0, len(raw), BLOCK)} - {b"\0" * BLOCK}


# The three scenarios the GPU e2e runs.


def test_fetch_only_hint_is_served_by_deliver_and_leaves_dram_alone(monkeypatch):
    pair = _Pair(monkeypatch)
    pair.seed(1)
    k1 = _key(1)
    pair.target.submit_hint(_envelope(_fetch("a1", [1])), "req1")
    assert _take(pair.results) == [("a1", HintStatus.ACCEPTED, None)]
    assert pair.target.query((k1,), "req1") == [
        (QueryStatus.FETCHABLE, CacheTier.REMOTE_G2)
    ]

    handle = pair.target.deliver({k1: [_mem_descriptor()]}, request_id="req1")
    pair.pump(lambda: bool(pair.target_completed))
    assert pair.target_completed == [(handle, _op_entries({k1: True}))]
    assert pair.target_primary.raw[:BLOCK] == _content(1)
    # A fetch hint moves nothing into KVCR DRAM.
    assert pair.target.query((k1,)) == [_MISS]
    assert pair.local_contents() == set()
    assert pair.events == []


def test_prefetch_only_lands_blocks_for_a_later_request(monkeypatch, caplog):
    monkeypatch.setenv("KVCR_KEY_HISTORY", "1")
    caplog.set_level(logging.DEBUG, logger="kvcr.core")
    pair = _Pair(monkeypatch)
    pair.seed(1, 2)
    k1, k2 = _key(1), _key(2)

    pair.target.submit_hint(_envelope(_prefetch("c1", [1, 2])), "req1")
    assert _take(pair.results) == [("c1", HintStatus.QUEUED, None)]
    # The carrier finishing must not cancel a prefetch made on another's behalf.
    pair.target.discard_hint("req1")

    pair.pump(lambda: bool(pair.terminal("c1")))
    assert pair.results == [
        HintResult("c1", "kv.fetch", HintStatus.COMPLETED, None, "req1", "m1")
    ]
    assert [m["type"] for m in pair.to_source] == ["resolve_keys", "start_write"]
    (start_write,) = pair.start_writes()
    assert start_write["keys"] == [k1, k2]
    assert pair.local_contents() == {_content(1), _content(2)}
    assert pair.events == [InventoryEvent((k1, k2), CacheTier.LOCAL_G2, False)]
    assert pair.target._core._remote_fw_dram._request_hints == {}

    # A later request finds the blocks locally and pulls nothing over the wire.
    assert pair.target.query((k1, k2), "req2") == [_HIT, _HIT]
    sent_before = len(pair.to_source)
    handle = pair.target.fetch((k1, k2), request_id="req2")
    pair.pump(lambda: bool(pair.target_completed))
    (completed,) = pair.target_completed
    assert completed[0] == handle
    assert all(
        r.success and r.release_handle is not None for r in completed[1].values()
    )
    assert len(pair.to_source) == sent_before
    assert [
        h["event"] for h in _key_history(caplog) if h["event"].startswith("remote_")
    ] == ["remote_fetch_completed"]


def test_fetch_and_prefetch_in_one_envelope(monkeypatch):
    pair = _Pair(monkeypatch)
    pair.seed(1, 2, 3)
    k1, k2, k3 = _key(1), _key(2), _key(3)
    entries = []
    inner = pair.target._core._state_lock

    class _CountingLock:
        def __enter__(self):
            entries.append("enter")
            return inner.__enter__()

        def __exit__(self, *exc):
            return inner.__exit__(*exc)

        def acquire(self, *args, **kwargs):
            entries.append("acquire")
            return inner.acquire(*args, **kwargs)

        def release(self):
            return inner.release()

    pair.target._core._state_lock = _CountingLock()
    pair.target.submit_hint(
        _envelope(_fetch("a1", [1]), _prefetch("c1", [2, 3])), "req1"
    )
    assert entries == []  # neither destination takes the lock at submit time
    pair.target._core._state_lock = inner
    assert _take(pair.results) == [
        ("a1", HintStatus.ACCEPTED, None),
        ("c1", HintStatus.QUEUED, None),
    ]

    handle = pair.target.deliver({k1: [_mem_descriptor()]}, request_id="req1")
    pair.pump(lambda: bool(pair.target_completed) and bool(pair.terminal("c1")))
    assert pair.target_completed == [(handle, _op_entries({k1: True}))]
    assert pair.target_primary.raw[:BLOCK] == _content(1)
    assert pair.terminal("c1") == [(HintStatus.COMPLETED, None)]
    assert pair.target.query((k2, k3), "req2") == [_HIT, _HIT]
    assert sorted(m["keys"] for m in pair.start_writes()) == sorted([[k1], [k2, k3]])


# Edge cases.


def test_partially_resolved_prefetch_reports_useful_partial_success(monkeypatch):
    pair = _Pair(monkeypatch)
    pair.seed(1)
    pair.target.submit_hint(_envelope(_prefetch("c1", [1, 9])), "req1")
    pair.pump(lambda: bool(pair.terminal("c1")))
    assert pair.terminal("c1") == [(HintStatus.COMPLETED, "partial")]
    assert pair.target.query((_key(1), _key(9)), "req2") == [_HIT, _MISS]
    assert pair.start_writes()[0]["keys"] == [_key(1)]


def test_prefetch_nothing_resolves(monkeypatch):
    pair = _Pair(monkeypatch)
    pair.seed(1)
    pair.target.submit_hint(_envelope(_prefetch("c1", [8, 9])), "req1")
    pair.pump(lambda: bool(pair.terminal("c1")))
    assert pair.terminal("c1") == [(HintStatus.FAILED, "unresolved")]
    assert pair.start_writes() == []
    assert pair.local_contents() == set()


def test_prefetch_skips_blocks_already_resident_or_filling(monkeypatch):
    pair = _Pair(monkeypatch)
    pair.seed(1, 2)
    k1 = _key(1)
    # k1 is already in the target's DRAM (deposited from its own framework tier).
    ctypes.memmove(pair.target_primary, _content(1), BLOCK)
    deposit = pair.target.deposit({k1: [_mem_descriptor()]})
    pair.target_agent.state = "DONE"
    assert _poll_until(pair.target, bool) == [(deposit, _op_entries({k1: True}))]
    pair.target_agent.state = "PROC"

    pair.target.submit_hint(_envelope(_prefetch("c1", [1, 2])), "req1")
    pair.pump(lambda: bool(pair.terminal("c1")))
    assert pair.terminal("c1") == [(HintStatus.COMPLETED, None)]
    assert pair.start_writes()[0]["keys"] == [_key(2)]  # only the missing block
    assert pair.target.query((k1, _key(2)), "req2") == [_HIT, _HIT]


class _DropPrefetchPolicy(FIFOPolicy):
    def decide_ingest(
        self, meta, source, required_local, router_hints=None, framework_hints=None
    ):
        if source is CacheTier.REMOTE_G2 and not required_local:
            return (PlacementAction.DROP, None)
        return super().decide_ingest(
            meta, source, required_local, router_hints, framework_hints
        )


@pytest.mark.parametrize("drop", [False, True], ids=["evicts", "policy_drops"])
def test_prefetch_into_a_full_pool_evicts_unless_the_policy_drops(monkeypatch, drop):
    pair = _Pair(
        monkeypatch,
        target_slots=1,
        target_policy=_DropPrefetchPolicy() if drop else None,
    )
    pair.seed(1)
    k9 = _key(9)
    ctypes.memmove(pair.target_primary, _content(9), BLOCK)
    deposit = pair.target.deposit({k9: [_mem_descriptor()]})
    pair.target_agent.state = "DONE"
    assert _poll_until(pair.target, bool) == [(deposit, _op_entries({k9: True}))]
    pair.target_agent.state = "PROC"
    pair.events.clear()

    pair.target.submit_hint(_envelope(_prefetch("c1", [1])), "req1")
    pair.pump(lambda: bool(pair.terminal("c1")))
    if drop:
        assert pair.terminal("c1") == [(HintStatus.FAILED, "fetch_failed")]
        assert pair.target.query((k9, _key(1))) == [_HIT, _MISS]
        assert pair.events == []
    else:
        assert pair.terminal("c1") == [(HintStatus.COMPLETED, None)]
        assert pair.target.query((k9, _key(1))) == [_MISS, _HIT]
        assert pair.events[0] == InventoryEvent((k9,), CacheTier.LOCAL_G2, True)


def test_close_settles_queued_and_in_flight_prefetches(monkeypatch):
    pair = _Pair(monkeypatch)
    pair.seed(1)
    pair.target.submit_hint(_envelope(_prefetch("c1", [1])), "req1")
    pair.target.poll_completed()  # c1 admitted, resolving
    pair.target.submit_hint(_envelope(_prefetch("c2", [1]), message_id="m2"), "req2")
    pair.results.clear()
    pair.target.close()
    assert sorted(pair.results, key=lambda result: result.action_id) == [
        HintResult("c1", "kv.fetch", HintStatus.FAILED, "closed", "req1", "m1"),
        HintResult("c2", "kv.fetch", HintStatus.DROPPED, "closed", "req2", "m2"),
    ]


def test_unanswered_resolve_fails_unresolved_at_the_deadline(monkeypatch):
    now = [100.0]
    pair = _Pair(
        monkeypatch,
        clock=lambda: now[0],
        target_config={"abandon_timeout_ms": 2000},  # the prefetch deadline
    )
    pair.target.submit_hint(_envelope(_prefetch("c1", [1])), "req1")
    pair.target.poll_completed()
    # Nothing is shuttled: the source never hears the question.
    now[0] = 102.5
    assert _poll_until(pair.target, lambda _: bool(pair.terminal("c1"))) == []
    assert pair.terminal("c1") == [(HintStatus.FAILED, "unresolved")]


def test_prefetch_clears_previous_remote_choices_and_delivers_local_bytes(monkeypatch):
    pair = _Pair(monkeypatch)
    pair.seed(1)
    key = _key(1)
    for request in ("old-a", "old-b"):
        pair.target.submit_hint(_envelope(_fetch(request, [1])), request)
        assert pair.target.query([key], request) == [
            (QueryStatus.FETCHABLE, CacheTier.REMOTE_G2)
        ]
    pair.target.submit_hint(_envelope(_prefetch("cpu", [1])))
    pair.pump(lambda: bool(pair.terminal("cpu")))
    pulls = len(pair.start_writes())
    pair.target_agent.state = "DONE"
    for request in ("old-a", "old-b", "new"):
        assert pair.target.query([key], request) == [_HIT]
        op = pair.target.deliver({key: [_mem_descriptor()]}, request)
        pair.pump(lambda: any(handle == op for handle, _ in pair.target_completed))
        assert pair.target_primary.raw[:BLOCK] == b"a" * BLOCK
    assert len(pair.start_writes()) == pulls == 1
    assert pair.target._core._local_dram._public_claims == {}


def test_concurrent_duplicate_hints_join_one_fill(monkeypatch):
    pair = _Pair(monkeypatch)
    pair.seed(1)
    pair.source_agent.state = "PROC"
    pair.target.submit_hint(_envelope(_prefetch("one", [1])))
    pair.pump(lambda: bool(pair.start_writes()))
    pair.target.submit_hint(_envelope(_prefetch("two", [1])))
    pair.pump(lambda: len(pair.target._core._local_dram._pending_residency_ops) == 2)
    pair.source_agent.state = "DONE"
    pair.pump(lambda: bool(pair.terminal("one")) and bool(pair.terminal("two")))
    assert (
        pair.terminal("one") == pair.terminal("two") == [(HintStatus.COMPLETED, None)]
    )
    assert len(pair.start_writes()) == 1
    assert pair.target_completed == []
    assert pair.target._core._local_dram._public_claims == {}
    assert pair.target._core._block_record_map[_key(1)].local_dram.claim_count == 0


def test_prefetch_full_claimed_pool_does_not_wait_for_capacity(monkeypatch):
    pair = _Pair(monkeypatch, target_slots=1)
    pair.seed(1)
    pair.target_agent.state = "DONE"
    op = pair.target.deposit({_key(9): [_mem_descriptor()]}, no_evict=True)
    entries = dict(_poll_until(pair.target, bool))[op]
    pair.target.submit_hint(_envelope(_prefetch("cpu", [1])))
    pair.pump(lambda: bool(pair.terminal("cpu")))
    assert pair.terminal("cpu") == [(HintStatus.FAILED, "fetch_failed")]
    assert not pair.target._core._local_dram._capacity_waiters
    assert pair.start_writes() == []
    pair.target.release([entries[_key(9)].release_handle])


def test_one_hash_resolves_multiple_cache_groups(monkeypatch):
    pair = _Pair(monkeypatch)
    pair.seed(1)
    second = _key(1, 1)
    op = pair.source.deposit({second: [_mem_descriptor(1, end_point_name="source")]})
    assert dict(_poll_until(pair.source, bool))[op][second].success
    pair.target.submit_hint(_envelope(_prefetch("cpu", [1])))
    pair.pump(lambda: bool(pair.terminal("cpu")))
    assert pair.terminal("cpu") == [(HintStatus.COMPLETED, None)]
    assert pair.target.query([_key(1), second]) == [_HIT, _HIT]
    assert pair.local_contents() == {b"a" * BLOCK, b"b" * BLOCK}


def test_all_layout_groups_exist_before_synchronous_completion(monkeypatch):
    pair = _Pair(monkeypatch)
    one, two = _key(1), _key(2)
    # Distinct complete layouts; all groups finish synchronously as local hits.
    for kvcr, agent, endpoint in (
        (pair.source, pair.source_agent, "source"),
        (pair.target, pair.target_agent, "target"),
    ):
        agent.state = "DONE"
        for key, labels in ((one, [""]), (two, ["", ""])):
            op = kvcr.deposit(
                {
                    key: [
                        MemoryRef(end_point_name=endpoint, element_index=i, label=label)
                        for i, label in enumerate(labels)
                    ]
                },
                expected_layout=labels,
            )
            assert dict(_poll_until(kvcr, bool))[op][key].success
    pair.target.submit_hint(_envelope(_prefetch("cpu", [1, 2])))
    pair.pump(lambda: bool(pair.terminal("cpu")))
    assert pair.terminal("cpu") == [(HintStatus.COMPLETED, None)]
    assert pair.start_writes() == []
    assert pair.target_completed == []
    assert pair.target._core._completion_owners == {}
    assert pair.target._core._remote_fw_dram._request_hints == {}


@pytest.mark.parametrize(
    "bad",
    [
        [[2, [_key(2)], [""]]],
        [[1, [_key(1), _key(1)], [""]]],
        [[1, [_key(1)], ["unknown"]]],
        [[1, [_key(1)], []]],
        [[1, [_key(2)], [""]]],
        [[1, [_key(1)], [":part", ":part"]]],
        [[True, [_key(1)], [""]]],
        [[1, ["text-key"], [""]]],
    ],
)
def test_resolution_rejects_invalid_answers_without_starting_pulls(monkeypatch, bad):
    pair = _Pair(monkeypatch)
    pair.target.submit_hint(_envelope(_prefetch("cpu", [1])))
    pair.target.poll_completed()
    _wait_until(lambda: bool(pair.target_control.sent))
    question = _decode_control_message(pair.target_control.sent[0][1])
    pair.target_control.incoming.append(
        msgspec.msgpack.encode(
            {
                "type": "keys_resolved",
                "sender_control_endpoint": SOURCE,
                "op_handle": question["op_handle"],
                "resolved": bad,
            }
        )
    )
    _poll_until(pair.target, lambda _: bool(pair.terminal("cpu")))
    assert pair.terminal("cpu") == [(HintStatus.FAILED, "unresolved")]
    assert not any(
        _decode_control_message(raw)["type"] == "start_write"
        for _, raw in pair.target_control.sent
    )
    assert pair.target.query([_key(1)]) == [_MISS]


@pytest.mark.parametrize("pinned", [False, True], ids=["new-pin", "already-pinned"])
def test_evicted_source_never_falls_back_to_framework_memory(monkeypatch, pinned):
    from kvcr.progress import _TransferRef
    from kvcr.remote_fw_dram import _FwMemResidency

    pair = _Pair(monkeypatch, source_slots=1)
    pair.seed(1)
    key = _key(1)
    source_core = pair.source._core
    remote = source_core._remote_fw_dram
    original_resolve = remote._handle_resolve_keys
    evicted = []

    def resolve_then_evict(progress, payload):
        original_resolve(progress, payload)
        # Run the real allocator's eviction path after it sends the advisory answer.
        with source_core._state_lock:
            record = source_core._block_record_map[key]
            residency = record.local_dram
            if pinned:
                record.fw_mem = _FwMemResidency([_TransferRef("source", 0, "")], "pin")
                source_core._framework_pin_keys["pin"] = {key}
            residency.retire_on_release = True
            source_core._local_dram._acquire_claim(key, residency)
            source_core._local_dram._release_claim(key, residency)
            evicted.append(key)

    monkeypatch.setattr(remote, "_handle_resolve_keys", resolve_then_evict)
    pair.target.submit_hint(_envelope(_prefetch("cpu", [1])))
    pair.pump(lambda: bool(pair.terminal("cpu")))
    assert evicted == [key]
    assert pair.start_writes()[0]["dram_only"] is True
    assert pair.terminal("cpu") == [(HintStatus.FAILED, "fetch_failed")]
    assert pair.target.query([key]) == [_MISS]
    assert pair.local_contents() == set()
    assert pair.source_agent.transfers == [1]  # only the original deposit
    assert source_core._request_pin_callback.__self__.searches == []


def test_resolution_ignores_wrong_peer_and_operation(monkeypatch):
    pair = _Pair(monkeypatch)
    pair.target.submit_hint(_envelope(_prefetch("cpu", [1])))
    pair.target.poll_completed()
    _wait_until(lambda: bool(pair.target_control.sent))
    question = _decode_control_message(pair.target_control.sent[0][1])
    reply = {
        "type": "keys_resolved",
        "sender_control_endpoint": SOURCE,
        "op_handle": question["op_handle"],
        "resolved": [],
    }
    pair.target_control.incoming.extend(
        [
            msgspec.msgpack.encode({**reply, "op_handle": question["op_handle"] - 1}),
            msgspec.msgpack.encode(
                {**reply, "sender_control_endpoint": "tcp://other:1"}
            ),
        ]
    )
    _wait_until(lambda: not pair.target_control.incoming)
    assert pair.target.poll_completed() == []
    assert pair.terminal("cpu") == []
    pair.target_control.incoming.append(msgspec.msgpack.encode(reply))
    _poll_until(pair.target, lambda _: bool(pair.terminal("cpu")))
    assert pair.terminal("cpu") == [(HintStatus.FAILED, "unresolved")]


def test_prefetch_timeout_preserves_quarantine_until_late_native_completion(
    monkeypatch,
):
    now = [100.0]
    pair = _Pair(
        monkeypatch,
        clock=lambda: now[0],
        target_config={"operation_timeout_ms": 1000, "abandon_timeout_ms": 3000},
    )
    pair.seed(1)
    pair.source_agent.state = "PROC"
    pair.target.submit_hint(_envelope(_prefetch("cpu", [1])))
    pair.pump(lambda: bool(pair.start_writes()))
    now[0] = 101.5
    pair.pump(lambda: bool(pair.terminal("cpu")))
    assert pair.terminal("cpu") == [(HintStatus.FAILED, "fetch_failed")]
    assert pair.target.query([_key(1)]) == [_MISS]
    assert len(pair.target._core._local_dram._free_slots[""]) == 3
    now[0] = 103.5
    pair.pump(
        lambda: any(
            op.uncertain
            for op in pair.target._core._progress._in_flight_ops.values()
            if hasattr(op, "uncertain")
        )
    )
    assert len(pair.target._core._local_dram._free_slots[""]) == 3
    pair.source_agent.state = "DONE"
    pair.pump(lambda: not _has_outstanding_operations(pair.target))
    assert pair.terminal("cpu") == [(HintStatus.FAILED, "fetch_failed")]
    assert len(pair.target._core._local_dram._free_slots[""]) == 4
    assert pair.events == []
    assert pair.target_completed == []


def test_close_settles_owned_fetch_once_and_releases_completion_owner(monkeypatch):
    pair = _Pair(monkeypatch)
    pair.seed(1)
    pair.target.submit_hint(_envelope(_prefetch("cpu", [1])))
    pair.pump(lambda: bool(pair.start_writes()))
    # Drain native work without polling target main: the internal fetch owner
    # still needs settlement when close consumes the pending notifications.
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        pair.shuttle()
        pair.source.poll_completed()
        if (
            len(pair.source_agent.released_xfers) == 2
            and not pair.target._core._progress._in_flight_ops
        ):
            break
        time.sleep(0.001)
    else:
        raise AssertionError("native pull did not finish")
    pair.target.close()
    assert pair.terminal("cpu") == [(HintStatus.FAILED, "closed")]
    assert pair.target._core._completion_owners == {}
    assert pair.target._core._remote_fw_dram._request_hints == {}
    pair.target.close()
    assert pair.terminal("cpu") == [(HintStatus.FAILED, "closed")]


@pytest.mark.parametrize("fail", [False, True])
def test_prefetch_only_copies_missing_pieces_and_preserves_committed_bytes(
    monkeypatch, fail
):
    pair = _Pair(monkeypatch, parts=True)
    key = _key(1)
    layout = [":a", ":b"]
    source_op = pair.source.deposit(
        {
            key: [
                _mem_descriptor(0, ":a", end_point_name="source"),
                _mem_descriptor(1, ":b", end_point_name="source"),
            ]
        },
        expected_layout=layout,
    )
    assert dict(_poll_until(pair.source, bool))[source_op][key].success
    ctypes.memmove(pair.target_primary, b"z" * BLOCK, BLOCK)
    pair.target_agent.state = "DONE"
    target_op = pair.target.deposit(
        {key: [_mem_descriptor(0, ":a")]}, expected_layout=layout
    )
    assert dict(_poll_until(pair.target, bool))[target_op][key].success
    assert pair.events == []  # incomplete blocks are not advertised
    assert pair.target._core._remote_fw_dram._hash_index == {}
    original = pair.target._core._block_record_map[key].local_dram
    if fail:
        pair.source_agent.transfer = lambda handle: "ERR"
    pair.target.submit_hint(_envelope(_prefetch("cpu", [1])))
    pair.pump(lambda: bool(pair.terminal("cpu")))
    assert pair.target._core._block_record_map[key].local_dram is original
    assert pair.target_local.raw[:BLOCK] == b"z" * BLOCK
    pull = pair.start_writes()[0]
    assert [ref["label"] for ref in pull["dst_descriptors"][0]] == [":b"]
    if fail:
        assert pair.terminal("cpu") == [(HintStatus.FAILED, "fetch_failed")]
        assert original.pending_labels == {":b"}
        assert pair.events == []
    else:
        assert pair.terminal("cpu") == [(HintStatus.COMPLETED, None)]
        assert not original.pending_labels
        assert pair.target_local.raw[BLOCK : 2 * BLOCK] == b"b" * BLOCK
        assert pair.events == [InventoryEvent((key,), CacheTier.LOCAL_G2, False)]
    assert original.claim_count == 0
    assert pair.target_completed == []


@pytest.mark.parametrize("keys", [[], [_key(1)]], ids=["empty", "no-local-backend"])
def test_internal_fetch_immediate_failure_never_reaches_engine_queue(keys):
    target = _new_kvcr(FakeNixlAgent(), FakePrimaryPinning(), FakeBytesControl())
    completed = []
    target._core.fetch(keys, on_complete=completed.append, claim_on_ready=False)
    assert len(completed) == 1
    assert set(completed[0]) == set(keys)
    assert all(not entry.success for entry in completed[0].values())
    assert target.poll_completed() == []
    assert not target._core._completion_owners


def test_native_setup_exception_settles_internal_owner(monkeypatch):
    pair = _Pair(monkeypatch)
    pair.seed(1)

    def fail_setup(*args, **kwargs):
        raise RuntimeError("injected native transfer setup failure")

    monkeypatch.setattr(pair.source_agent, "initialize_xfer", fail_setup)
    pair.target.submit_hint(_envelope(_prefetch("cpu", [1])))
    pair.pump(lambda: bool(pair.terminal("cpu")))
    assert pair.terminal("cpu") == [(HintStatus.FAILED, "fetch_failed")]
    assert not pair.target._core._completion_owners
    assert not pair.target._core._remote_fw_dram._request_hints
    assert pair.target_completed == []
    assert not _has_outstanding_operations(pair.target)
    assert len(pair.target._core._local_dram._free_slots[""]) == 4


def test_large_hint_uses_normal_fetch_batching(monkeypatch):
    pair = _Pair(monkeypatch, source_slots=65, target_slots=65)
    pair.seed(*range(1, 66))
    pair.target.submit_hint(_envelope(_prefetch("cpu", range(1, 66))))
    pair.pump(lambda: bool(pair.terminal("cpu")))
    assert pair.terminal("cpu") == [(HintStatus.COMPLETED, None)]
    assert len(pair.start_writes()) == 1
    assert len(pair.start_writes()[0]["keys"]) == 65
    assert pair.local_contents() == {_content(i) for i in range(1, 66)}


def test_dispatch_admission_needs_no_second_active_job_limit(monkeypatch):
    pair = _Pair(monkeypatch, clock=lambda: 100.0)
    for index in range(70):
        pair.target.submit_hint(_envelope(_prefetch(str(index), [1])), "carrier")
    _poll_until(pair.target, lambda _: len(pair.target_control.sent) == 70)
    assert len(pair.results) == 70
    assert all(result.status is HintStatus.QUEUED for result in pair.results)
    assert len(pair.target._core._remote_fw_dram._request_hints) == 70
    pair.target.discard_hint("carrier")
    assert len(pair.target._core._remote_fw_dram._request_hints) == 70
    pair.target.close()
    assert len(pair.results) == 140
    assert all(
        pair.terminal(str(index)) == [(HintStatus.FAILED, "closed")]
        for index in range(70)
    )
    assert not pair.target._core._remote_fw_dram._request_hints


def test_pending_prefetch_expires_without_spending_admission_budget(monkeypatch):
    monkeypatch.setattr("kvcr.hints.fetch._HINTS_PER_POLL", 1)
    now = [100.0]
    # Keep active resolution alive while pending hints cross their queue timeout.
    pair = _Pair(
        monkeypatch,
        clock=lambda: now[0],
        target_config={"operation_timeout_ms": 10000, "abandon_timeout_ms": 20000},
    )
    pair.target.submit_hint(_envelope(_prefetch("active", [1])))
    pair.target.submit_hint(_envelope(_prefetch("stale", [1])))
    pair.target.poll_completed()
    assert len(pair.target._core._remote_fw_dram._request_hints) == 1
    assert _take(pair.results) == [
        ("active", HintStatus.QUEUED, None),
        ("stale", HintStatus.QUEUED, None),
    ]

    now[0] += 6.0
    pair.target.submit_hint(_envelope(_prefetch("fresh", [1])))
    pair.target.poll_completed()
    assert _take(pair.results) == [
        ("fresh", HintStatus.QUEUED, None),
        ("stale", HintStatus.DROPPED, "stale"),
    ]
    # Expiry only applies before admission; both admitted jobs remain active.
    assert len(pair.target._core._remote_fw_dram._request_hints) == 2
    pair.target.close()
    assert _take(pair.results) == [
        ("active", HintStatus.FAILED, "closed"),
        ("fresh", HintStatus.FAILED, "closed"),
    ]


def test_failed_close_drops_pending_prefetch_and_preserves_active_job(monkeypatch):
    pair = _Pair(monkeypatch, clock=lambda: 100.0)
    pair.target.submit_hint(_envelope(_prefetch("active", [1])))
    pair.target.poll_completed()
    pair.target.submit_hint(_envelope(_prefetch("pending", [1])))
    pair.results.clear()
    with pytest.MonkeyPatch.context() as not_quiescent:
        not_quiescent.setattr(pair.target._core, "is_quiescent", lambda: False)
        with pytest.raises(RuntimeError, match="not quiescent"):
            pair.target.close()
    assert _take(pair.results) == [("pending", HintStatus.DROPPED, "closed")]
    assert len(pair.target._core._remote_fw_dram._request_hints) == 1

    pair.target.submit_hint(_envelope(_prefetch("retry", [1])))
    assert _take(pair.results) == [("retry", HintStatus.QUEUED, None)]
    pair.target.close()
    pair.target.close()
    assert _take(pair.results) == [
        ("retry", HintStatus.DROPPED, "closed"),
        ("active", HintStatus.FAILED, "closed"),
    ]
    assert not pair.target._core._remote_fw_dram._request_hints
    assert pair.target.poll_completed() == []


def test_multiple_groups_transfer_and_aggregate_mixed_results(monkeypatch):
    pair = _Pair(monkeypatch, target_slots=2)
    # The first group fills one slot; the second group needs two and fails now.
    for key, labels in ((_key(1), [""]), (_key(2), ["", ""])):
        op = pair.source.deposit(
            {
                key: [
                    _mem_descriptor(i, label, end_point_name="source")
                    for i, label in enumerate(labels)
                ]
            },
            expected_layout=labels,
        )
        assert dict(_poll_until(pair.source, bool))[op][key].success
    pair.target.submit_hint(_envelope(_prefetch("cpu", [1, 2])))
    pair.pump(lambda: bool(pair.terminal("cpu")))
    assert pair.terminal("cpu") == [(HintStatus.COMPLETED, "partial")]
    assert pair.target.query([_key(1), _key(2)]) == [_HIT, _MISS]
    assert pair.target_completed == []
    assert not pair.target._core._completion_owners
    assert not pair.target._core._remote_fw_dram._request_hints
