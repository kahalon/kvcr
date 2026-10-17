# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""kv.fetch hint handler tests: the engine-destination inline path."""

import logging

import pytest
from _kvcr_test_utils import (
    FakeBytesControl,
    FakeNixlAgent,
    FakePrimaryPinning,
    _HintAction,
    _HintEnvelope,
    _new_kvcr,
    _record_hint_results,
    _router_hint,
)

from kvcr.config import KVCRConfig, RemoteFWDramOptions
from kvcr.hints.dispatcher import _LOGGED_HANDLER_ERRORS
from kvcr.hints.parser import _LOGGED_HINT_ISSUES
from kvcr.types import BlockKey, CacheTier, HintStatus, QueryStatus


@pytest.fixture(autouse=True)
def _reset_once_logs():
    # Once-per-issue warnings are process-global; reset so tests that count
    # warnings do not depend on run order.
    _LOGGED_HANDLER_ERRORS.clear()
    _LOGGED_HINT_ISSUES.clear()


def _take(results) -> list[tuple[str | None, HintStatus, str | None]]:
    """Statuses reported since the previous take."""
    taken = [(r.action_id, r.status, r.reason) for r in results]
    results.clear()
    return taken


class _PrefixAdapter:
    def decode(self, key):
        return int.from_bytes(key[:8], "big")


def _key(block_hash: int) -> BlockKey:
    return BlockKey(block_hash.to_bytes(8, "big"))


def _fetch_kvcr(monkeypatch):
    kvcr = _new_kvcr(
        FakeNixlAgent(),
        FakePrimaryPinning(),
        FakeBytesControl(),
        KVCRConfig(nixl_agent_name="target", pool_layouts=[("", 16)]),
        key_adapter=_PrefixAdapter(),
        remote_options=RemoteFWDramOptions(
            eager_ctrl_connect=False, opportunistic_query=False
        ),
    )
    return kvcr, _record_hint_results(monkeypatch, kvcr)


def _fetch(action_id: str, source: str, hashes) -> _HintAction:
    return _HintAction(
        action_id, "kv.fetch", "1.0",
        {"source_control_endpoint": source, "block_hashes": list(hashes)},
    )  # fmt: skip


def _fetch_envelope(*hints: _HintAction) -> _HintEnvelope:
    return _HintEnvelope("0.1", "m1", list(hints))


_FETCHABLE = (QueryStatus.FETCHABLE, CacheTier.REMOTE_G2)
_MISS = (QueryStatus.MISS, None)


class _CountingLock:
    def __init__(self, inner) -> None:
        self._inner = inner
        self.entries = 0

    def __enter__(self):
        self.entries += 1
        return self._inner.__enter__()

    def __exit__(self, *exc):
        return self._inner.__exit__(*exc)

    def acquire(self, *args, **kwargs):
        self.entries += 1
        return self._inner.acquire(*args, **kwargs)

    def release(self):
        return self._inner.release()


def test_gpu_fetch_hint_takes_inline_path_without_lock(monkeypatch):
    kvcr, results = _fetch_kvcr(monkeypatch)
    lock = _CountingLock(kvcr._core._state_lock)
    kvcr._core._state_lock = lock

    kvcr.submit_hint(_router_hint("tcp://a:1", (1,)), "req")

    assert lock.entries == 0
    assert kvcr.query((_key(1),), "req") == [_FETCHABLE]
    assert [(r.status, r.request_id) for r in results] == [(HintStatus.ACCEPTED, "req")]


def test_same_source_fetch_hints_union_hashes(monkeypatch):
    kvcr, results = _fetch_kvcr(monkeypatch)
    kvcr.submit_hint(
        _fetch_envelope(_fetch("a1", "tcp://a:1", [1]), _fetch("a2", "tcp://a:1", [2])),
        "req",
    )
    assert kvcr.query((_key(1), _key(2)), "req") == [_FETCHABLE, _FETCHABLE]
    assert [r.status for r in results] == [HintStatus.ACCEPTED] * 2


def test_first_fetch_source_wins_within_envelope(monkeypatch, caplog):
    kvcr, results = _fetch_kvcr(monkeypatch)
    calls = []
    remote = kvcr._core._remote_fw_dram
    real = remote.submit_hint
    remote.submit_hint = lambda *a: (calls.append(a), real(*a))[1]
    with caplog.at_level(logging.WARNING):
        kvcr.submit_hint(
            _fetch_envelope(
                _fetch("a1", "tcp://a:1", [1]),
                _fetch("b1", "tcp://b:1", [2]),
                _fetch("a2", "tcp://a:1", [3]),
            ),
            "req",
        )
    assert len(calls) == 1
    hint = remote._request_hints["req"]
    assert hint.source == "tcp://a:1" and not hint.failed
    assert kvcr.query((_key(1), _key(2), _key(3)), "req") == [
        _FETCHABLE,
        _MISS,
        _FETCHABLE,
    ]
    assert _take(results) == [
        ("a1", HintStatus.ACCEPTED, None),
        ("b1", HintStatus.REJECTED, "multiple_sources"),
        ("a2", HintStatus.ACCEPTED, None),
    ]
    assert sum("multiple_sources" in m for m in caplog.messages) == 1


def test_second_source_in_later_hint_is_rejected_and_poisons(monkeypatch):
    kvcr, results = _fetch_kvcr(monkeypatch)
    kvcr.submit_hint(_fetch_envelope(_fetch("a1", "tcp://a:1", [1])), "req")
    assert _take(results) == [("a1", HintStatus.ACCEPTED, None)]
    kvcr.submit_hint(_fetch_envelope(_fetch("b1", "tcp://b:1", [2])), "req")
    assert _take(results) == [("b1", HintStatus.REJECTED, "multiple_sources")]
    assert kvcr._core._remote_fw_dram._request_hints["req"].failed is True


def test_mixed_valid_and_unknown_hints(monkeypatch):
    kvcr, results = _fetch_kvcr(monkeypatch)
    kvcr.submit_hint(
        _fetch_envelope(
            _fetch("a1", "tcp://a:1", [1]),
            _HintAction("f1", "kv.future_action", "1.0", {}),
        ),
        "req",
    )
    assert kvcr._core._remote_fw_dram._request_hints["req"].source == "tcp://a:1"
    assert set(_take(results)) == {
        ("a1", HintStatus.ACCEPTED, None),
        ("f1", HintStatus.REJECTED, "unsupported_action_type"),
    }


def test_all_invalid_envelope_does_not_raise_from_submit_hint(monkeypatch):
    kvcr, results = _fetch_kvcr(monkeypatch)
    bad = _HintAction("a1", "kv.fetch", "1.0", {"block_hashes": []})
    kvcr.submit_hint(_fetch_envelope(bad), "req")
    assert [r.status for r in results] == [HintStatus.REJECTED]


def test_fetch_hint_without_request_id_is_accepted(monkeypatch):
    kvcr, results = _fetch_kvcr(monkeypatch)
    kvcr.submit_hint(_router_hint("tcp://a:1", (1,)))
    assert [(r.status, r.request_id) for r in results] == [(HintStatus.ACCEPTED, None)]
    assert not kvcr._core._remote_fw_dram._request_hints
