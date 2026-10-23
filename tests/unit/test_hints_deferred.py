# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Deferred hint tests: per-handler queues and admission from ``poll_completed``."""

import logging
from collections.abc import Callable
from typing import ClassVar

import pytest
from test_hints_dispatcher import (
    _envelope,
    _reset_once_logs,  # noqa: F401  (autouse fixture; applies here too)
    _stub_kvcr,
    _StubHandler,
    _StubHint,
    _take,
    _tests,
)

from kvcr import STATE_METRIC
from kvcr.hints.parser import HintType
from kvcr.hints.results import HintResult, HintStatus


class _DeferringStubHandler(_StubHandler):
    """Defers every hint; ``take`` completes it, except hints listed in
    ``fail`` (raises)."""

    def __init__(self, core, dispatcher) -> None:
        super().__init__(core, dispatcher)
        self.taken: list[str | None] = []
        self.on_take: Callable[[_StubHint], None] | None = None

    def submit(self, hint, request_id, message_id) -> None:
        self.submitted.append(hint)
        self.dispatcher.defer(self, hint, request_id, message_id)

    def take(self, hint, request_id, message_id, now) -> None:
        self.taken.append(hint.action_id)
        if self.on_take is not None:
            self.on_take(hint)
        if hint.action_id in self.fail:
            raise RuntimeError("take boom")
        # The handler owns a taken hint and reports its terminal result.
        self.dispatcher.report(
            HintResult(
                action_id=hint.action_id,
                action_type=self.action_type,
                status=HintStatus.COMPLETED,
                request_id=request_id,
                message_id=message_id,
            )
        )


class _SecondStubHint(_StubHint, frozen=True):
    ACTION_TYPE: ClassVar[str] = "kv.test2"


def _second_deferring_stub(core, dispatcher) -> _DeferringStubHandler:
    handler = _DeferringStubHandler(core, dispatcher)
    handler.action_type = "kv.test2"
    return handler


_SECOND_DEFERRING_TYPE = HintType(
    "kv.test2", "1.0", lambda i, p: _SecondStubHint(i), _second_deferring_stub
)


def _deferring_kvcr(monkeypatch, **kwargs):
    return _stub_kvcr(monkeypatch, _DeferringStubHandler, **kwargs)


def _frozen_clock(kvcr) -> list[float]:
    """Replace the core clock with a settable one; returns its single cell."""
    now = [100.0]
    kvcr._core._clock = lambda: now[0]
    return now


def _queued(kvcr, action_type: str = "kv.test") -> list[str | None]:
    queue = kvcr._core._hints_dispatcher._queues[action_type]
    return [item.hint.action_id for item in queue]


def test_only_handlers_with_take_get_a_queue(monkeypatch, caplog):
    kvcr, handler, results = _stub_kvcr(monkeypatch)  # inline stub: no take
    dispatcher = kvcr._core._hints_dispatcher
    assert dispatcher._queues == {}
    # A handler without take has nowhere to defer to: handler_error.
    handler.submit = lambda hint, r, m: dispatcher.defer(handler, hint, r, m)
    with caplog.at_level(logging.WARNING, logger="kvcr.hints.dispatcher"):
        kvcr.submit_hint(_tests("a"))
    assert _take(results) == [("a", HintStatus.REJECTED, "handler_error")]
    assert len([r for r in caplog.records if r.name == "kvcr.hints.dispatcher"]) == 1

    deferring, _, _ = _deferring_kvcr(monkeypatch)
    assert list(deferring._core._hints_dispatcher._queues) == ["kv.test"]


def test_deferred_hints_are_admitted_in_order_within_the_budget(monkeypatch):
    monkeypatch.setattr("kvcr.hints.dispatcher._HINTS_PER_ADMIT", 2)
    kvcr, handler, results = _deferring_kvcr(monkeypatch)
    ids = [str(i) for i in range(5)]
    kvcr.submit_hint(_tests(*ids), request_id="req")
    assert all((r.request_id, r.message_id) == ("req", "m1") for r in results)
    assert _take(results) == [(i, HintStatus.QUEUED, None) for i in ids]

    for admitted in (ids[:2], ids[2:4], ids[4:]):
        kvcr.poll_completed()
        assert _take(results) == [(i, HintStatus.COMPLETED, None) for i in admitted]
    assert handler.taken == ids
    assert _queued(kvcr) == []


def test_stale_hints_are_dropped_before_taking(monkeypatch):
    kvcr, handler, results = _deferring_kvcr(monkeypatch)
    now = _frozen_clock(kvcr)
    kvcr.submit_hint(_tests("a"))
    now[0] += 6  # "a" is now older than the timeout; "b" is not.
    kvcr.submit_hint(_tests("b"))
    results.clear()

    kvcr.poll_completed()
    # The stale head is dropped without reaching the handler.
    assert _take(results) == [
        ("a", HintStatus.DROPPED, "stale"),
        ("b", HintStatus.COMPLETED, None),
    ]
    assert handler.taken == ["b"]
    assert _queued(kvcr) == []


def test_full_queue_drops_the_oldest_hint(monkeypatch):
    monkeypatch.setattr("kvcr.hints.dispatcher._MAX_QUEUED_HINTS", 2)
    kvcr, handler, results = _deferring_kvcr(monkeypatch)
    kvcr.submit_hint(_tests("a", "b", "c"))
    assert _take(results) == [
        ("a", HintStatus.QUEUED, None),
        ("b", HintStatus.QUEUED, None),
        ("c", HintStatus.QUEUED, None),
        ("a", HintStatus.DROPPED, "queue_full"),
    ]
    kvcr.poll_completed()
    assert _take(results) == [
        ("b", HintStatus.COMPLETED, None),
        ("c", HintStatus.COMPLETED, None),
    ]


def test_take_exception_fails_only_its_hint_and_counts_toward_budget(
    monkeypatch, caplog
):
    monkeypatch.setattr("kvcr.hints.dispatcher._HINTS_PER_ADMIT", 2)
    kvcr, handler, results = _deferring_kvcr(monkeypatch)
    handler.fail = {"a", "b"}
    kvcr.submit_hint(_tests("a", "b", "c"))
    results.clear()
    with caplog.at_level(logging.WARNING, logger="kvcr.hints.dispatcher"):
        assert list(kvcr.poll_completed()) == []
    assert _take(results) == [
        ("a", HintStatus.FAILED, "handler_error"),
        ("b", HintStatus.FAILED, "handler_error"),
    ]
    warnings = [r for r in caplog.records if r.name == "kvcr.hints.dispatcher"]
    assert len(warnings) == 1
    assert warnings[0].exc_info is not None

    kvcr.poll_completed()
    assert _take(results) == [("c", HintStatus.COMPLETED, None)]


def test_defer_from_take_fails_the_hint(monkeypatch):
    kvcr, handler, results = _deferring_kvcr(monkeypatch)
    dispatcher = kvcr._core._hints_dispatcher
    handler.on_take = lambda hint: dispatcher.defer(handler, hint, None, None)
    kvcr.submit_hint(_tests("a"))
    results.clear()
    kvcr.poll_completed()
    # Re-deferring would loop within the poll; the hint fails instead.
    assert _take(results) == [("a", HintStatus.FAILED, "handler_error")]
    assert handler.taken == ["a"]
    assert _queued(kvcr) == []


def test_progress_failure_during_take_is_not_a_handler_error(monkeypatch):
    kvcr, handler, results = _deferring_kvcr(monkeypatch)
    kvcr.submit_hint(_tests("a"))
    results.clear()

    def fail_then_raise(hint) -> None:
        kvcr._core._progress._failure = RuntimeError("boom")
        kvcr._core._progress.raise_if_failed()

    handler.on_take = fail_then_raise
    with pytest.raises(RuntimeError, match="boom"):
        kvcr.poll_completed()
    assert results == []
    kvcr._core._progress._failure = None


def test_queues_are_served_in_table_order_under_one_budget(monkeypatch):
    monkeypatch.setattr("kvcr.hints.dispatcher._HINTS_PER_ADMIT", 2)
    kvcr, first, results = _deferring_kvcr(
        monkeypatch, extra_types=(_SECOND_DEFERRING_TYPE,)
    )
    second = kvcr._core._hints_dispatcher._handlers["kv.test2"]
    kvcr.submit_hint(
        _envelope(
            ("a", "kv.test"), ("b", "kv.test2"), ("c", "kv.test2"), ("d", "kv.test2")
        )
    )
    results.clear()

    kvcr.poll_completed()
    # kv.test comes first in the table; what is left of the budget goes on.
    assert (first.taken, second.taken) == (["a"], ["b"])
    assert (_queued(kvcr), _queued(kvcr, "kv.test2")) == ([], ["c", "d"])
    assert _take(results) == [
        ("a", HintStatus.COMPLETED, None),
        ("b", HintStatus.COMPLETED, None),
    ]

    kvcr.poll_completed()
    assert second.taken == ["b", "c", "d"]
    assert _take(results) == [
        ("c", HintStatus.COMPLETED, None),
        ("d", HintStatus.COMPLETED, None),
    ]


def test_close_drops_queued_hints_before_handlers_close(monkeypatch):
    kvcr, handler, results = _deferring_kvcr(monkeypatch)
    kvcr.submit_hint(_tests("a", "b"))
    results.clear()
    with pytest.MonkeyPatch.context() as not_quiescent:
        not_quiescent.setattr(kvcr._core, "is_quiescent", lambda: False)
        with pytest.raises(RuntimeError, match="not quiescent"):
            kvcr.close()
    # Queued hints never started, so they are dropped even though close
    # failed; the handler may own in-flight work and stays open.
    assert _take(results) == [
        ("a", HintStatus.DROPPED, "closed"),
        ("b", HintStatus.DROPPED, "closed"),
    ]
    assert handler.close_calls == 0

    # A hint deferred before close is retried is dropped by the retry.
    kvcr.submit_hint(_tests("c"))
    assert _take(results) == [("c", HintStatus.QUEUED, None)]
    kvcr.close()
    assert _take(results) == [("c", HintStatus.DROPPED, "closed")]
    assert handler.close_calls == 1
    assert handler.taken == []


def test_deferred_results_emit_no_metric(monkeypatch):
    # Integrations forward KVCR stats and may reject names they do not
    # declare (vLLM asserts), so the deferred path adds no metric and no
    # state gauge either.
    kvcr, _, results = _deferring_kvcr(monkeypatch, enable_telemetry=True)
    kvcr.get_stats()  # start from an empty snapshot
    kvcr.submit_hint(_envelope(("x", "kv.whatever-1234"), ("a", "kv.test")))
    kvcr.poll_completed()
    assert [(r.action_type, r.status) for r in results] == [
        ("kv.whatever-1234", HintStatus.REJECTED),
        ("kv.test", HintStatus.QUEUED),
        ("kv.test", HintStatus.COMPLETED),
    ]
    records = kvcr.get_stats().records
    assert {name for _, name, _, _ in records} == {STATE_METRIC}
    assert not [labels for _, _, _, labels in records if "hint" in str(labels)]
