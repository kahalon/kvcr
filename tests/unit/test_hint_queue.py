# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Handler-owned pending hint queue behavior."""

import threading
from typing import ClassVar

import msgspec
import pytest

from kvcr.hints.deferred import _DeferredHint, _DeferredHintQueue
from kvcr.hints.results import HintResult, HintStatus


class _Hint(msgspec.Struct, frozen=True):
    ACTION_TYPE: ClassVar[str] = "kv.queue-test"

    action_id: str | None


def _queue():
    now = [100.0]
    results: list[HintResult] = []
    return _DeferredHintQueue(lambda: now[0], results.append), now, results


def _outcomes(results):
    return [(r.action_id, r.status, r.reason) for r in results]


def test_queue_preserves_fifo_timestamps_and_correlation():
    queue, now, results = _queue()
    first = _Hint("first")
    queue.defer(first, "request-1", "message-1")
    now[0] = 102.0
    queue.defer(_Hint(None), None, None)
    assert results == [
        HintResult(
            "first",
            "kv.queue-test",
            HintStatus.QUEUED,
            request_id="request-1",
            message_id="message-1",
        ),
        HintResult(None, "kv.queue-test", HintStatus.QUEUED),
    ]
    item = queue.pop()
    assert item == _DeferredHint(first, "request-1", "message-1", 100.0)
    assert item.result(HintStatus.FAILED, "handler_error") == HintResult(
        "first",
        "kv.queue-test",
        HintStatus.FAILED,
        "handler_error",
        "request-1",
        "message-1",
    )
    assert item.result(HintStatus.COMPLETED).reason is None
    assert queue.pop() == _DeferredHint(_Hint(None), None, None, 102.0)
    assert queue.pop() is None
    queue.close()
    # Popped items belong to the handler; the queue cannot settle them again.
    assert len(results) == 2


def test_full_queue_drops_oldest_once_and_keeps_newest(monkeypatch):
    monkeypatch.setattr("kvcr.hints.deferred._MAX_QUEUED_HINTS", 2)
    queue, _, results = _queue()
    for action_id in ("a", "b", "c", "d"):
        queue.defer(_Hint(action_id), f"request-{action_id}", "message")
    assert [queue.pop().hint.action_id, queue.pop().hint.action_id] == ["c", "d"]
    assert queue.pop() is None
    dropped = [r for r in results if r.status is HintStatus.DROPPED]
    assert _outcomes(dropped) == [
        ("a", HintStatus.DROPPED, "queue_full"),
        ("b", HintStatus.DROPPED, "queue_full"),
    ]
    assert [r.request_id for r in dropped] == ["request-a", "request-b"]
    for action_id in ("a", "b"):
        assert [r.status for r in results if r.action_id == action_id] == [
            HintStatus.QUEUED,
            HintStatus.DROPPED,
        ]


def test_expiry_keeps_exact_deadline_and_drops_stale_without_popping():
    queue, now, results = _queue()
    queue.defer(_Hint("old"), "request", "message")
    queue.defer(_Hint("also-old"), "request-2", "message-2")
    now[0] = 104.0
    queue.defer(_Hint("new"), None, None)
    queue.expire(105.0)
    assert len(results) == 3  # Age exactly five seconds is still valid.
    queue.expire(105.001)
    assert results[3:] == [
        HintResult(
            "old", "kv.queue-test", HintStatus.DROPPED, "stale", "request", "message"
        ),
        HintResult(
            "also-old",
            "kv.queue-test",
            HintStatus.DROPPED,
            "stale",
            "request-2",
            "message-2",
        ),
    ]
    queue.expire(105.001)
    assert len(results) == 5
    assert queue.pop().hint.action_id == "new"
    assert queue.pop() is None


def test_close_settles_pending_once_and_rejects_new_hints():
    queue, _, results = _queue()
    queue.defer(_Hint("a"), "request-a", "message-a")
    queue.defer(_Hint("b"), "request-b", "message-b")
    queue.close()
    queue.close()
    queue.discard_pending()
    queue.expire(1000.0)
    queue.defer(_Hint("c"), "request-c", "message-c")
    assert _outcomes(results) == [
        ("a", HintStatus.QUEUED, None),
        ("b", HintStatus.QUEUED, None),
        ("a", HintStatus.DROPPED, "closed"),
        ("b", HintStatus.DROPPED, "closed"),
        ("c", HintStatus.REJECTED, "closed"),
    ]
    assert [(r.request_id, r.message_id) for r in results] == [
        ("request-a", "message-a"),
        ("request-b", "message-b"),
        ("request-a", "message-a"),
        ("request-b", "message-b"),
        ("request-c", "message-c"),
    ]
    assert queue.pop() is None


@pytest.mark.parametrize("consumer", ["pop", "close"])
def test_queued_report_precedes_concurrent_consumption(consumer):
    reporting = threading.Event()
    allow_report = threading.Event()
    contended = threading.Event()
    consumed = threading.Event()
    results: list[HintResult] = []
    errors: list[BaseException] = []

    def report(result):
        if result.status is HintStatus.QUEUED:
            reporting.set()
            if not allow_report.wait(2):
                raise TimeoutError("report was not released")
        results.append(result)

    queue = _DeferredHintQueue(lambda: 100.0, report)

    class ObservedLock:
        def __init__(self):
            self.lock = threading.Lock()

        def __enter__(self):
            if not self.lock.acquire(blocking=False):
                contended.set()
                self.lock.acquire()

        def __exit__(self, *args):
            self.lock.release()

    # Observe a real failed acquisition instead of relying on a timing window.
    queue._lock = ObservedLock()

    def consume():
        if consumer == "pop":
            item = queue.pop()
            assert item is not None
            report(item.result(HintStatus.COMPLETED))
        else:
            queue.close()
        consumed.set()

    def run(fn):
        try:
            fn()
        except BaseException as error:
            errors.append(error)

    producer = threading.Thread(
        target=run, args=(lambda: queue.defer(_Hint("a"), None, None),), daemon=True
    )
    worker = threading.Thread(target=run, args=(consume,), daemon=True)
    producer.start()
    try:
        assert reporting.wait(2)
        worker.start()
        assert contended.wait(2)
        assert not consumed.is_set()
    finally:
        allow_report.set()
        producer.join(2)
        if worker.ident is not None:
            worker.join(2)
    assert not producer.is_alive() and not worker.is_alive()
    assert not errors
    terminal = HintStatus.COMPLETED if consumer == "pop" else HintStatus.DROPPED
    assert [r.status for r in results] == [HintStatus.QUEUED, terminal]
    assert queue.pop() is None
