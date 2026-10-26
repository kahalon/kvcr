# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Deferred hint ownership across core shutdown and retry."""

import logging
from typing import ClassVar

import pytest
from test_hints_dispatcher import (
    _envelope,
    _reset_once_logs,  # noqa: F401 (autouse fixture; applies here too)
    _stub_kvcr,
    _StubHandler,
    _StubHint,
    _take,
    _tests,
)

from kvcr.hints.deferred import _DeferredHintQueue
from kvcr.hints.parser import HintType
from kvcr.hints.results import HintStatus


class _DeferringStubHandler(_StubHandler):
    def __init__(self, core, dispatcher):
        super().__init__(core, dispatcher)
        self.queue = _DeferredHintQueue(
            lambda: core._clock(), lambda result: dispatcher.report(result)
        )
        self.active = []

    def submit(self, hint, request_id, message_id):
        self.queue.defer(hint, request_id, message_id)

    def poll_main(self, items):
        item = self.queue.pop()
        if item is not None:
            self.active.append(item)
        return list(items)

    def discard_pending(self):
        self.queue.discard_pending()

    def close(self):
        self.queue.close()
        for item in self.active:
            self.dispatcher.report(item.result(HintStatus.FAILED, "closed"))
        self.active.clear()
        super().close()


class _SecondStubHint(_StubHint, frozen=True):
    ACTION_TYPE: ClassVar[str] = "kv.test2"


def _second_deferring_stub(core, dispatcher):
    handler = _DeferringStubHandler(core, dispatcher)
    handler.action_type = "kv.test2"
    return handler


_SECOND_DEFERRING_TYPE = HintType(
    "kv.test2", "1.0", lambda i, p: _SecondStubHint(i), _second_deferring_stub
)


def test_close_drops_pending_before_quiescence_and_settles_active_later(monkeypatch):
    kvcr, handler, results = _stub_kvcr(monkeypatch, _DeferringStubHandler)
    kvcr.submit_hint(_tests("active"))
    assert _take(results) == [("active", HintStatus.QUEUED, None)]
    assert handler.active == []
    kvcr.poll_completed()
    kvcr.submit_hint(_tests("pending-a", "pending-b"))
    results.clear()
    with pytest.MonkeyPatch.context() as not_quiescent:
        not_quiescent.setattr(kvcr._core, "is_quiescent", lambda: False)
        with pytest.raises(RuntimeError, match="not quiescent"):
            kvcr.close()
    assert _take(results) == [
        ("pending-a", HintStatus.DROPPED, "closed"),
        ("pending-b", HintStatus.DROPPED, "closed"),
    ]
    assert handler.close_calls == 0
    assert [item.hint.action_id for item in handler.active] == ["active"]
    kvcr.submit_hint(_tests("retry-a", "retry-b"))
    assert _take(results) == [
        ("retry-a", HintStatus.QUEUED, None),
        ("retry-b", HintStatus.QUEUED, None),
    ]
    kvcr.close()
    kvcr.close()
    assert _take(results) == [
        ("retry-a", HintStatus.DROPPED, "closed"),
        ("retry-b", HintStatus.DROPPED, "closed"),
        ("active", HintStatus.FAILED, "closed"),
    ]
    assert handler.close_calls == 1
    assert handler.active == []
    kvcr.submit_hint(_tests("late"))
    assert _take(results) == [("late", HintStatus.REJECTED, "closed")]
    kvcr.poll_completed()
    assert results == []


def test_pending_cleanup_failure_does_not_skip_other_handlers(monkeypatch, caplog):
    class BrokenCleanup(_DeferringStubHandler):
        def discard_pending(self):
            raise RuntimeError("cleanup boom")

    kvcr, first, results = _stub_kvcr(
        monkeypatch,
        handler_factory=BrokenCleanup,
        extra_types=(_SECOND_DEFERRING_TYPE,),
    )
    kvcr.submit_hint(_envelope(("a", "kv.test"), ("b", "kv.test2")))
    results.clear()
    with caplog.at_level(logging.WARNING, logger="kvcr.hints.dispatcher"):
        kvcr.close()
    assert _take(results) == [
        ("b", HintStatus.DROPPED, "closed"),
        ("a", HintStatus.DROPPED, "closed"),
    ]
    assert first.close_calls == 1
    warnings = [r for r in caplog.records if r.name == "kvcr.hints.dispatcher"]
    assert len(warnings) == 1 and "discard_pending" in warnings[0].getMessage()
