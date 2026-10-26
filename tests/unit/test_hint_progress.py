# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Hint handlers consume the existing main-thread progress item stream."""

import logging
from dataclasses import dataclass

import pytest
from _kvcr_test_utils import _poll_until
from test_hints_dispatcher import (
    _reset_once_logs,  # noqa: F401 (autouse fixture; applies here too)
    _stub_kvcr,
    _StubHandler,
    _StubHint,
)

from kvcr.hints.parser import HintType
from kvcr.progress import _ProgressOp
from kvcr.remote_fw_dram import _ProgressUpdate


@dataclass
class _HintProgressOp(_ProgressOp):
    action_type: str

    def progress(self, progress, event):
        return True, True


class _ProgressHandler(_StubHandler):
    def __init__(self, core, dispatcher):
        super().__init__(core, dispatcher)
        self.batches = []
        self.finished = []
        self.connected_counts = []
        self.fail_after_first = False

    def poll_main(self, items):
        assert self.core._state_lock._is_owned()
        self.batches.append(list(items))
        self.connected_counts.append(self.core._remote_fw_dram._connected_remote_count)
        unhandled = []
        for item in items:
            if (
                isinstance(item, _HintProgressOp)
                and item.action_type == self.action_type
            ):
                self.finished.append(item)
                if self.fail_after_first:
                    raise RuntimeError("partially consumed")
            else:
                unhandled.append(item)
        return unhandled


def _second_handler(core, dispatcher):
    handler = _ProgressHandler(core, dispatcher)
    handler.action_type = "kv.test2"
    return handler


_SECOND_TYPE = HintType("kv.test2", "1.0", lambda i, p: _StubHint(i), _second_handler)


def _operation(name, action_type="kv.test"):
    return _HintProgressOp(("hint-test", name), frozenset(), action_type)


def test_remote_and_hint_handlers_consume_only_their_items_in_order(monkeypatch):
    kvcr, first, _ = _stub_kvcr(
        monkeypatch, handler_factory=_ProgressHandler, extra_types=(_SECOND_TYPE,)
    )
    second = kvcr._core._hints_dispatcher._handlers["kv.test2"]
    a, b, c = _operation("a"), _operation("b", "kv.test2"), _operation("c")
    for item in (a, _ProgressUpdate([], 7), b, c):
        kvcr._core._progress._completed.put(item)
    assert list(kvcr.poll_completed()) == []
    assert first.batches == [[a, b, c]]
    assert second.batches == [[b]]
    assert first.finished == [a, c]
    assert second.finished == [b]
    assert first.connected_counts == second.connected_counts == [7]
    d = _operation("d")
    kvcr._core._progress._completed.put(d)
    kvcr.poll_completed()
    assert first.batches[-1] == [d]
    assert second.batches[-1] == []
    kvcr.poll_completed()
    assert first.batches[-1] == second.batches[-1] == []
    assert first.finished == [a, c, d]
    assert second.finished == [b]


def test_progress_operation_completion_reaches_owning_handler(monkeypatch):
    kvcr, handler, _ = _stub_kvcr(monkeypatch, handler_factory=_ProgressHandler)
    item = _operation("async")
    kvcr._core._progress.submit(item)
    _poll_until(kvcr, lambda _: bool(handler.finished))
    assert handler.finished == [item]
    kvcr.poll_completed()
    assert handler.finished == [item]


@pytest.mark.parametrize("with_handlers", [False, True])
def test_unknown_progress_item_is_rejected_after_consumers(monkeypatch, with_handlers):
    kvcr, first, _ = _stub_kvcr(
        monkeypatch,
        handler_factory=_ProgressHandler if with_handlers else _StubHandler,
        extra_types=(_SECOND_TYPE,) if with_handlers else (),
    )
    known, unknown = _operation("known"), object()
    kvcr._core._progress._completed.put(unknown)
    if with_handlers:
        kvcr._core._progress._completed.put(known)
    with pytest.raises(TypeError, match="unsupported KVCR main item"):
        kvcr.poll_completed()
    if with_handlers:
        second = kvcr._core._hints_dispatcher._handlers["kv.test2"]
        assert first.finished == [known]
        assert second.batches == [[unknown]]


def test_partial_batch_failure_propagates_without_forwarding_or_replaying(
    monkeypatch, caplog
):
    kvcr, first, _ = _stub_kvcr(
        monkeypatch, handler_factory=_ProgressHandler, extra_types=(_SECOND_TYPE,)
    )
    second = kvcr._core._hints_dispatcher._handlers["kv.test2"]
    first.fail_after_first = True
    a, b, c = _operation("a"), _operation("b", "kv.test2"), _operation("c")
    with caplog.at_level(logging.WARNING, logger="kvcr.hints.dispatcher"):
        for batch in ((a, b), (c,)):
            for item in batch:
                kvcr._core._progress._completed.put(item)
            with pytest.raises(RuntimeError, match="partially consumed"):
                kvcr.poll_completed()
    assert first.finished == [a, c]
    assert second.batches == []
    warnings = [r for r in caplog.records if r.name == "kvcr.hints.dispatcher"]
    assert len(warnings) == 1 and warnings[0].exc_info is not None
    # No transaction/retry contract: the caller saw the failure. The next poll
    # must not replay an item whose side effects already occurred.
    first.fail_after_first = False
    kvcr.poll_completed()
    assert first.finished == [a, c]
    assert second.batches == [[]]


def test_progress_failure_takes_precedence_over_handler_error(monkeypatch, caplog):
    failure = RuntimeError("progress boom")

    class FailingHandler(_StubHandler):
        def poll_main(self, items):
            self.core._progress._failure = failure
            raise ValueError("handler boom")

    kvcr, _, results = _stub_kvcr(monkeypatch, FailingHandler)
    try:
        with caplog.at_level(logging.WARNING, logger="kvcr.hints.dispatcher"):
            with pytest.raises(RuntimeError, match="progress boom") as caught:
                kvcr.poll_completed()
        assert caught.value is failure
        assert not [r for r in caplog.records if r.name == "kvcr.hints.dispatcher"]
        assert results == []
    finally:
        kvcr._core._progress._failure = None
