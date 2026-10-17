# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Hint dispatch tests."""

import logging
from typing import ClassVar

import msgspec
import pytest
from _kvcr_test_utils import (
    FakeBytesControl,
    FakeNixlAgent,
    FakePrimaryPinning,
    _HintAction,
    _HintEnvelope,
    _new_kvcr,
    _record_hint_results,
)

from kvcr.config import KVCRConfig
from kvcr.hints.dispatcher import (
    _HANDLER_FACTORIES,
    _LOGGED_HANDLER_ERRORS,
    HINTS_METRIC,
    HintDispatcher,
)
from kvcr.hints.parser import _HINT_PARSERS, _LOGGED_HINT_ISSUES
from kvcr.types import HintResult, HintStatus


@pytest.fixture(autouse=True)
def _reset_once_logs():
    # Once-per-issue warnings are process-global; reset so tests that count
    # warnings do not depend on run order.
    _LOGGED_HANDLER_ERRORS.clear()
    _LOGGED_HINT_ISSUES.clear()


# Dispatch infrastructure tests use stub action types; no production handler.


class _StubHint(msgspec.Struct, frozen=True):
    ACTION_TYPE: ClassVar[str] = "kv.test"

    action_id: str | None


class _UnhandledHint(msgspec.Struct, frozen=True):
    ACTION_TYPE: ClassVar[str] = "kv.unhandled"

    action_id: str | None


class _StubHandler:
    """Inline handler that accepts every hint it is given."""

    action_type = "kv.test"

    def __init__(self, core, dispatcher) -> None:
        self.core = core
        self.dispatcher = dispatcher
        self.submitted: list[_StubHint] = []
        self.close_calls = 0
        self.raise_on_submit = False

    def submit(self, hint, request_id, message_id) -> None:
        if self.raise_on_submit:
            raise RuntimeError("submit boom")
        self.submitted.append(hint)
        self.dispatcher.report(
            HintResult(
                action_id=hint.action_id,
                action_type=self.action_type,
                status=HintStatus.ACCEPTED,
                request_id=request_id,
                message_id=message_id,
            )
        )

    def close(self) -> None:
        self.close_calls += 1


def _envelope(*hints: tuple[str, str], message_id: str = "m1") -> _HintEnvelope:
    return _HintEnvelope(
        "0.1", message_id, [_HintAction(i, t, "1.0", {}) for i, t in hints]
    )


def _tests(*ids: str) -> _HintEnvelope:
    return _envelope(*((i, "kv.test") for i in ids))


def _stub_kvcr(monkeypatch, **config_fields):
    monkeypatch.setitem(_HINT_PARSERS, "kv.test", ("1.0", lambda i, p: _StubHint(i)))
    monkeypatch.setitem(
        _HINT_PARSERS, "kv.unhandled", ("1.0", lambda i, p: _UnhandledHint(i))
    )
    config = KVCRConfig(
        nixl_agent_name="target", pool_layouts=[("", 16)], **config_fields
    )
    # Handlers come from the dispatcher's registry, instantiated per core.
    monkeypatch.setitem(
        _HANDLER_FACTORIES,
        "kv.test",
        lambda core, dispatcher: _StubHandler(core, dispatcher),
    )
    kvcr = _new_kvcr(
        FakeNixlAgent(), FakePrimaryPinning(), FakeBytesControl(), config=config
    )
    handler = kvcr._core._hints_dispatcher._handlers["kv.test"]
    return kvcr, handler, _record_hint_results(monkeypatch, kvcr)


def _take(results) -> list[tuple[str | None, HintStatus, str | None]]:
    """Statuses reported since the previous take."""
    taken = [(r.action_id, r.status, r.reason) for r in results]
    results.clear()
    return taken


def test_dispatcher_owns_the_handler_registry(monkeypatch):
    # Core registers nothing itself: the built-in handler comes from the
    # dispatcher module, and a newly registered factory appears in a fresh
    # dispatcher without any core change.
    kvcr = _new_kvcr(FakeNixlAgent(), FakePrimaryPinning(), FakeBytesControl())
    assert set(kvcr._core._hints_dispatcher._handlers) == {"kv.fetch"}

    built_for: list[tuple[object, object]] = []

    def factory(core, dispatcher):
        built_for.append((core, dispatcher))
        return _StubHandler(core, dispatcher)

    monkeypatch.setitem(_HANDLER_FACTORIES, "kv.test", factory)
    dispatcher = HintDispatcher(kvcr._core)
    assert set(dispatcher._handlers) == {"kv.fetch", "kv.test"}
    assert built_for == [(kvcr._core, dispatcher)]


def test_submit_hints_routes_to_handler_and_rejects_unknown(monkeypatch):
    kvcr, handler, results = _stub_kvcr(monkeypatch)
    kvcr.submit_hint(
        _envelope(("a", "kv.test"), ("b", "kv.unhandled")), request_id="req"
    )
    assert handler.submitted == [_StubHint("a")]
    assert [(r.action_id, r.action_type, r.status, r.reason) for r in results] == [
        ("a", "kv.test", HintStatus.ACCEPTED, None),
        ("b", "kv.unhandled", HintStatus.REJECTED, "unsupported_action_type"),
    ]
    assert all((r.request_id, r.message_id) == ("req", "m1") for r in results)


def test_close_closes_handlers_once(monkeypatch):
    kvcr, handler, _ = _stub_kvcr(monkeypatch)
    kvcr.close()
    assert handler.close_calls == 1
    kvcr.close()
    assert handler.close_calls == 1


def test_handlers_close_even_when_close_raises(monkeypatch):
    kvcr, handler, _ = _stub_kvcr(monkeypatch)
    monkeypatch.setattr(kvcr._core, "is_quiescent", lambda: False)
    with pytest.raises(RuntimeError, match="not quiescent"):
        kvcr.close()
    assert handler.close_calls == 1
    monkeypatch.undo()
    kvcr.close()
    assert handler.close_calls == 1


def test_submit_handler_exception_is_isolated(monkeypatch, caplog):
    kvcr, handler, results = _stub_kvcr(monkeypatch)
    healthy = _StubHandler(kvcr._core, kvcr._core._hints_dispatcher)
    healthy.action_type = "kv.unhandled"
    handler.raise_on_submit = True
    with caplog.at_level(logging.WARNING, logger="kvcr.hints.dispatcher"):
        kvcr.submit_hint(_tests("a", "b"), request_id="req")
        kvcr.submit_hint(_tests("c"), request_id="req")
    assert [(r.action_id, r.action_type, r.status, r.reason) for r in results] == [
        ("a", "kv.test", HintStatus.REJECTED, "handler_error"),
        ("b", "kv.test", HintStatus.REJECTED, "handler_error"),
        ("c", "kv.test", HintStatus.REJECTED, "handler_error"),
    ]
    assert all(r.request_id == "req" and r.message_id == "m1" for r in results)
    warnings = [r for r in caplog.records if r.name == "kvcr.hints.dispatcher"]
    assert len(warnings) == 1
    assert warnings[0].exc_info is not None
    results.clear()

    kvcr._core._hints_dispatcher._handlers["kv.unhandled"] = healthy
    kvcr.submit_hint(_envelope(("d", "kv.unhandled")))
    assert _take(results) == [("d", HintStatus.ACCEPTED, None)]
    assert healthy.submitted == [_UnhandledHint("d")]


def test_progress_failure_propagates_from_submit(monkeypatch):
    kvcr, handler, results = _stub_kvcr(monkeypatch)
    kvcr._core._progress._failure = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        kvcr.submit_hint(_tests("a"), request_id="req")
    with pytest.raises(RuntimeError, match="boom"):
        kvcr._core._hints_dispatcher.submit_hints([_StubHint("b")], "req")
    assert handler.submitted == []
    assert results == []
    kvcr._core._progress._failure = None


def test_results_carry_request_and_message_ids(monkeypatch):
    kvcr, _, results = _stub_kvcr(monkeypatch)
    kvcr.submit_hint(
        _envelope(("x", "kv.missing"), ("a", "kv.test"), message_id="m7"),
        request_id="req",
    )
    assert [(r.action_id, r.status) for r in results] == [
        ("x", HintStatus.REJECTED),
        ("a", HintStatus.ACCEPTED),
    ]
    assert all(r.request_id == "req" and r.message_id == "m7" for r in results)


def test_every_result_is_counted_in_telemetry(monkeypatch):
    kvcr, _, results = _stub_kvcr(monkeypatch, enable_telemetry=True)
    kvcr.submit_hint(_envelope(("x", "kv.whatever-1234"), ("a", "kv.test")))
    # The result keeps the original type; the metric label is bounded to the
    # registered handlers, so an unknown type is counted as "unknown".
    assert [(r.action_type, r.status) for r in results] == [
        ("kv.whatever-1234", HintStatus.REJECTED),
        ("kv.test", HintStatus.ACCEPTED),
    ]
    counters = [
        labels
        for kind, name, _, labels in kvcr.get_stats().records
        if kind == "counter" and name == HINTS_METRIC
    ]
    assert counters == [("unknown", "REJECTED"), ("kv.test", "ACCEPTED")]


def test_results_are_telemetry_only_without_stats(monkeypatch):
    # No stats object: reporting is a no-op rather than an error.
    kvcr, _, results = _stub_kvcr(monkeypatch)
    assert kvcr._core._stats is None
    kvcr.submit_hint(_tests("a", "b"))
    assert _take(results) == [
        ("a", HintStatus.ACCEPTED, None),
        ("b", HintStatus.ACCEPTED, None),
    ]
