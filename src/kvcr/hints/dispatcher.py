# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Hint dispatch infrastructure.

``HintDispatcher`` owns the handler table, the deferred lane and the result
queue; core holds one dispatcher and delegates to it. Hint modules register
their handler with ``register_hint_handler`` (and their parser with
``parser.register_hint_parser``); the built-in modules are imported at the
bottom of this file so that importing the dispatcher registers them.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import msgspec

from ..config import TelemetryStats
from ..types import HintResult, HintStatus
from .parser import _Hint, hint_message_id, parse_hints

if TYPE_CHECKING:
    from ..core import _KVCRCore

HINTS_METRIC = "kvcr_hints"
HINT_RESULTS_DISCARDED_METRIC = "kvcr_hint_results_discarded"

logger = logging.getLogger(__name__)
# Handlers run on hot paths; warn once per (phase, action type) about errors.
_LOGGED_HANDLER_ERRORS: set[str] = set()


def _warn_handler_error(action_type: str, phase: str) -> None:
    """Log the active exception once per ``(phase, action_type)``."""
    issue = f"{phase}:{action_type}"
    if issue in _LOGGED_HANDLER_ERRORS:
        return
    _LOGGED_HANDLER_ERRORS.add(issue)
    logger.warning(
        "KV hint hint handler failed during %s; action_type=%s",
        phase,
        action_type,
        exc_info=True,
    )


class _HintHandler(Protocol):
    """Executes one action type.

    ``submit`` runs on the caller's thread without the state lock, once per
    hint, and must report exactly one result for it, directly or by deferring
    it through ``dispatcher.defer``. ``admit`` runs under the state lock;
    ``True`` means the handler consumed the deferred hint and owns its
    terminal report, ``False`` means it cannot take it now.
    """

    action_type: str

    def submit(
        self,
        hint: _Hint,
        request_id: str | None,
        message_id: str | None,
    ) -> None: ...

    def admit(
        self,
        hint: _Hint,
        request_id: str | None,
        message_id: str | None,
        now: float,
    ) -> bool: ...

    def close(self) -> None: ...


class _HintResults:
    """Thread-safe bounded queue of undrained hint results.

    ``limit == 0`` retains nothing: results are only counted. ``known_types``
    bounds the metric's ``action_type`` label; any other type is counted as
    ``"unknown"`` (the result itself keeps the original string).
    """

    def __init__(
        self,
        limit: int,
        stats_getter: Callable[[], TelemetryStats | None],
        known_types: Callable[[], Collection[str]] | None = None,
    ) -> None:
        self._limit = limit
        self._stats_getter = stats_getter
        self._known_types = known_types
        self._lock = threading.Lock()
        self._results: deque[HintResult] = deque()

    def report(self, result: HintResult) -> None:
        discarded = False
        if self._limit > 0:
            with self._lock:
                discarded = len(self._results) >= self._limit
                if discarded:
                    self._results.popleft()
                self._results.append(result)
        stats = self._stats_getter()
        if stats is not None:
            action_type = result.action_type
            if self._known_types is not None and action_type not in (
                self._known_types()
            ):
                action_type = "unknown"
            stats.increase_counter(HINTS_METRIC, 1, (action_type, result.status.value))
            if discarded:
                stats.increase_counter(HINT_RESULTS_DISCARDED_METRIC, 1, ())

    def drain(self) -> list[HintResult]:
        with self._lock:
            results = list(self._results)
            self._results.clear()
        return results


def _result(
    item: _DeferredHint, status: HintStatus, reason: str | None = None
) -> HintResult:
    return HintResult(
        action_id=item.hint.action_id,
        action_type=item.handler.action_type,
        status=status,
        reason=reason,
        request_id=item.request_id,
        message_id=item.message_id,
    )


@dataclass
class _DeferredHint:
    handler: _HintHandler
    hint: _Hint
    request_id: str | None
    message_id: str | None
    submitted_at: float


class _DeferredHints:
    """Bounded FIFO of deferred hints, admitted in ``poll_completed``.

    The lock guards only the deque; handlers are always called without it, and
    the only result reported under it is ``ACCEPTED`` at enqueue (the results
    lock is a leaf). Every enqueued item gets exactly one terminal report:
    from the handler once admitted, or ``FAILED``/``DROPPED`` from the lane.
    """

    def __init__(
        self,
        *,
        max_queued: int,
        per_poll: int,
        timeout_s: float,
        dispatcher: HintDispatcher,
    ) -> None:
        self._max_queued = max_queued
        self._per_poll = per_poll
        self._timeout_s = timeout_s
        self._dispatcher = dispatcher  # reports go through its report()
        self._lock = threading.Lock()
        self._items: deque[_DeferredHint] = deque()
        self._closed = False

    def queued(self) -> int:
        return len(self._items)

    def enqueue(self, item: _DeferredHint) -> bool:
        with self._lock:
            closed = self._closed
            accepted = not closed and len(self._items) < self._max_queued
            if accepted:
                # Report before the item is visible to admission, so ACCEPTED
                # always precedes its terminal result. The results lock is a
                # leaf, so taking it here cannot deadlock.
                self._dispatcher.report(_result(item, HintStatus.ACCEPTED))
                self._items.append(item)
        if not accepted:
            reason = "closed" if closed else "queue_full"
            self._dispatcher.report(_result(item, HintStatus.DROPPED, reason))
        return accepted

    def admit(self, now: float) -> int:
        """Admit up to ``per_poll`` items; the caller holds the state lock.

        Handler calls that consume an item (admitted or failed) count toward
        the budget; stale drops do not call the handler and are not counted.
        Returns the number of items the handler admitted.
        """
        if not self._items:
            return 0
        admitted = 0
        handled = 0
        while handled < self._per_poll:
            with self._lock:
                if not self._items:
                    break
                item = self._items[0]
                stale = now - item.submitted_at > self._timeout_s
                if stale:
                    self._items.popleft()
            if stale:
                self._dispatcher.report(_result(item, HintStatus.DROPPED, "stale"))
                continue
            try:
                taken = item.handler.admit(
                    item.hint, item.request_id, item.message_id, now
                )
            except Exception:
                self._pop(item)
                handled += 1
                _warn_handler_error(item.handler.action_type, "admission")
                self._dispatcher.report(
                    _result(item, HintStatus.FAILED, "handler_error")
                )
                continue
            if not taken:
                break
            self._pop(item)
            handled += 1
            admitted += 1
        return admitted

    def close(self) -> None:
        """Drop every queued item; later enqueues are dropped as ``closed``."""
        with self._lock:
            self._closed = True
            items = list(self._items)
            self._items.clear()
        for item in items:
            self._dispatcher.report(_result(item, HintStatus.DROPPED, "closed"))

    def _pop(self, item: _DeferredHint) -> None:
        # Only admission removes from the head, and it runs under the state
        # lock, so the head is still ``item``.
        with self._lock:
            head = self._items.popleft()
        assert head is item


_HintHandlerFactory = Callable[["_KVCRCore", "HintDispatcher"], _HintHandler]

# action_type -> handler factory. Hint modules register at import time; a
# dispatcher instantiates every registered handler for its core.
_HANDLER_FACTORIES: dict[str, _HintHandlerFactory] = {}


def register_hint_handler(action_type: str, factory: _HintHandlerFactory) -> None:
    """Register the handler for ``action_type``.

    ``factory(core, dispatcher)`` builds it: ``core`` for backend access,
    ``dispatcher`` to ``report`` results and ``defer`` hints.
    """
    _HANDLER_FACTORIES[action_type] = factory


class HintDispatcher:
    """Routes parsed hints to their handlers and tracks their results.

    Core owns one dispatcher and calls ``submit`` from the caller's thread,
    ``admit`` under its state lock from ``poll_completed``, ``telemetry_state``
    under that lock from ``get_stats``, and ``close`` under it before anything
    that can raise. Which handlers exist is decided here, by the registry,
    not by core.
    """

    def __init__(self, core: _KVCRCore) -> None:
        config = core.config
        if config.max_queued_hints < 0:
            raise ValueError("max_queued_hints must be non-negative")
        if config.hints_per_poll <= 0:
            raise ValueError("hints_per_poll must be positive")
        if config.hint_timeout_ms <= 0:
            raise ValueError("hint_timeout_ms must be positive")
        if config.max_hint_results < 0:
            raise ValueError("max_hint_results must be non-negative")
        self._core = core
        self._results = _HintResults(
            config.max_hint_results,
            lambda: core._stats,
            known_types=lambda: self._handlers.keys(),
        )
        self._handlers: dict[str, _HintHandler] = {
            action_type: factory(core, self)
            for action_type, factory in _HANDLER_FACTORIES.items()
        }
        self._deferred = _DeferredHints(
            max_queued=config.max_queued_hints,
            per_poll=config.hints_per_poll,
            timeout_s=config.hint_timeout_ms / 1000,
            dispatcher=self,
        )
        self._closed = False

    # Caller thread, no state lock.

    def submit(
        self,
        envelope: Mapping[str, object] | msgspec.Struct,
        request_id: str | None,
    ) -> None:
        """Parse an envelope, report its rejections, dispatch its hints."""
        self._core._progress.raise_if_failed()
        hints, rejected = parse_hints(envelope)
        for result in rejected:
            self.report(msgspec.structs.replace(result, request_id=request_id))
        self.submit_hints(hints, request_id, hint_message_id(envelope))

    def submit_hints(
        self,
        hints: Sequence[_Hint],
        request_id: str | None = None,
        message_id: str | None = None,
    ) -> None:
        """Dispatch parsed hints to their handlers, one at a time, in order.

        This is also the side-band entry point for hints that did not arrive
        in an envelope.
        """
        self._core._progress.raise_if_failed()
        for hint in hints:
            action_type = type(hint).ACTION_TYPE
            handler = self._handlers.get(action_type)
            if handler is not None:
                self._submit_to_handler(handler, hint, request_id, message_id)
            else:
                self.report(
                    HintResult(
                        action_id=hint.action_id,
                        action_type=action_type,
                        status=HintStatus.REJECTED,
                        reason="unsupported_action_type",
                        request_id=request_id,
                        message_id=message_id,
                    )
                )

    def _submit_to_handler(
        self,
        handler: _HintHandler,
        hint: _Hint,
        request_id: str | None,
        message_id: str | None,
    ) -> None:
        # Contract: a handler reports (or defers) the hint exactly once, or
        # raises before doing so. Inline hints are terminal at submit time,
        # so a raise means REJECTED.
        try:
            handler.submit(hint, request_id, message_id)
        except Exception:
            _warn_handler_error(handler.action_type, "submit")
            self.report(
                HintResult(
                    action_id=hint.action_id,
                    action_type=handler.action_type,
                    status=HintStatus.REJECTED,
                    reason="handler_error",
                    request_id=request_id,
                    message_id=message_id,
                )
            )

    def defer(
        self,
        handler: _HintHandler,
        hint: _Hint,
        request_id: str | None,
        message_id: str | None,
    ) -> bool:
        """Queue ``hint`` for admission in ``poll_completed``."""
        return self._deferred.enqueue(
            _DeferredHint(handler, hint, request_id, message_id, self._core._clock())
        )

    def report(self, result: HintResult) -> None:
        """Record a result on behalf of a handler."""
        self._results.report(result)

    def drain_results(self) -> list[HintResult]:
        return self._results.drain()

    # Main thread, under the state lock.

    def admit(self, now: float) -> int:
        return self._deferred.admit(now)

    def queued(self) -> int:
        return self._deferred.queued()

    def telemetry_state(self) -> dict[str, int]:
        """Gauges for ``kvcr_state``, sampled by ``get_stats`` under the lock."""
        return {"hints_queued": self._deferred.queued()}

    def close(self) -> None:
        """Drop queued hints and close handlers; idempotent."""
        if self._closed:
            return
        self._closed = True
        self._deferred.close()
        for handler in self._handlers.values():
            try:
                handler.close()
            except Exception:
                _warn_handler_error(handler.action_type, "close")


# Built-in hint modules. Importing them registers their parser and handler;
# the import sits last so the protocol above is defined when they load.
from .fetch import _FetchHintHandler  # noqa: E402

register_hint_handler(_FetchHintHandler.action_type, _FetchHintHandler)
