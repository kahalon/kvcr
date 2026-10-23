# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Hint dispatch infrastructure.

``HintDispatcher`` owns the hint type table and the deferred queues, and
reports every hint's result; core holds one dispatcher and delegates to it.
Each hint module defines a ``HintType`` (its parser and handler);
``_HINT_TYPES`` lists the supported ones.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import msgspec

from .fetch import KV_FETCH
from .parser import HintType, _Hint, hint_message_id, parse_hints
from .results import HintResult, HintStatus

if TYPE_CHECKING:
    from ..core import _KVCRCore

# The supported hint types. Every dispatcher builds one handler per entry.
# The order is sensitive: deferred hints are admitted queue by queue in this
# order, under one budget per poll, so an earlier type's backlog is served
# before a later type's. Fairness across queues is left until it is a problem.
_HINT_TYPES: tuple[HintType, ...] = (KV_FETCH,)

# Deferred queue limits; constants until config is needed.
_MAX_QUEUED_HINTS = 1024  # per handler queue
_HINTS_PER_ADMIT = 64  # per poll, across queues
_HINT_TIMEOUT_MS = 5000

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
        "KV hint handler failed during %s; action_type=%s",
        phase,
        action_type,
        exc_info=True,
    )


class _HintHandler(Protocol):
    """Executes one action type.

    ``submit`` runs on the caller's thread without the state lock, once per
    hint. It decides, per hint, whether to report exactly one result now or
    to defer the hint with ``dispatcher.defer``.

    A handler that defers also defines ``take(hint, request_id, message_id,
    now)`` (see ``_Take``); the dispatcher keeps a queue only for handlers
    that do. ``take`` runs on the main thread under the state lock, once per
    deferred hint taken off the handler's queue, oldest first, up to the
    per-poll budget. It cannot refuse: from then on the handler owns the hint
    and its terminal report (``COMPLETED`` or ``FAILED``), and a handler with
    a limit of its own holds the hint until it has room. ``close`` must
    report every hint it took but has not finished as ``FAILED`` with reason
    ``closed``.
    """

    action_type: str

    def submit(
        self,
        hint: _Hint,
        request_id: str | None,
        message_id: str | None,
    ) -> None: ...

    def close(self) -> None: ...


# A deferring handler's ``take``: (hint, request_id, message_id, now).
_Take = Callable[[_Hint, str | None, str | None, float], None]


@dataclass(frozen=True)
class _DeferredHint:
    hint: _Hint
    request_id: str | None
    message_id: str | None
    submitted_at: float


def _deferred_result(
    item: _DeferredHint, status: HintStatus, reason: str | None = None
) -> HintResult:
    return HintResult(
        action_id=item.hint.action_id,
        action_type=type(item.hint).ACTION_TYPE,
        status=status,
        reason=reason,
        request_id=item.request_id,
        message_id=item.message_id,
    )


class HintDispatcher:
    """Routes parsed hints to their handlers and reports their results.

    Core owns one dispatcher. It calls ``submit`` from the caller's thread;
    ``admit`` from ``poll_completed`` under its state lock; ``close_queue``
    under that lock at the start of close; and ``close`` under it once the
    progress loop is quiescent, with the other main-thread resources.
    """

    def __init__(self, core: _KVCRCore) -> None:
        self._core = core
        self._types = {hint_type.action_type: hint_type for hint_type in _HINT_TYPES}
        self._handlers: dict[str, _HintHandler] = {
            action_type: hint_type.handler_factory(core, self)
            for action_type, hint_type in self._types.items()
        }
        # One bounded FIFO of deferred hints per handler that defines take, in
        # table order. The queues lock guards only the deques; handlers are
        # never called while it is held.
        self._takes: dict[str, _Take] = {}
        self._queues: dict[str, deque[_DeferredHint]] = {}
        for action_type, handler in self._handlers.items():
            take = getattr(handler, "take", None)
            if callable(take):
                self._takes[action_type] = take
                self._queues[action_type] = deque()
        self._queues_lock = threading.Lock()
        self._admitting: int | None = None  # thread id while admit() runs
        self._closed = False

    # Caller thread, no state lock.

    def submit(
        self,
        envelope: Mapping[str, object] | msgspec.Struct,
        request_id: str | None,
    ) -> None:
        """Parse an envelope, report its rejections, dispatch its hints."""
        self._core._progress.raise_if_failed()
        hints, rejected = parse_hints(envelope, self._types)
        for result in rejected:
            self.report(msgspec.structs.replace(result, request_id=request_id))
        message_id = hint_message_id(envelope)
        for hint in hints:
            handler = self._handlers[type(hint).ACTION_TYPE]
            self._submit_to_handler(handler, hint, request_id, message_id)

    def _submit_to_handler(
        self,
        handler: _HintHandler,
        hint: _Hint,
        request_id: str | None,
        message_id: str | None,
    ) -> None:
        # Contract: a handler reports or defers the hint exactly once, or
        # raises before doing so. A raise means REJECTED, unless the raise is
        # the fatal progress-thread failure, which propagates to the caller as
        # it does without hints.
        try:
            handler.submit(hint, request_id, message_id)
        except Exception:
            self._core._progress.raise_if_failed()
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
    ) -> None:
        """Put ``hint`` on ``handler``'s queue and report it ``QUEUED``.

        Called from ``handler.submit`` instead of reporting. A full queue
        discards its oldest hint as ``DROPPED`` with reason ``queue_full``.
        """
        if self._admitting == threading.get_ident():
            # Re-deferring would loop within one poll and reset the hint's
            # age; a handler schedules among the hints it holds instead.
            raise RuntimeError("defer() called from within admit()")
        queue = self._queues.get(handler.action_type)
        if queue is None:
            raise TypeError(f"{handler.action_type} handler defines no take()")
        discarded: _DeferredHint | None = None
        with self._queues_lock:
            # Stamped under the lock, so a queue's order is its hints' age order.
            item = _DeferredHint(hint, request_id, message_id, self._core._clock())
            if len(queue) >= _MAX_QUEUED_HINTS:
                discarded = queue.popleft()
            # Report before the hint is visible to admission, so QUEUED
            # always precedes its terminal result.
            self.report(_deferred_result(item, HintStatus.QUEUED))
            queue.append(item)
        if discarded is not None:
            self.report(_deferred_result(discarded, HintStatus.DROPPED, "queue_full"))

    def report(self, result: HintResult) -> None:
        """Record a hint's result; called by handlers and by the dispatcher.

        Results are logged at debug level only; rejections and handler errors
        already warn once per issue. No metric is emitted: integrations
        forward KVCR stats and may reject names they do not declare.
        """
        logger.debug(
            "KV hint result; action_id=%s action_type=%s status=%s reason=%s",
            result.action_id,
            result.action_type,
            result.status.value,
            result.reason,
        )

    # Main thread, under the state lock.

    def admit(self, now: float) -> None:
        """Take queued hints into their handlers, up to ``_HINTS_PER_ADMIT``.

        Queues are visited in ``_HINT_TYPES`` order under the one budget. In
        each queue, stale hints are dropped from the head first, without
        calling the handler and without counting toward the budget; then
        hints are taken, oldest first, while budget remains.
        """
        if not any(self._queues.values()):  # idle check on every poll
            return
        self._admitting = threading.get_ident()
        try:
            budget = _HINTS_PER_ADMIT
            for action_type, queue in self._queues.items():
                budget = self._admit_queue(action_type, queue, now, budget)
        finally:
            self._admitting = None

    def _admit_queue(
        self,
        action_type: str,
        queue: deque[_DeferredHint],
        now: float,
        budget: int,
    ) -> int:
        """Drop stale heads, then take hints while budget remains.

        Returns the budget left. Every head access is under the queues lock,
        because ``defer`` on caller threads pops heads too, to discard the
        oldest hint of a full queue.
        """
        take = self._takes[action_type]
        stale_before = now - _HINT_TIMEOUT_MS / 1000
        while True:
            with self._queues_lock:
                if not queue or queue[0].submitted_at >= stale_before:
                    break
                item = queue.popleft()
            self.report(_deferred_result(item, HintStatus.DROPPED, "stale"))
        while budget and queue:
            with self._queues_lock:
                if not queue:
                    break
                item = queue.popleft()
            budget -= 1
            try:
                take(item.hint, item.request_id, item.message_id, now)
            except Exception:
                self._core._progress.raise_if_failed()
                _warn_handler_error(action_type, "take")
                self.report(_deferred_result(item, HintStatus.FAILED, "handler_error"))
        return budget

    def close_queue(self) -> None:
        """Drop every queued hint as ``DROPPED`` with reason ``closed``; idempotent."""
        with self._queues_lock:
            dropped = [item for queue in self._queues.values() for item in queue]
            for queue in self._queues.values():
                queue.clear()
        for item in dropped:
            self.report(_deferred_result(item, HintStatus.DROPPED, "closed"))

    def close(self) -> None:
        """Drop queued hints, then close handlers; idempotent."""
        if self._closed:
            return
        self._closed = True
        self.close_queue()
        for handler in self._handlers.values():
            try:
                handler.close()
            except Exception:
                _warn_handler_error(handler.action_type, "close")
