# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Hint routing, result reporting, and handler lifecycle hooks."""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Mapping
from typing import TYPE_CHECKING, Protocol

import msgspec

from .fetch import KV_FETCH
from .parser import HintType, _Hint, hint_message_id, parse_hints
from .results import HintResult, HintStatus

if TYPE_CHECKING:
    from ..core import _KVCRCore

_HINT_TYPES: tuple[HintType, ...] = (KV_FETCH,)

logger = logging.getLogger(__name__)
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
    """Own one hint type from submission through settlement.

    ``submit`` runs without the state lock: report inline or queue with QUEUED
    before publication; raise only before either. Started hints get exactly
    one COMPLETED/FAILED result, including per-item failures.

    Optional ``poll_main(items)`` runs under the state lock after backends,
    even for empty batches. Return unhandled items in order; own clocks,
    expiry, capacity, and work budgets. Escaped errors propagate without replay.

    Both cleanup hooks run under the state lock. Optional ``discard_pending``
    drops queued work as DROPPED(closed) before quiescence, leaving shutdown
    retryable. ``close`` runs after quiescence: reject deferrals, drop queued
    work, and settle active hints as FAILED(closed).
    """

    action_type: str

    def submit(
        self,
        hint: _Hint,
        request_id: str | None,
        message_id: str | None,
    ) -> None: ...

    def close(self) -> None: ...


class HintDispatcher:
    """Route hints and invoke handlers; handlers own queues and active work."""

    def __init__(self, core: _KVCRCore) -> None:
        self._core = core
        self._types = {hint_type.action_type: hint_type for hint_type in _HINT_TYPES}
        self._handlers: dict[str, _HintHandler] = {
            action_type: hint_type.handler_factory(core, self)
            for action_type, hint_type in self._types.items()
        }
        self._polls: dict[str, Callable[[Collection[object]], list[object]]] = {}
        self._discard_pending: dict[str, Callable[[], None]] = {}
        for action_type, handler in self._handlers.items():
            poll = getattr(handler, "poll_main", None)
            if callable(poll):
                self._polls[action_type] = poll
            discard_pending = getattr(handler, "discard_pending", None)
            if callable(discard_pending):
                self._discard_pending[action_type] = discard_pending
        self._closed = False

    def submit(
        self,
        envelope: Mapping[str, object] | msgspec.Struct,
        request_id: str | None,
    ) -> None:
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

    def report(self, result: HintResult) -> None:
        """Log only; integrations may reject undeclared metrics."""
        logger.debug(
            "KV hint result; action_id=%s action_type=%s status=%s reason=%s",
            result.action_id,
            result.action_type,
            result.status.value,
            result.reason,
        )

    def poll_main(self, items: list[object]) -> list[object]:
        for action_type, poll in self._polls.items():
            try:
                items = poll(items)
            except Exception:
                self._core._progress.raise_if_failed()
                _warn_handler_error(action_type, "poll_main")
                raise
        return items

    def discard_pending(self) -> None:
        for action_type, discard_pending in self._discard_pending.items():
            try:
                discard_pending()
            except Exception:
                _warn_handler_error(action_type, "discard_pending")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for handler in self._handlers.values():
            try:
                handler.close()
            except Exception:
                _warn_handler_error(handler.action_type, "close")
