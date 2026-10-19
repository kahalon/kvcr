# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Hint dispatch infrastructure.

``HintDispatcher`` owns the handler table and reports every hint's result
as telemetry; core holds one dispatcher and delegates to it. Hint modules
register their parser and handler together with
``registry.register_hint_type``; the package imports the built-in modules, so
importing any of its modules registers them.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol

import msgspec

from ..types import HintResult, HintStatus
from .parser import _Hint, hint_message_id, parse_hints
from .registry import _HINT_TYPES

if TYPE_CHECKING:
    from ..core import _KVCRCore

HINTS_METRIC = "kvcr_hints"

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
    hint, and must report exactly one result for it.
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
    """Routes parsed hints to their handlers and reports their results.

    Core owns one dispatcher and calls ``submit`` from the caller's thread and
    ``close`` under its state lock once the progress loop is quiescent, with
    the other main-thread resources. Which handlers exist is decided here, by
    the registry, not by core.
    """

    def __init__(self, core: _KVCRCore) -> None:
        self._core = core
        self._handlers: dict[str, _HintHandler] = {
            action_type: hint_type.handler_factory(core, self)
            for action_type, hint_type in _HINT_TYPES.items()
        }
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
        # Contract: a handler reports the hint exactly once, or raises before
        # doing so. Inline hints are terminal at submit time, so a raise
        # means REJECTED, unless the raise is the fatal progress-thread
        # failure, which propagates to the caller as it does without hints.
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
        """Record a hint's result; called by handlers and by the dispatcher.

        Results are telemetry only. The metric's ``action_type`` label is
        bounded by the registered handlers; any other type is counted as
        ``"unknown"`` (the result itself keeps the original string).
        """
        stats = self._core._stats
        if stats is None:
            return
        action_type = result.action_type
        if action_type not in self._handlers:
            action_type = "unknown"
        stats.increase_counter(HINTS_METRIC, 1, (action_type, result.status.value))

    # Main thread, under the state lock.

    def close(self) -> None:
        """Close handlers; idempotent."""
        if self._closed:
            return
        self._closed = True
        for handler in self._handlers.values():
            try:
                handler.close()
            except Exception:
                _warn_handler_error(handler.action_type, "close")
