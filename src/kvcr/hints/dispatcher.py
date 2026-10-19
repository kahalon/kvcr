# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Hint dispatch infrastructure.

``HintDispatcher`` owns the handler table and reports every hint's result
as telemetry; core holds one dispatcher and delegates to it. Hint modules
register their handler with ``register_hint_handler`` (and their parser
with ``parser.register_hint_parser``); the built-in modules are imported at
the bottom of this file so that importing the dispatcher registers them.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol

import msgspec

from ..types import HintResult, HintStatus
from .parser import _Hint, hint_message_id, parse_hints

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


_HintHandlerFactory = Callable[["_KVCRCore", "HintDispatcher"], _HintHandler]

# action_type -> handler factory. Hint modules register at import time; a
# dispatcher instantiates every registered handler for its core.
_HANDLER_FACTORIES: dict[str, _HintHandlerFactory] = {}


def register_hint_handler(action_type: str, factory: _HintHandlerFactory) -> None:
    """Register the handler for ``action_type``.

    ``factory(core, dispatcher)`` builds it: ``core`` for backend access,
    ``dispatcher`` to ``report`` results.
    """
    _HANDLER_FACTORIES[action_type] = factory


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
            action_type: factory(core, self)
            for action_type, factory in _HANDLER_FACTORIES.items()
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


# Built-in hint modules. Importing them registers their parser and handler;
# the import sits last so the protocol above is defined when they load.
from .fetch import _FetchHintHandler  # noqa: E402

register_hint_handler(_FetchHintHandler.action_type, _FetchHintHandler)
