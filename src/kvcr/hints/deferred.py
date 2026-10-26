# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Pending hint queues owned by handlers, independent of dispatcher scheduling."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from .parser import _Hint
from .results import HintResult, HintStatus

# Pending limits; constants until a production deferring handler measures them.
_MAX_QUEUED_HINTS = 1024
_HINT_TIMEOUT_MS = 5000


@dataclass(frozen=True)
class _DeferredHint:
    hint: _Hint
    request_id: str | None
    message_id: str | None
    submitted_at: float

    def result(self, status: HintStatus, reason: str | None = None) -> HintResult:
        return HintResult(
            action_id=self.hint.action_id,
            action_type=type(self.hint).ACTION_TYPE,
            status=status,
            reason=reason,
            request_id=self.request_id,
            message_id=self.message_id,
        )


class _DeferredHintQueue:
    """Bounded FIFO for a handler's not-yet-started work.

    ``defer`` runs on caller threads. The handler calls ``expire(now)`` on
    every poll, even without capacity, then ``pop`` only when ready to start
    work, within its own budget. Once popped, the handler owns the hint and
    its exactly-once COMPLETED/FAILED report, including failures and shutdown.

    ``report`` must not raise or re-enter this queue: QUEUED is reported
    under the queue lock, before publishing the item. No backend work runs
    under this lock. The clock must be monotonic.
    """

    def __init__(
        self, clock: Callable[[], float], report: Callable[[HintResult], None]
    ) -> None:
        self._clock = clock
        self._report = report
        self._pending: deque[_DeferredHint] = deque()
        self._lock = threading.Lock()
        self._closed = False

    def defer(
        self, hint: _Hint, request_id: str | None, message_id: str | None
    ) -> None:
        discarded = None
        with self._lock:
            # Stamp under the lock so FIFO order is also age order.
            item = _DeferredHint(hint, request_id, message_id, self._clock())
            if self._closed:
                self._report(item.result(HintStatus.REJECTED, "closed"))
                return
            if len(self._pending) >= _MAX_QUEUED_HINTS:
                discarded = self._pending.popleft()
            self._report(item.result(HintStatus.QUEUED))
            self._pending.append(item)
        if discarded is not None:
            self._report(discarded.result(HintStatus.DROPPED, "queue_full"))

    def expire(self, now: float) -> None:
        """Drop stale pending hints without starting work or spending budget."""
        stale_before = now - _HINT_TIMEOUT_MS / 1000
        expired = []
        with self._lock:
            while self._pending and self._pending[0].submitted_at < stale_before:
                expired.append(self._pending.popleft())
        for item in expired:
            self._report(item.result(HintStatus.DROPPED, "stale"))

    def pop(self) -> _DeferredHint | None:
        """Transfer the oldest pending hint to the handler, if any."""
        with self._lock:
            return self._pending.popleft() if self._pending else None

    def discard_pending(self) -> None:
        """Drop pending work before quiescence; leave active work alone.

        This does not close the queue: core shutdown may fail and be retried.
        The handler must still call ``close`` during final cleanup.
        """
        with self._lock:
            dropped = self._pending
            self._pending = deque()
        for item in dropped:
            self._report(item.result(HintStatus.DROPPED, "closed"))

    def close(self) -> None:
        """Reject new deferrals and settle all remaining pending hints."""
        with self._lock:
            self._closed = True
        self.discard_pending()
