# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Execute parsed hint actions.

Core maps each ``action_type`` to a handler. A handler runs cheap metadata
updates inline at submit time, without the state lock, and queues anything
that touches block records or starts transfers. Queued work is admitted from
``poll_completed`` under the state lock, within a per-poll key budget.
"""

from __future__ import annotations

import logging
import threading
from collections import Counter, deque
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from .hint_parser import _HintAction
from .types import BlockKey, CacheTier

if TYPE_CHECKING:
    from .core import _KVCRCore
    from .progress import _TransferRef

HINT_ACTIONS_METRIC = "kvcr_hint_actions"
PREFETCH_BLOCKS_METRIC = "kvcr_prefetch_blocks"
PREFETCH_QUEUED_METRIC = "kvcr_prefetch_queued_blocks"
PREFETCH_RESERVED_METRIC = "kvcr_prefetch_reserved_blocks"

logger = logging.getLogger(__name__)

# Submission and admission run on hot paths; warn once per integration issue.
_LOGGED_ACTION_ISSUES: set[str] = set()

_DESTINATION_LABELS = {
    CacheTier.FW_G1: "GPU",
    CacheTier.LOCAL_G2: "CPU",
    CacheTier.G3: "STORAGE",
}


def _warn_once(issue: str, message: str, *args: object) -> None:
    if issue in _LOGGED_ACTION_ISSUES:
        return
    _LOGGED_ACTION_ISSUES.add(issue)
    logger.warning(message, *args)


class _ActionHandler(Protocol):
    """Executes every parsed action of one ``action_type``."""

    action_type: str

    def submit(self, actions: Sequence[_HintAction], request_id: str | None) -> None:
        """Run inline work and queue deferred work; called without the state lock."""
        ...

    def admit(self, budget: int, now: float) -> int:
        """Admit queued work under the state lock; return the budget consumed."""
        ...

    def close(self) -> None: ...


@dataclass
class _QueuedPrefetch:
    """A CPU-destination fetch action waiting for admission."""

    source: str
    hashes: deque[int]
    submitted_at: float


class _FetchActionHandler:
    """``kv.fetch``: engine hints inline, KVCR DRAM prefetch deferred."""

    action_type = "kv.fetch"

    def __init__(self, core: _KVCRCore) -> None:
        self._core = core
        options = core._remote_fw_dram._options
        self._options = options
        self._queue: deque[_QueuedPrefetch] = deque()
        # Guards _queue and _queued_blocks: submit may run on any thread.
        self._queue_lock = threading.Lock()
        self._queued_blocks = 0
        # Main-thread state, touched only under the core state lock.
        self._reserved: Counter[str] = Counter()
        self._prefetch_fill_keys: set[BlockKey] = set()
        self._gauges_dirty = False

        layout = core._prefetch_layout
        self._pool_counts: Counter[str] = Counter(
            label.partition(":")[0] for label in layout or ()
        )
        local_dram = core._local_dram
        self._reserve_limits: dict[str, int] = {
            name: int(
                length // slot_size * options.prefetch_reserve_limit_percent // 100
            )
            for name, (_, length, slot_size) in (
                local_dram._pools.items() if local_dram is not None else ()
            )
        }
        adapter = core._remote_fw_dram._key_adapter
        self._keys_for_block_hash = getattr(adapter, "keys_for_block_hash", None)
        self._unsupported: str | None
        if local_dram is None:
            self._unsupported = "no local DRAM is configured"
        elif layout is None:
            self._unsupported = "prefetch_layout is not configured"
        elif self._keys_for_block_hash is None:
            self._unsupported = "the key adapter does not implement keys_for_block_hash"
        else:
            self._unsupported = None

    # Submission: caller thread, no state lock.

    def submit(self, actions: Sequence[_HintAction], request_id: str | None) -> None:
        # The first engine-destination source in action order wins; a request
        # hint holds one source, and a second submit_hint would poison it.
        engine_source: str | None = None
        engine_hashes: frozenset[int] = frozenset()
        for action in actions:
            destination = action.destination
            if destination is CacheTier.FW_G1:
                source = action.source_endpoint
                if engine_source is None:
                    engine_source = source
                if source == engine_source:
                    # Same-source actions aggregate: storing each would replace
                    # the request's hash set and lose earlier actions.
                    engine_hashes = engine_hashes.union(action.block_hashes)
                    accepted = True
                else:
                    _warn_once(
                        "multiple-engine-sources",
                        "Rejected kv.fetch action: multiple engine-destination "
                        "sources in one envelope",
                    )
                    accepted = False
            elif destination is CacheTier.LOCAL_G2:
                accepted = self._enqueue(action)
            else:
                _warn_once(
                    f"destination:{destination.value}",
                    "Rejected kv.fetch action: destination %s is not supported",
                    _DESTINATION_LABELS.get(destination, destination.value),
                )
                accepted = False
            self._count_action(destination, accepted)
        if engine_source is not None:
            self._core._remote_fw_dram.submit_hint(
                engine_source, engine_hashes, request_id
            )

    def _enqueue(self, action: _HintAction) -> bool:
        if self._unsupported is not None:
            _warn_once(
                f"unsupported:{self._unsupported}",
                "Rejected CPU-destination kv.fetch action: %s",
                self._unsupported,
            )
            return False
        count = len(action.block_hashes)
        with self._queue_lock:
            full = (
                self._queued_blocks + count > self._options.max_queued_prefetch_blocks
            )
            if not full:
                self._queue.append(
                    _QueuedPrefetch(
                        action.source_endpoint,
                        deque(action.block_hashes),
                        self._core._clock(),
                    )
                )
                self._queued_blocks += count
                self._gauges_dirty = True
        if full:
            self._count_blocks("dropped_queue_full", count)
            return False
        if self._options.eager_ctrl_connect:
            from .remote_fw_dram import _TargetMetadataRequest

            self._core._progress.submit(_TargetMetadataRequest(action.source_endpoint))
        return True

    # Admission: main thread, under the core state lock.

    def admit(self, budget: int, now: float) -> int:
        if not self._queue and not self._gauges_dirty:
            return 0
        timeout = self._options.prefetch_timeout_ms / 1000
        consumed = 0
        while consumed < budget:
            with self._queue_lock:
                if not self._queue:
                    break
                head = self._queue[0]
                if now - head.submitted_at > timeout:
                    self._queue.popleft()
                    stale = len(head.hashes)
                    self._queued_blocks -= stale
                    self._gauges_dirty = True
                else:
                    stale = 0
                    take = min(budget - consumed, len(head.hashes))
                    hashes = [head.hashes.popleft() for _ in range(take)]
                    self._queued_blocks -= take
                    self._gauges_dirty = True
                    if not head.hashes:
                        self._queue.popleft()
            if stale:
                self._count_blocks("dropped_stale", stale)
                continue
            consumed += take
            self._admit_hashes(head.source, hashes, now, now + timeout)
        self._publish_gauges()
        return consumed

    def _admit_hashes(
        self, source: str, hashes: Sequence[int], now: float, deadline: float
    ) -> None:
        core = self._core
        local_dram = core._local_dram
        layout = core._prefetch_layout
        keys_for_block_hash = self._keys_for_block_hash
        if local_dram is None or layout is None or keys_for_block_hash is None:
            raise RuntimeError("prefetch admitted without local DRAM support")
        decode = core._remote_fw_dram._key_adapter.decode
        pool_counts = self._pool_counts
        limits = self._reserve_limits
        pending: Counter[str] = Counter()
        outcomes: Counter[str] = Counter()
        candidates: dict[BlockKey, None] = {}
        for block_hash in hashes:
            # Isolate the integration's adapter: a failure drops this hash only.
            # Nothing has touched block records yet, so there is nothing to prune.
            try:
                resolved = [
                    (key, decode(key) == block_hash)
                    for key in keys_for_block_hash(block_hash)
                ]
            except Exception:
                if "key_resolution_failed" not in _LOGGED_ACTION_ISSUES:
                    _LOGGED_ACTION_ISSUES.add("key_resolution_failed")
                    logger.warning(
                        "Dropped prefetch hash: key adapter failed to resolve it",
                        exc_info=True,
                    )
                outcomes["dropped_key_mismatch"] += 1
                continue
            for key, matches in resolved:
                if key in candidates:
                    continue
                if not matches:
                    _warn_once(
                        "key_mismatch",
                        "Dropped prefetch key: keys_for_block_hash returned a key "
                        "that does not decode to its block hash",
                    )
                    outcomes["dropped_key_mismatch"] += 1
                elif core._is_local_resident(key):
                    outcomes["skipped_resident"] += 1
                elif any(
                    self._reserved[pool] + pending[pool] + count > limits.get(pool, 0)
                    for pool, count in pool_counts.items()
                ):
                    outcomes["dropped_reserve_limit"] += 1
                else:
                    pending.update(pool_counts)
                    candidates[key] = None

        if candidates:
            for key in candidates:
                core._block_record(key)
            destinations, eviction_pending = local_dram.reserve_fill(
                candidates,
                sources=dict.fromkeys(candidates, CacheTier.REMOTE_G2),
                required_local=False,
                deadline=deadline,
                layout=layout,
                allow_evict=self._options.prefetch_may_evict,
            )
            for key in candidates:
                if key in destinations:
                    continue
                if key in eviction_pending:
                    outcomes["dropped_eviction_pending"] += 1
                else:
                    outcomes["dropped_no_capacity"] += 1
                core._prune_block_record(key)
            if destinations:
                self._start_prefetch(source, destinations, deadline)
                outcomes["admitted"] += len(destinations)

        for outcome, count in outcomes.items():
            self._count_blocks(outcome, count)

    def _start_prefetch(
        self,
        source: str,
        destinations: dict[BlockKey, list[_TransferRef]],
        deadline: float,
    ) -> None:
        core = self._core
        # Released only by on_fill_settled. A quarantined pull never settles, so its
        # keys stay counted until close: those slots really are unavailable.
        for key in destinations:
            self._prefetch_fill_keys.add(key)
            self._reserved.update(self._pool_counts)
        self._gauges_dirty = True
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "KVCR_EVENT prefetch_admitted source=%s blocks=%d",
                source,
                len(destinations),
            )
        items = list(destinations.items())
        per_op = self._options.prefetch_blocks_per_op
        for start in range(0, len(items), per_op):
            chunk = dict(items[start : start + per_op])
            handle = core._next_fill_handle
            core._next_fill_handle -= 1
            if not core._remote_fw_dram.prefetch(
                source, chunk, deadline, op_handle=handle
            ):
                core._complete_local_dram_fill(chunk, success=False)

    def on_fill_settled(self, keys: Collection[BlockKey]) -> None:
        """Release prefetch reservations for fills leaving FILLING."""
        fill_keys = self._prefetch_fill_keys
        if not fill_keys:
            return
        for key in keys:
            if key in fill_keys:
                fill_keys.remove(key)
                self._reserved.subtract(self._pool_counts)
                self._gauges_dirty = True

    def close(self) -> None:
        with self._queue_lock:
            self._queue.clear()
            self._queued_blocks = 0
        self._reserved.clear()
        self._prefetch_fill_keys.clear()

    # Telemetry.

    def _count_action(self, destination: CacheTier, accepted: bool) -> None:
        stats = self._core._stats
        if stats is not None:
            stats.increase_counter(
                HINT_ACTIONS_METRIC,
                1,
                (
                    self.action_type,
                    _DESTINATION_LABELS.get(destination, destination.value),
                    "accepted" if accepted else "rejected",
                ),
            )

    def _count_blocks(self, outcome: str, count: int) -> None:
        stats = self._core._stats
        if stats is not None and count:
            stats.increase_counter(PREFETCH_BLOCKS_METRIC, count, (outcome,))

    def _publish_gauges(self) -> None:
        if not self._gauges_dirty:
            return
        self._gauges_dirty = False
        stats = self._core._stats
        if stats is None:
            return
        stats.set_gauge(PREFETCH_QUEUED_METRIC, self._queued_blocks, ())
        for pool in self._reserve_limits:
            stats.set_gauge(PREFETCH_RESERVED_METRIC, self._reserved[pool], (pool,))
