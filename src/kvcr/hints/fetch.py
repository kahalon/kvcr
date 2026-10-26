# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Engine-directed fetch hints and autonomous CPU-destination fills.

A ``kv.fetch`` with a missing or GPU destination stores a source for later
engine ``fetch`` or ``deliver`` calls. With ``destination={"medium": "CPU"}``,
it enters this handler's deferred queue and autonomously fills KVCR-owned
DRAM as the integration continues calling ``poll_completed``. Fetch hints
require a source control endpoint, ``mode=copy``, and no ``no_retain`` field.
Discarding the carrier request's hint does not cancel a CPU prefetch.

CPU prefetch resolves router hashes to complete, committed keys and their
layouts at the source. Several cache-group keys may share one hash. Both
peers must support the resolution protocol and its DRAM-only source
restriction: an eviction between resolution and transfer fails the affected
key, without framework-memory fallback. The destination reuses normal fetch
reservations, compatible in-flight fills, missing-piece completion, and
local-disk promotion. Policy admission is optional, unavailable capacity
fails immediately, and successful copies carry no public claim and remain
ordinarily evictable. Resolution uses ``operation_timeout_ms``; fetch then
receives its normal operation timeout and native abandonment grace.

Successful complete blocks publish normal ``LOCAL_G2`` inventory events and
clear remembered remote selections across requests. The next lookup and
delivery can use local DRAM while that copy remains resident; transfers
already started retain their original route. Router visibility remains
eventual. An admitted CPU hint reports ``COMPLETED`` for full fulfillment,
``COMPLETED`` with reason ``partial`` when some useful keys succeed, and
``FAILED`` when none succeed. Internal fetch completions do not enter the
engine's completion queue. Closing KVCR settles outstanding hints once.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, ClassVar, cast
from uuid import uuid4

import msgspec

from ..types import BlockKey, OpEntryResult
from .deferred import _DeferredHint, _DeferredHintQueue
from .parser import HintType, _Hint
from .results import HintResult, HintStatus

if TYPE_CHECKING:
    from ..core import _KVCRCore
    from ..remote_fw_dram import _ResolvedKeys
    from .dispatcher import HintDispatcher

_KV_FETCH_ACTION_TYPE = "kv.fetch"
_KV_FETCH_ACTION_VERSION = "1.0"
# Preserve the previous dispatcher admission budget, now owned by this handler.
_HINTS_PER_POLL = 64


class _KVFetchHint(msgspec.Struct, frozen=True):
    """A parsed ``kv.fetch`` hint."""

    ACTION_TYPE: ClassVar[str] = _KV_FETCH_ACTION_TYPE

    action_id: str | None
    source_endpoint: str
    block_hashes: frozenset[int]
    destination: str = "GPU"


def _parse_kv_fetch(
    action_id: str | None, payload: Mapping[str, object]
) -> _KVFetchHint:
    # TODO: Add mode and no_retain when KVCR starts consuming them.
    if "no_retain" in payload:
        raise ValueError("no_retain is not currently supported")

    mode = payload.get("mode", "copy")
    if mode != "copy":
        raise ValueError("only copy mode is currently supported")

    source = payload.get("source_control_endpoint")
    if source is None:
        raise ValueError("source-less hints are not currently supported")

    block_hashes = payload.get("block_hashes")
    if not isinstance(source, str) or not source or not isinstance(block_hashes, list):
        raise ValueError("invalid router hint")

    hashes: set[int] = set()
    for block_hash in block_hashes:
        if (
            isinstance(block_hash, bool)
            or not isinstance(block_hash, int)
            or not 0 <= block_hash < 1 << 64
        ):
            raise ValueError("invalid router hint")
        hashes.add(block_hash)
    if not hashes:
        raise ValueError("invalid router hint")

    destination = payload.get("destination")
    medium = "GPU"
    if destination is not None:
        if not isinstance(destination, Mapping):
            raise ValueError("invalid destination")
        medium = destination.get("medium")
        if medium not in ("CPU", "GPU"):
            raise ValueError("unsupported destination medium")
    return _KVFetchHint(action_id, source, frozenset(hashes), medium)


@dataclass
class _PrefetchJob:
    item: _DeferredHint
    private_id: str = field(default_factory=lambda: f"kvcr-prefetch:{uuid4()}")
    pending_groups: int = 0
    unresolved: bool = False
    results: dict[BlockKey, OpEntryResult] = field(default_factory=dict)


class _FetchHintHandler:
    """CPU hints defer; other hints store a source for engine-directed work."""

    action_type = _KVFetchHint.ACTION_TYPE

    def __init__(self, core: _KVCRCore, dispatcher: HintDispatcher) -> None:
        self._core = core  # backend access only
        self._dispatcher = dispatcher  # results
        self._jobs: dict[str, _PrefetchJob] = {}
        self._pending = _DeferredHintQueue(
            lambda: core._clock(), lambda result: dispatcher.report(result)
        )

    def submit(
        self,
        hint: _Hint,
        request_id: str | None,
        message_id: str | None,
    ) -> None:
        hint: _KVFetchHint = hint  # type: ignore[assignment]
        if hint.destination == "CPU":
            self._pending.defer(hint, request_id, message_id)
            return
        # One store call per hint; the store's own rules apply (a later hint
        # replaces the request's hashes, a different source fails the request).
        self._core._remote_fw_dram.submit_hint(
            hint.source_endpoint, hint.block_hashes, request_id
        )
        self._dispatcher.report(
            HintResult(
                action_id=hint.action_id,
                action_type=self.action_type,
                status=HintStatus.ACCEPTED,
                request_id=request_id,
                message_id=message_id,
            )
        )

    def _start(self, item: _DeferredHint, now: float) -> None:
        job = _PrefetchJob(item)
        hint = cast(_KVFetchHint, item.hint)
        self._jobs[job.private_id] = job
        core = self._core
        try:
            if core._local_dram is None:
                self._finish(job, HintStatus.FAILED, "no_local_dram")
                return
            remote = core._remote_fw_dram
            remote.submit_hint(hint.source_endpoint, hint.block_hashes, job.private_id)
            handle = core._next_fill_handle
            core._next_fill_handle -= 1
            remote.resolve_keys(
                hint.source_endpoint,
                sorted(hint.block_hashes),
                now + core.config.operation_timeout_ms / 1000,
                op_handle=handle,
                on_resolved=partial(self._resolved, job),
            )
        except Exception:
            core._progress.raise_if_failed()
            self._finish(job, HintStatus.FAILED, "handler_error")

    def poll_main(self, items: Collection[object]) -> list[object]:
        # The remote backend consumes resolution replies and invokes callbacks;
        # core fetch owns fill completion. Only pending admission needs polling.
        now = self._core._clock()
        self._pending.expire(now)
        for _ in range(_HINTS_PER_POLL):
            item = self._pending.pop()
            if item is None:
                break
            self._start(item, now)
        return list(items)

    def discard_pending(self) -> None:
        self._pending.discard_pending()

    def _resolved(self, job: _PrefetchJob, resolved: _ResolvedKeys | None) -> None:
        if job.private_id not in self._jobs:
            return
        if not resolved:
            self._finish(job, HintStatus.FAILED, "unresolved")
            return
        groups: dict[tuple[str, ...], list[BlockKey]] = {}
        hashes = set()
        for block_hash, keys, layout in resolved:
            hashes.add(block_hash)
            groups.setdefault(layout, []).extend(keys)
        hint = cast(_KVFetchHint, job.item.hint)
        job.unresolved = hashes != hint.block_hashes
        # Register every group before any can complete synchronously.
        job.pending_groups = len(groups)
        for layout, keys in groups.items():
            try:
                self._core.fetch(
                    keys,
                    job.private_id,
                    expected_layout=list(layout),
                    claim_on_ready=False,
                    on_complete=partial(self._fetched, job),
                )
            except Exception:
                from ..progress import _FAILED_RESULT

                self._fetched(job, dict.fromkeys(keys, _FAILED_RESULT))

    def _fetched(
        self, job: _PrefetchJob, entries: Mapping[BlockKey, OpEntryResult]
    ) -> None:
        if job.private_id not in self._jobs:
            return
        job.results.update(entries)
        # A complete local copy supersedes remembered selections in every request.
        local = {
            key
            for key, result in entries.items()
            if result.success and self._core._has_complete_local(key)
        }
        for selected in self._core._remote_delivery_keys.values():
            selected.difference_update(local)
        job.pending_groups -= 1
        if job.pending_groups:
            return
        successful = sum(result.success for result in job.results.values())
        if not successful:
            self._finish(job, HintStatus.FAILED, "fetch_failed")
        else:
            partial_result = job.unresolved or successful != len(job.results)
            self._finish(
                job, HintStatus.COMPLETED, "partial" if partial_result else None
            )

    def _finish(
        self, job: _PrefetchJob, status: HintStatus, reason: str | None = None
    ) -> None:
        if self._jobs.pop(job.private_id, None) is None:
            return
        self._core.discard_hint(job.private_id)
        self._dispatcher.report(job.item.result(status, reason))

    def close(self) -> None:
        self._pending.close()
        for job in list(self._jobs.values()):
            self._finish(job, HintStatus.FAILED, "closed")


KV_FETCH = HintType(
    _KV_FETCH_ACTION_TYPE,
    _KV_FETCH_ACTION_VERSION,
    _parse_kv_fetch,
    _FetchHintHandler,
)
