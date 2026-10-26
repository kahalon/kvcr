# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Per-hint result types, internal to ``kvcr.hints``."""

from enum import Enum

import msgspec


class HintStatus(Enum):
    # Inline hints are terminal at submit time: ACCEPTED or REJECTED. A
    # deferred hint is QUEUED, then exactly one of COMPLETED, FAILED, DROPPED.
    ACCEPTED = "ACCEPTED"  # inline work done
    REJECTED = "REJECTED"  # invalid, unsupported, handler_error, closed
    QUEUED = "QUEUED"  # deferred to poll_completed
    COMPLETED = "COMPLETED"  # deferred work finished
    FAILED = "FAILED"  # deferred work failed
    DROPPED = "DROPPED"  # never started


class HintResult(msgspec.Struct, frozen=True):
    action_id: str | None
    action_type: str
    status: HintStatus
    reason: str | None = None
    request_id: str | None = None
    message_id: str | None = None
