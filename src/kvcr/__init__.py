# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Framework-neutral KV Cache Runner."""

from .api import (
    DURATION_METRIC,
    HINT_ACTIONS_METRIC,
    KVCR,
    PREFETCH_BLOCKS_METRIC,
    PREFETCH_QUEUED_METRIC,
    PREFETCH_RESERVED_METRIC,
    STATE_METRIC,
    TRANSFER_BLOCKS_METRIC,
    TRANSFER_BYTES_METRIC,
    KVCRBindings,
)
from .control_channels import (
    KVCRGuardProtocolError,
    KVCRMsgFramingError,
    KVCRServiceError,
    KVCRSocketError,
)
from .guard_protocol import (
    KVCRClient,
    KVCRPoolHold,
)
from .hint_parser import ROUTER_HINT_CAPABILITIES, ROUTER_HINT_KEY

__all__ = [
    "DURATION_METRIC",
    "HINT_ACTIONS_METRIC",
    "KVCR",
    "PREFETCH_BLOCKS_METRIC",
    "PREFETCH_QUEUED_METRIC",
    "PREFETCH_RESERVED_METRIC",
    "STATE_METRIC",
    "TRANSFER_BLOCKS_METRIC",
    "TRANSFER_BYTES_METRIC",
    "KVCRBindings",
    "KVCRClient",
    "KVCRMsgFramingError",
    "KVCRGuardProtocolError",
    "KVCRPoolHold",
    "KVCRServiceError",
    "ROUTER_HINT_CAPABILITIES",
    "ROUTER_HINT_KEY",
    "KVCRSocketError",
]
