# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Parse KV hint envelopes for KVCR fetch metadata.

Hints are advisory request metadata. This module validates versioned KV hint
envelopes, parses every valid action through a per-action-type registry,
and extracts the fields KVCR consumes. Integration constants are exported by
``kvcr.__init__``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import ClassVar

import msgspec

from .types import CacheTier

# Public integration constants exported by kvcr.__init__.
ROUTER_HINT_KEY = "kv_hint"
ROUTER_HINT_CAPABILITIES = frozenset({"router_hint"})

# Envelope/action versions currently understood by the parser.
_KV_HINT_PROTOCOL_VERSION = "0.1"
_KV_FETCH_ACTION_TYPE = "kv.fetch"
_KV_FETCH_ACTION_VERSION = "1.0"

# Hint parsing runs on the request path; suppress repeated warnings for the same
# malformed integration issue.
logger = logging.getLogger(__name__)
_LOGGED_HINT_ISSUES: set[str] = set()


def _warn_version_mismatch(
    *,
    supported: object,
    received: object,
    issue: str,
    message: str,
) -> None:
    """Warn once when a supported schema version does not match received value."""
    if received == supported or issue in _LOGGED_HINT_ISSUES:
        return
    _LOGGED_HINT_ISSUES.add(issue)
    logger.warning(message, supported, received)


def _hint_field(value: Mapping[str, object] | msgspec.Struct, name: str) -> object:
    return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)


class _KVFetchAction(msgspec.Struct, frozen=True):
    """A parsed ``kv.fetch`` action."""

    ACTION_TYPE: ClassVar[str] = _KV_FETCH_ACTION_TYPE

    action_id: str | None
    source_endpoint: str
    block_hashes: frozenset[int]
    destination: CacheTier


# Widen to a Union when more action types exist.
_HintAction = _KVFetchAction

_DESTINATION_MEDIA = {"GPU": CacheTier.FW_G1, "CPU": CacheTier.LOCAL_G2}


def _parse_destination(payload: Mapping[str, object]) -> CacheTier:
    """Map the optional ``destination`` payload object to a cache tier."""
    if "destination" not in payload:
        return CacheTier.FW_G1
    destination = payload["destination"]
    if not isinstance(destination, Mapping):
        raise ValueError("invalid destination")
    medium = destination.get("medium")
    if not isinstance(medium, str):
        raise ValueError("invalid destination medium")
    medium = medium.upper()
    if medium == "STORAGE":
        raise ValueError("STORAGE destination is not currently supported")
    tier = _DESTINATION_MEDIA.get(medium)
    if tier is None:
        raise ValueError("invalid destination medium")
    locality = destination.get("locality")
    if locality is not None and (
        not isinstance(locality, str) or locality.upper() != "LOCAL"
    ):
        raise ValueError("only LOCAL destination locality is supported")
    return tier


def _parse_kv_fetch(
    action_id: str | None, fetch_payload: Mapping[str, object]
) -> _KVFetchAction:
    # TODO: Add mode and no_retain when KVCR starts consuming them.
    if "no_retain" in fetch_payload:
        raise ValueError("no_retain is not currently supported")

    mode = fetch_payload.get("mode", "copy")
    if mode != "copy":
        raise ValueError("only copy mode is currently supported")

    source = fetch_payload.get("source_control_endpoint")
    if source is None:
        raise ValueError("source-less hints are not currently supported")

    block_hashes = fetch_payload.get("block_hashes")
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

    return _KVFetchAction(
        action_id, source, frozenset(hashes), _parse_destination(fetch_payload)
    )


_ActionParser = Callable[[str | None, Mapping[str, object]], _HintAction]

# action_type -> (supported action_version, parser).
_ACTION_PARSERS: dict[str, tuple[str, _ActionParser]] = {
    _KV_FETCH_ACTION_TYPE: (_KV_FETCH_ACTION_VERSION, _parse_kv_fetch),
}


def register_action_parser(
    action_type: str, action_version: str, parse: _ActionParser
) -> None:
    """Register ``parse(action_id, payload)``; it raises ValueError to reject."""
    _ACTION_PARSERS[action_type] = (action_version, parse)


def parse_hint_actions(
    hint: Mapping[str, object] | msgspec.Struct,
) -> list[_HintAction]:
    """Parse every valid action in a KV hint envelope.

    Invalid or unknown actions are skipped; ``ValueError`` is raised when the
    envelope is malformed or no action is valid.
    """
    if not isinstance(hint, (Mapping, msgspec.Struct)):
        raise ValueError("invalid router hint")

    _warn_version_mismatch(
        supported=_KV_HINT_PROTOCOL_VERSION,
        received=_hint_field(hint, "protocol_version"),
        issue="protocol-version-mismatch",
        message="KV hint protocol_version mismatch; supported=%r received=%r",
    )

    actions = _hint_field(hint, "actions")
    if not isinstance(actions, list):
        raise ValueError("invalid router hint")

    parsed: list[_HintAction] = []
    for action in actions:
        if not isinstance(action, (Mapping, msgspec.Struct)):
            continue
        action_type = _hint_field(action, "action_type")
        entry = (
            _ACTION_PARSERS.get(action_type) if isinstance(action_type, str) else None
        )
        if entry is None:
            continue
        supported_version, parse = entry
        _warn_version_mismatch(
            supported=supported_version,
            received=_hint_field(action, "action_version"),
            issue=f"{action_type}-version-mismatch",
            message=(
                f"{action_type} action_version mismatch; supported=%r received=%r; "
                "processing with supported schema"
            ),
        )
        payload = _hint_field(action, "payload")
        if not isinstance(payload, Mapping):
            continue
        action_id = _hint_field(action, "action_id")
        try:
            parsed.append(
                parse(action_id if isinstance(action_id, str) else None, payload)
            )
        except ValueError as error:
            issue = f"{action_type}:{error}"
            if issue not in _LOGGED_HINT_ISSUES:
                _LOGGED_HINT_ISSUES.add(issue)
                logger.warning("Rejected %s hint action: %s", action_type, error)

    if not parsed:
        raise ValueError("invalid router hint")
    return parsed


class _KVFetchHint(msgspec.Struct, frozen=True):
    """Fields KVCR currently supports from the first ``FW_G1`` fetch action."""

    source_endpoint: str
    block_hashes: frozenset[int]

    @classmethod
    def from_hint(cls, hint: Mapping[str, object] | msgspec.Struct) -> _KVFetchHint:
        """Parse and validate a KV hint envelope."""
        for action in parse_hint_actions(hint):
            if (
                isinstance(action, _KVFetchAction)
                and action.destination is CacheTier.FW_G1
            ):
                return cls(action.source_endpoint, action.block_hashes)
        raise ValueError("invalid router hint")
