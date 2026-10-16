# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging

import msgspec
import pytest
from _kvcr_test_utils import _HintAction, _HintEnvelope, _router_hint

from kvcr.hint_parser import (
    _ACTION_PARSERS,
    _LOGGED_HINT_ISSUES,
    _KVFetchAction,
    _KVFetchHint,
    parse_hint_actions,
)
from kvcr.types import CacheTier


@pytest.mark.parametrize("as_mapping", [False, True])
def test_kv_fetch_hint_parses_first_fetch_action(as_mapping) -> None:
    hint = _router_hint("tcp://source:1234", (123, 456, 123))
    hint.actions.insert(0, _HintAction("ignored", "kv.future_action", "9.0", {}))
    hint.actions.extend(_router_hint("tcp://other:1234", (789,)).actions)

    hint = msgspec.to_builtins(hint) if as_mapping else hint
    payload = hint["actions"][1]["payload"] if as_mapping else hint.actions[1].payload
    parsed = _KVFetchHint.from_hint(hint)
    payload["source_control_endpoint"] = "tcp://changed:1"
    payload["block_hashes"].clear()
    assert parsed == _KVFetchHint(
        source_endpoint="tcp://source:1234",
        block_hashes=frozenset({123, 456}),
    )


def test_kv_fetch_hint_warns_once_per_mismatched_version_field(
    caplog: pytest.LogCaptureFixture,
) -> None:
    _LOGGED_HINT_ISSUES.clear()
    kv_hint = _router_hint("tcp://source:1234", (123,))
    with caplog.at_level(logging.WARNING):
        for protocol, version in (("9.0", "2.0"), ("10.0", "3.0")):
            changed = msgspec.structs.replace(
                kv_hint,
                protocol_version=protocol,
                actions=[
                    msgspec.structs.replace(kv_hint.actions[0], action_version=version)
                ],
            )
            parsed = _KVFetchHint.from_hint(changed)

    assert parsed == _KVFetchHint("tcp://source:1234", frozenset({123}))
    assert caplog.text.count("KV hint protocol_version mismatch") == 1
    assert caplog.text.count("kv.fetch action_version mismatch") == 1


@pytest.mark.parametrize(
    "kv_hint",
    [
        None,
        {},
        _HintEnvelope("0.1", "test", "bad"),
        _HintEnvelope(
            "0.1", "test", [None, _HintAction("a", "kv.fetch", "1.0", "bad")]
        ),
        _HintEnvelope("0.1", "test", [_HintAction("a", "kv.future_action", "1.0", {})]),
    ],
)
def test_kv_fetch_hint_rejects_invalid_envelope(kv_hint: object) -> None:
    with pytest.raises(ValueError, match="invalid router hint"):
        _KVFetchHint.from_hint(kv_hint)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "fetch_payload",
    [
        {"block_hashes": [1]},
        {"source_control_endpoint": "", "block_hashes": [1]},
        {"source_control_endpoint": "tcp://source:1", "block_hashes": []},
        {"source_control_endpoint": "tcp://source:1", "block_hashes": [True]},
        {"source_control_endpoint": "tcp://source:1", "block_hashes": [-1]},
        {"source_control_endpoint": "tcp://source:1", "block_hashes": [1 << 64]},
        {
            "source_control_endpoint": "tcp://source:1",
            "block_hashes": [1],
            "mode": "move",
        },
        {
            "source_control_endpoint": "tcp://source:1",
            "block_hashes": [1],
            "no_retain": True,
        },
    ],
)
def test_kv_fetch_hint_rejects_invalid_payload(fetch_payload: object) -> None:
    """Reject malformed payloads on a matching ``kv.fetch`` action."""
    kv_hint = _router_hint("tcp://source:1")
    kv_hint.actions[0] = msgspec.structs.replace(
        kv_hint.actions[0], payload=fetch_payload
    )

    with pytest.raises(ValueError):
        _KVFetchHint.from_hint(kv_hint)


def _fetch_action(action_id, source, hashes, **extra):
    payload = {"source_control_endpoint": source, "block_hashes": list(hashes), **extra}
    return _HintAction(action_id, "kv.fetch", "1.0", payload)


def test_parse_hint_actions_returns_every_valid_fetch_action():
    hint = _HintEnvelope(
        "0.1",
        "m",
        [
            _fetch_action("a", "tcp://a:1", (1, 2)),
            _HintAction("x", "kv.future_action", "1.0", {}),
            _fetch_action("bad", "", (3,)),
            _fetch_action("b", "tcp://b:1", (4,), destination={"medium": "cpu"}),
        ],
    )
    assert parse_hint_actions(hint) == [
        _KVFetchAction("a", "tcp://a:1", frozenset({1, 2}), CacheTier.FW_G1),
        _KVFetchAction("b", "tcp://b:1", frozenset({4}), CacheTier.LOCAL_G2),
    ]


@pytest.mark.parametrize(
    "destination, expected",
    [
        (None, CacheTier.FW_G1),
        ({"medium": "GPU"}, CacheTier.FW_G1),
        ({"medium": "Cpu", "locality": "local"}, CacheTier.LOCAL_G2),
    ],
)
def test_parse_destination_media(destination, expected):
    extra = {} if destination is None else {"destination": destination}
    hint = _HintEnvelope("0.1", "m", [_fetch_action("a", "tcp://a:1", (1,), **extra)])
    assert parse_hint_actions(hint)[0].destination is expected


@pytest.mark.parametrize(
    "destination",
    [
        {"medium": "STORAGE"},
        {"medium": "TAPE"},
        {"medium": "CPU", "locality": "REMOTE"},
        "CPU",
        {},
    ],
)
def test_parse_destination_rejects_action_only(destination):
    hint = _HintEnvelope(
        "0.1",
        "m",
        [
            _fetch_action("bad", "tcp://a:1", (1,), destination=destination),
            _fetch_action("ok", "tcp://a:1", (2,)),
        ],
    )
    assert [a.action_id for a in parse_hint_actions(hint)] == ["ok"]


def test_parse_hint_actions_raises_when_nothing_valid():
    hint = _HintEnvelope("0.1", "m", [_fetch_action("bad", "", (1,))])
    with pytest.raises(ValueError):
        parse_hint_actions(hint)


def test_registered_action_type_is_parsed(monkeypatch):
    class _Stub(msgspec.Struct, frozen=True):
        action_id: str | None
        ACTION_TYPE = "kv.test"

    monkeypatch.setitem(_ACTION_PARSERS, "kv.test", ("1.0", lambda i, p: _Stub(i)))
    hint = _HintEnvelope("0.1", "m", [_HintAction("s", "kv.test", "1.0", {})])
    assert parse_hint_actions(hint) == [_Stub("s")]
