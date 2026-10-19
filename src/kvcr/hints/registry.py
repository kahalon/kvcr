# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The hint type registry, shared by the parser and the dispatcher.

A hint type is registered once, with its payload parser and its handler
factory, so the two cannot disagree. This module imports nothing from the
rest of the package at runtime; the parser, the dispatcher and the hint
modules all depend on it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core import _KVCRCore
    from .dispatcher import HintDispatcher, _HintHandler
    from .parser import _Hint

_HintParser = Callable[[str | None, Mapping[str, object]], "_Hint"]
_HintHandlerFactory = Callable[["_KVCRCore", "HintDispatcher"], "_HintHandler"]


@dataclass(frozen=True)
class HintType:
    action_type: str
    version: str
    parse: _HintParser
    handler_factory: _HintHandlerFactory


# action_type -> hint type. Hint modules register at import time; the parser
# looks types up per envelope and a dispatcher instantiates every registered
# handler for its core.
_HINT_TYPES: dict[str, HintType] = {}


def register_hint_type(
    action_type: str,
    version: str,
    parse: _HintParser,
    handler_factory: _HintHandlerFactory,
) -> None:
    """Register ``action_type`` at ``version``.

    ``parse(action_id, payload)`` returns the typed hint or raises
    ``ValueError(reason)``, which becomes the hint's rejection reason.
    ``handler_factory(core, dispatcher)`` builds the handler: ``core`` for
    backend access, ``dispatcher`` to ``report`` results. Hint modules call
    this once, at import time.
    """
    _HINT_TYPES[action_type] = HintType(action_type, version, parse, handler_factory)
