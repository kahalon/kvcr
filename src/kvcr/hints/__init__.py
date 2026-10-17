# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""KV hint handling: envelope parsing, per-type dispatch, and the hint handlers.

``parser`` turns a router envelope into typed hints, ``dispatcher`` routes each
hint to its handler inline or through the deferred lane, and one module per
hint type (``fetch`` today) implements the parser and the handler and
registers both at import time. Core holds one ``HintDispatcher`` and never
names a handler; adding a hint type is adding a module here and importing it
from ``dispatcher``.
"""

from .parser import ROUTER_HINT_CAPABILITIES, ROUTER_HINT_KEY

__all__ = ["ROUTER_HINT_CAPABILITIES", "ROUTER_HINT_KEY"]
