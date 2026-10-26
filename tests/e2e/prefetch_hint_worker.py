# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Start ``dynamo.vllm`` with a hook that adds a prefetch hint to a request.

The Dynamo KV router emits GPU-destination ``kv.fetch`` hints only. Until it
can ask a worker to prefetch, the test supplies the prefetch itself, as a spec::

    {"source_control_endpoint": "tcp://127.0.0.1:17771",
     "block_hashes": ["1234", "5678"]}

Hashes travel as strings so no JSON layer can round them. A spec reaches the
hook one of two ways: in the request, as ``nvext.kvcr_prefetch``, or as a
``*.prefetch.json`` file in the directory ``KVCR_E2E_PREFETCH_DIR`` names,
which the hook attaches to the next request it sees and renames to
``.attached``. The hook turns the spec into a CPU-destination ``kv.fetch``
action and appends it to the router's hint envelope for that request, or makes
an envelope of its own, so the target KVCR sees the fetch and the prefetch in
one envelope, the way a router would send them. Everything else is
``python -m dynamo.vllm``.
"""

import json
import logging
import os
import uuid
from pathlib import Path

if "PYTHONHASHSEED" not in os.environ:
    os.environ["PYTHONHASHSEED"] = "0"

# Under the kvcr logger hierarchy, so the test's logging config shows it.
logger = logging.getLogger("kvcr.e2e.prefetch")


def _spec_from_file():
    directory = os.environ.get("KVCR_E2E_PREFETCH_DIR")
    if not directory:
        return None
    pending = sorted(Path(directory).glob("*.prefetch.json"))
    if not pending:
        return None
    path = pending[0]
    spec = json.loads(path.read_text())
    path.rename(path.with_suffix(".attached"))
    return spec


def _install_prefetch_hook():
    from dynamo.vllm import handlers

    build_router_envelope = handlers._build_vllm_kv_hints

    def build_with_prefetch(request):
        envelope = build_router_envelope(request)
        spec = None
        for source in handlers._iter_nvext_sources(request):
            spec = source.get("kvcr_prefetch", spec)
        if spec is None:
            spec = _spec_from_file()
        if spec is None:
            return envelope
        action_type, envelope_type = handlers._vllm_kv_hints_types()
        action = action_type(
            action_id=str(spec.get("action_id") or f"prefetch-{uuid.uuid4().hex[:8]}"),
            action_type="kv.fetch",
            action_version="1.0",
            payload={
                "source_control_endpoint": spec["source_control_endpoint"],
                "block_hashes": [
                    int(block_hash) for block_hash in spec["block_hashes"]
                ],
                "destination": {"medium": "CPU"},
            },
        )
        if envelope is None:
            envelope = envelope_type(
                protocol_version="0.1",
                message_id=f"prefetch-{uuid.uuid4().hex}",
                actions=[],
            )
        logger.info(
            "kvcr_prefetch attached: action_id=%s hashes=%d router_actions=%d",
            action.action_id,
            len(action.payload["block_hashes"]),
            len(envelope.actions),
        )
        return envelope_type(
            protocol_version=envelope.protocol_version,
            message_id=envelope.message_id,
            actions=[*envelope.actions, action],
        )

    handlers._build_vllm_kv_hints = build_with_prefetch


if __name__ == "__main__":
    # Mirrors dynamo.vllm.__main__: the standby check runs before vLLM imports.
    from dynamo.common.snapshot.restore_context import maybe_run_restore_standby_mode

    maybe_run_restore_standby_mode()
    _install_prefetch_hook()
    from dynamo.vllm.main import main

    main()
