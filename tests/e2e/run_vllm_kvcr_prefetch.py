# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-rank check of a CPU-destination ``kv.fetch``: prefetch while a GPU fetch runs.

Built on ``run_vllm_kvcr.py``: the same Dynamo frontend and two-rank vLLM
worker, started through ``prefetch_hint_worker.py``, with KVCR local DRAM on
both ranks.

1. Seed two prefixes on rank 0, A and B. vLLM offloads their blocks to its CPU
   tier and cascades them into rank 0's KVCR DRAM, which indexes them by router
   hash. B's hashes are read off rank 0's KV event stream.
2. Send A to rank 1 with a prefetch of B attached by the worker hook (the spec
   is handed over as a file, or in ``nvext`` with ``--hint-channel nvext``).
   The router's ``kv.fetch`` for A (GPU destination) and the test's ``kv.fetch``
   for B (CPU destination) travel in one envelope, so A is delivered into rank
   1's GPU while B is pulled into rank 1's KVCR DRAM.
3. Wait for B's blocks to appear in rank 1's KVCR inventory, then send B to
   rank 1 with the exact prompt rank 0 answered. The completion must match
   rank 0's token for token with logprobs within tolerance, the prompt must be
   served from cache, and no peer delivery may happen for it.
"""

import argparse
import contextlib
import copy
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_vllm_kvcr as base  # noqa: E402

BLOCK_SIZE = base.BLOCK_SIZE
G2_SLOTS = 512  # KVCR DRAM slots per rank; a 2.7k-token prefix is 42 blocks
VERIFY_TOKENS = 32
LOGPROB_TOLERANCE = 0.1
SOURCE_CONTROL_ENDPOINT = "tcp://127.0.0.1:17771"  # rank 0, see TRANSFER_CONFIG
EVENT_ENDPOINTS = ("tcp://127.0.0.1:20081", "tcp://127.0.0.1:20082")  # per rank

METRIC = re.compile(
    r"kvcr_(duration_seconds|transfer_blocks|transfer_bytes):"
    r"\(([^)]*)\)(_count|_sum)?=([0-9.eE+-]+)"
)
HINT_RESULT = re.compile(
    r"KV hint result; action_id=(\S+) action_type=\S+ status=(\S+) reason=(\S+)"
)
TRANSFER_STARTED = re.compile(
    r"KVCR_EVENT target_transfer_started scope=(\w+) request_id=(\S+) op=\S+ "
    r"source=\S+ blocks=(\d+)"
)


def transfer_config():
    config = copy.deepcopy(base.TRANSFER_CONFIG)
    tier = config["kv_connector_extra_config"]["secondary_tiers"][0]
    tier["secondary_g2_slots"] = G2_SLOTS
    return config


def logging_config():
    """vLLM's default logging, plus the ``kvcr`` loggers at DEBUG.

    KVCR reports hint results and prefetch settlement at debug level only.
    """
    try:
        from vllm.logger import _DATE_FORMAT, _FORMAT
    except ImportError:  # pragma: no cover - older vLLM
        _FORMAT = "%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s"
        _DATE_FORMAT = "%m-%d %H:%M:%S"
    logger = {"handlers": ["vllm"], "level": "DEBUG", "propagate": False}
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "vllm": {
                "()": "vllm.logging_utils.NewLineFormatter",
                "datefmt": _DATE_FORMAT,
                "format": _FORMAT,
                "log_level": "DEBUG",
            }
        },
        "handlers": {
            "vllm": {
                "class": "logging.StreamHandler",
                "formatter": "vllm",
                "level": "DEBUG",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {"vllm": dict(logger), "kvcr": dict(logger)},
    }


class EventTap:
    """Follows one rank's KV event stream and keeps the stored hashes per owner.

    KVCR publishes its DRAM inventory as ``BlockStored`` / ``BlockRemoved``
    events with ``ownership="kvcr"``; vLLM's own tiers publish without it.
    """

    def __init__(self, endpoint, topic="kv-events"):
        import msgspec
        import zmq
        from vllm.distributed.kv_events import (
            AllBlocksCleared,
            BlockRemoved,
            BlockStored,
            KVEventBatch,
        )

        self._types = (BlockStored, BlockRemoved, AllBlocksCleared)
        self._decoder = msgspec.msgpack.Decoder(KVEventBatch)
        self._stored: dict[tuple[str | None, str | None], set[int]] = {}
        self._lock = threading.Lock()
        self._socket = zmq.Context.instance().socket(zmq.SUB)
        self._socket.connect(endpoint)
        self._socket.setsockopt(zmq.SUBSCRIBE, topic.encode())
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        stored, removed, cleared = self._types
        while not self._stop.is_set():
            if not self._socket.poll(200):
                continue
            _topic, _seq, payload = self._socket.recv_multipart()
            batch = self._decoder.decode(payload)
            with self._lock:
                for event in batch.events:
                    if isinstance(event, stored):
                        key = (event.ownership, event.medium)
                        self._stored.setdefault(key, set()).update(event.block_hashes)
                    elif isinstance(event, removed):
                        key = (event.ownership, event.medium)
                        self._stored.get(key, set()).difference_update(
                            event.block_hashes
                        )
                    elif isinstance(event, cleared):
                        # The engine's own GPU cache; KVCR inventory is unaffected.
                        self._stored.get((None, None), set()).clear()

    def stored(self, ownership="kvcr"):
        with self._lock:
            return set().union(
                *(
                    hashes
                    for (owner, _), hashes in self._stored.items()
                    if owner == ownership
                )
            )

    def summary(self):
        with self._lock:
            return {
                f"{owner}/{medium}": len(hashes)
                for (owner, medium), hashes in self._stored.items()
            }

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2)
        self._socket.close(linger=0)


def complete(
    base_url, model, prompt, worker_id, rank, *, max_tokens=8, logprobs=None, nvext=None
):
    headers = {
        "x-dynamo-worker-instance-id": str(worker_id),
        "x-dynamo-dp-rank": str(rank),
    }
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "nvext": {"extra_fields": ["worker_id"], **(nvext or {})},
    }
    if logprobs is not None:
        payload["logprobs"] = logprobs
    started = time.perf_counter()
    response = base.request_json(base_url + "/v1/completions", payload, headers)
    elapsed = time.perf_counter() - started
    if not response.get("choices"):
        raise RuntimeError(f"Completion has no choices: {response}")
    identity = response["nvext"]["worker_id"]
    if (identity["decode_worker_id"], identity["decode_dp_rank"]) != (worker_id, rank):
        raise RuntimeError(f"Request reached the wrong worker/rank: {identity}")
    choice = response["choices"][0]
    usage = response["usage"]
    result = {
        "rank": rank,
        "elapsed_s": elapsed,
        "prompt_tokens": usage["prompt_tokens"],
        "cached_tokens": usage["prompt_tokens_details"]["cached_tokens"],
        "text": choice["text"],
        "tokens": (choice.get("logprobs") or {}).get("tokens"),
        "token_logprobs": (choice.get("logprobs") or {}).get("token_logprobs"),
        "response": response,
    }
    print(
        f"completion: rank={rank} prompt_tokens={result['prompt_tokens']} "
        f"cached_tokens={result['cached_tokens']} elapsed={elapsed * 1e3:.0f}ms",
        flush=True,
    )
    return result


def kvcr_metrics(worker_log):
    """Sum every KVCR metric sample in the worker log, keyed by family and labels."""
    totals = {}
    for family, labels, suffix, value in METRIC.findall(
        worker_log.read_text(errors="replace")
    ):
        names = tuple(
            part.strip().strip("'\"") for part in labels.split(",") if part.strip()
        )
        key = f"{family}:{'/'.join(names)}{suffix}"
        totals[key] = totals.get(key, 0.0) + float(value)
    return totals


def hint_results(worker_log, action_id):
    text = worker_log.read_text(errors="replace")
    return [
        (status.rsplit(".", 1)[-1], None if reason == "None" else reason)
        for found, status, reason in HINT_RESULT.findall(text)
        if found == action_id
    ]


def target_transfers(worker_log):
    """Every pull a target started, as ``(scope, request_id, blocks)``, oldest first."""
    return [
        (scope, request_id, int(blocks))
        for scope, request_id, blocks in TRANSFER_STARTED.findall(
            worker_log.read_text(errors="replace")
        )
    ]


def run(args):
    import torch

    if torch.cuda.device_count() < 2:
        raise RuntimeError("This test requires two visible CUDA GPUs")

    output = args.output_dir
    worker_log = output / "worker.log"
    processes = []
    with contextlib.ExitStack() as resources:
        discovery = resources.enter_context(
            tempfile.TemporaryDirectory(prefix="kvcr-e2e-")
        )
        logging_path = output / "logging.json"
        base.write_json(logging_path, logging_config())
        env = {
            **base.serving_env(discovery),
            "VLLM_LOGGING_CONFIG_PATH": str(logging_path),
            # The worker hook picks up *.prefetch.json specs written here.
            "KVCR_E2E_PREFETCH_DIR": str(output),
        }

        def start(name, command):
            log = resources.enter_context((output / f"{name}.log").open("wb"))
            process = subprocess.Popen(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append((name, process))
            resources.callback(base.stop_process, process)
            print(f"Started {name} (PID {process.pid})", flush=True)

        start(
            "frontend",
            [
                sys.executable,
                "-m",
                "dynamo.frontend",
                "--router-mode",
                "kv",
                "--http-port",
                str(args.port),
            ],
        )
        start(
            "worker",
            [
                sys.executable,
                str(Path(__file__).with_name("prefetch_hint_worker.py")),
                "--model", args.model,
                "--tensor-parallel-size", "1",
                "--data-parallel-size", "2",
                "--block-size", str(BLOCK_SIZE),
                "--max-model-len", "8192",
                "--enable-prefix-caching",
                "--disable-hybrid-kv-cache-manager",
                "--enforce-eager",
                "--gpu-memory-utilization", "0.5",
                "--kv-transfer-config", json.dumps(transfer_config()),
                "--kv-events-config", json.dumps(base.EVENTS_CONFIG),
            ],
        )  # fmt: skip
        base_url = f"http://127.0.0.1:{args.port}"

        def model_ready():
            try:
                models = base.request_json(base_url + "/v1/models")
                return any(item["id"] == args.model for item in models.get("data", []))
            except (urllib.error.URLError, TimeoutError):
                return False

        print("Waiting for model registration and engine startup", flush=True)
        base.wait_until(
            model_ready, args.startup_timeout, processes, "model registration"
        )
        taps = [EventTap(endpoint) for endpoint in EVENT_ENDPOINTS]
        for tap in taps:
            resources.callback(tap.close)

        response = base.request_json(
            base_url + "/v1/completions",
            {
                "model": args.model,
                "prompt": f"Worker identity probe {uuid.uuid4().hex}",
                "max_tokens": 1,
                "nvext": {"extra_fields": ["worker_id"]},
            },
        )
        worker_id = response["nvext"]["worker_id"]["decode_worker_id"]

        # Fresh prefixes make a target-local hit impossible. A is the carrier
        # request's prefix, fetched into rank 1's GPU; B is prefetched into
        # rank 1's KVCR DRAM.
        prefix_a, prefix_b = base.make_prefix(), base.make_prefix()
        seed_a = complete(base_url, args.model, prefix_a + " seed source", worker_id, 0)
        if seed_a["cached_tokens"]:
            raise RuntimeError("Fresh seed unexpectedly reported cached tokens")
        blocks_a = seed_a["prompt_tokens"] // BLOCK_SIZE
        print("Waiting for A's blocks to reach rank 0's KVCR DRAM", flush=True)
        base.wait_until(
            lambda: len(taps[0].stored()) >= blocks_a or None,
            args.offload_wait * 4,
            processes,
            f"{blocks_a} KVCR-owned blocks on rank 0 "
            f"(events seen: {taps[0].summary()})",
        )
        before_b = taps[0].stored()

        verify_prompt = prefix_b + " verify"
        reference = complete(
            base_url, args.model, verify_prompt, worker_id, 0,
            max_tokens=VERIFY_TOKENS, logprobs=1,
        )  # fmt: skip
        base.write_json(output / "reference.json", reference["response"])
        blocks_b = reference["prompt_tokens"] // BLOCK_SIZE
        base.wait_until(
            lambda: len(taps[0].stored() - before_b) >= blocks_b or None,
            args.offload_wait * 4,
            processes,
            f"{blocks_b} KVCR-owned blocks of B on rank 0",
        )
        hashes_b = sorted(taps[0].stored() - before_b)
        print(f"B: {len(hashes_b)} blocks indexed on rank 0", flush=True)
        if hashes_b and taps[1].stored() & set(hashes_b):
            raise RuntimeError("B's blocks are already on rank 1")

        print("Waiting for router indexing", flush=True)
        time.sleep(args.offload_wait)

        # The carrier: A fetched into rank 1's GPU by the router's hint, B
        # prefetched into rank 1's KVCR DRAM by the test's, in one envelope.
        prefetch_id = f"prefetch-{uuid.uuid4().hex[:8]}"
        spec = {
            "action_id": prefetch_id,
            "source_control_endpoint": SOURCE_CONTROL_ENDPOINT,
            "block_hashes": [str(block_hash) for block_hash in hashes_b],
        }
        nvext = None
        if args.hint_channel == "nvext":
            nvext = {"kvcr_prefetch": spec}
        else:
            # No request is in flight, so the hook attaches it to the carrier.
            base.write_json(output / f"{prefetch_id}.prefetch.json", spec)
        carrier_started = time.perf_counter()
        carrier = complete(
            base_url, args.model, prefix_a + " retrieve on target", worker_id, 1,
            nvext=nvext,
        )  # fmt: skip
        base.write_json(output / "carrier.json", carrier["response"])
        if carrier["cached_tokens"] <= 0 or carrier["cached_tokens"] % BLOCK_SIZE:
            raise RuntimeError(
                "Expected block-aligned cached tokens for A, "
                f"got {carrier['cached_tokens']}"
            )
        base.wait_until(
            lambda: (
                f"kvcr_prefetch attached: action_id={prefetch_id}"
                in worker_log.read_text(errors="replace")
                or None
            ),
            30,
            processes,
            "the prefetch spec to reach the worker hook",
        )
        base.wait_until(
            lambda: set(hashes_b) <= taps[1].stored() or None,
            args.prefetch_timeout,
            processes,
            f"B's {len(hashes_b)} blocks in rank 1's KVCR inventory "
            f"(seen: {taps[1].summary()})",
        )
        landed_after_s = time.perf_counter() - carrier_started
        print(
            f"B landed on rank 1 {landed_after_s * 1e3:.0f}ms "
            "after the carrier was sent",
            flush=True,
        )

        results = base.wait_until(
            lambda: (
                (found if any(status != "QUEUED" for status, _ in found) else None)
                if (found := hint_results(worker_log, prefetch_id))
                else None
            ),
            30,
            processes,
            "the prefetch hint's terminal result",
        )
        if results != [("QUEUED", None), ("COMPLETED", None)]:
            raise RuntimeError(f"Unexpected hint results for the prefetch: {results}")
        transfers_before_verify = target_transfers(worker_log)

        verify = complete(
            base_url, args.model, verify_prompt, worker_id, 1,
            max_tokens=VERIFY_TOKENS, logprobs=1,
        )  # fmt: skip
        base.write_json(output / "verify.json", verify["response"])
        if verify["cached_tokens"] < len(hashes_b) * BLOCK_SIZE:
            raise RuntimeError(
                f"B was not served from the prefetched blocks: cached_tokens="
                f"{verify['cached_tokens']} < {len(hashes_b) * BLOCK_SIZE}"
            )
        if verify["text"] != reference["text"]:
            raise RuntimeError(
                "Completion from prefetched KV differs from rank 0's:\n"
                f"  rank 0: {reference['text']!r}\n  rank 1: {verify['text']!r}"
            )
        logprob_diff = None
        if reference["token_logprobs"] and verify["token_logprobs"]:
            logprob_diff = max(
                abs(a - b)
                for a, b in zip(
                    reference["token_logprobs"], verify["token_logprobs"], strict=True
                )
            )
            if logprob_diff > LOGPROB_TOLERANCE:
                raise RuntimeError(
                    f"Logprobs from prefetched KV drift by {logprob_diff:.4f} "
                    f"(tolerance {LOGPROB_TOLERANCE})"
                )
        print(
            f"verify: text matches rank 0; max |logprob diff|="
            f"{'n/a' if logprob_diff is None else f'{logprob_diff:.5f}'}",
            flush=True,
        )

        # Every pull rank 1 started: the carrier's peer delivery of A and the
        # prefetch of B, and nothing for B's request, which rank 1 served from
        # its KVCR DRAM. Counted from events, not the periodic telemetry line,
        # so the result does not depend on when that line was last printed.
        transfers = target_transfers(worker_log)
        if transfers != transfers_before_verify:
            raise RuntimeError("Verification triggered an additional peer transfer")
        deliveries = [b for scope, _, b in transfers if scope == "remote_deliver"]
        # Prefetch uses a private identity, independent of the carrier request.
        prefetches = [
            b
            for scope, rid, b in transfers
            if scope == "remote_fetch" and rid.startswith("kvcr-prefetch:")
        ]
        if deliveries != [blocks_a] or prefetches != [len(hashes_b)]:
            raise RuntimeError(
                f"Unexpected pulls on rank 1: peer deliveries of {deliveries} "
                f"blocks and prefetches of {prefetches} blocks; expected one "
                f"delivery of {blocks_a} (the carrier) and one prefetch of "
                f"{len(hashes_b)}, with nothing pulled for B's request"
            )

        # The telemetry line is periodic; give it time to include the prefetch.
        def prefetch_reported():
            metrics = kvcr_metrics(worker_log)
            return metrics if metrics.get("transfer_blocks:remote_fetch") else None

        metrics = base.wait_until(prefetch_reported, 60, processes, "KVCR telemetry")
        summary = {
            "status": "success",
            "blocks_a": blocks_a,
            "blocks_b": len(hashes_b),
            "carrier": {
                k: carrier[k] for k in ("elapsed_s", "prompt_tokens", "cached_tokens")
            },
            "prefetch": {
                "landed_after_s": landed_after_s,
                "results": results,
                "blocks_pulled": metrics.get("transfer_blocks:remote_fetch", 0),
            },
            "verify": {
                **{
                    k: verify[k]
                    for k in ("elapsed_s", "prompt_tokens", "cached_tokens")
                },
                "text_matches": True,
                "max_logprob_diff": logprob_diff,
            },
            "reference_elapsed_s": reference["elapsed_s"],
            "metrics": {
                k: v
                for k, v in metrics.items()
                if "remote" in k or k.startswith("transfer")
            },
        }
        base.write_json(output / "result.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
    print("KVCR prefetch validation: PASSED", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--offload-wait", type=int, default=15)
    parser.add_argument("--prefetch-timeout", type=int, default=60)
    parser.add_argument(
        "--hint-channel",
        choices=("file", "nvext"),
        default="file",
        help="how the prefetch spec reaches the worker hook",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("e2e-artifacts"))
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if image := os.environ.get("KVCR_TEST_IMAGE"):
        print(f"Test image: {image}", flush=True)
    print(f"Test model: {args.model}", flush=True)
    try:
        run(args)
    except Exception as error:
        print(
            f"KVCR prefetch validation: FAILED ({error})", file=sys.stderr, flush=True
        )
        base.write_json(
            args.output_dir / "result.json", {"status": "failed", "error": str(error)}
        )
        for log in ("frontend.log", "worker.log"):
            path = args.output_dir / log
            if path.exists():
                print(
                    f"\nLast lines of {log}:\n"
                    + "\n".join(path.read_text(errors="replace").splitlines()[-60:]),
                    file=sys.stderr,
                )
        raise


if __name__ == "__main__":
    main()
