# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for source access, route capture and the reduction-order floor variant.

Offline: pinned operator source is taken from an existing verified cache only
(skipped when absent), checkpoints are synthetic safetensors shards and the
HTTP layer is faked.  No CUDA, no network.
"""

from __future__ import annotations

import ast
import json
import os
import struct
import tempfile
import unittest
import urllib.error
from email.message import Message
from pathlib import Path
from unittest import mock

import upstream

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional for the pure tests
    torch = None


def _cached_sources() -> tuple[dict[str, str], list[dict[str, object]]] | None:
    candidates = [
        os.environ.get("HIPFIRE_QWEN4_ORACLE_CACHE"),
        Path.home() / ".hipfire" / "datasets" / "cachequality-qwen4-upstream",
        Path.home() / ".hipfire" / "datasets" / "qwen4-upstream",
        Path(__file__).resolve().parents[3] / ".codeinsight+research" / "qwen4" / "upstream-cache",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        texts: dict[str, str] = {}
        provenance: list[dict[str, object]] = []
        try:
            for spec in upstream.UPSTREAM_SOURCE_SPECS:
                loaded = upstream._load_cached_source(Path(candidate), spec)
                if loaded is None:
                    raise LookupError(spec["name"])
                texts[spec["name"]], identity = loaded
                provenance.append(identity)
        except (LookupError, upstream.FixtureError):
            continue
        return texts, provenance
    return None


SOURCES = _cached_sources()
needs_sources = unittest.skipUnless(SOURCES is not None, "verified pinned operator source cache not available")
needs_torch = unittest.skipUnless(torch is not None and SOURCES is not None, "torch + pinned source cache required")


def _pattern_bytes(count: int, salt: int) -> bytes:
    """Deterministic arbitrary 16-bit words (BF16 bit patterns are preserved, never interpreted)."""
    return struct.pack(f"<{count}H", *[((index * 2654435761 + salt) >> 7) & 0xFFFF for index in range(count)])


def _build_shard(tensors: dict[str, tuple[str, tuple[int, ...], bytes]]) -> bytes:
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    cursor = 0
    for name, (dtype, shape, payload) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [cursor, cursor + len(payload)]}
        cursor += len(payload)
    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return struct.pack("<Q", len(raw_header)) + raw_header + b"".join(item[2] for item in tensors.values())


EMBED = "model.language_model.embed_tokens.weight"
GATE_UP = "model.language_model.layers.0.mlp.experts.gate_up_proj"
DOWN = "model.language_model.layers.0.mlp.experts.down_proj"
MTP = "mtp.fc_hidden.weight"
PLE = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
HALF = "model.language_model.final_half.weight"
IDS = "model.language_model.ids"
SHARD = "model-00001-of-00001.safetensors"


def _fixture_tensors() -> dict[str, tuple[str, tuple[int, ...], bytes]]:
    return {
        EMBED: ("BF16", (10, 4), _pattern_bytes(40, 1)),
        GATE_UP: ("BF16", (3, 6, 5), _pattern_bytes(90, 2)),
        DOWN: ("BF16", (3, 5, 3), _pattern_bytes(45, 3)),
        MTP: ("BF16", (2, 2), _pattern_bytes(4, 4)),
        PLE: ("BF16", (12, 5), _pattern_bytes(60, 5)),
        HALF: ("F16", (2, 2), _pattern_bytes(4, 6)),
        IDS: ("I64", (4,), struct.pack("<4q", 5, -1, 77, 2**40)),
    }


def _checkpoint(
    root: Path,
    shard_bytes: bytes,
    *,
    local: bool,
    stage: bool = False,
    network_budget: int | None = None,
    disk_budget: int | None = None,
) -> upstream.PinnedCheckpoint:
    tensors = _fixture_tensors()
    budget = upstream._ByteBudget(network_budget)
    client = upstream._RangeClient(root / "cache", network_budget=budget)
    stream = None
    if disk_budget is not None:
        stream = upstream._RangeClient(
            root / "cache",
            subdir="stream_ranges",
            network_budget=budget,
            disk_budget_bytes=disk_budget,
        )
        stream.seals = client.seals
    local_shards = {}
    if local:
        path = root / "local" / SHARD
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(shard_bytes)
        local_shards[SHARD] = path
    return upstream.PinnedCheckpoint(
        client,
        index={"weight_map": {name: SHARD for name in tensors}, "metadata": {"total_size": 0}},
        config={"text_config": {"model_type": "qwen4_exp_text", "num_experts": 3, "num_hidden_layers": 1}},
        index_meta={},
        config_meta={},
        index_sha256="0" * 64,
        config_sha256="1" * 64,
        stream_client=stream,
        local_dir=root / "local" if local else None,
        local_shards=local_shards,
        network_budget=budget,
        staging_root=root / "stage" if stage else None,
    )


class _FakeResponse:
    def __init__(self, body: bytes, start: int, end: int, total: int):
        self.status = 206
        self._body = body
        self.headers = {
            "Content-Range": f"bytes {start}-{end}/{total}",
            "ETag": '"shard-etag"',
            "X-Repo-Commit": upstream.HF_CONFIG_COMMIT,
        }

    def getcode(self) -> int:
        return self.status

    def read(self, maximum: int = -1) -> bytes:
        return self._body if maximum < 0 else self._body[:maximum]


class _FakeHub:
    """Serves one shard through `urlopen`, recording every range request."""

    def __init__(self, shard_bytes: bytes):
        self.shard_bytes = shard_bytes
        self.requests: list[tuple[int, int]] = []

    def __call__(self, request: object, timeout: float = 0) -> _FakeResponse:
        del timeout
        assert request.full_url.endswith(SHARD), request.full_url  # type: ignore[attr-defined]
        first, last = request.headers["Range"].removeprefix("bytes=").split("-")  # type: ignore[attr-defined]
        start, end = int(first), min(int(last), len(self.shard_bytes) - 1)
        self.requests.append((start, end))
        return _FakeResponse(self.shard_bytes[start : end + 1], start, end, len(self.shard_bytes))


def _tensor_span(shard_bytes: bytes, tensors: dict, name: str) -> tuple[int, int]:
    header_length = struct.unpack("<Q", shard_bytes[:8])[0]
    offsets = json.loads(shard_bytes[8 : 8 + header_length])[name]["data_offsets"]
    return 8 + header_length + offsets[0], 8 + header_length + offsets[1]


class LocalSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.tensors = _fixture_tensors()
        self.shard = _build_shard(self.tensors)

    def tearDown(self) -> None:
        self._temp.cleanup()

    def test_local_rows_are_the_exact_source_bytes(self) -> None:
        checkpoint = _checkpoint(self.root, self.shard, local=True)
        record = checkpoint.tensor_rows_raw(EMBED, 2, 3)
        begin, _ = _tensor_span(self.shard, self.tensors, EMBED)
        expected = self.shard[begin + 2 * 8 : begin + 5 * 8]
        self.assertEqual(bytes(record["raw"]), expected)
        self.assertEqual(record["dtype"], "BF16")
        self.assertEqual(record["slice_shape"], [3, 4])
        self.assertEqual(record["source"], "local")
        self.assertEqual(record["sha256"], upstream._sha256_bytes(expected))
        self.assertEqual(checkpoint.local_read_bytes, len(expected))
        self.assertEqual(bytes(checkpoint.tensor_rows_raw(EMBED, 7)["raw"]), self.shard[begin + 7 * 8 : begin + 10 * 8])
        checkpoint.close()

    def test_expert_rows_use_fused_expert_offsets(self) -> None:
        checkpoint = _checkpoint(self.root, self.shard, local=True)
        record = checkpoint.expert_rows_raw(GATE_UP, 2, 1, 3)
        begin, _ = _tensor_span(self.shard, self.tensors, GATE_UP)
        first = begin + ((2 * 6 + 1) * 5) * 2
        self.assertEqual(bytes(record["raw"]), self.shard[first : first + 3 * 5 * 2])
        self.assertEqual(record["slice_shape"], [3, 5])
        with self.assertRaises(upstream.FixtureError):
            checkpoint.expert_rows_raw(GATE_UP, 3, 0, 1)
        with self.assertRaises(upstream.FixtureError):
            checkpoint.expert_rows_raw(EMBED, 0, 0, 1)
        checkpoint.close()

    def test_dtype_is_checked_exactly_and_f16_is_never_reinterpreted(self) -> None:
        checkpoint = _checkpoint(self.root, self.shard, local=True)
        with self.assertRaises(upstream.FixtureError):
            checkpoint.tensor_rows_raw(HALF, 0, 1)
        with self.assertRaises(upstream.FixtureError):
            checkpoint.tensor_rows_raw(EMBED, 0, 1, expect_dtype="F32")
        record = checkpoint.tensor_rows_raw(IDS, 0, None, expect_dtype="I64")
        self.assertEqual(struct.unpack("<4q", bytes(record["raw"])), (5, -1, 77, 2**40))
        checkpoint.close()

    def test_gather_keeps_order_and_duplicates(self) -> None:
        checkpoint = _checkpoint(self.root, self.shard, local=True)
        rows = [9, 0, 9, 4, 5, 5, 1]
        record = checkpoint.gather_rows_raw(EMBED, rows)
        begin, _ = _tensor_span(self.shard, self.tensors, EMBED)
        expected = b"".join(self.shard[begin + row * 8 : begin + row * 8 + 8] for row in rows)
        self.assertEqual(bytes(record["raw"]), expected)
        self.assertEqual(record["slice_shape"], [7, 4])
        # unique sorted rows are 0,1,4,5,9: adjacent ids coalesce into runs {0,1},{4,5},{9}
        self.assertEqual(record["range_count"], 3)
        with self.assertRaises(upstream.FixtureError):
            checkpoint.gather_rows_raw(EMBED, [10])
        checkpoint.close()

    def test_legacy_record_api_reads_local_shards_with_python_values(self) -> None:
        checkpoint = _checkpoint(self.root, self.shard, local=True)
        record = checkpoint.tensor_rows(EMBED, 1, 2)
        begin, _ = _tensor_span(self.shard, self.tensors, EMBED)
        words = struct.unpack("<8H", self.shard[begin + 8 : begin + 24])
        self.assertEqual(tuple(record["values"].data), words)
        self.assertEqual(record["dtype"], "BF16")
        checkpoint.close()

    def test_truncated_or_padded_shard_is_rejected(self) -> None:
        for damaged in (self.shard[:-1], self.shard + b"\0"):
            with tempfile.TemporaryDirectory() as directory:
                checkpoint = _checkpoint(Path(directory), damaged, local=True)
                with self.assertRaises(upstream.FixtureError):
                    checkpoint.tensor_descriptor(EMBED)

    def test_source_manifest_is_exact_and_header_only(self) -> None:
        checkpoint = _checkpoint(self.root, self.shard, local=True)
        manifest = checkpoint.source_manifest()
        self.assertEqual(manifest["tensor_count"], len(self.tensors))
        for name, (dtype, shape, payload) in self.tensors.items():
            entry = manifest["tensors"][name]
            self.assertEqual((entry["dtype"], tuple(entry["shape"]), entry["bytes"]), (dtype, shape, len(payload)))
        self.assertEqual(manifest["tensors"][GATE_UP]["role"], "routed_experts")
        self.assertEqual(manifest["tensors"][PLE]["role"], "ple_embedding")
        self.assertEqual(manifest["tensors"][MTP]["role"], "mtp")
        self.assertEqual(manifest["tensors"][EMBED]["role"], "embedding")
        self.assertEqual(manifest["bytes_by_dtype"]["BF16"], 80 + 180 + 90 + 8 + 120)
        self.assertEqual(manifest["bytes_by_role_dtype"]["routed_experts"], {"BF16": 270})
        self.assertEqual(manifest["bytes_by_dtype"]["F16"], 8)
        self.assertFalse(manifest["index_total_size_matches"])
        header_length = struct.unpack("<Q", self.shard[:8])[0]
        self.assertEqual(
            manifest["shards"][SHARD]["header_sha256"],
            upstream._sha256_bytes(self.shard[8 : 8 + header_length]),
        )
        self.assertEqual(checkpoint.local_read_bytes, 0, "a manifest must not read tensor payloads")
        self.assertEqual(manifest, checkpoint.source_manifest())
        checkpoint.close()

    def test_index_that_disagrees_with_shard_header_is_rejected(self) -> None:
        checkpoint = _checkpoint(self.root, self.shard, local=True)
        checkpoint.weight_map["model.language_model.not_in_shard"] = SHARD
        with self.assertRaises(upstream.FixtureError):
            checkpoint.source_manifest()
        checkpoint.close()

    def test_lfs_identity_records_sha_and_rejects_size_mismatch(self) -> None:
        checkpoint = _checkpoint(self.root, self.shard, local=True)
        headers = Message()
        headers["X-Linked-Etag"] = '"' + "ab" * 32 + '"'
        headers["X-Linked-Size"] = str(len(self.shard))
        headers["X-Repo-Commit"] = upstream.HF_CONFIG_COMMIT

        class Opener:
            def open(self, request: object, timeout: float = 0) -> None:
                raise urllib.error.HTTPError("u", 302, "Found", headers, None)

        with mock.patch.object(upstream.urllib.request, "build_opener", return_value=Opener()):
            identity = checkpoint.shard_lfs_identity(SHARD)
            self.assertEqual(identity["lfs_sha256"], "ab" * 32)
            self.assertEqual(identity["lfs_size"], len(self.shard))
            checkpoint._lfs.clear()
            headers.replace_header("X-Linked-Size", str(len(self.shard) + 1))
            with self.assertRaises(upstream.FixtureError):
                checkpoint.shard_lfs_identity(SHARD)
        checkpoint.close()

    def test_staged_rows_are_served_without_touching_the_source(self) -> None:
        checkpoint = _checkpoint(self.root, self.shard, local=True, stage=True)
        report = checkpoint.stage_rows(PLE, [3, 3, 7])
        self.assertEqual(report["newly_staged_rows"], 2)
        self.assertEqual(checkpoint.stage_rows(PLE, [3, 7])["newly_staged_rows"], 0)
        read_before = checkpoint.local_read_bytes
        record = checkpoint.gather_rows_raw(PLE, [7, 3, 7])
        begin, _ = _tensor_span(self.shard, self.tensors, PLE)
        row = lambda index: self.shard[begin + index * 10 : begin + index * 10 + 10]  # noqa: E731
        self.assertEqual(bytes(record["raw"]), row(7) + row(3) + row(7))
        self.assertEqual(record["source"], "staged")
        self.assertEqual(checkpoint.local_read_bytes, read_before)
        self.assertEqual(checkpoint.staged_read_bytes, 20, "two unique staged rows of 10 bytes")
        mixed = checkpoint.gather_rows_raw(PLE, [3, 4])
        self.assertEqual(mixed["source"], "mixed")
        self.assertEqual(bytes(mixed["raw"]), row(3) + row(4))
        checkpoint.close()

    def test_tampered_staged_rows_fail_their_digest(self) -> None:
        checkpoint = _checkpoint(self.root, self.shard, local=True, stage=True)
        checkpoint.stage_rows(PLE, [1, 2])
        checkpoint.close()
        data = next((self.root / "stage").glob("*/rows.bin"))
        raw = bytearray(data.read_bytes())
        raw[0] ^= 0xFF
        data.write_bytes(bytes(raw))
        reopened = _checkpoint(self.root, self.shard, local=True, stage=True)
        with self.assertRaises(upstream.FixtureError):
            reopened.gather_rows_raw(PLE, [1])
        reopened.close()

    def test_coverage_report_marks_partial_traversal_incomplete(self) -> None:
        checkpoint = _checkpoint(self.root, self.shard, local=True)
        checkpoint.expert_rows_raw(GATE_UP, 0, 0, 6)
        checkpoint.expert_rows_raw(GATE_UP, 2, 0, 6)
        checkpoint.expert_rows_raw(GATE_UP, 1, 0, 2)  # partial slice does not count as an expert read
        checkpoint.expert_rows_raw(DOWN, 1, 0, 5)
        report = checkpoint.coverage_report()
        self.assertEqual(report["experts_read_per_layer"][0], {"gate_up_proj": 2, "down_proj": 1})
        self.assertEqual(report["layers_partial"], [0])
        self.assertFalse(report["complete_expert_traversal"])
        for expert in (0, 1, 2):
            checkpoint.expert_rows_raw(GATE_UP, expert, 0, 6)
            checkpoint.expert_rows_raw(DOWN, expert, 0, 5)
        self.assertTrue(checkpoint.coverage_report()["complete_expert_traversal"])
        checkpoint.close()

    def test_read_ledger_records_range_hashes(self) -> None:
        checkpoint = _checkpoint(self.root, self.shard, local=True)
        record = checkpoint.expert_rows_raw(DOWN, 1, 0, 5)
        ledger = checkpoint.read_ledger()
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger[0]["sha256"], record["sha256"])
        self.assertEqual(ledger[0]["byte_start"], record["byte_start"])
        self.assertEqual(checkpoint.provenance()["read_ledger_sha256"], checkpoint.ledger_digest())
        checkpoint.close()

    @needs_torch
    def test_raw_record_to_torch_preserves_bf16_bits(self) -> None:
        checkpoint = _checkpoint(self.root, self.shard, local=True)
        record = checkpoint.tensor_rows_raw(EMBED, 0, 10)
        value = upstream.raw_record_to_torch(record, torch)
        self.assertEqual(value.dtype, torch.bfloat16)
        self.assertEqual(tuple(value.shape), (10, 4))
        begin, end = _tensor_span(self.shard, self.tensors, EMBED)
        expected = list(struct.unpack("<40H", self.shard[begin:end]))
        self.assertEqual(value.reshape(-1).view(torch.uint16).tolist(), expected)
        ids = upstream.raw_record_to_torch(checkpoint.tensor_rows_raw(IDS, 0), torch)
        self.assertEqual(ids.tolist(), [5, -1, 77, 2**40])
        checkpoint.close()


class RemoteStreamTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.tensors = _fixture_tensors()
        self.shard = _build_shard(self.tensors)
        self.hub = _FakeHub(self.shard)

    def tearDown(self) -> None:
        self._temp.cleanup()

    def _open(self, **kwargs: object) -> upstream.PinnedCheckpoint:
        return _checkpoint(self.root, self.shard, local=False, **kwargs)  # type: ignore[arg-type]

    def test_large_reads_are_coalesced_into_bounded_sealed_ranges_and_cached(self) -> None:
        with mock.patch.object(upstream.urllib.request, "urlopen", self.hub), mock.patch.object(
            upstream, "MAX_TENSOR_SLICE_BYTES", 32
        ), mock.patch.object(upstream, "RAW_STREAM_RANGE_BYTES", 64):
            checkpoint = self._open(disk_budget=10_000)
            checkpoint.tensor_descriptor(GATE_UP)
            self.hub.requests.clear()
            record = checkpoint.expert_rows_raw(GATE_UP, 1, 0, 6)  # 60 bytes -> one 64-byte-bounded range
            begin, _ = _tensor_span(self.shard, self.tensors, GATE_UP)
            self.assertEqual(bytes(record["raw"]), self.shard[begin + 60 : begin + 120])
            self.assertEqual(record["source"], "remote")
            big = checkpoint.tensor_rows_raw(GATE_UP, 0, 3)  # 180 bytes -> 3 ranges of <= 64 bytes
            self.assertEqual(bytes(big["raw"]), self.shard[begin : begin + 180])
            self.assertEqual(big["range_count"], 3)
            self.assertTrue(all(end - start + 1 <= 64 for start, end in self.hub.requests))
            self.assertEqual(len(self.hub.requests), 4)
            before = len(self.hub.requests)
            again = checkpoint.tensor_rows_raw(GATE_UP, 0, 3)
            self.assertEqual(bytes(again["raw"]), bytes(big["raw"]))
            self.assertEqual(len(self.hub.requests), before, "second read must be served from the stream cache")
            stats = checkpoint.transfer_stats()
            self.assertEqual(stats["stream_client"]["network_requests"], 4)
            self.assertEqual(stats["stream_client"]["cache_hit_bytes"], 180)
            self.assertEqual(stats["stream_client"]["network_bytes"], 240)
            checkpoint.close()

    def test_small_reads_use_the_legacy_sparse_range_cache(self) -> None:
        with mock.patch.object(upstream.urllib.request, "urlopen", self.hub):
            checkpoint = self._open(disk_budget=10_000)
            record = checkpoint.tensor_rows_raw(MTP, 0, 2)
            begin, _ = _tensor_span(self.shard, self.tensors, MTP)
            self.assertEqual(bytes(record["raw"]), self.shard[begin : begin + 8])
            self.assertEqual(checkpoint.stream_client.network_requests, 0)
            self.assertGreater(checkpoint.client.network_requests, 0)
            checkpoint.close()

    def test_layer_groups_are_evicted_and_accounted(self) -> None:
        with mock.patch.object(upstream.urllib.request, "urlopen", self.hub), mock.patch.object(
            upstream, "MAX_TENSOR_SLICE_BYTES", 32
        ), mock.patch.object(upstream, "RAW_STREAM_RANGE_BYTES", 64):
            checkpoint = self._open(disk_budget=10_000)
            stream = checkpoint.stream_client
            for layer, expert in enumerate((0, 1, 2)):
                self.assertEqual(checkpoint.begin_group(f"layer{layer}"), [] if layer < 2 else ["layer0"])
                checkpoint.expert_rows_raw(GATE_UP, expert, 0, 6)
            self.assertEqual(stream.live_groups, ["layer1", "layer2"])
            self.assertIn("layer0", stream.released_groups)
            self.assertEqual(len(list(stream.cache_dir.glob("*.bin"))), 2)
            tracked = sum(size for files in stream.group_files.values() for _, _, size in files)
            self.assertEqual(stream.cache_bytes, tracked)
            self.assertLessEqual(stream.peak_cache_bytes, 10_000)
            freed = checkpoint.release_group("layer1")
            self.assertGreater(freed, 0)
            self.assertEqual(len(list(stream.cache_dir.glob("*.bin"))), 1)
            checkpoint.close()
            self.assertFalse(stream.cache_dir.exists(), "close() must delete the whole stream cache")

    def test_disk_budget_is_enforced_before_downloading(self) -> None:
        with mock.patch.object(upstream.urllib.request, "urlopen", self.hub), mock.patch.object(
            upstream, "MAX_TENSOR_SLICE_BYTES", 32
        ), mock.patch.object(upstream, "RAW_STREAM_RANGE_BYTES", 64):
            checkpoint = self._open(disk_budget=100)
            checkpoint.tensor_descriptor(GATE_UP)
            self.hub.requests.clear()
            with self.assertRaises(upstream.FixtureError):
                checkpoint.tensor_rows_raw(GATE_UP, 0, 3)  # 3 x 64-byte ranges cannot fit in 100 bytes
            self.assertLessEqual(checkpoint.stream_client.cache_bytes, 100)
            checkpoint.close()

    def test_remote_byte_budget_stops_new_reads_only(self) -> None:
        with mock.patch.object(upstream.urllib.request, "urlopen", self.hub), mock.patch.object(
            upstream, "MAX_TENSOR_SLICE_BYTES", 32
        ), mock.patch.object(upstream, "RAW_STREAM_RANGE_BYTES", 64):
            checkpoint = self._open(network_budget=10**9, disk_budget=10_000)
            checkpoint.tensor_descriptor(GATE_UP)
            checkpoint.expert_rows_raw(GATE_UP, 0, 0, 6)
            checkpoint.network_budget.limit = checkpoint.network_budget.used + 10
            requests = len(self.hub.requests)
            checkpoint.expert_rows_raw(GATE_UP, 0, 0, 6)  # cached: free
            with self.assertRaises(upstream.FixtureError):
                checkpoint.expert_rows_raw(GATE_UP, 1, 0, 6)
            self.assertEqual(len(self.hub.requests), requests)
            self.assertEqual(
                checkpoint.network_budget.used,
                checkpoint.transfer_stats()["network_bytes"],
            )
            checkpoint.close()

    def test_unreadable_revision_in_response_is_rejected(self) -> None:
        bad = _FakeHub(self.shard)

        def wrong_revision(request: object, timeout: float = 0) -> _FakeResponse:
            response = bad(request, timeout)
            response.headers["X-Repo-Commit"] = "f" * 40
            return response

        with mock.patch.object(upstream.urllib.request, "urlopen", wrong_revision):
            checkpoint = self._open(disk_budget=10_000)
            with self.assertRaises(upstream.FixtureError):
                checkpoint.tensor_descriptor(EMBED)


class ReverseTransformTests(unittest.TestCase):
    EXPERTS = (
        "class Qwen4ExpTextExperts(nn.Module):\n"
        "    def forward(self, hidden_states, top_k_index, top_k_weights):\n"
        "        for expert_idx in expert_hit:\n"
        "            pass\n"
    )

    def test_loop_is_flipped_in_place_on_a_copy(self) -> None:
        module = ast.parse(self.EXPERTS)
        upstream._reverse_expert_loop_transform(module)
        self.assertIn("for expert_idx in expert_hit.flip(0):", ast.unparse(module))
        compile(module, "<test>", "exec")

    def test_unexpected_loop_shape_is_rejected(self) -> None:
        for source in (
            self.EXPERTS.replace("expert_hit", "other"),
            self.EXPERTS + "        for expert_idx in expert_hit:\n            pass\n",
        ):
            with self.assertRaises(upstream.FixtureError):
                upstream._reverse_expert_loop_transform(ast.parse(source))

    def test_unknown_reduction_order_is_rejected_before_use(self) -> None:
        with self.assertRaises(upstream.FixtureError):
            upstream.load_pinned_operators({}, [], object(), reduction_order="pairwise")

    @needs_sources
    def test_identity_records_variant_without_touching_trusted_hashes(self) -> None:
        texts, _ = SOURCES
        source = texts["transformers_modeling"]
        original = upstream._EXPECTED_OPERATOR_AST_SHA256["Qwen4ExpTextExperts.forward"]
        native = upstream._experts_reduction_identity(source, "native", original)
        self.assertFalse(native["transformed"])
        variant = upstream._experts_reduction_identity(source, "reverse-experts", original)
        self.assertTrue(variant["transformed"])
        self.assertEqual(variant["experts_forward_ast_sha256"], original)
        self.assertNotEqual(variant["variant_forward_ast_sha256"], original)
        diff_lines = variant["variant_diff"].splitlines()
        self.assertTrue(any(line.startswith("+") and line.endswith("for expert_idx in expert_hit.flip(0):") for line in diff_lines))
        self.assertTrue(any(line.startswith("-") and line.endswith("for expert_idx in expert_hit:") for line in diff_lines))
        self.assertEqual(variant["variant_diff_sha256"], upstream._sha256_bytes(variant["variant_diff"].encode()))
        spec = next(item for item in upstream.UPSTREAM_SOURCE_SPECS if item["name"] == "transformers_modeling")
        self.assertEqual(variant["trusted_source_sha256"], spec["sha256"])
        self.assertEqual(variant, upstream._experts_reduction_identity(source, "reverse-experts", original))
        # the pinned allowlisted AST is unchanged by building a variant
        self.assertEqual(upstream._pinned_operator_ast_digests(texts)["Qwen4ExpTextExperts.forward"], original)
        with self.assertRaises(upstream.FixtureError):
            upstream._experts_reduction_identity(source, "reverse-experts", "0" * 64)


HIDDEN, INTER, EXPERTS, TOP_K, TOKENS = 32, 8, 6, 3, 64
PREFIX = "model.language_model.layers.5.mlp"
CFG = {
    "hidden_size": HIDDEN,
    "num_attention_heads": 2,
    "shared_expert_intermediate_size": INTER,
    "moe_intermediate_size": INTER,
    "num_experts": EXPERTS,
    "num_experts_per_tok": TOP_K,
    "hidden_act": "silu",
}


class _MoeFixture:
    """Synthetic BF16 layer with zero shared expert so the output is exactly the routed accumulation."""

    def __init__(self, seed: int = 7):
        generator = torch.Generator().manual_seed(seed)

        def rand(*shape: int, scale: float = 0.5) -> "torch.Tensor":
            return (torch.randn(*shape, generator=generator) * scale).to(torch.bfloat16)

        self.hidden = rand(TOKENS, HIDDEN, scale=1.0)
        self.gate_up = rand(EXPERTS, 2 * INTER, HIDDEN)
        self.down = rand(EXPERTS, HIDDEN, INTER)
        self.full = {
            f"{PREFIX}.gate.weight": rand(EXPERTS, HIDDEN, scale=1.0),
            f"{PREFIX}.shared_expert.gate_proj.weight": torch.zeros(INTER, HIDDEN, dtype=torch.bfloat16),
            f"{PREFIX}.shared_expert.up_proj.weight": torch.zeros(INTER, HIDDEN, dtype=torch.bfloat16),
            f"{PREFIX}.shared_expert.down_proj.weight": torch.zeros(HIDDEN, INTER, dtype=torch.bfloat16),
            f"{PREFIX}.shared_expert_gate.weight": torch.zeros(1, HIDDEN, dtype=torch.bfloat16),
        }
        self.expert_reads: list[tuple[str, int]] = []

    def load_full(self, name: str) -> "torch.Tensor":
        return self.full[name]

    def load_expert(self, name: str, expert: int, row_start: int, row_count: int) -> "torch.Tensor":
        self.expert_reads.append((name, expert))
        table = self.gate_up if name.endswith("gate_up_proj") else self.down
        return table[expert, row_start : row_start + row_count].contiguous()

    def router(self) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        weight = self.full[f"{PREFIX}.gate.weight"]
        logits = torch.nn.functional.linear(self.hidden, weight)
        probabilities = torch.nn.functional.softmax(logits, dtype=torch.float, dim=-1)
        values, indices = torch.topk(probabilities, TOP_K, dim=-1)
        values = values / values.sum(dim=-1, keepdim=True)
        return logits, values.to(logits.dtype), indices

    def contributions(self, indices: "torch.Tensor", scores: "torch.Tensor") -> dict[int, tuple]:
        """Independently computed per-expert BF16 products, keyed by expert id."""
        result = {}
        for expert in sorted(set(indices.flatten().tolist())):
            top_pos, token_idx = torch.where((indices == expert).t())
            current = self.hidden[token_idx]
            gate, up = torch.nn.functional.linear(current, self.gate_up[expert]).chunk(2, dim=-1)
            product = torch.nn.functional.silu(gate) * up
            product = torch.nn.functional.linear(product, self.down[expert])
            result[expert] = (token_idx, product * scores[token_idx, top_pos, None])
        return result

    def accumulate(self, contributions: dict[int, tuple], order: list[int]) -> "torch.Tensor":
        total = torch.zeros_like(self.hidden)
        for expert in order:
            token_idx, product = contributions[expert]
            total[token_idx] = total[token_idx] + product.to(total.dtype)
        return total


@needs_torch
class RouteCaptureAndFloorTests(unittest.TestCase):
    def _operators(self, **kwargs: object) -> "upstream.PinnedQwen4Operators":
        texts, provenance = SOURCES
        return upstream.load_pinned_operators(texts, provenance, torch, **kwargs)  # type: ignore[arg-type]

    def _run(self, operators: "upstream.PinnedQwen4Operators", fixture: _MoeFixture) -> "torch.Tensor":
        return operators.moe(fixture.hidden, CFG, fixture.load_full, fixture.load_expert, 5)

    def test_route_sink_sees_the_actual_all_position_route_once(self) -> None:
        fixture = _MoeFixture()
        captured: list[tuple] = []
        operators = self._operators(route_sink=lambda *item: captured.append(item))
        output = self._run(operators, fixture)
        self.assertEqual(len(captured), 1)
        layer, logits, scores, indices = captured[0]
        self.assertEqual(layer, 5)
        self.assertEqual(tuple(logits.shape), (TOKENS, EXPERTS))
        self.assertEqual(tuple(scores.shape), (TOKENS, TOP_K))
        self.assertEqual(tuple(indices.shape), (TOKENS, TOP_K))
        self.assertEqual(indices.dtype, torch.int64)
        self.assertEqual(scores.dtype, torch.bfloat16)
        expected_logits, expected_scores, expected_indices = fixture.router()
        self.assertTrue(torch.equal(logits, expected_logits))
        self.assertTrue(torch.equal(indices, expected_indices))
        self.assertTrue(torch.equal(scores, expected_scores))
        # ordered top-k: scores are non-increasing per position
        self.assertTrue(bool((scores[:, :-1].float() >= scores[:, 1:].float()).all()))
        # the experts that were streamed are exactly the ones the captured route selected
        selected = set(indices.flatten().tolist())
        self.assertEqual({expert for _, expert in fixture.expert_reads}, selected)
        self.assertEqual(len(fixture.expert_reads), 2 * len(selected), "each routed expert read once per tensor")
        # capture does not perturb the forward
        plain = self._run(self._operators(), _MoeFixture())
        self.assertTrue(torch.equal(output, plain))
        self.assertEqual(operators.variant_provenance()["moe_calls"], 1)

    def test_native_matches_ascending_and_reverse_matches_descending_accumulation(self) -> None:
        fixture = _MoeFixture()
        _, scores, indices = fixture.router()
        products = fixture.contributions(indices, scores)
        ascending = sorted(products)
        native_expected = fixture.accumulate(products, ascending)
        reverse_expected = fixture.accumulate(products, ascending[::-1])
        self.assertFalse(
            torch.equal(native_expected, reverse_expected),
            "fixture must be order sensitive in BF16 or the variant is untestable",
        )
        native_routes: list[tuple] = []
        reverse_routes: list[tuple] = []
        native = self._run(self._operators(route_sink=lambda *item: native_routes.append(item)), _MoeFixture())
        reverse = self._run(
            self._operators(reduction_order="reverse-experts", route_sink=lambda *item: reverse_routes.append(item)),
            _MoeFixture(),
        )
        self.assertEqual(native.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(native, native_expected))
        self.assertTrue(torch.equal(reverse, reverse_expected))
        self.assertFalse(torch.equal(native, reverse))
        # routing ids/scores/logits are identical: only the accumulation order differs
        for native_item, reverse_item in zip(native_routes[0][1:], reverse_routes[0][1:]):
            self.assertTrue(torch.equal(native_item, reverse_item))
        # the difference is rounding-level, not a different computation
        self.assertLess(float((native.detach().float() - reverse.detach().float()).abs().max()), 0.5)

    def test_variant_provenance_and_default_provenance(self) -> None:
        default = self._operators()
        self.assertEqual(default.provenance(), self._operators().provenance())
        self.assertNotIn("variant", default.provenance())
        reverse = self._operators(reduction_order="reverse-experts", inference=True)
        provenance = reverse.provenance()
        variant = provenance["variant"][0]
        self.assertEqual(variant["reduction_order"], "reverse-experts")
        self.assertEqual(variant["grad_mode"], "no_grad")
        self.assertTrue(variant["reduction"]["transformed"])
        experts_row = next(row for row in provenance["moe"] if row["callable"] == "Qwen4ExpTextExperts.forward")
        self.assertEqual(experts_row["variant_diff_sha256"], variant["reduction"]["variant_diff_sha256"])
        self.assertEqual(experts_row["ast_sha256"], variant["reduction"]["experts_forward_ast_sha256"])
        self.assertEqual(experts_row["sha256"], default.provenance()["moe"][3]["sha256"])
        fixture = _MoeFixture()
        self._run(reverse, fixture)
        self.assertEqual(reverse.variant_provenance()["moe_hidden_dtypes"], ["torch.bfloat16"])

    def test_inference_mode_builds_no_autograd_graph(self) -> None:
        fixture = _MoeFixture()
        output = self._run(self._operators(inference=True), fixture)
        self.assertFalse(output.requires_grad)
        self.assertTrue(torch.is_grad_enabled())

    def test_route_sink_failure_and_bad_sink_are_not_swallowed(self) -> None:
        def failing(*_item: object) -> None:
            raise RuntimeError("sink failed")

        with self.assertRaises(RuntimeError):
            self._run(self._operators(route_sink=failing), _MoeFixture())
        with self.assertRaises(upstream.FixtureError):
            self._operators(route_sink="not callable")


if __name__ == "__main__":
    unittest.main()
