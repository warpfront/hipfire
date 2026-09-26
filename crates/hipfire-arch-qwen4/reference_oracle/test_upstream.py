# SPDX-License-Identifier: Apache-2.0
"""Regression tests for pinned checkpoint range provenance and bounds."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import upstream


class _Response:
    def __init__(self, body: bytes, *, start: int, end: int, total: int | None, etag: str = "etag-a"):
        self.status = 206
        self.headers = {
            "Content-Range": f"bytes {start}-{end}/{total if total is not None else '*'}",
            "ETag": etag,
            "X-Repo-Commit": upstream.HF_CONFIG_COMMIT,
        }
        self._body = body

    def getcode(self) -> int:
        return self.status

    def read(self, maximum: int = -1) -> bytes:
        return self._body if maximum < 0 else self._body[:maximum]


def _response(body: bytes, start: int, end: int, total: int | None = 16, etag: str = "etag-a") -> _Response:
    return _Response(body, start=start, end=end, total=total, etag=etag)


class RangeClientTests(unittest.TestCase):
    def _client(self) -> tuple[tempfile.TemporaryDirectory[str], upstream._RangeClient]:
        temp = tempfile.TemporaryDirectory()
        return temp, upstream._RangeClient(Path(temp.name))
    def test_connection_reset_retries_with_same_range_and_succeeds(self) -> None:
        temp, client = self._client()
        try:
            reset = upstream.urllib.error.URLError(ConnectionResetError(104, "Connection reset by peer"))
            with mock.patch.object(
                upstream.urllib.request,
                "urlopen",
                side_effect=[reset, _response(b"abcd", 0, 3)],
            ) as urlopen, mock.patch.object(upstream.time, "sleep") as sleep:
                body, _ = client.read("https://example.test/shard", 0, 3, maximum=8)
            self.assertEqual(body, b"abcd")
            self.assertEqual(urlopen.call_count, 2)
            self.assertEqual(
                [call.args[0].get_header("Range") for call in urlopen.call_args_list],
                ["bytes=0-3", "bytes=0-3"],
            )
            self.assertEqual(sleep.call_count, 1)
            sleep.assert_called_once_with(upstream.RANGE_RETRY_BACKOFF_SECONDS)
        finally:
            temp.cleanup()

    def test_connection_reset_retries_are_bounded(self) -> None:
        temp, client = self._client()
        try:
            reset = upstream.urllib.error.URLError(ConnectionResetError(104, "Connection reset by peer"))
            errors = [reset] * (upstream.RANGE_MAX_RETRIES + 1)
            with mock.patch.object(
                upstream.urllib.request,
                "urlopen",
                side_effect=errors,
            ) as urlopen, mock.patch.object(upstream.time, "sleep") as sleep:
                with self.assertRaises(upstream.FixtureError):
                    client.read("https://example.test/shard", 0, 3, maximum=8)
            self.assertEqual(urlopen.call_count, upstream.RANGE_MAX_RETRIES + 1)
            self.assertEqual(sleep.call_count, upstream.RANGE_MAX_RETRIES)
        finally:
            temp.cleanup()

    def test_identity_mismatch_is_not_retried(self) -> None:
        temp, client = self._client()
        try:
            with mock.patch.object(
                upstream.urllib.request,
                "urlopen",
                side_effect=[_response(b"abcd", 1, 4), _response(b"abcd", 0, 3)],
            ) as urlopen, mock.patch.object(upstream.time, "sleep") as sleep:
                with self.assertRaises(upstream.FixtureError):
                    client.read("https://example.test/shard", 0, 3, maximum=8)
            self.assertEqual(urlopen.call_count, 1)
            sleep.assert_not_called()
        finally:
            temp.cleanup()


    def test_short_response_is_rejected(self) -> None:
        temp, client = self._client()
        try:
            with mock.patch.object(upstream.urllib.request, "urlopen", return_value=_response(b"abc", 0, 3)):
                with self.assertRaises(upstream.FixtureError):
                    client.read("https://example.test/shard", 0, 3, maximum=8)
        finally:
            temp.cleanup()

    def test_long_response_is_rejected(self) -> None:
        temp, client = self._client()
        try:
            with mock.patch.object(upstream.urllib.request, "urlopen", return_value=_response(b"abcde", 0, 3)):
                with self.assertRaises(upstream.FixtureError):
                    client.read("https://example.test/shard", 0, 3, maximum=8)
        finally:
            temp.cleanup()

    def test_content_range_total_must_remain_consistent(self) -> None:
        temp, client = self._client()
        try:
            url = "https://example.test/shard"
            with mock.patch.object(
                upstream.urllib.request,
                "urlopen",
                side_effect=[_response(b"abcd", 0, 3, total=16), _response(b"efgh", 4, 7, total=17)],
            ):
                client.read(url, 0, 3, maximum=8)
                with self.assertRaises(upstream.FixtureError):
                    client.read(url, 4, 7, maximum=8)
        finally:
            temp.cleanup()

    def test_changed_etag_is_rejected(self) -> None:
        temp, client = self._client()
        try:
            url = "https://example.test/shard"
            with mock.patch.object(
                upstream.urllib.request,
                "urlopen",
                side_effect=[_response(b"abcd", 0, 3), _response(b"efgh", 4, 7, etag="etag-b")],
            ):
                client.read(url, 0, 3, maximum=8)
                with self.assertRaises(upstream.FixtureError):
                    client.read(url, 4, 7, maximum=8)
        finally:
            temp.cleanup()

    def test_wrong_content_range_end_and_total_are_rejected(self) -> None:
        temp, client = self._client()
        try:
            with mock.patch.object(
                upstream.urllib.request,
                "urlopen",
                return_value=_response(b"abcd", 0, 3, total=3),
            ):
                with self.assertRaises(upstream.FixtureError):
                    client.read("https://example.test/shard", 0, 3, maximum=8)
        finally:
            temp.cleanup()

    def test_cache_hit_rechecks_seals_and_length(self) -> None:
        temp, client = self._client()
        try:
            url = "https://example.test/shard"
            with mock.patch.object(upstream.urllib.request, "urlopen", return_value=_response(b"abcd", 0, 3)):
                client.read(url, 0, 3, maximum=8)
            key = upstream._sha256_bytes(f"{url}\0{0}\0{3}".encode("utf-8"))
            metadata_path = Path(temp.name) / "checkpoint_ranges" / f"{key}.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["etag"] = "etag-changed"
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaises(upstream.FixtureError):
                client.read(url, 0, 3, maximum=8)
        finally:
            temp.cleanup()

class PinnedOperatorTests(unittest.TestCase):
    def test_operator_signature_guard_rejects_source_change(self) -> None:
        spec = upstream._PINNED_OPERATOR_SIGNATURES[0]
        source = "class Qwen4ExpTextGatedResidual(nn.Module):\n    def forward(self, hyper_input):\n        return hyper_input\n"
        upstream._validate_pinned_operator_signatures(
            {"transformers_modeling": source},
            specs=(spec,),
        )
        changed = source.replace("hyper_input", "renamed")
        with self.assertRaises(upstream.FixtureError):
            upstream._validate_pinned_operator_signatures(
                {"transformers_modeling": changed},
                specs=(spec,),
            )

    def test_operator_base_tuple_is_exact(self) -> None:
        spec = upstream._PINNED_OPERATOR_SIGNATURES[0]
        source = "class Qwen4ExpTextGatedResidual(nn.Module):\n    def forward(self, hyper_input):\n        return hyper_input\n"
        changed = source.replace("(nn.Module)", "(nn.Module, Extra)")
        with self.assertRaises(upstream.FixtureError):
            upstream._validate_pinned_operator_signatures(
                {"transformers_modeling": changed},
                specs=(spec,),
            )

    def test_operator_ast_allowlist_rejects_body_mutation(self) -> None:
        cache = Path(__file__).resolve().parents[3] / ".codeinsight+research" / "qwen4" / "upstream-cache"
        source_text, _, _ = upstream.load_pinned_sources(cache)
        changed = source_text["transformers_modeling"].replace("self.hc_count", "self.hc_count + 0", 1)
        self.assertNotEqual(changed, source_text["transformers_modeling"])
        source_text = dict(source_text)
        source_text["transformers_modeling"] = changed
        with self.assertRaises(upstream.FixtureError):
            upstream._pinned_operator_ast_digests(source_text)

    def test_operator_loader_rejects_altered_source_with_copied_provenance(self) -> None:
        cache = Path(__file__).resolve().parents[3] / ".codeinsight+research" / "qwen4" / "upstream-cache"
        source_text, provenance, _ = upstream.load_pinned_sources(cache)
        changed = source_text["transformers_modeling"].replace("self.hc_count", "self.hc_count + 0", 1)
        self.assertNotEqual(changed, source_text["transformers_modeling"])
        source_text = dict(source_text)
        source_text["transformers_modeling"] = changed
        copied_provenance = [dict(item, origin="caller-copy", path="caller.py") for item in provenance]
        with self.assertRaises(upstream.FixtureError):
            upstream.load_pinned_operators(source_text, copied_provenance, object())

    def test_cached_source_hash_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = {
                "name": "fake",
                "project": "owner/project",
                "commit": "deadbeef",
                "path": "src/fake.py",
                "sha256": upstream._sha256_bytes(b"pinned"),
            }
            source_path, metadata_path = upstream._source_cache_paths(root, spec)
            source_path.parent.mkdir(parents=True)
            source_path.write_bytes(b"pinned")
            metadata_path.write_text(
                json.dumps({key: spec[key] for key in ("project", "commit", "path", "sha256")})
            )
            loaded = upstream._load_cached_source(root, spec)
            self.assertIsNotNone(loaded)
            source_path.write_bytes(b"changed")
            with self.assertRaises(upstream.FixtureError):
                upstream._load_cached_source(root, spec)


if __name__ == "__main__":
    unittest.main()
