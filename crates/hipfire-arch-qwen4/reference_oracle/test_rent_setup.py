# SPDX-License-Identifier: Apache-2.0
"""Offline tests for rent_setup: mocked wheel indexes, HTTP status, pins and hashes.

Nothing here touches the network: every HTTP call goes through
``rent_setup._urlopen``, which the tests replace.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import socket
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

import rent_setup as rs

TAGS = [
    "cp312-cp312-manylinux_2_28_x86_64",
    "cp312-cp312-linux_x86_64",
    "cp312-abi3-linux_x86_64",
    "py3-none-any",
]
ROCM = "10.1"


def link(filename: str, sha: str | None = None) -> rs.IndexLink:
    return rs.IndexLink(filename, f"https://idx.example/{filename}", sha)


def torch_links(*specs: str) -> list[rs.IndexLink]:
    return [link(f"torch-{s}.whl") for s in specs]


class _Response:
    def __init__(self, body: bytes, *, status: int = 200, headers: dict[str, str] | None = None, url: str = ""):
        self.status = status
        self.headers = dict(headers or {})
        self._body = body
        self._pos = 0
        self._url = url

    def getcode(self) -> int:
        return self.status

    def geturl(self) -> str:
        return self._url

    def read(self, n: int = -1) -> bytes:
        end = len(self._body) if n is None or n < 0 else self._pos + n
        chunk = self._body[self._pos : end]
        self._pos += len(chunk)
        return chunk

    def close(self) -> None:
        pass


def http_error(code: int, url: str = "https://x/") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, code, "err", {}, None)  # type: ignore[arg-type]


def fake_server(files: dict[str, bytes]):
    """urlopen replacement serving ``files`` with Range support; unknown URL -> 404."""

    def opener(request, timeout):
        url = request.full_url
        if url not in files:
            raise http_error(404, url)
        body = files[url]
        rng = request.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
            if start >= len(body):
                raise http_error(416, url)
            return _Response(
                body[start:],
                status=206,
                headers={"Content-Range": f"bytes {start}-{len(body) - 1}/{len(body)}", "Content-Length": str(len(body) - start)},
                url=url,
            )
        return _Response(body, headers={"Content-Length": str(len(body))}, url=url)

    return opener


class PinTests(unittest.TestCase):
    def test_pins_match_upstream_backend(self) -> None:
        import upstream

        self.assertEqual(rs.HF_REVISION, upstream.HF_CONFIG_COMMIT)
        self.assertEqual(rs.HF_MODEL, upstream.HF_MODEL)
        self.assertEqual(rs.MODEL_CONFIG_SHA256, upstream.MODEL_CONFIG_SHA256)
        self.assertEqual(rs.MODEL_INDEX_SHA256, upstream.MODEL_INDEX_SHA256)
        self.assertEqual(rs.HF_REVISION, "de4b8e4d43b917e7706784d8bb445c9af86a3540")
        self.assertEqual(rs.MODEL_CONFIG_SHA256, "889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b")
        self.assertEqual(rs.MODEL_INDEX_SHA256, "99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de")
        self.assertEqual(rs.EXPECTED_SHARD_COUNT, 131)

    def test_import_does_no_network(self) -> None:
        with mock.patch.object(socket.socket, "connect", side_effect=AssertionError("network at import")), mock.patch(
            "urllib.request.urlopen", side_effect=AssertionError("network at import")
        ):
            importlib.reload(rs)

    def test_cli_defaults(self) -> None:
        args = rs.build_parser().parse_args(["--root", "/tmp/x"])
        self.assertEqual(args.rocm_version, "10.1")
        self.assertEqual(args.torch_version, "2.14.0")
        self.assertEqual(args.fallback_rocm_version, "7.14")
        self.assertEqual(args.fallback_torch_version, "2.14.1")
        self.assertFalse(args.no_fallback)


class WheelSelectionTests(unittest.TestCase):
    def test_highest_compatible_rocm101_wheel_with_best_tag(self) -> None:
        links = torch_links(
            "2.13.0+rocm10.1.0-cp312-cp312-linux_x86_64",
            "2.14.0+rocm10.1.0-cp312-cp312-linux_x86_64",
            "2.14.0+rocm10.1.0-cp312-cp312-manylinux_2_28_x86_64",
            "2.15.0+rocm10.1.0-cp311-cp311-linux_x86_64",  # wrong ABI
            "2.16.0+rocm10.0.0-cp312-cp312-linux_x86_64",  # wrong ROCm
        )
        wheel = rs.select_wheel(links, TAGS, package="torch", rocm_version=ROCM)
        self.assertEqual(wheel.version, "2.14.0+rocm10.1.0")
        self.assertEqual(wheel.tag, "cp312-cp312-manylinux_2_28_x86_64")
        self.assertEqual(wheel.rocm, "10.1.0")

    def test_pinned_version_exact(self) -> None:
        links = torch_links(
            "2.13.0+rocm10.1.0-cp312-cp312-linux_x86_64",
            "2.14.0+rocm10.1.0-cp312-cp312-linux_x86_64",
            "2.15.0+rocm10.1.0-cp312-cp312-linux_x86_64",
        )
        wheel = rs.select_wheel(links, TAGS, package="torch", rocm_version=ROCM, version="2.14.0")
        self.assertEqual(wheel.public, "2.14.0")
        with self.assertRaises(rs.NoCompatibleWheel) as ctx:
            rs.select_wheel(links, TAGS, package="torch", rocm_version=ROCM, version="2.99.0")
        self.assertEqual(ctx.exception.code, "version_unavailable")

    def test_cpu_cuda_and_unlabelled_wheels_rejected(self) -> None:
        links = torch_links(
            "2.14.0+cpu-cp312-cp312-manylinux_2_28_x86_64",
            "2.14.0+cu128-cp312-cp312-manylinux_2_28_x86_64",
            "2.14.0-cp312-cp312-manylinux_2_28_x86_64",
            "2.14.0+rocm10.10.0-cp312-cp312-linux_x86_64",  # 10.10 is not 10.1
        )
        with self.assertRaises(rs.NoCompatibleWheel) as ctx:
            rs.select_wheel(links, TAGS, package="torch", rocm_version=ROCM)
        self.assertEqual(ctx.exception.code, "no_rocm_wheel")

    def test_no_abi_wheel(self) -> None:
        links = torch_links(
            "2.14.0+rocm10.1.0-cp312-cp312-win_amd64",
            "2.14.0+rocm10.1.0-cp310-cp310-linux_x86_64",
        )
        with self.assertRaises(rs.NoCompatibleWheel) as ctx:
            rs.select_wheel(links, TAGS, package="torch", rocm_version=ROCM)
        self.assertEqual(ctx.exception.code, "no_abi_compatible_wheel")

    def test_prerelease_and_yanked_ignored(self) -> None:
        links = torch_links("2.15.0.dev20260101+rocm10.1.0-cp312-cp312-linux_x86_64", "2.14.0+rocm10.1.0-cp312-cp312-linux_x86_64")
        links.append(rs.IndexLink("torch-2.20.0+rocm10.1.0-cp312-cp312-linux_x86_64.whl", "https://idx/y", None, yanked=True))
        self.assertEqual(rs.select_wheel(links, TAGS, package="torch", rocm_version=ROCM).public, "2.14.0")

    def test_device_wheel_must_match_torch_label(self) -> None:
        links = [link("amd_torch_device_gfx942-2.14.0+rocm10.1.0-cp312-cp312-linux_x86_64.whl")]
        wheel = rs.select_wheel(
            links, TAGS, package="amd-torch-device-gfx942", rocm_version=ROCM, version="2.14.0", local="rocm10.1.0"
        )
        self.assertEqual(wheel.package, "amd_torch_device_gfx942")
        with self.assertRaises(rs.NoCompatibleWheel):
            rs.select_wheel(links, TAGS, package="amd-torch-device-gfx942", rocm_version=ROCM, version="2.14.0", local="rocm10.1.1")

    def test_index_html_parsing(self) -> None:
        page = rs.parse_index_html(
            "https://stable.example/rocm/pytorch/whl-next/torch/",
            '<a href="torch-2.14.0%2Brocm10.1.0-cp312-cp312-linux_x86_64.whl">x</a><br/>'
            f'<a href="https://cdn.example/torch-2.14.1%2Brocm7.14-cp312-cp312-manylinux_2_28_x86_64.whl#sha256={"ab" * 32}">y</a>',
        )
        self.assertEqual(page.links[0].filename, "torch-2.14.0+rocm10.1.0-cp312-cp312-linux_x86_64.whl")
        self.assertEqual(page.links[0].url, "https://stable.example/rocm/pytorch/whl-next/torch/torch-2.14.0%2Brocm10.1.0-cp312-cp312-linux_x86_64.whl")
        self.assertIsNone(page.links[0].sha256)
        self.assertEqual(page.links[1].sha256, "ab" * 32)
        self.assertNotIn("#", page.links[1].url)


def amd_pages(*, torch: list[str] | None = None, device: list[str] | None = None, core: list[str] | None = None):
    good = "2.14.0+rocm10.1.0-cp312-cp312-linux_x86_64"
    pages = {
        "torch": [f"torch-{s}.whl" for s in (torch if torch is not None else [good])],
        "amd-torch-device-gfx942": [f"amd_torch_device_gfx942-{s}.whl" for s in (device if device is not None else [good])],
        "rocm": core if core is not None else ["rocm-10.0.0.tar.gz", "rocm-10.1.0.tar.gz"],
    }
    return pages


class ResolveTests(unittest.TestCase):
    def make_fetch(self, amd, fallback=None, *, amd_error=None, fallback_error=None):
        calls: list[tuple[str, str]] = []

        def fetch(index_url: str, package: str) -> rs.IndexPage:
            calls.append((index_url, package))
            is_amd = index_url == rs.AMD_INDEX_URL
            err = amd_error if is_amd else fallback_error
            if err is not None:
                raise err
            pages = amd if is_amd else fallback
            return rs.IndexPage(index_url, tuple(link(name) for name in pages[package]))

        return fetch, calls

    def resolve(self, fetch, **kw):
        return rs.resolve_torch(fetch, TAGS, **kw)

    FALLBACK = {"torch": ["torch-2.14.1+rocm7.14-cp312-cp312-manylinux_2_28_x86_64.whl"]}

    def test_primary_amd_stack_no_fallback(self) -> None:
        fetch, calls = self.make_fetch(amd_pages(), self.FALLBACK)
        plan, record = self.resolve(fetch)
        self.assertEqual(plan.source, "amd")
        self.assertEqual(plan.torch.version, "2.14.0+rocm10.1.0")
        self.assertEqual(plan.extras, ("device-gfx942",))
        self.assertEqual(plan.requirements, ("rocm[libraries,device-gfx942]==10.1.0",))
        self.assertFalse(record["fallback_used"])
        self.assertTrue(all(index == rs.AMD_INDEX_URL for index, _ in calls))

    def test_index_404_falls_back_with_reason(self) -> None:
        fetch, calls = self.make_fetch(amd_pages(), self.FALLBACK, amd_error=rs.NotFound("HTTP 404"))
        plan, record = self.resolve(fetch)
        self.assertEqual(plan.source, "fallback")
        self.assertEqual(plan.index_url, "https://download.pytorch.org/whl/rocm7.14/")
        self.assertEqual(plan.torch.version, "2.14.1+rocm7.14")
        self.assertTrue(record["fallback_used"])
        self.assertEqual(record["fallback_reason"]["code"], "index_404")
        self.assertEqual(record["fallback_rocm"], "7.14")

    def test_no_abi_wheel_falls_back(self) -> None:
        pages = amd_pages(torch=["2.14.0+rocm10.1.0-cp310-cp310-linux_x86_64"])
        plan, record = self.resolve(self.make_fetch(pages, self.FALLBACK)[0])
        self.assertEqual(plan.source, "fallback")
        self.assertEqual(record["fallback_reason"]["code"], "no_abi_compatible_wheel")

    def test_missing_device_package_or_core_falls_back(self) -> None:
        for pages, code in (
            (amd_pages(device=["2.14.0+rocm10.1.0-cp310-cp310-linux_x86_64"]), "no_abi_compatible_wheel"),
            (amd_pages(core=["rocm-10.0.0.tar.gz"]), "no_core_distribution"),
        ):
            plan, record = self.resolve(self.make_fetch(pages, self.FALLBACK)[0])
            self.assertEqual(plan.source, "fallback")
            self.assertEqual(record["fallback_reason"]["code"], code)

    def test_network_error_never_falls_back(self) -> None:
        fetch, calls = self.make_fetch(amd_pages(), self.FALLBACK, amd_error=rs.NetworkError("connection reset"))
        with self.assertRaises(rs.NetworkError):
            self.resolve(fetch)
        self.assertTrue(all(index == rs.AMD_INDEX_URL for index, _ in calls))

    def test_fallback_network_error_propagates(self) -> None:
        fetch, _ = self.make_fetch(amd_pages(), self.FALLBACK, amd_error=rs.NotFound("404"), fallback_error=rs.NetworkError("dns"))
        with self.assertRaises(rs.NetworkError):
            self.resolve(fetch)

    def test_fallback_disabled_or_unavailable_fails(self) -> None:
        fetch, _ = self.make_fetch(amd_pages(), self.FALLBACK, amd_error=rs.NotFound("404"))
        with self.assertRaises(rs.SetupError):
            self.resolve(fetch, fallback_rocm_version=None)
        cpu_only = {"torch": ["torch-2.14.1+cpu-cp312-cp312-manylinux_2_28_x86_64.whl"]}
        fetch, _ = self.make_fetch(amd_pages(), cpu_only, amd_error=rs.NotFound("404"))
        with self.assertRaises(rs.NoCompatibleWheel):
            self.resolve(fetch)

    def test_fetch_index_page_error_classification(self) -> None:
        with mock.patch.object(rs, "_sleep"):
            with mock.patch.object(rs, "_urlopen", side_effect=http_error(404)):
                with self.assertRaises(rs.NotFound):
                    rs.fetch_index_page("https://idx.example/", "torch")
            for exc in (urllib.error.URLError("unreachable"), http_error(503), http_error(429), socket.timeout("slow")):
                with mock.patch.object(rs, "_urlopen", side_effect=exc) as opener:
                    with self.assertRaises(rs.NetworkError):
                        rs.fetch_index_page("https://idx.example/", "torch")
                    self.assertEqual(opener.call_count, 4)  # retried, then fail closed

    def test_fetch_index_page_resolves_redirected_relative_links(self) -> None:
        body = b'<a href="torch-2.14.0%2Brocm10.1.0-cp312-cp312-linux_x86_64.whl">x</a>'
        final = "https://stable.example/rocm/pytorch/whl-next/torch/"
        with mock.patch.object(rs, "_urlopen", return_value=_Response(body, url=final)):
            page = rs.fetch_index_page("https://stable.example/rocm/whl-next/", "torch")
        self.assertTrue(page.links[0].url.startswith(final))


class DownloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        patcher = mock.patch.object(rs, "_sleep")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_download_verifies_and_stats(self) -> None:
        body = bytes(range(256)) * 100
        url = "https://h.example/f.bin"
        stats = rs.NetStats()
        with mock.patch.object(rs, "_urlopen", fake_server({url: body})):
            record = rs.download_file(url, self.dir / "f.bin", expected_size=len(body), expected_sha256=hashlib.sha256(body).hexdigest(), stats=stats)
        self.assertEqual((self.dir / "f.bin").read_bytes(), body)
        self.assertFalse((self.dir / "f.bin.part").exists())
        self.assertEqual(record["origin"], "downloaded")
        self.assertEqual(stats.summary()["total_bytes_downloaded"], len(body))

    def test_resume_from_part(self) -> None:
        body = bytes(range(256)) * 100
        url = "https://h.example/f.bin"
        (self.dir / "f.bin.part").write_bytes(body[:1000])
        with mock.patch.object(rs, "_urlopen", fake_server({url: body})):
            record = rs.download_file(url, self.dir / "f.bin", expected_size=len(body), expected_sha256=hashlib.sha256(body).hexdigest())
        self.assertEqual((self.dir / "f.bin").read_bytes(), body)
        self.assertEqual(record["bytes_resumed_from"], 1000)
        self.assertEqual(record["bytes_downloaded"], len(body) - 1000)

    def test_hash_mismatch_rejected_and_removed(self) -> None:
        body = b"shard-bytes" * 50
        url = "https://h.example/s.safetensors"
        with mock.patch.object(rs, "_urlopen", fake_server({url: body})):
            with self.assertRaises(rs.HashMismatch):
                rs.download_file(url, self.dir / "s.safetensors", expected_size=len(body), expected_sha256="0" * 64)
        self.assertFalse((self.dir / "s.safetensors").exists())
        self.assertFalse((self.dir / "s.safetensors.part").exists())

    def test_server_size_mismatch_rejected(self) -> None:
        body = b"abc" * 10
        url = "https://h.example/s.bin"
        with mock.patch.object(rs, "_urlopen", fake_server({url: body})):
            with self.assertRaises(rs.SetupError):
                rs.download_file(url, self.dir / "s.bin", expected_size=len(body) + 1)

    def test_corrupt_existing_file_is_replaced(self) -> None:
        body = b"good" * 100
        url = "https://h.example/s.bin"
        (self.dir / "s.bin").write_bytes(b"bad!" * 100)
        with mock.patch.object(rs, "_urlopen", fake_server({url: body})):
            record = rs.download_file(url, self.dir / "s.bin", expected_size=len(body), expected_sha256=hashlib.sha256(body).hexdigest())
        self.assertEqual(record["origin"], "downloaded")
        self.assertEqual((self.dir / "s.bin").read_bytes(), body)

    def test_valid_existing_file_reused_without_network(self) -> None:
        body = b"good" * 100
        (self.dir / "s.bin").write_bytes(body)
        with mock.patch.object(rs, "_urlopen", side_effect=AssertionError("network used")):
            record = rs.download_file("https://h.example/s.bin", self.dir / "s.bin", expected_size=len(body), expected_sha256=hashlib.sha256(body).hexdigest())
        self.assertEqual(record["origin"], "existing_verified")

    def test_persistent_network_error_fails_closed(self) -> None:
        with mock.patch.object(rs, "_urlopen", side_effect=urllib.error.URLError("down")):
            with self.assertRaises(rs.NetworkError):
                rs.download_file("https://h.example/s.bin", self.dir / "s.bin", expected_size=10)


def synthetic_repo(shards: int = 3, *, drop_lfs_for: str | None = None):
    """(index_bytes, shard bodies, siblings) where siblings mimic the HF API."""
    names = [f"model-{i:05d}-of-{shards:05d}.safetensors" for i in range(shards)]
    bodies = {name: (name.encode() * 40) for name in names}
    index = json.dumps({"weight_map": {f"t{i}": name for i, name in enumerate(names)}}).encode()
    config = b'{"model_type": "qwen"}'
    tokenizer = b'{"version": "1.0"}'
    siblings = {
        "config.json": {"size": len(config), "blob_id": rs.git_blob_sha1(config), "lfs_sha256": None, "lfs_size": None},
        rs.INDEX_NAME: {"size": len(index), "blob_id": rs.git_blob_sha1(index), "lfs_sha256": None, "lfs_size": None},
        "tokenizer.json": {"size": len(tokenizer), "blob_id": rs.git_blob_sha1(tokenizer), "lfs_sha256": None, "lfs_size": None},
        "README.md": {"size": 3, "blob_id": "a" * 40, "lfs_sha256": None, "lfs_size": None},
    }
    for name, body in bodies.items():
        siblings[name] = {
            "size": len(body),
            "blob_id": "b" * 40,
            "lfs_sha256": None if name == drop_lfs_for else hashlib.sha256(body).hexdigest(),
            "lfs_size": len(body),
        }
    return index, config, tokenizer, bodies, siblings


class PlanTests(unittest.TestCase):
    def patched(self, index: bytes, config: bytes, shards: int):
        return mock.patch.multiple(
            rs,
            MODEL_INDEX_SHA256=hashlib.sha256(index).hexdigest(),
            MODEL_CONFIG_SHA256=hashlib.sha256(config).hexdigest(),
            EXPECTED_SHARD_COUNT=shards,
        )

    def test_plan_downloads_exact_shards_and_tokenizer_only(self) -> None:
        index, config, _, bodies, siblings = synthetic_repo()
        siblings["model-99999-of-99999.safetensors"] = dict(siblings[next(iter(bodies))])  # not in index
        with self.patched(index, config, 3):
            specs = rs.plan_model_files(siblings, index)
        names = {s.name for s in specs}
        self.assertEqual({s.name for s in specs if s.role == "shard"}, set(bodies))
        self.assertIn("tokenizer.json", names)
        self.assertNotIn("README.md", names)
        self.assertNotIn("model-99999-of-99999.safetensors", names)

    def test_index_hash_must_match_pin(self) -> None:
        index, config, _, _, siblings = synthetic_repo()
        with self.patched(index, config, 3):
            with self.assertRaises(rs.HashMismatch):
                rs.plan_model_files(siblings, index + b" ")

    def test_wrong_shard_count_rejected(self) -> None:
        index, config, _, _, siblings = synthetic_repo()
        with self.patched(index, config, 131):
            with self.assertRaises(rs.SetupError):
                rs.plan_model_files(siblings, index)

    def test_shard_without_lfs_hash_rejected(self) -> None:
        index, config, _, bodies, siblings = synthetic_repo(drop_lfs_for="model-00001-of-00003.safetensors")
        with self.patched(index, config, 3):
            with self.assertRaisesRegex(rs.SetupError, "LFS sha256"):
                rs.plan_model_files(siblings, index)

    def test_missing_shard_or_tokenizer_rejected(self) -> None:
        index, config, _, bodies, siblings = synthetic_repo()
        with self.patched(index, config, 3):
            missing = dict(siblings)
            del missing["model-00002-of-00003.safetensors"]
            with self.assertRaises(rs.SetupError):
                rs.plan_model_files(missing, index)
            no_tok = dict(siblings)
            del no_tok["tokenizer.json"]
            with self.assertRaisesRegex(rs.SetupError, "tokenizer"):
                rs.plan_model_files(no_tok, index)

    def test_unsafe_shard_name_rejected(self) -> None:
        index = json.dumps({"weight_map": {"a": "../evil.safetensors"}}).encode()
        _, config, _, _, siblings = synthetic_repo()
        with mock.patch.multiple(rs, MODEL_INDEX_SHA256=hashlib.sha256(index).hexdigest(), EXPECTED_SHARD_COUNT=1):
            with self.assertRaisesRegex(rs.SetupError, "unsafe"):
                rs.plan_model_files(siblings, index)

    def test_api_revision_must_be_pinned(self) -> None:
        with self.assertRaises(rs.SetupError):
            rs.parse_siblings({"sha": "main", "siblings": []})
        self.assertEqual(rs.parse_siblings({"sha": rs.HF_REVISION, "siblings": []}), {})


class StageModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        patcher = mock.patch.object(rs, "_sleep")
        patcher.start()
        self.addCleanup(patcher.stop)

    def serve(self, shards: int = 3, *, corrupt: str | None = None):
        index, config, tokenizer, bodies, siblings = synthetic_repo(shards)
        files = {
            rs.hf_resolve_url("config.json"): config,
            rs.hf_resolve_url(rs.INDEX_NAME): index,
            rs.hf_resolve_url("tokenizer.json"): tokenizer,
        }
        for name, body in bodies.items():
            files[rs.hf_resolve_url(name)] = body[::-1] if name == corrupt else body
        patch = mock.patch.multiple(
            rs,
            MODEL_INDEX_SHA256=hashlib.sha256(index).hexdigest(),
            MODEL_CONFIG_SHA256=hashlib.sha256(config).hexdigest(),
            EXPECTED_SHARD_COUNT=shards,
        )
        return files, siblings, patch

    def test_stage_writes_hash_verified_manifest(self) -> None:
        files, siblings, patch = self.serve()
        with patch, mock.patch.object(rs, "_urlopen", fake_server(files)):
            result = rs.stage_model(self.root, siblings, rs.NetStats(), 2, disk=lambda specs: None)
        manifest = json.loads((self.root / "source" / "source-manifest.json").read_text())
        self.assertEqual(result["shard_count"], 3)
        self.assertEqual(manifest["revision"], rs.HF_REVISION)
        for entry in manifest["files"]:
            on_disk = hashlib.sha256((self.root / "source" / rs.HF_REVISION / entry["name"]).read_bytes()).hexdigest()
            self.assertEqual(entry["sha256"], on_disk)
            if entry["role"] == "shard":
                self.assertEqual(entry["sha256"], entry["pinned_sha256"])

    def test_shard_hash_mismatch_rejects_setup(self) -> None:
        files, siblings, patch = self.serve(corrupt="model-00001-of-00003.safetensors")
        with patch, mock.patch.object(rs, "_urlopen", fake_server(files)):
            with self.assertRaises(rs.HashMismatch):
                rs.stage_model(self.root, siblings, rs.NetStats(), 2, disk=lambda specs: None)
        self.assertFalse((self.root / "source" / rs.HF_REVISION / "model-00001-of-00003.safetensors").exists())
        self.assertFalse((self.root / "source" / "source-manifest.json").exists())

    def test_config_hash_pin_enforced(self) -> None:
        files, siblings, patch = self.serve()
        files[rs.hf_resolve_url("config.json")] = b'{"model_type": "qwex"}'  # same length, different bytes
        with patch, mock.patch.object(rs, "_urlopen", fake_server(files)):
            with self.assertRaises(rs.HashMismatch):
                rs.stage_model(self.root, siblings, rs.NetStats(), 1, disk=lambda specs: None)

    def test_disk_check_runs_before_shard_download(self) -> None:
        files, siblings, patch = self.serve()

        def reject(specs):
            raise rs.SetupError("insufficient disk")

        with patch, mock.patch.object(rs, "_urlopen", fake_server(files)):
            with self.assertRaisesRegex(rs.SetupError, "insufficient disk"):
                rs.stage_model(self.root, siblings, rs.NetStats(), 1, disk=reject)
        self.assertEqual(list((self.root / "source" / rs.HF_REVISION).glob("*.safetensors")), [])


class DiskTests(unittest.TestCase):
    def test_warns_above_600gb_but_allows_if_capacity(self) -> None:
        plan = rs.disk_plan(model_bytes=500 * 10**9, already_staged_bytes=0, corpus_bytes=100 * 10**9, output_bytes=50 * 10**9, free_bytes=10**13)
        self.assertGreater(plan["estimate_bytes"], rs.WARN_ESTIMATE_BYTES)
        self.assertEqual(len(plan["warnings"]), 1)

    def test_rejects_insufficient_capacity(self) -> None:
        with self.assertRaisesRegex(rs.SetupError, "insufficient disk"):
            rs.disk_plan(model_bytes=500 * 10**9, already_staged_bytes=0, corpus_bytes=0, output_bytes=0, free_bytes=400 * 10**9)

    def test_staged_bytes_reduce_requirement(self) -> None:
        plan = rs.disk_plan(model_bytes=500 * 10**9, already_staged_bytes=480 * 10**9, corpus_bytes=0, output_bytes=0, free_bytes=100 * 10**9)
        self.assertEqual(plan["still_needed_bytes"], 20 * 10**9 + rs.VENV_RESERVE_BYTES)


def good_runtime(**over):
    info = {
        "torch_version": "2.14.0+rocm10.1.0",
        "hip": "7.2.26015",
        "cuda": None,
        "cuda_available": True,
        "device_count": 1,
        "devices": [{"index": 0, "name": "AMD Instinct MI300X", "gcn_arch": "gfx942:sramecc+:xnack-", "total_memory": 192 << 30}],
        "distributions": {"torch": "2.14.0+rocm10.1.0", "rocm": "10.1.0", "amd-torch-device-gfx942": "2.14.0+rocm10.1.0"},
        "bf16_matmul_sum": float(64**3),
    }
    info.update(over)
    return info


class RuntimeTests(unittest.TestCase):
    def amd_plan(self) -> rs.TorchPlan:
        fetch = lambda index, pkg: rs.IndexPage(  # noqa: E731
            index, tuple(link(n) for n in amd_pages()[pkg])
        )
        return rs.resolve_torch(fetch, TAGS)[0]

    def test_good_rocm_runtime_records_raw_hip_without_requiring_equality(self) -> None:
        record = rs.validate_runtime(good_runtime(), self.amd_plan())
        self.assertEqual(record["raw_torch_version_hip"], "7.2.26015")
        self.assertFalse(record["hip_major_minor_equals_rocm_build"])
        self.assertIn("differs", record["note"])

    def test_cpu_build_rejected(self) -> None:
        with self.assertRaisesRegex(rs.SetupError, "not a ROCm build"):
            rs.validate_runtime(good_runtime(hip=None, cuda_available=False, devices=[], device_count=0), self.amd_plan())

    def test_cuda_build_rejected(self) -> None:
        with self.assertRaisesRegex(rs.SetupError, "CUDA build"):
            rs.validate_runtime(good_runtime(cuda="12.8"), self.amd_plan())

    def test_no_gpu_and_wrong_arch_rejected(self) -> None:
        with self.assertRaisesRegex(rs.SetupError, "no CPU fallback"):
            rs.validate_runtime(good_runtime(cuda_available=False, devices=[]), self.amd_plan())
        wrong = [{"index": 0, "name": "AMD Radeon", "gcn_arch": "gfx1100", "total_memory": 1}]
        with self.assertRaisesRegex(rs.SetupError, "gfx942"):
            rs.validate_runtime(good_runtime(devices=wrong), self.amd_plan())

    def test_core_and_device_distribution_identity_enforced(self) -> None:
        bad = good_runtime(distributions={"torch": "2.14.0+rocm10.1.0", "rocm": "10.0.0", "amd-torch-device-gfx942": None})
        with self.assertRaisesRegex(rs.SetupError, "rocm distribution"):
            rs.validate_runtime(bad, self.amd_plan())

    def test_wrong_torch_wheel_installed_rejected(self) -> None:
        with self.assertRaisesRegex(rs.SetupError, "selected wheel"):
            rs.validate_runtime(good_runtime(torch_version="2.14.0+cpu"), self.amd_plan())

    def test_bf16_sanity_failure_rejected(self) -> None:
        with self.assertRaisesRegex(rs.SetupError, "bf16"):
            rs.validate_runtime(good_runtime(bf16_matmul_sum=0.0), self.amd_plan())


class HostRocmTests(unittest.TestCase):
    def test_match_mismatch_unknown(self) -> None:
        self.assertEqual(rs.check_host_rocm({"rocm_version_mm": "10.1"}, "10.1.0", allow_mismatch=False)["status"], "match")
        with self.assertRaises(rs.SetupError):
            rs.check_host_rocm({"rocm_version_mm": "7.2"}, "10.1.0", allow_mismatch=False)
        with self.assertRaises(rs.SetupError):
            rs.check_host_rocm({"rocm_version_mm": None}, "10.1.0", allow_mismatch=False)
        record = rs.check_host_rocm({"rocm_version_mm": "7.2"}, "10.1.0", allow_mismatch=True)
        self.assertEqual((record["status"], record["allowed_by_flag"]), ("mismatch", True))

    def test_version_parsing_sources(self) -> None:
        self.assertEqual(rs.host_rocm_version({"rocm_info_version": "10.1.0-123"}), (10, 1))
        probe = {"commands": {"hipcc_version": {"stdout": "HIP version: 7.2.26015-abc\n"}}}
        self.assertEqual(rs.host_rocm_version(probe), (7, 2))
        self.assertIsNone(rs.host_rocm_version({"commands": {}}))

    def test_probe_host_records_commands_without_tools(self) -> None:
        run = mock.Mock(return_value={"argv": [], "found": False})
        info = rs.probe_host(run)
        self.assertIn("hipcc_version", info["commands"])
        self.assertIn("rocminfo", info["commands"])


class SourceLoaderTests(unittest.TestCase):
    def payload(self, **over):
        spec = {"name": "a", "project": "p", "commit": "c" * 40, "path": "x.py", "sha256": "1" * 64}
        data = {
            "expected": [spec],
            "provenance": [dict(spec, origin="downloaded_bounded_cache")],
            "hf_model": rs.HF_MODEL,
            "hf_config_commit": rs.HF_REVISION,
            "model_config_sha256": rs.MODEL_CONFIG_SHA256,
            "model_index_sha256": rs.MODEL_INDEX_SHA256,
        }
        data.update(over)
        return data

    def test_verify_accepts_consistent_payload(self) -> None:
        rs.verify_source_loader(self.payload())

    def test_verify_rejects_hash_or_pin_drift(self) -> None:
        bad = self.payload()
        bad["provenance"][0]["sha256"] = "2" * 64
        with self.assertRaises(rs.SetupError):
            rs.verify_source_loader(bad)
        with self.assertRaises(rs.SetupError):
            rs.verify_source_loader(self.payload(model_index_sha256="0" * 64))
        with self.assertRaises(rs.SetupError):
            rs.verify_source_loader(self.payload(provenance=[]))


if __name__ == "__main__":
    unittest.main()
