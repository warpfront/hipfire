#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fresh-rental setup for the device-agnostic Flash-Next BF16 teacher.

Run once on a freshly rented ROCm GPU box (MI300X / gfx942)::

    python3 rent_setup.py --root /mnt/work/flash-next

The script uses only the Python standard library, so it bootstraps on a clean
image.  Nothing here touches the network at import time; every network access
happens inside :func:`main`.

Stages (each one rewrites ``<root>/setup-manifest.json`` with fsync + atomic
rename, so a crash always leaves a parseable manifest):

1. record host/OS/ROCm toolchain state;
2. create a private venv at ``<root>/venv``;
3. resolve the ROCm PyTorch stack for the requested ROCm major.minor
   (default 10.1) from AMD's official index: ``torch`` + the split
   ``amd-torch-device-gfx942`` package + the ``rocm`` core distribution must all
   exist at the same ROCm build and be ABI-compatible with the venv interpreter.
   Only a missing index / missing or ABI-incompatible wheel (HTTP 404 or empty
   candidate set) may fall back, and only to the explicit, recorded
   ``--fallback-rocm-version`` / ``--fallback-torch-version`` PyTorch wheel.
   Network failures never fall back; they abort;
4. check disk capacity against the model + corpus + output estimate;
5. download the wheels (sha256 recorded, verified against index hash fragments
   when present), ``pip install`` them into the venv, then verify on the
   rental that torch is a ROCm build (``torch.version.hip`` set,
   ``torch.version.cuda`` unset), CUDA/HIP is available, every device is gfx942,
   and a bf16 matmul runs on the GPU;
6. stage the pinned Hugging Face revision (config + index hash-pinned, tokenizer
   files, all 131 shards) and verify every shard's full sha256 against the LFS
   sha256 reported by the HF API for that exact commit;
7. load the hash-pinned upstream operator sources through
   ``upstream.load_pinned_sources`` inside the venv.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import html.parser
import http.client
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

SCHEMA = "hipfire.flash_next.rent_setup.v1"

# ---------------------------------------------------------------------------
# Pins
# ---------------------------------------------------------------------------
HF_MODEL = "Qwen/Qwen3.8-Flash-Next"
HF_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
HF_ENDPOINT = "https://huggingface.co"
MODEL_CONFIG_SHA256 = "889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b"
MODEL_INDEX_SHA256 = "99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de"
EXPECTED_SHARD_COUNT = 131
CONFIG_NAME = "config.json"
INDEX_NAME = "model.safetensors.index.json"
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "generation_config.json",
    "chat_template.jinja",
    "chat_template.json",
    "preprocessor_config.json",
)
SHARD_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.safetensors$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")

# Official AMD ROCm PyTorch index (redirects into /rocm/pytorch/ and /rocm/core/).
AMD_INDEX_URL = "https://stable.repo.amd.com/rocm/whl-next/"
DEFAULT_ROCM_VERSION = "10.1"
DEFAULT_TORCH_VERSION = "2.14.0"
DEVICE_ARCH = "gfx942"
FALLBACK_INDEX_TEMPLATE = "https://download.pytorch.org/whl/rocm{version}/"
DEFAULT_FALLBACK_ROCM_VERSION = "7.14"
DEFAULT_FALLBACK_TORCH_VERSION = "2.14.1"

MIN_PYTHON = (3, 10)
WARN_ESTIMATE_BYTES = 600 * 10**9
VENV_RESERVE_BYTES = 40 * 10**9  # venv + wheels + pip cache headroom
HTTP_TIMEOUT_SECONDS = 60
INDEX_MAX_BYTES = 32 << 20
API_MAX_BYTES = 64 << 20
SMALL_FILE_MAX_BYTES = 64 << 20
STREAM_CHUNK = 1 << 20
HASH_CHUNK = 8 << 20
DOWNLOAD_ATTEMPTS = 5
DEFAULT_JOBS = 8
PROBE_OUTPUT_LIMIT = 64 << 10


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class SetupError(RuntimeError):
    """Fatal setup failure."""


class NetworkError(SetupError):
    """Transport failure or transient HTTP status (never triggers a fallback)."""


class NotFound(SetupError):
    """HTTP 404."""


class HashMismatch(SetupError):
    """A downloaded or staged file does not match its pinned hash."""


class NoCompatibleWheel(SetupError):
    """The index was readable but offers no acceptable wheel."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


# ---------------------------------------------------------------------------
# HTTP layer (the only place that touches the network; tests patch _urlopen)
# ---------------------------------------------------------------------------
class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    """Never forward credentials to a different host on redirect."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlparse(newurl).netloc != urllib.parse.urlparse(req.full_url).netloc:
            new.headers.pop("Authorization", None)
            new.unredirected_hdrs.pop("Authorization", None)
        return new


_OPENER: urllib.request.OpenerDirector | None = None
_sleep: Callable[[float], None] = time.sleep


def _urlopen(request: urllib.request.Request, timeout: float):
    global _OPENER
    if _OPENER is None:
        _OPENER = urllib.request.build_opener(_SafeRedirect)
    return _OPENER.open(request, timeout=timeout)


def http_open(url: str, *, headers: Mapping[str, str] | None = None, passthrough: Sequence[int] = ()):
    """Open ``url``; classify failures as NotFound / NetworkError / SetupError."""
    merged = {"Accept-Encoding": "identity", "User-Agent": "hipfire-rent-setup/1"}
    merged.update(headers or {})
    request = urllib.request.Request(url, headers=merged)
    try:
        return _urlopen(request, HTTP_TIMEOUT_SECONDS)
    except urllib.error.HTTPError as exc:
        if exc.code in passthrough:
            return exc
        if exc.code == 404:
            raise NotFound(f"HTTP 404 for {url}") from exc
        if exc.code in (401, 403):
            raise SetupError(f"HTTP {exc.code} for {url}: access denied (gated model? set HF_TOKEN)") from exc
        raise NetworkError(f"HTTP {exc.code} for {url}") from exc
    except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
        raise NetworkError(f"network error for {url}: {exc}") from exc


def _status(response: Any) -> int:
    status = getattr(response, "status", None)
    return int(status if status is not None else response.getcode())


def _retrying(fn: Callable[[], Any], *, attempts: int = 4) -> Any:
    for index in range(attempts):
        try:
            return fn()
        except NetworkError:
            if index == attempts - 1:
                raise
            _sleep(min(2**index, 8))


def http_get_bytes(url: str, *, headers: Mapping[str, str] | None = None, max_bytes: int) -> tuple[bytes, str]:
    """GET with retry on network errors; returns (body, final_url)."""

    def once() -> tuple[bytes, str]:
        response = http_open(url, headers=headers)
        try:
            if _status(response) != 200:
                raise NetworkError(f"HTTP {_status(response)} for {url}")
            body = response.read(max_bytes + 1)
            final = response.geturl() if hasattr(response, "geturl") else url
        except (OSError, http.client.HTTPException) as exc:
            raise NetworkError(f"read failed for {url}: {exc}") from exc
        finally:
            close = getattr(response, "close", None)
            if close:
                close()
        if len(body) > max_bytes:
            raise SetupError(f"response for {url} exceeds {max_bytes} bytes")
        return body, final

    return _retrying(once)


# ---------------------------------------------------------------------------
# Network accounting
# ---------------------------------------------------------------------------
class NetStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.entries: list[dict[str, Any]] = []
        self.started = time.monotonic()

    def record(self, label: str, nbytes: int, seconds: float, **extra: Any) -> None:
        entry = {
            "label": label,
            "bytes": nbytes,
            "seconds": round(seconds, 3),
            "mb_per_s": round(nbytes / seconds / 1e6, 2) if seconds > 0 and nbytes else None,
        }
        entry.update(extra)
        with self._lock:
            self.entries.append(entry)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            total = sum(int(e["bytes"]) for e in self.entries)
            busy = sum(float(e["seconds"]) for e in self.entries)
            wall = time.monotonic() - self.started
        return {
            "total_bytes_downloaded": total,
            "sum_transfer_seconds": round(busy, 3),
            "wall_seconds": round(wall, 3),
            "aggregate_mb_per_s_wall": round(total / wall / 1e6, 2) if wall > 0 and total else None,
            "transfers": len(self.entries),
        }


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------
def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def hash_file(path: Path, *, git_blob: bool = False) -> tuple[str, str | None, int]:
    """Return (sha256, git-blob sha1 or None, size) streaming from disk."""
    size = path.stat().st_size
    sha = hashlib.sha256()
    git = hashlib.sha1(b"blob %d\0" % size) if git_blob else None
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(HASH_CHUNK)
            if not chunk:
                break
            sha.update(chunk)
            if git is not None:
                git.update(chunk)
    return sha.hexdigest(), git.hexdigest() if git is not None else None, size


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    if entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                    else:
                        total += entry.stat(follow_symlinks=False).st_size
        except FileNotFoundError:
            continue
    return total


# ---------------------------------------------------------------------------
# Resumable, hash-verified download
# ---------------------------------------------------------------------------
_CONTENT_RANGE_RE = re.compile(r"^bytes (\d+)-(\d+)/(\d+|\*)$")
_UNSATISFIED_RE = re.compile(r"^bytes \*/(\d+)$")


def _header(response: Any, name: str) -> str | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get(name)
    return None if value is None else str(value)


def _stream_into_part(url: str, part: Path, headers: Mapping[str, str], expected_size: int | None) -> int:
    """One streaming attempt that appends to ``part``; returns bytes written."""
    offset = part.stat().st_size if part.exists() else 0
    if expected_size is not None and offset > expected_size:
        part.unlink()
        offset = 0
    if expected_size is not None and offset == expected_size:
        return 0
    request_headers = dict(headers)
    if offset:
        request_headers["Range"] = f"bytes={offset}-"
    response = http_open(url, headers=request_headers, passthrough=(416,))
    try:
        status = _status(response)
        total: int | None = expected_size
        mode = "ab"
        if status == 416:
            match = _UNSATISFIED_RE.match(_header(response, "Content-Range") or "")
            if match and int(match.group(1)) == offset:
                return 0
            part.unlink(missing_ok=True)
            raise NetworkError(f"range not satisfiable for {url}; restarting from zero")
        if status == 206:
            match = _CONTENT_RANGE_RE.match(_header(response, "Content-Range") or "")
            if not match or int(match.group(1)) != offset:
                raise SetupError(f"unexpected Content-Range {_header(response, 'Content-Range')!r} for {url}")
            if match.group(3) != "*":
                total = int(match.group(3))
        elif status == 200:
            mode = "wb"
            offset = 0
            length = _header(response, "Content-Length")
            if length is not None:
                total = int(length)
        else:
            raise NetworkError(f"HTTP {status} for {url}")
        if expected_size is not None and total is not None and total != expected_size:
            raise SetupError(f"size mismatch for {url}: server reports {total}, expected {expected_size}")
        written = 0
        try:
            with open(part, mode) as handle:
                while True:
                    chunk = response.read(STREAM_CHUNK)
                    if not chunk:
                        break
                    handle.write(chunk)
                    written += len(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        except (OSError, http.client.HTTPException) as exc:
            if isinstance(exc, OSError) and exc.errno in (28, 122):  # ENOSPC / EDQUOT
                raise SetupError(f"disk full while writing {part}: {exc}") from exc
            raise NetworkError(f"stream interrupted for {url}: {exc}") from exc
        if total is not None and offset + written != total:
            raise NetworkError(f"short read for {url}: have {offset + written} of {total}")
        return written
    finally:
        close = getattr(response, "close", None)
        if close:
            close()


def download_file(
    url: str,
    dest: Path,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    expected_git_sha1: str | None = None,
    headers: Mapping[str, str] | None = None,
    stats: NetStats | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """Download ``url`` to ``dest`` atomically; verify size and hashes from disk.

    A matching existing ``dest`` is re-hashed in full and reused; a stale one is
    discarded.  Partial downloads resume from ``dest.part``.  A hash mismatch
    deletes the file and raises :class:`HashMismatch`.
    """
    label = label or dest.name
    headers = dict(headers or {})
    dest.parent.mkdir(parents=True, exist_ok=True)

    def verify(path: Path) -> tuple[str, str | None, int]:
        sha, git, size = hash_file(path, git_blob=expected_git_sha1 is not None)
        problems = []
        if expected_size is not None and size != expected_size:
            problems.append(f"size {size} != {expected_size}")
        if expected_sha256 is not None and sha != expected_sha256:
            problems.append(f"sha256 {sha} != {expected_sha256}")
        if expected_git_sha1 is not None and git != expected_git_sha1:
            problems.append(f"git blob sha1 {git} != {expected_git_sha1}")
        if problems:
            raise HashMismatch(f"{label}: " + "; ".join(problems))
        return sha, git, size

    if dest.exists():
        started = time.monotonic()
        try:
            sha, git, size = verify(dest)
        except HashMismatch:
            dest.unlink()
        else:
            return {
                "name": dest.name,
                "url": url,
                "size": size,
                "sha256": sha,
                "git_blob_sha1": git,
                "origin": "existing_verified",
                "bytes_downloaded": 0,
                "bytes_resumed_from": 0,
                "network_seconds": 0.0,
                "verify_seconds": round(time.monotonic() - started, 3),
            }

    part = dest.with_name(dest.name + ".part")
    resumed_from = part.stat().st_size if part.exists() else 0
    downloaded = 0
    started = time.monotonic()
    last: NetworkError | None = None
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            downloaded += _stream_into_part(url, part, headers, expected_size)
            last = None
            break
        except NetworkError as exc:
            last = exc
            if attempt < DOWNLOAD_ATTEMPTS - 1:
                _sleep(min(2**attempt, 15))
    if last is not None:
        raise NetworkError(f"{label}: download failed after {DOWNLOAD_ATTEMPTS} attempts: {last}")
    network_seconds = time.monotonic() - started
    verify_started = time.monotonic()
    try:
        sha, git, size = verify(part)
    except HashMismatch:
        part.unlink(missing_ok=True)
        raise
    os.replace(part, dest)
    fsync_dir(dest.parent)
    if stats is not None:
        stats.record(label, downloaded, network_seconds, kind="download", resumed_from=resumed_from)
    return {
        "name": dest.name,
        "url": url,
        "size": size,
        "sha256": sha,
        "git_blob_sha1": git,
        "origin": "downloaded",
        "bytes_downloaded": downloaded,
        "bytes_resumed_from": resumed_from,
        "network_seconds": round(network_seconds, 3),
        "verify_seconds": round(time.monotonic() - verify_started, 3),
        "mb_per_s": round(downloaded / network_seconds / 1e6, 2) if network_seconds > 0 and downloaded else None,
    }


# ---------------------------------------------------------------------------
# Wheel index parsing / selection (pure)
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class IndexLink:
    filename: str
    url: str
    sha256: str | None
    yanked: bool = False


@dataclasses.dataclass(frozen=True)
class IndexPage:
    url: str
    links: tuple[IndexLink, ...]


@dataclasses.dataclass(frozen=True)
class Wheel:
    package: str
    filename: str
    url: str
    sha256: str | None
    version: str  # full, e.g. 2.14.0+rocm10.1.0
    public: str
    local: str
    rocm: str  # label without "rocm", e.g. 10.1.0
    tag: str  # best matching compatibility tag
    tag_rank: int


class _LinkParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[dict[str, str | None], str]] = []
        self._attrs: dict[str, str | None] | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            self._attrs = dict(attrs)
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._attrs is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._attrs is not None:
            self.links.append((self._attrs, "".join(self._text).strip()))
            self._attrs = None


def parse_index_html(page_url: str, body: str) -> IndexPage:
    parser = _LinkParser()
    parser.feed(body)
    links: list[IndexLink] = []
    for attrs, text in parser.links:
        href = attrs.get("href")
        if not href:
            continue
        absolute = urllib.parse.urljoin(page_url, href)
        parsed = urllib.parse.urlparse(absolute)
        filename = urllib.parse.unquote(parsed.path.rsplit("/", 1)[-1]) or text
        sha = None
        for part in parsed.fragment.split("&"):
            if part.startswith("sha256="):
                candidate = part[len("sha256="):].lower()
                sha = candidate if HEX64_RE.match(candidate) else None
        links.append(
            IndexLink(
                filename=filename,
                url=urllib.parse.urlunparse(parsed._replace(fragment="")),
                sha256=sha,
                yanked="data-yanked" in attrs,
            )
        )
    return IndexPage(url=page_url, links=tuple(links))


def fetch_index_page(index_url: str, package: str) -> IndexPage:
    """GET ``<index>/<package>/``.  404 -> NotFound; transport errors -> NetworkError."""
    url = f"{index_url.rstrip('/')}/{package}/"
    body, final = http_get_bytes(url, headers={"Accept": "text/html"}, max_bytes=INDEX_MAX_BYTES)
    return parse_index_html(final, body.decode("utf-8", "replace"))


def normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name).lower()


_NUMERIC_VERSION_RE = re.compile(r"^\d+(\.\d+)*$")
_ROCM_LOCAL_RE = re.compile(r"^rocm(\d+)\.(\d+)(?:\.(\d+))?$")


def numeric_version(text: str) -> tuple[int, ...] | None:
    """Stable release versions only (no pre/dev/post)."""
    return tuple(int(p) for p in text.split(".")) if _NUMERIC_VERSION_RE.match(text) else None


def parse_rocm_mm(text: str) -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)(?:\.\d+)*$", text)
    if not match:
        raise SetupError(f"invalid ROCm version {text!r}")
    return int(match.group(1)), int(match.group(2))


def parse_wheel_filename(filename: str) -> tuple[str, str, list[str]] | None:
    """(normalized name, version, expanded tags) or None if not a wheel."""
    if not filename.endswith(".whl"):
        return None
    parts = filename[:-4].split("-")
    if len(parts) == 6:
        parts.pop(2)  # build tag
    if len(parts) != 5:
        return None
    name, version, py, abi, plat = parts
    tags = [f"{p}-{a}-{t}" for p in py.split(".") for a in abi.split(".") for t in plat.split(".")]
    return normalize_name(name), version, tags


def select_wheel(
    links: Sequence[IndexLink],
    supported_tags: Sequence[str],
    *,
    package: str,
    rocm_version: str,
    version: str | None = None,
    local: str | None = None,
) -> Wheel:
    """Pick the best wheel for ``package`` built for ROCm ``rocm_version`` (major.minor).

    ``supported_tags`` is the venv interpreter's tag list in priority order.
    ``version`` pins the public version (otherwise the highest stable one wins);
    ``local`` pins the full local label (e.g. ``rocm10.1.0``).  CPU / CUDA /
    unlabelled wheels and other ROCm releases are never acceptable.
    """
    wanted = parse_rocm_mm(rocm_version)
    rank = {tag: index for index, tag in enumerate(supported_tags)}
    package_key = normalize_name(package)
    named = rocm_built = version_ok = 0
    candidates: list[tuple[tuple[int, ...], int, Wheel]] = []
    for link in links:
        if link.yanked:
            continue
        parsed = parse_wheel_filename(link.filename)
        if parsed is None or parsed[0] != package_key:
            continue
        named += 1
        _, full_version, tags = parsed
        public, _, label = full_version.partition("+")
        match = _ROCM_LOCAL_RE.match(label)
        if not match or (int(match.group(1)), int(match.group(2))) != wanted:
            continue
        if local is not None and label != local:
            continue
        rocm_built += 1
        key = numeric_version(public)
        if key is None or (version is not None and public != version):
            continue
        version_ok += 1
        usable = [(rank[t], t) for t in tags if t in rank]
        if not usable:
            continue
        best_rank, best_tag = min(usable)
        candidates.append(
            (
                key,
                -best_rank,
                Wheel(
                    package=package_key,
                    filename=link.filename,
                    url=link.url,
                    sha256=link.sha256,
                    version=full_version,
                    public=public,
                    local=label,
                    rocm=label[len("rocm"):],
                    tag=best_tag,
                    tag_rank=best_rank,
                ),
            )
        )
    if candidates:
        return max(candidates, key=lambda item: (item[0], item[1]))[2]
    want = f"{package} for ROCm {rocm_version}" + (f" version {version}" if version else "") + (f" label {local}" if local else "")
    if not rocm_built:
        raise NoCompatibleWheel("no_rocm_wheel", f"no {want}; {named} {package} files listed, none ROCm-{rocm_version} builds (CPU/CUDA/other-ROCm rejected)")
    if not version_ok:
        raise NoCompatibleWheel("version_unavailable", f"{want} not offered ({rocm_built} ROCm-{rocm_version} files with other versions)")
    raise NoCompatibleWheel("no_abi_compatible_wheel", f"{want} exists ({version_ok} files) but none matches this interpreter's tags")


def select_core_version(links: Sequence[IndexLink], rocm_label: str) -> IndexLink:
    """The ``rocm`` core distribution file whose version equals ``rocm_label`` exactly."""
    for link in links:
        if link.yanked:
            continue
        stem = re.sub(r"(\.tar\.gz|\.zip|\.whl)$", "", link.filename)
        if link.filename.endswith(".whl"):
            parsed = parse_wheel_filename(link.filename)
            if parsed and parsed[0] == "rocm" and parsed[1] == rocm_label:
                return link
        elif stem.rsplit("-", 1)[0].lower() == "rocm" and stem.rsplit("-", 1)[-1] == rocm_label:
            return link
    raise NoCompatibleWheel("no_core_distribution", f"rocm=={rocm_label} core distribution is not offered")


@dataclasses.dataclass(frozen=True)
class TorchPlan:
    source: str  # "amd" | "fallback"
    index_url: str
    rocm_version: str
    torch: Wheel
    extras: tuple[str, ...]
    requirements: tuple[str, ...]  # extra pip requirements (e.g. rocm core)
    device_wheel: Wheel | None
    core: IndexLink | None


def resolve_amd_stack(
    fetch: Callable[[str, str], IndexPage],
    supported_tags: Sequence[str],
    *,
    index_url: str,
    rocm_version: str,
    torch_version: str | None,
    device_arch: str,
) -> TorchPlan:
    """torch + split gfx device package + rocm core must all agree on the ROCm build."""
    torch = select_wheel(
        fetch(index_url, "torch").links, supported_tags, package="torch", rocm_version=rocm_version, version=torch_version
    )
    device_pkg = f"amd-torch-device-{device_arch}"
    device = select_wheel(
        fetch(index_url, device_pkg).links,
        supported_tags,
        package=device_pkg,
        rocm_version=rocm_version,
        version=torch.public,
        local=torch.local,
    )
    core = select_core_version(fetch(index_url, "rocm").links, torch.rocm)
    return TorchPlan(
        source="amd",
        index_url=index_url,
        rocm_version=rocm_version,
        torch=torch,
        extras=(f"device-{device_arch}",),
        requirements=(f"rocm[libraries,device-{device_arch}]=={torch.rocm}",),
        device_wheel=device,
        core=core,
    )


def resolve_fallback(
    fetch: Callable[[str, str], IndexPage],
    supported_tags: Sequence[str],
    *,
    rocm_version: str,
    torch_version: str,
) -> TorchPlan:
    index_url = FALLBACK_INDEX_TEMPLATE.format(version=rocm_version)
    torch = select_wheel(
        fetch(index_url, "torch").links, supported_tags, package="torch", rocm_version=rocm_version, version=torch_version
    )
    return TorchPlan("fallback", index_url, rocm_version, torch, (), (), None, None)


def resolve_torch(
    fetch: Callable[[str, str], IndexPage],
    supported_tags: Sequence[str],
    *,
    rocm_version: str = DEFAULT_ROCM_VERSION,
    torch_version: str | None = DEFAULT_TORCH_VERSION,
    device_arch: str = DEVICE_ARCH,
    amd_index_url: str = AMD_INDEX_URL,
    fallback_rocm_version: str | None = DEFAULT_FALLBACK_ROCM_VERSION,
    fallback_torch_version: str = DEFAULT_FALLBACK_TORCH_VERSION,
) -> tuple[TorchPlan, dict[str, Any]]:
    """Primary AMD stack; explicit fallback only on 404 / no compatible wheel.

    Any other failure (including NetworkError anywhere, also during the
    fallback lookup) propagates and aborts setup.
    """
    record: dict[str, Any] = {
        "requested_rocm": rocm_version,
        "requested_torch": torch_version or "latest",
        "device_arch": device_arch,
        "amd_index_url": amd_index_url,
        "fallback_used": False,
        "fallback_reason": None,
    }
    try:
        plan = resolve_amd_stack(
            fetch,
            supported_tags,
            index_url=amd_index_url,
            rocm_version=rocm_version,
            torch_version=torch_version,
            device_arch=device_arch,
        )
        return plan, record
    except NotFound as exc:
        reason = {"code": "index_404", "detail": str(exc)}
    except NoCompatibleWheel as exc:
        reason = {"code": exc.code, "detail": exc.detail}
    if fallback_rocm_version is None:
        raise SetupError(f"primary ROCm {rocm_version} stack unavailable ({reason['code']}: {reason['detail']}) and fallback is disabled")
    plan = resolve_fallback(fetch, supported_tags, rocm_version=fallback_rocm_version, torch_version=fallback_torch_version)
    record.update(fallback_used=True, fallback_reason=reason, fallback_rocm=fallback_rocm_version, fallback_torch=fallback_torch_version)
    return plan, record


def wheel_record(wheel: Wheel | None) -> dict[str, Any] | None:
    return None if wheel is None else dataclasses.asdict(wheel)


# ---------------------------------------------------------------------------
# Hugging Face metadata / file plan (pure)
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class FileSpec:
    name: str
    size: int
    role: str  # config | index | tokenizer | shard
    sha256: str | None  # pinned (config/index) or LFS sha256 (shard)
    git_blob_sha1: str | None  # for non-LFS files


def parse_siblings(api_json: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if api_json.get("sha") != HF_REVISION:
        raise SetupError(f"HF API returned revision {api_json.get('sha')!r}, expected pinned {HF_REVISION}")
    out: dict[str, dict[str, Any]] = {}
    for sibling in api_json.get("siblings", []):
        name = sibling.get("rfilename")
        if not isinstance(name, str):
            continue
        lfs = sibling.get("lfs") or {}
        out[name] = {
            "size": sibling.get("size"),
            "blob_id": sibling.get("blobId"),
            "lfs_sha256": (lfs.get("sha256") or "").lower() or None,
            "lfs_size": lfs.get("size"),
        }
    return out


def _small_spec(siblings: Mapping[str, Mapping[str, Any]], name: str, role: str, sha256: str | None) -> FileSpec:
    meta = siblings.get(name)
    if meta is None:
        raise SetupError(f"pinned revision has no {name}")
    size, blob = meta["size"], meta["blob_id"]
    if not isinstance(size, int) or size <= 0 or size > SMALL_FILE_MAX_BYTES:
        raise SetupError(f"{name}: implausible size {size!r}")
    if not isinstance(blob, str) or not HEX40_RE.match(blob):
        raise SetupError(f"{name}: HF API gave no git blob id")
    return FileSpec(name, size, role, sha256, blob)


def pinned_small_specs(siblings: Mapping[str, Mapping[str, Any]]) -> tuple[FileSpec, FileSpec]:
    return (
        _small_spec(siblings, CONFIG_NAME, "config", MODEL_CONFIG_SHA256),
        _small_spec(siblings, INDEX_NAME, "index", MODEL_INDEX_SHA256),
    )


def plan_model_files(siblings: Mapping[str, Mapping[str, Any]], index_bytes: bytes) -> list[FileSpec]:
    """Exact download set: config, index, tokenizer files, and the shards in ``weight_map``."""
    if hashlib.sha256(index_bytes).hexdigest() != MODEL_INDEX_SHA256:
        raise HashMismatch("model.safetensors.index.json does not match the pinned sha256")
    try:
        weight_map = json.loads(index_bytes)["weight_map"]
    except (ValueError, KeyError, TypeError) as exc:
        raise SetupError(f"unparseable safetensors index: {exc}") from exc
    shard_names = sorted({str(v) for v in weight_map.values()})
    if len(shard_names) != EXPECTED_SHARD_COUNT:
        raise SetupError(f"index maps {len(shard_names)} shards, expected {EXPECTED_SHARD_COUNT}")
    specs = list(pinned_small_specs(siblings))
    for name in shard_names:
        if not SHARD_NAME_RE.match(name):
            raise SetupError(f"refusing unsafe shard name {name!r}")
        meta = siblings.get(name)
        if meta is None:
            raise SetupError(f"shard {name} is in the index but not in the pinned revision")
        sha, size, lfs_size = meta["lfs_sha256"], meta["size"], meta["lfs_size"]
        if not sha or not HEX64_RE.match(sha):
            raise SetupError(f"shard {name}: HF API gave no LFS sha256")
        if not isinstance(size, int) or size <= 0 or (lfs_size is not None and lfs_size != size):
            raise SetupError(f"shard {name}: inconsistent size {size!r} / LFS size {lfs_size!r}")
        specs.append(FileSpec(name, size, "shard", sha, None))
    tokenizer = [name for name in TOKENIZER_FILES if name in siblings]
    if not ({"tokenizer.json"} <= set(tokenizer) or {"tokenizer.model"} <= set(tokenizer) or {"vocab.json", "merges.txt"} <= set(tokenizer)):
        raise SetupError(f"pinned revision has no usable tokenizer files (found {tokenizer})")
    for name in tokenizer:
        meta = siblings[name]
        if meta["lfs_sha256"]:
            if not isinstance(meta["size"], int):
                raise SetupError(f"{name}: missing size")
            specs.append(FileSpec(name, meta["size"], "tokenizer", meta["lfs_sha256"], None))
        else:
            specs.append(_small_spec(siblings, name, "tokenizer", None))
    return specs


# ---------------------------------------------------------------------------
# Disk planning (pure)
# ---------------------------------------------------------------------------
def disk_plan(
    *,
    model_bytes: int,
    already_staged_bytes: int,
    corpus_bytes: int,
    output_bytes: int,
    free_bytes: int,
    venv_reserve_bytes: int = VENV_RESERVE_BYTES,
) -> dict[str, Any]:
    estimate = model_bytes + corpus_bytes + output_bytes + venv_reserve_bytes
    still_needed = max(0, model_bytes - already_staged_bytes) + corpus_bytes + output_bytes + venv_reserve_bytes
    plan = {
        "model_bytes": model_bytes,
        "already_staged_bytes": already_staged_bytes,
        "corpus_bytes": corpus_bytes,
        "output_bytes": output_bytes,
        "venv_reserve_bytes": venv_reserve_bytes,
        "estimate_bytes": estimate,
        "still_needed_bytes": still_needed,
        "free_bytes": free_bytes,
        "warn_threshold_bytes": WARN_ESTIMATE_BYTES,
        "warnings": [],
    }
    if estimate > WARN_ESTIMATE_BYTES:
        plan["warnings"].append(f"disk estimate {estimate / 1e9:.0f} GB exceeds {WARN_ESTIMATE_BYTES / 1e9:.0f} GB")
    if free_bytes < still_needed:
        raise SetupError(
            f"insufficient disk: need {still_needed / 1e9:.1f} GB more, only {free_bytes / 1e9:.1f} GB free "
            f"(model {model_bytes / 1e9:.1f} + corpus {corpus_bytes / 1e9:.1f} + output {output_bytes / 1e9:.1f} "
            f"+ venv {venv_reserve_bytes / 1e9:.1f} GB estimate)"
        )
    return plan


# ---------------------------------------------------------------------------
# Host probing
# ---------------------------------------------------------------------------
def run_capture(argv: Sequence[str], *, timeout: int = 60) -> dict[str, Any]:
    try:
        result = subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout, errors="replace")
    except FileNotFoundError:
        return {"argv": list(argv), "found": False}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"argv": list(argv), "found": True, "error": str(exc)}
    return {
        "argv": list(argv),
        "found": True,
        "returncode": result.returncode,
        "stdout": result.stdout[:PROBE_OUTPUT_LIMIT],
        "stderr": result.stderr[:PROBE_OUTPUT_LIMIT],
    }


def host_rocm_version(probe: Mapping[str, Any]) -> tuple[int, int] | None:
    """major.minor from /opt/rocm/.info/version, else hipcc's ``HIP version``."""
    text = probe.get("rocm_info_version")
    if isinstance(text, str):
        match = re.match(r"\s*(\d+)\.(\d+)", text)
        if match:
            return int(match.group(1)), int(match.group(2))
    hipcc = probe.get("commands", {}).get("hipcc_version", {})
    match = re.search(r"HIP version:\s*(\d+)\.(\d+)", hipcc.get("stdout", "") if isinstance(hipcc, dict) else "")
    return (int(match.group(1)), int(match.group(2))) if match else None


def probe_host(run: Callable[[Sequence[str]], dict[str, Any]] = run_capture) -> dict[str, Any]:
    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "rocm_path": rocm_path,
        "dev_kfd": os.path.exists("/dev/kfd"),
        "dev_dri": os.path.isdir("/dev/dri"),
    }
    try:
        info["os_release"] = Path("/etc/os-release").read_text(encoding="utf-8")
    except OSError:
        info["os_release"] = None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                info["mem_total_kib"] = int(line.split()[1])
    except (OSError, ValueError):
        pass
    try:
        info["rocm_info_version"] = Path(rocm_path, ".info", "version").read_text(encoding="utf-8").strip()
    except OSError:
        info["rocm_info_version"] = None
    commands = {
        "hipcc_version": [os.path.join(rocm_path, "bin", "hipcc"), "--version"],
        "hipcc_path_version": ["hipcc", "--version"],
        "rocminfo": [os.path.join(rocm_path, "bin", "rocminfo")],
        "rocminfo_path": ["rocminfo"],
        "rocm_smi_driver": ["rocm-smi", "--showdriverversion"],
        "amd_smi_version": ["amd-smi", "version"],
        "amdgpu_module_version": ["cat", "/sys/module/amdgpu/version"],
    }
    info["commands"] = {name: run(argv) for name, argv in commands.items()}
    if not info["commands"]["hipcc_version"].get("found"):
        info["commands"]["hipcc_version"] = info["commands"]["hipcc_path_version"]
    rocminfo = info["commands"]["rocminfo"] if info["commands"]["rocminfo"].get("found") else info["commands"]["rocminfo_path"]
    info["gfx_agents"] = sorted(set(re.findall(r"\bgfx[0-9a-f]+\b", rocminfo.get("stdout", ""))))
    version = host_rocm_version(info)
    info["rocm_version_mm"] = None if version is None else f"{version[0]}.{version[1]}"
    return info


def check_host_rocm(host: Mapping[str, Any], wheel_rocm: str, *, allow_mismatch: bool) -> dict[str, Any]:
    """Host ROCm major.minor must match the wheel's ROCm build, or be explicitly allowed."""
    wanted = parse_rocm_mm(wheel_rocm)
    found = host.get("rocm_version_mm")
    if found is None:
        status = "unknown"
    else:
        status = "match" if parse_rocm_mm(found) == wanted else "mismatch"
    record = {"status": status, "host": found, "wheel": f"{wanted[0]}.{wanted[1]}", "allowed_by_flag": False}
    if status != "match":
        if not allow_mismatch:
            raise SetupError(
                f"host ROCm is {found or 'not detected'} but the selected wheel is ROCm {wanted[0]}.{wanted[1]} "
                "(use a matching image or pass --allow-rocm-mismatch to record the mismatch explicitly)"
            )
        record["allowed_by_flag"] = True
    return record


# ---------------------------------------------------------------------------
# Runtime verification inside the venv
# ---------------------------------------------------------------------------
SUPPORTED_TAGS_PROBE = """
import json
try:
    from packaging.tags import sys_tags
except ImportError:
    from pip._vendor.packaging.tags import sys_tags
import sys
print(json.dumps({"tags": [str(t) for t in sys_tags()],
                  "version": list(sys.version_info[:3]),
                  "implementation": sys.implementation.name,
                  "executable": sys.executable,
                  "prefix": sys.prefix}))
"""

RUNTIME_PROBE = """
import json
import importlib.metadata as md
import torch
out = {
    "torch_version": torch.__version__,
    "hip": getattr(torch.version, "hip", None),
    "cuda": getattr(torch.version, "cuda", None),
    "cuda_available": bool(torch.cuda.is_available()),
    "device_count": int(torch.cuda.device_count()),
    "devices": [],
    "distributions": {},
}
for name in ("torch", "rocm", "amd-torch-device-gfx942"):
    try:
        out["distributions"][name] = md.version(name)
    except md.PackageNotFoundError:
        out["distributions"][name] = None
for i in range(out["device_count"]):
    p = torch.cuda.get_device_properties(i)
    out["devices"].append({"index": i, "name": p.name, "gcn_arch": getattr(p, "gcnArchName", None),
                           "total_memory": int(p.total_memory)})
if out["device_count"]:
    x = torch.ones((64, 64), dtype=torch.bfloat16, device="cuda:0")
    y = x @ x
    torch.cuda.synchronize()
    out["bf16_matmul_sum"] = float(y.float().sum().item())
print(json.dumps(out))
"""

SOURCE_LOADER = """
import json, sys
sys.path.insert(0, sys.argv[1])
import upstream
texts, provenance, cache = upstream.load_pinned_sources()
operators = upstream._trusted_operator_provenance(texts)
print(json.dumps({
    "cache_dir": str(cache),
    "provenance": provenance,
    "operators": operators,
    "expected": [{k: s[k] for k in ("name", "project", "commit", "path", "sha256")} for s in upstream.UPSTREAM_SOURCE_SPECS],
    "hf_model": upstream.HF_MODEL,
    "hf_config_commit": upstream.HF_CONFIG_COMMIT,
    "model_config_sha256": upstream.MODEL_CONFIG_SHA256,
    "model_index_sha256": upstream.MODEL_INDEX_SHA256,
}))
"""


def validate_runtime(info: Mapping[str, Any], plan: TorchPlan, *, device_arch: str = DEVICE_ARCH) -> dict[str, Any]:
    """Raise unless torch is a ROCm build running bf16 on every gfx942 device.

    The raw ``torch.version.hip`` is recorded, not required to equal the ROCm
    distribution version (no verified mapping); the identity check is the
    wheel's local build label plus the installed ``rocm`` / device distributions.
    """
    problems: list[str] = []
    if not info.get("hip"):
        problems.append("torch.version.hip is empty: not a ROCm build of PyTorch")
    if info.get("cuda"):
        problems.append(f"torch.version.cuda={info.get('cuda')!r}: CUDA build installed")
    if info.get("torch_version") != plan.torch.version:
        problems.append(f"torch {info.get('torch_version')!r} != selected wheel {plan.torch.version!r}")
    dists = info.get("distributions", {})
    if plan.source == "amd":
        if dists.get("rocm") != plan.torch.rocm:
            problems.append(f"installed rocm distribution {dists.get('rocm')!r} != {plan.torch.rocm!r}")
        if dists.get("amd-torch-device-gfx942") != plan.torch.version:
            problems.append(f"amd-torch-device-{device_arch} {dists.get('amd-torch-device-gfx942')!r} != {plan.torch.version!r}")
    if not info.get("cuda_available"):
        problems.append("torch.cuda.is_available() is False (no CPU fallback allowed)")
    devices = info.get("devices", [])
    if not devices:
        problems.append("no GPU devices visible")
    bad = [d for d in devices if not str(d.get("gcn_arch") or "").startswith(device_arch)]
    if bad:
        problems.append(f"non-{device_arch} devices: {[(d.get('name'), d.get('gcn_arch')) for d in bad]}")
    if devices and info.get("bf16_matmul_sum") != float(64**3):
        problems.append(f"bf16 GPU matmul sanity check failed: {info.get('bf16_matmul_sum')!r}")
    if problems:
        raise SetupError("runtime verification failed: " + "; ".join(problems))
    wanted = parse_rocm_mm(plan.rocm_version)
    hip_match = re.match(r"^(\d+)\.(\d+)", str(info.get("hip")))
    hip_mm = (int(hip_match.group(1)), int(hip_match.group(2))) if hip_match else None
    return {
        "raw_torch_version_hip": info.get("hip"),
        "raw_torch_version_cuda": info.get("cuda"),
        "hip_major_minor_equals_rocm_build": hip_mm == wanted,
        "note": None
        if hip_mm == wanted
        else f"torch.version.hip {info.get('hip')!r} differs from ROCm build {wanted[0]}.{wanted[1]}; "
        "identity is established by the wheel local label and installed rocm distribution, not the HIP component version",
        "devices": devices,
        "distributions": dists,
    }


# ---------------------------------------------------------------------------
# Orchestration helpers
# ---------------------------------------------------------------------------
def hf_headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def hf_resolve_url(name: str) -> str:
    return f"{HF_ENDPOINT}/{HF_MODEL}/resolve/{HF_REVISION}/{urllib.parse.quote(name)}"


def fetch_hf_metadata(stats: NetStats) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    url = f"{HF_ENDPOINT}/api/models/{HF_MODEL}/revision/{HF_REVISION}?blobs=true"
    started = time.monotonic()
    body, _ = http_get_bytes(url, headers={**hf_headers(), "Accept": "application/json"}, max_bytes=API_MAX_BYTES)
    stats.record("hf_api_metadata", len(body), time.monotonic() - started, kind="api")
    try:
        api = json.loads(body)
    except ValueError as exc:
        raise SetupError(f"HF API returned invalid JSON: {exc}") from exc
    return parse_siblings(api), {"url": url, "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body), "authenticated": bool(hf_headers())}


def model_dir_for(root: Path) -> Path:
    """Staged snapshot dir ``<root>/source/<pinned revision>``.

    The revision is the final path component so the corpus helper's
    ``--strict-revision`` accepts the tokenizer path.
    """
    return root / "source" / HF_REVISION


def stage_model(root: Path, siblings: Mapping[str, Mapping[str, Any]], stats: NetStats, jobs: int, *, disk: Callable[[list[FileSpec]], None]) -> dict[str, Any]:
    model_dir = model_dir_for(root)
    model_dir.mkdir(parents=True, exist_ok=True)
    headers = hf_headers()
    config_spec, index_spec = pinned_small_specs(siblings)
    records: dict[str, dict[str, Any]] = {}
    for spec in (config_spec, index_spec):
        records[spec.name] = download_file(
            hf_resolve_url(spec.name),
            model_dir / spec.name,
            expected_size=spec.size,
            expected_sha256=spec.sha256,
            expected_git_sha1=spec.git_blob_sha1,
            headers=headers,
            stats=stats,
        )
    specs = plan_model_files(siblings, (model_dir / INDEX_NAME).read_bytes())
    disk(specs)
    pending = [s for s in specs if s.name not in records]
    # Tokenizer/small files first, then the large shards in parallel.
    small = [s for s in pending if s.role != "shard"]
    shards = [s for s in pending if s.role == "shard"]

    def fetch(spec: FileSpec) -> dict[str, Any]:
        record = download_file(
            hf_resolve_url(spec.name),
            model_dir / spec.name,
            expected_size=spec.size,
            expected_sha256=spec.sha256,
            expected_git_sha1=spec.git_blob_sha1,
            headers=headers,
            stats=stats,
        )
        print(f"[source] {spec.role} {spec.name} {record['origin']} {record['size']} B", file=sys.stderr, flush=True)
        return record

    for spec in small:
        records[spec.name] = fetch(spec)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = {pool.submit(fetch, spec): spec for spec in shards}
        try:
            for future in concurrent.futures.as_completed(futures):
                records[futures[future].name] = future.result()
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    ordered = []
    for spec in specs:
        record = records[spec.name]
        ordered.append(
            {
                "name": spec.name,
                "role": spec.role,
                "size": record["size"],
                "sha256": record["sha256"],
                "git_blob_sha1": record["git_blob_sha1"],
                "pinned_sha256": spec.sha256,
                "origin": record["origin"],
                "bytes_downloaded": record["bytes_downloaded"],
                "bytes_resumed_from": record["bytes_resumed_from"],
                "network_seconds": record["network_seconds"],
                "mb_per_s": record.get("mb_per_s"),
            }
        )
    manifest = {
        "schema": "hipfire.flash_next.source_manifest.v1",
        "model": HF_MODEL,
        "revision": HF_REVISION,
        "config_sha256": MODEL_CONFIG_SHA256,
        "index_sha256": MODEL_INDEX_SHA256,
        "shard_count": sum(1 for r in ordered if r["role"] == "shard"),
        "total_bytes": sum(r["size"] for r in ordered),
        "model_dir": str(model_dir),
        "files": ordered,
    }
    atomic_write_json(root / "source" / "source-manifest.json", manifest)
    return {"path": str(root / "source" / "source-manifest.json"), "source_dir": str(model_dir), "model_dir": str(model_dir), "total_bytes": manifest["total_bytes"], "shard_count": manifest["shard_count"], "files": len(ordered)}


def pip_run(venv_python: Path, args: Sequence[str], log_path: Path) -> float:
    cmd = [str(venv_python), "-m", "pip", *args]
    started = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    elapsed = time.monotonic() - started
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(f"$ {' '.join(cmd)}\n{result.stdout}\n{result.stderr}", encoding="utf-8")
    if result.returncode != 0:
        tail = (result.stderr or result.stdout)[-2000:]
        raise SetupError(f"pip {' '.join(args[:2])} failed (rc={result.returncode}); log {log_path}\n{tail}")
    return elapsed


def verify_source_loader(payload: Mapping[str, Any]) -> None:
    if payload.get("hf_model") != HF_MODEL or payload.get("hf_config_commit") != HF_REVISION:
        raise SetupError("upstream.py model/revision pins disagree with rent_setup")
    if payload.get("model_config_sha256") != MODEL_CONFIG_SHA256 or payload.get("model_index_sha256") != MODEL_INDEX_SHA256:
        raise SetupError("upstream.py config/index hash pins disagree with rent_setup")
    expected = {e["name"]: e for e in payload.get("expected", [])}
    seen = {p["name"]: p for p in payload.get("provenance", [])}
    if set(expected) != set(seen) or not expected:
        raise SetupError(f"loaded operator sources {sorted(seen)} != pinned specs {sorted(expected)}")
    for name, spec in expected.items():
        if seen[name].get("sha256") != spec["sha256"] or seen[name].get("commit") != spec["commit"]:
            raise SetupError(f"operator source {name} hash/commit mismatch after load")


def load_source_operators(venv_python: Path, oracle_dir: Path) -> dict[str, Any]:
    result = subprocess.run([str(venv_python), "-c", SOURCE_LOADER, str(oracle_dir)], capture_output=True, text=True, errors="replace", timeout=900)
    if result.returncode != 0:
        raise SetupError(f"pinned operator source load failed:\n{result.stderr[-2000:]}")
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    try:
        payload = json.loads(lines[-1])
    except (IndexError, ValueError) as exc:
        raise SetupError(f"operator loader printed no JSON: {result.stdout[-500:]}") from exc
    verify_source_loader(payload)
    return {
        "cache_dir": payload["cache_dir"],
        "sources": [{k: p.get(k) for k in ("name", "project", "commit", "path", "sha256", "origin", "url")} for p in payload["provenance"]],
        "operators_verified": len(payload["operators"]),
    }


def json_subprocess(venv_python: Path, code: str, *, timeout: int = 300) -> dict[str, Any]:
    result = subprocess.run([str(venv_python), "-c", code], capture_output=True, text=True, errors="replace", timeout=timeout)
    if result.returncode != 0:
        raise SetupError(f"venv python probe failed:\n{(result.stderr or result.stdout)[-2000:]}")
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError) as exc:
        raise SetupError(f"venv probe printed no JSON: {result.stdout[-500:]}") from exc


def check_host_python(python: str) -> dict[str, Any]:
    result = run_capture([python, "-c", "import sys,json;print(json.dumps({'version':list(sys.version_info[:3]),'impl':sys.implementation.name,'exe':sys.executable}))"])
    if not result.get("found") or result.get("returncode") != 0:
        raise SetupError(f"cannot run --python {python!r}: {result}")
    info = json.loads(result["stdout"])
    if tuple(info["version"][:2]) < MIN_PYTHON or info["impl"] != "cpython":
        raise SetupError(f"--python must be CPython >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}, got {info}")
    return info


def create_venv(python: str, venv_dir: Path) -> Path:
    venv_python = venv_dir / "bin" / "python"
    if not venv_python.exists():
        result = subprocess.run([python, "-m", "venv", str(venv_dir)], capture_output=True, text=True, errors="replace")
        if result.returncode != 0:
            raise SetupError(
                f"venv creation failed (on Debian/Ubuntu install python3-venv): {(result.stderr or result.stdout)[-1500:]}"
            )
    return venv_python


class Manifest:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, Any] = {}

    def update(self, **values: Any) -> None:
        self.data.update(values)
        atomic_write_json(self.path, self.data)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, type=Path, help="work root: venv/, source/, wheels/, logs/, setup-manifest.json")
    parser.add_argument("--python", default=sys.executable, help="host CPython >= 3.10 used to create the private venv")
    parser.add_argument("--rocm-version", default=DEFAULT_ROCM_VERSION, help="primary ROCm major.minor (default %(default)s)")
    parser.add_argument("--torch-version", default=DEFAULT_TORCH_VERSION, help="exact torch release, or 'latest' for the highest compatible (default %(default)s)")
    parser.add_argument("--fallback-rocm-version", default=DEFAULT_FALLBACK_ROCM_VERSION, help="explicit PyTorch-index ROCm fallback used only if the primary stack is 404/absent/ABI-incompatible (default %(default)s)")
    parser.add_argument("--fallback-torch-version", default=DEFAULT_FALLBACK_TORCH_VERSION, help="exact torch release for the fallback wheel (default %(default)s)")
    parser.add_argument("--no-fallback", action="store_true", help="never fall back")
    parser.add_argument("--allow-rocm-mismatch", action="store_true", help="record, instead of failing on, a host-ROCm vs wheel-ROCm mismatch or undetected host ROCm")
    parser.add_argument("--corpus-gb", type=float, default=0.0, help="expected corpus bytes in GB for the disk estimate")
    parser.add_argument("--output-gb", type=float, default=0.0, help="expected output bytes in GB for the disk estimate")
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS, help="parallel shard downloads (default %(default)s)")
    return parser


def run_setup(args: argparse.Namespace, *, fetch: Callable[[str, str], IndexPage] = fetch_index_page) -> dict[str, Any]:
    root = args.root.expanduser().resolve()
    for sub in ("", "source", "wheels", "logs"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    manifest = Manifest(root / "setup-manifest.json")
    stats = NetStats()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest.update(
        schema=SCHEMA,
        status="in_progress",
        started_at=started_at,
        argv=sys.argv,
        root=str(root),
        pins={
            "model": HF_MODEL,
            "revision": HF_REVISION,
            "config_sha256": MODEL_CONFIG_SHA256,
            "index_sha256": MODEL_INDEX_SHA256,
            "shard_count": EXPECTED_SHARD_COUNT,
        },
    )
    try:
        return _run_setup(args, root, manifest, stats, fetch)
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}", network=stats.summary(), network_log=stats.entries)
        raise


def _run_setup(args: argparse.Namespace, root: Path, manifest: Manifest, stats: NetStats, fetch: Callable[[str, str], IndexPage]) -> dict[str, Any]:
    oracle_dir = Path(__file__).resolve().parent
    host = probe_host()
    host_python = check_host_python(args.python)
    manifest.update(host=host, host_python=host_python)

    venv_python = create_venv(args.python, root / "venv")
    probe = json_subprocess(venv_python, SUPPORTED_TAGS_PROBE)
    manifest.update(venv={"path": str(root / "venv"), "python": probe["executable"], "version": probe["version"], "implementation": probe["implementation"]})

    torch_version = None if args.torch_version == "latest" else args.torch_version
    plan, selection = resolve_torch(
        fetch,
        probe["tags"],
        rocm_version=args.rocm_version,
        torch_version=torch_version,
        amd_index_url=AMD_INDEX_URL,
        fallback_rocm_version=None if args.no_fallback else args.fallback_rocm_version,
        fallback_torch_version=args.fallback_torch_version,
    )
    selection.update(
        source=plan.source,
        index_url=plan.index_url,
        selected_rocm=plan.rocm_version,
        torch_wheel=wheel_record(plan.torch),
        device_wheel=wheel_record(plan.device_wheel),
        core_distribution=None if plan.core is None else dataclasses.asdict(plan.core),
        extras=list(plan.extras),
        requirements=list(plan.requirements),
    )
    host_check = check_host_rocm(host, plan.torch.rocm, allow_mismatch=args.allow_rocm_mismatch)
    manifest.update(torch_selection=selection, host_rocm_check=host_check)
    print(f"[torch] {plan.source}: {plan.torch.filename}" + (f" (fallback: {selection['fallback_reason']['code']})" if selection["fallback_used"] else ""), file=sys.stderr, flush=True)

    siblings, api_meta = fetch_hf_metadata(stats)
    shard_bytes = sum(int(m["size"]) for n, m in siblings.items() if SHARD_NAME_RE.match(n) and m["lfs_sha256"])
    model_dir = model_dir_for(root)

    def check_disk(specs: Sequence[FileSpec] | None) -> None:
        names = [s.name for s in specs] if specs else []
        model_bytes = sum(s.size for s in specs) if specs else shard_bytes
        staged = 0
        for name in names:
            for candidate in (model_dir / name, model_dir / (name + ".part")):
                if candidate.exists():
                    staged += min(candidate.stat().st_size, next(s.size for s in specs if s.name == name))
        usage = shutil.disk_usage(root)
        plan_record = disk_plan(
            model_bytes=model_bytes,
            already_staged_bytes=staged,
            corpus_bytes=int(args.corpus_gb * 1e9),
            output_bytes=int(args.output_gb * 1e9),
            free_bytes=usage.free,
        )
        plan_record["total_bytes"] = usage.total
        for warning in plan_record["warnings"]:
            print(f"[disk] WARNING: {warning}", file=sys.stderr, flush=True)
        manifest.update(disk_plan=plan_record)

    check_disk(None)  # early reject from the HF-reported shard total, before any large transfer
    manifest.update(hf_metadata=api_meta)

    wheel_records = []
    wheel_paths = {}
    for wheel in (plan.torch, plan.device_wheel):
        if wheel is None:
            continue
        record = download_file(wheel.url, root / "wheels" / wheel.filename, expected_sha256=wheel.sha256, stats=stats, label=wheel.filename)
        record["index_sha256_fragment"] = wheel.sha256
        wheel_records.append(record)
        wheel_paths[wheel.package] = root / "wheels" / wheel.filename
    manifest.update(wheels=wheel_records)

    log_dir = root / "logs"
    pip_times = {}
    common = ["--no-input", "--disable-pip-version-check", "--index-url", plan.index_url]
    if plan.requirements:
        pip_times["core"] = pip_run(venv_python, ["install", *common, *plan.requirements], log_dir / "pip-install-core.log")
    torch_target = str(wheel_paths["torch"]) + (f"[{','.join(plan.extras)}]" if plan.extras else "")
    install = ["install", *common, "--report", str(log_dir / "pip-install-torch-report.json"), torch_target]
    if plan.device_wheel is not None:
        install.insert(len(install) - 1, str(wheel_paths[plan.device_wheel.package]))
    pip_times["torch"] = pip_run(venv_python, install, log_dir / "pip-install-torch.log")
    freeze = subprocess.run([str(venv_python), "-m", "pip", "list", "--format=json"], capture_output=True, text=True, errors="replace")
    manifest.update(pip={"seconds": {k: round(v, 3) for k, v in pip_times.items()}, "packages": json.loads(freeze.stdout) if freeze.returncode == 0 else None, "report": str(log_dir / "pip-install-torch-report.json")})

    runtime_info = json_subprocess(venv_python, RUNTIME_PROBE, timeout=600)
    runtime = validate_runtime(runtime_info, plan)
    manifest.update(runtime=runtime)

    source = stage_model(root, siblings, stats, args.jobs, disk=check_disk)
    manifest.update(source=source, network=stats.summary())

    operators = load_source_operators(venv_python, oracle_dir)
    usage = shutil.disk_usage(root)
    manifest.update(
        status="complete",
        operators=operators,
        disk_allocation={
            "venv_bytes": dir_size(root / "venv"),
            "wheels_bytes": dir_size(root / "wheels"),
            "source_bytes": dir_size(root / "source"),
            "free_bytes_after": usage.free,
            "total_bytes": usage.total,
        },
        network=stats.summary(),
        network_log=stats.entries,
        finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    return manifest.data


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        data = run_setup(args)
    except SetupError as exc:
        print(f"rent_setup: FAILED: {exc}", file=sys.stderr)
        return 1
    root = args.root.expanduser().resolve()
    print(json.dumps({"status": data["status"], "manifest": str(root / "setup-manifest.json"), "source_manifest": data["source"]["path"], "source_dir": data["source"]["source_dir"], "venv_python": str(root / "venv" / "bin" / "python")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
