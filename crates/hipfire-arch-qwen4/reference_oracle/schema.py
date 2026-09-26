#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Small dependency-free tensor/NPZ helpers for the Qwen4 reference oracle.

The oracle deliberately does not depend on torch or NumPy.  Generated NPZ files
are ordinary version-1 NPY members and can therefore be consumed by either
library, while the generator remains usable in a clean checkout.  Tensor values
are held as Python lists only while a compact fixture is being constructed; the
large real checkpoint is never opened or resident.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import math
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


class FixtureError(ValueError):
    """A malformed fixture or source slice was supplied."""


_DTYPE_INFO = {
    "float32": ("<f4", 4, "f"),
    "float64": ("<f8", 8, "d"),
    "int32": ("<i4", 4, "i"),
    "int64": ("<i8", 8, "q"),
    "uint16": ("<u2", 2, "H"),
    "uint32": ("<u4", 4, "I"),
    "uint64": ("<u8", 8, "Q"),
    "uint8": ("|u1", 1, "B"),
    "bool": ("|b1", 1, "?"),
}


@dataclass(frozen=True)
class Tensor:
    """A flat row-major tensor with an explicit storage dtype."""

    shape: tuple[int, ...]
    data: tuple[object, ...]
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.dtype not in _DTYPE_INFO:
            raise FixtureError(f"unsupported storage dtype {self.dtype!r}")
        if any(int(x) < 0 for x in self.shape):
            raise FixtureError(f"negative tensor shape {self.shape}")
        expected = 1
        for dim in self.shape:
            expected *= int(dim)
        if len(self.data) != expected:
            raise FixtureError(
                f"tensor {self.shape} contains {len(self.data)} values, expected {expected}"
            )

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return len(self.data)

    def reshape(self, *shape: int | Sequence[int]) -> "Tensor":
        if len(shape) == 1 and not isinstance(shape[0], int):
            target = tuple(int(x) for x in shape[0])
        else:
            target = tuple(int(x) for x in shape)
        return Tensor(target, self.data, self.dtype)

    def astype(self, dtype: str) -> "Tensor":
        if dtype == self.dtype:
            return self
        if dtype not in _DTYPE_INFO:
            raise FixtureError(f"unsupported storage dtype {dtype!r}")
        if dtype == "float32":
            values = tuple(float(x) for x in self.data)
        elif dtype == "float64":
            values = tuple(float(x) for x in self.data)
        elif dtype.startswith("int") or dtype.startswith("uint"):
            values = tuple(int(x) for x in self.data)
        else:
            values = tuple(bool(x) for x in self.data)
        return Tensor(self.shape, values, dtype)


def tensor(shape: Sequence[int], values: Iterable[object], dtype: str = "float32") -> Tensor:
    return Tensor(tuple(int(x) for x in shape), tuple(values), dtype)


def zeros(shape: Sequence[int], dtype: str = "float32") -> Tensor:
    n = 1
    for dim in shape:
        n *= int(dim)
    value: object = False if dtype == "bool" else 0
    return Tensor(tuple(int(x) for x in shape), (value,) * n, dtype)


def _pack_values(t: Tensor) -> bytes:
    descr, itemsize, code = _DTYPE_INFO[t.dtype]
    if t.dtype == "bool":
        return bytes(1 if bool(v) else 0 for v in t.data)
    out = io.BytesIO()
    fmt = "<" + code
    try:
        for value in t.data:
            out.write(struct.pack(fmt, value))
    except (OverflowError, struct.error) as exc:
        raise FixtureError(f"cannot pack {t.dtype} value in {t.shape}: {exc}") from exc
    raw = out.getvalue()
    if len(raw) != t.size * itemsize:
        raise FixtureError(f"packed {t.shape} has wrong byte count")
    return raw


def tensor_bytes(t: Tensor) -> bytes:
    """Return the exact little-endian storage bytes represented by *t*."""

    return _pack_values(t)


def _npy_member(t: Tensor) -> bytes:
    descr = _DTYPE_INFO[t.dtype][0]
    shape = tuple(int(x) for x in t.shape)
    shape_repr = repr(shape) if len(shape) != 1 else repr(shape)  # NumPy accepts (N,).
    if len(shape) == 1:
        shape_repr = f"({shape[0]},)"
    header = (
        "{'descr': "
        + repr(descr)
        + ", 'fortran_order': False, 'shape': "
        + shape_repr
        + ", }"
    ).encode("ascii")
    # NPY v1 requires the preamble + header length to be divisible by 16.
    prefix = b"\x93NUMPY\x01\x00"
    pad = 16 - ((len(prefix) + 2 + len(header) + 1) % 16)
    header += b" " * (pad - 1) + b"\n"
    if len(header) >= 65536:
        raise FixtureError("fixture member header is too large for deterministic NPY v1")
    return prefix + struct.pack("<H", len(header)) + header + _pack_values(t)


def write_npz(path: str | Path, arrays: Mapping[str, Tensor]) -> str:
    """Write a byte-stable, uncompressed NPZ and return its SHA-256 digest."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    members = {f"{name}.npy": _npy_member(arrays[name]) for name in sorted(arrays)}
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED, allowZip64=False) as archive:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, members[name])
    return sha256_file(destination)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _unpack_values(raw: bytes, descr: str, count: int) -> tuple[object, ...]:
    inverse = {info[0]: (dtype, info[1], info[2]) for dtype, info in _DTYPE_INFO.items()}
    if descr not in inverse:
        raise FixtureError(f"unsupported NPY dtype descriptor {descr!r}")
    dtype, itemsize, code = inverse[descr]
    if len(raw) != count * itemsize:
        raise FixtureError(f"NPY payload has {len(raw)} bytes, expected {count * itemsize}")
    if dtype == "bool":
        return tuple(bool(x) for x in raw)
    if not raw:
        return ()
    return tuple(struct.unpack("<" + code * count, raw))


def read_npy(raw: bytes) -> Tensor:
    if raw[:6] != b"\x93NUMPY":
        raise FixtureError("source member is not an NPY array")
    major, minor = raw[6], raw[7]
    if (major, minor) == (1, 0):
        if len(raw) < 10:
            raise FixtureError("truncated NPY v1 header")
        header_size = struct.unpack_from("<H", raw, 8)[0]
        offset = 10
    elif (major, minor) in ((2, 0), (3, 0)):
        if len(raw) < 12:
            raise FixtureError("truncated NPY v2/v3 header")
        header_size = struct.unpack_from("<I", raw, 8)[0]
        offset = 12
    else:
        raise FixtureError(f"unsupported NPY version {major}.{minor}")
    end = offset + header_size
    if end > len(raw):
        raise FixtureError("truncated NPY header payload")
    try:
        header = ast.literal_eval(raw[offset:end].decode("latin1").strip())
    except (SyntaxError, ValueError) as exc:
        raise FixtureError(f"invalid NPY header: {exc}") from exc
    if not isinstance(header, dict) or set(header) != {"descr", "fortran_order", "shape"}:
        raise FixtureError("NPY header does not have the required fields")
    if header["fortran_order"]:
        raise FixtureError("Fortran-order source slices are not accepted")
    shape = header["shape"]
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(int(x) for x in shape)
    count = math.prod(shape)
    values = _unpack_values(raw[end:], header["descr"], count)
    dtype = next(dtype for dtype, info in _DTYPE_INFO.items() if info[0] == header["descr"])
    return Tensor(shape, values, dtype)


def read_npz(path: str | Path) -> dict[str, Tensor]:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            names = archive.namelist()
            result: dict[str, Tensor] = {}
            for name in names:
                if not name.endswith(".npy") or "/" in name:
                    continue
                result[name[:-4]] = read_npy(archive.read(name))
    except (OSError, zipfile.BadZipFile) as exc:
        raise FixtureError(f"cannot read source NPZ {path}: {exc}") from exc
    if not result:
        raise FixtureError(f"source NPZ {path} contains no root NPY arrays")
    return result


def array_metadata(arrays: Mapping[str, Tensor]) -> list[dict[str, object]]:
    """Stable manifest metadata for a set of fixture arrays."""

    result = []
    for name in sorted(arrays):
        value = arrays[name]
        result.append(
            {
                "name": name,
                "storage_dtype": value.dtype,
                "shape": list(value.shape),
                "elements": value.size,
                "sha256": hashlib.sha256(tensor_bytes(value)).hexdigest(),
            }
        )
    return result


def write_json(path: str | Path, value: object) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, separators=(",", ": ")) + "\n",
        encoding="utf-8",
    )
