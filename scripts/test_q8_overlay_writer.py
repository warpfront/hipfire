#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""pytest for scripts/q8_overlay_writer.py (the Q8F16 trunk-overlay payload encoder)."""
import hashlib
import importlib.util
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("q8_overlay_writer", HERE / "q8_overlay_writer.py")
w = importlib.util.module_from_spec(spec)
sys.modules["q8_overlay_writer"] = w
spec.loader.exec_module(w)


def f32(x):
    return np.float32(x)


def scalar_f32_to_f16(val):
    """Independent scalar transcription of float16.rs f32_to_f16 on Python ints."""
    bits = val if isinstance(val, int) else struct.unpack("<I", struct.pack("<f", float(val)))[0]
    sign = (bits >> 31) & 1
    exp = (bits >> 23) & 0xFF
    frac = bits & 0x7FFFFF
    if exp == 0xFF:
        return (sign << 15) | (0x1F << 10) | (0 if frac == 0 else ((frac >> 13) | 1))
    new_exp = exp - 127 + 15
    if new_exp >= 31:
        return (sign << 15) | (0x1F << 10)
    if new_exp <= 0:
        if new_exp < -10:
            return sign << 15
        return (sign << 15) | ((frac | 0x800000) >> (1 - new_exp + 13))
    return (sign << 15) | (new_exp << 10) | (frac >> 13)


def scalar_round(x):
    """Rust f32::round (ties away from zero) on a float32 scalar."""
    t = np.trunc(x)
    return t + np.sign(x) if abs(x - t) >= f32(0.5) else t


def scalar_encode(values):
    """Independent scalar transcription of quantize_q8f16 (all arithmetic in float32)."""
    out = bytearray()
    values = [f32(v) for v in values]
    while len(values) % 32:
        values.append(f32(0))
    for b in range(0, len(values), 32):
        g = values[b : b + 32]
        max_abs = f32(0)
        for v in g:
            max_abs = max(max_abs, abs(v))
        scale = f32(max_abs / f32(127.0))
        inv = f32(f32(1.0) / scale) if scale > 0 else f32(0)
        out += struct.pack("<H", scalar_f32_to_f16(scale))
        for v in g:
            r = scalar_round(f32(v * inv))
            q = int(max(-128.0, min(127.0, float(r)))) if r == r else -128
            out += struct.pack("<b", q)
    return bytes(out)


def test_hand_case_and_tie_rounding():
    v = np.zeros(32, np.float32)
    v[:8] = [127, -127, 63.5, -63.5, 0.5, -0.5, 0.49, 126.5]
    b = w.encode_bytes(v)
    assert len(b) == 34
    assert b[:2] == b"\x00\x3c"  # scale exactly 1.0
    assert np.frombuffer(b[2:], np.int8)[:8].tolist() == [127, -127, 64, -64, 1, -1, 0, 127]


def test_scale_is_truncated_not_rounded_to_nearest():
    # f32 scale 1 + 2^-11 + 2^-12 sits above the f16 midpoint: RNE gives 0x3c01, the producer 0x3c00.
    scale = np.float32(1.0) + np.float32(2.0**-11) + np.float32(2.0**-12)
    v = np.zeros(32, np.float32)
    v[0] = scale * np.float32(127.0)
    b = w.encode_bytes(v)
    assert b[:2] == struct.pack("<H", scalar_f32_to_f16(float(np.float32(v[0] / np.float32(127.0)))))
    assert int.from_bytes(b[:2], "little") != int(np.float16(np.float32(v[0] / np.float32(127.0))).view(np.uint16))


def test_all_zero_block_and_tail_padding():
    assert w.encode_bytes(np.zeros(32, np.float32)) == bytes(34)
    v = np.linspace(-1, 1, 70, dtype=np.float32)  # 70 = two blocks + a 6-value tail, zero padded to 96
    assert len(w.encode_bytes(v)) == 3 * 34


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_vector_port_equals_the_scalar_transcription(seed):
    rng = np.random.default_rng(seed)
    vals = np.concatenate(
        [
            rng.standard_normal(32 * 64).astype(np.float32) * 0.02,
            (rng.standard_t(2.0, 32 * 64) * 0.01).astype(np.float32),
            (rng.standard_normal(32 * 64) * np.exp(rng.uniform(-30, 8, 32 * 64))).astype(np.float32),
            (rng.standard_normal(32 * 32) * 2.0 ** rng.integers(-40, -10, 32 * 32)).astype(np.float32),
        ]
    )
    assert w.encode_bytes(vals) == scalar_encode(vals)


def test_f32_to_f16_matches_scalar_over_many_bit_patterns():
    rng = np.random.default_rng(7)
    bits = np.concatenate(
        [rng.integers(0, 2**32, 1 << 16, dtype=np.uint64).astype(np.uint32), (np.arange(1 << 16, dtype=np.uint32) << 16)]
    )
    x = bits.view(np.float32)
    got = w.f32_to_f16_trunc(x)
    want = np.array([scalar_f32_to_f16(int(b)) for b in bits[:20000]], dtype=np.uint16)
    assert np.array_equal(got[:20000], want)


def test_expected_bytes_formula():
    assert w.expected_bytes(4, 256) == 4 * 8 * 34
    assert w.expected_bytes(2 * 4096, 2560) == 2 * 4096 * 80 * 34
    for bad_k in (0, 48, 100):
        with pytest.raises(ValueError):
            w.expected_bytes(2, bad_k)


RUSTC = shutil.which("rustc")


@pytest.mark.skipif(RUSTC is None, reason="rustc not available")
def test_bit_for_bit_against_the_verbatim_rust_encoder(tmp_path):
    exe = tmp_path / "q8f16_ref"
    subprocess.run([RUSTC, "-O", str(HERE / "q8_overlay_ref" / "q8f16_ref.rs"), "-o", str(exe)], check=True)
    rng = np.random.default_rng(20261101)
    cases = {
        "normal": rng.standard_normal(32 * 2048).astype(np.float32) * 0.02,
        "heavy": (rng.standard_t(2.0, 32 * 2048) * 0.01).astype(np.float32),
        "bf16": ((rng.standard_normal(32 * 2048).astype(np.float32) * 0.03).view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32),
        "wide": (rng.standard_normal(32 * 2048) * np.exp(rng.uniform(-30, 8, 32 * 2048))).astype(np.float32),
        "subnormal_scale": (rng.standard_normal(32 * 1024) * 2.0 ** rng.integers(-40, -10, 32 * 1024)).astype(np.float32),
        "trunc_vs_rne": (rng.uniform(0.5, 1.5, 32 * 2048) * 127.0).astype(np.float32),
        "ties": np.tile(np.array([127, -127, 63.5, -63.5, 0.5, -0.5, 1.5, -2.5] * 4, np.float32), 8),
    }
    for name, vals in cases.items():
        vals = np.nan_to_num(vals, posinf=0, neginf=0).astype(np.float32)
        (tmp_path / "in.f32").write_bytes(vals.astype("<f4").tobytes())
        subprocess.run([str(exe), str(tmp_path / "in.f32"), str(tmp_path / "out.bin")], check=True)
        assert w.encode_bytes(vals) == (tmp_path / "out.bin").read_bytes(), name


def write_safetensors(path, tensors):
    """tensors: name -> (shape, bf16 uint16 array)."""
    header, blobs, off = {}, [], 0
    for name, (shape, bits) in tensors.items():
        raw = bits.astype("<u2").tobytes()
        header[name] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    hdr = json.dumps(header).encode()
    path.write_bytes(len(hdr).to_bytes(8, "little") + hdr + b"".join(blobs))


def bf16_of(arr):
    return (np.ascontiguousarray(arr, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def test_end_to_end_names_sizes_manifest_and_mtp_exclusion(tmp_path):
    rng = np.random.default_rng(3)
    roles = ["q_proj", "k_proj", "v_proj", "o_proj", "indexer.index_qk_proj"]
    shapes = {"q_proj": (8, 64), "k_proj": (4, 64), "v_proj": (4, 64), "o_proj": (64, 32), "indexer.index_qk_proj": (6, 64)}
    tensors, weight_map, expect = {}, {}, {}
    for layer in (11, 3):
        for role in roles:
            name = f"model.language_model.layers.{layer}.self_attn.{role}.weight"
            vals = (rng.standard_normal(shapes[role]) * 0.05).astype(np.float32)
            bits = bf16_of(vals)
            tensors[name] = (shapes[role], bits)
            expect[name] = w.encode_bytes(w.bf16_bits_to_f32(bits))
    # Things that must NOT be selected: the MTP twin, a GDN projection, a norm.
    for other in (
        "mtp.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.5.linear_attn.in_proj_qkv.weight",
        "model.language_model.layers.3.self_attn.q_norm.weight",
    ):
        tensors[other] = ((2, 32), bf16_of(np.ones((2, 32), np.float32)))
    write_safetensors(tmp_path / "model-00001-of-00001.safetensors", tensors)
    weight_map = {n: "model-00001-of-00001.safetensors" for n in tensors}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))

    names = w.qsa_names(sorted(weight_map))
    assert len(names) == 10
    assert names[0].startswith("model.language_model.layers.3.") and names[-1].startswith("model.language_model.layers.11.")
    assert not any("mtp." in n or "linear_attn" in n or "norm" in n for n in names)

    out = tmp_path / "out"
    assert w.main(["encode", "--src", str(tmp_path), "--out", str(out)]) == 0
    manifest = json.loads((out / "q8_overlay_payloads.json").read_text())
    assert manifest["format"] == "qt3" and len(manifest["tensors"]) == 10
    for record in manifest["tensors"]:
        name = record["name"]
        data = (out / f"{name}.bin").read_bytes()
        m, k = record["shape"]
        assert len(data) == m * (k // 32) * 34 == record["bytes"]
        assert data == expect[name]
        assert hashlib.sha256(data).hexdigest() == record["sha256"]
    assert manifest["total_bytes"] == sum(r["bytes"] for r in manifest["tensors"])
    assert not list(out.glob("*.partial"))


def test_refuses_non_bf16_and_unaligned_k(tmp_path):
    name = "model.language_model.layers.3.self_attn.q_proj.weight"
    write_safetensors(tmp_path / "a.safetensors", {name: ((2, 48), bf16_of(np.ones((2, 48), np.float32)))})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {name: "a.safetensors"}}))
    src = w.Source(tmp_path)
    with pytest.raises(ValueError, match="multiple of 32"):
        w.encode_tensor(src, name, tmp_path)
    hdr = json.loads((tmp_path / "a.safetensors").read_bytes()[8 : 8 + int.from_bytes((tmp_path / "a.safetensors").read_bytes()[:8], "little")])
    hdr[name]["dtype"] = "F16"
    body = json.dumps(hdr).encode()
    raw = (tmp_path / "a.safetensors").read_bytes()
    n = int.from_bytes(raw[:8], "little")
    (tmp_path / "b.safetensors").write_bytes(len(body).to_bytes(8, "little") + body + raw[8 + n :])
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {name: "b.safetensors"}}))
    with pytest.raises(ValueError, match="BF16"):
        w.Source(tmp_path).rows(name, 0, 1)


def _flash_next_names():
    names = ["lm_head.weight", "mtp.layers.0.self_attn.q_proj.weight", "model.language_model.layers.0.mlp.gate.weight",
             "model.language_model.layers.0.mlp.shared_expert.up_proj.weight", "model.language_model.layers.0.linear_attn.conv1d.weight"]
    for layer in range(48):
        p = f"model.language_model.layers.{layer}."
        if layer % 4 == 3:
            names += [p + f"self_attn.{r}.weight" for r in ("q_proj", "k_proj", "v_proj", "o_proj", "indexer.index_qk_proj")]
        else:
            names += [p + f"linear_attn.{r}.weight" for r in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")]
    return names


def test_tiers_select_exactly_the_registered_mq6_trunk_classes():
    names = _flash_next_names()
    counts = {t: len(w.tier_names(names, t)) for t in w.TIERS}
    assert counts == {"attn": 60, "gdn_out": 36, "gdn_in": 144, "head": 1, "full": 241}
    full = w.tier_names(names, "full")
    assert len(set(full)) == 241
    assert full == w.tier_names(list(reversed(names)), "full")
    assert set(full) == set(w.tier_names(names, "attn")) | set(w.tier_names(names, "gdn_in")) | set(
        w.tier_names(names, "gdn_out")) | {"lm_head.weight"}
    # BF16-exact classes and the MTP head are never targets.
    assert not any("mtp." in n or "mlp." in n or "conv1d" in n for n in full)
    with pytest.raises(SystemExit):
        w.tier_names(names[:-1] + [], "full")
