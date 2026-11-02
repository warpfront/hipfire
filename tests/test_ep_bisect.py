#!/usr/bin/env python3
"""Synthetic-dump regression tests for tools/ep_bisect.py."""
import contextlib
import importlib.util
import io
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ep_bisect", REPO_ROOT / "tools" / "ep_bisect.py")
if spec is None or spec.loader is None:
    raise RuntimeError("failed to load tools/ep_bisect.py")
ep_bisect = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ep_bisect  # dataclasses resolves cls.__module__ via sys.modules
spec.loader.exec_module(ep_bisect)


def f32(*v):
    return struct.pack(f"<{len(v)}f", *v)


def i32(*v):
    return struct.pack(f"<{len(v)}i", *v)


def put(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def run(state: Path, moe: Path):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = ep_bisect.main(["--state-dir", str(state), "--moe-dir", str(moe)])
    return code, out.getvalue()


class EpBisectTest(unittest.TestCase):
    def build(self, root: Path, final_ep: bytes):
        state, moe = root / "state", root / "moe"
        for arm in ("single", "ep2"):
            put(state / arm / "layer-map.json", json.dumps({"dn_slot_to_physical_layer": [0, 1, 2]}).encode())
            put(state / arm / "position-0" / "token.u32", i32(7))
            put(state / arm / "position-0" / "position.u64", struct.pack("<Q", 0))
            rank = state / arm / "rank-0" / "position-0"
            put(rank / "kv-k-0.bin", b"\x00" * 8)
            put(rank / "dn-s-1.bin", b"\x01" * 8)
            put(rank / "root-logits.f32", f32(1.0, 2.0))
            layer = moe / arm / "rank-0" / "position-0" / "layer-0"
            put(layer / "route-indices.i32", i32(3, 1))
            put(layer / "route-weights.f32", f32(0.5, 0.5))
            put(layer / "routed-residual-input.f32", f32(1.0, 2.0))
        single = moe / "single" / "rank-0" / "position-0" / "layer-0"
        ep = moe / "ep2" / "rank-0" / "position-0" / "layer-0"
        put(single / "expanded-rows.ABSENT", b"fused down+fold")
        put(single / "meta-fold.json", b'{"path":"ninepath_d4","fold_base":"x_residual_after_shared","expanded":false}')
        put(ep / "expanded-rows.f32", f32(1.0, 1.0, 1.0, 1.0))
        put(ep / "meta-fold.json", b'{"path":"expanded","fold_base":"zeroed_partial","expanded":true}')
        put(single / "fold-output.f32", f32(5.0, 6.0))
        put(ep / "fold-output.f32", f32(1.0, 1.0))
        put(single / "final-residual.f32", f32(5.0, 6.0))
        put(ep / "final-residual.f32", final_ep)
        return state, moe

    def test_localizes_final_residual_and_ignores_structural_rows(self):
        with tempfile.TemporaryDirectory() as td:
            state, moe = self.build(Path(td), f32(5.0, 6.5))
            code, text = run(state, moe)
        self.assertEqual(code, 1, text)
        self.assertIn("position 0, layer 0, tensor final-residual, first differing byte 6", text)
        self.assertIn("max_abs 5.000e-01", text)
        self.assertIn("STRUCT", text)  # fold-output differs by design and is not the answer
        self.assertNotIn("tensor fold-output", text)

    def test_all_equal_exits_zero(self):
        with tempfile.TemporaryDirectory() as td:
            state, moe = self.build(Path(td), f32(5.0, 6.0))
            code, text = run(state, moe)
        self.assertEqual(code, 0, text)
        self.assertIn("EARLIEST DIVERGENT TENSOR: none", text)

    def test_missing_file_without_absence_marker_is_structural_error(self):
        with tempfile.TemporaryDirectory() as td:
            state, moe = self.build(Path(td), f32(5.0, 6.0))
            (moe / "ep2" / "rank-0" / "position-0" / "layer-0" / "route-weights.f32").unlink()
            code, text = run(state, moe)
        self.assertEqual(code, 2, text)
        self.assertIn("route-weights.f32 missing on ep", text)

    def test_dn_state_maps_to_physical_layer(self):
        with tempfile.TemporaryDirectory() as td:
            state, moe = self.build(Path(td), f32(5.0, 6.0))
            (state / "ep2" / "rank-0" / "position-0" / "dn-s-1.bin").write_bytes(b"\x01" * 7 + b"\x05")
            code, text = run(state, moe)
        self.assertEqual(code, 1, text)
        self.assertIn("position 0, layer 1, tensor dn-s-1 (layer 1), first differing byte 7", text)


if __name__ == "__main__":
    unittest.main()
