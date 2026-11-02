#!/usr/bin/env python3
"""Developer-only CPU bisection of a single-vs-EP qwen35 oracle dump pair.

Compares raw bytes, never loading a whole dump into memory (state files are
streamed in 1 MiB chunks). Python stdlib only; no runtime surface.

Inputs (both are produced by `route_oracle_mesh.rs::qwen35_ep2_vs_single_oracle`
and the opt-in `HIPFIRE_MOE_DUMP_DIR` hook):

  --state-dir  = $HIPFIRE_EP_ORACLE_DUMP
      {single,ep2}/layer-map.json
      {single,ep2}/position-P/{token.u32,position.u64}
      {single,ep2}/rank-R/position-P/{root-logits.f32,
          kv-k-L.bin kv-v-L.bin dn-s-S.bin dn-scales-S.bin dn-conv-S.bin
          dn-ef-S.bin}                      (POST-step state on both arms)
  --moe-dir    = $HIPFIRE_MOE_DUMP_DIR
      {single,ep2}/rank-R/position-P/layer-L/<tensor>.{f32,i32}

State suffixes: `kv-*-L` is the PHYSICAL layer; `dn-*-S` is the COMPACT
DeltaNet slot (S-th LinearAttention layer). layer-map.json translates S to the
physical layer so state and MoE rows sort into one layer order. State written
by layer L depends only on the residual entering layer L, so the first
differing layer-L state means the divergence is in layer L-1's MLP (or
between), which is why state rows sort *before* layer-L MoE rows.

Structural vs meaningful: the single arm's default fused down+fold (ninepath_d4)
never materialises expanded rows (hook writes `expanded-rows.ABSENT` and
meta-fold.json expanded=false -> NA, not a missing-file failure), and the EP
fold target is a zeroed partial (fold_base=zeroed_partial) rather than the
residual-after-shared. Fold input/output rows therefore print as STRUCT and do
not count as the divergence unless both arms declare the same fold_base in
meta-fold.json. `<stem>.DEFERRED` (deferred routed combine) is NA as well.
Common tensors (route, shared in/out, routed-residual-input, final-residual,
logits, state) are always meaningful.

Exit status: 0 all meaningful tensors equal, 1 meaningful divergence,
2 structural error (missing files / mismatched file sets / bad layout).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

CHUNK = 1 << 20  # multiple of every element size

_CODE = {"u8": "B", "i8": "b", "f16": "e", "f32": "f", "i32": "i"}
_ELEM = {"u8": 1, "i8": 1, "f16": 2, "f32": 4, "i32": 4}
_FLOAT = {"f16", "f32"}

STATE_RE = re.compile(r"^(kv-k|kv-v|dn-s|dn-scales|dn-conv|dn-ef)-(\d+)\.bin$")
# Known MoE tensors in dataflow order; unknown *.f32/*.i32 files sort after.
MOE_ORDER = [
    "shared-input",
    "route-indices",
    "route-weights",
    "shared-output",
    "routed-residual-input",
    "expanded-rows",
    "fold-input",
    "fold-output",
    "final-residual",
]
MOE_STAGE0 = 100  # MoE rows sort after the same layer's state rows
MARKER_SUFFIXES = (".ABSENT", ".DEFERRED")
# (dtype, packed). Packed = quantized/bit-packed storage whose byte difference is
# not a numeric error: fwht3 K (scale + 3-bit codes), Q8 V blocks, Q8 dn-s.
STATE_DTYPE = {
    "kv-k": ("u8", True),
    "kv-v": ("u8", True),
    "dn-s": ("i8", True),  # Q8 default; layer-map dn_s_format=f32 -> real f32
    "dn-scales": ("f32", False),
    "dn-conv": ("f32", False),
    "dn-ef": ("f16", False),
}


def state_dtype(kind: str, layer_map: dict | None) -> tuple[str, bool]:
    """Element type for max_abs plus whether the file is packed storage."""
    if kind == "dn-s" and (layer_map or {}).get("dn_s_format") == "f32":
        return "f32", False
    return STATE_DTYPE[kind]


STATE_ORDER = ["kv-k", "kv-v", "dn-s", "dn-scales", "dn-conv", "dn-ef"]
LOGITS_LAYER = 10**9


# ── byte diff core ──────────────────────────────────────────────────────────


@dataclass
class Diff:
    size_a: int
    size_b: int
    first_byte: int | None = None
    n_diff: int = 0  # differing elements within the compared window
    max_abs: float = 0.0
    nan_diffs: int = 0
    dtype: str = "u8"

    @property
    def sizes_match(self) -> bool:
        return self.size_a == self.size_b

    @property
    def equal(self) -> bool:
        return self.sizes_match and self.first_byte is None


def _accumulate(d: Diff, a: bytes, b: bytes, base: int) -> None:
    if d.first_byte is None:
        i = next(i for i in range(len(a)) if a[i] != b[i])
        d.first_byte = base + i
    es = _ELEM[d.dtype]
    n = len(a) // es
    fmt = "<%d%s" % (n, _CODE[d.dtype])
    xs = struct.unpack(fmt, a[: n * es])
    ys = struct.unpack(fmt, b[: n * es])
    is_float = d.dtype in _FLOAT
    for x, y in zip(xs, ys):
        if x == y:
            continue
        if is_float:
            if x != x or y != y:
                # NaN involved: bit-different (chunk compare said so) but no
                # meaningful magnitude; counted separately.
                d.n_diff += 1
                d.nan_diffs += 1
                continue
            if x in (float("inf"), float("-inf")) or y in (float("inf"), float("-inf")):
                d.n_diff += 1
                d.max_abs = float("inf")
                continue
        d.n_diff += 1
        delta = abs(x - y)
        if delta > d.max_abs:
            d.max_abs = float(delta)
    # +0.0 vs -0.0 compare equal as numbers but differ in bytes: first_byte is
    # set and n_diff stays 0 for them, which the report prints as "sign-only".


def diff_range(
    pa: Path,
    pb: Path,
    dtype: str,
    start: int = 0,
    length: int | None = None,
) -> Diff:
    sa, sb = os.path.getsize(pa), os.path.getsize(pb)
    end = min(sa, sb)
    if length is not None:
        end = min(end, start + length)
    d = Diff(sa, sb, dtype=dtype)
    pos = start
    with open(pa, "rb") as fa, open(pb, "rb") as fb:
        fa.seek(start)
        fb.seek(start)
        while pos < end:
            n = min(CHUNK, end - pos)
            a, b = fa.read(n), fb.read(n)
            if len(a) != n or len(b) != n:
                raise OSError(f"short read comparing {pa} / {pb} at offset {pos}")
            if a != b:
                _accumulate(d, a, b, pos)
            pos += n
    if length is None and sa != sb and d.first_byte is None:
        d.first_byte = min(sa, sb)  # equal prefix; sizes differ
    return d


# ── result rows ─────────────────────────────────────────────────────────────

# status: EQUAL | DIFF (meaningful) | STRUCT (differs, structural by design)
#         | NA (absent by metadata) | MISSING (file-set error)


@dataclass
class Row:
    pos: int
    layer: int
    stage: int
    name: str
    status: str
    first_byte: int | None = None
    first_elem: int | None = None
    n_diff: int = 0
    max_abs: float = 0.0
    note: str = ""
    packed: bool = False  # max_abs is a raw stored-byte delta, not a numeric error

    @property
    def key(self) -> tuple[int, int, int]:
        return (self.pos, self.layer, self.stage)


@dataclass
class Report:
    rows: list[Row] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    ignored: dict[str, int] = field(default_factory=dict)

    def error(self, msg: str) -> None:
        self.errors.append(msg)


def _row_from_diff(
    pos: int, layer: int, stage: int, name: str, d: Diff, status_if_diff: str
) -> Row:
    es = _ELEM[d.dtype]
    if d.equal:
        return Row(pos, layer, stage, name, "EQUAL")
    note = []
    if not d.sizes_match:
        note.append(f"size {d.size_a} != {d.size_b} (compared common prefix)")
    if d.n_diff == 0 and d.first_byte is not None and d.sizes_match:
        note.append("bytes differ, values compare equal (+0.0/-0.0)")
    if d.nan_diffs:
        note.append(f"{d.nan_diffs} NaN-involved elems")
    return Row(
        pos,
        layer,
        stage,
        name,
        status_if_diff,
        d.first_byte,
        None if d.first_byte is None else d.first_byte // es,
        d.n_diff,
        d.max_abs,
        "; ".join(note),
    )


# ── layout helpers ──────────────────────────────────────────────────────────


def _numbered(parent: Path, prefix: str) -> dict[int, Path]:
    out: dict[int, Path] = {}
    if parent.is_dir():
        for child in parent.iterdir():
            m = re.fullmatch(prefix + r"-(\d+)", child.name)
            if m and child.is_dir():
                out[int(m.group(1))] = child
    return out


def parse_positions(spec: str | None) -> set[int] | None:
    if not spec:
        return None
    out: set[int] = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def load_layer_map(state_dir: Path, arms: list[str], report: Report) -> dict | None:
    maps = []
    for arm in arms:
        p = state_dir / arm / "layer-map.json"
        if not p.is_file():
            report.error(f"MISSING {p}: dn-*-S slots cannot be mapped to physical layers")
            return None
        maps.append(json.loads(p.read_text()))
    if maps[0] != maps[1]:
        report.error(f"layer-map.json differs between arms {arms[0]} / {arms[1]}")
    return maps[0]


def state_layer(kind: str, idx: int, layer_map: dict | None) -> int:
    if kind.startswith("kv-"):
        return idx
    slots = (layer_map or {}).get("dn_slot_to_physical_layer")
    if slots is not None and idx < len(slots):
        return int(slots[idx])
    return LOGITS_LAYER - 1  # unmapped: sort after every real layer


def read_meta(layer_dir: Path) -> dict:
    """Hook meta-{shared,fold,final}.json keyed by suffix: meta['fold']['fold_base']."""
    return {
        p.name[len("meta-") : -len(".json")]: json.loads(p.read_text())
        for p in sorted(layer_dir.glob("meta-*.json"))
    }


def declared_absent(layer_dir: Path, meta: dict, stem: str) -> str | None:
    """Why `stem` is legitimately not comparable on this arm, else None.

    The hook writes `<stem>.ABSENT` (reason inside) when the stage did not
    materialise the tensor (e.g. fused down+fold has no expanded rows) and
    `<stem>.DEFERRED` when the routed combine is deferred to the next layer.
    """
    for suffix in MARKER_SUFFIXES:
        marker = layer_dir / (stem + suffix)
        if marker.is_file():
            text = marker.read_text(errors="replace").strip()
            return f"{stem}{suffix}: {text}" if text else f"{stem}{suffix}"
    if stem == "expanded-rows" and meta.get("fold", {}).get("expanded") is False:
        return "meta-fold.json expanded=false"
    return None


# ── state compare ───────────────────────────────────────────────────────────


def compare_state_position(
    pos: int, da: Path, db: Path, layer_map: dict | None, args, report: Report
) -> None:
    """da/db = <arm>/rank-R/position-P under --state-dir."""
    names: dict[Path, set[str]] = {}
    for d in (da, db):
        if not d.is_dir():
            report.error(f"MISSING state directory {d}")
            return
        names[d] = {p.name for p in d.iterdir() if p.is_file()}
    state_a = {n for n in names[da] if STATE_RE.match(n)}
    state_b = {n for n in names[db] if STATE_RE.match(n)}
    for n in sorted(state_a - state_b):
        report.error(f"pos {pos}: state file {n} only in {da.parent.parent.name}")
    for n in sorted(state_b - state_a):
        report.error(f"pos {pos}: state file {n} only in {db.parent.parent.name}")
    for d in (da, db):
        for n in names[d] - state_a - state_b:
            if n == "root-logits.f32":
                continue
            kind = "scratch" if n.startswith("scratch-") else "other"
            report.ignored[kind] = report.ignored.get(kind, 0) + 1
    for n in sorted(state_a & state_b):
        m = STATE_RE.match(n)
        kind, idx = m.group(1), int(m.group(2))
        dtype, packed = state_dtype(kind, layer_map)
        layer = state_layer(kind, idx, layer_map)
        stage = STATE_ORDER.index(kind)
        label = f"{kind}-{idx}"
        if kind.startswith("dn-") and layer < LOGITS_LAYER - 1:
            label += f" (layer {layer})"
        d = diff_range(da / n, db / n, dtype)
        r = _row_from_diff(pos, layer, stage, label, d, "DIFF")
        r.packed = packed
        if r.status == "DIFF" and not d.sizes_match:
            report.error(f"pos {pos}: {n} size {d.size_a} != {d.size_b}")
        report.rows.append(r)
    # Logits (after every layer).
    la, lb = da / "root-logits.f32", db / "root-logits.f32"
    if la.is_file() and lb.is_file():
        report.rows.append(
            _row_from_diff(pos, LOGITS_LAYER, 0, "root-logits", diff_range(la, lb, "f32"), "DIFF")
        )
    elif la.is_file() != lb.is_file():
        report.error(f"pos {pos}: root-logits.f32 missing in one arm")


def compare_tokens(pos: int, ta: Path, tb: Path, report: Report) -> None:
    for name in ("token.u32", "position.u64"):
        a, b = ta / name, tb / name
        if not (a.is_file() and b.is_file()):
            report.error(f"pos {pos}: {name} missing ({a.is_file()=}, {b.is_file()=})")
        elif a.read_bytes() != b.read_bytes():
            report.error(f"pos {pos}: {name} differs between arms — runs are not comparable")


# ── MoE compare ─────────────────────────────────────────────────────────────


def _stem_dtype(stem: str, ext: str) -> str:
    return {"f32": "f32", "i32": "i32"}.get(ext, "u8")


def _stems(layer_dirs: list[Path]) -> list[tuple[str, str]]:
    found: dict[str, str] = {}
    for d in layer_dirs:
        for p in d.iterdir():
            if p.name.startswith("meta-") and p.name.endswith(".json"):
                continue
            marker = next((s for s in MARKER_SUFFIXES if p.name.endswith(s)), None)
            if marker:
                found.setdefault(p.name[: -len(marker)], "f32")
                continue
            stem, _, ext = p.name.rpartition(".")
            if ext in ("f32", "i32", "bin") and stem:
                found[stem] = ext
    order = {s: i for i, s in enumerate(MOE_ORDER)}
    return sorted(found.items(), key=lambda kv: (order.get(kv[0], len(order)), kv[0]))


def _route_note(pa: Path, pb: Path) -> str:
    a, b = pa.read_bytes(), pb.read_bytes()
    n = min(len(a), len(b)) // 4
    xs = struct.unpack("<%di" % n, a[: n * 4])
    ys = struct.unpack("<%di" % n, b[: n * 4])
    slots = [i for i, (x, y) in enumerate(zip(xs, ys)) if x != y]
    kind = "same expert set, different slot order" if sorted(xs) == sorted(ys) else "different expert set"
    return f"{kind}; differing slots {slots}; single={list(xs)} ep={list(ys)}"


def compare_layer(
    pos: int, layer: int, da: Path, db: Path, args, report: Report
) -> None:
    ma, mb = read_meta(da), read_meta(db)
    n_slots = None
    for d in (da, db):
        ri = d / "route-indices.i32"
        if ri.is_file():
            n_slots = ri.stat().st_size // 4
            break
    for stem, ext in _stems([da, db]):
        fname = f"{stem}.{ext}"
        pa, pb = da / fname, db / fname
        stage = MOE_STAGE0 + (MOE_ORDER.index(stem) if stem in MOE_ORDER else len(MOE_ORDER))
        why_a = declared_absent(da, ma, stem)
        why_b = declared_absent(db, mb, stem)
        if why_a or why_b:
            who = [n for n, w in (("single", why_a), ("ep", why_b)) if w]
            reason = why_a or why_b
            report.rows.append(
                Row(pos, layer, stage, stem, "NA", note=f"not valid on {'+'.join(who)}: {reason}")
            )
            continue
        if not pa.is_file() or not pb.is_file():
            side = "single" if not pa.is_file() else "ep"
            if not pa.is_file() and not pb.is_file():
                side = "both arms"
            report.error(f"pos {pos} layer {layer}: {fname} missing on {side} with no absence metadata")
            report.rows.append(Row(pos, layer, stage, stem, "MISSING", note=f"missing on {side}"))
            continue
        dtype = _stem_dtype(stem, ext)
        d = diff_range(pa, pb, dtype)
        status = "DIFF"
        note_extra = ""
        if stem.startswith("fold-"):
            fa, fb = ma.get("fold", {}).get("fold_base"), mb.get("fold", {}).get("fold_base")
            if fa is None or fa != fb:
                status = "STRUCT"
                note_extra = (
                    f"fold target differs by design (single={fa or 'undeclared'}"
                    f"[{ma.get('fold', {}).get('path', '?')}], ep={fb or 'undeclared'}"
                    f"[{mb.get('fold', {}).get('path', '?')}])"
                )
        r = _row_from_diff(pos, layer, stage, stem, d, status)
        if r.status != "EQUAL":
            if note_extra:
                r.note = "; ".join(x for x in (note_extra, r.note) if x)
            if stem == "route-indices" and d.sizes_match:
                r.note = "; ".join(x for x in (_route_note(pa, pb), r.note) if x)
            if not d.sizes_match:
                report.error(f"pos {pos} layer {layer}: {fname} size {d.size_a} != {d.size_b}")
        report.rows.append(r)
        if stem == "expanded-rows" and not d.equal and d.sizes_match and n_slots:
            _slot_rows(pos, layer, pa, pb, d.size_a, n_slots, stage, report)


def _slot_rows(pos, layer, pa, pb, size, n_slots, stage, report: Report) -> None:
    if size % n_slots:
        report.error(f"pos {pos} layer {layer}: expanded-rows size {size} not divisible by {n_slots} slots")
        return
    slot_bytes = size // n_slots
    for s in range(n_slots):
        d = diff_range(pa, pb, "f32", s * slot_bytes, slot_bytes)
        if d.equal:
            continue
        # first_byte/elem here are absolute offsets within expanded-rows.f32.
        report.rows.append(
            _row_from_diff(pos, layer, stage, f"expanded-rows[slot {s}]", d, "DIFF")
        )


def compare_moe_position(pos: int, da: Path, db: Path, args, report: Report) -> None:
    la, lb = _numbered(da, "layer"), _numbered(db, "layer")
    if not la and not lb:
        report.error(f"pos {pos}: no layer-L directories under {da} or {db}")
        return
    for layer in sorted(set(la) - set(lb)):
        report.error(f"pos {pos}: layer-{layer} MoE dump only on single")
    for layer in sorted(set(lb) - set(la)):
        report.error(f"pos {pos}: layer-{layer} MoE dump only on ep")
    for layer in sorted(set(la) & set(lb)):
        compare_layer(pos, layer, la[layer], lb[layer], args, report)


# ── driver / output ─────────────────────────────────────────────────────────


def _fmt_abs(v: float) -> str:
    return "inf" if v == float("inf") else f"{v:.3e}"


def _abs_str(r: Row) -> str:
    if r.packed:
        return f"{r.max_abs:.0f}B*"
    return _fmt_abs(r.max_abs)


def print_report(report: Report, args) -> int:
    rows = sorted(report.rows, key=lambda r: r.key)
    positions = sorted({r.pos for r in rows})
    for pos in positions:
        prow = [r for r in rows if r.pos == pos]
        eq = sum(r.status == "EQUAL" for r in prow)
        print(f"\n=== position {pos}: {len(prow)} tensors, {eq} equal ===")
        shown = [r for r in prow if args.all or r.status != "EQUAL"]
        if not shown:
            print("  all compared tensors bitwise equal")
            continue
        print(f"  {'layer':>5} {'tensor':<34} {'status':<7} {'first_byte':>11} {'elem':>9} {'n_diff':>8} {'max_abs':>10}  note")
        for r in shown:
            layer = "head" if r.layer >= LOGITS_LAYER - 1 else str(r.layer)
            fb = "-" if r.first_byte is None else str(r.first_byte)
            fe = "-" if r.first_elem is None else str(r.first_elem)
            ma = "-" if r.status in ("EQUAL", "NA", "MISSING") else _abs_str(r)
            print(f"  {layer:>5} {r.name:<34} {r.status:<7} {fb:>11} {fe:>9} {r.n_diff:>8} {ma:>10}  {r.note}")

    print("\n=== summary ===")
    if report.ignored:
        print("ignored (not compared; scratch optional / EP-only agreement files): "
              + ", ".join(f"{k}={v}" for k, v in sorted(report.ignored.items())))
    first_by_pos = []
    for pos in positions:
        first = next((r for r in rows if r.pos == pos and r.status == "DIFF"), None)
        if first:
            first_by_pos.append(first)
    struct_rows = [r for r in rows if r.status == "STRUCT"]
    if struct_rows:
        print(f"structural (by design, not counted): {len(struct_rows)} rows "
              f"({', '.join(sorted({r.name for r in struct_rows}))}); use --all to list")
    na = sorted({r.name for r in rows if r.status == "NA"})
    if na:
        print(f"not valid on an arm (declared absent): {', '.join(na)}")
    print("first meaningful divergence per position:")
    if not first_by_pos:
        print("  none — every meaningful compared tensor is bitwise equal")
    for r in first_by_pos:
        layer = "head" if r.layer >= LOGITS_LAYER - 1 else r.layer
        print(f"  pos {r.pos}: layer {layer} {r.name} first_byte={r.first_byte} elem={r.first_elem} max_abs={_abs_str(r)}")
    exit_code = 0
    if first_by_pos:
        g = min(first_by_pos, key=lambda r: r.key)
        layer = "head" if g.layer >= LOGITS_LAYER - 1 else g.layer
        print(f"\nGLOBAL EARLIEST DIVERGENT TENSOR: position {g.pos}, layer {layer}, "
              f"tensor {g.name}, first differing byte {g.first_byte} (elem {g.first_elem}), "
              f"max_abs {_abs_str(g)}")
        if g.note:
            print(f"  note: {g.note}")
        if g.stage <= len(STATE_ORDER) and g.layer < LOGITS_LAYER - 1 and g.name.split("-")[0] in ("kv", "dn"):
            print(f"  hint: layer-{g.layer} state depends only on the residual entering layer {g.layer}; "
                  f"divergence is inside layer {g.layer - 1}'s MLP or between (check its MoE rows"
                  f"{'' if args.moe_dir else ' — pass --moe-dir'}).")
        elif g.name == "root-logits":
            print("  hint: every per-layer tensor equal and only logits differ -> head/final norm under EP, not MoE.")
        exit_code = 1
    else:
        print("\nGLOBAL EARLIEST DIVERGENT TENSOR: none")
    if any(r.packed and r.status == "DIFF" for r in rows):
        print("\nlegend: 'NB*' = largest raw stored-byte difference of a packed/quantized state file "
              "(fwht3 K, Q8 V, Q8 dn-s); NOT a numeric error. Use first_byte for those.")
    if report.errors:
        print(f"\n=== STRUCTURAL ERRORS ({len(report.errors)}) ===")
        for e in report.errors:
            print(f"  {e}")
        exit_code = 2
    return exit_code


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state-dir", type=Path, help="$HIPFIRE_EP_ORACLE_DUMP root (single/ and ep2/ inside)")
    ap.add_argument("--moe-dir", type=Path, help="$HIPFIRE_MOE_DUMP_DIR root (single/ and ep2/ inside)")
    ap.add_argument("--single-arm", default="single")
    ap.add_argument("--ep-arm", default="ep2")
    ap.add_argument("--single-rank", type=int, default=0)
    ap.add_argument("--ep-rank", type=int, default=0)
    ap.add_argument("--positions", help="e.g. 0,3,5-9 (default: all present)")
    ap.add_argument("--all", action="store_true", help="also list equal and structural rows")
    args = ap.parse_args(argv)
    if not args.state_dir and not args.moe_dir:
        ap.error("need --state-dir and/or --moe-dir")

    report = Report()
    want = parse_positions(args.positions)
    arms = [args.single_arm, args.ep_arm]
    ranks = [args.single_rank, args.ep_rank]

    def rank_pos(root: Path, arm: str, rank: int) -> Path:
        return root / arm / f"rank-{rank}"

    pos_sets = []
    for root in (args.state_dir, args.moe_dir):
        if root is None:
            continue
        for arm, rank in zip(arms, ranks):
            if not (root / arm).is_dir():
                print(f"error: {root / arm} does not exist", file=sys.stderr)
                return 2
        per_arm = [set(_numbered(rank_pos(root, a, r), "position")) for a, r in zip(arms, ranks)]
        if per_arm[0] != per_arm[1]:
            report.error(
                f"{root}: position sets differ: only {arms[0]}={sorted(per_arm[0] - per_arm[1])} "
                f"only {arms[1]}={sorted(per_arm[1] - per_arm[0])}"
            )
        pos_sets.append(per_arm[0] & per_arm[1])
    positions = sorted(set.union(*pos_sets) if pos_sets else set())
    if want is not None:
        positions = [p for p in positions if p in want]
    if not positions:
        print("error: no comparable positions found", file=sys.stderr)
        return 2

    layer_map = load_layer_map(args.state_dir, arms, report) if args.state_dir else None
    for pos in positions:
        if args.state_dir:
            sa = rank_pos(args.state_dir, arms[0], ranks[0]) / f"position-{pos}"
            sb = rank_pos(args.state_dir, arms[1], ranks[1]) / f"position-{pos}"
            compare_tokens(pos, args.state_dir / arms[0] / f"position-{pos}",
                           args.state_dir / arms[1] / f"position-{pos}", report)
            compare_state_position(pos, sa, sb, layer_map, args, report)
        if args.moe_dir:
            ma = rank_pos(args.moe_dir, arms[0], ranks[0]) / f"position-{pos}"
            mb = rank_pos(args.moe_dir, arms[1], ranks[1]) / f"position-{pos}"
            compare_moe_position(pos, ma, mb, args, report)
    return print_report(report, args)


if __name__ == "__main__":
    sys.exit(main())
