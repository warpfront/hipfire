#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""scripts/k0rank.py: dump parsing, ranking order, v1 schema, provenance, validation."""
import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import k0rank  # noqa: E402

LAYERS, EXPERTS, TOPK = 2, 4, 2
BIND = dict(
    model_sha256="a" * 64, base_size=1234, base_index_sha256="b" * 64,
    format_map="all-mq4-xts (test)", format_map_sha256="c" * 64,
    layers=LAYERS, experts_per_layer=EXPERTS, top_k=TOPK,
)


def call(layer, ids):
    return struct.pack("<BHH", 1, layer, len(ids)) + struct.pack(f"<{len(ids)}i", *ids)


def step_end(phase, idx):
    return struct.pack("<BBI", 2, phase, idx)


def dump(records):
    return k0rank.MAGIC + struct.pack("<III", LAYERS, 2, TOPK) + b"".join(records) + b""


DISCARD = b"\x03"


def write_dump(tmp, name, records, tokens=None):
    d = tmp / name
    d.mkdir()
    (d / "route.bin").write_bytes(dump(records))
    if tokens:
        (d / "run").mkdir()
        (d / "run" / "manifest.json").write_text(json.dumps(
            dict(tokens=tokens, distinct_prompts=2, spec={"spec": "ar"},
                 prompts=[{"sha256": "p1"}, {"sha256": "p2"}])))
    return str(d / "route.bin")


def bind_file(tmp):
    p = tmp / "base.expert-hotset"
    p.write_text(json.dumps(dict(BIND, extra="ignored")))
    return str(p)


@pytest.fixture
def corpus(tmp_path):
    # decode step, 1 row: layer0 -> {1,2}, layer1 -> {1,3}
    a = write_dump(tmp_path, "a", [
        call(0, [1, 2]), call(1, [1, 3]), step_end(0, 0),
        call(0, [1, -1]), call(1, [3, 3]), step_end(0, 1),
        call(0, [0, 0, 0]),  # open call with nslots != top_k: dropped by DISCARD
        DISCARD,
    ], tokens=8)
    # reference pass: never STEP_END, recovered at the first DISCARD (2 layers, 1 row)
    b = write_dump(tmp_path, "b", [
        call(0, [0, 1]), call(1, [0, 1]), DISCARD, call(0, [2, 2]), DISCARD,
    ], tokens=4)
    held = write_dump(tmp_path, "h", [call(0, [0, 1]), call(1, [2, 3]), step_end(0, 0)])
    m = tmp_path / "m.tsv"
    m.write_text(f"{a}\tcal\tcode\tar\n{b}\tcal\tprose\tar\n{held}\theld\tcode\tar\n")
    return tmp_path, str(m)


def test_parse_counts_follow_the_k0_rules(corpus):
    tmp, _ = corpus
    _, counts, info = k0rank.parse_dump(str(tmp / "a" / "route.bin"), EXPERTS)
    assert counts == [0, 2, 1, 0, 0, 1, 0, 3]  # layer-major, -1 padding skipped
    assert info["steps"] == 2 and info["discards"] == 1 and info["slot_reads"] == 7
    assert info["step_classes"]["ar1"] == 2


def test_reference_pass_is_recovered_at_first_discard(corpus):
    tmp, _ = corpus
    _, counts, info = k0rank.parse_dump(str(tmp / "b" / "route.bin"), EXPERTS)
    assert counts == [1, 1, 0, 0, 1, 1, 0, 0]  # the later (2,2) call is dropped
    assert info["recovered_reference"]["steps"] == 1
    assert info["step_classes"]["ar1"] == 1


def test_drop_class_removes_whole_steps(corpus):
    tmp, _ = corpus
    _, counts, info = k0rank.parse_dump(str(tmp / "a" / "route.bin"), EXPERTS, ("ar1",))
    assert sum(counts) == 0 and info["dropped_classes"] == {"ar1": 2}


def test_bad_magic_and_expert_range_are_refused(tmp_path):
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"NOTADUMP" + b"\0" * 16)
    with pytest.raises(ValueError, match="bad magic"):
        k0rank.parse_dump(str(bad), EXPERTS)
    oor = write_dump(tmp_path, "oor", [call(0, [9, 0]), step_end(0, 0)])
    with pytest.raises(ValueError, match="expert id 9"):
        k0rank.parse_dump(oor, EXPERTS)


def test_order_is_count_desc_then_id_asc():
    assert k0rank.order([5, 5, 9, 0, 5]) == [2, 0, 1, 4, 3]


def test_write_emits_a_closed_v1_file_and_provenance(corpus):
    tmp, manifest = corpus
    out = tmp / "k.expert-hotset"
    k0rank.main(["write", "--manifest", manifest, "--split", "cal", "--out", str(out),
                 "--bind-from", bind_file(tmp)])
    doc = json.loads(out.read_text())
    assert tuple(doc) == k0rank.SCHEMA_KEYS  # nothing the closed Rust schema would refuse
    k0rank.validate_doc(doc)
    assert doc["counts"] == [1, 3, 1, 0, 1, 2, 0, 3]
    assert doc["rank"] == [1, 7, 5, 0, 2, 4, 3, 6]
    assert doc["capture_tokens"] == 8 * 2 + 4 * 2
    assert {k: doc[k] for k in BIND} == BIND
    prov = json.loads((tmp / "k.expert-hotset.provenance.json").read_text())
    import hashlib
    assert prov["ranking_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert prov["split"] == "cal" and len(prov["dumps"]) == 2
    assert all(d["sha256"] and d["run"]["prompt_sha256"] for d in prov["dumps"])
    assert prov["tool"] == "k0rank" and prov["units"] == 8


def test_explicit_binding_flags_and_missing_binding(corpus):
    tmp, manifest = corpus
    out = tmp / "k2.expert-hotset"
    with pytest.raises(SystemExit, match="binding incomplete"):
        k0rank.main(["write", "--manifest", manifest, "--out", str(out)])
    k0rank.main(["write", "--manifest", manifest, "--out", str(out), "--model-sha256", "a" * 64,
                 "--base-size", "1234", "--base-index-sha256", "b" * 64, "--format-map", "x",
                 "--format-map-sha256", "c" * 64, "--layers", "2", "--experts-per-layer", "4",
                 "--top-k", "2"])
    assert json.loads(out.read_text())["base_size"] == 1234


def test_validate_rejects_broken_files(corpus):
    tmp, manifest = corpus
    out = tmp / "k3.expert-hotset"
    k0rank.main(["write", "--manifest", manifest, "--out", str(out), "--bind-from", bind_file(tmp)])
    good = json.loads(out.read_text())
    for mutate, msg in [
        (lambda d: d["rank"].__setitem__(0, 0), "permutation"),
        (lambda d: d["rank"].reverse(), "count desc"),
        (lambda d: d["counts"].__setitem__(0, 99), "counts_sha256|count desc"),
        (lambda d: d.update(extra=1), "keys"),
    ]:
        doc = json.loads(json.dumps(good))
        mutate(doc)
        with pytest.raises(ValueError, match=msg):
            k0rank.validate_doc(doc)


def test_coverage_of_held_reads(corpus, capsys):
    tmp, manifest = corpus
    out = tmp / "k4.expert-hotset"
    k0rank.main(["write", "--manifest", manifest, "--out", str(out), "--bind-from", bind_file(tmp)])
    k0rank.main(["validate", str(out), "--manifest", manifest, "--split", "held"])
    text = capsys.readouterr().out
    # held reads hit units 0,1 (layer 0) and 6,7 (layer 1); top 50% of 8 = ids 1,7,0,2
    assert "top 50.00% of units -> 75.00%" in text



def write_route_hist(tmp, name, matrix, tokens, split="cal"):
    np = pytest.importorskip("numpy")
    base = tmp / f"route_hist.pooled.{split}.{name}"
    doc = dict(
        schema="route_hist.v1", source={"path": "shards.tsv", "sha256": "s" * 64},
        model="m.mq4-xts", n_experts=EXPERTS, topk=TOPK,
        records=[{"layer": 0, "n": tokens, "k": TOPK, "position": 0}],
        per_layer=[dict(layer=i, tokens=tokens, counts=row) for i, row in enumerate(matrix)],
        total=dict(tokens=tokens), ids={"path": "shards.tsv", "sha256": "i" * 64}, id="r" * 64,
        x_miheat=dict(**{"class": "pooled"}, split=split, docs=1, head_counts=[1, 2, 3, 4]))
    Path(str(base) + ".json").write_text(json.dumps(doc))
    np.savez(str(base) + ".counts.npz", counts=np.array(matrix, dtype=np.int64))
    return str(base) + ".json"


def test_route_hist_reader_checks_the_npz_and_the_read_total(tmp_path):
    matrix = [[3, 2, 1, 0], [0, 4, 2, 0]]  # 6 slot reads/layer = 3 tokens x top_k 2
    path = write_route_hist(tmp_path, "ok", matrix, 3)
    rh = k0rank.read_route_hist(path)
    assert rh["counts"] == [3, 2, 1, 0, 0, 4, 2, 0]  # global id = layer * experts + expert
    assert (rh["layers"], rh["experts"], rh["tokens"]) == (2, 4, 3)
    assert "head_counts" not in rh["x_miheat"]
    bad = write_route_hist(tmp_path, "bad", [[3, 2, 1, 0], [0, 4, 2, 1]], 3)
    with pytest.raises(ValueError, match="slot reads"):
        k0rank.read_route_hist(bad)
    import numpy as np
    np.savez(path[:-5] + ".counts.npz", counts=np.zeros((2, 4), dtype=np.int64))
    with pytest.raises(ValueError, match="differ"):
        k0rank.read_route_hist(path)


def test_write_from_route_hist_labels_and_orders(tmp_path):
    path = write_route_hist(tmp_path, "w", [[3, 2, 1, 0], [0, 4, 2, 0]], 3)
    out = tmp_path / "mi.expert-hotset"
    k0rank.main(["write", "--route-hist", path, "--capture-label", "gfx942-derived",
                 "--out", str(out), "--bind-from", bind_file(tmp_path)])
    doc = json.loads(out.read_text())
    assert tuple(doc) == k0rank.SCHEMA_KEYS
    k0rank.validate_doc(doc)
    assert doc["counts"] == [3, 2, 1, 0, 0, 4, 2, 0]
    assert doc["rank"] == [5, 0, 1, 6, 2, 3, 4, 7]
    assert doc["capture_tokens"] == 3 and "gfx942-derived" in doc["corpus"]
    prov = json.loads((tmp_path / "mi.expert-hotset.provenance.json").read_text())
    assert prov["source_kind"] == "route_hist.v1" and prov["capture_label"] == "gfx942-derived"
    assert prov["route_hist"]["json_sha256"] and prov["route_hist"]["counts_npz_sha256"]
    assert prov["route_hist"]["id"] == "r" * 64 and prov["split"] == "cal"
    with pytest.raises(SystemExit, match="exactly one"):
        k0rank.main(["write", "--out", str(out), "--bind-from", bind_file(tmp_path)])


def test_static_hit_and_compare_on_route_hist(tmp_path, capsys):
    held_path = write_route_hist(tmp_path, "h", [[3, 2, 1, 0], [0, 4, 2, 0]], 3, split="held")
    rank = k0rank.order([3, 2, 1, 0, 0, 4, 2, 0])
    held = [3, 2, 1, 0, 0, 4, 2, 0]
    # parity ownership: rank 0 owns experts 0,2 ; rank 1 owns 1,3. one unit per rank.
    assert k0rank.static_hit(rank, held, EXPERTS, 1) == pytest.approx(7 / 12)
    assert k0rank.static_hit(rank, held, EXPERTS, 4) == 1.0
    assert k0rank.per_layer_hit(held, EXPERTS, 1) == pytest.approx(0.5)
    ranking = tmp_path / "r.expert-hotset"
    k0rank.main(["write", "--route-hist", held_path, "--out", str(ranking),
                 "--bind-from", bind_file(tmp_path)])
    k0rank.main(["compare", "--ranking", str(ranking), "--held", f"route_hist={held_path}",
                 "--per-rank-units", "1", "--layers-resident", "1"])
    assert "| 58.33% | 50.00% |" in capsys.readouterr().out
