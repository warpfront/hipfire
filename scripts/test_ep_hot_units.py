#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""scripts/ep_hot_units.py: skew-log parsing, forward grouping, counting, schema, binding."""
import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ep_hot_units  # noqa: E402

LAYERS, EXPERTS, RANKS = 2, 4, 2
RANKING = dict(
    version=1, kind="hipfire.qwen4.expert_hotset", model_sha256="a" * 64, base_size=1234,
    base_index_sha256="b" * 64, counts_sha256="c" * 64, format_map="x", format_map_sha256="d" * 64,
    layers=LAYERS, experts_per_layer=EXPERTS, top_k=2, capture_tokens=8, ranking_rule="r",
    corpus="c", rank=list(range(LAYERS * EXPERTS)), counts=[0] * (LAYERS * EXPERTS),
)


def line(rank, call, layer, tokens):
    return f"{rank} 0 {call} {1000 + call} {layer} {len(tokens)} {' '.join(tokens)}"


def write_log(directory, rank, forwards):
    """`forwards`: list of forwards, each a list of (layer, [tokens])."""
    directory.mkdir(parents=True, exist_ok=True)
    lines, call = [], 0
    for forward in forwards:
        for layer, tokens in forward:
            lines.append(line(rank, call, layer, tokens))
            call += 1
    (directory / f"skew-rank{rank}.log").write_text("\n".join(lines) + "\n")


@pytest.fixture
def dirs(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    # rank 0 owns experts 0 and 2. F1 and F2 are complete; the last forward is cut after layer 0.
    write_log(a, 0, [
        [(0, ["0T", "2C", "1F", "0T"]), (1, ["2H", "0L"])],
        [(0, ["0H"]), (1, ["2T", "3T"])],  # 3T: not owned by rank 0, never counted
        [(0, ["0T"])],  # incomplete forward
    ])
    # rank 1 owns experts 1 and 3.
    write_log(a, 1, [[(0, ["1T", "1T", "0H"]), (1, ["3H"])]])
    write_log(b, 0, [[(0, ["0T"]), (1, ["2T"])]])
    write_log(b, 1, [[(0, ["1H"]), (1, ["3F"])]])
    return a, b


def ranking_file(tmp_path, **override):
    path = tmp_path / "m.ep.expert-hotset"
    path.write_text(json.dumps(dict(RANKING, **override)))
    return str(path)


def test_forwards_are_grouped_by_layer_reset_and_incomplete_ones_dropped(dirs):
    a, _ = dirs
    counts, forwards, incomplete = ep_hot_units.parse_skew_file(
        str(a / "skew-rank0.log"), 0, LAYERS, EXPERTS, RANKS)
    # unit = layer * 4 + expert: (0,0)=0 twice, (1,2)=6 twice; (0,2) is class C, never counted.
    assert counts == [2, 0, 0, 0, 0, 0, 2, 0]
    assert (forwards, incomplete) == (2, 1)


def test_only_host_classes_of_owned_experts_count_once_per_forward(dirs):
    a, _ = dirs
    counts, forwards, _ = ep_hot_units.parse_skew_file(
        str(a / "skew-rank1.log"), 1, LAYERS, EXPERTS, RANKS)
    # `1T 1T` is one read in the forward; `0H` is the other rank's; `3H` is unit 7.
    assert counts == [0, 1, 0, 0, 0, 0, 0, 1] and forwards == 1


def test_directories_pool_per_rank(dirs):
    counts, forwards, incomplete = ep_hot_units.calibrate([str(d) for d in dirs], LAYERS, EXPERTS, RANKS)
    assert counts == [3, 2, 0, 0, 0, 0, 3, 1]
    assert forwards == [3, 2] and incomplete == [1, 0]


def test_a_cut_last_line_is_dropped_not_parsed(tmp_path):
    d = tmp_path / "cut"
    write_log(d, 0, [[(0, ["0T"]), (1, ["2T"])]])
    p = d / "skew-rank0.log"
    p.write_text(p.read_text() + "0 0 9 9 0 2 0T")  # no newline: a write cut mid-line
    counts, forwards, incomplete = ep_hot_units.parse_skew_file(str(p), 0, LAYERS, EXPERTS, RANKS)
    assert forwards == 1 and incomplete == 1 and counts[0] == 1


@pytest.mark.parametrize("bad,match", [
    ("0 0 0 1 0 1 9T", "expert 9"),
    ("0 0 0 1 0 1 0X", "bad token"),
    ("0 0 0 1 0 1 0", "bad token"),
    ("1 0 0 1 0 1 0T", "rank 1 in the log of rank 0"),
    ("0 0 0 1 0 2 0T", "2 slots but 1 tokens"),
    ("0 0 0 1 5 1 0T", "layer 5"),
    ("0 0 0 1 0", "expected"),
    ("0 0 x 1 y 1 0T", "non-integer"),
])
def test_malformed_lines_name_the_line(tmp_path, bad, match):
    p = tmp_path / "skew-rank0.log"
    p.write_text(bad + "\n")
    with pytest.raises(ValueError, match=match):
        ep_hot_units.parse_skew_file(str(p), 0, LAYERS, EXPERTS, RANKS)


def test_a_directory_missing_a_rank_log_is_refused(tmp_path):
    d = tmp_path / "half"
    write_log(d, 0, [[(0, ["0T"]), (1, ["2T"])]])
    with pytest.raises(ValueError, match="skew-rank1.log"):
        ep_hot_units.calibrate([str(d)], LAYERS, EXPERTS, RANKS)


def test_write_emits_a_closed_deterministic_one_line_v1_file(dirs, tmp_path, capsys):
    ranking = ranking_file(tmp_path)
    out1, out2 = tmp_path / "o1.ep.hot-units", tmp_path / "o2.ep.hot-units"
    for out in (out1, out2):
        argv = ["write", "--ranking", ranking, "--corpus", "held-out mixed", "--out", str(out)]
        for d in dirs:
            argv += ["--skew-dir", str(d)]
        assert ep_hot_units.main(argv) == 0
    assert out1.read_bytes() == out2.read_bytes()
    text = out1.read_text()
    assert text.count("\n") == 1 and text.endswith("\n")
    doc = json.loads(text)
    assert tuple(doc) == ep_hot_units.SCHEMA_KEYS  # nothing the closed Rust schema would refuse
    assert doc["kind"] == "hipfire.qwen4.ep_hot_units" and doc["version"] == 1
    assert doc["model_sha256"] == "a" * 64 and doc["base_size"] == 1234
    assert doc["base_index_sha256"] == "b" * 64
    assert doc["base_ranking_sha256"] == hashlib.sha256(Path(ranking).read_bytes()).hexdigest()
    assert (doc["unit_bytes"], doc["layers"], doc["experts_per_layer"], doc["ranks"]) == (
        2534400, LAYERS, EXPERTS, RANKS)
    assert doc["corpus"] == "held-out mixed"
    assert doc["capture_forwards"] == [3, 2]
    assert doc["counts"] == [3, 2, 0, 0, 0, 0, 3, 1]
    out = capsys.readouterr().out
    assert "rank 0: 3 forwards, 2 nonzero units" in out
    assert "rank 1: 2 forwards, 2 nonzero units" in out
    assert "loan  512 MiB" in out and "loan 1536 MiB" in out
    assert ep_hot_units.main(["validate", str(out1), "--ranking", ranking]) == 0


def test_unit_bytes_and_ranks_flags_are_written(dirs, tmp_path):
    ranking = ranking_file(tmp_path)
    out = tmp_path / "u.ep.hot-units"
    argv = ["write", "--ranking", ranking, "--corpus", "c", "--out", str(out),
            "--unit-bytes", "2611200", "--skew-dir", str(dirs[0]), "--skew-dir", str(dirs[1])]
    ep_hot_units.main(argv)
    assert json.loads(out.read_text())["unit_bytes"] == 2611200


def test_write_needs_a_skew_dir_and_a_ranking(tmp_path):
    ranking = ranking_file(tmp_path)
    with pytest.raises(SystemExit, match="--skew-dir"):
        ep_hot_units.main(["write", "--ranking", ranking, "--corpus", "c", "--out", str(tmp_path / "o")])
    with pytest.raises(SystemExit, match="ep_hot_units"):
        ep_hot_units.main(["write", "--ranking", str(tmp_path / "nope"), "--corpus", "c",
                           "--out", str(tmp_path / "o"), "--skew-dir", str(tmp_path)])
    not_ranking = tmp_path / "k.json"
    not_ranking.write_text(json.dumps({"kind": "other"}))
    with pytest.raises(SystemExit, match="not a hipfire.qwen4.expert_hotset"):
        ep_hot_units.main(["write", "--ranking", str(not_ranking), "--corpus", "c",
                           "--out", str(tmp_path / "o"), "--skew-dir", str(tmp_path)])


def good_doc(tmp_path):
    ranking_path = ranking_file(tmp_path)
    ranking, sha = ep_hot_units.read_ranking(ranking_path)
    doc = ep_hot_units.build_doc(ranking, sha, [3, 2, 0, 0, 0, 0, 3, 1], [3, 2], RANKS, 2534400, "corpus")
    return doc, ranking, sha


def test_validate_accepts_the_good_doc_and_refuses_each_break(tmp_path):
    doc, ranking, sha = good_doc(tmp_path)
    ep_hot_units.validate_doc(doc, ranking, sha)

    def broken(**change):
        bad = copy.deepcopy(doc)
        bad.update(change)
        return bad

    cases = [
        (broken(extra=1), "closed v1 schema"),
        (broken(kind="hipfire.qwen4.expert_hotset"), "kind"),
        (broken(version=2), "version"),
        (broken(model_sha256="A" * 64), "model_sha256"),
        (broken(base_index_sha256="abc"), "base_index_sha256"),
        (broken(base_ranking_sha256="z" * 64), "base_ranking_sha256"),
        (broken(counts=[1, 2, 3]), "counts has 3"),
        (broken(capture_forwards=[3]), "capture_forwards has 1"),
        (broken(capture_forwards=[3, -1]), "capture_forwards"),
        (broken(unit_bytes=0), "positive"),
        (broken(corpus="  "), "corpus"),
        (broken(counts=[4, 2, 0, 0, 0, 0, 3, 1]), "exceeds the 3 forwards"),
    ]
    for bad, match in cases:
        with pytest.raises(ValueError, match=match):
            ep_hot_units.validate_doc(bad, ranking, sha)
    missing = copy.deepcopy(doc)
    del missing["corpus"]
    with pytest.raises(ValueError, match="missing"):
        ep_hot_units.validate_doc(missing, ranking, sha)


def test_validate_checks_the_binding_to_the_ranking(tmp_path):
    doc, ranking, sha = good_doc(tmp_path)
    with pytest.raises(ValueError, match="base_ranking_sha256"):
        ep_hot_units.validate_doc(doc, ranking, "e" * 64)
    for key, value in (("model_sha256", "f" * 64), ("base_size", 99), ("base_index_sha256", "f" * 64),
                       ("layers", 3), ("experts_per_layer", 8)):
        other = dict(ranking, **{key: value})
        with pytest.raises(ValueError, match=key):
            ep_hot_units.validate_doc(doc, other, sha)


def test_validate_cli_exits_on_a_ranking_it_was_not_calibrated_under(dirs, tmp_path):
    ranking = ranking_file(tmp_path)
    out = tmp_path / "v.ep.hot-units"
    ep_hot_units.main(["write", "--ranking", ranking, "--corpus", "c", "--out", str(out),
                       "--skew-dir", str(dirs[0]), "--skew-dir", str(dirs[1])])
    other = tmp_path / "other.ep.expert-hotset"
    other.write_text(json.dumps(dict(RANKING, corpus="another ranking")))
    with pytest.raises(SystemExit, match="base_ranking_sha256"):
        ep_hot_units.main(["validate", str(out), "--ranking", str(other)])


def test_coverage_lines_report_units_and_mib_at_each_loan_size():
    doc = dict(experts_per_layer=EXPERTS, ranks=RANKS, unit_bytes=256 << 20, capture_forwards=[10, 5],
               counts=[6, 4, 0, 0, 0, 0, 3, 1])
    text = "\n".join(ep_hot_units.coverage_lines(doc))
    # Rank 0 owns units 0 and 6 (counts 6, 3): 512 MiB fits two 256 MiB units = all its reads.
    assert "rank 0: 10 forwards, 2 nonzero units (9 unit-reads)" in text
    assert "loan  512 MiB: top 2 units = 512.0 MiB cover 100.0%" in text
    # Rank 1 owns units 1 and 7 (counts 4, 1): the same loan covers both.
    assert "rank 1: 5 forwards, 2 nonzero units (5 unit-reads)" in text
    small = dict(doc, unit_bytes=300 << 20)
    text = "\n".join(ep_hot_units.coverage_lines(small))
    assert "loan  512 MiB: top 1 units = 300.0 MiB cover 66.7%" in text


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
