#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""scripts/post_land_smoke.py: parser, verdict, pins, plan gating, and an
end-to-end `run` against a fake daemon (CPU only)."""
import copy
import errno
import hashlib
import json
import os
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import post_land_smoke as psm  # noqa: E402

PROMPT = "Explain why the sky is blue in three short sentences."

# Event shapes as emitted by crates/hipfire-daemon/src/main.rs (loaded, error),
# hipfire-generate/src/common.rs emit_committed_event / emit_stream_event, and
# the staged two-phase terminal (commit_ready -> done) in hipfire-engine terminal.rs.
LOADED = '{"type":"loaded","arch":"qwen3_5","dim":4096,"layers":32,"vocab":248320,"vl":true}'
STDOUT_TWO_REQUESTS = [
    LOADED,
    '{"type":"token","id":"A","text":"The","attempt_id":1}',
    '{"type":"committed","id":"A","tok_id":760,"pos":12,"t_ms":30,"attempt_id":1}',
    '{"type":"token","id":"B","text":"Sky","attempt_id":2}',
    '{"type":"committed","id":"B","tok_id":9001,"pos":12,"t_ms":31,"attempt_id":2}',
    "this is not json at all",
    '{"type":"committed","id":"A","tok_id":2,"pos":13,"t_ms":50,"attempt_id":1}',
    '{"type":"token","id":"A","text":" sky","attempt_id":1}',
    '{"type":"commit_ready","id":"A","tokens":2,"tok_s":44.5,"finish_reason":"stop","attempt_id":1}',
    '{"type":"done","id":"A","tokens":2,"tok_s":44.5,"prefill_tokens":9,"decode_tok_s":44.5,'
    '"ttft_ms":12.0,"finish_reason":"stop","attempt_id":1}',
    '{"type":"error","id":"B","message":"boom","class":"internal","retryable":false,'
    '"rolled_back":false,"attempt_id":2}',
    '{"type":"error","message":"bad model","class":"validation","retryable":false,'
    '"rolled_back":false,"attempt_id":0}',
    '{"type":"unloaded"}',
    "[1, 2, 3]",
]


# --------------------------------------------------------------------------
# parser / helpers
# --------------------------------------------------------------------------


def test_parse_real_event_shapes():
    p = psm.parse_daemon_stdout(STDOUT_TWO_REQUESTS)
    assert p["loaded"]["arch"] == "qwen3_5" and p["loaded"]["vocab"] == 248320
    assert p["unloaded"] is True
    a, b = p["requests"]["A"], p["requests"]["B"]
    assert a["ids"] == [760, 2]  # interleaved with B and garbage, order kept
    assert a["positions"] == [12, 13]
    assert a["text"] == "The sky"
    assert a["finish"] == "stop" and a["done"]["tokens"] == 2 and a["done"]["tok_s"] == 44.5
    assert a["commit_ready"] is True and a["errors"] == []
    assert b["ids"] == [9001] and b["text"] == "Sky" and b["done"] is None
    assert b["errors"][0]["message"] == "boom" and b["errors"][0]["class"] == "internal"
    assert [e["message"] for e in p["errors"]] == ["bad model"]  # id-less error is global
    # non-JSON and non-object JSON are recorded, not fatal
    assert [g["line"] for g in p["garbage"]] == ["this is not json at all", "[1, 2, 3]"]
    assert p["garbage"][0]["line_no"] == 6


def test_parse_accepts_bytes_blank_lines_and_unknown_types():
    p = psm.parse_daemon_stdout([b'{"type":"info","message":"x"}\n', b"\n", b'{"type":"reasoning","id":"A","text":"hm"}\n'])
    assert p["other_types"] == {"info": 1}
    assert p["requests"]["A"]["reasoning"] == "hm" and p["requests"]["A"]["text"] == ""


def test_parse_committed_without_integer_tok_id_is_a_request_error():
    p = psm.parse_daemon_stdout(['{"type":"committed","id":"A","tok_id":"7"}'])
    assert p["requests"]["A"]["ids"] == []
    assert "tok_id" in p["requests"]["A"]["errors"][0]["message"]


def test_ids_md5_format():
    ids = [760, 2, 31337]
    want = hashlib.md5(b"760,2,31337").hexdigest()
    assert psm.ids_md5(ids) == want
    assert len(want) == 32 and psm.ids_md5([]) == hashlib.md5(b"").hexdigest()


def test_build_cell_result_shape():
    req = psm.parse_daemon_stdout(STDOUT_TWO_REQUESTS)["requests"]["A"]
    c = psm.build_cell_result(req, 1.25)
    assert c["ids"] == [760, 2] and c["n"] == 2 and c["ids_md5"] == psm.ids_md5([760, 2])
    assert c["text"] == "The sky" and c["finish"] == "stop" and c["gen_s"] == 1.25
    assert c["done_event"]["type"] == "done" and c["errors"] == []
    nr = psm.build_cell_result(None, None, "not run: x")
    assert nr["n"] == 0 and nr["errors"] == [{"message": "not run: x"}]


def test_scan_stderr():
    lines = ["ok line\n", "GPU fault at 0x1\n", "hipErrorOutOfMemory in x\n", "fine"]
    hits = psm.scan_stderr(lines, [r"GPU fault", r"hipError\w+", r"never"])
    assert [(h["pattern"], h["line"]) for h in hits] == [
        ("GPU fault", "GPU fault at 0x1"),
        (r"hipError\w+", "hipErrorOutOfMemory in x"),
    ]
    assert psm.scan_stderr(lines, []) == []
    capped = psm.scan_stderr(["boom"] * 100, ["boom"])
    assert len(capped) == psm.MAX_STDERR_HITS_PER_PATTERN


# --------------------------------------------------------------------------
# verdict
# --------------------------------------------------------------------------

IDS = [10, 20, 30, 40, 50, 60, 70, 80]


def cell_rec(ids=IDS, finish="length", done=True, errors=None, gen_s=0.5):
    return {
        "ids": list(ids),
        "n": len(ids),
        "ids_md5": psm.ids_md5(ids),
        "text": "t",
        "finish": finish if done else None,
        "done_event": {"type": "done", "finish_reason": finish} if done else None,
        "gen_s": gen_s,
        "errors": errors or [],
    }


def load_res(cells, **kw):
    r = {
        "load": "L",
        "exit_code": 0,
        "killed_on_deadline": False,
        "errors": [],
        "stderr_hits": [],
        "load_s": 4.0,
        "cells": cells,
    }
    r.update(kw)
    return r


def pin_of(ids):
    return {"ids": list(ids), "ids_md5": psm.ids_md5(ids), "sha": "old"}


def test_status_match_new_changed():
    ok = load_res({"c": cell_rec()})
    assert psm.cell_status(ok, ok["cells"]["c"], 8, pin_of(IDS))["status"] == "MATCH"
    assert psm.cell_status(ok, ok["cells"]["c"], 8, None)["status"] == "NEW"
    other = IDS[:3] + [999] + IDS[4:]
    st = psm.cell_status(ok, ok["cells"]["c"], 8, pin_of(other))
    assert st["status"] == "CHANGED"
    d = st["first_diff"]
    assert d["index"] == 3 and d["got"] == 40 and d["pin"] == 999
    assert d["got_context"] == [10, 20, 30, 40, 50, 60, 70]
    assert d["pin_context"] == [10, 20, 30, 999, 50, 60, 70]


def test_changed_when_one_is_a_prefix_of_the_other():
    ok = load_res({"c": cell_rec(IDS)})
    st = psm.cell_status(ok, ok["cells"]["c"], 8, pin_of(IDS[:5]))
    assert st["status"] == "CHANGED" and st["first_diff"]["index"] == 5
    assert st["first_diff"]["pin"] is None and st["first_diff"]["got"] == 60


@pytest.mark.parametrize(
    "load_kw,needle",
    [
        ({"killed_on_deadline": True, "exit_code": -9}, "killed on deadline"),
        ({"exit_code": 3}, "exit code 3"),
        ({"exit_code": None}, "exit code None"),
        ({"errors": [{"message": "qwen4 EP mesh is poisoned"}]}, "mesh is poisoned"),
        ({"stderr_hits": [{"pattern": "GPU fault", "line": "GPU fault!"}]}, "GPU fault"),
    ],
)
def test_status_fail_on_load_problems(load_kw, needle):
    lr = load_res({"c": cell_rec()}, **load_kw)
    st = psm.cell_status(lr, lr["cells"]["c"], 8, pin_of(IDS))
    assert st["status"] == "FAIL" and any(needle in r for r in st["reasons"])


def test_status_fail_on_missing_load_cell_and_cell_error():
    assert psm.cell_status(None, None, 8, None)["status"] == "FAIL"
    lr = load_res({})
    assert psm.cell_status(lr, None, 8, None)["status"] == "FAIL"
    lr = load_res({"c": cell_rec(errors=[{"message": "sealed_moe: live gate/up live buffer identity differs"}])})
    st = psm.cell_status(lr, lr["cells"]["c"], 8, pin_of(IDS))
    assert st["status"] == "FAIL" and "sealed_moe" in st["reasons"][0]


def test_status_fail_on_zero_tokens_no_done_and_aborted():
    lr = load_res({"c": cell_rec(ids=[], finish="stop")})
    assert psm.cell_status(lr, lr["cells"]["c"], 8, None)["status"] == "FAIL"
    lr = load_res({"c": cell_rec(done=False)})
    st = psm.cell_status(lr, lr["cells"]["c"], 8, None)
    assert st["status"] == "FAIL" and "no done event" in st["reasons"]
    lr = load_res({"c": cell_rec(finish="aborted")})
    assert psm.cell_status(lr, lr["cells"]["c"], 8, None)["status"] == "FAIL"


def test_status_short_without_natural_stop_fails_but_eos_accepted():
    short = IDS[:5]
    for finish in ("length", "tool_calls", None):
        lr = load_res({"c": cell_rec(ids=short, finish=finish or "")})
        lr["cells"]["c"]["finish"] = finish
        st = psm.cell_status(lr, lr["cells"]["c"], 8, None)
        assert st["status"] == "FAIL" and any("short" in r for r in st["reasons"]), finish
    for finish in ("stop", "eos"):
        lr = load_res({"c": cell_rec(ids=short, finish=finish)})
        assert psm.cell_status(lr, lr["cells"]["c"], 8, pin_of(short))["status"] == "MATCH"


def test_overall_precedence_and_exit_codes():
    assert psm.overall(["MATCH", "NEW", "SKIPPED"], []) == "PASS"
    assert psm.overall(["MATCH", "CHANGED"], []) == "CHANGED"
    assert psm.overall(["MATCH", "CHANGED", "FAIL"], []) == "FAIL"
    assert psm.overall(["MATCH", "MATCH"], [{"status": "INVARIANT"}]) == "FAIL"
    assert psm.overall([], []) == "PASS"
    assert psm.EXIT_FOR_OVERALL == {"PASS": 0, "FAIL": 1, "CHANGED": 3, "INCOMPLETE": 4} and psm.EXIT_USAGE == 2


def mk_plan(**over):
    plan = {
        "version": 1,
        "prompt": PROMPT,
        "max_tokens": 8,
        "stderr_fail_patterns": ["GPU fault"],
        "loads": [
            {
                "name": "fn1",
                "model": "qwen3.8:flash-next",
                "load_params": {"max_seq": 512},
                "env": {},
                "deadline_s": 30,
                "note": "single card",
                "cells": [
                    {"name": "fn1-ar", "generate": {"fake_seed": "a"}},
                    {"name": "fn1-mtp", "generate": {"fake_seed": "a"}},
                ],
            },
            {
                "name": "a3b",
                "model": "qwen3.6:35b-a3b",
                "deadline_s": 30,
                "cells": [{"name": "a3b-ar", "generate": {"fake_seed": "b"}}],
            },
        ],
        "invariants": [["fn1-ar", "fn1-mtp"]],
    }
    plan.update(over)
    return psm.validate_plan(plan)


def results_for(plan, **per_cell):
    return {
        "fn1": load_res({"fn1-ar": per_cell.get("fn1-ar", cell_rec()), "fn1-mtp": per_cell.get("fn1-mtp", cell_rec())}),
        "a3b": load_res({"a3b-ar": per_cell.get("a3b-ar", cell_rec([1, 2, 3, 4, 5, 6, 7, 8]))}),
    }


def test_verdict_pass_with_all_new_and_missing_pins():
    plan = mk_plan()
    v = psm.build_verdict(plan, results_for(plan), {}, "abc123", "/r", "psha")
    assert v["overall"] == "PASS" and v["sha"] == "abc123" and v["plan_sha256"] == "psha"
    assert {c["status"] for c in v["cells"].values()} == {"NEW"}
    assert v["invariants"] == [{"cells": ["fn1-ar", "fn1-mtp"], "status": "OK", "detail": "ids identical", "ids_md5": psm.ids_md5(IDS)}]
    assert v["results_dirs"] == ["/r"] and v["cells"]["fn1-ar"]["note"] == "single card"
    assert "note" not in v["cells"]["a3b-ar"]
    assert v["cells"]["fn1-ar"]["ids"] == IDS and v["cells"]["fn1-ar"]["load_s"] == 4.0


def test_verdict_changed_and_fail_precedence():
    plan = mk_plan()
    pins = {"cells": {"fn1-ar": pin_of(IDS), "fn1-mtp": pin_of(IDS), "a3b-ar": pin_of([9] * 8)}}
    v = psm.build_verdict(plan, results_for(plan), pins, "s")
    assert v["overall"] == "CHANGED"
    assert [c for c, e in v["cells"].items() if e["status"] == "CHANGED"] == ["a3b-ar"]
    res = results_for(plan)
    res["fn1"]["killed_on_deadline"] = True
    v = psm.build_verdict(plan, res, pins, "s")
    assert v["overall"] == "FAIL" and v["cells"]["fn1-ar"]["status"] == "FAIL"


def test_verdict_fail_on_kill_marks_every_cell_of_that_load():
    plan = mk_plan()
    res = results_for(plan)
    res["fn1"].update(killed_on_deadline=True, exit_code=-15)
    v = psm.build_verdict(plan, res, {}, "s")
    assert v["cells"]["fn1-ar"]["status"] == v["cells"]["fn1-mtp"]["status"] == "FAIL"
    assert v["cells"]["a3b-ar"]["status"] == "NEW"
    assert v["invariants"][0]["status"] == "UNCHECKED"  # FAIL side: not double-reported


def test_verdict_missing_load_result_is_fail():
    plan = mk_plan()
    res = results_for(plan)
    del res["a3b"]
    v = psm.build_verdict(plan, res, {}, "s")
    assert v["cells"]["a3b-ar"]["status"] == "FAIL" and v["overall"] == "FAIL"


def test_verdict_invariant_failure_is_overall_fail():
    plan = mk_plan()
    res = results_for(plan, **{"fn1-mtp": cell_rec(IDS[:4] + [77] + IDS[5:])})
    v = psm.build_verdict(plan, res, {}, "s")
    assert {c["status"] for c in v["cells"].values()} == {"NEW"}  # both cells fine on their own
    inv = v["invariants"][0]
    assert inv["status"] == "INVARIANT" and inv["first_diff"]["index"] == 4
    assert v["overall"] == "FAIL"
    text = psm.format_table(v)
    assert "INVARIANT fn1-ar != fn1-mtp" in text and "overall: FAIL" in text


def test_verdict_effective_max_tokens_honours_cell_override():
    plan = mk_plan()
    plan["loads"][0]["cells"][0]["generate"]["max_tokens"] = 4
    res = results_for(plan, **{"fn1-ar": cell_rec(IDS[:4], finish="length")})
    v = psm.build_verdict(plan, res, {}, "s")
    assert v["cells"]["fn1-ar"]["status"] == "NEW"  # 4 == its own max_tokens


def test_verdict_table_columns():
    plan = mk_plan()
    v = psm.build_verdict(plan, results_for(plan), {}, "s")
    lines = psm.format_table(v).splitlines()
    assert lines[0].split() == ["cell", "status", "n", "md5", "load_s", "gen_s"]
    assert lines[1].split() == ["fn1-ar", "NEW", "8", psm.ids_md5(IDS)[:12], "4.0", "0.5"]


# --------------------------------------------------------------------------
# pins
# --------------------------------------------------------------------------


def verdict_with(**statuses):
    cells = {}
    for name, st in statuses.items():
        ids = [1, 2, 3] if st != "SKIPPED" else []
        cells[name] = {"status": st, "ids": ids, "ids_md5": psm.ids_md5(ids), "reasons": []}
    over = psm.overall((c["status"] for c in cells.values()), [])
    return {"sha": "newsha", "overall": over, "cells": cells, "invariants": []}


def test_pin_refuses_fail():
    with pytest.raises(psm.PinRefused):
        psm.apply_pins({}, verdict_with(a="MATCH", b="FAIL"), "because")


def test_pin_refuses_changed_without_reason_and_accepts_with():
    v = verdict_with(a="MATCH", b="CHANGED", c="NEW")
    for bad in (None, "", "   "):
        with pytest.raises(psm.PinRefused):
            psm.apply_pins({}, v, bad)
    old = {"version": 1, "cells": {"b": pin_of([9, 9])}, "history": []}
    snapshot = copy.deepcopy(old)
    new = psm.apply_pins(old, v, "kernel retune lands new numerics", now="T1")
    assert old == snapshot  # input untouched
    assert new["cells"]["b"]["ids"] == [1, 2, 3] and new["cells"]["b"]["reason"] == "kernel retune lands new numerics"
    assert new["cells"]["b"]["sha"] == "newsha" and new["cells"]["b"]["pinned_utc"] == "T1"
    assert new["cells"]["a"]["reason"] == "match" and new["cells"]["c"]["reason"] == "new"
    assert new["cells"]["c"]["ids_md5"] == psm.ids_md5([1, 2, 3])
    assert new["history"] == [
        {"sha": "newsha", "overall": "CHANGED", "pinned_utc": "T1", "reason": "kernel retune lands new numerics", "changed": ["b"]}
    ]


def test_pin_pass_all_new_creates_fresh_file_structure():
    new = psm.apply_pins({}, verdict_with(a="NEW", b="NEW"), now="T0")
    assert new["version"] == 1 and set(new["cells"]) == {"a", "b"}
    assert new["history"][0]["changed"] == [] and new["history"][0]["overall"] == "PASS"


def test_fail_and_skipped_cells_never_overwrite_pins():
    v = verdict_with(a="MATCH", b="FAIL", c="SKIPPED")
    v["overall"] = "PASS"  # defensive path: even if a FAIL cell slips through
    old = {"version": 1, "cells": {"b": pin_of([5]), "c": pin_of([6])}, "history": []}
    new = psm.apply_pins(old, v)
    assert new["cells"]["b"]["ids"] == [5] and new["cells"]["c"]["ids"] == [6]
    assert new["cells"]["a"]["ids"] == [1, 2, 3]


def test_atomic_write_json_leaves_old_file_on_failure(tmp_path):
    target = tmp_path / "pins.json"
    psm.atomic_write_json(target, {"a": 1})
    assert json.loads(target.read_text()) == {"a": 1}

    class Boom:
        pass

    with pytest.raises(TypeError):
        psm.atomic_write_json(target, {"a": Boom()})
    assert json.loads(target.read_text()) == {"a": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["pins.json"]  # no tmp litter


def test_pin_cli_roundtrip_and_exit_codes(tmp_path):
    vpath, ppath = tmp_path / "v.json", tmp_path / "pins.json"
    vpath.write_text(json.dumps(verdict_with(a="NEW", b="NEW")))
    assert psm.main(["pin", "--verdict", str(vpath), "--pins", str(ppath)]) == 0
    assert set(json.loads(ppath.read_text())["cells"]) == {"a", "b"}
    vpath.write_text(json.dumps(verdict_with(a="CHANGED", b="MATCH")))
    before = ppath.read_text()
    assert psm.main(["pin", "--verdict", str(vpath), "--pins", str(ppath)]) == 1
    assert ppath.read_text() == before
    assert psm.main(["pin", "--verdict", str(vpath), "--pins", str(ppath), "--accept-changed", "numerics"]) == 0
    assert json.loads(ppath.read_text())["history"][-1]["reason"] == "numerics"
    vpath.write_text(json.dumps(verdict_with(a="FAIL")))
    assert psm.main(["pin", "--verdict", str(vpath), "--pins", str(ppath), "--accept-changed", "x"]) == 1
    assert psm.main(["pin", "--verdict", str(tmp_path / "nope.json"), "--pins", str(ppath)]) == 2


# --------------------------------------------------------------------------
# plan validation + load gating
# --------------------------------------------------------------------------


def base_raw():
    return {
        "version": 1,
        "prompt": PROMPT,
        "loads": [{"name": "L1", "model": "m", "deadline_s": 10, "cells": [{"name": "c1"}, {"name": "c2"}]}],
    }


def errs(raw):
    with pytest.raises(psm.PlanError) as ei:
        psm.validate_plan(raw)
    return "\n".join(ei.value.errors)


def test_validate_plan_defaults():
    p = psm.validate_plan(base_raw())
    assert p["max_tokens"] == 32 and p["stderr_fail_patterns"] == [] and p["invariants"] == []
    ld = p["loads"][0]
    assert ld["load_params"] == {} and ld["env"] == {} and ld["enabled"] is True
    assert ld["run_if_paths"] == [] and ld["note"] == "" and ld["cells"][0]["generate"] == {}


def test_validate_plan_errors():
    r = base_raw(); r["version"] = 2
    assert "version" in errs(r)
    r = base_raw(); r["prompt"] = ""
    assert "prompt" in errs(r)
    r = base_raw(); r["max_tokens"] = 0
    assert "max_tokens" in errs(r)
    r = base_raw(); r["loads"] = []
    assert "loads" in errs(r)
    r = base_raw(); r["loads"].append(copy.deepcopy(r["loads"][0]))
    e = errs(r)
    assert "duplicate load name" in e and "duplicate cell name" in e
    r = base_raw(); r["loads"][0]["cells"].append({"name": "c1"})
    assert "duplicate cell name" in errs(r)
    r = base_raw(); del r["loads"][0]["model"]
    assert "model" in errs(r)
    r = base_raw(); del r["loads"][0]["deadline_s"]
    assert "deadline_s" in errs(r)
    r = base_raw(); r["loads"][0]["name"] = "bad/name"
    assert "name" in errs(r)
    r = base_raw(); r["loads"][0]["cells"] = []
    assert "cells" in errs(r)
    r = base_raw(); r["loads"][0]["cells"][0]["generate"] = {"attempt_id": 3}
    assert "set by the runner" in errs(r)
    r = base_raw(); r["invariants"] = [["c1", "nope"]]
    assert "unknown cell 'nope'" in errs(r)
    r = base_raw(); r["invariants"] = [["c1"]]
    assert "pair" in errs(r)
    r = base_raw(); r["stderr_fail_patterns"] = ["(unclosed"]
    assert "bad regex" in errs(r)
    r = base_raw(); r["loads"][0]["run_if_paths"] = ["["]
    assert "bad regex" in errs(r)
    r = base_raw(); r["loads"][0]["name"] = "c1"
    assert "both a load and a cell" in errs(r)
    r = base_raw(); r["loads"][0]["env"] = {"A": [1]}
    assert "env" in errs(r)
    r = base_raw(); r["loads"][0]["enabled"] = "yes"
    assert "enabled" in errs(r)
    assert "JSON object" in errs([1])


def test_validate_plan_reports_all_errors_at_once():
    r = base_raw(); r["prompt"] = ""; r["version"] = 9; r["invariants"] = [["x", "y"]]
    assert len(psm.PlanError([]).errors) == 0
    with pytest.raises(psm.PlanError) as ei:
        psm.validate_plan(r)
    assert len(ei.value.errors) >= 4


def test_env_values_are_stringified():
    r = base_raw(); r["loads"][0]["env"] = {"HIPFIRE_TOPOLOGY_PP": 2}
    assert psm.validate_plan(r)["loads"][0]["env"] == {"HIPFIRE_TOPOLOGY_PP": "2"}


def gated(**kw):
    ld = {"name": "L", "model": "m", "deadline_s": 5, "cells": [{"name": "c"}]}
    ld.update(kw)
    return psm.validate_plan({"version": 1, "prompt": "p", "loads": [ld]})["loads"][0]


def test_load_selection_enabled_default_and_disabled():
    assert psm.load_selection(gated(), None) == (True, "enabled")
    assert psm.load_selection(gated(), []) == (True, "enabled")
    ok, why = psm.load_selection(gated(enabled=False), ["crates/x.rs"])
    assert ok is False and "disabled" in why


def test_load_selection_path_filter_match_and_nomatch():
    ld = gated(enabled=False, run_if_paths=[r"^crates/hipfire-arch-qwen35/", r"moe"])
    ok, why = psm.load_selection(ld, ["README.md", "crates/hipfire-arch-qwen35/src/a.rs"])
    assert ok is True and "crates/hipfire-arch-qwen35/src/a.rs" in why
    ok, why = psm.load_selection(ld, ["src/sealed_moe.rs"])  # re.search, not match
    assert ok is True
    ok, why = psm.load_selection(ld, ["README.md", "docs/a.md"])
    assert ok is False and "no changed path" in why


def test_load_selection_missing_changed_paths_skips_filtered_loads():
    ld = gated(enabled=False, run_if_paths=["moe"])
    ok, why = psm.load_selection(ld, None)
    assert ok is False and "--changed-paths not supplied" in why
    assert psm.load_selection(ld, [])[0] is False  # empty list given: nothing matches
    assert psm.load_selection(gated(enabled=True, run_if_paths=["moe"]), None)[0] is True


def test_read_changed_paths_ignores_blank_lines(tmp_path):
    f = tmp_path / "cp.txt"
    f.write_text("a/b.rs\n\n  c.md  \n")
    assert psm.read_changed_paths(f) == ["a/b.rs", "c.md"]


def test_select_cells_only():
    plan = mk_plan()
    assert set(psm.select_cells(plan, None)) == {"fn1", "a3b"}
    sel = psm.select_cells(plan, ["a3b", "fn1-mtp"])
    assert [c["name"] for c in sel["fn1"]] == ["fn1-mtp"] and [c["name"] for c in sel["a3b"]] == ["a3b-ar"]
    with pytest.raises(psm.PlanError):
        psm.select_cells(plan, ["nope"])


# --------------------------------------------------------------------------
# fake daemon + end-to-end run
# --------------------------------------------------------------------------

FAKE_DAEMON = r'''
import json, os, signal, subprocess, sys, time

mode = os.environ.get("FAKE_MODE", "ok")
if mode == "ignore_term":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def tok_ids(seed, n):
    base = sum(map(ord, seed))
    return [1000 + (base * 7 + i * 13) % 5000 for i in range(n)]


def log_request(msg):
    path = os.environ.get("FAKE_REQLOG")
    if path:
        with open(path, "a") as f:
            f.write(json.dumps(msg) + "\n")


if os.environ.get("FAKE_STDERR"):
    sys.stderr.write(os.environ["FAKE_STDERR"] + "\n")
    sys.stderr.flush()
if os.environ.get("FAKE_EXPECT_EMIT") and os.environ.get("HIPFIRE_EMIT_TOKEN_IDS") != "1":
    sys.exit(9)
if os.environ.get("FAKE_MARKER"):
    with open(os.environ["FAKE_MARKER"], "a") as f:
        f.write("started\n")
if os.environ.get("FAKE_ENVLOG"):
    with open(os.environ["FAKE_ENVLOG"], "a") as f:
        f.write(json.dumps({k: os.environ.get(k) for k in ("HOME", "HIPFIRE_NO_REGISTRY_FETCH", "HIPFIRE_EMIT_TOKEN_IDS")}) + "\n")

while True:
    line = sys.stdin.readline()
    if not line:
        break
    try:
        msg = json.loads(line)
    except ValueError:
        continue
    log_request(msg)
    t = msg.get("type")
    if t == "load":
        if mode == "load_error":
            emit({"type": "error", "message": "model not found", "class": "validation",
                  "retryable": False, "rolled_back": False, "attempt_id": 0})
            continue
        print("some garbage banner line", flush=True)
        loaded_ev = {"type": "loaded", "arch": "qwen3_5", "dim": 8, "layers": 2, "vocab": 100, "vl": False}
        loaded_ev.update(json.loads(os.environ.get("FAKE_LOADED", "{}")))
        emit(loaded_ev)
        if os.environ.get("FAKE_READY"):
            with open(os.environ["FAKE_READY"], "w") as f:
                f.write(str(os.getpid()))
        if mode in ("hang", "ignore_term"):
            gc = os.environ.get("FAKE_GRANDCHILD")
            if gc:
                p = subprocess.Popen(["sleep", "1000"])
                with open(gc, "w") as f:
                    f.write(str(p.pid))
            time.sleep(3600)
    elif t == "generate":
        aid = msg.get("attempt_id")
        cid = msg.get("id")
        if not isinstance(aid, int) or isinstance(aid, bool) or aid <= 0:
            emit({"type": "error", "id": cid, "message": "generate missing attempt_id", "class": "validation",
                  "retryable": False, "rolled_back": False, "attempt_id": 0})
            continue
        if mode == "exit3":
            sys.exit(3)
        if msg.get("fake_error"):
            emit({"type": "error", "id": cid, "message": msg["fake_error"], "class": "internal",
                  "retryable": False, "rolled_back": False, "attempt_id": aid})
            continue
        mt = msg["max_tokens"]
        n = int(msg.get("fake_tokens", mt))
        finish = msg.get("fake_finish") or ("length" if n >= mt else "stop")
        ids = tok_ids(msg.get("fake_seed", "s"), n)
        if "fake_flip" in msg:
            ids[int(msg["fake_flip"])] += 1
        for i, tid in enumerate(ids):
            emit({"type": "token", "id": cid, "text": f"w{i} ", "attempt_id": aid})
            emit({"type": "committed", "id": cid, "tok_id": tid, "pos": 10 + i, "t_ms": i, "attempt_id": aid})
        done = {"type": "done", "id": cid, "tokens": n, "tok_s": 12.5, "finish_reason": finish, "attempt_id": aid}
        emit(dict(done, type="commit_ready"))
        while True:  # two-phase terminal: wait for the matching commit
            c = sys.stdin.readline()
            if not c:
                sys.exit(4)
            cm = json.loads(c)
            log_request(cm)
            if cm.get("type") == "commit" and cm.get("id") == cid and cm.get("attempt_id") == aid:
                break
        emit(done)
    elif t == "unload":
        if mode == "orphan_grandchild":  # the leader exits normally, a grandchild stays alive in its group
            gc = os.environ.get("FAKE_GRANDCHILD")
            if gc:
                p = subprocess.Popen(["sleep", "1000"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
                with open(gc, "w") as f:
                    f.write(str(p.pid))
        emit({"type": "unloaded"})
        sys.exit(0)
'''


@pytest.fixture
def fake_daemon(tmp_path):
    p = tmp_path / "fake-daemon"
    p.write_text(f"#!{sys.executable}\n" + FAKE_DAEMON)
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return p


def write_plan(tmp_path, plan_dict, name="plan.json"):
    p = tmp_path / name
    p.write_text(json.dumps(plan_dict))
    return p


def e2e_plan(env=None, deadline=20, **over):
    plan = {
        "version": 1,
        "prompt": PROMPT,
        "max_tokens": 8,
        "stderr_fail_patterns": ["GPU fault"],
        "loads": [
            {
                "name": "L1",
                "model": "/nonexistent/model.hfq",
                "load_params": {"max_seq": 256},
                "env": env or {},
                "deadline_s": deadline,
                "note": "fake load",
                "cells": [
                    {"name": "ar", "generate": {"fake_seed": "x"}},
                    {"name": "mtp", "generate": {"fake_seed": "x", "mtp": True}},
                ],
            }
        ],
        "invariants": [["ar", "mtp"]],
    }
    plan.update(over)
    return plan


def run_cli(tmp_path, plan_path, daemon, out, *extra):
    return psm.main(["run", "--plan", str(plan_path), "--daemon", str(daemon), "--out", str(out), *extra])


def read(p):
    return json.loads(Path(p).read_text())


def test_e2e_run_ok_writes_logs_and_results(tmp_path, fake_daemon):
    reqlog = tmp_path / "req.log"
    plan = write_plan(tmp_path, e2e_plan(env={"FAKE_REQLOG": str(reqlog), "FAKE_STDERR": "daemon says hi", "FAKE_EXPECT_EMIT": "1"}))
    out = tmp_path / "out"
    assert run_cli(tmp_path, plan, fake_daemon, out) == 0

    res = read(out / "L1.json")
    assert res["load"] == "L1" and res["exit_code"] == 0 and res["killed_on_deadline"] is False
    assert res["env"] == {"FAKE_REQLOG": str(reqlog), "FAKE_STDERR": "daemon says hi", "FAKE_EXPECT_EMIT": "1"}  # overlay only
    assert res["load_params"] == {"max_seq": 256} and res["note"] == "fake load"
    assert res["daemon_path"] == str(fake_daemon)
    assert res["daemon_sha256"] == hashlib.sha256(fake_daemon.read_bytes()).hexdigest()
    assert res["model_bytes"] is None and res["errors"] == [] and res["stderr_hits"] == []
    assert res["load_s"] is not None and res["wall_s"] >= res["load_s"] and res["started_utc"].endswith("Z")
    assert res["garbage_lines"] == 1
    ar, mtp = res["cells"]["ar"], res["cells"]["mtp"]
    assert ar["n"] == 8 and ar["ids"] == mtp["ids"] and ar["ids_md5"] == psm.ids_md5(ar["ids"])
    assert ar["finish"] == "length" and ar["done_event"]["tok_s"] == 12.5 and ar["errors"] == []
    assert ar["text"] == "".join(f"w{i} " for i in range(8)) and ar["gen_s"] is not None

    # raw tees
    raw = (out / "L1.stdout.jsonl").read_text().splitlines()
    assert any('"type": "loaded"' in ln for ln in raw) and "some garbage banner line" in raw
    assert (out / "L1.stderr.log").read_text().strip() == "daemon says hi"

    # protocol the daemon actually saw: attempt_ids, sequential cells, commits, EMIT env
    sent = [json.loads(ln) for ln in reqlog.read_text().splitlines()]
    assert [m["type"] for m in sent] == ["load", "generate", "commit", "generate", "commit", "unload"]
    assert sent[0] == {"type": "load", "model": "/nonexistent/model.hfq", "params": {"max_seq": 256}}
    g1, g2 = sent[1], sent[3]
    assert (g1["id"], g1["attempt_id"], g2["id"], g2["attempt_id"]) == ("ar", 1, "mtp", 2)
    assert g1["prompt"] == PROMPT and g1["temperature"] == 0.0 and g1["max_tokens"] == 8 and g2["mtp"] is True
    assert sent[2] == {"type": "commit", "id": "ar", "attempt_id": 1}

    run = read(out / "run.json")
    assert run["ok"] is True and run["plan_sha256"] == hashlib.sha256(plan.read_bytes()).hexdigest()
    assert run["selected"] == {"L1": ["ar", "mtp"]} and run["loads"]["L1"]["note"] == "fake load"
    assert run["loads"]["L1"]["exit_code"] == 0 and run["daemon_sha256"] == res["daemon_sha256"]


def test_e2e_model_bytes_for_existing_file(tmp_path, fake_daemon):
    model = tmp_path / "m.hfq"
    model.write_bytes(b"x" * 1234)
    pd = e2e_plan()
    pd["loads"][0]["model"] = str(model)
    out = tmp_path / "out"
    assert run_cli(tmp_path, write_plan(tmp_path, pd), fake_daemon, out) == 0
    assert read(out / "L1.json")["model_bytes"] == 1234


def test_e2e_then_verdict_and_pin_flow(tmp_path, fake_daemon):
    plan = write_plan(tmp_path, e2e_plan())
    out, pins, vfile = tmp_path / "out", tmp_path / "pins.json", tmp_path / "VERDICT.json"
    assert run_cli(tmp_path, plan, fake_daemon, out) == 0
    vargs = ["verdict", "--results", str(out), "--plan", str(plan), "--pins", str(pins), "--sha", "land1", "--out", str(vfile)]
    assert psm.main(vargs) == 0  # missing pins = all NEW
    v = read(vfile)
    assert v["overall"] == "PASS" and v["sha"] == "land1" and {c["status"] for c in v["cells"].values()} == {"NEW"}
    assert v["plan_sha256"] == hashlib.sha256(plan.read_bytes()).hexdigest() and v["results_dirs"] == [str(out)]
    assert v["invariants"][0]["status"] == "OK"
    assert psm.main(["pin", "--verdict", str(vfile), "--pins", str(pins)]) == 0
    assert psm.main(vargs) == 0
    assert {c["status"] for c in read(vfile)["cells"].values()} == {"MATCH"}

    # same plan, one token id flipped in the AR cell -> CHANGED (exit 3), not FAIL
    changed = e2e_plan()
    changed["loads"][0]["cells"][0]["generate"]["fake_flip"] = 3
    changed["loads"][0]["cells"][1]["generate"]["fake_flip"] = 3  # keep the invariant pair identical
    plan2 = write_plan(tmp_path, changed, "plan2.json")
    out2 = tmp_path / "out2"
    assert run_cli(tmp_path, plan2, fake_daemon, out2) == 0
    vargs2 = ["verdict", "--results", str(out2), "--plan", str(plan2), "--pins", str(pins), "--sha", "land2", "--out", str(vfile)]
    assert psm.main(vargs2) == 3
    assert read(vfile)["cells"]["ar"]["first_diff"]["index"] == 3
    assert psm.main(["pin", "--verdict", str(vfile), "--pins", str(pins)]) == 1
    assert psm.main(["pin", "--verdict", str(vfile), "--pins", str(pins), "--accept-changed", "flip"]) == 0
    assert psm.main(vargs2) == 0


def test_e2e_invariant_failure_gives_fail_exit_1(tmp_path, fake_daemon):
    pd = e2e_plan()
    pd["loads"][0]["cells"][1]["generate"]["fake_flip"] = 5  # mtp diverges from ar
    plan = write_plan(tmp_path, pd)
    out = tmp_path / "out"
    assert run_cli(tmp_path, plan, fake_daemon, out) == 0  # run never compares
    rc = psm.main(["verdict", "--results", str(out), "--plan", str(plan), "--pins", str(tmp_path / "none.json"),
                   "--sha", "s", "--out", str(tmp_path / "v.json")])
    assert rc == 1 and read(tmp_path / "v.json")["invariants"][0]["status"] == "INVARIANT"


def test_e2e_natural_stop_short_cell_is_accepted_and_short_length_fails(tmp_path, fake_daemon):
    pd = e2e_plan()
    pd["invariants"] = []
    pd["loads"][0]["cells"] = [
        {"name": "eos", "generate": {"fake_seed": "x", "fake_tokens": 5}},  # finish=stop
        {"name": "shortlen", "generate": {"fake_seed": "x", "fake_tokens": 5, "fake_finish": "length"}},
    ]
    plan = write_plan(tmp_path, pd)
    out = tmp_path / "out"
    assert run_cli(tmp_path, plan, fake_daemon, out) == 0
    rc = psm.main(["verdict", "--results", str(out), "--plan", str(plan), "--pins", str(tmp_path / "n.json"),
                   "--sha", "s", "--out", str(tmp_path / "v.json")])
    v = read(tmp_path / "v.json")
    assert v["cells"]["eos"]["status"] == "NEW" and v["cells"]["shortlen"]["status"] == "FAIL" and rc == 1


def test_e2e_cell_error_does_not_stop_later_cells(tmp_path, fake_daemon):
    pd = e2e_plan()
    pd["invariants"] = []
    pd["loads"][0]["cells"][0]["generate"]["fake_error"] = "sealed_moe: live gate/up live buffer identity differs"
    plan = write_plan(tmp_path, pd)
    out = tmp_path / "out"
    assert run_cli(tmp_path, plan, fake_daemon, out) == 0
    res = read(out / "L1.json")
    assert res["cells"]["ar"]["errors"][0]["message"].startswith("sealed_moe")
    assert res["cells"]["ar"]["n"] == 0 and res["cells"]["mtp"]["n"] == 8
    rc = psm.main(["verdict", "--results", str(out), "--plan", str(plan), "--pins", str(tmp_path / "n.json"),
                   "--sha", "s", "--out", str(tmp_path / "v.json")])
    v = read(tmp_path / "v.json")
    assert rc == 1 and v["cells"]["ar"]["status"] == "FAIL" and v["cells"]["mtp"]["status"] == "NEW"


def test_e2e_stderr_fail_pattern_fails_load(tmp_path, fake_daemon):
    plan = write_plan(tmp_path, e2e_plan(env={"FAKE_STDERR": "amdgpu: GPU fault detected"}))
    out = tmp_path / "out"
    assert run_cli(tmp_path, plan, fake_daemon, out) == 0
    res = read(out / "L1.json")
    assert [h["pattern"] for h in res["stderr_hits"]] == ["GPU fault"]
    rc = psm.main(["verdict", "--results", str(out), "--plan", str(plan), "--pins", str(tmp_path / "n.json"),
                   "--sha", "s", "--out", str(tmp_path / "v.json")])
    assert rc == 1 and read(tmp_path / "v.json")["cells"]["ar"]["status"] == "FAIL"


def test_e2e_nonzero_exit_is_recorded_and_verdict_fails(tmp_path, fake_daemon, capsys):
    plan = write_plan(tmp_path, e2e_plan(env={"FAKE_MODE": "exit3"}))
    out = tmp_path / "out"
    assert run_cli(tmp_path, plan, fake_daemon, out) == 0  # ran to EOF, not killed
    res = read(out / "L1.json")
    assert res["exit_code"] == 3 and res["killed_on_deadline"] is False
    assert res["cells"]["ar"]["n"] == 0 and res["cells"]["mtp"]["n"] == 0
    assert any("daemon ended" in e["message"] or "no events" in e["message"] for e in res["cells"]["ar"]["errors"])
    rc = psm.main(["verdict", "--results", str(out), "--plan", str(plan), "--pins", str(tmp_path / "n.json"),
                   "--sha", "s", "--out", str(tmp_path / "v.json")])
    v = read(tmp_path / "v.json")
    assert rc == 1 and any("exit code 3" in r for r in v["cells"]["ar"]["reasons"])
    assert "FAIL" in capsys.readouterr().out


def test_e2e_load_error_skips_cells_and_unloads(tmp_path, fake_daemon):
    plan = write_plan(tmp_path, e2e_plan(env={"FAKE_MODE": "load_error"}))
    out = tmp_path / "out"
    t = time.monotonic()
    assert run_cli(tmp_path, plan, fake_daemon, out) == 0
    assert time.monotonic() - t < 15  # unload sent after the load error: no deadline wait
    res = read(out / "L1.json")
    assert res["load_s"] is None and any(e.get("message") == "model not found" for e in res["errors"])
    assert all("not run: load failed" in c["errors"][0]["message"] for c in res["cells"].values())


def test_e2e_deadline_kills_hung_daemon_and_its_group(tmp_path, fake_daemon):
    gc = tmp_path / "grandchild.pid"
    plan = write_plan(tmp_path, e2e_plan(env={"FAKE_MODE": "hang", "FAKE_GRANDCHILD": str(gc)}, deadline=2))
    out = tmp_path / "out"
    t = time.monotonic()
    assert run_cli(tmp_path, plan, fake_daemon, out) == 1
    assert time.monotonic() - t < 20
    res = read(out / "L1.json")
    assert res["killed_on_deadline"] is True and res["exit_code"] is not None and res["exit_code"] != 0
    assert res["load_s"] is not None  # loaded arrived before the hang
    assert any("deadline" in e["message"] for e in res["errors"])
    assert all(c["n"] == 0 for c in res["cells"].values())
    assert read(out / "run.json")["ok"] is False and read(out / "run.json")["loads"]["L1"]["killed_on_deadline"] is True
    pid = int(gc.read_text())
    for _ in range(50):  # group kill must take the grandchild too
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail("grandchild survived the process-group kill")
    rc = psm.main(["verdict", "--results", str(out), "--plan", str(plan), "--pins", str(tmp_path / "n.json"),
                   "--sha", "s", "--out", str(tmp_path / "v.json")])
    assert rc == 1 and read(tmp_path / "v.json")["cells"]["ar"]["status"] == "FAIL"


def test_e2e_deadline_escalates_to_sigkill_when_sigterm_ignored(tmp_path, fake_daemon, monkeypatch):
    monkeypatch.setattr(psm, "TERM_GRACE_S", 0.3)
    plan = write_plan(tmp_path, e2e_plan(env={"FAKE_MODE": "ignore_term"}, deadline=1))
    out = tmp_path / "out"
    assert run_cli(tmp_path, plan, fake_daemon, out) == 1
    res = read(out / "L1.json")
    assert res["killed_on_deadline"] is True and res["exit_code"] == -9


def test_e2e_only_restricts_loads_and_cells(tmp_path, fake_daemon):
    pd = e2e_plan()
    pd["loads"].append({"name": "L2", "model": "m2", "deadline_s": 20, "cells": [{"name": "other", "generate": {"fake_seed": "y"}}]})
    plan = write_plan(tmp_path, pd)
    out = tmp_path / "out"
    assert run_cli(tmp_path, plan, fake_daemon, out, "--only", "mtp") == 0
    assert (out / "L1.json").exists() and not (out / "L2.json").exists()
    assert list(read(out / "L1.json")["cells"]) == ["mtp"]
    assert read(out / "run.json")["selected"] == {"L1": ["mtp"]}
    # verdict honours the subset: unselected cells and the invariant are not judged
    rc = psm.main(["verdict", "--results", str(out), "--plan", str(plan), "--pins", str(tmp_path / "n.json"),
                   "--sha", "s", "--out", str(tmp_path / "v.json"), "--partial"])
    v = read(tmp_path / "v.json")
    assert rc == 0 and list(v["cells"]) == ["mtp"] and v["invariants"][0]["status"] == "SKIPPED"
    out2 = tmp_path / "out2"
    assert run_cli(tmp_path, plan, fake_daemon, out2, "--only", "L2") == 0
    assert (out2 / "L2.json").exists() and not (out2 / "L1.json").exists()
    assert run_cli(tmp_path, plan, fake_daemon, tmp_path / "out3", "--only", "bogus") == 2


def test_run_usage_errors_exit_2(tmp_path, fake_daemon, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"version": 1}))
    assert run_cli(tmp_path, bad, fake_daemon, tmp_path / "o") == 2
    assert "plan error" in capsys.readouterr().err
    notjson = tmp_path / "nj.json"
    notjson.write_text("{nope")
    assert run_cli(tmp_path, notjson, fake_daemon, tmp_path / "o") == 2
    good = write_plan(tmp_path, e2e_plan())
    assert run_cli(tmp_path, good, tmp_path / "no-such-daemon", tmp_path / "o") == 2
    assert run_cli(tmp_path, tmp_path / "missing.json", fake_daemon, tmp_path / "o") == 2


# --------------------------------------------------------------------------
# enabled / run_if_paths / SKIPPED
# --------------------------------------------------------------------------


def gated_plan(tmp_path):
    pd = e2e_plan()
    pd["invariants"] = []
    pd["loads"][0]["cells"] = [{"name": "ar", "generate": {"fake_seed": "x"}}]
    pd["loads"].append(
        {
            "name": "xtx",
            "model": "m",
            "deadline_s": 20,
            "enabled": False,
            "run_if_paths": ["^crates/rdna-compute/", "d2_decode"],
            "note": "D2 armed lane",
            "cells": [{"name": "xtx-d2", "generate": {"fake_seed": "z"}}],
        }
    )
    pd["loads"].append(
        {"name": "off", "model": "m", "deadline_s": 20, "enabled": False,
         "cells": [{"name": "off-ar", "generate": {"fake_seed": "q"}}]}
    )
    pd["invariants"] = [["ar", "xtx-d2"]]
    return write_plan(tmp_path, pd)


def test_run_skips_disabled_loads_and_filter_without_changed_paths(tmp_path, fake_daemon):
    plan = gated_plan(tmp_path)
    out = tmp_path / "out"
    assert run_cli(tmp_path, plan, fake_daemon, out) == 0
    assert (out / "L1.json").exists()
    sk = read(out / "xtx.json")
    assert sk["skipped"] is True and "--changed-paths not supplied" in sk["reason"] and sk["note"] == "D2 armed lane"
    assert read(out / "off.json")["skipped"] is True
    assert not (out / "xtx.stdout.jsonl").exists()  # no daemon was spawned
    run = read(out / "run.json")
    assert run["ok"] is True and run["loads"]["xtx"] == {"skipped": True, "reason": sk["reason"], "note": "D2 armed lane", "matched": []}
    assert run["changed_paths_given"] is False


def test_run_path_filter_match_runs_and_nomatch_skips(tmp_path, fake_daemon):
    plan = gated_plan(tmp_path)
    cp = tmp_path / "changed.txt"
    cp.write_text("docs/x.md\ncrates/rdna-compute/src/d2.rs\n")
    out = tmp_path / "out"
    assert run_cli(tmp_path, plan, fake_daemon, out, "--changed-paths", str(cp)) == 0
    assert read(out / "xtx.json").get("skipped") is None and read(out / "xtx.json")["cells"]["xtx-d2"]["n"] == 8
    assert read(out / "off.json")["skipped"] is True  # no run_if_paths: still disabled
    assert read(out / "run.json")["loads"]["xtx"]["skipped"] is False
    cp.write_text("docs/x.md\n")
    out2 = tmp_path / "out2"
    assert run_cli(tmp_path, plan, fake_daemon, out2, "--changed-paths", str(cp)) == 0
    assert "no changed path" in read(out2 / "xtx.json")["reason"]


def test_verdict_skipped_cells_are_neutral_and_never_pinned(tmp_path, fake_daemon):
    plan = gated_plan(tmp_path)
    out, pins, vfile = tmp_path / "out", tmp_path / "pins.json", tmp_path / "v.json"
    assert run_cli(tmp_path, plan, fake_daemon, out) == 0
    vargs = ["verdict", "--results", str(out), "--plan", str(plan), "--pins", str(pins), "--sha", "s", "--out", str(vfile)]
    assert psm.main(vargs) == 0
    v = read(vfile)
    assert v["overall"] == "PASS"
    assert v["cells"]["xtx-d2"]["status"] == "SKIPPED" and v["cells"]["off-ar"]["status"] == "SKIPPED"
    assert v["cells"]["ar"]["status"] == "NEW"
    assert v["cells"]["xtx-d2"]["note"] == "D2 armed lane"  # note carried into the verdict
    assert v["invariants"][0]["status"] == "SKIPPED"  # SKIPPED side: invariant not evaluated
    assert any("SKIPPED" in ln for ln in psm.format_table(v).splitlines())
    assert psm.main(["pin", "--verdict", str(vfile), "--pins", str(pins)]) == 0
    assert set(read(pins)["cells"]) == {"ar"}  # SKIPPED cells never write a pin
    # pre-existing pins for a skipped cell survive a later pin
    seeded = read(pins)
    seeded["cells"]["xtx-d2"] = pin_of([4, 5, 6])
    pins.write_text(json.dumps(seeded))
    assert psm.main(vargs) == 0 and read(vfile)["cells"]["xtx-d2"]["status"] == "SKIPPED"
    assert psm.main(["pin", "--verdict", str(vfile), "--pins", str(pins)]) == 0
    assert read(pins)["cells"]["xtx-d2"]["ids"] == [4, 5, 6]


def test_skipped_does_not_mask_fail_elsewhere():
    plan = mk_plan()
    res = {
        "fn1": {"load": "fn1", "skipped": True, "reason": "disabled", "note": ""},
        "a3b": load_res({"a3b-ar": cell_rec()}, killed_on_deadline=True),
    }
    v = psm.build_verdict(plan, res, {}, "s")
    assert v["cells"]["fn1-ar"]["status"] == "SKIPPED" and v["overall"] == "FAIL"
    res["a3b"]["killed_on_deadline"] = False
    assert psm.build_verdict(plan, res, {}, "s")["overall"] == "PASS"


def test_plan_subcommand_dry_run(tmp_path, capsys):
    plan = gated_plan(tmp_path)
    assert psm.main(["plan", "--plan", str(plan)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("RUN ") and "L1" in out[0] and "enabled" in out[0]
    assert out[1].startswith("SKIP") and "xtx" in out[1] and "--changed-paths not supplied" in out[1] and "D2 armed lane" in out[1]
    assert out[2].startswith("SKIP") and "off" in out[2]
    cp = tmp_path / "cp.txt"
    cp.write_text("crates/rdna-compute/src/a.rs\n")
    assert psm.main(["plan", "--plan", str(plan), "--changed-paths", str(cp)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[1].startswith("RUN ") and "crates/rdna-compute/src/a.rs" in out[1] and out[2].startswith("SKIP")
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    assert psm.main(["plan", "--plan", str(bad)]) == 2


# --------------------------------------------------------------------------
# CLI surface
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sub", [[], ["plan"], ["run"], ["verdict"], ["pin"]])
def test_help_works(sub):
    r = subprocess.run([sys.executable, str(SCRIPTS / "post_land_smoke.py"), *sub, "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "usage:" in r.stdout


def test_no_subcommand_is_usage_error():
    r = subprocess.run([sys.executable, str(SCRIPTS / "post_land_smoke.py")], capture_output=True, text=True)
    assert r.returncode == 2


# --------------------------------------------------------------------------
# run_if_diff / --diff
# --------------------------------------------------------------------------

SAMPLE_DIFF = """\
diff --git a/crates/hipfire-arch-qwen35/src/ep.rs b/crates/hipfire-arch-qwen35/src/ep.rs
index 1111111..2222222 100644
--- a/crates/hipfire-arch-qwen35/src/ep.rs
+++ b/crates/hipfire-arch-qwen35/src/ep.rs
@@ -120,2 +120,3 @@ fn sealed_moe_view(buf: &Buf) -> View {
-    let v = buf.view();
+    let v = buf.view_d2();
+    // sealed
@@ -300 +301 @@ impl Foo {
-    old_line();
+    new_line();
\\ No newline at end of file
diff --git a/old/name.rs b/new/name.rs
similarity index 100%
rename from old/name.rs
rename to new/name.rs
diff --git a/docs/new.md b/docs/new.md
new file mode 100644
index 0000000..3333333
--- /dev/null
+++ b/docs/new.md
@@ -0,0 +1 @@
+hello d2 decode
diff --git a/gone/legacy.rs b/gone/legacy.rs
deleted file mode 100644
index 4444444..0000000
--- a/gone/legacy.rs
+++ /dev/null
@@ -1,2 +0,0 @@ fn legacy_fn() {
-removed_marker();
---not a file header, a removed line starting with dashes
diff --git a/assets/logo.png b/assets/logo.png
index 5555555..6666666 100644
Binary files a/assets/logo.png and b/assets/logo.png differ
diff --git a/tail.rs b/tail.rs
index 7777777..8888888 100644
--- a/tail.rs
+++ b/tail.rs
@@ -1 +1 @@
-a
+b
"""


def diff_load(*rules, **kw):
    return gated(enabled=False, run_if_diff=[dict(r) for r in rules], **kw)


def diff_hit(rule, diff=SAMPLE_DIFF):
    return psm.match_diff_rule(rule, psm.parse_unified_diff(diff))


def test_parse_unified_diff_entries():
    es = psm.parse_unified_diff(SAMPLE_DIFF)
    assert [e["path"] for e in es] == [
        "crates/hipfire-arch-qwen35/src/ep.rs",
        "crates/hipfire-arch-qwen35/src/ep.rs",
        "new/name.rs",
        "docs/new.md",
        "gone/legacy.rs",
        "assets/logo.png",
        "tail.rs",
    ]
    assert es[0]["hunk"] == (
        "@@ -120,2 +120,3 @@ fn sealed_moe_view(buf: &Buf) -> View {\n"
        "-    let v = buf.view();\n+    let v = buf.view_d2();\n+    // sealed"
    )
    assert es[1]["hunk"] == "@@ -300 +301 @@ impl Foo {\n-    old_line();\n+    new_line();"  # no "\ No newline" line
    ren = es[2]
    assert (ren["a_path"], ren["b_path"]) == ("old/name.rs", "new/name.rs")
    assert "rename from old/name.rs" in ren["hunk"] and "similarity index 100%" in ren["hunk"]
    new = es[3]
    assert new["a_path"] is None and new["b_path"] == "docs/new.md" and new["hunk"].endswith("+hello d2 decode")
    gone = es[4]
    assert gone["b_path"] is None and gone["a_path"] == "gone/legacy.rs" and gone["path"] == "gone/legacy.rs"
    assert gone["hunk"].splitlines()[0] == "@@ -1,2 +0,0 @@ fn legacy_fn() {"
    assert "---not a file header, a removed line starting with dashes" in gone["hunk"]
    assert es[5]["hunk"] == "Binary files a/assets/logo.png and b/assets/logo.png differ"
    assert es[6]["hunk"] == "@@ -1 +1 @@\n-a\n+b"  # parsing continued past the binary entry


def test_parse_unified_diff_tolerates_garbage_empty_and_spaces():
    assert psm.parse_unified_diff("") == []
    assert psm.parse_unified_diff("warning: junk\nnot a diff\n") == []
    es = psm.parse_unified_diff(
        "junk before\ndiff --git a/dir with space/f.rs b/dir with space/f.rs\n--- a/dir with space/f.rs\t\n"
        "+++ b/dir with space/f.rs\t\n@@ -1 +1 @@\n-x\n+y\n"
    )
    assert [e["path"] for e in es] == ["dir with space/f.rs"]
    pure_mode = psm.parse_unified_diff("diff --git a/x.sh b/x.sh\nold mode 100644\nnew mode 100755\n")
    assert pure_mode[0]["path"] == "x.sh" and "new mode 100755" in pure_mode[0]["hunk"]
    empty_new = psm.parse_unified_diff("diff --git a/e b/e\nnew file mode 100644\nindex 0000000..e69de29\n")
    assert empty_new[0]["path"] == "e" and empty_new[0]["a_path"] is None


def test_diff_rule_hunk_header_funcname_match():
    hit = diff_hit({"path": r"ep\.rs$", "hunk": r"^@@ .*sealed_moe_view"})
    assert hit == {"path": "crates/hipfire-arch-qwen35/src/ep.rs", "line": "@@ -120,2 +120,3 @@ fn sealed_moe_view(buf: &Buf) -> View {"}
    assert diff_hit({"path": r"ep\.rs$", "hunk": r"@@ .*impl Foo"})["line"] == "@@ -300 +301 @@ impl Foo {"


def test_diff_rule_plus_minus_line_match():
    assert diff_hit({"path": "ep", "hunk": r"^\+.*view_d2"})["line"] == "+    let v = buf.view_d2();"
    assert diff_hit({"path": "ep", "hunk": r"^-.*old_line"})["line"] == "-    old_line();"
    assert diff_hit({"path": "ep"})["line"].startswith("@@ -120")  # no hunk regex: any hunk of the path
    # `+` lines only: a removed-only token must not match a `^\+` rule
    assert diff_hit({"path": "ep", "hunk": r"^\+.*old_line"}) is None


def test_diff_rule_is_per_file():
    assert diff_hit({"path": r"^docs/", "hunk": "view_d2"}) is None  # text lives in a different file's hunk
    assert diff_hit({"path": r"^docs/", "hunk": r"^\+hello"})["path"] == "docs/new.md"


def test_diff_rule_new_file_delete_rename_binary():
    assert diff_hit({"path": "docs/new.md", "hunk": r"d2 decode"}) is not None
    assert diff_hit({"path": "legacy", "hunk": "removed_marker"})["path"] == "gone/legacy.rs"  # a-side for deletions
    assert diff_hit({"path": "legacy", "hunk": r"^@@ .*legacy_fn"}) is not None
    assert diff_hit({"path": r"^new/name", "hunk": r"rename from old/"})["path"] == "new/name.rs"
    assert diff_hit({"path": r"^old/name"}) is None  # renames match on the b side
    assert diff_hit({"path": r"logo\.png", "hunk": "Binary files"})["line"].startswith("Binary files a/assets/logo.png")
    assert diff_hit({"path": r"^tail\.rs$", "hunk": r"^\+b$"}) is not None  # still parsed after the binary entry


def test_diff_rule_no_match_and_line_truncation():
    assert diff_hit({"path": "crates/", "hunk": "nothing_like_this"}) is None
    assert diff_hit({"path": "no/such/path"}) is None
    big = "diff --git a/f b/f\n--- a/f\n+++ b/f\n@@ -1 +1 @@\n+" + "x" * 500 + "\n"
    hit = diff_hit({"path": "f", "hunk": r"^\+x+"}, big)
    assert len(hit["line"]) == 200


def test_evaluate_load_diff_records_rule_index_path_and_line():
    ld = diff_load({"path": "nomatch", "hunk": "zzz"}, {"path": r"ep\.rs$", "hunk": r"^\+.*view_d2"})
    ev = psm.evaluate_load(ld, None, psm.parse_unified_diff(SAMPLE_DIFF))
    assert ev["run"] is True and len(ev["matched"]) == 1
    m = ev["matched"][0]
    assert m["kind"] == "run_if_diff" and m["rule_index"] == 1
    assert m["path"] == "crates/hipfire-arch-qwen35/src/ep.rs" and m["line"] == "+    let v = buf.view_d2();"
    assert "run_if_diff[1]" in ev["reason"] and "view_d2" in ev["reason"]
    assert psm.load_selection(ld, None, psm.parse_unified_diff(SAMPLE_DIFF)) == (True, ev["reason"])


def test_evaluate_load_diff_no_match_and_missing_diff():
    ld = diff_load({"path": "nomatch"})
    ev = psm.evaluate_load(ld, None, psm.parse_unified_diff(SAMPLE_DIFF))
    assert ev["run"] is False and ev["matched"] == [] and "no diff hunk matches run_if_diff" in ev["reason"]
    ev = psm.evaluate_load(ld, None, None)
    assert ev["run"] is False and "--diff not supplied" in ev["reason"]
    assert psm.evaluate_load(ld, None, [])["run"] is False  # empty diff given: nothing matches


def test_evaluate_load_or_semantics_and_absent_filters():
    es = psm.parse_unified_diff(SAMPLE_DIFF)
    both = gated(enabled=False, run_if_paths=["^never/"], run_if_diff=[{"path": "tail", "hunk": "^\\+b"}])
    assert psm.evaluate_load(both, ["README.md"], es)["run"] is True  # diff rule alone enables
    assert psm.evaluate_load(both, ["README.md"], None)["run"] is False
    ev = psm.evaluate_load(both, ["README.md"], None)
    assert "--diff not supplied" in ev["reason"] and "no changed path matches run_if_paths" in ev["reason"]
    both2 = gated(enabled=False, run_if_paths=["README"], run_if_diff=[{"path": "nomatch"}])
    ev = psm.evaluate_load(both2, ["README.md"], es)  # paths alone enables
    assert ev["run"] is True and [m["kind"] for m in ev["matched"]] == ["run_if_paths"]
    ev = psm.evaluate_load(gated(enabled=False, run_if_paths=["README", "docs"], run_if_diff=[{"path": "tail"}]), ["README.md", "docs/a"], es)
    assert [m["kind"] for m in ev["matched"]] == ["run_if_paths", "run_if_paths", "run_if_diff"]
    assert "(+2 more rule(s))" in ev["reason"]
    # neither filter given anywhere -> only `enabled` decides
    assert psm.evaluate_load(gated(enabled=False), None, None)["run"] is False
    assert psm.evaluate_load(gated(enabled=False), ["x"], es)["run"] is False
    assert psm.evaluate_load(gated(enabled=True), None, None) == {"run": True, "reason": "enabled", "matched": []}
    assert psm.evaluate_load(gated(enabled=False), None, None)["reason"].startswith("disabled")
    ev = psm.evaluate_load(gated(enabled=True, run_if_diff=[{"path": "nomatch"}]), None, None)
    assert ev["run"] is True and ev["matched"] == []  # enabled loads never need a rule


def test_validate_plan_run_if_diff():
    p = psm.validate_plan({**base_raw(), "loads": [{**base_raw()["loads"][0], "run_if_diff": [{"path": "a"}, {"path": "b", "hunk": "^\\+x"}]}]})
    assert p["loads"][0]["run_if_diff"] == [{"path": "a", "hunk": ""}, {"path": "b", "hunk": "^\\+x"}]
    assert psm.validate_plan(base_raw())["loads"][0]["run_if_diff"] == []
    for bad, needle in [
        ("nope", "list"),
        ([{"hunk": "x"}], "'path'"),
        ([{"path": "a", "extra": 1}], "'path'"),
        ([["a"]], "'path'"),
        ([{"path": "("}], "bad regex"),
        ([{"path": "a", "hunk": "["}], "bad regex"),
        ([{"path": "a", "hunk": 3}], "not a string"),
    ]:
        r = base_raw()
        r["loads"][0]["run_if_diff"] = bad
        assert needle in errs(r), bad


def diff_plan(tmp_path):
    pd = e2e_plan()
    pd["invariants"] = []
    pd["loads"][0]["cells"] = [{"name": "ar", "generate": {"fake_seed": "x"}}]
    pd["loads"].append(
        {
            "name": "ep2",
            "model": "m",
            "deadline_s": 20,
            "enabled": False,
            "run_if_diff": [{"path": r"ep\.rs$", "hunk": r"^\+.*view_d2"}],
            "cells": [{"name": "ep2-ar", "generate": {"fake_seed": "e"}}],
        }
    )
    pd["loads"].append(
        {
            "name": "pp2",
            "model": "m",
            "deadline_s": 20,
            "enabled": False,
            "run_if_diff": [{"path": "pp\\.rs$"}],
            "cells": [{"name": "pp2-ar", "generate": {"fake_seed": "p"}}],
        }
    )
    return write_plan(tmp_path, pd)


def test_run_with_diff_records_matched_rule_and_skips_others(tmp_path, fake_daemon):
    plan = diff_plan(tmp_path)
    diff = tmp_path / "land.diff"
    diff.write_text(SAMPLE_DIFF)
    out = tmp_path / "out"
    assert run_cli(tmp_path, plan, fake_daemon, out, "--diff", str(diff)) == 0
    res = read(out / "ep2.json")
    sel = res["selection"]
    assert sel["run"] is True and sel["matched"][0]["rule_index"] == 0
    assert sel["matched"][0]["path"] == "crates/hipfire-arch-qwen35/src/ep.rs"
    assert sel["matched"][0]["line"] == "+    let v = buf.view_d2();"
    assert res["cells"]["ep2-ar"]["n"] == 8
    sk = read(out / "pp2.json")
    assert sk["skipped"] is True and "no diff hunk matches run_if_diff" in sk["reason"] and sk["matched"] == []
    run = read(out / "run.json")
    assert run["diff_given"] is True and run["changed_paths_given"] is False
    assert run["loads"]["ep2"]["matched"][0]["kind"] == "run_if_diff" and run["loads"]["pp2"]["skipped"] is True
    assert read(out / "L1.json")["selection"] == {"run": True, "reason": "enabled", "matched": []}
    # without --diff the filtered loads are skipped with an explicit reason
    out2 = tmp_path / "out2"
    assert run_cli(tmp_path, plan, fake_daemon, out2) == 0
    assert "--diff not supplied" in read(out2 / "ep2.json")["reason"] and read(out2 / "run.json")["diff_given"] is False
    # and the verdict treats them as SKIPPED
    rc = psm.main(["verdict", "--results", str(out), "--plan", str(plan), "--pins", str(tmp_path / "n.json"),
                   "--sha", "s", "--out", str(tmp_path / "v.json")])
    v = read(tmp_path / "v.json")
    assert rc == 0 and v["cells"]["pp2-ar"]["status"] == "SKIPPED" and v["cells"]["ep2-ar"]["status"] == "NEW"


def test_plan_subcommand_with_diff(tmp_path, capsys):
    plan = diff_plan(tmp_path)
    diff = tmp_path / "land.diff"
    diff.write_text(SAMPLE_DIFF)
    assert psm.main(["plan", "--plan", str(plan), "--diff", str(diff)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("RUN ") and out[1].startswith("RUN ") and "run_if_diff[0]" in out[1]
    assert "crates/hipfire-arch-qwen35/src/ep.rs" in out[1] and "view_d2" in out[1]
    assert out[2].startswith("SKIP") and "no diff hunk matches" in out[2]
    assert psm.main(["plan", "--plan", str(plan)]) == 0
    assert "--diff not supplied" in capsys.readouterr().out
    assert psm.main(["plan", "--plan", str(plan), "--diff", str(tmp_path / "missing.diff")]) == 2


def test_help_mentions_diff_flag():
    for sub in ("plan", "run"):
        r = subprocess.run([sys.executable, str(SCRIPTS / "post_land_smoke.py"), sub, "--help"], capture_output=True, text=True)
        assert "--diff" in r.stdout


# --------------------------------------------------------------------------
# multi-lane verdict (--results DIR [DIR ...])
# --------------------------------------------------------------------------

PP_DIFF = """\
diff --git a/crates/x/src/pp.rs b/crates/x/src/pp.rs
--- a/crates/x/src/pp.rs
+++ b/crates/x/src/pp.rs
@@ -1 +1 @@ fn pp_window() {
-a
+b
"""


def lane_plan(tmp_path):
    pd = e2e_plan()
    pd["loads"][0]["cells"] = [
        {"name": "ar", "generate": {"fake_seed": "x"}},
        {"name": "mtp", "generate": {"fake_seed": "x"}},
    ]
    pd["loads"].append({"name": "L2", "model": "m2", "deadline_s": 20, "cells": [{"name": "other", "generate": {"fake_seed": "x"}}]})
    pd["loads"].append(
        {
            "name": "L3",
            "model": "m3",
            "deadline_s": 20,
            "enabled": False,
            "run_if_diff": [{"path": r"pp\.rs$", "hunk": r"pp_window"}],
            "note": "pp lane",
            "cells": [{"name": "pp", "generate": {"fake_seed": "p"}}],
        }
    )
    pd["invariants"] = [["ar", "mtp"], ["ar", "other"]]
    return write_plan(tmp_path, pd)


def lane_verdict(tmp_path, plan, *dirs, extra=()):
    return psm.main(["verdict", "--results", *map(str, dirs), "--plan", str(plan), "--pins", str(tmp_path / "none.json"),
                     "--sha", "s", "--out", str(tmp_path / "v.json"), *extra])


def test_verdict_merges_lane_dirs(tmp_path, fake_daemon):
    plan = lane_plan(tmp_path)
    a, b = tmp_path / "laneA", tmp_path / "laneB"
    assert run_cli(tmp_path, plan, fake_daemon, a, "--only", "L1") == 0
    assert run_cli(tmp_path, plan, fake_daemon, b, "--only", "L2", "L3") == 0  # L3 skipped: enabled=false
    assert lane_verdict(tmp_path, plan, a, b) == 0
    v = read(tmp_path / "v.json")
    assert v["results_dirs"] == [str(a), str(b)] and "results_dir" not in v
    assert {c: e["status"] for c, e in v["cells"].items()} == {"ar": "NEW", "mtp": "NEW", "other": "NEW", "pp": "SKIPPED"}
    assert v["cells"]["other"]["load"] == "L2" and v["cells"]["pp"]["note"] == "pp lane"
    # the cross-lane invariant is evaluated across the merged dirs
    assert [i["status"] for i in v["invariants"]] == ["OK", "OK"]
    # dir order does not matter
    assert lane_verdict(tmp_path, plan, b, a) == 0


def test_verdict_cross_lane_invariant_violation(tmp_path, fake_daemon):
    pd = read(lane_plan(tmp_path))
    pd["loads"][1]["cells"][0]["generate"]["fake_seed"] = "different"
    plan = write_plan(tmp_path, pd, "plan2.json")
    a, b = tmp_path / "laneA", tmp_path / "laneB"
    assert run_cli(tmp_path, plan, fake_daemon, a, "--only", "L1") == 0
    assert run_cli(tmp_path, plan, fake_daemon, b, "--only", "L2") == 0
    assert lane_verdict(tmp_path, plan, a, b) == 1
    assert [i["status"] for i in read(tmp_path / "v.json")["invariants"]] == ["OK", "INVARIANT"]


def test_verdict_overlapping_dirs_exit_2(tmp_path, fake_daemon, capsys):
    plan = lane_plan(tmp_path)
    a, b = tmp_path / "laneA", tmp_path / "laneB"
    assert run_cli(tmp_path, plan, fake_daemon, a, "--only", "L1") == 0
    assert run_cli(tmp_path, plan, fake_daemon, b, "--only", "mtp") == 0  # a cell of L1: same load
    capsys.readouterr()
    assert lane_verdict(tmp_path, plan, a, b) == 2
    assert "load 'L1' is covered by both" in capsys.readouterr().err
    assert not (tmp_path / "v.json").exists()


def test_verdict_dir_without_run_json_or_missing_dir_exit_2(tmp_path, fake_daemon, capsys):
    plan = lane_plan(tmp_path)
    a = tmp_path / "laneA"
    assert run_cli(tmp_path, plan, fake_daemon, a, "--only", "L1") == 0
    empty = tmp_path / "empty"
    empty.mkdir()
    capsys.readouterr()
    assert lane_verdict(tmp_path, plan, a, empty) == 2
    assert "no run.json" in capsys.readouterr().err
    assert lane_verdict(tmp_path, plan, a, tmp_path / "nope") == 2
    assert "not a directory" in capsys.readouterr().err
    (a / "run.json").unlink()  # a lone dir without run.json is also an error now
    assert lane_verdict(tmp_path, plan, a) == 2


def test_verdict_missing_lane_is_fail_not_run(tmp_path, fake_daemon):
    plan = lane_plan(tmp_path)
    a = tmp_path / "laneA"
    assert run_cli(tmp_path, plan, fake_daemon, a, "--only", "L1") == 0  # L2 lane never ran / was cancelled
    assert lane_verdict(tmp_path, plan, a) == 1
    v = read(tmp_path / "v.json")
    assert v["overall"] == "FAIL" and v["cells"]["other"]["status"] == "FAIL"
    assert "not run" in v["cells"]["other"]["reasons"][0] and "'L2'" in v["cells"]["other"]["reasons"][0]
    assert v["cells"]["ar"]["status"] == "NEW"
    # L3 is enabled=false with run_if_diff and no --diff was given: recomputed as skipped, not "not run"
    assert v["cells"]["pp"]["status"] == "SKIPPED" and "--diff not supplied" in v["cells"]["pp"]["reasons"][0]
    assert v["invariants"][1]["status"] == "UNCHECKED"  # ar vs other: other is FAIL
    # --partial judges only what ran
    assert lane_verdict(tmp_path, plan, a, extra=["--partial"]) == 0
    assert list(read(tmp_path / "v.json")["cells"]) == ["ar", "mtp"]


def test_verdict_cell_left_out_by_lane_only_is_fail_not_run(tmp_path, fake_daemon):
    plan = lane_plan(tmp_path)
    a, b = tmp_path / "laneA", tmp_path / "laneB"
    assert run_cli(tmp_path, plan, fake_daemon, a, "--only", "mtp") == 0
    assert run_cli(tmp_path, plan, fake_daemon, b, "--only", "L2") == 0
    assert lane_verdict(tmp_path, plan, a, b) == 1
    c = read(tmp_path / "v.json")["cells"]
    assert c["ar"]["status"] == "FAIL" and "not run" in c["ar"]["reasons"][0] and c["mtp"]["status"] == "NEW"


def test_verdict_filter_recomputed_from_run_json_inputs(tmp_path, fake_daemon):
    plan = lane_plan(tmp_path)
    diff_hit_f, diff_miss_f = tmp_path / "hit.diff", tmp_path / "miss.diff"
    diff_hit_f.write_text(PP_DIFF)
    diff_miss_f.write_text(SAMPLE_DIFF)
    # diff does NOT touch pp.rs: the unrun L3 lane is SKIPPED by the filter -> PASS
    a = tmp_path / "missA"
    assert run_cli(tmp_path, plan, fake_daemon, a, "--only", "L1", "L2", "--diff", str(diff_miss_f)) == 0
    info = read(a / "run.json")
    assert info["diff_given"] is True and info["diff_sha256"] == hashlib.sha256(diff_miss_f.read_bytes()).hexdigest()
    assert info["changed_paths"] is None and info["diff_entries"][0]["path"] == "crates/hipfire-arch-qwen35/src/ep.rs"
    assert lane_verdict(tmp_path, plan, a) == 0
    v = read(tmp_path / "v.json")
    assert v["cells"]["pp"]["status"] == "SKIPPED" and "no diff hunk matches run_if_diff" in v["cells"]["pp"]["reasons"][0]
    # diff touches pp.rs: L3 must have run, but no dir has it -> FAIL "not run"
    b = tmp_path / "hitA"
    assert run_cli(tmp_path, plan, fake_daemon, b, "--only", "L1", "L2", "--diff", str(diff_hit_f)) == 0
    assert lane_verdict(tmp_path, plan, b) == 1
    pp = read(tmp_path / "v.json")["cells"]["pp"]
    assert pp["status"] == "FAIL" and "not run" in pp["reasons"][0]
    # the L3 lane's own dir (same diff) fixes it
    c = tmp_path / "hitB"
    assert run_cli(tmp_path, plan, fake_daemon, c, "--only", "L3", "--diff", str(diff_hit_f)) == 0
    assert read(c / "L3.json")["selection"]["matched"][0]["rule_index"] == 0
    assert lane_verdict(tmp_path, plan, b, c) == 0
    assert read(tmp_path / "v.json")["cells"]["pp"]["status"] == "NEW"


def test_verdict_changed_paths_filter_recomputed(tmp_path, fake_daemon):
    pd = read(lane_plan(tmp_path))
    pd["loads"][2]["run_if_paths"] = ["^scripts/"]
    pd["loads"][2]["run_if_diff"] = []
    plan = write_plan(tmp_path, pd, "plan3.json")
    cp = tmp_path / "cp.txt"
    cp.write_text("scripts/post_land_smoke.py\n")
    a = tmp_path / "A"
    assert run_cli(tmp_path, plan, fake_daemon, a, "--only", "L1", "L2", "--changed-paths", str(cp)) == 0
    assert read(a / "run.json")["changed_paths"] == ["scripts/post_land_smoke.py"]
    assert lane_verdict(tmp_path, plan, a) == 1  # filter selects L3, nobody ran it
    assert "not run" in read(tmp_path / "v.json")["cells"]["pp"]["reasons"][0]


def test_verdict_dirs_disagreeing_on_inputs_exit_2(tmp_path, fake_daemon, capsys):
    plan = lane_plan(tmp_path)
    d1, d2 = tmp_path / "d1.diff", tmp_path / "d2.diff"
    d1.write_text(PP_DIFF)
    d2.write_text(SAMPLE_DIFF)
    a, b, c, d = (tmp_path / n for n in "abcd")
    assert run_cli(tmp_path, plan, fake_daemon, a, "--only", "L1", "--diff", str(d1)) == 0
    assert run_cli(tmp_path, plan, fake_daemon, b, "--only", "L2", "--diff", str(d2)) == 0
    capsys.readouterr()
    assert lane_verdict(tmp_path, plan, a, b) == 2
    assert "different --diff" in capsys.readouterr().err
    assert run_cli(tmp_path, plan, fake_daemon, c, "--only", "L2") == 0  # no --diff at all
    assert lane_verdict(tmp_path, plan, a, c) == 2
    cp1, cp2 = tmp_path / "cp1", tmp_path / "cp2"
    cp1.write_text("a\n")
    cp2.write_text("b\n")
    assert run_cli(tmp_path, plan, fake_daemon, d, "--only", "L1", "--changed-paths", str(cp1)) == 0
    e = tmp_path / "e"
    assert run_cli(tmp_path, plan, fake_daemon, e, "--only", "L2", "--changed-paths", str(cp2)) == 0
    capsys.readouterr()
    assert lane_verdict(tmp_path, plan, d, e) == 2
    assert "different --changed-paths" in capsys.readouterr().err
    # same content, different order is the same input
    cp3 = tmp_path / "cp3"
    cp3.write_text("b\na\n")
    cp4 = tmp_path / "cp4"
    cp4.write_text("a\nb\n")
    f, g = tmp_path / "f", tmp_path / "g"
    assert run_cli(tmp_path, plan, fake_daemon, f, "--only", "L1", "--changed-paths", str(cp3)) == 0
    assert run_cli(tmp_path, plan, fake_daemon, g, "--only", "L2", "--changed-paths", str(cp4)) == 0
    assert lane_verdict(tmp_path, plan, f, g) == 0


def test_build_verdict_absent_handling_unit():
    plan = mk_plan()
    res = {"fn1": load_res({"fn1-ar": cell_rec(), "fn1-mtp": cell_rec()})}
    sel = {"fn1": ["fn1-ar", "fn1-mtp"]}
    v = psm.build_verdict(plan, res, {}, "s", ["/a"], selected=sel, absent={"a3b": {"run": True, "reason": "enabled"}})
    assert v["cells"]["a3b-ar"]["status"] == "FAIL" and "not run" in v["cells"]["a3b-ar"]["reasons"][0]
    assert v["overall"] == "FAIL" and v["results_dirs"] == ["/a"]
    v = psm.build_verdict(plan, res, {}, "s", ["/a"], selected=sel, absent={"a3b": {"run": False, "reason": "disabled; x"}})
    assert v["cells"]["a3b-ar"]["status"] == "SKIPPED" and v["cells"]["a3b-ar"]["reasons"] == ["disabled; x"]
    assert v["overall"] == "PASS" and v["cells"]["a3b-ar"]["n"] is None
    v = psm.build_verdict(plan, res, {}, "s", ["/a"], selected=sel, absent=None)  # --partial
    assert "a3b-ar" not in v["cells"] and v["overall"] == "PASS"


# --------------------------------------------------------------------------
# stderr_require (route proof)
# --------------------------------------------------------------------------

ROUTE_STDERR = "[daemon] mtp: native drafter armed k=3\n[daemon] d2 decode path: forced off\nunrelated noise"


def route_plan(tmp_path, require, env=None, **load_kw):
    pd = e2e_plan(env={"FAKE_STDERR": ROUTE_STDERR, **(env or {})})
    pd["loads"][0]["stderr_require"] = require
    pd["loads"][0].update(load_kw)
    return write_plan(tmp_path, pd)


def vcall(tmp_path, plan, out):
    return psm.main(["verdict", "--results", str(out), "--plan", str(plan), "--pins", str(tmp_path / "n.json"),
                     "--sha", "s", "--out", str(tmp_path / "v.json")])


def test_scan_stderr_require_first_match_and_truncation():
    lines = ["noise\n", "mtp armed k=3\n", "mtp armed k=4\n", b"d2 forced off\n", "x" * 500 + " ROUTE\n"]
    proof = psm.scan_stderr_require(lines, [r"mtp armed k=\d", r"d2 forced off", r"ROUTE$", r"never"])
    assert proof[0] == {"pattern": r"mtp armed k=\d", "matched": True, "line": "mtp armed k=3"}  # first match wins
    assert proof[1]["line"] == "d2 forced off" and proof[2]["matched"] is True and len(proof[2]["line"]) == 200
    assert proof[3] == {"pattern": "never", "matched": False, "line": None}
    assert psm.scan_stderr_require(lines, []) == []


def test_validate_plan_stderr_require():
    assert psm.validate_plan(base_raw())["loads"][0]["stderr_require"] == []
    r = base_raw(); r["loads"][0]["stderr_require"] = ["a+", "b"]
    assert psm.validate_plan(r)["loads"][0]["stderr_require"] == ["a+", "b"]
    r = base_raw(); r["loads"][0]["stderr_require"] = ["(unclosed"]
    assert "stderr_require" in errs(r) and "bad regex" in errs(r)
    r = base_raw(); r["loads"][0]["stderr_require"] = "mtp"
    assert "stderr_require: must be a list" in errs(r)
    r = base_raw(); r["loads"][0]["stderr_require"] = [3]
    assert "not a string" in errs(r)


def test_cell_status_route_proof_missing_is_fail():
    lr = load_res({"c": cell_rec()}, route_proof=[
        {"pattern": "mtp armed", "matched": True, "line": "mtp armed"},
        {"pattern": "d2 forced off", "matched": False, "line": None},
    ])
    st = psm.cell_status(lr, lr["cells"]["c"], 8, pin_of(IDS))
    assert st["status"] == "FAIL" and st["reasons"] == ["required route line missing: d2 forced off"]
    lr["route_proof"][1].update(matched=True, line="d2 forced off")
    assert psm.cell_status(lr, lr["cells"]["c"], 8, pin_of(IDS))["status"] == "MATCH"
    assert psm.cell_status(load_res({"c": cell_rec()}), cell_rec(), 8, None)["status"] == "NEW"  # no route_proof key


def test_e2e_route_proof_all_present_passes(tmp_path, fake_daemon):
    plan = route_plan(tmp_path, [r"mtp: native drafter armed k=\d", r"d2 decode path: forced off"])
    out = tmp_path / "out"
    assert run_cli(tmp_path, plan, fake_daemon, out) == 0
    proof = read(out / "L1.json")["route_proof"]
    assert proof == [
        {"pattern": r"mtp: native drafter armed k=\d", "matched": True, "line": "[daemon] mtp: native drafter armed k=3"},
        {"pattern": "d2 decode path: forced off", "matched": True, "line": "[daemon] d2 decode path: forced off"},
    ]
    assert vcall(tmp_path, plan, out) == 0
    assert {c["status"] for c in read(tmp_path / "v.json")["cells"].values()} == {"NEW"}


def test_e2e_route_proof_one_missing_fails_every_cell(tmp_path, fake_daemon):
    # cells commit perfectly good ids, but the required route line never appeared (silent fallback)
    plan = route_plan(tmp_path, [r"mtp: native drafter armed", r"mtp: verify path engaged"])
    out = tmp_path / "out"
    assert run_cli(tmp_path, plan, fake_daemon, out) == 0
    res = read(out / "L1.json")
    assert [p["matched"] for p in res["route_proof"]] == [True, False] and res["route_proof"][1]["line"] is None
    assert all(c["n"] == 8 for c in res["cells"].values())
    assert vcall(tmp_path, plan, out) == 1
    v = read(tmp_path / "v.json")
    assert v["overall"] == "FAIL"
    for name in ("ar", "mtp"):
        assert v["cells"][name]["status"] == "FAIL"
        assert v["cells"][name]["reasons"] == ["required route line missing: mtp: verify path engaged"]
    assert v["invariants"][0]["status"] == "UNCHECKED"
    # a pinned FAIL load must not be pinnable
    assert psm.main(["pin", "--verdict", str(tmp_path / "v.json"), "--pins", str(tmp_path / "pins.json")]) == 1


def test_e2e_route_proof_empty_stderr_means_missing(tmp_path, fake_daemon):
    plan = route_plan(tmp_path, ["anything"], env={"FAKE_STDERR": ""})
    out = tmp_path / "out"
    assert run_cli(tmp_path, plan, fake_daemon, out) == 0
    assert read(out / "L1.json")["route_proof"] == [{"pattern": "anything", "matched": False, "line": None}]
    assert vcall(tmp_path, plan, out) == 1


def test_run_invalid_stderr_require_regex_exit_2(tmp_path, fake_daemon, capsys):
    plan = route_plan(tmp_path, ["(unclosed"])
    assert run_cli(tmp_path, plan, fake_daemon, tmp_path / "out") == 2
    assert "stderr_require" in capsys.readouterr().err and not (tmp_path / "out" / "L1.json").exists()
    assert psm.main(["plan", "--plan", str(plan)]) == 2


def test_route_proof_with_skipped_load(tmp_path, fake_daemon):
    # skipped load: no daemon, no route proof, cells SKIPPED (never FAIL for missing route lines)
    plan = route_plan(tmp_path, ["never printed"], enabled=False, run_if_paths=["^crates/"])
    out = tmp_path / "out"
    assert run_cli(tmp_path, plan, fake_daemon, out) == 0
    sk = read(out / "L1.json")
    assert sk["skipped"] is True and "route_proof" not in sk
    assert vcall(tmp_path, plan, out) == 0
    v = read(tmp_path / "v.json")
    assert {c["status"] for c in v["cells"].values()} == {"SKIPPED"} and v["overall"] == "PASS"
    # the same load enabled by a matching changed path runs and IS held to its route proof
    cp = tmp_path / "cp.txt"
    cp.write_text("crates/x.rs\n")
    out2 = tmp_path / "out2"
    assert run_cli(tmp_path, plan, fake_daemon, out2, "--changed-paths", str(cp)) == 0
    assert read(out2 / "L1.json")["route_proof"][0]["matched"] is False
    assert vcall(tmp_path, plan, out2) == 1
    assert read(tmp_path / "v.json")["cells"]["ar"]["status"] == "FAIL"


# --------------------------------------------------------------------------
# loaded_expect / loaded_expect_not
# --------------------------------------------------------------------------

LOADED_EV = {"type": "loaded", "arch": "qwen3_5", "pp": 2, "vl": False,
             "topology": {"pp": 2, "ep": 1, "ranks": [0, 1]}, "kv_backend": "q8"}


def test_check_loaded_unit():
    ck = psm.check_loaded(LOADED_EV, {"pp": 2, "topology.pp": 2, "topology.ranks.1": 1, "kv_backend": "q8"}, {"arch": "gemma4"})
    assert all(c["ok"] for c in ck) and [c["op"] for c in ck] == ["expect"] * 4 + ["expect_not"]
    bad = psm.check_loaded(LOADED_EV, {"pp": 1, "nope": 1, "vl": 0, "topology.ep": 2}, {"arch": "qwen3_5", "gone": 3})
    got = {(c["key"], c["op"]): (c["ok"], c["found"], c["got"]) for c in bad}
    assert got[("pp", "expect")] == (False, True, 2)
    assert got[("nope", "expect")] == (False, False, None)
    assert got[("vl", "expect")][0] is False  # False != 0: exact JSON equality, no truthiness coercion
    assert got[("topology.ep", "expect")] == (False, True, 1)
    assert got[("arch", "expect_not")][0] is False
    assert got[("gone", "expect_not")] == (False, False, None)  # expect_not also requires the key to exist
    assert psm.check_loaded(LOADED_EV, {"topology": {"pp": 2, "ep": 1, "ranks": [0, 1]}}, None)[0]["ok"]
    assert psm.check_loaded({"a.b": 1, "a": {"b": 2}}, {"a.b": 1}, None)[0]["ok"]  # literal key wins
    absent = psm.check_loaded(None, {"pp": 2}, {"x": 1})
    assert [c["ok"] for c in absent] == [False, False] and not any(c["found"] for c in absent)
    assert psm.check_loaded(LOADED_EV, None, None) == [] and psm.check_loaded(None, {}, {}) == []


def test_validate_plan_loaded_expect():
    p = psm.validate_plan(base_raw())["loads"][0]
    assert p["loaded_expect"] == {} and p["loaded_expect_not"] == {}
    r = base_raw(); r["loads"][0]["loaded_expect"] = {"topology.pp": 2, "vl": False}; r["loads"][0]["loaded_expect_not"] = {"arch": "x"}
    p = psm.validate_plan(r)["loads"][0]
    assert p["loaded_expect"] == {"topology.pp": 2, "vl": False} and p["loaded_expect_not"] == {"arch": "x"}
    for fld in ("loaded_expect", "loaded_expect_not"):
        r = base_raw(); r["loads"][0][fld] = ["pp"]
        assert f"{fld}: must be an object" in errs(r)
        r = base_raw(); r["loads"][0][fld] = {"a..b": 1}
        assert "bad key" in errs(r)
        r = base_raw(); r["loads"][0][fld] = {"": 1}
        assert "bad key" in errs(r)


def lcell_status(check):
    lr = load_res({"c": cell_rec()}, loaded_check=check)
    return psm.cell_status(lr, lr["cells"]["c"], 8, None)


def test_cell_status_loaded_mismatch_reasons():
    st = lcell_status(psm.check_loaded(LOADED_EV, {"topology.pp": 1, "pp": 2}, None))
    assert st["status"] == "FAIL" and st["reasons"] == ["loaded field mismatch: key=topology.pp want=1 got=2"]
    st = lcell_status(psm.check_loaded(LOADED_EV, {"nope": "x"}, None))
    assert st["reasons"] == ['loaded field mismatch: key=nope want="x" got=<missing>']
    st = lcell_status(psm.check_loaded(LOADED_EV, None, {"arch": "qwen3_5"}))
    assert st["reasons"] == ['loaded field mismatch: key=arch want=!="qwen3_5" got="qwen3_5"']
    assert lcell_status(psm.check_loaded(LOADED_EV, {"pp": 2}, {"arch": "x"}))["status"] == "NEW"


def loaded_plan(tmp_path, expect=None, expect_not=None, loaded=None, **kw):
    env = {"FAKE_LOADED": json.dumps(loaded if loaded is not None else {"pp": 2, "topology": {"pp": 2, "ranks": [0, 1]}})}
    env.update(kw.pop("env", {}))
    pd = e2e_plan(env=env)
    if expect is not None:
        pd["loads"][0]["loaded_expect"] = expect
    if expect_not is not None:
        pd["loads"][0]["loaded_expect_not"] = expect_not
    pd["loads"][0].update(kw)
    return write_plan(tmp_path, pd)


def test_e2e_loaded_expect_match_and_nested(tmp_path, fake_daemon):
    plan = loaded_plan(tmp_path, {"arch": "qwen3_5", "pp": 2, "topology.pp": 2, "topology.ranks.1": 1}, {"topology.pp": 1})
    out = tmp_path / "out"
    assert run_cli(tmp_path, plan, fake_daemon, out) == 0
    lc = read(out / "L1.json")["loaded_check"]
    assert len(lc) == 5 and all(c["ok"] for c in lc)
    assert {"key": "topology.pp", "op": "expect", "want": 2, "found": True, "got": 2, "ok": True} in lc
    assert vcall(tmp_path, plan, out) == 0


def test_e2e_loaded_expect_mismatch_fails_every_cell(tmp_path, fake_daemon):
    plan = loaded_plan(tmp_path, {"topology.pp": 4}, loaded={"topology": {"pp": 2}})
    out = tmp_path / "out"
    assert run_cli(tmp_path, plan, fake_daemon, out) == 0
    assert all(c["n"] == 8 for c in read(out / "L1.json")["cells"].values())
    assert vcall(tmp_path, plan, out) == 1
    v = read(tmp_path / "v.json")
    for name in ("ar", "mtp"):
        assert v["cells"][name]["status"] == "FAIL"
        assert v["cells"][name]["reasons"] == ["loaded field mismatch: key=topology.pp want=4 got=2"]


def test_e2e_loaded_expect_missing_key_and_expect_not(tmp_path, fake_daemon):
    plan = loaded_plan(tmp_path, {"topology.missing": True}, loaded={"topology": {"pp": 2}})
    out = tmp_path / "out"
    assert run_cli(tmp_path, plan, fake_daemon, out) == 0
    assert vcall(tmp_path, plan, out) == 1
    assert "got=<missing>" in read(tmp_path / "v.json")["cells"]["ar"]["reasons"][0]
    plan = loaded_plan(tmp_path, None, {"arch": "qwen3_5"})  # differs required; it is equal -> FAIL
    out = tmp_path / "out2"
    assert run_cli(tmp_path, plan, fake_daemon, out) == 0
    assert vcall(tmp_path, plan, out) == 1
    assert "want=!=" in read(tmp_path / "v.json")["cells"]["mtp"]["reasons"][0]


def test_e2e_loaded_event_absent_fails(tmp_path, fake_daemon):
    plan = loaded_plan(tmp_path, {"pp": 2}, env={"FAKE_MODE": "load_error"})
    out = tmp_path / "out"
    assert run_cli(tmp_path, plan, fake_daemon, out) == 0
    res = read(out / "L1.json")
    assert res["loaded_check"] == [{"key": "pp", "op": "expect", "want": 2, "found": False, "got": None, "ok": False}]
    assert vcall(tmp_path, plan, out) == 1
    reasons = read(tmp_path / "v.json")["cells"]["ar"]["reasons"]
    assert "loaded field mismatch: key=pp want=2 got=<missing>" in reasons


def test_loaded_expect_with_skipped_load_and_run_exit_2(tmp_path, fake_daemon, capsys):
    plan = loaded_plan(tmp_path, {"pp": 99}, enabled=False)
    out = tmp_path / "out"
    assert run_cli(tmp_path, plan, fake_daemon, out) == 0
    assert "loaded_check" not in read(out / "L1.json")
    assert vcall(tmp_path, plan, out) == 0 and read(tmp_path / "v.json")["cells"]["ar"]["status"] == "SKIPPED"
    pd = read(plan)
    pd["loads"][0]["loaded_expect"] = {"a..b": 1}
    bad = write_plan(tmp_path, pd, "bad.json")
    assert run_cli(tmp_path, bad, fake_daemon, tmp_path / "o2") == 2
    assert "bad key" in capsys.readouterr().err


# --------------------------------------------------------------------------
# model expansion / model_bytes
# --------------------------------------------------------------------------


def model_plan(model, marker=None, reqlog=None, **load_kw):
    pd = e2e_plan(env={k: str(v) for k, v in (("FAKE_MARKER", marker), ("FAKE_REQLOG", reqlog)) if v})
    pd["loads"][0]["model"] = model
    pd["loads"][0].update(load_kw)
    return pd


@pytest.mark.parametrize("form", ["$SMOKE_MD/m.hfq", "${SMOKE_MD}/m.hfq", "~/m.hfq"])
def test_model_env_and_tilde_expand_before_spawn(tmp_path, fake_daemon, monkeypatch, form):
    (tmp_path / "m.hfq").write_bytes(b"x" * 10)
    monkeypatch.setenv("SMOKE_MD", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    reqlog = tmp_path / "req.log"
    out = tmp_path / "out"
    plan = write_plan(tmp_path, model_plan(form, reqlog=reqlog, model_bytes=10))
    assert run_cli(tmp_path, plan, fake_daemon, out) == 0
    sent = [json.loads(ln) for ln in reqlog.read_text().splitlines()]
    assert sent[0]["model"] == str(tmp_path / "m.hfq")  # the daemon is asked to load the expanded path
    res = read(out / "L1.json")
    assert res["model"] == form and res["model_expanded"] == str(tmp_path / "m.hfq")
    assert res["model_bytes"] == 10 and res["model_bytes_want"] == 10 and res["spawned"] is True and res["errors"] == []


def test_unresolved_model_fails_before_spawn(tmp_path, fake_daemon, monkeypatch, capsys):
    monkeypatch.delenv("SMOKE_UNSET_XYZ", raising=False)
    marker = tmp_path / "marker"
    raw = "$SMOKE_UNSET_XYZ/m.hfq"
    plan = write_plan(tmp_path, model_plan(raw, marker=marker))
    out = tmp_path / "out"
    assert run_cli(tmp_path, plan, fake_daemon, out) == 1
    assert not marker.exists()  # the daemon was never started
    res = read(out / "L1.json")
    assert res["spawned"] is False and res["exit_code"] is None
    assert res["errors"] == [{"message": f"unresolved model path: {raw}"}]
    assert all(c["n"] == 0 and any("not run" in e["message"] for e in c["errors"]) for c in res["cells"].values())
    assert read(out / "run.json")["ok"] is False
    assert psm.main(["verdict", "--results", str(out), "--plan", str(plan), "--pins", str(tmp_path / "n.json"),
                     "--sha", "s", "--out", str(tmp_path / "v.json")]) == 1
    reasons = read(tmp_path / "v.json")["cells"]["ar"]["reasons"]
    assert any(f"unresolved model path: {raw}" in r for r in reasons)
    assert not any("daemon exit code" in r for r in reasons)
    capsys.readouterr()
    assert psm.main(["plan", "--plan", str(plan)]) == 0
    assert "UNRESOLVED" in capsys.readouterr().out


def test_model_bytes_mismatch_fails_before_spawn(tmp_path, fake_daemon, capsys):
    model = tmp_path / "m.hfq"
    model.write_bytes(b"x" * 10)
    marker = tmp_path / "marker"
    plan = write_plan(tmp_path, model_plan(str(model), marker=marker, model_bytes=11))
    out = tmp_path / "out"
    assert run_cli(tmp_path, plan, fake_daemon, out) == 1
    assert not marker.exists()
    res = read(out / "L1.json")
    assert res["errors"] == [{"message": "model size mismatch: want 11 got 10"}] and res["spawned"] is False
    assert psm.main(["verdict", "--results", str(out), "--plan", str(plan), "--pins", str(tmp_path / "n.json"),
                     "--sha", "s", "--out", str(tmp_path / "v.json")]) == 1
    assert any("model size mismatch: want 11 got 10" in r for r in read(tmp_path / "v.json")["cells"]["ar"]["reasons"])
    capsys.readouterr()
    psm.main(["plan", "--plan", str(plan)])
    assert "model size mismatch" in capsys.readouterr().out
    # a missing file is a mismatch too
    plan2 = write_plan(tmp_path, model_plan(str(tmp_path / "gone.hfq"), marker=marker, model_bytes=3), "p2.json")
    assert run_cli(tmp_path, plan2, fake_daemon, tmp_path / "out2") == 1
    assert read(tmp_path / "out2" / "L1.json")["errors"] == [{"message": "model size mismatch: want 3 got missing"}]
    assert not marker.exists()


def test_validate_plan_model_bytes():
    r = base_raw()
    r["loads"][0]["model_bytes"] = 12
    assert psm.validate_plan(r)["loads"][0]["model_bytes"] == 12
    assert psm.validate_plan(base_raw())["loads"][0]["model_bytes"] is None
    for bad in ("12", -1, True, 1.5):
        r = base_raw()
        r["loads"][0]["model_bytes"] = bad
        assert "model_bytes" in errs(r)


# --------------------------------------------------------------------------
# kernel_fault_patterns
# --------------------------------------------------------------------------


def test_kernel_fault_patterns_default_and_matching():
    p = psm.validate_plan(base_raw())
    assert p["kernel_fault_patterns"] == psm.DEFAULT_KERNEL_FAULT_PATTERNS and len(p["kernel_fault_patterns"]) == 9
    p["kernel_fault_patterns"].append("x")  # a copy, never the module constant
    assert len(psm.DEFAULT_KERNEL_FAULT_PATTERNS) == 9
    rx = psm.compile_fault_regex(psm.DEFAULT_KERNEL_FAULT_PATTERNS)
    for bad in (
        "amdgpu 0000:03:00.0: amdgpu: ring gfx_0.0.0 timeout",
        "AMDGPU 0000:03:00.0: AMDGPU: RING GFX_0.0.0 TIMEOUT",  # case-insensitive
        "amdgpu: GPU reset begin!",
        "amdgpu 0000:03:00.0: [gfxhub] page fault (src_id:0 ring:0 vmid:8 pasid:32777)",
        "pcieport 0000:00:01.1: AER: Uncorrectable (Fatal) error",
        "Kernel panic - not syncing",
    ):
        assert rx.search(bad), bad
    for ok in ("usb 1-1: new high-speed USB device number 4", "amdgpu 0000:03:00.0: amdgpu: SMU is initialized successfully!"):
        assert not rx.search(ok), ok


def test_kernel_fault_patterns_override_and_errors():
    r = base_raw()
    r["kernel_fault_patterns"] = ["custom-fault", r"ring \d+"]
    assert psm.validate_plan(r)["kernel_fault_patterns"] == ["custom-fault", r"ring \d+"]
    for bad, needle in (([], "non-empty"), (["("], "bad regex"), ("x", "list of regex"), ([1], "not a string")):
        r = base_raw()
        r["kernel_fault_patterns"] = bad
        assert needle in errs(r)


# --------------------------------------------------------------------------
# lanes: validation + `lanes` subcommand
# --------------------------------------------------------------------------


def lane_obj(name, loads, **kw):
    d = {"name": name, "host": "hiptrx", "cards": "A", "mem_gb": 30.3, "hostmem_gb": 60, "hostmem_lock": True,
         "max_secs": 600, "loads": loads}
    d.update(kw)
    return d


def lanes_raw(lanes=None, **over):
    def load(name, **kw):
        d = {"name": name, "model": f"m-{name}", "deadline_s": 10, "cells": [{"name": f"{name}-cell", "generate": {}}]}
        d.update(kw)
        return d

    raw = {
        "version": 1,
        "prompt": PROMPT,
        "loads": [
            load("A"),
            load("B", enabled=False, run_if_paths=[r"b\.rs$"]),
            load("C", enabled=False, run_if_paths=[r"c\.rs$"]),
            load("D"),
        ],
        "lanes": lanes
        or [
            lane_obj("l1", ["A", "B"]),
            lane_obj("l2", ["C"], host="hipx", cards="xtx", mem_gb=22.4, hostmem_gb=64.0, hostmem_lock=False, max_secs=900),
            lane_obj("l3", ["D"]),
        ],
    }
    raw.update(over)
    return raw


def run_lanes(tmp_path, raw, capsys, *extra):
    plan = write_plan(tmp_path, raw)
    capsys.readouterr()
    rc = psm.main(["lanes", "--plan", str(plan), *extra])
    cap = capsys.readouterr()
    return rc, cap


def test_validate_lanes_ok_and_normalised():
    p = psm.validate_plan(lanes_raw())
    assert [ln["name"] for ln in p["lanes"]] == ["l1", "l2", "l3"]
    assert p["lanes"][1]["hostmem_lock"] is False and p["lanes"][0]["loads"] == ["A", "B"]
    assert "lanes" not in psm.validate_plan(base_raw())


def test_validate_lanes_errors():
    assert "'A' is in two lanes" in errs(lanes_raw([lane_obj("l1", ["A", "B"]), lane_obj("l2", ["A", "C", "D"])]))
    assert "unknown load 'Z'" in errs(lanes_raw([lane_obj("l1", ["A", "B", "Z"]), lane_obj("l2", ["C", "D"])]))
    e = errs(lanes_raw([lane_obj("l1", ["A", "B"]), lane_obj("l2", ["C"])]))
    assert "load 'D' is in no lane" in e
    assert "duplicate lane name" in errs(lanes_raw([lane_obj("l1", ["A", "B"]), lane_obj("l1", ["C", "D"])]))
    assert "listed twice" in errs(lanes_raw([lane_obj("l1", ["A", "A", "B", "C", "D"])]))
    for bad, needle in (
        ({"host": ""}, ".host"),
        ({"cards": 3}, ".cards"),
        ({"mem_gb": 0}, ".mem_gb"),
        ({"hostmem_gb": -1}, ".hostmem_gb"),
        ({"hostmem_lock": "yes"}, ".hostmem_lock"),
        ({"max_secs": 0}, ".max_secs"),
        ({"max_secs": 1.5}, ".max_secs"),
        ({"loads": []}, ".loads"),
        ({"name": "bad name"}, ".name"),
    ):
        lane = lane_obj("l1", ["A", "B", "C", "D"])
        lane.update(bad)
        assert needle in errs(lanes_raw([lane])), bad
    r = lanes_raw()
    r["lanes"] = []
    assert "non-empty list of lane objects" in errs(r)
    r["lanes"] = "x"
    assert "lanes" in errs(r)


def test_validate_lanes_rejects_loads_that_can_outrun_the_hold_cap():
    # Loads run one after another inside the hold; the fabric cap must never be what ends a load.
    raw = lanes_raw([lane_obj("l1", ["A", "B", "C", "D"], max_secs=75)])
    psm.validate_plan(raw)  # 4 x (10 s deadline + 5 s grace) + 15 s slack = 75 s fits 75
    raw = lanes_raw([lane_obj("l1", ["A", "B", "C", "D"], max_secs=74)])
    assert "worst-case time 75s" in errs(raw) and "exceeds max_secs 74" in errs(raw)


def test_shipped_plan_validates_and_every_lane_fits_its_hold_cap():
    plan, _ = psm.load_plan(SCRIPTS / "post-land-smoke" / "plan.json")
    deadlines = {ld["name"]: ld["deadline_s"] for ld in plan["loads"]}
    for lane in plan["lanes"]:
        assert lane["max_secs"] <= 600, lane["name"]  # fabric holds are capped at 600 s
        worst = sum(deadlines[x] + psm.TERM_GRACE_S for x in lane["loads"]) + psm.LANE_SLACK_S
        assert worst <= lane["max_secs"], lane["name"]
    loads = {ld["name"]: ld for ld in plan["loads"]}
    assert loads["xtx-ar-load"]["env"]["HIPFIRE_QWEN4_D2_DEFAULT"] == "0" and loads["xtx-ar-load"]["enabled"]
    assert loads["xtx-d2-load"]["env"]["HIPFIRE_QWEN4_D2_DEFAULT"] == "1" and not loads["xtx-d2-load"]["enabled"]


def test_lanes_subcommand_json_fabric_args_and_dropped_lane(tmp_path, capsys):
    rc, cap = run_lanes(tmp_path, lanes_raw(), capsys)
    assert rc == 0 and cap.err == ""
    got = json.loads(cap.out)
    assert [ln["name"] for ln in got] == ["l1", "l3"]  # l2 (only C, gated, no match) is dropped
    l1, l3 = got
    assert l1["loads"] == ["A"] and l1["skipped_loads"] == ["B"] and l1["host"] == "hiptrx" and l1["unresolved_loads"] == []
    assert l1["fabric_args"] == ["--class", "correct", "--card", "A", "--mem-gb", "30.3", "--hostmem-gb", "60",
                                 "--hostmem-lock", "--max-secs", "600"]
    assert (l1["cards"], l1["mem_gb"], l1["hostmem_gb"], l1["hostmem_lock"], l1["max_secs"]) == ("A", 30.3, 60, True, 600)
    assert l3["loads"] == ["D"] and l3["skipped_loads"] == []

    cp = tmp_path / "changed.txt"
    cp.write_text("src/c.rs\nsrc/b.rs\n")
    rc, cap = run_lanes(tmp_path, lanes_raw(), capsys, "--changed-paths", str(cp))
    got = {ln["name"]: ln for ln in json.loads(cap.out)}
    assert list(got) == ["l1", "l2", "l3"] and got["l1"]["loads"] == ["A", "B"] and got["l1"]["skipped_loads"] == []
    # no --hostmem-lock when the lane does not ask for it; float 64.0 prints as 64
    assert got["l2"]["fabric_args"] == ["--class", "correct", "--card", "xtx", "--mem-gb", "22.4", "--hostmem-gb", "64",
                                        "--max-secs", "900"]
    assert got["l2"]["host"] == "hipx" and got["l2"]["loads"] == ["C"]


def test_lanes_with_diff_and_unresolved_and_usage_errors(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("SMOKE_UNSET_XYZ", raising=False)
    raw = lanes_raw()
    raw["loads"][0]["model"] = "$SMOKE_UNSET_XYZ/a.hfq"
    raw["loads"][2]["enabled"] = False
    raw["loads"][2]["run_if_diff"] = [{"path": r"pp\.rs$", "hunk": "pp_window"}]
    d = tmp_path / "x.diff"
    d.write_text(PP_DIFF)
    rc, cap = run_lanes(tmp_path, raw, capsys, "--diff", str(d))
    got = {ln["name"]: ln for ln in json.loads(cap.out)}
    assert rc == 0 and got["l2"]["loads"] == ["C"]  # run_if_diff selected it
    assert got["l1"]["unresolved_loads"] == ["A"] and "UNRESOLVED load 'A'" in cap.err
    # usage errors: no lanes in the plan, invalid lanes, missing file
    rc, cap = run_lanes(tmp_path, base_raw(), capsys)
    assert rc == 2 and "no `lanes`" in cap.err
    rc, cap = run_lanes(tmp_path, lanes_raw([lane_obj("l1", ["A"])]), capsys)
    assert rc == 2 and "in no lane" in cap.err
    assert psm.main(["lanes", "--plan", str(tmp_path / "nope.json")]) == 2


# --------------------------------------------------------------------------
# hold: kernel watcher + lane run
# --------------------------------------------------------------------------

FAKE_WATCH = r'''
import json, os, sys, time
n, trigger, mode = int(sys.argv[1]), sys.argv[2], sys.argv[3]
fdlog = sys.argv[4] if len(sys.argv) > 4 else None
if fdlog:
    fds = sorted(os.listdir("/proc/self/fd"))  # includes the directory fd of this very listing
    with open(fdlog, "w") as f:
        f.write(json.dumps(fds))
for i in range(n):
    print(f"kernel: harmless line {i}", flush=True)
if trigger != "-":
    t0 = time.time()
    while not os.path.exists(trigger) and time.time() - t0 < 60:
        time.sleep(0.05)
if mode == "fault":
    print("amdgpu 0000:03:00.0: amdgpu: ring gfx_0.0.0 timeout", flush=True)
elif mode == "custom":
    print("kernel: my-custom-fault happened", flush=True)
elif mode == "exit":
    sys.exit(0)
time.sleep(3600)
'''


@pytest.fixture
def fabric_env(monkeypatch):
    monkeypatch.setenv("FABRIC_JOB", "j4242")
    monkeypatch.setenv("FABRIC_CARDS", "A")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("HIPFIRE_DEVICES", "0")
    monkeypatch.setattr(psm, "WATCH_FLUSH_S", 0.05)
    monkeypatch.setattr(psm, "WATCH_SETTLE_S", 0.2)


@pytest.fixture
def watch_script(tmp_path):
    p = tmp_path / "fake_watch.py"
    p.write_text(FAKE_WATCH)
    return p


def hold_plan_dict(envs=None, deadlines=None, **over):
    envs = envs or {}
    deadlines = deadlines or {}
    loads = []
    for n in ("H1", "H2", "H3"):
        loads.append({
            "name": n,
            "model": "/nonexistent/model.hfq",
            "load_params": {"max_seq": 64},
            "env": envs.get(n, {}),
            "deadline_s": deadlines.get(n, 30),
            "cells": [{"name": n.lower() + "-ar", "generate": {"fake_seed": n}}],
        })
    plan = {
        "version": 1,
        "prompt": PROMPT,
        "max_tokens": 8,
        "loads": loads,
        "lanes": [lane_obj("lane1", ["H1", "H2", "H3"])],
    }
    plan.update(over)
    return plan


def hold_cli(plan, daemon, out, *extra, watch=None, watch_args=("3", "-", "none"), lane="lane1"):
    argv = ["hold", "--plan", str(plan), "--lane", lane, "--daemon", str(daemon), "--out", str(out), *extra]
    if watch is not None:
        argv += ["--watch-cmd", sys.executable, str(watch), *watch_args]
    return psm.main(argv)


def group_gone(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    return False


def verdict_of(tmp_path, plan, *dirs, name="v.json", extra=()):
    rc = psm.main(["verdict", "--results", *map(str, dirs), "--plan", str(plan), "--pins", str(tmp_path / "none.json"),
                   "--sha", "s", "--out", str(tmp_path / name), *extra])
    return rc, read(tmp_path / name)


def test_hold_complete_no_fault(tmp_path, fake_daemon, watch_script, fabric_env):
    plan = write_plan(tmp_path, hold_plan_dict())
    out = tmp_path / "hold"
    assert hold_cli(plan, fake_daemon, out, watch=watch_script, watch_args=("5", "-", "none")) == 0
    h = read(out / "hold.json")
    assert h["status"] == "complete" and h["reason"] is None and h["lane"] == "lane1" and h["fabric_job"] == "j4242"
    assert h["cards_env"] == {"FABRIC_CARDS": "A", "ROCR_VISIBLE_DEVICES": "0", "HIPFIRE_DEVICES": "0"}
    assert h["started_utc"].endswith("Z") and h["ended_utc"].endswith("Z")
    assert [(x["load"], x["status"]) for x in h["loads"]] == [("H1", "ok"), ("H2", "ok"), ("H3", "ok")]
    w = h["watcher"]
    assert w["cmd"][0] == sys.executable and w["lines"] == 5 and w["hits"] == [] and w["gone"] is True
    assert w["pid"] == w["pgid"]  # own session: PGID == PID
    assert group_gone(w["pgid"])  # the watcher's process group is really gone after hold returns
    log = (out / "kernel.log").read_text().splitlines()
    assert log == [f"kernel: harmless line {i}" for i in range(5)]
    assert not (out / "STOP").exists()
    run = read(out / "run.json")
    assert run["ok"] is True and run["lane"] == "lane1" and sorted(run["selected"]) == ["H1", "H2", "H3"]
    assert run["changed_paths"] is None and run["diff_sha256"] is None
    for n in ("H1", "H2", "H3"):
        assert read(out / f"{n}.json")["exit_code"] == 0
    rc, v = verdict_of(tmp_path, plan, out)
    assert rc == 0 and v["overall"] == "PASS" and v["incomplete"] == []


def test_hold_records_changed_paths_and_skips_unselected_loads(tmp_path, fake_daemon, watch_script, fabric_env):
    pd = hold_plan_dict()
    pd["loads"][1]["enabled"] = False
    pd["loads"][1]["run_if_paths"] = [r"h2\.rs$"]
    plan = write_plan(tmp_path, pd)
    cp = tmp_path / "changed.txt"
    cp.write_text("src/other.rs\n")
    out = tmp_path / "hold"
    assert hold_cli(plan, fake_daemon, out, "--changed-paths", str(cp), watch=watch_script) == 0
    h = read(out / "hold.json")
    assert h["skipped_loads"] == ["H2"] and [x["load"] for x in h["loads"]] == ["H1", "H3"]
    assert not (out / "H2.json").exists()
    run = read(out / "run.json")
    assert run["changed_paths"] == ["src/other.rs"] and sorted(run["selected"]) == ["H1", "H3"] and run["only"] == ["H1", "H3"]
    rc, v = verdict_of(tmp_path, plan, out, extra=["--partial"])
    assert rc == 0
    rc, v = verdict_of(tmp_path, plan, out)  # H2 recomputed from run.json inputs -> SKIPPED, not "not run"
    assert rc == 0 and v["cells"]["h2-ar"]["status"] == "SKIPPED"


def test_hold_fault_during_second_load_stops_the_lane(tmp_path, fake_daemon, watch_script, fabric_env):
    ready, gc, marker = tmp_path / "h2.ready", tmp_path / "h2.gc", tmp_path / "h3.marker"
    pd = hold_plan_dict(envs={
        "H2": {"FAKE_MODE": "hang", "FAKE_READY": str(ready), "FAKE_GRANDCHILD": str(gc)},
        "H3": {"FAKE_MARKER": str(marker)},
    })
    plan = write_plan(tmp_path, pd)
    out = tmp_path / "hold"
    t = time.monotonic()
    rc = hold_cli(plan, fake_daemon, out, watch=watch_script, watch_args=("3", str(ready), "fault"))
    assert rc == 4 and time.monotonic() - t < 30
    stop = (out / "STOP").read_text()
    assert "ring gfx_0.0.0 timeout" in stop
    h = read(out / "hold.json")
    assert h["status"] == "incomplete" and h["reason"].startswith("kernel fault: ") and "ring gfx_0.0.0 timeout" in h["reason"]
    assert [(x["load"], x["status"]) for x in h["loads"]] == [("H1", "ok"), ("H2", "aborted"), ("H3", "not_run")]
    assert h["watcher"]["gone"] is True and group_gone(h["watcher"]["pgid"])
    assert len(h["watcher"]["hits"]) == 1 and h["watcher"]["hits"][0]["line"].endswith("ring gfx_0.0.0 timeout")
    assert h["watcher"]["lines"] == 4
    assert "ring gfx_0.0.0 timeout" in (out / "kernel.log").read_text()
    assert not marker.exists()  # H3 never started
    assert not (out / "H3.json").exists()
    # the daemon of H2 and its grandchild are gone (group killed)
    for pidfile in (ready, gc):
        pid = int(pidfile.read_text())
        for _ in range(50):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            pytest.fail(f"{pidfile.name} pid {pid} survived the abort")
    h2 = read(out / "H2.json")
    assert h2["aborted"] is True and h2["killed_on_deadline"] is False and any("aborted" in e["message"] for e in h2["errors"])
    run = read(out / "run.json")
    assert run["ok"] is False and run["loads"]["H3"]["not_run"] is True and run["selected"]["H3"] == ["h3-ar"]

    rc, v = verdict_of(tmp_path, plan, out)
    assert rc == 4 and v["overall"] == "INCOMPLETE" and v["incomplete"] == ["lane1"]
    assert all(c["status"] == "FAIL" and any(r.startswith("INCOMPLETE: kernel fault") for r in c["reasons"]) for c in v["cells"].values())
    assert set(v["cells"]) == {"h1-ar", "h2-ar", "h3-ar"}  # even H1 (finished before the fault) is FAIL
    pins = tmp_path / "pins.json"
    for extra in ((), ("--allow-failed-cells", "because")):
        assert psm.main(["pin", "--verdict", str(tmp_path / "v.json"), "--pins", str(pins), *extra]) == 1
    assert not pins.exists()


@pytest.mark.parametrize("when", ["early", "mid-run"])
def test_hold_watcher_dies_is_incomplete(tmp_path, fake_daemon, watch_script, fabric_env, when):
    ready, marker = tmp_path / "h2.ready", tmp_path / "h3.marker"
    envs = {"H3": {"FAKE_MARKER": str(marker)}}
    if when == "mid-run":
        envs["H2"] = {"FAKE_MODE": "hang", "FAKE_READY": str(ready)}
    plan = write_plan(tmp_path, hold_plan_dict(envs=envs))
    out = tmp_path / "hold"
    trigger = "-" if when == "early" else str(ready)
    assert hold_cli(plan, fake_daemon, out, watch=watch_script, watch_args=("1", trigger, "exit")) == 4
    h = read(out / "hold.json")
    assert h["status"] == "incomplete" and h["reason"].startswith("watcher died")
    assert (out / "STOP").read_text().startswith("watcher died")
    assert h["watcher"]["gone"] is True and group_gone(h["watcher"]["pgid"]) and h["watcher"]["hits"] == []
    assert not marker.exists()
    st = {x["load"]: x["status"] for x in h["loads"]}
    assert st["H3"] == "not_run" and (when == "early" or st["H2"] == "aborted")
    rc, v = verdict_of(tmp_path, plan, out)
    assert rc == 4 and v["overall"] == "INCOMPLETE"


def test_hold_requires_fabric_job(tmp_path, fake_daemon, watch_script, monkeypatch, capsys):
    monkeypatch.delenv("FABRIC_JOB", raising=False)
    monkeypatch.setattr(psm, "WATCH_FLUSH_S", 0.05)
    monkeypatch.setattr(psm, "WATCH_SETTLE_S", 0.1)
    plan = write_plan(tmp_path, hold_plan_dict())
    out = tmp_path / "hold"
    assert hold_cli(plan, fake_daemon, out, watch=watch_script) == 2
    assert "FABRIC_JOB" in capsys.readouterr().err
    assert not out.exists()
    assert hold_cli(plan, fake_daemon, out, "--no-fabric-check", watch=watch_script) == 0
    assert read(out / "hold.json")["fabric_job"] is None and read(out / "hold.json")["status"] == "complete"
    # a dir that already holds a hold.json is refused
    assert hold_cli(plan, fake_daemon, out, "--no-fabric-check", watch=watch_script) == 2


def test_hold_usage_errors(tmp_path, fake_daemon, watch_script, fabric_env, capsys):
    plan = write_plan(tmp_path, hold_plan_dict())
    assert hold_cli(plan, fake_daemon, tmp_path / "o1", watch=watch_script, lane="nope") == 2
    assert "unknown lane 'nope'" in capsys.readouterr().err
    nolanes = write_plan(tmp_path, base_raw(), "nolanes.json")
    assert hold_cli(nolanes, fake_daemon, tmp_path / "o2", watch=watch_script) == 2
    assert hold_cli(plan, tmp_path / "missing-daemon", tmp_path / "o3", watch=watch_script) == 2
    assert hold_cli(plan, fake_daemon, tmp_path / "o4", "--home-models", str(tmp_path / "nodir"), watch=watch_script) == 2
    assert psm.main(["hold", "--plan", str(plan), "--lane", "lane1", "--daemon", str(fake_daemon), "--out", str(tmp_path / "o5"),
                     "--watch-cmd"]) == 2
    assert not any((tmp_path / d).exists() for d in ("o1", "o2", "o3", "o4", "o5"))


def test_hold_home_models_symlink_and_env(tmp_path, fake_daemon, watch_script, fabric_env, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    (models / "x.bin").write_bytes(b"1")
    envlog = tmp_path / "env.log"
    envs = {n: {"FAKE_ENVLOG": str(envlog)} for n in ("H1", "H2", "H3")}
    plan = write_plan(tmp_path, hold_plan_dict(envs=envs))
    out = tmp_path / "hold"
    assert hold_cli(plan, fake_daemon, out, "--home-models", str(models), watch=watch_script) == 0
    seen = [json.loads(ln) for ln in envlog.read_text().splitlines()]
    assert len(seen) == 3
    for n, e in zip(("H1", "H2", "H3"), seen):
        home = out / n / "home"
        assert e["HOME"] == str(home) and e["HIPFIRE_NO_REGISTRY_FETCH"] == "1" and e["HIPFIRE_EMIT_TOKEN_IDS"] == "1"
        link = home / ".hipfire" / "models"
        assert link.is_symlink() and os.readlink(link) == str(models)
        assert (link / "x.bin").read_bytes() == b"1"
        assert sorted(p.name for p in home.iterdir()) == [".hipfire"]  # models symlink only
        assert read(out / f"{n}.json")["env_extra"]["HOME"] == str(home)
    # without --home-models the daemon inherits HOME but still gets the registry/emit env
    envlog.unlink()
    monkeypatch.setenv("HOME", str(tmp_path / "realhome"))
    out2 = tmp_path / "hold2"
    assert hold_cli(plan, fake_daemon, out2, watch=watch_script) == 0
    seen = [json.loads(ln) for ln in envlog.read_text().splitlines()]
    assert all(e["HOME"] == str(tmp_path / "realhome") and e["HIPFIRE_NO_REGISTRY_FETCH"] == "1" for e in seen)
    assert not (out2 / "H1").exists()


def test_hold_watcher_inherits_no_fds_above_2(tmp_path, fake_daemon, watch_script, fabric_env):
    r, w = os.pipe()  # inheritable: would leak into the watcher without close_fds=True
    os.set_inheritable(r, True)
    os.set_inheritable(w, True)
    try:
        fdlog = tmp_path / "fds.json"
        plan = write_plan(tmp_path, hold_plan_dict())
        out = tmp_path / "hold"
        assert hold_cli(plan, fake_daemon, out, watch=watch_script, watch_args=("1", "-", "none", str(fdlog))) == 0
        fds = json.loads(fdlog.read_text())
        assert {"0", "1", "2"} <= set(fds)
        extra = [f for f in fds if int(f) > 2]
        assert len(extra) <= 1, fds  # only the fd of the /proc/self/fd listing itself
        assert str(r) not in extra and str(w) not in extra
    finally:
        os.close(r)
        os.close(w)


def test_hold_custom_kernel_fault_pattern_from_plan(tmp_path, fake_daemon, watch_script, fabric_env):
    ready = tmp_path / "h2.ready"
    pd = hold_plan_dict(envs={"H2": {"FAKE_MODE": "hang", "FAKE_READY": str(ready)}}, kernel_fault_patterns=["MY-CUSTOM-FAULT"])
    plan = write_plan(tmp_path, pd)
    out = tmp_path / "hold"
    assert hold_cli(plan, fake_daemon, out, watch=watch_script, watch_args=("1", str(ready), "custom")) == 4
    h = read(out / "hold.json")
    assert h["status"] == "incomplete" and "my-custom-fault" in h["reason"]
    assert h["watcher"]["hits"][0]["pattern"] == "MY-CUSTOM-FAULT"


def test_hold_default_patterns_ignore_harmless_lines(tmp_path, fake_daemon, watch_script, fabric_env):
    # a 'custom' fault line does not match the default list: the hold completes
    plan = write_plan(tmp_path, hold_plan_dict())
    out = tmp_path / "hold"
    assert hold_cli(plan, fake_daemon, out, watch=watch_script, watch_args=("1", "-", "custom")) == 0
    assert read(out / "hold.json")["status"] == "complete" and not (out / "STOP").exists()


def test_hold_load_killed_on_deadline_is_status_deadline_exit_1(tmp_path, fake_daemon, watch_script, fabric_env):
    pd = hold_plan_dict(envs={"H2": {"FAKE_MODE": "hang"}}, deadlines={"H2": 1})
    plan = write_plan(tmp_path, pd)
    out = tmp_path / "hold"
    assert hold_cli(plan, fake_daemon, out, watch=watch_script) == 1
    h = read(out / "hold.json")
    assert h["status"] == "deadline" and "H2" in h["reason"]
    assert {x["load"]: x["status"] for x in h["loads"]} == {"H1": "ok", "H2": "killed_on_deadline", "H3": "ok"}
    assert h["watcher"]["gone"] is True
    rc, v = verdict_of(tmp_path, plan, out)  # a deadline is a plain FAIL, not INCOMPLETE
    assert rc == 1 and v["overall"] == "FAIL" and v["incomplete"] == []


def test_hold_unresolved_model_is_status_deadline_and_never_spawns(tmp_path, fake_daemon, watch_script, fabric_env, monkeypatch):
    monkeypatch.delenv("SMOKE_UNSET_XYZ", raising=False)
    marker = tmp_path / "m"
    pd = hold_plan_dict(envs={"H1": {"FAKE_MARKER": str(marker)}})
    pd["loads"][0]["model"] = "$SMOKE_UNSET_XYZ/a.hfq"
    plan = write_plan(tmp_path, pd)
    out = tmp_path / "hold"
    assert hold_cli(plan, fake_daemon, out, watch=watch_script) == 1
    h = read(out / "hold.json")
    assert h["status"] == "deadline" and "unresolved model path" in h["reason"]
    assert {x["load"]: x["status"] for x in h["loads"]}["H1"] == "failed_before_spawn" and not marker.exists()


def test_hold_sigterm_tears_everything_down(tmp_path, fake_daemon, watch_script, fabric_env):
    import signal as _signal
    import threading

    ready = tmp_path / "h2.ready"
    plan = write_plan(tmp_path, hold_plan_dict(envs={"H2": {"FAKE_MODE": "hang", "FAKE_READY": str(ready)}}))
    out = tmp_path / "hold"
    prev = _signal.getsignal(_signal.SIGTERM)

    def poke():  # signals only this test process, whose hold handler is installed once H2 is running
        for _ in range(600):
            if ready.exists():
                break
            time.sleep(0.05)
        os.kill(os.getpid(), _signal.SIGTERM)

    th = threading.Thread(target=poke)
    th.start()
    try:
        rc = hold_cli(plan, fake_daemon, out, watch=watch_script)
    finally:
        th.join()
    assert rc == 4 and _signal.getsignal(_signal.SIGTERM) is prev  # handlers restored
    h = read(out / "hold.json")
    assert h["status"] == "incomplete" and h["reason"] == "signalled: SIGTERM"
    assert h["watcher"]["gone"] is True and group_gone(h["watcher"]["pgid"])
    assert {x["load"]: x["status"] for x in h["loads"]} == {"H1": "ok", "H2": "aborted", "H3": "not_run"}


def test_hold_probe_failure_is_exit_2_with_incomplete_hold_json(tmp_path, fake_daemon, fabric_env, monkeypatch, capsys):
    marker = tmp_path / "m"
    plan = write_plan(tmp_path, hold_plan_dict(envs={"H1": {"FAKE_MARKER": str(marker)}}))
    monkeypatch.setattr(psm, "WATCH_PROBE_CMD", [sys.executable, "-c", "import sys; print('Hint: insufficient permissions'); sys.exit(0)"])
    out = tmp_path / "hold"
    assert hold_cli(plan, fake_daemon, out) == 2
    assert "watcher unavailable" in capsys.readouterr().err
    h = read(out / "hold.json")
    assert h["status"] == "incomplete" and h["reason"].startswith("watcher unavailable") and h["watcher"]["pid"] is None
    assert not marker.exists() and (out / "run.json").exists() and not (out / "H1.json").exists()
    rc, v = verdict_of(tmp_path, plan, out)
    assert rc == 4 and v["overall"] == "INCOMPLETE"


def test_hold_default_watch_cmd_after_probe(tmp_path, fake_daemon, watch_script, fabric_env, monkeypatch):
    monkeypatch.setattr(psm, "WATCH_PROBE_CMD", [sys.executable, "-c", "pass"])
    monkeypatch.setattr(psm, "DEFAULT_WATCH_CMD", [sys.executable, str(watch_script), "2", "-", "none"])
    plan = write_plan(tmp_path, hold_plan_dict())
    out = tmp_path / "hold"
    assert hold_cli(plan, fake_daemon, out) == 0
    assert read(out / "hold.json")["watcher"]["lines"] == 2 and read(out / "hold.json")["watcher"]["gone"] is True


def test_verdict_hold_json_running_or_missing_status_is_incomplete(tmp_path, fake_daemon):
    plan = write_plan(tmp_path, e2e_plan())
    out = tmp_path / "out"
    assert run_cli(tmp_path, plan, fake_daemon, out) == 0
    assert verdict_of(tmp_path, plan, out)[0] == 0
    (out / "hold.json").write_text(json.dumps({"lane": "x", "status": "running"}))
    rc, v = verdict_of(tmp_path, plan, out)
    assert rc == 4 and v["incomplete"] == ["x"] and v["overall"] == "INCOMPLETE"
    assert any("hold status 'running'" in r for r in v["cells"]["ar"]["reasons"])
    (out / "hold.json").write_text(json.dumps({"lane": "x", "status": "complete"}))
    assert verdict_of(tmp_path, plan, out)[0] == 0
    (out / "hold.json").write_text("not json")
    assert verdict_of(tmp_path, plan, out)[0] == 2


def test_build_verdict_incomplete_beats_fail_and_only_marks_its_loads():
    plan = mk_plan()
    res = results_for(plan, **{"a3b-ar": cell_rec([9, 9, 9, 9, 9, 9, 9, 9])})
    res["a3b"]["errors"] = [{"message": "boom"}]  # an ordinary FAIL elsewhere
    v = psm.build_verdict(plan, res, {}, "s", incomplete=[{"lane": "hiptrx-a", "reason": "kernel fault: x", "loads": ["fn1"]}])
    assert v["overall"] == "INCOMPLETE" and v["incomplete"] == ["hiptrx-a"]
    assert v["cells"]["fn1-ar"]["status"] == "FAIL" and v["cells"]["fn1-ar"]["reasons"] == ["INCOMPLETE: kernel fault: x"]
    assert v["cells"]["fn1-mtp"]["reasons"] == ["INCOMPLETE: kernel fault: x"]
    assert v["cells"]["a3b-ar"]["status"] == "FAIL" and "INCOMPLETE" not in v["cells"]["a3b-ar"]["reasons"][0]
    assert "INCOMPLETE" in psm.format_table(v) and "incomplete lanes: hiptrx-a" in psm.format_table(v)
    assert psm.build_verdict(plan, results_for(plan), {}, "s")["incomplete"] == []


# --------------------------------------------------------------------------
# pin --allow-failed-cells
# --------------------------------------------------------------------------


def test_pin_allow_failed_cells_pins_new_and_match_only():
    v = verdict_with(a="NEW", b="FAIL", c="MATCH", d="SKIPPED")
    assert v["overall"] == "FAIL"
    with pytest.raises(psm.PinRefused):
        psm.apply_pins({}, v)
    old = {"version": 1, "cells": {"b": {"ids": [7], "ids_md5": psm.ids_md5([7]), "reason": "old"}}, "history": []}
    new = psm.apply_pins(old, v, allow_failed="b is a known flaky cell", now="T1")
    assert sorted(new["cells"]) == ["a", "b", "c"] and new["cells"]["b"]["ids"] == [7]  # FAIL never overwrites
    assert new["cells"]["a"]["reason"] == "new" and new["cells"]["c"]["reason"] == "match" and "d" not in new["cells"]
    hist = new["history"][-1]
    assert hist["overall"] == "FAIL" and hist["allow_failed_cells"] == "b is a known flaky cell"
    assert hist["failed_cells"] == ["b"] and hist["changed"] == []
    assert old["history"] == []  # input untouched


def test_pin_allow_failed_cells_changed_still_needs_accept_changed():
    v = verdict_with(a="NEW", b="FAIL", c="CHANGED")
    with pytest.raises(psm.PinRefused, match="CHANGED"):
        psm.apply_pins({}, v, allow_failed="why")
    new = psm.apply_pins({}, v, "retune", allow_failed="why", now="T1")
    assert sorted(new["cells"]) == ["a", "c"] and new["cells"]["c"]["reason"] == "retune"
    assert new["history"][-1]["changed"] == ["c"] and new["history"][-1]["failed_cells"] == ["b"]


def test_pin_allow_failed_cells_refuses_incomplete_empty_reason_and_failed_invariants():
    inc = {"sha": "s", "overall": "INCOMPLETE", "cells": {"a": {"status": "FAIL", "ids": [], "ids_md5": "x", "reasons": []}}, "invariants": []}
    for kw in ({}, {"allow_failed": "why"}, {"accept_changed": "r", "allow_failed": "why"}):
        with pytest.raises(psm.PinRefused, match="INCOMPLETE"):
            psm.apply_pins({}, inc, **kw)
    with pytest.raises(psm.PinRefused):
        psm.apply_pins({}, verdict_with(a="NEW", b="FAIL"), allow_failed="  ")
    v = verdict_with(a="NEW", b="NEW", c="MATCH")
    v["invariants"] = [{"cells": ["a", "b"], "status": "INVARIANT"}]
    v["overall"] = "FAIL"
    new = psm.apply_pins({}, v, allow_failed="why")
    assert sorted(new["cells"]) == ["c"] and new["history"][-1]["failed_cells"] == ["a", "b"]
    # on a PASS verdict the flag changes nothing
    ok = psm.apply_pins({}, verdict_with(a="NEW"), allow_failed="why")
    assert "allow_failed_cells" not in ok["history"][-1] and sorted(ok["cells"]) == ["a"]


def test_pin_cli_allow_failed_cells_roundtrip(tmp_path, capsys):
    vp, pp = tmp_path / "v.json", tmp_path / "pins.json"
    vp.write_text(json.dumps(verdict_with(a="NEW", b="FAIL", c="CHANGED")))
    assert psm.main(["pin", "--verdict", str(vp), "--pins", str(pp), "--allow-failed-cells", "known b"]) == 1  # CHANGED, no reason
    assert "CHANGED" in capsys.readouterr().err and not pp.exists()
    assert psm.main(["pin", "--verdict", str(vp), "--pins", str(pp)]) == 1  # FAIL without the flag
    vp.write_text(json.dumps(verdict_with(a="NEW", b="FAIL", c="MATCH")))
    assert psm.main(["pin", "--verdict", str(vp), "--pins", str(pp), "--allow-failed-cells", "known b"]) == 0
    pins = read(pp)
    assert sorted(pins["cells"]) == ["a", "c"] and pins["history"][-1]["allow_failed_cells"] == "known b"
    assert pins["history"][-1]["failed_cells"] == ["b"]
    assert psm.main(["pin", "--verdict", str(vp), "--pins", str(pp), "--allow-failed-cells", " "]) == 1


@pytest.mark.parametrize("sub", ["hold", "lanes"])
def test_new_subcommand_help_works(sub, capsys):
    with pytest.raises(SystemExit) as ei:
        psm.main([sub, "--help"])
    assert ei.value.code == 0
    assert "--plan" in capsys.readouterr().out


# --------------------------------------------------------------------------
# group-leader reap order: never signal a process group after its leader was reaped
# --------------------------------------------------------------------------


@pytest.fixture
def reap_events(monkeypatch):
    """Ordered record of os.killpg(pgid, sig) calls and of every successful os.waitpid reap."""
    events: list[tuple] = []
    real_killpg, real_waitpid = os.killpg, os.waitpid

    def killpg(pgid, sig):
        events.append(("killpg", pgid, int(sig)))
        return real_killpg(pgid, sig)

    def waitpid(pid, opts):
        res = real_waitpid(pid, opts)
        if res[0] == pid:
            events.append(("reap", pid))
        return res

    monkeypatch.setattr(os, "killpg", killpg)
    monkeypatch.setattr(os, "waitpid", waitpid)
    return events


def assert_signals_precede_reap(events, leaders=None):
    """No real group signal (sig != 0) to a pgid after its leader was reaped; each reaped leader got a
    final group signal first (so the check is not vacuous). `leaders`: pids that must appear reaped."""
    reaped = []
    signalled_before = set()
    for ev in events:
        if ev[0] == "reap":
            assert ev[1] in signalled_before, f"leader {ev[1]} reaped without a prior group signal: {events}"
            reaped.append(ev[1])
        elif ev[2] != 0:
            assert ev[1] not in reaped, f"killpg({ev[1]}, {ev[2]}) after its leader was reaped: {events}"
            signalled_before.add(ev[1])
    for pid in leaders or ():
        assert pid in reaped, f"leader {pid} was never reaped: {events}"
    assert reaped, events


def wait_gone(pid, what):
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    pytest.fail(what)


def test_reap_order_normal_end_kills_orphan_grandchild(tmp_path, fake_daemon, reap_events):
    gc = tmp_path / "grandchild.pid"
    plan = write_plan(tmp_path, e2e_plan(env={"FAKE_MODE": "orphan_grandchild", "FAKE_GRANDCHILD": str(gc)}))
    out = tmp_path / "out"
    assert run_cli(tmp_path, plan, fake_daemon, out) == 0
    res = read(out / "L1.json")
    assert res["exit_code"] == 0 and res["killed_on_deadline"] is False and res["aborted"] is False
    wait_gone(int(gc.read_text()), "grandchild survived the normal-end group kill")
    assert_signals_precede_reap(reap_events)
    kills = [e for e in reap_events if e[0] == "killpg" and e[2] == int(psm.signal.SIGKILL)]
    assert len(kills) == 1  # exactly the final group signal of the one daemon leader
    assert reap_events.index(kills[0]) < reap_events.index(("reap", kills[0][1]))


def test_reap_order_deadline_kill(tmp_path, fake_daemon, reap_events, monkeypatch):
    monkeypatch.setattr(psm, "TERM_GRACE_S", 0.3)
    gc = tmp_path / "grandchild.pid"
    plan = write_plan(tmp_path, e2e_plan(env={"FAKE_MODE": "ignore_term", "FAKE_GRANDCHILD": str(gc)}, deadline=1))
    out = tmp_path / "out"
    assert run_cli(tmp_path, plan, fake_daemon, out) == 1
    res = read(out / "L1.json")
    assert res["killed_on_deadline"] is True and res["exit_code"] == -9
    wait_gone(int(gc.read_text()), "grandchild survived the deadline group kill")
    assert_signals_precede_reap(reap_events)
    sigs = [e[2] for e in reap_events if e[0] == "killpg"]
    assert sigs[:2] == [int(psm.signal.SIGTERM), int(psm.signal.SIGKILL)]


def test_reap_order_abort_path(tmp_path, fake_daemon, reap_events):
    ready, gc = tmp_path / "ready", tmp_path / "grandchild.pid"
    path = write_plan(tmp_path, e2e_plan(env={"FAKE_MODE": "hang", "FAKE_READY": str(ready),
                                              "FAKE_GRANDCHILD": str(gc)}, deadline=30))
    plan, _ = psm.load_plan(path)
    out = tmp_path / "out"
    out.mkdir()
    abort = psm.AbortFlag()

    def fire():
        for _ in range(300):
            if ready.exists() and gc.exists():
                break
            time.sleep(0.05)
        abort.set("test abort")

    th = threading.Thread(target=fire)
    th.start()
    res = psm.run_load(plan["loads"][0], plan, fake_daemon, out, abort=abort)
    th.join()
    assert res["aborted"] is True and res["killed_on_deadline"] is False
    wait_gone(int(gc.read_text()), "grandchild survived the abort group kill")
    assert_signals_precede_reap(reap_events)


def test_reap_order_watcher_stop_and_daemons(tmp_path, fake_daemon, watch_script, fabric_env, reap_events):
    plan = write_plan(tmp_path, hold_plan_dict())
    out = tmp_path / "hold"
    assert hold_cli(plan, fake_daemon, out, watch=watch_script, watch_args=("3", "-", "none")) == 0
    w = read(out / "hold.json")["watcher"]
    assert w["gone"] is True
    assert_signals_precede_reap(reap_events, leaders=[w["pid"]])
    wkills = [e for e in reap_events if e[0] == "killpg" and e[1] == w["pgid"] and e[2] != 0]
    assert [e[2] for e in wkills] == [int(psm.signal.SIGTERM), int(psm.signal.SIGKILL)]


def test_reap_order_watcher_died_then_stop(tmp_path, fake_daemon, watch_script, fabric_env, reap_events):
    # the watcher exits by itself: its reader thread must not reap the leader before stop() signals the group
    plan = write_plan(tmp_path, hold_plan_dict())
    out = tmp_path / "hold"
    assert hold_cli(plan, fake_daemon, out, watch=watch_script, watch_args=("1", "-", "exit")) == 4
    h = read(out / "hold.json")
    assert h["reason"].startswith("watcher died (exit 0)") and h["watcher"]["gone"] is True
    assert_signals_precede_reap(reap_events, leaders=[h["watcher"]["pid"]])


def test_wait_exit_unreaped_never_reaps():
    proc = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    try:
        assert psm._wait_exit_unreaped(proc, 10) is True
        assert proc.returncode is None  # still a zombie: not reaped
        assert psm._unreaped_rc(proc) == 0
        os.killpg(proc.pid, 0)  # the zombie leader still owns its group
    finally:
        proc.wait()
    sleeper = subprocess.Popen(["sleep", "1000"], start_new_session=True)
    try:
        assert psm._wait_exit_unreaped(sleeper, 0.1) is False
        os.killpg(sleeper.pid, psm.signal.SIGKILL)
        assert psm._wait_exit_unreaped(sleeper, 10) is True
        assert psm._unreaped_rc(sleeper) == -9 and sleeper.returncode is None
    finally:
        sleeper.wait()


# --------------------------------------------------------------------------
# lost group leader (ECHILD / reaped elsewhere): never signalled, never a fake pass
# --------------------------------------------------------------------------


def no_waitid(*a, **k):
    raise ChildProcessError(errno.ECHILD, "No child processes")


def killpg_calls(events):
    return [e for e in events if e[0] == "killpg"]


@pytest.fixture
def popen_log(monkeypatch):
    """Records every Popen; on teardown SIGKILLs (by exact pid, our own children) and reaps leftovers."""
    procs = []
    real = subprocess.Popen

    class Rec(real):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            procs.append(self)

    monkeypatch.setattr(subprocess, "Popen", Rec)
    yield procs
    for p in procs:
        if p.returncode is None:
            try:
                os.kill(p.pid, psm.signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait()


def test_lost_leader_kill_group_reaped_elsewhere(reap_events):
    proc = subprocess.Popen(["sleep", "1000"], start_new_session=True)
    os.kill(proc.pid, psm.signal.SIGKILL)
    os.waitpid(proc.pid, 0)  # someone else reaps the leader: real ECHILD for us
    reap_events.clear()
    assert psm.kill_group(proc) is False
    assert killpg_calls(reap_events) == []  # not even the initial TERM
    assert proc.returncode is None  # no fabricated status


def test_lost_leader_known_rc_is_kept_but_never_signalled(reap_events):
    proc = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    assert proc.wait() == 0  # reaped through Popen: the real rc is known
    reap_events.clear()
    with pytest.raises(psm.LeaderLost) as ei:
        psm._wait_exit_unreaped(proc, 1)
    assert ei.value.rc == 0
    with pytest.raises(psm.LeaderLost):
        psm._killpg_owned(proc, proc.pid, psm.signal.SIGKILL)
    assert psm.kill_group(proc) is False
    assert killpg_calls(reap_events) == []


@pytest.mark.parametrize("path", ["normal", "deadline", "abort"])
def test_lost_leader_run_load_never_signals_and_never_passes(tmp_path, fake_daemon, reap_events, popen_log, monkeypatch, path):
    ready = tmp_path / "ready"
    env = {"FAKE_READY": str(ready)}
    if path != "normal":
        env["FAKE_MODE"] = "hang"
    pp = write_plan(tmp_path, e2e_plan(env=env, deadline=2 if path == "deadline" else 30))
    plan, _ = psm.load_plan(pp)
    out = tmp_path / "out"
    out.mkdir()
    abort = psm.AbortFlag()
    if path == "abort":
        def fire():
            for _ in range(300):
                if ready.exists():
                    break
                time.sleep(0.05)
            abort.set("test abort")

        threading.Thread(target=fire, daemon=True).start()
    with monkeypatch.context() as m:
        m.setattr(os, "waitid", no_waitid)
        res = psm.run_load(plan["loads"][0], plan, fake_daemon, out, abort=abort)
    assert killpg_calls(reap_events) == []  # zero group signals, the initial TERM included
    assert res["ownership_lost"] is True and res["exit_code"] is None
    assert any(e["message"] == "process group ownership lost; not signalled" for e in res["errors"])
    assert abort.is_set()  # no further load may run
    assert res["killed_on_deadline"] is (path == "deadline") and res["aborted"] is (path == "abort")
    cell = psm.cell_status(res, next(iter(res["cells"].values())), 8, None)
    assert cell["status"] == "FAIL"
    assert any("ownership lost" in r for r in cell["reasons"])
    assert read(out / "L1.json")["ownership_lost"] is True


def test_lost_leader_watcher_stop_never_signals(tmp_path, watch_script, reap_events, popen_log):
    out = tmp_path / "w"
    out.mkdir()
    w = psm.KernelWatcher([sys.executable, str(watch_script), "1", "-", "none"], out, ["fault"], psm.AbortFlag())
    w.start()
    w._stopping = True  # the reader thread must not treat the coming EOF as a death
    os.kill(w.pid, psm.signal.SIGKILL)
    os.waitpid(w.pid, 0)  # reaped by someone else: real ECHILD
    reap_events.clear()
    assert w.stop() is False
    assert w.lost is True and w.gone is False
    assert killpg_calls(reap_events) == []  # no TERM, no KILL, no killpg(pgid, 0) alive-check


def test_lost_leader_watcher_reader_never_signals(tmp_path, watch_script, reap_events, popen_log, monkeypatch):
    out = tmp_path / "w"
    out.mkdir()
    abort = psm.AbortFlag()
    monkeypatch.setattr(os, "waitid", no_waitid)
    w = psm.KernelWatcher([sys.executable, str(watch_script), "1", "-", "exit"], out, ["fault"], abort)
    w.start()
    assert abort._ev.wait(10)
    assert "ownership lost" in abort.reason and w.lost is True
    assert (out / "STOP").read_text().startswith("watcher died (process group ownership lost")
    assert w.stop() is False and w.gone is False
    assert killpg_calls(reap_events) == []


def test_lost_leader_hold_watcher_stop_is_incomplete(tmp_path, fake_daemon, watch_script, fabric_env, reap_events,
                                                     popen_log, monkeypatch):
    lost_pids = set()
    real_start, real_waitid = psm.KernelWatcher.start, os.waitid

    def start(self):
        real_start(self)
        lost_pids.add(self.pid)

    def waitid(idtype, pid, opts):
        if pid in lost_pids:
            raise ChildProcessError(errno.ECHILD, "No child processes")
        return real_waitid(idtype, pid, opts)

    monkeypatch.setattr(psm.KernelWatcher, "start", start)
    monkeypatch.setattr(os, "waitid", waitid)
    plan = write_plan(tmp_path, hold_plan_dict())
    out = tmp_path / "hold"
    rc = hold_cli(plan, fake_daemon, out, watch=watch_script, watch_args=("3", "-", "none"))
    h = read(out / "hold.json")
    assert rc == 4 and h["status"] == "incomplete" and "ownership lost" in h["reason"]
    assert h["watcher"]["ownership_lost"] is True and h["watcher"]["gone"] is False
    assert not [e for e in killpg_calls(reap_events) if e[1] == h["watcher"]["pgid"]]
