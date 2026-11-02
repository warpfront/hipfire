#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""Flash-Next PLE row sidecars (the `ple` map: `q4`, `q8`, any format token).

A split base publishes its PLE rows as `<file>.ple-<key>` (or, for the
Flash-Next family only, the shared `qwen3.8-flash-next.ple-<key>`). The curated
overlay names the sidecars in the `ple` map; `registry_gen.py` pins each against
the live Hugging Face tree (LFS oid + size), exactly like the other sidecar
kinds. Entries without a `ple` map come out unchanged.

No network: `repo_tree` is stubbed.
"""
import copy
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "registry_gen", Path(__file__).parent / "registry_gen.py"
)
rg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rg)

REPO = "test/split-flash"
BASE = "qwen3.8-flash-next-base.mq3-xts"
Q4 = f"{BASE}.ple-q4"
Q8 = f"{BASE}.ple-q8"
BASE_BYTES = 60_000_000_000
Q4_BYTES = 28_800_000_000
Q8_BYTES = 54_400_000_000


def _tree() -> dict:
    return {
        BASE: {"lfs": {"oid": "base-oid", "size": BASE_BYTES}, "size": BASE_BYTES},
        Q4: {"lfs": {"oid": "q4-oid", "size": Q4_BYTES}, "size": Q4_BYTES},
        Q8: {"lfs": {"oid": "q8-oid", "size": Q8_BYTES}, "size": Q8_BYTES},
    }


def _curated() -> dict:
    entry = {
        "repo": REPO,
        "file": BASE,
        "size_gb": 60.0,
        "min_vram_gb": 8,
        "desc": "test split base",
        "ple": {"q4": {"file": Q4}, "q8": {"file": Q8}},
    }
    return {"models": {"qwen3.8:flash-next-mq3": entry}, "aliases": {}}


def _build(monkeypatch, curated: dict, tree: dict | None = None):
    monkeypatch.setattr(rg, "repo_tree", lambda repo, token: tree if tree is not None else _tree())
    return rg.build_registry(curated, None)


def test_ple_slots_are_pinned_from_the_live_tree(monkeypatch):
    registry, errors = _build(monkeypatch, _curated())
    assert errors == [], errors
    entry = registry["models"]["qwen3.8:flash-next-mq3"]
    assert entry["ple"]["q4"] == {"file": Q4, "sha256": "q4-oid", "size_bytes": Q4_BYTES}
    assert entry["ple"]["q8"] == {"file": Q8, "sha256": "q8-oid", "size_bytes": Q8_BYTES}
    assert entry["sha256"] == "base-oid"
    assert entry["quant"] == "mq3"
    assert entry["arch_id"] == 16


def test_one_slot_entry_carries_only_that_slot(monkeypatch):
    curated = _curated()
    del curated["models"]["qwen3.8:flash-next-mq3"]["ple"]["q8"]
    registry, errors = _build(monkeypatch, curated)
    assert errors == [], errors
    entry = registry["models"]["qwen3.8:flash-next-mq3"]
    assert list(entry["ple"]) == ["q4"]
    assert entry["ple"]["q4"]["sha256"] == "q4-oid"


def test_entries_without_a_ple_map_are_untouched(monkeypatch):
    curated = _curated()
    del curated["models"]["qwen3.8:flash-next-mq3"]["ple"]
    registry, errors = _build(monkeypatch, curated)
    assert errors == [], errors
    entry = registry["models"]["qwen3.8:flash-next-mq3"]
    assert not any(key.startswith("ple") for key in entry), entry


def test_missing_slot_file_fails_closed(monkeypatch):
    tree = _tree()
    del tree[Q4]
    registry, errors = _build(monkeypatch, _curated(), tree)
    assert registry is None
    assert any("ple.q4" in e and "not found" in e for e in errors), errors


def test_non_lfs_slot_has_no_digest_pin_and_fails_closed(monkeypatch):
    tree = _tree()
    tree[Q8] = {"size": Q8_BYTES}
    registry, errors = _build(monkeypatch, _curated(), tree)
    assert registry is None
    assert any("ple.q8" in e and "sha256" in e for e in errors), errors


def test_build_does_not_mutate_the_curated_overlay(monkeypatch):
    curated = _curated()
    before = copy.deepcopy(curated)
    _build(monkeypatch, curated)
    assert curated == before


def test_unknown_format_keys_are_pinned_like_any_other(monkeypatch):
    q2 = f"{BASE}.ple-q2"
    tree = _tree()
    tree[q2] = {"lfs": {"oid": "q2-oid", "size": 14_000_000_000}, "size": 14_000_000_000}
    curated = _curated()
    curated["models"]["qwen3.8:flash-next-mq3"]["ple"]["q2"] = {"file": q2}
    registry, errors = _build(monkeypatch, curated, tree)
    assert errors == [], errors
    ple = registry["models"]["qwen3.8:flash-next-mq3"]["ple"]
    assert list(ple) == ["q2", "q4", "q8"] or sorted(ple) == ["q2", "q4", "q8"]
    assert ple["q2"] == {"file": q2, "sha256": "q2-oid", "size_bytes": 14_000_000_000}


def test_invalid_keys_and_misplaced_files_fail_closed(monkeypatch):
    for key in ("Q4", "q-4", "q_4", ""):
        curated = _curated()
        curated["models"]["qwen3.8:flash-next-mq3"]["ple"][key] = {"file": f"{BASE}.ple-{key}"}
        registry, errors = _build(monkeypatch, curated)
        assert registry is None
        assert any("invalid PLE format key" in e for e in errors), (key, errors)

    # The sidecar must be `<file>.ple-<key>` (or the Flash-Next shared name).
    curated = _curated()
    curated["models"]["qwen3.8:flash-next-mq3"]["ple"]["q4"] = {"file": Q8}
    registry, errors = _build(monkeypatch, curated)
    assert registry is None
    assert any("must be" in e for e in errors), errors

    # The entry's own file must be a split base.
    curated = _curated()
    curated["models"]["qwen3.8:flash-next-mq3"]["file"] = "plain.mq3-xts"
    registry, errors = _build(monkeypatch, curated)
    assert registry is None
    assert any("split base" in e for e in errors), errors

    curated = _curated()
    curated["models"]["qwen3.8:flash-next-mq3"]["ple"] = ["q4"]
    registry, errors = _build(monkeypatch, curated)
    assert registry is None
    assert any("ple must be an object" in e for e in errors), errors


def test_shared_flash_next_files_are_accepted_and_pinned_identically(monkeypatch):
    shared_q4, shared_q8 = "qwen3.8-flash-next.ple-q4", "qwen3.8-flash-next.ple-q8"
    tree = _tree()
    tree[shared_q4] = {"lfs": {"oid": "sq4", "size": 1}, "size": 1}
    tree[shared_q8] = {"lfs": {"oid": "sq8", "size": 2}, "size": 2}
    curated = _curated()
    entry = curated["models"]["qwen3.8:flash-next-mq3"]
    entry["ple"] = {"q4": {"file": shared_q4}, "q8": {"file": shared_q8}}
    registry, errors = _build(monkeypatch, curated, tree)
    assert errors == [], errors
    assert registry["models"]["qwen3.8:flash-next-mq3"]["ple"]["q4"]["sha256"] == "sq4"

    # Another family cannot borrow the shared pair.
    other = copy.deepcopy(curated)
    other["models"]["qwen3.8:flash-next-mq3"]["file"] = "other-model-base.mq3-xts"
    registry, errors = _build(monkeypatch, other, tree)
    assert registry is None
    assert any("must be" in e for e in errors), errors


def test_shared_pins_must_agree_across_entries():
    models = {
        "a": {"ple": {"q4": {"file": "qwen3.8-flash-next.ple-q4", "sha256": "x", "size_bytes": 1}}},
        "b": {"ple": {"q4": {"file": "qwen3.8-flash-next.ple-q4", "sha256": "y", "size_bytes": 1}}},
    }
    errors = []
    rg.validate_shared_ple_pins(models, errors)
    assert any("pinned differently" in e for e in errors), errors
    models["b"]["ple"]["q4"]["sha256"] = "x"
    errors = []
    rg.validate_shared_ple_pins(models, errors)
    assert errors == []


def test_committed_registry_publishes_ple_sidecars_only_through_the_map():
    import json

    root = Path(__file__).resolve().parent.parent / "registry"
    for name in ("models.json", "v1.json"):
        models = json.loads((root / name).read_text())["models"]
        for tag, entry in models.items():
            assert "ple_q4" not in entry and "ple_q8" not in entry, (name, tag)
        mq3 = models["qwen3.8:flash-next-mq3-xts"]
        assert sorted(mq3["ple"]) == ["q4", "q8"], name
        for key, sidecar in mq3["ple"].items():
            assert sidecar["file"] == f"qwen3.8-flash-next.ple-{key}"
    v1 = json.loads((root / "v1.json").read_text())["models"]
    for sidecar in v1["qwen3.8:flash-next-mq3-xts"]["ple"].values():
        assert len(sidecar["sha256"]) == 64 and sidecar["size_bytes"] > 0
