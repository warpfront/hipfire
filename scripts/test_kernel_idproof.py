#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""kernel_idproof.py's pure classifier (the build itself needs hipcc)."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "kernel_idproof", Path(__file__).parent / "kernel_idproof.py"
)
kid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kid)

BASE = {
    "gemv.hsaco": "a1", "gemv.hash": "a2", "gemv.index.json": "a3",
    "attn.hsaco": "b1", "attn.hash": "b2",
    "old.hsaco": "c1",
}


def _classify(new_files):
    return kid.classify(kid.module_digests(BASE), kid.module_digests(new_files))


def test_same_listing_is_all_identical():
    assert _classify(dict(BASE)) == {
        "identical": ["attn", "gemv", "old"], "changed": [], "removed": [], "added": [],
    }


def test_each_class_and_the_differing_parts_are_reported():
    new = dict(BASE)
    new["gemv.hsaco"] = "zz"
    new["gemv.hash"] = "zz"
    del new["old.hsaco"]
    new["fresh.hsaco"] = "d1"
    assert _classify(new) == {
        "identical": ["attn"],
        "changed": ["gemv (.hash,.hsaco)"],
        "removed": ["old"],
        "added": ["fresh"],
    }


def test_a_module_losing_one_part_is_changed():
    new = dict(BASE)
    del new["attn.hash"]
    assert _classify(new)["changed"] == ["attn (.hash)"]


def test_require_identical_flags_only_matching_changed_or_removed_modules():
    new = dict(BASE)
    new["gemv.hsaco"] = "zz"
    del new["old.hsaco"]
    new["fresh.hsaco"] = "d1"
    result = _classify(new)
    assert kid.violations(result, ["attn*"]) == []
    assert kid.violations(result, ["gemv*", "old"]) == ["changed: gemv (.hsaco)", "removed: old"]
    assert kid.violations(result, ["fresh"]) == []
