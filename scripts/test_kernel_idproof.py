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
        "identical": ["attn", "gemv", "old"], "changed": [], "nondeterministic": [],
        "removed": [], "added": [],
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
        "nondeterministic": [],
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


INPUTS = kid.INPUTS


def _with_inputs(files, inputs):
    mods = kid.module_digests(files)
    for module, sha in inputs.items():
        mods[module][INPUTS] = sha
    return mods


def test_index_inputs_ignore_only_the_object_sha():
    a = b'{"module": "m", "object_sha256": "aa", "flags": ["-O3"], "source_sha256": "s"}'
    b = b'{"source_sha256": "s", "flags": ["-O3"], "object_sha256": "bb", "module": "m"}'
    assert kid.index_inputs(a) == kid.index_inputs(b)
    assert kid.index_inputs(a) != kid.index_inputs(a.replace(b'-O3', b'-O2'))
    assert kid.index_inputs(a) != kid.index_inputs(a.replace(b'"s"', b'"t"'))


def test_same_inputs_different_object_is_nondeterministic_not_changed():
    base = _with_inputs(BASE, {"gemv": "i1", "attn": "i2"})
    new_files = dict(BASE)
    new_files["gemv.hsaco"] = "zz"
    new_files["gemv.index.json"] = "zz"  # only its object_sha256 differs
    result = kid.classify(base, _with_inputs(new_files, {"gemv": "i1", "attn": "i2"}))
    assert result["nondeterministic"] == ["gemv (.hsaco,.index.json)"]
    assert result["changed"] == []
    assert result["identical"] == ["attn", "old"]
    # Not a failure, even for a module the caller requires identical.
    assert kid.violations(result, ["gemv*"]) == []


def test_changed_inputs_or_hash_or_missing_inputs_stay_changed():
    new_files = dict(BASE)
    new_files["gemv.hsaco"] = "zz"
    new_files["gemv.index.json"] = "zz"
    base = _with_inputs(BASE, {"gemv": "i1"})
    # The compile's inputs differ: a real change.
    assert kid.classify(base, _with_inputs(new_files, {"gemv": "i9"}))["changed"] == [
        "gemv (.hsaco,.index.json)"
    ]
    # The key's `.hash` differs although the inputs agree: not noise.
    hashed = dict(new_files)
    hashed["gemv.hash"] = "zz"
    assert kid.classify(base, _with_inputs(hashed, {"gemv": "i1"}))["changed"] == [
        "gemv (.hash,.hsaco,.index.json)"
    ]
    # An old pack without inputs for the module cannot be called noise.
    assert kid.classify(base, kid.module_digests(new_files))["changed"] == [
        "gemv (.hsaco,.index.json)"
    ]
    # The index changed without the object: not noise either.
    index_only = dict(BASE)
    index_only["gemv.index.json"] = "zz"
    assert kid.classify(base, _with_inputs(index_only, {"gemv": "i1"}))["changed"] == [
        "gemv (.index.json)"
    ]
