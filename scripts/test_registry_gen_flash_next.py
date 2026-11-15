#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""Flash-Next MQ4 XTS is ONE split base on the shared PLE pair.

`qwen3.8:flash-next` must be the split base `qwen3.8-flash-next-base.mq4-xts`
with the family's shared `ple` map (`q4` / `q8`); the other stock spellings
(`qwen3.8:flash`, `-mq4`, `-gptq3`, `-mq4-xts`) must be aliases of it, never
duplicate rows; and no entry may download the removed monolithic
(`qwen3.8-flash-next.mq4-xts`, `qwen3.8-flash-next-gptq3.mq4`); the client keeps
a compiled-in pin for a verified local copy.

No network: `repo_tree` is stubbed, the catalog is built in memory.
"""
import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "registry_gen", Path(__file__).parent / "registry_gen.py"
)
rg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rg)

BASE = rg.FLASH_NEXT_XTS_SPLIT_BASE_FILE
Q4 = rg.FLASH_NEXT_XTS_SHARED_PLE["q4"]
Q8 = rg.FLASH_NEXT_XTS_SHARED_PLE["q8"]
Q4_BYTES = 29_120_146_432
Q8_BYTES = 54_400_269_312


def _stock_entry() -> dict:
    return {
        "repo": rg.FLASH_NEXT_XTS_REPO,
        "file": BASE,
        "size_gb": 70.9,
        "min_vram_gb": 72,
        "desc": "Qwen3.8-Flash-Next MQ4 XTS split base",
        "arch_id": 16,
        "quant": "mq4",
        "sha256": rg.FLASH_NEXT_XTS_SPLIT_BASE_SHA256,
        "size_bytes": rg.FLASH_NEXT_XTS_SPLIT_BASE_SIZE,
        "ple": {
            "q4": {
                "file": Q4,
                "sha256": "fc78d30b9d2376db2b713732e80de2069f1cd1d6e1e40b50c716927dcc18ee1e",
                "size_bytes": Q4_BYTES,
            },
            "q8": {
                "file": Q8,
                "sha256": "a655edc25402d0dfbef276cb3c5ca57724fa1074976d8a9029371a3a52eca89c",
                "size_bytes": Q8_BYTES,
            },
        },
        "recommended_settings": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
    }


def _stock_models() -> dict:
    return {rg.FLASH_NEXT_XTS_TAG: _stock_entry()}


def _stock_aliases() -> dict:
    return {alias: rg.FLASH_NEXT_XTS_TAG for alias in rg.FLASH_NEXT_XTS_ALIAS_TAGS}


def _errors(models: dict, aliases: dict | None = None) -> list[str]:
    errors: list[str] = []
    rg.validate_flash_next_xts(
        models, _stock_aliases() if aliases is None else aliases, errors
    )
    return errors


def test_the_stock_split_base_passes():
    assert _errors(_stock_models()) == []


def test_unrelated_catalogs_are_not_checked():
    assert _errors({"x:1": {"repo": "other/repo", "file": "x.mq4"}}) == []


@pytest.mark.parametrize(
    "key,value",
    [
        ("file", "qwen3.8-flash-next-base.mq3-xts"),
        ("repo", "other/flash-next"),
        ("sha256", "a" * 64),
        ("size_bytes", rg.FLASH_NEXT_XTS_SPLIT_BASE_SIZE - 1),
        ("arch_id", 5),
        ("quant", "mq6"),
        ("desc", "Qwen3.8-Flash-Next split base"),
        ("desc", "Qwen3.8-Flash-Next MQ4 XTS GPTQ3"),
    ],
)
def test_split_base_invariants_reject_drift(key, value):
    models = _stock_models()
    models["qwen3.8:flash-next"][key] = value
    assert _errors(models)


def test_the_stock_tag_is_required():
    other = {"other": {"repo": rg.FLASH_NEXT_XTS_REPO, "file": BASE, "desc": "x"}}
    assert any("missing" in error for error in _errors(other))


@pytest.mark.parametrize("alias", rg.FLASH_NEXT_XTS_ALIAS_TAGS)
def test_every_stock_spelling_must_alias_the_split_tag(alias):
    aliases = _stock_aliases()
    del aliases[alias]
    assert any(f"{alias} must alias" in error for error in _errors(_stock_models(), aliases))
    aliases[alias] = "qwen3.8:flash-next-mq4-xts-symtrunk"
    assert any(f"{alias} must alias" in error for error in _errors(_stock_models(), aliases))


@pytest.mark.parametrize("key", ["q4", "q8"])
@pytest.mark.parametrize(
    "mutate",
    [
        lambda ple, key: ple.pop(key),
        lambda ple, key: ple[key].update(file=f"{BASE}.ple-{key}"),
        lambda ple, key: ple[key].update(file="other.ple"),
    ],
    ids=["missing", "per-base name", "foreign file"],
)
def test_ple_sidecars_must_be_the_shared_files(key, mutate):
    models = _stock_models()
    mutate(models["qwen3.8:flash-next"]["ple"], key)
    assert any(f"ple.{key}" in error for error in _errors(models))


@pytest.mark.parametrize("alias", rg.FLASH_NEXT_XTS_ALIAS_TAGS)
def test_a_stock_spelling_may_not_be_a_duplicate_row(alias):
    models = _stock_models()
    models[alias] = _stock_entry()
    assert any(alias in error and "duplicate" in error for error in _errors(models))


@pytest.mark.parametrize("file", rg.FLASH_NEXT_XTS_MONOLITHIC_FILES)
def test_no_entry_may_name_the_removed_monolithic(file):
    # Even an unrelated tag of the repo, or a tag outside the four.
    models = _stock_models()
    models["qwen3.8:flash-next-extra"] = {
        "repo": rg.FLASH_NEXT_XTS_REPO,
        "file": file,
        "desc": "x",
    }
    assert any(file in error and "monolithic" in error for error in _errors(models))
    # And a stock tag re-pointed at it fails both the ban and the file invariant.
    models = _stock_models()
    models["qwen3.8:flash-next"]["file"] = file
    errors = _errors(models)
    assert any("monolithic" in error for error in errors)
    assert any("file must be" in error for error in errors)


def test_old_wire_registry_shape_is_rejected():
    """The pre-split wire defaults (legacy filename on three tags) are gone."""
    models = _stock_models()
    for tag in rg.FLASH_NEXT_XTS_ALIAS_TAGS[1:]:
        models[tag] = _stock_entry()
        models[tag]["file"] = rg.FLASH_NEXT_XTS_LEGACY_FILE
    models["qwen3.8:flash-next"]["file"] = rg.FLASH_NEXT_XTS_CANONICAL_FILE
    errors = _errors(models, {})
    assert sum("removed monolithic" in error for error in errors) == 4


def _tree() -> dict:
    def item(path, oid, size):
        return {"type": "file", "path": path, "size": size, "lfs": {"oid": oid, "size": size}}

    entry = _stock_entry()
    return {
        BASE: item(BASE, entry["sha256"], entry["size_bytes"]),
        Q4: item(Q4, entry["ple"]["q4"]["sha256"], Q4_BYTES),
        Q8: item(Q8, entry["ple"]["q8"]["sha256"], Q8_BYTES),
    }


def _curated(models: dict) -> dict:
    curated_models = {}
    for tag, entry in models.items():
        entry = copy.deepcopy(entry)
        # The curated overlay names the sidecars; the generator pins them from the tree.
        for key in ("arch_id", "quant", "sha256", "size_bytes"):
            entry.pop(key, None)
        entry["ple"] = {key: {"file": sidecar["file"]} for key, sidecar in entry["ple"].items()}
        curated_models[tag] = entry
    return {"models": curated_models, "aliases": _stock_aliases()}


def test_build_registry_accepts_the_split_only_catalog(monkeypatch):
    monkeypatch.setattr(rg, "repo_tree", lambda repo, token: _tree())
    monkeypatch.setattr(rg, "log", lambda msg: None)
    registry, errors = rg.build_registry(_curated(_stock_models()), None)
    assert errors == [], errors
    built = registry["models"][rg.FLASH_NEXT_XTS_TAG]
    assert built["file"] == BASE
    assert built["sha256"] == rg.FLASH_NEXT_XTS_SPLIT_BASE_SHA256
    assert built["size_bytes"] == rg.FLASH_NEXT_XTS_SPLIT_BASE_SIZE
    assert built["ple"]["q4"]["file"] == Q4 and built["ple"]["q8"]["file"] == Q8
    for alias in rg.FLASH_NEXT_XTS_ALIAS_TAGS:
        assert alias not in registry["models"]
        assert registry["aliases"][alias] == rg.FLASH_NEXT_XTS_TAG


def test_build_registry_fails_when_a_tag_still_names_the_monolithic(monkeypatch):
    tree = _tree()
    tree[rg.FLASH_NEXT_XTS_CANONICAL_FILE] = {
        "type": "file",
        "path": rg.FLASH_NEXT_XTS_CANONICAL_FILE,
        "size": 125_288_544_792,
        "lfs": {"oid": "8b15b6fede7d7c5bfed0db4720a8295bedda51bc93e545fa242bd50d0f200972",
                "size": 125_288_544_792},
    }
    monkeypatch.setattr(rg, "repo_tree", lambda repo, token: tree)
    monkeypatch.setattr(rg, "log", lambda msg: None)
    curated = _curated(_stock_models())
    curated["models"]["qwen3.8:flash-next"]["file"] = rg.FLASH_NEXT_XTS_CANONICAL_FILE
    registry, errors = rg.build_registry(curated, None)
    assert registry is None
    assert any("monolithic" in error for error in errors), errors
