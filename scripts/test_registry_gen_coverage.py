#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""Offline gates on registry_gen.py against the committed registry data.

- every curated entry maps to an arch_id and a quant;
- the committed registry/v1.json is exactly what build_registry produces from
  registry/models.json (hipfire-registry include_str!s v1.json, so a stale
  v1.json ships a stale bundled registry);
- the reasoning-effort allow-list matches hipfire-config.
"""
import importlib.util
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

spec = importlib.util.spec_from_file_location(
    "registry_gen", Path(__file__).parent / "registry_gen.py"
)
rg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rg)

SIDECAR_KINDS = ("triattn", "mtp", "dflash", "vision", "t5", "clip", "qwen3", "vae")


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_every_curated_entry_is_generatable():
    missing = []
    for tag, entry in _load(REPO_ROOT / "registry" / "models.json")["models"].items():
        if rg.arch_id_for(tag, entry) is None:
            missing.append(f"{tag}: no arch_id")
        if rg.quant_for(entry.get("file", "")) is None:
            missing.append(f"{tag}: unknown quant for {entry.get('file')!r}")
    assert not missing, "\n".join(missing)


def _tree_item(path: str, sha256, size) -> dict:
    item = {"type": "file", "path": path, "size": size}
    if sha256 is not None:
        item["lfs"] = {"oid": sha256, "size": size}
    return item


def _trees_from(v1: dict) -> dict[str, dict[str, dict]]:
    """Rebuild the HF tree each repo must have had, from v1.json's own digests."""
    trees: dict[str, dict[str, dict]] = {}
    for entry in v1["models"].values():
        repo = entry.get("repo")
        if not repo:
            continue
        tree: dict[str, dict] = {}
        if entry.get("sha256") is not None:
            tree[entry["file"]] = _tree_item(entry["file"], entry["sha256"], entry["size_bytes"])
        sidecars = [entry[k] for k in SIDECAR_KINDS if isinstance(entry.get(k), dict)]
        sidecars += [sc for sc in (entry.get("heads") or {}).values() if isinstance(sc, dict)]
        sidecars += [sc for sc in (entry.get("ple") or {}).values() if isinstance(sc, dict)]
        for sc in sidecars:
            if "size_bytes" in sc:
                tree[sc["file"]] = _tree_item(sc["file"], sc.get("sha256"), sc["size_bytes"])
        if tree:
            trees.setdefault(repo, {}).update(tree)
    return trees


def test_committed_v1_is_reproducible_from_models_json(monkeypatch):
    v1 = _load(REPO_ROOT / "registry" / "v1.json")
    curated = _load(REPO_ROOT / "registry" / "models.json")
    trees = _trees_from(v1)

    def fake_repo_tree(repo, token):
        # An image repo with nothing annotated in v1 was unpublished when v1 was
        # generated; the generator tolerates exactly that failure.
        if repo not in trees:
            raise RuntimeError(f"{repo}: not published")
        return trees[repo]

    monkeypatch.setattr(rg, "repo_tree", fake_repo_tree)
    monkeypatch.setattr(rg, "log", lambda msg: None)
    built, errors = rg.build_registry(curated, None)
    assert not errors, "\n".join(errors)
    built, committed = rg.strip_generated_at(built), rg.strip_generated_at(v1)
    stale = sorted(
        set(built["models"]).symmetric_difference(committed["models"])
        | {t for t in built["models"] if built["models"][t] != committed["models"].get(t)}
    )
    assert not stale, f"registry/v1.json is stale for {stale}; re-run scripts/registry_gen.py"
    assert built == committed, "registry/v1.json is stale; re-run scripts/registry_gen.py"


def test_flash_next_xts_wire_contract():
    models = _load(REPO_ROOT / "registry" / "v1.json")["models"]
    errors = []
    rg.validate_flash_next_xts(models, errors)
    assert not errors, "\n".join(errors)
    for file in (
        rg.FLASH_NEXT_XTS_SPLIT_BASE_FILE,
        *rg.FLASH_NEXT_XTS_MONOLITHIC_FILES,
    ):
        assert rg.quant_for(file) == "mq4"
        assert rg.arch_id_for("qwen3.8:flash-next", {"file": file}) == 16
    # The monolithic is gone from the published registry (the client keeps a pin).
    assert not any(
        entry.get("file") in rg.FLASH_NEXT_XTS_MONOLITHIC_FILES for entry in models.values()
    )
    # The retired rungs are gone from the published registry too; the client
    # keeps compiled pins for a verified local copy (hipfire-registry RETIRED_MODELS).
    curated = _load(REPO_ROOT / "registry" / "models.json")["models"]
    for tag in ("qwen3.8:flash-next-rtn-asym", "qwen3.8:flash-next-mq6q8-pleq8"):
        assert tag not in models and tag not in curated
    for retired_file in ("qwen3.8-flash-next.mq4", "qwen3.8-flash-next.mq6q8-pleq8"):
        assert not any(entry.get("file") == retired_file for entry in models.values())
    # The obsolete local-only ZAB tier is removed outright (no retired pin either).
    assert "qwen3.8:flash-next-mq4-xts-zab" not in models
    assert "qwen3.8:flash-next-mq4-xts-zab" not in curated
    assert not any(
        entry.get("file") == "qwen3.8-flash-next.mq4-xts-zab" for entry in models.values()
    )
    v1 = _load(REPO_ROOT / "registry" / "v1.json")
    assert v1["aliases"]["qwen3.8:flash"] == "qwen3.8:flash-next"
    assert models["qwen3.8:flash-next"]["desc"].startswith("Qwen3.8-Flash-Next MQ4 XTS")
    assert all("GPTQ3" not in entry["desc"] for entry in models.values())


def test_flash_next_mq2_xts_split_base_quant_and_arch():
    for tag, file in (
        ("qwen3.8:flash-next-mq2-xts", "qwen3.8-flash-next-base.mq2-xts"),
        ("qwen3.8:flash-next-mq2-xts-4500", "qwen3.8-flash-next-base.mq2-xts-4500"),
    ):
        assert rg.quant_for(file) == "mq2"
        assert rg.arch_id_for(tag, {"file": file}) == 16
    assert rg.quant_for("qwen3.8-flash-next-base.mq3-xts") == "mq3"
    assert rg.quant_for("x.mq2-xts-abc") is None


@pytest.mark.parametrize("target", [None, "qwen3.8:flash-next-mq4"])
def test_flash_next_short_alias_invariant_rejects_drift(monkeypatch, target):
    v1 = _load(REPO_ROOT / "registry" / "v1.json")
    curated = _load(REPO_ROOT / "registry" / "models.json")
    trees = _trees_from(v1)
    monkeypatch.setattr(rg, "repo_tree", lambda repo, token: trees[repo])
    monkeypatch.setattr(rg, "log", lambda msg: None)
    if target is None:
        del curated["aliases"]["qwen3.8:flash"]
    else:
        curated["aliases"]["qwen3.8:flash"] = target
    registry, errors = rg.build_registry(curated, None)
    assert registry is None
    assert any("qwen3.8:flash must alias" in error for error in errors)


def _rust_str_list(name: str) -> set[str]:
    src = (REPO_ROOT / "crates" / "hipfire-config" / "src" / "lib.rs").read_text()
    m = re.search(rf"const {name}: &\[&str\] = &\[(.*?)\];", src, re.S)
    assert m, f"{name} not found in hipfire-config"
    body = re.sub(r"//[^\n]*", "", m.group(1))
    return set(re.findall(r'"([^"]*)"', body))


@pytest.mark.parametrize(
    "py_name,rust_name",
    [("REASONING_EFFORTS", "REASONING_EFFORTS")],
)
def test_python_allow_lists_match_hipfire_config(py_name, rust_name):
    assert set(getattr(rg, py_name)) == _rust_str_list(rust_name)
