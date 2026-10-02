#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""Offline gates on registry_gen.py against the committed registry data.

- every curated entry maps to an arch_id and a quant;
- the committed registry/v1.json is exactly what build_registry produces from
  registry/models.json (hipfire-registry include_str!s v1.json, so a stale
  v1.json ships a stale bundled registry);
- the Python allow-lists that "MUST stay in sync" with hipfire-config do.
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


def _rust_str_list(name: str) -> set[str]:
    src = (REPO_ROOT / "crates" / "hipfire-config" / "src" / "lib.rs").read_text()
    m = re.search(rf"const {name}: &\[&str\] = &\[(.*?)\];", src, re.S)
    assert m, f"{name} not found in hipfire-config"
    body = re.sub(r"//[^\n]*", "", m.group(1))
    return set(re.findall(r'"([^"]*)"', body))


@pytest.mark.parametrize(
    "py_name,rust_name",
    [("KNOWN_KV_MODES", "KV_MODES"), ("REASONING_EFFORTS", "REASONING_EFFORTS")],
)
def test_python_allow_lists_match_hipfire_config(py_name, rust_name):
    assert set(getattr(rg, py_name)) == _rust_str_list(rust_name)
