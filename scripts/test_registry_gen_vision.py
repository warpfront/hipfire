#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""The shared Qwen3.8-27B vision-tower sidecar slot (`vision` kind).

Every non-draft `qwen3.8:27b*` target tier declares the same
`qwen3.8-27b-vision.hfq` file (llama.cpp mmproj-style) so each text quant
tier serves images without
requantizing the trunk. The pack ships after the registry slot: sha256 and
size_bytes stay absent until the file lands on HF, at which point
`registry_gen.py` annotates them like any other sidecar kind.

Unlike dflash, the vision file is standalone — no registry entry's `file`
names it — so there is no "pairs with an entry file" invariant to pin here.

Tags containing `-draft` (`qwen3.8:27b-draft` alias and
`qwen3.8:27b-draft-mq3`..`-mq6`) are text-only DFlash controller sidecars,
not target tiers, and carry no vision slot; `_tiers` excludes them.
"""
import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

spec = importlib.util.spec_from_file_location(
    "registry_gen", Path(__file__).parent / "registry_gen.py"
)
rg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rg)

VISION_FILE = "qwen3.8-27b-vision.hfq"


def _curated() -> dict:
    return json.loads((REPO_ROOT / "registry" / "models.json").read_text())


def _tiers(models: dict) -> dict:
    return {k: v for k, v in models.items() if k.startswith("qwen3.8:27b") and "-draft" not in k}


def test_every_qwen38_27b_tier_declares_the_shared_vision_file():
    models = _tiers(_curated()["models"])
    assert len(models) > 0, "expected qwen3.8:27b* tiers in curated models"
    digests = set()
    for tag, entry in models.items():
        vision = entry.get("vision")
        assert isinstance(vision, dict), f"{tag} must declare a vision slot"
        assert vision.get("file") == VISION_FILE, f"{tag} must share {VISION_FILE}"
        assert set(vision) == {"file", "sha256", "size_bytes"}, f"{tag}: vision slot must be digest-pinned"
        digests.add((vision["sha256"], vision["size_bytes"]))
    # One physical pack serves every tier: the pins must agree, or `pull` for
    # one tier would verify a different file than `rm`'s shared-keeper logic
    # protects for another.
    assert len(digests) == 1, f"qwen3.8 tiers disagree on the vision pack digest: {digests}"


def test_tiers_excludes_text_only_draft_sidecars():
    # `-draft` (alias) and `-draft-mq3..mq6` are text-only DFlash controller
    # sidecars, not vision-bearing target tiers.
    models = {
        "qwen3.8:27b": {},
        "qwen3.8:27b-mq3": {},
        "qwen3.8:27b-draft": {},
        "qwen3.8:27b-draft-mq3": {},
        "qwen3.8:27b-draft-mq4": {},
        "qwen3.8:27b-draft-mq5": {},
        "qwen3.8:27b-draft-mq6": {},
        "qwen3.6:35b-a3b": {},
    }
    assert set(_tiers(models)) == {"qwen3.8:27b", "qwen3.8:27b-mq3"}


def test_annotate_sidecar_resolves_vision_digest_from_tree():
    tree = {
        VISION_FILE: {"lfs": {"oid": "abc123", "size": 1_000_000_000}, "size": 1_000_000_000}
    }
    errors: list = []
    out = rg.annotate_sidecar({"file": VISION_FILE}, tree, "qwen3.8:27b", "vision", errors)
    assert errors == []
    assert out == {"file": VISION_FILE, "sha256": "abc123", "size_bytes": 1_000_000_000}


def test_annotate_sidecar_fails_closed_for_missing_vision_file():
    errors: list = []
    out = rg.annotate_sidecar({"file": VISION_FILE}, {}, "qwen3.8:27b", "vision", errors)
    assert out == {"file": VISION_FILE}
    assert any("vision" in e for e in errors), errors


def test_build_registry_annotates_the_vision_kind(monkeypatch):
    # Proves the `vision` kind is wired into the sidecar loop (no network:
    # repo_tree is stubbed). Mirrors what the daily run emits once the pack
    # is uploaded: file + HF digest, carried through the strict-superset gate.
    size = 1_000_000_000
    tree = {
        "qwen3.8-27b.mq4": {"lfs": {"oid": "trunk", "size": 15_660_000_000}, "size": 15_660_000_000},
        VISION_FILE: {"lfs": {"oid": "tower", "size": size}, "size": size},
    }
    monkeypatch.setattr(rg, "repo_tree", lambda repo, token: tree)
    curated = {
        "models": {
            "qwen3.8:27b": {
                "repo": "hipfire-models/qwen3.8-27b",
                "file": "qwen3.8-27b.mq4",
                "size_gb": 15.66,
                "min_vram_gb": 17,
                "desc": "test trunk",
                "vision": {"file": VISION_FILE},
            }
        },
        "aliases": {},
    }
    registry, errors = rg.build_registry(curated, None)
    assert errors == [], errors
    assert registry is not None
    assert registry["models"]["qwen3.8:27b"]["vision"] == {
        "file": VISION_FILE,
        "sha256": "tower",
        "size_bytes": size,
    }
