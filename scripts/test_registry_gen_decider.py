#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""`decider:4b` registry entry and its arch mapping.

Decider-4B is a dense Qwen3.5-4B finetune published as a `/v1/systemone` judge
(MQ4G256V2, symmetric, uncalibrated). The daily registry workflow is
fail-closed: an unmapped tag family makes `arch_id_for` return None, which
aborts the ENTIRE run, every other model's entry included. So the `decider`
family must be mapped explicitly, to dense arch 5.

The entry is checked against `registry/models.json` only; the pinned
sha256/size_bytes land in `registry/v1.json` when the generator runs after the
upload, and `test_registry_gen_coverage.py` covers that file.
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

TAG = "decider:4b"
UPLOADED_BYTES = 2_595_461_120


def _curated() -> dict:
    return json.loads((REPO_ROOT / "registry" / "models.json").read_text())


def test_decider_family_is_dense_arch5():
    assert rg.arch_id_for(TAG, {"file": "decider-4b.mq4"}) == 5


def test_mq4_file_is_a_known_quant():
    assert rg.quant_for("decider-4b.mq4") == "mq4"


def test_curated_entry_matches_the_published_artifact():
    entry = _curated()["models"][TAG]
    assert entry["repo"] == "hipfire-models/decider-4b"
    assert entry["file"] == "decider-4b.mq4"
    # Declared size must sit within the generator's drift tolerance of the bytes
    # actually uploaded (decimal GB, same convention as every other entry).
    assert abs(entry["size_gb"] - UPLOADED_BYTES / 1e9) <= rg.SIZE_TOLERANCE
    assert entry["min_vram_gb"] > entry["size_gb"]
    assert rg.arch_id_for(TAG, entry) == 5
    assert rg.quant_for(entry["file"]) == "mq4"


def test_curated_entry_is_uncalibrated_and_carries_no_sampling_overrides():
    # The artifact is symmetric/uncalibrated; a calibration profile is a
    # models.toml decision for later (spec 2026-09-28-jev-decide-design.md
    # section 13.9), not a registry field. The judge reads one next-token
    # distribution, so it declares no sampling presets either.
    entry = _curated()["models"][TAG]
    assert "uncalibrated" in entry["quant_recipe"]
    assert "recommended_settings" not in entry
    assert "sampling_profiles" not in entry
    assert "calibration" not in entry


def test_bare_decider_aliases_to_the_4b_tag():
    assert _curated()["aliases"].get("decider") == TAG
