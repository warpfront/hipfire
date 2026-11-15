#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""PP-mode prefill-chunk logic of scripts/fn-ep-gate.sh (both arms of a pair load one chunk) and the
trunk-shard policy of its arms (exactness arms pin HIPFIRE_QWEN4_EP_TRUNK_SHARD=0, the rest run the default).

The helper is tested directly; the gate's own bash functions (`arm_env`, `run_oracle_arm`, `run_bench_cell`,
`oracle_arm_ok`, `pp_choose_chunk_user`, `pp_choose_chunk_from_reference`) are lifted out of the real script
and run against a fake reference-arm log or a stub oracle / bench binary that prints the environment it
received, so no GPU, build or model is needed.
"""
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
GATE = HERE / "fn-ep-gate.sh"
HELPER = HERE / "fn_ep_gate_chunk.py"
sys.path.insert(0, str(HERE))
import fn_ep_gate_chunk as chunk  # noqa: E402

# Last lines of the field report's ref-pp5 arm (pre-fix build), plus the new negotiation line.
DG_STDERR = """\
  qwen4 QSA BF16 split selector: 80 MiB score/filter scratch reserved
  qwen4 prefill chunk: 8192 rows (requested 8192)
qwen4 PP band layers 29..39 (input false, output false): loaded on device 4 (max_seq 8192)
qwen4 PP canonical chunk: 4096 rows, lowered from 8192 for the whole mesh by band 4 (per-band largest admissible rung: band 0 (layers 0..10) 8192, band 4 (layers 39..48) 4096)
qwen4 PP plan: 5 bands [0..10, 10..19] on devices [1, 2], edge payload up to 320 MiB (4096 canonical rows)
"""


def test_parse_takes_the_pp_canonical_line_over_band_chunk_lines() -> None:
    assert chunk.load_chunk(DG_STDERR) == (4096, "PP canonical chunk line")


def test_parse_falls_back_to_a_single_card_load_and_none_without_a_line() -> None:
    single = "  qwen4 prefill chunk: 8192 rows (requested 8192)\n  qwen4 prefill chunk: 4096 rows (requested 8192)\n"
    assert chunk.load_chunk(single) == (4096, "single-card prefill chunk line")
    assert chunk.load_chunk("nothing here\n") is None


def test_user_pin_parses_extra_env() -> None:
    assert chunk.user_chunk("HIPFIRE_PLE_FORMAT=q4 HIPFIRE_PREFILL_CHUNK_ROWS=4096 HIPFIRE_MTP_K=auto") == 4096
    assert chunk.user_chunk("HIPFIRE_PREFILL_CHUNK_ROWS=4000") == 3840  # whole 256-row tiles, as the loader rounds
    assert chunk.user_chunk("HIPFIRE_PREFILL_CHUNK_ROWS=100") is None
    assert chunk.user_chunk("HIPFIRE_PLE_FORMAT=q4 OTHER_HIPFIRE_PREFILL_CHUNK_ROWS=4096") is None
    assert chunk.user_chunk("") is None


def _lift(script: str, start: str, end: str) -> str:
    a = script.index(start)
    return script[a : script.index(end, a)]


def _flow(tmp_path: Path, extra_env: str, ref_rc: int, ref_stderr: str) -> dict:
    """Run the gate's chunk functions after a (fake) reference arm and report what every arm would get."""
    rec = tmp_path / "oracle" / "ref-pp5-receipt"
    rec.mkdir(parents=True)
    (rec / "rc.txt").write_text(f"{ref_rc}\n")
    (rec / "stderr.txt").write_text(ref_stderr)
    (tmp_path / "oracle" / "ref-pp5").mkdir()
    (tmp_path / "oracle" / "ref-pp5" / "meta.json").write_text("{}")
    script = GATE.read_text()
    lifted = "\n".join(
        [
            _lift(script, "ARM_ENV=()\narm_env() {", "start_vram_sampler() {"),
            _lift(script, "oracle_arm_ok() {", "\n}\n") + "\n}",
            _lift(script, "# An explicit HIPFIRE_PREFILL_CHUNK_ROWS in EXTRA_ENV", "run_pp_gate() {"),
        ]
    )
    harness = f"""
set -uo pipefail
log() {{ echo "log: $*" >&2; }}
MODE=pp DEVS_SOURCE=flag NO_ISOLATE=1 TRANSPORT=host DAEMON=/d HIPFIRE_KERNEL_CACHE=/k REF_ARM=ref-pp5
OUT={tmp_path} EXTRA_ENV={extra_env!r} PP_LAYERS="" MTP=0
PP_PINNED_MTP=(HIPFIRE_MTP_AR_FLOOR=0 HIPFIRE_MTP_INCREMENTAL=0 HIPFIRE_MTP_NGRAM=off)
PP_CHUNK="" PP_CHUNK_HOW="not applicable" PP_CHUNK_DERIVED="" PP_CHUNK_USER=""
CHUNK_HELPER={HELPER}
{lifted}
pp_choose_chunk_user
arm_env ref-pp5 1 0,1,2,3,4 5 0 "" 0; echo "REF: ${{ARM_ENV[*]}}"
pp_choose_chunk_from_reference
arm_env pp6 1 0,1,2,3,4,5 6 0 "" 0; echo "TGT: ${{ARM_ENV[*]}}"
echo "CHUNK=$PP_CHUNK"; echo "HOW=$PP_CHUNK_HOW"
"""
    out = subprocess.run(["bash", "-c", harness], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    got = {}
    for line in out.stdout.splitlines():
        key, sep, value = line.partition("=" if line.startswith(("CHUNK=", "HOW=")) else ": ")
        got[key] = value
    return got


def _chunk_words(env_line: str) -> list:
    return [w for w in env_line.split() if w.startswith("HIPFIRE_PREFILL_CHUNK_ROWS=")]


def test_the_reference_arms_chunk_is_pinned_on_the_target_and_recorded(tmp_path: Path) -> None:
    got = _flow(tmp_path, "HIPFIRE_PLE_FORMAT=q4", 0, DG_STDERR)
    assert "-u HIPFIRE_PREFILL_CHUNK_ROWS" in got["REF"] and _chunk_words(got["REF"]) == []
    assert _chunk_words(got["TGT"]) == ["HIPFIRE_PREFILL_CHUNK_ROWS=4096"]
    assert got["CHUNK"] == "4096"
    assert "reference arm" in got["HOW"] and "PP canonical chunk line" in got["HOW"]


def test_an_explicit_chunk_in_extra_env_wins_on_every_arm(tmp_path: Path) -> None:
    got = _flow(tmp_path, "HIPFIRE_PREFILL_CHUNK_ROWS=2048 HIPFIRE_PLE_FORMAT=q4", 0, DG_STDERR)
    for arm in ("REF", "TGT"):
        assert _chunk_words(got[arm]) == ["HIPFIRE_PREFILL_CHUNK_ROWS=2048"], got[arm]
    assert got["CHUNK"] == "2048" and "explicit" in got["HOW"]


def test_a_failed_reference_leaves_the_target_unpinned_and_says_so(tmp_path: Path) -> None:
    got = _flow(tmp_path, "", 1, DG_STDERR)
    assert _chunk_words(got["TGT"]) == [] and got["CHUNK"] == ""
    assert "not pinned" in got["HOW"]


SHARD = "HIPFIRE_QWEN4_EP_TRUNK_SHARD"


def _arm_env_of(tmp_path: Path, mode: str, call: str, extra_env: str = "", inherited: Optional[str] = "1") -> str:
    """What the real `run_oracle_arm` / `run_bench_cell` hand the child: its HIPFIRE_QWEN4_EP_TRUNK_SHARD."""
    stub = tmp_path / "stub.sh"
    stub.write_text(f'#!/bin/sh\necho "SHARD=${{{SHARD}-unset}}"\n')
    stub.chmod(0o755)
    script = GATE.read_text()
    lifted = _lift(script, "ARM_ENV=()\narm_env() {", "oracle_arm_ok() {")
    harness = f"""
set -uo pipefail
log() {{ echo "log: $*" >&2; }}
MODE={mode} DEVS_SOURCE=flag NO_ISOLATE=1 TRANSPORT=host DAEMON=/d HIPFIRE_KERNEL_CACHE=/k
OUT={tmp_path} EXTRA_ENV={extra_env!r} PP_LAYERS="" MTP=0 PP_CHUNK_DERIVED="" SAMPLER="" SPEED_FAIL=0
PP_PINNED_MTP=(HIPFIRE_MTP_AR_FLOOR=0 HIPFIRE_MTP_INCREMENTAL=0 HIPFIRE_MTP_NGRAM=off)
ORACLE={stub} HIPFIRE={stub} MODEL=/m MAX_SEQ=64 TOKENS=1 STATE_STEPS=1 SEED=1
ARM_TIMEOUT=60 BENCH_TIMEOUT=60 PROMPT_ARGS=() BDFS=(none)
SHORT_PROMPT=/p
{lifted}
{call}
"""
    env = {k: v for k, v in os.environ.items() if k != SHARD}
    if inherited is not None:
        env[SHARD] = inherited
    out = subprocess.run(["bash", "-c", harness], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    got = [p.read_text().strip() for p in sorted(tmp_path.rglob("stdout.txt"))]
    got += [p.read_text().strip() for p in sorted(tmp_path.rglob("*.json"))]
    assert len(got) == 1, got
    return got[0]


def test_ep_and_pp_oracle_arms_pin_the_shard_off_whatever_is_inherited(tmp_path: Path) -> None:
    cases = [("ep", "single 1 0 1"), ("ep", "ep2 2 0,1 2"), ("pp", "ppref 1 0 1 1")]
    for n, (mode, call) in enumerate(cases):
        for inherited in ("1", "on", None):
            d = tmp_path / f"{n}-{inherited}"
            d.mkdir()
            got = _arm_env_of(d, mode, f"run_oracle_arm {call}", inherited=inherited)
            assert got == "SHARD=0", (mode, call, inherited, got)


def test_the_pin_is_last_so_extra_env_cannot_turn_the_shard_on_for_an_exactness_arm(tmp_path: Path) -> None:
    got = _arm_env_of(tmp_path, "ep", "run_oracle_arm ep2 2 0,1 2", extra_env=f"{SHARD}=1")
    assert got == "SHARD=0"


def test_speed_cells_clear_an_inherited_shard_value_and_run_the_default_route(tmp_path: Path) -> None:
    for n, inherited in enumerate(("1", "0", None)):
        d = tmp_path / str(n)
        d.mkdir()
        got = _arm_env_of(d, "ep", 'run_bench_cell ep2 2 0,1 decode 0 "$SHORT_PROMPT" 128', inherited=inherited)
        assert got == "SHARD=unset", (inherited, got)


def test_reserve_arms_run_the_default_route_and_may_take_an_explicit_value_from_extra_env(tmp_path: Path) -> None:
    for n, (call, extra, want) in enumerate(
        [
            ("run_oracle_arm ep2-off 2 0,1 2 0 0 '' 0", "", "SHARD=unset"),
            ("run_oracle_arm ep2-on 2 0,1 2 0 0 '' 1", "", "SHARD=unset"),
            ("run_oracle_arm ep2-on 2 0,1 2 0 0 '' 1", f"{SHARD}=0", "SHARD=0"),
        ]
    ):
        d = tmp_path / str(n)
        d.mkdir()
        assert _arm_env_of(d, "reserve", call, extra_env=extra, inherited="1") == want


def test_extra_env_may_not_set_the_shard_outside_reserve_mode(tmp_path: Path) -> None:
    model = tmp_path / "model.stub"
    model.write_text("")
    for mode in ("ep", "pp"):
        out = subprocess.run(
            ["bash", str(GATE), "--mode", mode, "--skip-build", str(model), "2"],
            capture_output=True,
            text=True,
            env={**os.environ, "EXTRA_ENV": f"HIPFIRE_PLE_FORMAT=q4 {SHARD}=1"},
            cwd=tmp_path,
        )
        assert out.returncode != 0
        assert f"EXTRA_ENV may not set {SHARD}" in out.stdout + out.stderr
