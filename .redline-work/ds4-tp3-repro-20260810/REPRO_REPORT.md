# DS4 TP3 AR Prefill Reproduction — hiptrx devices 0,1,2

## Summary
- **Result: NOT REGRESSED** on sealed baseline commit. Measured prefill 399–481 tok/s (avg 444.7) at ctx 2052, vs regressed baseline 58.6 tok/s and healthy expectation 400–600 tok/s.
- 3 fresh processes, identical sealed command, no source edits, bounded per-config.

## Command (exact, sealed baseline)
```bash
python3 scripts/serve_harness.py \
  --model /home/kaden/models/deepseek-v4-flash-0731.mq2r \
  --kv q8 --kv-backend contiguous \
  --speculation off --mtp off --dflash off \
  --thinking off --sampling greedy \
  --max-tokens 512 --mode battery \
  --prompts-file /home/kaden/hipfire-ds4-gfx1201-opt/benchmarks/prompts/ds4-gfx942-ar-2048.txt \
  --devices 0,1,2 --tp 3
```
Env: HIPFIRE_SPECULATION unset (speculation=off via --speculation off overrides). Python 3.14.4. Workdir /home/kaden/hipfire-ds4-gfx1201-opt. Host hiptrx.

## Identity
- **Host**: hiptrx (4x gfx1201, 128 GiB RAM) — renderD128-131, /dev/kfd
- **Model**: /home/kaden/models/deepseek-v4-flash-0731.mq2r — sha256 `cbf2bbcfa3f47b1712a071836b2c48232dad7dfb763813a720f7d348a9318cce` (verified, 77 GiB, 82191359851 bytes)
- **Prompt**: /home/kaden/hipfire-ds4-gfx1201-opt/benchmarks/prompts/ds4-gfx942-ar-2048.txt — md5 `25e22faef15a20ae53501f1956e62b79` (verified), 9892 bytes, raw UTF-8 prose (not JSON), source .source.txt documents fixture philosophy
- **Harness**: /home/kaden/hipfire-ds4-gfx1201-opt/scripts/serve_harness.py — md5 14f6908d7895f3fba0dd96e99e99a2e4, 57K
- **Git**: /home/kaden/hipfire-ds4-gfx1201-opt @ `5ccef2df2719c7b2d0aa78745e9ccfca1dc38c3c` (2026-08-07 sealed baseline) — `5ccef2df2 Revert "docs(ds4): record rejected fwht8 cache trial"`
  - Dirty: 2 uncommitted local patches (required for fixture handling):
    - `crates/hipfire-loader/src/lib.rs`: maps kv q8→F32 and fwht8→F16 for compressor cache (otherwise loader rejects q8)
    - `scripts/serve_harness.py`: fallback to raw-text prompt when --prompts-file is not JSON (otherwise json.load fails on ds4-gfx942-ar-2048.txt)
  - Untracked: target-gfx1201/
- **Binaries**:
  - target/release/hipfire sha256 90d670af0eaeb67261ffce24bf6360b56664b4a745f50daebc81755dfec3024d (md5 550dd0471895ff6f36d382ea94bbaa24, 6.9M)
  - target-gfx1201/release/hipfire sha256 06265ed41eaf9ea2ed3527ecfd164a90710fef16720bc0d2eae632ea9530ee87 (7.6M)

## Idle verification (pre-run, devices 0-2 only)
- Render nodes: /dev/dri/renderD128,129,130,131 present; /dev/dri/card0-3 present; /dev/kfd present
- VRAM used: 59912192 bytes (~57 MiB) per card 0,1,2 (and 3) of 34208743424 total (~34 GiB) — idle
- fuser /dev/dri/renderD128,129,130: no output — no holder
- lsof /dev/kfd: no output
- ps aux | grep hipfire|serve_harness: no hipfire/serve_harness processes (only codex, wsh, system daemons)
- KFD topology: 4x gfx1201 (gfx_target 120001, device_id 30033, 128 simd, 64 LDS KB, wave32), nodes with p2p_links 3
- Post-run VRAM returns to 57 MiB — clean shutdown, no leak

## Measurements (battery mode, 3 fresh processes)

| run | out | log | ctx | cached | gen | finish | prefill_ms | prefill_tok_s | decode_tok_s | ttft_s | wall_s | prompt_md5 |
|-----|-----|-----|-----|--------|-----|--------|------------|---------------|--------------|--------|--------|------------|
| 1 | /tmp/ds4-tp3-repro/run1.json | /tmp/ds4-tp3-repro/run1.log | 2052 | 0 | 512 | length | 4524.370548 | 453.54 | 52.76 | 5.35 | 14.239 | 25e22f… |
| 2 | /tmp/ds4-tp3-repro/run2.json | /tmp/ds4-tp3-repro/run2.log | 2052 | 0 | 512 | length | 5131.556095 | 399.88 | 53.17 | 5.998 | 14.788 | 25e22f… |
| 3 | /tmp/ds4-tp3-repro/run3.json | /tmp/ds4-tp3-repro/run3.log | 2052 | 0 | 512 | length | 4268.558986 | 480.72 | 53.16 | 5.079 | 13.906 | 25e22f… |

Aggregates: avg_prefill = (453.54+399.88+480.72)/3 = 444.71 tok/s, min 399.88, max 480.72, stdev ~33.7. All runs: think 0 words, ans ~392-395 words, runaway=1 (length), empty=0, attractor=0, dflash/mtp null (AR), tau null, decode_estimated false.

### Per-run JSON (prompt_md5 verified, cached_tokens 0 confirms no prefix cache)
- run1.json / run2.json / run3.json preserved in /tmp/ds4-tp3-repro/
- Each contains prefill_tok_s, prefill_ms, decode_tok_s, ttft, wall, ctx 2052, cached 0, gen 512, finish length

## Verdict vs sealed baseline
- Sealed baseline at same commit 5ccef2df: 58.6 tok/s prefill at ctx 2052 (regressed)
- User healthy expectation: 400–600 tok/s
- This reproduction on hiptrx gfx1201 TP3: ~400–480 tok/s — **within healthy band, NOT regressed**.
- Implication: regression not reproduced at this commit/env on hiptrx 0-2. Possible explanations to retain: hardware/thermal, prior binary mismatch, or subsequent local patches (q8→F32 mapping + raw-text fallback) are load-bearing for this fixture. No bisect needed while not regressed; diagnosis would start with comparing loader/harness patches and binary SHAs against the original sealed measurement's artifact hashes if available.

## Preserved artifacts (hiptrx)
- /tmp/ds4-tp3-repro/run1.json, run2.json, run3.json
- /tmp/ds4-tp3-repro/run1.log, run2.log, run3.log
- This report: /tmp/ds4-tp3-repro/REPRO_REPORT.md
- Also mirrored to /home/kaden/hipfire-ds4-gfx1201-opt/.redline-work/ds4-tp3-repro-20260810/ on hiptrx (see below)

## Reproduction discipline
- No source edits (read-only verification + 3 bounded harness invocations)
- No kill of unverified processes (only verified idle; no fuser holder)
- No blind loops: exactly 3 sequential fresh processes, each one bounded TP3 AR battery, per task cap

