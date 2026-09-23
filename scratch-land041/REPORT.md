# land-041 composite gate report (mq4-lloyd @ 0.4.0)

## Provenance

- Base: `e5f944a3a` (0.4.0), `mq4-lloyd` fast-forward target.
- Branches merged into `land-041` (all based on `e5f944a3a`, merged conflict-free):
  - `gfx11-fa2-q16` @ `e61919a37` — FA2 16-query tiling on gfx11 (XTX ship pass +1.99% pp8192; Halo stackable; bit-exact).
  - `gfx11-rope-fast` @ `3d2c114aa` — exact RoPE rewrite (vectorized across heads, shared angles; legacy kernel behind `HIPFIRE_ROPE_INTERLEAVED_LEGACY=1`).
  - `fix-tiny-qwen-q8-reduce` @ `c86862b89` (+`65f3b6a07`) — tile sizing bounding the q8 flash reducer LDS below 64 KiB. Pushed to hipx before merging.
- Worktrees: hipx `/home/kaden/hipfire-land041`, local `/home/kaden/ClaudeCode/warpfront/wt-land041`, both branched from `e5f944a3a`.
- FA2 and tiny-Qwen both touch `crates/rdna-compute/src/attention.rs` in disjoint regions (FA2 launcher ~L5615, tile policy ~L85-115): all three merges applied with zero conflicts, semantics of each side preserved verbatim.
- CHANGELOG: one entry per branch added under Unreleased `### v0.4.0` in the existing style (commit `f22e8e802`); the RoPE line was missing and is now present. No version bump (stays 0.4.0).
- Clean-room: case-insensitive `git log -p e5f944a3a..HEAD` scan for the forbidden token reports 0 occurrences.
## Local gates (gfx1201) — PASS

- Scoped tests: `cargo test -p rdna-compute q8_flash` → 3 passed, 0 failed, including the branch-added `q8_flash_tile_grows_when_reducer_lds_would_exceed_capacity` (2K/262K/1M reservation boundaries).
- Card-B `GPU-e475645fe0200397` guard (`scripts/guard_gfx1201_baseline.py --output-dir scratch-land041/guard --home /home/kaden/.hipfire-homes/land041-guard`): **PASS — actual fp8 backend and baseline guard**. After one discarded fresh-process warmup, three independent pp8192 process medians 3685.80 / 3658.90 / 3645.20 tok/s (all ≥ 3,620 floor); decode medians 36.56 / 36.53 / 36.50 tok/s (all within 1% of 36.5). Script verified `Fp8 vmm` + `kv=fp8` in each process log. Evidence: `scratch-land041/guard/`.
- Card-A `GPU-9eb7aeda51c88ffd` tiny-Qwen repro (qwen3.5-0.8b, `--max-tokens 96 --kv-mode q8 --kv-backend vmm --spec off`, `HIPFIRE_GRAPH=1`, private HOME with `reasoning.mode=off`): exit 0, coherent two-sentence Rayleigh-scattering answer, no GPU fault. Log: `GPU dev 0: gfx1201`, `KV cache: Q8 vmm (`, `physical_cap=262144 / max_seq=262144`. Full log: `scratch-land041/tinyqwen-run.log`. Never touched card-D `GPU-6109a4cb5f833235`.
- Hashes (local): 27B fixture MD5 `2cfe88923b3671ca16a8de6ec1122fde` (pinned, verified by guard); tiny model MD5 `19fb3ef3be67e09277b691ba8500257a`; land041 CLI SHA256 `301cc3455a0412958180c3133b73e0a2e69b213556df293a448678a864a3ad3b`, daemon `d8312d958ca33665d101e613040e941e21cce9b902b0a6f8ec93d1d3a49a9a07`.
## hipx gates (gfx1100 / Halo) — PASS

- Lease discipline: announced acquire/pre/post/release via hub around X5Integ, R256Screen, BetaAlpha, GemmCalib windows. Precheck: KFD only `gpusentry` PID 2970, 0% VRAM. No hipx compile during anyone's timing; gfx1100 = HIP 0 (`ROCR_VISIBLE_DEVICES=0`), Halo = HIP 1 (`ROCR_VISIBLE_DEVICES=1`), `HIP_VISIBLE_DEVICES=0` always, never HIP 3. Every process attested by `/proc` env (both visibility vars) plus the `GPU dev 0:` arch log line and `KV cache: Q8 vmm (`.
- Build: native `cargo build --release` in `/home/kaden/hipfire-land041` green (warnings only); scoped `cargo test --release -p rdna-compute q8_flash` 3/3 pass on hipx; `eval_hipfire` example built. Landed baseline reused from `/home/kaden/hipfire-prof040` (CLI SHA256 `0c10c3a6e1954b86e37c21ade0ea3822fdf63291df3ac3d4d6e1da56450e85f2`, matching the FA2 record) and `/home/kaden/hipfire-land040` @ `e5f944a3a` for WT2-A.
- Matrix (`hipfire bench <model> --matrix --pp 512,8192 --ctx 128 --tg 128 --spec off --runs 3 --warmups 1 --kv-mode q8 --kv-backend vmm --json`, one fresh-process A1/B1 pair per card, runner `scratch-land041/run_matrix.py`):
  - gfx1100: A1 pp512 2143.9 / pp8192 2183.4 / tg128@128 49.073266; B1 pp512 2130.0 (-0.65%) / pp8192 2222.0 (+1.77%) / tg 49.126387 (+0.11%). No row >1% below landed. B1 vs FA2-B (2130.35/2224.30): -0.02%/-0.10%, within noise.
  - Halo: A1 pp512 814.4 / pp8192 762.5 / tg 14.852115; B1 pp512 808.1 (-0.77%) / pp8192 758.1 (-0.58%) / tg 14.855923 (+0.03%). No row >1% below landed. B1 vs FA2-B (797.25/751.45): +1.36%/+0.88%, at/above FA2 values (stackable profile consistent).
- WT2 c24 q8/q8 (`run_wt2.py`, A from land040, B from land041): gfx1100 A/B KLD 0.076879/0.076879 (diff 0, equals pinned 0.076879); Halo A/B 0.076901/0.076901 (diff 0, equals pinned 0.076901). Gate B<=0.10, diff<=0.0005, exact pin match.
- Battery: manual `hipfire serve` on gfx1100 (isolated HOME, `GPU dev 0: gfx1100`, Q8 vmm, max_seq 249344 card-bound) + `scripts/serve_harness.py --no-spawn --mode battery --kv q8 --kv-backend vmm --port 11520 --max-think-tokens 1 --max-tokens 512` → **5/5 finish=stop, runaway=0, empty=0, attractor=0, retrieval_miss=0**. JSON: `scratch-land041/gfx1100-battery.json` (also mirrored locally under `scratch-land041/hipx/`). Graceful shutdown; postcheck KFD only `gpusentry` PID 2970, PIDs gone.
- Hashes (hipx): model MD5 `2cfe88923b3671ca16a8de6ec1122fde` (== pinned); land041 CLI SHA256 `92ba8888082f52feebbbe9881f5075fa198605964fdc9f9ca1cb2a9869b41c30`, daemon `ad804bc8bfe6926e72340e1abc055f2ba80c521f5b1ccfa59b78619220219559`.
- Evidence: `/home/kaden/hipfire-land041/scratch-land041/` (`{gfx1100,gfx1151}-matrix-{A1,B1}.{json,log,env,pre-pids,post-pids}`, `{gfx1100,gfx1151}-wt2-{A,B}.{log,kldseq}`, `gfx1100-battery.json`, `battery-serve.log`); matrix JSONs + battery JSON mirrored at `scratch-land041/hipx/` in the local worktree.

## Finish

- All gates above passed with zero failures; no scope was reduced. `mq4-lloyd` fast-forwarded to this commit on hipx and GitHub `Kaden-Schutt/hipfire` `mq4-lloyd`. The local `mq4-lloyd` checkout (`wt-lloyd`) stays at `34ccae74b`: it holds uncommitted user edits (`hipfire-arch-qwen35/src/speculative.rs`, `hipfire-runtime/examples/dflash_spec_demo.rs`) from outside this session, so no stash/commit/ref move was made there. No formatters or workspace suites were run; only the scoped `q8_flash` CPU tests.
