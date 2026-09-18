# iu4stage — staging-block census + b64 rewrite (Step 1–2)

HEAD base: 9fbaa98c1 (A3 double-buffer + B LDS-broadcast). Only file changed:
`kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx12.hip` (gemm.rs LDS untouched).

## Step 1 — ISA census of the staging block (gfx1201 `-save-temps` + unbundled disasm)

Method: TU = `block_i4_128_quant.hip` + kernel (byte-identical to iu4db
`tu-changeAB.hip`), `hipcc --offload-arch=gfx1201 -O3 -save-temps -c`,
device code unbundled via `clang-offload-bundler`, `llvm-objdump -mcpu=gfx1201`.
h-loop is rolled, so one static trip = one dynamic (g,h) half-block = 2 slabs.
Trip located by WMMA clusters (full_add body).

### BEFORE (per (g,h) trip, dynamic; 2 slabs, 256 lanes)

| op | count | widths | bytes/instr |
|---|---|---|---|
| global loads | 18 | 18x b32 | 4 |
| LDS stores | 18 | 18x b32 | 4 |
| vector addr VALU | ~47 (11x v_mad + 36x v_add_co + lshl/mul) | — | — |
| EXEC masking | 10x s_and_saveexec + 10x s_cbranch_execz (+ restore) | — | — |
| waits in staging path | 4 (2x s_wait_loadcnt_dscnt + 2x s_wait_loadcnt) | — | — |
| barriers | 4 (B0/B1/B2a/B2b) | — | — |
| v_readfirstlane / SGPR base loads | 0 | — | — |

Per slab per lane: 9x b32 loads + 9x b32 stores, ~24 addr VALU, ~5 masked
branches. Every load recomputes a 64-bit vector address (`v_mad_co_i64_i32` +
2x `v_add_co`) under an EXEC mask from `if (tok < N)` / loop predicates.

### Alignment verdict (brief's b128 ideal is impossible)

- Xq nibbles: 72 B blocks (8 B header) → every 16 B chunk at 8-mod-16. b128
  misaligned; b64 chunks all 8-aligned. Ceiling = b64.
- W nibbles: 136 B groups (8 B header) → slab starts at +8/+40/+72/+104, all
  8-mod-16. Ceiling = b64.
- LDS stride-40 rows: row base 40r = 0/8-mod-16 alternating. Ceiling = b64.
- True ideal: 4x b64 loads + 4x b64 stores per lane per slab (+1 b32 DS/SZ),
  zero per-load VALU, zero masks. (Brief's 2x b128 assumed 16 B alignment
  that the 8-byte headers rule out.)

## Step 2 — rewrite (b64-coalesced staging, hoisted offsets)

- `u32x2_t` (ext_vector) staging words; `A_pf/W_pf` 4x u32 → 2x u32x2.
- New lane map: 2 rounds; round r covers 64 rows (r*64 + tid>>2), quad tid&3.
  Consecutive lanes share a row with consecutive quads (coalesced globals,
  sequential LDS banks, same 2-way floor as before). Slab covered exactly once
  (same LDS bytes as before → oracle-safe).
- Offsets (`st_avoff/st_wvoff/st_ldsoff` + `st_aok`) computed ONCE at kernel
  top (loop-invariant); per-trip SGPR bases carry kb/g/h. OOB clamps to last
  valid index; A/DS/SZ values zero-select via branch-free ternary (exact old
  semantics incl. partial tiles); W keeps clamped value as before.
- DS/SZ-next converted to clamp-not-branch (1 word each, stay b32).
- K-loop, fold, SB=1 bundle, barriers, LDS layout all untouched.

### AFTER (per (g,h) trip, dynamic)

| op | count | widths | bytes/instr |
|---|---|---|---|
| global loads | 8x b64 + 2x b32 (DS/SZ) | b64 + b32 | 8 / 4 |
| LDS stores | 4x dual-b64 (stride64, 8 values) + 2x b32 | 16 B/instr publishes | 16 / 4 |
| vector addr VALU | ~20x v_add_co (1-2/load, SGPR+VGPR form), 0 MAD | — | — |
| EXEC masking | 0 | — | — |
| cndmask selects | ~11 (A/DS/SZ zero-selects) | — | — |
| waits in staging path | ~10 fine-grained loadcnt (per-pair dep waits, not drains) | — | — |
| barriers | 4, unchanged | — | — |

Whole-body static: global_load_b32 83→68, +12 b64; ds_store b32-family 46→35,
+6 stride64_b64; WMMA 32, barriers 21, fragment ds_loads — all unchanged.
Two b64 loads use the SGPR-base form (`v, s[6:7]`); the rest use the
vector-address form with uniform parts in SGPRs (compiler split the base but
kept 1-2 VADDs per load — residual VALU, acceptable).

Step 3 (ablations) NOT needed: staging was far from ideal and the rewrite
closed it. No DS/SZ-plane or bandwidth ablation required.

## Gates 1–4

1. Oracle 9/9 PASS (`oracle-new.log`): cpu_bitwise=OK mism=0 all 9 (gate/set,
   gate/add, down/set, down/add, m48tail, n80cols, k256/set, k256/add,
   m100n100); repeat_identical=OK.
2. Metadata: VGPR 196 (was 212) ≤ 256, spill 0/0, scratch 0, LDS 19456
   unchanged, OCC 3 WG/CU both symbols (`occ-new.log`).
3. Census before/after: tables above (`base-dis.txt`/`body-add.txt` vs
   `new-dis.txt`/`body-new.txt`).
4. TIME N=512, medians over 4 runs (us/call):

| case | 9fbaa98c1 ref | new (r1..r4) | ratio |
|---|---|---|---|
| gate set | 657 | 525.4 / 518.7 / 525.3 / 524.8 | 0.80 |
| gate add | 669 | 540.0 / 537.1 / 539.5 / 543.2 | 0.81 |
| down add | 638 | 508.6 / 510.6 / 517.5 / 510.8 | 0.80 |
| qkvza set | 636 | 508.1 / 511.7 / 511.2 / 515.1 | 0.80 |
| qkv set | 538 | 426.9 / 429.2 / 431.2 / 434.0 | 0.80 |

−20% uniform (±1.5% run spread). 131–138 TOPS → 168–180 TOPS (31–33% of peak).
Gate_up −20% ≥ 15% → gate 5 TRIGGERED (below).

vs incumbent (same-session refs from iu4db: s2bt8 gate_up fused 1679.3,
residual 850.1): gate per-matrix 2x525/1679 = 0.63x; residual 511/850 = 0.60x.

## Gate 5 — full validation (triggered)

### KLD pins (bit-for-bit) — PASS

Ref `/home/kaden/kldrefs/qwen3.8-27b.ref_wt2.bin` (sha256
`8c545178…a43234a`, HFKLDR v1, 2048 ctx, 24 chunks — the same file every pin
today was made with). Tree-built eval in wt-iu4, plan §1.2 env
(`HIPFIRE_GRAPH=0 HIPFIRE_NORMALIZE_PROMPT=0 HIPFIRE_LLOYD_GFX12=1
HIPFIRE_GFX11_MQ4V2_IU4=1`, `--kv-mode q8 --kv-v q8 --scoring-mode prefill`).
Receipts: `iu4stage/kldpin/c{1,2,24}.{kldseq,stdout,stderr}`.

| chunks | pin | got | verdict |
|---|---|---|---|
| 1 | `032ebad84f2c1dd5e2980fa805215ac0` | `032ebad84f2c1dd5e2980fa805215ac0` | BIT-IDENTICAL |
| 2 | `dc7e53181662271780374f0a85fd7732` | `dc7e53181662271780374f0a85fd7732` | BIT-IDENTICAL |
| 24 | KLD `0.063410` | KLD `0.063410` (NLL 1.864044, PPL 6.4498) | EXACT |

### Interleaved bench OFF/ON/OFF/ON — medians (tok/s)

`HIPFIRE_GRAPH=1 HIPFIRE_LLOYD_GFX12=1 [HIPFIRE_GFX11_MQ4V2_IU4=1]
./target/release/hipfire bench qwen3.8:27b-mq4-xt --matrix
--pp 512,2048,8192,32768 --ctx 128 --tg 64 --spec off --runs 3 --warmups 1
--kv-mode q8 --json`. Receipts: `iu4stage/bench/{off1,on_a,off_b,on_b}.log`
(`on1.log` DISCARDED — see stale-daemon note).

| row | OFF-1 | ON_a | OFF_b | ON_b | prior ON | ON/OFF |
|---|---|---|---|---|---|---|
| pp512 | 1475.3 | 2037.1 | 1472.4 | 2029.3 | 1627 | +38% |
| pp2048 | 1406.3 | 1977.9 | 1442.7 | 1972.6 | 1595 | +39% |
| pp8192 | 1298.8 | 1791.2 | 1338.8 | 1786.0 | 1469 | +36% |
| pp32768 | 1008.4 | 1285.3 | 1033.8 | 1283.9 | 1111 | +26% |
| tg64@128 | 28.79* | 36.50 | 36.48 | 36.49 | — | — |

ON vs prior-ON: +25% / +24% / +22% / +16% (all >> 1.5% reality threshold).
ON_a vs ON_b agree to ≤0.4%; OFF pair agrees to ≤3% (session drift on long
rows). *OFF-1 decode 28.79 is lone-run noise (its pp rows match OFF_b; the
other three decodes agree at 36.48–36.50).

Stale-daemon note (finding, no data lost): the first ON run (`on1.log`,
1629/1593/1468/1112) used `target/release/daemon` built 03:04, predating
Change A — the kernel cache proves it compiled V2 source (no
`Double-buffered` marker). `hipfire` is NOT the daemon: bench spawns
`target/release/daemon` via `find_daemon`. Rebuilt post-commit; md5s
`hipfire 17fcd6b07f141b5451de7a2d628d01db`,
`daemon 32d48a086ebebc9a380d3e14a13b350d`. Freshness proof: the ON runs hit
cache entry `….3271eb5c00362293.hip`, byte-identical to the committed TU
(`st_avoff` marker present). Lesson for the gate: any ON row within 2% of
1627/1595/1469/1111 is stale — ON_a/ON_b at ~2030/1975 are the real rows.
Side observation: V2 end-to-end (1629) ≈ A+B (1627) — the overlap work did
not move prefill; only the staging-width cut did (+25%).

### decode ≥ 36.4 — PASS (36.50 / 36.49, clean logs, 0 errors)

### serve battery ON (real harness) — 5 turns, 0 empty, 0 attractor

`python3 scripts/serve_harness.py --model …/qwen3.8-27b.mq4-xt
--tag qwen3.8:27b-mq4-xt --kv q8 --mtp off --dflash off --mode battery
--sampling registry --thinking off --max-tokens 256 --seed 0 … --port 11531`
with `HIPFIRE_DAEMON_BIN=…/wt-iu4/target/release/daemon` (=md5
`32d48a08…`, the post-commit rebuild) and `HIPFIRE_GFX11_MQ4V2_IU4=1`.
Receipts: `iu4stage/serve-battery/battery.{json,log}`,
`harness.{stdout,stderr}`. (An earlier `test-serve.sh` run is NOT the
acceptance route and is superseded; its 7/8 (pre-existing Test-4 think-span
failure, identical with iu4 OFF) is retained only in `serve-on*.log`.)

| turn | finish | gen | think/ans words | decode | flags |
|---|---|---|---|---|---|
| t1 code | stop | 175 | 42/47 | 36.4 | — |
| t2 reason | stop | 156 | 22/45 | 36.4 | — |
| t3 factual | stop | 171 | 80/42 | 36.5 | — |
| t4 prose | length | 256 | 150/44 | 36.4 | RUNAWAY* |
| t5 instruct | stop | 220 | 103/55 | 36.4 | — |

Turns 5, runaway 1 (*t4 hit max-tokens 256 — truncation, text clean),
empty 0, attractor 0, retrieval_miss 0. Avg prefill 570.6 tok/s, decode 36.4.
Excerpts: t1 `def merge_sorted(a, b):` — correct two-pointer merge with
docstring and tail extend; t3 "Seasons are caused by Earth's axial tilt of
about 23.5 degrees…" — correct. Full texts in `battery.json`
(`assistant_content` per turn).
