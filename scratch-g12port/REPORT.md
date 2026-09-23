# G12Port: gfx11 non-GEMM levers ported to exact gfx1201 (R9700)

Branch `g12-port` from `mq4-lloyd` `2afc4a294`, local worktree `wt-g12port`. Current artifact
`qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq` (local path), fp8 KV default (VMM for serve/bench,
legacy fp8 in the evaluator).

The branch carries the gfx11 lever commits it builds on, cherry-picked unchanged so a merge with
`stack-1` is clean: F1-lite `fb5aa2b96` and the β/α fold `bb3d19cd7`/`3a9925fb4`/`beb78dd3b`/`0c48c5919`.
Both are gated to gfx1100/gfx1151 there, so the gfx1201 behaviour of the base build equals
`2afc4a294`'s: its WT2 sequence is byte-identical to G12Prof's `2afc4a294` run (md5
`9d0e860f41db992820ebdc9483c0a041`). The `gfx11-smallfixes` commits are **not** carried (SmallFixes3
killed both on gfx11; neither survives on gfx1201 either).

## Verdicts

| Lever | gfx1201 form | Exact | Standalone (N=8192) | pp8192 trace (card-A) | WT2 c24 fp8 | A/B | Verdict |
|---|---|---|---|---|---|---|---|
| **F1-lite h-stream** | SILU epilogue in the gfx12 iu4 SET tile (gate/up 16-row groups interleaved by address, 2-group raster) + `_gfx12` h-producer | yes | producer 2.13 → 1.15 ms/call; fused GEMM ≈ SET pair | GPU 2244.3 → 2197.1 ms (**−47.2**); server 3676.1 → 3751.6 tok/s | 0.083278, byte-identical | card-A ABBA 3606.2 → **3692.1** (**+2.38%**) | **KEEP** (`94d4032fc`) |
| β/α fold into Z SET | widened Z SET whose tail tile scatters β/α in its epilogue (no split kernel, no widened buffer) | yes (byte-identical on gfx1201) | Z+β+α 2.15–2.21 → 2.02–2.19 ms/layer | GPU +3.8 ms; 3676.1 → 3657.8 | byte-identical | card-A 3619.1 → 3624.1 (+0.14%); card-E on top of F1, 2 pairs: 3719.4 → 3719.2 (**−0.006%**) | **KILL** (flat; patch kept in `dropped/`) |
| Residual GEMV tight grid | launch ceil(M/2) blocks of the two-row wave32 kernel | yes | down+out ≈ −1% (0.0713–0.0759 → 0.0709–0.0752 ms) | decode only | n/a (prefill scoring) | card-E 2 pairs: tg128 +0.26% (slow card state) / −0.04% (fast state); pp8192 −0.01% | **KILL** (< 1% tg128 bar) |
| RMSNorm keep-X | `HIPFIRE_RMSNORM_KEEP_X` AWQ-i4 producer twin | yes | **0.415–0.488 → 0.562–0.624 ms (+29–35%)** | — | — | — | **KILL** (standalone) |

Kill switches: `HIPFIRE_F1LITE=0` (shared with the gfx11 lever; gfx1201 arm gated on exact
`gfx1201` + symmetric iu4 route + `HIPFIRE_IU4_SYMFOLD`), `HIPFIRE_MQ4V2_RESIDUAL_TIGHT_GRID=0`.

## 1. F1-lite on gfx1201 (kept)

**Design.** The gfx11 variant (f32 h + slimmer producer) ports directly; an IU4-emitting epilogue is
impossible for the same reason (FWHT-256 group > the tile's 64 h rows).
- `kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx12.hip`: the body becomes `template <int EPI>`
  (SET/ADD/SILU). SILU: virtual row group g8 of a 128-row tile is gate (even) or up (odd) of h rows
  `blockIdx.x*64 + 16*(g8>>1)`, so each wave's rg=0/rg=1 accumulators are g and u of the same 16 h rows
  with the same lane map. W staging and SZ metadata pick G or U by a wave-uniform bit
  (`readfirstlane`), so loads keep SGPR bases. The epilogue computes `IU4_SILU_MUL` (spelled as the
  producer's `SILU_MUL`) and streams h through the SET's LDS transpose in 4 phases.
- A two-group raster (`IU4_SILU_RGROUPS=2`): without it the fused grid streams 2×47 MB of weights per
  token-tile column and loses the MALL residency one SET has; standalone the ungrouped kernel ran
  13.4 ms vs the pair's 12.2 ms, grouped 12.3/11.6 ms (rg 2/4/8 equal).
- Existing `full_set`/`full_add`/residual entries are **instruction-identical** to `2afc4a294`
  (`build_mods.sh` disassembly diff; 182 VGPR, 0 spill for all entries).
- `fused_silu_mul_mq_rotate_awq_i4_hin_gfx12` (`FUSED_SILU_MUL_MQ_ROTATE_AWQ_I4_HIN_GFX12_SRC`), the
  `_gfx12` twin of the gfx11 h-producer; `fused_silu_mul_mq_rotate_awq_i4_gfx12` itself is
  instruction-identical.
- Host: `Gpu::gemm_gate_up_silu_mq4g256v2_iu4_gfx12` (M%128, K%256, symmetric route);
  `f1lite_ffn_eligible` also admits the gfx12 silu-quant producer gate.

**Exactness.** Standalone oracle (`g12-host oracle`, JIT argv/concatenation): `ORACLE_PASS`, 0
differing bytes on `block_i4_128` (with and without the x_rot store) and on x_rot f32 for
M17408/K5120/N1024, partial N=1000, N=64 odd tiles, and a silu-specials case with 516,555 |g|>88.
Production: WT2 c24 fp8 md5 `9d0e860f…` byte-identical with the runtime-JIT'd h-producer present in
the kernel cache; the traced pp8192 request emits the same token.

**Trace (card-A, one warmed uncached pp8192 request each).**

| Kernel | base calls / ms | F1 calls / ms |
|---|---:|---:|
| gate/up `full_set_symfold` | 128 / 742.2 | — |
| `gemm_mq4g256v2_gate_up_silu_mmq_iu4_symfold` | — | 64 / 758.3 (+16.1) |
| `fused_silu_mul_mq_rotate_awq_i4_gfx12` | 64 / 135.7 | — |
| `fused_silu_mul_mq_rotate_awq_i4_hin_gfx12` | — | 64 / 73.0 (−62.6) |
| Total GPU | 3729 / 2244.3 | 3665 / 2197.1 (−47.2) |

The fused GEMM costs +2.2%: the exact SILU (ocml `expf` + IEEE divide, 64 per lane per tile) is VALU
the SET pair doesn't pay (standalone the SILU-free variant is 1.5% faster), partly offset by storing one
stream instead of two. It cannot be cheapened without changing h's bits.

**A/B (card-A, `hipfire bench --matrix --pp 8192 --ctx 128 --tg 128`, warmup + ABBA).** Process medians
base 3597.8 / 3614.6, F1 3687.0 / 3697.2 → **+2.38%**; tg128 flat (36.42/36.47 vs 36.41/36.45).

## 2. β/α fold (killed on gfx1201)

gfx1201 already runs β/α through the iu4 SET (96 single-row-tile launches, ~5.8 ms), so the gfx11
saving (MW4 tail + convert) does not exist here, and the gfx11 split kernel would cost ~30 ms (a full
Z copy). The port therefore scattered β/α from the Z SET's tail tile in-epilogue (exact, byte-identical
WT2). It is still flat: the extra row tile adds half a dispatch round to a 24-round grid (2 WG/CU ×
64 CU), which costs about what the two small launches did (trace: fold SET 102.5 ms vs Z+β+α 101.2 ms).
Card-E A/B on top of F1 (2 pairs): 3719.4 → 3719.2 (−0.006%). Dropped; the code is preserved as
`dropped/gfx1201-betaalpha-fold.patch` (applies on `94d4032fc`).

## 3. RMSNorm keep-X (killed)

Standalone, byte-identical, but 29–35% slower in every process (0.415–0.488 → 0.562–0.624 ms/call,
K5120 N8192): the (K+256)·4 B LDS staging cuts occupancy, while gfx1201's re-read of the 20 KB row is
served from cache. Not ported.

## 4. Residual GEMV tight grid (killed)

The generic `gemv_mq4g256v2_residual` owns rows `2*blockIdx.x` and `+1`, so half of gfx1201's M-block
grid exits immediately. Contracting to ceil(M/2) is exact (oracle: M5120 K17408/K6144, odd M2049) and
~1% faster per call standalone, i.e. ~45 µs/token at most. Card-E, bin-cand2 with
`HIPFIRE_MQ4V2_RESIDUAL_TIGHT_GRID=0` vs default, warmup + 2 ABBA pairs. tg128 is bimodal on this card
(28.7 vs 36.5 tok/s per process, independent of arm; G12Prof saw the same state on card-C):

| State | off (process medians) | on |
|---|---|---|
| slow | 28.69, 28.72 | 28.77, 28.80 (+0.26%) |
| fast | 36.49, 36.52 | 36.48, 36.50 (−0.04%) |
| pp8192 | 3727.9, 3730.7, 3724.5, 3722.3 | 3726.0, 3733.9, 3723.6, 3720.1 (−0.01%) |

Below Main's ≥1% tg128 bar: dropped (not committed).

## Method and files

- Cards: card-A (standalone, WT2, traces, first A/Bs) until Main's split; card-E (fold and grid A/Bs).
  Env per process: private `HOME=/home/kaden/.hipfire-homes/g12port-<card>`, kernel cache under it,
  `HIPFIRE_MODELS_DIR`, `HIPFIRE_GRAPH=1`, UUID visibility; every other `HIPFIRE_*` scrubbed.
- `build_mods.sh`: builds the gfx1201 modules exactly as the JIT concatenates them, diffs unchanged
  entries' disassembly against `2afc4a294`. `g12-host.cpp`: oracle and timing harness.
- `run_wt2.py`, `run_trace.py` (rocprof-wrapped serve, fp8 VMM asserted), `trace_delta.py`,
  `run_ab.py` (warmup + ABBA fresh processes, daemon pinned per arm).
- Evidence: `logs-oracle-A.txt`, `logs-time-A.txt`, `logs-time-raster-A.txt`, `wt2/`, `trace/`
  (`deltas.txt`), `ab/`.
- Binaries: `bin-base` (2afc4a294-equivalent on gfx1201), `bin-cand` (F1+fold), `bin-cand2`
  (F1+fold+grid), `bin-f1` (F1 commit; also `target/release` for the card-B guard). Binaries, modules and raw rocprof CSVs are git-ignored.

## Card-B guard (`scripts/guard_gfx1201_baseline.py`, floor 3,620, decode ±1% of 36.5)

Binaries: `target/release` built from `94d4032fc` (= `bin-f1`). Card-B `GPU-e475645fe0200397`, private
home `g12port-guard`.
- Run 1 (`guard-run1/`): pp8192 medians 3,728.7 / 3,717.5 / 3,714.5, decode 36.44 / 36.48 / **28.74** →
  FAIL on the decode check only. The 28.7 tok/s state is the card-level bimodal decode seen on every
  card today in both arms of every A/B (G12Prof card-C at `2afc4a294`; card-E nogrid/grid and
  nofold/fold arms alike); F1-lite does not run in decode (prefill, n ≥ 64 only).
- Run 2 (`guard/`): pp8192 medians **3,725.4 / 3,717.6 / 3,724.0**, decode 36.44 / 36.48 / 36.50 →
  **PASS**. Post-run `rocm-smi --showpids`: no G12Port process on any card.
