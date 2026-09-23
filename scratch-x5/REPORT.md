# X5 gfx11 production integration

Branch `gfx11-x5`, based on `mq4-lloyd` at `e5f944a3a`. Baseline binary: `/home/kaden/hipfire-prof040/target/release/{hipfire,daemon}` from that pinned production tree. Candidate: this worktree's native hipx release binaries. Model: `/home/kaden/.hipfire/models/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq`. Design, original standalone oracle, and rejected alternatives: `/home/kaden/hipfire-prof040/scratch-halogemm2/DESIGN.md`.

**Decision:** enable measured full-prefetch X5 SET and ADD on gfx1151 and gfx1100 for full M128/N128 tiles on the symmetric, eager column path; other routes retain the incumbent. The independent whole-model ship gate passes on both cards. This is a useful smaller win, not a claim to reach 1,000 tokens/s: the standalone 1.35x SET headroom gate in DESIGN failed. The R256 successor screen was independently killed on resource and large-K ADD gates; it is not integrated.

## Cutover and invariants

- `kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip` has the measured X5 body isolated under `IU4_STAGE_X5`, keeping the ordinary LF16 symfold implementation in the non-X5 compilation. Separate private SET/ADD symbols retain the existing ABI, M128×N128 output ownership, `[32,16,1]` block, 30,720-byte LDS, weight layout, four group barriers, ascending K128 integer/FP folds and single ADD residual. Aligned b128 X packet loads and a fifth dword in half the waves stage X; the next half is prefetched before current WMMAs and retired with `vmcnt(0)` afterward. No final-half overread.
- `kernels.rs` registers a separate `IU4_SYMMETRIC_FOLD=1, IU4_STAGE_X5=1` module. `gemm.rs` selects it only when gfx1151 or gfx1100 LF16 shape, symfold active, `M%128==N%128==0`, positive dimensions, K%256==0 (existing precondition), and non-capture eager column route. `HIPFIRE_IU4_X5=0` retains the incumbent. No shared default translation-unit or compiler-policy change. The gfx1100 K6144 packet-only aggregate advantage was order-sensitive; no unvalidated shape split was added.
- The standalone finite oracle in DESIGN compared 262,144 outputs bit-exact with unequal scales and nonzero residual; that is not a universal proof. The integrated paired WT2 sequence comparison below is an independent model-level check. The compiler's actual FP fold contraction was preserved; no `-ffp-contract=off` was introduced for this GEMM.

## Resource and trace gates

Native `cargo build --release` succeeded after integration. Runtime uses HIP `--genco --offload-arch=<arch> -O3 --no-offload-compress`; an isolated host/device translation unit altered register allocation, so **runtime-compiled production HSACOs** in fresh HOME caches were probed with `scratch-x5/resource.cpp`, not that misleading combined object. Both actual SET and ADD entries on each architecture: **92 VGPR, zero private/spill, 2 blocks/MP, 32 waves, LDS 30,720 bytes**. The standalone exact-source compiled ISA also has 92 VGPR/zero spill; see `x5-set-gfx1151.isa` for lookahead load and WMMA ordering. Packet ownership enumeration covered 2,304 distinct X dwords and aligned full-N rows.

| Card / environment | Runtime X5 HSACO in `scratch-x5/<arch>-8192/home/.hipfire_kernels/<arch>/` | Traced X5 SET | Traced X5 ADD | Sum vs pinned production GEMM | Uncached 8192 trace wall |
|---|---|---:|---:|---:|---:|
| gfx1151, ROCR=1 HIP=0 | `gemm_mq4g256v2_residual_mmq_iu4_gfx11_x5_symfold.c4e902d85a7eb0e1.hsaco` | 272 calls, 4,878.067 ms | 128 calls, 2,460.373 ms | 7,338.441 vs 8,239.284 ms; 900.843 ms saved (10.93%) | 10,042.243 ms |
| gfx1100, ROCR=0 HIP=0 | `gemm_mq4g256v2_residual_mmq_iu4_gfx11_x5_symfold.936038d287cc3b6f.hsaco` | 544 calls, 1,805.340 ms | 256 calls, 861.997 ms | 2,667.336 vs 2,838.894 ms; 171.558 ms saved (6.04%) | 3,608.909 ms |

`run_profile.py`/`inventory.py` drive and record traces; both processes asserted `GPU dev 0: <arch>`, `KV cache: Q8 vmm (` and both visibility variables. Trace reports and raw profiles are under `scratch-x5/<arch>-8192/`. Pinned sums are the prior production traces in DESIGN, not a contemporaneous clock-normalized experiment.

## Whole-model matrix gate

Each card ran fresh-process `A B B A A B B A` against the pinned baseline, with fresh HOME and default runtime knobs. Every arm used `hipfire bench <model> --matrix --pp 512,8192 --ctx 128 --tg 128 --spec off --runs 3 --warmups 1 --kv-mode q8 --kv-backend vmm --json`; the script asserted effective q8/vmm and exact device arch. `scratch-x5/abba.py` and `summarize_abba.py` retain per-run JSON/logs and `scratch-x5/<arch>/abba/summary.json`. Values below are pooled mean candidate throughput divided by pooled mean baseline throughput, with per-cycle ratios in parentheses.

| Card | pp512 | pp8192 | tg128 decode | Gate |
|---|---:|---:|---:|---|
| gfx1151 | **1.18514×** (1.19055, 1.17973) | **1.09131×** (1.08422, 1.09851) | 0.99979× (−0.021%) | Pass |
| gfx1100 | **1.05908×** (1.05694, 1.06123) | **1.04744×** (1.04577, 1.04913) | 0.99952× (−0.048%) | Pass |

Both prefill rows exceed the +1.5% ship threshold; no row is >1% slower, and decode is within 1%. The gfx1151 pp8192 pooled result is approximately 813.6 vs 745.5 tok/s, not 1,000 tok/s. The matrix includes synthetic prefill/decode, not a guarantee for every prompt length: N not divisible by 128 falls back.

## Exact quality and fallback

The paired baseline/candidate `target/release/examples/eval_hipfire` runs used `--model <above> --ref /home/kaden/kldrefs/qwen3.8-27b.ref_wt2.bin --output <arm>.kldseq --max-chunks 24 --kv-mode q8 --kv-v q8`, each with its card-specific visibility environment and separate HOME. Both scored 24,552 tokens in 24 chunks. Baseline and candidate **byte-identical** KLD sequence files (`cmp` pass):

| Card | Baseline and X5 displayed slice-mean KLD | Shared SHA256 of each KLD sequence |
|---|---:|---|
| gfx1151 | 0.076901 | `eefd7d257d736b9bdaddf1246e93217c710eb4291dfec3cafd7f62c863aaeb25` |
| gfx1100 | 0.076879 | `8f2b94bb904603bc53d17f7877c306ee4aacd0afc7c71acbf80a60bea0b1e2b2` |

An initial Halo evaluator invocation used evaluator defaults (asym3/legacy) and yielded 0.088467; it was excluded, then rerun with the required explicit q8/q8 recipe. The missing `/home/kaden/harness/manifest.json` only skipped the evaluator's reference SHA check; both arms loaded the same named reference and produced identical sequences.

Each architecture then ran a manual X5 server with its visibility variables and fresh HOME, verified `GPU dev 0: <arch>` and `KV cache: Q8 vmm (` in logs and both variables in the daemon's `/proc` environment. `scripts/serve_harness.py --no-spawn --mode battery --max-think-tokens 1 --max-tokens 512` gave **5/5** coherent, nonempty `finish=stop` responses for code, arithmetic, factual, prose and instruction prompts on each card; no runaway, attractor or retrieval miss. Full outputs: `scratch-x5/gfx1151-battery.json`, `gfx1100-battery.json`. `odd_fill.py` submitted `benchmarks/prompts/ttft_511.txt` (511 prompt tokens); each card returned a coherent `finish=stop` answer identifying the truncated merge-function question. This checks the N%128 fallback, not X5 timing. Responses: `scratch-x5/<arch>-ttft511.json`.

GPU access was leased/coordinated with Main and peer timers. Halo used `ROCR_VISIBLE_DEVICES=1 HIP_VISIBLE_DEVICES=0`, XTX used `0/0`; no HIP3. Native builds were done outside other agents' timing. The final `rocm-smi --showpids --showuse --showmemuse` postcheck showed zero VRAM allocated on both cards and only the persistent gpusentry process; the lease was released.
