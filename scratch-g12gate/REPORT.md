# gfx1201 integrated landing gate, 2026-09-23

**Decision: PASS.** `g12-stack-v2` merged the official gfx11/shared `kaden/mq4-lloyd` `4bd33ce40d0718281e42b85e55d0e3037e7bf2f3` into the gfx1201 stack `29d06b5d0`; merge commit `232049b2f1b5114376e307726ba3c22c4e1f5928`. Both release binaries built successfully in dedicated worktrees (`cargo build --release -p hipfire-cli -p hipfire-daemon`), followed by a route-exact evaluator build (`cargo build --release -p hipfire-runtime --example eval_hipfire --features deltanet`). Baseline tree `wt-g12base` is detached at exactly `4bd33ce40`.

Model: `/home/kaden/qcal/qat/v25/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq`, 14,987,185,152 bytes; guard-pinned MD5 `2cfe88923b3671ca16a8de6ec1122fde`. Baseline CLI/daemon MD5 `49560375a08e77c05e48817b2faeb939` / `45289a0bc0f2ae607b9216c9fce3744d`; candidate `db8ddad2430a30814768db935e437f20` / `3881e0945acce62e57be43db6c820f11`. Process-private homes and kernel caches, `HIPFIRE_MODELS_DIR=/home/kaden/.hipfire/models`, `HIPFIRE_GRAPH=1`, both visible-device variables pinned to the card UUID; stale `daemon.pid` removed. Fresh matrix logs confirm gfx1201 and `KV cache: Fp8 vmm (...)`. No performance work was timed on the shared card-D. Pre/post gate process inventories found no owned process remaining; a separate G12Gdn2 daemon ran on card-C later.

## Merge conflicts (all 22 marker blocks in eight files)

The merge retained the already-landed gfx11/shared behavior and added only gfx1201 arms. `ours` means `29d06b5d0`, `theirs` means official `4bd33ce40`. The marker line positions are those reported by Git's initial merge, before resolution:

| File | Conflict blocks and resolution |
|---|---|
| `CHANGELOG.md` | #2, lines 24–32: both, preserving the gfx1201 F1 note and all four official gfx11 entries (beta/alpha fold, F1, GDN, V2C ADD touch). |
| `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs` | #18, lines 158–263: ours, which is the landed F1 eligibility/producer body extended with the exact-gfx1201 AWQ-IU4 alternative; gfx11 sidecar branch unchanged. |
| `crates/rdna-compute/src/gemm.rs` | #20, lines 19903–19945: theirs, retaining official V2C ADD-touch selection. #21, lines 19992–20031: theirs, retaining official V2B ADD-epilogue/residual-fold selection. #22, lines 31798–32069: ours for the gfx1201 early return and gfx12-only F1 helper, with the official `flush_residual_fold()` and its original comment restored exactly in the unchanged gfx11 path. |
| `crates/rdna-compute/src/gemv.rs` | #19, lines 3971–4128: ours, the official h-producer unchanged on gfx11, with an exact-gfx1201 source/symbol branch. |
| `crates/rdna-compute/src/kernels.rs` | #17, lines 1437–1475: ours, retaining the official common h-producer export and adding the separate gfx12 export. |
| `kernels/src/gemm_mq4g256v2_residual_iu4_v2b.gfx11.hip` | #3 (39–70), #4 (159–180), #5 (184–192), #6 (448–454), #7 (458–464), #8 (466–476): **theirs for each**; all official gfx1151 V2B F1/ADD-epilogue changes retained. |
| `kernels/src/gemm_mq4g256v2_residual_iu4_v2c.gfx11.hip` | #9 (38–67), #10 (155–176), #11 (180–188), #12 (356–406), #13 (452–458), #14 (462–468), #15 (470–480), #16 (482–501): **theirs for each**; all official gfx1100 V2C F1/ADD-touch changes retained. |
| `scratch-f1lite/f1-host.cpp` | #1, lines 1–885 (add/add): theirs, preserving the official gfx11 standalone oracle host. |

**gfx11 audit:** `git diff 4bd33ce40 g12-stack-v2 -- 'kernels/src/*gfx11*'` is empty. The sole `gemm.rs` diff relative to official is 90 added/4 deleted lines: a gfx1201-only early-return admission and its helper plus matching API documentation; the executed gfx11 method body, including residual-fold flush, is unchanged. **gfx11-affecting changes: zero.** `git log -p 4bd33ce40..HEAD` contains zero case-insensitive `r4d` occurrences.

## Card-A fresh-process ABBA ×2

`run_abba.py` compares the two release binary pairs on `GPU-9eb7aeda51c88ffd`. One independent discarded warmup for *each binary*, then `A B B A A B B A` (A official, B candidate); every measured process executes `hipfire bench <model> --matrix --pp 512,8192 --ctx 128 --tg 128 --spec off --runs 3 --warmups 1 --kv-mode fp8 --json`. The process value below is the median of its three resident samples; full per-process samples and VMM/arch logs are in `abba/`. Both arms use the same fixture; 8,192 synthetic little-endian u32 IDs are `10+i%1000`, stream MD5 `c596ea04a31b9977adeca9eda88f622c`.

| Process | Arm | pp512 tok/s | pp8192 tok/s | tg128 tok/s |
|---|---|---:|---:|---:|
| 1 | A | 3333.8 | 3645.5 | 36.4818 |
| 2 | B | 3487.6 | 3848.7 | 36.4824 |
| 3 | B | 3492.6 | 3847.8 | 36.4878 |
| 4 | A | 3334.9 | 3633.3 | 36.4969 |
| 5 | A | 3338.5 | 3633.3 | 36.5042 |
| 6 | B | 3472.5 | 3852.0 | 36.5075 |
| 7 | B | 3484.7 | 3853.0 | 36.4688 |
| 8 | A | 3329.6 | 3633.1 | 36.4776 |
| **Median** | **A → B** | **3334.35 → 3486.15 (+4.553%)** | **3633.30 → 3850.35 (+5.974%)** | **36.48934 → 36.48512 (−0.012%)** |

Every pp row improves; no row >1% slower. All eight decode processes are in the fast ~36.5 mode, with the fast-mode medians within 1%; this comparison did **not** observe a 28.7-mode process, although that mode has been observed historically in both arms. No decode is silently removed here. **ABBA gate passes.**

## Card-D WT2 quality

Candidate evaluator MD5 `1e35f7db4fd5762941a180d9a5602e52`. `run_wt2.py` on `GPU-6109a4cb5f833235` uses `eval_hipfire --model <model> --ref /home/kaden/kldrefs/qwen3.8-27b.ref_wt2.bin --kv-mode fp8 --kv-v q8 --scoring-mode prefill --max-chunks 24 --output wt2/candidate.kldseq`, scrubbed inherited `HIPFIRE_*`, graph off per evaluator. Log confirms `GPU dev 0: gfx1201`, `fp8/fp8 native KV`, **`KV cache: fp8-e4m3 backend=legacy`** (not the bench's VMM), `prefill_chunk requested=8192 admitted=8192`. All 24 chunks score 24,552 tokens: slice-mean KLD **0.083278**, mean NLL **1.886010**, PPL **6.5930**. Sequence MD5 **`9d0e860f41db992820ebdc9483c0a041`**, byte-identical to the pinned baseline. **WT2 passes.**

## Card-A manual-serve battery and TTFT-511

Manual `hipfire serve --model <model> --kv-mode fp8 127.0.0.1:11520` on card-A, candidate daemon pinned, private HOME/cache, graph on. Server log confirmed `KV cache: Fp8 vmm` and completed model prewarm. `python3 scripts/serve_harness.py --model <model> --kv fp8 --no-spawn --port 11520 --max-think-tokens 1 --max-tokens 512 --out scratch-g12gate/battery.json` uses the registry's recommended sampling. The five fresh-conversation genre prompts (code/reason/factual/prose/instruct) are **5/5 coherent** and finish `stop`; zero empty, runaway, attractor or retrieval miss. Code correctly merges already-sorted lists; reasoning gives 150+60=210 miles; factual answers Earth axial tilt in three sentences; prose is a four-sentence lighthouse story; instruction gives exactly five numbered tips. The first request includes cold JIT (TTFT 16.227s; decode 12.9tok/s), and is not used as a performance comparison.

`--prompt-file benchmarks/prompts/ttft_511.txt` admitted exactly 511 tokens, zero cached. With registry sampling it generated relevant analysis but hit the 512-token cap mid-sentence (`finish=length`, 5,846.4ms prefill, 87.4tok/s); this is recorded in `ttft-511.json`, not hidden. With the same exact fixture and cap but `--sampling greedy`, it completed a coherent explanation of the repeated `merge_sorted` and truncated final `a`, **finish=stop**, 294 generated tokens, prefill **145.1ms / 3522.0tok/s**, decode **36.4tok/s**, no empty/runaway/attractor (`ttft-511-greedy.json`). Manual server stopped cleanly afterward. **Battery and coherent TTFT-511 pass**, with the registry-sampling length caveat above.

## Card-B guard

`scripts/guard_gfx1201_baseline.py --output-dir scratch-g12gate/guard --home /home/kaden/.hipfire-homes/g12gate-b`, `GPU-e475645fe0200397`, floor **3620**. One discarded fresh process, then three measured independent processes. pp8192 medians **3881.1 / 3846.6 / 3844.1 tok/s**, minimum **3844.1** (> floor by **224.1**). Decode **36.4856 / 36.4517 / 36.4892 tok/s**, each within 1% of 36.5; logs confirm automatic fp8 VMM. **Guard passes.** The card was explicitly released to Stacker2 after the post-run inventory.

## Evidence

`run_abba.py`, `abba/summary.json` and all `abba/` process JSON/logs; `run_wt2.py`, `wt2/summary.json`, `candidate.log`, and `candidate.kldseq`; `battery.json`, both TTFT transcripts; `guard/summary.json` and fresh-process logs. All measured process records are retained alongside the report for source/fixture verification. No formatter, linter, or project-wide test suite was run during integration.
