# Credits

hipfire is an LLM inference engine for AMD RDNA GPUs. It builds on a long
chain of upstream research, drivers, and runtime code. This file
documents every significant source that informed the architecture, the
ROCm/HIP path, the kernels, and the ship-line behavior.

The shape of this file is lifted from
[ncdrone/rustane's CREDITS.md](https://github.com/ncdrone/rustane/blob/master/CREDITS.md).

## Foundational Sources

| Project | Author | What We Learned |
|---------|--------|-----------------|
| [autoresearch](https://github.com/karpathy/autoresearch) | Andrej Karpathy | Methodology pattern: `program.md` (strategy) > agent modifies one file > fixed eval > keep/discard > repeat. We adapt it for hardware/driver exploration; the "fixed eval" equivalent is the route matrix in [`docs/VALIDATION.md`](docs/VALIDATION.md) plus the serve/Redline harnesses it names (`scripts/serve_harness.py`, `scripts/redline_daemon_harness.py`, and related gates). |
| [rustane](https://github.com/ncdrone/rustane) | ncdrone | Rust-native FFI to private/undocumented hardware APIs via `dlopen`. Their `ane-bridge` > `metal-decode` > `engine` decomposition is what we adapted into `hip-bridge` > `rdna-compute` > `engine`. The CREDITS.md shape here is also lifted from theirs. |
| [Mesa (radeonsi / radv)](https://gitlab.freedesktop.org/mesa/mesa) | Mesa contributors | Open AMD GPU driver source. The gfx10 register headers (`sid.h`, `gfx10_format_table.h`) and the compute-relevant register documentation that ROCm does not publish. Reference for what gfx1010 can actually do at the hardware level vs. what ROCm chooses to expose. |
| [amdgpu kernel driver](https://gitlab.freedesktop.org/agd5f/linux) | AMD + upstream contributors | KMD ioctl surface for `/dev/dri/renderD*`, PM4 command buffer format, doorbell semantics. Backstop for the redline (direct-KMD) crate and for diagnosing firmware/driver mismatches. |
| [ROCm / HSA runtime](https://github.com/ROCm/ROCm) | AMD | The runtime stack hipfire FFIs into. `libhsa-runtime64.so` and `libamdhip64.so` loaded via `libloading`; we deliberately stay off the ROCm Python and ROCm userspace stacks at runtime. |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | ggerganov + upstream contributors | GGUF format reference for the import path; the MQ4/MQ3/MQ2 family is a parallel Magnum Quants line, not a fork, but the GGUF reader cribs from llama.cpp's parser. Also our standing prefill/decode comparison baseline and the source of most of the "what does this RDNA card do under llama.cpp" intuition. |
| [candle](https://github.com/huggingface/candle) | Hugging Face | Rust ML reference for tensor layout, safetensors import, and quantization-format plumbing. We do not depend on candle at runtime; it is the closest existing Rust-native reference for "what a clean inference engine looks like" and informs the engine crate's API shape. |
| [Lucebox DFlash on ggml](https://www.lucebox.com/blog/dflash27b) | Lucebox | Standalone C++/ggml/CUDA DFlash for Qwen 3.5-27B on a single RTX 3090. Concrete published numbers to target, n_gen-aware bench methodology, and the shape of Path C (DDTree wire-up). |
| [ds4](https://github.com/antirez/ds4) | antirez | Standalone C99 reference inference for DeepSeek V4 Flash. Source of truth for `crates/hipfire-arch-deepseek4`: MTP head wiring, Hyper-Connections head-reduction algebra, raw-SWA + compressed-KV cache layout, and the tail-only YaRN RoPE convention. Our forward pass matches its numeric outputs at temp=0 within FMA-order noise. |

## Rust Crates and Runtimes

| Crate / Runtime | Use |
|-----------------|-----|
| [libloading](https://docs.rs/libloading) | `dlopen` of `libhsa-runtime64.so` / `libamdhip64.so` from `hip-bridge` and `hsa-bridge`. The whole "no ROCm install pain" story rests on this. |
| [memmap2](https://docs.rs/memmap2) | Zero-copy weight load for safetensors / GGUF / HFQ4 blobs in the engine and quantize crates. |
| [serde](https://docs.rs/serde) / [serde_json](https://docs.rs/serde_json) | Config files, registry JSON, OpenAI-compatible HTTP API, daemon IPC. |
| [rayon](https://docs.rs/rayon) | CPU-side parallelism for quantization passes and tokenizer batch encode. |
| [byteorder](https://docs.rs/byteorder) | GGUF / safetensors little-endian readers. |
| [thiserror](https://docs.rs/thiserror) | Error type derivation in the bridge crates. |
| [image](https://docs.rs/image) | PNG / JPEG decode for vision-model preprocessing. |
| [libc](https://docs.rs/libc) | ioctl / syscall plumbing for the redline direct-KMD path. |

## Papers

Only papers whose findings shipped as concrete behavior. Read-and-mined
papers without ship-line impact are deliberately omitted.

Author attributions are intentionally omitted: arxiv author lists are
the authoritative record, and listing a single name here invites
miscredit. Click through for the canonical author list.

| Paper | Relevance |
|-------|-----------|
| [DFlash (arXiv:2602.06036)](https://arxiv.org/abs/2602.06036) | Speculative-decode method that ships in `crates/engine/src/dflash.rs`. Target layer fusion, non-causal bidirectional attention within block, post-FFN residual hidden extraction. Our 9B / 27B DFlash perf headlines come from this. |
| [DDTree (arXiv:2604.12989)](https://arxiv.org/abs/2604.12989) | Block-diffusion draft tree, best-first heap, ancestor-only verify mask. Algorithm 1 ships in `crates/engine/src/speculative.rs`; informed the Path C PRD and the gfx1100 tree-mode FA tuning. |
| [CACTUS (arXiv:2604.04987)](https://arxiv.org/abs/2604.04987) | KL-bumped acceptance threshold replacing Leviathan `min(1, q/p)`. Shipped as the `temp>0` rejection-acceptance path so DFlash on creative content is no longer penalized for the draft being distilled on argmax. |
| [Fail-Fast drafting (arXiv:2512.20573)](https://arxiv.org/abs/2512.20573) | Per-block confidence-gated speculation length. Informs the A3B DFlash default-off gate and the dynamic draft-length policy that collapses to AR when the draft is uncertain. |
| [MoBiLE (arXiv:2510.12357)](https://arxiv.org/abs/2510.12357) | Per-token big/little MoE expert switching. Frames the A3B 24 GB consumer-card OOM mitigation; the eviction-aware sidecar work tracks this paper. |

Papers we deliberately do NOT credit: Orion (rustane-relevant, not load-bearing for hipfire); S2D2, Fast-dVLM, MineDraft (read during DFlash recon, none shipped). Performative credit is worse than no credit.

## Contributors

Listed by merged-PR count, then PR date. Core author Kaden Schutt is
omitted from this section by convention; this list is for everyone else
who has shipped code.

This section is regenerated by `scripts/refresh-credits.sh` (run after
new PRs merge). Hand-edits inside the auto block will be overwritten.

<!-- contributors:auto-start -->
### Kevin Read ([@unverbraucht](https://github.com/unverbraucht)) - 39 PRs

- #379: Vision pipeline cleanup (issue #326)
- #369: feat(attn): dots.ocr vision + decode attention kernels (v4/v5/gqa_warp/mb8) + investigation results
- #370: feat(tooling): splice_layers.py — graft F16 vision layers into a trunk quant
- #371: feat(rdna-compute): WMMA flash attention for asym4 text prefill (issue #237 item 2)
- #336: feat(dots-ocr): e2e OCR serving + text prefill + decode attention + graph-capture prep
- #335: perf(gfx10): HFQ4 + HFQ3 MMQ prefill default-on for RDNA2 — env-gates removed (#300)
- #333: data(benchmarks): KLD measurement cohorts + master-doc restructure (27B, 9B MQ3/MQ4, lmhead-a100)
- #331: feat(pflash): Hetero PFlash support + Serve path hardening
- #321: dots-ocr: phase 2 — vision tower + end-to-end OCR validated
- #197: MQ4-Lloyd WMMA prefill (#182 Phase 5b) — Phases A+B1+B2+B3
- #325: fix(vision): Qwen3.5-VL parity with HF + llama.cpp (closes #324)
- #315: perf(gfx10): HFQ4 MMQ family + HFQ3 polish — RDNA2 prefill post-#298
- #327: perf(gfx906): HFQ4 fused-projection MMQ — +7.3% prefill on Qwen3.5 9B MQ4
- #323: feat(multi-gpu): hetero PP=2 prereqs — env-gate mixed arch, per-arch JIT cache, init_layers VRAM bypass
- #314: refactor(awq): unify sidecar loader (#268 items 2 + 3)
- #298: feat(gfx10): MQ3 prefill on RDNA1/RDNA2 — full HFQ3 batched-prefill family with MMQ
- #297: Qwen2 standalone architecture (arch_id=7) — phase 1
- #230: refactor(tokenizer): interned merge symbols + loud OOV at construction
- #312: feat(vision): OpenAI /v1/chat/completions image inputs (supersedes #234)
- #228: fix(qwen35): MoE final-norm GemmaRMSNorm + drop now-redundant spiral workaround
- #273: feat(quantize+runtime): AWQ Stage A F2 — output-side whitelist expansion (o_proj, down_proj)
- #281: feat(gfx906): HFQ4 dp4a parity — b128 cliff + residual + 3 fused (#276)
- #266: feat(quantize+runtime): AWQ Stage A — Activation-aware Weight Quantization for MQ4G256 (F1 input-side)
- #267: fix(daemon): default repeat_penalty 1.3 → 1.0 (fixes #258 bug B)
- #242: feat(qwen35): F16 lm_head storage + WMMA-batched eval fan-out
- #263: feat(bench): Phase A Step 0+4 — cohort runner, imatrix collector, MSE harness
- #243: feat(runtime): eval_hipfire_llama — KLD eval harness for llama-arch models
- #251: feat(quantize): default conv1d weight to Q8 (KLD 0.30 → 0.25)
- #248: feat(q8): Tier 3 fused WMMA prefill — 1069 tok/s on gfx1100
- #236: quality-eval: act on claude + glm-5 review (followup to #233)
- #227: perf(mq3): _mb4 batch-tile fanout — +77% gfx1151 / +17-27% gfx1100
- #229: perf(tokenizer): hot-path hardening — cache merge_rank, allocator-free SP scan
- #233: quant-quality eval: KLD harness + prefill scoring + GGUF anchors
- #210: fix(qwen35): symmetric MQ3-in-MoE refusal across prefill paths (#179)
- #187: feat(gfx906): HFQ6/MQ6 Phase A — wave64 + dp4a kernel stack (+41% decode, +248% prefill)
- #206: fix(agentic-gate): VRAM-headroom skip for undersized hosts
- #195: MQ3-Lloyd WMMA prefill (#116 Phase 5) — Phases A+B1+B2+B3+C
- #189: feat(mq3-lloyd): enable fast variants on gfx1151 (Strix Halo APU) — parity with gfx1100
- #186: fix(kernels): port gemv_mq8g256 from sudot4 to sdot4 — fixes gfx906 build

### Björn Bösel ([@fivetide](https://github.com/fivetide)) - 25 PRs

- #535: feat: agentic PR review workflow
- #483: feat(spec): qwen35 DDTree distribution-correct temp>0 spec decode (SWOR) + target-generic DFlash for LLaMA/Qwen3
- #498: feat(dspark): consolidated DSpark speculative decode — deepseek4 + qwen3 + qwen35 (supersedes #484, #492, #493)
- #477: feat(spec): arch-generic speculative-decode seam + unified speculation config
- #463: Unified model loading + KV-cache usage unification
- #455: Unified model loading: transparent quant + auxiliary-detail management (carrier registry + WeightBackend)
- #454: chore(fmt): add fmt-changed helper + pre-commit rustfmt gate
- #451: fix(gpu-lock): flock-based stale-proof GPU mutex
- #453: fix(rccl): mark all_reduce_sum_f32 unsafe (clippy not_unsafe_ptr_arg_deref)
- #359: perf: barrier-free gate_up + MoE grouped WMMA kernels (+43-53%)
- #378: refactor(paro): remove default-off and unwired HIPFIRE_PARO_* env vars
- #342: refactor(rdna-compute): decompose Gpu God Object — 12 domain files, 5 state structs
- #349: feat: wire mtp_mode/mtp_k into global and per-model config
- #348: fix(cli): add .mq2lloyd extension to listLocal and findModel
- #337: Arch routing centralization -- ArchCaps atom/molecule/capability hierarchy (#328, task 2)
- #334: feat: consolidate HIPFIRE_ env vars into FeatureFlags + RuntimeConfig structs (#328, task 1)
- #330: perf(gfx1151): MQ4-Lloyd K4 prefill kernels (non-mb4 + mb4)
- #319: ParoQuant batched prefill (Phases 0-4) + MQ4G128 LA gates + g256-perfmax + rebase repair
- #318: feat(paroquant-moe): full hipGraph capture support for Qwen3.6-A3B-PARO (follow-up to #316 + #317)
- #316: feat(paroquant): Qwen3.6-A3B-MoE working — KLD 0.93 → 0.09 (10×)
- #317: fix(qwen35-moe): atomic-free MoE down for hipGraph determinism (task #100)
- #240: Fix/nix deps
- #214: feat(quantize): K-map alternating mode — same PPL, 17% smaller MoE (#196)
- #205: fix(quantize): K-map edge layers — FFN-only for dense, attn+FFN for MoE
- #199: feat(quantize): per-tensor mixed precision K-map (#196)

### xynexus ([@xynexus](https://github.com/xynexus)) - 18 PRs

- #414: tool(hfq): inject chat template into existing HFQ
- #412: docs: add FWHT residual QJL TODO
- #418: feat(dflash): add MQ6 draft conversion and dispatch
- #415: feat(awq): support AWQ on sub-4-bit quant arms
- #426: chore(scripts): harden Python diagnostic imports
- #425: chore(scripts): harden cleanup and benchmark argument handling
- #423: test(qwen35): cover MoE prefill admission matrix
- #422: ci: add no-GPU validation workflow
- #420: fix(runtime): accept no for prompt normalization opt-out
- #419: tools: add TurboQuant calibration tooling
- #416: feat(runtime): add HIP wait scheduling knob
- #411: chore: ignore local agent and quantization artifacts
- #410: docs: document embedded DFlash draft packaging
- #409: feat(cli): add local model catalog
- #405: tools: add AMD matrix instruction calculator submodule
- #404: docs: park hipfire eval harness plan
- #403: docs: add local model family support matrix
- #295: Skill hygiene

### Nick Woolmer ([@nwoolmer](https://github.com/nwoolmer)) - 12 PRs

- #629: maple: serve Maple-Preview 20B-A1B natively-ternary MoE (arch 15, qt=51 MQ2G256LloydU)
- #610: quant: Ornith 1.5 35B-A3B — SKU, MTP sidecar, VL, and qt44 MoE kernels incl. RDNA4/R9700
- #597: quant: add Bonsai TQ2G128 (qt=40) and BQ1G128 (qt=41)
- #508: fix(qwen35): require outer object brace before closing a tool call
- #445: feat(reap): generic MoE REAP — selective expert pruning + selective re-quant (SP1–SP4 + SP2)
- #446: feat(cohere2moe): Cohere2-MoE / North-Mini-Code-1.0 support (arch_id 12)
- #384: Qwen3.6-A3B serving + DeltaNet-state fixes + KLD eval-tooling
- #380: fix(kernel-cache): version cache key + self-heal invalid device images
- #372: fix(qwen35): truncation-safe DeltaNet resume + bounded thinking for agentic serve
- #367: Qwen3.5/3.6: grammar-guided tool calling + reliable prompt caching for agentic workloads
- #357: perf(rdna-compute): MMQ cutoff 256→128 on RDNA3+ for HFQ4 prefill (+118% Strix Halo)
- #322: feat(arch-deepseek4): add DeepSeek V4 Flash support (arch_id=9)

### alpineQ ([@alpineQ](https://github.com/alpineQ)) - 6 PRs

- #609: feat(serve): OpenAI client parity on the multi-slot path — tools, reasoning_effort, sampling, usage, tool-iteration reuse
- #608: fix(serve): multi-slot engine fits 24 GB cards — skip GDN S-tape, chunk prefill scratch, apply device visibility
- #607: fix(config): register serve.multi_slot_slots, multi_slot_ctx and multi_slot_prefill_chunk
- #606: fix(scripts): close unterminated quote that broke serve_concurrency_gate.sh parsing
- #222: feat(kernel): tiled online-softmax for attention_dflash_f32 (LDS-overflow fix at L≥16128)
- #201: fix(tokenizer): O(N²) → O(N log N) GPT-2 BPE — long-prompt prefill no longer stalls

### Tomás Gutiérrez L. ([@0x00cl](https://github.com/0x00cl)) - 3 PRs

- #364: fix(cli): Missing config options in record
- #231: fix(cli): adds .mq3 to the list of available local models filter
- #200: feat(cli): add hf identifier and token

### Sergio Sánchez Vallés ([@SergioSV96](https://github.com/SergioSV96)) - 3 PRs

- #632: fix(installer): run under Windows PowerShell 5.1 and prefer the discrete GPU
- #630: fix(scripts): prepend weight-cache preamble in compile-kernels.ps1
- #633: fix(cli): reserve a 32 MB main-thread stack on Windows

### Ghazanfar Ansari ([@ghazni101](https://github.com/ghazni101)) - 3 PRs

- #617: perf(loader): UMA-gated eviction + parallel warmer + zero-copy uploads — 2-3.4x faster dense loads, ~6x faster A3B MoE loads on dGPUs; fix sequential sampler seed
- #627: fix(generate): seed sequential AR sampler per request instead of fixed 0x13579BDF
- #637: fix(serve/quantize): context-cap decode guard; MoE expert-3d guard; standalone quantizer image

### aldrouil ([@aldrouil](https://github.com/aldrouil)) - 2 PRs

- #320: feat(cli): add hipfire sidecar-gen command for TriAttention calibration sidecar generation
- #269: fix(qwen35): guard hipGraph capture against TriAttention eviction stream conflict

### BearJew ([@DrBearJew](https://github.com/DrBearJew)) - 1 PR

- #221: feat(daemon,cli): ClosedThink prefix + thinking-on budget routing + response diagnostics

### HUSRCF-HKUST ([@HUSRCF](https://github.com/HUSRCF)) - 1 PR

- #638: feat(ck): add optional runtime and capability ABI

### Big God ([@creazyboyone](https://github.com/creazyboyone)) - 1 PR

- #615: fix(windows): daemon discovery, diag compiler probe, and quantize build

### George ([@noctrex](https://github.com/noctrex)) - 1 PR

- #613: fix(quantize): gemma4 GGUF config + tensor-name translation

### KBS ([@youdie006](https://github.com/youdie006)) - 1 PR

- #581: hfq: return Err instead of panicking on a truncated container (#578)

<!-- contributors:auto-end -->

### Co-authors (no merged PR of their own)

- **beanssec** ([@beanssec](https://github.com/beanssec)) - co-author on PR #35 (vision color misidentification fix) and PR #48 (`triattn_validate` r̄ contamination surfacing).
- **Dominik** (`git@domko.sbs`) - co-author on PR #9 (OpenAI-compatible serve, streaming lock lifecycle, SSE headers).

### Raon-1-mini-preview (arch 19)

- **u/Queasy_Asparagus69** (Hugging Face: datayoda) - the original community port of KRAFTON Raon-1-mini-preview to hipfire, and the bug reports that shaped the shipped route: the stock serve stream refusing generations (`stream must begin with gen_start`), thinking never closing (answers landing in reasoning), the broken `enable_thinking: false` frame, and needle-probe prompt-echo at greedy.

## License

hipfire is licensed under Apache-2.0 as of v0.3.0; individual files
whose authors have not elected Apache-2.0 remain MIT-licensed per
their SPDX header (see [LICENSE](LICENSE), [LICENSE-MIT](LICENSE-MIT),
[LICENSE-APACHE](LICENSE-APACHE), and [NOTICE](NOTICE)). The
canonical repository transitioned MIT-only -> dual in May 2026, then
dual -> outbound Apache-2.0 for v0.3.0; see
[docs/governance/relicense-2026-05.md](docs/governance/relicense-2026-05.md)
for the full decision record including the course correction from
a unilateral Apache-2.0 relicense to dual licensing.

This CREDITS.md is the authoritative contributor inventory referenced
by NOTICE; Apache-2.0 § 4(c) requires preservation of attribution
notices in the Source form, which includes this file when
distribution is under Apache-2.0. Upstream sources listed above
retain their own licenses.
