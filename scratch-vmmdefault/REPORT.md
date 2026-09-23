# VMM runtime KV default — local implementation and proof

Branch `vmm-default` from `34ccae74b`. Rust control plane only; Python developer serve harness. No HIP kernel changes. This report records the VMM default and subsequent sequence-capacity cutover; raw GPU artifacts remain local/uncommitted.

## Diff inventory

- `crates/hipfire-config/src/lib.rs`: default schema VMM; built-in default profile no longer pins contiguous.
- `crates/hipfire-registry/src/lib.rs`: remove registry-tag backend and `memory.max_seq` overrides; retain `generation.max_tokens` policy.
- `crates/hipfire-cli/src/main.rs`: preserve source provenance when projecting load params; omit backend and `max_seq` on built-in/registry automatic requests and emit explicit flag/global/model/one-shot overrides.
- `crates/hipfire-runtime/src/{kv_backend,kv_mode}.rs`, `crates/saddle-core/src/kv.rs`: VMM runtime default, gfx1201 native-fp8/gfx11 Q8 auto mode and VMM mapping-growth diagnostics.
- `crates/hipfire-loader/src/admission.rs`, `lib.rs`, `carriers.rs`: source/backend and trained context preflight before teardown; conservative weight projection is only a model-fit refusal. The Qwen carrier remeasures VRAM after actual weight upload, then chooses the card-bound VMM `max_seq` before KV reservation and returns the measured bound in loaded diagnostics.
- `crates/hipfire-daemon/src/main.rs`: log resolved sequence bound/model/card/mode and report them in loaded JSON beside effective backend/fallback; preserve slot-engine gate.
- `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs`: leave larger PBS in request-time rung admission after KV mapping; release widened reusable PBS between requests if future physical KV growth would otherwise exceed free VRAM.
- `crates/hipfire-arch-deepseek4/src/{carrier,deepseek4,ep,forward}.rs`, `crates/hipfire-arch-gemma4/src/carrier.rs`, `crates/hipfire-arch-llama/src/carrier.rs`: privately owned DeepSeek compressor caches honor admitted backend; unsupported carrier backstops prevent falsely labeled VMM.
- `crates/hipfire-generate/tests/vision_lifecycle_tests.rs`: migrate admission call to typed signature.
- `scripts/serve_harness.py`: automatic requested backend and `max_seq` by default; emit explicit overrides only, with effective markers and inline selftests.
- `scripts/guard_gfx1201_baseline.py`, `scripts/fixtures/vmmdefault_serve_prompts.json`: pinned card-B raw guard and five-prompt byte fixture.
- `docs/CONFIG.md` and dated perf checkpoint: current default/capability semantics and measured provenance.

The post-`4418013e1` uncommitted sequence-capacity inventory was 18 tracked files (`git diff --stat`: 662 insertions, 243 deletions) plus this untracked report and local raw artifacts. In addition to the grouped entries above, it includes `crates/hipfire-runtime/src/loader_api.rs` and three example constructors (`crates/hipfire-arch-llama/examples/qwen3_dspark_bench.rs`, `crates/hipfire-runtime/examples/{llama_dflash_hidden_capture,llama_lm_head_logits_parity}.rs`). The runtime default, DeepSeek/Gemma adaptations, guard script, fixture, and dated perf checkpoint were already committed in `4418013e1`; they were not reintroduced by this diff.

## Sequence-capacity accounting

The trained window comes from already-open HFQ JSON `config.text_config.max_position_embeddings` (`admission.rs:393-407`); the 27B fixture reports 262,144. Pre-teardown admission estimates HFQ bytes + 12.5% + 512 MiB solely to refuse an obviously impossible load without evicting a resident model. **It never sets the effective card cap from that projection.** In `qwen35/carrier.rs`, actual weight upload precedes the HIP free-VRAM query, which precedes KV construction. Measured free *after weights* less the exact minimum 512-row PBS/projection/Q16 reservation (including 1 GiB safety headroom) and 128 MiB page rounding is divided by `vmm_kv_token_bytes` (only 16/64 full-attention layers); this gives card capacity. The default is `min(model_ctx, card_cap)`, an explicit positive `max_seq` wins even above trained context, and loaded JSON/logging uses this postweight value. Without eviction VMM `physical_cap=max_seq`; CASK may use `physical_cap=min(max_seq, eviction_window)`.

KV reserves virtual addresses at load and grows physical pages on demand. The real prefill width is chosen by `memory_admitted_rung` (`prefill.rs:1392-1450`), which calls `gpu.hip.get_vram_info` **after** `kv_cache.ensure_mapped_capacity` (`prefill.rs:2062-2063`) and charges full PBS/projection/Q16 scratch only against currently free, mapped-KV VRAM. Thus short prompts can use 8192 rows; long contexts may narrow to 512. Optional widened PBS reuse is released between requests when keeping it would prevent the unmapped KV reserve from becoming physical (`can_retain_widened_pbs`); this also protects later decode growth.

PBS is lazy (widened cache allocated on first prefill), after KV construction (`carrier.rs` weight→KV→DeltaNet→scratch). The separate gfx11 C/lean stack is absent here; because the context bound charges only the 512-row minimum, lean versus full 8192-row PBS no longer materially changes it. Qwen MoE has no dense PBS budget and retains the 32,768 schema built-in (`hipfire-config/src/lib.rs:4779`) on omitted context, with a loaded reason; MoE-specific capacity accounting is a known gap rather than a load refusal.

## Verification

- `cargo build --release` passed. Scoped loader admission/resident-preservation and CLI/backend provenance tests passed. Harness `--self-test` passed. `hipfire config get memory.kv_backend` on fresh isolated home printed `vmm`; explicit set contiguous printed `contiguous`.
- Card-B `vmm_arena_smoke` passed all five lifecycle cases. The final A-D binary (`hipfire` MD5 `125231ec864f219a9a7758016b83430c`, daemon MD5 `bc16355db9dd8370cc64d75acccdbc36`) passed three independent fresh-process automatic pp8192 medians `3648.2,3642.9,3642.5` (>3620) and decode medians `36.50,36.51,36.52` within 1% of 36.5; logs explicitly show VMM allocation. Matched explicit contiguous comparator medians `3577.8,3573.4,3574.5`, contiguous allocation marker, no VMM marker. Local exact-model tag alias also loaded automatic VMM on the immediately preceding daemon build (MD5 `dc56b8bae3bed030fdbc937db25bfc5c`); only the non-gfx1201/Windows certification check changed after that probe.
- Native serve output five of five coherent, nonempty; VMM and explicit contiguous deterministic greedy seeded transcripts byte-identical (MD5 `98cd8d89d3213491a06c7c88489a14a7`), with matching request MD5s on preceding daemon build `dc56b8bae3bed030fdbc937db25bfc5c`. Raw `scratch-vmmdefault/guard-certified/`, `guard-contiguous-certified/`, `serve/`, and `serve-contiguous/` remain local.
- Scope: exact logits/KV oracle, independent graph replay trace, native fp8/bf16 adaptive oracle, and other devices/OSes/topologies were not exercised; admission conservatively declines unvalidated combinations. No claimed promotion/ABBA gain.

## Postweight validation follow-up

- The first manual pre-warmed serve returned HTTP 500 for a long request: CLI `ServeRuntime` cached the *omitted* request `max_seq` as zero, then unnecessarily reloaded the already resident model; the pre-teardown projection correctly refused loading another copy. Separate fix `a3531e96e` caches the daemon's effective `loaded.max_seq` and logs the actual requested minimum rather than a misleading zero. On the rebuilt CLI, the long request proceeded from the existing pre-warmed model into physical KV mapping without a second weight load; this confirms the reload regression is gone.
- Prior 235,930-token experiment timed out at 1,200 seconds. No request or daemon chunk log was retained in `scratch-vmmdefault/` or the isolated `.hipfire-homes/vmmdefault*` homes, so neither completed chunk count nor per-chunk time trend can honestly be reconstructed; it was **not rerun**. Treat as an open performance finding, not a capacity failure. The instrumented partial ~64k run below provides the available depth-dependent timing evidence.

### Card-B manual serve growth observation (GPU-e475645fe0200397)

Fresh isolated `HIPFIRE_HOME=/home/kaden/.hipfire-homes/vmmdefault-growth`, `ROCR_VISIBLE_DEVICES=HIP_VISIBLE_DEVICES=GPU-e475645fe0200397`, daemon from this branch, automatic backend and automatic `max_seq` (no corresponding serve flags). Source committed prompt: `benchmarks/prompts/qwen38_issue693_longcode_20676.txt`, MD5 `b4d0b63cddcac872648ddf3cdd92cac2`; three repetitions plus a short run prefix/suffix produced 225,813 bytes, MD5 `2b85a3a0aaa0e5c5e942e7fd1b681b81` (request JSON MD5 `b2cc45ca41ea2dc9a4664c202e5fdab3`). Approximately 64k intended tokens; no completed response/usage token count exists. Timestamped daemon log: `growth/serve.clock.log`; 250-ms card-B sysfs samples: `growth/repeat3.vram.csv`; result: `growth/repeat3.result.json`.

| Observed mapped prefix (tokens) | Time to next mapping (s; approximately one 8,192-row chunk) |
| ---: | ---: |
| 18,289 → 26,417 | 2.169 |
| 26,417 → 34,546 | 2.488 |
| 34,546 → 42,674 | 2.787 |
| 42,674 → 50,803 | 3.129 |
| 50,803 → 58,931 | 158.538 |
| 58,931 → stop | >86.855 (unfinished) |

Initial mapped prefix at load was 2,032; the request expanded it to 58,931 without OOM. Peak card-B VRAM used was 25,415,127,040 bytes (34,208,743,424 total); prefill **did not complete**, so total successful prefill time and token count are unavailable. The request elapsed 256.166 s before deliberate termination; the connection closed because its owned serve was stopped, not because of an observed OOM. The depth cliff beyond the old 32k default is disproportionately large; **suspected**, not demonstrated, is gfx1201 fp8 fast-attention admission `max_ctx ≤ 32768` falling back to a slower kernel. Main directed stopping this slow 64k attempt and skipping the planned 128k request pending separate investigation, rather than waiting 20 minutes per attempt. Card-B was clean afterward (`rocm-smi --showpids`: no KFD PIDs; used VRAM 59,912,192 bytes). This is a partial growth observation, **not** a claim that either 64k or 128k completed.

### Explicit overrides and gfx1100

The card-B serve harness used explicit `--max-seq` and automatic KV backend; the registry selected Q8 KV. Both loads succeeded with VMM, and both pre-warm receipts preserve the user bound (raw logs under `overrides/`):

| Explicit override | Loaded `max_seq` | `bound` | Trained `model_ctx` | Measured Q8 `card_cap` |
| ---: | ---: | :--- | ---: | ---: |
| `--max-seq 300000` | 300,000 | user | 262,144 | 490,948 |
| `--max-seq 16384` | 16,384 | user | 262,144 | 490,948 |

The 300k harness's 16-token generation ended with an open-think validation error **after** the successful load; this does not qualify as a completed generation. The 16k harness produced a 64-token `finish=length` response. Their `bound=user` load receipts are the override proof.

After `Gfx11Fa2Gates` granted HIP0, the committed sequence-capacity branch was fetched into a **fresh native hipx worktree** `/home/kaden/hipfire-vmmdefault` and built with `cargo build --release -p hipfire-cli -p hipfire-daemon`. The older binaries previously under that path were preserved as `/home/kaden/hipfire-vmmdefault-prior-binaries`. A detached gfx1100 bench load with Q8 KV and no backend/max-seq overrides completed; `/home/kaden/hipfire-vmmdefault/gfx1100-load.log` reports `max_seq: 249344 (bound=card; model_ctx=262144, card_cap=249344, kv=q8)` and `KV cache: Q8 vmm`. HIP0 was released to `Gfx11Fa2Gates` immediately afterward. No gfx11 ABBA/performance claim is made.

### Build and clean-room

`cargo build --release` passed again after the serve context-cache fix (warnings only). The code commits are `294c4ad9d` (measured sequence-capacity cutover) and `a3531e96e` (pre-warmed serve context tracking). Only this report is committed as evidence; raw logs/JSON/VRAM CSV remain local and uncommitted. The requested clean-room search over `kaden/mq4-lloyd..HEAD` returns zero matches.
