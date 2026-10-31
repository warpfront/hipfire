# Speculation support inventory

Inventory of speculative-decode **capability** per architecture in hipfire
source. Capability is not product admission, not a performance floor, and not a
validation pass.

| Field | Value |
|---|---|
| Page state | **shipped / ref-pinned** — source-derived capability inventory for the audited ref (see [`INDEX.md`](INDEX.md)) |
| Inventory date | 2026-10-13 |
| Audited source ref | `c7a732ad5b921706519cdba30fdb43a7e62aa2d4` |
| Comparison base | `origin/beta` @ `c7a732ad5b921706519cdba30fdb43a7e62aa2d4` |
| Arch id table | [`architecture-ids.md`](architecture-ids.md) |
| Validation routes | [`VALIDATION.md`](VALIDATION.md) only |

**Fail closed:** before any product claim, re-read the symbols in
[Source anchors](#source-anchors). If this page disagrees with source, source
wins. [`admissions.yml`](admissions.yml) records exactly one admitted row (an
LFM plain-AR route, no speculation); no speculation mechanism on any
architecture is admitted by this page or by that registry.

## Lifecycle

- **DFlash is not deprecated.** It is a supported mechanism; its default and
  per-model resolution are in [`CONFIG.md`](CONFIG.md).
- **Deprecated since 0.4.0, removal in 0.5.0:** only the legacy contiguous
  DFlash opt-in knobs `HIPFIRE_DFLASH_WINDOW=0` and `HIPFIRE_DFLASH_CTX_CAP`.
  Setting either warns at DFlash load; the draft-declared window supersedes
  them. Do not recommend them. Source: the `lifecycle: deprecated since 0.4.0,
  removal 0.5.0` marker on `warn_deprecated_legacy_dflash_knobs` in
  `crates/hipfire-arch-qwen35/src/dflash_spec.rs`; status rows in
  [`env-vars.md`](env-vars.md) and [`CONFIG.md`](CONFIG.md).
- **Deprecated since 0.4.0, removal in 0.5.0, adjacent to this page:** CASK /
  TriAttention KV eviction and PFlash prefill compression. They still load and
  warn, are unsupported, and are not a basis for any speculation claim. Qwen4
  admission refuses them outright.

## Vocabulary

| Term | Meaning |
|---|---|
| **n-gram drafter** | Model-free, arch-generic (`crates/hipfire-runtime/src/spec_ngram.rs`). Opt-in via `HIPFIRE_NGRAM_DRAFT=1` (env wins) or config `speculation.ngram` (default `off`), folded into `SpecLoadCfg`. Requires a `SpecTarget` verify seam. Distinct from the n-gram *takeover windows* (`speculation.mtp_ngram`, ngram-mod) that run inside native Qwen4 MTP. |
| **MTP** | Learned multi-token-prediction head shipped with model weights. qwen35 (arch 5/6): a bundled `.mq4-mtp` trailer or a `<trunk>.mtp` sidecar, attached only on an explicit `speculation.mtp = on` / `--spec mtp` (`auto` stays AR: no measured AR floor). qwen4 / Flash-Next (arch 16): native in-trunk MTP layer, armed by `auto` (declares a measured AR floor). deepseek4 (arch 9): in-trunk MTP layer, explicit only under the same gate (its layer is not even loaded under `auto`). The gate is `hipfire_config::mtp_mode_admits` over each producer's `MTP_MEASURED_AR_FLOOR` (`Carrier::mtp_measured_ar_floor`, default `false`). |
| **DFlash** | Block-diffusion drafter (arXiv 2602.06036). Qwen35 path: `crates/hipfire-arch-qwen35/src/dflash_spec.rs`. Dense llama/plain-Qwen3 targets: generic chain/tree path `crates/hipfire-runtime/src/dflash_generic.rs` with an external arch_id=20 HFQ draft. Muse Glimmer (arch 14): its own arch_id=23 drafter (`crates/hipfire-arch-muse-glimmer/src/drafter.rs`). |
| **DSpark** | Draft-sidecar path (`<stem>-dspark.<ext>`). Implemented for llama/plain-Qwen3 (arch 0/1), qwen35 (5/6), and deepseek4 (9). Preferred over DFlash/MTP/n-gram (or in-trunk MTP on deepseek4) when the sidecar loads and `ctx.spec.dspark != Some(false)`. |
| **DDTree** | Tree-masked verify + SWOR sampling. Opt-in (`speculation.dflash_tree` / `HIPFIRE_DFLASH_TREE=1`, experimental); knobs `speculation.ddtree_budget` / `ddtree_topk` (`HIPFIRE_DDTREE_BUDGET` / `HIPFIRE_DDTREE_TOPK`; config default budget `0` = chain). Refused on qwen4. |
| **SpecTarget verify seam** | Arch-generic verify interface (`crates/hipfire-runtime/src/spec.rs`). Arch `spec_impl.rs` (or `mtp_spec.rs` for qwen4) plugs the target. Sequential `verify_block` is a correct baseline; block-parallel verify is optional perf. |
| **Speculator registry** | Load-time `build_speculator` in `crates/hipfire-loader/src/spec_build.rs` (DFlash → qwen35 MTP head → n-gram cascade when each is eligible). Carriers may short-circuit **before** this with DSpark (llama 0/1, qwen35 5/6, deepseek4 9) or build their own speculator (deepseek4 MTP, qwen4 native MTP, muse-glimmer DFlash). |

How to add speculation to a new arch (interface contract): executable skill
[`.agents/skills/hipfire-arch-port/`](../.agents/skills/hipfire-arch-port/) (speculation step).

## Default posture (config schema)

All values are from [`CONFIG.md`](CONFIG.md) / `crates/hipfire-config/src/lib.rs`; re-read them before citing.

| Surface | Schema default | Note |
|---|---|---|
| `speculation` selector | `auto` | `off` / `auto` / `ngram` / `dflash` / `mtp` / `dspark`; forced names bypass heuristics, missing prerequisites fall back to AR with a warning |
| `speculation.dflash` | `off` | One registry-scoped exception: Qwen3.8-27B tiers whose registry entry declares a paired `dflash` draft resolve `auto` (kill switch `dflash_mode off`). Daemon: `dflash_mode == "off"` skips draft load even if a draft path is supplied. |
| `speculation.mtp` | `auto` | Arms a present head only for a producer that declares a measured AR floor (`MTP_MEASURED_AR_FLOOR`: qwen4 `true`; qwen35, deepseek4 and every other carrier `false`, so `auto` stays AR and logs `MTP: off by default (no measured AR floor); opt in with --spec mtp`); `on` arms any present head and fails the load without one; DSpark / DFlash selection is separate |
| `speculation.mtp_k` | `auto` | Native Qwen4 MTP: `min(GPU row-exact verify rows, window of the CPU routed-expert mode that owns layers) − 1`, resolved at load after placement (7 on every GPU today with no CPU-owned experts, 3 on a D1/D2 CPU-placed load; `hipfire config` shows 7); else 3 (`MTP_K_DEFAULT`); explicit 1–10 overrides (`4` restores the previous gfx1151 cap) |
| `speculation.mtp_sampled` | `true` | Documented for native Qwen4 MTP sampled serving |
| `speculation.mtp_ngram` | `auto` | On for native Qwen4 MTP on gfx1151 / gfx1201; other producers off unless `on` |
| `speculation.ngram` | `off` | Opt-in; `ngram_k` 12, `ngram_min_count` 2 |
| DSpark | auto (`dspark` mode not `off`) | Loads whenever a `-dspark` sidecar sits next to the trunk |

## Master table (source-derived)

n-gram arch gate in `build_speculator`:
`matches!(arch_id, 0 | 1 | 5 | 6 | 7 | 8 | 10 | 11 | 12)`.

| Arch crate | arch_id | Family | In-repo mechanisms | Native learned drafter? | Load / route wiring (summary) |
|---|---|---|---|---|---|
| llama | 0 / 1 | LLaMA / Mistral / plain Qwen3 dense | **DSpark** sidecar + DFlash generic (arch_id=20 draft) + n-gram + SpecTarget | External DSpark (`-dspark`) and/or DFlash HFQ draft when present and mode allows | Carrier precedence **DSpark > DFlash > n-gram**. Route `LlamaSpec` when a speculator exists: greedy without penalties, or the sampled chain / DDTree SWOR gates |
| qwen35 | 5 / 6 | Qwen3.5/3.6/3.8 DeltaNet hybrid (dense + A3B MoE) | **DSpark** sidecar + DFlash + native MTP head + n-gram + SpecTarget | DSpark; DFlash draft; native MTP head (bundled `.mq4-mtp` trailer or sibling `.mtp`) | Load precedence **DSpark > DFlash > MTP > n-gram** (one cascade; see MTP). MTP is explicit-only (`--spec mtp` / `speculation.mtp = on`): qwen35 declares no measured AR floor, so `auto` stays AR and loads no head; DFlash needs `dflash_mode` on/auto (registry `auto` for paired Qwen3.8-27B tiers). Route `QwenDflash` carries both DFlash and MTP; sampled MTP routes on temperature>0 (the qwen35 MTP drafter reports temperature verify) |
| qwen2 | 7 | Qwen2/2.5, VibeThinker | n-gram + SpecTarget (block-parallel verify available) | No | n-gram opt-in; route `Qwen2Spec` when a speculator exists (greedy, or n-gram can sample) |
| dots-ocr | 8 | rednote dots.ocr (Qwen2-1.5B decoder) | n-gram on **decode phase only** after vision prefill | No | `generate_vl_dots_ocr` → n-gram decode loop when a speculator is built; vision prefill unchanged |
| deepseek4 | 9 | DeepSeek-V4 MLA+MoE | **DSpark** and/or in-trunk MTP + SpecTarget | Yes — `-dspark` sidecar and/or in-trunk MTP | Precedence **DSpark > MTP**; route `Deepseek4Spec` requires `deepseek4_spec_requested` (installed DSpark, or `mtp_mode` `on` with weights present; `auto` stays AR: the native MTP has no measured AR floor, so its layer is not loaded) and a temperature-compatible drafter; **n-gram not in the n-gram arch gate** |
| minimax | 10 | MiniMax-M2 MoE | n-gram + SpecTarget | No | n-gram opt-in; route `MiniMaxSpec` when a speculator exists |
| lfm2moe | 11 | LFM2.5 hybrid conv+attn (+MoE variants) | n-gram + SpecTarget (conv-state snapshot/rollback) | No | n-gram opt-in; route `LfmSpec` when a speculator exists. **Not** a Redline or batched-prefill admission |
| cohere2moe | 12 | Cohere2-MoE / North-Mini-Code | n-gram + SpecTarget + `Cohere2MoeEmit` | No | n-gram opt-in; route `CohereSpec` when a speculator exists |
| gemma4 | 13 (+22 drafter) | Gemma 4 dense text | EAGLE draft (arch 22) via `params.drafter` only; **no** `build_speculator` mechanism (not in the n-gram gate; `params.draft` DFlash refused) | Arch-22 EAGLE drafter exists but is **gated off** | The EAGLE path runs only with `HIPFIRE_GEMMA4_EAGLE=1` (developer; it is slower than AR on gfx1151). Greedy requests verify by argmax; sampled and penalized requests by naive sampling (each verify row drawn with the AR sampler against the request's penalty history); `logprobs` come from the verify rows. E2B/E4B targets refuse a drafter. Default serve is plain AR |
| muse-glimmer | 14 (+23 drafter) | Muse Glimmer dense text | DFlash with the arch-23 `muse_glimmer_assistant` draft | DFlash arch-23 drafter when supplied | Draft is loaded only when a draft path is set and `dflash_mode != off`; load failure falls back to AR. Route `GlimmerSpec` when a speculator exists. Gemma EAGLE `params.drafter` refused; n-gram not in the gate |
| maple | 15 | Maple-Preview natively-ternary MoE | None | No | `MapleCarrier::caps()` declares no DFlash and no MTP; route is always `MapleAr` |
| qwen4 | 16 | Qwen3.8-Flash-Next (Qwen4) | **Native MTP** (in-trunk layer) + MTP/ngram-mod takeover windows + **DFlash2** (arch-20 draft, explicit only) + SpecTarget (`mtp_spec.rs`, `dflash_target.rs`) | Yes — native MTP; external DFlash2 draft | Native greedy MTP attaches by default on admitted loads (`.mq4r` retained route keeps AR; `speculation.mtp off` disables). DFlash2 runs only on `--spec dflash` / `speculation.dflash = on` with a draft (`--model-draft`), single card, GPU-resident experts. Admission **refuses** DSpark, standalone n-gram, DDTree, EAGLE, adaptive KV, CASK, PFlash, a draft path without `dflash = on`, and DFlash under EP/PP. Route `Qwen4Spec` greedy, and sampled via rejection sampling (MTP when `mtp_sampled` is on; DFlash2 always, unpenalized requests). EP (2..=8 ranks) admits MTP only on explicit `--spec mtp` (see [`CHANGELOG.md`](../CHANGELOG.md)) |
| raon | 19 | Raon-1-mini-preview (RDA + MLA + 4-stream mHC MoE) | **DFlash2** (KRAFTON `Raon-1-mini-preview-dflash2` draft through `dflash_convert`, explicit only) + SpecTarget (`spec_impl.rs`) | External DFlash2 draft | Opt-in only: `--spec dflash` / `speculation.dflash = on` with `--model-draft`, single card. `RaonCarrier` refuses a draft without `dflash = on`, `dflash = on` without a draft, and native MTP / DSpark / n-gram. Route `RaonSpec` (`generate_dflash` over the Raon carrier's `Qwen35Emit`, grammar off) for greedy requests and sampled requests without a non-neutral penalty (the card's presence default counts for an unset sampled request, so those stay `RaonAr`); the prompt render, reasoning split, tool router and stop handling are `RaonAr`'s |

**Has a learned or diffusion drafter path in-tree today:** qwen35 (DSpark ± DFlash ± MTP),
qwen4 (native MTP, or DFlash2 on request), raon (DFlash2 on request), deepseek4 (DSpark/MTP), llama/plain-Qwen3 dense (DSpark and/or generic
DFlash when draft/sidecar loaded), muse-glimmer (DFlash arch-23 draft). gemma4 has an EAGLE
drafter path that is gated off. **n-gram-capable with SpecTarget:** arch ids in the gate above
(qwen4 excluded by admission). Everything else remains plain autoregressive unless source gains
a new arm.

## Mechanism notes

### DFlash (qwen35, dense generic, qwen4 DFlash2, muse-glimmer)

- **Default:** `speculation.dflash = "off"` in `crates/hipfire-config/src/lib.rs`. `"auto"` turns dense
  Qwen3.5-style targets on and skips known-loss A3B/MoE cases ([`CONFIG.md`](CONFIG.md)). The registry
  layer lowers `auto` for the Qwen3.8-27B trunk tiers that declare a paired draft
  (`crates/hipfire-registry/src/lib.rs`); an absent draft file wires nothing and the load keeps MTP `auto` (AR for qwen35, which declares no measured AR floor).
- **Daemon:** `params.dflash_mode == "off"` skips draft load even if a draft
  path is supplied (`crates/hipfire-daemon/src/main.rs`).
- **qwen35:** a loaded draft builds `DflashSpeculator` (chain by default; DDTree opt-in). Chain-mode
  temp>0 verifies use the selector path; DFlash verifiers implement no repeat/presence/frequency
  penalties, so a penalized request runs AR, and `min_p` is ignored (the daemon warns). Env
  opt-outs are documented (`HIPFIRE_DFLASH_TEMP_SPEC=0`, `HIPFIRE_DFLASH_CHAT=0`, etc.).
  Adaptive KV suppresses DFlash.
- **Legacy contiguous knobs are deprecated since 0.4.0, removal 0.5.0:**
  `HIPFIRE_DFLASH_WINDOW=0` and `HIPFIRE_DFLASH_CTX_CAP`. The default draft-declared window is the
  supported path; see [Lifecycle](#lifecycle).
- **llama 0/1:** `dflash_generic` requires arch_id=20 HFQ from `dflash_convert`.
  Chain is default; the tree arm is opt-in (`HIPFIRE_DFLASH_TREE=1`).
- **qwen4 16 (DFlash2):** `Qwen4Carrier` builds the shared
  `hipfire_runtime::dflash2_spec::Dflash2Speculator` (name `dflash2`) from an arch-20 DFlash2 draft
  (`dflash_convert`, candidate selector + dynamic convs) only on `--spec dflash` /
  `speculation.dflash = on` with `--model-draft`; native MTP stays the default and the two never load
  together. The target captures the drafter's `target_layer_ids` as each layer's completed output
  averaged over the 4 hyper-connection streams (`hc_mean_capture_f32`, run between interpreter steps
  so the verify program is unchanged) in prefill and every verify; a capturing prefill is planned
  GPU-only (the NPU trunk pair executor and the S1 layer-major prefill build their own steps, so
  they are bypassed). Block width: `speculation.dflash_block`, Qwen4 `auto` = 8 (with the confidence
  cut that verifies only the draft's confident prefix, the fastest measured on gfx1151), an explicit width clamped to the GPU's row-exact verify rows
  (8 on gfx1151). The host keeps only the draft's 2048-row window of feature rows (≤ 2 × 2048 ×
  51,200 B). Chain only: greedy is the selector's argmax walk (output = AR); sampled draws from the
  selector's softmax over its top-16 and accepts with sparse-q rejection sampling against the AR
  sampler's law (temperature, top_k, top_p, min_p). Penalized requests stay AR. EP, PP and
  CPU-hosted routed experts refuse DFlash2. **Known gap:** the DFlash2 route never reuses the Qwen4
  session / prefix cache (every request prefills cold; HIPF-143).
- **raon 19 (DFlash2):** `RaonCarrier` builds the shared `Dflash2Speculator` from the KRAFTON
  `Raon-1-mini-preview-dflash2` draft (`dflash_convert`; `target_streams_per_layer = 4`) only on
  `--spec dflash` / `speculation.dflash = on` with `--model-draft`. Unlike Qwen4, which averages
  the 4 mHC streams, the target captures the draft's `target_layer_ids` as the full post-FFN
  carrier (all 4 streams, stream-major) per extract layer; the drafter's feature width is
  `num_extract × target_streams_per_layer × hidden` (51,200 for this pairing). Block width:
  an explicit `speculation.dflash_block`, else the unbatched `auto` width (16: Raon has no
  continuous batching). There is no prefix reuse on either Raon route: `RaonSpec` clears the
  device state, the conversation mirror and the assistant-turn cache first and plans a cold
  prefill of the full render. A request whose context does not fit the drafter's capacity falls
  through to `RaonAr`.
- **muse-glimmer 14:** a distinct arch-23 draft (not a `dflash_convert` arch-20 artifact), loaded via
  `HIPFIRE_DFLASH_DRAFT` / `params.draft` while `dflash_mode != off`.
- **Genre conditionality** is a measured property (see the historical table in
  [`BENCHMARKS.md`](BENCHMARKS.md)), not a universal win.

### DSpark

- Sidecar discovery: `<stem>-dspark.<ext>` next to the trunk when
  `ctx.spec.dspark != Some(false)` (`None` = auto keeps default-on load; forced
  `--spec dspark` with no sidecar warns and falls back).
- **llama 0/1:** `LlamaCarrier` loads and builds DSpark **before**
  `build_speculator`. Precedence **DSpark > DFlash > n-gram**.
- **qwen35 5/6:** `finish_qwen35_load` builds `dspark_speculator` first; it wins
  over DFlash and the `build_speculator` cascade (adaptive KV suppresses it).
  Precedence **DSpark > DFlash > MTP > n-gram**.
- **deepseek4 9:** `-dspark` sidecar preferred over in-trunk MTP at load; sampled verify is enabled.
  Precedence **DSpark > MTP**.
- Refused by qwen4 admission and on the qwen35 dense-TP mesh.

### MTP

**qwen35 (arch 5/6) — one load path.** `finish_qwen35_load` resolves the head
(`resolve_qwen35_mtp_head_sourced` in `crates/hipfire-loader/src/lib.rs`): a bundled
`.mq4-mtp` trailer first, then the sidecar (CLI-resolved `mtp_path`, else `<trunk>.mtp`);
a head whose hidden size or vocab differs from the trunk is refused before upload. The head
is attempted when arch is 5/6, no adaptive KV, no eviction, the producer-capability gate admits `speculation.mtp` (`on`; `auto` only for a producer with a declared measured AR floor, which qwen35 lacks, so `auto` never loads a head and the load logs `qwen35 MTP: off by default (no measured AR floor); opt in with --spec mtp`; `off` never), no
DFlash draft and no DSpark speculator. It is handed to the `build_speculator` MTP arm
(`build_qwen35_mtp_speculator`, cascade DFlash → MTP → n-gram). `mtp=on` with no head fails
the load; `auto` never reaches the head (see above). K from `speculation.mtp_k` (`HIPFIRE_MTP_K`),
default 3, clamped 1–10. There is no separate opt-in env gate: `HIPFIRE_QWEN35_MTP` and
`HIPFIRE_QWEN_MTP` have no readers under `crates/`, and the daemon has no `generate_qwen35_mtp`
serve path. Route: `QwenDflash` (generic wrapper); the qwen35 MTP drafter has
`requires_greedy() == false` and `supports_temp_verify() == true`, so sampled requests route
through MTP. Penalties are applied to every verify row.

- **qwen35 dense tensor-parallel mesh (arch 5):** the MTP head is replicated on rank 0 only for
  an explicit `speculation.mtp = on`; `auto` and `off` keep the AR mesh loop. Block-verify
  drafters (n-gram, DFlash, DSpark) are refused on the mesh.
- **qwen4 / Flash-Next (arch 16):** `Qwen4Carrier` builds a native MTP speculator
  (`hipfire_arch_qwen4::mtp_spec::build_qwen4_mtp_speculator`) when
  `qwen4_native_mtp` admits it (default on because Qwen4 declares a measured AR floor, `MTP_MEASURED_AR_FLOOR = true`, unchanged; off for retained-Redline `.mq4r`; `--spec mtp`
  forced on is refused on `.mq4r` and on a profile with no MTP head). K via `mtp_k`
  (`auto` = `min(row-exact verify rows, CPU-owned-expert window) − 1`: 7 with no CPU-owned experts, 3 on a CPU-placed load). Sampled requests use speculative rejection sampling when
  `speculation.mtp_sampled` is on. MTP+ngram-mod takeover windows are a separate switch
  (`speculation.mtp_ngram`).
- **deepseek4:** `Deepseek4MtpDrafter` reports `requires_greedy() == true`.
  Spec request from `HIPFIRE_DEEPSEEK4_SPEC_DECODE=1` or
  `mtp_mode`/`HIPFIRE_MTP_MODE` `on` with weights present (`auto` stays AR: no measured AR floor, and `DeepseekV4Config::load_mtp` leaves the MTP layer unloaded). `spec_k` resolves at load
  `HIPFIRE_DEEPSEEK4_SPEC_K` → configured `mtp_k` → `MTP_K_DEFAULT` (3).
  A DSpark sidecar, when present, is preferred over this in-trunk MTP at load.

### n-gram

- Enable: `HIPFIRE_NGRAM_DRAFT=1` or CLI-resolved `speculation.ngram` (`ngram_draft`).
- Window: `HIPFIRE_NGRAM_DRAFT_K` / config (default K=12 in `build_speculator`).
- deepseek4 is **intentionally outside** the n-gram gate; its SpecTarget n-gram
  primitives are not the production draft path (MTP/DSpark instead).
- qwen4 refuses the standalone n-gram drafter at admission (n-gram appears only as MTP takeover
  windows). The qwen35 dense-TP mesh refuses it too.
- lfm2moe verify must snapshot/restore conv rings; keep graph/capture modes
  consistent with the campaign contract when measuring parity.
- Historical τ/perf probes (2026-06-23/24) showed n-gram often slower than AR on
  compute-bound small decoders and only modest upside on large BW-bound MoE at
  low τ — correctness wiring ≠ speedup admission.

## Validation (mandatory routing)

Use [`VALIDATION.md`](VALIDATION.md) as the sole claim → route selector.

| Claim class | Minimum route (from VALIDATION) | Not sufficient |
|---|---|---|
| Spec wiring / compile-only | No-GPU routes when no GPU behavior claimed | Assuming GPU parity |
| Numerical / state parity for a verify path | Path-specific oracle for that arch/surface; **blocked** if none exists | `serve_harness.py` alone |
| User-facing serve semantics | `scripts/serve_harness.py` with the exact model | Numerical proof |
| LFM2.5 chat framing / thinking | `scripts/serve_harness.py` with the exact `lfm2.5:*` tag | Numerical parity |
| Perf improvement from a drafter | [`methodology/perf-benchmarking.md`](methodology/perf-benchmarking.md) + stationary matched runs | Single warm run; genre-mismatched prompts |
| Product default / admission | Row in [`admissions.yml`](admissions.yml) | Speculator present at load; harness exit 0 |
| Redline-attributed claim | [`REDLINE.md`](REDLINE.md) ladder | Spec-decode success |

### Retired coherence gates (historical only)

`scripts/coherence-gate-*.sh` (including the DFlash and per-arch batteries) are
**retired as current acceptance evidence** and deprecated since 0.4.0, removal in 0.5.0
(with `_coherence_runner.py`, `awq_coherence_check.sh`, `pflash-gate.sh`). They must not be
required for merge, promotion, or benchmark claims. They may be used only for historical
reproduction. See [`VALIDATION.md`](VALIDATION.md) § Retired gates (historical only).

Campaign-specific omissions of coherence gates (for example an LFM effort) are
**not** a universal rule for every arch.

## Explicit non-claims

| Anti-pattern | Disposition |
|---|---|
| “n-gram is wired ⇒ ship as default” | **Rejected** — opt-in; no admission row |
| “native MTP / paired DFlash is on by default ⇒ admitted product route” | **Rejected** — config defaults and registry layers are not rows in `admissions.yml` |
| “DFlash is deprecated” | **Rejected** — only the legacy contiguous knobs (`HIPFIRE_DFLASH_WINDOW=0`, `HIPFIRE_DFLASH_CTX_CAP`) are deprecated (0.4.0 → 0.5.0) |
| “LFM SpecTarget ⇒ LFM Redline / batched prefill promoted” | **Rejected** — separate surfaces; see [`REDLINE.md`](REDLINE.md) §11.4 |
| Coherence-gate green as current acceptance | **Rejected** |
| Stitching Redline product bench + daemon harness into route proof | **Rejected** — [`REDLINE.md`](REDLINE.md) §5/§7 |
| Treating the [`BENCHMARKS.md`](BENCHMARKS.md) DFlash genre table as current baseline | **Rejected** — superseded methodology |
| Treating [`spec-decode-durability-2026-06-23.md`](spec-decode-durability-2026-06-23.md) as current baseline or procedure | **Rejected** — historical record |
| Inferred `admissions.yml` row from this inventory | **Rejected** |

## Upstream learned-drafter survey (context only)

Not in-tree support. Useful when planning; does not admit a route.

| Family | Upstream notes (as of original 2026-06 survey) |
|---|---|
| llama / qwen3 / mistral (0/1) | EAGLE-3 heads and small same-family draft models exist publicly |
| qwen2/2.5 (7) | EAGLE-3 heads; no native MTP in base weights |
| minimax (10) | Vendor declined MTP; community EAGLE-3 heads exist |
| qwen35-vl | VL research drafters (SpecVLM, ViSpec, …) need training |
| lfm2moe (11), cohere2moe (12), dots-ocr (8) | No public MTP/EAGLE suitable as drop-in; n-gram remains the only model-free path |

## Source anchors

| Concern | Path / symbol |
|---|---|
| SpecTarget + accept loop | `crates/hipfire-runtime/src/spec.rs` |
| n-gram drafter | `crates/hipfire-runtime/src/spec_ngram.rs` |
| Generic DFlash + DDTree | `crates/hipfire-runtime/src/dflash_generic.rs` |
| DSpark core / llama body | `crates/hipfire-runtime/src/dspark_core.rs`; `crates/hipfire-arch-llama/src/dspark_body.rs` |
| build_speculator / n-gram gate | `crates/hipfire-loader/src/spec_build.rs` |
| Carriers (per-arch load + DSpark/build_speculator) | `crates/hipfire-loader/src/carriers.rs` (llama 0/1 DSpark>DFlash>n-gram; deepseek4 DSpark>MTP; `Qwen4Carrier` native MTP and `admit_options`; `Gemma4Carrier` EAGLE; `MuseGlimmerCarrier` arch-23 draft; `MapleCarrier` no speculation) |
| qwen35 DSpark / DFlash / MTP load | `crates/hipfire-loader/src/lib.rs` (`finish_qwen35_load`, `dspark_speculator`, `resolve_qwen35_mtp_head_sourced`; dense-TP MTP head resolution) |
| qwen35 DFlash + deprecated contiguous knobs | `crates/hipfire-arch-qwen35/src/dflash_spec.rs` (`warn_deprecated_legacy_dflash_knobs`) |
| qwen35 / qwen4 MTP speculators | `crates/hipfire-arch-qwen35/src/mtp_speculator.rs`; `crates/hipfire-arch-qwen4/src/mtp_spec.rs`; qwen4 policy `crates/hipfire-loader/src/admission.rs` (`qwen4_native_mtp`) |
| Generation route selection | `crates/hipfire-generate/src/ar.rs` (`GenerationRoute`, `select_generation_route`, `deepseek4_spec_requested_from_policy`); `crates/hipfire-generate/src/qwen.rs` (`generate_dflash`, `generate_spec`) |
| Daemon draft guard / MTP presence | `crates/hipfire-daemon/src/main.rs` (`dflash_mode == "off"` draft skip; `mtp_weights_present`) |
| Config defaults (`speculation.*`, `MTP_K_DEFAULT`) | `crates/hipfire-config/src/lib.rs`; registry paired-DFlash layer `crates/hipfire-registry/src/lib.rs` |
| Per-arch SpecTarget | `crates/hipfire-arch-*/src/spec_impl.rs` |
| deepseek4 MTP greedy requirement | `crates/hipfire-arch-deepseek4/src/mtp_speculator.rs` — `requires_greedy` |
| Gemma4 EAGLE gate | `crates/hipfire-generate/src/dense.rs` (`HIPFIRE_GEMMA4_EAGLE`) |
| Arch ids | [`architecture-ids.md`](architecture-ids.md) |

## Historical appendix — n-gram seam work log (2026-06)

Retained for provenance of the four-arch n-gram bring-up. **Not** current
procedure and **not** acceptance evidence.

| Arch | Wiring outcome (2026-06-23) | Notes retained from that campaign |
|---|---|---|
| minimax (10) | Loader + daemon + GPU greedy AR vs n-gram token-identical on fixture | Emitter delimiter cosmetic difference vs bespoke AR possible |
| lfm2moe (11) | Loader + daemon + conv-state rollback; AR == n-gram byte-identical on fixture | Keep graph flags consistent with parity contract |
| cohere2moe (12) | Loader + daemon + `Cohere2MoeEmit`; AR == n-gram byte-identical | Fixed double `n_tokens` advance bug during bring-up |
| dots-ocr (8) | VL decode-phase n-gram after vision prefill; AR == n-gram byte-identical | Then slower than AR on measured OCR fixture (low τ / small GEMMs) |

2026-06-24 τ probes (gfx1151, campaign-local) falsified “batched verify fixes
all n-gram losses” for compute-bound small decoders; large BW-bound MoE remained
a modest/uncertain case only. Decision at that time: keep n-gram correct +
opt-in; prefer learned drafters for real speedups.

Shared integration pattern: arch-crate bundle owns `SpecTarget` (orphan rule);
loader re-exports; carrier `spec_target_guard` returns `InPlaceGuard`; the generation
route selector sends a spec-capable request to the arch's `*Spec` route **before** the
arch's plain-AR producer.

For any new arch, re-derive steps from source and the arch-port skill — do not
treat this appendix as an executable checklist that bypasses
[`VALIDATION.md`](VALIDATION.md).
