<!-- SPDX-License-Identifier: Apache-2.0 -->

# Narrow spec: Qwen3.5-VL vision serve bring-up (arch 5 carrier, HTTP stream contract)

Status: **branch-implemented on `feat/qwen35-vl` (2026-08-27)** — see the
as-built blockquote below; quality claims beyond the cited smokes remain
fail-closed until a `docs/VALIDATION.md` VL row exists. The VL route rows
landed 2026-08-27 (see the final blockquote); running a claim under its
route with fresh identity hashes is still required per [`VALIDATION.md`](../VALIDATION.md).
Author: engineering (hipfire agent)
Date: 2026-08-27
Parent family spec: [`qwen35-vl-mq4v2-spec.md`](../qwen35-vl-mq4v2-spec.md)
(§3 "Runtime — verify-only" plus §4 shared artifact contract; this record is
the scoped fix for the one part of §3 verification that FAILED live)
Sibling narrow spec:
[`docs/specs/2026-08-27-lfm2-vl-vision-runtime.md`](2026-08-27-lfm2-vl-vision-runtime.md) on
`feat/lfm2-vl` — the same two stream-contract fixes were landed there for the
arch-11 carrier; this branch ports them and nothing else from that branch.

## 0. Problem statement

The 2026-08-27 validation ledger
(`.codeinsight+research/vl-validation-2026-08-27/RESULTS.md`) recorded that
arch-5 VL bring-up passed locally (`run --image` OCR on the requantized
NuExtract3 artifact reads all fixtures correctly) while the HTTP image path
on `vl-serve-qwen35` (:6901) delivered **zero response bytes** — the daemon
ran tower + prefill + generation, but an OpenAI-client image turn saw an
empty stream, and repeated attempts could leave the single admission slot
held.

Root cause (finding b of the ledger, root-caused the same day on the sibling
branch): `generate_vl` and `generate_vl_dots_ocr` never emitted the
**`gen_start` stream-contract opener**. The HTTP layer's StreamContractGate
rejects every event that arrives without a preceding `gen_start` for the
request id, so all VL events after the opener-less stream began are dropped
before reaching the client socket. Text-path `generate()` has emitted the
opener since the e99583afa-class fixes; the VL entry points predate the gate
and were never retrofitted.

## 1. Goal and scope

1. `generate_vl` (the arch 5/6 qwen-VL arm reached via
   `VisionRoute::QwenVl`) emits `gen_start` as the FIRST event of its
   stream — before any GPU work, error path, or info event.
2. `generate_vl_dots_ocr` (arch 8 arm sharing the same HTTP surface)
   gets the identical opener.
3. Verify live over HTTP on :6901 that image turns stream visible bytes,
   and that client disconnect mid-image-turn releases the slot for the next
   request (ledger finding c re-check for THIS carrier).
4. Record binary md5s, artifact sha256, fixture hashes in the evidence
   ledger per §7 of the parent spec.

**Deliberately NOT ported from `feat/lfm2-vl`:**

- `VisionRoute::Lfm2Vl`, `LoadedModel::lfm2_vision()`,
  `has_vision_encoder()` — arch-11 loader/carrier API living on the sibling
  branch. The daemon `has_vl` gate here stays
  `m.vision_config().is_some() || m.dots_ocr().is_some()`, which already
  covers arch 5/6 (`vision_config` populated for `has_vision` artifacts —
  proven by the 2026-08-27 ledger container smokes on these binaries) and
  arch 8. Gate unification lands when the branches merge to master, not by
  cross-porting half an API.
- The lfm2-VL body (`generate_lfm2_vl`) and its abort-pair corrections —
  no arch-11 code belongs on this branch.
- `docs/VALIDATION.md` VL rows (§6 of parent spec): evidence from this
  bring-up justifies a future proposal; it does not write the rows.

## 2. Change specification

### 2.1 Opener placement invariant

The opener MUST be the first event written to the request stream. In both
functions it goes immediately after the CPU-sampler RNG reset (generate_vl)
or at function top (generate_vl_dots_ocr), before any early-return/error
path:

```rust
let gen_contract = crate::common::gen_start_contract_version_for_arch(m.arch_id);
emit_gen_start(stdout, params.id, false, gen_contract);
```

Contract-version selection goes through the shared helper — never a literal.
`params.id` (not the ambient attempt id) identifies the stream, matching the
text path.

### 2.2 Files touched

- `crates/hipfire-generate/src/vision.rs` — import `emit_gen_start`; insert
  one opener block per function (§2.1). No other change.
- Nothing else. No loader, daemon, engine, or carrier edits on this branch.

### 2.3 Abort-terminal audit (verification item, potential follow-up)

Finding (c) of the ledger concerned the *newly written* lfm2-VL body. On
this branch the existing abort polls already emit the canonical pair where
they exist (e.g. dots n-gram prefill: `emit_qwen_ar_cancelled`). Whether the
main generate_vl decode loop checks disconnects mid-generation at all was
NOT audited before the fix window. This spec makes it a live verification
item (§3 case D): disconnect mid-image-turn must free the slot within
generation-completion time. If it wedges permanently, the follow-up (abort
polls + canonical terminals inside the qwen-VL loops) gets its own scoped
commit and spec amendment — not folded silently into this one.

## 3. Verification plan (bring-up tier, gfx1101)

Route class: serving smoke of the hipfire-tester quick-start tier. Not a
VALIDATION.md claim route; no perf numbers.

| case | steps | pass condition |
|---|---|---|
| A build identity | `cargo build --release` (daemon AND cli); md5sum both | binaries differ from deployed :6901 snapshot; md5s recorded |
| B listing + text | swap container; `GET /v1/models`; text completion | model listed under filename-id; math prompt → correct greedy answer (append `/no_think` per ledger quirk) |
| C image turn streams | `/v1/chat/completions` with `image_url` data URI = committed fixture + OCR/desc prompts | streamed SSE chunks arrive incrementally (≥ several data events, not one empty terminal); decoded text matches the 2026-05-23 comparison readings incl. known-variance notes |
| D disconnect recovery | start image turn, kill client mid-stream, immediately issue text turn | text turn answers without restart; no permanent slot wedge |
| E local parity spot-check | `hipfire run --image` against same fixture | still coherent (opener added must not disturb the stdio contract consumers) |

Fail-closed notes: fixtures pinned by SHA-256; artifact remains
`~/AI/models/nuextract3/NuExtract3-mq4v2-vlval.mq4`
(sha256 `5bfea8b7…fe68c6` full hash re-recorded at swap time). Read every
decoded output by eye — the eyeball rule applies to image batteries even at
bring-up tier.

## 4. As-built

> **Implemented on `feat/qwen35-vl` (2026-08-27), same day.** All of §2
> landed: `gen_start` openers in `generate_vl` and `generate_vl_dots_ocr`
> (§2.1 verbatim, shared helper, no literals). §2.3's audit fired for real:
> the live disconnect probe found that arch-5 VL has NO abort polls at all,
> and a client killed mid-encode **wedged the single admission slot
> permanently** (three consecutive 30 s serve-queue timeouts spanning >3 min;
> daemon logs showed `[daemon-control] received abort` arriving while
> generation continued to completion with no recognized terminal ever
> emitted) — the finding-(c) class, reached through a different door than on
> arch 11. Scoped follow-up commit added two cancel polls to `generate_vl`
> (first prefill iteration + top of decode loop) emitting the canonical
> `emit_qwen_ar_cancelled` pair, mirroring the dots.ocr prefill-cancel
> precedent; partial state relies on the proven next-dispatch non-zero-
> seq_pos reset.
>
> Verified over HTTP :6901 (gfx1101, `NuExtract3-mq4v2-vlval.mq4`, fixtures
> sha256-pinned): text "391" ✓; doge OCR streams 186 content chunks /
> 6/6 captions ✓; scene_2 desc matches baseline incl. the one documented
> "MQUEEN" fine-print miss ✓; mid-encode disconnect → immediate follow-up
> answers in **8.81 s**, mid-decode → **1.04 s**, no wedge ✓. Binaries:
> daemon `ddd0baad…`, cli `fc954e92…`. Evidence:
> `.codeinsight+research/qwen35-vl-vision-serve-2026-08-27/RESULTS.md`
> (pre-fix wedge runs preserved as case_d/d2/d3 artifacts).
>
> Owed, fail-closed until landed: VL stream think-splitting — reasoning
> arrives inside `content` deltas with literal `</think>` / `<|im_end|>`
> marker chunks on this path while non-stream separates `reasoning_content`
> properly; the dots.ocr arm's n-gram decode-loop abort break still falls
> through to a done terminal (same class as §2.3, untested here — no dots
> fixture deployed); `docs/VALIDATION.md` VL rows per parent spec §5.7.

> **Follow-up landed same day (2026-08-27 audit pass): the think-splitting
> debt is closed.** `generate_vl` decode emission now routes through the
> shared text-path machinery — `EosFilter` (`qwen_ar_eos_filter_config`:
> UTF-8 boundaries + `<|im_end|>`/`<|endoftext|>` suppression) →
> `ThinkOutputRouter` (chunk-boundary-invariant channel split) → typed
> `emit_visible_token` / `emit_reasoning_token` envelopes — at both emission
> sites (main decode loop + think-cap force-close) with a `finish_into`
> flush before the terminal, and `gen_start.started_in_think` now reflects
> the real assistant prefix instead of a hardcoded false. This replaces the
> hand-rolled byte emission that violated the v2 typed-channel contract the
> opener itself claims (arch 5 advertises contract 2, under which the CLI
> appends `token` text to `content` verbatim — no marker scan; the old
> envelope also spliced the user-supplied `id` into the JSON unescaped —
> the typed emitters escape it). The dots.ocr decode-loop abort now emits
> the canonical cancelled pair + return instead of falling through to a
> done terminal (§2.3 class; still live-untested — no dots fixture
> deployed). Quantizer companion fix: `has_vision` latches only after every
> name-based skip gate, so a gemma4 unified checkpoint quantized with
> `--include-vision` can no longer record `has_vision: true` while the
> gemma4 text-only gate drops every vision tensor. 5 unit tests cover the
> routing (`vision::tests`). Verified over HTTP :6901 (daemon
> `d4d63ce0…`, cli `25bedc84…`): thinking-on doge desc stream 363
> reasoning / 31 content chunks, zero markers, answer correct (pre-fix
> baseline same turn: 0 reasoning chunks, `</think>`+`<|im_end|>` in
> content); non-stream separates `reasoning_content`, finish stop; doge OCR
> 6/6 captions with content byte-identical to the pre-fix answer segment;
> text math "391" unchanged at max_tokens=256; mid-decode kill → follow-up
> in 1.03 s, no wedge; `run --image` stdio parity in a one-off container
> (6/6 captions). Evidence:
> `.codeinsight+research/qwen35-vl-audit-2026-08-27/RESULTS.md`.
>
> The `docs/VALIDATION.md` VL rows per parent spec §5.7 landed 2026-08-27
> (same day), confirmed by a from-scratch container rebuild at HEAD
> `c91ae3db` (daemon md5 unchanged `d4d63ce0…`): stream typed-emission
> census 156 reasoning / 27 content chunks, zero marker leakage; doge OCR
> 6/6 captions; mid-encode kill → follow-up 4.69 s and true mid-decode
> kill (252 chunks already streamed) → follow-up 1.04 s, no wedge.
> Evidence: `.codeinsight+research/vl-serve-qwen35-rebuild-2026-08-27/RESULTS.md`.
> Still owed, fail-closed: dots.ocr live abort probe.
