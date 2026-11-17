# Pick a model for your GPU

A starting point, not a guarantee. VRAM and size come from the registry
(`registry/models.json`, `min_vram_gb` / `size_gb`); `hipfire list -r` is the
authoritative list. The tok/s figures in [By VRAM](#by-vram) are copied from
[BENCHMARKS.md](BENCHMARKS.md), whose tables are **historical**: each was
measured once on an older build under the fixture named in its column, not on
your card or the current binary. [Measured on v0.4.4](#measured-on-v044) has
dated measurements of the current release. Use either to compare sizes and
routes, not as a promise.

## By VRAM

| Your VRAM | Tag | Size | Speculation setting | Expected decode (historical) |
|---|---|---:|---|---|
| 2–4 GB | `qwen3.5:0.8b` | 0.55 GB | defaults (`speculation=auto`, `dflash_mode=off`) | 391 tok/s on 7900 XTX [A]; 250 RX 6800 XT, 190 RX 5700 XT, 231 MI50 [C] |
| 4–6 GB | `qwen3.5:4b` | 2.59 GB | defaults | 180 tok/s on 7900 XTX [A]; 61 RX 5700 XT, 61 MI50 [C] |
| 8–12 GB | `qwen3.5:9b` | 5.31 GB | defaults; for code, `dflash_mode auto` with `qwen3.5:9b-draft` | AR 132 tok/s on 7900 XTX [A]; 65 RX 6800 XT, 59 MI50 [C]. DFlash on code: 329–373 tok/s vs 122–124 AR; on prose it lost (98–99 vs 123–125) [B] |
| 16–24 GB | `qwen3.5:27b` | 15.0 GB | defaults; for code, `dflash_mode auto` with `qwen3.5:27b-draft` | AR 47 tok/s on 7900 XTX [A]; 22 RX 6800 XT, 21 MI50 [C]. DFlash on code: 196.0 (3.5) / 185.5 (the retired 3.6) tok/s vs ≈44 AR; prose/instruct ≈1.0–1.1× [B]. The current 27B flagship is `qwen3.8:27b-mq4-xts`; see [Measured on v0.4.4](#measured-on-v044) |

`min_vram_gb` in the registry: 0.8B 2, 4B 4.1, 9B 6.8, 27B (Qwen 3.5 / 3.6 MQ4) 16.
A 27B MQ4 trunk does not fit on 8 GB cards (OOM on RX 5700 XT and BC-250 in
[C]); the lowest 27B floors in the registry are the 1-bit and ternary Bonsai
tags (5 and 8 GB, below). Long context needs more room than the weights
alone; see the KV notes in
[GETTING_STARTED.md](GETTING_STARTED.md#long-context-optional).

### Qwen3.8-27B by card

The default `qwen3.8:27b` is `qwen3.8:27b-mq4-xts` (14.99 GB). On the 27B tiers
that pair a DFlash draft (`mq3*`, `mq4*`), `dflash_mode` resolves to `auto`, so
a pulled draft (1.21 GB for mq4, 0.98 GB for mq3) loads beside the weights and
takes context room. The `mq3*` tiers pair their draft only on a card with at
least 18 GiB of VRAM (`dflash_min_vram_gb`): a 16 GB card loads them without it.
Each result below is one load running an 8-turn document
chain that grows to a 35.5K-token prompt (greedy, thinking off, default KV). The
card's capacity was emulated with `HIPFIRE_EMU` on a 7900 XTX (gfx1100) or an
R9700 (gfx1201), headless, on build `bbe7d51a2c`. The emulation caps hipfire's
managed allocations, not the whole card: under the 20 GiB cap the whole card
peaked at 20.8–21.8 GiB. A real card with a display attached can therefore
admit less context than these numbers.

| Card | Pick | Measured |
|---|---|---|
| 7900 XTX (24 GB) | `qwen3.8:27b` (defaults, paired DFlash) | Admitted context 104,086 (no emulation). The 35.5K chain completes; DFlash decode averaged 50.1 tok/s. |
| 7900 XT (20 GB) | `qwen3.8:27b` with `hipfire config qwen3.8:27b set dflash_mode off`, or `qwen3.8:27b-mq3-xts` | Defaults (mq4-xts + draft): admitted context 15,420, out of memory on a 17.7K-token prompt. mq4-xts AR: context 80,956, the chain completes at 47.2 tok/s. mq3-xts + draft: context 123,361, the chain completes at 26.7 tok/s; gfx1100 has no PM4 decode route for MQ3 + DFlash yet. |
| 16 GB (7800 XT, 9070 XT) | `qwen3.8:27b-mq3-xts` (defaults: below the 18 GiB floor its draft is not paired), or `qwen3.5:9b` | `qwen3.8:27b` (mq4-xts) is refused at load: AR on both 7800 XT and 9070 XT emulation, and with the draft on the 9070 XT. 9070 XT emulation, mq3-xts AR (fp8 KV): context 77,220, the chain completes at 43.7 tok/s. mq3-xts + draft (forced with `dflash_mode on`): context 8,128, out of memory on a 21.8K-token prompt. |

`hipfire fit qwen3.8:27b --ctx N` evaluates every `qwen3.8:27b-*` tier on your
GPU, with the paired draft charged against free memory, and names the
largest tier that fits ([CLI.md](CLI.md#hipfire-fit)).

## Measured on v0.4.4

Copied from the README's
[Measured performance (v0.4.4)](../README.md#measured-performance-v044)
table, which names the fixture, method and source of each row. Each is one
measurement on the named card, date and build, not a floor for your prompts or
host. Decode is tok/s per request.

| Card | Tag or file | Route | Result | Date, build |
|---|---|---|---|---|
| 7900 XTX (24 GB) | `qwen3.8:27b-mq4-xts` | native MTP (`--spec mtp`), Q8 KV | decode 136.5–136.8 (code 1K), 141.1 (prose 1K), 129.5–129.9 (code 32K), 79.7–80.1 (prose 32K) | 2026-10-26, beta `085dd189b5` + #998 |
| R9700 (32 GB) | `qwen3.8:27b-mq4-xts` (+ paired DFlash draft, + MTP head) | AR / DFlash / native MTP, committed `humaneval_0_has_close_elements` (code) and `fiction_lighthouse` (prose) prompts | decode AR 41.2; DFlash 313.6 (code), 63.4 (prose); MTP 95.0 (code), 55.5 (prose); prefill 4,936 (pp512), 5,278 (pp2048), 4,888 (pp16384) | 2026-10-27, published v0.4.4 binary |
| R9700 (32 GB) | `qwen3.8:27b-mq4-xts` + paired DFlash draft, fp8 KV | DFlash, long context | decode 59.60 (16K), 51.40 (32K), 43.37 (64K), 33.70 (128K) | 2026-10-26, 0.4.4 cycle beta |
| Strix Halo | `qwen3.8:27b-mq4-xts` + paired DFlash draft | DFlash | decode 135.3 (code) | 2026-10-13, v0.4.2 |
| R9700 (32 GB) + host RAM | `qwen3.8:flash-next` (split base) | AR / native MTP (`--spec mtp`), hot set, same two prompts, 640 tokens | decode AR 49.7 (code), 45.6 (prose); MTP 82.9 (code), 48.5 (prose); prefill 6,647.8 (pp16384) | 2026-10-27, published v0.4.4 binary |
| Strix Halo + XDNA2 | `qwen3.8-flash-next-gptq3.mq4` | NPU trunk prefill | 16,355-token prefill 2,472.7 tok/s | 2026-10-27, `943b5323da` |
| R9700 (32 GB) | `qwen3.6:35b-a3b-mq4rs` (measured on gfx1201) | AR, `bench --matrix` | decode 242.2 (tg128 at 128 context); prefill 11,788 (pp512), 19,867 (pp4096), 18,567 (pp16384) | 2026-10-27, published v0.4.4 binary |

## Larger and newer families (registry floors)

These tags have no rows in [BENCHMARKS.md](BENCHMARKS.md). The ones measured
on v0.4.4 are in [Measured on v0.4.4](#measured-on-v044); for the rest only
size and the registry VRAM floor are given and no speed is claimed. Pick the
tier whose `min_vram_gb` fits with headroom for KV, and read
[MODELS.md](MODELS.md) for the family notes.

| Your VRAM | Tag (`min_vram_gb`) | Size | Note |
|---|---|---:|---|
| 5–8 GB | `bonsai:27b-bq1` (5) / `bonsai:27b-tq2` (8) | 3.8 / 7.16 GB | 27B at 1.14 / 2.125 bpw; heavy quantization, see the registry `desc` for its KLD |
| 12 GB | `qwen3.5:27b-mq3` (12), `qwen3.6:27b-mq3` (12) | 10.7 GB | MQ3 alpha, gfx11 / gfx12 |
| 13–15 GB | `qwen3.8:27b-mq3-xts` (13, symmetric), `qwen3.8:27b-mq3-xt` (13), `qwen3.8:27b-mq3` (14), `qwen3.8:27b-mq3-pro` (15) | 11.78–13.18 GB | MQ3V2 tiers; `-mq3-xts` runs the W3A4 prefill on gfx1100 and gfx1201. For the 27B on 16 and 20 GB cards see [Qwen3.8-27B by card](#qwen38-27b-by-card) |
| 16–18 GB | `qwen3.8:27b-mq4-xt` (16, speed), `qwen3.8:27b-mq4` (17, asymmetric, the pre-0.5.0 default), `qwen3.8:27b-mq4-pro` (18) | 14.99–16.46 GB | MQ4V2 tiers; `qwen3.8:27b-mq4-xts` (16) is the registry default (`qwen3.8`, `qwen3.8:27b`) and the canonical dense fixture: paired KLD +0.0008 vs `qwen3.8:27b-mq4` (gfx1201 A4 route), decode 4.8% faster, 0.67 GB smaller |
| 20–24 GB | `qwen3.8:27b-mq5*` (20–21), `qwen3.8:27b-mq6*` (23–24) | 18.18–22.17 GB | higher-quality V2 tiers |
| 22 GB | `qwen3.6:35b-a3b` (22), `ornith-1.5:35b-a3b` (22), `nex-n2:mini` (22) | 19–19.8 GB | 35B / 3B-active MoE |
| 24 GB | `muse-glimmer:fast` (24) / `muse-glimmer` (26) | 16.26 / 18.61 GB | 30B dense, thinking model |
| 48–72 GB, or less with host RAM | `qwen3.8:flash` → `qwen3.8:flash-next` (72), `-mq3-xts` (64), `-mq2-xts` / `-mq2-xts-4500` (48) | 70.9 / 59.1 / 46.9 / 44.5 GB, plus the shared PLE sidecar (29.1 GB q4 rows) | Flash-Next split base (arch 16). A smaller card maps the remaining experts from host RAM (the 0.4.4 release smoke ran it on a 32 GB R9700 and a 24 GB 7900 XTX); a 128 GB Strix Halo holds it in unified memory. See the [README](../README.md#quickstart) |
| 96 GB | `deepseek-v4-flash` (96), `minimax-m2.7` (96) | 82 / 79.2 GB | MQ2-class MoE |

Qwen 3.8 27B tiers also pull the shared vision tower (`vision_mode`, default
`auto`, streamed from host/disk per image; `resident` keeps it in ~0.94 GiB of VRAM) and the shared MTP head. Image generation (`flux.schnell:1`, 24 GB) is a
separate surface: [IMAGEGEN.md](IMAGEGEN.md).

## Fixtures

- **[A]** [Historical AR decode, 7900 XTX (gfx1100)](BENCHMARKS.md#historical-autoregressive-decode-no-spec--7900-xtx-gfx1100):
  MQ4, then-default asym3 KV (removed in 0.5.0; on Qwen `asym3` was an alias
  of `fwht3`), median of 5 runs, no speculation.
- **[B]** [Historical DFlash by genre, 7900 XTX](BENCHMARKS.md#historical-dflash-speculative-decode-by-genre--7900-xtx):
  asym3 KV (same removal), `--no-chatml`, `max_tokens=120`, DFlash forced on, median of 5.
  Superseded methodology; the speedup depends on the prompt genre.
- **[C]** [Historical other arches](BENCHMARKS.md#historical-other-arches-decode-toks):
  then-default configuration, decode tok/s.

## Speculation settings

- **Defaults** (`speculation=auto`, `dflash_mode=off`): native MTP engages
  under `auto` only on a producer with a measured AR floor (today Flash-Next /
  Qwen4); Qwen3.5-class and DeepSeek V4 stay AR until `--spec mtp` opts in.
  DFlash stays off. Force plain AR with
  `hipfire config set speculation off`. Exception: `qwen3.8:27b` and its
  `mq4*` tiers, the `mq3*` tiers on a card with at least 18 GiB of VRAM,
  and the `mq5*` tiers on a card with at least 28 GiB, resolve
  `dflash_mode=auto` from the registry, so a
  paired draft file that is present is used (absent: MTP `auto` as before,
  which is AR for this Qwen35 producer);
  `hipfire config <tag> set dflash_mode off` opts out
  ([CONFIG.md](CONFIG.md#qwen38-27b-paired-dflash-default)).
- **DFlash** elsewhere is opt-in: `hipfire config <tag> set dflash_mode auto`.
  `hipfire pull <tag>` already fetches a draft the registry declares for that
  tag (for example `qwen3.5:9b`, `qwen3.5:27b`, `qwen3.8:27b`), and the
  separate `-draft` tags are listed in [MODELS.md](MODELS.md). It helped code
  prompts and hurt prose in fixture [B]; measure your own workload before
  leaving it on.

## Not recommended

The legacy contiguous-DFlash knobs and the Givens `legacy-asymN` / `asymN` /
`turboN` KV names were removed in 0.5.0; use a
`kv_cache` of `auto`, `q8` or `fwhtN`. See
[Measured on v0.4.4](#measured-on-v044) and the dated
[perf checkpoints](perf-checkpoints/) for newer-family measurements.
