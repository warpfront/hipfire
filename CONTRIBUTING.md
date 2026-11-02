# Contributing to hipfire

hipfire is under active development; see [CHANGELOG.md](CHANGELOG.md) for
what is shipping. Real-world testing on cards we don't have, kernel work
on archs we don't ship for, bug reports with full reproduction, and new
model architecture support — all welcome.

## Two ways to help, no Rust required

Both paths below use only the installed binaries (the `hipfire` CLI,
`daemon`, and `hipfire-quantize` when its build succeeds, placed in
`~/.hipfire/bin/` by `scripts/install.sh`). They need no source edits and
no Rust knowledge; the installer itself builds from source, so it needs
`cargo` and a ROCm install (see
[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)).

### 1. Run the bench matrix on your GPU

If you have an RDNA card the maintainer doesn't, the highest-leverage
thing is running the standard tester workflow and posting numbers.

```bash
hipfire diag                            # ROCm + arch detection sanity check
hipfire pull qwen3.5:0.8b               # ~0.55 GB; needs 2 GB+ VRAM
hipfire pull qwen3.5:9b                 # ~5.3 GB; needs 6.8 GB+ VRAM
hipfire bench qwen3.5:0.8b --runs 5     # decode + prefill tok/s over 5 runs
hipfire bench qwen3.5:9b   --runs 5
```

For 16 GB+ cards, also pull and bench `qwen3.5:27b` (~15 GB). For WSL2,
native Windows or multi-GPU setups, see
[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) and
[docs/multi-gpu.md](docs/multi-gpu.md). Live tags and VRAM floors:
[docs/MODELS.md](docs/MODELS.md).

Open an issue titled `Benchmarks: <your GPU>` and paste the `diag`
output + each `bench` block (attach `--json` output if you can). A report
is evidence for the maintainers to read; it is not automatically added to
[docs/BENCHMARKS.md](docs/BENCHMARKS.md), which is a historical page. The
`hipfire-tester` skill in `.agents/skills/hipfire-tester/` walks an AI
agent through bring-up and drafting the report if you want help.

### 2. Diagnose and report a bug

```
hipfire diag                            # capture everything first
```

Open an issue with: GPU + ROCm version, exact command, full error
output (not just the last line), and the diag output. The
`hipfire-autoheal` skill (in `.agents/skills/hipfire-autoheal/`) is a
fix-catalog walkthrough that an agent can apply on your behalf for
common runtime issues; if it doesn't resolve cleanly, that's exactly
the case we want filed.

---

## Developer workflow

### Setup

```bash
git clone https://github.com/warpfront/hipfire
cd hipfire
cargo build --release --locked -p hipfire-daemon
cargo build --release --locked -p hipfire-cli
cargo build --release --features deltanet --example test_kernels -p hipfire-runtime
cargo build --release -p hipfire-quantize
./scripts/install-hooks.sh
```
Requires current stable Rust (CI tracks `stable`; 1.98 is what the
maintainers build with — no MSRV is declared yet) and ROCm 6+ (ROCm 6.4+
for RDNA4, 7.2+ for gfx115x; the dev workflow needs `hipcc` for kernel
JIT). Release tags ship prebuilt kernel packs for the admitted GPUs
(gfx1201, gfx1100, gfx1151, gfx906, gfx942); other arches JIT-compile on
first load. See
[Prebuilt kernel packs](docs/GETTING_STARTED.md#prebuilt-kernel-packs).

`scripts/install-hooks.sh` is idempotent; it sets
`core.hooksPath=.githooks` and makes the local pre-commit hook
executable.

### No-GPU CI subset

The default CI path intentionally avoids AMD GPU access and model
downloads:

```bash
./scripts/no-gpu-ci.sh
```

It runs `cargo check --workspace --examples`, no-GPU Rust unit tests,
CPU Python tests, the env/docs drift check and the lifecycle check.
The jobs in [`ci.yml`](.github/workflows/ci.yml) include
`build (workspace, no GPU)`, `unit + integration tests (no GPU)` and
`gates (ratchets, layering, registers)` (ratchets, layering, crate maps,
changelog; not part of `no-gpu-ci.sh`). Merge also requires an approving
review — see
[docs/VALIDATION.md](docs/VALIDATION.md) § Merge bar. Hardware-relevant
changes still owe claim-matched GPU/model evidence for direct review; that
evidence is not a separate required CI check.

### Tokenizer differential test (HF oracle)

`hipfire_runtime::tokenizer::Tokenizer::from_hf_json` is checked against the
HuggingFace `tokenizers` crate for every tokenizer in
`crates/hipfire-runtime/tests/fixtures/hf_tokenizer_differential/manifest.json`
(`family`, `file`, `md5`, `source`, `revision`; `file` is relative to that
directory, and the Qwen3.8 entries reuse
`../hf_tokenizer_corpus/tokenizer.json`). The oracle is pinned
`tokenizers =0.22.2` with the pure-Rust `fancy-regex` backend plus `md5`, both
`[dev-dependencies]` of `hipfire-runtime` only: no C/C++ regex library, no
Python, no network and nothing in the shipped binaries. `tokenizers`,
`esaxx-rs` and `spm_precompiled` are Apache-2.0; the remaining resolved
licenses are permissive and allowed by `deny.toml`.

```bash
cargo test --release -p hipfire-runtime --test hf_tokenizer_differential   # all families
cargo test --release -p hipfire-runtime --test hf_tokenizer_differential qwen35 -- --nocapture
```

Each family is its own test (`qwen3`, `qwen25`, `qwen35`, `qwen36`, `qwen38`,
`qwen38_flash_next`, `dots_ocr`, `deepseek4`, `minimax`, `minimax27`, `lfm2`,
`lfm25`, `lfm25_moe`, `llama_bpe`, `muse_glimmer`, `north_mini_code`, `maple`,
`ornith`, `vibethinker`), so one failing family never skips another;
`manifest_integrity` verifies every MD5, and `unlisted_manifest_families` runs
any manifest family without a dedicated test. Per tokenizer the test compares
exact IDs (`add_special_tokens=false`) on all 945 cases in
`hf_tokenizer_corpus/cases.jsonl`, every sorted `benchmarks/prompts/*.txt`,
and a deterministic seeded adversarial corpus (whitespace, tabs, CRLF, unicode
spaces, digit runs, combining marks, emoji/ZWJ, CJK, code, JSON, added-token
boundaries). Where HF `decode(ids)` reproduces the input it also checks
hipfire decode and decode→re-encode. Mismatches never stop the run: every one
is recorded and minimised, and the test fails at the end.

| Env var | Default | Meaning |
|---|---|---|
| `HIPFIRE_TOKENIZER_FUZZ_CASES` | `50000` | adversarial cases per tokenizer |
| `HIPFIRE_TOKENIZER_FUZZ_SEED` | built-in | u64 (decimal or `0x`); xor-ed with a per-family hash |
| `HIPFIRE_TOKENIZER_DIFF_REPORT_DIR` | `<temp>/hipfire_tokenizer_differential` | where reports are written |
| `HIPFIRE_TOKENIZER_DIFF_MIN_PROBES` | `5000` | minimisation budget per unique signature |
| `HIPFIRE_TOKENIZER_DIFF_MIN_LIMIT` | `2000` | max signatures minimised per family (rest reported as `skipped_limit`) |
| `HIPFIRE_TOKENIZER_DIFF_PRINT_LIMIT` | `50` | root signatures printed per family |

Reports per family (complete JSONL, nothing capped or dropped):
`<family>.failures.jsonl` (every failing case/kind with full input and both ID
streams), `<family>.minimized.jsonl` (one line per unique signature: minimal
repro, ID streams, every contributing case source) and `<family>.summary.json`
(counts and one minimal repro per distinct root signature). The console prints
counts and the minimal repro for each root signature.

Fixtures are immutable inputs: refresh a tokenizer by replacing the file *and*
its `md5`/`revision` in the manifest. When a runtime fix lands, append the
minimal repro to `hf_tokenizer_differential/regressions.jsonl`
(`{"family","text","ids"}`); the test then pins HF and hipfire to those exact
IDs. Do not shrink the 945-case corpus.

#### Encode throughput benchmark (opt-in, single thread)

`crates/hipfire-runtime/tests/hf_tokenizer_encode_bench.rs` is an `#[ignore]`d
release-only benchmark of hipfire vs HF `encode` (and HF `encode_fast`) on the
pinned Qwen3.8 tokenizer: the committed 138-, 55- and 14-request corpus
groups, 64K- and 256K-token documents cycled from `benchmarks/prompts/*.txt`,
and 64K NFC-clean non-ASCII / NFC-adversarial documents. The shipped NFC
normalizer is compared both to an in-memory no-normalization variant
(`normalizer: null`) and to an **eager NFC** arm that normalizes every input
with HF's Unicode 9 tables before encoding with the no-normalizer tokenizer.
The eager arm isolates the allocation-free NFC fast-path benefit without
changing normalization semantics. It prints one JSON line per group/nfc/engine
(requests, bytes, tokens, median `elapsed_ns`, `mb_per_s`, `tokens_per_s`,
`ms_per_request`, `ids_equal`) plus `hermes_estimate` records that scale
the committed 138-request group to a 115-request pass. It is not part of
CI.

```bash
cargo test --release -p hipfire-runtime --test hf_tokenizer_encode_bench \
    -- --ignored --nocapture --test-threads=1
# HIPFIRE_TOKBENCH_REPS=7 (default), HIPFIRE_TOKBENCH_ONLY=tc55,doc64k
```

### GPU kernel correctness check

```bash
./target/release/examples/test_kernels      # ~30s, no model needed
```

Validates every dispatched kernel against a CPU reference on the
detected arch. This is the load-bearing correctness gate for any
arch port; if it fails on your hardware we want to hear about it
(see issue template / autoheal skill).

### Runtime and performance validation

Any change to kernels, dispatch, fusion, rotation, rmsnorm, sampling,
the spec-decode path, loader/daemon, or the forward pass MUST validate the
actual path under test and attach claim-matched evidence for **direct
maintainer review**. Use the harnesses below (and the routes in
[docs/VALIDATION.md](docs/VALIDATION.md)). Optional automation
([`.github/workflows/hw-gate.yml`](.github/workflows/hw-gate.yml),
[`scripts/hw-gate/`](scripts/hw-gate/)) may deliver the same class of
evidence when it runs — a maintainer can apply `hw-run` to force a hardware
pass — but hw-gate is **not** a required status check, not automatic
acceptance, and not a substitute for the approving review. The fixed
`coherence-gate*.sh` batteries and the retired `tools/change_gate` /
agentic-review route are historical only and must not be used as acceptance
evidence. No local planning tool is merge evidence.

```bash
python3 scripts/redline_daemon_harness.py --model /path/to/model --pm4
python3 scripts/serve_harness.py --model /path/to/model --mode battery --sampling greedy
./scripts/speed-gate.sh --fast          # 4B prefill+decode regression vs tests/speed-baselines/<arch>.txt
```

**Don't bypass with `--no-verify`.** A regression the gate catches is
information. Authorized exceptions need explicit written sign-off from
the maintainer for that specific change. Read
[docs/methodology/perf-benchmarking.md](docs/methodology/perf-benchmarking.md)
before claiming any perf win — there is no universal noise band; you
characterize noise on your own fixture, and the bench harness has a
stale-binary trap that has bitten us before.

### New kernel files

```bash
# Kernel source: kernels/src/<name>.hip
# Per-arch overrides: kernels/src/<name>.gfx12.hip       (family tag)
#                     kernels/src/<name>.gfx1100.hip     (chip tag)

# Register the source in: crates/rdna-compute/src/kernels.rs
# Packaged module names come from the exact registry in
#   crates/rdna-compute/src/kernel_registry.rs (never from .hip file names)
# Wire dispatch in: crates/rdna-compute (dispatch.rs, gemm.rs, gemv.rs, ...)
#   and the hipfire-dispatch family tables

# Compile-check the arches your change touches (packages from the exact
# registry into kernels/compiled/<arch>/, which is gitignored; with no
# arguments it builds the default admitted set):
./scripts/compile-kernels.sh gfx1100 gfx1201
```

Without a matching packaged index the runtime JIT-compiles with `hipcc`
(see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) § Kernel build).

Architecture deep-dive: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
Quantization design (MQ4 / HF4 / asym KV math):
[docs/QUANTIZATION.md](docs/QUANTIZATION.md). Tuning an existing
kernel for perf (multi-row, K-tile depth, wave64 port, prefetch,
ISA flags) — see [`.agents/skills/hipfire-kernel-tuning/`](.agents/skills/hipfire-kernel-tuning/),
which catalogs the empirical methodology + every lever this repo's
git log has actually used.

### Porting to a new GPU arch

The `.agents/skills/hipfire-arch-port/` directory is the canonical entry
point — playbook, WMMA matrix, validation procedure, contributor
onboarding. Don't write code before reading it; getting WMMA
C-mappings wrong produces silent corruption.

Reference: PR #56 (gfx12 WMMA port and channel tests) walked the skill
end-to-end.

### New model architectures

Start with [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (§ Model sources,
§ Carriers, § Architecture crates) and
[`.agents/skills/hipfire-arch-port/`](.agents/skills/hipfire-arch-port/).
A new family is a new `crates/hipfire-arch-<name>/` crate that implements
`hipfire_runtime::arch::Architecture` (copy `crates/hipfire-arch-toy/`,
and see `crates/hipfire-arch-qwen35/src/arch.rs` for a production
implementation), plus a carrier registered in `hipfire-loader` and an id
in [docs/architecture-ids.md](docs/architecture-ids.md).

Quantize from the original safetensors; GGUF weight input is deprecated
and will be removed in 0.5.0 (see [docs/QUANTIZE.md](docs/QUANTIZE.md)).

### Branch naming

| Type | Pattern |
|---|---|
| Feature | `feature/<short-name>` |
| Bug fix | `fix/<short-name>` |
| Arch port | `port/<arch>-<kernel>` |
| Benchmark contribution | `bench/<gpu-name>` |

### PR template

Concise description, before/after numbers if perf-sensitive, mention
which gates passed. One logical change per PR; the PR template
([`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md))
lists the evidence each surface owes. Format only the files you changed
with `scripts/fmt-changed.sh <master|beta>` (base ref = the branch you
cut from); bare `cargo fmt` is blocked on purpose. `beta` is rustfmt-clean
since the 0.4.3 sweep and `scripts/pre-commit-ci.sh` checks the whole
workspace, so formatting your files is all it takes; a branch cut before
the sweep replays onto it with `scripts/fmt-sweep-rebase.sh <sweep-sha>`
(then rebase onto `beta` as usual). CI's rustfmt
and clippy jobs are advisory; the pre-commit hook still checks rustfmt on
staged files.

For perf claims: include the binary md5
(`md5sum target/release/examples/bench_qwen35_mq4`) and the prompt
md5 if the bench is prompt-dependent. Without these, the result is
unreproducible — see
[docs/methodology/perf-benchmarking.md](docs/methodology/perf-benchmarking.md).

### Code style

- `scripts/fmt-changed.sh <master|beta>` — formats only your changed
  files; never bare `cargo fmt` (it is blocked in this repo).
- `cargo clippy` — no new warnings (advisory in CI).
- **No Python in the inference hot path.** Python is fine for
  tooling, benchmarks, and offline analysis; never in the engine.
- Comment HIP kernel parameters: VGPR budget, wave occupancy, LDS
  usage, K-tile depth — anything a reader needs to understand the
  perf shape without inspecting `--save-temps` output.

### Release cadence and PR cutoff

Starting after 0.4.0, hipfire releases on a fixed, published schedule. That way contributors know exactly when a merged PR will ship and can plan around it.

- **One release per week, alternating between Tuesday and Sunday.** Weeks run Sunday to Saturday, and a week never has both a Sunday and a Tuesday release. So a Tuesday release is followed by the next Sunday (5 days later), and a Sunday release by the Tuesday of the *following* week (9 days later). Each release bumps the patch version by 0.0.1:

  | Version | Scheduled release date |
  |---|---|
  | 0.4.1 | Tuesday 2026-10-06 |
  | 0.4.2 | Sunday 2026-10-11 — **exception:** actual cut Wednesday 2026-10-14 04:00 UTC |
  | 0.4.3 | Tuesday 2026-10-20 |
  | 0.4.4 | Sunday 2026-10-25 |
  | 0.4.5 | Tuesday 2026-11-03 |
  | 0.4.6 | Sunday 2026-11-08 |

  **0.4.2 exception:** 0.4.2 is cut on Wednesday 2026-10-14 at 04:00 UTC (9:00 PM MST, Tuesday 2026-10-13) instead of its scheduled Sunday slot. This is a one-off; the dates for 0.4.3 and later are unchanged. The same pattern continues after the table.
- **Packaging deadline:** each release is packaged and published by **11:59 PM MST on its release day**. That's 06:59 UTC the following day; for 0.4.1, it's by 11:59 PM MST Tuesday 10/06, which is 06:59 UTC Wednesday 10/07. For 0.4.2 the 04:00 UTC cut above applies instead.
- **Features** always go into the *upcoming* release. They land on `beta`, which is packaged and released on the next release day.
- **Bug fixes** can go into the current release and the upcoming one.
- **Cutoff:** a PR filed before **11:59 PM MST (UTC−7) on the day before a release date** is incorporated into that release. MST is fixed year-round (no daylight saving), so the cutoff is always **06:59 UTC on release day**:

  | Zone | Summer time (Europe until Oct 25, US until Nov 1, 2026) | Winter time (after that, until spring) |
  |---|---|---|
  | UTC | 06:59, release day | 06:59, release day |
  | US Pacific | 11:59 PM PDT, day before | 10:59 PM PST, day before |
  | US Mountain | 12:59 AM MDT, release day | 11:59 PM MST, day before |
  | US Eastern | 2:59 AM EDT, release day | 1:59 AM EST, release day |
  | Central Europe | 08:59 CEST, release day | 07:59 CET, release day |
  | India | 12:29 IST, release day | 12:29 IST, release day |
  | China | 14:59 CST, release day | 14:59 CST, release day |

  For 0.4.1 that means before 06:59 UTC on Tuesday 2026-10-06 (11:59 PM MST, Monday 10/05).
- **Every PR still has to pass the gates in this document.** A PR filed before the cutoff that doesn't pass rolls to the next release.

---

## Crate topology

The workspace map, layering rules and request lifecycle are owned by
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); per-crate maps follow
[docs/crate-maps.md](docs/crate-maps.md) and arch ids are in
[docs/architecture-ids.md](docs/architecture-ids.md). Short summary:

```
crates/
  hip-bridge / hsa-bridge  HIP / HSA FFI (dlopen, no link-time ROCm)
  rdna-compute             kernel dispatch, JIT / packaged-kernel loader,
                           arch capability predicates
  hipfire-dispatch         per-kernel-family dispatch tables
  saddle-core              grammar, KV, caps, sampling policy
  hipfire-runtime          arch-agnostic runtime + `Architecture` trait:
                           HFQ/safetensors, tokenizer, sampler, loop_guard,
                           prompt_frame, eos_filter, spec primitives
  hipfire-arch-*           one crate per model family, keyed by arch_id
                           (qwen35, qwen35-vl, qwen2, gemma4, ...);
                           `hipfire-arch-llama` is a facade (the dense
                           forward lives in hipfire-runtime);
                           `hipfire-arch-toy` is the new-port template
  hipfire-loader           composition root: carriers, load_model
  hipfire-engine           arch-free serve engine
  hipfire-generate         generation bodies needing arch + engine
  hipfire-daemon           the `daemon` binary (JSON-lines / HTTP server)
  hipfire-quantize         safetensors → .mq4 / .hfq quantizer CLI
  hipfire-cli / -config / -registry / -client / -tui
                           operator surface: CLI, typed config, model
                           registry, daemon/HTTP client, terminal UI
```

### Where does X go?

- **"I want to add a new model architecture"** → new
  `crates/hipfire-arch-<name>/` crate, implement the `Architecture`
  trait. Copy `crates/hipfire-arch-toy/` as a template.
- **"I want to fix a kernel bug or add a kernel"** → `kernels/src/*.hip`
  for the kernel + `crates/rdna-compute` (`kernels.rs`, `kernel_registry.rs`,
  dispatch) for registration and wiring. Stays in rdna-compute regardless
  of which model family uses it.
- **"I want to tune sampler / repeat_penalty / blocked tokens
  behavior"** → `crates/hipfire-runtime/src/sampler.rs`.
- **"I want to add an end-of-turn marker for an arch"** → arch crate's
  `eos_filter_overrides()` returning
  `EosFilterOverrides { stop_at: ..., holdback_prefixes: ..., strip_think: ... }`.
- **"I want to add a daemon request / API endpoint"** → the JSON-lines
  message dispatch in `crates/hipfire-daemon/src/main.rs`; the HTTP
  gateway is `crates/hipfire-cli/src/serve/`. See
  [docs/SERVE.md](docs/SERVE.md).
- **"I want to add a CLI feature"** → `crates/hipfire-cli` (TUI:
  `crates/hipfire-tui`). See [docs/CLI.md](docs/CLI.md).
- **"I want to optimize for a specific RDNA generation"** →
  `crates/rdna-compute` / `crates/hipfire-dispatch` (capability
  predicates and per-arch kernel selection). Not inside a model-family
  crate (that fragments per-arch knowledge across the workspace).
- **"I want to add a new quant format"** → `kernels/src/` for the
  kernel + `crates/rdna-compute` for dispatch routing +
  `crates/hipfire-quantize` for the quantizer CLI, plus the runtime load
  path and `docs/quant-formats/qt-register.txt`. See
  [docs/QUANTIZATION.md](docs/QUANTIZATION.md).

### Per-arch behavior via the `Architecture` trait

Every arch crate `impl Architecture for Foo`. Defaults assume Qwen3.5
family conventions (ChatML prompt frame, `<think>` strip); the sampler,
prompt frame and loop guard are owned by `hipfire-runtime`, not
overridden per arch. The only override hook on the trait today is
`eos_filter_overrides`:

| Override | When to use | Example |
|---|---|---|
| `EosFilterOverrides` | Arch-specific end-of-turn markers (Gemma's `<end_of_turn>`, etc.) and `<think>` strip policy | `EosFilterOverrides { stop_at: vec![b"<end_of_turn>".to_vec()], holdback_prefixes: vec![b"<end_".to_vec()], strip_think: Some(false) }` |

Field-level docs live on `crates/hipfire-runtime/src/arch.rs`; worked
examples are in `crates/hipfire-arch-toy/src/arch.rs` and
`crates/hipfire-arch-gemma4/src/arch.rs`.

---

## Skills (agent-driven workflows)

| Skill | When to use |
|---|---|
| [`hipfire-tester`](.agents/skills/hipfire-tester/) | Bring-up, claim-scoped harness routes and a benchmark/tester report on a new GPU. |
| [`hipfire-diag`](.agents/skills/hipfire-diag/) | "Hipfire isn't working — what's wrong?" Captures GPU/HIP/kernel state. |
| [`hipfire-autoheal`](.agents/skills/hipfire-autoheal/) | Runtime issue triage: daemon hangs, JIT failures, port conflicts, OOM. |
| [`hipfire-arch-port`](.agents/skills/hipfire-arch-port/) | Porting hipfire to a new GPU arch (gfx12, gfx1152, gfx94x, …). |
| [`hipfire-kernel-tuning`](.agents/skills/hipfire-kernel-tuning/) | Optimize an existing kernel — pick a lever (multi-row, K-tile depth, prefetch, wave64 port, WMMA/MFMA, fused projections, ISA flags) and validate the win across the supported arch matrix. |
| [`hipfire-kernel-atlas`](.agents/skills/hipfire-kernel-atlas/) | Collect phase-aware Atlas measurements and render ISA Fit views (measurement evidence only). |
| [`astrea`](.agents/skills/astrea/) | Quant calibration, KLD/PPL evaluation and format-selection experiments. |
| [`redline-retained-replay`](.agents/skills/redline-retained-replay/) | Discovery hook for retained AQL/PM4 (Redline) work; routes to [docs/REDLINE.md](docs/REDLINE.md) and [docs/VALIDATION.md](docs/VALIDATION.md). |
| [`rebase-onto-modular`](.agents/skills/rebase-onto-modular/) | Porting a branch authored against pre-0.1.20 master onto the modular layout. |

Each skill has a `SKILL.md` (some also carry `skill.json` and sibling
`.md` files) that an agent can load.

---

## Where help is needed

Open work is tracked on the GitHub issue tracker. One standing request:

- **Issue #50** — gfx1152 testing (Strix Halo class APU). If you have
  the hardware, comment there with your `hipfire diag` output and what
  you ran.

Beyond that, `hipfire list -r` + a benchmark issue on any arch without a
committed baseline in `tests/speed-baselines/` is welcome.

---

## Licensing & attribution

hipfire is licensed under the **Apache License 2.0** (see
[LICENSE-APACHE](LICENSE-APACHE)) as of v0.3.0.

Individual files whose substantive authors have not elected Apache-2.0
remain under the **MIT License** (see [LICENSE-MIT](LICENSE-MIT)) and
are identified by their per-file SPDX header. Those grants are
preserved unchanged — nothing is relicensed in absentia. See
[LICENSE](LICENSE) and [NOTICE](NOTICE) for attribution details and
per-file SPDX semantics. The decision record (the 2026-05-19 course
correction from a unilateral relicense to dual licensing, and the
v0.3.0 move to outbound Apache-2.0) is at
[docs/governance/relicense-2026-05.md](docs/governance/relicense-2026-05.md).

### For new contributors

- **New contributions default to Apache-2.0.** By submitting a
  contribution and signing off via `git commit -s` (Developer
  Certificate of Origin — <https://developercertificate.org/>), you
  certify that you have the right to license your contribution under
  Apache-2.0 and that you intend to do so.
- All commits MUST be signed off via `git commit -s`. PRs without a
  DCO sign-off line on every commit will be asked to amend.
- **Contributors may explicitly elect MIT-only for their
  contribution.** State this in the PR description (a short note like
  "license: MIT only" is enough) and the merger will tag the relevant
  files accordingly. The SPDX header will read
  `SPDX-License-Identifier: MIT`. The project ships under Apache-2.0
  overall; your specific files stay MIT-only and keep that grant.
- Add an SPDX header to every new source file you create. Templates
  live in [docs/governance/relicense-2026-05.md](docs/governance/relicense-2026-05.md).
  For sole-author files the default (Apache-2.0) template is:
  ```
  // SPDX-License-Identifier: Apache-2.0
  // Copyright (c) 2026 <Your Name>
  // hipfire — see LICENSE and NOTICE in the project root.
  ```
- For substantial modifications (>30% of lines rewritten in an
  existing file), add your own copyright line BELOW existing ones.
  Do NOT remove existing copyright lines.

### For existing (pre-2026-05-19) contributors

Your prior contributions remain licensed exactly as you originally
submitted them — under the MIT license that hipfire used at the time.
Nothing in the dual-licensing transition revokes or modifies that
grant.

If you would like your prior contributions to be available under
Apache-2.0 as well (so that hipfire's NOTICE-backed attribution
machinery applies to them and downstream forks pulling under
Apache-2.0 get your code too), you can opt in by commenting on the
[relicense tracking issue (#305)](https://github.com/warpfront/hipfire/issues/305).
After opt-in, the maintainer will re-run
`scripts/governance/apply_spdx_headers.py --rewrite-spdx` to refresh
the SPDX tags on the files where you are a substantive author.

Opt-in is **entirely voluntary**. Files where you are the sole
substantive author currently carry `SPDX-License-Identifier: MIT`;
that stays unless you elect otherwise. Files of mixed authorship
where you are one of multiple substantive authors carry
`SPDX-License-Identifier: MIT OR Apache-2.0` until everyone
involved on that file has either opted in or declined.

### For downstream users / forks

hipfire as a whole is offered under Apache-2.0, so redistribution must
satisfy Apache License 2.0 § 4 for the work — including shipping a
readable copy of NOTICE. Individual MIT-tagged files carry an
additional, narrower obligation:

- For files whose SPDX header reads **MIT**, the LICENSE-MIT text
  applies: preserve the copyright notice and permission text.
- If you redistribute under **Apache-2.0**, the LICENSE-APACHE text
  applies, including § 4 obligations:
  - (a) Include a copy of the Apache-2.0 license to recipients.
  - (b) Mark modified files prominently as modified.
  - (c) Preserve per-file copyright, patent, trademark, and
        attribution notices in the Source form. Do not strip SPDX
        headers, copyright lines, or the CREDITS.md inventory.
  - (d) Include a readable copy of NOTICE in derivative-work
        distributions.

For files tagged `SPDX-License-Identifier: MIT OR Apache-2.0` you
may pick either license for your use of that file. Files tagged
`SPDX-License-Identifier: MIT` are MIT-only — you do not get an
Apache-2.0 patent grant from them — and files tagged
`SPDX-License-Identifier: Apache-2.0` are Apache-only.

Stripping attribution when redistributing hipfire code is a license
violation, treated as copyright infringement per `Jacobsen v.
Katzer`, 535 F.3d 1373 (Fed. Cir. 2008). The point of the
dual-license transition is **accreditation protection, not IP
control**: forks remain welcome under either license at the
recipient's option, but attribution MUST travel with the code.

See [AGENTS.md](AGENTS.md) for the project-level notice addressed
to AI agents helping users derive from this codebase, and
[PRIOR-ART.md](PRIOR-ART.md) for the inventory of original
architectural innovations originating in hipfire (with dates and
canonical commit hashes) that derivative works should attribute even
when no code is copied verbatim.
