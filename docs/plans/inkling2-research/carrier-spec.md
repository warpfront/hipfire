# Inkling-2-Mini carrier + bundle skeleton SPEC

Read-only spec for worker `feat/inkling2-mini-support`. All line anchors are
against base `origin/beta`; worker checkout already contains the arch crate
Phase 0 (`config.rs` / `weights.rs` / `forward.rs` / `oracle.rs`,
`ARCH_ID = 24`, `MTP_SIDECAR_ARCH_ID = 25`, `LoopConfig::mini()` = 24 phys /
48 visits / core `[6,18)` x3) plus `arch_mapping.rs` rows for 24
(worker commit `803d54324`) and the `architecture-ids.md` row 24 reservation.
Do NOT edit `crates/*` in either checkout while drafting — this doc is the
only write.

Goal: make an Inkling-2-Mini HFQ file **loadable** (bundle in `LoadedModel`,
correct caps/topology/admission, template resolves) with **no generate path
yet**. The daemon will load arch 24 and refuse `generate` until the forward
route lands.

## 0. Decisions (locked)

- `arch_id = 24` (provisional, already stamped by worker `arch_mapping.rs`).
  MTP sidecar id 25 is reserved only — nothing claims it.
- `INKLING2_MAX_CONTEXT: usize = 32768` (matches the loader's omitted
  `max_seq` default, `admission.rs` pins `omitted.max_seq == 32768`).
- HFQ-only. Safetensors dirs are refused at load (no `derive_arch_id` row,
  no dir config parser — upgrade path, not this change).
- `ArchModel::n_layers` returns `loop_cfg.n_slot()` (48): state is sized per
  visit slot, not per physical layer. PP banding never reads it (PP refused).
- `arch_key = "inkling2"`, `arch_label(24) = "inkling2"`.
- Caps = `ArchCaps::default()` verbatim (text-only, no batch/PP/EP/MTP/DFlash,
  `ReasoningContract::Unsupported`). Topology = `TopologyCaps::default()`
  (no override at all — every multi-card request refuses at admission).
- eos = `cfg.eos_token` from HFQ metadata (verified `200006` on Mini).
  No candidate-list scan; fall back to `1` only if metadata lacks it (the
  parser already errors, so this is unreachable — keep the literal).
- Chat template: **no code change**. `resolve_chat_template` falls through
  to the HFQ-embedded template for any unlisted arch id; Inkling-2 artifacts
  ship one. A built-in fallback arm is a later change, only if an artifact
  without an embedded template appears.

## 1. New-file map (`crates/hipfire-arch-inkling2/src/`)

Worker crate today: `lib.rs`, `config.rs`, `weights.rs`, `forward.rs`,
`oracle.rs`, `Cargo.toml`. Add exactly three files; extend `lib.rs` and
`Cargo.toml`.

### 1a. `arch.rs` — `Architecture` impl (mirror `hipfire-arch-minimax/src/arch.rs:1-65`)

```rust
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `Architecture` trait impl for Inkling-2-Mini (arch_id = 24).
//!
//! Thin marker + delegation, mirroring `hipfire-arch-minimax`'s `arch.rs`.
//! The decode forward is NOT on the trait — it lives as free functions in
//! `crate::forward` (CPU oracle today, GPU port later).

use crate::arch_model::{Inkling2Bundle, Inkling2State};
use crate::config::Inkling2Config;
use crate::weights::Inkling2Weights;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

/// Zero-sized type marker for Inkling-2-Mini. Trait dispatch uses the type.
pub struct Inkling2;

impl Architecture for Inkling2 {
    type Weights = Inkling2Weights;
    type State = Inkling2State;
    type Config = Inkling2Config;

    /// Provisional id, reserved in docs/architecture-ids.md row 24.
    fn arch_id() -> u32 {
        crate::config::ARCH_ID
    }

    fn name() -> &'static str {
        "inkling2"
    }

    fn config_from_hfq(hfq: &HfqFile) -> Result<Self::Config, String> {
        Inkling2Config::from_metadata_json(&hfq.metadata_json)
    }

    fn load_weights(
        hfq: &mut HfqFile,
        cfg: &Self::Config,
        _gpu: &mut Gpu,
    ) -> Result<Self::Weights, String> {
        // Phase 1: host-side index + validation only (no GPU upload yet).
        // `Inkling2Weights::load` takes `&HfqFile`; no `&mut` needed.
        Inkling2Weights::load(hfq, cfg)
    }

    fn new_state(_gpu: &mut Gpu, _cfg: &Self::Config) -> Result<Self::State, String> {
        Ok(Inkling2State { seq_pos: 0 })
    }
}
```

Note: `Inkling2Weights::load(&HfqFile, &Inkling2Config)` already exists
(worker `weights.rs:44`) — no signature change. The unused `Gpu` arg stays
for trait conformance (`_gpu`).

### 1b. `arch_model.rs` — bundle + `ArchModel` (mirror `hipfire-arch-minimax/src/arch_model.rs:1-47`)

```rust
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::llama::KvCache;
use rdna_compute::Gpu;

use crate::config::Inkling2Config;
use crate::weights::Inkling2Weights;

/// Skeleton decode state: cursor only. Per-visit KV/conv state lands here
/// with the GPU port; `reset_session_state` already resets the full shape.
#[derive(Default)]
pub struct Inkling2State {
    pub seq_pos: usize,
}

/// Owned skeleton bundle: config + host-side weight index + stub state + eos.
/// GPU scratch/KV attach in the port phase; teardown stays correct because
/// there is nothing device-resident to free yet.
pub struct Inkling2Bundle {
    pub config: Inkling2Config,
    pub weights: Inkling2Weights,
    pub state: Inkling2State,
    pub eos_tok: u32,
}

impl ArchModel for Inkling2Bundle {
    fn dim(&self) -> usize {
        self.config.n_embd
    }

    fn n_layers(&self) -> usize {
        // Visit slots (48 on Mini), not physical layers (24): state is
        // sized per slot. PP banding never reads this (PP refused).
        self.config.loop_cfg.n_slot()
    }

    fn vocab_size(&self) -> usize {
        self.config.unpadded_vocab_size
    }

    fn arch_key(&self) -> &'static str {
        "inkling2"
    }

    fn kv_cache_mut(&mut self) -> Option<&mut KvCache> {
        None
    }

    fn reset_session_state(&mut self, _gpu: &mut Gpu) -> Result<(), String> {
        self.state.seq_pos = 0;
        Ok(())
    }

    fn free_gpu(self: Box<Self>, _gpu: &mut Gpu) {
        // Host-side weight index only — drop is the whole teardown.
        // When GPU buffers attach, free EVERY allocation here (cf. PR #566).
        drop(self);
    }
}
```

### 1c. `carrier.rs` — `load_inkling2_bundle` (mirror `hipfire-arch-minimax/src/carrier.rs:43-89`, HFQ arm only)

```rust
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `load_inkling2_bundle` entry point for the loader's `Carrier` impl.
//!
//! HFQ-only skeleton: parses config from `hfq.metadata_json`, indexes +
//! validates weights via `Inkling2Weights::load`, resolves the tokenizer
//! from HFQ metadata, takes `eos_token` from the parsed config. Dir sources
//! are refused (no dir config parser yet). No GPU upload, no state alloc.

use crate::arch::Inkling2;
use crate::arch_model::Inkling2Bundle;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::loader_api::{LoadCtx, ModelSource};

/// Admitted context window. Matches the loader's omitted-`max_seq` default.
pub const INKLING2_MAX_CONTEXT: usize = 32768;

/// Build the Inkling-2-Mini skeleton bundle from an HFQ source.
pub fn load_inkling2_bundle(
    src: ModelSource,
    ctx: &mut LoadCtx,
) -> Result<Inkling2Bundle, String> {
    if ctx.pp > 1 {
        return Err(match &src {
            ModelSource::Hfq(_) => "inkling2: pipeline-parallel (pp>1) unsupported",
            ModelSource::Dir(_) => "inkling2: safetensors + pp>1 unsupported",
        }
        .into());
    }
    match src {
        ModelSource::Hfq(hfq_file) => {
            if ctx.max_seq == 0 || ctx.max_seq > INKLING2_MAX_CONTEXT {
                return Err(format!(
                    "inkling2: max_seq {} is outside the admitted context range 1..={INKLING2_MAX_CONTEXT}",
                    ctx.max_seq
                ));
            }
            let config = <Inkling2 as Architecture>::config_from_hfq(&hfq_file)?;
            // `load_weights` needs `&mut HfqFile`; rebind here (trait shape).
            let mut hfq_file = hfq_file;
            let weights =
                <Inkling2 as Architecture>::load_weights(&mut hfq_file, &config, ctx.gpu)?;
            let tokenizer =
                hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq_file.metadata_json)
                    .map_err(|e| format!("tokenizer not found: {e}"))?;
            let _ = tokenizer;
            Ok(Inkling2Bundle {
                eos_tok: config.eos_token,
                config,
                weights,
                state: crate::arch_model::Inkling2State { seq_pos: 0 },
            })
        }
        ModelSource::Dir(_) => Err(
            "inkling2: safetensors directory sources are not supported; load a single HFQ file"
                .into(),
        ),
    }
}
```

(`tokenizer` is unused by the bundle — the loader's `resolve_source_meta`
owns the served tokenizer. The `from_hfq_metadata` call stays as the
fail-closed "artifact must carry a tokenizer" gate, mirroring minimax
`carrier.rs:59-61`. Drop the `let _ =` if clippy complains; keep the call.)

### 1d. `lib.rs` additions

```rust
pub mod arch;
pub mod arch_model;
pub mod carrier;

pub use arch::Inkling2;
pub use arch_model::{Inkling2Bundle, Inkling2State};
pub use carrier::{load_inkling2_bundle, INKLING2_MAX_CONTEXT};
```

### 1e. `crates/hipfire-arch-inkling2/Cargo.toml` addition

```toml
rdna-compute = { path = "../rdna-compute" }
```

(`hipfire-runtime` dep already present. `serde_json` already present for
`config.rs`. No other new deps — ponytail: no `hipfire-config`,
no `safetensors` use in the new files.)

## 2. Loader changes (`crates/hipfire-loader/`)

### 2a. `Cargo.toml` — add non-optional dep (follows `hipfire-arch-minimax`, line 43)

```toml
hipfire-arch-inkling2 = { path = "../hipfire-arch-inkling2" }
```

### 2b. `src/lib.rs` — import + re-export + accessors

Next to `use hipfire_arch_lfm2moe as lfm2moe;` (`lib.rs:35`):

```rust
use hipfire_arch_inkling2 as inkling2;
```

Next to `pub use minimax::MiniMaxBundle;` (`lib.rs:969`):

```rust
pub use inkling2::Inkling2Bundle;
```

Next to the `minimax()`/`minimax_mut()` accessors (`lib.rs:1512-1523`):

```rust
/// Inkling-2-Mini bundle if this model is arch_id=24, else None.
pub fn inkling2(&self) -> Option<&Inkling2Bundle> {
    self.state
        .as_deref()
        .and_then(|s| (s as &dyn Any).downcast_ref::<Inkling2Bundle>())
}

pub fn inkling2_mut(&mut self) -> Option<&mut Inkling2Bundle> {
    self.state
        .as_deref_mut()
        .and_then(|s| (s as &mut dyn Any).downcast_mut::<Inkling2Bundle>())
}
```

### 2c. `src/carriers.rs` — `Inkling2Carrier` (place after `MinimaxCarrier`, ends line 12785)

```rust
// ─── Inkling2Carrier ──────────────────────────────────────────────────

pub struct Inkling2Carrier;
impl Carrier for Inkling2Carrier {
    fn name(&self) -> &'static str {
        "inkling2"
    }
    fn claims_arch_id(&self, arch_id: u32, _is_dir: bool) -> bool {
        // 24 in both namespaces today (worker arch_mapping.rs rows); the
        // Dir arm is refused at load until a dir config parser lands.
        arch_id == hipfire_arch_inkling2::ARCH_ID
    }
    fn caps(&self) -> saddle_core::caps::ArchCaps {
        // Text-only skeleton: the default IS the declaration (no batch,
        // no MTP, no DFlash, no images, reasoning Unsupported). Spelled
        // via default() so a new capability is opt-in, never inherited.
        saddle_core::caps::ArchCaps::default()
    }
    fn sampling_defaults(&self) -> saddle_core::sampling::SamplingDefaults {
        saddle_core::sampling::SamplingDefaults::new(1.0, 1.0, 1.0)
    }
    // No topology_caps override: the default declares no axis, so every
    // pp/ep/tp request refuses at admission with the uniform sentence.
    // No spec_target_guard / make_spec_emitter override: the defaults
    // refuse spec-decode, so the skeleton is AR-only by construction.
    // No bench_prefill/prime/run override: daemon falls through (None).
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
        if ctx.pp > 1 {
            return Err(crate::pp_unsupported_message(
                "inkling2",
                matches!(src, ModelSource::Dir(_)),
                ctx.pp,
            ));
        }
        dir_diag(&src);
        let meta = resolve_source_meta(&src, ctx.path)?;
        let bundle = hipfire_arch_inkling2::load_inkling2_bundle(src, ctx)?;
        let speculator =
            crate::spec_build::build_speculator(meta.arch_id, None, None, ctx.max_seq, ctx.spec);
        Ok(LoadedModel {
            state: Some(Box::new(bundle)),
            speculator,
            ..LoadedModel::skeleton(
                meta.arch_id,
                meta.tokenizer,
                ctx.max_seq,
                ctx.max_seq,
                ctx.path.to_string(),
                meta.chat_template,
                meta.chat_template_source,
            )
        })
    }
}
```

(`dir_diag`, `resolve_source_meta`, `pp_unsupported_message` are
`carriers.rs`-private helpers — the new block must live in `carriers.rs`,
not in the arch crate, exactly as shown.)

REGISTRY (`lib.rs:676-693`): append `&Inkling2Carrier,` to the list.

### 2d. Routing rows

- `arch_mapping.rs` (`MODEL_TYPE_TO_ARCH_ID`): **already present in worker**
  (`803d54324`): `inkling2loopforconditionalgeneration`, `inkling2_loop`,
  `inkling2_loop_text`, `inkling2`, `inkling2_text` → 24. If porting onto a
  base without them, copy those five rows verbatim — do not invent keys.
- `arch_label` (`lib.rs:478-498`): add `24 => "inkling2",` (insertion order:
  after the `19 => "raon1_mini",` arm).
- `bench_decode_route` (`lib.rs:463-474`): **exhaustive match — 24 MUST get an
  arm or the build fails**. Add `24 => BenchDecodeRoute::Unsupported,`.
- `continuous_batch_route`, `wire_verb`, `vision_route`: no change
  (`_ => None` / `Generate` / `supports_images == false → None` already do
  the right thing).
- `arch_default_template` (`carriers.rs:11226`): no change (embedded template
  path; see §3).
- `reset_core.rs`: two REQUIRED additions (the `inventory_covers_every_arch_
  in_arch_mapping` test fails without both):
  ```rust
  const INKLING2: ResetCoreCoverage = ResetCoreCoverage {
      arch: "inkling2",
      recurrent_or_conv: true,
      s_ef_residual: true,
      kv_or_aux_caches: true,
      graphs: false,
      drafter: false,
      adaptive: false,
      host_position_and_conversation: true,
      eligibility: RetryResetEligibility::Ineligible {
          reason: "inkling2 skeleton has no generate route yet",
      },
  };
  ```
  append `INKLING2,` to the returned slice, and add `24 => Some("inkling2"),`
  next to `19 => Some("raon1_mini"),` in the test's `arch_key_for_id`.

## 3. Chat-template resolve snippet

No new code. Precedent — `resolve_chat_template` (`lib.rs:1917-1963`): arch
5/6, 11, 12 take explicit arms; **every other id falls to the `_ => {}` arm
which returns `hfq.chat_template()`** (embedded `tokenizer_config`
template). Arch 24 rides that fallthrough today. The served provenance then
reads `SRC_EMBEDDED_HFQ`. Only add an explicit arm if an artifact without an
embedded template appears:

```rust
// ONLY if needed later — NOT part of this change:
24 => {
    if let Some(t) = hfq.chat_template() {
        return (Some(t), Some(SRC_EMBEDDED_HFQ.to_string()));
    }
    return (None, None); // no built-in fallback for inkling2 yet
}
```

Dir sources never reach this (refused in `load_inkling2_bundle`), so no
`arch_default_template` arm.

## 4. Compile checklist

From the repo root (Windows shell):

```
cargo check -p hipfire-arch-inkling2
cargo check -p hipfire-loader
cargo check -p hipfire-runtime
cargo test -p hipfire-runtime arch_mapping reset_core
cargo test -p hipfire-loader carriers_are_disjoint known_ids_route_as_expected
cargo test -p hipfire-arch-inkling2
```

Then `cargo clippy -p hipfire-arch-inkling2 -p hipfire-loader` and
`scripts/pre-commit-ci.sh origin/beta` on the final commit (per AGENTS.md §7;
install `cargo-deny` first). No GPU needed for any of the above.

Expected load smoke (needs the `inkling2-mini-mq4.hfq` artifact, CPU-side
only — asserts load/admission, not numerics):

```
hipfire bench <path-to-inkling2-mini-mq4.hfq> --runs 1 --warmups 0 --max-tokens 1
```

expect a load-time success followed by a fail-closed "no generate route for
arch 24" (exact refusal string TBD by the daemon arm — the spec deliberately
leaves the daemon untouched; an `"unknown arch"`-class error is acceptable
here, a VRAM leak or mis-route to another carrier is not).

## 5. Explicitly NOT included

- No GPU kernels, no `forward` GPU port, no `WeightTensor` upload
  (`Inkling2Weights` stays a host-side index; `Inkling2State` a cursor).
- No daemon generate route (`generate_*` arch-24 branch), no `bench_*`
  overrides, no `decide_*` hooks.
- No MTP floor: no `mtp_measured_ar_floor` override (default `false` keeps
  `speculation.mtp = auto` on AR; explicit `--spec mtp` finds no head and
  must refuse, not attach).
- No DFlash / n-gram / DDTree / DSpark / EAGLE (no `spec_target_guard` /
  `make_spec_emitter` overrides; `build_speculator` args stay
  `(id, None, None, …)` like minimax).
- No batch / PP / EP / TP (no `topology_caps` override; no slot engine).
- No vision (`supports_images` stays `false`), no reasoning contract
  (`Unsupported` — and therefore no `arch_replays_history_reasoning` entry).
- No safetensors-dir support, no quantizer changes, no `derive_arch_id`
  changes, no registry model/tag entries, no benchmark claims.
- No `semantic_contract_version`, no `has_deltanet` (both stay unset/false).

Skipped / not verified: daemon-side refusal text for arch 24 (worker owns the
generate dispatch); exact `bench` behavior pre-generate-route (smoke command
above is a prediction, not a measured result); whether `clippy` wants the
`let _ = tokenizer;` removed (leave to the worker's clippy run).
Risk: `LoadedModel::skeleton` + `REGISTRY` + `bench_decode_route` +
`reset_core` inventory are four separate arch-24 touch points with no single
compiler error covering all of them — the checklist's test lines are the
safety net, run all five.
