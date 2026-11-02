# Documentation index

Single hub for navigation, lifecycle classification, and ownership.
Domain prose lives in the linked owners; this file does not duplicate it.

| Field | Value |
|---|---|
| Inventory date | 2026-09-07 (rows not named in the entry re-audit below) |
| Working branch | `beta` |
| Audited source ref | `20ba9ee3e33bddfa443ef1e5ba0c989de629fedb` (rows not named in the entry re-audit below) |
| Comparison base | `origin/beta` @ `20ba9ee3e33bddfa443ef1e5ba0c989de629fedb` |
| Entry re-audit (2026-10-13) | `GETTING_STARTED.md` (install, quickstart and tag claims) and this file's file-presence coverage of `docs/`, checked against `origin/beta` @ `c7a732ad5b921706519cdba30fdb43a7e62aa2d4` (0.4.2 line). Row states for the 0.4.2 Flash-Next route defaults and for the pages and collections added in that pass are new; every other row state was not re-audited. |
| Integrated commit / tree / source hashes | Supplied externally by Git/CI after cutover. Never self-referenced here. |

## Truth states

Exactly four **active** truth states apply to a current claim or surface.
Pick exactly one:

| State | Meaning |
|---|---|
| **shipped / ref-pinned** | Present on the audited source ref and treated as current product or contributor authority for that concern. |
| **branch-implemented** | Implemented or documented on a named working branch but not present in the audited beta ref. |
| **measured** | Observation tied to named fixture, binary/model identity, and date. Not a default, floor, or admission. |
| **planned** | Intent only. No executable authority and no implied schedule. |

**Historical** is a **lifecycle / evidence disposition**, not a fifth active
truth state. It means a retained record useful for provenance — not current
procedure, baseline, or product authority — unless a shipped owner re-states
the claim under one of the four active states above.

Directory and collection rows use **exactly one** canonical classification:
one of the four active states, a historical disposition, or an explicit
fail-closed marker below. When members vary, put the directory policy
classification in the State cell and explain member exceptions in
Notes/Policy. Never invent another truth state or a slash-combined policy
state. Each individual claim still takes exactly one active label, a
historical disposition, or fails closed.

Lack of authority is not an active truth state:

| Marker | Meaning |
|---|---|
| **blocked** | No single canonical owner, missing executable route, or unresolved contradiction. Fail closed. |
| **superseded / rejected** | Replaced or declined; keep only as history. Do not promote. |
| **unknown** | Not classified. Fail closed — do not invent an owner, admission, or gate. |
| **historical** (disposition) | Retained evidence or archive. Not an active authority label. |

**Fail-closed:** if a concern has no row below, or a row is `blocked` /
`unknown`, do not treat any nearby doc, skill, measurement, or runtime default
as authority. Open an explicit owner or keep the claim out of product language.

Admissions are machine-recorded only in [`admissions.yml`](admissions.yml) (schema v2). Exactly one evidence-bound row is admitted; every other route stays unadmitted. Runtime gating or a passing measurement alone never admits a route.

## How to use this index

1. Find the concern in [Ownership](#ownership).
2. Open that owner for mutable facts.
3. For validation or promotion evidence, use only [`VALIDATION.md`](VALIDATION.md).
4. Treat collection rows as directory policy; do not promote a file inside a historical collection by recency alone.

## Ownership

Exactly one canonical owner (or explicit `BLOCKED`) per concern.

| Concern | Canonical owner | State | Notes |
|---|---|---|---|
| Docs navigation, lifecycle labels, ownership map | [`docs/INDEX.md`](INDEX.md) | shipped / ref-pinned | This file. |
| Human validation route selection | [`docs/VALIDATION.md`](VALIDATION.md) | shipped / ref-pinned | Sole route selector. |
| Machine admission registry | [`docs/admissions.yml`](admissions.yml) | shipped / ref-pinned | Schema v2; exactly one earned record (LFM2.5-350M MQ4 gfx1201 retained-PM4). |
| Product onboarding | [`docs/GETTING_STARTED.md`](GETTING_STARTED.md) | shipped / ref-pinned | Install, first run, uninstall; points at the 0.4.2 beta Flash-Next sections without owning their facts. |
| CLI surface and model lifecycle commands | [`docs/CLI.md`](CLI.md) | shipped / ref-pinned | |
| Daemon / user config keys | [`docs/CONFIG.md`](CONFIG.md) | shipped / ref-pinned | |
| Copyable TOML config profiles | [`docs/configs/`](configs/README.md) | shipped / ref-pinned | User, developer, and retained-PM4 examples. |
| Environment variables | [`docs/env-vars.md`](env-vars.md) | shipped / ref-pinned | Env coverage enforced by [`scripts/check-env-docs.py`](../scripts/check-env-docs.py). |
| Registry models, VRAM, sampling, sidecars | [`docs/MODELS.md`](MODELS.md) | shipped / ref-pinned | Registry presence ≠ runtime admission. |
| Serve HTTP API | [`docs/SERVE.md`](SERVE.md) | shipped / ref-pinned | |
| Chat UX and daemon attach behavior | [`docs/CHAT.md`](CHAT.md) | shipped / ref-pinned | |
| Post-land GPU smoke (cells, invariants, lanes, verdict and pin policy) | [`docs/post-land-smoke.md`](post-land-smoke.md) | shipped / ref-pinned | Orchestrator `scripts/post-land-smoke.sh`, runner `scripts/post_land_smoke.py`, plan `scripts/post-land-smoke/plan.json`. |
| Open field faults (known issues) and what to capture | [`docs/KNOWN-ISSUES.md`](KNOWN-ISSUES.md) | measured | Dated observations with host versions; root causes unresolved unless stated. Not a default, workaround or admission. |
| Crate layout and request lifecycle (overview) | [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) | shipped / ref-pinned | Runtime source wins on conflict. |
| Architecture id table | [`docs/architecture-ids.md`](architecture-ids.md) | shipped / ref-pinned | |
| Quantization formats and math | [`docs/QUANTIZATION.md`](QUANTIZATION.md) | shipped / ref-pinned | |
| `hipfire quantize` operator guide | [`docs/QUANTIZE.md`](QUANTIZE.md) | shipped / ref-pinned | |
| Image generation (FLUX) | [`docs/IMAGEGEN.md`](IMAGEGEN.md) | shipped / ref-pinned | First release; RDNA3/3.5 measured. CLI `hipfire img`, HTTP `/v1/images/generations` + `/edits`. |
| Multi-GPU operator guide | [`docs/multi-gpu.md`](multi-gpu.md) | shipped / ref-pinned | |
| SKU application-capacity emulation (`HIPFIRE_EMU`, `hipfire emu`, `bench --emu`) | [`docs/EMU.md`](EMU.md) | branch-implemented | Developer opt-in; capacity cap plus a CPU-only derating estimate. No CU/clock/power emulation. Does not create admission. |
| Container install / run | [`docs/CONTAINER.md`](CONTAINER.md) | shipped / ref-pinned | |
| NixOS notes | [`docs/NIXOS.md`](NIXOS.md) | shipped / ref-pinned | |
| Perf claim protocol (warmup, fresh-process, noise) | [`docs/methodology/perf-benchmarking.md`](methodology/perf-benchmarking.md) | shipped / ref-pinned | Numbers live in measured owners, not here. |
| Bench-suite layout | [`docs/methodology/bench-suite.md`](methodology/bench-suite.md) | shipped / ref-pinned | |
| CPU-exec offload benchmark handoff (`memory.offload_exec`) | [`docs/methodology/cpu-exec-offload-benchmark-handoff.md`](methodology/cpu-exec-offload-benchmark-handoff.md) | shipped / ref-pinned | Method, fixture identity and gotchas for the pcie-vs-cpu A/B; results live in `docs/perf-checkpoints/`. |
| Arch-port validation procedure (channel / speed) | [`docs/methodology/arch-port-validation.md`](methodology/arch-port-validation.md) | shipped / ref-pinned | Does not restore retired coherence-gate batteries. |
| Perf-arch working discipline | [`docs/methodology/perf-arch-discipline.md`](methodology/perf-arch-discipline.md) | shipped / ref-pinned | |
| Kernel Atlas methodology | [`docs/methodology/kernel-atlas.md`](methodology/kernel-atlas.md) | shipped / ref-pinned | |
| Redline contributor certification and route-proof policy | [`docs/REDLINE.md`](REDLINE.md) | shipped / ref-pinned | Normative contributor policy. |
| Golden Redline fixture reproduction and client setup | [`docs/GOLDEN-REDLINE.md`](GOLDEN-REDLINE.md) | branch-implemented | Sealed measured fixtures; does not create admission. |
| Thin Redline skill hook (workflow only) | [`.agents/skills/redline-retained-replay/`](../.agents/skills/redline-retained-replay/) | shipped / ref-pinned | Must not fork policy from `REDLINE.md`. |
| Executable agent skills root | [`.agents/skills/`](../.agents/skills/) | shipped / ref-pinned | Sole executable skill root. |
| Speculation feature inventory | [`docs/speculation-support-inventory.md`](speculation-support-inventory.md) | shipped / ref-pinned | Current speculation capability inventory. Capability is not admission or a performance floor; verify in source before product claims. |
| Model-load refusal wording inventory | [`docs/load-refusal-wording.md`](load-refusal-wording.md) | shipped / ref-pinned | Audit of the user-reachable model-load refusal strings (before/after, setting named, excluded areas). |
| Spec-decode durability note (2026-06-23) | [`docs/spec-decode-durability-2026-06-23.md`](spec-decode-durability-2026-06-23.md) | historical | Dated 2026-06-23 fixture/tables report; not current validation. |
| Published benchmark tables | [`docs/BENCHMARKS.md`](BENCHMARKS.md) | historical | Unmanifested / incomplete evidence identity on tables → historical only; not live floors or admissions. |
| Dated perf campaign checkpoints | [`docs/perf-checkpoints/`](perf-checkpoints/) | measured | Immutable bodies; amend only via new dated files. |
| Dependency adoption log | [`docs/dependency-adoption-log.md`](dependency-adoption-log.md) | historical | |
| Upstream merge journal | [`docs/upstream-merge-journal.md`](upstream-merge-journal.md) | historical | |
| DeepSeek4 PR body archive | [`docs/deepseek4-pr-body.md`](deepseek4-pr-body.md) | historical | |
| Multi-GPU bring-up lessons | [`docs/multi-gpu-bringup-lessons.md`](multi-gpu-bringup-lessons.md) | historical | |
| HFP4 format note | [`docs/quant-formats/hfp4.md`](quant-formats/hfp4.md) | shipped / ref-pinned | Current HFP4/MFP4 owners are shipped / ref-pinned; some reserved aliases remain planned. Broader quant authority stays `QUANTIZATION.md`. |
| CPU SIMD coverage map | [`docs/quant-formats/cpu-simd-coverage.md`](quant-formats/cpu-simd-coverage.md) | shipped / ref-pinned | Which AVX2 row dot each `CpuQuant` gets and under which gate. Authority for the CPU-exec offload path's vector coverage; the qt allocation itself stays `qt-register.txt`. |
| MoE AWQ working notes | [`docs/moe-awq/`](moe-awq/) | historical | |
| MI300X rental runbook | [`docs/rental/MI300X-RENTAL-RUNBOOK.md`](rental/MI300X-RENTAL-RUNBOOK.md) | historical | |
| Relicense / governance records | [`docs/governance/`](governance/) | historical | Legal/historical; do not rewrite. |
| Design-time architecture drafts | [`docs/design/`](design/) | planned | Directory policy is planned intent. Member exceptions: some branch-implemented LFM/Redline designs and measured baseline notes may coexist; none are product defaults. Historical member bodies stay disposition-only. |
| Implementation plans and PRDs | [`docs/plans/`](plans/) | planned | Directory policy is planned intent. Member exceptions: measured results ledgers may appear; recency ≠ authority. |
| Narrow specs | [`docs/specs/`](specs/) | planned | Intent/spec records; promote only via shipped owners. |
| Shared-prefix serving integration | [`docs/specs/2026-09-05-serving-cache-scheduler-spec.md`](specs/2026-09-05-serving-cache-scheduler-spec.md) | planned | Paged ownership, hybrid-state reuse, fair scheduling and structured generation; extends the multi-slot spec, not a runtime admission. |
| Investigations | [`docs/investigations/`](investigations/) | historical | Discovery trails including measured research; not product defaults. |
| Reviews | [`docs/reviews/`](reviews/) | historical | Review archives. |
| Lessons learned | [`docs/lessons_learned/`](lessons_learned/) | historical | Postmortems. |
| Superpowers nested plans/specs | [`docs/superpowers/`](superpowers/) | planned | Nested plans/specs only. Not an executable skill root. |
| Flash-Next 0.4.2 route defaults (YaRN `attention.yarn_factor`, Strix Halo NPU trunk `fusions.npu_trunk`, EP owner-routed prefill and expert staging) | [`docs/CONFIG.md`](CONFIG.md) | shipped / ref-pinned | Present on the audited beta ref, not in the tagged v0.4.1. EP setup and gates: [`multi-gpu.md`](multi-gpu.md); variable semantics: [`env-vars.md`](env-vars.md). Defaults are not admissions. |
| Named subsystems and attribution pointers | [`docs/GLOSSARY.md`](GLOSSARY.md) | shipped / ref-pinned | Origination records live in `PRIOR-ART.md`, `NOTICE`, `CREDITS.md`, not here. |
| Per-crate maps (`crates/<name>/map.md`) | [`docs/crate-maps.md`](crate-maps.md) | shipped / ref-pinned | Convention and generator markers; checked by `scripts/check-crate-maps.py`. |
| VL quantization design specs | [`docs/qwen35-vl-mq4v2-spec.md`](qwen35-vl-mq4v2-spec.md), [`docs/lfm2-vl-mq4v2-spec.md`](lfm2-vl-mq4v2-spec.md) | historical | Design specs retained for provenance; not current procedure or product authority. |
| Historical GEMMA4 E-series validation records | [`GEMMA4_ESERIES_GFX1100_PREFILL.md`](GEMMA4_ESERIES_GFX1100_PREFILL.md), [`GEMMA4_ESERIES_GFX1201_VALIDATION.md`](GEMMA4_ESERIES_GFX1201_VALIDATION.md), [`GEMMA4_ESERIES_LOGIT_STABILITY.md`](GEMMA4_ESERIES_LOGIT_STABILITY.md) | historical | Dated provenance records; not current policy, default, or admission. Current env and config semantics stay in [`env-vars.md`](env-vars.md) and [`CONFIG.md`](CONFIG.md). |
| Unclassified top-level pages: [`SCS-SUITE.md`](SCS-SUITE.md), [`jit-g0-inventory.md`](jit-g0-inventory.md) | — | **unknown** | Listed for discovery only; not audited. Treat none as authority, floor, default or admission. |
| Unclassified collections: [`npu/`](npu/README.md), [`pm-union/`](pm-union/OWNERS.md) | — | **unknown** | Listed for discovery only; not audited. The product NPU trunk route is a `CONFIG.md` concern (`fusions.npu_trunk`), not defined by `npu/`. |
| Universal replacement gate for all GPU changes | — | **BLOCKED** | No universal gate. Route per [`VALIDATION.md`](VALIDATION.md). |
| Inferred model/route admissions without registry rows | — | **BLOCKED** | [`admissions.yml`](admissions.yml) schema v2 forbids inferred/wildcard rows; only the exact earned record is admitted. |
| Stitching manual Redline capture to product timed-arm proof | — | **BLOCKED** | See `REDLINE.md` certification ladder; fail closed. |

## Top-level page classification

Every current top-level page, exactly once.

| Page | State | Owner role |
|---|---|---|
| [`INDEX.md`](INDEX.md) | shipped / ref-pinned | Navigation, lifecycle, ownership. |
| [`VALIDATION.md`](VALIDATION.md) | shipped / ref-pinned | Validation route selector. |
| [`admissions.yml`](admissions.yml) | shipped / ref-pinned | Admission registry (schema v2; exactly one record). |
| [`GETTING_STARTED.md`](GETTING_STARTED.md) | shipped / ref-pinned | Onboarding. |
| [`PICK-A-MODEL.md`](PICK-A-MODEL.md) | shipped / ref-pinned | VRAM tier → tag → speculation setting; tok/s copied from historical BENCHMARKS.md fixtures. |
| [`CLI.md`](CLI.md) | shipped / ref-pinned | CLI. |
| [`CONFIG.md`](CONFIG.md) | shipped / ref-pinned | Config keys. |
| [`env-vars.md`](env-vars.md) | shipped / ref-pinned | Environment variables. |
| [`MODELS.md`](MODELS.md) | shipped / ref-pinned | Models. |
| [`SERVE.md`](SERVE.md) | shipped / ref-pinned | Serve API. |
| [`CHAT.md`](CHAT.md) | shipped / ref-pinned | Chat. |
| [`KNOWN-ISSUES.md`](KNOWN-ISSUES.md) | measured | Open field faults and what to capture. |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | shipped / ref-pinned | Architecture overview. |
| [`architecture-ids.md`](architecture-ids.md) | shipped / ref-pinned | Arch id table. |
| [`QUANTIZATION.md`](QUANTIZATION.md) | shipped / ref-pinned | Quant design. |
| [`QUANTIZE.md`](QUANTIZE.md) | shipped / ref-pinned | Quantize tool. |
| [`IMAGEGEN.md`](IMAGEGEN.md) | shipped / ref-pinned | Image generation (FLUX); first release RDNA3/3.5. |
| [`multi-gpu.md`](multi-gpu.md) | shipped / ref-pinned | Multi-GPU ops. |
| [`CONTAINER.md`](CONTAINER.md) | shipped / ref-pinned | Containers. |
| [`NIXOS.md`](NIXOS.md) | shipped / ref-pinned | NixOS. |
| [`REDLINE.md`](REDLINE.md) | shipped / ref-pinned | Redline certification policy. |
| [`GOLDEN-REDLINE.md`](GOLDEN-REDLINE.md) | branch-implemented | Sealed fixture reproduction and OpenAI-client setup. |
| [`BENCHMARKS.md`](BENCHMARKS.md) | historical | Dated/historical bench tables; incomplete manifests stay historical-only. |
| [`speculation-support-inventory.md`](speculation-support-inventory.md) | shipped / ref-pinned | Current speculation capability inventory. |
| [`load-refusal-wording.md`](load-refusal-wording.md) | shipped / ref-pinned | Model-load refusal wording inventory. |
| [`spec-decode-durability-2026-06-23.md`](spec-decode-durability-2026-06-23.md) | historical | Dated 2026-06-23 durability fixture/tables report. |
| [`dependency-adoption-log.md`](dependency-adoption-log.md) | historical | Dependency log. |
| [`upstream-merge-journal.md`](upstream-merge-journal.md) | historical | Merge journal. |
| [`deepseek4-pr-body.md`](deepseek4-pr-body.md) | historical | PR body archive. |
| [`multi-gpu-bringup-lessons.md`](multi-gpu-bringup-lessons.md) | historical | Bring-up narrative. |
| [`GLOSSARY.md`](GLOSSARY.md) | shipped / ref-pinned | Named subsystems. |
| [`crate-maps.md`](crate-maps.md) | shipped / ref-pinned | Crate-map convention. |
| [`GEMMA4_ESERIES_GFX1100_PREFILL.md`](GEMMA4_ESERIES_GFX1100_PREFILL.md) | historical | Dated gfx1100 prefill validation record. |
| [`GEMMA4_ESERIES_GFX1201_VALIDATION.md`](GEMMA4_ESERIES_GFX1201_VALIDATION.md) | historical | Dated gfx1201 validation record. |
| [`GEMMA4_ESERIES_LOGIT_STABILITY.md`](GEMMA4_ESERIES_LOGIT_STABILITY.md) | historical | Dated logit-stability record. |
| [`qwen35-vl-mq4v2-spec.md`](qwen35-vl-mq4v2-spec.md) | historical | Qwen3.5-VL design spec. |
| [`lfm2-vl-mq4v2-spec.md`](lfm2-vl-mq4v2-spec.md) | historical | LFM2.5-VL design spec. |
| [`SCS-SUITE.md`](SCS-SUITE.md) | unknown | Not classified. |
| [`jit-g0-inventory.md`](jit-g0-inventory.md) | unknown | Not classified. |

## Collection classification

Every current top-level collection, exactly once. Directory policy applies to members that lack their own stronger metadata. One canonical policy state per collection; member exceptions go in Policy only.

| Collection | State | Policy |
|---|---|---|
| [`methodology/`](methodology/) | shipped / ref-pinned | Active methodology owners. Link; do not copy matrices into root guides. **Member exceptions:** [`astrea-atlas-pareto-workflow.md`](methodology/astrea-atlas-pareto-workflow.md) and [`astrea-model-policy.md`](methodology/astrea-model-policy.md) are **planned / blocked** (proposed intent; not current INDEX/VALIDATION owners; `candidate_id` join unsupported in current Atlas schema). |
| [`perf-checkpoints/`](perf-checkpoints/) | measured | Immutable dated evidence. Correct only with a new dated amendment file. |
| [`design/`](design/) | planned | Approved intent and drafts. Member exceptions: branch-implemented LFM/Redline docs and measured baseline notes may appear; directory is not runtime truth. |
| [`plans/`](plans/) | planned | Execution plans/PRDs. Member exceptions: measured results ledgers may appear. Recency ≠ authority. |
| [`specs/`](specs/) | planned | Narrow specs and intent; promote only via shipped owners. |
| [`investigations/`](investigations/) | historical | Historical discovery plus measured research trails. |
| [`reviews/`](reviews/) | historical | Review archives. |
| [`lessons_learned/`](lessons_learned/) | historical | Postmortems. |
| [`moe-awq/`](moe-awq/) | historical | MoE AWQ session notes. |
| [`quant-formats/`](quant-formats/) | shipped / ref-pinned | Format notes under quant umbrella; HFP4/MFP4 current surface is shipped / ref-pinned while some aliases stay planned. `QUANTIZATION.md` remains primary. |
| [`rental/`](rental/) | historical | Rental runbooks. |
| [`governance/`](governance/) | historical | Legal/governance records; bodies stay untouched. |
| [`superpowers/`](superpowers/) | planned | Nested plans/specs only; not an executable skill root. |
| [`configs/`](configs/README.md) | shipped / ref-pinned | Copyable TOML profiles; `configs/README.md` owns usage. |
| [`npu/`](npu/README.md) | unknown | Not classified; not an owner of the product NPU trunk route. |
| [`pm-union/`](pm-union/OWNERS.md) | unknown | Not classified. |
| [`assets/`](assets/readme/) | shipped / ref-pinned | Static README figures; embedded measurements carry no admission. |

## Audit scope

- **Audited ref** (`20ba9ee3e33bddfa443ef1e5ba0c989de629fedb`): beta behavior pin for the inventory rows not named in the entry re-audit.
- **Entry re-audit ref** (`c7a732ad5b921706519cdba30fdb43a7e62aa2d4`, 2026-10-13): covers `GETTING_STARTED.md` claims and file presence of every top-level `docs/` page and collection. It is not a re-audit of every row state in this index.
- **Comparison base** (`origin/beta` @ `20ba9ee3e33bddfa443ef1e5ba0c989de629fedb`): use when separating branch-only work from current beta facts.
- Branch-implemented surfaces must not be phrased as beta product facts.
- Historical, legal, and measured checkpoint bodies are not rewritten by this index.

## Explicit non-goals

- No mutable fact duplication across owners.
- No self-referencing “final” commit or tree hash in this file.
- No universal GPU/coherence replacement gate.
- No admissions inferred from benches, harness exits, or registry tags.
