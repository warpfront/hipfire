# Bench and quality fixtures

**Lifecycle:** shipped / reference. **Values:** [`fixtures/pins.toml`](../fixtures/pins.toml)
(checked by [`scripts/check-pins.py`](../scripts/check-pins.py), run by
`scripts/ci-gates.sh`). This page says what each fixture is and how its pins
are reproduced; every digest, size and KLD value lives in the data file and is
cited here as `pin:<kind>/<id>`. Rules about when a fixture is mandatory are in
[`AGENTS.md`](../AGENTS.md) § "Fixtures and pins".

Look a value up with `scripts/check-pins.py get <kind>/<id> <field>`; check a
file against a pin with `scripts/check-pins.py verify <kind>/<id> PATH` (for a
`.kldseq` the `kld/<id>` pin compares the sequence digest).

## Pinned Hugging Face bench fixture (dense Qwen3.8-27B)

For dense Qwen3.8-27B MQ4V2/DFlash perf work, do not identify the
canonical trunk by local filename. Local filenames drift and lookalike
MQ4/MQ4V2 files are not comparable.

The canonical dense trunk is whichever local artifact byte-matches
`pin:model/qwen3.8-27b.mq4-xts`: `qwen3.8-27b.mq4-xts` from HF repo
`hipfire-models/qwen3.8-27b` (registry tag `qwen3.8:27b-mq4-xts`).
`qwen3.8-27b.mq4-xt` has exactly the same size, so only the digest tells them
apart. The checkpoint is H2, symmetric MQ4V2 XT (qt44) with per-group AWQ alpha
and three GPTQ scale-refit rounds on the v5 Hessian. It replaces the
2026-09-23 `mq4-xts` test upload (QAT r7s200,
`pin:model/qwen3.8-27b.mq4-xts-qat-r7s200`), which is not comparable.

Paired draft: the registry sidecar `qwen38-27b-dflash-mq4.hfq`, or a local
copy of the registry draft (the same bytes), `pin:draft/qwen38-27b-dflash-mq4.hfq`.

Before reporting dense 3.8 MQ4V2/DFlash results, verify the candidate
trunk with `scripts/check_fixture.sh [PATH]` (size + SHA-256 against the pin)
and require the pinned digest. Reports that use a trunk with a different digest
are not comparable and should be discarded.

Speculation policy: since SPEC-AR-FLOOR S2a a greedy DFlash request on this
trunk runs the measured speculate-vs-AR floor (the `done` envelope's `floor`
object), so a decode number is comparable only with the same policy. Compare
DFlash rows by `client_span_tok_s` (`hipfire bench --json`, client wall time)
as well as `decode_tok_s`; `HIPFIRE_SPEC_RETIRE=1.5` reproduces the pre-floor
policy on the same binary.

Historical: the previous dense pin was the asymmetric Qwen3.8-27B MQ4V2 XT
(`pin:model/qwen3.8-27b.mq4-xt`; re-issued 2026-09-15 with AWQ sidecars, the
prior upload `pin:model/qwen3.8-27b.mq4-xt-no-awq` lacked them, WT2 KLD 0.057
vs 0.048). Before that it was Qwen3.6-27B (`pin:model/qwen3.6-27b.mq4`).

### Quality pins

`eval_hipfire --scoring-mode prefill --max-chunks 24`
with `HIPFIRE_GRAPH=0`, `HIPFIRE_NORMALIZE_PROMPT=0` and a fresh HOME
and kernel cache. References: WT2 `pin:reference/qwen3.8-27b.ref_wt2.bin`,
code24 `pin:reference/heldout-code24.gptqsym.kldref.bin`.
A change that claims to be bit-exact must reproduce its arch's `.kldseq`
byte for byte. A change that moves a pin on purpose re-pins it in
`fixtures/pins.toml`. A re-pin must record, in the same commit: the evaluator
build commit, the evaluator binary md5, and the receipt path (the `evaluator`
and `receipt` fields; `scripts/check-pins.py` refuses a current KLD pin without
them). The public references and one-command reproduction live in
`kld/VERIFY.md` on the model's Hugging Face repo (`hipfire-models/qwen3.8-27b`,
`hipfire-models/qwen3.8-flash-next`); update it with the pin.

Toolchain: **measured on ROCm 10.1 / HIP 7.16 / clang 24** (lld 24,
code object v6). Each pin is reproduced in two fresh processes with separate
HOME directories. gfx1201/gfx1100 use empty kernel caches; gfx1201 repeats
use different R9700 cards. gfx1100/gfx1151 run serialized under a shared timing
lock. **gfx1151 pins require a warm kernel cache**: populate it once, then
reuse it for two scored runs with fresh HOME directories. Cold-JIT Halo runs
can differ in chunk 0 only; warm-cache WT2/code24 repeats agree byte for byte
(the cold-cache discriminator runs are recorded in
[`docs/perf-checkpoints/2026-11-15-agents-md-measurements.md`](perf-checkpoints/2026-11-15-agents-md-measurements.md)).

Current pins (`status = "current"`), one per arch × route × corpus:

| Arch | Route, KV flags | WT2 | code24 |
|---|---|---|---|
| gfx1201 | A4 (default), `--kv-mode fp8 --kv-v q8` | `pin:kld/gfx1201.a4.wt2` | `pin:kld/gfx1201.a4.code24` |
| gfx1201 | native fp8 (`HIPFIRE_IU4_PREFILL=0`), `--kv-mode fp8 --kv-v q8` | `pin:kld/gfx1201.fp8.wt2` | `pin:kld/gfx1201.fp8.code24` |
| gfx1100 | A4 (default), `--kv-mode q8 --kv-v q8` | `pin:kld/gfx1100.a4.wt2` | `pin:kld/gfx1100.a4.code24` |
| gfx1151 | A4 (default), `--kv-mode q8 --kv-v q8`, warm kernel cache | `pin:kld/gfx1151.a4.wt2` | `pin:kld/gfx1151.a4.code24` |
| gfx942 | Flash-Next stock FP8 AR (default route, `qwen4_kld eval --decode`), BF16 teacher | `pin:kld/gfx942.fn-fp8.wt2` | `pin:kld/gfx942.fn-fp8.code24` |

The RDNA rows are re-pinned for the canonical GDN prefill (chunk- and
VRAM-independent numerics; CHANGELOG `## Unreleased`): the prefill GDN recipe
and the scored-region route moved the sequences, paired against the previous
pins on the same model, references, toolchain and cards (per-chunk paired-t 95%
CI contains zero on every cell; the only bootstrap CI that excludes zero is the
gfx1201 native-FP8 WT2 one, on the better side; absolute KLD below 0.08
everywhere). Evaluator builds: `pin:evaluator/eval_hipfire-40c52c783a-gfx1201`
(gfx1201) and `pin:evaluator/eval_hipfire-40c52c783a-gfx11` (gfx1100/gfx1151),
both commit `40c52c783a` (branch `canon-cand`). Receipts:
`/home/kaden/qcal/release-0.5.0/reliability/chunkfix/runs-canon/G4/` (`base/`,
`cand/`, `cand/paired-*.json`). The gfx942 rows are unchanged: the candidate
`qwen4_kld` `pin:evaluator/qwen4_kld-8842531d0e` reproduces both pinned
sequences byte for byte. The gfx1100 and gfx1151 rows round to the same KLD
(means agree to about 9 digits: the canonical route gives both gfx11 archs the
same arithmetic) but are distinct sequences from distinct devices; their
`.kldseq` hashes differ.

Previous (pre-canonical) RDNA pins are the `status = "previous"` rows
(`pin:kld/gfx1201.a4.wt2.pre-canonical` and its siblings, id suffix
`.pre-canonical`). Historical HIP 7.15 pins are the `status = "historical"` rows
with id suffix `.hip715` (e.g. `pin:kld/gfx1201.a4.wt2.hip715`).

**gfx942 (MI300X).** The gfx942 rows are not `eval_hipfire` runs and not the
27B model. Model: `pin:model/qwen3.8-flash-next.stock-fp8.hfq`. Evaluator:
`qwen4_kld eval --model MODEL --ref REF --output OUT.kldseq --decode` with
`HIPFIRE_GRAPH=0` and `HIPFIRE_NORMALIZE_PROMPT=0`, all other `HIPFIRE_*` unset,
scoring the production AR n=1 route, 24 chunks. References are BF16-teacher
`.kldref` files: WT2 `pin:reference/fn-bf16-wt2-24x2048.v3.kldref` (24 x 2048)
and code24 `pin:reference/fn-bf16-code24-24x512.v3.kldref` (24 x 512). Pinned
at evaluator build `pin:evaluator/qwen4_kld-165cf0e962` (commit `165cf0e962`,
v0.4.6), ROCm 10.1 / HIP 7.16.26385 / clang 24. Each cell was reproduced in two
fresh processes with separate HOME and empty `HIPFIRE_KERNEL_CACHE`
(byte-identical), and once more on a warm cache (byte-identical to the cold
pair). v0.4.6 already contains the HC quarters-gates route
(`HIPFIRE_QWEN4_G942_HC_QUARTERS_GATES`, default on), so these are its
numerics. Planted control: `HIPFIRE_G942_TRUNK_GEMV=1` moves the code24 hash
(the `control_*` fields of `pin:kld/gfx942.fn-fp8.code24`). Receipt:
`/home/kaden/qcal/release-0.5.0/s0/PIN-GFX942.md`.

## Pinned A3B MoE DFlash fixtures

For Qwen3.6-35B-A3B MoE DFlash perf/profiling work, use the
following command shape and do not substitute other prompts unless the
user explicitly updates this fixture section:

```bash
hipfire bench ~/.hipfire/models/qwen3.6-35b-a3b.mq4-awq-mi300x \
  --spec dflash --runs 5 --warmups 3 --max-tokens 256 \
  --kv-mode q8 --backend noslots --workload stateless --json
```

Pinned artifacts: target `pin:model/qwen3.6-35b-a3b.mq4-awq-mi300x`, draft
`pin:draft/qwen36-35b-a3b-dflash-mq4.hfq` (md5 pins).

The only permitted prompt fixtures for this A3B MoE DFlash thread are
`pin:prompt/merge_sort_thinking_off` (`benchmarks/prompts/merge_sort_thinking_off.txt`)
and `pin:prompt/humaneval_3_below_zero` (`benchmarks/prompts/humaneval_3_below_zero.txt`);
`scripts/check-pins.py` checks their md5 against the committed files. The best
observed results on them are recorded in
[`docs/perf-checkpoints/2026-11-15-agents-md-measurements.md`](perf-checkpoints/2026-11-15-agents-md-measurements.md).

Runs using any other prompt are exploratory only and must not be compared
against the A3B MoE DFlash perfmaxx line.

## Pinned Flash-Next bench fixture

The canonical Flash-Next trunk (MQ4 XTS) is the split base
`qwen3.8-flash-next-base.mq4-xts` from HF repo `hipfire-models/qwen3.8-flash-next`
(registry tag `qwen3.8:flash-next-mq4-xts`; the tags `qwen3.8:flash-next`,
`qwen3.8:flash-next-mq4` and `qwen3.8:flash-next-gptq3` resolve to the same
file), `pin:model/qwen3.8-flash-next-base.mq4-xts`. The short user-facing name
`qwen3.8:flash` aliases `qwen3.8:flash-next`. The base has no PLE rows; they are
the shared sidecar rows below.

- Shared PLE rows (same repo, the same pair the MQ3/MQ2 split tiers use):
  `pin:sidecar/qwen3.8-flash-next.ple-q4` (4-bit Lloyd-Max, rank order) and
  `pin:sidecar/qwen3.8-flash-next.ple-q8` (exact Q8F16). They attach to the base
  by source rows `18e2fef1` (derived=false). `hipfire pull --ple auto` fetches
  q4 on gfx1151 and q8 elsewhere (download sizes: the
  `download_bytes_with_ple_q4` / `download_bytes_with_ple_q8` fields of the
  base pin).
- Provenance: the base is `hipfire ple split` of the previous monolithic pin
  below. Two independent splits were byte-identical; a tensor walk found 1,197
  base tensors (`70873841176` B) byte-identical to the monolithic file, with
  only the 128 PLE n-gram shards (`54400261120` B) removed and a
  `hipfire_ple_rows` marker added. The shared q8 rows are byte-identical to the
  monolithic file's embedded rows, so **base + shared q8 rows is byte-equivalent
  to the previous monolithic pin**; with the q4 rows the rows are not exact.
- Equivalent monolithic (the previous pin, and the grandfathered identity the
  client still serves from a verified local copy): `qwen3.8-flash-next.mq4-xts`
  (also `qwen3.8-flash-next-gptq3.mq4`), `pin:model/qwen3.8-flash-next.mq4-xts`.
- Recipe: MQ4 XTS, the r2 tiers (MQ6G256V2 trunk (240 tensors) and language head,
  MQ4G256V2/MQ4G128V2 experts, Q8F16 embed/MTP-attention and PLE n-gram rows,
  128 shards, external-resident, 54,400,261,120 B), with the routed experts
  symmetric and GPTQ3-solved on the 262,144-token calibration corpus (no AWQ;
  553/25,088 experts keep RTN). All 1227 non-expert tensors are byte-identical
  to the prior pin. Needs a build at or after `cb566dab9` (qt=54 I64 metadata
  records); validated on `fe77c0837`.

The recipe above describes the monolithic file; the base is the same minus the
PLE shards. Re-bound sidecars for the base: trunk overlay
`pin:sidecar/qwen3.8-flash-next-base.mq4-xts.trunk-overlay` and expert hot set
`pin:sidecar/qwen3.8-flash-next-base.mq4-xts.expert-hotset`; the monolith-bound
overlay (`pin:sidecar/qwen3.8-flash-next.mq4-xts.trunk-overlay`) and hot set
(`pin:sidecar/qwen3.8-flash-next.mq4-xts.expert-hotset`) serve grandfathered
monolithic installs.

**EP2 serve fixture (2x R9700).** The 0.4.5 Flash-Next EP2 serve measurements
(cp20) use the symmetric-trunk variant `qwen3.8-flash-next-base.mq4-xts-symtrunk`
with its `.hc8`, `.ep.hot-units` and `.ep.expert-hotset` sidecars; fixture,
hashes, cells and method are in
[`docs/perf-checkpoints/2026-11-11-flash-next-ep2-cp20.md`](perf-checkpoints/2026-11-11-flash-next-ep2-cp20.md).
Use that cell plan (story/code x C1/C4/C8, fresh serves, pooled medians with
bootstrap CIs) for any EP2 serve claim, and the rules in
[`lessons-0.4.5.md`](methodology/lessons-0.4.5.md) § 4.

HF preservation policy: the monolithic HF objects (`qwen3.8-flash-next.mq4-xts`,
`qwen3.8-flash-next-gptq3.mq4` and the two retired artifacts below) must stay
published while 0.4.2 clients exist, because their bundles reference them.
Removing current registry entries is not permission to delete those objects.
The monolith-bound overlay, hot set,
`kld/qwen3.8-flash-next.mq4-xts.plehot` and the shared q4 stay on HF. This is
packaging, not a recipe change.

Before reporting Flash-Next results, verify the candidate trunk with
`scripts/check-pins.py verify model/qwen3.8-flash-next-base.mq4-xts PATH` (or
`sha256sum`) and require the pinned digest (the split base, or the equivalent
monolithic file with its own digest, `pin:model/qwen3.8-flash-next.mq4-xts`).

Gate receipts must quote the `qwen4 ple rows:` line (rule: Flash-Next gates on
mq4-xts use `HIPFIRE_PLE_FORMAT=q4` explicitly; R9700 receipts without q4 do
not count for gates).

Historical: from 2026-09-28 to 2026-10-02 the pin was the RTN-asymmetric
`qwen3.8-flash-next.mq4` (still on HF under that name, tag
`qwen3.8:flash-next-rtn-asym`; `pin:model/qwen3.8-flash-next.mq4`, whose
`sha256_before_qt54_retag` field is the digest before the 2026-09-29 qt=54
re-tag, same payloads). Measurements labelled `qwen3.8-flash-next.mq4` were
taken on those bytes.

Historical: the pin before that was `qwen3.8-flash-next.mq6q8-pleq8` (still on
HF, tag `qwen3.8:flash-next-mq6q8-pleq8`; `pin:model/qwen3.8-flash-next.mq6q8-pleq8`,
digest after the same qt=54 re-tag, `sha256_before_qt54_retag` before it); it
differs from the RTN-asymmetric pin only in its Q8F16 head. Caveats and
measurements below without a date were taken on that pin unless they say
otherwise.

These two historical tags (`qwen3.8:flash-next-rtn-asym` and
`qwen3.8:flash-next-mq6q8-pleq8`) are retired, not deleted fixtures.
New `hipfire pull` requests refuse them and name `qwen3.8:flash-next` as the
replacement. Existing SHA-256- and size-verified installed files remain
supported by old tag, bare filename or installed path in `run`/`serve`, without
redownload; missing files error rather than auto-download. Explicit paths to
copies elsewhere remain ordinary plain-file loads. Local `list` marks installed
models retired; the new available view excludes them. Compiled `RETIRED_MODELS`
pins retain repo/file/SHA-256/size, reusing `VerifiedCache` rather than rehashing
unchanged files. Neither artifact had sidecar, overlay or hotset pins.
No overlay/hotset pins are invented, and no Hugging Face files are deleted or
altered by retirement. Both retired entries are removed from the raw remote
registry and new bundled/loaded available views, without new wire fields.
Code inspection shows 0.4.2 rejects the generic `ple` schema and falls back
to cache or bundle; this is not an exercised old-binary test. Monolithic HF
files **must stay published while 0.4.2 clients exist**, because their bundled
entries still reference them.
The obsolete local-only `qwen3.8:flash-next-mq4-xts-zab` entry is removed
outright from the registry with no retired pin (its trunk tier ships as the
default overlay on the stock tags); the canonical split-base aliases, MQ2/MQ3
fixture rows and all overlay pins are unchanged.

### Flash-Next fixture caveats

Caveats that are part of the fixture, not trivia (measured results on this
fixture are in
[`docs/perf-checkpoints/2026-11-15-agents-md-measurements.md`](perf-checkpoints/2026-11-15-agents-md-measurements.md)):

- Loading it needs a build whose qwen4 trunk source contract admits **both**
  packed trunk tiers and whose external-PLE admission accepts both PLE tiers.
  Older builds refuse at load; that refusal is correct, not a corrupt file.
- **Flash-Next row-exact wide verify (arch 16, eager single target):** the
  physical executor/storage bound is **64 total rows** (seed plus at most 63
  native drafts), not a default tuning cap, a forced K=63 or a new algorithmic
  tuning cap; feasible offer depths and full-window costs still govern selection.
  Fixed eight-token PM tiles cover logical rows 1..64, including partial tiles;
  tile8 is not admission8. The global shipping qualified capability remains
  **8 GPU rows**, and CPU-owned-expert loads retain their independent **4-row**
  qualification. Actual allocated scratch/logit/capture capacity can narrow a
  window further. Explicit lab CPU reserve supports 64 rows through the common
  D1/D2 capacity and census, with exact input/mask/output/scratch slots (640
  slots); it is not CPU64 qualification. CPU routing stays per-call eager with
  whole-window forward/capture; the proven CPU default stays 4 until hardware,
  state and performance proof land. No PP/EP,
  lane-executor, codec or model-family widening follows from this contract.
  Verify remains eager: actual recording or graph capture is ineligible
  for wide verify. At each window entry bind its QSA reference to the
  **actual singleton AR execution mode** (Eager or Recorded), not just the
  backend request flag; Recorded selects the recorded singleton arithmetic
  without recording verify. Select attention policy per row's own position,
  batch only adjacent equal-policy rows, and restore the reference and row
  scope on success and error. Recompute after AR replay-state transitions.
  Qualification must compare raw bytes against the same architecture's
  actual singleton object for **every width 1..64**, with poisoned guards
  and tails; PM C1 or a CPU DAG transcription alone is not that proof.
  Stateful stages require sequential singleton transitions, all logit words
  and logical-state bytes, every kept prefix through both native repair and
  generic commit, abort and subsequent continuation; greedy MTP/copy needs
  identical token IDs and receipts proving the requested widths executed.
  Missing devices, unsupported instructions/arches, refused widths and
  missing cells are refusals, never passing skips. Finite-input CPU evidence
  does not prove arbitrary exceptional inputs or GPU behavior.
  The dedicated legacy aggregate qt44 gate/up `[512,1280,2560]` and qt53 down
  `[512,2560,640]` source targets rows 1..64 using a fixed **8-token** tile:
  C1 GU ROWS8 denotes eight output rows, not eight tokens. GU's 56-byte ABI
  preserves the original 52-byte prefix and adds total R at offset 52
  (M=1280); down keeps its original 52-byte ABI with total R in N at offset 40
  (M=2560, K=640). Each architecture (gfx1151/gfx1201) has two distinct exports
  in one image; GU uses block 256/static LDS 10,240 B and down block 32/LDS 0.
  This is neither Slab K1 nor descriptor reinterpretation, and changes no
  weights, math, codec or precision. The source G1 consumer keeps the real
  model alive, borrows immutable weights and compares original C1 with the
  candidate at every R=1..64 on real inputs plus two pathologies (192 cells),
  wordwise through GU, activation, down and combine; missing cells or mismatches
  fail. Source implementation is not exercised proof: assets/native CPU G0
  and G1–G4 remain pending, with no new emitted-image or whole-route PASS claim
  and no default promotion. Private frozen Slab 1928/96 native two-architecture
  all-row proofs do not qualify legacy aggregate or whole-model execution.
  Native capture COMMIT-TICKET/state coverage remains required by G2/G3;
  eager projection eligibility does not qualify HIP recording or graph capture.
  The existing `bf16_rows_oracle` accepts repeatable
  `--captured-ordinary MANIFEST` inputs: real canonical BF16 single/x4
  eager64 AR captures feed its R=1..64, full-word, poisoned-guard and actual
  C1 raw-code-object checks; the legacy HC input path remains available.
  The QSA pool primitive covers every kept prefix 0..R and GPU continuation
  at actual head dimension 128 and compression 4/8. Its raw 8193 and pooled
  2049 boundaries are distinct from selected 2051; this is not full
  selector/attention or captured-ticket G2 qualification.
  Qualification requires genuine `VerifiedCache` provenance for the prior
  payload SHA: exact canonical path, device/inode, size and mtime seconds/
  nanoseconds must match; missing, stale or wrong-digest records refuse.
  SHA-pin the cache artifact itself and record payload recomputation as false;
  do not repeat the consumed one-time full-model/Q4 hash or fall back to a
  fresh payload hash. Every submitted qualified-family launch must bind its
  actual loaded device ELF to the original G1 raw-code-object pins, not merely
  a container digest or source/module label. These consumers do not complete
  original G1–G4 qualification or promote the GPU8/CPU4 defaults.
  For partial G1 runs, use explicit `--rows 1-16`, `--rows 17-32`,
  `--rows 33-48` and `--rows 49-64` shards, with capture high-water rows
  16/32/48/64 respectively. Each shard covers 48 cells (real inputs plus
  two pathologies per width); one shard is not whole G1, nor evidence that
  four full models fit concurrently. The original 12 CPU producer cases
  compose nine prior passes with three affected-case passes; native legacy
  GU/down M7 is byte-exact on both gfx1151/gfx1201. These are software
  producer proofs, not a fresh all-12 run, whole FN G0 or hardware G1–G4.
  Shipping eight-GPU-row/four-CPU-row bounds remain qualification bounds,
  not arbitrary economic caps; the physical 64-row extent is not a K63 policy.
  Use `--preload-plan-only` for an unmeasured source plan before full load,
  not the post-load production RAW from `--resource-plan-only`. For the
  accepted bench PP512/context128/TG1 invocation, max sequence and placement
  chunk are both 544, not assumed 32768/4096; the plan does not qualify
  actual allocation, capture or runtime resource use.
  S1 `--execution-recipe` must bind the explicit target, tool/source ELF,
  source-bound sealed pack, reference origin and expected placement/resource
  contract. Keep genuine verified-model provenance in `HIPFIRE_HOME` while
  the kernel resolver uses its cold private `HOME`; never seed, copy or
  freshly hash a payload to manufacture a cache hit. The original gfx1100
  teacher and seven primitive cases remain unchanged: a gfx1201 target is
  not a new teacher or cross-architecture greedy proof. Keep original KL
  limits (16K: 0.03948458079446915; 4K: 0.027284316982763658), not +0.010.
  Symmetric IU4 is requested ON by default on gfx1100, gfx1151 and gfx1201;
  pending qualification never justifies switching it OFF (`=0` is an isolated
  control). Active execution still requires actual eligible headers,
  descriptor, row and resource admission; a request is not an active-route
  receipt. The gfx1100 default (regular prefill and the S1 `aggregate-sym-iu4`
  window read the same `Gpu::qwen4_moe_sym_iu4_requested`) was flipped on
  with its numeric acceptance recorded in the CHANGELOG entry (S1 context-mean
  KL within the limits above; WT2/code24 BF16-teacher KLD deltas +0.0293 /
  +0.0296 vs gfx1201 +0.0306 / +0.0285, the code24 excess of 0.0011 waived by
  Kaden 2026-10-29; pp8192/pp32768 A/B +1.2 / +1.4%). FP8 KV and FA2 are
  independent, stackable choices; A3B H16/KV2/D256 FA2 work is not an applied
  Flash-Next H24/KV2/D256 QSA FA2 port or a Flash-Next performance gain.
  **Widened byte/state/ID and performance qualification is pending.** Before
  any per-architecture table promotion, require pinned source/object/model
  receipts and paired full-window cost curves against incumbent
  prefill-kernel-routed verify at identical widths and starting state, with
  an uncertainty-supported cost-per-row win at each promoted cell. A host
  singleton loop, faster component kernel or MTP tok/s is not that result.
  Keep the per-architecture widened qualification cells in
  `docs/design/qwen4-program-retained-pm4.md`'s eager-wide-verify section
  separate from its historical single-token retained receipts.

- **MTP `auto` (`speculation.mtp = auto`, the schema default) arms native MTP
  only for a producer that declares a measured AR floor. Today that is Qwen4
  (Flash-Next, arch 16), except on MQ4R-SKU loads (HFQ `hipfire_route_sku`
  metadata or the registry card, never the file extension), which keep the
  retained Redline AR route. Qwen3.5-class (arch 5/6, dense and MoE), DeepSeek V4 and
  Gemma 4 declare no floor (the DS4 native-MTP audit found none), so `auto`
  resolves to AR there even when a head is present. An explicit
  `speculation.mtp = on` / `--spec mtp` still attaches any present head on
  every producer, and DSpark / DFlash selection is unchanged. Qwen4 MTP serves greedy and sampled
  requests: sampled requests are verified by
  speculative rejection sampling against the AR sampler's distribution
  (oracle: `crates/hipfire-arch-qwen4/tests/sampled_mtp_distribution_hw.rs`).
  Sampled MTP output matches AR in distribution, not byte for byte.
  Non-neutral penalties keep native MTP for sampled requests when
  `supports_temp_verify` is true (since `9f1ea2171`, shipped in v0.4.1);
  greedy requests with penalties run AR
  (`crates/hipfire-generate/src/ar.rs:1715–1734`).
  Flash-Next batch lanes refuse non-neutral repeat/presence/frequency penalties
  (`crates/hipfire-generate/src/batch.rs`, `qwen4_lane_request_supported`), but
  batch sampled neutral-penalty requests under `serve.batch_sampled`: AR lanes
  draw with the AR singleton sampler and request seed, MTP lanes verify sampled
  windows with naive acceptance.** Each MTP
  window picks its verification route: a batched `(K+1)`-row verify at the
  draft depth `K` (up to the `mtp_k` cap; `auto` for native Qwen4 MTP =
  `min(GPU row-exact verify rows, CPU-owned-expert window) − 1` resolved after
  placement: 7 today, 3 on a CPU-placed load; else 3) that maximizes expected emitted tokens per window cost
  (per-depth draft agreement, decayed), or the interleaved route (one target
  row per draft, stop at the first rejection) when no depth pays.
  A same-request AR floor guards the default route: the request's first
  window is one ordinary target-only token, timed (a sampled draw included)
  as the AR price and followed by one head append (the drafter's first such
  token pays the decode route's one-time kernel binds, so it prices nothing
  and the next window calibrates again); up to three speculative
  probe windows then run under the table-ordered route. A batched depth is
  priced by the depth (cap) it was routed at: the cap's decayed mean window
  time over the agreement-predicted tokens of its windows' drafted-depth mix,
  so the windows the margin prune cuts short count at their real, lower, cost
  and yield (the drafted-depth model kept Halo prose at depth 2 while forced
  K=4 with the prune ran +8.4%); interleaved by its time per emitted token.
  The AR verdict (retire or keep drafting) then reads only options measured on
  this request; once a measured option beats AR the window runs the cheapest
  candidate: every cap, its evidence blended with the neighbouring cap's
  (d±1, the one with more windows) for what that neighbour holds beyond it,
  shifted by one row on the neighbour's windows that drafted its whole cap.
  The row is the request's own window-cost fit (a decayed least-squares line
  of each priced batched window's wall time on its drafted depth, ridge-pulled
  toward `MTP_ROW_COST_PRIOR`, 0.27 AR steps per row, until windows at two
  depths are priced), and every cap's evidence decays with every batched
  window, so an unmeasured cap or one measured only during a poor start is
  priced from the cap that runs. A request climbs one cap at a time, up to 7
  (the 8-row verify), while the next row pays, and stays shallow where it does
  not. On the singleton drafter the first batched window at each drafted
  depth (across requests) is observed unpriced and spends a probe: the
  process's first speculative window binds the whole verify route's kernels
  (gfx1151: ~+17-20 ms once per process). Chooser candidates and the fitted
  per-row cost appear in `HIPFIRE_MTP_FLOOR_TRACE` lines (`capN`, `modeled`,
  `drafted`, `row_us`). When none costs
  strictly less per emitted token than AR (a
  tie, or an invalid AR timing, counts as no win) the request retires to
  head-free AR for good, unless n-gram windows keep it alive: a request stays
  alive on a losing native route only while external (n-gram) windows are
  measured and unblocked (cost per token below AR) and the decayed realized
  per-token cost of all its windows, native and external, is strictly below
  AR; it retires at the first window where that stops holding.
  Before that sticky retirement of a request (singleton or continuous-batching
  lane floor) whose draft budget is above 3 and that has run a deep (cap > 3)
  window (it runs after the AR re-price probes where they exist; a budget of 3
  or less never re-probes), the floor requires fresh shallow (K ≤ 3: batched
  cap ≤ 3 or interleaved) native evidence: 3 priced shallow windows since that
  last deep window. Without them it routes the request at cap ≤ 3 for a
  bounded re-probe from the existing probe budget, which ends at 3 priced
  windows or 6 in all (a first-use unpriced window and external windows spend
  the bound; no new environment switch). AR is re-priced first where it can
  be; a lane with a 0 AR probe budget still gets the bounded shallow control,
  starting at its first would-be retirement. A shallow cost
  below AR holds the request on its best measured shallow depth, route capped
  at 3 until a deeper candidate prices below both the shallow depth and AR for
  3 consecutive windows (positive evidence); a shallow cost at or above AR, or
  no priced shallow window at all (fail closed), retires it.
  `HIPFIRE_MTP_FLOOR_TRACE` lines carry `budget_k`, `deep_seen`,
  `shallow_state` (`idle|probing|holding|lost|unpriced`), `shallow_windows`,
  `shallow_priced`, `shallow_budget`, `shallow_since_deep`,
  `shallow_us_per_tok` and `climb_streak`. The `auto` cap rule above is
  unchanged, as are explicit `mtp_k` overrides and the D1/D2 hard gates.
  A retired request runs with no head append, copy, snapshot or replay, forced
  advances fall back to the target alone, and only `configure_request`
  (a new request) resets it, not `mtp_reset` or a prefill realign. At most one
  calibration head append and three probe windows precede the first measured
  decision; this bounds the speculation work, not the full-request time. A
  floor-observed native window is priced from its start to the next window's
  plan, an interval that holds its whole GPU tail without a device sync at
  every window end, and the retired token host-syncs its top-1, which the AR
  producer does not.
  With `HIPFIRE_MTP_PHASE_TIMING=1`, batched `window_us` includes the snapshot
  and commit tail; diagnostic forced routes synchronize for cost accuracy.
  `HIPFIRE_MTP_INCREMENTAL=0` forces batched at the full `mtp_k`; `1` forces
  interleaved; either forced route is a diagnostic override that bypasses the
  floor entirely. `HIPFIRE_MTP_AR_FLOOR=0` opts out of the floor (diagnostic
  A/B): no calibration token, probe windows or retirement, and the
  per-window chooser routes as before the floor; the forced routes win if both
  are set. Interleaved verification (`HIPFIRE_MTP_INCREMENTAL=1`) uses the
  AR target's single-row route. With Q8 GDN recurrent state, the row-capture
  verify/rollback kernel (gfx11+ SIMT, `kCapture && kQ8`) applies AR's
  per-token Q8 boundary at each row's absolute position, so every captured
  verify row, the final slot and every rollback keep-prefix equal sequential
  single-row decode. Oracles: `gdn_q8_verify_matches_single_row`
  (rdna-compute) and the ignored
  `crates/hipfire-arch-qwen4/tests/greedy_mtp_identity_hw.rs` (canonical
  GPTQ3 MQ4 Flash-Next, p1, 1200 greedy IDs): on gfx1151 forced batched,
  forced interleaved, adaptive and full-head batched MTP all match AR in
  three fresh-process runs (before the repair, batched first differed at
  index 112 and adaptive at 130). This guarantee covers only that capture
  path: it is not a claim for the unarmed few-row Q8 prefill/replay route
  (used where rows are not captured) or for architectures the oracle has not
  run on. The captured gfx11+ SIMT kernel writes only the final recurrent
  state, not a retained state for every row: native partial acceptance
  reruns the kept prefix's recurrence from the saved live slot and repairs
  causal histories/marks. Generic restore-and-replay commit is a separate
  path and needs its own token-canonical capture gate.
  Drafts rank the vocabulary with an MQ2 copy of the LM head and re-score its top 8 exactly
  against the head's own Q8_0 or MQ6G256V2 rows (`HIPFIRE_MTP_DRAFT_HEAD`,
  default `mq2r`).
  Teacher-forced prompt and replay steps advance MTP state without computing
  an unused language-head prediction; prompt target chunks emit only their
  final logit row while retaining every wide hidden row.
- **MTP + ngram-mod takeover (`speculation.mtp_ngram`, `HIPFIRE_MTP_NGRAM`,
  default `auto`):** on for native Qwen4 MTP (arch 16) on gfx1151 and gfx1201
  and off for every other producer; `on`/`1` arms every MTP producer that
  supports it, `off`/`0` is the opt-out (env beats TOML beats default). It
  serves greedy and sampled requests, thinking on or off (Qwen4 and
  Qwen3.5/3.x). The Qwen4 proposal source is switched by
  `speculation.mtp_ngram_copy` / `HIPFIRE_MTP_NGRAM_COPY` (`on`/`off`/`auto`,
  also `1`/`0`; default `auto`, which is **on for native Qwen4 MTP (arch 16)
  on gfx1151 and gfx1201** and off on every other GPU; `off`/`0` is the kill
  switch; env beats TOML beats default). Default history: the lane shipped
  with `auto` off on every GPU (`448a471ea4`); `auto` arms gfx1151 and gfx1201
  since `c921903686` (`copy_lane_arch_admitted` in
  `crates/hipfire-config/src/lib.rs`). Flash-Next EP mesh lanes stage the same
  copy lane by default wherever `auto` arms it, in steps of at most
  `HIPFIRE_QWEN4_LANE_COPY_LANES` (2) lanes (`047ed76122`; `=0` turns it
  off). **Off is the 5/3/3 pool lane
  exactly.** On Qwen4 a pool
  hit (`MtpNgramContext`, the 5/3/3 pool; request-local, cleared and reseeded
  from the full prompt every request, so earlier requests never influence
  candidates, and cleared after five low-acceptance windows) takes the window
  only when the head's own first draft (row-0 argmax, before any sampled draw)
  equals the first candidate (the row-0 confirmation runs on every takeover)
  and the takeover's decayed emitted tokens per window over its verify cost
  (the measured mean, else the request's window-cost fit, not the static
  per-arch row table; the yield gate, `ngram_yield` / `ngram_wins` /
  `ngram_takeover_depth`) is at least the native chooser's expected rate (the
  floor's cheapest candidate once the floor is measured); otherwise the window
  is native (the batched route reuses that row-0 step). A takeover ends when
  the AR floor retires the request, and load admission sizes the verify rows
  and row capture by `max_k`. The drafter holds `ngram:
  Option<MtpNgramContext>` in this lane and the floor observes takeovers
  through `MtpFloor::observe_external`; no `ngram_plan` / `ngram_confirm`
  trace events are emitted.
  **`on` arms the copy lane** (`Qwen4MtpDrafter::with_copy_lane(true)`, the
  drafter's `ngram_copy: Option<MtpCopyContext>` instead of `ngram`, floor
  observation through `MtpFloor::observe_external_copy`, triple 5/1/63). In
  that lane the proposal source is `MtpCopyContext`
  (`hipfire_runtime::ngram_mod`), a request-local verbatim-copy
  (prompt-lookup) matcher: one direct-mapped table per key length (32, 16, 8
  and `n_match`, longest first), each hash → the position just past the key's
  most recent occurrence in the request's history (prompt plus retained
  output), the match checked token for token (a hash collision never
  proposes), and the proposal copies `history[pos..]`. It is seeded from the
  full prompt every request and its tables are never cleared (a slot from an
  earlier request can never propose, so earlier requests never influence
  candidates); there is no low-acceptance clear on Qwen4 (the old pool's
  five-low-window clear wiped a request's seeded prompt, up to 200K tokens,
  for the rest of the request), the take-length chooser backs off on measured
  acceptance instead. On a CPU proxy of the agentic code-edit chain
  (`release-0.4.3/ngram-copy-research`; proxy, not a hardware measurement)
  conditional per-depth acceptance was 0.991-0.997 with these keys against
  0.90-0.96 for the 5-token chain of the earlier pool. Each window with a
  proposal (up to `n_max` tokens) takes a length `d` in `1..take_cap` chosen
  from the request's own shadow-measured per-depth n-gram acceptance `a_j`
  (the proposal is compared with the tokens the window committed, whichever
  source ran it: on a greedy stream exactly the verify's acceptance, on a
  sampled stream an unbiased point-mass estimate) and measured takeover costs:
  the `d` maximizing `(1 + Σ_i Π_{j<=i} a_j)` over the takeover's window cost
  (the request's measured mean for that candidate count, else the nearest
  measured count moved by the fit's per-row slope, else the native window
  cost model; a drafter's first takeover of a row count is observed unpriced
  like the first native window of a depth), taken when at least the native
  chooser's rate; otherwise the window is native. The head's own first draft
  (row-0 argmax, before any sampled draw) must equal the first candidate only
  until the copy source is established: a take of at least 5 candidates with
  depth-1 acceptance at least 0.9 over at least 4 windows of evidence
  (checked against agentic-chain replays, `NGRAM_UNCONFIRMED_MIN_*`)
  **and** the last shadow observation having accepted
  its first candidate (`ngram_last_first_accepted`, so a divergence costs at
  most one blind window) runs with no head row-0 step.
  `take_cap = min(n_max, row-exact verify rows - 1)`, where the rows are the
  per-GPU `rdna_compute::prefill_invariant::qwen4_row_exact_verify_rows` (8
  on every GPU today, so 7) narrowed per window by
  `Qwen4Bundle::row_exact_verify_rows`: a verify forward is row-exact against
  single-row decode (Q8 decode copy, K1 lane-union MoE, rows_x4 GEMV, k4 HC
  gate, QSA prologue, fused router+shared) only up to those rows; more run
  prefill kernels, which break greedy byte-identity. Raising the one per-GPU
  value raises the cap.
  Admission sizes the verify rows and the GDN row capture for `take_cap + 1`
  rows when the n-gram source arms (`native_mtp_proposal_drafts`, which the
  loader passes to `native_mtp_device_bytes` / `native_mtp_request_bytes`),
  and the drafter grows its row capture and verify scratch to the armed cap.
  The capture ring itself is charged at the wider of that window and the
  arch's row-exact verify rows (`native_mtp_capture`, shared with the
  runtime's `native_mtp_capture_rows`).
  **A request the floor retired keeps the copy lane.** Retirement says native
  drafting loses to AR, nothing about the copy source, whose acceptance the
  shadow table measures independently (on R9700 at the agentic chain's
  sampling, T1.0 thinking on, the floor retired every request during the
  thinking prose and the answer's verbatim code copy ran as AR; smoke:
  `release-0.4.3/agentchain/runs/smoke-r9700/def/res.json`). Only the floor's
  AR calibration (and its probes) bars the pool; on a retired request the
  comparator is AR itself (1.0 token per AR step) instead of the native rate.
  A retired takeover first catches the head up (`catch_up_head`: the rows the
  retired AR tokens owe are appended teacher-forced exactly as a forced
  advance does), then confirms the first candidate with the head's row-0
  probe unless the source is established, falling back to the retired AR step
  on a mismatch, else runs the takeover with no head step. The floor keeps
  pricing external windows after retirement (`external`, `external_depth`,
  `external_blocked`) without un-retiring, counting probes or changing native
  routing, and reports `external_blocked` alone, so the lane disarms only when
  takeovers themselves cost at least AR per token. `stats.mtp_retired` stays
  true and the `ngram_mod_*` counters count retired takeovers.
  A takeover verifies `[seed, candidates..]` on the same batched verify
  executor as native MTP windows (interleaved one-row route under
  `HIPFIRE_MTP_INCREMENTAL=1`), rolls the target back to the consumed rows,
  and appends exactly those rows to the head teacher-forced (token p, hidden
  p). Greedy takeovers are exact;
  sampled takeovers verify each candidate with the point-mass draft
  distribution δ(candidate) through the shared rejection verify, so output
  matches MTP-only in distribution, not byte for byte. The native chooser's
  per-depth agreement relaxes, for depths a window did not compare, toward the
  next-shallower depth's agreement (depth 0 toward the 0.8 prior): at
  `MTP_AGREEMENT_RELAX_UP` (0.8) per window while that agreement is above the
  depth's own, `MTP_AGREEMENT_RELAX` (0.95) when below, so a depth rejected
  once is re-explored instead of staying frozen below its threshold, and an
  unreached depth is priced at the request's own deepest observed acceptance
  (conditional acceptance is flat in depth past the first draft); with the
  copy lane on the n-gram table uses the same rule. Triple
  `HIPFIRE_NGRAM_MOD_{N_MATCH,N_MIN,N_MAX}` is 5/3/3 on admitted Qwen4
  targets with the copy lane off and 5/1/63 with it on (24/48/64 elsewhere;
  per-component overrides; Qwen4 `n_max` capped at 63, the 64-row spec verify
  capacity; with the copy lane on it is the proposal and observation length
  and the take cap is `min(n_max, 7)`). The load log prints
  `qwen4 native MTP n-gram: on|off (n_match=.. n_min=.. n_max=..)`, plus
  `qwen4 native MTP n-gram copy lane: on (take_cap=N)` only when the n-gram
  source is on and the copy lane is armed;
  `HIPFIRE_MTP_TRACE=1` window events carry `source` (`ngram`/`mtp`) and
  `verify_route`; with the copy lane on they also carry `ngram_plan`
  (`emitted`, `available`, `take`, `confirm`, `rate`, `native_rate`, `a1`,
  `plan_us`, `retired`) per window with a proposal (retired plans included)
  and `ngram_confirm` (`position`, `match`) wherever the row-0 confirmation
  ran; with it off neither event is emitted.
- **Concurrent MTP output depends on batch composition by default (Flash-Next
  EP lanes, exact gfx1201).** With `HIPFIRE_QWEN4_LANE_ROW_EXACT` off (the
  default, `20c7b6de0e`), lane verify steps of 9..=64 rows take routes whose
  per-row arithmetic depends on the co-batched rows, so C>1 MTP output may
  differ from the singleton's; C1 and AR stay exact. `greedy`/`seeded` opts
  into batch-invariant output (code C8 step +82%, C4 +41% on the measured
  fixture). Bitwise gates for lane changes therefore compare candidate against
  incumbent under the same setting, never a lane against the singleton
  ([`lessons-0.4.5.md`](methodology/lessons-0.4.5.md) § 7).
