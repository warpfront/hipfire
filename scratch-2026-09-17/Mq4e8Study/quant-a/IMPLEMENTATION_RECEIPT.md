# MQ4E8 slice A (quantizer/requant) — implementation receipt

## New CLI flags (`--format`, pipeline.rs:325-334)
`mq4e8-c0/c1/c2/c3/c4/c4z/c4s` → `Mq4HeaderConstraint::{Unconstrained,
HalfPow2,HalfInt16,GroupShared,RowPow2,RowPow2IntZero,RowPow2KMinus8}`.
Typo guard (`mq4e8-*` unrecognized → exit 2) adjacent to the arch gate.

## Exported signatures (crates/hipfire-quantize/src/quant_mq4e8.rs)
- `enum Mq4HeaderConstraint` (:24) — 7 variants (C0 + C1..C4 + C4z + C4s)
- `fn mq4e8_suffix(c) -> &'static str` (:48)
- `struct Mq4e8Stats { c2_clamp_lo/hi, master_floor, zero_master_rows: Vec<u32>, zero_scale_halves }` (:63)
- `fn quantize_mq4g256v2_constrained(w:&[f32], m, k, signs1, signs2, constraint) -> Result<(Vec<u8>, Mq4e8Stats), String>` (:648)
- internal: `base_half_steps` (:134, fixture-identical minmax), `c1_scales` (:159),
  `c2_scales` (:197), `lloyd_free_zeros` (:239), `lloyd_int_zero` (:301),
  `assign_fixed_kminus8` (:375), `choose_row_exp` (:470), `row_master_b` (:491),
  shared `assign_q` DAG (:108, bit-identical to `quantize_mq4g256v2`)

## Callsites (pipeline.rs)
- import (:35); `MainQuantFlags::mq4e8` field (:113), construction (:2704), test literal (:7066)
- arch gate: `use_mq4e8 && arch_id != 5` → exit 2 (:1142)
- dispatch: `} else if let Some(mq4e8_constraint) = flags.mq4e8 {` ahead of the
  primary dense `use_mq4v2` arm (~:5886); same AWQ recipe (alpha/imatrix/
  eligibility/sidecars, seeds 42/1042); K%256 enforced (exit 2, no fallback);
  emits `QuantType::MQ4G256V2`/label `MQ4G256V2` (census unchanged)
- embed stays Q8 via existing guard (+`flags.mq4e8.is_some()`, :5790)
- metadata: `hipfire_base_format=mq4e8-cN` (automatic) + `hipfire_mq4e8_study`
  provenance (:2911); GGUF path fails closed on unknown formats (no change)

## Ownership
Slice A owns `crates/hipfire-quantize` (quant_mq4e8.rs, pipeline format/dispatch
above, cli unchanged). Untouched: default `quantize_mq4g256v2` body (gate-2
byte-identity md5 85f5fa20dd45afa9754bfeb8e78b29a4 + in-tree fnv tripwire),
loader/GEMV/fp8, kernels/, rdna-compute/. No global default mutation.

## Unsupported behavior (all exit 2, no artifact)
- arch_id != 5 (MoE included): `mq4e8-c* is dense-qwen3_5-only`
- K%256 != 0 in study mode: no HFQ fallback
- unknown `mq4e8-*` suffix: typo guard; GGUF input: existing "not recognized" error
- inexact scale/zero (assert on every group/row): encode Err → exit 2, nothing published

## Commits (branch mq4e8-study)
- 2fd1416bb gates 1-2 (encoder + oracle + byte identity)
- 200af2e48 CLI formats + dense dispatch
- 6344f3a94 exact AWQ reference-scales dump test

## Tests (non-project-wide) & logs (all under Mq4e8Study/quant-a/)
- `cargo test -p hipfire-quantize quant_mq4e8` (5 tests: oracle 7/7 MATCH,
  default-bytes tripwire, scale relations, C1 ratio/C2 clamps, K reject);
  release-mode oracle 7/7 MATCH; `pipeline::` 13/13 pass
- gate1_oracle.log (Rust 7/7 + Python oracle_py.py 7/7 md5 MATCH vs planner files)
- gate2_byteid.log (pre/post md5 identical)
- gate3_byteid_cN.log ×7 (all PASS: 496/496 sidecars + 354/354 others identical)
- requant_cN.log + requant_cN.cmd ×7 (census 497/801/49 each, ~100s each)
- awq_scales.bin (496 AWQ tensors; truncates 0-mismatch to fixture sidecars)
- spotcheck.py (3-tensor chain validation: C0 +11–17%, C2 +25–39%, C4 +21–30× tail)

## Artifact table (/home/kaden/qcal/ladder-v2/artifacts/, 14987185152 B each)
| tag | file | md5 | elapsed |
| c0 | qwen3.8-27b.mq4e8-c0.hfq | ccf14c7cbfe2e7602f22e56a2463f170 | 98s |
| c4 | qwen3.8-27b.mq4e8-c4.hfq | e5a522a0a27a779c3db15949010e0178 | 98s |
| c2 | qwen3.8-27b.mq4e8-c2.hfq | 6582bc9b0d8cb4915c877711bc2a1d02 | 104s |
| c4z | qwen3.8-27b.mq4e8-c4z.hfq | 0c0278958f144bd4af5e9a5354e5679c | 83s |
| c4s | qwen3.8-27b.mq4e8-c4s.hfq | 48e3d81b49e1723b57de1b1f6471b1c5 | 86s |
| c1 | qwen3.8-27b.mq4e8-c1.hfq | 3676676ea4e6f135926675a93e267075 | 101s |
| c3 | qwen3.8-27b.mq4e8-c3.hfq | fac22847faa912ea1cd9c150ea9e91dd | 99s |
Recipe: `--format mq4e8-cN --imatrix …/Qwen3.8-27B-imatrix.gguf --awq-alpha 0.55 --tier xt`
from `/home/kaden/qcal/parents/qwen3.8-27b/` (no --include-vision).

## Constraint event report (from requant logs; cross-matches planner sweep)
- C2: m-cap clamps lo=0/hi=769; B-floor→0x0010 226 groups
- C4: master floor→0x0001 3 rows; zero-master rows 0
- C4z: cleared-master floor→0x0010 28 rows; zero-master rows 0
- C4s: master floor 3 rows (same masters as C4)
- C0/C1/C3: no clamps/floors (C1 candidate restriction never empty: G always valid)
- C1 excluded-candidate / full tail tables: planner's sweep (avoided duplicate)

## Scope notes
- Gate-4 full-sweep numerics: planner-owned (their tables complete); slice A
  provides clamp/floor reports + exact-reference tooling only.
- Gate-5 1-chunk smokes: superseded by Main's override — planner's full-model
  1/2/24 KLD is the load/route smoke (no duplicate GPU use).
