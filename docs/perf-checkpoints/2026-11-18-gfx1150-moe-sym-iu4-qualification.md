# gfx1150 symmetric-IU4 MoE (NT1) qualification — 2026-11-18

Machine: Ryzen/Radeon iGPU gfx1150 (Radeon 890M class, 16 CU RDNA3.5), ROCm
`/opt/rocm/core` (HIP 7.16, clang 24), 61 GB UMA RAM (42.9 GB visible to HIP).
Model: `qwen3.6-35b-a3b-d53.mq4rs` (registry `hipfire-models/qwen3.6-35b-a3b`,
quant_recipe `mq4v2-symmetric-no-q8-router-down-mq4g128v2-symmetric@7da11034`),
18 700 570 368 B, sha256
`8806a742d558ebe43256010096cd82cfac3a20be5218ad56a3f74267db001894` (verified).
Symmetric qt44 gate_up (`MQ4G256V2`) + qt53 down (`MQ4G128V2`), routed experts
fully resident, headers loader-verified symmetric (`zp == -8*sc`).

Branch work: `feat/gfx1150-moe-sym-iu4` @ `21832d4de` (P1 `2440cc1f9`, P2
`af881c8ed`, P3 `21832d4de`, plus `252be11c5` — the C2 producer flag fix found
by this qualification, see below). Base: `origin/beta` `625c491d1`.
Binaries: `target/release/hipfire` and `target/release/daemon` built from this
tree (build commit `21832d4de`), served from an isolated `HIPFIRE_HOME`
(`/tmp/opencode/hipfire-home-p4`); the installed `/home/david/.hipfire/bin/*`
(built from `b0718ec43`) was not used. `HIPFIRE_HOME` pointed at the isolated
copy; kernels compiled once per arch into the shared
`/home/david/.hipfire_kernels/gfx1150`.

## Why this checkpoint exists

The gfx1150 890M-class iGPU was admitted to the symmetric-IU4 grouped MoE
route for Qwen3.5/3.6-MoE artifacts whose routed experts are symmetric
qt44+qt53 (this file). Before this work, the artifact loaded only onto the
qt53 F32 fallback (~3.7x slower prefill). The route is admitted behind
`HIPFIRE_MOE_SYM_IU4` (default OFF on gfx1150 until KLD-backed promotion):
`1` opts in, `0` kills it with one WARNING.

## P4 test round (hardware, this machine)

1. `cargo build --release` of `hipfire-cli` and `hipfire-daemon` from the
   worktree: OK (rc 0). gfx1150 kernels compile on load (first request
   compiles the hipcc modules; results in the shared arch kernel cache).
2. Parity, greedy 48 tokens, pinned prompt `benchmarks/prompts/bare_factual.txt`
   (159 B, md5 `1d32df5f12c414d3e34c7b35b6611e6c`), `-t 0 --top-p 1`,
   `--spec off`, `--no-stream`, both arms `HIPFIRE_MOE_SYM_IU4` = 0 and 1:
   **byte-identical output**
   (`The capital of France is **Paris**.\n\n*   **River:** The **Seine** River
   runs through Paris.\n*   **Famous Landmark:** The **Eiffel Tower** is one
   of the most iconic landmarks located in`), finish_reason `length`.
   First 48 of more tokens verified equal; full distribution not yet pinned.
3. Route admission evidence (IU4 arm logs):
   `moe expert IU4 prefill (HIPFIRE_MOE_SYM_IU4): requested=true admitted
   host_mapped=false graph_capture=false arch=gfx1150 rows=…
   route_rows=8192 gate_up=MQ4G256V2 down=MQ4G128V2
   numerics=SymmetricBf16SlotsF32OrderedCombine
   modules=qwen4_moe_iu4_sym_pm_gfx1150[…]+moe_slot_ops_pm_gfx1150[…]`,
   `qwen35 symmetric IU4 MoE: INFO: 40 MoE layers admitted by capability
   (default on; HIPFIRE_MOE_SYM_IU4=0 kills it); headers verified symmetric,
   scratch reserved`, `scratch reserved for 8192-row widened resident MoE
   prefill`.
4. KLD: **not available**. `fixtures/pins.toml` carries A3B pins only for
   `qwen3.6-35b-a3b.mq4-awq-mi300x` (and its DFlash draft), not for this
   symmetric artifact; no `.kldseq` reference exists for it on any arch.
   A KLD-backed claim is therefore **withheld**; the evidence here is the
   bounded greedy check (48 tokens, identical) plus the battery below.
   Any promotion (e.g. `default_requested: true`) needs new pins + KLD runs
   first, via Kaden's promotion lane.
5. Coherence (`scripts/serve_harness.py --mode battery --thinking off`,
   kv q8, mtp off, temp 1.0 registry sampling, `--max-tokens 512`):
   all five genres (`code`/`reason`/`factual`/`prose`/`instruct`) coherent in
   both arms (IU4: finish=stop, gen 68–306 tokens; incumbent: finish=stop,
   gen 68–329 tokens). Both arms read as normal sampled output; no runaway,
   no empty completions. (Raw dumps: `/tmp/opencode/p4-battery-iu4.json`,
   `/tmp/opencode/p4-battery-incumbent.json` — note sampled runs differ
   token-for-token between arms by construction; this is a coherence check,
   not a bitwise one.)
6. Perf: fresh processes, byte-identical prompt
   `benchmarks/prompts/agentic_pi_system.txt` (md5
   `2d4892884dfeec9fbddfde2a2f47220f`), `--spec off --max-tokens 8
   --backend noslots --workload stateless --kv-mode q8 --temperature 0`,
   draft off, 4 interleaved fresh loads per arm (A/B/A/B order per round).
   Host load average at start: 0.93 (1 min).

   | round | arm 0 (WMMA incumbent) prefill tok/s | arm 1 (IU4 NT1) prefill tok/s | TTFT arm0 (ms) | TTFT arm1 (ms) |
   |---|---|---|---|---|
   | 1 | 281.10 | 757.70 | 4394.0 | 1630.0 |
   | 2 | 281.00 | 756.10 | 4395.6 | 1633.5 |
   | 3 | 281.40 | 760.10 | 4388.6 | 1624.8 |
   | 4 | 281.30 | 757.90 | 4389.6 | 1629.6 |

   Median arm0 = **281.2 tok/s**, arm1 = **757.8 tok/s**, speedup **2.695x**
   (per-round ratios 2.694/2.690/2.700/2.695; bootstrap 95% CI over the ratio
   of medians, 20 000 resamples: **[2.688, 2.695]**). Decode ~36 tok/s both
   arms. TTFT median 4392 -> 1629 ms (2.70x). Within-arm sd ≈ 0.17/1.7 tok/s.

   Caveats: both arms spill a few hundred MiB GTT (the model is 18.7 GB on a
   42.9 GB visible UMA iGPU) and both arms report fast_ppt power throttling
   >10% of measured runs; numbers compare only against runs in the same
   thermal/power regime. Decode is unaffected by the route (~36 tok/s both
   arms).

## The load-bearing bug qualification found (commit `252be11c5`)

The first `HIPFIRE_MOE_SYM_IU4=1` load refused with
`moe.iu4-arch-incapable` even though the descriptor, objects, slot-ops and
checker were all in place: `FeatureFlags` appended
`-DIU4_A4_CANDIDATES=2` only under `matches!(arch, "gfx1100" | "gfx1151")`
(and the gfx1201 arm), never on gfx1150, so
`iu4_a4_candidates_c2_only(hipcc_extra_flags)` was false and
`moe_sym_iu4_requested()` never became true. Fixed by adding `"gfx1150"` to
that arm and naming gfx1150 in the CONFIG.md paragraph
(`kernel.gfx11_a4_candidates`). This is why hardware qualification is not
optional: the static wiring tests were all green.

## Not done / next

- KLD qualification pin(s) for this artifact (would need a `.kldseq` pin in
  `fixtures/pins.toml` for gfx1150 and the reference `.kldref`).
- `default_requested: true` flip — requires KLD data + ratchet + promotion
  via Kaden's promotion lane. NEVER flip on this evidence.
- serve_harness chain/session modes; the `--tool-chain` transcript.
- Served-path perf (`hipfire bench --backend slots`, continuous batching).
- gfx1150 flagship pack fixture (`flagship-pack-gfx1150.tsv`) if/when the
  route is exercised by CI.
