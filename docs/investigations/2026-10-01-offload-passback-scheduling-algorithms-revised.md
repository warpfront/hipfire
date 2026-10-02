# Passback Scheduling: Algorithms — with Constraints and Sources
**Lifecycle:** `historical`
**Date:** 2026-10-01 (body rewritten 2026-10-02 per user request)
**Purpose:** Survey scheduling algorithms for hipfire's row-split passback scheduler, filtered against the actual constraints of the system. Methods that don't fit are explicitly called out with reasons; methods that do are grounded in cited sources.

______________________________________________________________________

## 0 · Hipfire's actual problem — what makes it hard

### 0.1 The control variable and process variable

- **Control variable (CO):** `share` = fraction of output rows sent to GPU, in `[0.05, 0.50]` (`SHARE_MIN`/`SHARE_MAX`, `crates/hipfire-dispatch/src/offload_split.rs:89-90`).
- **Process variable (PV):** join time = wall time of the slower arm (GPU or CPU) when both arms run concurrently. The objective is to minimise this.
- **Setpoint:** the theoretical optimum `share* = r_gpu/(r_gpu + r_cpu)` — the balance point where both arms finish simultaneously.

### 0.2 The plant is memoryless, not FOPDT

The share for step *k* is chosen at the top of step *k*, and its effect appears in step *k*'s own join time. There is no transport delay between the control action and the process response: the static map

```
share (CO)  ──▶  join_ns (PV)
```

is evaluated within the same step. The one-step delay in the loop is *feedback measurement* latency, not FOPDT dead time. No time constant is measured anywhere in the investigation tree. Two consequences:

1. Ratio-based reasoning (the APEX inequality, §3.4) is the *natural* formalism here, because it needs no plant model — only the two rates, which are measured directly.
2. Rejections of tuning methods should turn on *operational* cost (forcing oscillation, per-machine step tests), not on an unmeasured `L/T` ratio.

### 0.3 Constraints that rule out part of the control toolbox

| Constraint | Why it matters | Methods ruled out |
|---|---|---|
| **Production path may not be degraded to measure it.** The one-time seeding probe (`measure_both`, `offload_split.rs:584-628`) runs once per shape at first use; there is no further calibration phase and no mechanism to deliberately excite the loop at runtime. | ZN closed-loop and relay ATV *find* their gains by forcing oscillation or a relay cycle. Doing that during serving is a production regression, by construction. | ZN closed-loop, relay ATV |
| **The gain is directly measurable, not just identifiable.** `measure_both` times both engines on the real weight buffer and returns `cpu_bytes_per_s` / `gpu_bytes_per_s`. | A ZN reaction curve would recover, less directly, a quantity the probe already yields. | ZN open-loop, Cohen-Coon (relative to this specific alternative) |
| **Layers never move.** Weights are host-mapped at load and stay there. `pcie`/`cpu`/`passback` decide *who multiplies*, not where data lives. | No PCIe thrashing. The plant is purely computational. | Methods modelling data-movement dynamics as part of the plant |
| **Noisy measurements.** Join times carry timer jitter, PCIe queueing variance, and host-load noise. The controller already hedges this with `JOIN_MARGIN_NS` hysteresis and a per-shape `min_join_ns` floor (`offload_split.rs:398-417`). | Derivative action amplifies noise. Setpoint-weighted P and derivative-on-measurement do not. | Unfiltered derivative-on-error PID |
| **Cross-machine robustness required.** Plant gains vary 5-10× across RX 9070 XT + 7800X3D, MI300X, different PCIe gens and CPUs. | Any method needing per-machine tuning is a deployment burden — though note the rate probe already re-runs per machine. | Methods requiring per-machine tuning *when the gain is otherwise unmeasured* |
| **Per-step decision in \<1 ms.** A new share is chosen every decode step, applied at most once per `APPLY_EVERY = 4` samples. | Must be O(1) per step. No iterative solvers, no matrix inversions. | MRAC, any method requiring online optimisation |

### 0.4 What the problem actually is

Not a tracking control problem (make PV follow a time-varying setpoint). It is a **real-time optimisation** problem: find and maintain the share that minimises join time, where the optimum is static on a given machine, directly measurable by a probe, and slow-drifting with host load.

## 1 · Current scheduler

**Source:** `crates/hipfire-dispatch/src/offload_split.rs:427-435`

```rust
fn next_share(cur: f64, r_cpu: Option<f64>, r_gpu: Option<f64>, gpu_waited: bool) -> f64 {
    match (r_cpu, r_gpu) {
        (Some(cpu), Some(gpu)) if cpu + gpu > 0.0 =>
            clamp_share(cur + ALPHA * (clamp_share(gpu / (gpu + cpu)) - cur)),
        _ if !gpu_waited => clamp_share(cur + 0.02),
        _ => cur,
    }
}
```

with `ALPHA = 0.3`, `APPLY_EVERY = 4`, `EWMA_ALPHA = 0.25`, `FROZEN_APPLIED = 64`, `FREEZE_EPS = 0.005` (`offload_split.rs:99-107`).

**Strengths:** simple; self-seeding (the probe gives a defensible starting share); adapts per shape via `(dtype, k)` keying (`offload_split.rs:252-262`); and the GPU-arm rate is reconstructed **relative to the shape's own no-wait join floor**, with hysteresis, rather than against an absolute threshold or the CPU's own duration (`observe`, `offload_split.rs:450-498`). That is what keeps the estimate honest on a host where the D2H/H2D copies alone dominate, and it must be preserved by anything that replaces the estimator.

### 1.1 There is no steady-state error

`share ← share + 0.3·(target − share)` is an exponential smoother toward the setpoint. For a constant target, error decays as `e_n = 0.7ⁿ·e_0`. From 0.375 toward a target of 0.42:

| adjustments | steps (`×4`) | share |
|---:|---:|---:|
| 0 | 0 | 0.37500 |
| 4 | 16 | 0.40920 |
| 10 | 40 | 0.41873 |
| 20 | 80 | 0.41996 |
| 64 | 256 | 0.42000 (residual ≈6e-12) |

The asymptote is **0.42 exactly**; 0.409 at four adjustments is a mid-trajectory sample, not a steady state.

Integral action is the textbook remedy for offset. There is no offset, so the remedy addresses nothing.

### 1.2 Real weaknesses

1. **Lag, not offset.** The target is an EWMA (`α = 0.25`) of two EWMA rates, and the controller fires at most once per 4 samples (`offload_split.rs:99-107`, `:501-503`). Under host-load drift the share trails the true balance point by roughly one EWMA time constant times the drift rate.
2. **Conditional upward ratchet when the GPU arm is never observed to wait.** If `r_gpu` is never sampled — the CPU is the straggler on every step, so joins sit at the no-wait floor — the fallback nudges the share up by 0.02 per adjustment indefinitely. On a host where the GPU is genuinely faster per byte, this walks to `SHARE_MAX = 0.50` while learning nothing. This branch is conditional on `!gpu_waited` (`offload_split.rs:432`); it is not unconditional. [`perf-gap.md`](./2026-10-01-offload-passback-perf-gap.md) §5 records it as latent and notes it **did not trigger** on the 9B or 27B fixtures — both had joins exceeding the floor and an observable `r_gpu`.
3. **Freeze gate.** `applied ≥ 64` **and** all four trailing deltas `< 0.005` (`offload_split.rs:519-524`) — a convergence gate, not a cap. A converged decode stops being perturbed — the intent — but a shape that freezes while the host is quiet never reopens. `gpu_samples` counts waiting joins only and is *not* a step count ([`perf-gap.md`](./2026-10-01-offload-passback-perf-gap.md) §5; `offload_split.rs:490-498`).

## 2 · Methods that DON'T apply (and why)

### 2.1 Ziegler-Nichols closed-loop

**What it does:** drive `Kp` up until the loop oscillates, record `Ku` and `Pu`, compute gains.

**Why it doesn't fit:** requires forcing the system into sustained oscillation to find `Ku` — deliberately running bad shares on every machine tested, during production. Woolf lists as an explicit disadvantage: "Can venture into unstable regions while testing the P controller, which could cause the system to become out of control" (Woolf §9.3, ZN closed-loop disadvantages). It is also limited to "processes that cannot run in an open-loop environment."

**Verdict:** ❌ Unacceptable for a production inference path.

### 2.2 Ziegler-Nichols open-loop (reaction curve)

**What it does:** step the share, record the join-time response, measure `L`, `T`, `K`, compute gains.

**Why it doesn't fit:** requires a step test per machine. The seeding probe already times both engines on the real weight buffer and returns both rates; `share* = r_gpu/(r_gpu + r_cpu)` follows by arithmetic (`offload_split.rs:622`). A reaction curve estimates the same balance point less directly, per machine, by perturbing a live path. It is *redundant*, not merely tedious. Woolf lists as an explicit disadvantage: "Approximations for the `Kc`, `Ti`, and `Td` values might not be entirely accurate for different systems" (Woolf §9.3, ZN open-loop disadvantages).

**Verdict:** ❌ Redundant with the probe.

### 2.3 Cohen-Coon

**What it does:** same reaction-curve test as ZN open-loop, formulas derived for 1/4 decay ratio.

**Why it doesn't fit:** same redundancy ground as §2.2 — the probe already yields both rates directly. Woolf states "a large process delay is necessary to make this method practical because otherwise unreasonably large controller gains will be predicted" and classifies it an "offline" method where "a step change can be introduced to the input once it is at steady-state" (Woolf §9.3, Cohen-Coon) — a decode loop never provides that steady state. Florisera concurs: "only works for first-order models with a time delay," "can only be done when the system is in steady-state," and "might lead to unstable closed-loop systems" (Florisera §4, Cohen-Coon disadvantages).

**Verdict:** ❌ Rejected on redundancy, not on any measured dead-time ratio.

### 2.4 Relay Auto-Tuning (ATV)

**What it does:** inject a relay into the share, measure the limit cycle, derive `Ku` and `Pu`, apply ZN formulas.

**Why it doesn't fit:** same instability risk as ZN closed-loop — it finds `Ku` by forcing oscillation by design. Noise makes limit-cycle amplitude unreliable: describing-function analysis neglects higher harmonics, giving 5–20% `Ku` errors on typical transfer functions (Yu 1999, via Hornsey §4). Hornsey's own rig found false relay switching from noise "was counteracted through first-order filtering, rather than hysteresis, as the former method was found to give superior results" (Hornsey §6, Notes on experimental methods). Saturation-relay identification is also time-intensive (test duration at least double; −π/2-point tests 30–60 min per point), and Hornsey concludes "its implementation should not be necessary in the majority of cases, as the benefit it gives over the preload relay may often not be sufficient" (Hornsey §7). Woolf adds that ATV "will only work on systems that have significant dead time or the ultimate period, `Pu`, will be equal to the sampling period" (Woolf §9.3, Auto Tune Variation). Florisera carries no relay/ATV section at all — it covers heuristic, ZN closed-loop, ZN open-loop, and Cohen-Coon only.

**Verdict:** ❌ Correct conclusion to reject for a production path.

### 2.5 Model Reference Adaptive Control (MRAC)

**What it does:** adapt controller parameters online via a Lyapunov-based adaptation law so the plant output tracks a reference model.

**Why it doesn't fit:**

- Requires knowing plant order `n` and delay `d`. Hipfire's plant is not a standard DARMA process; it is a ratio of two measured throughputs that varies with shape, dtype, and host load. (Akhtar & Bernstein's discrete-time MRAC assumes a SISO DARMA model with known `n`, `m`, delays, minimum-phase zeros, and coprime polynomials — Assumptions 2.1–2.4.)
- Even a modest second-order model gives a multi-element regressor updated every step — the paper's parameter vector and regressor are both in ℝ²ⁿ⁻¹ (§2, eqs. 2.18–2.19) — for a "parameter" that is a scalar gain ratio.
- Persistent excitation (injecting probing signals) conflicts with running productive inference.
- A Lyapunov-stable discrete-time MRAC is provably stable for its plant class, but proving it for *this* plant requires a formal analysis nobody has done.

**Verdict:** ⚠️ A research project, not a drop-in replacement.

### 2.6 Self-Tuning Regulators (STR)

**What it does:** combine recursive least-squares parameter estimation with pole-placement controller redesign.

**Why it doesn't fit:**

- Least-squares estimates are biased when model errors correlate with the regressor. Join-time measurements are noisy and, once EWMA-smoothed, autocorrelated.
- Requires a parametric plant model hipfire's black-box plant does not supply — and the one parameter that matters is *directly measured* by the probe, so there is nothing for RLS to identify that the probe has not already measured.
- Without persistent excitation, RLS converges to a *wrong* (input-distribution-dependent) estimate rather than the true parameter — biased estimates plus an excitation requirement that conflicts with productive operation.

**Verdict:** ❌ Biased estimates plus an excitation requirement, on a quantity already measured directly.

## 3 · Methods that DO apply

### 3.1 IMC-tuned PI, not untuned incremental PID

**What it does:** Internal Model Control tuning — the standard Lambda/IMC FOPDT PI rule is `Kc = τ/(K·(λ + θ))`, `Ti = τ`, with `λ` a chosen closed-loop time constant, typically 1–3× `τ` (Dataforth AN-124 §Procedure; Rivera–Morari–Skogestad IMC design). The `λ` knob trades response speed against robustness from a single parameter.

**Why IMC specifically:** it is the one classical route whose stated purpose is robustness to plant variation, which is exactly hipfire's cross-machine condition. Dataforth's stated advantages, verified verbatim: "the process variable will not overshoot its set point after a set point change"; "much less sensitive to possible errors made when determining the process dead time"; "the control loop will remain stable even if the process characteristics change substantially from the ones used for tuning" (AN-124, Advantages). Woolf concurs on the contrast: "The Ziegler-Nichols open loop and Cohen-Coon methods give large controller gain and short integral time, which isn't conducive to chemical engineering applications" (Woolf §9.3, Internal Model Control).

**Implementation, if pursued:** incremental (velocity) form for numerical stability, with conditional integration as the anti-windup mechanism. Pick the anti-windup scheme deliberately (back-calculation vs. conditional integration vs. hybrid) rather than by reflex — Caparroza et al. compare back-calculation, conditional integration, and hybrid schemes for FOPDT plants and find commonly used heuristic rules (e.g. tracking time = integral time) suboptimal, developing systematic tuning rules for the tracking time constant instead.

**Expected gain:** **unknown, and plausibly zero.** The gain an integral term would target — steady-state error — does not exist (§1.1). IMC's *actual* contribution here is its robustness margin under host-load drift, which is a smaller and harder-to-size benefit. Sizing it honestly requires the §5 Phase 0 measurement of the drift magnitude. Do not book 1-2 pp.

> **Sourcing note.** A setpoint-filtering formula of the form
> `α = τc(τp + 0.5θp) / (τp(τc + θp))` is sometimes attributed to IMC/setpoint-weighting references. It is **unsupported as cited — do not use it** without an independent source. The underlying *technique* — setpoint weighting and derivative-on-measurement to avoid kick and noise amplification — is standard.

### 3.2 Online plant-gain estimation (ratio tracker)

**What it does:** track the ratio `ρ = r_gpu/r_cpu` directly via an EWMA, and scale the proportional gain inversely: `Kp_eff = Kp_base / max(ρ̂, ε)`.

**Why it might fit:** no step test; adapts across machines automatically; O(1) per step.

**Three problems:**

1. **It would delete the shipped estimator.** The two rates are not redundant — the GPU rate is reconstructed from the *excess of join over the shape's own no-wait floor*, with `JOIN_MARGIN_NS` hysteresis (`observe`, `offload_split.rs:450-498`). Replacing them with a single tracked ratio discards the mechanism that keeps the GPU-arm estimate from collapsing to an order-of-magnitude error when the copies dominate the multiply. That is a regression, not a refactor.
2. **The gain scaling is nearly circular.** `ρ̂` *is* the setpoint (`share* = ρ/(1+ρ)`). Deriving `Kp` from it needs a stated derivation; none is given.
3. **It is drift-lag-prone in the same way the current loop is**, since an EWMA of a ratio inherits the same lag. It does not obviously fix §1.2 weakness 1.

**Risk:** moderate-to-high.

### 3.3 Two-tier host-load tracker (global cap adjustment)

**What it does:** a second EWMA tracks join time across all shapes; when `avg_join_ns` rises, lower `SHARE_MAX` globally.

**Why it fits:**

- Addresses host-load transients, which the per-shape EWMA smooths away.
- Low risk: the cap only tightens.
- ~20 lines.

**Expected gain:** small, and unmeasured. Note this is the only proposal aimed at the *actual* documented defect (§1.2 weakness 1), so it deserves more attention than its size suggests.

### 3.4 Split predicate (APEX-style inequality)

**What it does:** before attempting a split, check whether expected overlap gain exceeds the fixed sync cost (D2H + H2D join); skip sampling where split is intrinsically negative.

**Form borrowed:** APEX derives, for asymmetric pipelining over attention workloads, `N_G/N_C < 2·(T_glinear/T_gatt) + 3 + T_gatt/T_glinear` — verified verbatim as eq. (6) (APEX §3.2).

**What the source actually concludes:** APEX finds this inequality "is rarely satisfied in decode-only scenarios" (APEX §3.2) because measured CPU attention throughput is typically under 10% of GPU's (APEX §2.3: "CPU performance typically under 10% of GPU throughput"), and it therefore uses the inequality mainly to decide **not** to offload. Hipfire's regime is different — it splits weight GEMV rows where the CPU is competitive — so the inequality's conclusion does not transfer, and only its *form* is borrowed. (APEX's own worked threshold: for typical `T_gatt/T_glinear` of 0.5–1.5, `N_G/N_C` must be under ~7.5, i.e. CPU at least ~13% of GPU — APEX §3.2.)

**Open derivation:** a hipfire predicate of the form `(r_gpu + r_cpu)/max(r_gpu, r_cpu) > fixed_cost / min(gpu_time, cpu_time)` is **not derived** from APEX. "The analog is simpler" is an assertion where a derivation is required. The inputs it needs already exist in the trace: [`perf-gap.md`](./2026-10-01-offload-passback-perf-gap.md) §4b gives per-step `d2h` 0.03 / `join` 0.05 ms at `m=12288` and `d2h` 0.03 / `join` 0.03 ms at `m=1024`, and §4c notes the bounds are solo-rate bounds (CPU retention measured 86-94%). Derive the predicate from those, or drop it.

**Risk:** low. A bad predicate skips sampling; it never produces a wrong share.

**Expected gain:** unknown.

## 4 · Comparison matrix

| Algorithm | Lines | Risk | Gap closed | Cross-machine? | Needs step test? |
|---|---|---|---|---|---|
| **Current** EWMA filter (shipped) | — | — | baseline | ✅ (probe re-runs) | No |
| ❌ ZN closed-loop | — | — | — | Poor | Yes (oscillation) |
| ❌ ZN open-loop | — | — | — | Redundant w/ probe | Yes (step) |
| ❌ Cohen-Coon | — | — | — | Redundant w/ probe | Yes (step) |
| ❌ Relay ATV | — | — | — | Same as ZN CL | Yes (relay) |
| ⚠️ MRAC | ~200+ | High (unproven for plant) | Unknown | Best in theory | No |
| ❌ STR/RLS | ~100+ | Medium (biased estimates) | — | Yes (if excited) | Probing signals |
| **⚠️ IMC-tuned PI** | ~60 | Low–moderate | **Unknown; targets a non-existent error** | ✅ (designed for it) | No |
| **⚠️ Ratio tracker** | ~30 | Moderate–high | Unclear | ✅ | No |
| **✅ Two-tier host tracker** | ~20 | Low | Unquantified; targets the real defect | ✅ | No |
| **⚠️ Split predicate** | ~15 | Low | Unquantified | ✅ | No |

Every "gap closed" cell above is **unknown**, and that is the honest state. Filling these with 1-3 pp figures would require measurements that do not exist.

## 5 · Recommended path

**Phase 0 — run the measurement that is already specified.** [`perf-gap.md`](./2026-10-01-offload-passback-perf-gap.md) §7 names the experiment: CPU-step wall time bucketed splittable vs. refused per shape, from the `gpu_offload_probe.txt` trace, with `HIPFIRE_OFFLOAD_EXEC=cpu` giving the split-vs-whole comparison directly. This is the gate. Until it runs:

- [`perf-gap.md`](./2026-10-01-offload-passback-perf-gap.md) §5 reports the scheduler's converged shares at **0.279-0.360** against an independent `offload-bench` expectation of **0.28-0.45** — *"the share is where the probe says it should be."* On this fixture, share placement is **not** the defect.
- §6 tags the coverage term — "32% of steps refuse to split" — as **`[INFERENCE]`, "an assumption, not a measurement,"** noting the refused steps may be small (DeltaNet beta/alpha rows at ~64 KB, far below `MIN_SPLIT_BYTES = 2 MiB`) in which case the term "collapses toward zero." The two branches "differ by most of the shortfall."

**Phase 1 (conditional) — fix lag, not offset.** Only if Phase 0 shows a real drift cost. The target is §1.2 weakness 1: EWMA lag plus `APPLY_EVERY = 4` quantization. Candidate levers, in order of evidence: reduce the target EWMA's lag; add a load feed-forward term; make the freeze gate reopen on host-load change. **Not** an integral term.

**Phase 2 (conditional) — close the ratchet.** Add a downward branch for "GPU arm finished first, and by a lot," so a shape that has never sampled `r_gpu` cannot walk to `SHARE_MAX`. Cheap, and it removes a defect [`perf-gap.md`](./2026-10-01-offload-passback-perf-gap.md) §5 flags as latent but real "on a faster link or a slower CPU."

**Phase 3 (conditional) — coverage, or stop.** If Phase 0 shows refused steps carry disproportionate wall time, coverage is the whole remaining deficit and it needs structure (padding small shapes, or Asynchronous Overlap per APEX), not scheduler tuning. If it shows they are nearly free, coverage is a non-issue and the correct action is to stop.

**Status of projected numbers:** projections such as "4-8 pp of the 9B gap" or fixed coverage fractions are withdrawn. The perf-gap table records 9B realized **+8.5% to +11.7%** at 8/32 and 27B **+18.8%**; the 50/50 coverage split is an untested assumption, not a result.

## 6 · References

Code and in-tree sources (verified against this checkout):

- `crates/hipfire-dispatch/src/offload_split.rs:89-90` — `SHARE_MIN`/`SHARE_MAX`.
- `crates/hipfire-dispatch/src/offload_split.rs:99-107` — `ALPHA`, `APPLY_EVERY`, `EWMA_ALPHA`, `FROZEN_APPLIED`, `FREEZE_EPS`.
- `crates/hipfire-dispatch/src/offload_split.rs:166-178` — `plan_rows` refusal gates (`m < 16`, `MIN_SPLIT_BYTES`).
- `crates/hipfire-dispatch/src/offload_split.rs:398-417` — `JOIN_MARGIN_NS`, `waited_extra_ns`.
- `crates/hipfire-dispatch/src/offload_split.rs:427-435` — `next_share`.
- `crates/hipfire-dispatch/src/offload_split.rs:450-498` — `observe`: floor-relative GPU-rate reconstruction.
- `crates/hipfire-dispatch/src/offload_split.rs:501-524` — apply gating and freeze gate.
- `crates/hipfire-dispatch/src/offload_split.rs:584-628` — `measure_both` seeding probe (solo, sequential, median rates).

External control-theory sources (all quotations below verified verbatim against the primary on 2026-10-02 unless marked `[SECOND-HAND]`):

- Woolf et al., *Chemical Process Dynamics and Controls* (Engineering LibreTexts), §9.3 "PID Tuning via Classical Methods" (<https://eng.libretexts.org/Bookshelves/Industrial_and_Systems_Engineering/Chemical_Process_Dynamics_and_Controls_(Woolf)/09%3A_Proportional-Integral-Derivative_(PID)_Control/9.03%3A_PID_Tuning_via_Classical_Methods>). Verified: ZN closed-loop disadvantage "Can venture into unstable regions while testing the P controller, which could cause the system to become out of control"; ZN open-loop disadvantage "Approximations for the `Kc`, `Ti`, and `Td` values might not be entirely accurate for different systems"; Cohen-Coon "a large process delay is necessary to make this method practical because otherwise unreasonably large controller gains will be predicted" + "Offline method" with step introduced "once it is at steady-state"; ATV "will only work on systems that have significant dead time or the ultimate period, `Pu`, will be equal to the sampling period"; IMC contrast "give large controller gain and short integral time, which isn't conducive to chemical engineering applications".
- Florisera, "PID Tuning Methods" (<https://florisera.com/pid-tuning-methods/>). Verified: FOPDT sensitivity note ("very sensitive to any discrepancies with respect to the assumed process … Deviations to the time delay will greatly degrade the PID performance"); Cohen-Coon disadvantages ("Might lead to unstable closed-loop systems", "Can only be used for first order systems with large process delays", "Can only be done when the system is in steady-state"). **The page carries no relay/ATV section** — heuristic, ZN closed-loop, ZN open-loop, Cohen-Coon only.
- Dataforth AN-124, "Tuning Control Loops with the IMC Tuning Method" (<https://www.dataforth.com/tuning-control-loops-with-imc-tuning-method>). Verified advantages: "process variable will not overshoot its set point after a set point change"; "much less sensitive to possible errors made when determining the process dead time"; "the control loop will remain stable even if the process characteristics change substantially from the ones used for tuning". Procedure: closed-loop time constant chosen 1–3× τ. Theoretical basis: Rivera, Morari & Skogestad, "Internal Model Control. 4. PID Controller Design" (1986).
- Hornsey, "A Review of Relay Auto-tuning Methods for the Tuning of PID-type Controllers," *Reinvention* Vol. 5 Issue 2 (2012) (<https://warwick.ac.uk/fac/cross_fac/iatl/research/reinvention/archive/volume5issue2/hornsey/>). Verified: describing-function harmonic-neglect basis (§3, after Atherton); 5–20% `Ku` errors for typical transfer functions (Yu 1999, via Hornsey §4); false-switching fix "counteracted through first-order filtering, rather than hysteresis, as the former method was found to give superior results" (§6); saturation-relay duration cost and conclusion "its implementation should not be necessary in the majority of cases, as the benefit it gives over the preload relay may often not be sufficient" (§6–7).
- Akhtar & Bernstein, "Lyapunov-Stable Discrete-Time Model Reference Adaptive Control," ACC 2005 (<https://ieeexplore.ieee.org/document/1470460>, preprint <https://dsbaero.engin.umich.edu/wp-content/uploads/sites/441/2019/06/conference276.pdf>). Verified: parameter vector and regressor both in ℝ²ⁿ⁻¹ (§2, eqs. 2.18–2.19); DARMA plant assumptions 2.1–2.4 (known `n`/`m`, minimum-phase, coprime).
- Caparroza, Soltesz, Hägglund & Guzmán, "Anti-Windup in PID Control: Review, Analysis, and New Tuning Directions," arXiv:2606.01959 (2026) (<https://arxiv.org/html/2606.01959v1>). Verified: real paper; compares dynamic/instantaneous back-calculation, conditional integration, and hybrid schemes on FOPDT plants; finds heuristic tracking-time rules suboptimal and derives systematic tuning rules. Note: Hägglund is an author; Åström is not — cite the four authors exactly.
- Control Guru, "Using Signal Filters In Our PID Loop" (<https://controlguru.com/using-signal-filters-in-our-pid-loop/>) and "PID Control and Derivative on Measurement" (<https://controlguru.com/pid-control-and-derivative-on-measurement/>). Verified: "integral action is unaffected by noise because the constant summing of error literally averages the random variations in the signal"; derivative action "cause[s] the noise in the PV measurement to be reflected and amplified in the controller output (CO) signal"; derivative-on-error equals negative-derivative-on-PV except at setpoint steps, where it produces "derivative kick" — hence "derivative on measured PV" is recommended. Hornsey §1 concurs: "in many practical applications the derivative term is placed in the feedback path, hence taking the derivative of the output and resulting in a smoother response."
- Fan, Zhang, Li & Nikolopoulos, "APEX: Asynchronous Parallel CPU-GPU Execution for Online LLM Inference on Constrained GPUs," arXiv:2506.03296 (<https://arxiv.org/html/2506.03296>). Verified: eq. (6) `N_G/N_C < 2·T_glinear/T_gatt + 3 + T_gatt/T_glinear` (§3.2); "rarely satisfied in decode-only scenarios" (§3.2); "CPU performance typically under 10% of GPU throughput" (§2.3); ~7.5 / ~13% worked threshold (§3.2).
- `[SECOND-HAND]` I.D. Landau, "Controls, Adaptive Systems," in *Encyclopedia of Physical Science and Technology* (3rd ed.), 2003 — MRAC background; not checked against the primary.
- `[SECOND-HAND]` Åström & Hägglund, *Advanced PID Control*, ISA, 2006 — incremental-form PID background; not checked against the primary.

### Related records
- [`2026-10-01-offload-passback-perf-gap.md`](./2026-10-01-offload-passback-perf-gap.md) — ceiling vs. realized; §5 (share placement not the defect) and §7 (the measurement that would settle it) govern most of what is claimed here.
