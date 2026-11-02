// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//! Lean persistent submit of the V8 / V9 / V10 / G80 pair designs: the dynamic TXN of [`ArrayDesign`] (`emit_dynamic_body`)
//! minus every reset / lock / descriptor write whose target already holds the right value after a completed submit.
//!
//! # Contract
//! `ArrayDesign::lean_insts` is valid **only** for a submit that directly follows a *completed* submit (its C-token
//! SYNCs returned and the array quiesced) of the same context (same PDI, same design), made with either
//! [`ArrayDesign::insts`] or a previous lean TXN. The first submit after the PDI is loaded MUST use the full
//! `insts` (the PDI initialises no lock, queues no task and enables no core, and the cores / DMA channels are not
//! yet in the post-run state derived below). Both flavours produce byte-identical outputs and leave the array in the
//! same state (every `Config::state_snapshot` field), so any mix of full and lean submits is valid.
//!
//! # Method
//! A full submit initialises the *canonical start state* `S0` (cores reset, channels reset and requeued at their
//! first BD, resident descriptors rewritten, every lock at its initial value, cores enabled). The lean TXN needs
//! `S0` as well, but starts from the end state `S_end` of the previous run. Both runs execute the same
//! deterministic schedule, so `S_end` is derived analytically from per-channel counts of completed BD loads and
//! compared with `S0`; an element is repaired only if `S_end != S0` **and** the difference is not erased by the
//! equivalence below. The prefix option additionally clears its finite iteration fields on every submit, explicitly
//! preserving the prefix descriptor contract even though each complete group wraps them back to zero.
//!
//! ## Equivalence used (parked channel == queued channel)
//! A channel whose ring ends *at its first BD* (`BD index == start`) is parked there: either the descriptor is
//! latched and its lock not yet acquired (MM2S: acquire of a full lock that is never granted), or acquired with the
//! empty lock already decremented and waiting for stream data (S2MM "lookahead" acquire; a BD acquires its lock
//! before data arrives). That is exactly the state a freshly queued channel reaches after its first tick, so the
//! channel, its locks and its descriptor registers are left untouched. A channel parked at a *different* BD, or
//! whose latched descriptor carries a stale iteration `current`, cannot be used by the restarted cores / the
//! canonical schedule: it is reset (CTRL bit1 pulse, drops the engine and queue, NOT the BD registers), its locks
//! are overwritten with their initial values (this also returns a lookahead credit the reset dropped, whether or not
//! the hardware had taken it), the iteration `current` of its BDs is cleared if nonzero, and the task is requeued
//! (+ ENABLE on compute tiles, as the full TXN).
//!
//! ## Counts (per memtile / per core, `J = kc*waves` chunks, `W = waves`)
//! `lookahead(c)` = the load of BD `c mod len` after `c` completed loads; ring BD index `= c mod 2` (rings),
//! `c mod len` (chains).
//!
//! | element | completed loads | parked at | returns to start iff |
//! |---|---|---|---|
//! | core S2MM0 A ring {0,1}, S2MM1 B ring {2,3} (all designs) | `J` each | BD `J mod 2` (+ `A/B_EMPTY[J mod 2]` lookahead) | `J` even |
//! | core MM2S0 C, BD 4 self-cyclic | `W` | BD 4, `C_FULL` unacquired | always |
//! | V8 memtile A/B S2MM {28,29},{4,5}; MM2S {14,15},{30,31} | `J` | BD `J mod 2` | `J` even |
//! | V9 A S2MM5 {28,29}, A MM2S2 {14,15} | `J` | BD `J mod 2` | `J` even |
//! | V9 B fill S2MM4 {4,5} | `NW` fills | BD `4 + NW mod 2`, shared `B_EMPTY` lookahead (`2 -> 1`) | `NW` even |
//! | V9 B replay MM2S1 chain of `MW` BDs | `MW*NW` (each BD `NW`) | first chain BD, `B_FULL = NW-NW = 0` unacquired, every BD's `current = NW mod 2` | `NW` even (else stale latch **and** register) |
//! | V10 non-prefix B fill S2MM4 BD 4 (finite, no `next`) | 1 | idle, queue empty | never (must requeue) |
//! | V10 non-prefix B replay MM2S1 BD 30 (self-cyclic) | `MW` passes | BD 30, `B_FULL = MW-MW = 0` unacquired, no iteration | always |
//! | V10 prefix B fill S2MM4 BD `4+g`, group length `L_g <= 64` | `L_g` per finite task, `T = NW*kc` total | idle, queue empty; each register `current = L_g mod L_g = 0` | never: requeue groups in order, `repeat = L_g`; clear their currents |
//! | V10 prefix guarded B replay MM2S1 BD `30+g` | `L_g` per finite task, `T` total | idle after guarded tasks (or after BD 36); each register `current = 0` | never: requeue groups in order, `repeat = L_g`; clear their currents |
//! | V10 prefix lockfree B replay MM2S1 BD 36 | `MW-1` whole-segment passes | idle, queue empty; no iteration or locks | requeue after guarded groups iff `MW > 1`; omit for `MW = 1` |
//! | V10 A fill S2MM5 {28,29} | `MW` fills | BD `28 + MW mod 2`, shared `A_EMPTY` lookahead (`2 -> 1`) | `MW` even |
//! | V10 A replay MM2S2 chain of `NW` BDs | `NW*MW` (each BD `MW`) | first chain BD, `A_FULL = MW-MW = 0`, every BD's `current = MW mod 2` | `MW` even |
//! | G80 A S2MM5 {28,29}, A MM2S2 {14,15} (**columns 0..4 only**) | `J` each | BD `J mod 2` (+ `A_EMPTY[J mod 2]` lookahead) | `J` even |
//! | G80 B fill S2MM4 BD 4 -> 5 (finite, `repeat = kc`) | `kc` loads of each BD | idle, queue empty; BD 4 / 5 `current = kc mod kc = 0`; ready locks 4 / 6 `kc - kc = 0` | never: requeue `repeat = kc` |
//! | G80 B guard MM2S1 BD 30 / MM2S5 BD 42 (finite, `repeat = kc`, ready lock -1) | `kc` loads | idle, queue empty; `current = 0` | never: requeue `repeat = kc` |
//! | G80 B whole MM2S1 BD 31 / MM2S5 BD 43 (finite, `repeat = mw - 1`, no lock, no iteration) | `mw - 1` passes | idle, queue empty | requeue after its guard iff `MW > 1` (omit for `MW = 1`) |
//! | C rows S2MM {0,1},{24,25},{2,3},{26,27} (V8 / G80 int8: all rows; V9/V10 rows 0,2) | `W` tiles | BD `W mod 2`, `c_lock(W mod 2, r)` lookahead | `W` even |
//! | V9/V10 C rows 1,3 S2MM single BD 24/26, iteration wrap 2 | `W` (+1 lookahead acquire) | same BD; register `current = (W+1) mod 2`, latched `W mod 2`, shared pair lock `2 -> 1` | `W` even (odd: engine latch stale, register already 0) |
//! | C drains MM2S0 {6..9}, MM2S3 {32..35} (V8 / G80 int8: two BDs per wave) | `2W` | BD `2W mod 4`, full locks unacquired | `W` even |
//! | G80 I32 C producers S2MM {0,1},{24,25},{2,3},{26,27} (two-stripe chain, finite, `repeat = W`) | `W` passes | idle, queue empty; `c_lock(0, r)` back at 1 / 0 | never: requeue `repeat = W` |
//! | G80 I32 C drains MM2S0 BD 6..9, MM2S3 BD 32..35 (four-stripe chain, finite, `repeat = W`) | `W` passes | idle, queue empty; full locks unacquired | never: requeue `repeat = W` |
//!
//! Locks at the end: every `*_FULL` and `PEER_*` lock is 0 (each release is matched by exactly one acquire, the
//! pair protocol being symmetric), every `*_EMPTY` lock is initial except the single lookahead credit of each parked
//! S2MM channel, `C_EMPTY = 1`, `C_FULL = 0`. Hence the only *non-equivalent* lock differences arise on channels that
//! are reset, and those are overwritten with the initial value.
//! Prefix locks need no writes: fill releases `B_FULL[0]` exactly `T` times, guarded readers acquire it exactly `T`
//! times (`0 + T - T = 0`). Above 63 chunks only, fill acquires `B_EMPTY[0]` `T` times and guarded readers release it
//! `T` times (`63 - T + T = 63`); below that bound neither uses it. Finite tasks have no next-BD lookahead, so neither
//! prefix channel holds a dropped credit after completion. The last load of each group wraps its register current to 0
//! before that finite task completes; explicit current clears are idempotent, not a repair of a stale latch.
//!
//! ## G80 (`J = kc * mw`, `W = mw`)
//! The G80 memtile map, BDs, locks and finite tasks are those of `gemm_g80` (module documentation, "Completed-submit end
//! state"); the plan below only re-derives which of them differ from the canonical start `S0` after a completed submit.
//! * **A (columns 0..4)** is the V8 ring: `J` odd resets S2MM5 / MM2S2 to BD 28 / 14 and rewrites the pong `A_EMPTY[1]`
//!   (lock 2) to its initial 1. Columns 4..8 own no A channel and no A lock, so their A resets and lock writes are filtered
//!   out at emission (per column), never issued.
//! * **B** is entirely finite: fill BD 4 -> 5 (`repeat = kc`), guard BD 30 / 42 (`repeat = kc`), whole BD 31 / 43
//!   (`repeat = mw - 1`). After a completed submit every one of these channels is idle with an empty queue (no BD is loaded
//!   after the last repeat, so there is no lookahead credit and no parked BD), the iteration `current` of BD 4, 5, 30, 42 has
//!   wrapped to 0 (`kc` loads, `wrap = kc`), and the ready locks 4 / 6 are `kc - kc = 0` (fill releases `kc` times, the guard
//!   acquires `kc` times). Hence **no B channel reset and no B lock write is ever needed**, but every submit MUST requeue, in
//!   this order per channel: fill `repeat = kc`; guard 30 `repeat = kc` then whole 31 `repeat = mw - 1` (MM2S1); guard 42 then
//!   whole 43 (MM2S5). Whole tasks are omitted for `mw = 1`. The current of BD 4, 5, 30, 42 is cleared (mask bits 23..28)
//!   on every submit: idempotent after a completed run, required if a reset ever interrupted a guard / fill mid-iteration.
//!   The finite-queue shape (guard, whole) is the same for every `kc`, `mw` and J / W parity, so the lean state snapshot after
//!   changing operands or odd / even J / W equals the full submit's.
//! * **C (int8)** is the V8 non-shared pair: `W` odd resets S2MM0..3 and the drains MM2S0 / MM2S3 to BD `C_S2MM_BD[r][0]`, 6, 32 and
//!   returns the slot-1 empty lock `c_lock(1, r)` (initial 1). No shared pair C lock exists for G80, rows 1 and 3 included.
//! * **C (`Epilogue::I32`)** is finite like B: per row the producer S2MM `r` runs the two-stripe chain `C_S2MM_BD[r][0] -> [1]`
//!   and each drain MM2S0 / MM2S3 the four-stripe chain `6 -> 7 -> 8 -> 9` / `32 -> .. -> 35`, every task `repeat = W`. Each ends
//!   idle with an empty queue (no BD is loaded after the last repeat, so no parked BD and no lookahead credit), the single-slot
//!   `c_lock(0, r)` empty / full pairs are back at 1 / 0 for **every** wave parity, and the BDs carry no iteration. Hence no C
//!   reset, no C lock write and no parity repair at all, but all six C tasks per column are requeued on every submit with
//!   `repeat = W` (`W <= 256`); a forced MM2S0 / MM2S3 reset keeps that full-repeat requeue instead of one repeat of BD 6 / 32.
//! * **Cores**: as every design (`J` odd resets A / B rings and returns `A_EMPTY[1]`, `B_EMPTY[1]`); the core C ring is parked
//!   at BD 4 for any `W` (`Epilogue::I32`: the two-BD chain 4 -> 5 -> 4 completes `2W` loads and is parked at BD 4 with
//!   `C_FULL` unacquired as well); `C_EMPTY`, acquired before chunk 0 of each wave, is released back by the drain (I32: by BD 5,
//!   after both stripes), so it is 1 at the end and needs no write.
//!
//! ## IEF15 (`J = epochs * waves` via `geo.kc = epochs`, `W = waves`)
//! The IEF15 memtile map, BD ids, locks and tasks are V8's streaming pair (`Topo` kind `Ief15` takes the V8 arms: no
//! iteration, no finite tasks), only the ring BD lengths differ, so the V8 ring parity plan applies unchanged: `J` odd
//! resets the memtile A / B rings and the core A / B rings and returns the pong `A_EMPTY` / `B_EMPTY` credits, `W` odd
//! repairs the C rings and drains.
//!
//! Prefix verification in `crates/peacemaker-npu/tests/lean.rs` compares three changing-operand lean submits and full/lean
//! interleaves to both the CPU reference and a fresh full-TXN configuration, including every retained snapshot field.
//! It covers one/two groups and `MW = 1`/`MW > 1`, forced channel resets, and missing finite-task requeues.
//! For Fast int8 (`shift = 12`), V10 prefix down `4096x2560x640` is 8,816 B lean versus 49,136 B full;
//! the padded `160x2560x640` design is 15,824 B lean versus 48,944 B full.
//!
//! ## Why the cores always restart
//! A finished core is `DONE` (status bit, PC past the program); only a `CORE_CONTROL` reset clears it, and the core
//! program restarts its ping/pong phase `r4 = 0`. So every core gets halt+reset, release + `CORE_PC = 0` (same
//! sequence as the full TXN), and the enable last.
//!
//! ## Why shim and SYNC are always emitted
//! Shim BDs carry DDR-patched host addresses (the host may use new buffers per submit), the token controller id and
//! the finite tasks are consumed per run; they are emitted exactly as [`emit_shim`] does. SYNC is
//! [`emit_sync`]`(pair, per_column = false)`.
//!
//! ## Op order
//! 1. halt+reset of all 32 cores; 2. CTRL RESET assert then release on exactly the repaired channels;
//! 3. BD word 6 `current := 0` of exactly the iteration BDs listed above (mask bits 23..28);
//! 4. lock writes of exactly the dropped lookahead credits; 5. core reset release + `CORE_PC`;
//! 6. requeue (+ ENABLE on compute tiles) of exactly the reset channels, plus V10's completed finite B tasks (prefix:
//! fill groups then guarded groups then optional BD 36, using the full submit's task repeats);
//! 7. shim; 8. enable cores; 9. SYNC. A non-prefix design whose parities are all even therefore needs only 4 core ops per
//! core (+ V10's single fill requeue per column), the shim and the SYNC.
//!
//! The quiescence assumption (lookahead credits already taken at the C token) holds with a margin of thousands of
//! cycles: the credit of a channel is taken as soon as the consumer of its previous use released, which is before
//! the last C tile leaves the memtile; and every repair writes an absolute initial value, so a hardware that does not
//! take the lookahead is repaired equally.
//!
//! # Forced channel repair (`lean_insts_requeue`, `append_lean_run_body`)
//! The contract above also assumes nothing but the design's own run touched the array since the completed submit. The
//! persistent ring (`ring.rs`) inserts a POLL (memtile col 0 S2MM4 reset, then a self-looping 1-word task that stays
//! running) and a DONE (col 0 MM2S0 reset, one finite 16-word task) between two bodies, so S2MM4 and MM2S0 are no
//! longer "parked at their first BD" even when the shape's parity says so. The caller names such channels as
//! `(column, direction, channel)` and each one gets the reset path of the plan, regardless of parity:
//!
//! | forced channel | reset + requeue BD | credit lock rewritten to its initial value |
//! |---|---|---|
//! | memtile S2MM4 (V8 / V9 B fill) | BD 4 | `B_EMPTY[0]`: the BD-4 lookahead credit of the even-parity park (V9: shared lock, initial 2; V8: initial 1) |
//! | memtile S2MM4 (V10 finite fill) | BD 4 | none (no acquire); the completed requeue is promoted to reset + requeue |
//! | memtile S2MM4 (V10 prefix finite fill) | reset once, keep every BD `4+g` task in group order with `repeat = L_g` | none: completed finite tasks hold no lookahead credit, including above 63 chunks |
//! | memtile S2MM4 (G80 finite fill) | reset once; the plan's fill task `repeat = kc` is kept (not replaced by one repeat) | none: no acquire, ready locks 4 / 6 already 0 |
//! | memtile MM2S0 / MM2S3 (C drains) | BD 6 / BD 32 | none: FULL locks are acquired only on a grant, so a reset parks nothing |
//!
//! Equivalence: the lock write is absolute, so it restores the credit whether or not the reset dropped one (the same
//! argument as for odd-parity repairs, which use the lock of BD 5 because they park there); the BD registers of BD 4 / 6
//! are never rewritten by POLL / DONE (they use other BDs), so the requeue loads the unchanged canonical descriptor;
//! the reset also kills a still running self-loop task. A channel the plan already resets at this shape keeps the
//! plan's repair (parity lock, `current` clears): the forced entry is skipped, and a credit lock the plan already
//! rewrites is not written twice. Forced ops join steps 2 (reset), 4 (lock) and 6 (requeue) of the op order and add,
//! per forced channel, 56 B (reset + release) + 24 B (requeue, absent for V10 finite fills already requeued) + 24 B
//! (credit lock, V8/V9 S2MM4 only). Unsupported channels or columns return Err before any op is appended.
//! `lean_insts()` is `lean_insts_requeue(&[])`.
use super::*;

/// Parity facts that decide which elements are repaired.
#[derive(Clone, Copy, Debug)]
struct Parity {
    /// `kc * waves` odd: core (all designs) / V8 / V9-A rings end on their second BD.
    chunks_odd: bool,
    /// `waves` odd: C rings, single-BD C iteration and C drain chains end off-phase.
    waves_odd: bool,
}

/// One memtile channel reset + requeue.
#[derive(Clone, Copy, Debug)]
struct MemChannel {
    direction: dma::Direction,
    channel: u32,
    bd: u32,
}

/// Per-column memtile repairs (identical for every column).
#[derive(Default, Debug)]
struct MemPlan {
    /// Channels reset, then requeued at `bd`.
    resets: Vec<MemChannel>,
    /// Finite tasks requeued with their canonical order and repeat counts.
    requeues: Vec<Task>,
    /// Memtile lock ids overwritten with their initial value.
    locks: Vec<u32>,
    /// Memtile BD ids whose iteration `current` is cleared.
    currents: Vec<u32>,
}

impl MemPlan {
    fn reset(&mut self, direction: dma::Direction, channel: u32, bd: u32) {
        self.resets.push(MemChannel {
            direction,
            channel,
            bd,
        });
    }
    fn lock(&mut self, id: u32) {
        if !self.locks.contains(&id) {
            self.locks.push(id);
        }
    }
}

/// Everything the lean TXN repairs.
#[derive(Debug)]
struct LeanPlan {
    /// Per core: reset + requeue A (S2MM0) and B (S2MM1) rings and return their dropped lookahead credits.
    core_ab: bool,
    mem: MemPlan,
}

impl ArrayDesign {
    fn lean_plan(&self) -> Result<LeanPlan, String> {
        let geo = &self.geo;
        let topo = geo.topo;
        if !matches!(
            self.variant,
            Variant::V8 | Variant::V9 | Variant::V10 | Variant::G80 | Variant::Ief15
        ) || !topo.is_pair()
        {
            return Err(format!(
                "lean submit is derived for the V8/V9/V10/G80/IEF15 pair designs only, not {}",
                self.variant
            ));
        }
        let parity = Parity {
            chunks_odd: (geo.kc * geo.waves) % 2 == 1,
            waves_odd: geo.waves % 2 == 1,
        };
        let ((a_empty, _), (b_empty, _)) = ab_locks();
        let mut mem = MemPlan::default();
        match topo.resident() {
            // G80: A ring (columns 0..4 only, filtered per column at emission), finite B fill / guard / whole tasks that are
            // requeued on every submit with their canonical repeats, and the iteration `current` of the BDs that carry one.
            None if topo.is_g80() => {
                if parity.chunks_odd {
                    mem.reset(S2mm, A_S2MM_CH, A_S2MM_BD[0]);
                    mem.reset(Mm2s, A_MM2S_CH, A_MM2S_BD[0]);
                    mem.lock(a_empty[1]);
                }
                let task = |direction, channel, bd, repeat| Task {
                    direction,
                    channel,
                    bd,
                    repeat,
                    issue_token: false,
                };
                if topo.is_g80_n512() {
                    // N512: the B fill / guard groups (BD 4, 5 / 30, 31, `repeat = group length`) and the whole-panel BD 36.
                    let (requeues, currents) = gemm_g80_n512::lean_b_tasks(geo.kc, geo.waves);
                    mem.requeues.extend(requeues);
                    mem.currents.extend(currents);
                } else {
                    let (kc, whole) = (geo.kc as u32, (geo.waves - 1) as u32);
                    mem.requeues.push(task(S2mm, B_S2MM_CH, B_S2MM_BD[0], kc));
                    for p in 0..2 {
                        mem.requeues.push(task(
                            Mm2s,
                            gemm_g80::G80_B_MM2S_CH[p],
                            gemm_g80::G80_B_GUARD_BD[p],
                            kc,
                        ));
                    }
                    if whole > 0 {
                        for p in 0..2 {
                            mem.requeues.push(task(
                                Mm2s,
                                gemm_g80::G80_B_MM2S_CH[p],
                                gemm_g80::G80_B_WHOLE_BD[p],
                                whole,
                            ));
                        }
                    }
                    mem.currents.extend([
                        B_S2MM_BD[0],
                        B_S2MM_BD[1],
                        gemm_g80::G80_B_GUARD_BD[0],
                        gemm_g80::G80_B_GUARD_BD[1],
                    ]);
                }
                // G80 I32: the C producers (two-stripe chain per row) and the C drains (four-stripe chain per drain channel) are
                // finite and run `mw` passes, like the B tasks: after a completed submit they are idle with empty queues and
                // every C lock is back at its initial value (no lookahead, single slot), for every wave parity. Requeue them.
                if topo.is_g80_i32() || topo.is_g80_bf16() {
                    for r in 0..ROWS {
                        mem.requeues
                            .push(task(S2mm, r as u32, C_S2MM_BD[r][0], geo.waves as u32));
                    }
                    for g in 0..2 {
                        mem.requeues.push(task(
                            Mm2s,
                            PAIR_C_MM2S_CH[g],
                            PAIR_C_MM2S_BD0[g],
                            geo.waves as u32,
                        ));
                    }
                }
            }
            None => {
                if parity.chunks_odd {
                    mem.reset(S2mm, B_S2MM_CH, B_S2MM_BD[0]);
                    mem.reset(S2mm, A_S2MM_CH, A_S2MM_BD[0]);
                    mem.reset(Mm2s, B_MM2S_CH, B_MM2S_BD[0]);
                    mem.reset(Mm2s, A_MM2S_CH, A_MM2S_BD[0]);
                    mem.lock(b_empty[1]);
                    mem.lock(a_empty[1]);
                }
            }
            Some(res) => match res.mode {
                ResidentMode::NwOuter => {
                    if parity.chunks_odd {
                        mem.reset(S2mm, A_S2MM_CH, A_S2MM_BD[0]);
                        mem.reset(Mm2s, A_MM2S_CH, A_MM2S_BD[0]);
                        mem.lock(a_empty[1]);
                    }
                    if res.nw % 2 == 1 {
                        mem.reset(S2mm, B_S2MM_CH, B_S2MM_BD[0]);
                        mem.lock(b_empty[0]);
                        mem.reset(Mm2s, B_MM2S_CH, R_B_REPLAY_BD[0]);
                        mem.currents.extend_from_slice(&R_B_REPLAY_BD[..res.mw]);
                    }
                }
                ResidentMode::MwOuter => {
                    if res.b_prefix {
                        // Same finite B tasks as ring_tasks, without building unrelated core / A / C tasks.
                        let task = |direction, channel, bd, repeat| Task {
                            direction,
                            channel,
                            bd,
                            repeat,
                            issue_token: false,
                        };
                        for (g, (_, len)) in res.b_prefix_groups().enumerate() {
                            mem.requeues
                                .push(task(S2mm, B_S2MM_CH, B_S2MM_BD[g], len as u32));
                            mem.currents.extend([B_S2MM_BD[g], B_MM2S_BD[g]]);
                        }
                        for (g, (_, len)) in res.b_prefix_groups().enumerate() {
                            mem.requeues
                                .push(task(Mm2s, B_MM2S_CH, B_MM2S_BD[g], len as u32));
                        }
                        if res.mw > 1 {
                            mem.requeues.push(task(
                                Mm2s,
                                B_MM2S_CH,
                                V10_B_LOCKFREE_BD,
                                (res.mw - 1) as u32,
                            ));
                        }
                    } else {
                        mem.requeues.push(Task {
                            direction: S2mm,
                            channel: B_S2MM_CH,
                            bd: B_S2MM_BD[0],
                            repeat: 1,
                            issue_token: false,
                        });
                    }
                    if res.mw % 2 == 1 {
                        mem.reset(S2mm, A_S2MM_CH, A_S2MM_BD[0]);
                        mem.lock(a_empty[0]);
                        mem.reset(Mm2s, A_MM2S_CH, V10_A_CHAIN_BD0);
                        mem.currents
                            .extend((0..res.nw as u32).map(|i| V10_A_CHAIN_BD0 + i));
                    }
                }
            },
        }
        if parity.waves_odd && !topo.is_g80_i32() && !topo.is_g80_bf16() {
            for r in 0..ROWS {
                mem.reset(S2mm, r as u32, C_S2MM_BD[r][0]);
            }
            for g in 0..2 {
                mem.reset(Mm2s, PAIR_C_MM2S_CH[g], PAIR_C_MM2S_BD0[g]);
            }
            for r in 0..ROWS {
                // Rings (V8 all rows, V9/V10 rows 0,2): the dropped lookahead credit is the slot-1 empty lock.
                // V9/V10 rows 1,3 share one pair lock between both slots (credit returned to slot-0 empty).
                let shared_pair = topo.resident().is_some() && r % 2 == 1;
                mem.lock(c_lock(if shared_pair { 0 } else { 1 }, r));
            }
        }
        Ok(LeanPlan {
            core_ab: parity.chunks_odd,
            mem,
        })
    }

    /// Memtile channel `(direction, channel)` of a column forced through reset + requeue although the plan of the design
    /// does not repair it at this parity (see "Forced channel repair" in the module documentation): the first BD of the
    /// channel and the lock that held the dropped lookahead credit while the channel was parked at that BD.
    fn forced_channel(
        &self,
        direction: dma::Direction,
        channel: u32,
    ) -> Result<(u32, Option<u32>), String> {
        let topo = self.geo.topo;
        if !matches!(
            self.variant,
            Variant::V8 | Variant::V9 | Variant::V10 | Variant::G80 | Variant::Ief15
        ) || !topo.is_pair()
        {
            return Err(format!(
                "lean submit is derived for the V8/V9/V10/G80/IEF15 pair designs only, not {}",
                self.variant
            ));
        }
        let (_, (b_empty, _)) = ab_locks();
        match (direction, channel) {
            // B fill ring {4, 5}: an EVEN-parity run parks it at BD 4 holding the `B_EMPTY[0]` lookahead credit (V8 ring
            // of two locks, V9 one shared lock). Both V10 finite fill forms finish without a lookahead credit.
            (S2mm, B_S2MM_CH) => Ok((B_S2MM_BD[0], if topo.mw_outer() || topo.is_g80() { None } else { Some(b_empty[0]) })),
            // C drain chains: acquire of a FULL lock that is never granted when parked; no credit, no iteration.
            (Mm2s, ch) if PAIR_C_MM2S_CH.contains(&ch) => {
                let g = PAIR_C_MM2S_CH.iter().position(|&c| c == ch).unwrap();
                Ok((PAIR_C_MM2S_BD0[g], None))
            }
            _ => Err(format!("forced lean repair is derived for memtile S2MM{B_S2MM_CH} and the C drain MM2S channels \
                {PAIR_C_MM2S_CH:?} only, not {direction:?} channel {channel}")),
        }
    }

    /// Lean persistent TXN: see the module documentation (`gemm_array_lean.rs`). Valid only directly after a
    /// completed submit of this design (full or lean); the first submit after the PDI must use [`ArrayDesign::insts`].
    /// Err for any design other than V8 / V9 / V10 / G80.
    pub fn lean_insts(&self) -> Result<Vec<u8>, String> {
        self.lean_insts_requeue(&[])
    }

    /// [`ArrayDesign::lean_insts`] plus a forced reset + requeue of each `(column, direction, memtile channel)` of
    /// `extra` (module documentation, "Forced channel repair"). Use it when something other than the design's own run
    /// has clobbered those channels since the last completed run (the persistent ring's poll / done tasks). Channels the
    /// plan already repairs at this shape are not emitted twice. Supported channels: memtile B fill S2MM4 and the C
    /// drains MM2S0 / MM2S3; any other channel, or a column outside the topology, is an Err.
    pub fn lean_insts_requeue(
        &self,
        extra: &[(u32, dma::Direction, u32)],
    ) -> Result<Vec<u8>, String> {
        let mut txn = Txn::aie2p_8col();
        self.append_lean_run_body(&mut txn, [0; 3], extra)?;
        Ok(txn.to_bytes())
    }

    /// Append one lean body (ends with the C token SYNCs) to `txn`, every shim DDR patch advanced by `arena_base[arg]`
    /// (args 0..=2) as [`ArrayDesign::append_run_body`] does for the full body. `extra` as in
    /// [`ArrayDesign::lean_insts_requeue`]. On Err `txn` is unchanged.
    pub fn append_lean_run_body(
        &self,
        txn: &mut Txn,
        arena_base: [u64; 3],
        extra: &[(u32, dma::Direction, u32)],
    ) -> Result<(), String> {
        self.lean_body(txn, arena_base, extra, true, false)
    }

    /// [`ArrayDesign::append_lean_run_body`] that does not rewrite the shim BD words / S2MM controller ids (only DDR
    /// patches and tasks). Valid ONLY when the previous body executed on this hardware context was a body of this SAME
    /// design (not merely the same PDI: e.g. [`ArrayDesign::with_a_repeat`] shares the PDI but not the shim BDs).
    pub fn append_lean_run_body_same_design(
        &self,
        txn: &mut Txn,
        arena_base: [u64; 3],
        extra: &[(u32, dma::Direction, u32)],
    ) -> Result<(), String> {
        self.lean_body(txn, arena_base, extra, false, false)
    }

    /// [`ArrayDesign::append_lean_run_body_same_design`] that keeps B resident (G80 only, whose whole-K B stays in the
    /// memtile for a submit): no shim B stream, no memtile B fill; the B ready locks are set to the `kc` credits the
    /// fill would have released. Valid ONLY when the previous body on this hardware context was a body of this same
    /// design whose B argument held the same bytes (the memtile B regions still hold them); `arena_base[1]` is unused.
    pub fn append_lean_run_body_b_resident(
        &self,
        txn: &mut Txn,
        arena_base: [u64; 3],
        extra: &[(u32, dma::Direction, u32)],
    ) -> Result<(), String> {
        if self.variant != Variant::G80 || self.geo.topo.is_g80_n512() {
            return Err(format!(
                "B-resident lean body is derived for G80 only, not {}",
                self.variant
            ));
        }
        self.lean_body(txn, arena_base, extra, false, true)
    }

    fn lean_body(
        &self,
        txn: &mut Txn,
        arena_base: [u64; 3],
        extra: &[(u32, dma::Direction, u32)],
        shim_bd_words: bool,
        b_resident: bool,
    ) -> Result<(), String> {
        let mut plan = self.lean_plan()?;
        // B resident: the fill task is not requeued (nothing streams B in); the guards get the fill's credits instead.
        if b_resident {
            plan.mem
                .requeues
                .retain(|t| !(t.direction == S2mm && t.channel == B_S2MM_CH));
        }
        let geo = &self.geo;
        let topo = geo.topo;
        let cores: Vec<Location> = topo.cores().collect();
        let core_task = |direction, channel, bd| Task {
            direction,
            channel,
            bd,
            repeat: 1,
            issue_token: false,
        };
        let core_chans = [
            core_task(S2mm, gemm_core::A_CHANNEL, gemm_core::A_BD[0]),
            core_task(S2mm, gemm_core::B_CHANNEL, gemm_core::B_BD[0]),
        ];
        let mem_task = |c: &MemChannel| Task {
            direction: c.direction,
            channel: c.channel,
            bd: c.bd,
            repeat: 1,
            issue_token: false,
        };
        // G80 injects A in columns 0..4 only: the A channels / locks of the other columns do not exist and are never repaired.
        let ((a_empty, a_full), _) = ab_locks();
        let skip_chan = |col: u32, t: &Task| {
            !topo.has_a(col)
                && ((t.direction == S2mm && t.channel == A_S2MM_CH)
                    || (t.direction == Mm2s && t.channel == A_MM2S_CH))
        };
        let skip_lock =
            |col: u32, id: u32| !topo.has_a(col) && (a_empty.contains(&id) || a_full.contains(&id));
        let mut mem_resets: Vec<(Location, Task)> = Vec::new();
        let mut mem_requeues: Vec<(Location, Task)> = Vec::new();
        for col in 0..topo.cols as u32 {
            for c in &plan.mem.resets {
                let t = mem_task(c);
                if !skip_chan(col, &t) {
                    mem_resets.push((mem_loc(col), t));
                }
            }
            for &t in &plan.mem.requeues {
                mem_requeues.push((mem_loc(col), t));
            }
        }
        // Forced channels, validated before the first op is appended. A channel the plan already resets (same column,
        // direction, channel) keeps its parity-derived repair and is skipped. Finite tasks retain ALL their repeats and
        // queue order; the forced reset must not replace them with one repeat of the first BD.
        let mut extra_locks: Vec<(u32, u32)> = Vec::new();
        let mut extra_resets: Vec<(Location, Task)> = Vec::new();
        for &(col, direction, channel) in extra {
            if col as usize >= topo.cols {
                return Err(format!(
                    "forced lean repair: column {col} outside the {} active columns",
                    topo.cols
                ));
            }
            let (bd, lock) = self.forced_channel(direction, channel)?;
            let loc = mem_loc(col);
            let same = |&(l, t): &(Location, Task)| {
                l == loc && t.direction == direction && t.channel == channel
            };
            if mem_resets.iter().chain(&extra_resets).any(same) {
                continue;
            }
            // Non-prefix V10 has one repeat of BD 4: promote it as before, preserving the existing TXN order.
            // Prefix tasks must remain intact instead of being replaced by the single reset-channel task.
            if !topo.is_g80() && !topo.resident().is_some_and(|r| r.b_prefix) {
                mem_requeues.retain(|e| !same(e));
            }
            extra_resets.push((
                loc,
                Task {
                    direction,
                    channel,
                    bd,
                    repeat: 1,
                    issue_token: false,
                },
            ));
            if let Some(id) = lock {
                if !plan.mem.locks.contains(&id) && !extra_locks.contains(&(col, id)) {
                    extra_locks.push((col, id));
                }
            }
        }
        mem_resets.extend(extra_resets);

        // 1. halt + reset every core (clears DONE / status; the program restarts its phase at 0).
        for &tile in &cores {
            txn.mask_write(tile.address(regs::core::CORE_CONTROL), 2, 3);
        }
        // 2. channel reset, only the channels whose parked BD / latched descriptor is not the canonical start.
        let mut resets: Vec<(Location, Task)> = Vec::new();
        if plan.core_ab {
            for &tile in &cores {
                for t in core_chans {
                    resets.push((tile, t));
                }
            }
        }
        resets.extend(mem_resets.iter().copied());
        for &(loc, t) in &resets {
            txn.mask_write(channel_ctrl(loc, t.direction, t.channel), 2, 2);
        }
        for &(loc, t) in &resets {
            txn.mask_write(channel_ctrl(loc, t.direction, t.channel), 0, 2);
        }
        // 3. iteration `current := 0` (memtile BD word 6 bits 23..28); includes every prefix group on every submit.
        for col in 0..topo.cols as u32 {
            for &bd in &plan.mem.currents {
                txn.mask_write(Bd::address(mem_loc(col), bd) + 6 * 4, 0, 63 << 23);
            }
        }
        // 4. dropped lookahead credits: absolute initial values.
        if plan.core_ab {
            let init = core_config(topo).1;
            for &tile in &cores {
                for id in [gemm_core::A_EMPTY[1], gemm_core::B_EMPTY[1]] {
                    let (addr, value) = dma::lock_write(tile, id, init[id as usize] as u32);
                    txn.write32(addr, value);
                }
            }
        }
        let mut initial_mem_locks = [0u32; 64];
        if !plan.mem.locks.is_empty() || !extra_locks.is_empty() {
            for (id, value) in memtile_locks(topo, 0) {
                initial_mem_locks[id as usize] = value;
            }
        }
        for col in 0..topo.cols as u32 {
            for &id in &plan.mem.locks {
                if skip_lock(col, id) {
                    continue;
                }
                let (addr, value) =
                    dma::lock_write(mem_loc(col), id, initial_mem_locks[id as usize]);
                txn.write32(addr, value);
            }
        }
        for &(col, id) in &extra_locks {
            let (addr, value) = dma::lock_write(mem_loc(col), id, initial_mem_locks[id as usize]);
            txn.write32(addr, value);
        }
        if b_resident {
            for col in 0..topo.cols as u32 {
                for id in gemm_g80::B_READY {
                    let (addr, value) = dma::lock_write(mem_loc(col), id, geo.kc as u32);
                    txn.write32(addr, value);
                }
            }
        }
        // 5. release core reset, PC = 0.
        for &tile in &cores {
            txn.mask_write(tile.address(regs::core::CORE_CONTROL), 0, 2);
            txn.write32(tile.address(regs::core::CORE_PC), 0);
        }
        // 6. requeue the reset channels (+ ENABLE on compute tiles) and the completed finite memtile tasks.
        if plan.core_ab {
            for &tile in &cores {
                for t in core_chans {
                    t.emit_txn(tile, txn);
                    t.enable_txn(tile, txn);
                }
            }
        }
        for &(loc, t) in &mem_resets {
            // B resident: a reset B fill channel stays idle (requeueing it would park a fill waiting for data).
            if b_resident && t.direction == S2mm && t.channel == B_S2MM_CH {
                continue;
            }
            if !mem_requeues
                .iter()
                .any(|&(l, q)| l == loc && q.direction == t.direction && q.channel == t.channel)
            {
                t.emit_txn(loc, txn);
            }
        }
        for &(loc, t) in &mem_requeues {
            t.emit_txn(loc, txn);
        }
        // 7. shim BDs / DDR patches / tokens / tasks.
        emit_shim_at(txn, geo, arena_base, shim_bd_words, b_resident);
        // 8. cores last, then wait for the C tokens.
        for &tile in &cores {
            txn.mask_write(tile.address(regs::core::CORE_CONTROL), 1, 1);
        }
        emit_output_tail(txn, geo, false, arena_base[2]);
        Ok(())
    }
}
