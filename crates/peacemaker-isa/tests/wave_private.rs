// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Wave-private LDS (`peacemaker_author::WaveRegion`, `Driver::lds_private`)
//! on the production backend: the builder's slot machine, an emitted
//! multi-wave kernel, the object-level ownership proof
//! (`pm_check::lds_wave_private`) and the emulator's gate model.
//!
//! The kernel (block 256, eight wave32 waves; `*_demo`): every odd wave, under
//! wave-uniform control, stores `3 * tid + 7` to its own slice of a private
//! window, publishes it, reads its lane-neighbour's slot, retires, adds 100,
//! stores that back, publishes, reads its own slot and retires; every wave
//! then stores its result to a shared tile, publishes the tile with a
//! workgroup barrier and reads the result of the thread 32 away (another
//! wave); the value goes to `out[tid]`. No wait and no barrier sits between a
//! private store and the load that follows it: one wave's DS instructions
//! execute in issue order (RDNA3 ISA 12.5 p.125, RDNA3.5 ISA 12.5 p.127, RDNA4
//! ISA 12.5 and 5.7.1.4).
use peacemaker_author::{ready, Gfx1100, Gfx1151, Gfx1201, Target, WaveRegion, Workgroup};
use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_isa::insn::{Instruction, MemoryClass};
use peacemaker_isa::lds::{private_windows, Lds, PrivateWindow, SlotState, Transition};
#[cfg(feature = "toolchain")]
use peacemaker_isa::pm_check::{lds_bounds, lds_wave_private};
use peacemaker_isa::reg::{Live, RegRef};
use peacemaker_isa::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan, V};

#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;

fn v(n: u8) -> RegRef {
    V::<1>(n).reg()
}
fn sr(n: u8, len: u8) -> RegRef {
    RegRef {
        kind: peacemaker_isa::reg::Kind::S,
        base: n,
        len,
    }
}

// ---- the slot machine ---------------------------------------------------------

fn err<T>(r: Result<T, String>) -> String {
    match r {
        Ok(_) => panic!("expected a refusal"),
        Err(e) => e,
    }
}

#[test]
fn private_slot_layout_is_validated_and_never_overlaps_another_slot() {
    let mut l = Lds::default();
    assert!(err(l.add_wave_private("p", 0, 0, 4)).contains("nonzero"));
    assert!(err(l.add_wave_private("p", 0, 64, 0)).contains("nonzero"));
    assert!(err(l.add_wave_private("p", 8, 64, 4)).contains("multiples of 16"));
    assert!(err(l.add_wave_private("p", 0, 40, 4)).contains("multiples of 16"));
    assert!(err(l.add_wave_private("p", 0, 1 << 30, 8)).contains("overflows"));
    let shared = l.add("shared", 1024, 512).unwrap();
    // Neither a window over a shared slot nor a shared slot over a window.
    assert!(err(l.add_wave_private("p", 512, 512, 2)).contains("overlapping"));
    let p = l.add_wave_private("p", 0, 256, 4).unwrap();
    assert!(err(l.add("alias", 768, 16)).contains("overlapping"));
    assert!(err(l.add_wave_private("q", 1024, 16, 1)).contains("overlapping"));
    let w = l.slots[p].wave_private.unwrap();
    assert_eq!((w.stride, w.waves, l.slots[p].len), (256, 4, 1024));
    assert!(l.slots[shared].wave_private.is_none());
}

#[test]
fn private_states_step_by_wave_local_transitions_only() {
    let mut l = Lds::default();
    let p = l.add_wave_private("p", 0, 256, 4).unwrap();
    // The protocol: Free -> Publishing -> Published -> Reading -> Free.
    assert!(err(l.private_load(p)).contains("not published to its wave"));
    assert!(err(l.private_publish(p)).contains("needs a store first"));
    assert!(err(l.private_retire(p)).contains("nothing was published"));
    l.private_store(p).unwrap();
    l.private_store(p).unwrap();
    assert_eq!(l.slots[p].state, SlotState::Publishing);
    assert!(err(l.private_load(p)).contains("not published"));
    assert!(err(l.private_retire(p)).contains("nothing was published"));
    l.private_publish(p).unwrap();
    assert!(err(l.private_store(p)).contains("cannot be stored while Published"));
    l.private_load(p).unwrap();
    l.private_load(p).unwrap();
    assert_eq!(l.slots[p].state, SlotState::Reading);
    l.private_retire(p).unwrap();
    assert_eq!(l.slots[p].state, SlotState::Free);
    // A published slice that was never read retires too.
    l.private_store(p).unwrap();
    l.private_publish(p).unwrap();
    l.private_retire(p).unwrap();
    l.relayout().unwrap();
}

#[test]
fn no_cross_wave_path_reaches_a_private_slot() {
    let mut l = Lds::default();
    let shared = l.add("shared", 4096, 256).unwrap();
    let p = l.add_wave_private("p", 0, 256, 4).unwrap();
    // A barrier transition would publish or retire it for every wave.
    l.private_store(p).unwrap();
    for t in [Transition::Ready(p), Transition::Retire(p)] {
        let e = err(l.barrier(&[t], true));
        assert!(e.contains("wave-private") && e.contains("barrier transition"), "{e}");
    }
    // Shared store/load entry points refuse it (a store naming it, or a load).
    assert!(err(l.store(p)).contains("wave-private"));
    assert!(err(l.load(p)).contains("wave-private"));
    // ...and the private entry points refuse a shared slot.
    assert!(err(l.private_store(shared)).contains("workgroup-shared"));
    assert!(err(l.private_load(shared)).contains("workgroup-shared"));
    assert!(err(l.private_publish(shared)).contains("workgroup-shared"));
    assert!(err(l.private_retire(shared)).contains("workgroup-shared"));
    // The shared slot's own barrier protocol is untouched.
    l.store(shared).unwrap();
    l.barrier(&[Transition::Ready(shared)], true).unwrap();
    l.load(shared).unwrap();
    l.barrier(&[Transition::Retire(shared)], true).unwrap();
}

#[test]
fn paths_join_a_private_slot_only_in_the_same_state() {
    let mut shared_a = Lds::default();
    let s = shared_a.add("s", 0, 64).unwrap();
    let mut shared_b = shared_a.clone();
    shared_b.store(s).unwrap();
    // A shared region stored on one path is `Writing` on both (a peer stored).
    shared_a.join(&shared_b).unwrap();
    assert_eq!(shared_a.slots[s].state, SlotState::Publishing);

    let mut a = Lds::default();
    let p = a.add_wave_private("p", 0, 64, 2).unwrap();
    let mut b = a.clone();
    b.private_store(p).unwrap();
    // A private slice is each wave's own: nothing a peer stored.
    let e = err(a.join(&b));
    assert!(e.contains("Free on one and Publishing on the other"), "{e}");
    // Published and Reading merge (both readable by the owning wave).
    let (mut c, mut d) = (b.clone(), b.clone());
    c.private_publish(p).unwrap();
    d.private_publish(p).unwrap();
    d.private_load(p).unwrap();
    c.join(&d).unwrap();
    assert_eq!(c.slots[p].state, SlotState::Reading);
    // A different layout never joins.
    let mut other = Lds::default();
    other.add("p", 0, 128).unwrap();
    assert!(err(a.join(&other)).contains("different LDS layouts"));
}

// ---- an emitted kernel ----------------------------------------------------------

enum Tile {}
enum Scratch {}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Variant {
    /// Every access in the issuing wave's own slice.
    Good,
    /// Each odd wave works in another odd wave's slice (`(wave + 2) & 7`):
    /// race-free (nobody else touches that slice) but not wave-private.
    CrossWave,
    /// Lane stride 32 B instead of 4 B: wave w's slots overrun its 512 B slice.
    Overrun,
    /// Every odd wave works in slice 1: the same bytes from four waves, no barrier.
    Collide,
}

const STRIDE: u32 = 512;
const WAVES: u32 = 8;
const TILE_BASE: u32 = (STRIDE * WAVES) as u32; // 4096
const TILE_BYTES: u32 = 1024;
const SYMBOL: &str = "wave_private_demo";

fn alu(b: &mut Builder, text: &str, defs: &[RegRef], uses: &[RegRef]) -> Result<(), String> {
    b.push(Instruction::new(text, defs.to_vec(), uses.to_vec()))
}

fn kernel<T: Target>(
    wg: &mut Workgroup<'_, T, Builder>,
    gfx12: bool,
    variant: Variant,
) -> Result<(), String> {
    let p: WaveRegion<Scratch, _> = wg.lds_private("scratch", 0, STRIDE, WAVES)?;
    let tile = wg.lds::<Tile>("tile", TILE_BASE, TILE_BYTES)?;
    let end = wg.exit(".Lend")?;
    {
        let b = wg.isa();
        let off = if gfx12 { "0x0" } else { "null" };
        b.push(
            Instruction::new(
                format!("s_load_b64 s[2:3], s[0:1], {off}"),
                vec![sr(2, 2)],
                vec![sr(0, 2)],
            )
            .memory(MemoryClass::SmemLoad),
        )?;
        alu(b, "v_and_b32_e32 v1, 31, v0", &[v(1)], &[v(0)])?; // lane
        alu(b, "v_lshrrev_b32_e32 v2, 5, v0", &[v(2)], &[v(0)])?; // wave
        match variant {
            Variant::CrossWave => {
                alu(b, "v_add_nc_u32_e32 v4, 2, v2", &[v(4)], &[v(2)])?;
                alu(b, "v_and_b32_e32 v4, 7, v4", &[v(4)], &[v(4)])?;
                alu(b, "v_lshlrev_b32_e32 v4, 9, v4", &[v(4)], &[v(4)])?;
            }
            Variant::Collide => alu(b, "v_mov_b32_e32 v4, 0x200", &[v(4)], &[])?,
            _ => alu(b, "v_lshlrev_b32_e32 v4, 9, v2", &[v(4)], &[v(2)])?, // wave * 512
        }
        let own_lane = if variant == Variant::Overrun { 5 } else { 2 };
        alu(
            b,
            &format!("v_lshl_add_u32 v3, v1, {own_lane}, v4"),
            &[v(3)],
            &[v(1), v(4)],
        )?; // slot address
        alu(b, "v_xor_b32_e32 v5, 1, v1", &[v(5)], &[v(1)])?; // lane ^ 1
        alu(b, "v_lshl_add_u32 v6, v5, 2, v4", &[v(6)], &[v(5), v(4)])?;
        alu(b, "v_mul_u32_u24_e32 v7, 3, v0", &[v(7)], &[v(0)])?;
        alu(b, "v_add_nc_u32_e32 v7, 7, v7", &[v(7)], &[v(7)])?;
        alu(b, "v_mov_b32_e32 v10, 0", &[v(10)], &[])?;
        alu(b, "v_lshl_add_u32 v11, v0, 2, 0x1000", &[v(11)], &[v(0)])?;
        alu(b, "v_readfirstlane_b32 s4, v2", &[sr(4, 1)], &[v(2)])?;
        alu(b, "s_and_b32 s5, s4, 1", &[sr(5, 1)], &[sr(4, 1)])?;
    }
    // Even waves skip the body: wave-uniform control, no barrier inside.
    let c = wg.scmp(Instruction::new(
        "s_cmp_eq_u32 s5, 0",
        vec![],
        vec![sr(5, 1)],
    ))?;
    let _p = wg.skip_if(c, ".Lskip", p, |w, p| {
        let st = |text: &str, a: u8, d: u8| {
            Instruction::new(text, vec![], vec![v(a), v(d)]).memory(MemoryClass::DsStore)
        };
        let ld = |text: &str, d: u8, a: u8| {
            Instruction::new(text, vec![v(d)], vec![v(a)]).memory(MemoryClass::DsLoad)
        };
        let writing = w.private_store(p, st("ds_store_b32 v3, v7", 3, 7))?;
        let owned = w.private_publish(writing)?;
        w.private_load(&owned, ld("ds_load_b32 v8, v6", 8, 6))?;
        let p = w.private_retire(owned)?;
        alu(w.isa(), "v_add_nc_u32_e32 v9, 0x64, v8", &[v(9)], &[v(8)])?;
        // The retired slice is writable again; the wave reads its own slot back.
        let writing = w.private_store(p, st("ds_store_b32 v3, v9", 3, 9))?;
        let owned = w.private_publish(writing)?;
        w.private_load(&owned, ld("ds_load_b32 v10, v3", 10, 3))?;
        let p = w.private_retire(owned)?;
        alu(w.isa(), "v_add_nc_u32_e32 v10, 0, v10", &[v(10)], &[v(10)])?;
        Ok(p)
    })?;
    // Every wave: publish its result through the shared tile and read another wave's.
    let (tile, pend) = wg.ds_store(
        tile,
        Instruction::new("ds_store_b32 v11, v10", vec![], vec![v(11), v(10)])
            .memory(MemoryClass::DsStore),
    )?;
    let drained = wg.wait(pend)?;
    let (tile,) = wg.barrier((ready(tile, drained),))?;
    {
        let b = wg.isa();
        alu(b, "v_add_nc_u32_e32 v12, 32, v0", &[v(12)], &[v(0)])?;
        alu(b, "v_and_b32_e32 v12, 0xff, v12", &[v(12)], &[v(12)])?;
        alu(b, "v_lshl_add_u32 v12, v12, 2, 0x1000", &[v(12)], &[v(12)])?;
        alu(b, "v_lshlrev_b32_e32 v14, 2, v0", &[v(14)], &[v(0)])?;
    }
    wg.ds_load(
        &tile,
        Instruction::new("ds_load_b32 v13, v12", vec![v(13)], vec![v(12)])
            .memory(MemoryClass::DsLoad),
    )?;
    wg.isa().push(
        Instruction::new(
            "global_store_b32 v14, v13, s[2:3]",
            vec![],
            vec![v(14), v(13), sr(2, 2)],
        )
        .memory(MemoryClass::VmemStore),
    )?;
    wg.isa().wait_all()?;
    wg.end(end)
}

fn emit(arch: Arch, variant: Variant) -> Result<Emitted, String> {
    let mut plan = RegPlan::new(16, 104)?;
    const NAMES: [&str; 16] = [
        "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
    ];
    for (n, name) in NAMES.iter().enumerate() {
        plan.v::<1>(name, n as u8, Live::Whole)?;
    }
    plan.s::<2>("karg", 0, Live::Whole)?;
    plan.s::<2>("optr", 2, Live::Whole)?;
    plan.s::<1>("wave", 4, Live::Whole)?;
    plan.s::<1>("odd", 5, Live::Whole)?;
    let mut b = Builder::new(
        KernelSpec {
            kernel_id: "wave_private_demo".into(),
            variant: format!("{variant:?}"),
            arch,
            symbol: SYMBOL.into(),
            kernargs: KernargLayout::new(8).pointer("out", 0),
            user_sgpr_count: 2,
            system_sgpr_workgroup_id_y: false,
            workgroup_size: 256,
            group_segment_fixed_size: TILE_BASE + TILE_BYTES,
            wave32: true,
            cu_mode: false,
        },
        plan,
    );
    let gfx12 = arch.gfx12();
    match arch {
        Arch::Gfx1100 => kernel(&mut Workgroup::<Gfx1100, Builder>::new(&mut b)?, gfx12, variant)?,
        Arch::Gfx1151 => kernel(&mut Workgroup::<Gfx1151, Builder>::new(&mut b)?, gfx12, variant)?,
        Arch::Gfx1201 => kernel(&mut Workgroup::<Gfx1201, Builder>::new(&mut b)?, gfx12, variant)?,
        a => return Err(format!("{a:?}")),
    }
    b.finish()
}

const ARCHES: [Arch; 3] = [Arch::Gfx1100, Arch::Gfx1151, Arch::Gfx1201];

fn lines(e: &Emitted) -> Vec<&str> {
    e.s_text.lines().map(str::trim).collect()
}
fn mnemonic(l: &str) -> &str {
    l.split_whitespace().next().unwrap_or("")
}

#[test]
fn builder_emits_no_barrier_and_no_wait_inside_the_private_protocol() {
    for arch in ARCHES {
        let e = emit(arch, Variant::Good).unwrap();
        let text = lines(&e);
        // The wave-uniform body is [s_cbranch_scc1 .Lskip, .Lskip:).
        let from = text.iter().position(|l| *l == "s_cbranch_scc1 .Lskip").unwrap();
        let to = text.iter().position(|l| *l == ".Lskip:").unwrap();
        let body = &text[from + 1..to];
        for l in body {
            assert!(
                !mnemonic(l).starts_with("s_barrier"),
                "{arch:?}: barrier under wave-uniform control: {l}"
            );
        }
        // Between a private store and the load after it: nothing (no wait, no barrier).
        for (st, ld) in [
            ("ds_store_b32 v3, v7", "ds_load_b32 v8, v6"),
            ("ds_store_b32 v3, v9", "ds_load_b32 v10, v3"),
        ] {
            let i = body.iter().position(|l| *l == st).unwrap();
            assert_eq!(body[i + 1], ld, "{arch:?}: {body:?}");
        }
        // The only waits in the body are the use waits of the loaded registers.
        let waits: Vec<&&str> = body.iter().filter(|l| l.starts_with("s_wait")).collect();
        assert_eq!(waits.len(), 2, "{arch:?}: {waits:?}");
        // One workgroup barrier in the kernel: the shared tile's publication.
        let barriers = text.iter().filter(|l| mnemonic(l) == "s_barrier" || mnemonic(l) == "s_barrier_signal").count();
        assert_eq!(barriers, 1, "{arch:?}");
        // The proof records the private window for a certifier.
        let slots = &e.proof.lds_slots;
        let w = private_windows(slots);
        assert_eq!(
            w,
            [PrivateWindow {
                base: 0,
                stride: STRIDE,
                waves: WAVES
            }]
        );
        assert!(slots.iter().any(|s| s.name == "tile" && s.wave_private.is_none()));
    }
}

// ---- the object-level ownership proof ------------------------------------------

#[cfg(feature = "toolchain")]
#[test]
fn certifier_proves_each_wave_stays_in_its_own_slice() {
    for arch in ARCHES {
        let e = emit(arch, Variant::Good).unwrap();
        let w = private_windows(&e.proof.lds_slots);
        let limit = TILE_BASE + TILE_BYTES;
        let end = lds_wave_private(&e.s_text, SYMBOL, 8, limit, &w).unwrap();
        assert_eq!(end, limit, "{arch:?}");
        // The plain bound check of the same kernel agrees.
        assert_eq!(lds_bounds(&e.s_text, SYMBOL, 8, limit).unwrap(), limit);
        // The window must fit and be nonempty.
        assert!(lds_wave_private(&e.s_text, SYMBOL, 8, limit - 1024, &w).is_err());

        // A foreign-slice access, a slot overrun and a collision of waves in
        // one slice are object-level refusals naming the wave, the lane and the
        // slice; the builder (which sees slots, not addresses) accepted all.
        for bad in [Variant::CrossWave, Variant::Overrun, Variant::Collide] {
            let e = emit(arch, bad).unwrap();
            let w = private_windows(&e.proof.lds_slots);
            let msg = err(lds_wave_private(&e.s_text, SYMBOL, 8, limit, &w));
            assert!(
                msg.contains("outside its own slice"),
                "{arch:?} {bad:?}: {msg}"
            );
            // Plain bounds alone cannot tell: both stay inside the allocation.
            lds_bounds(&e.s_text, SYMBOL, 8, limit).unwrap();
        }
    }
}

#[cfg(feature = "toolchain")]
#[test]
fn certifier_refuses_masked_addresses_that_can_reach_a_window() {
    // A masked address is any submask of its mask: it may land in a window.
    let text = "\nk:\n\
        v_and_b32_e32 v1, 31, v0\n\
        s_branch .L1\n\
        .L1:\n\
        v_and_b32_e32 v2, 0xfc, v1\n\
        ds_load_b32 v3, v2\n\
        s_endpgm\n\
        .Lk_end:\n";
    let w = [PrivateWindow {
        base: 64,
        stride: 64,
        waves: 2,
    }];
    assert!(lds_bounds(text, "k", 1, 4096).is_ok());
    let msg = err(lds_wave_private(text, "k", 1, 4096, &w));
    assert!(msg.contains("masked address"), "{msg}");
    // Below the window it is fine.
    let w = [PrivateWindow {
        base: 256,
        stride: 64,
        waves: 2,
    }];
    lds_wave_private(text, "k", 1, 4096, &w).unwrap();
}

// ---- the emulator's gate model -------------------------------------------------

fn expected_out() -> Vec<u32> {
    // v10 of thread t: odd wave -> (3 * (wave*32 + (lane ^ 1)) + 7) + 100, else 0.
    let v10 = |t: u32| {
        let (wave, lane) = (t / 32, t % 32);
        if wave % 2 == 1 {
            3 * (wave * 32 + (lane ^ 1)) + 7 + 100
        } else {
            0
        }
    };
    (0..256).map(|t| v10((t + 32) % 256)).collect()
}

const KERNARG_VA: u64 = 0x1_0000;
const OUT_VA: u64 = 0x7f00_0000_0000;

fn run(arch: Arch, variant: Variant, private_window: bool) -> Result<Vec<u32>, String> {
    let e = emit(arch, variant)?;
    let (text, _) = peacemaker_isa::kernels::iu4_gemm::module(&[e], "wave_private_demo")?;
    let elf = peacemaker_isa::native::assemble(&text, arch)?;
    let program = peacemaker_lift::lift_object(
        &elf,
        peacemaker_lift::Options {
            frontend: peacemaker_ir::inst::Frontend::Builder,
        },
    )
    .map_err(|e| e.to_string())?
    .program;
    let mut emu = Emulator::new(&program, SYMBOL)?.hw_model();
    if private_window {
        emu = emu.wave_private_lds(0, STRIDE, WAVES);
    }
    let mut mem = Memory::default();
    mem.map("kernarg".into(), KERNARG_VA, OUT_VA.to_le_bytes().to_vec())?;
    mem.map("out".into(), OUT_VA, vec![0xAA; 256 * 4])?;
    emu.run_workgroup(
        &Launch {
            kernarg_ptr: KERNARG_VA,
            workgroup: [0, 0, 0],
            block: [256, 1, 1],
            dynamic_lds: 0,
        },
        &mut mem,
    )?;
    let out = mem
        .regions
        .iter()
        .find(|r| r.name == "out")
        .unwrap()
        .bytes
        .chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    Ok(out)
}

#[test]
fn multi_wave_kernel_is_race_free_and_exact_with_no_private_wait() {
    // The gate model runs the workgroup under round-robin, every
    // wave-runs-ahead schedule and two seeded-random interleavings, with LDS
    // race detection, in-flight loads/stores and the private-window ownership
    // check; the stores must be bit-identical across schedules.
    for arch in ARCHES {
        let got = run(arch, Variant::Good, true).unwrap_or_else(|e| panic!("{arch:?}: {e}"));
        assert_eq!(got, expected_out(), "{arch:?}");
        // Without declaring the window the race detector alone also passes: no
        // wave ever touches another wave's slice.
        assert_eq!(run(arch, Variant::Good, false).unwrap(), expected_out());
    }
}

#[test]
fn emulator_refuses_a_wave_touching_another_waves_slice() {
    for arch in ARCHES {
        // Declared private: refused at the access, naming the wave and slice.
        for bad in [Variant::CrossWave, Variant::Overrun, Variant::Collide] {
            let e = run(arch, bad, true).unwrap_err();
            assert!(e.contains("[private]"), "{arch:?} {bad:?}: {e}");
            assert!(e.contains("leaves its own slice"), "{arch:?} {bad:?}: {e}");
        }
        // Not declared: four waves writing the same bytes with no barrier is an
        // ordinary LDS race...
        let e = run(arch, Variant::Collide, false).unwrap_err();
        assert!(e.contains("[race]"), "{arch:?}: {e}");
        // ...while each odd wave working alone in another odd wave's slice is
        // race-free and exact: only the ownership check refuses it.
        assert_eq!(
            run(arch, Variant::CrossWave, false).unwrap(),
            expected_out(),
            "{arch:?}"
        );
    }
}

// ---- the contract pipeline (`toolchain::certify`) ---------------------------------

#[cfg(feature = "toolchain")]
mod pipeline {
    use super::*;
    use peacemaker_isa::toolchain::{build, certify, IsaShapeContract, Toolchain};

    fn contract(e: &Emitted, windows: bool) -> IsaShapeContract {
        IsaShapeContract {
            symbol: SYMBOL.into(),
            launch_dynamic_lds_bytes: Some(0),
            group_segment_fixed_bytes: TILE_BASE + TILE_BYTES,
            require_wave32: true,
            wave_private_windows: if windows {
                private_windows(&e.proof.lds_slots)
            } else {
                vec![]
            },
            ..Default::default()
        }
    }

    fn run_pipeline(arch: Arch, variant: Variant, windows: bool) -> Result<(), String> {
        let e = emit(arch, variant)?;
        let (text, _) = peacemaker_isa::kernels::iu4_gemm::module(&[e.clone()], "wave_private_demo")?;
        let dir = std::env::temp_dir().join(format!(
            "peacemaker-isa-wave-private-{}-{}-{variant:?}-{windows}",
            std::process::id(),
            arch.name()
        ));
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let s = dir.join("k.s");
        std::fs::write(&s, &text).map_err(|e| e.to_string())?;
        let toolchain = Toolchain::oracle();
        let b = build(&toolchain, &s, &dir.join("k.hsaco"), arch.name())?;
        certify(
            &toolchain,
            &b,
            &s,
            arch.name(),
            &dir.join("manifest.json"),
            Some(&contract(&e, windows)),
            "test",
            "test",
        )
    }

    macro_rules! need_rocm {
        () => {
            for t in ["llvm-mc", "ld.lld", "clang-offload-bundler", "llvm-objdump", "llvm-readobj"] {
                let _ = require_rocm_tool!(t);
            }
        };
    }

    #[test]
    fn pipeline_certifies_an_own_slice_kernel_and_refuses_foreign_slice_emits() {
        need_rocm!();
        for arch in [Arch::Gfx1100, Arch::Gfx1151] {
            run_pipeline(arch, Variant::Good, true).unwrap_or_else(|e| panic!("{arch:?}: {e}"));
            for bad in [Variant::CrossWave, Variant::Overrun, Variant::Collide] {
                let e = err(run_pipeline(arch, bad, true));
                assert!(e.contains("outside its own slice"), "{arch:?} {bad:?}: {e}");
            }
        }
    }

    #[test]
    fn pipeline_refuses_windows_it_cannot_prove() {
        need_rocm!();
        // gfx1201 `wave_private_demo` is not an M7-certified symbol: nothing would prove ownership.
        let e = err(run_pipeline(Arch::Gfx1201, Variant::Good, true));
        assert!(e.contains("wave-private LDS windows"), "{e}");
    }
}

