//! The rolled K-loop. One trip is a whole 256-K group (two 128-K blocks,
//! plane parity 0 then 1), so every LDS address is an immediate and the
//! trip ends in the state it started in. The last group is peeled: its
//! second block fetches and publishes nothing.
//!
//! Block (plane `h`), after hipcc K1-ILP's measured order:
//! - Phase A: slab-1 prefetch clause; fragment loads from slot 0; slab-0
//!   WMMAs (the first seeds every accumulator with the magic); publish slab 1
//!   into slot 1; B1 (retire slot 0, publish slot 1).
//! - Phase B: next-block fetch clause; fragment loads from slot 1 and the
//!   token scales; slab-1 WMMAs with the scale rows loaded into fragment
//!   registers freed by slab 0, and the fold's `subrev` stage between them;
//!   publish the next block into slot 0 and planes `h^1`; B2 signal; the
//!   fold's `mul`/`fmac` stage; B2 wait.
use super::{Gen, K_LOOP, K_LOOP_END, EPI, fold, op, publish::{self, ds_load}, s, sr};
use crate::{Builder, V, insn::Wmma, lds::Transition, reg::Live};

fn between(a: &str, b: &str) -> Live { Live::Between(a.into(), b.into()) }

pub(crate) fn emit(b: &mut Builder, g: &Gen) -> Result<(), String> {
    b.label(super::K_BEGIN)?;
    // K == 256: no full trip; go straight to the peeled group.
    op(b, format!("s_cmp_eq_u32 s{}, 0", g.trips), &[], &[s(g.trips)])?;
    op(b, format!("s_cbranch_scc1 {}", tail_label(0)), &[], &[])?;
    let step = |b: &mut Builder, srd: u8| op(b, format!("s_add_nc_u64 s[{0}:{1}], s[{0}:{1}], s[{2}:{3}]", srd, srd + 1, g.step2, g.step2 + 1),
        &[sr(srd, 2)], &[sr(srd, 2), sr(g.step2, 2)]);
    b.loop_(K_LOOP, |b| {
        block(b, g, "m0", 0, true, &label("m1"))?;
        // The even-block descriptor was last read by block 0's slab-1 fetch;
        // move it to block 2g+2 before block 1's next-block fetch reads it.
        block_with(b, g, "m1", 1, true, K_LOOP_END, |b| step(b, g.srd_a[0]))?;
        // Trip end: the odd-block descriptor and the group offset move on
        // after every read of them in this trip.
        step(b, g.srd_a[1])?;
        op(b, format!("s_add_co_i32 s{0}, s{0}, 0x88", g.goff), &[s(g.goff)], &[s(g.goff)])?;
        op(b, format!("s_add_co_i32 s{0}, s{0}, -1", g.trips), &[s(g.trips)], &[s(g.trips)])?;
        op(b, format!("s_cmp_lg_u32 s{}, 0", g.trips), &[], &[s(g.trips)])?;
        op(b, format!("s_cbranch_scc1 {K_LOOP}"), &[], &[])
    })?;
    b.label(K_LOOP_END)?;
    block(b, g, "t0", 0, true, &label("t1"))?;
    block(b, g, "t1", 1, false, EPI)
}

fn label(tag: &str) -> String { format!(".Liu4_{tag}") }
fn tail_label(h: usize) -> String { label(&format!("t{h}")) }

fn block(b: &mut Builder, g: &Gen, tag: &str, h: usize, fetch_next: bool, end: &str) -> Result<(), String> {
    block_with(b, g, tag, h, fetch_next, end, |_| Ok(()))
}
/// One 128-K block; `after_slab1_fetch` runs right after the phase-A clause.
fn block_with(b: &mut Builder, g: &Gen, tag: &str, h: usize, fetch_next: bool, end: &str,
    after_slab1_fetch: impl Fn(&mut Builder) -> Result<(), String>) -> Result<(), String> {
    let (x, x0, x1) = (label(tag), label(&format!("{tag}_s0")), label(&format!("{tag}_s1")));
    b.label(&x)?;
    let [f0, f1] = g.f;
    b.regs.v::<4>("frag0_w", f0, between(&x, &x0))?;
    b.regs.v::<8>("frag0_a", f0 + 4, between(&x, &x0))?;
    b.regs.v::<4>("frag1_w", f1, between(&x, &x1))?;
    b.regs.v::<8>("frag1_a", f1 + 4, between(&x, &x1))?;
    b.regs.v::<8>("sc_rg0", f0 + 4, between(&x0, end))?;
    b.regs.v::<4>("sc_rg1_lo", f0, between(&x0, end))?;
    b.regs.v::<4>("sc_rg1_hi", f1, between(&x1, end))?;
    b.regs.v::<8>("fold_t", f1 + 4, between(&x1, end))?;

    // Phase A: slab 0 from slot 0, slab 1 staged into slot 1.
    publish::fetch_slab1(b, g, h)?;
    after_slab1_fetch(b)?;
    fragments(b, g, 0)?;
    for sb in 0..2 { bundle(b, g, sb, sb == 0, |_, _| Ok(()))?; }
    publish::publish_slab(b, g, 1)?;
    b.barrier(&[Transition::Retire(g.slot_a[0]), Transition::Retire(g.slot_w[0]), Transition::Ready(g.slot_a[1]), Transition::Ready(g.slot_w[1])])?;

    // Phase B: slab 1 from slot 1, the fold, next block staged into slot 0.
    if fetch_next { publish::fetch_next(b, g, h)?; }
    fragments(b, g, 1)?;
    let l = g.layout;
    let dp = h as u32 * (l.ds[1] - l.ds[0]) / 4;
    for pair in 0..2u32 {
        ds_load(b, g.slot_ds[h], format!("ds_load_2addr_b32 v[{}:{}], v{}{}", g.d + 2 * pair as u8, g.d + 2 * pair as u8 + 1, g.d_addr,
            super::ds_offsets(dp + 32 * pair, dp + 32 * pair + 16)), super::vr(g.d + 2 * pair as u8, 2), g.d_addr)?;
    }
    bundle(b, g, 0, false, |_, _| Ok(()))?;
    b.label(&x0)?;
    let sz = l.sz[h];
    for (dst, off) in [(f0 + 4, 0), (f0 + 8, 16), (f0, 64)] {
        ds_load(b, g.slot_sz[h], format!("ds_load_b128 v[{}:{}], v{}{}", dst, dst + 3, g.sc_addr, super::ds_offset(sz + off)), super::vr(dst, 4), g.sc_addr)?;
    }
    bundle(b, g, 1, false, |b, i| fold::unbias_after_wmma(b, g, i))?;
    b.label(&x1)?;
    ds_load(b, g.slot_sz[h], format!("ds_load_b128 v[{}:{}], v{}{}", f1, f1 + 3, g.sc_addr, super::ds_offset(sz + 80)), super::vr(f1, 4), g.sc_addr)?;
    fold::unbias_tail(b, g)?;
    if fetch_next {
        publish::publish_slab(b, g, 0)?;
        publish::publish_meta(b, g, h ^ 1)?;
        b.barrier_signal(&[Transition::Retire(g.slot_a[1]), Transition::Retire(g.slot_w[1]), Transition::Retire(g.slot_ds[h]), Transition::Retire(g.slot_sz[h]),
            Transition::Ready(g.slot_a[0]), Transition::Ready(g.slot_w[0]), Transition::Ready(g.slot_ds[h ^ 1]), Transition::Ready(g.slot_sz[h ^ 1])])?;
    }
    fold::scale(b, g)?;
    if fetch_next { b.barrier_wait()?; }
    Ok(())
}

/// Fragment loads of both 32-K sub-blocks from slot `slot`: per sub-block one
/// W load (both row groups) and two A loads (column blocks 0-1, 2-3).
fn fragments(b: &mut Builder, g: &Gen, slot: usize) -> Result<(), String> {
    let l = g.layout;
    let (a, w) = (l.a[slot] / 512, l.w[slot] / 512);
    for sb in 0..2 {
        let f = g.f[sb];
        ds_load(b, g.slot_w[slot], format!("ds_load_2addr_stride64_b64 v[{}:{}], v{}{}", f, f + 3, g.fr_w[sb], super::ds_offsets(w, w + 1)), super::vr(f, 4), g.fr_w[sb])?;
        for half in 0..2u8 {
            let dst = f + 4 + 4 * half;
            let o = a + 2 * u32::from(half);
            ds_load(b, g.slot_a[slot], format!("ds_load_2addr_stride64_b64 v[{}:{}], v{}{}", dst, dst + 3, g.fr_a[sb], super::ds_offsets(o, o + 1)), super::vr(dst, 4), g.fr_a[sb])?;
        }
    }
    Ok(())
}

/// Eight WMMAs of sub-block `sb` in `fold::BUNDLE_ORDER`; `after(b, i)` runs
/// after the `i`-th.
fn bundle(b: &mut Builder, g: &Gen, sb: usize, first: bool, after: impl Fn(&mut Builder, usize) -> Result<(), String>) -> Result<(), String> {
    let f = g.f[sb];
    for (i, &k) in fold::BUNDLE_ORDER.iter().enumerate() {
        let (nb, rg) = (k / 2, k % 2);
        let dst = V::<8>(g.cacc + 8 * k as u8);
        let c = if first { V::<8>(g.magic) } else { dst };
        b.push(Wmma::iu4(g.spec.arch, dst, V::<2>(f + 2 * rg as u8), V::<2>(f + 4 + 2 * nb as u8), Some(c)))?;
        after(b, i)?;
    }
    Ok(())
}
