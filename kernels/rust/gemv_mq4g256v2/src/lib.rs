//! Rust-native port of `kernels/src/gemv_mq4g256v2.hip` lines 47-232
//! (the generic kernel: runtime K, plain pointer loads, `y[row] = acc`).
//!
//! Same 4-accumulator interleaving, DOG association order, tail distribution,
//! and shuffle reduction as the HIP source of truth. Kernel symbol and
//! kernarg layout are byte-identical to the hipcc build.

#![no_std]
#![feature(abi_gpu_kernel, link_llvm_intrinsics, f16)]
#![allow(internal_features)]

use core::panic::PanicInfo;

#[panic_handler]
fn panic(_: &PanicInfo) -> ! {
    loop {}
}

extern "C" {
    #[link_name = "llvm.amdgcn.workitem.id.x"]
    fn tid_x() -> u32;
    #[link_name = "llvm.amdgcn.workgroup.id.x"]
    fn wg_x() -> u32;
    #[link_name = "llvm.amdgcn.ds.bpermute"]
    fn ds_bpermute(addr: i32, data: i32) -> i32;
}

#[inline(always)]
unsafe fn half2float(bits: u16) -> f32 {
    f16::from_bits(bits) as f32
}

/// `__shfl_down` on wave32 via `ds_bpermute`. The workgroup is exactly one
/// wave32, so `tid_x()` is the lane id.
#[inline(always)]
unsafe fn shfl_down(v: f32, offset: u32, lane: u32) -> f32 {
    let src = if lane + offset < 32 {
        lane + offset
    } else {
        lane
    };
    f32::from_bits(ds_bpermute((src << 2) as i32, v.to_bits() as i32) as u32)
}

/// The DOG expression with the SAME association order as the HIP macro:
/// `a += t0 + t1 + ... + t7` parses as `a = a + (((t0 + t1) + t2) ... + t7)`.
/// One expression — do NOT split into separate `+=`.
macro_rules! dog {
    ($x:expr, $pk:expr, $sc:expr, $zp:expr, $b:expr, $a:expr) => {
        // One base pointer, constant lane offsets: keeps the eight loads
        // provably contiguous (Rust i32 adds have no nsw, so `($b + i) as usize`
        // per element cannot be split and the loads stay scalar b32).
        let xp = $x.add($b as usize);
        $a += ($sc * (($pk & 0xF) as f32) + $zp) * *xp
            + ($sc * ((($pk >> 4) & 0xF) as f32) + $zp) * *xp.add(1)
            + ($sc * ((($pk >> 8) & 0xF) as f32) + $zp) * *xp.add(2)
            + ($sc * ((($pk >> 12) & 0xF) as f32) + $zp) * *xp.add(3)
            + ($sc * ((($pk >> 16) & 0xF) as f32) + $zp) * *xp.add(4)
            + ($sc * ((($pk >> 20) & 0xF) as f32) + $zp) * *xp.add(5)
            + ($sc * ((($pk >> 24) & 0xF) as f32) + $zp) * *xp.add(6)
            + ($sc * ((($pk >> 28) & 0xF) as f32) + $zp) * *xp.add(7)
    };
}

#[no_mangle]
#[allow(non_snake_case)]
pub unsafe extern "gpu-kernel" fn gemv_mq4g256v2(
    A: *const u8,
    x: *const f32,
    y: *mut f32,
    M: i32,
    K: i32,
) {
    let lane = tid_x();
    let row = wg_x() as i32;
    if row >= M {
        return;
    }
    let tid = lane as i32;

    let groups_per_row = K / 256;
    let row_ptr = A.offset((row as i64 * groups_per_row as i64 * 136) as isize);
    let row_weight_offset = (row as i64 * groups_per_row as i64 * 136) as u32;
    let boff = tid * 4;

    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;
    let quads = groups_per_row >> 2;
    let tail = groups_per_row & 3;

    let mut q = 0;
    while q < quads {
        let g = q << 2;
        let gp0 = row_ptr.add((g * 136) as usize);
        let gp1 = gp0.add(136);
        let gp2 = gp1.add(136);
        let gp3 = gp2.add(136);

        let goff = row_weight_offset + g as u32 * 136u32;
        // One uniform 8-byte header load (s_load_b64), then select the VALUE per
        // lane. Selecting the address instead turns it into a divergent VMEM load.
        let h0 = *(gp0 as *const u64);
        let hs0 = if tid < 16 { h0 as u32 } else { (h0 >> 32) as u32 };
        let sc0 = half2float((hs0 & 0xFFFFu32) as u16);
        let zp0 = half2float((hs0 >> 16) as u16);
        // One uniform 8-byte header load (s_load_b64), then select the VALUE per
        // lane. Selecting the address instead turns it into a divergent VMEM load.
        let h1 = *(gp1 as *const u64);
        let hs1 = if tid < 16 { h1 as u32 } else { (h1 >> 32) as u32 };
        let sc1 = half2float((hs1 & 0xFFFFu32) as u16);
        let zp1 = half2float((hs1 >> 16) as u16);
        // One uniform 8-byte header load (s_load_b64), then select the VALUE per
        // lane. Selecting the address instead turns it into a divergent VMEM load.
        let h2 = *(gp2 as *const u64);
        let hs2 = if tid < 16 { h2 as u32 } else { (h2 >> 32) as u32 };
        let sc2 = half2float((hs2 & 0xFFFFu32) as u16);
        let zp2 = half2float((hs2 >> 16) as u16);
        // One uniform 8-byte header load (s_load_b64), then select the VALUE per
        // lane. Selecting the address instead turns it into a divergent VMEM load.
        let h3 = *(gp3 as *const u64);
        let hs3 = if tid < 16 { h3 as u32 } else { (h3 >> 32) as u32 };
        let sc3 = half2float((hs3 & 0xFFFFu32) as u16);
        let zp3 = half2float((hs3 >> 16) as u16);

        let pk0 = *(gp0.add((8 + boff) as usize) as *const u32);
        let pk1 = *(gp1.add((8 + boff) as usize) as *const u32);
        let pk2 = *(gp2.add((8 + boff) as usize) as *const u32);
        let pk3 = *(gp3.add((8 + boff) as usize) as *const u32);

        let base = g * 256 + tid * 8;

        dog!(x, pk0, sc0, zp0, base, acc0);
        dog!(x, pk1, sc1, zp1, base + 256, acc1);
        dog!(x, pk2, sc2, zp2, base + 512, acc2);
        dog!(x, pk3, sc3, zp3, base + 768, acc3);

        let _ = goff;
        q += 1;
    }

    if tail >= 1 {
        let g = quads << 2;
        let gp = row_ptr.add((g * 136) as usize);
        let goff = row_weight_offset + g as u32 * 136u32;
        // One uniform 8-byte header load (s_load_b64), then select the VALUE per
        // lane. Selecting the address instead turns it into a divergent VMEM load.
        let h = *(gp as *const u64);
        let hs = if tid < 16 { h as u32 } else { (h >> 32) as u32 };
        let sc = half2float((hs & 0xFFFFu32) as u16);
        let zp = half2float((hs >> 16) as u16);
        let pk = *(gp.add((8 + boff) as usize) as *const u32);
        let base = g * 256 + tid * 8;
        dog!(x, pk, sc, zp, base, acc0);
        let _ = goff;
    }
    if tail >= 2 {
        let g = (quads << 2) + 1;
        let gp = row_ptr.add((g * 136) as usize);
        let goff = row_weight_offset + g as u32 * 136u32;
        // One uniform 8-byte header load (s_load_b64), then select the VALUE per
        // lane. Selecting the address instead turns it into a divergent VMEM load.
        let h = *(gp as *const u64);
        let hs = if tid < 16 { h as u32 } else { (h >> 32) as u32 };
        let sc = half2float((hs & 0xFFFFu32) as u16);
        let zp = half2float((hs >> 16) as u16);
        let pk = *(gp.add((8 + boff) as usize) as *const u32);
        let base = g * 256 + tid * 8;
        dog!(x, pk, sc, zp, base, acc1);
        let _ = goff;
    }
    if tail >= 3 {
        let g = (quads << 2) + 2;
        let gp = row_ptr.add((g * 136) as usize);
        let goff = row_weight_offset + g as u32 * 136u32;
        // One uniform 8-byte header load (s_load_b64), then select the VALUE per
        // lane. Selecting the address instead turns it into a divergent VMEM load.
        let h = *(gp as *const u64);
        let hs = if tid < 16 { h as u32 } else { (h >> 32) as u32 };
        let sc = half2float((hs & 0xFFFFu32) as u16);
        let zp = half2float((hs >> 16) as u16);
        let pk = *(gp.add((8 + boff) as usize) as *const u32);
        let base = g * 256 + tid * 8;
        dog!(x, pk, sc, zp, base, acc2);
        let _ = goff;
    }

    let mut acc = (acc0 + acc1) + (acc2 + acc3);
    let mut offset = 16;
    while offset > 0 {
        acc += shfl_down(acc, offset, lane);
        offset >>= 1;
    }

    if tid == 0 {
        *y.add(row as usize) = acc;
    }
}
