//! Cost of data-driven / control-flow PM4 primitives on the compute ring.
//!
//! `poc_pm4_adaptive <hsaco-with-lds768> [render-node]`
//!
//! For each IB shape, N=1000 dispatches of a ~1 us kernel (each followed by
//! the redline barrier, like a retained tape) and the same with N=1; reports
//! (t(N)-t(1))/(N-1) wall microseconds per dispatch:
//!   direct     SET_SH_REG state + DISPATCH_DIRECT               (baseline)
//!   indirect   SET_BASE once + DISPATCH_INDIRECT (dims from VRAM)
//!   loadsh     LOAD_SH_REG_INDEX (USER_DATA kernarg ptr from VRAM) + DIRECT
//!   condexec1  COND_EXEC(pred!=0) guarding each dispatch block (executes)
//!   condexec0  COND_EXEC(pred==0) guarding each dispatch block (skips)
//!   ibchain    each dispatch block in its own IB2, called via INDIRECT_BUFFER
//! Each variant also checks that the kernel really ran (or really did not).
use redline::device::Device;
use redline::dispatch::{lds_granularity, CommandBuffer, DispatchQueue, Kernel, KernargBuilder};
use std::time::Instant;

const N: usize = 250; // ~55 dwords per block; IB is 64 KB
const DISPATCH_INDIRECT: u32 = 0x16;
const COND_EXEC: u32 = 0x22;
const INDIRECT_BUFFER: u32 = 0x3F;
const LOAD_SH_REG_INDEX: u32 = 0x63;

fn pkt3(op: u32, body: u32, shader_type: bool) -> u32 {
    (3u32 << 30) | ((body - 1) << 16) | (op << 8) | ((shader_type as u32) << 1)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dev = Device::open(args.get(1).map(|s| s.as_str())).unwrap();
    let dq = DispatchQueue::new(&dev).unwrap();
    let gran = lds_granularity(&dev.info.gfx_arch);
    eprintln!("gfx_arch={}", dev.info.gfx_arch);
    let m = dev.load_module_file(&args[0]).unwrap();
    let k = Kernel::find(&m, "lds768").expect("lds768");
    let n_out = 192usize;

    let out = dev.alloc_vram(4096).unwrap();
    let fence = dev.alloc_vram(4096).unwrap();
    let ka = dev.alloc_vram(4096).unwrap();
    let mut kb = KernargBuilder::new(8);
    kb.write_ptr(0, out.gpu_addr);
    dev.upload(&ka, kb.as_bytes()).unwrap();
    // Data-driven operands live here: dims at +0, USER_DATA image at +64, predicate at +128.
    let data = dev.alloc_vram(4096).unwrap();
    let mut dbuf = vec![0u8; 4096];
    dbuf[0..12].copy_from_slice(&[1u32, 1, 1].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
    dbuf[64..72].copy_from_slice(&ka.gpu_addr.to_le_bytes());
    dev.upload(&data, &dbuf).unwrap();
    let pred_va = data.gpu_addr + 128;
    // IB2 pool for the chain variant: one small IB per dispatch.
    let ib2_stride = 256u64; // bytes
    let ib2 = dev.alloc_vram(ib2_stride * (N as u64 + 1)).unwrap();
    // Production gfx12 tape inter-node barrier (redline-rocr pm4.rs): EVENT_WRITE
    // CS_PARTIAL_FLUSH + ACQUIRE_MEM with the inter-node GCR (0x10180).
    let prod_barrier = |cb: &mut CommandBuffer| {
        cb.push_raw(&[pkt3(0x46, 1, false), 0x407]);
        cb.push_raw(&[pkt3(0x58, 7, false), 0, u32::MAX, 0x00ff_ffff, 0, 0, 0x0000_000a, 0x10180]);
    };
    // MEC has no nested-IB call; it can only CHAIN (tail-jump). Build N small
    // IBs, each = dispatch block [+ barrier] + INDIRECT_BUFFER(CHAIN) to the
    // next; the last has no chain. IB1 ("ibchain" variant) just chains to #0.
    let build_chain = |n: usize, barrier: u8| {
        let mut pool = vec![0u8; (ib2_stride as usize) * (N + 1)];
        for i in 0..n {
            let mut one = CommandBuffer::new();
            one.dispatch_lds(k, [1, 1, 1], [256, 1, 1], ka.gpu_addr, 0, gran);
            match barrier {
                1 => one.barrier(fence.gpu_addr, (i + 1) as u32),
                2 => prod_barrier(&mut one),
                _ => {}
            }
            if i + 1 < n {
                let nva = ib2.gpu_addr + ib2_stride * (i as u64 + 1);
                // size of the next IB in dwords is filled below once known; all IBs are the same size
                one.push_raw(&[pkt3(INDIRECT_BUFFER, 3, false), nva as u32, (nva >> 32) as u32, 0]);
            }
            let bytes = one.as_bytes();
            assert!(bytes.len() as u64 <= ib2_stride, "ib2 block {} > stride", bytes.len());
            pool[i * ib2_stride as usize..i * ib2_stride as usize + bytes.len()].copy_from_slice(&bytes);
        }
        // Patch chain sizes: block i (i>0) length in dwords.
        let len_of = |i: usize| -> u32 {
            let mut one = CommandBuffer::new();
            one.dispatch_lds(k, [1, 1, 1], [256, 1, 1], ka.gpu_addr, 0, gran);
            match barrier { 1 => one.barrier(fence.gpu_addr, 1), 2 => prod_barrier(&mut one), _ => {} }
            if i + 1 < n { one.push_raw(&[0, 0, 0, 0]); }
            one.len_dwords()
        };
        for i in 0..n.saturating_sub(1) {
            let blk = i * ib2_stride as usize;
            let l = len_of(i) as usize * 4;
            let sz = len_of(i + 1) | (1 << 20) | (1 << 23);
            pool[blk + l - 4..blk + l].copy_from_slice(&sz.to_le_bytes());
        }
        dev.upload(&ib2, &pool).unwrap();
        len_of(0) | (1 << 20) | (1 << 23)
    };

    let check = |expect_ran: bool, label: &str| {
        let mut raw = vec![0u8; n_out * 4];
        dev.download(&out, &mut raw).unwrap();
        let o: &[f32] = unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, n_out) };
        let bad = (0..n_out).filter(|&t| o[t] != (n_out - 1 - t) as f32).count();
        let ran = bad == 0;
        eprintln!("    {label}: kernel {} (expected {})", if ran { "ran" } else { "did not run" }, if expect_ran { "ran" } else { "skip" });
        ran == expect_ran
    };
    let clear = || dev.upload(&out, &vec![0u8; n_out * 4]).unwrap();

    // Builds an IB with `n` dispatch blocks of the given variant.
    let build = |variant: &str, n: usize, barrier: u8| -> CommandBuffer {
        let mut cb = CommandBuffer::new();
        for i in 0..n {
            match variant {
                "direct" => cb.dispatch_lds(k, [1, 1, 1], [256, 1, 1], ka.gpu_addr, 0, gran),
                "indirect" => {
                    cb.set_kernel_state(k, [256, 1, 1], ka.gpu_addr, 0, gran);
                    // MEC form (radv radv_cmd_buffer.c, uses_mec branch): VA inline, no SET_BASE.
                    let va = data.gpu_addr;
                    cb.push_raw(&[pkt3(DISPATCH_INDIRECT, 3, true), va as u32, (va >> 32) as u32, (1 << 0) | (1 << 15)]);
                }
                "loadsh" => {
                    cb.set_kernel_state(k, [256, 1, 1], 0, 0, gran); // USER_DATA written as zeros...
                    let va = data.gpu_addr + 64; // ...then overwritten from memory:
                    let reg_off = 0x0240 + k.kernarg_sgpr_idx.unwrap();
                    cb.push_raw(&[pkt3(LOAD_SH_REG_INDEX, 4, true), va as u32, (va >> 32) as u32, reg_off, 2]);
                    cb.push_raw(&[pkt3(0x15, 4, true), 1, 1, 1, (1 << 0) | (1 << 15)]);
                }
                "condexec1" | "condexec0" => {
                    let mut inner = CommandBuffer::new();
                    inner.dispatch_lds(k, [1, 1, 1], [256, 1, 1], ka.gpu_addr, 0, gran);
                    let len = inner.len_dwords();
                    cb.push_raw(&[pkt3(COND_EXEC, 4, false), pred_va as u32, (pred_va >> 32) as u32, 0, len]);
                    cb.push_raw(&inner.as_bytes().chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<_>>());
                }
                "ibchain" => {
                    if i == 0 {
                        let first = build_chain(n, barrier);
                        let va = ib2.gpu_addr;
                        cb.push_raw(&[pkt3(INDIRECT_BUFFER, 3, false), va as u32, (va >> 32) as u32, first]);
                    }
                    continue; // barrier lives inside the chained IBs
                }
                _ => unreachable!(),
            }
            match barrier {
                1 => cb.barrier(fence.gpu_addr, (i + 1) as u32),
                2 => prod_barrier(&mut cb),
                _ => {}
            }
        }
        cb
    };

    for barrier in [0u8, 2, 1] {
    eprintln!("== barrier after each dispatch: {}", ["none", "RELEASE_MEM+WAIT_REG_MEM (redline crate)", "CS_PARTIAL_FLUSH+ACQUIRE_MEM (production tape)"][barrier as usize]);
    for (variant, pred, expect) in [
        ("direct", 1u32, true),
        ("indirect", 1, true),
        ("loadsh", 1, true),
        ("condexec1", 1, true),
        ("condexec0", 0, false),
        ("ibchain", 1, true),
    ] {
        dbuf[128..132].copy_from_slice(&pred.to_le_bytes());
        dev.upload(&data, &dbuf).unwrap();
        let bos = [&m.code_buf, &out, &fence, &ka, &data, &ib2];
        let time = |n: usize| -> f64 {
            let cb = build(variant, n, barrier);
            dq.submit(&dev, &cb, &bos).unwrap();
            let mut best = f64::MAX;
            for _ in 0..3 {
                let t = Instant::now();
                dq.submit(&dev, &cb, &bos).unwrap();
                best = best.min(t.elapsed().as_secs_f64() * 1e6);
            }
            best
        };
        clear();
        let t1 = time(1);
        let ok = check(expect, variant);
        let tn = time(N);
        eprintln!("{variant:10} {:.3} us/dispatch  (t1={t1:.0}us tN={tn:.0}us) {}", (tn - t1) / (N as f64 - 1.0), if ok { "OK" } else { "MISMATCH" });
    }
    }
}
