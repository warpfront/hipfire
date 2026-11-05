//! Probe: is the Qwen4 symmetric-MoE int4 activation producer
//! (`qwen4_moe_rotate256_i4`) bitwise identical on gfx1201 and gfx1151? If so
//! the hetero split may send the primary's int4 sidecar across the link.
//!
//! ```text
//! HIPFIRE_DEVICES=gfx1201,gfx1151 cargo run --release -p rdna-compute --features lab --example hetero_iu4_xarch_probe
//! ```

use rdna_compute::Gpu;

const DIM: usize = 2560;

fn sidecar(gpu: &mut Gpu, x: &[f32], rows: usize) -> Vec<u8> {
    let t = gpu.upload_f32(x, &[rows, DIM]).unwrap();
    let prepared = gpu.qwen4_moe_rotate256_i4(&t, DIM, rows).unwrap();
    let ptr = gpu.int4_mmq_prepared_ptr(&prepared, DIM, rows).unwrap();
    let bytes = rdna_compute::scratch::int4_mmq_reservation_bytes(DIM, rows).unwrap();
    gpu.hip.device_synchronize().unwrap();
    // SAFETY: a non-owning view of the live sidecar.
    let view = unsafe { hip_bridge::DeviceBuffer::from_raw(ptr, bytes) };
    let mut out = vec![0u8; bytes];
    gpu.hip.memcpy_dtoh(&mut out, &view).unwrap();
    out
}

fn main() {
    let mut a = Gpu::init_with_device(0).expect("device 0");
    let mut b = Gpu::init_with_device(1).expect("device 1");
    println!("{} vs {}", a.arch, b.arch);
    let mut seed = 0x1234_5678_9abc_def0u64;
    let mut unit = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 40) as f32 / (1u64 << 24) as f32
    };
    for rows in [1usize, 512, 2048] {
        // Normal-ish rows with outliers (post-norm hidden states have a few).
        let x: Vec<f32> = (0..rows * DIM)
            .map(|i| {
                let v = (unit() - 0.5) * 2.0;
                if i % 997 == 0 { v * 40.0 } else { v }
            })
            .collect();
        let (sa, sb) = (sidecar(&mut a, &x, rows), sidecar(&mut b, &x, rows));
        let diff = sa.iter().zip(&sb).filter(|(p, q)| p != q).count();
        println!("rows {rows}: {} bytes, {diff} differ", sa.len());
        // Layout: is the sidecar row-major (a row block is a contiguous byte
        // range), so a sub-chunk can be sliced out of a whole chunk's sidecar?
        if rows > 1 {
            let per = sa.len() / rows;
            let (r0, r1) = (rows / 3, rows / 3 + 7.min(rows - rows / 3));
            let part = sidecar(&mut a, &x[r0 * DIM..r1 * DIM], r1 - r0);
            let sliced = &sa[r0 * per..r1 * per];
            println!("rows {rows}: rows {r0}..{r1} re-quantized alone == byte slice of the chunk: {}", part == sliced);
            // Block-major ([K/128 block][row] records of 72 B): gather each
            // block's run of the sub-range (one 2D copy on the device).
            let blocks = DIM / 128;
            let gathered: Vec<u8> =
                (0..blocks).flat_map(|b| sa[(b * rows + r0) * 72..(b * rows + r1) * 72].to_vec()).collect();
            println!("rows {rows}: rows {r0}..{r1} re-quantized alone == block-major gather: {}", part == gathered);
        }
    }
}
