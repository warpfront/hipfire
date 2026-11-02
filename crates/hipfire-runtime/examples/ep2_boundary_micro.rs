// SPDX-License-Identifier: Apache-2.0
//! EP2 kill-test peer visibility and boundary micro (fabric A+E only).
use hipfire_runtime::ep_row_gather::{EpRowGather, EpRowGatherSpec};
use rdna_compute::{DType, Gpu, GpuTensor};
type Result<T, E = Box<dyn std::error::Error>> = std::result::Result<T, E>;
fn download(g: &Gpu, t: &GpuTensor, n: usize) -> Result<Vec<u8>> {
    g.bind_thread()?;
    let mut b = vec![0; n];
    g.hip.memcpy_dtoh(&mut b, &t.buf)?;
    Ok(b)
}
fn peer_probe(after: bool) -> Result<()> {
    let mut gs = [Gpu::init_with_device(0)?, Gpu::init_with_device(1)?];
    let bytes = (4096usize * 10 + 512 * 15).div_ceil(16) * 16 * 2560 * 2;
    let mut bufs = Vec::new();
    let mut builder = EpRowGather::builder(EpRowGatherSpec {
        ranks: 2,
        slots: 65536,
        row_bytes: 16,
    });
    for (rank, g) in gs.iter_mut().enumerate() {
        assert_eq!(g.arch, "gfx1201");
        g.reserve_qwen4_moe_sym(4096, 10, 512, 2560, 640)?;
        let a = g.alloc_tensor(&[bytes / 4], DType::F32)?;
        let b = g.alloc_tensor(&[bytes / 4], DType::F32)?;
        builder.register(rank, g, &[&a, &b])?;
        bufs.push([a, b]);
    }
    let transport = builder.build()?;
    for (r, g) in gs.iter_mut().enumerate() {
        transport.enable_peers(r, g)?;
    }
    // The after arm deliberately grows a new allocation after enable_peers.
    let mut late = Vec::new();
    if after {
        for g in &mut gs {
            late.push(g.alloc_tensor(&[bytes / 4], DType::F32)?);
        }
    }
    let mut bases = [0; 8];
    let mut expected = Vec::new();
    for (r, g) in gs.iter_mut().enumerate() {
        g.bind_thread()?;
        let src = if after { &late[r] } else { &bufs[r][0] };
        g.hip.memset(&src.buf, 0x31 + r as i32, bytes)?;
        g.hip.device_synchronize()?;
        expected.push(download(g, src, 1 << 20)?);
        bases[r] = src.buf.as_ptr() as usize;
    }
    for (r, g) in gs.iter_mut().enumerate() {
        g.bind_thread()?;
        let ids = g.upload_raw(
            &(0..65536)
                .flat_map(|_| ((1 - r) as i32).to_le_bytes())
                .collect::<Vec<_>>(),
            &[65536],
        )?;
        g.ep_gather_rows_by_owner(
            &ids,
            bases,
            bufs[r][1].buf.as_ptr() as usize,
            65536,
            2,
            r,
            1,
            512,
        )?;
        g.hip.device_synchronize()?;
        let got = download(g, &bufs[r][1], 1 << 20)?;
        let mismatch = got
            .chunks_exact(4)
            .zip(expected[1 - r].chunks_exact(4))
            .filter(|(a, b)| a != b)
            .count();
        println!(
            "M0_PEER allocation={} rank={r} bytes={} mismatching_u32={mismatch}",
            if after { "after" } else { "before" },
            1 << 20
        );
        assert_eq!(mismatch, 0);
    }
    for (r, g) in gs.iter_mut().enumerate() {
        transport.release(r, g)?;
    }
    Ok(())
}
fn main() -> Result<()> {
    match std::env::args().nth(1).as_deref() {
        Some("peer-before") => peer_probe(false),
        Some("peer-after") => peer_probe(true),
        _ => Err("expected peer-before or peer-after".into()),
    }
}
