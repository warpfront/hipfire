// SPDX-License-Identifier: Apache-2.0
//! M0: real-artifact grouped IU4 owner-mask oracle. Run ignored on fabric card A.
use hip_bridge::KernargBlob;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
};
const MODEL: &str =
    "/home/kaden/qcal/release-0.4.1/fn-requant/artifact/qwen3.8-flash-next-gptq3.mq4";
const E: usize = 512;
const H: usize = 2560;
const I: usize = 640;
type Result<T, Error = Box<dyn std::error::Error>> = std::result::Result<T, Error>;
struct Tensor {
    name: String,
    shape: Vec<usize>,
    offset: u64,
    len: usize,
}
fn index(f: &mut File) -> Result<Vec<Tensor>> {
    let mut h = [0; 32];
    f.read_exact(&mut h)?;
    assert_eq!(&h[..4], b"HFQM");
    let u32_at = |b: &[u8]| u32::from_le_bytes(b.try_into().unwrap()) as usize;
    let u64_at = |b: &[u8]| u64::from_le_bytes(b.try_into().unwrap());
    let meta = u64_at(&h[16..24]);
    let data = u64_at(&h[24..32]);
    f.seek(SeekFrom::Start(meta))?;
    let mut b = vec![0; (data - meta) as usize];
    f.read_exact(&mut b)?;
    let (mut depth, mut quoted, mut escape, mut p) = (0i32, false, false, 0);
    for (i, &c) in b.iter().enumerate() {
        if escape {
            escape = false;
            continue;
        }
        if quoted && c == b'\\' {
            escape = true;
            continue;
        }
        if c == b'"' {
            quoted = !quoted;
        }
        if !quoted {
            if c == b'{' {
                depth += 1;
            }
            if c == b'}' {
                depth -= 1;
                if depth == 0 {
                    p = i + 1;
                    break;
                }
            }
        }
    }
    assert!(p > 0);
    let n = u32_at(&b[p..p + 4]);
    p += 4;
    assert_eq!(n, u32_at(&h[12..16]));
    let mut offset = data;
    let mut out = Vec::new();
    for _ in 0..n {
        let l = u16::from_le_bytes(b[p..p + 2].try_into().unwrap()) as usize;
        p += 2;
        let name = String::from_utf8(b[p..p + l].to_vec())?;
        p += l + 1;
        let nd = b[p] as usize;
        p += 1;
        let mut shape = Vec::new();
        for _ in 0..nd {
            shape.push(u32_at(&b[p..p + 4]));
            p += 4;
        }
        p += 4;
        let len = u64_at(&b[p..p + 8]) as usize;
        p += 8;
        out.push(Tensor {
            name,
            shape,
            offset,
            len,
        });
        offset += len as u64;
    }
    Ok(out)
}
fn payload(f: &mut File, t: &Tensor) -> Result<Vec<u8>> {
    f.seek(SeekFrom::Start(t.offset))?;
    let mut b = vec![0; t.len];
    f.read_exact(&mut b)?;
    Ok(b)
}
fn raw(g: &mut Gpu, n: usize) -> Result<GpuTensor> {
    Ok(g.alloc_tensor(&[n.div_ceil(4)], DType::F32)?)
}
fn bytes(g: &Gpu, t: &GpuTensor, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0; n];
    g.hip.memcpy_dtoh(&mut b, &t.buf)?;
    Ok(b)
}
fn ints(v: impl Iterator<Item = i32>) -> Vec<u8> {
    v.flat_map(i32::to_le_bytes).collect()
}
fn gemm(
    g: &mut Gpu,
    nt: usize,
    down: bool,
    ptrs: &GpuTensor,
    tiles: &GpuTensor,
    sorted: &GpuTensor,
    y: &GpuTensor,
    p: usize,
    n: usize,
) -> Result<()> {
    let base = if down {
        "qwen4_moe_down_iu4_sym_pm_gfx1201"
    } else {
        "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201"
    };
    let name = if nt == 1 {
        base.to_string()
    } else {
        format!("{base}_nt{nt}")
    };
    g.ensure_qwen4_moe_sym_pm_entry(&name)?;
    let x = if down {
        g.scratch
            .qwen4_moe_down_i4_scratch
            .as_ref()
            .unwrap()
            .as_ptr()
    } else {
        g.scratch.int4_mmq_x_scratch.as_ref().unwrap().as_ptr()
    };
    let mut a = KernargBlob::new();
    for ptr in [
        ptrs.buf.as_ptr(),
        tiles.buf.as_ptr(),
        sorted.buf.as_ptr(),
        x,
        y.buf.as_ptr(),
    ] {
        a.push_ptr(ptr as *const _);
    }
    for v in if down {
        [H, I, 1, p, n * 10]
    } else {
        [2 * I, H, 10, p, n]
    } {
        a.push_i32(v as i32);
    }
    a.push_i32(E as i32);
    a.pad_to(16);
    g.launch_kernel_blob(
        &name,
        [
            (if down { H } else { 2 * I }) as u32 / 64,
            (p / 16) as u32,
            1,
        ],
        [if !down && nt == 1 { 64 } else { 128 }, 1, 1],
        0,
        a.as_mut_slice(),
    )?;
    Ok(())
}
#[test]
#[ignore = "real FN artifact and gfx1201 fabric card A"]
fn owner_mask_preserves_owned_and_skips_unowned() -> Result<()> {
    let mut f = File::open(std::env::var("FN_MODEL").unwrap_or(MODEL.into()))?;
    let ts = index(&mut f)?;
    let mut g = Gpu::init()?;
    assert_eq!(g.arch, "gfx1201");
    let mut failures = 0;
    for (layer, host) in [(0, false), (47, true)] {
        let gu = ts
            .iter()
            .find(|t| {
                t.name.starts_with("model.language_model.")
                    && t.name.contains(&format!("layers.{layer}."))
                    && t.name.ends_with("experts.gate_up_proj")
            })
            .expect("real gate/up tensor");
        let dn = ts
            .iter()
            .find(|t| {
                t.name.starts_with("model.language_model.")
                    && t.name.contains(&format!("layers.{layer}."))
                    && t.name.ends_with("experts.down_proj")
            })
            .expect("real down tensor");
        eprintln!(
            "M0 weights {} {:?}; {} {:?}",
            gu.name, gu.shape, dn.name, dn.shape
        );
        assert_eq!(gu.len, E * 1740800);
        assert_eq!(dn.len, E * 870400);
        let gu_b = payload(&mut f, gu)?;
        let dn_b = payload(&mut f, dn)?;
        let gu_w = if host {
            g.upload_raw_host_mapped(&gu_b, &[gu_b.len()])?
        } else {
            g.upload_raw(&gu_b, &[gu_b.len()])?
        };
        let dn_w = if host {
            g.upload_raw_host_mapped(&dn_b, &[dn_b.len()])?
        } else {
            g.upload_raw(&dn_b, &[dn_b.len()])?
        };
        drop(gu_b);
        drop(dn_b);
        let table = |w: &GpuTensor, stride: usize| {
            (0..E)
                .flat_map(|e| ((w.buf.as_ptr() as u64) + (e * stride) as u64).to_le_bytes())
                .collect::<Vec<_>>()
        };
        let gu_p = g.upload_raw(&table(&gu_w, 1740800), &[E * 8])?;
        let dn_p = g.upload_raw(&table(&dn_w, 870400), &[E * 8])?;
        for n in [512, 1127, 4055, 4096] {
            let p = (n * 10 + E * 15).div_ceil(16) * 16;
            g.reserve_qwen4_moe_sym(n, 10, E, H, I)?;
            let top = g.upload_raw(
                &ints((0..n * 10).map(|s| ((s * 37 + s / 10 * 13) % E) as i32)),
                &[n * 10],
            )?;
            let x = g.upload_f32(
                &(0..n * H)
                    .map(|s| ((s * 17 % 251) as f32 - 125.) / 128.)
                    .collect::<Vec<_>>(),
                &[n * H],
            )?;
            let counts = raw(&mut g, E * 4)?;
            let offsets = raw(&mut g, (E + 1) * 4)?;
            let sorted = raw(&mut g, p * 4)?;
            let tiles = raw(&mut g, p / 16 * 4)?;
            let inverse = raw(&mut g, n * 10 * 4)?;
            g.moe_scatter_stable_top10(
                &top,
                &counts,
                &offsets,
                &sorted,
                &tiles,
                &inverse,
                n * 10,
                E,
                p,
            )?;
            g.hip.device_synchronize()?;
            let ids: Vec<i32> = bytes(&g, &tiles, p / 16 * 4)?
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            assert!(ids.iter().any(|&e| e >= 0 && e % 2 == 0));
            assert!(ids.iter().any(|&e| e >= 0 && e % 2 == 1));
            let owned = g.upload_raw(
                &ints(
                    ids.iter()
                        .map(|&e| if e >= 0 && e % 2 == 0 { e } else { -1 }),
                ),
                &[p / 16],
            )?;
            // S1: the production PM mask kernel (owner table `e % 2`, rank 0) must equal the host mask word for word, and its copy is what the GEMMs below read.
            let owner = g.upload_raw(&ints((0..E).map(|e| (e % 2) as i32)), &[E])?;
            let owned_pm = raw(&mut g, p / 16 * 4)?;
            g.hip.memset(&owned_pm.buf, 0x5a, p / 16 * 4)?;
            g.ep_tile_mask(&tiles, &owned_pm, &owner, p / 16, E, 0)?;
            g.hip.device_synchronize()?;
            let (host_owned, pm_owned, source_after) = (
                bytes(&g, &owned, p / 16 * 4)?,
                bytes(&g, &owned_pm, p / 16 * 4)?,
                bytes(&g, &tiles, p / 16 * 4)?,
            );
            let mask_bad = host_owned
                .chunks_exact(4)
                .zip(pm_owned.chunks_exact(4))
                .filter(|(a, b)| a != b)
                .count()
                + source_after
                    .chunks_exact(4)
                    .zip(ids.iter())
                    .filter(|(a, &e)| i32::from_le_bytes((*a).try_into().unwrap()) != e)
                    .count();
            println!(
                "M0_PM_MASK rows={n} layer={layer} tiles={} mismatching_u32={mask_bad}",
                p / 16
            );
            failures += mask_bad;
            let owned = owned_pm;
            let ygu = raw(&mut g, p * I * 2)?;
            let ydn = raw(&mut g, p * H * 2)?;
            for nt in [1, 4, 8] {
                let mut reference = Vec::new();
                for masked in [false, true] {
                    g.hip.memset(&ygu.buf, 0xa5, p * I * 2)?;
                    g.hip.memset(&ydn.buf, 0xa5, p * H * 2)?;
                    let _gate = g.qwen4_moe_rotate256_i4(&x, H, n)?;
                    let tid = if masked { &owned } else { &tiles };
                    // Production selectors for NT4 resident / NT8 mapped; reachable NT1 and cross-residency anchors use their shipped entries.
                    if nt == if host { 8 } else { 4 } {
                        let a = g.qwen4_moe_rotate256_i4(&x, H, n)?;
                        g.gemm_qwen4_moe_gate_up_silu_iu4_sym(
                            &gu_p,
                            tid,
                            &sorted,
                            &a,
                            &ygu,
                            2 * I,
                            H,
                            10,
                            p,
                            n,
                            host,
                        )?;
                    } else {
                        gemm(&mut g, nt, false, &gu_p, tid, &sorted, &ygu, p, n)?;
                    }
                    let d = g.qwen4_moe_rotate128_i4(&ygu, &sorted, I, p, n * 10)?;
                    if nt == if host { 8 } else { 4 } {
                        g.gemm_qwen4_moe_down_iu4_sym(
                            &dn_p,
                            tid,
                            &sorted,
                            &d,
                            &ydn,
                            H,
                            I,
                            1,
                            p,
                            n * 10,
                            host,
                        )?;
                    } else {
                        gemm(&mut g, nt, true, &dn_p, tid, &sorted, &ydn, p, n)?;
                    }
                    g.hip.device_synchronize()?;
                    let got = bytes(&g, &ydn, p * H * 2)?;
                    if !masked {
                        reference = got;
                        continue;
                    }
                    let (mut own, mut skipped) = (0, 0);
                    for (t, &e) in ids.iter().enumerate() {
                        for w in t * 16 * H * 2 / 4..(t + 1) * 16 * H * 2 / 4 {
                            let a = &got[w * 4..w * 4 + 4];
                            if e >= 0 && e % 2 == 0 {
                                own += usize::from(a != &reference[w * 4..w * 4 + 4]);
                            } else {
                                skipped += usize::from(a != [0xa5; 4]);
                            }
                        }
                    }
                    println!("M0 rows={n} layer={layer} host={host} entry=NT{nt} owned_u32={own} skipped_u32={skipped}");
                    failures += own + skipped;
                    if own + skipped > 0 {
                        if let Ok(dir) = std::env::var("FAIL_DIR") {
                            std::fs::create_dir_all(&dir)?;
                            std::fs::write(format!("{dir}/m0-{layer}-{n}-{nt}-got.bin"), got)?;
                            std::fs::write(
                                format!("{dir}/m0-{layer}-{n}-{nt}-reference.bin"),
                                &reference,
                            )?;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(failures, 0, "M0 mismatching u32 words");
    Ok(())
}
