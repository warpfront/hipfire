// gfx11 WMMA zero-sign probe: output element 0 = A[0][0]*B[0][0] + masked partners A[0][k>=1] * (+0) + C. mode 0 = single WMMA, mode 1 = A split by sign (A- then A+). Masked partner = V of a masked key, B = masked P = +0.
use hip_bridge::KernargBlob;
use rdna_compute::{DType, Gpu, GpuTensor, KernelCompiler};
type R<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn raw(g: &mut Gpu, n: usize) -> R<GpuTensor> {
    Ok(g.zeros(&[n.max(16)], DType::Raw)?)
}
fn dl(g: &Gpu, t: &GpuTensor) -> R<Vec<u8>> {
    g.hip.device_synchronize()?;
    let mut d = vec![0u8; t.byte_size()];
    g.hip.memcpy_dtoh(&mut d, &t.buf)?;
    Ok(d)
}
const SRC: &str = r#"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
typedef _Float16 frag __attribute__((ext_vector_type(16)));
typedef float f8 __attribute__((ext_vector_type(8)));
// mode 0: one mma(A,B,C); mode 1: split A by sign (A- then A+) as two mmas.
extern "C" __global__ void wmma_probe(const unsigned short* A, const unsigned short* B, const float* C, float* out, int mode){
  int lane=threadIdx.x&31, lo=lane&15, hi=lane>>4;
  union{unsigned short h[16];frag v;}ua,ub,up,um;
  for(int e=0;e<16;++e){ua.h[e]=A[lo*16+e];ub.h[e]=B[lo*16+e];
    _Float16 x=ua.v[e]; up.v[e]=(x>(_Float16)0)?x:(_Float16)0.0f; um.v[e]=(x<(_Float16)0)?x:(_Float16)(-0.0f);}
  f8 c; for(int e=0;e<8;++e) c[e]=C[(2*e+hi)*16+lo];
  if(mode==0) c=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ua.v,ub.v,c);
  else { c=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(um.v,ub.v,c); c=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(up.v,ub.v,c); }
  for(int e=0;e<8;++e) out[(2*e+hi)*16+lo]=c[e];
}
"#;
fn h16(x: f32) -> u16 {
    let b = x.to_bits();
    let s = ((b >> 31) & 1) as u16;
    let e = ((b >> 23) & 0xff) as i32;
    let m = b & 0x7fffff;
    if e == 0 {
        return s << 15;
    }
    let ne = e - 127 + 15;
    if ne >= 31 {
        return (s << 15) | 0x7bff;
    }
    if ne <= 0 {
        return s << 15;
    }
    (s << 15) | ((ne as u16) << 10) | (m >> 13) as u16
}
fn main() -> R<()> {
    let mut g = Gpu::init()?;
    let flags = format!("{} -ffp-contract=off", g.flags.hipcc_extra_flags);
    let mut comp = KernelCompiler::new(&g.arch, flags)?;
    let path = comp
        .compile_for_symbol("wmma_probe_pv", SRC, "wmma_probe")?
        .to_path_buf();
    let module = g.hip.module_load(path.to_str().unwrap())?;
    let func = g.hip.module_get_function(&module, "wmma_probe")?;
    let mut s = 0x1234_5678_9abc_def0u64;
    let mut rnd = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let (at, bt, ct, ot) = (
        raw(&mut g, 512)?,
        raw(&mut g, 512)?,
        raw(&mut g, 1024)?,
        raw(&mut g, 1024)?,
    );
    for mode in [0i32, 1] {
        let (mut cases, mut bad) = (0, 0);
        for trial in 0..400 {
            // valid data: d0, C, s0 random magnitudes/signs; masked partners k>=1: random sign/magnitude patterns, compared across 8 variants
            let d0 = ((rnd() % 2000) as f32 / 1000.0 - 1.0) * 0.05;
            let cv = ((rnd() % 2000) as f32 / 1000.0 - 1.0) * 0.05;
            let s0 = ((rnd() % 2000) as f32 / 1000.0 - 1.0) * 0.01;
            let mut results = Vec::new();
            for _var in 0..8 {
                let mut a = vec![0u16; 256];
                let mut b = vec![0u16; 256];
                a[0] = h16(d0);
                for k in 1..16 {
                    a[k] = h16(((rnd() % 2000) as f32 / 1000.0 - 1.0) * 0.1);
                }
                b[0] = h16(s0);
                let c: Vec<f32> = (0..256).map(|i| if i == 0 { cv } else { 0.0 }).collect();
                g.memcpy_htod_auto(
                    &at.buf,
                    &a.iter().flat_map(|x| x.to_ne_bytes()).collect::<Vec<u8>>(),
                )?;
                g.memcpy_htod_auto(
                    &bt.buf,
                    &b.iter().flat_map(|x| x.to_ne_bytes()).collect::<Vec<u8>>(),
                )?;
                g.memcpy_htod_auto(
                    &ct.buf,
                    &c.iter().flat_map(|x| x.to_ne_bytes()).collect::<Vec<u8>>(),
                )?;
                let mut blob = KernargBlob::new();
                for t in [&at, &bt, &ct, &ot] {
                    blob.push_ptr(t.buf.as_ptr());
                }
                blob.push_i32(mode);
                unsafe {
                    g.hip.launch_kernel_blob(
                        &func,
                        [1, 1, 1],
                        [32, 1, 1],
                        0,
                        None,
                        blob.as_mut_slice(),
                    )?;
                }
                let o = dl(&g, &ot)?;
                results.push(u32::from_ne_bytes([o[0], o[1], o[2], o[3]]));
            }
            cases += 1;
            if results.iter().any(|r| *r != results[0]) {
                bad += 1;
            }
            let _ = trial;
        }
        println!(
            "mode {mode} ({}): {bad}/{cases} cases changed by masked-partner values",
            if mode == 0 {
                "single mma"
            } else {
                "sign-split mma"
            }
        );
    }
    println!("arch {}", g.arch);
    Ok(())
}
