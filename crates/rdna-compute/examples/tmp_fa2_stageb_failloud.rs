// THROWAWAY StageB host-gate probe. DELETE BEFORE COMMIT. Not for production.
//
// Fail-loud contract for the stage-b launchers (S3):
//   flag off -> Err naming HIPFIRE_GFX12_FA2_FP8 (never an f16 fallback)
//   flag on  -> Err out of ensure_kernel (stage-b pre-convert symbols are
//              absent pre-B2), never a silent fallback and never a launch.
// Exit 0 iff every expectation holds; nonzero with a diagnostic otherwise.
// Usage: HIPFIRE_GFX12_FA2_FP8=0|1 tmp_fa2_stageb_failloud
use rdna_compute::{DType, Gpu};

const NH: usize = 24;
const NKV: usize = 4;
const HD: usize = 256;
const BATCH: usize = 8;
const CTX: usize = 64;

fn bytes_of_i32(v: &[i32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    if gpu.arch != "gfx1201" {
        eprintln!("SKIP: exact gfx1201 only (arch={})", gpu.arch);
        return;
    }
    let flag_on = gpu.flags.gfx12_fa2_fp8_enabled();
    eprintln!("gfx12_fa2_fp8_enabled={flag_on}");

    let q = vec![0.5f32; BATCH * NH * HD];
    let pos = vec![(CTX as i32) - 1; BATCH];
    let kv_dummy = vec![0u8; 4096];
    let d_q = gpu.upload_f32(&q, &[BATCH, NH, HD]).unwrap();
    let d_k = gpu.upload_raw(&kv_dummy, &[kv_dummy.len()]).unwrap();
    let d_v = gpu.upload_raw(&kv_dummy, &[kv_dummy.len()]).unwrap();
    let d_pos = gpu.upload_raw(bytes_of_i32(&pos), &[BATCH * 4]).unwrap();
    let d_out = gpu.zeros(&[BATCH * NH * HD], DType::F32).unwrap();

    let mut fail = 0;
    let mut check = |name: &str, r: Result<(), _>| {
        match r {
            Ok(()) => {
                eprintln!("FAIL: {name} unexpectedly launched");
                fail += 1;
            }
            Err(e) => {
                let msg = format!("{e:?}");
                if !flag_on {
                    // Flag off: must be the explicit no-fallback refusal.
                    if !msg.contains("HIPFIRE_GFX12_FA2_FP8=1") {
                        eprintln!("FAIL: {name} flag-off error is not the no-fallback refusal");
                        fail += 1;
                    }
                } else {
                    // Flag on, pre-B2: must be loud (missing stage-b
                    // pre-convert symbol), never the flag refusal and never
                    // a silent f16 run (which would return Ok).
                    if msg.contains("no f16 fallback") {
                        eprintln!("FAIL: {name} flag-on hit the flag-off path");
                        fail += 1;
                    }
                }
            }
        }
    };

    check(
        "route-Q",
        gpu.attention_q8_0_fa2_gqa_fp8_gfx1201(
            &d_q, &d_k, &d_v, &d_out, &d_pos, NH, NKV, HD, CTX, BATCH,
        ),
    );
    check(
        "route-N",
        gpu.attention_fp8_e4m3_fa2_gqa_fp8_gfx1201(
            &d_q, &d_k, &d_v, &d_out, &d_pos, NH, NKV, HD, CTX, BATCH,
        ),
    );

    if fail == 0 {
        eprintln!("PASS: stage-b fail-loud contract holds (flag_on={flag_on})");
    } else {
        eprintln!("FAIL: {fail} expectation(s) broken (flag_on={flag_on})");
        std::process::exit(1);
    }
}
