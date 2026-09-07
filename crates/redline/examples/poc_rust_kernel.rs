//! Redline: dispatch a kernel compiled by rustc (nightly, amdgcn-amd-amdhsa
//! target) through bare libdrm. No HIP, no hipcc, no C anywhere in the path.
use redline::device::Device;
use redline::dispatch::{DispatchQueue, KernargBuilder, Kernel};

fn as_bytes<T>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn main() {
    let elf = std::env::args().nth(1).expect("path to rustc-built .elf");
    let dev = Device::open(None).unwrap();
    let dq = DispatchQueue::new(&dev).unwrap();
    let module = dev.load_module_file(&elf).unwrap();
    let kernel = Kernel::find(&module, "vec_add").expect("vec_add not found");
    eprintln!("kernel: {} (kernarg={}, lds={})", kernel.name, kernel.kernarg_size, kernel.group_segment_size);

    let n = 1u32 << 20;
    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
    let b: Vec<f32> = (0..n).map(|i| 1000.0 - i as f32 * 0.25).collect();
    let nbytes = n as usize * 4;
    let a_buf = dev.alloc_vram(nbytes as u64).unwrap();
    let b_buf = dev.alloc_vram(nbytes as u64).unwrap();
    let o_buf = dev.alloc_vram(nbytes as u64).unwrap();
    dev.upload(&a_buf, as_bytes(&a)).unwrap();
    dev.upload(&b_buf, as_bytes(&b)).unwrap();
    dev.upload(&o_buf, &vec![0u8; nbytes]).unwrap();

    let mut ka = KernargBuilder::new(28);
    ka.write_ptr(0, a_buf.gpu_addr).write_ptr(8, b_buf.gpu_addr).write_ptr(16, o_buf.gpu_addr).write_u32(24, n);
    let t = std::time::Instant::now();
    dq.dispatch(&dev, kernel, [(n + 255) / 256, 1, 1], [256, 1, 1], ka.as_bytes(), &[&module.code_buf, &a_buf, &b_buf, &o_buf]).unwrap();
    let dt = t.elapsed();

    let mut raw = vec![0u8; nbytes];
    dev.download(&o_buf, &mut raw).unwrap();
    let out: &[f32] = unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, n as usize) };
    let bad = (0..n as usize).filter(|&i| out[i] != a[i] + b[i]).count();
    eprintln!("vec_add n={n}: {bad} wrong, dispatch+wait {:.3} ms", dt.as_secs_f64() * 1e3);
    if bad != 0 { std::process::exit(1); }
    eprintln!("RUSTC-BUILT KERNEL RAN VIA BARE DRM: exact on all {n} elements");

    // LDS kernel: block-reverse through shared memory + s_barrier.
    let k2 = Kernel::find(&module, "lds_reverse").expect("lds_reverse not found");
    eprintln!("kernel: {} (kernarg={}, lds={})", k2.name, k2.kernarg_size, k2.group_segment_size);
    let lds_override = std::env::args().nth(2).and_then(|s| s.parse::<u32>().ok());
    let mut k2 = k2.clone();
    if let Some(l) = lds_override { eprintln!("  overriding group_segment_size {} -> {}", k2.group_segment_size, l); k2.group_segment_size = l; }
    dev.upload(&o_buf, &vec![0u8; nbytes]).unwrap();
    let mut ka = KernargBuilder::new(20);
    ka.write_ptr(0, a_buf.gpu_addr).write_ptr(8, o_buf.gpu_addr).write_u32(16, n);
    dq.dispatch(&dev, &k2, [(n + 255) / 256, 1, 1], [256, 1, 1], ka.as_bytes(), &[&module.code_buf, &a_buf, &o_buf]).unwrap();
    let mut raw = vec![0u8; nbytes];
    dev.download(&o_buf, &mut raw).unwrap();
    let out: &[f32] = unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, n as usize) };
    let bad = (0..n as usize).filter(|&i| { let wg = i / 256; let t = i % 256; out[i] != a[wg * 256 + (255 - t)] }).count();
    eprintln!("lds_reverse n={n}: {bad} wrong");
    // Which lanes are wrong within the first workgroup?
    let wrong_t: Vec<usize> = (0..256).filter(|&t| out[t] != a[255 - t]).collect();
    eprintln!("  wg0 wrong lanes: {} (first {:?} .. last {:?})", wrong_t.len(), &wrong_t[..wrong_t.len().min(4)], wrong_t.last());
    eprintln!("  wg0 out[0..4]={:?} out[128..132]={:?} out[252..256]={:?}", &out[..4], &out[128..132], &out[252..256]);
    if bad != 0 { std::process::exit(2); }
    eprintln!("RUSTC-BUILT LDS+BARRIER KERNEL: exact");
}
