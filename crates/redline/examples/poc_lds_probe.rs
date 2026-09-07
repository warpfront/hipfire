use redline::device::Device;
use redline::dispatch::{DispatchQueue, KernargBuilder, Kernel};
fn main() {
    let node = std::env::args().nth(2);
    let dev = Device::open(node.as_deref()).unwrap(); let dq = DispatchQueue::new(&dev).unwrap();
    eprintln!("family_id={} gfx_arch={:?} gran={}", dev.info.family_id, dev.info.gfx_arch, redline::dispatch::lds_granularity(&dev.info.gfx_arch));
    let m = dev.load_module_file(&std::env::args().nth(1).unwrap()).unwrap();
    for (name, n) in [("lds1024", 256usize), ("lds4096", 1024usize)] {
        let k = Kernel::find(&m, name).unwrap();
        let o = dev.alloc_vram((n * 4) as u64).unwrap(); dev.upload(&o, &vec![0u8; n * 4]).unwrap();
        let mut ka = KernargBuilder::new(8); ka.write_ptr(0, o.gpu_addr);
        dq.dispatch(&dev, k, [1, 1, 1], [n as u32, 1, 1], ka.as_bytes(), &[&m.code_buf, &o]).unwrap();
        let mut raw = vec![0u8; n * 4]; dev.download(&o, &mut raw).unwrap();
        let out: &[f32] = unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, n) };
        let bad = (0..n).filter(|&t| out[t] != (n - 1 - t) as f32).count();
        eprintln!("{name} (kd lds={}): {bad}/{n} wrong", k.group_segment_size);
    }
}
