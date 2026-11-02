//! Micro-benchmark: H2D upload from (a) a warm heap buffer vs (b/c/d) mmap'd
//! file-backed pages (the zero-copy loader path). Decides where the
//! remaining load-time wall lives.
//!
//! Phases:
//! - `HEAP_SRC`      — 64 MiB warm heap chunks (the old staging-copy path's
//!                     best case).
//! - `MMAP_COLD`     — first sweep over a freshly-dropped mapping: pages are
//!                     NOT resident and PTEs absent, so every chunk pays disk
//!                     read + soft fault + H2D. The honest cold-start number.
//! - `MMAP_WARM_PTE` — second sweep, everything resident with PTEs installed.
//!                     The steady-state zero-copy number.
//! - `MMAP_POPULATE` — sweep after `MADV_POPULATE_READ`; isolates any
//!                     remaining per-chunk fault cost by removing it.
//!
//! Byte accounting: every phase counts its own completed iterations inside
//! its own timing window and reports bytes actually transferred.
//!
//! Run: cargo run --release -p hipfire-runtime --example h2d_mmap_bench -- <file>

use rdna_compute::Gpu;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: h2d_mmap_bench <file>");
    let mut gpu = Gpu::init().expect("gpu init");

    let size = std::fs::metadata(&path).expect("stat").len() as usize;
    println!("file: {} MiB", size >> 20);

    let chunk: usize = 64 << 20;
    if size < chunk {
        eprintln!("file smaller than one {chunk} MiB chunk; nothing to measure");
        return;
    }

    // One reusable destination; each upload overwrites it in full.
    let dst = {
        let warm = vec![0u8; chunk];
        gpu.upload_raw(&warm, &[chunk]).expect("alloc dst")
    };

    // (a) heap source — the old staging-copy path.
    let heap = vec![0x5au8; chunk];
    let mut reps = 0usize;
    let t0 = std::time::Instant::now();
    while reps * chunk < size {
        gpu.memcpy_htod_auto(&dst.buf, &heap).expect("up");
        reps += 1;
    }
    report("HEAP_SRC     ", reps * chunk, t0.elapsed());

    let file = std::fs::File::open(&path).expect("open");

    // (b) COLD: drop resident pages for this inode BEFORE mapping, then map
    // fresh so PTEs are absent too. `posix_fadvise(DONTNEED)` is only honored
    // while the file is NOT mmap'd, which is why the drop runs on the plain
    // handle before `Mmap::map`.
    {
        use std::os::unix::io::AsRawFd;
        let fd = file.as_raw_fd();
        let rc = unsafe { libc::posix_fadvise(fd, 0, size as _, libc::POSIX_FADV_DONTNEED) };
        if rc != 0 {
            eprintln!("warn: fadvise DONTNEED failed (rc={rc}); MMAP_COLD may be partially warm");
        }
    }
    let mmap = unsafe { memmap2::Mmap::map(&file).expect("mmap") };
    let (reps, dt) = sweep_mmap(&gpu, &dst, &mmap, chunk);
    report("MMAP_COLD    ", reps * chunk, dt);

    // (c) WARM PTEs: second sweep, pages resident + PTEs installed by (b).
    let (reps, dt) = sweep_mmap(&gpu, &dst, &mmap, chunk);
    report("MMAP_WARM_PTE", reps * chunk, dt);

    // (d) POPULATE_READ then sweep: pre-faults everything up front, so the
    // sweep itself carries zero fault cost. Delta vs (c) shows how much of
    // even the "warm" pass was still fault handling.
    unsafe {
        libc::madvise(
            mmap.as_ptr() as *mut libc::c_void,
            mmap.len(),
            libc::MADV_POPULATE_READ,
        );
    }
    let (reps, dt) = sweep_mmap(&gpu, &dst, &mmap, chunk);
    report("MMAP_POPULATE", reps * chunk, dt);
}

/// Sequential full-file sweep through the GPU copy engine, timed inside.
/// Returns completed chunk transfers and elapsed time so callers report
/// exact byte counts for exactly the measured window.
fn sweep_mmap(
    gpu: &Gpu,
    dst: &rdna_compute::GpuTensor,
    mmap: &[u8],
    chunk: usize,
) -> (usize, std::time::Duration) {
    let mut n = 0usize;
    let t0 = std::time::Instant::now();
    while n + chunk <= mmap.len() {
        gpu.memcpy_htod_auto(&dst.buf, &mmap[n..n + chunk])
            .expect("up");
        n += chunk;
    }
    let dt = t0.elapsed();
    (n / chunk, dt)
}

fn report(label: &str, bytes: usize, dt: std::time::Duration) {
    println!(
        "{label} {:8.2} GB/s ({} x 64 MiB in {:.3}s)",
        bytes as f64 / dt.as_secs_f64() / 1e9,
        bytes >> 26,
        dt.as_secs_f64()
    );
}
