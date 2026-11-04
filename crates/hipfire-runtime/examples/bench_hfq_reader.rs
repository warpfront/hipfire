// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Throughput bench and byte-identity check of the HFQ direct reader
//! (`hfq_parallel::read_ranges_into_host` / `upload_ranges_to_device`):
//! `O_DIRECT` reads of a model file at a grid of lane counts and read sizes,
//! printing GB/s. Without `--gpu` no GPU is touched (plain heap destination);
//! with `--gpu` (run it inside a fabric grant) the destinations are the loader's
//! real ones: a pinned host-mapped owner (`host`) and a device buffer through
//! the pinned staging ring (`device`).
//!
//! ```text
//! cargo run --release -p hipfire-runtime --example bench_hfq_reader -- \
//!     --file MODEL.mq4 [--lanes 4,8,16,32] [--chunk-mib 4,8,16] \
//!     [--range-mib 300,18] [--gib 6] [--reps 2] [--verify] [--gpu]
//! ```
//!
//! `--range-mib` is the size of each `read_ranges_into_host` call: ~300 is the
//! host-mapped expert entries (`weight_store::read_host_range`), ~18 is a
//! VRAM tensor (`upload_ranges_to_device` reads that many bytes per call).
//! Before every run the region is dropped from the page cache, so every read is
//! a cold `O_DIRECT` read; successive runs use fresh regions of the file.
//! `--verify` reads six tensors with every grid tuning and compares SHA-256
//! against a plain buffered `pread` of the same bytes.

use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::hfq_parallel::{
    read_ranges_into_host, upload_ranges_to_device, DeviceRangeCopy, DirectReadStats, DirectSource,
    HostRangeCopy, ReaderTuning,
};
use hipfire_runtime::weight_store::sha256_hex;
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::path::PathBuf;
use std::time::Instant;

const MIB: usize = 1024 * 1024;
const GIB: u64 = 1 << 30;

struct Args {
    file: PathBuf,
    lanes: Vec<usize>,
    chunks_mib: Vec<usize>,
    range_mib: Vec<usize>,
    gib: u64,
    reps: usize,
    verify: bool,
    gpu: bool,
}

fn list(raw: &str) -> Vec<usize> {
    raw.split(',')
        .map(|part| part.trim().parse().expect("integer list"))
        .collect()
}

fn parse_args() -> Args {
    let mut args = Args {
        file: PathBuf::new(),
        lanes: vec![4, 8, 16, 32],
        chunks_mib: vec![4, 8, 16],
        range_mib: vec![300, 18],
        gib: 6,
        reps: 2,
        verify: false,
        gpu: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().unwrap_or_else(|| panic!("{flag} needs a value"));
        match flag.as_str() {
            "--file" => args.file = PathBuf::from(value()),
            "--lanes" => args.lanes = list(&value()),
            "--chunk-mib" => args.chunks_mib = list(&value()),
            "--range-mib" => args.range_mib = list(&value()),
            "--gib" => args.gib = value().parse().expect("--gib integer"),
            "--reps" => args.reps = value().parse().expect("--reps integer"),
            "--verify" => args.verify = true,
            "--gpu" => args.gpu = true,
            other => panic!("unknown argument {other}"),
        }
    }
    assert!(args.file.is_file(), "--file MODEL is required");
    args
}

fn drop_cache(file: &std::fs::File, offset: u64, len: u64) {
    // SAFETY: plain advisory syscall on an open descriptor.
    unsafe {
        libc::posix_fadvise(
            file.as_raw_fd(),
            offset as libc::off_t,
            len as libc::off_t,
            libc::POSIX_FADV_DONTNEED,
        );
    }
}

fn main() {
    let args = parse_args();
    let source = DirectSource::open_path(&args.file).expect("open source");
    let buffered = std::fs::File::open(&args.file).expect("open file");
    let file_len = source.file_identity().len;
    eprintln!(
        "file {} ({:.1} GiB), O_DIRECT available: {}",
        args.file.display(),
        file_len as f64 / GIB as f64,
        source.is_direct_available()
    );

    let region = args.gib * GIB;
    let slots = ((file_len - GIB) / region).max(1);
    let mut run_index = 0u64;
    let mut gpu = args
        .gpu
        .then(|| rdna_compute::Gpu::init().expect("initialize GPU"));
    println!("dest,range_mib,lanes,chunk_mib,rep,gbps,direct_pct,secs");
    let dests: &[&str] = if args.gpu {
        &["heap", "host", "device"]
    } else {
        &["heap"]
    };
    for &dest in dests {
        for &range_mib in &args.range_mib {
            let range = range_mib * MIB;
            let calls = (region as usize / range).max(1);
            // One destination per (dest, range): reused by every call, like an owner.
            let mut heap = vec![0u8; if dest == "heap" { range } else { 0 }];
            let mut host_tensor = None;
            let mut host_ptr = std::ptr::null_mut();
            let mut device_buf = None;
            match dest {
                "host" => {
                    let gpu = gpu.as_mut().unwrap();
                    let (tensor, ptr) = gpu.alloc_raw_host_mapped(range, &[range]).expect("pinned");
                    host_tensor = Some(tensor);
                    host_ptr = ptr;
                }
                "device" => {
                    device_buf = Some(gpu.as_ref().unwrap().hip.malloc(range).expect("device"));
                }
                _ => {}
            }
            for &chunk_mib in &args.chunks_mib {
                for &lanes in &args.lanes {
                    let tuning = ReaderTuning::new(lanes, chunk_mib * MIB);
                    for rep in 0..args.reps {
                        // Fresh 4 KiB-aligned region per run, past the header.
                        let base = GIB + (run_index % slots) * region;
                        run_index += 1;
                        drop_cache(&buffered, base, region);
                        let started = Instant::now();
                        let (mut direct, mut total) = (0u64, 0u64);
                        for call in 0..calls {
                            let offset = base + (call * range) as u64;
                            if offset + range as u64 > file_len {
                                break;
                            }
                            let stats: DirectReadStats = match dest {
                                "heap" | "host" => {
                                    let dst = if dest == "heap" {
                                        heap.as_mut_ptr()
                                    } else {
                                        host_ptr
                                    };
                                    let copies = [HostRangeCopy {
                                        file_offset: offset,
                                        len: range,
                                        dst,
                                    }];
                                    // SAFETY: `dst` is `range` writable bytes owned by this call.
                                    unsafe { read_ranges_into_host(&source, &copies, tuning) }
                                        .expect("read")
                                }
                                _ => {
                                    let copies = [DeviceRangeCopy {
                                        file_offset: offset,
                                        len: range,
                                        dst: device_buf.as_ref().unwrap(),
                                        dst_offset: 0,
                                    }];
                                    upload_ranges_to_device(
                                        gpu.as_mut().unwrap(),
                                        &source,
                                        &copies,
                                        tuning,
                                    )
                                    .expect("upload")
                                }
                            };
                            direct += stats.bytes_direct;
                            total += stats.bytes_direct + stats.bytes_buffered;
                        }
                        let secs = started.elapsed().as_secs_f64();
                        println!(
                            "{dest},{range_mib},{lanes},{chunk_mib},{rep},{:.2},{:.0},{secs:.2}",
                            total as f64 / secs / 1e9,
                            direct as f64 * 100.0 / total.max(1) as f64
                        );
                    }
                }
            }
            if let (Some(tensor), Some(gpu)) = (host_tensor, gpu.as_mut()) {
                let _ = gpu.free_tensor(tensor);
            }
        }
    }

    if args.verify {
        verify(&args, &source, &buffered);
    }
}

fn verify(args: &Args, source: &DirectSource, buffered: &std::fs::File) {
    let hfq = HfqFile::open(&args.file).expect("open HFQ");
    let tensors = hfq.tensors();
    let big: Vec<_> = tensors
        .iter()
        .filter(|tensor| tensor.data_size >= MIB)
        .collect();
    assert!(big.len() >= 6, "model has too few tensors to verify");
    let mut tunings = vec![ReaderTuning::LEGACY, ReaderTuning::DEFAULT];
    for &lanes in &args.lanes {
        for &chunk_mib in &args.chunks_mib {
            tunings.push(ReaderTuning::new(lanes, chunk_mib * MIB));
        }
    }
    tunings.dedup();
    let mut mismatches = 0;
    for pick in 0..6 {
        // Spread across the file; cap each tensor at 256 MiB to bound the run.
        let tensor = big[pick * (big.len() - 1) / 5];
        let len = tensor.data_size.min(256 * MIB);
        let offset = tensor.data_offset as u64;
        let mut want = vec![0u8; len];
        buffered.read_exact_at(&mut want, offset).expect("pread");
        let want_hash = sha256_hex(&want);
        for tuning in &tunings {
            drop_cache(buffered, offset, len as u64);
            let mut got = vec![0u8; len];
            let copies = [HostRangeCopy {
                file_offset: offset,
                len,
                dst: got.as_mut_ptr(),
            }];
            // SAFETY: `got` is `len` writable bytes owned by this call.
            unsafe { read_ranges_into_host(source, &copies, *tuning) }.expect("read");
            if sha256_hex(&got) != want_hash {
                mismatches += 1;
                println!(
                    "MISMATCH tensor {} lanes {} chunk {} MiB",
                    tensor.name,
                    tuning.lanes,
                    tuning.chunk_bytes / MIB
                );
            }
        }
        println!(
            "verify {} offset {offset} len {len} sha256 {want_hash} ({} tunings)",
            tensor.name,
            tunings.len()
        );
    }
    println!("verify: {mismatches} mismatches");
    assert_eq!(mismatches, 0);
}
