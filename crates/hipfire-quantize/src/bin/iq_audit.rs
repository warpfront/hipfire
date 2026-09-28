//! iq_audit — 反量化内核差分台架（数据导出端）。
//!
//! 两种模式：
//!
//! 1. `iq_audit <model.gguf> <outdir> [win_elems=32768] [per_type=3]`
//!    对源 GGUF 里每种量化类型挑代表张量，按窗口（头/中/尾）抽块，导出：
//!      `<tag>.q`     窗口内的原始量化块字节（逐字节取自源文件）
//!      `<tag>.f32`   本仓库内核反量化出的 f32（raw little-endian）
//!      `<tag>.meta`  元信息
//!      `manifest.tsv`
//!
//! 2. `iq_audit --fromq <dir>`
//!    读 `<dir>/manifest.tsv` 里列出的每个 `<tag>.q`，用本仓库内核反量化，
//!    只写出 `<tag>.f32`。用途：合成窗口（模型里没有的类型，如 IQ1_S /
//!    IQ4_NL）由参考侧 `from_float_ref` 生成 `.q`，再由本模式补齐候选侧。
//!
//! 参考侧由 `107_iq_diff.py` 用 llama.cpp ggml 的 `to_float`（`libiqref.so`）
//! 在**同一份 `.q` 字节**上跑一遍，逐元素比对 —— 两侧输入完全一致，
//! 唯一变量就是"谁的数学对"。
//!
//! 为什么用 `#[path]` 引入模块：`iq_tables` / `iq_dequant` / `gguf_input` 目前是
//! 主 bin（`src/main.rs`）的私有模块。把本审计单独做成一个 bin，是为了随时可以
//! 整文件删除，且不改动生产代码的模块结构（便于上游 rebase）。

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

#[path = "../iq_tables.rs"]
mod iq_tables;
#[path = "../iq_dequant.rs"]
mod iq_dequant;
#[path = "../gguf_input.rs"]
mod gguf_input;

use gguf_input::{GgmlType, GgufFile, TensorInfo};

/// 把张量名转成安全的文件名片段（保留可读性，便于人眼定位）。
fn sanitize(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

/// 用给定字节反量化 n 个元素（走生产代码的装量分派）。
fn dequant_bytes(dtype: GgmlType, name: &str, data: &[u8], n: usize) -> Vec<f32> {
    let synth = TensorInfo {
        name: name.to_string(),
        shape: vec![n],
        dtype,
        offset: 0,
    };
    gguf_input::tensor_to_f32(&synth, data)
}

fn write_f32(path: &Path, out: &[f32]) {
    let mut fb = Vec::with_capacity(out.len() * 4);
    for v in out {
        fb.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(path, &fb).expect("write .f32");
}

/// 本仓库 `gguf_input::tensor_to_f32` 是否实现了该类型。
///
/// 台架侧复制一份这个判据（而不是给生产代码加 API），原因有二：
///   1. `--fromq` 遇到未实现类型时应当**跳过并报告**，而不是让整批 panic 掉
///      —— 合成集里可能故意塞未实现类型来探测"边界在哪"；
///   2. 保持审计文件自包含、可整目录删除，不改动生产模块的结构。
fn is_implemented(dtype: GgmlType) -> bool {
    matches!(
        dtype,
        GgmlType::F32
            | GgmlType::F16
            | GgmlType::BF16
            | GgmlType::Q4_0
            | GgmlType::Q8_0
            | GgmlType::Q1_0
            | GgmlType::Q2_0
            | GgmlType::Q2K
            | GgmlType::Q4K
            | GgmlType::Q5K
            | GgmlType::Q6K
    ) || dtype.iq_kind().is_some()
}

/// 模式 2：只对已存在的 `.q` 补出 `.f32`（合成窗口用）。
fn run_fromq(dir: &Path) {
    let man = fs::read_to_string(dir.join("manifest.tsv")).expect("read manifest.tsv");
    let mut n = 0usize;
    let mut skipped = 0usize;
    for line in man.lines().skip(1) {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 9 {
            continue;
        }
        let tag = f[0];
        let qpath = dir.join(format!("{tag}.q"));
        if !qpath.exists() {
            continue;
        }
        let tid: u32 = f[1].parse().expect("type id");
        let n_elems: usize = f[8].parse().expect("n_elems");
        let dtype = GgmlType::from_u32(tid).unwrap_or_else(|| panic!("unknown ggml type {tid}"));
        if !is_implemented(dtype) {
            eprintln!("  fromq: SKIP {tag} —— 本仓库未实现 {:?}", dtype);
            skipped += 1;
            continue;
        }
        let q = fs::read(&qpath).expect("read .q");
        let out = dequant_bytes(dtype, tag, &q, n_elems);
        assert_eq!(out.len(), n_elems, "dequant element count mismatch");
        write_f32(&dir.join(format!("{tag}.f32")), &out);
        eprintln!("  fromq: {tag} n={n_elems}");
        n += 1;
    }
    eprintln!(
        "fromq: wrote {n} .f32 into {}  (skipped {skipped})",
        dir.display()
    );
}

/// 模式 1：从 GGUF 抽窗口，导出 .q / .f32 / .meta / manifest。
fn run_gguf(gguf_path: &Path, outdir: &Path, win_elems: usize, per_type: usize) {
    fs::create_dir_all(outdir).expect("create outdir");
    let g = GgufFile::open(gguf_path).expect("open gguf");

    eprintln!(
        "iq_audit: {} ({} tensors) -> {}  win_elems={} per_type={}",
        gguf_path.display(),
        g.tensors.len(),
        outdir.display(),
        win_elems,
        per_type
    );

    let mut by_type: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (i, t) in g.tensors.iter().enumerate() {
        by_type.entry(t.dtype as u32).or_default().push(i);
    }

    let mut manifest = String::from(
        "tag\ttype_id\ttype_name\ttensor\tblock_elems\tblock_bytes\tblock_start\tnblocks\tn_elems\n",
    );
    let mut n_win = 0usize;

    for (type_id, idxs) in &by_type {
        // 未实现的类型直接跳过并报告 —— 台架的意义是验证已实现的内核，
        // 不该因为模型里多出一种没实现的类型而整批 panic。
        let dtype0 = g.tensors[idxs[0]].dtype;
        if !is_implemented(dtype0) {
            eprintln!(
                "  type {:>3}: SKIP {} 张量 —— 本仓库未实现 {:?}",
                type_id,
                idxs.len(),
                dtype0
            );
            continue;
        }

        // 代表张量选取：token_embd / output 优先（关键表），其余按字节数降序。
        let mut cand = idxs.clone();
        cand.sort_by_key(|&i| std::cmp::Reverse(g.tensors[i].byte_size()));
        let mut chosen: Vec<usize> = cand
            .iter()
            .copied()
            .filter(|&i| {
                let nm = &g.tensors[i].name;
                nm.contains("token_embd") || nm == "output.weight"
            })
            .collect();
        for &i in &cand {
            if chosen.len() >= per_type {
                break;
            }
            if !chosen.contains(&i) {
                chosen.push(i);
            }
        }
        chosen.truncate(per_type.max(1));

        eprintln!(
            "  type {:>3}: {} tensors, audit {:?}",
            type_id,
            idxs.len(),
            chosen
                .iter()
                .map(|&i| g.tensors[i].name.as_str())
                .collect::<Vec<_>>()
        );

        for &i in &chosen {
            let t = &g.tensors[i];
            let bs = t.dtype.block_size();
            let bb = t.dtype.block_bytes();
            if bs == 0 || bb == 0 {
                continue;
            }
            let raw = g.tensor_data(t);
            let nblocks = raw.len() / bb;
            if nblocks == 0 {
                continue;
            }
            let win_blocks = (win_elems / bs).max(1).min(nblocks);
            // 小张量整取；大张量取头 / 中 / 尾三个窗口（错位布局类 bug 只在某段暴露）
            let starts: Vec<usize> = if nblocks <= win_blocks * 3 {
                vec![0]
            } else {
                vec![0, (nblocks - win_blocks) / 2, nblocks - win_blocks]
            };

            for (wi, &s) in starts.iter().enumerate() {
                let qbytes = &raw[s * bb..(s + win_blocks) * bb];
                let n = win_blocks * bs;
                let out = dequant_bytes(t.dtype, &t.name, qbytes, n);
                assert_eq!(out.len(), n, "dequant element count mismatch");

                let tag = format!("{}_{}_w{}", type_id, sanitize(&t.name), wi);
                fs::write(outdir.join(format!("{tag}.q")), qbytes).expect("write .q");
                write_f32(&outdir.join(format!("{tag}.f32")), &out);
                let meta_txt = format!(
                    "type_id={}\ntype_name={}\ntensor={}\nblock_elems={}\n\
                     block_bytes={}\nblock_start={}\nnblocks={}\nn_elems={}\n\
                     total_blocks={}\ntotal_bytes={}\n",
                    type_id,
                    format!("{:?}", t.dtype),
                    t.name,
                    bs,
                    bb,
                    s,
                    win_blocks,
                    n,
                    nblocks,
                    raw.len()
                );
                fs::write(outdir.join(format!("{tag}.meta")), meta_txt).expect("write .meta");

                manifest.push_str(&format!(
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                    tag,
                    type_id,
                    format!("{:?}", t.dtype),
                    t.name,
                    bs,
                    bb,
                    s,
                    win_blocks,
                    n
                ));
                n_win += 1;
            }
        }
    }

    fs::write(outdir.join("manifest.tsv"), &manifest).expect("write manifest");
    eprintln!("iq_audit: wrote {n_win} windows to {}", outdir.display());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 3 && args[1] == "--fromq" {
        run_fromq(Path::new(&args[2]));
        return;
    }
    if args.len() < 3 {
        eprintln!("usage: iq_audit <model.gguf> <outdir> [win_elems] [per_type]");
        eprintln!("       iq_audit --fromq <dir>");
        std::process::exit(64);
    }
    let win_elems: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(32768);
    let per_type: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(3);
    run_gguf(Path::new(&args[1]), Path::new(&args[2]), win_elems, per_type);
}
