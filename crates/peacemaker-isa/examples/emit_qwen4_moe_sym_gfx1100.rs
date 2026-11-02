// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! B4-B emitter, certifier and P1 receipt writer for the gfx11 symmetric IU4 MoE kernels (gfx1100; the E8P-I4
//! module also for gfx1151) (`src/kernels/qwen4_moe_sym_gfx1100`, included below by `#[path]`).
//!
//! ```text
//! cargo run -p peacemaker-isa --release --features toolchain --example emit_qwen4_moe_sym_gfx1100 -- \
//!     [--arch gfx1100|gfx1151] [--codec legacy|e8p|all] [--out-dir REPO_ROOT] [--receipt FILE] [--work-dir DIR] [--evidence-dir DIR] [--seed N] [--p1-only]
//! ```
//!
//! `--codec legacy` (default, the original behaviour): three modules `kernels/qwen4_moe_iu4_sym{,_w2,_w3}_pm_gfx1100.hxaco`;
//! `--codec e8p`: the one E8P-I4 module `kernels/qwen4_moe_iu4_sym_e8p_pm_gfx1100.hxaco` (entries
//! `qwen4_moe_{gate_up_silu,down}_iu4_sym_e8p_pm_gfx1100[_acc]`, 1,536 B static LDS stage, default receipt
//! `rcpt/k2e8p.p1.json`); `all`: both. `--arch gfx1151` (default gfx1100) emits the same E8P module for the Halo
//! (`kernels/qwen4_moe_iu4_sym_e8p_pm_gfx1151.hxaco`, contracts `crates/peacemaker-isa/kernels/qwen4_moe_sym.gfx1151.*_e8p*.contract.json`,
//! negatives under `kernels/qwen4_moe_gfx1151_negative/`) and requires `--codec e8p`. Re-running `legacy` against the committed tree proves the old bytes are unchanged
//! (an identical existing artifact is accepted, a different one is an error). Each module has the
//! four entries `{gate_up, down} x {BF16, _acc}`. Per module: builder-proof inspection, independent
//! source wait/hazard replay, native build (+ ROCm assembler/linker `.text` identity), per-symbol M7
//! (byte-exact lift, wait/hazard/barrier/window analyses, no obligations), the committed-shape
//! contract certified through `toolchain::certify` (radiowave manifest, spill counts), the kernel
//! descriptor (denorm 3/3, wave32, no private/LDS), and assertions over the *real* `llvm-objdump`
//! disassembly (IU4 16x16x16 shape, signed operands, wait ledger replay, drained back-edges and
//! exit, no gfx12 waits / scratch / FP-mode writes).
//!
//! Mutations (`Negative`): extra modules per codec (legacy: four; E8P: `wrong-lut`, `wrong-sign-mask`, `reversed-epoch`,
//! `wrong-fold-epoch`, `wrong-scale-row`, `wrong-expert-pointer`), written to
//! `kernels/qwen4_moe_gfx1100_negative/<negative>/<module>.hxaco` with the *same entry names*. They are
//! intentionally broken, audited separately (structure only, no candidate contract / certificate)
//! and reported under `negatives`, never as candidate evidence.
//!
//! P1 arithmetic/address proofs come from `qwen4_moe_sym_gfx1100_p1.rs`. Existing artifacts are never
//! overwritten (an identical existing file is accepted, a different one is an error). The receipt is
//! written even when something fails; the exit code is 1 on any failure. No device is used.
#[cfg(not(feature = "toolchain"))]
fn main() {
    eprintln!("emit_qwen4_moe_sym_gfx1100 needs --features toolchain");
    std::process::exit(2);
}

#[cfg(feature = "toolchain")]
#[allow(dead_code, unused_imports)]
#[path = "../src/kernels/qwen4_moe_sym_gfx1100/mod.rs"]
mod gfx1100;

#[cfg(feature = "toolchain")]
#[allow(dead_code, unused_imports)]
#[path = "qwen4_moe_sym_gfx1100_p1.rs"]
mod p1;

#[cfg(feature = "toolchain")]
mod app {
    use super::gfx1100::{self, Codec, Kind, Negative, Spec};
    use super::p1;
    use peacemaker_isa::kernels::{iu4_gemm, qwen4_moe_sym};
    use peacemaker_isa::toolchain::{self, CountBound, IsaShapeContract, Toolchain};
    use peacemaker_isa::{ledger_replay, pm_check, Arch, Emitted};
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    /// The default target; `--arch gfx1151` selects the Halo E8P module (`--codec e8p` only).
    const DEFAULT_ARCH: Arch = Arch::Gfx1100;
    const DEFAULT_RECEIPT: &str = "/home/kaden/qcal/release-0.4.3/consumer24/rcpt/b4.p1.json";
    const DEFAULT_RECEIPT_E8P: &str =
        "/home/kaden/qcal/release-0.4.3/consumer24/rcpt/k2e8p.p1.json";
    const DEFAULT_RECEIPT_E8P_GFX1151: &str =
        "/home/kaden/qcal/release-0.4.3/halo-e8p-rcpt/k2e8p.gfx1151.p1.json";
    /// Immutable baselines (lead): the gfx1201 objects and the shared sources. Re-pinned on land-rc5 over beta 12daf4fe9:
    /// the objects are the exact beta 12daf4fe9 blobs (n_experts@60 guard; rc3/rc4 pins were
    /// fea044d00e2989172fb440aef1af2304c70ccbf5c663508c0c0ec912afb15481 / 83822312b26a2f90fa688e39f610f1b95528e035c5118946b7060df3cc254833).
    /// The gfx1201 pin was then moved by the gfx1201 NT8 `gate_up` epoch-lock bake (was
    /// 0eea1673a530880586dc0cd179b7a3ef3dfd491660dff439f7b54e921792791e).
    const PINNED_OBJECTS: [(&str, &str); 2] = [
        (
            "qwen4_moe_iu4_sym_pm_gfx1201.hxaco",
            "44177c21de25bec8502f63f318e682dc4fa1948420723c75dec14fa63cfed723",
        ),
        (
            "qwen4_moe_iu4_sym_w3_pm_gfx1201.hxaco",
            "b1c05e645d424491e3d1dd69e6145ab2d6081fb97474662fd1fa14351ba5200f",
        ),
    ];
    const PINNED_SOURCES: [(&str, &str); 2] = [
        // Re-pinned on land2 (gfx1201 NT8 gate_up lock bake, 75f0d90173): qwen4_moe_sym.rs is
        // f75b549cfc019fb6680b1fd7200a36285342a25ad757b66bde3bf5fb23b7864c; earlier pin below.
        // Re-pinned on land-rc5: qwen4_moe_sym.rs is beta 12daf4fe9 (cdb4f9c6a9ca0c4017aa17e88d134cf3e22fe3fced04e3896f2dc7b5a04c446f)
        // plus the retained consumer24 E8P dispatch; qwen4_moe_sym_w3.rs is exactly beta 12daf4fe9. rc3/rc4 pins:
        // 02ac151d8c548badb646ada96eec4516d1c3258528dad8f1fa4984ed7043c00e, b15ef478422178342b6cba2a021555dcc837aba604e95e3ce0a2008911b3bbce.
        (
            "src/kernels/qwen4_moe_sym.rs",
            "f75b549cfc019fb6680b1fd7200a36285342a25ad757b66bde3bf5fb23b7864c",
        ),
        (
            "src/kernels/qwen4_moe_sym_w3.rs",
            "e236113e61660dfc4a6fb0523661af748cdd3446b19c05110b3f5b1b7d1469c6",
        ),
    ];

    type R<T> = Result<T, String>;

    /// Which codec modules a run emits: the committed MQ4/W2/W3 set (default), the E8P module, or both.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum CodecSet {
        Legacy,
        E8p,
        All,
    }

    impl CodecSet {
        fn codecs(self) -> Vec<Codec> {
            match self {
                Self::Legacy => Codec::ALL.to_vec(),
                Self::E8p => vec![Codec::E8p],
                Self::All => Codec::ALL.iter().copied().chain([Codec::E8p]).collect(),
            }
        }
    }

    /// The mutations built for `codec` (E8P: `wrong-lut`, `wrong-sign-mask`, `reversed-epoch`, `wrong-fold-epoch`,
    /// `wrong-scale-row`, `wrong-expert-pointer`; the others: the four B4 mutations).
    fn negatives_of(codec: Codec) -> Vec<Negative> {
        if codec == Codec::E8p {
            Negative::E8P.to_vec()
        } else {
            Negative::ALL.to_vec()
        }
    }

    struct Opts {
        arch: Arch,
        out_dir: PathBuf,
        receipt: PathBuf,
        work_dir: PathBuf,
        evidence_dir: Option<PathBuf>,
        seed: u64,
        p1_only: bool,
        codecs: CodecSet,
    }

    struct Ctx {
        arch: Arch,
        root: PathBuf,
        work: PathBuf,
        evidence: Option<PathBuf>,
        tc: Toolchain,
    }

    struct Failure {
        scope: String,
        message: String,
    }

    // --------------------------------------------------------------------------------------
    // small utilities

    fn sha256(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn sha256_file(path: &Path) -> R<String> {
        std::fs::read(path)
            .map(|b| sha256(&b))
            .map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Create `path` with `bytes`. An existing file is never replaced: identical content is accepted,
    /// different content is an error.
    fn write_new(path: &Path, bytes: &[u8]) -> R<&'static str> {
        use std::io::Write;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
        {
            Ok(mut f) => {
                f.write_all(bytes)
                    .map_err(|e| format!("{}: {e}", path.display()))?;
                Ok("created")
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                let existing =
                    std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
                if existing == bytes {
                    Ok("identical-existing")
                } else {
                    Err(format!("{}: exists with different content (existing artifacts are never overwritten)", path.display()))
                }
            }
            Err(e) => Err(format!("{}: {e}", path.display())),
        }
    }

    /// `.text` of an ELF, or of the device ELF inside a clang offload bundle.
    fn text_section(bytes: &[u8]) -> R<Vec<u8>> {
        let start = bytes
            .windows(4)
            .position(|w| w == b"\x7fELF")
            .ok_or("no ELF in object")?;
        let e = &bytes[start..];
        let rd = |o: usize, n: usize| e.get(o..o + n).ok_or_else(|| "truncated ELF".to_string());
        let u16_ = |o: usize| rd(o, 2).map(|b| u16::from_le_bytes([b[0], b[1]]) as usize);
        let u32_ = |o: usize| rd(o, 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize);
        let u64_ = |o: usize| rd(o, 8).map(|b| u64::from_le_bytes(b.try_into().unwrap()) as usize);
        let (shoff, shentsize, shnum, shstrndx) =
            (u64_(0x28)?, u16_(0x3a)?, u16_(0x3c)?, u16_(0x3e)?);
        let sh = |i: usize| shoff + i * shentsize;
        let names = u64_(sh(shstrndx) + 24)?;
        for i in 0..shnum {
            let name = e
                .get(names + u32_(sh(i))?..)
                .ok_or("truncated section names")?;
            if name.starts_with(b".text\0") {
                let (off, len) = (u64_(sh(i) + 24)?, u64_(sh(i) + 32)?);
                return e
                    .get(off..off + len)
                    .map(<[u8]>::to_vec)
                    .ok_or_else(|| "truncated .text".into());
            }
        }
        Err("no .text section".into())
    }

    /// `.max_flat_workgroup_size` / `.kernarg_segment_size` of `symbol` in the module's code-object metadata.
    fn metadata_u32(source: &str, symbol: &str, key: &str) -> R<u32> {
        let lines: Vec<&str> = source.lines().map(str::trim).collect();
        let name = lines
            .iter()
            .position(|l| l.strip_prefix(".name:").is_some_and(|n| n.trim() == symbol))
            .ok_or_else(|| format!("{symbol}: no metadata entry"))?;
        lines[..name]
            .iter()
            .rev()
            .take_while(|l| !l.starts_with("- "))
            .find_map(|l| l.strip_prefix(key))
            .and_then(|n| n.trim().parse().ok())
            .ok_or_else(|| format!("{symbol}: no {key} in metadata"))
    }

    fn find_key<'a>(v: &'a Value, key: &str) -> Option<&'a Value> {
        match v {
            Value::Object(m) => m
                .get(key)
                .or_else(|| m.values().find_map(|x| find_key(x, key))),
            Value::Array(a) => a.iter().find_map(|x| find_key(x, key)),
            _ => None,
        }
    }

    fn git_head_blob(root: &Path, rel: &str) -> Option<Vec<u8>> {
        let out = Command::new("git")
            .arg("-C")
            .arg(root)
            .arg("show")
            .arg(format!("HEAD:{rel}"))
            .output()
            .ok()?;
        out.status.success().then_some(out.stdout)
    }

    // --------------------------------------------------------------------------------------
    // entries, contracts
    struct Entry {
        arch: Arch,
        codec: Codec,
        kind: Kind,
        acc: bool,
        symbol: String,
        threads: u32,
    }

    /// Opcode families only the E8P entries (LDS table stage) may contain.
    const E8P_ONLY_FORBIDDEN: [&str; 6] = [
        "ds_load*",
        "ds_store*",
        "ds_read*",
        "ds_write*",
        "s_barrier*",
        "v_writelane*",
    ];

    fn mk(arch: Arch, codec: Codec, kind: Kind, acc: bool, negative: Negative) -> Spec {
        Spec {
            arch,
            codec,
            kind,
            acc,
            negative,
        }
    }

    /// Module order: BF16 gate/up, BF16 down, then the `_acc` pair.
    fn entry_keys() -> Vec<(Kind, bool)> {
        let mut v = Vec::new();
        for acc in [false, true] {
            for kind in Kind::ALL {
                v.push((kind, acc));
            }
        }
        v
    }

    fn variant_tag(codec: Codec, kind: Kind, acc: bool) -> String {
        format!(
            "{}{}{}",
            kind.tag(),
            codec.infix(),
            if acc { "_acc" } else { "" }
        )
    }

    fn tensors(kind: Kind) -> u32 {
        if kind == Kind::GateUp {
            2
        } else {
            1
        }
    }

    const COUNTED: [&str; 23] = [
        "v_wmma_i32_16x16x16_iu4",
        "buffer_load_b32",
        "buffer_load_b64",
        "buffer_load_b96",
        "buffer_load_b128",
        "buffer_load_u8",
        "buffer_load_i8",
        "buffer_load_u16",
        "buffer_load_i16",
        "buffer_store_b8",
        "buffer_store_b16",
        "buffer_store_b32",
        "buffer_store_b64",
        "buffer_store_b128",
        "ds_swizzle_b32",
        "v_permlanex16_b32",
        "s_load_b32",
        "s_load_b64",
        "s_load_b128",
        "s_load_b256",
        "s_waitcnt",
        "s_waitcnt_vscnt",
        "s_endpgm",
    ];
    /// Additionally counted in E8P entries only (the committed MQ4/W2/W3 contracts are unchanged).
    const E8P_COUNTED: [&str; 5] = [
        "ds_load_b32",
        "ds_store_b32",
        "s_barrier",
        "v_writelane_b32",
        "v_add3_u32",
    ];

    /// Disassembler spelling of the legacy mnemonic aliases.
    fn canonical(name: &str) -> &str {
        match name {
            "buffer_load_dword" => "buffer_load_b32",
            "buffer_load_dwordx2" => "buffer_load_b64",
            "buffer_load_dwordx3" => "buffer_load_b96",
            "buffer_load_dwordx4" => "buffer_load_b128",
            "buffer_load_ubyte" => "buffer_load_u8",
            "buffer_load_sbyte" => "buffer_load_i8",
            "buffer_load_ushort" => "buffer_load_u16",
            "buffer_load_sshort" => "buffer_load_i16",
            "buffer_store_byte" => "buffer_store_b8",
            "buffer_store_short" => "buffer_store_b16",
            "buffer_store_dword" => "buffer_store_b32",
            "buffer_store_dwordx2" => "buffer_store_b64",
            "buffer_store_dwordx4" => "buffer_store_b128",
            other => other,
        }
    }

    /// The per-entry shape contract, in the repository's convention (`kernels/*.contract.json`):
    /// exact counts of the characteristic instructions of the authored source, the forbidden families
    /// and the resource ceilings. `toolchain::certify` checks it against the linked object's
    /// disassembly, so a source/object disagreement is a certification failure.
    fn contract(spec: &Entry, e: &Emitted) -> IsaShapeContract {
        let mut counts: BTreeMap<String, u32> = BTreeMap::new();
        let mut vopd = 0u32;
        for line in e.s_text.lines() {
            let insn = line
                .split("//")
                .next()
                .unwrap_or("")
                .split(';')
                .next()
                .unwrap_or("")
                .trim();
            let name = canonical(insn.split_whitespace().next().unwrap_or(""));
            if COUNTED.contains(&name) || (spec.codec == Codec::E8p && E8P_COUNTED.contains(&name))
            {
                *counts.entry(name.to_owned()).or_default() += 1;
            } else if name.starts_with("v_dual_") {
                vopd += 1;
                *counts.entry(name.to_owned()).or_default() += 1;
            }
        }
        if vopd > 0 {
            counts.insert("vopd_packets".into(), vopd);
        }
        IsaShapeContract {
            symbol: spec.symbol.clone(),
            start_label: None,
            end_label: None,
            counts: counts
                .into_iter()
                .map(|(k, v)| (k, CountBound::Exact(v)))
                .collect(),
            forbidden: [
                "scratch_*",
                "flat_*",
                "global_*",
                "ds_load*",
                "ds_store*",
                "ds_read*",
                "ds_write*",
                "s_barrier*",
                "v_readlane*",
                "v_writelane*",
                "s_wait_*",
                "s_setpc*",
                "s_swappc*",
                "s_denorm_mode*",
                "s_round_mode*",
                "s_setreg*",
                "v_wmma_i32_16x16x32*",
                "v_wmma_i32_16x16x16_iu8*",
                "v_wmma_f*",
                "v_wmma_bf16*",
                "v_swmmac*",
            ]
            .iter()
            .filter(|s| spec.codec != Codec::E8p || !E8P_ONLY_FORBIDDEN.contains(*s))
            // The BF16 MQ4 GateUp/Down entries own workgroup barriers (lockstep / K=640 pipeline).
            .filter(|s| !(spec.codec == Codec::Mq4 && !spec.acc && **s == "s_barrier*"))
            .map(|s| (*s).to_owned())
            .collect(),
            vgpr_max: Some(u32::from(e.shape.next_free_vgpr).div_ceil(8) * 8),
            sgpr_max: Some((u32::from(e.shape.next_free_sgpr).div_ceil(8) * 8).max(104)),
            require_wave32: true,
            require_zero_spills: true,
            require_zero_private: true,
            launch_dynamic_lds_bytes: Some(0),
            group_segment_fixed_bytes: spec.codec.lds_bytes(),
        }
    }

    // --------------------------------------------------------------------------------------
    // disassembly audit

    struct Insn {
        addr: u64,
        mnemonic: String,
        operands: String,
    }

    /// Instructions of `symbol` in `llvm-objdump --disassemble` output (`<insn> // <addr>: <bytes>`).
    fn symbol_insns(dis: &str, symbol: &str) -> R<Vec<Insn>> {
        let mut inside = false;
        let mut saw = false;
        let mut out = Vec::new();
        for line in dis.lines() {
            let t = line.trim();
            if t.ends_with(">:") {
                if let Some((_, name)) = t.split_once('<') {
                    let name = name.trim_end_matches(">:");
                    if name == symbol {
                        inside = true;
                        saw = true;
                        continue;
                    } else if inside && !name.starts_with(".L") && !name.contains('+') {
                        break;
                    }
                }
            }
            if !inside {
                continue;
            }
            let Some((text, comment)) = t.split_once("//") else {
                continue;
            };
            let text = text.trim();
            if text.is_empty() || text.starts_with('.') {
                continue;
            }
            let addr_text = comment.trim().split(':').next().unwrap_or("");
            let addr = u64::from_str_radix(addr_text.trim(), 16)
                .map_err(|_| format!("{symbol}: unparsable disassembly line `{t}`"))?;
            let (mnemonic, operands) = match text.split_once(char::is_whitespace) {
                Some((m, o)) => (m.to_owned(), o.trim().to_owned()),
                None => (text.to_owned(), String::new()),
            };
            out.push(Insn {
                addr,
                mnemonic,
                operands,
            });
        }
        if !saw {
            return Err(format!("{symbol}: not found in the object's disassembly"));
        }
        if out.is_empty() {
            return Err(format!(
                "{symbol}: no instructions in the object's disassembly"
            ));
        }
        Ok(out)
    }

    fn parse_int(t: &str) -> Option<i64> {
        let t = t.trim().trim_end_matches(',');
        if let Some(h) = t.strip_prefix("0x") {
            i64::from_str_radix(h, 16).ok()
        } else {
            t.parse().ok()
        }
    }

    /// `(vmcnt, lgkmcnt)` a gfx11 `s_waitcnt` waits down to (`None` = not waited).
    fn waitcnt_fields(operands: &str) -> Option<(Option<u32>, Option<u32>)> {
        let t = operands.trim();
        if t.starts_with("0x") || t.chars().next().is_some_and(|c| c.is_ascii_digit()) {
            let imm = parse_int(t)? as u32;
            let (vm, lgkm) = ((imm >> 10) & 0x3f, (imm >> 4) & 0x3f);
            return Some(((vm != 0x3f).then_some(vm), (lgkm != 0x3f).then_some(lgkm)));
        }
        let (mut vm, mut lgkm) = (None, None);
        for field in t
            .split(|c: char| c.is_whitespace() || c == '&' || c == ',')
            .filter(|f| !f.is_empty())
        {
            let (name, value) = field.split_once('(')?;
            let value: u32 = value.strip_suffix(')')?.parse().ok()?;
            match name {
                "vmcnt" => vm = Some(value),
                "lgkmcnt" => lgkm = Some(value),
                "expcnt" => {}
                _ => return None,
            }
        }
        Some((vm, lgkm))
    }

    struct ObjectAudit {
        json: Value,
    }

    /// Assertions over the real linked object of one entry: its disassembly and kernel descriptor.
    fn audit_object(
        spec: &Entry,
        symbol: &str,
        dis: &str,
        elf: &Path,
        module_text: &str,
        negative: Negative,
    ) -> R<ObjectAudit> {
        let insns = symbol_insns(dis, symbol)?;
        let mut mnemonics: BTreeMap<&str, u32> = BTreeMap::new();
        for i in &insns {
            *mnemonics.entry(i.mnemonic.as_str()).or_default() += 1;
        }
        let count = |m: &str| mnemonics.get(m).copied().unwrap_or(0);
        let tensor_n = tensors(spec.kind);

        // Matrix shape: only the signed 16x16x16 IU4 WMMA, a whole number of K128 epochs of 8 K16 steps.
        let mut wmma = 0u32;
        for i in insns
            .iter()
            .filter(|i| i.mnemonic.starts_with("v_wmma_") || i.mnemonic.starts_with("v_swmmac_"))
        {
            if i.mnemonic != "v_wmma_i32_16x16x16_iu4" {
                return Err(format!(
                    "{symbol}: unexpected matrix instruction {} (only v_wmma_i32_16x16x16_iu4)",
                    i.mnemonic
                ));
            }
            if !i.operands.contains("neg_lo:[1,1") {
                return Err(format!(
                    "{symbol}: WMMA at {:#x} is not signed A and B: `{}`",
                    i.addr, i.operands
                ));
            }
            wmma += 1;
        }
        if wmma == 0 || wmma % (8 * tensor_n) != 0 {
            return Err(format!("{symbol}: {wmma} WMMA instructions, not a positive multiple of {} (8 K16 steps x {tensor_n} tensors)", 8 * tensor_n));
        }

        // No gfx12 waits, scratch, flat/global, barriers, lane moves, FP-mode writes; LDS data and barriers only in E8P.
        let e8p = spec.codec == Codec::E8p;
        // The BF16 MQ4 GateUp entry owns the per-epoch lockstep barrier (2 per K loop trip) and the
        // BF16 MQ4 Down entry the unrolled K=640 pipeline (5, one after each epoch's issue).
        let lockstep = spec.codec == Codec::Mq4 && !spec.acc;
        let barriers_expected = match (lockstep, spec.kind) {
            (true, Kind::GateUp) => 2,
            (true, Kind::Down) if negative == Negative::None => 5,
            _ => 0,
        };
        for i in &insns {
            let m = i.mnemonic.as_str();
            let bad = m.starts_with("s_wait_")
                || m.starts_with("scratch_")
                || m.starts_with("flat_")
                || m.starts_with("global_")
                || (!e8p && m.starts_with("ds_load"))
                || (!e8p && m.starts_with("ds_store"))
                || (!e8p && barriers_expected == 0 && m.starts_with("s_barrier"))
                || m.starts_with("v_readlane")
                || (!e8p && m.starts_with("v_writelane"))
                || m == "s_denorm_mode"
                || m == "s_round_mode"
                || m.starts_with("s_setreg");
            if bad {
                return Err(format!(
                    "{symbol}: forbidden instruction {m} at {:#x}",
                    i.addr
                ));
            }
        }
        if !e8p && barriers_expected != 0 && count("s_barrier") != barriers_expected {
            return Err(format!(
                "{symbol}: {} s_barrier, the lockstep design requires exactly {barriers_expected}",
                count("s_barrier")
            ));
        }
        // E8P: exactly the stage and unpack census of the authored design (see `e8p.rs`).
        if e8p {
            let waves = spec.threads / 32;
            let words = 8 * 2 * tensor_n; // eight K16 steps x two words x tensors
            let want = [
                ("ds_load_b32", 2 * words),
                ("ds_store_b32", 128 / (32 * waves) + 8),
                ("s_barrier", 1),
                ("v_writelane_b32", 8 * 32),
                ("v_wmma_i32_16x16x16_iu4", 8 * tensor_n),
            ];
            for (name, expected) in want {
                if count(name) != expected {
                    return Err(format!(
                        "{symbol}: {} {name}, E8P census requires exactly {expected}",
                        count(name)
                    ));
                }
            }
        }

        // 16-bit scale loads exist for every tensor.
        let u16_loads = count("buffer_load_u16");
        if u16_loads < tensor_n {
            return Err(format!("{symbol}: {u16_loads} buffer_load_u16, need at least {tensor_n} (one 16-bit scale load per tensor)"));
        }

        // Wait ledger replayed over the *disassembled* object (not the authored source).
        let normalised: String = insns
            .iter()
            .map(|i| format!("{} {}\n", i.mnemonic, i.operands))
            .collect();
        ledger_replay::replay_waits(&normalised, spec.arch)
            .map_err(|e| format!("{symbol}: object wait replay: {e}"))?;
        let hazards = ledger_replay::replay_hazards(&normalised, spec.arch)
            .map_err(|e| format!("{symbol}: object hazard replay: {e}"))?;
        if !hazards.is_empty() {
            return Err(format!(
                "{symbol}: object hazard replay found {}: {:?}",
                hazards.len(),
                hazards.first()
            ));
        }

        // Drained at every back-edge and at the exit (linear scan; M7 is the path-exact analysis).
        let (mut vm, mut st, mut lgkm, mut smem) = (0u32, 0u32, 0u32, 0u32);
        let (mut back_edges, mut exits, mut waits) = (0u32, 0u32, 0u32);
        for i in &insns {
            let m = i.mnemonic.as_str();
            if m.starts_with("buffer_load") {
                vm += 1;
            } else if m.starts_with("buffer_store") {
                st += 1;
            } else if m.starts_with("ds_") {
                lgkm += 1;
            } else if m.starts_with("s_load")
                || m.starts_with("s_buffer_load")
                || m.starts_with("s_sendmsg_rtn")
            {
                lgkm += 1;
                smem += 1;
            } else if m == "s_waitcnt" {
                waits += 1;
                let (v, l) = waitcnt_fields(&i.operands).ok_or_else(|| {
                    format!(
                        "{symbol}: unparsable wait `{}` at {:#x}",
                        i.operands, i.addr
                    )
                })?;
                if let Some(n) = v {
                    vm = vm.min(n);
                }
                if let Some(n) = l {
                    if n == 0 {
                        lgkm = 0;
                        smem = 0;
                    } else if smem == 0 {
                        lgkm = lgkm.min(n);
                    }
                }
            } else if m == "s_waitcnt_vscnt" {
                waits += 1;
                let n = i
                    .operands
                    .rsplit(',')
                    .next()
                    .and_then(parse_int)
                    .ok_or_else(|| format!("{symbol}: unparsable vscnt wait `{}`", i.operands))?;
                st = st.min(n as u32);
            } else if m == "s_branch" || m.starts_with("s_cbranch_") {
                let simm = i.operands.split_whitespace().next().and_then(parse_int);
                let Some(simm) = simm else {
                    return Err(format!(
                        "{symbol}: unresolved branch target `{}` at {:#x}",
                        i.operands, i.addr
                    ));
                };
                let target = i.addr as i64 + 4 + 4 * i64::from(simm as u16 as i16);
                if target <= i.addr as i64 {
                    back_edges += 1;
                    if vm != 0 || lgkm != 0 || st != 0 {
                        return Err(format!("{symbol}: back-edge at {:#x} with outstanding vm={vm} lgkm={lgkm} stores={st}", i.addr));
                    }
                }
            } else if m == "s_endpgm" {
                exits += 1;
                if vm != 0 || lgkm != 0 {
                    return Err(format!(
                        "{symbol}: s_endpgm at {:#x} with outstanding vm={vm} lgkm={lgkm}",
                        i.addr
                    ));
                }
            }
        }
        if exits == 0 {
            return Err(format!("{symbol}: no s_endpgm"));
        }
        if back_edges == 0 {
            return Err(format!("{symbol}: no K-loop back-edge"));
        }
        if count("buffer_load_b32")
            + count("buffer_load_b64")
            + count("buffer_load_b128")
            + u16_loads
            > 0
            && waits == 0
        {
            return Err(format!("{symbol}: loads without any wait"));
        }

        // Kernel descriptor of the linked ELF: denorm 3/3, wave32, no private memory or LDS.
        let kd = toolchain::read_kd(elf, symbol)?;
        let (d32, d1664) = (
            (kd.compute_pgm_rsrc1 >> 16) & 3,
            (kd.compute_pgm_rsrc1 >> 18) & 3,
        );
        if (d32, d1664) != (3, 3) {
            return Err(format!(
                "{symbol}: descriptor denorm modes {d32}/{d1664}, required 3/3 (rsrc1 {:#010x})",
                kd.compute_pgm_rsrc1
            ));
        }
        // The static LDS is exactly the E8P table stage and nothing else; no private memory.
        if !kd.wave32
            || kd.private_segment_size != 0
            || kd.group_segment_size != spec.codec.lds_bytes()
        {
            return Err(format!(
                "{symbol}: descriptor wave32={} private={} group={} (expected group {})",
                kd.wave32,
                kd.private_segment_size,
                kd.group_segment_size,
                spec.codec.lds_bytes()
            ));
        }
        // rc5 guard: `n_experts` (kernarg 0x3c) is loaded into s5 and the routed expert id s23 is range-checked with ONE
        // unsigned compare, which branches to the no-store exit before any expert-pointer load (`s[58:59]`), buffer/LDS
        // access or barrier. The old signed negative compare is gone. The compare reads the actual routing id (s23),
        // never a pointer-index mutation, so `wrong-expert-pointer` is guarded identically.
        let norm = |s: &str| s.chars().filter(|c| !c.is_whitespace()).collect::<String>();
        let guard = insns
            .iter()
            .position(|i| i.mnemonic == "s_cmp_ge_u32" && norm(&i.operands) == "s23,s5")
            .ok_or_else(|| format!("{symbol}: no `s_cmp_ge_u32 s23, s5` expert-range guard"))?;
        if insns
            .iter()
            .filter(|i| i.mnemonic == "s_cmp_ge_u32" && norm(&i.operands) == "s23,s5")
            .count()
            != 1
        {
            return Err(format!("{symbol}: expert-range guard is not unique"));
        }
        if insns
            .iter()
            .any(|i| i.mnemonic == "s_cmp_lt_i32" && norm(&i.operands) == "s23,0")
        {
            return Err(format!(
                "{symbol}: the old signed negative expert compare survives"
            ));
        }
        let narg = insns
            .iter()
            .position(|i| i.mnemonic == "s_load_b32" && norm(&i.operands) == "s5,s[0:1],0x3c")
            .ok_or_else(|| {
                format!("{symbol}: no n_experts load (`s_load_b32 s5, s[0:1], 0x3c`)")
            })?;
        if narg >= guard {
            return Err(format!(
                "{symbol}: n_experts load follows the expert-range guard"
            ));
        }
        let branch = insns
            .get(guard + 1)
            .filter(|i| i.mnemonic == "s_cbranch_scc1" || i.mnemonic == "s_cbranch_scc0");
        if branch.is_none() {
            return Err(format!(
                "{symbol}: the expert-range guard is not followed by its exit branch"
            ));
        }
        for i in &insns[..=guard] {
            let m = i.mnemonic.as_str();
            if m.starts_with("buffer_")
                || m.starts_with("ds_")
                || m.starts_with("s_barrier")
                || (m.starts_with("s_load") && norm(&i.operands).starts_with("s[58:59],"))
            {
                return Err(format!(
                    "{symbol}: {m} at {:#x} precedes the expert-range guard",
                    i.addr
                ));
            }
        }
        if metadata_u32(module_text, symbol, ".kernarg_segment_size:")? != 64 {
            return Err(format!(
                "{symbol}: kernarg segment is not the frozen 64 bytes"
            ));
        }
        if metadata_u32(module_text, symbol, ".max_flat_workgroup_size:")? != spec.threads {
            return Err(format!(
                "{symbol}: workgroup size differs from Spec::threads"
            ));
        }

        Ok(ObjectAudit {
            json: json!({
                "instructions": insns.len(), "wmma": wmma, "wmma_shape": "v_wmma_i32_16x16x16_iu4 neg_lo:[1,1,0]",
                "wmma_k16_steps_per_tensor_epoch": 8, "tensors": tensor_n,
                "buffer_load_u16": u16_loads, "buffer_load_b32": count("buffer_load_b32"), "buffer_load_b64": count("buffer_load_b64"),
                "buffer_load_b128": count("buffer_load_b128"), "buffer_store_b128": count("buffer_store_b128"),
                "ds_swizzle_b32": count("ds_swizzle_b32"), "v_permlanex16_b32": count("v_permlanex16_b32"),
                "ds_load_b32": count("ds_load_b32"), "ds_store_b32": count("ds_store_b32"), "s_barrier": count("s_barrier"),
                "v_writelane_b32": count("v_writelane_b32"),
                "literal_0x88888888_operands": insns.iter().filter(|i| i.operands.contains("0x88888888")).count(),
                "waits": waits, "back_edges": back_edges, "s_endpgm": exits,
                "object_wait_replay": "clean", "object_hazard_replay": "clean",
                "descriptor": {
                    "compute_pgm_rsrc1": format!("{:#010x}", kd.compute_pgm_rsrc1),
                    "float_denorm_mode_32": d32, "float_denorm_mode_16_64": d1664,
                    "wave32": kd.wave32, "private_segment_size": kd.private_segment_size, "group_segment_size": kd.group_segment_size,
                },
                "kernarg_segment_size": 64,
            }),
        })
    }

    // --------------------------------------------------------------------------------------
    // one module

    struct Built {
        json: Value,
        /// symbol -> source text sha256 of the entry (for mutation comparison)
        entry_sha: BTreeMap<String, String>,
        module_text_sha: String,
    }

    fn frozen_kernargs(kind: Kind) -> Value {
        let frozen = qwen4_moe_sym::Spec {
            arch: Arch::Gfx1151,
            kind,
            nt: 1,
            rr: 1,
            codec: qwen4_moe_sym::WeightCodec::Mq4Sym,
        };
        serde_json::to_value(frozen.kernargs()).expect("kernargs serialise")
    }

    /// Emit, audit and build one module. `candidate == false` audits an intentionally broken module:
    /// same structural audits, no candidate shape contract or certificate, outputs under the negative directory.
    fn build_module(
        ctx: &Ctx,
        codec: Codec,
        negative: Negative,
        reference: Option<&BTreeMap<String, String>>,
    ) -> R<Built> {
        let candidate = negative == Negative::None;
        let module = codec.module(ctx.arch);
        let want_module = format!("qwen4_moe_iu4_sym{}_pm_{}", codec.infix(), ctx.arch.name());
        if module != want_module {
            return Err(format!(
                "Codec::module() = {module}, expected {want_module}"
            ));
        }
        let dir = ctx.work.join(negative.tag());
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;

        // 1. Emit the four entries.
        let mut emitted = Vec::new();
        let mut entry_notes: Vec<Option<&'static str>> = Vec::new();
        for (kind, acc) in entry_keys() {
            let symbol = mk(ctx.arch, codec, kind, acc, negative).symbol();
            let mut note = None;
            let e = match gfx1100::emit(mk(ctx.arch, codec, kind, acc, negative)) {
                Ok(e) => e,
                Err(err) if acc && negative == Negative::WrongPackSelector => {
                    // The `_acc` entries have no pack exchange: nothing to mutate; emitted as the candidate.
                    note = Some(
                        "not-applicable: the _acc entry has no pack exchange; emitted unmutated",
                    );
                    gfx1100::emit(mk(ctx.arch, codec, kind, acc, Negative::None))
                        .map_err(|e2| format!("{symbol}: emit failed for the mutation ({err}) and the unmutated entry ({e2})"))?
                }
                Err(err) => return Err(format!("{symbol}: emit failed: {err}")),
            };
            // Same entry names in every module of every mutation.
            if !e.s_text.lines().any(|l| {
                l.trim()
                    .strip_prefix(".name:")
                    .is_some_and(|n| n.trim() == symbol)
            }) {
                return Err(format!(
                    "{symbol}: emitted source has no kernel entry of that name"
                ));
            }
            if e.proof.arch != ctx.arch {
                return Err(format!(
                    "{symbol}: built for {}, not {}",
                    e.proof.arch.name(),
                    ctx.arch.name()
                ));
            }
            // Frozen 64-byte ABI (60 bytes plus n_experts@60), identical to the gfx1151 MQ4 entry's layout.
            let layout = mk(ctx.arch, codec, kind, acc, negative).kernargs();
            if layout.size != 64
                || serde_json::to_value(&layout).map_err(|e| e.to_string())?
                    != frozen_kernargs(kind)
            {
                return Err(format!(
                    "{symbol}: kernarg layout differs from the frozen 64-byte ABI"
                ));
            }
            emitted.push(e);
            entry_notes.push(note);
        }
        let specs: Vec<Entry> = entry_keys()
            .into_iter()
            .map(|(k, a)| {
                let s = mk(ctx.arch, codec, k, a, negative);
                let threads = mk(ctx.arch, codec, k, a, negative).threads();
                Entry {
                    arch: ctx.arch,
                    codec,
                    kind: k,
                    acc: a,
                    symbol: s.symbol(),
                    threads,
                }
            })
            .collect();

        // 2. Independent source replay per entry (wait ledger and hazards).
        for (spec, e) in specs.iter().zip(&emitted) {
            let symbol = spec.symbol.clone();
            ledger_replay::replay_waits(&e.s_text, ctx.arch)
                .map_err(|err| format!("{symbol}: source wait replay: {err}"))?;
            let hz = ledger_replay::replay_hazards(&e.s_text, ctx.arch)
                .map_err(|err| format!("{symbol}: source hazard replay: {err}"))?;
            if !hz.is_empty() {
                return Err(format!(
                    "{symbol}: source hazard replay found {}: {:?}",
                    hz.len(),
                    hz.first()
                ));
            }
        }

        // 3. The module.
        let (text, mproof) = iu4_gemm::module(&emitted, &module)
            .map_err(|e| format!("{module}: module assembly: {e}"))?;
        ledger_replay::replay_waits(&text, ctx.arch)
            .map_err(|e| format!("{module}: module wait replay: {e}"))?;
        let denorm32 = text.matches(".amdhsa_float_denorm_mode_32 3").count();
        let denorm1664 = text.matches(".amdhsa_float_denorm_mode_16_64 3").count();
        if denorm32 != 4 || denorm1664 != 4 {
            return Err(format!(
                "{module}: source declares denorm 3/3 for {denorm32}/{denorm1664} of 4 entries"
            ));
        }
        let source = dir.join(format!("{module}.s"));
        std::fs::write(&source, &text).map_err(|e| format!("{}: {e}", source.display()))?;
        let hxaco = dir.join(format!("{module}.hxaco"));

        // 4. Native build; objdump parse-back is mandatory (oracle toolchain).
        let build = toolchain::build(&ctx.tc, &source, &hxaco, ctx.arch.name())
            .map_err(|e| format!("{module}: build: {e}"))?;
        if !build.parse_back_checked || build.disassembly.is_empty() {
            return Err(format!(
                "{module}: object disassembly/parse-back was skipped"
            ));
        }
        let elf_bytes =
            std::fs::read(&build.elf).map_err(|e| format!("{}: {e}", build.elf.display()))?;
        let native_text = text_section(&elf_bytes)?;
        // The native writer's .text must equal the ROCm assembler/linker's.
        let oracle_dir = dir.join("oracle");
        std::fs::create_dir_all(&oracle_dir).map_err(|e| e.to_string())?;
        let oracle = toolchain::oracle_assemble_link_bundle(
            &ctx.tc,
            &source,
            &oracle_dir.join(format!("{module}.hxaco")),
            ctx.arch.name(),
        )
        .map_err(|e| format!("{module}: oracle build: {e}"))?;
        let oracle_text = text_section(&std::fs::read(&oracle.elf).map_err(|e| e.to_string())?)?;
        if native_text != oracle_text {
            return Err(format!(
                "{module}: native .text {} differs from the ROCm oracle {}",
                sha256(&native_text),
                sha256(&oracle_text)
            ));
        }
        let dis_path = dir.join(format!("{module}.dis"));
        std::fs::write(&dis_path, &build.disassembly).map_err(|e| e.to_string())?;

        // 5. Per-symbol: M7, object audit, and (candidate only) shape contract + certificate.
        let mut entries = Vec::new();
        let mut entry_sha = BTreeMap::new();
        let mut contract_files: Vec<(PathBuf, Vec<u8>)> = Vec::new();
        for (idx, (spec, e)) in specs.iter().zip(&emitted).enumerate() {
            let symbol = spec.symbol.clone();
            let variant = variant_tag(codec, spec.kind, spec.acc);
            let m7 = pm_check::m7(&build.elf, ctx.arch.name(), &symbol)
                .map_err(|e| format!("{symbol}: M7: {e}"))?;
            if m7["lift"].as_str() != Some("byte-exact") || m7["obligations"] != json!({}) {
                return Err(format!("{symbol}: M7 not clean: {m7}"));
            }
            let audit = audit_object(
                spec,
                &symbol,
                &build.disassembly,
                &build.elf,
                &text,
                negative,
            )?;
            let mut row = json!({
                "symbol": symbol, "variant": variant, "threads": spec.threads,
                "source_sha256": sha256(e.s_text.as_bytes()),
                "builder_proof_sha256": e.proof.s_text_sha256,
                "builder": {
                    "instructions": e.shape.instructions, "wmma": e.shape.wmma, "vmem": e.shape.vmem, "waits": e.shape.waits,
                    "barriers": e.shape.barriers, "next_free_vgpr": e.shape.next_free_vgpr, "next_free_sgpr": e.shape.next_free_sgpr,
                    "wait_proofs": e.proof.waits.len(), "hazard_proofs": e.proof.hazards.len(),
                    "hazard_rules": e.proof.hazards.iter().map(|h| h.rule.clone()).collect::<std::collections::BTreeSet<_>>(),
                    "loop_fixpoints": e.proof.loop_fixpoints.len(),
                },
                "m7": m7, "object": audit.json,
            });
            entry_sha.insert(symbol.clone(), sha256(e.s_text.as_bytes()));
            if let Some(note) = entry_notes[idx] {
                row["mutation_note"] = json!(note);
            }
            if let Some(reference) = reference {
                let same = reference
                    .get(&symbol)
                    .is_some_and(|s| *s == sha256(e.s_text.as_bytes()));
                row["identical_to_candidate_source"] = json!(same);
                // Only the `_acc` pack-selector mutation may be a no-op (those entries have no pack exchange).
                let may_be_unchanged = negative == Negative::WrongPackSelector && spec.acc;
                if !same {
                    row["mutation_effective"] = json!(true);
                } else if !may_be_unchanged {
                    return Err(format!("{symbol}: the {} mutation left the entry's source byte-identical to the candidate", negative.tag()));
                }
            }
            if candidate {
                let shape = contract(spec, e);
                let manifest = dir.join(format!("{variant}.manifest.json"));
                toolchain::certify(
                    &ctx.tc,
                    &build,
                    &source,
                    ctx.arch.name(),
                    &manifest,
                    Some(&shape),
                    &mproof.s_text_sha256,
                    &mproof.builder_git_sha,
                )
                .map_err(|err| format!("{symbol}: certification: {err}"))?;
                let manifest_json: Value =
                    serde_json::from_slice(&std::fs::read(&manifest).map_err(|e| e.to_string())?)
                        .map_err(|e| format!("{}: {e}", manifest.display()))?;
                let spills = (
                    find_key(&manifest_json, "vgpr_spill_count").and_then(Value::as_u64),
                    find_key(&manifest_json, "sgpr_spill_count").and_then(Value::as_u64),
                );
                if spills != (Some(0), Some(0)) {
                    return Err(format!(
                        "{symbol}: certificate spill counts {spills:?}, required (0, 0)"
                    ));
                }
                let contract_bytes =
                    serde_json::to_vec_pretty(&shape).map_err(|e| e.to_string())?;
                row["certificate"] = json!({
                    "manifest": manifest, "manifest_sha256": sha256_file(&manifest)?,
                    "vgpr_count": find_key(&manifest_json, "vgpr_count"), "sgpr_count": find_key(&manifest_json, "sgpr_count"),
                    "vgpr_spill_count": 0, "sgpr_spill_count": 0,
                });
                row["contract_sha256"] = json!(sha256(&contract_bytes));
                contract_files.push((
                    ctx.root.join(format!(
                        "crates/peacemaker-isa/kernels/qwen4_moe_sym.{}.{variant}.contract.json",
                        ctx.arch.name()
                    )),
                    contract_bytes,
                ));
            }
            entries.push(row);
        }
        if let Some(reference) = reference {
            if emitted.iter().zip(&specs).all(|(e, s)| {
                reference
                    .get(&s.symbol)
                    .is_some_and(|x| *x == sha256(e.s_text.as_bytes()))
            }) {
                return Err(format!(
                    "{module}: the {} module is identical to the candidate in every entry",
                    negative.tag()
                ));
            }
        }

        // 6. Outputs (only after every audit passed). Never overwrite.
        let hxaco_bytes = std::fs::read(&hxaco).map_err(|e| format!("{}: {e}", hxaco.display()))?;
        let dest = if candidate {
            ctx.root.join(format!("kernels/{module}.hxaco"))
        } else {
            ctx.root.join(format!(
                "kernels/qwen4_moe_{}_negative/{}/{module}.hxaco",
                ctx.arch.name(),
                negative.tag()
            ))
        };
        let mut written = vec![json!({"path": dest, "status": write_new(&dest, &hxaco_bytes)?})];
        for (path, bytes) in &contract_files {
            written.push(json!({"path": path, "status": write_new(path, bytes)?}));
        }
        if let Some(ev) = &ctx.evidence {
            let ev = ev.join(negative.tag());
            let proof = serde_json::to_vec_pretty(&mproof).map_err(|e| e.to_string())?;
            for (name, bytes) in [
                (format!("{module}.s"), text.as_bytes().to_vec()),
                (format!("{module}.proof.json"), proof),
                (
                    format!("{module}.dis"),
                    build.disassembly.as_bytes().to_vec(),
                ),
            ] {
                written.push(
                    json!({"path": ev.join(&name), "status": write_new(&ev.join(&name), &bytes)?}),
                );
            }
        }

        let module_text_sha = sha256(text.as_bytes());
        Ok(Built {
            entry_sha,
            module_text_sha: module_text_sha.clone(),
            json: json!({
                "module": module, "codec": format!("{codec:?}"), "negative": negative.tag(),
                "intentionally_broken": !candidate, "candidate_proof": candidate,
                "source": source, "source_sha256": module_text_sha, "module_proof_sha256": mproof.s_text_sha256,
                "builder_git_sha": mproof.builder_git_sha,
                "elf": build.elf, "elf_sha256": sha256(&elf_bytes), "text_section_sha256": sha256(&native_text),
                "oracle_text_section_sha256": sha256(&oracle_text), "text_identical_to_rocm_oracle": true,
                "hxaco": dest, "hxaco_sha256": sha256(&hxaco_bytes),
                "disassembly": dis_path, "disassembly_sha256": sha256_file(&dis_path)?,
                "tools": build.tools.iter().map(|t| json!({"path": t.path, "version": t.version, "sha256": t.sha256})).collect::<Vec<_>>(),
                "entries": entries, "outputs": written,
            }),
        })
    }

    // --------------------------------------------------------------------------------------
    // source identity, frozen objects

    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(rd) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in rd.flatten() {
            let p = entry.path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().is_some_and(|e| e == "rs") {
                out.push(p);
            }
        }
    }

    fn source_identity(root: &Path) -> R<Value> {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut files = Vec::new();
        walk(&manifest.join("src"), &mut files);
        files.sort();
        let mut tree = Sha256::new();
        let mut kernel = Vec::new();
        for f in &files {
            let rel = f
                .strip_prefix(root)
                .unwrap_or(f.as_path())
                .display()
                .to_string();
            let h = sha256_file(f)?;
            tree.update(format!("{rel}\0{h}\n"));
            if rel.contains("qwen4_moe_sym_gfx1100") {
                kernel.push(json!({"path": rel, "sha256": h}));
            }
        }
        let mine = [
            "examples/emit_qwen4_moe_sym_gfx1100.rs",
            "examples/qwen4_moe_sym_gfx1100_p1.rs",
        ];
        let mut emit = Vec::new();
        for m in mine {
            let p = manifest.join(m);
            emit.push(json!({"path": m, "sha256": sha256_file(&p)?}));
        }
        let mut pinned = Vec::new();
        for (rel, want) in PINNED_SOURCES {
            let got = sha256_file(&manifest.join(rel))?;
            if got != want {
                return Err(format!(
                    "pinned source {rel} changed: {got}, baseline {want}"
                ));
            }
            pinned.push(json!({"path": rel, "sha256": got, "equals_baseline": true}));
        }
        Ok(json!({
            "pinned_sources": pinned,
            "peacemaker_isa_src_tree_sha256": format!("{:x}", tree.finalize()), "src_files": files.len(),
            "gfx1100_kernel_files": kernel, "emitter_files": emit,
            "probe_binary": std::env::current_exe().map_err(|e| e.to_string())?,
            "probe_binary_sha256": sha256_file(&std::env::current_exe().map_err(|e| e.to_string())?)?,
        }))
    }

    /// Hash every top-level `kernels/*.hxaco`; gfx1201 ones must equal their `HEAD` blob.
    fn frozen_objects(root: &Path, failures: &mut Vec<Failure>) -> BTreeMap<String, Value> {
        let mut out = BTreeMap::new();
        let Ok(rd) = std::fs::read_dir(root.join("kernels")) else {
            return out;
        };
        let mut names: Vec<String> = rd
            .flatten()
            .filter(|e| e.path().is_file())
            .filter_map(|e| e.file_name().into_string().ok())
            .filter(|n| n.ends_with(".hxaco"))
            .collect();
        names.sort();
        for name in names {
            let path = root.join("kernels").join(&name);
            let Ok(bytes) = std::fs::read(&path) else {
                continue;
            };
            let sha = sha256(&bytes);
            if let Some((_, want)) = PINNED_OBJECTS.iter().find(|(n, _)| *n == name) {
                if sha != *want {
                    failures.push(Failure {
                        scope: "frozen:gfx1201".into(),
                        message: format!("kernels/{name} is {sha}, baseline {want}"),
                    });
                }
            }
            let mut row = json!({"sha256": sha});
            if name.contains("gfx1201") {
                match git_head_blob(root, &format!("kernels/{name}")) {
                    Some(blob) => {
                        let same = sha256(&blob) == sha;
                        row["head_sha256"] = json!(sha256(&blob));
                        row["equals_head"] = json!(same);
                        if !same {
                            failures.push(Failure {
                                scope: "frozen:gfx1201".into(),
                                message: format!("kernels/{name} differs from HEAD"),
                            });
                        }
                    }
                    None => row["head_sha256"] = json!(null),
                }
            }
            out.insert(name, row);
        }
        out
    }

    // --------------------------------------------------------------------------------------

    fn parse_args(argv: &[String]) -> R<Opts> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let root = root
            .canonicalize()
            .map_err(|e| format!("{}: {e}", root.display()))?;
        let mut o = Opts {
            arch: DEFAULT_ARCH,
            out_dir: root,
            receipt: PathBuf::from(DEFAULT_RECEIPT),
            work_dir: PathBuf::new(),
            evidence_dir: None,
            seed: 0x4234_5001,
            p1_only: false,
            codecs: CodecSet::Legacy,
        };
        let (mut receipt_given, mut work_given) = (false, false);
        let mut it = argv.iter();
        while let Some(a) = it.next() {
            let mut value = |name: &str| {
                it.next()
                    .cloned()
                    .ok_or_else(|| format!("{name} needs a value"))
            };
            match a.as_str() {
                "--arch" => {
                    o.arch = match value("--arch")?.as_str() {
                        "gfx1100" => Arch::Gfx1100,
                        "gfx1151" => Arch::Gfx1151,
                        other => return Err(format!("--arch {other} (gfx1100|gfx1151)")),
                    }
                }
                "--out-dir" => o.out_dir = PathBuf::from(value("--out-dir")?),
                "--receipt" => {
                    o.receipt = PathBuf::from(value("--receipt")?);
                    receipt_given = true
                }
                "--codec" => {
                    o.codecs = match value("--codec")?.as_str() {
                        "legacy" => CodecSet::Legacy,
                        "e8p" => CodecSet::E8p,
                        "all" => CodecSet::All,
                        other => return Err(format!("--codec {other} (legacy|e8p|all)")),
                    }
                }
                "--work-dir" => {
                    o.work_dir = PathBuf::from(value("--work-dir")?);
                    work_given = true
                }
                "--evidence-dir" => o.evidence_dir = Some(PathBuf::from(value("--evidence-dir")?)),
                "--seed" => {
                    o.seed = value("--seed")?
                        .parse()
                        .map_err(|e| format!("--seed: {e}"))?
                }
                "--p1-only" => o.p1_only = true,
                other => return Err(format!("unknown argument {other}")),
            }
        }
        // gfx1151 builds the E8P module only (its MQ4/W2/W3 are other designs); the legacy trio stays gfx1100.
        if o.arch == Arch::Gfx1151 && o.codecs != CodecSet::E8p {
            return Err(
                "--arch gfx1151 requires --codec e8p (the MQ4/W2/W3 modules are gfx1100-only)"
                    .into(),
            );
        }
        if !work_given {
            o.work_dir = std::env::temp_dir().join(format!(
                "peacemaker-b4-{}-{}",
                o.arch.name(),
                std::process::id()
            ));
        }
        // The E8P-only run must not collide with the committed B4 receipt.
        if o.codecs == CodecSet::E8p && !receipt_given {
            o.receipt = PathBuf::from(if o.arch == Arch::Gfx1151 {
                DEFAULT_RECEIPT_E8P_GFX1151
            } else {
                DEFAULT_RECEIPT_E8P
            });
        }
        Ok(o)
    }

    /// The kernel's launch admission (`validate_launch`): the frozen sizes are accepted, every other
    /// shape the contract forbids (generic tails, zero or unaligned dimensions, 32-bit extent and gather
    /// overflows) is refused.
    fn admission(arch: Arch, codecs: &[Codec]) -> (Value, Vec<String>) {
        let mut failures = Vec::new();
        let (mut accepted, mut refused) = (0u32, 0u32);
        for &codec in codecs {
            for acc in [false, true] {
                for kind in Kind::ALL {
                    let spec = || mk(arch, codec, kind, acc, Negative::None);
                    let (m, k) = if kind == Kind::GateUp {
                        (2560, 2560)
                    } else {
                        (2560, 640)
                    };
                    for (x_row_div, x_src_rows, m_total) in [
                        (1u32, 1536u32, 1536u32),
                        (10, 1536, 1536),
                        (1, 1, 16),
                        (10, 17, 32),
                    ] {
                        match gfx1100::validate_launch(spec(), m, k, x_row_div, m_total, x_src_rows)
                        {
                            Ok(()) => accepted += 1,
                            Err(e) => failures
                                .push(format!("{}: frozen launch refused: {e}", spec().symbol())),
                        }
                    }
                    let kmul = if kind == Kind::GateUp { 256 } else { 128 };
                    let refusals: [(&str, u32, u32, u32, u32, u32); 9] = [
                        ("k zero", m, 0, 1, 1536, 1536),
                        ("k not a multiple", m, k + 64, 1, 1536, 1536),
                        ("k below the multiple", m, kmul - 64, 1, 1536, 1536),
                        ("m zero", 0, k, 1, 1536, 1536),
                        ("m not a multiple of 64", 100, k, 1, 1536, 1536),
                        ("m_total zero", m, k, 1, 0, 1536),
                        ("m_total not a multiple of 16", m, k, 1, 1537, 1536),
                        ("x_row_div zero", m, k, 0, 1536, 1536),
                        ("gather product overflow", m, k, u32::MAX, 1536, u32::MAX),
                    ];
                    for (what, m, k, x_row_div, m_total, x_src_rows) in refusals {
                        if gfx1100::validate_launch(spec(), m, k, x_row_div, m_total, x_src_rows)
                            .is_ok()
                        {
                            failures.push(format!(
                                "{}: launch admitted despite {what}",
                                spec().symbol()
                            ));
                        } else {
                            refused += 1;
                        }
                    }
                    // A tensor beyond a 32-bit descriptor extent.
                    if gfx1100::validate_launch(spec(), 1 << 24, 2560, 1, 1536, 1536).is_ok() {
                        failures.push(format!(
                            "{}: launch admitted with a >4 GiB tensor",
                            spec().symbol()
                        ));
                    } else {
                        refused += 1;
                    }
                }
            }
        }
        (
            json!({"accepted_frozen_launches": accepted, "refused_forbidden_launches": refused, "scope": format!("{}::validate_launch", arch.name())}),
            failures,
        )
    }

    pub fn run() -> i32 {
        let argv: Vec<String> = std::env::args().collect();
        let opts = match parse_args(&argv[1..]) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("{e}");
                return 2;
            }
        };
        let mut failures: Vec<Failure> = Vec::new();
        let ctx = Ctx {
            arch: opts.arch,
            root: opts.out_dir.clone(),
            work: opts.work_dir.clone(),
            evidence: opts.evidence_dir.clone(),
            tc: Toolchain::oracle(),
        };

        let frozen_before = frozen_objects(&ctx.root, &mut failures);

        // P1 arithmetic / address proofs.
        let (admission_json, admission_failures) = admission(opts.arch, &opts.codecs.codecs());
        for f in admission_failures {
            failures.push(Failure {
                scope: "p1:admission".into(),
                message: f,
            });
        }
        let codec_of = |n: &str| match n {
            "mq4" => Codec::Mq4,
            "w2" => Codec::W2,
            _ => Codec::W3,
        };
        let kind_of = |n: &str| {
            if n == "gate_up" {
                Kind::GateUp
            } else {
                Kind::Down
            }
        };
        let (w2, w3, mq4) = (
            |x: u32| gfx1100::w2_spread_model(x),
            |x: u32| gfx1100::w3_spread_model(x),
            |x: u32| gfx1100::mq4_unpack_model(x),
        );
        let code_offset =
            |c: &str, k: &str, e: u32| gfx1100::code_offset(codec_of(c), kind_of(k), e);
        let scale_offset =
            |c: &str, k: &str, e: u32| gfx1100::scale_offset(codec_of(c), kind_of(k), e);
        let code_loads = |c: &str| gfx1100::code_loads(codec_of(c));
        let row_code_bytes = |c: &str, k: u32| gfx1100::row_code_bytes(codec_of(c), k);
        let row_scale_bytes = |c: &str, k: u32| gfx1100::row_scale_bytes(codec_of(c), k);
        let tensor_bytes = |c: &str, m: u32, k: u32| gfx1100::tensor_bytes(codec_of(c), m, k);
        let dot_bound = |c: &str| gfx1100::dot_bound(codec_of(c));
        let scale_row = |lr: u32, hi: u32, wrong: bool| gfx1100::scale_row(lr, hi, wrong);
        let accumulator_row = |j: u32, hi: u32| gfx1100::accumulator_row(j, hi);
        let swizzle_lane = |j: u32| gfx1100::swizzle_lane(j);
        let kernel = p1::Kernel {
            w2_spread: &w2,
            w3_spread: &w3,
            mq4_unpack: &mq4,
            code_offset: &code_offset,
            scale_offset: &scale_offset,
            code_loads: &code_loads,
            row_code_bytes: &row_code_bytes,
            row_scale_bytes: &row_scale_bytes,
            tensor_bytes: &tensor_bytes,
            dot_bound: &dot_bound,
            scale_row: &scale_row,
            accumulator_row: &accumulator_row,
            swizzle_lane: &swizzle_lane,
        };
        let p1_out = p1::prove(opts.seed, Some(&kernel));
        for f in &p1_out.failures {
            failures.push(Failure {
                scope: "p1".into(),
                message: f.clone(),
            });
        }
        // E8P CPU proofs: arithmetic (2C+s fold, signed nibble unpack), K assignment, table stage, address windows, register layout.
        let codecs = opts.codecs.codecs();
        let p1_e8p = if codecs.contains(&Codec::E8p) {
            let pr = gfx1100::e8p::prove();
            for f in &pr.failures {
                failures.push(Failure {
                    scope: "p1:e8p".into(),
                    message: f.clone(),
                });
            }
            json!({"checks": pr.checks.iter().map(|(n, c)| json!({"check": n, "cases": c})).collect::<Vec<_>>(), "failures": pr.failures.len()})
        } else {
            json!(null)
        };

        let mut candidates = Vec::new();
        let mut negatives = Vec::new();
        if !opts.p1_only {
            let mut reference: BTreeMap<Codec_, BTreeMap<String, String>> = BTreeMap::new();
            let mut cand_text: BTreeMap<Codec_, String> = BTreeMap::new();
            for &codec in &codecs {
                match build_module(&ctx, codec, Negative::None, None) {
                    Ok(b) => {
                        reference.insert(Codec_::of(codec), b.entry_sha);
                        cand_text.insert(Codec_::of(codec), b.module_text_sha);
                        candidates.push(b.json);
                    }
                    Err(e) => failures.push(Failure {
                        scope: format!("candidate:{}", codec.module(ctx.arch)),
                        message: e,
                    }),
                }
            }
            let mut neg_order: Vec<Negative> = Vec::new();
            for &c in &codecs {
                for n in negatives_of(c) {
                    if !neg_order.contains(&n) {
                        neg_order.push(n);
                    }
                }
            }
            for negative in neg_order {
                let mut modules = Vec::new();
                for &codec in &codecs {
                    if !negatives_of(codec).contains(&negative) {
                        continue;
                    }
                    let Some(r) = reference.get(&Codec_::of(codec)) else {
                        failures.push(Failure {
                            scope: format!("negative:{}:{}", negative.tag(), codec.module(ctx.arch)),
                            message: "skipped: the candidate module of this codec failed, no reference to compare".into(),
                        });
                        continue;
                    };
                    match build_module(&ctx, codec, negative, Some(r)) {
                        Ok(b) => {
                            let mut j = b.json;
                            j["module_source_differs_from_candidate"] = json!(
                                cand_text.get(&Codec_::of(codec)) != Some(&b.module_text_sha)
                            );
                            modules.push(j);
                        }
                        Err(e) => failures.push(Failure {
                            scope: format!(
                                "negative:{}:{}",
                                negative.tag(),
                                codec.module(ctx.arch)
                            ),
                            message: e,
                        }),
                    }
                }
                negatives.push(json!({
                    "negative": negative.tag(), "intentionally_broken": true, "candidate_proof": false,
                    "audit": "structure only: source replay, M7, object disassembly/descriptor; no candidate contract or certificate",
                    "modules": modules,
                }));
            }
        }

        let frozen_after = frozen_objects(&ctx.root, &mut failures);
        // Only objects that existed before the run must be unchanged (the run adds new files).
        for (name, row) in &frozen_before {
            if frozen_after.get(name) != Some(row) {
                failures.push(Failure {
                    scope: "frozen".into(),
                    message: format!("existing kernels/{name} changed during the run"),
                });
            }
        }

        let sources = match source_identity(&ctx.root) {
            Ok(v) => v,
            Err(e) => {
                failures.push(Failure {
                    scope: "inputs".into(),
                    message: e,
                });
                json!(null)
            }
        };

        let scope_failed = |prefix: &str| failures.iter().any(|f| f.scope.starts_with(prefix));
        let exit_code = i32::from(!failures.is_empty());
        let verdict = |failed: bool| if failed { "fail" } else { "pass" };
        let receipt = json!({
            "schema": "b4-p1-receipt-v1", "gate": "B4-P1", "arch": opts.arch.name(), "device": null,
            "scope": "static/CPU only: arithmetic+address model proofs, emitted-object audits; no device execution, no G1 (synthetic or real)",
            "host_link_caveat": "PCIe 4.0 x4 (reported by the lead; not measured by this probe)",
            "command": {
                "argv": argv, "cwd": std::env::current_dir().ok(), "exit_code": exit_code,
                "generated_unix": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).ok(),
            },
            "options": {
                "out_dir": ctx.root, "work_dir": ctx.work, "evidence_dir": ctx.evidence, "seed": opts.seed, "p1_only": opts.p1_only,
                "codecs": codecs.iter().map(|c| c.module(opts.arch)).collect::<Vec<_>>(),
                "receipt": opts.receipt,
            },
            "inputs": sources,
            "denorm_modes": "required 3/3; read from each linked object's kernel descriptor (see candidates[].entries[].object.descriptor)",
            "p1_arithmetic": p1_out.receipt,
            "p1_e8p": p1_e8p,
            "launch_admission": admission_json,
            "candidates": candidates,
            "negatives": negatives,
            "frozen_objects": {
                "before": frozen_before, "after": frozen_after,
                "gfx1201_rule": "every kernels/*gfx1201*.hxaco must equal its git HEAD blob and be unchanged by this run",
            },
            "verdict": {
                "p1_arithmetic": verdict(scope_failed("p1")),
                "candidate_modules": if opts.p1_only { "not-run" } else { verdict(scope_failed("candidate")) },
                "negative_audit": if opts.p1_only { "not-run" } else { verdict(scope_failed("negative")) },
                "frozen_objects": verdict(scope_failed("frozen")),
                "overall": verdict(!failures.is_empty()),
            },
            "failures": failures.iter().map(|f| json!({"scope": f.scope, "message": f.message})).collect::<Vec<_>>(),
        });
        let text = serde_json::to_string_pretty(&receipt).expect("receipt serialises") + "\n";
        let mut code = exit_code;
        match write_new(&opts.receipt, text.as_bytes()) {
            Ok(status) => eprintln!("receipt {}: {status}", opts.receipt.display()),
            Err(e) => {
                eprintln!("{e}");
                print!("{text}");
                code = code.max(2);
            }
        }
        for f in &failures {
            eprintln!("FAIL [{}] {}", f.scope, f.message);
        }
        code
    }

    /// `Codec` is `Copy + Eq` but not guaranteed `Ord`: key maps by its symbol infix.
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
    struct Codec_(String);
    impl Codec_ {
        fn of(c: Codec) -> Self {
            Codec_(c.infix().to_owned())
        }
    }
}

#[cfg(feature = "toolchain")]
fn main() {
    std::process::exit(app::run());
}
