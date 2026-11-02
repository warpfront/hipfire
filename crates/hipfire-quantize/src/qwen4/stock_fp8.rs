// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! `hipfire-quantize --recipe stock-fp8`: the stock-FP8 passthrough producer
//! for the Qwen3.8-Flash-Next FP8 checkpoint.
//!
//! # Usage
//!
//! ```text
//! hipfire-quantize --recipe stock-fp8 --input <fp8-checkpoint-dir> --output <model.hfq>
//! ```
//!
//! `--recipe stock-fp8` selects this converter by itself (no
//! `--qwen4-flash-next`); the legacy recipe flags, `--format` and the
//! Flash-Next route are refused rather than ignored.  The only other flag the
//! route reads is `--qwen4-component-mode compact-fixture`, an explicit
//! non-production mode used by the fixture tests (no pin, no census, no
//! 48 x 512 geometry).
//!
//! # What is written
//!
//! The artifact is an HFQM package (arch 16) whose `qwen4_recipe.id` is
//! `"stock-fp8"`.  Nothing is re-quantized: FP8 codes are copied verbatim and
//! every other tensor keeps its source BF16 bytes.
//!
//! * Routed experts.  The checkpoint stores
//!   `<layer>.mlp.experts.E.{gate_proj,up_proj,down_proj}.weight` (F8_E4M3
//!   `[rows, K]`) plus `.weight_scale_inv` (BF16 `[ceil(rows/128), ceil(K/128)]`).
//!   They are stacked under the existing names `<layer>.mlp.experts.gate_up_proj`
//!   (`[E, 2*rows, K]`) and `<layer>.mlp.experts.down_proj` (`[E, R, K]`) as
//!   qt=55 `FP8E4M3B128`, group_size 128.  One block per expert:
//!   `codes ‖ s_inv ‖ zero pad to a multiple of 256` where, for gate/up,
//!   codes = gate codes then up codes and s_inv = gate scales then up scales
//!   (the gate half is a whole number of 128-row scale blocks, so the stacked
//!   scale grid is exact).  Production: 48 layers x 512 experts; gate_up block
//!   stride 3,277,312, down block stride 1,638,656.
//! * PLE.  Every `…ngram_embedding.shard_N.weight` (F8_E4M3 `[2,500,012, 160]`)
//!   becomes qt=56 `FP8E4M3PT`, group_size 0: a 16-byte header (the BF16 bits of
//!   the table's shared `…ngram_embedding.weight_scale`, little endian, then 14
//!   zero bytes) followed by the codes verbatim.  The scale source may be shape
//!   `[]`, `[1]` or `[1, 1]`.
//! * Everything else under the text manifest (trunk, router, shared expert,
//!   embeddings, head, norms, hyper-connection, …) is BF16 qt=16, verbatim.
//!   The three I64 PLE metadata records are copied as qt=54 and drive
//!   `qwen4_ple`, exactly as in the Flash-Next producer.
//! * Omitted: all `mtp.*` tensors (this is an AR-only artifact; the MTP experts
//!   are FP8 in the source and their scales BF16), and the vision tower
//!   (`model.visual.*`, …): the Qwen4 loader's text-only boundary refuses
//!   vision records, so they cannot ride along.  Both omissions are recorded in
//!   the receipt.
//!
//! # Gates
//!
//! 1. The destination and its receipt must not exist.
//! 2. Production: `<input>/fp8.sha256` must hash to the pinned manifest SHA-256
//!    (`2f9001c5…5a`, the fp8 teacher's `PINNED_MANIFEST_SHA256`) before any
//!    source tensor is read.  The shard file names must equal the manifest's
//!    `*.safetensors` set, and `config.json` / tokenizer sidecars read by the
//!    producer must match their manifest hashes.
//! 3. Production census: 131 shards, 75,392 F8_E4M3 (73,728 routed + 128 PLE +
//!    1,536 MTP), 76,694 BF16, 3 I64; the BF16 set must equal the Flash-Next
//!    manifest's text tensors, the stacked expert and PLE names/shapes must
//!    equal the manifest's, and no part may be missing.
//! 4. EVERY F8_E4M3 tensor (including the omitted MTP ones) is scanned in
//!    bounded 8 MiB reads before any output byte is written.  A `0x7F` or
//!    `0xFF` code is refused with the source tensor named.  `0x80` (-0) codes
//!    are counted and never rewritten.
//! 5. Free-space preflight, temporary file + fsync + reopen validation, and
//!    no-replace publication of both artifact and receipt.
//!
//! # Receipt
//!
//! `<output>.receipt.json` (e.g. `model.hfq.receipt.json`) records: the
//! pinned and verified manifest SHA-256, the SHA-256 of the artifact
//! (`artifact_sha256`, one streaming read of the finished temporary), the source
//! dtype census, emitted/omitted FP8 counts and `0x80` counts, all byte
//! counts, wall-clock seconds, and for every HFQM tensor its qt, shape,
//! artifact offset/length and each source tensor with its shard file, absolute
//! byte range `[start, end)` inside that file, `sha256` of exactly those bytes
//! and (for FP8) its `0x80` count.  The range hashes are taken while the bytes
//! are copied and, for FP8, must equal the hashes taken by the pre-write scan.

use super::*;
use rayon::prelude::*;
use rdna_compute::fp8_e4m3::{
    b128_block_stride, b128_scale_count, b128_tensor_bytes, pt_header, pt_tensor_bytes, scan_codes,
    B128_BLOCK, PT_HEADER_BYTES,
};
use sha2::{Digest, Sha256};
use std::time::Instant;

/// `qwen4_recipe.id` written into the artifact.
pub(crate) const STOCK_FP8_RECIPE_ID: &str = "stock-fp8";
const STOCK_FP8_MANIFEST_NAME: &str = "fp8.sha256";
/// SHA-256 of the pinned `fp8.sha256` (fp8 teacher `PINNED_MANIFEST_SHA256`).
pub(crate) const STOCK_FP8_PINNED_MANIFEST_SHA256: &str =
    "2f9001c5f2bbff2b728e6e21409f11bdbee6353e4ca93b79fb1499936259615a";
const MAX_MANIFEST_BYTES: u64 = 4 * 1024 * 1024;
/// qt=55 `FP8E4M3B128`.
const FP8E4M3B128_QUANT_TYPE: u8 = 55;
/// qt=56 `FP8E4M3PT`.
const FP8E4M3PT_QUANT_TYPE: u8 = 56;
const FP8_DTYPE: &str = "F8_E4M3";
const RECEIPT_SUFFIX: &str = ".receipt.json";
const PLE_SCALE_SUFFIX: &str = ".ngram_embedding.weight_scale";
/// Bounded read size of the FP8 scan and of the payload copy.
const IO_CHUNK_BYTES: usize = 8 * 1024 * 1024;
const STOCK_EXPERT_LAYERS: usize = 48;
const STOCK_F8_TENSORS: u64 = 75_392;
const STOCK_BF16_TENSORS: u64 = 76_694;
const STOCK_I64_TENSORS: u64 = 3;
const STOCK_F8_ROUTED: u64 = 73_728;
const STOCK_F8_MTP: u64 = 1_536;
/// Conservative receipt size: per source record plus fixed overhead.
const RECEIPT_BYTES_PER_SOURCE: u64 = 512;
const RECEIPT_FIXED_BYTES: u64 = 16 * 1024 * 1024;
static ZEROS: [u8; 256] = [0; 256];

#[derive(Debug)]
pub(crate) struct StockFp8Options<'a> {
    pub(crate) input: &'a Path,
    pub(crate) output: &'a Path,
    pub(crate) mode: Qwen4Mode,
    /// Pinned SHA-256 of `<input>/fp8.sha256`.  `Production` requires exactly
    /// [`STOCK_FP8_PINNED_MANIFEST_SHA256`]; the compact fixture mode may omit
    /// it or pin its own manifest (tests).
    pub(crate) pin: Option<&'a str>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct StockFp8Summary {
    pub(crate) entries: usize,
    pub(crate) expert_entries: usize,
    pub(crate) ple_shards: usize,
    pub(crate) fp8_scanned: usize,
    pub(crate) mtp_omitted: usize,
    pub(crate) vision_omitted: usize,
    pub(crate) predicted_output_bytes: u64,
    pub(crate) artifact_bytes: u64,
    pub(crate) artifact_sha256: String,
    pub(crate) receipt_path: PathBuf,
    pub(crate) elapsed_seconds: f64,
}

/// CLI entry point.
pub(crate) fn run_stock_fp8_cli(
    input: &Path,
    output: &Path,
    mode: Qwen4Mode,
) -> Result<StockFp8Summary, Qwen4Error> {
    let pin = match mode {
        Qwen4Mode::Production => Some(STOCK_FP8_PINNED_MANIFEST_SHA256),
        Qwen4Mode::CompactFixture => None,
    };
    write_stock_fp8_artifact(&StockFp8Options {
        input,
        output,
        mode,
        pin,
    })
}

fn invalid<T>(message: impl Into<String>) -> Result<T, Qwen4Error> {
    Err(Qwen4Error::Invalid(message.into()))
}

fn hex_lower(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(DIGITS[(byte >> 4) as usize] as char);
        out.push(DIGITS[(byte & 15) as usize] as char);
    }
    out
}

fn receipt_path_for(output: &Path) -> PathBuf {
    let mut name = output
        .file_name()
        .map(|name| name.to_os_string())
        .unwrap_or_else(|| "qwen4.hfq".into());
    name.push(RECEIPT_SUFFIX);
    output.with_file_name(name)
}

fn absolute(path: &Path) -> String {
    std::path::absolute(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .into_owned()
}

fn list_sample(names: &[String]) -> String {
    let shown: Vec<&str> = names.iter().take(8).map(String::as_str).collect();
    if names.len() > shown.len() {
        format!("{shown:?} (+{} more)", names.len() - shown.len())
    } else {
        format!("{shown:?}")
    }
}

// ---------------------------------------------------------------- manifest pin

struct ManifestCheck {
    path: PathBuf,
    sha256: String,
    files: BTreeMap<String, String>,
}

/// Hash `<dir>/fp8.sha256` and refuse anything but the pin, then parse its
/// `sha256sum`-format lines.  Reads no tensor.
fn verify_manifest(dir: &Path, pin: &str) -> Result<ManifestCheck, Qwen4Error> {
    let path = dir.join(STOCK_FP8_MANIFEST_NAME);
    let bytes = read_local_bytes(&path, MAX_MANIFEST_BYTES, true)?
        .ok_or_else(|| Qwen4Error::Invalid(format!("{} is missing", path.display())))?;
    let sha256 = hex_lower(Sha256::digest(&bytes).as_slice());
    if sha256 != pin {
        return invalid(format!(
            "manifest {} sha256 {sha256} refused: only {pin} is accepted",
            path.display()
        ));
    }
    let text = String::from_utf8(bytes).map_err(|error| {
        Qwen4Error::Invalid(format!("{} is not UTF-8: {error}", path.display()))
    })?;
    let mut files = BTreeMap::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let malformed = || Qwen4Error::Invalid(format!("malformed manifest line {line:?}"));
        let (sha, rest) = line
            .trim_start()
            .split_once(char::is_whitespace)
            .ok_or_else(malformed)?;
        let name = rest.trim_start().trim_start_matches('*').trim();
        if sha.len() != 64
            || !sha.bytes().all(|byte| byte.is_ascii_hexdigit())
            || name.is_empty()
            || name.contains('/')
            || name.starts_with('.')
        {
            return Err(malformed());
        }
        if files
            .insert(name.to_string(), sha.to_ascii_lowercase())
            .is_some()
        {
            return invalid(format!("manifest lists {name} twice"));
        }
    }
    Ok(ManifestCheck {
        path,
        sha256,
        files,
    })
}

/// Shard names must be exactly the manifest's `*.safetensors`; the config and
/// tokenizer sidecars the producer reads must be listed and match.
fn verify_manifest_files(
    dir: &Path,
    manifest: &ManifestCheck,
    shard_paths: &[PathBuf],
) -> Result<(), Qwen4Error> {
    let on_disk: BTreeSet<String> = shard_paths
        .iter()
        .filter_map(|path| path.file_name())
        .map(|name| name.to_string_lossy().into_owned())
        .collect();
    let listed: BTreeSet<String> = manifest
        .files
        .keys()
        .filter(|name| name.ends_with(".safetensors"))
        .cloned()
        .collect();
    if on_disk != listed {
        let missing: Vec<String> = listed.difference(&on_disk).cloned().collect();
        let extra: Vec<String> = on_disk.difference(&listed).cloned().collect();
        return invalid(format!(
            "safetensors shards disagree with {}: missing {} extra {}",
            manifest.path.display(),
            list_sample(&missing),
            list_sample(&extra)
        ));
    }
    if !manifest.files.contains_key("config.json") {
        return invalid(format!(
            "{} does not list config.json",
            manifest.path.display()
        ));
    }
    for name in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "chat_template.jinja",
    ] {
        let path = dir.join(name);
        if !path.is_file() {
            continue;
        }
        let want = manifest.files.get(name).ok_or_else(|| {
            Qwen4Error::Invalid(format!(
                "{} is read by the producer but is not listed in {}",
                path.display(),
                manifest.path.display()
            ))
        })?;
        let got = crate::calibration::sha256_file_hex(&path).map_err(Qwen4Error::Invalid)?;
        if &got != want {
            return invalid(format!(
                "{} sha256 {got} does not match the manifest's {want}",
                path.display()
            ));
        }
    }
    Ok(())
}

// ------------------------------------------------------------------ source plan

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Proj {
    Gate = 0,
    Up = 1,
    Down = 2,
}

struct ExpertName<'a> {
    prefix: &'a str,
    expert: usize,
    proj: Proj,
    scale: bool,
}

/// `<prefix>.mlp.experts.<E>.{gate,up,down}_proj.{weight,weight_scale_inv}`.
fn parse_expert_name(name: &str) -> Option<ExpertName<'_>> {
    const MARKER: &str = ".mlp.experts.";
    let at = name.find(MARKER)?;
    let prefix = &name[..at];
    let mut parts = name[at + MARKER.len()..].splitn(3, '.');
    let expert = parts.next()?;
    if expert.is_empty() || !expert.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let proj = match parts.next()? {
        "gate_proj" => Proj::Gate,
        "up_proj" => Proj::Up,
        "down_proj" => Proj::Down,
        _ => return None,
    };
    let scale = match parts.next()? {
        "weight" => false,
        "weight_scale_inv" => true,
        _ => return None,
    };
    Some(ExpertName {
        prefix,
        expert: expert.parse().ok()?,
        proj,
        scale,
    })
}

#[derive(Default)]
struct LayerExperts {
    weights: [BTreeMap<usize, SourceTensor>; 3],
    scales: [BTreeMap<usize, SourceTensor>; 3],
}

#[derive(Debug, Default)]
struct SourceCensus {
    total: u64,
    bytes: u64,
    by_dtype: BTreeMap<String, u64>,
    f8_routed: u64,
    f8_ple: u64,
    f8_mtp: u64,
}

struct Omitted {
    tensor: SourceTensor,
    reason: &'static str,
}

struct ExpertBlock {
    codes: Vec<SourceTensor>,
    scales: Vec<SourceTensor>,
}

enum StockPayload {
    /// A whole source tensor, byte for byte (BF16 qt=16, I64 qt=54).
    Verbatim(SourceTensor),
    /// Stacked experts: per block `codes ‖ s_inv ‖ pad` (zero pad bytes `pad`).
    Experts { blocks: Vec<ExpertBlock>, pad: u64 },
    /// qt=56: 16-byte header then the codes verbatim.
    Ple {
        header: [u8; PT_HEADER_BYTES],
        codes: SourceTensor,
        scale: SourceTensor,
    },
}

struct StockEntry {
    name: String,
    quant_type: u8,
    shape: Vec<u32>,
    group_size: u32,
    data_len: u64,
    payload: StockPayload,
}

struct SourceUse<'a> {
    role: &'static str,
    expert: Option<usize>,
    tensor: &'a SourceTensor,
}

impl StockEntry {
    /// Source tensors in payload order.
    fn sources(&self) -> Vec<SourceUse<'_>> {
        match &self.payload {
            StockPayload::Verbatim(tensor) => vec![SourceUse {
                role: if self.quant_type == QWEN4_I64_QUANT_TYPE {
                    "i64"
                } else {
                    "bf16"
                },
                expert: None,
                tensor,
            }],
            StockPayload::Experts { blocks, .. } => {
                let mut out = Vec::new();
                for (expert, block) in blocks.iter().enumerate() {
                    for tensor in &block.codes {
                        out.push(SourceUse {
                            role: "codes",
                            expert: Some(expert),
                            tensor,
                        });
                    }
                    for tensor in &block.scales {
                        out.push(SourceUse {
                            role: "s_inv",
                            expert: Some(expert),
                            tensor,
                        });
                    }
                }
                out
            }
            StockPayload::Ple { codes, scale, .. } => vec![
                SourceUse {
                    role: "ple_scale",
                    expert: None,
                    tensor: scale,
                },
                SourceUse {
                    role: "codes",
                    expert: None,
                    tensor: codes,
                },
            ],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ExpertGeometry {
    experts: u64,
    gate_rows: u64,
    hidden: u64,
}

struct StockInventory {
    entries: Vec<StockEntry>,
    ple_metadata: PleMetadata,
    omitted: Vec<Omitted>,
    fp8: Vec<SourceTensor>,
    census: SourceCensus,
    geometry: ExpertGeometry,
    expert_entries: usize,
    ple_shards: usize,
    /// SHA-256 of each PLE scale record (the two bytes read for the header).
    ple_scale_digests: BTreeMap<String, [u8; 32]>,
}

fn is_scalar_shape(shape: &[u64]) -> bool {
    shape.is_empty() || shape == [1] || shape == [1, 1]
}

fn require_dtype(tensor: &SourceTensor, dtype: &str, what: &str) -> Result<(), Qwen4Error> {
    if tensor.dtype != dtype {
        return invalid(format!(
            "{what} {} has dtype {}, expected {dtype}",
            tensor.name, tensor.dtype
        ));
    }
    Ok(())
}

fn scale_grid(rows: u64, k: u64) -> Vec<u64> {
    vec![
        rows.div_ceil(B128_BLOCK as u64),
        k.div_ceil(B128_BLOCK as u64),
    ]
}

fn check_expert_tensor(
    tensor: &SourceTensor,
    shape: &[u64],
    bytes_per_element: u64,
) -> Result<(), Qwen4Error> {
    if tensor.shape != shape {
        return invalid(format!(
            "{} has shape {:?}, expected {:?}",
            tensor.name, tensor.shape, shape
        ));
    }
    let expected = checked_product(shape, &tensor.name)?
        .checked_mul(bytes_per_element)
        .ok_or_else(|| Qwen4Error::Invalid(format!("{} length overflows", tensor.name)))?;
    if tensor.data_len() != expected {
        return invalid(format!(
            "{} has {} payload bytes, expected {expected}",
            tensor.name,
            tensor.data_len()
        ));
    }
    Ok(())
}

fn usize_of(value: u64, what: &str) -> Result<usize, Qwen4Error> {
    usize::try_from(value)
        .map_err(|_| Qwen4Error::Invalid(format!("{what} {value} does not fit usize")))
}

/// Stack one layer's separate experts into the two existing stacked names.
fn assemble_layer(
    prefix: &str,
    layer: LayerExperts,
    mode: Qwen4Mode,
) -> Result<([StockEntry; 2], ExpertGeometry), Qwen4Error> {
    let LayerExperts {
        mut weights,
        mut scales,
    } = layer;
    let experts = weights[0].len();
    if experts == 0 {
        return invalid(format!("{prefix}: no routed expert tensors"));
    }
    if mode == Qwen4Mode::Production && experts as u64 != ROUTED_EXPERTS {
        return invalid(format!(
            "{prefix}: has {experts} routed experts, production requires {ROUTED_EXPERTS}"
        ));
    }
    for (label, map) in [
        ("gate_proj.weight", &weights[0]),
        ("up_proj.weight", &weights[1]),
        ("down_proj.weight", &weights[2]),
        ("gate_proj.weight_scale_inv", &scales[0]),
        ("up_proj.weight_scale_inv", &scales[1]),
        ("down_proj.weight_scale_inv", &scales[2]),
    ] {
        if !map.keys().copied().eq(0..experts) {
            let missing: Vec<String> = (0..experts)
                .filter(|index| !map.contains_key(index))
                .map(|index| format!("{prefix}.mlp.experts.{index}.{label}"))
                .collect();
            return invalid(format!(
                "{prefix}: {label} set is not experts 0..{experts}; missing {} (extra indexes {:?})",
                list_sample(&missing),
                map.keys().filter(|index| **index >= experts).collect::<Vec<_>>()
            ));
        }
    }
    let gate0 = &weights[0][&0];
    if gate0.shape.len() != 2 {
        return invalid(format!(
            "{} must be rank 2, got {:?}",
            gate0.name, gate0.shape
        ));
    }
    let (gate_rows, hidden) = (gate0.shape[0], gate0.shape[1]);
    let block = B128_BLOCK as u64;
    if gate_rows == 0 || hidden == 0 || gate_rows % block != 0 {
        return invalid(format!(
            "{}: gate rows {gate_rows} must be a nonzero multiple of {block} so the stacked \
             gate+up scale grid is exact (K={hidden})",
            gate0.name
        ));
    }
    if mode == Qwen4Mode::Production
        && (gate_rows != GATE_UP_INTERMEDIATE / 2 || hidden != HIDDEN_WIDTH)
    {
        return invalid(format!(
            "{prefix}: gate_proj is [{gate_rows}, {hidden}], production requires [{}, {HIDDEN_WIDTH}]",
            GATE_UP_INTERMEDIATE / 2
        ));
    }
    let gate_shape = [gate_rows, hidden];
    let down_shape = [hidden, gate_rows];
    let gate_scale_shape = scale_grid(gate_rows, hidden);
    let down_scale_shape = scale_grid(hidden, gate_rows);
    for expert in 0..experts {
        check_expert_tensor(&weights[0][&expert], &gate_shape, 1)?;
        check_expert_tensor(&weights[1][&expert], &gate_shape, 1)?;
        check_expert_tensor(&weights[2][&expert], &down_shape, 1)?;
        check_expert_tensor(&scales[0][&expert], &gate_scale_shape, 2)?;
        check_expert_tensor(&scales[1][&expert], &gate_scale_shape, 2)?;
        check_expert_tensor(&scales[2][&expert], &down_scale_shape, 2)?;
    }
    let geometry = ExpertGeometry {
        experts: experts as u64,
        gate_rows,
        hidden,
    };

    let take = |maps: &mut [BTreeMap<usize, SourceTensor>; 3], proj: usize, expert: usize| {
        maps[proj].remove(&expert).expect("expert set validated")
    };
    let mut gate_up_blocks = Vec::with_capacity(experts);
    let mut down_blocks = Vec::with_capacity(experts);
    for expert in 0..experts {
        gate_up_blocks.push(ExpertBlock {
            codes: vec![take(&mut weights, 0, expert), take(&mut weights, 1, expert)],
            scales: vec![take(&mut scales, 0, expert), take(&mut scales, 1, expert)],
        });
        down_blocks.push(ExpertBlock {
            codes: vec![take(&mut weights, 2, expert)],
            scales: vec![take(&mut scales, 2, expert)],
        });
    }

    let make = |name: String,
                rows: u64,
                k: u64,
                blocks: Vec<ExpertBlock>|
     -> Result<StockEntry, Qwen4Error> {
        let (rows_us, k_us, experts_us) = (
            usize_of(rows, "rows")?,
            usize_of(k, "K")?,
            usize_of(experts as u64, "experts")?,
        );
        let stride = b128_block_stride(rows_us, k_us)
            .ok_or_else(|| Qwen4Error::Invalid(format!("{name} block stride overflows")))?;
        let data_len = b128_tensor_bytes(experts_us, rows_us, k_us)
            .ok_or_else(|| Qwen4Error::Invalid(format!("{name} payload length overflows")))?;
        let used = rows_us * k_us
            + 2 * b128_scale_count(rows_us, k_us)
                .ok_or_else(|| Qwen4Error::Invalid(format!("{name} scale count overflows")))?;
        let pad = (stride - used) as u64;
        let shape = shape_u32(&[experts as u64, rows, k], &name)?;
        Ok(StockEntry {
            name,
            quant_type: FP8E4M3B128_QUANT_TYPE,
            shape,
            group_size: B128_BLOCK as u32,
            data_len: data_len as u64,
            payload: StockPayload::Experts { blocks, pad },
        })
    };
    let gate_up = make(
        format!("{prefix}.mlp.experts.gate_up_proj"),
        2 * gate_rows,
        hidden,
        gate_up_blocks,
    )?;
    let down = make(
        format!("{prefix}.mlp.experts.down_proj"),
        hidden,
        gate_rows,
        down_blocks,
    )?;
    Ok(([gate_up, down], geometry))
}

/// Classify every source tensor and plan the stock-FP8 entries.  Reads only
/// the three I64 records and the PLE scale records.  `index` is the Flash-Next
/// manifest (production only).
fn plan_stock_entries(
    tensors: Vec<SourceTensor>,
    mode: Qwen4Mode,
    index: Option<&ManifestIndex>,
) -> Result<StockInventory, Qwen4Error> {
    let mut census = SourceCensus::default();
    let mut fp8 = Vec::new();
    let mut omitted = Vec::new();
    let mut metadata_sources = BTreeMap::<I64Role, SourceTensor>::new();
    let mut ple_codes = BTreeMap::<usize, SourceTensor>::new();
    let mut ple_scales = BTreeMap::<String, SourceTensor>::new();
    let mut layers = BTreeMap::<String, LayerExperts>::new();
    let mut bf16 = Vec::<SourceTensor>::new();

    for tensor in tensors {
        census.total += 1;
        census.bytes += tensor.data_len();
        *census.by_dtype.entry(tensor.dtype.clone()).or_default() += 1;
        if tensor.dtype == FP8_DTYPE {
            fp8.push(tensor.clone());
        }
        let name = tensor.name.clone();
        if name.starts_with("mtp.") {
            match tensor.dtype.as_str() {
                FP8_DTYPE => census.f8_mtp += 1,
                "BF16" => {}
                other => {
                    return invalid(format!(
                        "mtp tensor {name} has dtype {other}, expected F8_E4M3 or BF16"
                    ));
                }
            }
            omitted.push(Omitted {
                tensor,
                reason: "mtp",
            });
            continue;
        }
        if is_vision_tensor(&name) {
            require_dtype(&tensor, "BF16", "vision tensor")?;
            omitted.push(Omitted {
                tensor,
                reason: "vision",
            });
            continue;
        }
        if let Some(role) = i64_role(&name) {
            require_dtype(&tensor, "I64", "PLE metadata record")?;
            if metadata_sources.insert(role, tensor).is_some() {
                return invalid(format!("duplicate PLE metadata role {role:?} ({name})"));
            }
            continue;
        }
        if tensor.dtype == "I64" {
            return invalid(format!("unexpected I64 tensor {name}"));
        }
        if let Some(shard) = ple_shard_index(&name) {
            require_dtype(&tensor, FP8_DTYPE, "PLE shard")?;
            if tensor.shape.len() != 2 || tensor.shape[1] != PLE_ROW_WIDTH || tensor.shape[0] == 0 {
                return invalid(format!(
                    "PLE shard {name} has shape {:?}, expected [rows, {PLE_ROW_WIDTH}]",
                    tensor.shape
                ));
            }
            census.f8_ple += 1;
            if ple_codes.insert(shard, tensor).is_some() {
                return invalid(format!("duplicate PLE shard index {shard}"));
            }
            continue;
        }
        if name.ends_with(PLE_SCALE_SUFFIX) {
            require_dtype(&tensor, "BF16", "PLE scale")?;
            if !is_scalar_shape(&tensor.shape) || tensor.data_len() != 2 {
                return invalid(format!(
                    "PLE scale {name} has shape {:?} ({} bytes), expected a single BF16 ([], [1] or [1,1])",
                    tensor.shape,
                    tensor.data_len()
                ));
            }
            ple_scales.insert(name, tensor);
            continue;
        }
        if name.contains(".mlp.experts.") {
            let parsed = parse_expert_name(&name).ok_or_else(|| {
                Qwen4Error::Invalid(format!(
                    "{name}: not a separate routed-expert tensor \
                     (<layer>.mlp.experts.E.{{gate,up,down}}_proj.{{weight,weight_scale_inv}})"
                ))
            })?;
            if parsed.scale {
                require_dtype(&tensor, "BF16", "expert scale")?;
            } else {
                require_dtype(&tensor, FP8_DTYPE, "expert weight")?;
                census.f8_routed += 1;
            }
            let layer = layers.entry(parsed.prefix.to_string()).or_default();
            let slot = if parsed.scale {
                &mut layer.scales
            } else {
                &mut layer.weights
            };
            if slot[parsed.proj as usize]
                .insert(parsed.expert, tensor)
                .is_some()
            {
                return invalid(format!("duplicate expert tensor {name}"));
            }
            continue;
        }
        if tensor.dtype != "BF16" {
            return invalid(format!(
                "{name} has dtype {}; stock-fp8 carries only BF16 outside routed experts, PLE and mtp.*",
                tensor.dtype
            ));
        }
        bf16.push(tensor);
    }

    if mode == Qwen4Mode::Production {
        let count = |dtype: &str| census.by_dtype.get(dtype).copied().unwrap_or(0);
        let f8 = count(FP8_DTYPE);
        let bf = count("BF16");
        let i64s = count("I64");
        let other: Vec<_> = census
            .by_dtype
            .iter()
            .filter(|(dtype, _)| !matches!(dtype.as_str(), "F8_E4M3" | "BF16" | "I64"))
            .collect();
        if f8 != STOCK_F8_TENSORS
            || bf != STOCK_BF16_TENSORS
            || i64s != STOCK_I64_TENSORS
            || !other.is_empty()
            || census.f8_routed != STOCK_F8_ROUTED
            || census.f8_ple != PLE_SHARD_COUNT as u64
            || census.f8_mtp != STOCK_F8_MTP
        {
            return invalid(format!(
                "stock FP8 production census mismatch: F8_E4M3 {f8} (routed {}, PLE {}, mtp {}), \
                 BF16 {bf}, I64 {i64s}, other {other:?}; expected F8_E4M3 {STOCK_F8_TENSORS} \
                 (routed {STOCK_F8_ROUTED}, PLE {PLE_SHARD_COUNT}, mtp {STOCK_F8_MTP}), \
                 BF16 {STOCK_BF16_TENSORS}, I64 {STOCK_I64_TENSORS}",
                census.f8_routed, census.f8_ple, census.f8_mtp
            ));
        }
    }

    // Tied aliases are logical, not stored (as in the Flash-Next producer).
    if let Some(index) = index {
        let mut kept = Vec::with_capacity(bf16.len());
        for tensor in bf16 {
            if let Some(expected) = index.aliases.get(&tensor.name) {
                if tensor.shape != expected.shape {
                    return invalid(format!(
                        "tied alias {} has shape {:?}, expected {:?}",
                        tensor.name, tensor.shape, expected.shape
                    ));
                }
                omitted.push(Omitted {
                    tensor,
                    reason: "tied_alias",
                });
            } else {
                kept.push(tensor);
            }
        }
        bf16 = kept;
    }

    // Routed experts.
    if layers.is_empty() {
        return invalid("stock-fp8 input has no routed experts");
    }
    if mode == Qwen4Mode::Production && layers.len() != STOCK_EXPERT_LAYERS {
        return invalid(format!(
            "stock-fp8 production requires {STOCK_EXPERT_LAYERS} routed-expert layers, found {}",
            layers.len()
        ));
    }
    let mut resident = Vec::<StockEntry>::new();
    let mut geometry: Option<ExpertGeometry> = None;
    let mut expert_entries = 0usize;
    for (prefix, layer) in layers {
        let (pair, layer_geometry) = assemble_layer(&prefix, layer, mode)?;
        match geometry {
            None => geometry = Some(layer_geometry),
            Some(first) if first != layer_geometry => {
                return invalid(format!(
                    "{prefix}: expert geometry {layer_geometry:?} differs from {first:?}"
                ));
            }
            Some(_) => {}
        }
        expert_entries += pair.len();
        resident.extend(pair);
    }
    let geometry = geometry.expect("at least one layer");

    // BF16 verbatim.
    for tensor in bf16 {
        let data_len = checked_bf16_bytes(&tensor.shape, &tensor.name)?;
        if tensor.data_len() != data_len {
            return invalid(format!(
                "{} has {} payload bytes, expected {data_len}",
                tensor.name,
                tensor.data_len()
            ));
        }
        resident.push(StockEntry {
            name: tensor.name.clone(),
            quant_type: 16,
            shape: shape_u32(&tensor.shape, &tensor.name)?,
            group_size: 0,
            data_len,
            payload: StockPayload::Verbatim(tensor),
        });
    }

    // Manifest agreement + order.
    if let Some(index) = index {
        let mut expected = BTreeMap::<&str, &[u64]>::new();
        for (name, entry) in &index.required {
            if entry.is_mtp || entry.role == ManifestRole::Ple {
                continue;
            }
            expected.insert(name.as_str(), entry.shape.as_slice());
        }
        let actual: BTreeMap<&str, Vec<u64>> = resident
            .iter()
            .map(|entry| {
                (
                    entry.name.as_str(),
                    entry.shape.iter().map(|&d| u64::from(d)).collect(),
                )
            })
            .collect();
        let missing: Vec<String> = expected
            .keys()
            .filter(|name| !actual.contains_key(*name))
            .map(|name| name.to_string())
            .collect();
        let extra: Vec<String> = actual
            .keys()
            .filter(|name| !expected.contains_key(*name))
            .map(|name| name.to_string())
            .collect();
        if !missing.is_empty() || !extra.is_empty() {
            return invalid(format!(
                "stock-fp8 source disagrees with the Flash-Next manifest: {} missing {}, {} undeclared {}",
                missing.len(),
                list_sample(&missing),
                extra.len(),
                list_sample(&extra)
            ));
        }
        for (name, shape) in &actual {
            if expected[name] != shape.as_slice() {
                return invalid(format!(
                    "{name} has shape {shape:?}, manifest expects {:?}",
                    expected[name]
                ));
            }
        }
        resident.sort_by_key(|entry| {
            index
                .manifest_order
                .get(&entry.name)
                .copied()
                .unwrap_or(usize::MAX)
        });
    } else {
        resident.sort_by(|left, right| left.name.cmp(&right.name));
    }

    // PLE shards.
    if ple_codes.is_empty() {
        return invalid("stock-fp8 input has no PLE shards");
    }
    for (expected, actual) in ple_codes.keys().enumerate() {
        if *actual != expected {
            return invalid(format!(
                "PLE shard indexes must be contiguous from zero, found {actual} at position {expected}"
            ));
        }
    }
    if mode == Qwen4Mode::Production {
        if ple_codes.len() != PLE_SHARD_COUNT {
            return invalid(format!(
                "stock-fp8 requires {PLE_SHARD_COUNT} PLE shards, found {}",
                ple_codes.len()
            ));
        }
        for (shard, tensor) in &ple_codes {
            if tensor.shape != [PLE_ROWS_PER_SHARD, PLE_ROW_WIDTH] {
                return invalid(format!(
                    "PLE shard {} has shape {:?}, expected [{PLE_ROWS_PER_SHARD}, {PLE_ROW_WIDTH}]",
                    tensor.name, tensor.shape
                ));
            }
            if let Some(index) = index {
                if index.ple_names.get(shard) != Some(&tensor.name) {
                    return invalid(format!(
                        "PLE shard {shard} is {}, manifest declares {:?}",
                        tensor.name,
                        index.ple_names.get(shard)
                    ));
                }
            }
        }
    }
    let mut ple_scale_cache = BTreeMap::<String, ([u8; 2], [u8; 32])>::new();
    let mut ple_entries = Vec::with_capacity(ple_codes.len());
    for (_, codes) in ple_codes {
        let table = codes
            .name
            .find(".shard_")
            .map(|at| &codes.name[..at])
            .ok_or_else(|| {
                Qwen4Error::Invalid(format!("{} is not a PLE shard name", codes.name))
            })?;
        let scale_name = format!("{table}.weight_scale");
        let scale = ple_scales.get(&scale_name).ok_or_else(|| {
            Qwen4Error::Invalid(format!(
                "PLE shard {} has no scale tensor {scale_name}",
                codes.name
            ))
        })?;
        if !ple_scale_cache.contains_key(&scale_name) {
            let mut bits = [0u8; 2];
            scale.read_range(0, &mut bits)?;
            let mut digest = [0u8; 32];
            digest.copy_from_slice(Sha256::digest(bits).as_slice());
            ple_scale_cache.insert(scale_name.clone(), (bits, digest));
        }
        let (bits, _) = ple_scale_cache[&scale_name];
        let rows = codes.shape[0];
        let data_len = pt_tensor_bytes(usize_of(rows, "PLE rows")?, PLE_ROW_WIDTH as usize)
            .ok_or_else(|| Qwen4Error::Invalid(format!("{} payload overflows", codes.name)))?
            as u64;
        ple_entries.push(StockEntry {
            name: codes.name.clone(),
            quant_type: FP8E4M3PT_QUANT_TYPE,
            shape: shape_u32(&codes.shape, &codes.name)?,
            group_size: 0,
            data_len,
            payload: StockPayload::Ple {
                header: pt_header(u16::from_le_bytes(bits)),
                codes,
                scale: scale.clone(),
            },
        });
    }
    if let Some(unused) = ple_scales
        .keys()
        .find(|name| !ple_scale_cache.contains_key(*name))
    {
        return invalid(format!("PLE scale {unused} belongs to no shard"));
    }
    let ple_scale_digests: BTreeMap<String, [u8; 32]> = ple_scale_cache
        .into_iter()
        .map(|(name, (_, digest))| (name, digest))
        .collect();

    // The three I64 PLE records (qt=54), validated exactly as Flash-Next does.
    let mut ple_metadata = PleMetadata {
        multipliers: Vec::new(),
        vocab_sizes: Vec::new(),
        prefix_offsets: Vec::new(),
    };
    let mut metadata_entries = Vec::new();
    for role in [I64Role::Multipliers, I64Role::VocabSizes, I64Role::Offsets] {
        let source = metadata_sources.remove(&role).ok_or_else(|| {
            Qwen4Error::Invalid(format!(
                "stock-fp8 input is missing PLE metadata role {role:?}"
            ))
        })?;
        let expected_len = match role {
            I64Role::Multipliers => 3,
            I64Role::VocabSizes | I64Role::Offsets => PLE_HEAD_COUNT,
        };
        let values = read_i64_array(&source, expected_len)?;
        if let Some(index) = index {
            let declared = index.metadata.get(&source.name);
            if declared.map(|d| (d.role, d.shape.as_slice()))
                != Some((role, source.shape.as_slice()))
            {
                return invalid(format!(
                    "{} is not the manifest's {role:?} metadata record",
                    source.name
                ));
            }
        }
        match role {
            I64Role::Multipliers => ple_metadata.multipliers = values,
            I64Role::VocabSizes => ple_metadata.vocab_sizes = values,
            I64Role::Offsets => ple_metadata.prefix_offsets = values,
        }
        metadata_entries.push(StockEntry {
            name: source.name.clone(),
            quant_type: QWEN4_I64_QUANT_TYPE,
            shape: shape_u32(&source.shape, &source.name)?,
            group_size: 0,
            data_len: source.data_len(),
            payload: StockPayload::Verbatim(source),
        });
    }
    validate_ple_metadata(
        &ple_metadata.multipliers,
        &ple_metadata.vocab_sizes,
        &ple_metadata.prefix_offsets,
    )?;

    let ple_shards = ple_entries.len();
    let mut entries = ple_entries;
    entries.extend(resident);
    entries.extend(metadata_entries);
    Ok(StockInventory {
        entries,
        ple_metadata,
        omitted,
        fp8,
        census,
        geometry,
        expert_entries,
        ple_shards,
        ple_scale_digests,
    })
}

/// Flash-Next manifest index for the production cross-check.  Only names and
/// shapes are used; every role (including the packed trunk tiers it declares)
/// is carried as BF16 by this recipe.
fn production_index(config: &Value) -> Result<ManifestIndex, Qwen4Error> {
    let config = Qwen4Config::from_value(config)
        .map_err(|error| Qwen4Error::Invalid(format!("Qwen4 config is invalid: {error}")))?;
    let manifest = Qwen4Manifest::build(&config).map_err(|error| {
        Qwen4Error::Invalid(format!("Qwen4 manifest could not be built: {error}"))
    })?;
    ManifestIndex::build(&manifest)
}

// --------------------------------------------------------------------- FP8 scan

#[derive(Clone, Copy, Debug)]
struct ScanDigest {
    sha256: [u8; 32],
    count_0x80: u64,
}

struct ScanJob {
    file: Arc<File>,
    path: PathBuf,
    /// `(name, absolute start, absolute end)`, ascending.
    tensors: Vec<(String, u64, u64)>,
}

fn scan_job(job: ScanJob) -> Result<Vec<(String, ScanDigest)>, Qwen4Error> {
    let mut buf = vec![0u8; IO_CHUNK_BYTES];
    let mut out = Vec::with_capacity(job.tensors.len());
    for (name, start, end) in &job.tensors {
        let mut hasher = Sha256::new();
        let mut count_0x80 = 0u64;
        let mut position = *start;
        while position < *end {
            let n = (buf.len() as u64).min(end - position) as usize;
            read_exact_at(&job.file, position, &mut buf[..n]).map_err(|error| {
                Qwen4Error::io(format!("scan {name} in {}", job.path.display()), error)
            })?;
            let scan = scan_codes(&buf[..n]);
            if scan.nan > 0 {
                let at = buf[..n]
                    .iter()
                    .position(|byte| byte & 0x7f == 0x7f)
                    .unwrap_or(0);
                return invalid(format!(
                    "source tensor {name} ({}) contains the E4M3 NaN code 0x{:02X} at tensor byte {}; \
                     {} NaN code(s) in the scanned chunk; refusing to convert",
                    job.path.display(),
                    buf[at],
                    position - start + at as u64,
                    scan.nan
                ));
            }
            count_0x80 += scan.neg_zero;
            hasher.update(&buf[..n]);
            position += n as u64;
        }
        let mut sha256 = [0u8; 32];
        sha256.copy_from_slice(hasher.finalize().as_slice());
        out.push((name.clone(), ScanDigest { sha256, count_0x80 }));
    }
    Ok(out)
}

/// Scan EVERY F8_E4M3 source tensor (emitted and omitted) in bounded reads,
/// shards in parallel.  No output exists yet.
fn scan_fp8_sources(fp8: &[SourceTensor]) -> Result<HashMap<String, ScanDigest>, Qwen4Error> {
    let mut by_file = BTreeMap::<PathBuf, ScanJob>::new();
    for tensor in fp8 {
        let SourceKind::Local { file } = &tensor.shard.kind else {
            return invalid("stock-fp8 scans local source files only");
        };
        by_file
            .entry(tensor.shard.path.clone())
            .or_insert_with(|| ScanJob {
                file: Arc::clone(file),
                path: tensor.shard.path.clone(),
                tensors: Vec::new(),
            })
            .tensors
            .push((tensor.name.clone(), tensor.data_start, tensor.data_end));
    }
    let mut jobs: Vec<ScanJob> = by_file.into_values().collect();
    for job in &mut jobs {
        job.tensors.sort_by_key(|(_, start, _)| *start);
    }
    let scanned = jobs
        .into_par_iter()
        .map(scan_job)
        .collect::<Result<Vec<_>, Qwen4Error>>()?;
    Ok(scanned.into_iter().flatten().collect())
}

// ------------------------------------------------------------------- payload IO

struct Copier<'a> {
    buf: Vec<u8>,
    copied: HashMap<String, [u8; 32]>,
    scan: &'a HashMap<String, ScanDigest>,
}

impl Copier<'_> {
    fn copy(&mut self, tensor: &SourceTensor, writer: &mut dyn Write) -> Result<(), Qwen4Error> {
        let total = tensor.data_len();
        let mut hasher = Sha256::new();
        let mut done = 0u64;
        while done < total {
            let n = (self.buf.len() as u64).min(total - done) as usize;
            tensor.read_range(done, &mut self.buf[..n])?;
            hasher.update(&self.buf[..n]);
            writer.write_all(&self.buf[..n]).map_err(|error| {
                Qwen4Error::io(format!("write payload of {}", tensor.name), error)
            })?;
            done += n as u64;
        }
        let mut digest = [0u8; 32];
        digest.copy_from_slice(hasher.finalize().as_slice());
        match self.scan.get(&tensor.name) {
            Some(scan) if scan.sha256 != digest => {
                return invalid(format!(
                    "source tensor {} changed between the NaN scan and the copy",
                    tensor.name
                ));
            }
            None if tensor.dtype == FP8_DTYPE => {
                return invalid(format!(
                    "FP8 source tensor {} was never scanned",
                    tensor.name
                ));
            }
            _ => {}
        }
        self.copied.insert(tensor.name.clone(), digest);
        Ok(())
    }

    fn write_entry(
        &mut self,
        entry: &StockEntry,
        writer: &mut dyn Write,
    ) -> Result<(), Qwen4Error> {
        match &entry.payload {
            StockPayload::Verbatim(tensor) => self.copy(tensor, writer),
            StockPayload::Experts { blocks, pad } => {
                for block in blocks {
                    for tensor in block.codes.iter().chain(&block.scales) {
                        self.copy(tensor, writer)?;
                    }
                    writer.write_all(&ZEROS[..*pad as usize]).map_err(|error| {
                        Qwen4Error::io(format!("write pad of {}", entry.name), error)
                    })?;
                }
                Ok(())
            }
            StockPayload::Ple { header, codes, .. } => {
                writer.write_all(header).map_err(|error| {
                    Qwen4Error::io(format!("write PLE header of {}", entry.name), error)
                })?;
                self.copy(codes, writer)
            }
        }
    }
}

// ---------------------------------------------------------------- reopen + hash

struct IndexCursor<'a> {
    region: &'a [u8],
    pos: usize,
}

impl<'a> IndexCursor<'a> {
    fn take(&mut self, len: usize, what: &str) -> Result<&'a [u8], Qwen4Error> {
        if self.region.len().saturating_sub(self.pos) < len {
            return invalid(format!("index truncated reading {what}"));
        }
        let slice = &self.region[self.pos..self.pos + len];
        self.pos += len;
        Ok(slice)
    }

    fn u16(&mut self, what: &str) -> Result<u16, Qwen4Error> {
        Ok(u16::from_le_bytes(
            self.take(2, what)?.try_into().expect("2 bytes"),
        ))
    }

    fn u32(&mut self, what: &str) -> Result<u32, Qwen4Error> {
        Ok(u32::from_le_bytes(
            self.take(4, what)?.try_into().expect("4 bytes"),
        ))
    }

    fn u64(&mut self, what: &str) -> Result<u64, Qwen4Error> {
        Ok(u64::from_le_bytes(
            self.take(8, what)?.try_into().expect("8 bytes"),
        ))
    }
}

/// Reopen the finished temporary with the HFQM container rules and the
/// stock quant types.  (`Qwen4ReopenPlan::open` is the Flash-Next reader and
/// deliberately refuses qt 55/56; this one is dedicated and does not weaken it.)
fn reopen_stock_artifact(path: &Path) -> Result<Qwen4ReopenPlan, Qwen4Error> {
    let file =
        File::open(path).map_err(|error| Qwen4Error::io(path.display().to_string(), error))?;
    let file_len = file
        .metadata()
        .map_err(|error| Qwen4Error::io(format!("stat {}", path.display()), error))?
        .len();
    if file_len < 32 {
        return invalid(format!(
            "{} is shorter than its HFQM header",
            path.display()
        ));
    }
    let mut header = [0u8; 32];
    read_exact_at(&file, 0, &mut header)
        .map_err(|error| Qwen4Error::io(format!("read header of {}", path.display()), error))?;
    let word = |at: usize| u32::from_le_bytes(header[at..at + 4].try_into().expect("4 bytes"));
    let long = |at: usize| u64::from_le_bytes(header[at..at + 8].try_into().expect("8 bytes"));
    if &header[..4] != b"HFQM" || word(4) != 1 {
        return invalid(format!("{} has a bad HFQM magic/version", path.display()));
    }
    let arch_id = word(8);
    if arch_id != QWEN4_ARCH_ID {
        return invalid(format!(
            "{} has arch id {arch_id}, expected {QWEN4_ARCH_ID}",
            path.display()
        ));
    }
    let tensor_count = word(12) as usize;
    let (metadata_offset, data_offset) = (long(16), long(24));
    if metadata_offset < 32 || metadata_offset > data_offset || data_offset > file_len {
        return invalid(format!(
            "{} has invalid metadata/data offsets {metadata_offset}/{data_offset}",
            path.display()
        ));
    }
    let region_len = data_offset - metadata_offset;
    if region_len > MAX_REOPEN_REGION_BYTES {
        return invalid(format!(
            "metadata/index region is {region_len} bytes, exceeds {MAX_REOPEN_REGION_BYTES}"
        ));
    }
    let mut region = vec![0u8; usize_of(region_len, "metadata/index region")?];
    read_exact_at(&file, metadata_offset, &mut region)
        .map_err(|error| Qwen4Error::io(format!("read index of {}", path.display()), error))?;
    let json_end = json_blob_end(&region)
        .ok_or_else(|| Qwen4Error::Invalid("metadata JSON is not terminated".to_string()))?;
    let metadata_json = String::from_utf8(region[..json_end].to_vec())
        .map_err(|error| Qwen4Error::Invalid(format!("metadata is not UTF-8: {error}")))?;

    let mut cursor = IndexCursor {
        region: &region,
        pos: json_end,
    };
    let index_count = cursor.u32("count")? as usize;
    if index_count != tensor_count {
        return invalid(format!(
            "header has {tensor_count} tensors but the index has {index_count}"
        ));
    }
    let mut entries = Vec::with_capacity(index_count);
    let mut seen = BTreeSet::new();
    let mut cumulative = data_offset;
    for _ in 0..index_count {
        let name_len = cursor.u16("name length")? as usize;
        let name = String::from_utf8(cursor.take(name_len, "name")?.to_vec())
            .map_err(|error| Qwen4Error::Invalid(format!("tensor name is not UTF-8: {error}")))?;
        let quant_type = cursor.take(1, "quant type")?[0];
        let dims = cursor.take(1, "rank")?[0] as usize;
        let mut shape = Vec::with_capacity(dims);
        for _ in 0..dims {
            shape.push(cursor.u32("dimension")?);
        }
        let group_size = cursor.u32("group size")?;
        let data_len = cursor.u64("data length")?;
        let end = cumulative
            .checked_add(data_len)
            .filter(|end| *end <= file_len)
            .ok_or_else(|| {
                Qwen4Error::Invalid(format!(
                    "{name} payload exceeds the file ({file_len} bytes)"
                ))
            })?;
        if !seen.insert(name.clone()) {
            return invalid(format!("duplicate output tensor {name}"));
        }
        validate_stock_entry_len(&name, quant_type, &shape, data_len)?;
        entries.push(Qwen4PlanEntry {
            name,
            quant_type,
            shape,
            group_size,
            data_offset: cumulative,
            data_len,
        });
        cumulative = end;
    }
    if cumulative != file_len {
        return invalid(format!(
            "artifact payload ends at {cumulative}, file ends at {file_len}"
        ));
    }
    Ok(Qwen4ReopenPlan {
        arch_id,
        metadata_json,
        entries,
    })
}

fn validate_stock_entry_len(
    name: &str,
    quant_type: u8,
    shape: &[u32],
    data_len: u64,
) -> Result<(), Qwen4Error> {
    match quant_type {
        16 | QWEN4_I64_QUANT_TYPE => validate_output_entry_len(name, quant_type, shape, data_len),
        FP8E4M3B128_QUANT_TYPE => {
            let [experts, rows, k] = shape else {
                return invalid(format!("{name} qt=55 shape {shape:?} is not [E, R, K]"));
            };
            let expected = b128_tensor_bytes(*experts as usize, *rows as usize, *k as usize)
                .ok_or_else(|| Qwen4Error::Invalid(format!("{name} qt=55 length overflows")))?;
            if expected as u64 != data_len {
                return invalid(format!(
                    "{name} qt=55 payload is {data_len} bytes, expected {expected}"
                ));
            }
            Ok(())
        }
        FP8E4M3PT_QUANT_TYPE => {
            let [rows, k] = shape else {
                return invalid(format!("{name} qt=56 shape {shape:?} is not [rows, K]"));
            };
            let expected = pt_tensor_bytes(*rows as usize, *k as usize)
                .ok_or_else(|| Qwen4Error::Invalid(format!("{name} qt=56 length overflows")))?;
            if expected as u64 != data_len {
                return invalid(format!(
                    "{name} qt=56 payload is {data_len} bytes, expected {expected}"
                ));
            }
            Ok(())
        }
        other => invalid(format!(
            "{name} uses unsupported stock-fp8 quant type {other}"
        )),
    }
}

fn hash_file_streaming(path: &Path) -> Result<([u8; 32], u64), Qwen4Error> {
    let mut file =
        File::open(path).map_err(|error| Qwen4Error::io(path.display().to_string(), error))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; IO_CHUNK_BYTES];
    let mut total = 0u64;
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|error| Qwen4Error::io(format!("hash {}", path.display()), error))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        total += n as u64;
    }
    let mut digest = [0u8; 32];
    digest.copy_from_slice(hasher.finalize().as_slice());
    Ok((digest, total))
}

/// Removes its paths on drop; published files are already renamed away.
#[derive(Default)]
struct TempCleanup(Vec<PathBuf>);

impl Drop for TempCleanup {
    fn drop(&mut self) {
        for path in &self.0 {
            let _ = fs::remove_file(path);
        }
    }
}

// ---------------------------------------------------------------------- driver

fn build_recipe(
    inventory: &StockInventory,
    manifest_sha256: Option<&str>,
    mode: Qwen4Mode,
) -> Value {
    let geometry = inventory.geometry;
    let rows_gate_up = 2 * geometry.gate_rows;
    json!({
        "version": 3,
        "id": STOCK_FP8_RECIPE_ID,
        "stacked_experts": true,
        "routing": {"num_experts": geometry.experts, "top_k": ROUTER_TOP_K},
        "matrix": {"trunk": []},
        "gate_up": {
            "quant_type": FP8E4M3B128_QUANT_TYPE, "format": "FP8E4M3B128",
            "group_size": B128_BLOCK, "rows": rows_gate_up, "k": geometry.hidden,
        },
        "down": {
            "quant_type": FP8E4M3B128_QUANT_TYPE, "format": "FP8E4M3B128",
            "group_size": B128_BLOCK, "rows": geometry.hidden, "k": geometry.gate_rows,
        },
        "ple": {
            "quant_type": FP8E4M3PT_QUANT_TYPE, "format": "FP8E4M3PT",
            "group_size": 0, "header_bytes": PT_HEADER_BYTES,
        },
        "trunk": "bf16",
        "mtp": "omitted-ar-only",
        "vision": "omitted-text-only",
        "source_manifest_sha256": manifest_sha256,
        "component_mode": match mode {
            Qwen4Mode::Production => "production",
            Qwen4Mode::CompactFixture => "compact-fixture",
        },
    })
}

fn source_json(
    use_: &SourceUse<'_>,
    scan: &HashMap<String, ScanDigest>,
    copied: &HashMap<String, [u8; 32]>,
) -> Value {
    let tensor = use_.tensor;
    let digest = copied
        .get(&tensor.name)
        .map(|digest| hex_lower(digest))
        .unwrap_or_default();
    let file = tensor
        .shard
        .path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut record = json!({
        "name": tensor.name,
        "role": use_.role,
        "dtype": tensor.dtype,
        "shape": tensor.shape,
        "file": file,
        "start": tensor.data_start,
        "end": tensor.data_end,
        "bytes": tensor.data_len(),
        "sha256": digest,
    });
    if let Some(expert) = use_.expert {
        record["expert"] = json!(expert);
    }
    if let Some(scan) = scan.get(&tensor.name) {
        record["count_0x80"] = json!(scan.count_0x80);
    }
    record
}

/// Convert the stock FP8 checkpoint to a stock-FP8 HFQM artifact plus receipt.
pub(crate) fn write_stock_fp8_artifact(
    options: &StockFp8Options<'_>,
) -> Result<StockFp8Summary, Qwen4Error> {
    let started = Instant::now();
    if options
        .input
        .to_str()
        .is_some_and(|input| input.starts_with("hf://"))
    {
        return invalid("stock-fp8 needs a local checkpoint directory (fp8.sha256 must be hashed)");
    }
    if options.mode == Qwen4Mode::Production
        && options.pin != Some(STOCK_FP8_PINNED_MANIFEST_SHA256)
    {
        return invalid("stock-fp8 production mode requires the pinned manifest SHA-256");
    }
    let receipt_path = receipt_path_for(options.output);
    for path in [options.output, receipt_path.as_path()] {
        if path.exists() {
            return invalid(format!(
                "refusing to replace existing stock-fp8 output {}",
                path.display()
            ));
        }
    }

    // Gate 2: the manifest pin, before any source tensor or header is read.
    let manifest = match options.pin {
        Some(pin) => {
            if !options.input.is_dir() {
                return invalid(format!(
                    "stock-fp8 needs the checkpoint directory holding {STOCK_FP8_MANIFEST_NAME}: {}",
                    options.input.display()
                ));
            }
            Some(verify_manifest(options.input, pin)?)
        }
        None => None,
    };

    let source = source_paths(options.input)?;
    let SourceSet::Local { paths, .. } = &source else {
        return invalid("stock-fp8 needs a local source");
    };
    validate_source_shard_count(&source, options.mode)?;
    if let Some(manifest) = &manifest {
        verify_manifest_files(options.input, manifest, paths)?;
    }
    let tensors = load_inventory(&source)?;
    let config_value = load_optional_config(&source)?.ok_or_else(|| {
        Qwen4Error::Invalid("stock-fp8 input is missing required config.json".to_string())
    })?;
    let (index, tokenizer) = if options.mode == Qwen4Mode::Production {
        (
            Some(production_index(&config_value)?),
            Some(load_tokenizer_metadata(&source)?),
        )
    } else {
        (None, None)
    };

    let inventory = plan_stock_entries(tensors, options.mode, index.as_ref())?;
    let manifest_sha256 = manifest.as_ref().map(|manifest| manifest.sha256.as_str());
    let recipe = build_recipe(&inventory, manifest_sha256, options.mode);
    let streaming = json!({
        "version": 1,
        "recipe": STOCK_FP8_RECIPE_ID,
        "resident_entries": inventory.entries.len() - inventory.ple_shards,
        "expert_entries": inventory.expert_entries,
        "external_ple_entries": inventory.ple_shards,
        "io_chunk_bytes": IO_CHUNK_BYTES,
    });
    let metadata_json = build_metadata_document(
        Some(&config_value),
        recipe,
        streaming,
        &inventory.ple_metadata,
        tokenizer.as_ref(),
    )?;
    let stream_entries: Vec<hipfire_runtime::hfq::HfqStreamEntry> = inventory
        .entries
        .iter()
        .map(|entry| hipfire_runtime::hfq::HfqStreamEntry {
            name: entry.name.clone(),
            quant_type: entry.quant_type,
            shape: entry.shape.clone(),
            group_size: entry.group_size,
            data_len: entry.data_len,
        })
        .collect();
    if stream_entries.len() > u32::MAX as usize {
        return invalid(format!("too many entries: {}", stream_entries.len()));
    }
    let predicted_output_bytes = predicted_output_bytes(&metadata_json, &stream_entries)?;
    let source_records: u64 = inventory
        .entries
        .iter()
        .map(|entry| entry.sources().len() as u64)
        .sum();
    let receipt_estimate = RECEIPT_FIXED_BYTES + RECEIPT_BYTES_PER_SOURCE * source_records;
    capacity_preflight(
        options.output,
        predicted_output_bytes
            .checked_add(receipt_estimate)
            .ok_or_else(|| Qwen4Error::Invalid("predicted output size overflows".to_string()))?,
    )?;

    // Gate 4: every FP8 tensor scanned before any output exists.
    let scan = scan_fp8_sources(&inventory.fp8)?;
    source.verify_identity()?;

    // Write, fsync, verify.
    let temporary = temporary_output_path(options.output)?;
    let mut cleanup = TempCleanup(vec![temporary.clone()]);
    let mut copier = Copier {
        buf: vec![0u8; IO_CHUNK_BYTES],
        copied: HashMap::new(),
        scan: &scan,
    };
    for (name, digest) in &inventory.ple_scale_digests {
        copier.copied.insert(name.clone(), *digest);
    }
    hipfire_runtime::hfq::write_hfqm_package_streaming(
        &temporary,
        QWEN4_ARCH_ID,
        &metadata_json,
        &stream_entries,
        |index, writer| {
            copier
                .write_entry(&inventory.entries[index], writer)
                .map_err(Qwen4Error::into_io)
        },
    )
    .map_err(|error| {
        Qwen4Error::io(
            format!("write stock-fp8 artifact {}", options.output.display()),
            error,
        )
    })?;
    let copied = copier.copied;
    sync_file(&temporary)?;
    source.verify_identity()?;
    let reopened = reopen_stock_artifact(&temporary)?;
    reopened.validate_against(&metadata_json, &stream_entries)?;
    let (artifact_digest, artifact_bytes) = hash_file_streaming(&temporary)?;
    let artifact_sha256 = hex_lower(&artifact_digest);

    // Receipt.
    let omitted_bytes = |reason: &str| -> u64 {
        inventory
            .omitted
            .iter()
            .filter(|omitted| omitted.reason == reason)
            .map(|omitted| omitted.tensor.data_len())
            .sum()
    };
    let omitted_count = |reason: &str| -> usize {
        inventory
            .omitted
            .iter()
            .filter(|omitted| omitted.reason == reason)
            .count()
    };
    let omitted_fp8: Vec<&Omitted> = inventory
        .omitted
        .iter()
        .filter(|omitted| omitted.tensor.dtype == FP8_DTYPE)
        .collect();
    let omitted_fp8_json: Vec<Value> = omitted_fp8
        .iter()
        .map(|omitted| {
            let tensor = &omitted.tensor;
            let scanned = scan.get(&tensor.name);
            json!({
                "name": tensor.name,
                "reason": omitted.reason,
                "file": tensor.shard.path.file_name().map(|n| n.to_string_lossy().into_owned()),
                "start": tensor.data_start,
                "end": tensor.data_end,
                "sha256": scanned.map(|s| hex_lower(&s.sha256)),
                "count_0x80": scanned.map(|s| s.count_0x80),
            })
        })
        .collect();
    let count80_total: u64 = scan.values().map(|digest| digest.count_0x80).sum();
    let count80_omitted: u64 = omitted_fp8
        .iter()
        .filter_map(|omitted| scan.get(&omitted.tensor.name))
        .map(|digest| digest.count_0x80)
        .sum();
    let fp8_emitted = inventory.census.f8_routed + inventory.census.f8_ple;
    let pad_bytes: u64 = inventory
        .entries
        .iter()
        .map(|entry| match &entry.payload {
            StockPayload::Experts { blocks, pad } => pad * blocks.len() as u64,
            _ => 0,
        })
        .sum();
    let source_files: Vec<Value> = paths
        .iter()
        .map(|path| {
            json!({
                "name": path.file_name().map(|n| n.to_string_lossy().into_owned()),
                "path": absolute(path),
                "bytes": fs::metadata(path).map(|m| m.len()).ok(),
                "manifest_sha256": path
                    .file_name()
                    .and_then(|name| manifest.as_ref()?.files.get(&*name.to_string_lossy())),
            })
        })
        .collect();
    let tensors_json: Vec<Value> = inventory
        .entries
        .iter()
        .zip(&reopened.entries)
        .map(|(entry, placed)| {
            let sources: Vec<Value> = entry
                .sources()
                .iter()
                .map(|use_| source_json(use_, &scan, &copied))
                .collect();
            json!({
                "name": entry.name,
                "quant_type": entry.quant_type,
                "shape": entry.shape,
                "group_size": entry.group_size,
                "artifact_offset": placed.data_offset,
                "artifact_len": placed.data_len,
                "sources": sources,
            })
        })
        .collect();
    let receipt = json!({
        "receipt_version": 1,
        "recipe": STOCK_FP8_RECIPE_ID,
        "component_mode": match options.mode {
            Qwen4Mode::Production => "production",
            Qwen4Mode::CompactFixture => "compact-fixture",
        },
        "artifact_sha256": artifact_sha256,
        "artifact": {
            "path": absolute(options.output),
            "bytes": artifact_bytes,
            "sha256": artifact_sha256,
            "entries": inventory.entries.len(),
        },
        "manifest_sha256": manifest_sha256,
        "pinned_manifest_sha256": options.pin,
        "manifest_verified": manifest.is_some(),
        "input": absolute(options.input),
        "source_files": source_files,
        "census": {
            "source_tensors": inventory.census.total,
            "dtypes": inventory.census.by_dtype,
            "fp8": {
                "total": inventory.fp8.len(),
                "scanned": scan.len(),
                "nan_codes": 0,
                "emitted": fp8_emitted,
                "emitted_routed_experts": inventory.census.f8_routed,
                "emitted_ple_shards": inventory.census.f8_ple,
                "omitted": inventory.census.f8_mtp,
                "omitted_mtp": inventory.census.f8_mtp,
                "count_0x80": {
                    "total": count80_total,
                    "emitted": count80_total - count80_omitted,
                    "omitted": count80_omitted,
                },
            },
            "omitted": {
                "mtp": {"tensors": omitted_count("mtp"), "bytes": omitted_bytes("mtp")},
                "vision": {"tensors": omitted_count("vision"), "bytes": omitted_bytes("vision")},
                "tied_alias": {
                    "tensors": omitted_count("tied_alias"),
                    "bytes": omitted_bytes("tied_alias"),
                },
            },
            "emitted_entries": inventory.entries.len(),
            "expert_entries": inventory.expert_entries,
            "ple_shards": inventory.ple_shards,
        },
        "bytes": {
            "source_total": inventory.census.bytes,
            "source_omitted": omitted_bytes("mtp") + omitted_bytes("vision") + omitted_bytes("tied_alias"),
            "source_emitted": inventory.census.bytes
                - omitted_bytes("mtp") - omitted_bytes("vision") - omitted_bytes("tied_alias"),
            "artifact_total": artifact_bytes,
            "artifact_header_and_index": reopened.entries.first().map(|e| e.data_offset),
            "artifact_expert_pad_zero": pad_bytes,
            "artifact_ple_headers": PT_HEADER_BYTES as u64 * inventory.ple_shards as u64,
            "predicted_artifact": predicted_output_bytes,
        },
        "omitted_fp8": omitted_fp8_json,
        "tensors": tensors_json,
        "elapsed_seconds": started.elapsed().as_secs_f64(),
    });
    let receipt_temporary = temporary_output_path(&receipt_path)?;
    cleanup.0.push(receipt_temporary.clone());
    {
        let file = File::create(&receipt_temporary).map_err(|error| {
            Qwen4Error::io(format!("create {}", receipt_temporary.display()), error)
        })?;
        let mut writer = io::BufWriter::new(file);
        serde_json::to_writer(&mut writer, &receipt)
            .map_err(|error| Qwen4Error::json("serialize stock-fp8 receipt", error))?;
        writer.write_all(b"\n").map_err(|error| {
            Qwen4Error::io(format!("write {}", receipt_temporary.display()), error)
        })?;
        writer.flush().map_err(|error| {
            Qwen4Error::io(format!("flush {}", receipt_temporary.display()), error)
        })?;
    }
    sync_file(&receipt_temporary)?;

    // Publish: artifact first, then receipt; a failed receipt takes the
    // just-published artifact back so no artifact exists without its receipt.
    publish_qwen4(&temporary, options.output)?;
    if let Err(error) = publish_qwen4(&receipt_temporary, &receipt_path) {
        let _ = fs::remove_file(options.output);
        return Err(error);
    }

    Ok(StockFp8Summary {
        entries: inventory.entries.len(),
        expert_entries: inventory.expert_entries,
        ple_shards: inventory.ple_shards,
        fp8_scanned: scan.len(),
        mtp_omitted: omitted_count("mtp"),
        vision_omitted: omitted_count("vision"),
        predicted_output_bytes,
        artifact_bytes,
        artifact_sha256,
        receipt_path,
        elapsed_seconds: started.elapsed().as_secs_f64(),
    })
}

#[cfg(test)]
mod tests;
