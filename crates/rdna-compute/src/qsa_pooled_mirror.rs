// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Derived BF16 mirror of the QSA pooled index keys
//! (`HIPFIRE_QWEN4_QSA_POOLED_BF16`, on unless `0`; admitted only on exact gfx1151).
//!
//! `qsa_pool_rope` rounds every pooled key to BF16 before it stores the value
//! as F32, so the high 16 bits of each canonical F32 element are exactly its
//! BF16 value and the low 16 bits are zero. The mirror holds those high halves
//! (`[block][index_dim]`, BF16) so the builder BF16 score kernel
//! (`qsa_select_bf16_pm_gfx1151`) reads half the bytes of the F32 scorer while
//! ranking identical values. The canonical F32 pooled arena stays
//! authoritative: the mirror is derived, never snapshotted, and its block
//! watermark ([`QsaPooledBf16ValidBlocks`]) follows the semantics of the F16
//! arena's `QsaF16ValidLen`, counted in pooled blocks: owners clamp it
//! whenever pooled blocks are rewritten or rewound, only a successful
//! converter ([`qsa_pooled_bf16_sync`]) advances it.

use std::sync::atomic::{AtomicUsize, Ordering};

use hip_bridge::{HipError, HipResult, KernargBlob};

use crate::tensor_ops::QsaKvFormat;
use crate::{DType, Gpu, GpuTensor};

/// Pooled index width the mirror covers (the PM score kernels' fixed shape).
pub const QSA_POOLED_BF16_INDEX_DIM: usize = 128;

const MODULE: &str = "qsa_pooled_bf16_mirror";
const CONVERT: &str = "qsa_pooled_f32_to_bf16_mirror";
const PROBE: &str = "qsa_pooled_f32_low_nonzero_count";
const THREADS: usize = 256;
/// Elements per thread of both kernels.
const PER_THREAD: usize = 4;

/// `HIPFIRE_QWEN4_QSA_POOLED_BF16` (developer bool, on unless `0`), read once.
static QWEN4_QSA_POOLED_BF16: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    hipfire_config::developer_bool("HIPFIRE_QWEN4_QSA_POOLED_BF16", true)
});
/// The same switch `tensor_ops` reads for the PM score kernels
/// (`HIPFIRE_QWEN4_QSA_SCORE_PM`, on unless `0`): the mirror is only consumed
/// by the PM BF16 score kernel, so it is not admitted without it.
static QWEN4_QSA_SCORE_PM: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| hipfire_config::developer_bool("HIPFIRE_QWEN4_QSA_SCORE_PM", true));

/// Load-time admission, no GPU launch: the switch is on, exact gfx1151,
/// F32 QSA state, `index_dim` 128, and the builder BF16 score module
/// (`kernels::QSA_SELECT_BF16_PM_GFX1151`, the PM scorer's switch on) exists.
pub fn qsa_pooled_bf16_mirror_enabled(gpu: &Gpu, format: QsaKvFormat, index_dim: usize) -> bool {
    qsa_pooled_bf16_mirror_admits(
        *QWEN4_QSA_POOLED_BF16,
        *QWEN4_QSA_SCORE_PM,
        !crate::kernels::QSA_SELECT_BF16_PM_GFX1151.is_empty(),
        gpu.arch_caps.is_gfx1151(),
        format,
        index_dim,
    )
}

fn qsa_pooled_bf16_mirror_admits(
    switch: bool,
    score_pm: bool,
    module: bool,
    gfx1151: bool,
    format: QsaKvFormat,
    index_dim: usize,
) -> bool {
    switch
        && score_pm
        && module
        && gfx1151
        && format == QsaKvFormat::F32
        && index_dim == QSA_POOLED_BF16_INDEX_DIM
}

/// Host watermark, in pooled blocks, of the leading blocks whose mirror
/// elements equal the high halves of the canonical F32 pooled keys. Same
/// contract as `QsaF16ValidLen`: it travels with its owner, is never
/// snapshotted, owners [`clamp`](Self::clamp) it on any rewrite or rewind of
/// pooled blocks (and [`invalidate`](Self::invalidate) it when the pooled
/// bytes are replaced), and only the converter [`advance`](Self::advance)s it.
#[derive(Default)]
pub struct QsaPooledBf16ValidBlocks(AtomicUsize);

impl QsaPooledBf16ValidBlocks {
    /// The raw mark (not clamped to any committed count).
    pub fn get(&self) -> usize {
        self.0.load(Ordering::Relaxed)
    }

    /// Leading blocks that are valid while the F32 arena commits
    /// `committed_blocks` complete blocks.
    pub fn valid(&self, committed_blocks: usize) -> usize {
        self.get().min(committed_blocks)
    }

    /// Never claim more than `committed_blocks` after a rewind or rewrite.
    pub fn clamp(&self, committed_blocks: usize) {
        self.0.fetch_min(committed_blocks, Ordering::Relaxed);
    }

    /// The pooled bytes were replaced, restored or failed mid-write.
    pub fn invalidate(&self) {
        self.0.store(0, Ordering::Relaxed);
    }

    /// Record `end_blocks` valid leading blocks. Converter-only in practice;
    /// public so the owner can bind the contract in tests.
    pub fn advance(&self, end_blocks: usize) {
        self.0.store(end_blocks, Ordering::Relaxed);
    }
}

/// A layer's mirror as the select route reads it: BF16
/// `[pooled_capacity][index_dim]` plus its watermark.
#[derive(Clone, Copy)]
pub struct QsaPooledBf16MirrorRef<'a> {
    pub mirror: &'a GpuTensor,
    pub valid_blocks: &'a QsaPooledBf16ValidBlocks,
}

impl QsaPooledBf16MirrorRef<'_> {
    /// Leading blocks that are valid while the F32 arena holds
    /// `committed_blocks` complete blocks.
    pub fn valid(&self, committed_blocks: usize) -> usize {
        self.valid_blocks.valid(committed_blocks)
    }
}

fn shape_err(what: &str) -> HipError {
    HipError::new(0, &format!("QSA pooled BF16 mirror: {what}"))
}

/// Shared launch plan of the converter and the probe over blocks
/// `[first, end)`: `(i32 first, i32 end, i32 dim, grid_x)`.
fn plan(first: usize, end: usize, index_dim: usize) -> HipResult<(i32, i32, i32, u32)> {
    if index_dim == 0 || !index_dim.is_multiple_of(PER_THREAD) {
        return Err(shape_err("index_dim must be a nonzero multiple of 4"));
    }
    if first >= end {
        return Err(shape_err("empty block range"));
    }
    let as_i32 = |n: usize| i32::try_from(n).map_err(|_| shape_err("block count overflow"));
    let threads = (end - first)
        .checked_mul(index_dim / PER_THREAD)
        .ok_or_else(|| shape_err("grid overflow"))?;
    let grid = u32::try_from(threads.div_ceil(THREADS)).map_err(|_| shape_err("grid overflow"))?;
    Ok((as_i32(first)?, as_i32(end)?, as_i32(index_dim)?, grid))
}

/// The F32 pooled tensor and BF16 mirror hold at least `end` blocks.
fn check_tensors(
    pooled_f32: &GpuTensor,
    mirror: Option<&GpuTensor>,
    end: usize,
    index_dim: usize,
) -> HipResult<()> {
    let elems = end
        .checked_mul(index_dim)
        .ok_or_else(|| shape_err("extent overflow"))?;
    if pooled_f32.dtype != DType::F32
        || pooled_f32.numel() < elems
        || pooled_f32.buf.size() < elems * 4
    {
        return Err(shape_err(
            "pooled keys must be F32 and cover the block range",
        ));
    }
    if let Some(mirror) = mirror {
        if mirror.dtype != DType::BF16 || mirror.numel() < elems || mirror.buf.size() < elems * 2 {
            return Err(shape_err("mirror must be BF16 and cover the block range"));
        }
    }
    Ok(())
}

/// Convert pooled blocks `[mark.get(), complete_blocks)` into the mirror when
/// the mark is behind `complete_blocks`, then advance the mark to
/// `complete_blocks`. No launch (and no mark change) when the mirror already
/// covers `complete_blocks`.
///
/// Direct (unrecorded, uncaptured) launches only: the host block range is
/// baked into the launch. Under recording or capture the call neither
/// launches nor advances the mark: blocks pooled then lie at or past the mark
/// (pooling appends; owners clamp on every rewind), the next direct sync
/// converts them, and until then `mark.valid(complete) < complete` makes the
/// select batch use the canonical F32 keys. Exact gfx1151 only; any other arch
/// is an error.
pub fn qsa_pooled_bf16_sync(
    gpu: &mut Gpu,
    pooled_f32: &GpuTensor,
    mirror: &GpuTensor,
    mark: &QsaPooledBf16ValidBlocks,
    complete_blocks: usize,
    index_dim: usize,
) -> HipResult<()> {
    let valid = mark.get();
    if complete_blocks <= valid {
        return Ok(());
    }
    if !gpu.arch_caps.is_gfx1151() {
        return Err(shape_err("sync needs exact gfx1151"));
    }
    if gpu.replay.is_recording() || gpu.graphs.capture_mode {
        return Ok(());
    }
    check_tensors(pooled_f32, Some(mirror), complete_blocks, index_dim)?;
    let (first, end, dim, grid) = plan(valid, complete_blocks, index_dim)?;
    gpu.ensure_kernel_public(MODULE, crate::kernels::QSA_POOLED_BF16_MIRROR_SRC, CONVERT)?;
    let mut args = KernargBlob::new();
    args.push_ptr(pooled_f32.buf.as_ptr());
    args.push_ptr(mirror.buf.as_ptr());
    args.push_i32(first);
    args.push_i32(end);
    args.push_i32(dim);
    args.pad_to(16);
    // Extents (`qsa_pooled_f32_to_bf16_mirror`): thread `q < (end - first) *
    // index_dim / 4` converts one quad, reading 16 F32 bytes at `first * index_dim *
    // 4 + q * 16` and writing 8 BF16 bytes at `first * index_dim * 2 + q * 8`;
    // the `grid_x` round-up to 256 threads never passes that guard, and
    // `index_dim % 4 == 0` (`plan`) makes the byte lengths exact:
    // pooled read `(end - first) * index_dim * 4`, mirror write
    // `(end - first) * index_dim * 2`. Both pointers are whole-tensor bases, so
    // the tensor helper resolves the backing owner at call time.
    if gpu.flags.qwen4_qsa_extent_check {
        let depth = crate::dispatch::QsaExtentDepth {
            rows: complete_blocks - valid,
            end_position: 0,
            block_count: complete_blocks,
        };
        gpu.qsa_extent_check(
            "qsa_pooled_f32",
            pooled_f32,
            valid * index_dim * 4,
            (complete_blocks - valid) * index_dim * 4,
            depth,
            "qsa_pooled_bf16_sync",
        );
        gpu.qsa_extent_check(
            "qsa_pooled_bf16_mirror",
            mirror,
            valid * index_dim * 2,
            (complete_blocks - valid) * index_dim * 2,
            depth,
            "qsa_pooled_bf16_sync",
        );
    }
    gpu.launch_blob_recorded(
        CONVERT,
        [grid, 1, 1],
        [THREADS as u32, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )?;
    if gpu.flags.qwen4_qsa_extent_check {
        let depth = crate::dispatch::QsaExtentDepth {
            rows: complete_blocks - valid,
            end_position: 0,
            block_count: complete_blocks,
        };
        gpu.qsa_extent_post_launch(
            "qsa_pooled_f32",
            pooled_f32,
            valid * index_dim * 4,
            (complete_blocks - valid) * index_dim * 4,
            depth,
            "qsa_pooled_bf16_sync",
        );
        gpu.qsa_extent_post_launch(
            "qsa_pooled_bf16_mirror",
            mirror,
            valid * index_dim * 2,
            (complete_blocks - valid) * index_dim * 2,
            depth,
            "qsa_pooled_bf16_sync",
        );
    }
    mark.advance(complete_blocks);
    Ok(())
}

/// Lab exactness probe: the number of F32 elements of pooled blocks
/// `[first_block, end_block)` whose low 16 bits are nonzero, i.e. that the
/// mirror's high-half copy would not reproduce. `0` for every BF16-rounded
/// pooled arena. Direct launch on exact gfx1151; synchronizes on the readback.
pub fn qsa_pooled_bf16_low_nonzero_count(
    gpu: &mut Gpu,
    pooled_f32: &GpuTensor,
    first_block: usize,
    end_block: usize,
    index_dim: usize,
) -> HipResult<u32> {
    if first_block >= end_block {
        return Ok(0);
    }
    if !gpu.arch_caps.is_gfx1151() || gpu.replay.is_recording() || gpu.graphs.capture_mode {
        return Err(shape_err("probe needs a direct launch on gfx1151"));
    }
    check_tensors(pooled_f32, None, end_block, index_dim)?;
    let (first, end, dim, grid) = plan(first_block, end_block, index_dim)?;
    gpu.ensure_kernel_public(MODULE, crate::kernels::QSA_POOLED_BF16_MIRROR_SRC, PROBE)?;
    // Zeroed F32 words are zero u32 counters.
    let count = gpu.zeros(&[1], DType::F32)?;
    let mut args = KernargBlob::new();
    args.push_ptr(pooled_f32.buf.as_ptr());
    args.push_ptr(count.buf.as_ptr());
    args.push_i32(first);
    args.push_i32(end);
    args.push_i32(dim);
    args.pad_to(16);
    // Extents (`qsa_pooled_f32_low_nonzero_count`): thread `q < (end - first) *
    // index_dim / 4` reads one 16-byte F32 quad at `first * index_dim + q * 4`;
    // the grid round-up to 256 threads never passes that guard. `count` takes
    // one atomic u32 word.
    if gpu.flags.qwen4_qsa_extent_check {
        let depth = crate::dispatch::QsaExtentDepth {
            rows: end_block - first_block,
            end_position: 0,
            block_count: end_block,
        };
        gpu.qsa_extent_check(
            "qsa_pooled_f32",
            pooled_f32,
            first_block * index_dim * 4,
            (end_block - first_block) * index_dim * 4,
            depth,
            "qsa_pooled_bf16_low_nonzero_count",
        );
        gpu.qsa_extent_check(
            "qsa_pooled_probe_count",
            &count,
            0,
            4,
            depth,
            "qsa_pooled_bf16_low_nonzero_count",
        );
    }
    let launched = gpu.launch_blob_recorded(
        PROBE,
        [grid, 1, 1],
        [THREADS as u32, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    );
    if launched.is_ok() && gpu.flags.qwen4_qsa_extent_check {
        let depth = crate::dispatch::QsaExtentDepth {
            rows: end_block - first_block,
            end_position: 0,
            block_count: end_block,
        };
        gpu.qsa_extent_post_launch(
            "qsa_pooled_f32",
            pooled_f32,
            first_block * index_dim * 4,
            (end_block - first_block) * index_dim * 4,
            depth,
            "qsa_pooled_bf16_low_nonzero_count",
        );
        gpu.qsa_extent_post_launch(
            "qsa_pooled_probe_count",
            &count,
            0,
            4,
            depth,
            "qsa_pooled_bf16_low_nonzero_count",
        );
    }
    let bits = launched.and_then(|()| gpu.download_f32(&count));
    let freed = gpu.free_tensor(count);
    let bits = bits?;
    freed?;
    Ok(bits[0].to_bits())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn watermark_matches_f16_semantics_in_blocks() {
        let mark = QsaPooledBf16ValidBlocks::default();
        assert_eq!(mark.get(), 0);
        mark.advance(2049);
        assert_eq!(mark.valid(2048), 2048);
        assert_eq!(mark.get(), 2049);
        mark.clamp(2047); // Rollback keeps only the retained complete blocks.
        assert_eq!(mark.get(), 2047);
        mark.clamp(4000); // An append does not claim converted blocks.
        assert_eq!(mark.get(), 2047);
        mark.advance(4000);
        let moved = mark; // The watermark moves with its storage owner.
        assert_eq!(moved.get(), 4000);
        moved.invalidate(); // Replaced prefix, reset, restore or failed append.
        assert_eq!(moved.valid(4000), 0);
        moved.advance(3);
        moved.clamp(0);
        assert_eq!(moved.get(), 0);
    }

    #[test]
    fn mirror_ref_valid_is_clamped_to_committed() {
        // `QsaPooledBf16MirrorRef::valid` is the mark's `valid`; no GPU needed
        // for the mark half of the reference.
        let mark = QsaPooledBf16ValidBlocks::default();
        mark.advance(10);
        assert_eq!(mark.valid(7), 7);
        assert_eq!(mark.valid(10), 10);
        assert_eq!(mark.valid(11), 10);
    }

    #[test]
    fn admission_requires_every_condition() {
        let f32f = QsaKvFormat::F32;
        assert!(qsa_pooled_bf16_mirror_admits(
            true, true, true, true, f32f, 128
        ));
        assert!(!qsa_pooled_bf16_mirror_admits(
            false, true, true, true, f32f, 128
        ));
        assert!(!qsa_pooled_bf16_mirror_admits(
            true, false, true, true, f32f, 128
        ));
        assert!(!qsa_pooled_bf16_mirror_admits(
            true, true, false, true, f32f, 128
        ));
        assert!(!qsa_pooled_bf16_mirror_admits(
            true, true, true, false, f32f, 128
        ));
        assert!(!qsa_pooled_bf16_mirror_admits(
            true, true, true, true, f32f, 64
        ));
        assert!(!qsa_pooled_bf16_mirror_admits(
            true, true, true, true, f32f, 256
        ));
        assert!(!qsa_pooled_bf16_mirror_admits(
            true,
            true,
            true,
            true,
            QsaKvFormat::Q8,
            128
        ));
        assert!(!qsa_pooled_bf16_mirror_admits(
            true,
            true,
            true,
            true,
            QsaKvFormat::Fp8,
            128
        ));
    }

    #[test]
    fn layout_bytes_per_block_default_off() {
        use crate::tensor_ops::QsaStateLayout;
        assert!(!QsaStateLayout::F32.pooled_bf16);
        assert_eq!(QsaStateLayout::F32.pooled_bf16_bytes_per_block(128), 0);
        let on = QsaStateLayout {
            pooled_bf16: true,
            ..QsaStateLayout::F32
        };
        assert_eq!(on.pooled_bf16_bytes_per_block(128), 256);
        // 12 layers at 32K / 256K context, compress 4: bytes of the mirrors.
        for (tokens, per_layer) in [(32 * 1024usize, 2 << 20), (256 * 1024, 16 << 20)] {
            assert_eq!(
                tokens.div_ceil(4) * on.pooled_bf16_bytes_per_block(128),
                per_layer
            );
        }
        // Independent of the F16 arena.
        let both = QsaStateLayout {
            pooled_bf16: true,
            f16_arena: true,
            ..QsaStateLayout::F32
        };
        assert_eq!(both.pooled_bf16_bytes_per_block(128), 256);
        assert_eq!(both.f16_arena_bytes_per_token(2, 256), 2048);
    }

    #[test]
    fn launch_plan_covers_exactly_the_range() {
        assert!(plan(0, 0, 128).is_err());
        assert!(plan(5, 5, 128).is_err());
        assert!(plan(0, 4, 0).is_err());
        assert!(plan(0, 4, 6).is_err());
        // One block of 128 = 32 threads = one workgroup.
        assert_eq!(plan(7, 8, 128).unwrap(), (7, 8, 128, 1));
        // 8 blocks = 256 threads: exactly one group; 9 blocks needs two.
        assert_eq!(plan(0, 8, 128).unwrap().3, 1);
        assert_eq!(plan(0, 9, 128).unwrap().3, 2);
        // 256K context at compress 4: 65536 blocks.
        assert_eq!(plan(0, 65536, 128).unwrap().3, 8192);
        assert!(plan(0, usize::MAX, 128).is_err());
    }

    fn try_gpu() -> Option<crate::gpu_test::TestGpu> {
        let gpu = crate::gpu_test::init().ok()?;
        if gpu.arch_caps.is_gfx1151() {
            Some(gpu)
        } else {
            eprintln!("skip: needs exact gfx1151, found {}", gpu.arch);
            None
        }
    }

    /// BF16-rounded F32 values (low half zero), deterministic.
    fn pooled_values(blocks: usize, dim: usize, salt: u64) -> Vec<f32> {
        let mut s = 0x9e37_79b9_7f4a_7c15u64 ^ salt;
        (0..blocks * dim)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let v = f32::from_bits((s >> 32) as u32);
                // Finite, BF16-rounded: clear the low half, force a normal exponent.
                f32::from_bits((v.to_bits() & 0xFFFF_0000 & !0x4000_0000) | 0x3000_0000)
            })
            .collect()
    }

    /// Mirror bytes as u16 words.
    fn words(gpu: &Gpu, t: &GpuTensor) -> Vec<u16> {
        gpu.download_raw_bytes(t)
            .expect("download mirror")
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    }

    /// The mirror's converted prefix equals the canonical high halves, the
    /// range past it keeps its sentinel, and the sync advances / skips like
    /// the watermark says.
    #[test]
    #[ignore = "needs an exact-gfx1151 GPU"]
    fn sync_converts_only_the_gap_and_advances_the_mark() {
        let Some(mut gpu) = try_gpu() else { return };
        const DIM: usize = 128;
        const CAP: usize = 64;
        let pooled = pooled_values(CAP, DIM, 1);
        let pooled_t = gpu.upload_f32(&pooled, &[CAP * DIM]).unwrap();
        let sentinel = vec![0xA5A5u16; CAP * DIM];
        let bytes: Vec<u8> = sentinel.iter().flat_map(|w| w.to_le_bytes()).collect();
        let mut mirror = gpu.upload_raw(&bytes, &[CAP * DIM]).unwrap();
        mirror.dtype = DType::BF16;
        let mark = QsaPooledBf16ValidBlocks::default();

        let expect = |upto: usize| -> Vec<u16> {
            (0..CAP * DIM)
                .map(|i| {
                    if i < upto * DIM {
                        (pooled[i].to_bits() >> 16) as u16
                    } else {
                        0xA5A5
                    }
                })
                .collect()
        };

        qsa_pooled_bf16_sync(&mut gpu, &pooled_t, &mirror, &mark, 0, DIM).unwrap();
        assert_eq!(mark.get(), 0);
        assert_eq!(words(&gpu, &mirror), expect(0));

        qsa_pooled_bf16_sync(&mut gpu, &pooled_t, &mirror, &mark, 5, DIM).unwrap();
        assert_eq!(mark.get(), 5);
        assert_eq!(words(&gpu, &mirror), expect(5));

        // Already covered: no launch, so a corrupted tail would stay as is.
        qsa_pooled_bf16_sync(&mut gpu, &pooled_t, &mirror, &mark, 3, DIM).unwrap();
        assert_eq!(mark.get(), 5);

        // Incremental: only [5, 41) converts.
        qsa_pooled_bf16_sync(&mut gpu, &pooled_t, &mirror, &mark, 41, DIM).unwrap();
        assert_eq!(mark.get(), 41);
        assert_eq!(words(&gpu, &mirror), expect(41));

        // Rewind plus rewritten canonical blocks: clamp, then re-convert.
        let rewritten = pooled_values(CAP, DIM, 2);
        let rewritten_t = gpu.upload_f32(&rewritten, &[CAP * DIM]).unwrap();
        mark.clamp(10);
        qsa_pooled_bf16_sync(&mut gpu, &rewritten_t, &mirror, &mark, 20, DIM).unwrap();
        assert_eq!(mark.get(), 20);
        let got = words(&gpu, &mirror);
        for (i, w) in got.iter().enumerate().take(20 * DIM).skip(10 * DIM) {
            assert_eq!(*w, (rewritten[i].to_bits() >> 16) as u16, "element {i}");
        }
        for (i, w) in got.iter().enumerate().take(10 * DIM) {
            assert_eq!(*w, (pooled[i].to_bits() >> 16) as u16, "kept element {i}");
        }
    }

    /// The probe reports zero on BF16-rounded keys and counts exactly the
    /// elements that carry low bits.
    #[test]
    #[ignore = "needs an exact-gfx1151 GPU"]
    fn probe_counts_nonzero_low_halves() {
        let Some(mut gpu) = try_gpu() else { return };
        const DIM: usize = 128;
        const CAP: usize = 40;
        let mut pooled = pooled_values(CAP, DIM, 3);
        let t = gpu.upload_f32(&pooled, &[CAP * DIM]).unwrap();
        assert_eq!(
            qsa_pooled_bf16_low_nonzero_count(&mut gpu, &t, 0, CAP, DIM).unwrap(),
            0
        );
        for i in [3usize, 4, 127, 128, 129, 5000] {
            pooled[i] = f32::from_bits(pooled[i].to_bits() | 1);
        }
        let t = gpu.upload_f32(&pooled, &[CAP * DIM]).unwrap();
        assert_eq!(
            qsa_pooled_bf16_low_nonzero_count(&mut gpu, &t, 0, CAP, DIM).unwrap(),
            6
        );
        // Element 5000 is in block 39; ranges exclude other blocks' elements.
        assert_eq!(
            qsa_pooled_bf16_low_nonzero_count(&mut gpu, &t, 1, 2, DIM).unwrap(),
            2
        );
        assert_eq!(
            qsa_pooled_bf16_low_nonzero_count(&mut gpu, &t, 39, 40, DIM).unwrap(),
            1
        );
        assert_eq!(
            qsa_pooled_bf16_low_nonzero_count(&mut gpu, &t, 10, 39, DIM).unwrap(),
            0
        );
    }
}
