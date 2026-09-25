//! Fold wiring for the K-loop: which physical ranges hold the scale rows and
//! products at fold time, the phase-B placement of the two fold stages, and
//! the (closed, never emitted into a product module) `K256Pow2` re-seed.
use super::{Gen, op, v};
use crate::{Builder, V, kernels::iu4_k1::{self, FoldRegisters}};

/// Accumulator order of a WMMA bundle: row group 0 over the four column
/// blocks, then row group 1 (`k = nb * 2 + rg`).
pub(crate) const BUNDLE_ORDER: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];
/// A stage-1 (`subrev`) group trails the WMMA that completed its accumulator
/// by this many WMMAs, so the dependent VALU does not wait on WMMA latency.
pub(crate) const UNBIAS_LAG: usize = 2;

/// Registers of one block's fold. Scale rows of row group 0 alias F0's A
/// fragments, row group 1 aliases F0's and F1's W fragments, products alias
/// F1's A fragments: every alias is written only after the WMMAs that read
/// the fragment have issued (proved by per-block lifetimes in `kloop`).
pub(crate) fn registers(g: &Gen) -> FoldRegisters {
    let [f0, f1] = g.f;
    FoldRegisters {
        cacc: std::array::from_fn(|k| V(g.cacc + 8 * k as u8)),
        acc: std::array::from_fn(|k| V(g.acc + 8 * k as u8)),
        magic: V(g.magic),
        sc: [[V(f0 + 4), V(f0 + 8)], [V(f0), V(f1)]],
        t: V(f1 + 4),
        d: std::array::from_fn(|nb| V(g.d + nb as u8)),
    }
}

/// Stage 1 for the accumulator completed `UNBIAS_LAG` WMMAs ago, if any;
/// called after each phase-B slab-1 WMMA `i` of `BUNDLE_ORDER`.
pub(crate) fn unbias_after_wmma(b: &mut Builder, g: &Gen, i: usize) -> Result<(), String> {
    if i >= UNBIAS_LAG { iu4_k1::emit_fold_unbias(b, registers(g), BUNDLE_ORDER[i - UNBIAS_LAG])?; }
    Ok(())
}
/// Stage 1 for the accumulators the last WMMAs completed.
pub(crate) fn unbias_tail(b: &mut Builder, g: &Gen) -> Result<(), String> {
    for &k in &BUNDLE_ORDER[BUNDLE_ORDER.len() - UNBIAS_LAG..] { iu4_k1::emit_fold_unbias(b, registers(g), k)?; }
    Ok(())
}
/// Stage 2 (64 packets): row group 0 first, whose scale rows landed during
/// the slab-1 WMMAs.
pub(crate) fn scale(b: &mut Builder, g: &Gen) -> Result<(), String> {
    for &k in &BUNDLE_ORDER { iu4_k1::emit_fold_scale(b, registers(g), k)?; }
    Ok(())
}

/// `K256Pow2` half-B re-seed: `cacc = (cacc << s) + magic` for every element
/// after half A accumulated from zero (plan §3.3). The fold-256 track is
/// closed at Q0; this body exists for the record and `Spec::validate` keeps
/// it out of every product module.
pub fn emit_pow2_reseed(b: &mut Builder, cacc: [V<8>; 8], shift: [V<1>; 8], magic: V<8>) -> Result<(), String> {
    for c in cacc {
        for j in 0..8u8 {
            let (r, sh, m) = (c.base() + j, shift[usize::from(j)], magic.base() + j);
            op(b, format!("v_lshl_add_u32 v{r}, v{r}, v{}, v{m}", sh.base()), &[v(r)], &[v(r), sh.reg(), v(m)])?;
        }
    }
    Ok(())
}
