// Halo VerifyAttn (gfx1151) `6f708aef3`: the K tile's `ds_store_b128` at the
// loop tail (`k_stage`, attention_verify_wmma.gfx1151.hip:270) reaches the
// loop-head `s_barrier` (:248) on the back edge with no `s_waitcnt lgkmcnt`.
use peacemaker_author::{rotate, Backend, Gfx1151, Ring, Workgroup};

enum KTile {}

fn k_tile_loop<B: Backend<Insn = String>>(wg: &mut Workgroup<Gfx1151, B>) -> Result<(), String> {
    let k0 = wg.lds::<KTile>("K0", 0, 8704)?;
    let k1 = wg.lds::<KTile>("K1", 8704, 8704)?;
    // Pre-loop k_stage(0).
    let (ring, st) = wg.ds_store(Ring::new(k0, k1), "ds_store_b128 v1, v[2:5]".into())?;
    let _ = wg.loop_carried(".Lk_tile", (ring.into_steady(), st), |wg, (ring, st)| {
        // Loop head: publish the tile staged on the previous trip.
        let (ring,) = wg.barrier((rotate(ring, st),))?;
        wg.ds_load_cur(&ring, "ds_load_b128 v[8:11], v6".into())?;
        // k_stage(t + 1) at the loop tail.
        let next = wg.ds_store(ring, "ds_store_b128 v1, v[2:5]".into())?;
        Ok((next, wg.scmp_wg_uniform("s_cmp_lg_u32 s4, 0".into())?))
    })?;
    Ok(())
}

fn main() {
    let _ = k_tile_loop::<peacemaker_author::trace::Trace>;
}
