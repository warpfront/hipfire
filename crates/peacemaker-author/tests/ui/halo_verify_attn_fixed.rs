// The Halo fix (`1ba84942a`): drain the tile's LDS store before the
// loop-head barrier, exactly one `s_waitcnt lgkmcnt(0)` per trip.
use peacemaker_author::{rotate, trace::Trace, Backend, Gfx1151, Ring, Workgroup};

enum KTile {}

fn k_tile_loop<B: Backend<Insn = String>>(wg: &mut Workgroup<Gfx1151, B>) -> Result<(), String> {
    let k0 = wg.lds::<KTile>("K0", 0, 8704)?;
    let k1 = wg.lds::<KTile>("K1", 8704, 8704)?;
    let (ring, st) = wg.ds_store(Ring::new(k0, k1), "ds_store_b128 v1, v[2:5]".into())?;
    let _ = wg.loop_carried(".Lk_tile", (ring.into_steady(), st), |wg, (ring, st)| {
        let drained = wg.wait(st)?;
        let (ring,) = wg.barrier((rotate(ring, drained),))?;
        wg.ds_load_cur(&ring, "ds_load_b128 v[8:11], v6".into())?;
        let next = wg.ds_store(ring, "ds_store_b128 v1, v[2:5]".into())?;
        Ok((next, wg.scmp_wg_uniform("s_cmp_lg_u32 s4, 0".into())?))
    })?;
    Ok(())
}

fn main() {
    let mut b = Trace::new("gfx1151");
    k_tile_loop(&mut Workgroup::<Gfx1151, Trace>::new(&mut b).unwrap()).unwrap();
    let barrier = b.text.iter().position(|t| t == "s_barrier").unwrap();
    assert_eq!(b.text[barrier - 1], "s_waitcnt lgkmcnt(0)");
    assert_eq!((b.count("s_waitcnt"), b.count("s_barrier")), (1, 1));
    assert_eq!(b.text.last().map(String::as_str), Some("s_cbranch_scc1 .Lk_tile"));
}
