// Both FA2 bodies with the mailbox retired before the first V fill:
// - non-fill: the extra barrier `f284a3b8a` added (hip:613-620);
// - fill: the retire rides the barrier that already precedes the first V
//   store, so the safe body costs no extra barrier.
use peacemaker_author::{join, ready, retire, trace::Trace, Backend, Gfx1100, LdsRegion, Published, Workgroup};

enum KTile {}
enum VBody {}
enum Mailbox {}
enum VPlane {}

fn publish_mailbox<B: Backend<Insn = String>>(wg: &mut Workgroup<Gfx1100, B>) -> Result<LdsRegion<Mailbox, Published>, String> {
    let mbox = wg.lds::<Mailbox>("mailbox", 32760, 8)?;
    let (mbox, st) = wg.ds_store(mbox, "ds_store_2addr_b32 v0, v1, v2 offset1:1".into())?;
    let drained = wg.wait(st)?;
    let (mbox,) = wg.barrier((ready(mbox, drained),))?;
    wg.ds_load(&mbox, "ds_load_2addr_b32 v[3:4], v0 offset1:1".into())?;
    Ok(mbox)
}

fn non_fill<B: Backend<Insn = String>>(wg: &mut Workgroup<Gfx1100, B>) -> Result<(), String> {
    let v_body = wg.lds::<VBody>("V", 16384, 16376)?;
    let mbox = publish_mailbox(wg)?;
    let (mbox,) = wg.barrier((retire(mbox),))?;
    let v_plane = join::<VPlane, _, _>(v_body, mbox)?;
    let (_v, st) = wg.ds_store(v_plane, "ds_store_b128 v5, v[8:11]".into())?;
    let _ = wg.wait(st)?;
    Ok(())
}

fn fill<B: Backend<Insn = String>>(wg: &mut Workgroup<Gfx1100, B>) -> Result<(), String> {
    let k = wg.lds::<KTile>("K", 0, 16384)?;
    let v_body = wg.lds::<VBody>("V", 16384, 16376)?;
    let mbox = publish_mailbox(wg)?;
    // Fill waves stage the first K tile; its existing barrier also retires the mailbox.
    let (k, st) = wg.ds_store(k, "ds_store_b128 v7, v[12:15]".into())?;
    let drained = wg.wait(st)?;
    let (_k, mbox) = wg.barrier((ready(k, drained), retire(mbox)))?;
    let v_plane = join::<VPlane, _, _>(v_body, mbox)?;
    let (_v, st) = wg.ds_store(v_plane, "ds_store_b128 v5, v[8:11]".into())?;
    let _ = wg.wait(st)?;
    Ok(())
}

fn main() {
    let mut b = Trace::new("gfx1100");
    non_fill(&mut Workgroup::<Gfx1100, Trace>::new(&mut b).unwrap()).unwrap();
    assert_eq!(b.count("s_barrier"), 2);
    let mut b = Trace::new("gfx1100");
    fill(&mut Workgroup::<Gfx1100, Trace>::new(&mut b).unwrap()).unwrap();
    assert_eq!(b.count("s_barrier"), 2, "the mailbox retire rides the K-tile barrier");
}
