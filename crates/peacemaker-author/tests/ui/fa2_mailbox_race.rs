// gfx11 FA2 mailbox (non-fill body) `f284a3b8a`: lane 0 publishes gmax/gmin
// in the V plane's last two dwords (attention_q8_0_fa2_gqa.gfx11.hip:603-611);
// the tile loop's first V fill (:951) rewrites them while a slow wave may
// still be loading them.
use peacemaker_author::{join, ready, Backend, Gfx1100, Workgroup};

enum VBody {}
enum Mailbox {}
enum VPlane {}

fn mailbox<B: Backend<Insn = String>>(wg: &mut Workgroup<Gfx1100, B>) -> Result<(), String> {
    let v_body = wg.lds::<VBody>("V", 16384, 16376)?;
    let mbox = wg.lds::<Mailbox>("mailbox", 32760, 8)?;
    let (mbox, st) = wg.ds_store(mbox, "ds_store_2addr_b32 v0, v1, v2 offset1:1".into())?;
    let drained = wg.wait(st)?;
    let (mbox,) = wg.barrier((ready(mbox, drained),))?;
    wg.ds_load(&mbox, "ds_load_2addr_b32 v[3:4], v0 offset1:1".into())?;
    // First V fill of the tile loop: the whole plane, mailbox included.
    let v_plane = join::<VPlane, _, _>(v_body, mbox)?;
    let (_v, st) = wg.ds_store(v_plane, "ds_store_b128 v5, v[8:11]".into())?;
    let _ = wg.wait(st)?;
    Ok(())
}

fn main() {
    let _ = mailbox::<peacemaker_author::trace::Trace>;
}
