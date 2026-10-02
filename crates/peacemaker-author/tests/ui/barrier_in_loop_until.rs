// `loop_until` is wave scope: its break condition may differ between waves,
// so its body has no barrier.
use peacemaker_author::{Backend, Gfx1151, Workgroup};

fn walk<B: Backend<Insn = String>>(wg: &mut Workgroup<Gfx1151, B>) -> Result<(), String> {
    wg.loop_until(".Lwalk", ".Lfound", |w, exit| {
        let hit = w.scmp("s_cmp_lg_u32 s4, 0".into())?;
        w.break_if(hit, exit)?;
        w.barrier(())?;
        Ok(())
    })
}

fn main() {
    let _ = walk::<peacemaker_author::trace::Trace>;
}
