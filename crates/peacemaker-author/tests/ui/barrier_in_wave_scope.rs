// A barrier under wave-dependent control: the branch body sees only a
// `Wave`, which has no barrier.
use peacemaker_author::{Backend, Gfx1100, Workgroup};

fn divergent<B: Backend<Insn = String>>(wg: &mut Workgroup<Gfx1100, B>) -> Result<(), String> {
    let first_wave = wg.scmp("s_cmp_eq_u32 s18, 0".into())?;
    wg.skip_if(first_wave, ".Lskip", (), |w, ()| {
        w.barrier(())?;
        Ok(())
    })
}

fn main() {
    let _ = divergent::<peacemaker_author::trace::Trace>;
}
