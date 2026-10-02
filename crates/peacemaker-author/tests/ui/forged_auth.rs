// An `Auth` cannot be built outside the typed core.
use peacemaker_author::{Auth, Backend, Gfx1100, Workgroup};

fn divergent<B: Backend<Insn = String>>(wg: &mut Workgroup<Gfx1100, B>) -> Result<(), String> {
    let first_wave = wg.scmp("s_cmp_eq_u32 s18, 0".into())?;
    wg.skip_if(first_wave, ".Lskip", (), |w, ()| {
        let forged = Auth { session: 1 };
        w.isa().barrier(&forged, &[])
    })
}

fn main() {
    let _ = divergent::<peacemaker_author::trace::Trace>;
}
