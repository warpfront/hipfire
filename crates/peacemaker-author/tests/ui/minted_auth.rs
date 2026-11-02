// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Nor minted: only `Workgroup::new` starts a session.
use peacemaker_author::{Auth, Backend, Gfx1100, Workgroup};

fn divergent<B: Backend<Insn = String>>(wg: &mut Workgroup<Gfx1100, B>) -> Result<(), String> {
    let first_wave = wg.scmp("s_cmp_eq_u32 s18, 0".into())?;
    wg.skip_if(first_wave, ".Lskip", (), |w, ()| {
        let minted = Auth::mint();
        w.isa().barrier(&minted, &[])
    })
}

fn main() {
    let _ = divergent::<peacemaker_author::trace::Trace>;
}
