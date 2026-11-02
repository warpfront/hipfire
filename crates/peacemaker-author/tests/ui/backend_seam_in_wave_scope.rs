// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// The reviewer's seam counterexample: a wave-only body reaching the
// backend through `isa` and calling its barrier entry point directly. Every
// state-changing entry point takes an `Auth` the wave scope never sees.
use peacemaker_author::{Backend, Gfx1100, Workgroup};

fn divergent<B: Backend<Insn = String>>(wg: &mut Workgroup<Gfx1100, B>) -> Result<(), String> {
    let first_wave = wg.scmp("s_cmp_eq_u32 s18, 0".into())?;
    wg.skip_if(first_wave, ".Lskip", (), |w, ()| {
        w.isa().barrier(&[])?;
        Ok(())
    })
}

fn main() {
    let _ = divergent::<peacemaker_author::trace::Trace>;
}
