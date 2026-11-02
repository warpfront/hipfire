// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// The reviewer's handoff counterexample: after the handoff only the reading
// waves run, so a barrier there would wait on writers that have left. The
// reader continuation is wave scope and has no barrier.
use peacemaker_author::{retire, Backend, Gfx1151, Workgroup};

enum Gate {}

fn kernel<B: Backend<Insn = String>>(wg: &mut Workgroup<Gfx1151, B>) -> Result<(), String> {
    let gate = wg.lds::<Gate>("gate", 0, 2048)?;
    let end = wg.exit(".Lend")?;
    let readers = wg.scmp("s_cmp_ge_u32 s4, 2".into())?;
    wg.handoff(readers, ".Lup", end, gate, |w, gate| w.ds_store(gate, "ds_store_b128 v1, v[2:5]".into()), |w, gate, _| {
        w.ds_load(&gate, "ds_load_b128 v[8:11], v1".into())?;
        w.barrier((retire(gate),))?;
        Ok(())
    })
}

fn main() {
    let _ = kernel::<peacemaker_author::trace::Trace>;
}
