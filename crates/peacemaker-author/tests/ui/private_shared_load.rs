// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// The shared `ds_load` reads a workgroup-published `LdsRegion`; a wave-private
// region is not one, so no other wave can read it through it.
use peacemaker_author::trace::Trace;
use peacemaker_author::{Gfx1151, Workgroup};

enum Scratch {}

fn main() {
    let _ = || -> Result<(), String> {
        let mut b = Trace::new("gfx1151");
        let mut wg = Workgroup::<Gfx1151, Trace>::new(&mut b)?;
        let p = wg.lds_private::<Scratch>("scratch", 0, 512, 8)?;
        let w = wg.private_store(p, "ds_store_b32 v1, v2".into())?;
        let o = wg.private_publish(w)?;
        wg.ds_load(&o, "ds_load_b32 v3, v1".into())?;
        Ok(())
    };
}
