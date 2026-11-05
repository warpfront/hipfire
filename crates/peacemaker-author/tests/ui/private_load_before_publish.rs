// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// A wave reads its slice only once it is `Owned` (published to that wave).
use peacemaker_author::trace::Trace;
use peacemaker_author::{Gfx1151, Workgroup};

enum Scratch {}

fn main() {
    let _ = || -> Result<(), String> {
        let mut b = Trace::new("gfx1151");
        let mut wg = Workgroup::<Gfx1151, Trace>::new(&mut b)?;
        let p = wg.lds_private::<Scratch>("scratch", 0, 512, 8)?;
        let w = wg.private_store(p, "ds_store_b32 v1, v2".into())?;
        wg.private_load(&w, "ds_load_b32 v3, v1".into())?;
        Ok(())
    };
}
