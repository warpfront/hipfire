// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// A wave-private region cannot be published to the other waves: `retire` (and
// `ready`) take shared `LdsRegion`s, so no barrier transition can name it.
use peacemaker_author::trace::Trace;
use peacemaker_author::{retire, Gfx1151, Workgroup};

enum Scratch {}

fn main() {
    let _ = || -> Result<(), String> {
        let mut b = Trace::new("gfx1151");
        let mut wg = Workgroup::<Gfx1151, Trace>::new(&mut b)?;
        let p = wg.lds_private::<Scratch>("scratch", 0, 512, 8)?;
        let w = wg.private_store(p, "ds_store_b32 v1, v2".into())?;
        let o = wg.private_publish(w)?;
        // Another wave reading it would need a barrier transition.
        let _ = wg.barrier((retire(o),))?;
        Ok(())
    };
}
