use super::{ActScale, Builder, Spec, ds_load};
use crate::kernels::common::{op, v};
use crate::ledger::Counter;

/// Row-scale ratio staging for one row group (8 ratios, rows 8i..8i+7):
/// two b128 loads per group, alternating between the staging alias
/// v168..v175 and v[183:186] + v[188:191]. The second set holds nothing in a
/// Row K region: v184 carries the next block's ratio only from its fetch
/// before slab 0 to its publish before this fold, and the others are
/// epilogue temporaries.
const RATIO_SETS: [[u8; 8]; 2] = [[168, 169, 170, 171, 172, 173, 174, 175], [183, 184, 185, 186, 188, 189, 190, 191]];

/// A half's C is already complete when this runs. Published ratios for the
/// *next* half scale the accumulators; each weight ratio scales four token
/// chains. The first half omits the fold because C starts at literal zero.
pub(super) fn emit(b:&mut Builder,spec:Spec,plane:usize)->Result<(),String>{
    if spec.act_scale==ActScale::K128 {return emit_k128(b,plane)}
    // Row: the eight ratios of row group i arrive in two b128 loads, one
    // group ahead of their 16 packets. The A-publish stores still read
    // v168..v175, so every store has drained before the first load; each
    // group then waits only for its own two loads.
    let group=|b:&mut Builder,i:usize|->Result<(),String>{
        let regs=RATIO_SETS[i%2];
        let offset=(16384+plane*1024+i*64) as u32;
        ds_load(b,2+plane,regs[0],4,180,offset)?;
        ds_load(b,2+plane,regs[4],4,180,offset+16)
    };
    b.wait(Counter::Ds,0)?;
    group(b,0)?;
    for i in 0..4 {
        if i+1<4 {group(b,i+1)?;}
        b.wait(Counter::Ds,if i+1<4 {2} else {0})?;
        let regs=RATIO_SETS[i%2];
        for e in (0..8).step_by(2) {
            let (r0,r1)=(regs[e],regs[e+1]);
            for tt in 0..4 {
                let (a,z)=(8*(i*4+tt)+e,8*(i*4+tt)+e+1);
                op(b,format!("v_dual_mul_f32 v{a}, v{a}, v{r0} :: v_dual_mul_f32 v{z}, v{z}, v{r1}"),
                    &[v(a as u8),v(z as u8)],&[v(a as u8),v(z as u8),v(r0),v(r1)])?;
            }
        }
    }
    Ok(())
}

/// K128: published ratios are read in 16 row pairs and multiplied by the
/// per-token D ratio before they scale the accumulators.
fn emit_k128(b:&mut Builder,plane:usize)->Result<(),String>{
    for tt in 0..4u8 {ds_load(b,4+plane,172+tt,1,181,(18432+plane*512+usize::from(tt)*64) as u32)?;}
    for i in 0..4 {
        for e in (0..8).step_by(2) {
            let offset=16384+plane*1024+i*64+e*4;
            ds_load(b,2+plane,168,1,180,offset as u32)?;
            ds_load(b,2+plane,169,1,180,(offset+4) as u32)?;
            for tt in 0..4 {
                let (a,z)=(8*(i*4+tt)+e,8*(i*4+tt)+e+1);
                op(b,format!("v_dual_mul_f32 v170, v168, v{} :: v_dual_mul_f32 v171, v169, v{}",172+tt,172+tt),&[v(170),v(171)],&[v(168),v(169),v((172+tt) as u8)])?;
                op(b,format!("v_dual_mul_f32 v{a}, v{a}, v170 :: v_dual_mul_f32 v{z}, v{z}, v171"),
                    &[v(a as u8),v(z as u8)],&[v(a as u8),v(z as u8),v(170),v(171)])?;
            }
        }
    }
    Ok(())
}
