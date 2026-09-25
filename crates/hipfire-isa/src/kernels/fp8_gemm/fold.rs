use super::{ActScale, Builder, Spec, ds_load, op, v};

/// A half's C is already complete when this runs. Published ratios for the
/// *next* half are read in 16 row pairs; each weight ratio scales four token
/// chains. The first half omits the fold because C starts at literal zero.
pub(super) fn emit(b:&mut Builder,spec:Spec,plane:usize)->Result<(),String>{
    if spec.act_scale==ActScale::K128 {
        for tt in 0..4u8 {ds_load(b,4+plane,172+tt,1,181,(18432+plane*512+usize::from(tt)*64) as u32)?;}
    }
    for i in 0..4 {
        for e in (0..8).step_by(2) {
            let offset=16384+plane*1024+i*64+e*4;
            ds_load(b,2+plane,168,1,180,offset as u32)?;
            ds_load(b,2+plane,169,1,180,(offset+4) as u32)?;
            for tt in 0..4 {
                let (a,z)=(8*(i*4+tt)+e,8*(i*4+tt)+e+1);
                let (r0,r1)=if spec.act_scale==ActScale::K128 {
                    op(b,format!("v_dual_mul_f32 v170, v168, v{} :: v_dual_mul_f32 v171, v169, v{}",172+tt,172+tt),&[v(170),v(171)],&[v(168),v(169),v((172+tt) as u8)])?;
                    (170,171)
                } else {(168,169)};
                op(b,format!("v_dual_mul_f32 v{a}, v{a}, v{r0} :: v_dual_mul_f32 v{z}, v{z}, v{r1}"),
                    &[v(a as u8),v(z as u8)],&[v(a as u8),v(z as u8),v(r0),v(r1)])?;
            }
        }
    }
    Ok(())
}
