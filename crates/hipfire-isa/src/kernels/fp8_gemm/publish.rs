use super::{ActScale, Builder, Spec, ds_store, op, s, so, v, vo, vload};

/// Address and prefetch every fragment byte of one A slot. All four b64
/// transactions are retained until the prior slot's WMMA readers are done.
pub(super) fn fetch_a(b:&mut Builder,kb_offset:u32,slab:u32)->Result<(),String>{
    so(b,format!("s_lshl_b32 s91, s85, 7"),&[91],&[85])?;
    if kb_offset!=0 {so(b,format!("s_add_co_i32 s91, s91, {:#x}",kb_offset*128),&[91],&[91])?;}
    if slab!=0 {so(b,format!("s_add_co_i32 s91, s91, {}",slab*64),&[91],&[91])?;}
    b.clause(|b| {
        for j in 0..4 {vload(b,168+2*j,2,177,36,Some(91),j as u32*8)?;}
        Ok(())
    })
}
pub(super) fn publish_a(b:&mut Builder,slot:usize)->Result<(),String>{
    for j in 0..4 {
        ds_store(b,slot,168+2*j,2,178,(slot as u32)*8192+u32::from(j/2)*2048+u32::from(j%2)*128)?;
    }
    Ok(())
}
pub(super) fn stage_a(b:&mut Builder,slot:usize,slab:u32)->Result<(),String>{
    fetch_a(b,0,slab)?;
    publish_a(b,slot)
}

/// Repack Rw planes are indexed by (row_tile, half, virtual_row). Each lane
/// writes exactly its own virtual row. For block scaling only tid<128 writes
/// the per-token D ratio, so its slot has a unique producer for every word.
pub(super) fn publish_next_ratios(b:&mut Builder,spec:Spec,next_kb:u32,plane:usize)->Result<(),String>{
    so(b,"s_add_co_i32 s92, s94, s85",&[92],&[94,85])?;
    so(b,format!("s_add_co_i32 s92, s92, {next_kb}"),&[92],&[92])?;
    so(b,"s_lshl_b32 s92, s92, 10",&[92],&[92])?;
    vload(b,168,1,182,40,Some(92),0)?;
    ds_store(b,2+plane,168,1,182,16384+plane as u32*1024)?;
    if spec.act_scale==ActScale::K128 {
        so(b,"s_add_co_i32 s93, s85, 0",&[93],&[85])?;
        so(b,"s_lshl_b32 s93, s93, 2",&[93],&[93])?;
        vload(b,168,1,183,48,Some(93),0)?;
        vload(b,169,1,183,48,Some(93),4*next_kb)?;
        // Positive powers of two: floating exponents are affine. The bit
        // subtraction produces the exactly represented d[b]/d[b+1].
        vo(b,"v_sub_nc_u32_e32 v170, v168, v169",&[170],&[168,169],&[])?;
        vo(b,"v_add_nc_u32_e32 v170, 1.0, v170",&[170],&[170],&[])?;
        op(b,"v_cmp_gt_u32_e64 s72, 0x80, v187",&[s(72)],&[v(187)])?;
        op(b,"s_mov_b32 s74, exec_lo",&[s(74)],&[])?;
        op(b,"s_mov_b32 exec_lo, s72",&[],&[s(72)])?;
        ds_store(b,4+plane,170,1,182,18432+plane as u32*512)?;
        op(b,"s_mov_b32 exec_lo, s74",&[],&[s(74)])?;
    }
    Ok(())
}

/// Weight ring unit u: four 128-bit transactions, one for each row16 tile,
/// with the probe's `lane*16` per-lane address. The two K16 fragments within
/// each b128 are selected by st*8. No W is ever staged into LDS or decoded.
pub(super) fn weight_base(b:&mut Builder)->Result<(),String>{
    so(b,"s_add_co_i32 s90, s94, s85",&[90],&[94,85])?;
    so(b,"s_lshl_b32 s90, s90, 15",&[90],&[90])?;
    so(b,"s_add_co_i32 s90, s90, s95",&[90],&[90,95])
}
pub(super) fn fetch_w(b:&mut Builder,unit:usize)->Result<(),String>{
    b.clause(|b|{
        for i in 0..4 {vload(b,128+((unit&1)*16+i*4) as u8,4,176,32,Some(90),(unit*2048+i*512) as u32)?;}
        Ok(())
    })
}

/// Four conflict-free A fragments from the current slot, one per 16-column
/// tile. No A fragment destination aliases the next slab's prefetch bank.
pub(super) fn fragments(b:&mut Builder,slot:usize,step:usize)->Result<(),String>{
    for tt in 0..4 {
        ds_load(b,slot,160+2*tt,2,179,(slot*8192+step*2048+usize::from(tt)*256) as u32)?;
    }
    Ok(())
}
use super::ds_load;
