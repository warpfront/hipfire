use super::{ActScale, Builder, EPILOGUE, Epi, Spec, mem, op, s, so, sr, v, vo, vload};
use crate::{insn::MemoryClass, kernels::iu4_gemm::region::{self, Binding, Region}};

/// A single virtual row is selected against segment boundaries independently
/// of the wave: M%16 tails may straddle a fragment, and SiLU's paired order
/// must not be treated as linear concatenation.
fn store(b:&mut Builder,seg:usize,row:u8,token:u8,val:u8)->Result<(),String>{
    let mi=26+seg as u8;
    let desc=52+4*seg as u8;
    // s72 = prefix, s73 = prefix+Mi, v184 = absolute virtual row.
    so(b,format!("s_add_co_i32 s73, s72, s{mi}"),&[73],&[72,mi])?;
    op(b,format!("v_cmp_le_u32_e64 s74, s72, v{row}"),&[s(74)],&[s(72),v(row)])?;
    op(b,format!("v_cmp_gt_u32_e64 s76, s73, v{row}"),&[s(76)],&[s(73),v(row)])?;
    op(b,format!("v_cmp_gt_u32_e64 s78, s31, v{token}"),&[s(78)],&[s(31),v(token)])?;
    so(b,"s_and_b32 s74, s74, s76",&[74],&[74,76])?;
    so(b,"s_and_b32 s74, s74, s78",&[74],&[74,78])?;
    vo(b,format!("v_mul_lo_u32 v185, v{token}, s{mi}"),&[185],&[token],&[mi])?;
    vo(b,format!("v_subrev_nc_u32_e32 v186, s72, v{row}"),&[186],&[row],&[72])?;
    vo(b,"v_add_nc_u32_e32 v185, v185, v186",&[185],&[185,186],&[])?;
    vo(b,"v_lshlrev_b32_e32 v185, 2, v185",&[185],&[185],&[])?;
    op(b,"s_mov_b32 s79, exec_lo",&[s(79)],&[])?;
    op(b,"s_mov_b32 exec_lo, s74",&[],&[s(74)])?;
    let dest=format!("s[{desc}:{}]",desc+3);
    if seg==0 && b.spec.symbol.contains("_add_") {
        mem(b,format!("buffer_load_b32 v167, v185, {dest}, null offen"),&[v(167)],&[v(185),sr(desc,4)],MemoryClass::VmemLoad)?;
        vo(b,format!("v_add_f32_e32 v167, v167, v{val}"),&[167],&[167,val],&[])?;
        mem(b,format!("buffer_store_b32 v167, v185, {dest}, null offen"),&[],&[v(185),v(167),sr(desc,4)],MemoryClass::VmemStore)?;
    } else {
        mem(b,format!("buffer_store_b32 v{val}, v185, {dest}, null offen"),&[],&[v(185),v(val),sr(desc,4)],MemoryClass::VmemStore)?;
    }
    op(b,"s_mov_b32 exec_lo, s79",&[],&[s(79)])?;
    Ok(())
}

pub(super) fn emit(b:&mut Builder,spec:Spec)->Result<(),String>{
    b.label(EPILOGUE)?;
    // D is reused across all row16 tiles; Ew is reused across the four token
    // tiles. Both loads are out-of-range safe for masked padded rows/tokens.
    so(b,"s_lshl_b32 s72, s83, 10",&[72],&[83])?;
    for i in 0..4 {for e in 0..8 {
        let dst=128+(i*8+e) as u8;
        vload(b,dst,1,180,44,Some(72),(i*64+e*4) as u32)?;
    }}
    if spec.act_scale==ActScale::K128 {
        vo(b,"v_lshrrev_b32_e32 v181, 2, v181",&[181],&[181],&[])?;
        vo(b,"v_mul_lo_u32 v181, v181, s84",&[181],&[181],&[84])?;
        vo(b,"v_lshlrev_b32_e32 v181, 2, v181",&[181],&[181],&[])?;
        so(b,"s_mul_i32 s72, s82, s84",&[72],&[82,84])?;
        so(b,"s_add_co_i32 s72, s72, s84",&[72],&[72,84])?;
        so(b,"s_add_co_i32 s72, s72, -1",&[72],&[72])?;
        so(b,"s_lshl_b32 s72, s72, 2",&[72],&[72])?;
    }else{so(b,"s_lshl_b32 s72, s82, 2",&[72],&[82])?;}
    for tt in 0..4 {
        if tt==0 {vo(b,"v_mov_b32_e32 v190, v181",&[190],&[181],&[])?;}
        else if spec.act_scale==ActScale::K128 {
            so(b,format!("s_mul_i32 s73, s84, {}",crate::kernels::iu4_gemm::lit((tt*64) as u32)),&[73],&[84])?;
            vo(b,"v_add_nc_u32_e32 v190, s73, v181",&[190],&[181],&[73])?;
        } else {vo(b,format!("v_add_nc_u32_e32 v190, {}, v181",crate::kernels::iu4_gemm::lit((tt*64) as u32)),&[190],&[181],&[])?;}
        vload(b,160+tt,1,190,48,Some(72),0)?;
    }
    // Precisely RN(D * RN(Ew * C)): no fma or reciprocal contraction.
    for i in 0..4 {for tt in 0..4 {for e in 0..8 {
        let acc=(8*(i*4+tt)+e) as u8;
        let ew=128+(i*8+e) as u8;
        vo(b,format!("v_mul_f32_e32 v{acc}, v{acc}, v{ew}"),&[acc],&[acc,ew],&[])?;
        vo(b,format!("v_mul_f32_e32 v{acc}, v{acc}, v{}",160+tt),&[acc],&[acc,160+tt as u8],&[])?;
    }}}
    // Absolute virtual row and token coordinates of the native WMMA output.
    // The lane's low four bits are token columns; high bit owns 8 row values.
    so(b,"s_lshl_b32 s72, s83, 8",&[72],&[83])?;
    so(b,"s_lshl_b32 s73, s88, 6",&[73],&[88])?;
    so(b,"s_add_co_i32 s72, s72, s73",&[72],&[72,73])?;
    vo(b,"v_and_b32_e32 v184, 31, v187",&[184],&[187],&[])?;
    vo(b,"v_lshrrev_b32_e32 v184, 4, v184",&[184],&[184],&[])?;
    vo(b,"v_lshlrev_b32_e32 v184, 3, v184",&[184],&[184],&[])?;
    vo(b,"v_add_nc_u32_e32 v184, s72, v184",&[184],&[184],&[72])?;
    vo(b,"v_and_b32_e32 v183, 15, v187",&[183],&[187],&[])?;
    so(b,"s_lshl_b32 s73, s89, 6",&[73],&[89])?;
    so(b,"s_add_co_i32 s73, s73, s82",&[73],&[73,82])?;
    vo(b,"v_add_nc_u32_e32 v183, s73, v183",&[183],&[183],&[73])?;
    if spec.epi==Epi::GateUpSilu {
        let region=Region::silu()?;
        if region.temps>7||region.masks>2 {return Err("SiLU region exceeds F2 borrowed registers".into())}
        // The two input row16 tiles of each pair share the same h coordinate.
        for pair in 0..2 {for tt in 0..4 {for e in 0..8 {
            let g=(8*((2*pair)*4+tt)+e) as u8;
            let u=(8*((2*pair+1)*4+tt)+e) as u8;
            region::emit_interleaved(b,&region,&[Binding {g,u,out:175,temps:(168..175).collect(),masks:vec![80,81]}])?;
            vo(b,format!("v_add_nc_u32_e32 v188, {}, v184",pair*16+e),&[188],&[184],&[])?;
            // v184 contains rb*256+rg*64+lane_row; divide rb/rg
            // contributions by two, but retain paired row16 selection.
            so(b,"s_lshl_b32 s72, s83, 7",&[72],&[83])?;
            so(b,"s_lshl_b32 s73, s88, 5",&[73],&[88])?;
            so(b,"s_add_co_i32 s72, s72, s73",&[72],&[72,73])?;
            vo(b,"v_and_b32_e32 v188, 31, v187",&[188],&[187],&[])?;
            vo(b,"v_lshrrev_b32_e32 v188, 4, v188",&[188],&[188],&[])?;
            vo(b,"v_lshlrev_b32_e32 v188, 3, v188",&[188],&[188],&[])?;
            vo(b,format!("v_add_nc_u32_e32 v188, {}, v188",pair*16+e),&[188],&[188],&[])?;
            vo(b,"v_add_nc_u32_e32 v188, s72, v188",&[188],&[188],&[72])?;
            vo(b,format!("v_add_nc_u32_e32 v189, {}, v183",tt*16),&[189],&[183],&[])?;
            so(b,"s_mov_b32 s72, 0",&[72],&[])?;
            store(b,0,188,189,175)?;
        }}}
    } else {
        for i in 0..4 {for tt in 0..4 {for e in 0..8 {
            let acc=(8*(i*4+tt)+e) as u8;
            vo(b,format!("v_add_nc_u32_e32 v188, {}, v184",i*16+e),&[188],&[184],&[])?;
            vo(b,format!("v_add_nc_u32_e32 v189, {}, v183",tt*16),&[189],&[183],&[])?;
            so(b,"s_mov_b32 s72, 0",&[72],&[])?;
            for seg in 0..spec.epi.count() {
                store(b,seg,188,189,acc)?;
                so(b,format!("s_add_co_i32 s72, s72, s{}",26+seg),&[72],&[72,26+seg as u8])?;
            }
        }}}
    }
    op(b,"s_mov_b32 exec_lo, -1",&[],&[])?;
    b.wait_all()?;
    Ok(())
}
