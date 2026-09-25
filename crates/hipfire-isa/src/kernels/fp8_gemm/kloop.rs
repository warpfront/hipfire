use super::{BEGIN, LOOP, LOOP_END, EPILOGUE, Builder, Spec, fold, op, publish, so};
use crate::{insn::Wmma, lds::Transition, reg::V};

fn step(b:&mut Builder,unit:usize,st:usize,first:bool,slot:usize)->Result<(),String>{
    publish::fragments(b,slot,2*(unit%2)+st)?;
    for i in 0..4 {for tt in 0..4 {
        let dst=(i*4+tt)*8;
        let w=128+((unit&1)*16+i*4+st*2) as u8;
        let x=160+(tt*2) as u8;
        b.push(Wmma::fp8(b.spec.arch,V::<8>(dst as u8),V::<2>(w),V::<2>(x),
            if first {None}else{Some(V::<8>(dst as u8))})?)?;
    }}
    Ok(())
}

/// On entry A0 is Published. A1's VMEM payload is fetched before slab 0,
/// published after all slab-0 fragment readers, then a uniform barrier retires
/// A0. Metadata for b+1 is published in that same barrier. The final block
/// omits next-block prefetch and fold. No packed weight decode is performed.
fn block(b:&mut Builder,spec:Spec,next:bool,first:bool)->Result<(),String>{
    publish::weight_base(b)?;
    publish::fetch_w(b,0)?;
    publish::fetch_w(b,1)?;
    publish::fetch_a(b,0,1)?;
    for u in 0..2 {
        for st in 0..2 {step(b,u,st,first&&u==0&&st==0,0)?;}
        // Two-unit 16-VGPR ring: the unit just consumed is dead.
        publish::fetch_w(b,u+2)?;
    }
    publish::publish_a(b,1)?;
    if next {publish::publish_next_ratios(b,spec,1,((first as usize)+1)&1)?;}
    let mut b1=vec![Transition::Retire(0),Transition::Ready(1)];
    if next {
        let p=((first as usize)+1)&1;
        b1.push(Transition::Ready(2+p));
        if spec.act_scale==super::ActScale::K128 {b1.push(Transition::Ready(4+p));}
    }
    b.barrier(&b1)?;
    if next {publish::fetch_a(b,1,0)?;}
    for u in 2..4 {for st in 0..2 {step(b,u,st,false,1)?;}}
    if next {
        // The next A0 payload was prefetched before slab 1. Its stores are
        // deliberately delayed until all A1 reads have issued.
        publish::publish_a(b,0)?;
        let p=((first as usize)+1)&1;
        fold::emit(b,spec,p)?;
        let mut b2=vec![Transition::Retire(1),Transition::Retire(2+p),Transition::Ready(0)];
        if spec.act_scale==super::ActScale::K128 {b2.push(Transition::Retire(4+p));}
        b.barrier_signal(&b2)?;
        b.barrier_wait()?;
    } else {
        b.barrier(&[Transition::Retire(1)])?;
    }
    so(b,"s_add_co_i32 s85, s85, 1",&[85],&[85])?;
    b.wait_all()?;
    Ok(())
}

pub(super) fn emit(b:&mut Builder,spec:Spec)->Result<(),String>{
    b.label(BEGIN)?;
    // C=0 is the hardware WMMA's inline operand only for the first K16;
    // subsequent instructions carry the same 128 VGPRs as their SrcC.
    block(b,spec,true,true)?;
    so(b,"s_add_co_i32 s86, s84, -2",&[86],&[84])?;
    so(b,"s_lshr_b32 s86, s86, 1",&[86],&[86])?;
    so(b,"s_cmp_eq_u32 s86, 0",&[],&[86])?;
    op(b,format!("s_cbranch_scc1 {LOOP_END}"),&[],&[])?;
    b.loop_(LOOP,|b|{
        block(b,spec,true,false)?;
        block(b,spec,true,false)?;
        so(b,"s_add_co_i32 s86, s86, -1",&[86],&[86])?;
        so(b,"s_cmp_lg_u32 s86, 0",&[],&[86])?;
        op(b,format!("s_cbranch_scc1 {LOOP}"),&[],&[])
    })?;
    b.label(LOOP_END)?;
    block(b,spec,false,false)?;
    // Final global values are read only after all WMMA users of the fragment
    // ring have retired; no descriptor or LDS payload survives the epilogue.
    b.wait_all()?;
    op(b,format!("s_branch {EPILOGUE}"),&[],&[])
}
