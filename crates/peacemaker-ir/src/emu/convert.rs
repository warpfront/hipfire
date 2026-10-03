//! Integer bit-level round-to-nearest-even conversions, independent of host FP mode.
pub fn f16_to_f32(h:u16)->u32 {
    let sign=u32::from(h&0x8000)<<16;let exp=(h>>10)&31;let frac=u32::from(h&1023);
    match exp {0=>{if frac==0{sign}else{let shift=frac.leading_zeros()-21;sign|((113-shift)<<23)|((frac<<(shift+13))&0x7fffff)}},31=>sign|0x7f800000|(frac<<13),_=>sign|((u32::from(exp)+112)<<23)|(frac<<13)}
}
fn rne(x:u32,shift:u32)->u32 {if shift==0{return x;}if shift>32{return 0;}if shift==32{return u32::from(x>0x80000000);}let q=x>>shift;let mask=(1u32<<shift)-1;let r=x&mask;let half=1u32<<(shift-1);q+u32::from(r>half || r==half && q&1!=0)}
pub fn f32_to_f16(x:u32)->u16 {
    let sign=((x>>16)&0x8000)as u16;let e=((x>>23)&255)as i32;let f=x&0x7fffff;
    if e==255{return sign|0x7c00|if f==0{0}else{((f>>13)as u16)|0x200};}
    let he=e-112;if he>=31{return sign|0x7c00;}if he<=0 {if he < -10{return sign;}return sign|rne(f|0x800000,(14-he)as u32)as u16;}
    sign|(((he as u32)<<10)+rne(f,13))as u16
}
pub fn f32_to_bf16(x:u32)->u16 {if x&0x7fffffff>0x7f800000 {((x>>16)|0x40)as u16}else{(x.wrapping_add(0x7fff+((x>>16)&1))>>16)as u16}}
/// AMD gfx12 OCP E4M3: finite top exponent, signed zero, canonical NaN.
pub fn fp8_to_f32(x:u8)->u32 {
    let sign=u32::from(x&128)<<24;let e=(x>>3)&15;let f=u32::from(x&7);
    if e==15 && f==7 {return sign|0x7fc00000;}
    if e==0 {if f==0{return sign;}let shift=f.leading_zeros()-28;return sign|((121-shift)<<23)|((f<<(shift+20))&0x7fffff);}
    sign|((u32::from(e)+120)<<23)|(f<<20)
}
pub fn ldexp_f32(x:u32,n:i32)->u32 {
    let sign=x&0x80000000;let e=(x>>23)&255;let frac=x&0x7fffff;
    if e==255 {return if frac==0{x}else{x|0x400000};}
    if e==0 && frac==0 {return x;}
    let (mant,e)=if e==0 {let shift=frac.leading_zeros()-8;(frac<<shift,1-i64::from(shift))}else{(frac|0x800000,i64::from(e))};
    let exp=e+i64::from(n);
    if exp>=255 {return sign|0x7f800000;}
    if exp<=0 {return sign|rne(mant,(1-exp).min(33)as u32);}
    sign|((exp as u32)<<23)|(mant&0x7fffff)
}
/// `v_cvt_i32_f32`: truncate toward zero, saturate out-of-range values including infinity, NaN -> 0.
pub fn f32_to_i32(x:u32)->u32 {f32::from_bits(x) as i32 as u32}
/// `v_cvt_u32_f32`: truncate toward zero, saturate (negative values and -inf give 0), NaN -> 0.
pub fn f32_to_u32(x:u32)->u32 {f32::from_bits(x) as u32}
#[cfg(test)] mod tests {
    use super::*;
    #[test] fn half_boundaries_and_ties() {assert_eq!(f16_to_f32(1),2f32.powi(-24).to_bits());assert_eq!(f16_to_f32(0x3c00),1f32.to_bits());assert_eq!(f32_to_f16(1.00048828125f32.to_bits()),0x3c00);assert_eq!(f32_to_f16(1.00146484375f32.to_bits()),0x3c02);assert_eq!(f32_to_f16(65520f32.to_bits()),0x7c00);assert_eq!(f32_to_f16((-0f32).to_bits()),0x8000);}
    #[test] fn bf16_rounding_and_nan() {assert_eq!(f32_to_bf16(0x3f808000),0x3f80);assert_eq!(f32_to_bf16(0x3f818000),0x3f82);assert_eq!(f32_to_bf16(0x7f800001),0x7fc0);}
    #[test] fn fp8_ocp_extremes() {assert_eq!(fp8_to_f32(1),2f32.powi(-9).to_bits());assert_eq!(fp8_to_f32(0x7e),448f32.to_bits());assert_eq!(fp8_to_f32(0x80),0x80000000);assert!(f32::from_bits(fp8_to_f32(0x7f)).is_nan());}
    #[test] fn ldexp_preserves_signed_zero_and_rounds_subnormals() {
        assert_eq!(ldexp_f32(1f32.to_bits(),-149),1);
        assert_eq!(ldexp_f32(1f32.to_bits(),-150),0);
        assert_eq!(ldexp_f32(3f32.to_bits(),-150),2);
        assert_eq!(ldexp_f32(1,23),0x800000);
        assert_eq!(ldexp_f32(0x80000000,i32::MAX),0x80000000);
        assert_eq!(ldexp_f32(1f32.to_bits(),i32::MAX),0x7f800000);
    }
    #[test] fn ldexp_matches_exact_scaling_across_rounding_boundaries() {
        // Hand-computed in denormal units of 2^-149: 2^-127 = 0x400000, so 1.5*2^-127 = 0x600000 and 1.75*2^-127 = 0x700000.
        assert_eq!(ldexp_f32(1.5f32.to_bits(),-127),0x600000);
        assert_eq!(ldexp_f32(1.75f32.to_bits(),-127),0x700000);
        // Ties to even at the denormal LSB: 2.5*2^-149 -> 2 (tie, even), 3.5*2^-149 -> 4, 2.75*2^-149 -> 3.
        assert_eq!(ldexp_f32(1.25f32.to_bits(),-148),2);
        assert_eq!(ldexp_f32(1.75f32.to_bits(),-148),4);
        assert_eq!(ldexp_f32(1.375f32.to_bits(),-148),3);
        // 0x3fffffff * 2^-127 = 8388607.5 units: a tie, odd quotient rounds up onto the minimum normal.
        assert_eq!(ldexp_f32(0x3fffffff,-127),0x00800000);
        // Denormal input scaled into the normal range is exact; overflow rounds to infinity.
        assert_eq!(ldexp_f32(1,149),1f32.to_bits());
        assert_eq!(ldexp_f32(0x007fffff,1),0x00fffffe);
        assert_eq!(ldexp_f32(0xff7fffff,1),0xff800000);
        assert_eq!(ldexp_f32(0x7f800001,3),0x7fc00001);
        // Independent oracle: binary64 scaling is exact, `as f32` is one correctly rounded step.
        let mut x=0x1234_5678u32;
        for _ in 0..20000 {
            x^=x<<13;x^=x>>17;x^=x<<5;
            let bits=(x&0x807f_ffff)|(((x>>9)%255)<<23);
            let n=((x>>3)%460) as i32-300;
            let want=(f64::from(f32::from_bits(bits))*2f64.powi(n)) as f32;
            assert_eq!(ldexp_f32(bits,n),want.to_bits(),"{bits:#x} * 2^{n}");
        }
    }
    #[test] fn f32_to_integer_edges() {
        assert_eq!(f32_to_i32(f32::NAN.to_bits()),0);
        assert_eq!(f32_to_u32(f32::NAN.to_bits()),0);
        assert_eq!(f32_to_i32(f32::INFINITY.to_bits()),i32::MAX as u32);
        assert_eq!(f32_to_i32(f32::NEG_INFINITY.to_bits()),i32::MIN as u32);
        assert_eq!(f32_to_i32(2147483648f32.to_bits()),i32::MAX as u32);
        assert_eq!(f32_to_i32((-2147483648f32).to_bits()),i32::MIN as u32);
        assert_eq!(f32_to_i32((-1.9f32).to_bits()),(-1i32) as u32);
        assert_eq!(f32_to_u32(f32::INFINITY.to_bits()),u32::MAX);
        assert_eq!(f32_to_u32(f32::NEG_INFINITY.to_bits()),0);
        assert_eq!(f32_to_u32((-0.5f32).to_bits()),0);
        assert_eq!(f32_to_u32(4294967296f32.to_bits()),u32::MAX);
        assert_eq!(f32_to_u32(4294967040f32.to_bits()),4294967040);
    }
}
