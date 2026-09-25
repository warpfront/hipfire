use serde::Serialize;
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind { V, S }
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct RegRef { pub kind: Kind, pub base: u8, pub len: u8 }
impl RegRef {
    pub fn overlaps(self, other: Self) -> bool { self.kind == other.kind && u16::from(self.base) < u16::from(other.base) + u16::from(other.len) && u16::from(other.base) < u16::from(self.base) + u16::from(self.len) }
}
impl fmt::Display for RegRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let prefix = if self.kind == Kind::V { 'v' } else { 's' };
        if self.len == 1 { write!(f, "{prefix}{}", self.base) } else { write!(f, "{prefix}[{}:{}]", self.base, u16::from(self.base) + u16::from(self.len) - 1) }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct V<const N: u8>(pub u8);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S<const N: u8>(pub u8);
impl<const N: u8> V<N> { pub fn reg(self) -> RegRef { RegRef { kind: Kind::V, base: self.0, len: N } } pub fn base(self) -> u8 { self.0 } }
impl<const N: u8> S<N> { pub fn reg(self) -> RegRef { RegRef { kind: Kind::S, base: self.0, len: N } } pub fn base(self) -> u8 { self.0 } }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Special { VccLo, ExecLo, Null, Ttmp7, Ttmp9, M0 }
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum Live { Whole, Between(String, String) }
impl Live { pub fn contains(&self, label: &str, order: &[String]) -> bool { match self { Self::Whole => true, Self::Between(start, end) => { let at = order.iter().position(|l| l == label); let first = order.iter().position(|l| l == start); let last = order.iter().position(|l| l == end).unwrap_or(order.len()); matches!((at,first), (Some(at),Some(first)) if first <= at && at < last) } } } }
#[derive(Clone, Debug, Serialize)]
pub struct Range { pub name: &'static str, pub kind: Kind, pub base: u8, pub len: u8, pub live: Live }
impl Range { pub fn reg(&self) -> RegRef { RegRef { kind: self.kind, base: self.base, len: self.len } } }
#[derive(Clone, Debug)]
pub struct Scope { pub start: String, pub end: String }
#[derive(Clone, Debug)]
pub struct RegPlan { pub vgpr_budget: u16, pub sgpr_budget: u16, pub ranges: Vec<Range>, pub scratch: Option<(u8,u8)> }
impl RegPlan {
    pub fn new(vgpr_budget: u16, sgpr_budget: u16) -> Result<Self,String> { if vgpr_budget > 256 || sgpr_budget > 104 { return Err("register budget exceeds architectural limit".into()) } Ok(Self { vgpr_budget, sgpr_budget, ranges: Vec::new(), scratch: None }) }
    pub fn scratch_pool(&mut self, base: u8, len: u8) -> Result<(),String> { if u16::from(base)+u16::from(len)>self.vgpr_budget { return Err("scratch pool exceeds VGPR budget".into()) } self.scratch=Some((base,len)); Ok(()) }
    fn add(&mut self,name:&'static str,kind:Kind,base:u8,len:u8,live:Live)->Result<(),String> {
        if !matches!(len,1|2|4|8) { return Err("register width must be 1, 2, 4 or 8".into()) }
        if kind==Kind::S && len>1 && base%len!=0 { return Err(format!("misaligned SGPR {base}:{len}")) }
        let budget=if kind==Kind::V { self.vgpr_budget } else { self.sgpr_budget };
        if u16::from(base)+u16::from(len)>budget { return Err(format!("{name} exceeds register budget")) }
        let reg=RegRef { kind,base,len };
        for existing in &self.ranges { if reg.overlaps(existing.reg()) && (live==Live::Whole || existing.live==Live::Whole || live==existing.live) { return Err(format!("{name} overlaps {} with a shared lifetime",existing.name)) } }
        self.ranges.push(Range { name,kind,base,len,live }); Ok(())
    }
    pub fn v<const N:u8>(&mut self,name:&'static str,base:u8,live:Live)->Result<V<N>,String> { self.add(name,Kind::V,base,N,live)?; Ok(V(base)) }
    pub fn s<const N:u8>(&mut self,name:&'static str,base:u8,live:Live)->Result<S<N>,String> { self.add(name,Kind::S,base,N,live)?; Ok(S(base)) }
    pub fn temp_v<const N:u8>(&mut self,scope:&Scope)->Result<V<N>,String> { let (base,len)=self.scratch.ok_or("no scratch pool declared")?; for b in base..base.saturating_add(len) { if u16::from(b)+u16::from(N)>u16::from(base)+u16::from(len) { break } let live=Live::Between(scope.start.clone(),scope.end.clone()); if self.add("temp",Kind::V,b,N,live).is_ok() { return Ok(V(b)) } } Err("scratch pool has no free range".into()) }
    pub fn next_free_vgpr(&self)->u16 { self.ranges.iter().filter(|r|r.kind==Kind::V).map(|r|u16::from(r.base)+u16::from(r.len)).max().unwrap_or(0) }
    pub fn next_free_sgpr(&self)->u16 { self.ranges.iter().filter(|r|r.kind==Kind::S).map(|r|u16::from(r.base)+u16::from(r.len)).max().unwrap_or(0) }
    pub fn verify_access(&self,reg:RegRef,label:&str,order:&[String])->Result<(),String> { let owners:Vec<_>=self.ranges.iter().filter(|r|r.reg().overlaps(reg)&&r.live.contains(label,order)).collect(); if owners.len()!=1 || owners[0].kind!=reg.kind || owners[0].base>reg.base || u16::from(owners[0].base)+u16::from(owners[0].len)<u16::from(reg.base)+u16::from(reg.len) { return Err(format!("{reg} at {label} has no unique live owner")) } Ok(()) }
    pub fn verify_lifetimes(&self,order:&[String])->Result<(),String> { for r in &self.ranges { if let Live::Between(first,last)=&r.live {let a=order.iter().position(|l|l==first).ok_or_else(||format!("missing lifetime label {first}"))?;let b=order.iter().position(|l|l==last).ok_or_else(||format!("missing lifetime label {last}"))?;if a>=b{return Err(format!("invalid lifetime for {}",r.name))}}}for (i,a) in self.ranges.iter().enumerate(){for b in self.ranges.iter().skip(i+1){if a.reg().overlaps(b.reg())&&order.iter().any(|label|a.live.contains(label,order)&&b.live.contains(label,order)){return Err(format!("live ranges {} and {} overlap",a.name,b.name))}}}Ok(()) }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub struct Vb<const B:u8>(pub u8);
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub struct Vp<const P:u8>(pub u8);
impl<const B:u8> Vb<B> { pub fn checked(base:u8)->Result<Self,String> { if B>=4 || base%4!=B { Err("VGPR bank mismatch".into()) } else { Ok(Self(base)) } } }
impl<const P:u8> Vp<P> { pub fn checked(base:u8)->Result<Self,String> { if P>=2 || base%2!=P { Err("VGPR parity mismatch".into()) } else { Ok(Self(base)) } } }
pub trait DistinctBanks<const OTHER:u8> {}
macro_rules! distinct { ($b:literal: $($other:literal),+) => { $(impl DistinctBanks<$other> for Vb<$b> {})+ }; }
distinct!(0:1,2,3); distinct!(1:0,2,3); distinct!(2:0,1,3); distinct!(3:0,1,2);
pub trait OppositeParity<const OTHER:u8> {}
impl OppositeParity<1> for Vp<0> {} impl OppositeParity<0> for Vp<1> {}
impl V<8> { pub fn pair(self,j:u8)->Result<(V<1>,V<1>),String> { if j>=8 || j%2!=0 || u16::from(self.0)+u16::from(j)+1>255 { Err("invalid V8 pair".into()) } else { Ok((V(self.0+j),V(self.0+j+1))) } } }
