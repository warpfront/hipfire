use serde::Serialize;
#[derive(Clone,Copy,Debug,PartialEq,Eq,Serialize)] pub enum SlotState {Free,Publishing,Published,Reading}
#[derive(Clone,Debug,PartialEq,Serialize)] pub struct LdsSlot {pub name:String,pub base:u32,pub len:u32,pub state:SlotState}
/// One barrier-carried slot transition (shared with the typed core).
pub use peacemaker_author::SlotTransition as Transition;
#[derive(Clone,Debug,Default,PartialEq)] pub struct Lds {pub slots:Vec<LdsSlot>,in_flight:Vec<Transition>,signalled:bool,retired:Vec<LdsSlot>}
impl Lds {
 pub fn add(&mut self,name:impl Into<String>,base:u32,len:u32)->Result<usize,String>{ if len==0||base.checked_add(len).is_none(){return Err("invalid LDS slot".into())} if self.slots.iter().any(|s|base<s.base+s.len&&s.base<base+len){return Err("overlapping LDS slots".into())} self.slots.push(LdsSlot{name:name.into(),base,len,state:SlotState::Free});Ok(self.slots.len()-1) }
 fn slot(&mut self,id:usize)->Result<&mut LdsSlot,String>{if self.signalled && self.in_flight.iter().any(|t|matches!(t,Transition::Retire(n)|Transition::Ready(n) if *n==id)){return Err("LDS slot in split-barrier transition".into())}self.slots.get_mut(id).ok_or("unknown LDS slot".into())}
 pub fn store(&mut self,id:usize)->Result<(),String>{let s=self.slot(id)?;if !matches!(s.state,SlotState::Free|SlotState::Publishing){return Err(format!("{} cannot be stored while {:?}",s.name,s.state))}s.state=SlotState::Publishing;Ok(())}
 pub fn load(&mut self,id:usize)->Result<(),String>{let s=self.slot(id)?;if !matches!(s.state,SlotState::Published|SlotState::Reading){return Err(format!("{} is not published",s.name))}s.state=SlotState::Reading;Ok(())}
 pub fn barrier_signal(&mut self,transitions:&[Transition],stores_drained:bool)->Result<(),String>{if self.signalled{return Err("split barrier already in flight".into())}if !stores_drained{return Err("LDS stores must be drained before barrier signal".into())}for t in transitions {let (id,expected)=match t {Transition::Retire(id)=>(*id,SlotState::Reading),Transition::Ready(id)=>(*id,SlotState::Publishing)};if self.slots.get(id).map(|s|s.state)!=Some(expected){return Err(format!("invalid LDS transition {t:?}"))}}self.in_flight=transitions.to_vec();self.signalled=true;Ok(())}
 pub fn barrier_wait(&mut self)->Result<(),String>{if !self.signalled{return Err("no barrier signal in flight".into())}for t in self.in_flight.drain(..){match t {Transition::Retire(id)=>self.slots[id].state=SlotState::Free,Transition::Ready(id)=>self.slots[id].state=SlotState::Published}}self.signalled=false;Ok(())}
 pub fn barrier(&mut self,transitions:&[Transition],stores_drained:bool)->Result<(),String>{self.barrier_signal(transitions,stores_drained)?;self.barrier_wait()}
 /// End the current slot layout so a later program phase can declare its
 /// own (possibly overlapping) slots. Every slot must be Free (its readers
 /// retired by a barrier) and no split barrier may be in flight; LDS bytes
 /// carry no typed meaning across the boundary. Slot ids restart at zero.
 pub fn relayout(&mut self)->Result<(),String>{if self.signalled{return Err("relayout with a split barrier in flight".into())}if let Some(s)=self.slots.iter().find(|s|s.state!=SlotState::Free){return Err(format!("{} is {:?} at relayout",s.name,s.state))}self.retired.append(&mut self.slots);Ok(())}
 /// Join another control path's slot states into this point. Equal states
 /// stay; a slot published on both paths and read on one is `Reading` (its
 /// reads still need a barrier to retire); a slot stored on one path and
 /// untouched on the other is `Publishing` (a typed region `Writing` on both:
 /// the next barrier publishes whatever this and the other waves stored);
 /// any other disagreement, or a different layout or barrier in flight, is
 /// refused.
 pub fn join(&mut self,other:&Lds)->Result<(),String>{
  if self.signalled!=other.signalled||self.in_flight!=other.in_flight{return Err("paths join with different split barriers in flight".into())}
  if self.slots.len()!=other.slots.len()||self.retired.len()!=other.retired.len(){return Err("paths join with different LDS layouts".into())}
  for (mine,theirs) in self.slots.iter_mut().zip(&other.slots){
   if (&mine.name,mine.base,mine.len)!=(&theirs.name,theirs.base,theirs.len){return Err("paths join with different LDS layouts".into())}
   mine.state=match (mine.state,theirs.state){
    (a,b) if a==b=>a,
    (SlotState::Published|SlotState::Reading,SlotState::Published|SlotState::Reading)=>SlotState::Reading,
    (SlotState::Free|SlotState::Publishing,SlotState::Free|SlotState::Publishing)=>SlotState::Publishing,
    (a,b)=>return Err(format!("paths join with {} {a:?} on one and {b:?} on the other",mine.name)),
   };
  }
  Ok(())
 }
 /// Every slot of every layout, retired phases first (the proof record).
 pub fn all_slots(&self)->Vec<LdsSlot>{self.retired.iter().chain(&self.slots).cloned().collect()}
}
