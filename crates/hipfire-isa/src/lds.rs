use serde::Serialize;
#[derive(Clone,Copy,Debug,PartialEq,Eq,Serialize)] pub enum SlotState {Free,Publishing,Published,Reading}
#[derive(Clone,Debug,Serialize)] pub struct LdsSlot {pub name:String,pub base:u32,pub len:u32,pub state:SlotState}
#[derive(Clone,Copy,Debug,PartialEq,Eq,Serialize)] pub enum Transition {Retire(usize),Ready(usize)}
#[derive(Clone,Debug,Default)] pub struct Lds {pub slots:Vec<LdsSlot>,in_flight:Vec<Transition>,signalled:bool}
impl Lds {
 pub fn add(&mut self,name:impl Into<String>,base:u32,len:u32)->Result<usize,String>{ if len==0||base.checked_add(len).is_none(){return Err("invalid LDS slot".into())} if self.slots.iter().any(|s|base<s.base+s.len&&s.base<base+len){return Err("overlapping LDS slots".into())} self.slots.push(LdsSlot{name:name.into(),base,len,state:SlotState::Free});Ok(self.slots.len()-1) }
 fn slot(&mut self,id:usize)->Result<&mut LdsSlot,String>{if self.signalled && self.in_flight.iter().any(|t|matches!(t,Transition::Retire(n)|Transition::Ready(n) if *n==id)){return Err("LDS slot in split-barrier transition".into())}self.slots.get_mut(id).ok_or("unknown LDS slot".into())}
 pub fn store(&mut self,id:usize)->Result<(),String>{let s=self.slot(id)?;if !matches!(s.state,SlotState::Free|SlotState::Publishing){return Err(format!("{} cannot be stored while {:?}",s.name,s.state))}s.state=SlotState::Publishing;Ok(())}
 pub fn load(&mut self,id:usize)->Result<(),String>{let s=self.slot(id)?;if !matches!(s.state,SlotState::Published|SlotState::Reading){return Err(format!("{} is not published",s.name))}s.state=SlotState::Reading;Ok(())}
 pub fn barrier_signal(&mut self,transitions:&[Transition],stores_drained:bool)->Result<(),String>{if self.signalled{return Err("split barrier already in flight".into())}if !stores_drained{return Err("LDS stores must be drained before barrier signal".into())}for t in transitions {let (id,expected)=match t {Transition::Retire(id)=>(*id,SlotState::Reading),Transition::Ready(id)=>(*id,SlotState::Publishing)};if self.slots.get(id).map(|s|s.state)!=Some(expected){return Err(format!("invalid LDS transition {t:?}"))}}self.in_flight=transitions.to_vec();self.signalled=true;Ok(())}
 pub fn barrier_wait(&mut self)->Result<(),String>{if !self.signalled{return Err("no barrier signal in flight".into())}for t in self.in_flight.drain(..){match t {Transition::Retire(id)=>self.slots[id].state=SlotState::Free,Transition::Ready(id)=>self.slots[id].state=SlotState::Published}}self.signalled=false;Ok(())}
 pub fn barrier(&mut self,transitions:&[Transition],stores_drained:bool)->Result<(),String>{self.barrier_signal(transitions,stores_drained)?;self.barrier_wait()}
}
