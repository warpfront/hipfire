use smallvec::SmallVec;
use crate::{cfg::{BarrierKind, Cond}, inst::{Arch, Form, Inst, Opcode, ValidateError}, lds::LdsAccess, operand::{CachePolicy, Operand, Special}, reg::{Kind, RegRef}, wait::{Counter, CounterSet}};
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ImplicitSet { pub reads: u8, pub writes: u8 }
impl ImplicitSet {
    pub const SCC: u8 = 1 << 0;
    pub const VCC: u8 = 1 << 1;
    pub const EXEC: u8 = 1 << 2;
    pub const M0: u8 = 1 << 3;
    pub const MODE: u8 = 1 << 4;
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemClass { VmemLoad, VmemStore, VmemAtomic { returns: bool }, DsLoad, DsStore, DsAtomic { returns: bool }, SmemLoad, SmemStore, Export, LdsDma, FlatLoad, FlatStore, FlatAtomic { returns: bool } }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OrderType { Load, Store, Ds, Smem, Sample, Bvh, Exp, Mixed }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SrcRead { AtIssue, Deferred }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemEffect { pub class: MemClass, pub counters: CounterSet, pub in_order_type: OrderType, pub src_read: SrcRead }
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Control { #[default] None, Branch { cond: Cond }, Jump, EndPgm, Barrier(BarrierKind), Wait, Clause, Delay, Halt, Trap }
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Effects { pub defs: SmallVec<[RegRef; 2]>, pub uses: SmallVec<[RegRef; 4]>, pub implicit: ImplicitSet, pub mem: Option<MemEffect>, pub control: Control, pub lds: Option<LdsAccess> }
/// The old VGPR value preserved by a lane/half write.  This is not a
/// full-register definition: without lane-sensitive proof, an undefined old
/// destination must remain an open obligation rather than a proved read.
pub fn preserved_vgpr_destination(arch: Arch, inst: &Inst) -> Option<RegRef> {
    let writelane = inst.op.name(arch) == Some("v_writelane_b32");
    inst.operands.iter().enumerate().find_map(|(i, operand)| {
        let reg = match operand {
            Operand::Half(reg, _) => reg,
            Operand::Reg(reg) if writelane && i == 0 => reg,
            _ => return None,
        };
        if reg.kind != Kind::V || !inst.effects.defs.contains(reg) { return None; }
        // An explicit source overlapping the old destination remains a real
        // undefined read, not merely the value preserved by the partial write.
        let overlap = inst.operands.iter().enumerate().any(|(j, src)| {
            if j == i { return false; }
            let other = match src { Operand::Reg(r) | Operand::Half(r, _) => r, _ => return false };
            other.kind == reg.kind && other.base < reg.base + u16::from(reg.len) && reg.base < other.base + u16::from(other.len)
        });
        (!overlap).then_some(*reg)
    })
}


impl Effects {
    /// Single table source for authoring and lifted instruction effects.
    /// Physical register widths come from decoded operands; implicit reads and
    /// writes, counter membership and opcode-specific accumulator uses come
    /// from the target's opcode table. `cpol` selects a VMEM atomic's returning
    /// or non-returning form (its `@rtn` destination slot and counter rules).
    pub fn from_table(arch: Arch, op: Opcode, form: Form, operands: &[Operand], cpol: &CachePolicy) -> Result<Self, ValidateError> {
        let row = crate::isa::lookup(arch, op, form)
            .ok_or(ValidateError::UnknownOpcode { op, form })?;
        let returns = crate::isa::atomic_returns(arch, cpol);
        let mut out = Self::default();
        let mut field_names = row.slots(returns).map(|(name, _)| name);
        // CMPX encodes the fixed EXEC destination as VDST, but it is not an operand.
        if form == Form::Vop3 && row.name.starts_with("v_cmpx_") {
            field_names.next();
        }
        // VOPC's explicit printed `vcc_lo` destination is fieldless in XML.
        let prefix = usize::from(form == Form::Vopc && row.name.starts_with("v_cmp_")
            && matches!(operands.first(), Some(Operand::Special(Special::Vcc | Special::VccLo))));
        for operand in operands.iter().skip(prefix) {
            let name = field_names.next().unwrap_or("");
            let def = row.defs.split(',').any(|s| s == name);
            // A half-register destination leaves the other half intact.
            let used = row.uses.split(',').any(|s| s == name)
                || def && matches!(operand, Operand::Half(..));
            match operand {
                Operand::Reg(reg) | Operand::Half(reg, _) => {
                    if def { out.defs.push(*reg); }
                    if used { out.uses.push(*reg); }
                }
                Operand::Special(special) => {
                    let bit = match special {
                        Special::Scc => ImplicitSet::SCC,
                        Special::Vcc | Special::VccLo | Special::VccHi => ImplicitSet::VCC,
                        Special::Exec | Special::ExecLo | Special::ExecHi => ImplicitSet::EXEC,
                        Special::M0 => ImplicitSet::M0,
                        _ => 0,
                    };
                    if def { out.implicit.writes |= bit; }
                    if used { out.implicit.reads |= bit; }
                }
                _ => {}
            }
        }
        for effect in row.implicit.split(',') {
            let (reg, mode) = effect.split_once(':').unwrap_or(("", ""));
            let bit = match reg { "scc" => ImplicitSet::SCC, "vcc" => ImplicitSet::VCC, "exec" => ImplicitSet::EXEC, "m0" => ImplicitSet::M0, "mode" => ImplicitSet::MODE, _ => continue };
            if mode.contains('r') { out.implicit.reads |= bit; }
            if mode.contains('w') { out.implicit.writes |= bit; }
        }
        let name = row.name;
        out.control = if name == "s_endpgm" { Control::EndPgm }
            else if name == "s_branch" { Control::Jump }
            else if name.starts_with("s_cbranch") {
                let cond = if name.ends_with("scc0") { Cond::Scc0 }
                    else if name.ends_with("scc1") { Cond::Scc1 }
                    else if name.ends_with("vccz") { Cond::Vccz }
                    else if name.ends_with("vccnz") { Cond::Vccnz }
                    else if name.ends_with("execnz") { Cond::Execnz } else { Cond::Execz };
                Control::Branch { cond }
            } else if name == "s_barrier" { Control::Barrier(BarrierKind::Full)
            } else if name == "s_barrier_signal" { Control::Barrier(BarrierKind::Signal(crate::cfg::BarrierId(0))) }
            else if name == "s_barrier_wait" { Control::Barrier(BarrierKind::Wait) }
            else if name.starts_with("s_wait") { Control::Wait }
            else if name == "s_clause" { Control::Clause }
            else if name == "s_delay_alu" { Control::Delay }
            else { Control::None };
        for (counter, units) in row.counter_rules(returns) {
            let counter = match counter {
                "Km" => Counter::Km, "Ds" => Counter::Ds, "Load" => Counter::Load,
                "Store" => Counter::Store, "Sample" => Counter::Sample, "Bvh" => Counter::Bvh,
                "Exp" => Counter::Exp, "Vm" => Counter::Vm, "Vs" => Counter::Vs,
                "Lgkm" => Counter::Lgkm,
                _ => return Err(ValidateError::Operand("unknown counter rule".into())),
            };
            if !matches!(units, "1" | "2") { return Err(ValidateError::Operand("unsupported counter unit weight".into())); }
            let (class, order) = if name.starts_with("s_load") { (MemClass::SmemLoad, OrderType::Smem) }
                else if name.starts_with("s_sendmsg_rtn") { (MemClass::SmemLoad, OrderType::Smem) }
                else if name.starts_with("ds_load") { (MemClass::DsLoad, OrderType::Ds) }
                else if name.starts_with("ds_store") { (MemClass::DsStore, OrderType::Ds) }
                else if name.starts_with("ds_") { (MemClass::DsAtomic { returns: !out.defs.is_empty() }, OrderType::Ds) }
                else if name.starts_with("flat_load") { (MemClass::FlatLoad, OrderType::Load) }
                else if name.starts_with("flat_store") { (MemClass::FlatStore, OrderType::Store) }
                else if name.starts_with("global_load") || name.starts_with("buffer_load") || name.starts_with("scratch_load") { (MemClass::VmemLoad, OrderType::Load) }
                else if name.starts_with("global_store") || name.starts_with("buffer_store") || name.starts_with("scratch_store") { (MemClass::VmemStore, OrderType::Store) }
                else if name.starts_with("global_atomic") || name.starts_with("buffer_atomic") {
                    (MemClass::VmemAtomic { returns }, if returns { OrderType::Load } else { OrderType::Store })
                }
                else { return Err(ValidateError::Operand(format!("unclassified memory rule for {name}"))); };
            let mem = out.mem.get_or_insert_with(|| MemEffect {
                class, counters: CounterSet::default(), in_order_type: order,
                src_read: if matches!(class, MemClass::DsStore) { SrcRead::Deferred } else { SrcRead::AtIssue },
            });
            mem.counters.insert(counter);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use crate::{codec::gfx12, reg::Kind};

    #[test]
    fn partial_writes_depend_on_the_previous_destination() {
        for words in [
            &[0x7e00_1500][..],                 // v_cvt_f16_f32_e32 v0.l, v0
            &[0xda98_08c0, 0x3e00_0032][..],  // ds_load_u16_d16 v62, v50
            &[0xcc22_0011, 0x244e_2f1a][..],  // v_fma_mixhi_f16 v17, ...
            &[0xd761_0061, 0x0201_0000][..],  // v_writelane_b32 v97, ...
        ] {
            let (inst, _) = gfx12::decode(words).unwrap();
            let dest = inst.effects.defs.iter().find(|r| r.kind == Kind::V).unwrap();
            assert!(inst.effects.uses.contains(dest), "{:?} fails to read its preserved destination", inst.op);
        }
    }
}
