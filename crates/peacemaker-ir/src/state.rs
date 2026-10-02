use crate::{cfg::InstId, inst::Program, kernarg::KernargFacts, lds::LdsFacts, wait::WaitFact};
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RoundTripProof {
    pub input_sha256: [u8; 32], pub tools: Vec<String>, pub streams: Vec<StreamProof>,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StreamProof { pub entry: u64, pub size: u64, pub stream_sha256: [u8; 32] }
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LiftReport { pub rejections: Vec<(u64, String)> }
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Lifted<P> { pub program: P, pub proof: RoundTripProof, pub report: LiftReport }
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ResourceSummary { pub max_vgpr: u16, pub max_sgpr: u16, pub uses_vcc: bool, pub uses_flat_scratch: bool, pub lds_fixed: u32, pub lds_dynamic_max: Option<u32> }
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Facts { pub waits: Vec<WaitFact>, pub lds: LdsFacts, pub resources: Vec<ResourceSummary>, pub kernargs: KernargFacts }
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ObligationKind { SrcReadTiming(crate::effects::MemClass), LdsBounds, Hazard, Definedness, Unknown }
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Obligation { pub kind: ObligationKind, pub insts: Vec<InstId>, pub rule_id: String, pub text: String }
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Analyzed<P> { pub program: P, pub facts: Facts, pub obligations: Vec<Obligation>, pub revision: u32 }
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Receipt { pub bytes: Vec<u8> }
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Certified<P> { pub program: P, pub receipt: Receipt }
impl Lifted<Program> {
    pub fn validate(&self) -> Result<(), crate::inst::ValidateError> { self.program.validate() }
}
