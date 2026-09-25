use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Arch { Gfx1201, Gfx1100, Gfx1151 }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WgIdSource { Ttmp7, Ttmp9, Sgpr(u8) }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WgIds { pub x: WgIdSource, pub y: WgIdSource }

impl Arch {
    pub fn name(self) -> &'static str { match self { Self::Gfx1201 => "gfx1201", Self::Gfx1100 => "gfx1100", Self::Gfx1151 => "gfx1151" } }
    pub fn gfx12(self) -> bool { matches!(self, Self::Gfx1201) }
    pub fn buffer_offset_max(self) -> u32 { if self.gfx12() { (1 << 23) - 1 } else { 4095 } }
    pub fn wmma_iu4(self) -> &'static str { if self.gfx12() { "v_wmma_i32_16x16x32_iu4" } else { "v_wmma_i32_16x16x16_iu4" } }
    pub fn dependency_wait(self) -> &'static str { if self.gfx12() { "s_wait_alu" } else { "s_waitcnt_depctr" } }
    pub fn barrier(self) -> &'static [&'static str] { if self.gfx12() { &["s_barrier_signal -1", "s_barrier_wait 0xffff"] } else { &["s_barrier"] } }
    pub fn workgroup_ids(self, user_sgpr_count: u8) -> Result<WgIds, String> {
        if self.gfx12() {
            Ok(WgIds { x: WgIdSource::Ttmp9, y: WgIdSource::Ttmp7 })
        } else if user_sgpr_count <= 102 {
            Ok(WgIds { x: WgIdSource::Sgpr(user_sgpr_count), y: WgIdSource::Sgpr(user_sgpr_count + 1) })
        } else {
            Err("workgroup ID SGPRs exceed architectural range".into())
        }
    }
    pub fn check_mnemonic(self, mnemonic: &str) -> Result<(), String> {
        if (self.gfx12() && (mnemonic == "s_waitcnt" || mnemonic == "s_waitcnt_vscnt")) || (!self.gfx12() && (mnemonic.starts_with("s_wait_loadcnt") || mnemonic.starts_with("s_wait_storecnt") || mnemonic.starts_with("s_wait_dscnt") || mnemonic.starts_with("s_wait_kmcnt") || mnemonic == "s_wait_alu" || mnemonic.starts_with("v_swmmac") || mnemonic == "v_wmma_i32_16x16x32_iu4" || mnemonic == "global_inv" || mnemonic == "s_barrier_signal" || mnemonic == "s_barrier_wait" || mnemonic == "s_add_f32" || mnemonic == "s_mul_f32" || mnemonic == "v_s_rcp_f32")) || (self.gfx12() && (mnemonic == "s_barrier" || mnemonic.starts_with("buffer_gl") || mnemonic == "s_waitcnt_depctr")) {
            return Err(format!("{mnemonic} is unsupported on {}", self.name()));
        }
        Ok(())
    }
}
impl std::str::FromStr for Arch {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> { match s { "gfx1201" => Ok(Self::Gfx1201), "gfx1100" => Ok(Self::Gfx1100), "gfx1151" => Ok(Self::Gfx1151), _ => Err(format!("unsupported architecture {s}")) } }
}
