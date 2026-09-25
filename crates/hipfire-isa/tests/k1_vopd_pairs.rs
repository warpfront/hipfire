use hipfire_isa::{Arch, Builder, KernargLayout, KernelSpec, RegPlan};
use hipfire_isa::kernels::iu4_k1::{emit_fold, FoldRegisters, Variant};
use hipfire_isa::reg::Live;
use std::io::Write;
use std::process::{Command, Stdio};

#[test]
fn k1_explicit_and_hidden_arguments_match_hipcc() {
    for (variant, size, first_hidden, dynamic_lds) in [
        (Variant::FullSet, 296, 40, 160),
        (Variant::FullAdd, 296, 40, 160),
        (Variant::GateUpSilu, 304, 48, 168),
    ] {
        let args = variant.kernargs();
        args.validate().unwrap();
        assert_eq!(args.size, size);
        assert_eq!(args.args.iter().find(|a| a.value_kind == "hidden_block_count_x").unwrap().offset,
            first_hidden);
        assert_eq!(args.args.iter().find(|a| a.value_kind == "hidden_dynamic_lds_size").unwrap().offset,
            dynamic_lds);
    }
}

#[test]
fn all_96_k1_fold_packets_encode_on_gfx1201() {
    let mut plan = RegPlan::new(200, 48).unwrap();
    let cacc = std::array::from_fn(|i| {
        plan.v::<8>("cacc", (i * 8) as u8, Live::Whole).unwrap()
    });
    let acc = std::array::from_fn(|i| {
        plan.v::<8>("acc", (64 + i * 8) as u8, Live::Whole).unwrap()
    });
    let magic = plan.v::<8>("magic8", 128, Live::Whole).unwrap();
    let sc_row = plan.v::<8>("sc_row", 136, Live::Whole).unwrap();
    let sc_quads = [hipfire_isa::V::<4>(sc_row.base()), hipfire_isa::V::<4>(sc_row.base() + 4)];
    let t = plan.v::<8>("t", 144, Live::Whole).unwrap();
    let d = std::array::from_fn(|i| plan.v::<1>("d_x", (152 + i) as u8, Live::Whole).unwrap());
    let spec = KernelSpec {
        kernel_id: "iu4_k1_fold_test".into(), variant: "cacc2=0".into(),
        arch: Arch::Gfx1201, symbol: "iu4_k1_fold_test".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2,
        workgroup_size: 256, group_segment_fixed_size: 0, wave32: true,
        system_sgpr_workgroup_id_y: false,
    };
    let mut b = Builder::new(spec, plan);
    emit_fold(&mut b, FoldRegisters { cacc, acc, magic, sc: [sc_quads; 2], t, d }).unwrap();
    let packets: Vec<_> = b.program.instructions.iter().map(|i| i.text.as_str()).collect();
    assert_eq!(packets.len(), 96);
    for op in ["v_dual_subrev_f32", "v_dual_mul_f32", "v_dual_fmac_f32"] {
        assert_eq!(packets.iter().filter(|p| p.starts_with(op)).count(), 32, "{op}");
    }
    let mut mc = Command::new("/opt/rocm/core-10.0/lib/llvm/bin/llvm-mc")
        .args(["-triple=amdgcn-amd-amdhsa", "-mcpu=gfx1201", "-show-encoding"])
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped())
        .spawn().expect("ROCm llvm-mc");
    {
        let mut stdin = mc.stdin.take().unwrap();
        for packet in packets { writeln!(stdin, "{packet}").unwrap(); }
    }
    let output = mc.wait_with_output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let encoded = String::from_utf8(output.stdout).unwrap();
    assert_eq!(encoded.matches("encoding:").count(), 96);
}
