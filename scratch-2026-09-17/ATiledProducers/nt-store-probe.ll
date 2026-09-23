; ModuleID = 'scratch-2026-09-17/ATiledProducers/nt-store-probe.hip'
source_filename = "scratch-2026-09-17/ATiledProducers/nt-store-probe.hip"
target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@__hip_cuid_a5dce4c69cb1a8d1 = addrspace(1) global i8 0
@llvm.compiler.used = appending addrspace(1) global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @__hip_cuid_a5dce4c69cb1a8d1 to ptr)], section "llvm.metadata"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define protected amdgpu_kernel void @nt_store_probe(ptr addrspace(1) noundef writeonly captures(none) %0, float noundef %1) local_unnamed_addr #0 {
  %3 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x()
  %4 = zext nneg i32 %3 to i64
  %5 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %4
  store float %1, ptr addrspace(1) %5, align 4, !tbaa !14, !nontemporal !16
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define protected amdgpu_kernel void @normal_store_probe(ptr addrspace(1) noundef writeonly captures(none) %0, float noundef %1) local_unnamed_addr #0 {
  %3 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x()
  %4 = zext nneg i32 %3 to i64
  %5 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %4
  store float %1, ptr addrspace(1) %5, align 4, !tbaa !14
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable "amdgpu-flat-work-group-size"="1,1024" "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-no-wwm" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1201" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}
!llvm.errno.tbaa = !{!5, !9}
!opencl.ocl.version = !{!13}

!0 = !{i32 1, !"amdhsa_code_object_version", i32 600}
!1 = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
!2 = !{i32 8, !"PIC Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = !{!10, !10, i64 0}
!10 = !{!"int", !11, i64 0}
!11 = !{!"omnipotent char", !12, i64 0}
!12 = !{!"Simple C/C++ TBAA"}
!13 = !{i32 2, i32 0}
!14 = !{!15, !15, i64 0}
!15 = !{!"float", !7, i64 0}
!16 = !{i32 1}
