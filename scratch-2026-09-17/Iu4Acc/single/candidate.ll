; ModuleID = 'iu4_combined-hip-amdgcn-amd-amdhsa-gfx1201.tmp.bc'
source_filename = "../iu4_combined.hip"
target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

%struct.__hip_builtin_blockIdx_t = type { i8 }
%struct.__hip_builtin_blockDim_t = type { i8 }
%struct.__hip_builtin_threadIdx_t = type { i8 }
%struct.anon = type { i8 }
%struct.HIP_vector_type = type { %struct.HIP_vector_base }
%struct.HIP_vector_base = type { float, float, float, float }
%struct.block_i4_128 = type { float, i32, [64 x i8] }
%struct.__half = type { %union.anon.0 }
%union.anon.0 = type { half }
%union.anon = type { i32 }
%struct.__half_raw = type { %union.anon.1 }
%union.anon.1 = type { half }

$_ZN24__hip_builtin_blockIdx_t7__get_yEv = comdat any

$_ZN24__hip_builtin_blockIdx_t7__get_xEv = comdat any

$_Z26quantize_block_i4_128_wavePKfP12block_i4_128i = comdat any

$_ZN15HIP_vector_typeIfLj4EEC2IJffffETnPN14__hip_internal9enable_ifIXaagtLj4ELi1EeqsZT_Lj4EEvE4typeELPv0EEEDpT_ = comdat any

$_ZN15HIP_vector_baseIfLj4EEC2Effff = comdat any

$_Z10__shfl_xorfii = comdat any

$_Z10__shfl_xoriii = comdat any

$_Z12__half2float6__half = comdat any

$_Z16__ushort_as_halft = comdat any

$_Z13__syncthreadsv = comdat any

$_ZNK6__halfcv10__half_rawEv = comdat any

$_ZN6__halfC2ERK10__half_raw = comdat any

@__const.__assert_fail.fmt = private unnamed_addr addrspace(4) constant [47 x i8] c"%s:%u: %s: Device-side assertion `%s' failed.\0A\00", align 16
@blockIdx = extern_weak protected addrspace(1) global %struct.__hip_builtin_blockIdx_t, align 1
@blockDim = extern_weak protected addrspace(1) global %struct.__hip_builtin_blockDim_t, align 1
@threadIdx = extern_weak protected addrspace(1) global %struct.__hip_builtin_threadIdx_t, align 1
@warpSize = internal addrspace(4) constant %struct.anon undef, align 1
@LDS = external hidden addrspace(3) global [0 x i8], align 1
@.str = private unnamed_addr addrspace(4) constant [10 x i8] c"workgroup\00", align 1
@.str.1 = private unnamed_addr addrspace(4) constant [7 x i8] c"global\00", align 1
@.str.2 = private unnamed_addr addrspace(4) constant [6 x i8] c"local\00", align 1
@__hip_cuid_18e58617ebcb2f7e = addrspace(1) global i8 0
@llvm.compiler.used = appending addrspace(1) global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @__hip_cuid_18e58617ebcb2f7e to ptr)], section "llvm.metadata"
@__oclc_ISA_version = internal local_unnamed_addr addrspace(4) constant i32 12001, align 4
@__oclc_ABI_version = internal local_unnamed_addr addrspace(4) constant i32 600, align 4

; Function Attrs: convergent mustprogress noreturn nounwind uwtable
define weak void @__cxa_pure_virtual() #0 !dbg !17 {
  call void @llvm.trap(), !dbg !21
  unreachable, !dbg !22
}

; Function Attrs: cold noreturn nounwind memory(inaccessiblemem: write)
declare void @llvm.trap() #1

; Function Attrs: convergent mustprogress noreturn nounwind uwtable
define weak void @__cxa_deleted_virtual() #0 !dbg !23 {
  call void @llvm.trap(), !dbg !24
  unreachable, !dbg !25
}

; Function Attrs: convergent mustprogress noinline nounwind uwtable
define weak hidden void @__assert_fail(ptr noundef %0, ptr noundef %1, i32 noundef %2, ptr noundef %3) #2 !dbg !26 {
  %5 = alloca ptr, align 8, addrspace(5)
  %6 = alloca ptr, align 8, addrspace(5)
  %7 = alloca i32, align 4, addrspace(5)
  %8 = alloca ptr, align 8, addrspace(5)
  %9 = alloca [47 x i8], align 16, addrspace(5)
  %10 = alloca i64, align 8, addrspace(5)
  %11 = alloca i32, align 4, addrspace(5)
  %12 = alloca ptr, align 8, addrspace(5)
  %13 = alloca ptr, align 8, addrspace(5)
  %14 = alloca ptr, align 8, addrspace(5)
  %15 = alloca ptr, align 8, addrspace(5)
  %16 = addrspacecast ptr addrspace(5) %5 to ptr
  %17 = addrspacecast ptr addrspace(5) %6 to ptr
  %18 = addrspacecast ptr addrspace(5) %7 to ptr
  %19 = addrspacecast ptr addrspace(5) %8 to ptr
  %20 = addrspacecast ptr addrspace(5) %9 to ptr
  %21 = addrspacecast ptr addrspace(5) %10 to ptr
  %22 = addrspacecast ptr addrspace(5) %11 to ptr
  %23 = addrspacecast ptr addrspace(5) %12 to ptr
  %24 = addrspacecast ptr addrspace(5) %13 to ptr
  %25 = addrspacecast ptr addrspace(5) %14 to ptr
  %26 = addrspacecast ptr addrspace(5) %15 to ptr
  store ptr %0, ptr %16, align 8, !tbaa !28
  store ptr %1, ptr %17, align 8, !tbaa !28
  store i32 %2, ptr %18, align 4, !tbaa !8
  store ptr %3, ptr %19, align 8, !tbaa !28
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %9) #25, !dbg !31
  call void @llvm.memcpy.p0.p4.i64(ptr align 16 %20, ptr addrspace(4) align 16 @__const.__assert_fail.fmt, i64 47, i1 false), !dbg !32
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %10) #25, !dbg !33
  %27 = call i64 @__ockl_fprintf_stderr_begin() #26, !dbg !34
  store i64 %27, ptr %21, align 8, !dbg !35, !tbaa !36
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %11) #25, !dbg !38
  store i32 0, ptr %22, align 4, !dbg !39, !tbaa !8
  br label %28, !dbg !40

28:                                               ; preds = %4
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %12) #25, !dbg !41
  %29 = getelementptr inbounds [47 x i8], ptr %20, i64 0, i64 0, !dbg !42
  store ptr %29, ptr %23, align 8, !dbg !43, !tbaa !28
  br label %30, !dbg !44

30:                                               ; preds = %35, %28
  %31 = load ptr, ptr %23, align 8, !dbg !45, !tbaa !28
  %32 = getelementptr inbounds nuw i8, ptr %31, i32 1, !dbg !45
  store ptr %32, ptr %23, align 8, !dbg !45, !tbaa !28
  %33 = load i8, ptr %31, align 1, !dbg !46, !tbaa !47
  %34 = icmp ne i8 %33, 0, !dbg !46
  br i1 %34, label %35, label %36, !dbg !44

35:                                               ; preds = %30
  br label %30, !dbg !44, !llvm.loop !48

36:                                               ; preds = %30
  %37 = load ptr, ptr %23, align 8, !dbg !51, !tbaa !28
  %38 = getelementptr inbounds [47 x i8], ptr %20, i64 0, i64 0, !dbg !52
  %39 = ptrtoint ptr %37 to i64, !dbg !53
  %40 = ptrtoint ptr %38 to i64, !dbg !53
  %41 = sub i64 %39, %40, !dbg !53
  %42 = trunc i64 %41 to i32, !dbg !51
  store i32 %42, ptr %22, align 4, !dbg !54, !tbaa !8
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %12) #25, !dbg !55
  br label %43, !dbg !55

43:                                               ; preds = %36
  br label %44, !dbg !55

44:                                               ; preds = %43
  %45 = load i64, ptr %21, align 8, !dbg !56, !tbaa !36
  %46 = getelementptr inbounds [47 x i8], ptr %20, i64 0, i64 0, !dbg !57
  %47 = load i32, ptr %22, align 4, !dbg !58, !tbaa !8
  %48 = sext i32 %47 to i64, !dbg !58
  %49 = call i64 @__ockl_fprintf_append_string_n(i64 noundef %45, ptr noundef %46, i64 noundef %48, i32 noundef 0) #26, !dbg !59
  store i64 %49, ptr %21, align 8, !dbg !60, !tbaa !36
  br label %50, !dbg !61

50:                                               ; preds = %44
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %13) #25, !dbg !62
  %51 = load ptr, ptr %17, align 8, !dbg !63, !tbaa !28
  store ptr %51, ptr %24, align 8, !dbg !64, !tbaa !28
  br label %52, !dbg !65

52:                                               ; preds = %57, %50
  %53 = load ptr, ptr %24, align 8, !dbg !66, !tbaa !28
  %54 = getelementptr inbounds nuw i8, ptr %53, i32 1, !dbg !66
  store ptr %54, ptr %24, align 8, !dbg !66, !tbaa !28
  %55 = load i8, ptr %53, align 1, !dbg !67, !tbaa !47
  %56 = icmp ne i8 %55, 0, !dbg !67
  br i1 %56, label %57, label %58, !dbg !65

57:                                               ; preds = %52
  br label %52, !dbg !65, !llvm.loop !68

58:                                               ; preds = %52
  %59 = load ptr, ptr %24, align 8, !dbg !70, !tbaa !28
  %60 = load ptr, ptr %17, align 8, !dbg !71, !tbaa !28
  %61 = ptrtoint ptr %59 to i64, !dbg !72
  %62 = ptrtoint ptr %60 to i64, !dbg !72
  %63 = sub i64 %61, %62, !dbg !72
  %64 = trunc i64 %63 to i32, !dbg !70
  store i32 %64, ptr %22, align 4, !dbg !73, !tbaa !8
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %13) #25, !dbg !74
  br label %65, !dbg !74

65:                                               ; preds = %58
  br label %66, !dbg !74

66:                                               ; preds = %65
  %67 = load i64, ptr %21, align 8, !dbg !75, !tbaa !36
  %68 = load ptr, ptr %17, align 8, !dbg !76, !tbaa !28
  %69 = load i32, ptr %22, align 4, !dbg !77, !tbaa !8
  %70 = sext i32 %69 to i64, !dbg !77
  %71 = call i64 @__ockl_fprintf_append_string_n(i64 noundef %67, ptr noundef %68, i64 noundef %70, i32 noundef 0) #26, !dbg !78
  store i64 %71, ptr %21, align 8, !dbg !79, !tbaa !36
  %72 = load i64, ptr %21, align 8, !dbg !80, !tbaa !36
  %73 = load i32, ptr %18, align 4, !dbg !81, !tbaa !8
  %74 = zext i32 %73 to i64, !dbg !81
  %75 = call i64 @__ockl_fprintf_append_args(i64 noundef %72, i32 noundef 1, i64 noundef %74, i64 noundef 0, i64 noundef 0, i64 noundef 0, i64 noundef 0, i64 noundef 0, i64 noundef 0, i32 noundef 0) #26, !dbg !82
  store i64 %75, ptr %21, align 8, !dbg !83, !tbaa !36
  br label %76, !dbg !84

76:                                               ; preds = %66
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %14) #25, !dbg !85
  %77 = load ptr, ptr %19, align 8, !dbg !86, !tbaa !28
  store ptr %77, ptr %25, align 8, !dbg !87, !tbaa !28
  br label %78, !dbg !88

78:                                               ; preds = %83, %76
  %79 = load ptr, ptr %25, align 8, !dbg !89, !tbaa !28
  %80 = getelementptr inbounds nuw i8, ptr %79, i32 1, !dbg !89
  store ptr %80, ptr %25, align 8, !dbg !89, !tbaa !28
  %81 = load i8, ptr %79, align 1, !dbg !90, !tbaa !47
  %82 = icmp ne i8 %81, 0, !dbg !90
  br i1 %82, label %83, label %84, !dbg !88

83:                                               ; preds = %78
  br label %78, !dbg !88, !llvm.loop !91

84:                                               ; preds = %78
  %85 = load ptr, ptr %25, align 8, !dbg !93, !tbaa !28
  %86 = load ptr, ptr %19, align 8, !dbg !94, !tbaa !28
  %87 = ptrtoint ptr %85 to i64, !dbg !95
  %88 = ptrtoint ptr %86 to i64, !dbg !95
  %89 = sub i64 %87, %88, !dbg !95
  %90 = trunc i64 %89 to i32, !dbg !93
  store i32 %90, ptr %22, align 4, !dbg !96, !tbaa !8
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %14) #25, !dbg !97
  br label %91, !dbg !97

91:                                               ; preds = %84
  br label %92, !dbg !97

92:                                               ; preds = %91
  %93 = load i64, ptr %21, align 8, !dbg !98, !tbaa !36
  %94 = load ptr, ptr %19, align 8, !dbg !99, !tbaa !28
  %95 = load i32, ptr %22, align 4, !dbg !100, !tbaa !8
  %96 = sext i32 %95 to i64, !dbg !100
  %97 = call i64 @__ockl_fprintf_append_string_n(i64 noundef %93, ptr noundef %94, i64 noundef %96, i32 noundef 0) #26, !dbg !101
  store i64 %97, ptr %21, align 8, !dbg !102, !tbaa !36
  br label %98, !dbg !103

98:                                               ; preds = %92
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %15) #25, !dbg !104
  %99 = load ptr, ptr %16, align 8, !dbg !105, !tbaa !28
  store ptr %99, ptr %26, align 8, !dbg !106, !tbaa !28
  br label %100, !dbg !107

100:                                              ; preds = %105, %98
  %101 = load ptr, ptr %26, align 8, !dbg !108, !tbaa !28
  %102 = getelementptr inbounds nuw i8, ptr %101, i32 1, !dbg !108
  store ptr %102, ptr %26, align 8, !dbg !108, !tbaa !28
  %103 = load i8, ptr %101, align 1, !dbg !109, !tbaa !47
  %104 = icmp ne i8 %103, 0, !dbg !109
  br i1 %104, label %105, label %106, !dbg !107

105:                                              ; preds = %100
  br label %100, !dbg !107, !llvm.loop !110

106:                                              ; preds = %100
  %107 = load ptr, ptr %26, align 8, !dbg !112, !tbaa !28
  %108 = load ptr, ptr %16, align 8, !dbg !113, !tbaa !28
  %109 = ptrtoint ptr %107 to i64, !dbg !114
  %110 = ptrtoint ptr %108 to i64, !dbg !114
  %111 = sub i64 %109, %110, !dbg !114
  %112 = trunc i64 %111 to i32, !dbg !112
  store i32 %112, ptr %22, align 4, !dbg !115, !tbaa !8
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %15) #25, !dbg !116
  br label %113, !dbg !116

113:                                              ; preds = %106
  br label %114, !dbg !116

114:                                              ; preds = %113
  %115 = load i64, ptr %21, align 8, !dbg !117, !tbaa !36
  %116 = load ptr, ptr %16, align 8, !dbg !118, !tbaa !28
  %117 = load i32, ptr %22, align 4, !dbg !119, !tbaa !8
  %118 = sext i32 %117 to i64, !dbg !119
  %119 = call i64 @__ockl_fprintf_append_string_n(i64 noundef %115, ptr noundef %116, i64 noundef %118, i32 noundef 1) #26, !dbg !120
  call void @llvm.trap(), !dbg !121
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %11) #25, !dbg !122
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %10) #25, !dbg !122
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %9) #25, !dbg !122
  ret void, !dbg !122
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p5(ptr addrspace(5) captures(none)) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p4.i64(ptr noalias writeonly captures(none), ptr addrspace(4) noalias readonly captures(none), i64, i1 immarg) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p5(ptr addrspace(5) captures(none)) #3

; Function Attrs: convergent mustprogress noinline nounwind uwtable
define weak hidden void @__assertfail() #2 !dbg !123 {
  call void @llvm.trap(), !dbg !124
  ret void, !dbg !125
}

; Function Attrs: convergent mustprogress norecurse nounwind uwtable
define protected amdgpu_kernel void @quantize_int4_mmq_ds128(ptr addrspace(1) noalias noundef %0, ptr addrspace(1) noalias noundef %1, i32 noundef %2, i32 noundef %3) #4 !dbg !126 {
  %5 = alloca ptr, align 8, addrspace(5)
  %6 = alloca ptr, align 8, addrspace(5)
  %7 = alloca ptr, align 8, addrspace(5)
  %8 = alloca ptr, align 8, addrspace(5)
  %9 = alloca i32, align 4, addrspace(5)
  %10 = alloca i32, align 4, addrspace(5)
  %11 = alloca i32, align 4, addrspace(5)
  %12 = alloca i32, align 4, addrspace(5)
  %13 = alloca i32, align 4, addrspace(5)
  %14 = alloca i32, align 4, addrspace(5)
  %15 = alloca i32, align 4, addrspace(5)
  %16 = alloca %struct.HIP_vector_type, align 16, addrspace(5)
  %17 = alloca [4 x float], align 16, addrspace(5)
  %18 = alloca i32, align 4, addrspace(5)
  %19 = alloca ptr, align 8, addrspace(5)
  %20 = addrspacecast ptr addrspace(5) %7 to ptr
  %21 = addrspacecast ptr addrspace(5) %8 to ptr
  %22 = addrspacecast ptr addrspace(5) %9 to ptr
  %23 = addrspacecast ptr addrspace(5) %10 to ptr
  %24 = addrspacecast ptr addrspace(5) %11 to ptr
  %25 = addrspacecast ptr addrspace(5) %12 to ptr
  %26 = addrspacecast ptr addrspace(5) %13 to ptr
  %27 = addrspacecast ptr addrspace(5) %14 to ptr
  %28 = addrspacecast ptr addrspace(5) %15 to ptr
  %29 = addrspacecast ptr addrspace(5) %16 to ptr
  %30 = addrspacecast ptr addrspace(5) %17 to ptr
  %31 = addrspacecast ptr addrspace(5) %18 to ptr
  %32 = addrspacecast ptr addrspace(5) %19 to ptr
  store ptr addrspace(1) %0, ptr addrspace(5) %5, align 8
  %33 = load ptr, ptr addrspace(5) %5, align 8, !tbaa !128
  store ptr addrspace(1) %1, ptr addrspace(5) %6, align 8
  %34 = load ptr, ptr addrspace(5) %6, align 8, !tbaa !130
  store ptr %33, ptr %20, align 8, !tbaa !128
  store ptr %34, ptr %21, align 8, !tbaa !130
  store i32 %2, ptr %22, align 4, !tbaa !8
  store i32 %3, ptr %23, align 4, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %11) #25, !dbg !132
  %35 = call noundef i32 @_ZN24__hip_builtin_blockIdx_t7__get_yEv() #26, !dbg !133
  store i32 %35, ptr %24, align 4, !dbg !134, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %12) #25, !dbg !135
  %36 = call noundef i32 @_ZN24__hip_builtin_blockIdx_t7__get_xEv() #26, !dbg !136
  %37 = call noundef i32 @_ZN24__hip_builtin_blockDim_t7__get_xEv() #26, !dbg !137
  %38 = mul i32 %36, %37, !dbg !138
  %39 = call noundef i32 @_ZN25__hip_builtin_threadIdx_t7__get_xEv() #26, !dbg !139
  %40 = add i32 %38, %39, !dbg !140
  store i32 %40, ptr %25, align 4, !dbg !141, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %13) #25, !dbg !142
  %41 = load i32, ptr %25, align 4, !dbg !143, !tbaa !8
  %42 = mul nsw i32 %41, 4, !dbg !144
  store i32 %42, ptr %26, align 4, !dbg !145, !tbaa !8
  %43 = load i32, ptr %24, align 4, !dbg !146, !tbaa !8
  %44 = load i32, ptr %23, align 4, !dbg !147, !tbaa !8
  %45 = icmp sge i32 %43, %44, !dbg !148
  br i1 %45, label %50, label %46, !dbg !149

46:                                               ; preds = %4
  %47 = load i32, ptr %26, align 4, !dbg !150, !tbaa !8
  %48 = load i32, ptr %22, align 4, !dbg !151, !tbaa !8
  %49 = icmp sge i32 %47, %48, !dbg !152
  br i1 %49, label %50, label %51, !dbg !149

50:                                               ; preds = %46, %4
  store i32 1, ptr %27, align 4
  br label %101, !dbg !153

51:                                               ; preds = %46
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %15) #25, !dbg !154
  %52 = call noundef i32 @_ZN25__hip_builtin_threadIdx_t7__get_xEv() #26, !dbg !155
  %53 = and i32 %52, 31, !dbg !156
  store i32 %53, ptr %28, align 4, !dbg !157, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %16) #25, !dbg !158
  %54 = load i32, ptr %26, align 4, !dbg !159, !tbaa !8
  %55 = add nsw i32 %54, 3, !dbg !160
  %56 = load i32, ptr %22, align 4, !dbg !161, !tbaa !8
  %57 = icmp slt i32 %55, %56, !dbg !162
  br i1 %57, label %58, label %69, !dbg !163

58:                                               ; preds = %51
  %59 = load ptr, ptr %20, align 8, !dbg !164, !tbaa !128
  %60 = load i32, ptr %24, align 4, !dbg !165, !tbaa !8
  %61 = sext i32 %60 to i64, !dbg !165
  %62 = load i32, ptr %22, align 4, !dbg !166, !tbaa !8
  %63 = sext i32 %62 to i64, !dbg !166
  %64 = mul nsw i64 %61, %63, !dbg !167
  %65 = getelementptr inbounds float, ptr %59, i64 %64, !dbg !168
  %66 = load i32, ptr %26, align 4, !dbg !169, !tbaa !8
  %67 = sext i32 %66 to i64, !dbg !168
  %68 = getelementptr inbounds float, ptr %65, i64 %67, !dbg !168
  call void @llvm.memcpy.p0.p0.i64(ptr align 16 %29, ptr align 16 %68, i64 16, i1 false), !dbg !168
  br label %74, !dbg !163

69:                                               ; preds = %51
  %70 = addrspacecast ptr %29 to ptr addrspace(5), !dbg !170
  %71 = call %struct.HIP_vector_type @_ZL11make_float4ffff(float noundef 0.000000e+00, float noundef 0.000000e+00, float noundef 0.000000e+00, float noundef 0.000000e+00) #26, !dbg !170
  %72 = getelementptr inbounds nuw %struct.HIP_vector_type, ptr addrspace(5) %70, i32 0, i32 0, !dbg !170
  %73 = extractvalue %struct.HIP_vector_type %71, 0, !dbg !170
  store %struct.HIP_vector_base %73, ptr addrspace(5) %72, align 16, !dbg !170
  br label %74, !dbg !163

74:                                               ; preds = %69, %58
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %17) #25, !dbg !171
  %75 = getelementptr inbounds nuw %struct.HIP_vector_base, ptr %29, i32 0, i32 0, !dbg !172
  %76 = load float, ptr %75, align 16, !dbg !172, !tbaa !173
  store float %76, ptr %30, align 4, !dbg !176, !tbaa !177
  %77 = getelementptr inbounds float, ptr %30, i64 1, !dbg !176
  %78 = getelementptr inbounds nuw %struct.HIP_vector_base, ptr %29, i32 0, i32 1, !dbg !178
  %79 = load float, ptr %78, align 4, !dbg !178, !tbaa !179
  store float %79, ptr %77, align 4, !dbg !176, !tbaa !177
  %80 = getelementptr inbounds float, ptr %30, i64 2, !dbg !176
  %81 = getelementptr inbounds nuw %struct.HIP_vector_base, ptr %29, i32 0, i32 2, !dbg !180
  %82 = load float, ptr %81, align 8, !dbg !180, !tbaa !181
  store float %82, ptr %80, align 4, !dbg !176, !tbaa !177
  %83 = getelementptr inbounds float, ptr %30, i64 3, !dbg !176
  %84 = getelementptr inbounds nuw %struct.HIP_vector_base, ptr %29, i32 0, i32 3, !dbg !182
  %85 = load float, ptr %84, align 4, !dbg !182, !tbaa !183
  store float %85, ptr %83, align 4, !dbg !176, !tbaa !177
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %18) #25, !dbg !184
  %86 = load i32, ptr %26, align 4, !dbg !185, !tbaa !8
  %87 = sdiv i32 %86, 128, !dbg !186
  store i32 %87, ptr %31, align 4, !dbg !187, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %19) #25, !dbg !188
  %88 = load ptr, ptr %21, align 8, !dbg !189, !tbaa !130
  %89 = load i32, ptr %31, align 4, !dbg !190, !tbaa !8
  %90 = sext i32 %89 to i64, !dbg !190
  %91 = load i32, ptr %23, align 4, !dbg !191, !tbaa !8
  %92 = sext i32 %91 to i64, !dbg !191
  %93 = mul nsw i64 %90, %92, !dbg !192
  %94 = getelementptr inbounds %struct.block_i4_128, ptr %88, i64 %93, !dbg !193
  %95 = load i32, ptr %24, align 4, !dbg !194, !tbaa !8
  %96 = sext i32 %95 to i64, !dbg !195
  %97 = getelementptr inbounds %struct.block_i4_128, ptr %94, i64 %96, !dbg !195
  store ptr %97, ptr %32, align 8, !dbg !196, !tbaa !130
  %98 = getelementptr inbounds [4 x float], ptr %30, i64 0, i64 0, !dbg !197
  %99 = load ptr, ptr %32, align 8, !dbg !198, !tbaa !130
  %100 = load i32, ptr %28, align 4, !dbg !199, !tbaa !8
  call void @_Z26quantize_block_i4_128_wavePKfP12block_i4_128i(ptr noundef %98, ptr noundef %99, i32 noundef %100) #26, !dbg !200
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %19) #25, !dbg !201
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %18) #25, !dbg !201
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %17) #25, !dbg !201
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %16) #25, !dbg !201
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %15) #25, !dbg !201
  store i32 0, ptr %27, align 4, !dbg !201
  br label %101, !dbg !201

101:                                              ; preds = %74, %50
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %13) #25, !dbg !201
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %12) #25, !dbg !201
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %11) #25, !dbg !201
  %102 = load i32, ptr %27, align 4
  switch i32 %102, label %104 [
    i32 0, label %103
    i32 1, label %103
  ]

103:                                              ; preds = %101, %101
  ret void, !dbg !201

104:                                              ; preds = %101
  unreachable
}

; Function Attrs: alwaysinline convergent mustprogress nounwind uwtable
define linkonce_odr hidden noundef i32 @_ZN24__hip_builtin_blockIdx_t7__get_yEv() #5 comdat align 2 !dbg !202 {
  %1 = call noundef i32 @_ZL21__hip_get_block_idx_yv() #26, !dbg !204
  ret i32 %1, !dbg !205
}

; Function Attrs: alwaysinline convergent mustprogress nounwind uwtable
define linkonce_odr hidden noundef i32 @_ZN24__hip_builtin_blockIdx_t7__get_xEv() #5 comdat align 2 !dbg !206 {
  %1 = call noundef i32 @_ZL21__hip_get_block_idx_xv() #26, !dbg !207
  ret i32 %1, !dbg !208
}

; Function Attrs: alwaysinline convergent mustprogress nounwind uwtable
define internal noundef i32 @_ZN24__hip_builtin_blockDim_t7__get_xEv() #5 align 2 !dbg !209 {
  %1 = call noundef i32 @_ZL21__hip_get_block_dim_xv() #26, !dbg !210
  ret i32 %1, !dbg !211
}

; Function Attrs: alwaysinline convergent mustprogress nounwind uwtable
define internal noundef i32 @_ZN25__hip_builtin_threadIdx_t7__get_xEv() #5 align 2 !dbg !212 {
  %1 = call noundef i32 @_ZL22__hip_get_thread_idx_xv() #26, !dbg !213
  ret i32 %1, !dbg !214
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #3

; Function Attrs: convergent inlinehint mustprogress nounwind uwtable
define internal %struct.HIP_vector_type @_ZL11make_float4ffff(float noundef %0, float noundef %1, float noundef %2, float noundef %3) #6 !dbg !215 {
  %5 = alloca %struct.HIP_vector_type, align 16, addrspace(5)
  %6 = alloca float, align 4, addrspace(5)
  %7 = alloca float, align 4, addrspace(5)
  %8 = alloca float, align 4, addrspace(5)
  %9 = alloca float, align 4, addrspace(5)
  %10 = addrspacecast ptr addrspace(5) %6 to ptr
  %11 = addrspacecast ptr addrspace(5) %7 to ptr
  %12 = addrspacecast ptr addrspace(5) %8 to ptr
  %13 = addrspacecast ptr addrspace(5) %9 to ptr
  %14 = addrspacecast ptr addrspace(5) %5 to ptr
  store float %0, ptr %10, align 4, !tbaa !177
  store float %1, ptr %11, align 4, !tbaa !177
  store float %2, ptr %12, align 4, !tbaa !177
  store float %3, ptr %13, align 4, !tbaa !177
  %15 = load float, ptr %10, align 4, !dbg !217, !tbaa !177
  %16 = load float, ptr %11, align 4, !dbg !218, !tbaa !177
  %17 = load float, ptr %12, align 4, !dbg !219, !tbaa !177
  %18 = load float, ptr %13, align 4, !dbg !220, !tbaa !177
  call void @_ZN15HIP_vector_typeIfLj4EEC2IJffffETnPN14__hip_internal9enable_ifIXaagtLj4ELi1EeqsZT_Lj4EEvE4typeELPv0EEEDpT_(ptr noundef nonnull align 16 dereferenceable(16) %14, float noundef %15, float noundef %16, float noundef %17, float noundef %18) #26, !dbg !221
  %19 = load %struct.HIP_vector_type, ptr addrspace(5) %5, align 16, !dbg !222
  ret %struct.HIP_vector_type %19, !dbg !222
}

; Function Attrs: alwaysinline convergent mustprogress nounwind uwtable
define linkonce_odr hidden void @_Z26quantize_block_i4_128_wavePKfP12block_i4_128i(ptr noundef %0, ptr noalias noundef %1, i32 noundef %2) #5 comdat !dbg !223 {
  %4 = alloca ptr, align 8, addrspace(5)
  %5 = alloca ptr, align 8, addrspace(5)
  %6 = alloca i32, align 4, addrspace(5)
  %7 = alloca float, align 4, addrspace(5)
  %8 = alloca i32, align 4, addrspace(5)
  %9 = alloca i32, align 4, addrspace(5)
  %10 = alloca float, align 4, addrspace(5)
  %11 = alloca float, align 4, addrspace(5)
  %12 = alloca i32, align 4, addrspace(5)
  %13 = alloca i32, align 4, addrspace(5)
  %14 = alloca float, align 4, addrspace(5)
  %15 = alloca float, align 4, addrspace(5)
  %16 = alloca i32, align 4, addrspace(5)
  %17 = alloca float, align 4, addrspace(5)
  %18 = alloca float, align 4, addrspace(5)
  %19 = alloca i32, align 4, addrspace(5)
  %20 = alloca [4 x i32], align 16, addrspace(5)
  %21 = alloca i32, align 4, addrspace(5)
  %22 = alloca float, align 4, addrspace(5)
  %23 = alloca i32, align 4, addrspace(5)
  %24 = alloca i32, align 4, addrspace(5)
  %25 = alloca ptr, align 8, addrspace(5)
  %26 = addrspacecast ptr addrspace(5) %4 to ptr
  %27 = addrspacecast ptr addrspace(5) %5 to ptr
  %28 = addrspacecast ptr addrspace(5) %6 to ptr
  %29 = addrspacecast ptr addrspace(5) %7 to ptr
  %30 = addrspacecast ptr addrspace(5) %8 to ptr
  %31 = addrspacecast ptr addrspace(5) %10 to ptr
  %32 = addrspacecast ptr addrspace(5) %11 to ptr
  %33 = addrspacecast ptr addrspace(5) %12 to ptr
  %34 = addrspacecast ptr addrspace(5) %13 to ptr
  %35 = addrspacecast ptr addrspace(5) %14 to ptr
  %36 = addrspacecast ptr addrspace(5) %15 to ptr
  %37 = addrspacecast ptr addrspace(5) %16 to ptr
  %38 = addrspacecast ptr addrspace(5) %17 to ptr
  %39 = addrspacecast ptr addrspace(5) %18 to ptr
  %40 = addrspacecast ptr addrspace(5) %19 to ptr
  %41 = addrspacecast ptr addrspace(5) %20 to ptr
  %42 = addrspacecast ptr addrspace(5) %21 to ptr
  %43 = addrspacecast ptr addrspace(5) %22 to ptr
  %44 = addrspacecast ptr addrspace(5) %23 to ptr
  %45 = addrspacecast ptr addrspace(5) %24 to ptr
  %46 = addrspacecast ptr addrspace(5) %25 to ptr
  store ptr %0, ptr %26, align 8, !tbaa !128
  store ptr %1, ptr %27, align 8, !tbaa !130
  store i32 %2, ptr %28, align 4, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %7) #25, !dbg !224
  %47 = load ptr, ptr %26, align 8, !dbg !225, !tbaa !128
  %48 = getelementptr inbounds float, ptr %47, i64 0, !dbg !225
  %49 = load float, ptr %48, align 4, !dbg !225, !tbaa !177
  %50 = call contract noundef float @_ZL5fabsff(float noundef %49) #26, !dbg !226
  %51 = load ptr, ptr %26, align 8, !dbg !227, !tbaa !128
  %52 = getelementptr inbounds float, ptr %51, i64 1, !dbg !227
  %53 = load float, ptr %52, align 4, !dbg !227, !tbaa !177
  %54 = call contract noundef float @_ZL5fabsff(float noundef %53) #26, !dbg !228
  %55 = call contract noundef float @_ZL5fmaxfff(float noundef %50, float noundef %54) #26, !dbg !229
  %56 = load ptr, ptr %26, align 8, !dbg !230, !tbaa !128
  %57 = getelementptr inbounds float, ptr %56, i64 2, !dbg !230
  %58 = load float, ptr %57, align 4, !dbg !230, !tbaa !177
  %59 = call contract noundef float @_ZL5fabsff(float noundef %58) #26, !dbg !231
  %60 = load ptr, ptr %26, align 8, !dbg !232, !tbaa !128
  %61 = getelementptr inbounds float, ptr %60, i64 3, !dbg !232
  %62 = load float, ptr %61, align 4, !dbg !232, !tbaa !177
  %63 = call contract noundef float @_ZL5fabsff(float noundef %62) #26, !dbg !233
  %64 = call contract noundef float @_ZL5fmaxfff(float noundef %59, float noundef %63) #26, !dbg !234
  %65 = call contract noundef float @_ZL5fmaxfff(float noundef %55, float noundef %64) #26, !dbg !235
  store float %65, ptr %29, align 4, !dbg !236, !tbaa !177
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %8) #25, !dbg !237
  store i32 16, ptr %30, align 4, !dbg !238, !tbaa !8
  br label %66, !dbg !237

66:                                               ; preds = %77, %3
  %67 = load i32, ptr %30, align 4, !dbg !239, !tbaa !8
  %68 = icmp sgt i32 %67, 0, !dbg !240
  br i1 %68, label %70, label %69, !dbg !241

69:                                               ; preds = %66
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %8) #25, !dbg !241
  br label %80

70:                                               ; preds = %66
  %71 = load float, ptr %29, align 4, !dbg !242, !tbaa !177
  %72 = load float, ptr %29, align 4, !dbg !243, !tbaa !177
  %73 = load i32, ptr %30, align 4, !dbg !244, !tbaa !8
  %74 = freeze float %72, !dbg !245
  %75 = call contract noundef float @_Z10__shfl_xorfii(float noundef %74, i32 noundef %73, i32 noundef 32) #26, !dbg !245
  %76 = call contract noundef float @_ZL5fmaxfff(float noundef %71, float noundef %75) #26, !dbg !246
  store float %76, ptr %29, align 4, !dbg !247, !tbaa !177
  br label %77, !dbg !248

77:                                               ; preds = %70
  %78 = load i32, ptr %30, align 4, !dbg !249, !tbaa !8
  %79 = ashr i32 %78, 1, !dbg !249
  store i32 %79, ptr %30, align 4, !dbg !249, !tbaa !8
  br label %66, !dbg !241, !llvm.loop !250

80:                                               ; preds = %69
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %10) #25, !dbg !251
  store float 1.000000e+00, ptr %31, align 4, !dbg !252, !tbaa !177
  %81 = load float, ptr %29, align 4, !dbg !253, !tbaa !177
  %82 = fcmp contract une float %81, 0.000000e+00, !dbg !254
  br i1 %82, label %83, label %156, !dbg !254

83:                                               ; preds = %80
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %11) #25, !dbg !255
  store float 1.000000e+30, ptr %32, align 4, !dbg !256, !tbaa !177
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %12) #25, !dbg !257
  store i32 0, ptr %33, align 4, !dbg !258, !tbaa !8
  br label %84, !dbg !257

84:                                               ; preds = %152, %83
  %85 = load i32, ptr %33, align 4, !dbg !259, !tbaa !8
  %86 = icmp slt i32 %85, 8, !dbg !260
  br i1 %86, label %88, label %87, !dbg !261

87:                                               ; preds = %84
  store i32 5, ptr %34, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %12) #25, !dbg !261
  br label %155

88:                                               ; preds = %84
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %14) #25, !dbg !262
  %89 = load float, ptr %29, align 4, !dbg !263, !tbaa !177
  %90 = fdiv contract float %89, 7.000000e+00, !dbg !264
  %91 = fmul contract float %90, 5.000000e-01, !dbg !265
  %92 = load i32, ptr %33, align 4, !dbg !266, !tbaa !8
  %93 = sitofp i32 %92 to float, !dbg !266
  %94 = fdiv contract float %93, 7.000000e+00, !dbg !267
  %95 = fadd contract float 1.000000e+00, %94, !dbg !268
  %96 = fmul contract float %91, %95, !dbg !269
  store float %96, ptr %35, align 4, !dbg !270, !tbaa !177
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %15) #25, !dbg !271
  store float 0.000000e+00, ptr %36, align 4, !dbg !272, !tbaa !177
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %16) #25, !dbg !273
  store i32 0, ptr %37, align 4, !dbg !274, !tbaa !8
  br label %97, !dbg !273

97:                                               ; preds = %126, %88
  %98 = load i32, ptr %37, align 4, !dbg !275, !tbaa !8
  %99 = icmp slt i32 %98, 4, !dbg !276
  br i1 %99, label %101, label %100, !dbg !277

100:                                              ; preds = %97
  store i32 8, ptr %34, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %16) #25, !dbg !277
  br label %129

101:                                              ; preds = %97
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %17) #25, !dbg !278
  %102 = load ptr, ptr %26, align 8, !dbg !279, !tbaa !128
  %103 = load i32, ptr %37, align 4, !dbg !280, !tbaa !8
  %104 = sext i32 %103 to i64, !dbg !279
  %105 = getelementptr inbounds float, ptr %102, i64 %104, !dbg !279
  %106 = load float, ptr %105, align 4, !dbg !279, !tbaa !177
  %107 = load float, ptr %35, align 4, !dbg !281, !tbaa !177
  %108 = fdiv contract float %106, %107, !dbg !282
  %109 = call contract noundef float @_ZL5rintff(float noundef %108) #26, !dbg !283
  store float %109, ptr %38, align 4, !dbg !284, !tbaa !177
  %110 = load float, ptr %38, align 4, !dbg !285, !tbaa !177
  %111 = call contract noundef float @_ZL5fmaxfff(float noundef %110, float noundef -8.000000e+00) #26, !dbg !286
  %112 = call contract noundef float @_ZL5fminfff(float noundef %111, float noundef 7.000000e+00) #26, !dbg !287
  store float %112, ptr %38, align 4, !dbg !288, !tbaa !177
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %18) #25, !dbg !289
  %113 = load float, ptr %38, align 4, !dbg !290, !tbaa !177
  %114 = fneg contract float %113, !dbg !291
  %115 = load float, ptr %35, align 4, !dbg !292, !tbaa !177
  %116 = load ptr, ptr %26, align 8, !dbg !293, !tbaa !128
  %117 = load i32, ptr %37, align 4, !dbg !294, !tbaa !8
  %118 = sext i32 %117 to i64, !dbg !293
  %119 = getelementptr inbounds float, ptr %116, i64 %118, !dbg !293
  %120 = load float, ptr %119, align 4, !dbg !293, !tbaa !177
  %121 = call contract noundef float @_ZL9__fmaf_rnfff(float noundef %114, float noundef %115, float noundef %120) #26, !dbg !295
  store float %121, ptr %39, align 4, !dbg !296, !tbaa !177
  %122 = load float, ptr %39, align 4, !dbg !297, !tbaa !177
  %123 = load float, ptr %39, align 4, !dbg !298, !tbaa !177
  %124 = load float, ptr %36, align 4, !dbg !299, !tbaa !177
  %125 = call contract noundef float @_ZL9__fmaf_rnfff(float noundef %122, float noundef %123, float noundef %124) #26, !dbg !300
  store float %125, ptr %36, align 4, !dbg !301, !tbaa !177
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %18) #25, !dbg !302
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %17) #25, !dbg !302
  br label %126, !dbg !302

126:                                              ; preds = %101
  %127 = load i32, ptr %37, align 4, !dbg !303, !tbaa !8
  %128 = add nsw i32 %127, 1, !dbg !303
  store i32 %128, ptr %37, align 4, !dbg !303, !tbaa !8
  br label %97, !dbg !277, !llvm.loop !304

129:                                              ; preds = %100
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %19) #25, !dbg !306
  store i32 16, ptr %40, align 4, !dbg !307, !tbaa !8
  br label %130, !dbg !306

130:                                              ; preds = %141, %129
  %131 = load i32, ptr %40, align 4, !dbg !308, !tbaa !8
  %132 = icmp sgt i32 %131, 0, !dbg !309
  br i1 %132, label %134, label %133, !dbg !310

133:                                              ; preds = %130
  store i32 11, ptr %34, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %19) #25, !dbg !310
  br label %144

134:                                              ; preds = %130
  %135 = load float, ptr %36, align 4, !dbg !311, !tbaa !177
  %136 = load i32, ptr %40, align 4, !dbg !312, !tbaa !8
  %137 = freeze float %135, !dbg !313
  %138 = call contract noundef float @_Z10__shfl_xorfii(float noundef %137, i32 noundef %136, i32 noundef 32) #26, !dbg !313
  %139 = load float, ptr %36, align 4, !dbg !314, !tbaa !177
  %140 = fadd contract float %139, %138, !dbg !314
  store float %140, ptr %36, align 4, !dbg !314, !tbaa !177
  br label %141, !dbg !315

141:                                              ; preds = %134
  %142 = load i32, ptr %40, align 4, !dbg !316, !tbaa !8
  %143 = ashr i32 %142, 1, !dbg !316
  store i32 %143, ptr %40, align 4, !dbg !316, !tbaa !8
  br label %130, !dbg !310, !llvm.loop !317

144:                                              ; preds = %133
  %145 = load float, ptr %36, align 4, !dbg !318, !tbaa !177
  %146 = load float, ptr %32, align 4, !dbg !319, !tbaa !177
  %147 = fcmp contract olt float %145, %146, !dbg !320
  br i1 %147, label %148, label %151, !dbg !320

148:                                              ; preds = %144
  %149 = load float, ptr %36, align 4, !dbg !321, !tbaa !177
  store float %149, ptr %32, align 4, !dbg !322, !tbaa !177
  %150 = load float, ptr %35, align 4, !dbg !323, !tbaa !177
  store float %150, ptr %31, align 4, !dbg !324, !tbaa !177
  br label %151, !dbg !325

151:                                              ; preds = %148, %144
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %15) #25, !dbg !326
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %14) #25, !dbg !326
  br label %152, !dbg !326

152:                                              ; preds = %151
  %153 = load i32, ptr %33, align 4, !dbg !327, !tbaa !8
  %154 = add nsw i32 %153, 1, !dbg !327
  store i32 %154, ptr %33, align 4, !dbg !327, !tbaa !8
  br label %84, !dbg !261, !llvm.loop !328

155:                                              ; preds = %87
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %11) #25, !dbg !329
  br label %156, !dbg !329

156:                                              ; preds = %155, %80
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %20) #25, !dbg !330
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %21) #25, !dbg !331
  store i32 0, ptr %42, align 4, !dbg !332, !tbaa !8
  br label %157, !dbg !331

157:                                              ; preds = %184, %156
  %158 = load i32, ptr %42, align 4, !dbg !333, !tbaa !8
  %159 = icmp slt i32 %158, 4, !dbg !334
  br i1 %159, label %161, label %160, !dbg !335

160:                                              ; preds = %157
  store i32 14, ptr %34, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %21) #25, !dbg !335
  br label %187

161:                                              ; preds = %157
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %22) #25, !dbg !336
  %162 = load float, ptr %29, align 4, !dbg !337, !tbaa !177
  %163 = fcmp contract oeq float %162, 0.000000e+00, !dbg !338
  br i1 %163, label %164, label %165, !dbg !339

164:                                              ; preds = %161
  br label %174, !dbg !339

165:                                              ; preds = %161
  %166 = load ptr, ptr %26, align 8, !dbg !340, !tbaa !128
  %167 = load i32, ptr %42, align 4, !dbg !341, !tbaa !8
  %168 = sext i32 %167 to i64, !dbg !340
  %169 = getelementptr inbounds float, ptr %166, i64 %168, !dbg !340
  %170 = load float, ptr %169, align 4, !dbg !340, !tbaa !177
  %171 = load float, ptr %31, align 4, !dbg !342, !tbaa !177
  %172 = fdiv contract float %170, %171, !dbg !343
  %173 = call contract noundef float @_ZL5rintff(float noundef %172) #26, !dbg !344
  br label %174, !dbg !339

174:                                              ; preds = %165, %164
  %175 = phi contract float [ 0.000000e+00, %164 ], [ %173, %165 ], !dbg !339
  store float %175, ptr %43, align 4, !dbg !345, !tbaa !177
  %176 = load float, ptr %43, align 4, !dbg !346, !tbaa !177
  %177 = call contract noundef float @_ZL5fmaxfff(float noundef %176, float noundef -8.000000e+00) #26, !dbg !347
  %178 = call contract noundef float @_ZL5fminfff(float noundef %177, float noundef 7.000000e+00) #26, !dbg !348
  store float %178, ptr %43, align 4, !dbg !349, !tbaa !177
  %179 = load float, ptr %43, align 4, !dbg !350, !tbaa !177
  %180 = fptosi float %179 to i32, !dbg !350
  %181 = load i32, ptr %42, align 4, !dbg !351, !tbaa !8
  %182 = sext i32 %181 to i64, !dbg !352
  %183 = getelementptr inbounds [4 x i32], ptr %41, i64 0, i64 %182, !dbg !352
  store i32 %180, ptr %183, align 4, !dbg !353, !tbaa !8
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %22) #25, !dbg !354
  br label %184, !dbg !354

184:                                              ; preds = %174
  %185 = load i32, ptr %42, align 4, !dbg !355, !tbaa !8
  %186 = add nsw i32 %185, 1, !dbg !355
  store i32 %186, ptr %42, align 4, !dbg !355, !tbaa !8
  br label %157, !dbg !335, !llvm.loop !356

187:                                              ; preds = %160
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %23) #25, !dbg !357
  %188 = getelementptr inbounds [4 x i32], ptr %41, i64 0, i64 0, !dbg !358
  %189 = load i32, ptr %188, align 16, !dbg !358, !tbaa !8
  %190 = getelementptr inbounds [4 x i32], ptr %41, i64 0, i64 1, !dbg !359
  %191 = load i32, ptr %190, align 4, !dbg !359, !tbaa !8
  %192 = add nsw i32 %189, %191, !dbg !360
  %193 = getelementptr inbounds [4 x i32], ptr %41, i64 0, i64 2, !dbg !361
  %194 = load i32, ptr %193, align 8, !dbg !361, !tbaa !8
  %195 = add nsw i32 %192, %194, !dbg !362
  %196 = getelementptr inbounds [4 x i32], ptr %41, i64 0, i64 3, !dbg !363
  %197 = load i32, ptr %196, align 4, !dbg !363, !tbaa !8
  %198 = add nsw i32 %195, %197, !dbg !364
  store i32 %198, ptr %44, align 4, !dbg !365, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %24) #25, !dbg !366
  store i32 16, ptr %45, align 4, !dbg !367, !tbaa !8
  br label %199, !dbg !366

199:                                              ; preds = %210, %187
  %200 = load i32, ptr %45, align 4, !dbg !368, !tbaa !8
  %201 = icmp sgt i32 %200, 0, !dbg !369
  br i1 %201, label %203, label %202, !dbg !370

202:                                              ; preds = %199
  store i32 17, ptr %34, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %24) #25, !dbg !370
  br label %213

203:                                              ; preds = %199
  %204 = load i32, ptr %44, align 4, !dbg !371, !tbaa !8
  %205 = load i32, ptr %45, align 4, !dbg !372, !tbaa !8
  %206 = freeze i32 %204, !dbg !373
  %207 = call noundef i32 @_Z10__shfl_xoriii(i32 noundef %206, i32 noundef %205, i32 noundef 32) #26, !dbg !373
  %208 = load i32, ptr %44, align 4, !dbg !374, !tbaa !8
  %209 = add nsw i32 %208, %207, !dbg !374
  store i32 %209, ptr %44, align 4, !dbg !374, !tbaa !8
  br label %210, !dbg !375

210:                                              ; preds = %203
  %211 = load i32, ptr %45, align 4, !dbg !376, !tbaa !8
  %212 = ashr i32 %211, 1, !dbg !376
  store i32 %212, ptr %45, align 4, !dbg !376, !tbaa !8
  br label %199, !dbg !370, !llvm.loop !377

213:                                              ; preds = %202
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %25) #25, !dbg !378
  %214 = load ptr, ptr %27, align 8, !dbg !379, !tbaa !130
  %215 = getelementptr inbounds nuw %struct.block_i4_128, ptr %214, i32 0, i32 2, !dbg !380
  %216 = getelementptr inbounds [64 x i8], ptr %215, i64 0, i64 0, !dbg !379
  store ptr %216, ptr %46, align 8, !dbg !381, !tbaa !28
  %217 = getelementptr inbounds [4 x i32], ptr %41, i64 0, i64 0, !dbg !382
  %218 = load i32, ptr %217, align 16, !dbg !382, !tbaa !8
  %219 = and i32 %218, 15, !dbg !383
  %220 = getelementptr inbounds [4 x i32], ptr %41, i64 0, i64 1, !dbg !384
  %221 = load i32, ptr %220, align 4, !dbg !384, !tbaa !8
  %222 = and i32 %221, 15, !dbg !385
  %223 = shl i32 %222, 4, !dbg !386
  %224 = or i32 %219, %223, !dbg !387
  %225 = trunc i32 %224 to i8, !dbg !388
  %226 = load ptr, ptr %46, align 8, !dbg !389, !tbaa !28
  %227 = load i32, ptr %28, align 4, !dbg !390, !tbaa !8
  %228 = mul nsw i32 %227, 2, !dbg !391
  %229 = add nsw i32 %228, 0, !dbg !392
  %230 = sext i32 %229 to i64, !dbg !389
  %231 = getelementptr inbounds i8, ptr %226, i64 %230, !dbg !389
  store i8 %225, ptr %231, align 1, !dbg !393, !tbaa !47
  %232 = getelementptr inbounds [4 x i32], ptr %41, i64 0, i64 2, !dbg !394
  %233 = load i32, ptr %232, align 8, !dbg !394, !tbaa !8
  %234 = and i32 %233, 15, !dbg !395
  %235 = getelementptr inbounds [4 x i32], ptr %41, i64 0, i64 3, !dbg !396
  %236 = load i32, ptr %235, align 4, !dbg !396, !tbaa !8
  %237 = and i32 %236, 15, !dbg !397
  %238 = shl i32 %237, 4, !dbg !398
  %239 = or i32 %234, %238, !dbg !399
  %240 = trunc i32 %239 to i8, !dbg !400
  %241 = load ptr, ptr %46, align 8, !dbg !401, !tbaa !28
  %242 = load i32, ptr %28, align 4, !dbg !402, !tbaa !8
  %243 = mul nsw i32 %242, 2, !dbg !403
  %244 = add nsw i32 %243, 1, !dbg !404
  %245 = sext i32 %244 to i64, !dbg !401
  %246 = getelementptr inbounds i8, ptr %241, i64 %245, !dbg !401
  store i8 %240, ptr %246, align 1, !dbg !405, !tbaa !47
  %247 = load i32, ptr %28, align 4, !dbg !406, !tbaa !8
  %248 = icmp eq i32 %247, 0, !dbg !407
  br i1 %248, label %249, label %256, !dbg !407

249:                                              ; preds = %213
  %250 = load float, ptr %31, align 4, !dbg !408, !tbaa !177
  %251 = load ptr, ptr %27, align 8, !dbg !409, !tbaa !130
  %252 = getelementptr inbounds nuw %struct.block_i4_128, ptr %251, i32 0, i32 0, !dbg !410
  store float %250, ptr %252, align 4, !dbg !411, !tbaa !412
  %253 = load i32, ptr %44, align 4, !dbg !414, !tbaa !8
  %254 = load ptr, ptr %27, align 8, !dbg !415, !tbaa !130
  %255 = getelementptr inbounds nuw %struct.block_i4_128, ptr %254, i32 0, i32 1, !dbg !416
  store i32 %253, ptr %255, align 4, !dbg !417, !tbaa !418
  br label %256, !dbg !419

256:                                              ; preds = %249, %213
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %25) #25, !dbg !420
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %23) #25, !dbg !420
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %20) #25, !dbg !420
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %10) #25, !dbg !420
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %7) #25, !dbg !420
  ret void, !dbg !420
}

; Function Attrs: convergent mustprogress norecurse nounwind uwtable
define protected amdgpu_kernel void @gemm_mq4g256v2_residual_mmq_iu4(ptr addrspace(1) noalias noundef %0, ptr addrspace(1) noalias noundef %1, ptr addrspace(1) noalias noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5, i32 noundef %6) #7 !dbg !421 {
  %8 = alloca ptr, align 8, addrspace(5)
  %9 = alloca ptr, align 8, addrspace(5)
  %10 = alloca ptr, align 8, addrspace(5)
  %11 = alloca ptr, align 8, addrspace(5)
  %12 = alloca ptr, align 8, addrspace(5)
  %13 = alloca ptr, align 8, addrspace(5)
  %14 = alloca i32, align 4, addrspace(5)
  %15 = alloca i32, align 4, addrspace(5)
  %16 = alloca i32, align 4, addrspace(5)
  %17 = alloca i32, align 4, addrspace(5)
  %18 = addrspacecast ptr addrspace(5) %11 to ptr
  %19 = addrspacecast ptr addrspace(5) %12 to ptr
  %20 = addrspacecast ptr addrspace(5) %13 to ptr
  %21 = addrspacecast ptr addrspace(5) %14 to ptr
  %22 = addrspacecast ptr addrspace(5) %15 to ptr
  %23 = addrspacecast ptr addrspace(5) %16 to ptr
  %24 = addrspacecast ptr addrspace(5) %17 to ptr
  store ptr addrspace(1) %0, ptr addrspace(5) %8, align 8
  %25 = load ptr, ptr addrspace(5) %8, align 8, !tbaa !28
  store ptr addrspace(1) %1, ptr addrspace(5) %9, align 8
  %26 = load ptr, ptr addrspace(5) %9, align 8, !tbaa !130
  store ptr addrspace(1) %2, ptr addrspace(5) %10, align 8
  %27 = load ptr, ptr addrspace(5) %10, align 8, !tbaa !128
  store ptr %25, ptr %18, align 8, !tbaa !28
  store ptr %26, ptr %19, align 8, !tbaa !130
  store ptr %27, ptr %20, align 8, !tbaa !128
  store i32 %3, ptr %21, align 4, !tbaa !8
  store i32 %4, ptr %22, align 4, !tbaa !8
  store i32 %5, ptr %23, align 4, !tbaa !8
  store i32 %6, ptr %24, align 4, !tbaa !8
  %28 = load i32, ptr %24, align 4, !dbg !422, !tbaa !8
  %29 = icmp ne i32 %28, 0, !dbg !422
  br i1 %29, label %30, label %37, !dbg !422

30:                                               ; preds = %7
  %31 = load ptr, ptr %18, align 8, !dbg !423, !tbaa !28
  %32 = load ptr, ptr %19, align 8, !dbg !424, !tbaa !130
  %33 = load ptr, ptr %20, align 8, !dbg !425, !tbaa !128
  %34 = load i32, ptr %21, align 4, !dbg !426, !tbaa !8
  %35 = load i32, ptr %22, align 4, !dbg !427, !tbaa !8
  %36 = load i32, ptr %23, align 4, !dbg !428, !tbaa !8
  call void @_ZL42gemm_mq4g256v2_residual_mmq_iu4_gfx12_bodyILb1EEvPKcPK12block_i4_128Pfiii(ptr noundef %31, ptr noundef %32, ptr noundef %33, i32 noundef %34, i32 noundef %35, i32 noundef %36) #26, !dbg !429
  br label %44, !dbg !430

37:                                               ; preds = %7
  %38 = load ptr, ptr %18, align 8, !dbg !431, !tbaa !28
  %39 = load ptr, ptr %19, align 8, !dbg !432, !tbaa !130
  %40 = load ptr, ptr %20, align 8, !dbg !433, !tbaa !128
  %41 = load i32, ptr %21, align 4, !dbg !434, !tbaa !8
  %42 = load i32, ptr %22, align 4, !dbg !435, !tbaa !8
  %43 = load i32, ptr %23, align 4, !dbg !436, !tbaa !8
  call void @_ZL42gemm_mq4g256v2_residual_mmq_iu4_gfx12_bodyILb0EEvPKcPK12block_i4_128Pfiii(ptr noundef %38, ptr noundef %39, ptr noundef %40, i32 noundef %41, i32 noundef %42, i32 noundef %43) #26, !dbg !437
  br label %44

44:                                               ; preds = %37, %30
  ret void, !dbg !438
}

; Function Attrs: alwaysinline convergent mustprogress nounwind uwtable
define internal void @_ZL42gemm_mq4g256v2_residual_mmq_iu4_gfx12_bodyILb1EEvPKcPK12block_i4_128Pfiii(ptr noalias noundef %0, ptr noalias noundef %1, ptr noalias noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5) #5 !dbg !439 {
  %7 = alloca ptr, align 8, addrspace(5)
  %8 = alloca ptr, align 8, addrspace(5)
  %9 = alloca ptr, align 8, addrspace(5)
  %10 = alloca i32, align 4, addrspace(5)
  %11 = alloca i32, align 4, addrspace(5)
  %12 = alloca i32, align 4, addrspace(5)
  %13 = alloca i32, align 4, addrspace(5)
  %14 = alloca i32, align 4, addrspace(5)
  %15 = alloca i32, align 4, addrspace(5)
  %16 = alloca i32, align 4, addrspace(5)
  %17 = alloca i32, align 4, addrspace(5)
  %18 = alloca i32, align 4, addrspace(5)
  %19 = alloca i32, align 4, addrspace(5)
  %20 = alloca i32, align 4, addrspace(5)
  %21 = alloca i32, align 4, addrspace(5)
  %22 = alloca i32, align 4, addrspace(5)
  %23 = alloca i32, align 4, addrspace(5)
  %24 = alloca i32, align 4, addrspace(5)
  %25 = alloca i32, align 4, addrspace(5)
  %26 = alloca ptr, align 8, addrspace(5)
  %27 = alloca ptr, align 8, addrspace(5)
  %28 = alloca [4 x [2 x i32]], align 16, addrspace(5)
  %29 = alloca [2 x [2 x i32]], align 16, addrspace(5)
  %30 = alloca i32, align 4, addrspace(5)
  %31 = alloca i32, align 4, addrspace(5)
  %32 = alloca i32, align 4, addrspace(5)
  %33 = alloca i32, align 4, addrspace(5)
  %34 = alloca i32, align 4, addrspace(5)
  %35 = alloca i32, align 4, addrspace(5)
  %36 = alloca i32, align 4, addrspace(5)
  %37 = alloca [4 x [2 x <8 x float>]], align 32, addrspace(5)
  %38 = alloca [2 x i32], align 4, addrspace(5)
  %39 = alloca [2 x i32], align 4, addrspace(5)
  %40 = alloca [2 x i32], align 4, addrspace(5)
  %41 = alloca [2 x i32], align 4, addrspace(5)
  %42 = alloca i32, align 4, addrspace(5)
  %43 = alloca i32, align 4, addrspace(5)
  %44 = alloca i32, align 4, addrspace(5)
  %45 = alloca i32, align 4, addrspace(5)
  %46 = alloca i32, align 4, addrspace(5)
  %47 = alloca i32, align 4, addrspace(5)
  %48 = alloca i32, align 4, addrspace(5)
  %49 = alloca <2 x i32>, align 8, addrspace(5)
  %50 = alloca i32, align 4, addrspace(5)
  %51 = alloca i32, align 4, addrspace(5)
  %52 = alloca <8 x float>, align 32, addrspace(5)
  %53 = alloca <8 x i32>, align 32, addrspace(5)
  %54 = alloca <8 x i32>, align 32, addrspace(5)
  %55 = alloca [2 x <8 x float>], align 32, addrspace(5)
  %56 = alloca [4 x float], align 16, addrspace(5)
  %57 = alloca [2 x <2 x i32>], align 16, addrspace(5)
  %58 = alloca [2 x <2 x i32>], align 16, addrspace(5)
  %59 = alloca i32, align 4, addrspace(5)
  %60 = alloca i32, align 4, addrspace(5)
  %61 = alloca i32, align 4, addrspace(5)
  %62 = alloca i32, align 4, addrspace(5)
  %63 = alloca i32, align 4, addrspace(5)
  %64 = alloca i32, align 4, addrspace(5)
  %65 = alloca i32, align 4, addrspace(5)
  %66 = alloca i32, align 4, addrspace(5)
  %67 = alloca i32, align 4, addrspace(5)
  %68 = alloca i32, align 4, addrspace(5)
  %69 = alloca i32, align 4, addrspace(5)
  %70 = alloca i32, align 4, addrspace(5)
  %71 = alloca i32, align 4, addrspace(5)
  %72 = alloca float, align 4, addrspace(5)
  %73 = alloca %struct.__half, align 2, addrspace(5)
  %74 = alloca %struct.__half, align 2, addrspace(5)
  %75 = alloca ptr, align 8, addrspace(5)
  %76 = alloca ptr, align 8, addrspace(5)
  %77 = alloca i32, align 4, addrspace(5)
  %78 = alloca <2 x i32>, align 8, addrspace(5)
  %79 = alloca i32, align 4, addrspace(5)
  %80 = alloca i32, align 4, addrspace(5)
  %81 = alloca i32, align 4, addrspace(5)
  %82 = alloca i8, align 1, addrspace(5)
  %83 = alloca ptr, align 8, addrspace(5)
  %84 = alloca ptr, align 8, addrspace(5)
  %85 = alloca i32, align 4, addrspace(5)
  %86 = alloca i32, align 4, addrspace(5)
  %87 = alloca i32, align 4, addrspace(5)
  %88 = alloca i32, align 4, addrspace(5)
  %89 = alloca i32, align 4, addrspace(5)
  %90 = alloca ptr, align 8, addrspace(5)
  %91 = alloca ptr, align 8, addrspace(5)
  %92 = alloca i32, align 4, addrspace(5)
  %93 = alloca <2 x i32>, align 8, addrspace(5)
  %94 = alloca i32, align 4, addrspace(5)
  %95 = alloca i32, align 4, addrspace(5)
  %96 = alloca ptr, align 8, addrspace(5)
  %97 = alloca i32, align 4, addrspace(5)
  %98 = alloca i32, align 4, addrspace(5)
  %99 = alloca i32, align 4, addrspace(5)
  %100 = alloca ptr, align 8, addrspace(5)
  %101 = alloca i32, align 4, addrspace(5)
  %102 = alloca i32, align 4, addrspace(5)
  %103 = alloca i32, align 4, addrspace(5)
  %104 = alloca i32, align 4, addrspace(5)
  %105 = alloca i32, align 4, addrspace(5)
  %106 = alloca i32, align 4, addrspace(5)
  %107 = alloca [8 x float], align 16, addrspace(5)
  %108 = alloca i32, align 4, addrspace(5)
  %109 = alloca i32, align 4, addrspace(5)
  %110 = alloca i32, align 4, addrspace(5)
  %111 = alloca i32, align 4, addrspace(5)
  %112 = alloca float, align 4, addrspace(5)
  %113 = alloca float, align 4, addrspace(5)
  %114 = alloca i32, align 4, addrspace(5)
  %115 = alloca float, align 4, addrspace(5)
  %116 = alloca i32, align 4, addrspace(5)
  %117 = alloca i32, align 4, addrspace(5)
  %118 = alloca ptr, align 8, addrspace(5)
  %119 = alloca i32, align 4, addrspace(5)
  %120 = alloca ptr, align 8, addrspace(5)
  %121 = alloca i32, align 4, addrspace(5)
  %122 = alloca i32, align 4, addrspace(5)
  %123 = alloca float, align 4, addrspace(5)
  %124 = alloca %struct.__half, align 2, addrspace(5)
  %125 = alloca %struct.__half, align 2, addrspace(5)
  %126 = alloca ptr, align 8, addrspace(5)
  %127 = alloca i32, align 4, addrspace(5)
  %128 = alloca i32, align 4, addrspace(5)
  %129 = alloca i32, align 4, addrspace(5)
  %130 = alloca i32, align 4, addrspace(5)
  %131 = alloca i32, align 4, addrspace(5)
  %132 = alloca i32, align 4, addrspace(5)
  %133 = alloca i32, align 4, addrspace(5)
  %134 = alloca i32, align 4, addrspace(5)
  %135 = alloca i32, align 4, addrspace(5)
  %136 = alloca i32, align 4, addrspace(5)
  %137 = alloca i32, align 4, addrspace(5)
  %138 = alloca float, align 4, addrspace(5)
  %139 = alloca i32, align 4, addrspace(5)
  %140 = alloca i32, align 4, addrspace(5)
  %141 = alloca ptr, align 8, addrspace(5)
  %142 = addrspacecast ptr addrspace(5) %7 to ptr
  %143 = addrspacecast ptr addrspace(5) %8 to ptr
  %144 = addrspacecast ptr addrspace(5) %9 to ptr
  %145 = addrspacecast ptr addrspace(5) %10 to ptr
  %146 = addrspacecast ptr addrspace(5) %11 to ptr
  %147 = addrspacecast ptr addrspace(5) %12 to ptr
  %148 = addrspacecast ptr addrspace(5) %13 to ptr
  %149 = addrspacecast ptr addrspace(5) %14 to ptr
  %150 = addrspacecast ptr addrspace(5) %15 to ptr
  %151 = addrspacecast ptr addrspace(5) %16 to ptr
  %152 = addrspacecast ptr addrspace(5) %17 to ptr
  %153 = addrspacecast ptr addrspace(5) %18 to ptr
  %154 = addrspacecast ptr addrspace(5) %19 to ptr
  %155 = addrspacecast ptr addrspace(5) %20 to ptr
  %156 = addrspacecast ptr addrspace(5) %21 to ptr
  %157 = addrspacecast ptr addrspace(5) %22 to ptr
  %158 = addrspacecast ptr addrspace(5) %23 to ptr
  %159 = addrspacecast ptr addrspace(5) %24 to ptr
  %160 = addrspacecast ptr addrspace(5) %25 to ptr
  %161 = addrspacecast ptr addrspace(5) %26 to ptr
  %162 = addrspacecast ptr addrspace(5) %27 to ptr
  %163 = addrspacecast ptr addrspace(5) %28 to ptr
  %164 = addrspacecast ptr addrspace(5) %29 to ptr
  %165 = addrspacecast ptr addrspace(5) %30 to ptr
  %166 = addrspacecast ptr addrspace(5) %31 to ptr
  %167 = addrspacecast ptr addrspace(5) %32 to ptr
  %168 = addrspacecast ptr addrspace(5) %33 to ptr
  %169 = addrspacecast ptr addrspace(5) %34 to ptr
  %170 = addrspacecast ptr addrspace(5) %35 to ptr
  %171 = addrspacecast ptr addrspace(5) %36 to ptr
  %172 = addrspacecast ptr addrspace(5) %37 to ptr
  %173 = addrspacecast ptr addrspace(5) %38 to ptr
  %174 = addrspacecast ptr addrspace(5) %39 to ptr
  %175 = addrspacecast ptr addrspace(5) %40 to ptr
  %176 = addrspacecast ptr addrspace(5) %41 to ptr
  %177 = addrspacecast ptr addrspace(5) %42 to ptr
  %178 = addrspacecast ptr addrspace(5) %43 to ptr
  %179 = addrspacecast ptr addrspace(5) %44 to ptr
  %180 = addrspacecast ptr addrspace(5) %45 to ptr
  %181 = addrspacecast ptr addrspace(5) %46 to ptr
  %182 = addrspacecast ptr addrspace(5) %47 to ptr
  %183 = addrspacecast ptr addrspace(5) %48 to ptr
  %184 = addrspacecast ptr addrspace(5) %49 to ptr
  %185 = addrspacecast ptr addrspace(5) %50 to ptr
  %186 = addrspacecast ptr addrspace(5) %51 to ptr
  %187 = addrspacecast ptr addrspace(5) %53 to ptr
  %188 = addrspacecast ptr addrspace(5) %55 to ptr
  %189 = addrspacecast ptr addrspace(5) %56 to ptr
  %190 = addrspacecast ptr addrspace(5) %57 to ptr
  %191 = addrspacecast ptr addrspace(5) %58 to ptr
  %192 = addrspacecast ptr addrspace(5) %59 to ptr
  %193 = addrspacecast ptr addrspace(5) %60 to ptr
  %194 = addrspacecast ptr addrspace(5) %61 to ptr
  %195 = addrspacecast ptr addrspace(5) %62 to ptr
  %196 = addrspacecast ptr addrspace(5) %63 to ptr
  %197 = addrspacecast ptr addrspace(5) %64 to ptr
  %198 = addrspacecast ptr addrspace(5) %65 to ptr
  %199 = addrspacecast ptr addrspace(5) %66 to ptr
  %200 = addrspacecast ptr addrspace(5) %67 to ptr
  %201 = addrspacecast ptr addrspace(5) %68 to ptr
  %202 = addrspacecast ptr addrspace(5) %69 to ptr
  %203 = addrspacecast ptr addrspace(5) %70 to ptr
  %204 = addrspacecast ptr addrspace(5) %71 to ptr
  %205 = addrspacecast ptr addrspace(5) %72 to ptr
  %206 = addrspacecast ptr addrspace(5) %73 to ptr
  %207 = addrspacecast ptr addrspace(5) %74 to ptr
  %208 = addrspacecast ptr addrspace(5) %75 to ptr
  %209 = addrspacecast ptr addrspace(5) %76 to ptr
  %210 = addrspacecast ptr addrspace(5) %77 to ptr
  %211 = addrspacecast ptr addrspace(5) %78 to ptr
  %212 = addrspacecast ptr addrspace(5) %79 to ptr
  %213 = addrspacecast ptr addrspace(5) %80 to ptr
  %214 = addrspacecast ptr addrspace(5) %81 to ptr
  %215 = addrspacecast ptr addrspace(5) %82 to ptr
  %216 = addrspacecast ptr addrspace(5) %83 to ptr
  %217 = addrspacecast ptr addrspace(5) %84 to ptr
  %218 = addrspacecast ptr addrspace(5) %85 to ptr
  %219 = addrspacecast ptr addrspace(5) %86 to ptr
  %220 = addrspacecast ptr addrspace(5) %87 to ptr
  %221 = addrspacecast ptr addrspace(5) %88 to ptr
  %222 = addrspacecast ptr addrspace(5) %89 to ptr
  %223 = addrspacecast ptr addrspace(5) %90 to ptr
  %224 = addrspacecast ptr addrspace(5) %91 to ptr
  %225 = addrspacecast ptr addrspace(5) %92 to ptr
  %226 = addrspacecast ptr addrspace(5) %93 to ptr
  %227 = addrspacecast ptr addrspace(5) %94 to ptr
  %228 = addrspacecast ptr addrspace(5) %95 to ptr
  %229 = addrspacecast ptr addrspace(5) %96 to ptr
  %230 = addrspacecast ptr addrspace(5) %97 to ptr
  %231 = addrspacecast ptr addrspace(5) %98 to ptr
  %232 = addrspacecast ptr addrspace(5) %99 to ptr
  %233 = addrspacecast ptr addrspace(5) %100 to ptr
  %234 = addrspacecast ptr addrspace(5) %101 to ptr
  %235 = addrspacecast ptr addrspace(5) %102 to ptr
  %236 = addrspacecast ptr addrspace(5) %103 to ptr
  %237 = addrspacecast ptr addrspace(5) %104 to ptr
  %238 = addrspacecast ptr addrspace(5) %105 to ptr
  %239 = addrspacecast ptr addrspace(5) %106 to ptr
  %240 = addrspacecast ptr addrspace(5) %107 to ptr
  %241 = addrspacecast ptr addrspace(5) %108 to ptr
  %242 = addrspacecast ptr addrspace(5) %109 to ptr
  %243 = addrspacecast ptr addrspace(5) %110 to ptr
  %244 = addrspacecast ptr addrspace(5) %111 to ptr
  %245 = addrspacecast ptr addrspace(5) %112 to ptr
  %246 = addrspacecast ptr addrspace(5) %113 to ptr
  %247 = addrspacecast ptr addrspace(5) %114 to ptr
  %248 = addrspacecast ptr addrspace(5) %115 to ptr
  %249 = addrspacecast ptr addrspace(5) %116 to ptr
  %250 = addrspacecast ptr addrspace(5) %117 to ptr
  %251 = addrspacecast ptr addrspace(5) %118 to ptr
  %252 = addrspacecast ptr addrspace(5) %119 to ptr
  %253 = addrspacecast ptr addrspace(5) %120 to ptr
  %254 = addrspacecast ptr addrspace(5) %121 to ptr
  %255 = addrspacecast ptr addrspace(5) %122 to ptr
  %256 = addrspacecast ptr addrspace(5) %123 to ptr
  %257 = addrspacecast ptr addrspace(5) %124 to ptr
  %258 = addrspacecast ptr addrspace(5) %125 to ptr
  %259 = addrspacecast ptr addrspace(5) %126 to ptr
  %260 = addrspacecast ptr addrspace(5) %127 to ptr
  %261 = addrspacecast ptr addrspace(5) %128 to ptr
  %262 = addrspacecast ptr addrspace(5) %129 to ptr
  %263 = addrspacecast ptr addrspace(5) %130 to ptr
  %264 = addrspacecast ptr addrspace(5) %131 to ptr
  %265 = addrspacecast ptr addrspace(5) %132 to ptr
  %266 = addrspacecast ptr addrspace(5) %133 to ptr
  %267 = addrspacecast ptr addrspace(5) %134 to ptr
  %268 = addrspacecast ptr addrspace(5) %135 to ptr
  %269 = addrspacecast ptr addrspace(5) %136 to ptr
  %270 = addrspacecast ptr addrspace(5) %137 to ptr
  %271 = addrspacecast ptr addrspace(5) %138 to ptr
  %272 = addrspacecast ptr addrspace(5) %139 to ptr
  %273 = addrspacecast ptr addrspace(5) %140 to ptr
  %274 = addrspacecast ptr addrspace(5) %141 to ptr
  store ptr %0, ptr %142, align 8, !tbaa !28
  store ptr %1, ptr %143, align 8, !tbaa !130
  store ptr %2, ptr %144, align 8, !tbaa !128
  store i32 %3, ptr %145, align 4, !tbaa !8
  store i32 %4, ptr %146, align 4, !tbaa !8
  store i32 %5, ptr %147, align 4, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %13) #25, !dbg !440
  %275 = call noundef i32 @_ZN25__hip_builtin_threadIdx_t7__get_xEv() #26, !dbg !441
  store i32 %275, ptr %148, align 4, !dbg !442, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %14) #25, !dbg !443
  %276 = load i32, ptr %148, align 4, !dbg !444, !tbaa !8
  %277 = ashr i32 %276, 5, !dbg !445
  store i32 %277, ptr %149, align 4, !dbg !446, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %15) #25, !dbg !447
  %278 = load i32, ptr %148, align 4, !dbg !448, !tbaa !8
  %279 = and i32 %278, 31, !dbg !449
  store i32 %279, ptr %150, align 4, !dbg !450, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %16) #25, !dbg !451
  %280 = call noundef i32 @_ZN24__hip_builtin_blockIdx_t7__get_xEv() #26, !dbg !452
  %281 = mul i32 %280, 128, !dbg !453
  store i32 %281, ptr %151, align 4, !dbg !454, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %17) #25, !dbg !455
  %282 = call noundef i32 @_ZN24__hip_builtin_blockIdx_t7__get_yEv() #26, !dbg !456
  %283 = mul i32 %282, 128, !dbg !457
  store i32 %283, ptr %152, align 4, !dbg !458, !tbaa !8
  %284 = load i32, ptr %151, align 4, !dbg !459, !tbaa !8
  %285 = load i32, ptr %145, align 4, !dbg !460, !tbaa !8
  %286 = icmp sge i32 %284, %285, !dbg !461
  br i1 %286, label %291, label %287, !dbg !462

287:                                              ; preds = %6
  %288 = load i32, ptr %152, align 4, !dbg !463, !tbaa !8
  %289 = load i32, ptr %147, align 4, !dbg !464, !tbaa !8
  %290 = icmp sge i32 %288, %289, !dbg !465
  br i1 %290, label %291, label %292, !dbg !462

291:                                              ; preds = %287, %6
  store i32 1, ptr %153, align 4
  br label %1375, !dbg !466

292:                                              ; preds = %287
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %19) #25, !dbg !467
  %293 = load i32, ptr %150, align 4, !dbg !468, !tbaa !8
  %294 = and i32 %293, 15, !dbg !469
  store i32 %294, ptr %154, align 4, !dbg !470, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %20) #25, !dbg !471
  %295 = load i32, ptr %150, align 4, !dbg !472, !tbaa !8
  %296 = ashr i32 %295, 4, !dbg !473
  store i32 %296, ptr %155, align 4, !dbg !474, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %21) #25, !dbg !475
  %297 = load i32, ptr %149, align 4, !dbg !476, !tbaa !8
  %298 = ashr i32 %297, 1, !dbg !477
  store i32 %298, ptr %156, align 4, !dbg !478, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %22) #25, !dbg !479
  %299 = load i32, ptr %149, align 4, !dbg !480, !tbaa !8
  %300 = and i32 %299, 1, !dbg !481
  store i32 %300, ptr %157, align 4, !dbg !482, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %23) #25, !dbg !483
  %301 = load i32, ptr %156, align 4, !dbg !484, !tbaa !8
  %302 = mul nsw i32 %301, 32, !dbg !485
  store i32 %302, ptr %158, align 4, !dbg !486, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %24) #25, !dbg !487
  %303 = load i32, ptr %157, align 4, !dbg !488, !tbaa !8
  %304 = mul nsw i32 %303, 64, !dbg !489
  store i32 %304, ptr %159, align 4, !dbg !490, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %25) #25, !dbg !491
  %305 = load i32, ptr %146, align 4, !dbg !492, !tbaa !8
  %306 = sdiv i32 %305, 256, !dbg !493
  store i32 %306, ptr %160, align 4, !dbg !494, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %26) #25, !dbg !495
  store ptr addrspacecast (ptr addrspace(3) @LDS to ptr), ptr %161, align 8, !dbg !496, !tbaa !28
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %27) #25, !dbg !497
  store ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @LDS to ptr), i64 4096), ptr %162, align 8, !dbg !498, !tbaa !28
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %28) #25, !dbg !499
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %29) #25, !dbg !500
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %30) #25, !dbg !501
  %307 = load i32, ptr %150, align 4, !dbg !502, !tbaa !8
  %308 = mul i32 %307, 8, !dbg !503
  store i32 %308, ptr %165, align 4, !dbg !504, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %31) #25, !dbg !505
  store i32 0, ptr %166, align 4, !dbg !506, !tbaa !8
  br label %309, !dbg !505

309:                                              ; preds = %340, %292
  %310 = load i32, ptr %166, align 4, !dbg !507, !tbaa !8
  %311 = icmp slt i32 %310, 4, !dbg !508
  br i1 %311, label %313, label %312, !dbg !509

312:                                              ; preds = %309
  store i32 2, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %31) #25, !dbg !509
  br label %343

313:                                              ; preds = %309
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %32) #25, !dbg !510
  %314 = load i32, ptr %157, align 4, !dbg !511, !tbaa !8
  %315 = mul nsw i32 %314, 4, !dbg !512
  %316 = load i32, ptr %166, align 4, !dbg !513, !tbaa !8
  %317 = add nsw i32 %315, %316, !dbg !514
  store i32 %317, ptr %167, align 4, !dbg !515, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %33) #25, !dbg !516
  store i32 0, ptr %168, align 4, !dbg !517, !tbaa !8
  br label %318, !dbg !516

318:                                              ; preds = %336, %313
  %319 = load i32, ptr %168, align 4, !dbg !518, !tbaa !8
  %320 = icmp slt i32 %319, 2, !dbg !519
  br i1 %320, label %322, label %321, !dbg !520

321:                                              ; preds = %318
  store i32 5, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %33) #25, !dbg !520
  br label %339

322:                                              ; preds = %318
  %323 = load i32, ptr %167, align 4, !dbg !521, !tbaa !8
  %324 = mul i32 %323, 2, !dbg !522
  %325 = load i32, ptr %168, align 4, !dbg !523, !tbaa !8
  %326 = add i32 %324, %325, !dbg !524
  %327 = mul i32 %326, 256, !dbg !525
  %328 = load i32, ptr %165, align 4, !dbg !526, !tbaa !8
  %329 = add i32 %327, %328, !dbg !527
  %330 = load i32, ptr %166, align 4, !dbg !528, !tbaa !8
  %331 = sext i32 %330 to i64, !dbg !529
  %332 = getelementptr inbounds [4 x [2 x i32]], ptr %163, i64 0, i64 %331, !dbg !529
  %333 = load i32, ptr %168, align 4, !dbg !530, !tbaa !8
  %334 = sext i32 %333 to i64, !dbg !529
  %335 = getelementptr inbounds [2 x i32], ptr %332, i64 0, i64 %334, !dbg !529
  store i32 %329, ptr %335, align 4, !dbg !531, !tbaa !8
  br label %336, !dbg !532

336:                                              ; preds = %322
  %337 = load i32, ptr %168, align 4, !dbg !533, !tbaa !8
  %338 = add nsw i32 %337, 1, !dbg !533
  store i32 %338, ptr %168, align 4, !dbg !533, !tbaa !8
  br label %318, !dbg !520, !llvm.loop !534

339:                                              ; preds = %321
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %32) #25, !dbg !535
  br label %340, !dbg !535

340:                                              ; preds = %339
  %341 = load i32, ptr %166, align 4, !dbg !536, !tbaa !8
  %342 = add nsw i32 %341, 1, !dbg !536
  store i32 %342, ptr %166, align 4, !dbg !536, !tbaa !8
  br label %309, !dbg !509, !llvm.loop !537

343:                                              ; preds = %312
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %34) #25, !dbg !538
  store i32 0, ptr %169, align 4, !dbg !539, !tbaa !8
  br label %344, !dbg !538

344:                                              ; preds = %375, %343
  %345 = load i32, ptr %169, align 4, !dbg !540, !tbaa !8
  %346 = icmp slt i32 %345, 2, !dbg !541
  br i1 %346, label %348, label %347, !dbg !542

347:                                              ; preds = %344
  store i32 8, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %34) #25, !dbg !542
  br label %378

348:                                              ; preds = %344
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %35) #25, !dbg !543
  %349 = load i32, ptr %156, align 4, !dbg !544, !tbaa !8
  %350 = mul nsw i32 %349, 2, !dbg !545
  %351 = load i32, ptr %169, align 4, !dbg !546, !tbaa !8
  %352 = add nsw i32 %350, %351, !dbg !547
  store i32 %352, ptr %170, align 4, !dbg !548, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %36) #25, !dbg !549
  store i32 0, ptr %171, align 4, !dbg !550, !tbaa !8
  br label %353, !dbg !549

353:                                              ; preds = %371, %348
  %354 = load i32, ptr %171, align 4, !dbg !551, !tbaa !8
  %355 = icmp slt i32 %354, 2, !dbg !552
  br i1 %355, label %357, label %356, !dbg !553

356:                                              ; preds = %353
  store i32 11, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %36) #25, !dbg !553
  br label %374

357:                                              ; preds = %353
  %358 = load i32, ptr %170, align 4, !dbg !554, !tbaa !8
  %359 = mul i32 %358, 2, !dbg !555
  %360 = load i32, ptr %171, align 4, !dbg !556, !tbaa !8
  %361 = add i32 %359, %360, !dbg !557
  %362 = mul i32 %361, 256, !dbg !558
  %363 = load i32, ptr %165, align 4, !dbg !559, !tbaa !8
  %364 = add i32 %362, %363, !dbg !560
  %365 = load i32, ptr %169, align 4, !dbg !561, !tbaa !8
  %366 = sext i32 %365 to i64, !dbg !562
  %367 = getelementptr inbounds [2 x [2 x i32]], ptr %164, i64 0, i64 %366, !dbg !562
  %368 = load i32, ptr %171, align 4, !dbg !563, !tbaa !8
  %369 = sext i32 %368 to i64, !dbg !562
  %370 = getelementptr inbounds [2 x i32], ptr %367, i64 0, i64 %369, !dbg !562
  store i32 %364, ptr %370, align 4, !dbg !564, !tbaa !8
  br label %371, !dbg !565

371:                                              ; preds = %357
  %372 = load i32, ptr %171, align 4, !dbg !566, !tbaa !8
  %373 = add nsw i32 %372, 1, !dbg !566
  store i32 %373, ptr %171, align 4, !dbg !566, !tbaa !8
  br label %353, !dbg !553, !llvm.loop !567

374:                                              ; preds = %356
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %35) #25, !dbg !568
  br label %375, !dbg !568

375:                                              ; preds = %374
  %376 = load i32, ptr %169, align 4, !dbg !569, !tbaa !8
  %377 = add nsw i32 %376, 1, !dbg !569
  store i32 %377, ptr %169, align 4, !dbg !569, !tbaa !8
  br label %344, !dbg !542, !llvm.loop !570

378:                                              ; preds = %347
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %37) #25, !dbg !571
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %38) #25, !dbg !572
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %39) #25, !dbg !573
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %40) #25, !dbg !574
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %41) #25, !dbg !575
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %42) #25, !dbg !576
  store i32 0, ptr %177, align 4, !dbg !577, !tbaa !8
  br label %379, !dbg !576

379:                                              ; preds = %471, %378
  %380 = load i32, ptr %177, align 4, !dbg !578, !tbaa !8
  %381 = icmp slt i32 %380, 2, !dbg !579
  br i1 %381, label %383, label %382, !dbg !580

382:                                              ; preds = %379
  store i32 14, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %42) #25, !dbg !580
  br label %474

383:                                              ; preds = %379
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %43) #25, !dbg !581
  %384 = load i32, ptr %152, align 4, !dbg !582, !tbaa !8
  %385 = load i32, ptr %177, align 4, !dbg !583, !tbaa !8
  %386 = mul nsw i32 %385, 64, !dbg !584
  %387 = add nsw i32 %384, %386, !dbg !585
  %388 = load i32, ptr %148, align 4, !dbg !586, !tbaa !8
  %389 = ashr i32 %388, 2, !dbg !587
  %390 = add nsw i32 %387, %389, !dbg !588
  store i32 %390, ptr %178, align 4, !dbg !589, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %44) #25, !dbg !590
  %391 = load i32, ptr %178, align 4, !dbg !591, !tbaa !8
  %392 = load i32, ptr %147, align 4, !dbg !592, !tbaa !8
  %393 = icmp slt i32 %391, %392, !dbg !593
  br i1 %393, label %394, label %396, !dbg !594

394:                                              ; preds = %383
  %395 = load i32, ptr %178, align 4, !dbg !595, !tbaa !8
  br label %399, !dbg !594

396:                                              ; preds = %383
  %397 = load i32, ptr %147, align 4, !dbg !596, !tbaa !8
  %398 = sub nsw i32 %397, 1, !dbg !597
  br label %399, !dbg !594

399:                                              ; preds = %396, %394
  %400 = phi i32 [ %395, %394 ], [ %398, %396 ], !dbg !594
  store i32 %400, ptr %179, align 4, !dbg !598, !tbaa !8
  %401 = load i32, ptr %178, align 4, !dbg !599, !tbaa !8
  %402 = load i32, ptr %147, align 4, !dbg !600, !tbaa !8
  %403 = icmp slt i32 %401, %402, !dbg !601
  %404 = zext i1 %403 to i64, !dbg !602
  %405 = select i1 %403, i32 1, i32 0, !dbg !602
  %406 = load i32, ptr %177, align 4, !dbg !603, !tbaa !8
  %407 = sext i32 %406 to i64, !dbg !604
  %408 = getelementptr inbounds [2 x i32], ptr %176, i64 0, i64 %407, !dbg !604
  store i32 %405, ptr %408, align 4, !dbg !605, !tbaa !8
  %409 = load i32, ptr %179, align 4, !dbg !606, !tbaa !8
  %410 = mul i32 %409, 72, !dbg !607
  %411 = load i32, ptr %148, align 4, !dbg !608, !tbaa !8
  %412 = and i32 %411, 3, !dbg !609
  %413 = mul i32 %412, 8, !dbg !610
  %414 = add i32 %410, %413, !dbg !611
  %415 = load i32, ptr %177, align 4, !dbg !612, !tbaa !8
  %416 = sext i32 %415 to i64, !dbg !613
  %417 = getelementptr inbounds [2 x i32], ptr %173, i64 0, i64 %416, !dbg !613
  store i32 %414, ptr %417, align 4, !dbg !614, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %45) #25, !dbg !615
  %418 = load i32, ptr %151, align 4, !dbg !616, !tbaa !8
  %419 = load i32, ptr %177, align 4, !dbg !617, !tbaa !8
  %420 = mul nsw i32 %419, 64, !dbg !618
  %421 = add nsw i32 %418, %420, !dbg !619
  %422 = load i32, ptr %148, align 4, !dbg !620, !tbaa !8
  %423 = ashr i32 %422, 2, !dbg !621
  %424 = add nsw i32 %421, %423, !dbg !622
  store i32 %424, ptr %180, align 4, !dbg !623, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %46) #25, !dbg !624
  %425 = load i32, ptr %180, align 4, !dbg !625, !tbaa !8
  %426 = load i32, ptr %145, align 4, !dbg !626, !tbaa !8
  %427 = icmp slt i32 %425, %426, !dbg !627
  br i1 %427, label %428, label %430, !dbg !628

428:                                              ; preds = %399
  %429 = load i32, ptr %180, align 4, !dbg !629, !tbaa !8
  br label %433, !dbg !628

430:                                              ; preds = %399
  %431 = load i32, ptr %145, align 4, !dbg !630, !tbaa !8
  %432 = sub nsw i32 %431, 1, !dbg !631
  br label %433, !dbg !628

433:                                              ; preds = %430, %428
  %434 = phi i32 [ %429, %428 ], [ %432, %430 ], !dbg !628
  store i32 %434, ptr %181, align 4, !dbg !632, !tbaa !8
  %435 = load i32, ptr %181, align 4, !dbg !633, !tbaa !8
  %436 = load i32, ptr %160, align 4, !dbg !634, !tbaa !8
  %437 = mul i32 %435, %436, !dbg !635
  %438 = mul i32 %437, 136, !dbg !636
  %439 = load i32, ptr %148, align 4, !dbg !637, !tbaa !8
  %440 = and i32 %439, 3, !dbg !638
  %441 = mul i32 %440, 8, !dbg !639
  %442 = add i32 %438, %441, !dbg !640
  %443 = load i32, ptr %177, align 4, !dbg !641, !tbaa !8
  %444 = sext i32 %443 to i64, !dbg !642
  %445 = getelementptr inbounds [2 x i32], ptr %174, i64 0, i64 %444, !dbg !642
  store i32 %442, ptr %445, align 4, !dbg !643, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %47) #25, !dbg !644
  %446 = load i32, ptr %177, align 4, !dbg !645, !tbaa !8
  %447 = mul nsw i32 %446, 64, !dbg !646
  %448 = load i32, ptr %148, align 4, !dbg !647, !tbaa !8
  %449 = ashr i32 %448, 2, !dbg !648
  %450 = add nsw i32 %447, %449, !dbg !649
  store i32 %450, ptr %182, align 4, !dbg !650, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %48) #25, !dbg !651
  %451 = load i32, ptr %148, align 4, !dbg !652, !tbaa !8
  %452 = and i32 %451, 3, !dbg !653
  store i32 %452, ptr %183, align 4, !dbg !654, !tbaa !8
  %453 = load i32, ptr %182, align 4, !dbg !655, !tbaa !8
  %454 = lshr i32 %453, 4, !dbg !656
  %455 = mul i32 %454, 2, !dbg !657
  %456 = load i32, ptr %183, align 4, !dbg !658, !tbaa !8
  %457 = lshr i32 %456, 1, !dbg !659
  %458 = add i32 %455, %457, !dbg !660
  %459 = mul i32 %458, 256, !dbg !661
  %460 = load i32, ptr %182, align 4, !dbg !662, !tbaa !8
  %461 = and i32 %460, 15, !dbg !663
  %462 = load i32, ptr %183, align 4, !dbg !664, !tbaa !8
  %463 = and i32 %462, 1, !dbg !665
  %464 = mul i32 %463, 16, !dbg !666
  %465 = add i32 %461, %464, !dbg !667
  %466 = mul i32 %465, 8, !dbg !668
  %467 = add i32 %459, %466, !dbg !669
  %468 = load i32, ptr %177, align 4, !dbg !670, !tbaa !8
  %469 = sext i32 %468 to i64, !dbg !671
  %470 = getelementptr inbounds [2 x i32], ptr %175, i64 0, i64 %469, !dbg !671
  store i32 %467, ptr %470, align 4, !dbg !672, !tbaa !8
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %48) #25, !dbg !673
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %47) #25, !dbg !673
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %46) #25, !dbg !673
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %45) #25, !dbg !673
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %44) #25, !dbg !673
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %43) #25, !dbg !673
  br label %471, !dbg !673

471:                                              ; preds = %433
  %472 = load i32, ptr %177, align 4, !dbg !674, !tbaa !8
  %473 = add nsw i32 %472, 1, !dbg !674
  store i32 %473, ptr %177, align 4, !dbg !674, !tbaa !8
  br label %379, !dbg !580, !llvm.loop !675

474:                                              ; preds = %382
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %49) #25, !dbg !676
  store <2 x i32> zeroinitializer, ptr %184, align 8, !dbg !677, !tbaa !47
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %50) #25, !dbg !678
  store i32 0, ptr %185, align 4, !dbg !679, !tbaa !8
  br label %475, !dbg !678

475:                                              ; preds = %496, %474
  %476 = load i32, ptr %185, align 4, !dbg !680, !tbaa !8
  %477 = icmp slt i32 %476, 4, !dbg !681
  br i1 %477, label %479, label %478, !dbg !682

478:                                              ; preds = %475
  store i32 17, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %50) #25, !dbg !682
  br label %499

479:                                              ; preds = %475
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %51) #25, !dbg !683
  store i32 0, ptr %186, align 4, !dbg !684, !tbaa !8
  br label %480, !dbg !683

480:                                              ; preds = %492, %479
  %481 = load i32, ptr %186, align 4, !dbg !685, !tbaa !8
  %482 = icmp slt i32 %481, 2, !dbg !686
  br i1 %482, label %484, label %483, !dbg !687

483:                                              ; preds = %480
  store i32 20, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %51) #25, !dbg !687
  br label %495

484:                                              ; preds = %480
  store <8 x float> zeroinitializer, ptr addrspace(5) %52, align 32, !dbg !688, !tbaa !47
  %485 = load <8 x float>, ptr addrspace(5) %52, align 32, !dbg !688, !tbaa !47
  %486 = load i32, ptr %185, align 4, !dbg !689, !tbaa !8
  %487 = sext i32 %486 to i64, !dbg !690
  %488 = getelementptr inbounds [4 x [2 x <8 x float>]], ptr %172, i64 0, i64 %487, !dbg !690
  %489 = load i32, ptr %186, align 4, !dbg !691, !tbaa !8
  %490 = sext i32 %489 to i64, !dbg !690
  %491 = getelementptr inbounds [2 x <8 x float>], ptr %488, i64 0, i64 %490, !dbg !690
  store <8 x float> %485, ptr %491, align 32, !dbg !692, !tbaa !47
  br label %492, !dbg !693

492:                                              ; preds = %484
  %493 = load i32, ptr %186, align 4, !dbg !694, !tbaa !8
  %494 = add nsw i32 %493, 1, !dbg !694
  store i32 %494, ptr %186, align 4, !dbg !694, !tbaa !8
  br label %480, !dbg !687, !llvm.loop !695

495:                                              ; preds = %483
  br label %496, !dbg !696

496:                                              ; preds = %495
  %497 = load i32, ptr %185, align 4, !dbg !697, !tbaa !8
  %498 = add nsw i32 %497, 1, !dbg !697
  store i32 %498, ptr %185, align 4, !dbg !697, !tbaa !8
  br label %475, !dbg !682, !llvm.loop !698

499:                                              ; preds = %478
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %53) #25, !dbg !699
  store <8 x i32> zeroinitializer, ptr addrspace(5) %54, align 32, !dbg !700, !tbaa !47
  %500 = load <8 x i32>, ptr addrspace(5) %54, align 32, !dbg !700, !tbaa !47
  store <8 x i32> %500, ptr %187, align 32, !dbg !701, !tbaa !47
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %55) #25, !dbg !702
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %56) #25, !dbg !703
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %57) #25, !dbg !704
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %58) #25, !dbg !705
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %59) #25, !dbg !706
  store i32 0, ptr %192, align 4, !dbg !707, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %60) #25, !dbg !708
  store i32 0, ptr %193, align 4, !dbg !709, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %61) #25, !dbg !710
  %501 = load i32, ptr %148, align 4, !dbg !711, !tbaa !8
  store i32 %501, ptr %194, align 4, !dbg !712, !tbaa !8
  br label %502, !dbg !710

502:                                              ; preds = %537, %499
  %503 = load i32, ptr %194, align 4, !dbg !713, !tbaa !8
  %504 = icmp slt i32 %503, 256, !dbg !714
  br i1 %504, label %506, label %505, !dbg !715

505:                                              ; preds = %502
  store i32 23, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %61) #25, !dbg !715
  br label %540

506:                                              ; preds = %502
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %62) #25, !dbg !716
  %507 = load i32, ptr %194, align 4, !dbg !717, !tbaa !8
  %508 = ashr i32 %507, 1, !dbg !718
  store i32 %508, ptr %195, align 4, !dbg !719, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %63) #25, !dbg !720
  %509 = load i32, ptr %194, align 4, !dbg !721, !tbaa !8
  %510 = and i32 %509, 1, !dbg !722
  store i32 %510, ptr %196, align 4, !dbg !723, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %64) #25, !dbg !724
  %511 = load i32, ptr %152, align 4, !dbg !725, !tbaa !8
  %512 = load i32, ptr %195, align 4, !dbg !726, !tbaa !8
  %513 = add nsw i32 %511, %512, !dbg !727
  store i32 %513, ptr %197, align 4, !dbg !728, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %65) #25, !dbg !729
  %514 = load i32, ptr %197, align 4, !dbg !730, !tbaa !8
  %515 = load i32, ptr %147, align 4, !dbg !731, !tbaa !8
  %516 = icmp slt i32 %514, %515, !dbg !732
  br i1 %516, label %517, label %526, !dbg !733

517:                                              ; preds = %506
  %518 = load ptr, ptr %143, align 8, !dbg !734, !tbaa !130
  %519 = load i32, ptr %197, align 4, !dbg !735, !tbaa !8
  %520 = sext i32 %519 to i64, !dbg !736
  %521 = getelementptr inbounds %struct.block_i4_128, ptr %518, i64 %520, !dbg !736
  %522 = load i32, ptr %196, align 4, !dbg !737, !tbaa !8
  %523 = sext i32 %522 to i64, !dbg !736
  %524 = getelementptr inbounds i32, ptr %521, i64 %523, !dbg !736
  %525 = load i32, ptr %524, align 4, !dbg !736, !tbaa !8
  br label %527, !dbg !733

526:                                              ; preds = %506
  br label %527, !dbg !733

527:                                              ; preds = %526, %517
  %528 = phi i32 [ %525, %517 ], [ 0, %526 ], !dbg !733
  store i32 %528, ptr %198, align 4, !dbg !738, !tbaa !8
  %529 = load i32, ptr %198, align 4, !dbg !739, !tbaa !8
  %530 = load i32, ptr %195, align 4, !dbg !740, !tbaa !8
  %531 = mul nsw i32 %530, 8, !dbg !741
  %532 = sext i32 %531 to i64, !dbg !742
  %533 = getelementptr inbounds i8, ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @LDS to ptr), i64 12288), i64 %532, !dbg !742
  %534 = load i32, ptr %196, align 4, !dbg !743, !tbaa !8
  %535 = sext i32 %534 to i64, !dbg !742
  %536 = getelementptr inbounds i32, ptr %533, i64 %535, !dbg !742
  store i32 %529, ptr %536, align 1, !dbg !744, !tbaa !8
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %65) #25, !dbg !745
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %64) #25, !dbg !745
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %63) #25, !dbg !745
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %62) #25, !dbg !745
  br label %537, !dbg !745

537:                                              ; preds = %527
  %538 = load i32, ptr %194, align 4, !dbg !746, !tbaa !8
  %539 = add nsw i32 %538, 256, !dbg !746
  store i32 %539, ptr %194, align 4, !dbg !746, !tbaa !8
  br label %502, !dbg !715, !llvm.loop !747

540:                                              ; preds = %505
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %66) #25, !dbg !748
  %541 = load i32, ptr %148, align 4, !dbg !749, !tbaa !8
  store i32 %541, ptr %199, align 4, !dbg !750, !tbaa !8
  br label %542, !dbg !748

542:                                              ; preds = %616, %540
  %543 = load i32, ptr %199, align 4, !dbg !751, !tbaa !8
  %544 = icmp slt i32 %543, 256, !dbg !752
  br i1 %544, label %546, label %545, !dbg !753

545:                                              ; preds = %542
  store i32 26, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %66) #25, !dbg !753
  br label %619

546:                                              ; preds = %542
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %67) #25, !dbg !754
  %547 = load i32, ptr %199, align 4, !dbg !755, !tbaa !8
  %548 = ashr i32 %547, 1, !dbg !756
  store i32 %548, ptr %200, align 4, !dbg !757, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %68) #25, !dbg !758
  %549 = load i32, ptr %199, align 4, !dbg !759, !tbaa !8
  %550 = and i32 %549, 1, !dbg !760
  store i32 %550, ptr %201, align 4, !dbg !761, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %69) #25, !dbg !762
  %551 = load i32, ptr %151, align 4, !dbg !763, !tbaa !8
  %552 = load i32, ptr %200, align 4, !dbg !764, !tbaa !8
  %553 = add nsw i32 %551, %552, !dbg !765
  store i32 %553, ptr %202, align 4, !dbg !766, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %70) #25, !dbg !767
  %554 = load i32, ptr %202, align 4, !dbg !768, !tbaa !8
  %555 = load i32, ptr %145, align 4, !dbg !769, !tbaa !8
  %556 = icmp slt i32 %554, %555, !dbg !770
  br i1 %556, label %557, label %559, !dbg !771

557:                                              ; preds = %546
  %558 = load i32, ptr %202, align 4, !dbg !772, !tbaa !8
  br label %562, !dbg !771

559:                                              ; preds = %546
  %560 = load i32, ptr %145, align 4, !dbg !773, !tbaa !8
  %561 = sub nsw i32 %560, 1, !dbg !774
  br label %562, !dbg !771

562:                                              ; preds = %559, %557
  %563 = phi i32 [ %558, %557 ], [ %561, %559 ], !dbg !771
  store i32 %563, ptr %203, align 4, !dbg !775, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %71) #25, !dbg !776
  %564 = load ptr, ptr %142, align 8, !dbg !777, !tbaa !28
  %565 = load i32, ptr %203, align 4, !dbg !778, !tbaa !8
  %566 = sext i32 %565 to i64, !dbg !778
  %567 = load i32, ptr %160, align 4, !dbg !779, !tbaa !8
  %568 = sext i32 %567 to i64, !dbg !779
  %569 = mul nsw i64 %566, %568, !dbg !780
  %570 = mul nsw i64 %569, 136, !dbg !781
  %571 = getelementptr inbounds i8, ptr %564, i64 %570, !dbg !782
  %572 = load i32, ptr %571, align 4, !dbg !782, !tbaa !8
  store i32 %572, ptr %204, align 4, !dbg !783, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %72) #25, !dbg !784
  %573 = load i32, ptr %201, align 4, !dbg !785, !tbaa !8
  %574 = icmp eq i32 %573, 0, !dbg !786
  br i1 %574, label %575, label %587, !dbg !787

575:                                              ; preds = %562
  %576 = addrspacecast ptr %206 to ptr addrspace(5), !dbg !788
  %577 = load i32, ptr %204, align 4, !dbg !789, !tbaa !8
  %578 = and i32 %577, 65535, !dbg !790
  %579 = trunc i32 %578 to i16, !dbg !791
  %580 = call i16 @_Z16__ushort_as_halft(i16 noundef zeroext %579) #26, !dbg !788
  %581 = getelementptr inbounds nuw %struct.__half, ptr addrspace(5) %576, i32 0, i32 0, !dbg !788
  %582 = getelementptr inbounds nuw %union.anon.0, ptr addrspace(5) %581, i32 0, i32 0, !dbg !788
  store i16 %580, ptr addrspace(5) %582, align 2, !dbg !788
  %583 = getelementptr inbounds nuw %struct.__half, ptr %206, i32 0, i32 0, !dbg !792
  %584 = getelementptr inbounds nuw %union.anon.0, ptr %583, i32 0, i32 0, !dbg !792
  %585 = load i16, ptr %584, align 2, !dbg !792
  %586 = call contract noundef float @_Z12__half2float6__half(i16 %585) #26, !dbg !792
  br label %599, !dbg !787

587:                                              ; preds = %562
  %588 = addrspacecast ptr %207 to ptr addrspace(5), !dbg !793
  %589 = load i32, ptr %204, align 4, !dbg !794, !tbaa !8
  %590 = lshr i32 %589, 16, !dbg !795
  %591 = trunc i32 %590 to i16, !dbg !796
  %592 = call i16 @_Z16__ushort_as_halft(i16 noundef zeroext %591) #26, !dbg !793
  %593 = getelementptr inbounds nuw %struct.__half, ptr addrspace(5) %588, i32 0, i32 0, !dbg !793
  %594 = getelementptr inbounds nuw %union.anon.0, ptr addrspace(5) %593, i32 0, i32 0, !dbg !793
  store i16 %592, ptr addrspace(5) %594, align 2, !dbg !793
  %595 = getelementptr inbounds nuw %struct.__half, ptr %207, i32 0, i32 0, !dbg !797
  %596 = getelementptr inbounds nuw %union.anon.0, ptr %595, i32 0, i32 0, !dbg !797
  %597 = load i16, ptr %596, align 2, !dbg !797
  %598 = call contract noundef float @_Z12__half2float6__half(i16 %597) #26, !dbg !797
  br label %599, !dbg !787

599:                                              ; preds = %587, %575
  %600 = phi contract float [ %586, %575 ], [ %598, %587 ], !dbg !787
  store float %600, ptr %205, align 4, !dbg !798, !tbaa !177
  %601 = load i32, ptr %202, align 4, !dbg !799, !tbaa !8
  %602 = load i32, ptr %145, align 4, !dbg !800, !tbaa !8
  %603 = icmp slt i32 %601, %602, !dbg !801
  br i1 %603, label %604, label %606, !dbg !802

604:                                              ; preds = %599
  %605 = load float, ptr %205, align 4, !dbg !803, !tbaa !177
  br label %607, !dbg !802

606:                                              ; preds = %599
  br label %607, !dbg !802

607:                                              ; preds = %606, %604
  %608 = phi contract float [ %605, %604 ], [ 0.000000e+00, %606 ], !dbg !802
  %609 = load i32, ptr %200, align 4, !dbg !804, !tbaa !8
  %610 = mul nsw i32 %609, 8, !dbg !805
  %611 = sext i32 %610 to i64, !dbg !806
  %612 = getelementptr inbounds i8, ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @LDS to ptr), i64 14336), i64 %611, !dbg !806
  %613 = load i32, ptr %201, align 4, !dbg !807, !tbaa !8
  %614 = sext i32 %613 to i64, !dbg !806
  %615 = getelementptr inbounds float, ptr %612, i64 %614, !dbg !806
  store float %608, ptr %615, align 1, !dbg !808, !tbaa !177
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %72) #25, !dbg !809
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %71) #25, !dbg !809
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %70) #25, !dbg !809
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %69) #25, !dbg !809
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %68) #25, !dbg !809
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %67) #25, !dbg !809
  br label %616, !dbg !809

616:                                              ; preds = %607
  %617 = load i32, ptr %199, align 4, !dbg !810, !tbaa !8
  %618 = add nsw i32 %617, 256, !dbg !810
  store i32 %618, ptr %199, align 4, !dbg !810, !tbaa !8
  br label %542, !dbg !753, !llvm.loop !811

619:                                              ; preds = %545
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %75) #25, !dbg !812
  %620 = load ptr, ptr %143, align 8, !dbg !813, !tbaa !130
  %621 = getelementptr inbounds i8, ptr %620, i64 8, !dbg !814
  store ptr %621, ptr %208, align 8, !dbg !815, !tbaa !28
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %76) #25, !dbg !816
  %622 = load ptr, ptr %142, align 8, !dbg !817, !tbaa !28
  %623 = getelementptr inbounds i8, ptr %622, i64 8, !dbg !818
  store ptr %623, ptr %209, align 8, !dbg !819, !tbaa !28
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %77) #25, !dbg !820
  store i32 0, ptr %210, align 4, !dbg !821, !tbaa !8
  br label %624, !dbg !820

624:                                              ; preds = %669, %619
  %625 = load i32, ptr %210, align 4, !dbg !822, !tbaa !8
  %626 = icmp slt i32 %625, 2, !dbg !823
  br i1 %626, label %628, label %627, !dbg !824

627:                                              ; preds = %624
  store i32 29, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %77) #25, !dbg !824
  br label %672

628:                                              ; preds = %624
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %78) #25, !dbg !825
  %629 = load ptr, ptr %208, align 8, !dbg !826, !tbaa !28
  %630 = load i32, ptr %210, align 4, !dbg !827, !tbaa !8
  %631 = sext i32 %630 to i64, !dbg !828
  %632 = getelementptr inbounds [2 x i32], ptr %173, i64 0, i64 %631, !dbg !828
  %633 = load i32, ptr %632, align 4, !dbg !828, !tbaa !8
  %634 = zext i32 %633 to i64, !dbg !829
  %635 = getelementptr inbounds nuw i8, ptr %629, i64 %634, !dbg !829
  %636 = load <2 x i32>, ptr %635, align 8, !dbg !829, !tbaa !47
  store <2 x i32> %636, ptr %211, align 8, !dbg !830, !tbaa !47
  %637 = load i32, ptr %210, align 4, !dbg !831, !tbaa !8
  %638 = sext i32 %637 to i64, !dbg !832
  %639 = getelementptr inbounds [2 x i32], ptr %176, i64 0, i64 %638, !dbg !832
  %640 = load i32, ptr %639, align 4, !dbg !832, !tbaa !8
  %641 = icmp ne i32 %640, 0, !dbg !832
  br i1 %641, label %642, label %644, !dbg !832

642:                                              ; preds = %628
  %643 = load <2 x i32>, ptr %211, align 8, !dbg !833, !tbaa !47
  br label %645, !dbg !832

644:                                              ; preds = %628
  br label %645, !dbg !832

645:                                              ; preds = %644, %642
  %646 = phi <2 x i32> [ %643, %642 ], [ zeroinitializer, %644 ], !dbg !832
  %647 = load ptr, ptr %161, align 8, !dbg !834, !tbaa !28
  %648 = load i32, ptr %210, align 4, !dbg !835, !tbaa !8
  %649 = sext i32 %648 to i64, !dbg !836
  %650 = getelementptr inbounds [2 x i32], ptr %175, i64 0, i64 %649, !dbg !836
  %651 = load i32, ptr %650, align 4, !dbg !836, !tbaa !8
  %652 = zext i32 %651 to i64, !dbg !837
  %653 = getelementptr inbounds nuw i8, ptr %647, i64 %652, !dbg !837
  store <2 x i32> %646, ptr %653, align 8, !dbg !838, !tbaa !47
  %654 = load ptr, ptr %209, align 8, !dbg !839, !tbaa !28
  %655 = load i32, ptr %210, align 4, !dbg !840, !tbaa !8
  %656 = sext i32 %655 to i64, !dbg !841
  %657 = getelementptr inbounds [2 x i32], ptr %174, i64 0, i64 %656, !dbg !841
  %658 = load i32, ptr %657, align 4, !dbg !841, !tbaa !8
  %659 = zext i32 %658 to i64, !dbg !842
  %660 = getelementptr inbounds nuw i8, ptr %654, i64 %659, !dbg !842
  %661 = load <2 x i32>, ptr %660, align 8, !dbg !842, !tbaa !47
  %662 = load ptr, ptr %162, align 8, !dbg !843, !tbaa !28
  %663 = load i32, ptr %210, align 4, !dbg !844, !tbaa !8
  %664 = sext i32 %663 to i64, !dbg !845
  %665 = getelementptr inbounds [2 x i32], ptr %175, i64 0, i64 %664, !dbg !845
  %666 = load i32, ptr %665, align 4, !dbg !845, !tbaa !8
  %667 = zext i32 %666 to i64, !dbg !846
  %668 = getelementptr inbounds nuw i8, ptr %662, i64 %667, !dbg !846
  store <2 x i32> %661, ptr %668, align 8, !dbg !847, !tbaa !47
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %78) #25, !dbg !848
  br label %669, !dbg !848

669:                                              ; preds = %645
  %670 = load i32, ptr %210, align 4, !dbg !849, !tbaa !8
  %671 = add nsw i32 %670, 1, !dbg !849
  store i32 %671, ptr %210, align 4, !dbg !849, !tbaa !8
  br label %624, !dbg !824, !llvm.loop !850

672:                                              ; preds = %627
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %76) #25, !dbg !851
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %75) #25, !dbg !851
  call void @_Z13__syncthreadsv() #26, !dbg !852
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %79) #25, !dbg !853
  store i32 0, ptr %212, align 4, !dbg !854, !tbaa !8
  br label %673, !dbg !853

673:                                              ; preds = %1246, %672
  %674 = load i32, ptr %212, align 4, !dbg !855, !tbaa !8
  %675 = load i32, ptr %160, align 4, !dbg !856, !tbaa !8
  %676 = icmp slt i32 %674, %675, !dbg !857
  br i1 %676, label %678, label %677, !dbg !858

677:                                              ; preds = %673
  store i32 32, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %79) #25, !dbg !858
  br label %1249

678:                                              ; preds = %673
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %80) #25, !dbg !859
  store i32 0, ptr %213, align 4, !dbg !860, !tbaa !8
  br label %679, !dbg !859

679:                                              ; preds = %1242, %678
  %680 = load i32, ptr %213, align 4, !dbg !861, !tbaa !8
  %681 = icmp slt i32 %680, 2, !dbg !862
  br i1 %681, label %683, label %682, !dbg !863

682:                                              ; preds = %679
  store i32 35, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %80) #25, !dbg !863
  br label %1245

683:                                              ; preds = %679
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %81) #25, !dbg !864
  %684 = load i32, ptr %212, align 4, !dbg !865, !tbaa !8
  %685 = mul nsw i32 2, %684, !dbg !866
  %686 = load i32, ptr %213, align 4, !dbg !867, !tbaa !8
  %687 = add nsw i32 %685, %686, !dbg !868
  store i32 %687, ptr %214, align 4, !dbg !869, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %82) #25, !dbg !870
  %688 = load i32, ptr %213, align 4, !dbg !871, !tbaa !8
  %689 = icmp eq i32 %688, 1, !dbg !872
  br i1 %689, label %690, label %695, !dbg !873

690:                                              ; preds = %683
  %691 = load i32, ptr %212, align 4, !dbg !874, !tbaa !8
  %692 = add nsw i32 %691, 1, !dbg !875
  %693 = load i32, ptr %160, align 4, !dbg !876, !tbaa !8
  %694 = icmp eq i32 %692, %693, !dbg !877
  br label %695

695:                                              ; preds = %690, %683
  %696 = phi i1 [ false, %683 ], [ %694, %690 ], !dbg !878
  %697 = zext i1 %696 to i8, !dbg !879
  store i8 %697, ptr %215, align 1, !dbg !879, !tbaa !880
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %83) #25, !dbg !882
  %698 = load i32, ptr %213, align 4, !dbg !883, !tbaa !8
  %699 = icmp eq i32 %698, 0, !dbg !884
  %700 = zext i1 %699 to i64, !dbg !885
  %701 = select i1 %699, i32 12288, i32 13312, !dbg !885
  %702 = sext i32 %701 to i64, !dbg !886
  %703 = getelementptr inbounds i8, ptr addrspacecast (ptr addrspace(3) @LDS to ptr), i64 %702, !dbg !886
  store ptr %703, ptr %216, align 8, !dbg !887, !tbaa !28
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %84) #25, !dbg !888
  %704 = load i32, ptr %213, align 4, !dbg !889, !tbaa !8
  %705 = icmp eq i32 %704, 0, !dbg !890
  %706 = zext i1 %705 to i64, !dbg !891
  %707 = select i1 %705, i32 14336, i32 15360, !dbg !891
  %708 = sext i32 %707 to i64, !dbg !892
  %709 = getelementptr inbounds i8, ptr addrspacecast (ptr addrspace(3) @LDS to ptr), i64 %708, !dbg !892
  store ptr %709, ptr %217, align 8, !dbg !893, !tbaa !28
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %85) #25, !dbg !894
  store i32 0, ptr %218, align 4, !dbg !895, !tbaa !8
  br label %710, !dbg !894

710:                                              ; preds = %745, %695
  %711 = load i32, ptr %218, align 4, !dbg !896, !tbaa !8
  %712 = icmp slt i32 %711, 2, !dbg !897
  br i1 %712, label %714, label %713, !dbg !898

713:                                              ; preds = %710
  store i32 38, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %85) #25, !dbg !898
  br label %748

714:                                              ; preds = %710
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %86) #25, !dbg !899
  store i32 0, ptr %219, align 4, !dbg !900, !tbaa !8
  br label %715, !dbg !899

715:                                              ; preds = %741, %714
  %716 = load i32, ptr %219, align 4, !dbg !901, !tbaa !8
  %717 = icmp slt i32 %716, 8, !dbg !902
  br i1 %717, label %719, label %718, !dbg !903

718:                                              ; preds = %715
  store i32 41, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %86) #25, !dbg !903
  br label %744

719:                                              ; preds = %715
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %87) #25, !dbg !904
  %720 = load i32, ptr %158, align 4, !dbg !905, !tbaa !8
  %721 = load i32, ptr %218, align 4, !dbg !906, !tbaa !8
  %722 = mul nsw i32 %721, 16, !dbg !907
  %723 = add nsw i32 %720, %722, !dbg !908
  %724 = load i32, ptr %155, align 4, !dbg !909, !tbaa !8
  %725 = mul nsw i32 8, %724, !dbg !910
  %726 = add nsw i32 %723, %725, !dbg !911
  %727 = load i32, ptr %219, align 4, !dbg !912, !tbaa !8
  %728 = add nsw i32 %726, %727, !dbg !913
  %729 = mul i32 %728, 8, !dbg !914
  store i32 %729, ptr %220, align 4, !dbg !915, !tbaa !8
  %730 = load ptr, ptr %217, align 8, !dbg !916, !tbaa !28
  %731 = load i32, ptr %220, align 4, !dbg !917, !tbaa !8
  %732 = zext i32 %731 to i64, !dbg !918
  %733 = getelementptr inbounds nuw i8, ptr %730, i64 %732, !dbg !918
  %734 = load float, ptr %733, align 4, !dbg !918, !tbaa !177
  %735 = load i32, ptr %218, align 4, !dbg !919, !tbaa !8
  %736 = sext i32 %735 to i64, !dbg !920
  %737 = getelementptr inbounds [2 x <8 x float>], ptr %188, i64 0, i64 %736, !dbg !920
  %738 = load i32, ptr %219, align 4, !dbg !921, !tbaa !8
  %739 = load <8 x float>, ptr %737, align 32, !dbg !922
  %740 = insertelement <8 x float> %739, float %734, i32 %738, !dbg !922
  store <8 x float> %740, ptr %737, align 32, !dbg !922
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %87) #25, !dbg !923
  br label %741, !dbg !923

741:                                              ; preds = %719
  %742 = load i32, ptr %219, align 4, !dbg !924, !tbaa !8
  %743 = add nsw i32 %742, 1, !dbg !924
  store i32 %743, ptr %219, align 4, !dbg !924, !tbaa !8
  br label %715, !dbg !903, !llvm.loop !925

744:                                              ; preds = %718
  br label %745, !dbg !926

745:                                              ; preds = %744
  %746 = load i32, ptr %218, align 4, !dbg !927, !tbaa !8
  %747 = add nsw i32 %746, 1, !dbg !927
  store i32 %747, ptr %218, align 4, !dbg !927, !tbaa !8
  br label %710, !dbg !898, !llvm.loop !928

748:                                              ; preds = %713
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %88) #25, !dbg !929
  store i32 0, ptr %221, align 4, !dbg !930, !tbaa !8
  br label %749, !dbg !929

749:                                              ; preds = %769, %748
  %750 = load i32, ptr %221, align 4, !dbg !931, !tbaa !8
  %751 = icmp slt i32 %750, 4, !dbg !932
  br i1 %751, label %753, label %752, !dbg !933

752:                                              ; preds = %749
  store i32 44, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %88) #25, !dbg !933
  br label %772

753:                                              ; preds = %749
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %89) #25, !dbg !934
  %754 = load i32, ptr %159, align 4, !dbg !935, !tbaa !8
  %755 = load i32, ptr %221, align 4, !dbg !936, !tbaa !8
  %756 = mul nsw i32 %755, 16, !dbg !937
  %757 = add nsw i32 %754, %756, !dbg !938
  %758 = load i32, ptr %154, align 4, !dbg !939, !tbaa !8
  %759 = add nsw i32 %757, %758, !dbg !940
  %760 = mul i32 %759, 8, !dbg !941
  store i32 %760, ptr %222, align 4, !dbg !942, !tbaa !8
  %761 = load ptr, ptr %216, align 8, !dbg !943, !tbaa !28
  %762 = load i32, ptr %222, align 4, !dbg !944, !tbaa !8
  %763 = zext i32 %762 to i64, !dbg !945
  %764 = getelementptr inbounds nuw i8, ptr %761, i64 %763, !dbg !945
  %765 = load float, ptr %764, align 4, !dbg !945, !tbaa !177
  %766 = load i32, ptr %221, align 4, !dbg !946, !tbaa !8
  %767 = sext i32 %766 to i64, !dbg !947
  %768 = getelementptr inbounds [4 x float], ptr %189, i64 0, i64 %767, !dbg !947
  store float %765, ptr %768, align 4, !dbg !948, !tbaa !177
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %89) #25, !dbg !949
  br label %769, !dbg !949

769:                                              ; preds = %753
  %770 = load i32, ptr %221, align 4, !dbg !950, !tbaa !8
  %771 = add nsw i32 %770, 1, !dbg !950
  store i32 %771, ptr %221, align 4, !dbg !950, !tbaa !8
  br label %749, !dbg !933, !llvm.loop !951

772:                                              ; preds = %752
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %90) #25, !dbg !952
  %773 = load ptr, ptr %143, align 8, !dbg !953, !tbaa !130
  %774 = load i32, ptr %214, align 4, !dbg !954, !tbaa !8
  %775 = sext i32 %774 to i64, !dbg !954
  %776 = load i32, ptr %147, align 4, !dbg !955, !tbaa !8
  %777 = sext i32 %776 to i64, !dbg !955
  %778 = mul nsw i64 %775, %777, !dbg !956
  %779 = getelementptr inbounds %struct.block_i4_128, ptr %773, i64 %778, !dbg !957
  %780 = getelementptr inbounds i8, ptr %779, i64 8, !dbg !958
  %781 = getelementptr inbounds i8, ptr %780, i64 32, !dbg !959
  store ptr %781, ptr %223, align 8, !dbg !960, !tbaa !28
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %91) #25, !dbg !961
  %782 = load ptr, ptr %142, align 8, !dbg !962, !tbaa !28
  %783 = load i32, ptr %212, align 4, !dbg !963, !tbaa !8
  %784 = sext i32 %783 to i64, !dbg !963
  %785 = mul nsw i64 %784, 136, !dbg !964
  %786 = getelementptr inbounds i8, ptr %782, i64 %785, !dbg !965
  %787 = getelementptr inbounds i8, ptr %786, i64 8, !dbg !966
  %788 = load i32, ptr %213, align 4, !dbg !967, !tbaa !8
  %789 = sext i32 %788 to i64, !dbg !967
  %790 = mul nsw i64 %789, 64, !dbg !968
  %791 = getelementptr inbounds i8, ptr %787, i64 %790, !dbg !969
  %792 = getelementptr inbounds i8, ptr %791, i64 32, !dbg !970
  store ptr %792, ptr %224, align 8, !dbg !971, !tbaa !28
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %92) #25, !dbg !972
  store i32 0, ptr %225, align 4, !dbg !973, !tbaa !8
  br label %793, !dbg !972

793:                                              ; preds = %830, %772
  %794 = load i32, ptr %225, align 4, !dbg !974, !tbaa !8
  %795 = icmp slt i32 %794, 2, !dbg !975
  br i1 %795, label %797, label %796, !dbg !976

796:                                              ; preds = %793
  store i32 47, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %92) #25, !dbg !976
  br label %833

797:                                              ; preds = %793
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %93) #25, !dbg !977
  %798 = load ptr, ptr %223, align 8, !dbg !978, !tbaa !28
  %799 = load i32, ptr %225, align 4, !dbg !979, !tbaa !8
  %800 = sext i32 %799 to i64, !dbg !980
  %801 = getelementptr inbounds [2 x i32], ptr %173, i64 0, i64 %800, !dbg !980
  %802 = load i32, ptr %801, align 4, !dbg !980, !tbaa !8
  %803 = zext i32 %802 to i64, !dbg !981
  %804 = getelementptr inbounds nuw i8, ptr %798, i64 %803, !dbg !981
  %805 = load <2 x i32>, ptr %804, align 8, !dbg !981, !tbaa !47
  store <2 x i32> %805, ptr %226, align 8, !dbg !982, !tbaa !47
  %806 = load i32, ptr %225, align 4, !dbg !983, !tbaa !8
  %807 = sext i32 %806 to i64, !dbg !984
  %808 = getelementptr inbounds [2 x i32], ptr %176, i64 0, i64 %807, !dbg !984
  %809 = load i32, ptr %808, align 4, !dbg !984, !tbaa !8
  %810 = icmp ne i32 %809, 0, !dbg !984
  br i1 %810, label %811, label %813, !dbg !984

811:                                              ; preds = %797
  %812 = load <2 x i32>, ptr %226, align 8, !dbg !985, !tbaa !47
  br label %814, !dbg !984

813:                                              ; preds = %797
  br label %814, !dbg !984

814:                                              ; preds = %813, %811
  %815 = phi <2 x i32> [ %812, %811 ], [ zeroinitializer, %813 ], !dbg !984
  %816 = load i32, ptr %225, align 4, !dbg !986, !tbaa !8
  %817 = sext i32 %816 to i64, !dbg !987
  %818 = getelementptr inbounds [2 x <2 x i32>], ptr %190, i64 0, i64 %817, !dbg !987
  store <2 x i32> %815, ptr %818, align 8, !dbg !988, !tbaa !47
  %819 = load ptr, ptr %224, align 8, !dbg !989, !tbaa !28
  %820 = load i32, ptr %225, align 4, !dbg !990, !tbaa !8
  %821 = sext i32 %820 to i64, !dbg !991
  %822 = getelementptr inbounds [2 x i32], ptr %174, i64 0, i64 %821, !dbg !991
  %823 = load i32, ptr %822, align 4, !dbg !991, !tbaa !8
  %824 = zext i32 %823 to i64, !dbg !992
  %825 = getelementptr inbounds nuw i8, ptr %819, i64 %824, !dbg !992
  %826 = load <2 x i32>, ptr %825, align 8, !dbg !992, !tbaa !47
  %827 = load i32, ptr %225, align 4, !dbg !993, !tbaa !8
  %828 = sext i32 %827 to i64, !dbg !994
  %829 = getelementptr inbounds [2 x <2 x i32>], ptr %191, i64 0, i64 %828, !dbg !994
  store <2 x i32> %826, ptr %829, align 8, !dbg !995, !tbaa !47
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %93) #25, !dbg !996
  br label %830, !dbg !996

830:                                              ; preds = %814
  %831 = load i32, ptr %225, align 4, !dbg !997, !tbaa !8
  %832 = add nsw i32 %831, 1, !dbg !997
  store i32 %832, ptr %225, align 4, !dbg !997, !tbaa !8
  br label %793, !dbg !976, !llvm.loop !998

833:                                              ; preds = %796
  %834 = load ptr, ptr %162, align 8, !dbg !999, !tbaa !28
  %835 = load ptr, ptr %161, align 8, !dbg !1000, !tbaa !28
  %836 = getelementptr inbounds [4 x [2 x <8 x float>]], ptr %172, i64 0, i64 0, !dbg !1001
  %837 = getelementptr inbounds [2 x <8 x float>], ptr %188, i64 0, i64 0, !dbg !1002
  call void @_ZL21iu4_bundle_single_accILi0EEvPKcS1_RA2_A2_KjRA4_S3_PA2_Dv8_fPKS8_RA4_KfRKDv8_i(ptr noundef %834, ptr noundef %835, ptr noundef nonnull align 4 dereferenceable(16) %164, ptr noundef nonnull align 4 dereferenceable(32) %163, ptr noundef %836, ptr noundef %837, ptr noundef nonnull align 4 dereferenceable(16) %189, ptr noundef nonnull align 32 dereferenceable(32) %187) #26, !dbg !1003
  %838 = load ptr, ptr %162, align 8, !dbg !1004, !tbaa !28
  %839 = load ptr, ptr %161, align 8, !dbg !1005, !tbaa !28
  %840 = getelementptr inbounds [4 x [2 x <8 x float>]], ptr %172, i64 0, i64 0, !dbg !1006
  %841 = getelementptr inbounds [2 x <8 x float>], ptr %188, i64 0, i64 0, !dbg !1007
  call void @_ZL21iu4_bundle_single_accILi1EEvPKcS1_RA2_A2_KjRA4_S3_PA2_Dv8_fPKS8_RA4_KfRKDv8_i(ptr noundef %838, ptr noundef %839, ptr noundef nonnull align 4 dereferenceable(16) %164, ptr noundef nonnull align 4 dereferenceable(32) %163, ptr noundef %840, ptr noundef %841, ptr noundef nonnull align 4 dereferenceable(16) %189, ptr noundef nonnull align 32 dereferenceable(32) %187) #26, !dbg !1008
  call void @_Z13__syncthreadsv() #26, !dbg !1009
  store ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @LDS to ptr), i64 8192), ptr %162, align 8, !dbg !1010, !tbaa !28
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %94) #25, !dbg !1011
  store i32 0, ptr %227, align 4, !dbg !1012, !tbaa !8
  br label %842, !dbg !1011

842:                                              ; preds = %869, %833
  %843 = load i32, ptr %227, align 4, !dbg !1013, !tbaa !8
  %844 = icmp slt i32 %843, 2, !dbg !1014
  br i1 %844, label %846, label %845, !dbg !1015

845:                                              ; preds = %842
  store i32 50, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %94) #25, !dbg !1015
  br label %872

846:                                              ; preds = %842
  %847 = load i32, ptr %227, align 4, !dbg !1016, !tbaa !8
  %848 = sext i32 %847 to i64, !dbg !1017
  %849 = getelementptr inbounds [2 x <2 x i32>], ptr %190, i64 0, i64 %848, !dbg !1017
  %850 = load <2 x i32>, ptr %849, align 8, !dbg !1017, !tbaa !47
  %851 = load ptr, ptr %161, align 8, !dbg !1018, !tbaa !28
  %852 = load i32, ptr %227, align 4, !dbg !1019, !tbaa !8
  %853 = sext i32 %852 to i64, !dbg !1020
  %854 = getelementptr inbounds [2 x i32], ptr %175, i64 0, i64 %853, !dbg !1020
  %855 = load i32, ptr %854, align 4, !dbg !1020, !tbaa !8
  %856 = zext i32 %855 to i64, !dbg !1021
  %857 = getelementptr inbounds nuw i8, ptr %851, i64 %856, !dbg !1021
  store <2 x i32> %850, ptr %857, align 8, !dbg !1022, !tbaa !47
  %858 = load i32, ptr %227, align 4, !dbg !1023, !tbaa !8
  %859 = sext i32 %858 to i64, !dbg !1024
  %860 = getelementptr inbounds [2 x <2 x i32>], ptr %191, i64 0, i64 %859, !dbg !1024
  %861 = load <2 x i32>, ptr %860, align 8, !dbg !1024, !tbaa !47
  %862 = load ptr, ptr %162, align 8, !dbg !1025, !tbaa !28
  %863 = load i32, ptr %227, align 4, !dbg !1026, !tbaa !8
  %864 = sext i32 %863 to i64, !dbg !1027
  %865 = getelementptr inbounds [2 x i32], ptr %175, i64 0, i64 %864, !dbg !1027
  %866 = load i32, ptr %865, align 4, !dbg !1027, !tbaa !8
  %867 = zext i32 %866 to i64, !dbg !1028
  %868 = getelementptr inbounds nuw i8, ptr %862, i64 %867, !dbg !1028
  store <2 x i32> %861, ptr %868, align 8, !dbg !1029, !tbaa !47
  br label %869, !dbg !1030

869:                                              ; preds = %846
  %870 = load i32, ptr %227, align 4, !dbg !1031, !tbaa !8
  %871 = add nsw i32 %870, 1, !dbg !1031
  store i32 %871, ptr %227, align 4, !dbg !1031, !tbaa !8
  br label %842, !dbg !1015, !llvm.loop !1032

872:                                              ; preds = %845
  call void @_Z13__syncthreadsv() #26, !dbg !1033
  %873 = load i8, ptr %215, align 1, !dbg !1034, !tbaa !880, !range !1035, !noundef !20
  %874 = trunc i8 %873 to i1, !dbg !1034
  br i1 %874, label %1001, label %875, !dbg !1036

875:                                              ; preds = %872
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %95) #25, !dbg !1037
  %876 = load i32, ptr %214, align 4, !dbg !1038, !tbaa !8
  %877 = add nsw i32 %876, 1, !dbg !1039
  store i32 %877, ptr %228, align 4, !dbg !1040, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %96) #25, !dbg !1041
  %878 = load ptr, ptr %143, align 8, !dbg !1042, !tbaa !130
  %879 = load i32, ptr %228, align 4, !dbg !1043, !tbaa !8
  %880 = sext i32 %879 to i64, !dbg !1043
  %881 = load i32, ptr %147, align 4, !dbg !1044, !tbaa !8
  %882 = sext i32 %881 to i64, !dbg !1044
  %883 = mul nsw i64 %880, %882, !dbg !1045
  %884 = getelementptr inbounds %struct.block_i4_128, ptr %878, i64 %883, !dbg !1046
  %885 = getelementptr inbounds i8, ptr %884, i64 8, !dbg !1047
  store ptr %885, ptr %229, align 8, !dbg !1048, !tbaa !28
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %97) #25, !dbg !1049
  store i32 0, ptr %230, align 4, !dbg !1050, !tbaa !8
  br label %886, !dbg !1049

886:                                              ; preds = %902, %875
  %887 = load i32, ptr %230, align 4, !dbg !1051, !tbaa !8
  %888 = icmp slt i32 %887, 2, !dbg !1052
  br i1 %888, label %890, label %889, !dbg !1053

889:                                              ; preds = %886
  store i32 53, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %97) #25, !dbg !1053
  br label %905

890:                                              ; preds = %886
  %891 = load ptr, ptr %229, align 8, !dbg !1054, !tbaa !28
  %892 = load i32, ptr %230, align 4, !dbg !1055, !tbaa !8
  %893 = sext i32 %892 to i64, !dbg !1056
  %894 = getelementptr inbounds [2 x i32], ptr %173, i64 0, i64 %893, !dbg !1056
  %895 = load i32, ptr %894, align 4, !dbg !1056, !tbaa !8
  %896 = zext i32 %895 to i64, !dbg !1057
  %897 = getelementptr inbounds nuw i8, ptr %891, i64 %896, !dbg !1057
  %898 = load <2 x i32>, ptr %897, align 8, !dbg !1057, !tbaa !47
  %899 = load i32, ptr %230, align 4, !dbg !1058, !tbaa !8
  %900 = sext i32 %899 to i64, !dbg !1059
  %901 = getelementptr inbounds [2 x <2 x i32>], ptr %190, i64 0, i64 %900, !dbg !1059
  store <2 x i32> %898, ptr %901, align 8, !dbg !1060, !tbaa !47
  br label %902, !dbg !1061

902:                                              ; preds = %890
  %903 = load i32, ptr %230, align 4, !dbg !1062, !tbaa !8
  %904 = add nsw i32 %903, 1, !dbg !1062
  store i32 %904, ptr %230, align 4, !dbg !1062, !tbaa !8
  br label %886, !dbg !1053, !llvm.loop !1063

905:                                              ; preds = %889
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %98) #25, !dbg !1064
  %906 = load i32, ptr %212, align 4, !dbg !1065, !tbaa !8
  %907 = load i32, ptr %213, align 4, !dbg !1066, !tbaa !8
  %908 = add nsw i32 %906, %907, !dbg !1067
  store i32 %908, ptr %231, align 4, !dbg !1068, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %99) #25, !dbg !1069
  %909 = load i32, ptr %213, align 4, !dbg !1070, !tbaa !8
  %910 = xor i32 %909, 1, !dbg !1071
  store i32 %910, ptr %232, align 4, !dbg !1072, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %100) #25, !dbg !1073
  %911 = load ptr, ptr %142, align 8, !dbg !1074, !tbaa !28
  %912 = load i32, ptr %231, align 4, !dbg !1075, !tbaa !8
  %913 = sext i32 %912 to i64, !dbg !1075
  %914 = mul nsw i64 %913, 136, !dbg !1076
  %915 = getelementptr inbounds i8, ptr %911, i64 %914, !dbg !1077
  %916 = getelementptr inbounds i8, ptr %915, i64 8, !dbg !1078
  %917 = load i32, ptr %232, align 4, !dbg !1079, !tbaa !8
  %918 = sext i32 %917 to i64, !dbg !1079
  %919 = mul nsw i64 %918, 64, !dbg !1080
  %920 = getelementptr inbounds i8, ptr %916, i64 %919, !dbg !1081
  store ptr %920, ptr %233, align 8, !dbg !1082, !tbaa !28
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %101) #25, !dbg !1083
  store i32 0, ptr %234, align 4, !dbg !1084, !tbaa !8
  br label %921, !dbg !1083

921:                                              ; preds = %937, %905
  %922 = load i32, ptr %234, align 4, !dbg !1085, !tbaa !8
  %923 = icmp slt i32 %922, 2, !dbg !1086
  br i1 %923, label %925, label %924, !dbg !1087

924:                                              ; preds = %921
  store i32 56, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %101) #25, !dbg !1087
  br label %940

925:                                              ; preds = %921
  %926 = load ptr, ptr %233, align 8, !dbg !1088, !tbaa !28
  %927 = load i32, ptr %234, align 4, !dbg !1089, !tbaa !8
  %928 = sext i32 %927 to i64, !dbg !1090
  %929 = getelementptr inbounds [2 x i32], ptr %174, i64 0, i64 %928, !dbg !1090
  %930 = load i32, ptr %929, align 4, !dbg !1090, !tbaa !8
  %931 = zext i32 %930 to i64, !dbg !1091
  %932 = getelementptr inbounds nuw i8, ptr %926, i64 %931, !dbg !1091
  %933 = load <2 x i32>, ptr %932, align 8, !dbg !1091, !tbaa !47
  %934 = load i32, ptr %234, align 4, !dbg !1092, !tbaa !8
  %935 = sext i32 %934 to i64, !dbg !1093
  %936 = getelementptr inbounds [2 x <2 x i32>], ptr %191, i64 0, i64 %935, !dbg !1093
  store <2 x i32> %933, ptr %936, align 8, !dbg !1094, !tbaa !47
  br label %937, !dbg !1095

937:                                              ; preds = %925
  %938 = load i32, ptr %234, align 4, !dbg !1096, !tbaa !8
  %939 = add nsw i32 %938, 1, !dbg !1096
  store i32 %939, ptr %234, align 4, !dbg !1096, !tbaa !8
  br label %921, !dbg !1087, !llvm.loop !1097

940:                                              ; preds = %924
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %102) #25, !dbg !1098
  %941 = load i32, ptr %152, align 4, !dbg !1099, !tbaa !8
  %942 = load i32, ptr %148, align 4, !dbg !1100, !tbaa !8
  %943 = ashr i32 %942, 1, !dbg !1101
  %944 = add nsw i32 %941, %943, !dbg !1102
  store i32 %944, ptr %235, align 4, !dbg !1103, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %103) #25, !dbg !1104
  %945 = load i32, ptr %235, align 4, !dbg !1105, !tbaa !8
  %946 = load i32, ptr %147, align 4, !dbg !1106, !tbaa !8
  %947 = icmp slt i32 %945, %946, !dbg !1107
  br i1 %947, label %948, label %950, !dbg !1108

948:                                              ; preds = %940
  %949 = load i32, ptr %235, align 4, !dbg !1109, !tbaa !8
  br label %953, !dbg !1108

950:                                              ; preds = %940
  %951 = load i32, ptr %147, align 4, !dbg !1110, !tbaa !8
  %952 = sub nsw i32 %951, 1, !dbg !1111
  br label %953, !dbg !1108

953:                                              ; preds = %950, %948
  %954 = phi i32 [ %949, %948 ], [ %952, %950 ], !dbg !1108
  store i32 %954, ptr %236, align 4, !dbg !1112, !tbaa !8
  %955 = load ptr, ptr %143, align 8, !dbg !1113, !tbaa !130
  %956 = load i32, ptr %228, align 4, !dbg !1114, !tbaa !8
  %957 = sext i32 %956 to i64, !dbg !1114
  %958 = load i32, ptr %147, align 4, !dbg !1115, !tbaa !8
  %959 = sext i32 %958 to i64, !dbg !1115
  %960 = mul nsw i64 %957, %959, !dbg !1116
  %961 = getelementptr inbounds %struct.block_i4_128, ptr %955, i64 %960, !dbg !1117
  %962 = load i32, ptr %236, align 4, !dbg !1118, !tbaa !8
  %963 = sext i32 %962 to i64, !dbg !1117
  %964 = getelementptr inbounds %struct.block_i4_128, ptr %961, i64 %963, !dbg !1117
  %965 = load i32, ptr %148, align 4, !dbg !1119, !tbaa !8
  %966 = and i32 %965, 1, !dbg !1120
  %967 = sext i32 %966 to i64, !dbg !1117
  %968 = getelementptr inbounds i32, ptr %964, i64 %967, !dbg !1117
  %969 = load i32, ptr %968, align 4, !dbg !1117, !tbaa !8
  store i32 %969, ptr %192, align 4, !dbg !1121, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %104) #25, !dbg !1122
  %970 = load i32, ptr %151, align 4, !dbg !1123, !tbaa !8
  %971 = load i32, ptr %148, align 4, !dbg !1124, !tbaa !8
  %972 = ashr i32 %971, 1, !dbg !1125
  %973 = add nsw i32 %970, %972, !dbg !1126
  store i32 %973, ptr %237, align 4, !dbg !1127, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %105) #25, !dbg !1128
  %974 = load i32, ptr %237, align 4, !dbg !1129, !tbaa !8
  %975 = load i32, ptr %145, align 4, !dbg !1130, !tbaa !8
  %976 = icmp slt i32 %974, %975, !dbg !1131
  br i1 %976, label %977, label %979, !dbg !1132

977:                                              ; preds = %953
  %978 = load i32, ptr %237, align 4, !dbg !1133, !tbaa !8
  br label %982, !dbg !1132

979:                                              ; preds = %953
  %980 = load i32, ptr %145, align 4, !dbg !1134, !tbaa !8
  %981 = sub nsw i32 %980, 1, !dbg !1135
  br label %982, !dbg !1132

982:                                              ; preds = %979, %977
  %983 = phi i32 [ %978, %977 ], [ %981, %979 ], !dbg !1132
  store i32 %983, ptr %238, align 4, !dbg !1136, !tbaa !8
  %984 = load ptr, ptr %142, align 8, !dbg !1137, !tbaa !28
  %985 = load i32, ptr %238, align 4, !dbg !1138, !tbaa !8
  %986 = sext i32 %985 to i64, !dbg !1138
  %987 = load i32, ptr %160, align 4, !dbg !1139, !tbaa !8
  %988 = sext i32 %987 to i64, !dbg !1139
  %989 = mul nsw i64 %986, %988, !dbg !1140
  %990 = mul nsw i64 %989, 136, !dbg !1141
  %991 = getelementptr inbounds i8, ptr %984, i64 %990, !dbg !1142
  %992 = load i32, ptr %231, align 4, !dbg !1143, !tbaa !8
  %993 = mul nsw i32 %992, 136, !dbg !1144
  %994 = sext i32 %993 to i64, !dbg !1142
  %995 = getelementptr inbounds i8, ptr %991, i64 %994, !dbg !1142
  %996 = load i32, ptr %232, align 4, !dbg !1145, !tbaa !8
  %997 = mul nsw i32 %996, 4, !dbg !1146
  %998 = sext i32 %997 to i64, !dbg !1142
  %999 = getelementptr inbounds i8, ptr %995, i64 %998, !dbg !1142
  %1000 = load i32, ptr %999, align 4, !dbg !1142, !tbaa !8
  store i32 %1000, ptr %193, align 4, !dbg !1147, !tbaa !8
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %105) #25, !dbg !1148
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %104) #25, !dbg !1148
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %103) #25, !dbg !1148
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %102) #25, !dbg !1148
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %100) #25, !dbg !1148
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %99) #25, !dbg !1148
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %98) #25, !dbg !1148
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %96) #25, !dbg !1148
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %95) #25, !dbg !1148
  br label %1001, !dbg !1148

1001:                                             ; preds = %982, %872
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %91) #25, !dbg !1149
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %90) #25, !dbg !1149
  %1002 = load ptr, ptr %162, align 8, !dbg !1150, !tbaa !28
  %1003 = load ptr, ptr %161, align 8, !dbg !1151, !tbaa !28
  %1004 = getelementptr inbounds [4 x [2 x <8 x float>]], ptr %172, i64 0, i64 0, !dbg !1152
  %1005 = getelementptr inbounds [2 x <8 x float>], ptr %188, i64 0, i64 0, !dbg !1153
  call void @_ZL21iu4_bundle_single_accILi0EEvPKcS1_RA2_A2_KjRA4_S3_PA2_Dv8_fPKS8_RA4_KfRKDv8_i(ptr noundef %1002, ptr noundef %1003, ptr noundef nonnull align 4 dereferenceable(16) %164, ptr noundef nonnull align 4 dereferenceable(32) %163, ptr noundef %1004, ptr noundef %1005, ptr noundef nonnull align 4 dereferenceable(16) %189, ptr noundef nonnull align 32 dereferenceable(32) %187) #26, !dbg !1154
  %1006 = load ptr, ptr %162, align 8, !dbg !1155, !tbaa !28
  %1007 = load ptr, ptr %161, align 8, !dbg !1156, !tbaa !28
  %1008 = getelementptr inbounds [4 x [2 x <8 x float>]], ptr %172, i64 0, i64 0, !dbg !1157
  %1009 = getelementptr inbounds [2 x <8 x float>], ptr %188, i64 0, i64 0, !dbg !1158
  call void @_ZL21iu4_bundle_single_accILi1EEvPKcS1_RA2_A2_KjRA4_S3_PA2_Dv8_fPKS8_RA4_KfRKDv8_i(ptr noundef %1006, ptr noundef %1007, ptr noundef nonnull align 4 dereferenceable(16) %164, ptr noundef nonnull align 4 dereferenceable(32) %163, ptr noundef %1008, ptr noundef %1009, ptr noundef nonnull align 4 dereferenceable(16) %189, ptr noundef nonnull align 32 dereferenceable(32) %187) #26, !dbg !1159
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %106) #25, !dbg !1160
  store i32 0, ptr %239, align 4, !dbg !1161, !tbaa !8
  br label %1010, !dbg !1160

1010:                                             ; preds = %1107, %1001
  %1011 = load i32, ptr %239, align 4, !dbg !1162, !tbaa !8
  %1012 = icmp slt i32 %1011, 2, !dbg !1163
  br i1 %1012, label %1014, label %1013, !dbg !1164

1013:                                             ; preds = %1010
  store i32 59, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %106) #25, !dbg !1164
  br label %1110

1014:                                             ; preds = %1010
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %107) #25, !dbg !1165
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %108) #25, !dbg !1166
  store i32 0, ptr %241, align 4, !dbg !1167, !tbaa !8
  br label %1015, !dbg !1166

1015:                                             ; preds = %1039, %1014
  %1016 = load i32, ptr %241, align 4, !dbg !1168, !tbaa !8
  %1017 = icmp slt i32 %1016, 8, !dbg !1169
  br i1 %1017, label %1019, label %1018, !dbg !1170

1018:                                             ; preds = %1015
  store i32 62, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %108) #25, !dbg !1170
  br label %1042

1019:                                             ; preds = %1015
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %109) #25, !dbg !1171
  %1020 = load i32, ptr %158, align 4, !dbg !1172, !tbaa !8
  %1021 = load i32, ptr %239, align 4, !dbg !1173, !tbaa !8
  %1022 = mul nsw i32 %1021, 16, !dbg !1174
  %1023 = add nsw i32 %1020, %1022, !dbg !1175
  %1024 = load i32, ptr %155, align 4, !dbg !1176, !tbaa !8
  %1025 = mul nsw i32 8, %1024, !dbg !1177
  %1026 = add nsw i32 %1023, %1025, !dbg !1178
  %1027 = load i32, ptr %241, align 4, !dbg !1179, !tbaa !8
  %1028 = add nsw i32 %1026, %1027, !dbg !1180
  %1029 = mul i32 %1028, 8, !dbg !1181
  store i32 %1029, ptr %242, align 4, !dbg !1182, !tbaa !8
  %1030 = load ptr, ptr %217, align 8, !dbg !1183, !tbaa !28
  %1031 = load i32, ptr %242, align 4, !dbg !1184, !tbaa !8
  %1032 = zext i32 %1031 to i64, !dbg !1185
  %1033 = getelementptr inbounds nuw i8, ptr %1030, i64 %1032, !dbg !1185
  %1034 = getelementptr inbounds nuw i8, ptr %1033, i64 4, !dbg !1185
  %1035 = load float, ptr %1034, align 4, !dbg !1185, !tbaa !177
  %1036 = load i32, ptr %241, align 4, !dbg !1186, !tbaa !8
  %1037 = sext i32 %1036 to i64, !dbg !1187
  %1038 = getelementptr inbounds [8 x float], ptr %240, i64 0, i64 %1037, !dbg !1187
  store float %1035, ptr %1038, align 4, !dbg !1188, !tbaa !177
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %109) #25, !dbg !1189
  br label %1039, !dbg !1189

1039:                                             ; preds = %1019
  %1040 = load i32, ptr %241, align 4, !dbg !1190, !tbaa !8
  %1041 = add nsw i32 %1040, 1, !dbg !1190
  store i32 %1041, ptr %241, align 4, !dbg !1190, !tbaa !8
  br label %1015, !dbg !1170, !llvm.loop !1191

1042:                                             ; preds = %1018
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %110) #25, !dbg !1192
  store i32 0, ptr %243, align 4, !dbg !1193, !tbaa !8
  br label %1043, !dbg !1192

1043:                                             ; preds = %1103, %1042
  %1044 = load i32, ptr %243, align 4, !dbg !1194, !tbaa !8
  %1045 = icmp slt i32 %1044, 4, !dbg !1195
  br i1 %1045, label %1047, label %1046, !dbg !1196

1046:                                             ; preds = %1043
  store i32 65, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %110) #25, !dbg !1196
  br label %1106

1047:                                             ; preds = %1043
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %111) #25, !dbg !1197
  %1048 = load i32, ptr %159, align 4, !dbg !1198, !tbaa !8
  %1049 = load i32, ptr %243, align 4, !dbg !1199, !tbaa !8
  %1050 = mul nsw i32 %1049, 16, !dbg !1200
  %1051 = add nsw i32 %1048, %1050, !dbg !1201
  %1052 = load i32, ptr %154, align 4, !dbg !1202, !tbaa !8
  %1053 = add nsw i32 %1051, %1052, !dbg !1203
  %1054 = mul i32 %1053, 8, !dbg !1204
  store i32 %1054, ptr %244, align 4, !dbg !1205, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %112) #25, !dbg !1206
  %1055 = load ptr, ptr %216, align 8, !dbg !1207, !tbaa !28
  %1056 = load i32, ptr %244, align 4, !dbg !1208, !tbaa !8
  %1057 = zext i32 %1056 to i64, !dbg !1209
  %1058 = getelementptr inbounds nuw i8, ptr %1055, i64 %1057, !dbg !1209
  %1059 = load float, ptr %1058, align 4, !dbg !1209, !tbaa !177
  store float %1059, ptr %245, align 4, !dbg !1210, !tbaa !177
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %113) #25, !dbg !1211
  %1060 = load ptr, ptr %216, align 8, !dbg !1212, !tbaa !28
  %1061 = load i32, ptr %244, align 4, !dbg !1213, !tbaa !8
  %1062 = zext i32 %1061 to i64, !dbg !1214
  %1063 = getelementptr inbounds nuw i8, ptr %1060, i64 %1062, !dbg !1214
  %1064 = getelementptr inbounds nuw i8, ptr %1063, i64 4, !dbg !1214
  %1065 = load i32, ptr %1064, align 4, !dbg !1214, !tbaa !8
  %1066 = sitofp i32 %1065 to float, !dbg !1214
  store float %1066, ptr %246, align 4, !dbg !1215, !tbaa !177
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %114) #25, !dbg !1216
  store i32 0, ptr %247, align 4, !dbg !1217, !tbaa !8
  br label %1067, !dbg !1216

1067:                                             ; preds = %1099, %1047
  %1068 = load i32, ptr %247, align 4, !dbg !1218, !tbaa !8
  %1069 = icmp slt i32 %1068, 8, !dbg !1219
  br i1 %1069, label %1071, label %1070, !dbg !1220

1070:                                             ; preds = %1067
  store i32 68, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %114) #25, !dbg !1220
  br label %1102

1071:                                             ; preds = %1067
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %115) #25, !dbg !1221
  %1072 = load i32, ptr %247, align 4, !dbg !1222, !tbaa !8
  %1073 = sext i32 %1072 to i64, !dbg !1223
  %1074 = getelementptr inbounds [8 x float], ptr %240, i64 0, i64 %1073, !dbg !1223
  %1075 = load float, ptr %1074, align 4, !dbg !1223, !tbaa !177
  %1076 = load float, ptr %245, align 4, !dbg !1224, !tbaa !177
  %1077 = call contract noundef float @_ZL9__fmul_rnff(float noundef %1075, float noundef %1076) #26, !dbg !1225
  store float %1077, ptr %248, align 4, !dbg !1226, !tbaa !177
  %1078 = load float, ptr %248, align 4, !dbg !1227, !tbaa !177
  %1079 = load float, ptr %246, align 4, !dbg !1228, !tbaa !177
  %1080 = load i32, ptr %243, align 4, !dbg !1229, !tbaa !8
  %1081 = sext i32 %1080 to i64, !dbg !1230
  %1082 = getelementptr inbounds [4 x [2 x <8 x float>]], ptr %172, i64 0, i64 %1081, !dbg !1230
  %1083 = load i32, ptr %239, align 4, !dbg !1231, !tbaa !8
  %1084 = sext i32 %1083 to i64, !dbg !1230
  %1085 = getelementptr inbounds [2 x <8 x float>], ptr %1082, i64 0, i64 %1084, !dbg !1230
  %1086 = load <8 x float>, ptr %1085, align 32, !dbg !1230, !tbaa !47
  %1087 = load i32, ptr %247, align 4, !dbg !1232, !tbaa !8
  %1088 = extractelement <8 x float> %1086, i32 %1087, !dbg !1230
  %1089 = call contract noundef float @_ZL9__fmaf_rnfff(float noundef %1078, float noundef %1079, float noundef %1088) #26, !dbg !1233
  %1090 = load i32, ptr %243, align 4, !dbg !1234, !tbaa !8
  %1091 = sext i32 %1090 to i64, !dbg !1235
  %1092 = getelementptr inbounds [4 x [2 x <8 x float>]], ptr %172, i64 0, i64 %1091, !dbg !1235
  %1093 = load i32, ptr %239, align 4, !dbg !1236, !tbaa !8
  %1094 = sext i32 %1093 to i64, !dbg !1235
  %1095 = getelementptr inbounds [2 x <8 x float>], ptr %1092, i64 0, i64 %1094, !dbg !1235
  %1096 = load i32, ptr %247, align 4, !dbg !1237, !tbaa !8
  %1097 = load <8 x float>, ptr %1095, align 32, !dbg !1238
  %1098 = insertelement <8 x float> %1097, float %1089, i32 %1096, !dbg !1238
  store <8 x float> %1098, ptr %1095, align 32, !dbg !1238
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %115) #25, !dbg !1239
  br label %1099, !dbg !1239

1099:                                             ; preds = %1071
  %1100 = load i32, ptr %247, align 4, !dbg !1240, !tbaa !8
  %1101 = add nsw i32 %1100, 1, !dbg !1240
  store i32 %1101, ptr %247, align 4, !dbg !1240, !tbaa !8
  br label %1067, !dbg !1220, !llvm.loop !1241

1102:                                             ; preds = %1070
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %113) #25, !dbg !1242
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %112) #25, !dbg !1242
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %111) #25, !dbg !1242
  br label %1103, !dbg !1242

1103:                                             ; preds = %1102
  %1104 = load i32, ptr %243, align 4, !dbg !1243, !tbaa !8
  %1105 = add nsw i32 %1104, 1, !dbg !1243
  store i32 %1105, ptr %243, align 4, !dbg !1243, !tbaa !8
  br label %1043, !dbg !1196, !llvm.loop !1244

1106:                                             ; preds = %1046
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %107) #25, !dbg !1245
  br label %1107, !dbg !1245

1107:                                             ; preds = %1106
  %1108 = load i32, ptr %239, align 4, !dbg !1246, !tbaa !8
  %1109 = add nsw i32 %1108, 1, !dbg !1246
  store i32 %1109, ptr %239, align 4, !dbg !1246, !tbaa !8
  br label %1010, !dbg !1164, !llvm.loop !1247

1110:                                             ; preds = %1013
  call void @_Z13__syncthreadsv() #26, !dbg !1248
  %1111 = load i8, ptr %215, align 1, !dbg !1249, !tbaa !880, !range !1035, !noundef !20
  %1112 = trunc i8 %1111 to i1, !dbg !1249
  br i1 %1112, label %1241, label %1113, !dbg !1250

1113:                                             ; preds = %1110
  store ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @LDS to ptr), i64 4096), ptr %162, align 8, !dbg !1251, !tbaa !28
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %116) #25, !dbg !1252
  store i32 0, ptr %249, align 4, !dbg !1253, !tbaa !8
  br label %1114, !dbg !1252

1114:                                             ; preds = %1150, %1113
  %1115 = load i32, ptr %249, align 4, !dbg !1254, !tbaa !8
  %1116 = icmp slt i32 %1115, 2, !dbg !1255
  br i1 %1116, label %1118, label %1117, !dbg !1256

1117:                                             ; preds = %1114
  store i32 71, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %116) #25, !dbg !1256
  br label %1153

1118:                                             ; preds = %1114
  %1119 = load i32, ptr %249, align 4, !dbg !1257, !tbaa !8
  %1120 = sext i32 %1119 to i64, !dbg !1258
  %1121 = getelementptr inbounds [2 x i32], ptr %176, i64 0, i64 %1120, !dbg !1258
  %1122 = load i32, ptr %1121, align 4, !dbg !1258, !tbaa !8
  %1123 = icmp ne i32 %1122, 0, !dbg !1258
  br i1 %1123, label %1124, label %1129, !dbg !1258

1124:                                             ; preds = %1118
  %1125 = load i32, ptr %249, align 4, !dbg !1259, !tbaa !8
  %1126 = sext i32 %1125 to i64, !dbg !1260
  %1127 = getelementptr inbounds [2 x <2 x i32>], ptr %190, i64 0, i64 %1126, !dbg !1260
  %1128 = load <2 x i32>, ptr %1127, align 8, !dbg !1260, !tbaa !47
  br label %1130, !dbg !1258

1129:                                             ; preds = %1118
  br label %1130, !dbg !1258

1130:                                             ; preds = %1129, %1124
  %1131 = phi <2 x i32> [ %1128, %1124 ], [ zeroinitializer, %1129 ], !dbg !1258
  %1132 = load ptr, ptr %161, align 8, !dbg !1261, !tbaa !28
  %1133 = load i32, ptr %249, align 4, !dbg !1262, !tbaa !8
  %1134 = sext i32 %1133 to i64, !dbg !1263
  %1135 = getelementptr inbounds [2 x i32], ptr %175, i64 0, i64 %1134, !dbg !1263
  %1136 = load i32, ptr %1135, align 4, !dbg !1263, !tbaa !8
  %1137 = zext i32 %1136 to i64, !dbg !1264
  %1138 = getelementptr inbounds nuw i8, ptr %1132, i64 %1137, !dbg !1264
  store <2 x i32> %1131, ptr %1138, align 8, !dbg !1265, !tbaa !47
  %1139 = load i32, ptr %249, align 4, !dbg !1266, !tbaa !8
  %1140 = sext i32 %1139 to i64, !dbg !1267
  %1141 = getelementptr inbounds [2 x <2 x i32>], ptr %191, i64 0, i64 %1140, !dbg !1267
  %1142 = load <2 x i32>, ptr %1141, align 8, !dbg !1267, !tbaa !47
  %1143 = load ptr, ptr %162, align 8, !dbg !1268, !tbaa !28
  %1144 = load i32, ptr %249, align 4, !dbg !1269, !tbaa !8
  %1145 = sext i32 %1144 to i64, !dbg !1270
  %1146 = getelementptr inbounds [2 x i32], ptr %175, i64 0, i64 %1145, !dbg !1270
  %1147 = load i32, ptr %1146, align 4, !dbg !1270, !tbaa !8
  %1148 = zext i32 %1147 to i64, !dbg !1271
  %1149 = getelementptr inbounds nuw i8, ptr %1143, i64 %1148, !dbg !1271
  store <2 x i32> %1142, ptr %1149, align 8, !dbg !1272, !tbaa !47
  br label %1150, !dbg !1273

1150:                                             ; preds = %1130
  %1151 = load i32, ptr %249, align 4, !dbg !1274, !tbaa !8
  %1152 = add nsw i32 %1151, 1, !dbg !1274
  store i32 %1152, ptr %249, align 4, !dbg !1274, !tbaa !8
  br label %1114, !dbg !1256, !llvm.loop !1275

1153:                                             ; preds = %1117
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %117) #25, !dbg !1276
  %1154 = load i32, ptr %213, align 4, !dbg !1277, !tbaa !8
  %1155 = xor i32 %1154, 1, !dbg !1278
  store i32 %1155, ptr %250, align 4, !dbg !1279, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %118) #25, !dbg !1280
  %1156 = load i32, ptr %250, align 4, !dbg !1281, !tbaa !8
  %1157 = icmp eq i32 %1156, 0, !dbg !1282
  %1158 = zext i1 %1157 to i64, !dbg !1283
  %1159 = select i1 %1157, i32 12288, i32 13312, !dbg !1283
  %1160 = sext i32 %1159 to i64, !dbg !1284
  %1161 = getelementptr inbounds i8, ptr addrspacecast (ptr addrspace(3) @LDS to ptr), i64 %1160, !dbg !1284
  store ptr %1161, ptr %251, align 8, !dbg !1285, !tbaa !28
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %119) #25, !dbg !1286
  %1162 = load i32, ptr %152, align 4, !dbg !1287, !tbaa !8
  %1163 = load i32, ptr %148, align 4, !dbg !1288, !tbaa !8
  %1164 = ashr i32 %1163, 1, !dbg !1289
  %1165 = add nsw i32 %1162, %1164, !dbg !1290
  store i32 %1165, ptr %252, align 4, !dbg !1291, !tbaa !8
  %1166 = load i32, ptr %252, align 4, !dbg !1292, !tbaa !8
  %1167 = load i32, ptr %147, align 4, !dbg !1293, !tbaa !8
  %1168 = icmp slt i32 %1166, %1167, !dbg !1294
  br i1 %1168, label %1169, label %1171, !dbg !1295

1169:                                             ; preds = %1153
  %1170 = load i32, ptr %192, align 4, !dbg !1296, !tbaa !8
  br label %1172, !dbg !1295

1171:                                             ; preds = %1153
  br label %1172, !dbg !1295

1172:                                             ; preds = %1171, %1169
  %1173 = phi i32 [ %1170, %1169 ], [ 0, %1171 ], !dbg !1295
  %1174 = load ptr, ptr %251, align 8, !dbg !1297, !tbaa !28
  %1175 = load i32, ptr %148, align 4, !dbg !1298, !tbaa !8
  %1176 = ashr i32 %1175, 1, !dbg !1299
  %1177 = mul nsw i32 %1176, 8, !dbg !1300
  %1178 = sext i32 %1177 to i64, !dbg !1301
  %1179 = getelementptr inbounds i8, ptr %1174, i64 %1178, !dbg !1301
  %1180 = load i32, ptr %148, align 4, !dbg !1302, !tbaa !8
  %1181 = and i32 %1180, 1, !dbg !1303
  %1182 = sext i32 %1181 to i64, !dbg !1301
  %1183 = getelementptr inbounds i32, ptr %1179, i64 %1182, !dbg !1301
  store i32 %1173, ptr %1183, align 4, !dbg !1304, !tbaa !8
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %119) #25, !dbg !1305
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %120) #25, !dbg !1306
  %1184 = load i32, ptr %250, align 4, !dbg !1307, !tbaa !8
  %1185 = icmp eq i32 %1184, 0, !dbg !1308
  %1186 = zext i1 %1185 to i64, !dbg !1309
  %1187 = select i1 %1185, i32 14336, i32 15360, !dbg !1309
  %1188 = sext i32 %1187 to i64, !dbg !1310
  %1189 = getelementptr inbounds i8, ptr addrspacecast (ptr addrspace(3) @LDS to ptr), i64 %1188, !dbg !1310
  store ptr %1189, ptr %253, align 8, !dbg !1311, !tbaa !28
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %121) #25, !dbg !1312
  %1190 = load i32, ptr %151, align 4, !dbg !1313, !tbaa !8
  %1191 = load i32, ptr %148, align 4, !dbg !1314, !tbaa !8
  %1192 = ashr i32 %1191, 1, !dbg !1315
  %1193 = add nsw i32 %1190, %1192, !dbg !1316
  store i32 %1193, ptr %254, align 4, !dbg !1317, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %122) #25, !dbg !1318
  %1194 = load i32, ptr %148, align 4, !dbg !1319, !tbaa !8
  %1195 = and i32 %1194, 1, !dbg !1320
  store i32 %1195, ptr %255, align 4, !dbg !1321, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %123) #25, !dbg !1322
  %1196 = load i32, ptr %255, align 4, !dbg !1323, !tbaa !8
  %1197 = icmp eq i32 %1196, 0, !dbg !1324
  br i1 %1197, label %1198, label %1210, !dbg !1325

1198:                                             ; preds = %1172
  %1199 = addrspacecast ptr %257 to ptr addrspace(5), !dbg !1326
  %1200 = load i32, ptr %193, align 4, !dbg !1327, !tbaa !8
  %1201 = and i32 %1200, 65535, !dbg !1328
  %1202 = trunc i32 %1201 to i16, !dbg !1329
  %1203 = call i16 @_Z16__ushort_as_halft(i16 noundef zeroext %1202) #26, !dbg !1326
  %1204 = getelementptr inbounds nuw %struct.__half, ptr addrspace(5) %1199, i32 0, i32 0, !dbg !1326
  %1205 = getelementptr inbounds nuw %union.anon.0, ptr addrspace(5) %1204, i32 0, i32 0, !dbg !1326
  store i16 %1203, ptr addrspace(5) %1205, align 2, !dbg !1326
  %1206 = getelementptr inbounds nuw %struct.__half, ptr %257, i32 0, i32 0, !dbg !1330
  %1207 = getelementptr inbounds nuw %union.anon.0, ptr %1206, i32 0, i32 0, !dbg !1330
  %1208 = load i16, ptr %1207, align 2, !dbg !1330
  %1209 = call contract noundef float @_Z12__half2float6__half(i16 %1208) #26, !dbg !1330
  br label %1222, !dbg !1325

1210:                                             ; preds = %1172
  %1211 = addrspacecast ptr %258 to ptr addrspace(5), !dbg !1331
  %1212 = load i32, ptr %193, align 4, !dbg !1332, !tbaa !8
  %1213 = lshr i32 %1212, 16, !dbg !1333
  %1214 = trunc i32 %1213 to i16, !dbg !1334
  %1215 = call i16 @_Z16__ushort_as_halft(i16 noundef zeroext %1214) #26, !dbg !1331
  %1216 = getelementptr inbounds nuw %struct.__half, ptr addrspace(5) %1211, i32 0, i32 0, !dbg !1331
  %1217 = getelementptr inbounds nuw %union.anon.0, ptr addrspace(5) %1216, i32 0, i32 0, !dbg !1331
  store i16 %1215, ptr addrspace(5) %1217, align 2, !dbg !1331
  %1218 = getelementptr inbounds nuw %struct.__half, ptr %258, i32 0, i32 0, !dbg !1335
  %1219 = getelementptr inbounds nuw %union.anon.0, ptr %1218, i32 0, i32 0, !dbg !1335
  %1220 = load i16, ptr %1219, align 2, !dbg !1335
  %1221 = call contract noundef float @_Z12__half2float6__half(i16 %1220) #26, !dbg !1335
  br label %1222, !dbg !1325

1222:                                             ; preds = %1210, %1198
  %1223 = phi contract float [ %1209, %1198 ], [ %1221, %1210 ], !dbg !1325
  store float %1223, ptr %256, align 4, !dbg !1336, !tbaa !177
  %1224 = load i32, ptr %254, align 4, !dbg !1337, !tbaa !8
  %1225 = load i32, ptr %145, align 4, !dbg !1338, !tbaa !8
  %1226 = icmp slt i32 %1224, %1225, !dbg !1339
  br i1 %1226, label %1227, label %1229, !dbg !1340

1227:                                             ; preds = %1222
  %1228 = load float, ptr %256, align 4, !dbg !1341, !tbaa !177
  br label %1230, !dbg !1340

1229:                                             ; preds = %1222
  br label %1230, !dbg !1340

1230:                                             ; preds = %1229, %1227
  %1231 = phi contract float [ %1228, %1227 ], [ 0.000000e+00, %1229 ], !dbg !1340
  %1232 = load ptr, ptr %253, align 8, !dbg !1342, !tbaa !28
  %1233 = load i32, ptr %148, align 4, !dbg !1343, !tbaa !8
  %1234 = ashr i32 %1233, 1, !dbg !1344
  %1235 = mul nsw i32 %1234, 8, !dbg !1345
  %1236 = sext i32 %1235 to i64, !dbg !1346
  %1237 = getelementptr inbounds i8, ptr %1232, i64 %1236, !dbg !1346
  %1238 = load i32, ptr %255, align 4, !dbg !1347, !tbaa !8
  %1239 = sext i32 %1238 to i64, !dbg !1346
  %1240 = getelementptr inbounds float, ptr %1237, i64 %1239, !dbg !1346
  store float %1231, ptr %1240, align 4, !dbg !1348, !tbaa !177
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %123) #25, !dbg !1349
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %122) #25, !dbg !1349
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %121) #25, !dbg !1349
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %120) #25, !dbg !1350
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %118) #25, !dbg !1350
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %117) #25, !dbg !1350
  br label %1241, !dbg !1350

1241:                                             ; preds = %1230, %1110
  call void @_Z13__syncthreadsv() #26, !dbg !1351
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %84) #25, !dbg !1352
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %83) #25, !dbg !1352
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %82) #25, !dbg !1352
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %81) #25, !dbg !1352
  br label %1242, !dbg !1352

1242:                                             ; preds = %1241
  %1243 = load i32, ptr %213, align 4, !dbg !1353, !tbaa !8
  %1244 = add nsw i32 %1243, 1, !dbg !1353
  store i32 %1244, ptr %213, align 4, !dbg !1353, !tbaa !8
  br label %679, !dbg !863, !llvm.loop !1354

1245:                                             ; preds = %682
  br label %1246, !dbg !1356

1246:                                             ; preds = %1245
  %1247 = load i32, ptr %212, align 4, !dbg !1357, !tbaa !8
  %1248 = add nsw i32 %1247, 1, !dbg !1357
  store i32 %1248, ptr %212, align 4, !dbg !1357, !tbaa !8
  br label %673, !dbg !858, !llvm.loop !1358

1249:                                             ; preds = %677
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %126) #25, !dbg !1359
  store ptr addrspacecast (ptr addrspace(3) @LDS to ptr), ptr %259, align 8, !dbg !1360, !tbaa !128
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %127) #25, !dbg !1361
  store i32 0, ptr %260, align 4, !dbg !1362, !tbaa !8
  br label %1250, !dbg !1361

1250:                                             ; preds = %1371, %1249
  %1251 = load i32, ptr %260, align 4, !dbg !1363, !tbaa !8
  %1252 = icmp slt i32 %1251, 2, !dbg !1364
  br i1 %1252, label %1254, label %1253, !dbg !1365

1253:                                             ; preds = %1250
  store i32 74, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %127) #25, !dbg !1365
  br label %1374

1254:                                             ; preds = %1250
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %128) #25, !dbg !1366
  store i32 0, ptr %261, align 4, !dbg !1367, !tbaa !8
  br label %1255, !dbg !1366

1255:                                             ; preds = %1367, %1254
  %1256 = load i32, ptr %261, align 4, !dbg !1368, !tbaa !8
  %1257 = icmp slt i32 %1256, 4, !dbg !1369
  br i1 %1257, label %1259, label %1258, !dbg !1370

1258:                                             ; preds = %1255
  store i32 77, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %128) #25, !dbg !1370
  br label %1370

1259:                                             ; preds = %1255
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %129) #25, !dbg !1371
  store i32 0, ptr %262, align 4, !dbg !1372, !tbaa !8
  br label %1260, !dbg !1371

1260:                                             ; preds = %1287, %1259
  %1261 = load i32, ptr %262, align 4, !dbg !1373, !tbaa !8
  %1262 = icmp slt i32 %1261, 8, !dbg !1374
  br i1 %1262, label %1264, label %1263, !dbg !1375

1263:                                             ; preds = %1260
  store i32 80, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %129) #25, !dbg !1375
  br label %1290

1264:                                             ; preds = %1260
  %1265 = load i32, ptr %261, align 4, !dbg !1376, !tbaa !8
  %1266 = sext i32 %1265 to i64, !dbg !1377
  %1267 = getelementptr inbounds [4 x [2 x <8 x float>]], ptr %172, i64 0, i64 %1266, !dbg !1377
  %1268 = load i32, ptr %260, align 4, !dbg !1378, !tbaa !8
  %1269 = sext i32 %1268 to i64, !dbg !1377
  %1270 = getelementptr inbounds [2 x <8 x float>], ptr %1267, i64 0, i64 %1269, !dbg !1377
  %1271 = load <8 x float>, ptr %1270, align 32, !dbg !1377, !tbaa !47
  %1272 = load i32, ptr %262, align 4, !dbg !1379, !tbaa !8
  %1273 = extractelement <8 x float> %1271, i32 %1272, !dbg !1377
  %1274 = load ptr, ptr %259, align 8, !dbg !1380, !tbaa !128
  %1275 = load i32, ptr %149, align 4, !dbg !1381, !tbaa !8
  %1276 = mul nsw i32 %1275, 320, !dbg !1382
  %1277 = load i32, ptr %155, align 4, !dbg !1383, !tbaa !8
  %1278 = mul nsw i32 8, %1277, !dbg !1384
  %1279 = load i32, ptr %262, align 4, !dbg !1385, !tbaa !8
  %1280 = add nsw i32 %1278, %1279, !dbg !1386
  %1281 = mul nsw i32 %1280, 20, !dbg !1387
  %1282 = add nsw i32 %1276, %1281, !dbg !1388
  %1283 = load i32, ptr %154, align 4, !dbg !1389, !tbaa !8
  %1284 = add nsw i32 %1282, %1283, !dbg !1390
  %1285 = zext i32 %1284 to i64, !dbg !1380
  %1286 = getelementptr inbounds nuw float, ptr %1274, i64 %1285, !dbg !1380
  store float %1273, ptr %1286, align 4, !dbg !1391, !tbaa !177
  br label %1287, !dbg !1392

1287:                                             ; preds = %1264
  %1288 = load i32, ptr %262, align 4, !dbg !1393, !tbaa !8
  %1289 = add nsw i32 %1288, 1, !dbg !1393
  store i32 %1289, ptr %262, align 4, !dbg !1393, !tbaa !8
  br label %1260, !dbg !1375, !llvm.loop !1394

1290:                                             ; preds = %1263
  call void @_Z13__syncthreadsv() #26, !dbg !1395
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %130) #25, !dbg !1396
  store i32 0, ptr %263, align 4, !dbg !1397, !tbaa !8
  br label %1291, !dbg !1396

1291:                                             ; preds = %1363, %1290
  %1292 = load i32, ptr %263, align 4, !dbg !1398, !tbaa !8
  %1293 = icmp slt i32 %1292, 8, !dbg !1399
  br i1 %1293, label %1295, label %1294, !dbg !1400

1294:                                             ; preds = %1291
  store i32 83, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %130) #25, !dbg !1400
  br label %1366

1295:                                             ; preds = %1291
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %131) #25, !dbg !1401
  %1296 = load i32, ptr %263, align 4, !dbg !1402, !tbaa !8
  %1297 = ashr i32 %1296, 1, !dbg !1403
  store i32 %1297, ptr %264, align 4, !dbg !1404, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %132) #25, !dbg !1405
  %1298 = load i32, ptr %263, align 4, !dbg !1406, !tbaa !8
  %1299 = and i32 %1298, 1, !dbg !1407
  store i32 %1299, ptr %265, align 4, !dbg !1408, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %133) #25, !dbg !1409
  %1300 = load i32, ptr %148, align 4, !dbg !1410, !tbaa !8
  %1301 = and i32 %1300, 15, !dbg !1411
  store i32 %1301, ptr %266, align 4, !dbg !1412, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %134) #25, !dbg !1413
  %1302 = load i32, ptr %148, align 4, !dbg !1414, !tbaa !8
  %1303 = ashr i32 %1302, 4, !dbg !1415
  %1304 = and i32 %1303, 15, !dbg !1416
  store i32 %1304, ptr %267, align 4, !dbg !1417, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %135) #25, !dbg !1418
  %1305 = load i32, ptr %264, align 4, !dbg !1419, !tbaa !8
  %1306 = mul nsw i32 %1305, 32, !dbg !1420
  %1307 = load i32, ptr %260, align 4, !dbg !1421, !tbaa !8
  %1308 = mul nsw i32 %1307, 16, !dbg !1422
  %1309 = add nsw i32 %1306, %1308, !dbg !1423
  %1310 = load i32, ptr %266, align 4, !dbg !1424, !tbaa !8
  %1311 = add nsw i32 %1309, %1310, !dbg !1425
  store i32 %1311, ptr %268, align 4, !dbg !1426, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %136) #25, !dbg !1427
  %1312 = load i32, ptr %265, align 4, !dbg !1428, !tbaa !8
  %1313 = mul nsw i32 %1312, 64, !dbg !1429
  %1314 = load i32, ptr %261, align 4, !dbg !1430, !tbaa !8
  %1315 = mul nsw i32 %1314, 16, !dbg !1431
  %1316 = add nsw i32 %1313, %1315, !dbg !1432
  %1317 = load i32, ptr %267, align 4, !dbg !1433, !tbaa !8
  %1318 = add nsw i32 %1316, %1317, !dbg !1434
  store i32 %1318, ptr %269, align 4, !dbg !1435, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %137) #25, !dbg !1436
  %1319 = load i32, ptr %264, align 4, !dbg !1437, !tbaa !8
  %1320 = shl i32 %1319, 1, !dbg !1438
  %1321 = load i32, ptr %265, align 4, !dbg !1439, !tbaa !8
  %1322 = or i32 %1320, %1321, !dbg !1440
  store i32 %1322, ptr %270, align 4, !dbg !1441, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %138) #25, !dbg !1442
  %1323 = load ptr, ptr %259, align 8, !dbg !1443, !tbaa !128
  %1324 = load i32, ptr %270, align 4, !dbg !1444, !tbaa !8
  %1325 = mul nsw i32 %1324, 320, !dbg !1445
  %1326 = load i32, ptr %266, align 4, !dbg !1446, !tbaa !8
  %1327 = mul nsw i32 %1326, 20, !dbg !1447
  %1328 = add nsw i32 %1325, %1327, !dbg !1448
  %1329 = load i32, ptr %267, align 4, !dbg !1449, !tbaa !8
  %1330 = add nsw i32 %1328, %1329, !dbg !1450
  %1331 = zext i32 %1330 to i64, !dbg !1443
  %1332 = getelementptr inbounds nuw float, ptr %1323, i64 %1331, !dbg !1443
  %1333 = load float, ptr %1332, align 4, !dbg !1443, !tbaa !177
  store float %1333, ptr %271, align 4, !dbg !1451, !tbaa !177
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %139) #25, !dbg !1452
  %1334 = load i32, ptr %151, align 4, !dbg !1453, !tbaa !8
  %1335 = load i32, ptr %268, align 4, !dbg !1454, !tbaa !8
  %1336 = add nsw i32 %1334, %1335, !dbg !1455
  store i32 %1336, ptr %272, align 4, !dbg !1456, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %140) #25, !dbg !1457
  %1337 = load i32, ptr %152, align 4, !dbg !1458, !tbaa !8
  %1338 = load i32, ptr %269, align 4, !dbg !1459, !tbaa !8
  %1339 = add nsw i32 %1337, %1338, !dbg !1460
  store i32 %1339, ptr %273, align 4, !dbg !1461, !tbaa !8
  %1340 = load i32, ptr %272, align 4, !dbg !1462, !tbaa !8
  %1341 = load i32, ptr %145, align 4, !dbg !1463, !tbaa !8
  %1342 = icmp slt i32 %1340, %1341, !dbg !1464
  br i1 %1342, label %1343, label %1362, !dbg !1465

1343:                                             ; preds = %1295
  %1344 = load i32, ptr %273, align 4, !dbg !1466, !tbaa !8
  %1345 = load i32, ptr %147, align 4, !dbg !1467, !tbaa !8
  %1346 = icmp slt i32 %1344, %1345, !dbg !1468
  br i1 %1346, label %1347, label %1362, !dbg !1465

1347:                                             ; preds = %1343
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %141) #25, !dbg !1469
  %1348 = load ptr, ptr %144, align 8, !dbg !1470, !tbaa !128
  %1349 = load i32, ptr %273, align 4, !dbg !1471, !tbaa !8
  %1350 = sext i32 %1349 to i64, !dbg !1471
  %1351 = load i32, ptr %145, align 4, !dbg !1472, !tbaa !8
  %1352 = sext i32 %1351 to i64, !dbg !1472
  %1353 = mul nsw i64 %1350, %1352, !dbg !1473
  %1354 = getelementptr inbounds float, ptr %1348, i64 %1353, !dbg !1474
  %1355 = load i32, ptr %272, align 4, !dbg !1475, !tbaa !8
  %1356 = sext i32 %1355 to i64, !dbg !1476
  %1357 = getelementptr inbounds float, ptr %1354, i64 %1356, !dbg !1476
  store ptr %1357, ptr %274, align 8, !dbg !1477, !tbaa !128
  %1358 = load float, ptr %271, align 4, !dbg !1478, !tbaa !177
  %1359 = load ptr, ptr %274, align 8, !dbg !1479, !tbaa !128
  %1360 = load float, ptr %1359, align 4, !dbg !1480, !tbaa !177
  %1361 = fadd contract float %1360, %1358, !dbg !1480
  store float %1361, ptr %1359, align 4, !dbg !1480, !tbaa !177
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %141) #25, !dbg !1481
  br label %1362, !dbg !1481

1362:                                             ; preds = %1347, %1343, %1295
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %140) #25, !dbg !1482
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %139) #25, !dbg !1482
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %138) #25, !dbg !1482
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %137) #25, !dbg !1482
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %136) #25, !dbg !1482
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %135) #25, !dbg !1482
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %134) #25, !dbg !1482
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %133) #25, !dbg !1482
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %132) #25, !dbg !1482
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %131) #25, !dbg !1482
  br label %1363, !dbg !1482

1363:                                             ; preds = %1362
  %1364 = load i32, ptr %263, align 4, !dbg !1483, !tbaa !8
  %1365 = add nsw i32 %1364, 1, !dbg !1483
  store i32 %1365, ptr %263, align 4, !dbg !1483, !tbaa !8
  br label %1291, !dbg !1400, !llvm.loop !1484

1366:                                             ; preds = %1294
  call void @_Z13__syncthreadsv() #26, !dbg !1485
  br label %1367, !dbg !1486

1367:                                             ; preds = %1366
  %1368 = load i32, ptr %261, align 4, !dbg !1487, !tbaa !8
  %1369 = add nsw i32 %1368, 1, !dbg !1487
  store i32 %1369, ptr %261, align 4, !dbg !1487, !tbaa !8
  br label %1255, !dbg !1370, !llvm.loop !1488

1370:                                             ; preds = %1258
  br label %1371, !dbg !1489

1371:                                             ; preds = %1370
  %1372 = load i32, ptr %260, align 4, !dbg !1490, !tbaa !8
  %1373 = add nsw i32 %1372, 1, !dbg !1490
  store i32 %1373, ptr %260, align 4, !dbg !1490, !tbaa !8
  br label %1250, !dbg !1365, !llvm.loop !1491

1374:                                             ; preds = %1253
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %126) #25, !dbg !1492
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %60) #25, !dbg !1492
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %59) #25, !dbg !1492
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %58) #25, !dbg !1492
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %57) #25, !dbg !1492
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %56) #25, !dbg !1492
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %55) #25, !dbg !1492
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %53) #25, !dbg !1492
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %49) #25, !dbg !1492
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %41) #25, !dbg !1492
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %40) #25, !dbg !1492
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %39) #25, !dbg !1492
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %38) #25, !dbg !1492
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %37) #25, !dbg !1492
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %30) #25, !dbg !1492
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %29) #25, !dbg !1492
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %28) #25, !dbg !1492
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %27) #25, !dbg !1492
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %26) #25, !dbg !1492
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %25) #25, !dbg !1492
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %24) #25, !dbg !1492
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %23) #25, !dbg !1492
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %22) #25, !dbg !1492
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %21) #25, !dbg !1492
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %20) #25, !dbg !1492
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %19) #25, !dbg !1492
  store i32 0, ptr %153, align 4, !dbg !1492
  br label %1375, !dbg !1492

1375:                                             ; preds = %1374, %291
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %17) #25, !dbg !1492
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %16) #25, !dbg !1492
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %15) #25, !dbg !1492
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %14) #25, !dbg !1492
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %13) #25, !dbg !1492
  %1376 = load i32, ptr %153, align 4
  switch i32 %1376, label %1378 [
    i32 0, label %1377
    i32 1, label %1377
  ]

1377:                                             ; preds = %1375, %1375
  ret void, !dbg !1492

1378:                                             ; preds = %1375
  unreachable
}

; Function Attrs: alwaysinline convergent mustprogress nounwind uwtable
define internal void @_ZL42gemm_mq4g256v2_residual_mmq_iu4_gfx12_bodyILb0EEvPKcPK12block_i4_128Pfiii(ptr noalias noundef %0, ptr noalias noundef %1, ptr noalias noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5) #5 !dbg !1493 {
  %7 = alloca ptr, align 8, addrspace(5)
  %8 = alloca ptr, align 8, addrspace(5)
  %9 = alloca ptr, align 8, addrspace(5)
  %10 = alloca i32, align 4, addrspace(5)
  %11 = alloca i32, align 4, addrspace(5)
  %12 = alloca i32, align 4, addrspace(5)
  %13 = alloca i32, align 4, addrspace(5)
  %14 = alloca i32, align 4, addrspace(5)
  %15 = alloca i32, align 4, addrspace(5)
  %16 = alloca i32, align 4, addrspace(5)
  %17 = alloca i32, align 4, addrspace(5)
  %18 = alloca i32, align 4, addrspace(5)
  %19 = alloca i32, align 4, addrspace(5)
  %20 = alloca i32, align 4, addrspace(5)
  %21 = alloca i32, align 4, addrspace(5)
  %22 = alloca i32, align 4, addrspace(5)
  %23 = alloca i32, align 4, addrspace(5)
  %24 = alloca i32, align 4, addrspace(5)
  %25 = alloca i32, align 4, addrspace(5)
  %26 = alloca ptr, align 8, addrspace(5)
  %27 = alloca ptr, align 8, addrspace(5)
  %28 = alloca [4 x [2 x i32]], align 16, addrspace(5)
  %29 = alloca [2 x [2 x i32]], align 16, addrspace(5)
  %30 = alloca i32, align 4, addrspace(5)
  %31 = alloca i32, align 4, addrspace(5)
  %32 = alloca i32, align 4, addrspace(5)
  %33 = alloca i32, align 4, addrspace(5)
  %34 = alloca i32, align 4, addrspace(5)
  %35 = alloca i32, align 4, addrspace(5)
  %36 = alloca i32, align 4, addrspace(5)
  %37 = alloca [4 x [2 x <8 x float>]], align 32, addrspace(5)
  %38 = alloca [2 x i32], align 4, addrspace(5)
  %39 = alloca [2 x i32], align 4, addrspace(5)
  %40 = alloca [2 x i32], align 4, addrspace(5)
  %41 = alloca [2 x i32], align 4, addrspace(5)
  %42 = alloca i32, align 4, addrspace(5)
  %43 = alloca i32, align 4, addrspace(5)
  %44 = alloca i32, align 4, addrspace(5)
  %45 = alloca i32, align 4, addrspace(5)
  %46 = alloca i32, align 4, addrspace(5)
  %47 = alloca i32, align 4, addrspace(5)
  %48 = alloca i32, align 4, addrspace(5)
  %49 = alloca <2 x i32>, align 8, addrspace(5)
  %50 = alloca i32, align 4, addrspace(5)
  %51 = alloca i32, align 4, addrspace(5)
  %52 = alloca <8 x float>, align 32, addrspace(5)
  %53 = alloca <8 x i32>, align 32, addrspace(5)
  %54 = alloca <8 x i32>, align 32, addrspace(5)
  %55 = alloca [2 x <8 x float>], align 32, addrspace(5)
  %56 = alloca [4 x float], align 16, addrspace(5)
  %57 = alloca [2 x <2 x i32>], align 16, addrspace(5)
  %58 = alloca [2 x <2 x i32>], align 16, addrspace(5)
  %59 = alloca i32, align 4, addrspace(5)
  %60 = alloca i32, align 4, addrspace(5)
  %61 = alloca i32, align 4, addrspace(5)
  %62 = alloca i32, align 4, addrspace(5)
  %63 = alloca i32, align 4, addrspace(5)
  %64 = alloca i32, align 4, addrspace(5)
  %65 = alloca i32, align 4, addrspace(5)
  %66 = alloca i32, align 4, addrspace(5)
  %67 = alloca i32, align 4, addrspace(5)
  %68 = alloca i32, align 4, addrspace(5)
  %69 = alloca i32, align 4, addrspace(5)
  %70 = alloca i32, align 4, addrspace(5)
  %71 = alloca i32, align 4, addrspace(5)
  %72 = alloca float, align 4, addrspace(5)
  %73 = alloca %struct.__half, align 2, addrspace(5)
  %74 = alloca %struct.__half, align 2, addrspace(5)
  %75 = alloca ptr, align 8, addrspace(5)
  %76 = alloca ptr, align 8, addrspace(5)
  %77 = alloca i32, align 4, addrspace(5)
  %78 = alloca <2 x i32>, align 8, addrspace(5)
  %79 = alloca i32, align 4, addrspace(5)
  %80 = alloca i32, align 4, addrspace(5)
  %81 = alloca i32, align 4, addrspace(5)
  %82 = alloca i8, align 1, addrspace(5)
  %83 = alloca ptr, align 8, addrspace(5)
  %84 = alloca ptr, align 8, addrspace(5)
  %85 = alloca i32, align 4, addrspace(5)
  %86 = alloca i32, align 4, addrspace(5)
  %87 = alloca i32, align 4, addrspace(5)
  %88 = alloca i32, align 4, addrspace(5)
  %89 = alloca i32, align 4, addrspace(5)
  %90 = alloca ptr, align 8, addrspace(5)
  %91 = alloca ptr, align 8, addrspace(5)
  %92 = alloca i32, align 4, addrspace(5)
  %93 = alloca <2 x i32>, align 8, addrspace(5)
  %94 = alloca i32, align 4, addrspace(5)
  %95 = alloca i32, align 4, addrspace(5)
  %96 = alloca ptr, align 8, addrspace(5)
  %97 = alloca i32, align 4, addrspace(5)
  %98 = alloca i32, align 4, addrspace(5)
  %99 = alloca i32, align 4, addrspace(5)
  %100 = alloca ptr, align 8, addrspace(5)
  %101 = alloca i32, align 4, addrspace(5)
  %102 = alloca i32, align 4, addrspace(5)
  %103 = alloca i32, align 4, addrspace(5)
  %104 = alloca i32, align 4, addrspace(5)
  %105 = alloca i32, align 4, addrspace(5)
  %106 = alloca i32, align 4, addrspace(5)
  %107 = alloca [8 x float], align 16, addrspace(5)
  %108 = alloca i32, align 4, addrspace(5)
  %109 = alloca i32, align 4, addrspace(5)
  %110 = alloca i32, align 4, addrspace(5)
  %111 = alloca i32, align 4, addrspace(5)
  %112 = alloca float, align 4, addrspace(5)
  %113 = alloca float, align 4, addrspace(5)
  %114 = alloca i32, align 4, addrspace(5)
  %115 = alloca float, align 4, addrspace(5)
  %116 = alloca i32, align 4, addrspace(5)
  %117 = alloca i32, align 4, addrspace(5)
  %118 = alloca ptr, align 8, addrspace(5)
  %119 = alloca i32, align 4, addrspace(5)
  %120 = alloca ptr, align 8, addrspace(5)
  %121 = alloca i32, align 4, addrspace(5)
  %122 = alloca i32, align 4, addrspace(5)
  %123 = alloca float, align 4, addrspace(5)
  %124 = alloca %struct.__half, align 2, addrspace(5)
  %125 = alloca %struct.__half, align 2, addrspace(5)
  %126 = alloca ptr, align 8, addrspace(5)
  %127 = alloca i32, align 4, addrspace(5)
  %128 = alloca i32, align 4, addrspace(5)
  %129 = alloca i32, align 4, addrspace(5)
  %130 = alloca i32, align 4, addrspace(5)
  %131 = alloca i32, align 4, addrspace(5)
  %132 = alloca i32, align 4, addrspace(5)
  %133 = alloca i32, align 4, addrspace(5)
  %134 = alloca i32, align 4, addrspace(5)
  %135 = alloca i32, align 4, addrspace(5)
  %136 = alloca i32, align 4, addrspace(5)
  %137 = alloca i32, align 4, addrspace(5)
  %138 = alloca float, align 4, addrspace(5)
  %139 = alloca i32, align 4, addrspace(5)
  %140 = alloca i32, align 4, addrspace(5)
  %141 = alloca ptr, align 8, addrspace(5)
  %142 = addrspacecast ptr addrspace(5) %7 to ptr
  %143 = addrspacecast ptr addrspace(5) %8 to ptr
  %144 = addrspacecast ptr addrspace(5) %9 to ptr
  %145 = addrspacecast ptr addrspace(5) %10 to ptr
  %146 = addrspacecast ptr addrspace(5) %11 to ptr
  %147 = addrspacecast ptr addrspace(5) %12 to ptr
  %148 = addrspacecast ptr addrspace(5) %13 to ptr
  %149 = addrspacecast ptr addrspace(5) %14 to ptr
  %150 = addrspacecast ptr addrspace(5) %15 to ptr
  %151 = addrspacecast ptr addrspace(5) %16 to ptr
  %152 = addrspacecast ptr addrspace(5) %17 to ptr
  %153 = addrspacecast ptr addrspace(5) %18 to ptr
  %154 = addrspacecast ptr addrspace(5) %19 to ptr
  %155 = addrspacecast ptr addrspace(5) %20 to ptr
  %156 = addrspacecast ptr addrspace(5) %21 to ptr
  %157 = addrspacecast ptr addrspace(5) %22 to ptr
  %158 = addrspacecast ptr addrspace(5) %23 to ptr
  %159 = addrspacecast ptr addrspace(5) %24 to ptr
  %160 = addrspacecast ptr addrspace(5) %25 to ptr
  %161 = addrspacecast ptr addrspace(5) %26 to ptr
  %162 = addrspacecast ptr addrspace(5) %27 to ptr
  %163 = addrspacecast ptr addrspace(5) %28 to ptr
  %164 = addrspacecast ptr addrspace(5) %29 to ptr
  %165 = addrspacecast ptr addrspace(5) %30 to ptr
  %166 = addrspacecast ptr addrspace(5) %31 to ptr
  %167 = addrspacecast ptr addrspace(5) %32 to ptr
  %168 = addrspacecast ptr addrspace(5) %33 to ptr
  %169 = addrspacecast ptr addrspace(5) %34 to ptr
  %170 = addrspacecast ptr addrspace(5) %35 to ptr
  %171 = addrspacecast ptr addrspace(5) %36 to ptr
  %172 = addrspacecast ptr addrspace(5) %37 to ptr
  %173 = addrspacecast ptr addrspace(5) %38 to ptr
  %174 = addrspacecast ptr addrspace(5) %39 to ptr
  %175 = addrspacecast ptr addrspace(5) %40 to ptr
  %176 = addrspacecast ptr addrspace(5) %41 to ptr
  %177 = addrspacecast ptr addrspace(5) %42 to ptr
  %178 = addrspacecast ptr addrspace(5) %43 to ptr
  %179 = addrspacecast ptr addrspace(5) %44 to ptr
  %180 = addrspacecast ptr addrspace(5) %45 to ptr
  %181 = addrspacecast ptr addrspace(5) %46 to ptr
  %182 = addrspacecast ptr addrspace(5) %47 to ptr
  %183 = addrspacecast ptr addrspace(5) %48 to ptr
  %184 = addrspacecast ptr addrspace(5) %49 to ptr
  %185 = addrspacecast ptr addrspace(5) %50 to ptr
  %186 = addrspacecast ptr addrspace(5) %51 to ptr
  %187 = addrspacecast ptr addrspace(5) %53 to ptr
  %188 = addrspacecast ptr addrspace(5) %55 to ptr
  %189 = addrspacecast ptr addrspace(5) %56 to ptr
  %190 = addrspacecast ptr addrspace(5) %57 to ptr
  %191 = addrspacecast ptr addrspace(5) %58 to ptr
  %192 = addrspacecast ptr addrspace(5) %59 to ptr
  %193 = addrspacecast ptr addrspace(5) %60 to ptr
  %194 = addrspacecast ptr addrspace(5) %61 to ptr
  %195 = addrspacecast ptr addrspace(5) %62 to ptr
  %196 = addrspacecast ptr addrspace(5) %63 to ptr
  %197 = addrspacecast ptr addrspace(5) %64 to ptr
  %198 = addrspacecast ptr addrspace(5) %65 to ptr
  %199 = addrspacecast ptr addrspace(5) %66 to ptr
  %200 = addrspacecast ptr addrspace(5) %67 to ptr
  %201 = addrspacecast ptr addrspace(5) %68 to ptr
  %202 = addrspacecast ptr addrspace(5) %69 to ptr
  %203 = addrspacecast ptr addrspace(5) %70 to ptr
  %204 = addrspacecast ptr addrspace(5) %71 to ptr
  %205 = addrspacecast ptr addrspace(5) %72 to ptr
  %206 = addrspacecast ptr addrspace(5) %73 to ptr
  %207 = addrspacecast ptr addrspace(5) %74 to ptr
  %208 = addrspacecast ptr addrspace(5) %75 to ptr
  %209 = addrspacecast ptr addrspace(5) %76 to ptr
  %210 = addrspacecast ptr addrspace(5) %77 to ptr
  %211 = addrspacecast ptr addrspace(5) %78 to ptr
  %212 = addrspacecast ptr addrspace(5) %79 to ptr
  %213 = addrspacecast ptr addrspace(5) %80 to ptr
  %214 = addrspacecast ptr addrspace(5) %81 to ptr
  %215 = addrspacecast ptr addrspace(5) %82 to ptr
  %216 = addrspacecast ptr addrspace(5) %83 to ptr
  %217 = addrspacecast ptr addrspace(5) %84 to ptr
  %218 = addrspacecast ptr addrspace(5) %85 to ptr
  %219 = addrspacecast ptr addrspace(5) %86 to ptr
  %220 = addrspacecast ptr addrspace(5) %87 to ptr
  %221 = addrspacecast ptr addrspace(5) %88 to ptr
  %222 = addrspacecast ptr addrspace(5) %89 to ptr
  %223 = addrspacecast ptr addrspace(5) %90 to ptr
  %224 = addrspacecast ptr addrspace(5) %91 to ptr
  %225 = addrspacecast ptr addrspace(5) %92 to ptr
  %226 = addrspacecast ptr addrspace(5) %93 to ptr
  %227 = addrspacecast ptr addrspace(5) %94 to ptr
  %228 = addrspacecast ptr addrspace(5) %95 to ptr
  %229 = addrspacecast ptr addrspace(5) %96 to ptr
  %230 = addrspacecast ptr addrspace(5) %97 to ptr
  %231 = addrspacecast ptr addrspace(5) %98 to ptr
  %232 = addrspacecast ptr addrspace(5) %99 to ptr
  %233 = addrspacecast ptr addrspace(5) %100 to ptr
  %234 = addrspacecast ptr addrspace(5) %101 to ptr
  %235 = addrspacecast ptr addrspace(5) %102 to ptr
  %236 = addrspacecast ptr addrspace(5) %103 to ptr
  %237 = addrspacecast ptr addrspace(5) %104 to ptr
  %238 = addrspacecast ptr addrspace(5) %105 to ptr
  %239 = addrspacecast ptr addrspace(5) %106 to ptr
  %240 = addrspacecast ptr addrspace(5) %107 to ptr
  %241 = addrspacecast ptr addrspace(5) %108 to ptr
  %242 = addrspacecast ptr addrspace(5) %109 to ptr
  %243 = addrspacecast ptr addrspace(5) %110 to ptr
  %244 = addrspacecast ptr addrspace(5) %111 to ptr
  %245 = addrspacecast ptr addrspace(5) %112 to ptr
  %246 = addrspacecast ptr addrspace(5) %113 to ptr
  %247 = addrspacecast ptr addrspace(5) %114 to ptr
  %248 = addrspacecast ptr addrspace(5) %115 to ptr
  %249 = addrspacecast ptr addrspace(5) %116 to ptr
  %250 = addrspacecast ptr addrspace(5) %117 to ptr
  %251 = addrspacecast ptr addrspace(5) %118 to ptr
  %252 = addrspacecast ptr addrspace(5) %119 to ptr
  %253 = addrspacecast ptr addrspace(5) %120 to ptr
  %254 = addrspacecast ptr addrspace(5) %121 to ptr
  %255 = addrspacecast ptr addrspace(5) %122 to ptr
  %256 = addrspacecast ptr addrspace(5) %123 to ptr
  %257 = addrspacecast ptr addrspace(5) %124 to ptr
  %258 = addrspacecast ptr addrspace(5) %125 to ptr
  %259 = addrspacecast ptr addrspace(5) %126 to ptr
  %260 = addrspacecast ptr addrspace(5) %127 to ptr
  %261 = addrspacecast ptr addrspace(5) %128 to ptr
  %262 = addrspacecast ptr addrspace(5) %129 to ptr
  %263 = addrspacecast ptr addrspace(5) %130 to ptr
  %264 = addrspacecast ptr addrspace(5) %131 to ptr
  %265 = addrspacecast ptr addrspace(5) %132 to ptr
  %266 = addrspacecast ptr addrspace(5) %133 to ptr
  %267 = addrspacecast ptr addrspace(5) %134 to ptr
  %268 = addrspacecast ptr addrspace(5) %135 to ptr
  %269 = addrspacecast ptr addrspace(5) %136 to ptr
  %270 = addrspacecast ptr addrspace(5) %137 to ptr
  %271 = addrspacecast ptr addrspace(5) %138 to ptr
  %272 = addrspacecast ptr addrspace(5) %139 to ptr
  %273 = addrspacecast ptr addrspace(5) %140 to ptr
  %274 = addrspacecast ptr addrspace(5) %141 to ptr
  store ptr %0, ptr %142, align 8, !tbaa !28
  store ptr %1, ptr %143, align 8, !tbaa !130
  store ptr %2, ptr %144, align 8, !tbaa !128
  store i32 %3, ptr %145, align 4, !tbaa !8
  store i32 %4, ptr %146, align 4, !tbaa !8
  store i32 %5, ptr %147, align 4, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %13) #25, !dbg !1494
  %275 = call noundef i32 @_ZN25__hip_builtin_threadIdx_t7__get_xEv() #26, !dbg !1495
  store i32 %275, ptr %148, align 4, !dbg !1496, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %14) #25, !dbg !1497
  %276 = load i32, ptr %148, align 4, !dbg !1498, !tbaa !8
  %277 = ashr i32 %276, 5, !dbg !1499
  store i32 %277, ptr %149, align 4, !dbg !1500, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %15) #25, !dbg !1501
  %278 = load i32, ptr %148, align 4, !dbg !1502, !tbaa !8
  %279 = and i32 %278, 31, !dbg !1503
  store i32 %279, ptr %150, align 4, !dbg !1504, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %16) #25, !dbg !1505
  %280 = call noundef i32 @_ZN24__hip_builtin_blockIdx_t7__get_xEv() #26, !dbg !1506
  %281 = mul i32 %280, 128, !dbg !1507
  store i32 %281, ptr %151, align 4, !dbg !1508, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %17) #25, !dbg !1509
  %282 = call noundef i32 @_ZN24__hip_builtin_blockIdx_t7__get_yEv() #26, !dbg !1510
  %283 = mul i32 %282, 128, !dbg !1511
  store i32 %283, ptr %152, align 4, !dbg !1512, !tbaa !8
  %284 = load i32, ptr %151, align 4, !dbg !1513, !tbaa !8
  %285 = load i32, ptr %145, align 4, !dbg !1514, !tbaa !8
  %286 = icmp sge i32 %284, %285, !dbg !1515
  br i1 %286, label %291, label %287, !dbg !1516

287:                                              ; preds = %6
  %288 = load i32, ptr %152, align 4, !dbg !1517, !tbaa !8
  %289 = load i32, ptr %147, align 4, !dbg !1518, !tbaa !8
  %290 = icmp sge i32 %288, %289, !dbg !1519
  br i1 %290, label %291, label %292, !dbg !1516

291:                                              ; preds = %287, %6
  store i32 1, ptr %153, align 4
  br label %1373, !dbg !1520

292:                                              ; preds = %287
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %19) #25, !dbg !1521
  %293 = load i32, ptr %150, align 4, !dbg !1522, !tbaa !8
  %294 = and i32 %293, 15, !dbg !1523
  store i32 %294, ptr %154, align 4, !dbg !1524, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %20) #25, !dbg !1525
  %295 = load i32, ptr %150, align 4, !dbg !1526, !tbaa !8
  %296 = ashr i32 %295, 4, !dbg !1527
  store i32 %296, ptr %155, align 4, !dbg !1528, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %21) #25, !dbg !1529
  %297 = load i32, ptr %149, align 4, !dbg !1530, !tbaa !8
  %298 = ashr i32 %297, 1, !dbg !1531
  store i32 %298, ptr %156, align 4, !dbg !1532, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %22) #25, !dbg !1533
  %299 = load i32, ptr %149, align 4, !dbg !1534, !tbaa !8
  %300 = and i32 %299, 1, !dbg !1535
  store i32 %300, ptr %157, align 4, !dbg !1536, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %23) #25, !dbg !1537
  %301 = load i32, ptr %156, align 4, !dbg !1538, !tbaa !8
  %302 = mul nsw i32 %301, 32, !dbg !1539
  store i32 %302, ptr %158, align 4, !dbg !1540, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %24) #25, !dbg !1541
  %303 = load i32, ptr %157, align 4, !dbg !1542, !tbaa !8
  %304 = mul nsw i32 %303, 64, !dbg !1543
  store i32 %304, ptr %159, align 4, !dbg !1544, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %25) #25, !dbg !1545
  %305 = load i32, ptr %146, align 4, !dbg !1546, !tbaa !8
  %306 = sdiv i32 %305, 256, !dbg !1547
  store i32 %306, ptr %160, align 4, !dbg !1548, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %26) #25, !dbg !1549
  store ptr addrspacecast (ptr addrspace(3) @LDS to ptr), ptr %161, align 8, !dbg !1550, !tbaa !28
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %27) #25, !dbg !1551
  store ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @LDS to ptr), i64 4096), ptr %162, align 8, !dbg !1552, !tbaa !28
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %28) #25, !dbg !1553
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %29) #25, !dbg !1554
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %30) #25, !dbg !1555
  %307 = load i32, ptr %150, align 4, !dbg !1556, !tbaa !8
  %308 = mul i32 %307, 8, !dbg !1557
  store i32 %308, ptr %165, align 4, !dbg !1558, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %31) #25, !dbg !1559
  store i32 0, ptr %166, align 4, !dbg !1560, !tbaa !8
  br label %309, !dbg !1559

309:                                              ; preds = %340, %292
  %310 = load i32, ptr %166, align 4, !dbg !1561, !tbaa !8
  %311 = icmp slt i32 %310, 4, !dbg !1562
  br i1 %311, label %313, label %312, !dbg !1563

312:                                              ; preds = %309
  store i32 2, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %31) #25, !dbg !1563
  br label %343

313:                                              ; preds = %309
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %32) #25, !dbg !1564
  %314 = load i32, ptr %157, align 4, !dbg !1565, !tbaa !8
  %315 = mul nsw i32 %314, 4, !dbg !1566
  %316 = load i32, ptr %166, align 4, !dbg !1567, !tbaa !8
  %317 = add nsw i32 %315, %316, !dbg !1568
  store i32 %317, ptr %167, align 4, !dbg !1569, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %33) #25, !dbg !1570
  store i32 0, ptr %168, align 4, !dbg !1571, !tbaa !8
  br label %318, !dbg !1570

318:                                              ; preds = %336, %313
  %319 = load i32, ptr %168, align 4, !dbg !1572, !tbaa !8
  %320 = icmp slt i32 %319, 2, !dbg !1573
  br i1 %320, label %322, label %321, !dbg !1574

321:                                              ; preds = %318
  store i32 5, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %33) #25, !dbg !1574
  br label %339

322:                                              ; preds = %318
  %323 = load i32, ptr %167, align 4, !dbg !1575, !tbaa !8
  %324 = mul i32 %323, 2, !dbg !1576
  %325 = load i32, ptr %168, align 4, !dbg !1577, !tbaa !8
  %326 = add i32 %324, %325, !dbg !1578
  %327 = mul i32 %326, 256, !dbg !1579
  %328 = load i32, ptr %165, align 4, !dbg !1580, !tbaa !8
  %329 = add i32 %327, %328, !dbg !1581
  %330 = load i32, ptr %166, align 4, !dbg !1582, !tbaa !8
  %331 = sext i32 %330 to i64, !dbg !1583
  %332 = getelementptr inbounds [4 x [2 x i32]], ptr %163, i64 0, i64 %331, !dbg !1583
  %333 = load i32, ptr %168, align 4, !dbg !1584, !tbaa !8
  %334 = sext i32 %333 to i64, !dbg !1583
  %335 = getelementptr inbounds [2 x i32], ptr %332, i64 0, i64 %334, !dbg !1583
  store i32 %329, ptr %335, align 4, !dbg !1585, !tbaa !8
  br label %336, !dbg !1586

336:                                              ; preds = %322
  %337 = load i32, ptr %168, align 4, !dbg !1587, !tbaa !8
  %338 = add nsw i32 %337, 1, !dbg !1587
  store i32 %338, ptr %168, align 4, !dbg !1587, !tbaa !8
  br label %318, !dbg !1574, !llvm.loop !1588

339:                                              ; preds = %321
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %32) #25, !dbg !1589
  br label %340, !dbg !1589

340:                                              ; preds = %339
  %341 = load i32, ptr %166, align 4, !dbg !1590, !tbaa !8
  %342 = add nsw i32 %341, 1, !dbg !1590
  store i32 %342, ptr %166, align 4, !dbg !1590, !tbaa !8
  br label %309, !dbg !1563, !llvm.loop !1591

343:                                              ; preds = %312
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %34) #25, !dbg !1592
  store i32 0, ptr %169, align 4, !dbg !1593, !tbaa !8
  br label %344, !dbg !1592

344:                                              ; preds = %375, %343
  %345 = load i32, ptr %169, align 4, !dbg !1594, !tbaa !8
  %346 = icmp slt i32 %345, 2, !dbg !1595
  br i1 %346, label %348, label %347, !dbg !1596

347:                                              ; preds = %344
  store i32 8, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %34) #25, !dbg !1596
  br label %378

348:                                              ; preds = %344
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %35) #25, !dbg !1597
  %349 = load i32, ptr %156, align 4, !dbg !1598, !tbaa !8
  %350 = mul nsw i32 %349, 2, !dbg !1599
  %351 = load i32, ptr %169, align 4, !dbg !1600, !tbaa !8
  %352 = add nsw i32 %350, %351, !dbg !1601
  store i32 %352, ptr %170, align 4, !dbg !1602, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %36) #25, !dbg !1603
  store i32 0, ptr %171, align 4, !dbg !1604, !tbaa !8
  br label %353, !dbg !1603

353:                                              ; preds = %371, %348
  %354 = load i32, ptr %171, align 4, !dbg !1605, !tbaa !8
  %355 = icmp slt i32 %354, 2, !dbg !1606
  br i1 %355, label %357, label %356, !dbg !1607

356:                                              ; preds = %353
  store i32 11, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %36) #25, !dbg !1607
  br label %374

357:                                              ; preds = %353
  %358 = load i32, ptr %170, align 4, !dbg !1608, !tbaa !8
  %359 = mul i32 %358, 2, !dbg !1609
  %360 = load i32, ptr %171, align 4, !dbg !1610, !tbaa !8
  %361 = add i32 %359, %360, !dbg !1611
  %362 = mul i32 %361, 256, !dbg !1612
  %363 = load i32, ptr %165, align 4, !dbg !1613, !tbaa !8
  %364 = add i32 %362, %363, !dbg !1614
  %365 = load i32, ptr %169, align 4, !dbg !1615, !tbaa !8
  %366 = sext i32 %365 to i64, !dbg !1616
  %367 = getelementptr inbounds [2 x [2 x i32]], ptr %164, i64 0, i64 %366, !dbg !1616
  %368 = load i32, ptr %171, align 4, !dbg !1617, !tbaa !8
  %369 = sext i32 %368 to i64, !dbg !1616
  %370 = getelementptr inbounds [2 x i32], ptr %367, i64 0, i64 %369, !dbg !1616
  store i32 %364, ptr %370, align 4, !dbg !1618, !tbaa !8
  br label %371, !dbg !1619

371:                                              ; preds = %357
  %372 = load i32, ptr %171, align 4, !dbg !1620, !tbaa !8
  %373 = add nsw i32 %372, 1, !dbg !1620
  store i32 %373, ptr %171, align 4, !dbg !1620, !tbaa !8
  br label %353, !dbg !1607, !llvm.loop !1621

374:                                              ; preds = %356
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %35) #25, !dbg !1622
  br label %375, !dbg !1622

375:                                              ; preds = %374
  %376 = load i32, ptr %169, align 4, !dbg !1623, !tbaa !8
  %377 = add nsw i32 %376, 1, !dbg !1623
  store i32 %377, ptr %169, align 4, !dbg !1623, !tbaa !8
  br label %344, !dbg !1596, !llvm.loop !1624

378:                                              ; preds = %347
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %37) #25, !dbg !1625
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %38) #25, !dbg !1626
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %39) #25, !dbg !1627
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %40) #25, !dbg !1628
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %41) #25, !dbg !1629
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %42) #25, !dbg !1630
  store i32 0, ptr %177, align 4, !dbg !1631, !tbaa !8
  br label %379, !dbg !1630

379:                                              ; preds = %471, %378
  %380 = load i32, ptr %177, align 4, !dbg !1632, !tbaa !8
  %381 = icmp slt i32 %380, 2, !dbg !1633
  br i1 %381, label %383, label %382, !dbg !1634

382:                                              ; preds = %379
  store i32 14, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %42) #25, !dbg !1634
  br label %474

383:                                              ; preds = %379
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %43) #25, !dbg !1635
  %384 = load i32, ptr %152, align 4, !dbg !1636, !tbaa !8
  %385 = load i32, ptr %177, align 4, !dbg !1637, !tbaa !8
  %386 = mul nsw i32 %385, 64, !dbg !1638
  %387 = add nsw i32 %384, %386, !dbg !1639
  %388 = load i32, ptr %148, align 4, !dbg !1640, !tbaa !8
  %389 = ashr i32 %388, 2, !dbg !1641
  %390 = add nsw i32 %387, %389, !dbg !1642
  store i32 %390, ptr %178, align 4, !dbg !1643, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %44) #25, !dbg !1644
  %391 = load i32, ptr %178, align 4, !dbg !1645, !tbaa !8
  %392 = load i32, ptr %147, align 4, !dbg !1646, !tbaa !8
  %393 = icmp slt i32 %391, %392, !dbg !1647
  br i1 %393, label %394, label %396, !dbg !1648

394:                                              ; preds = %383
  %395 = load i32, ptr %178, align 4, !dbg !1649, !tbaa !8
  br label %399, !dbg !1648

396:                                              ; preds = %383
  %397 = load i32, ptr %147, align 4, !dbg !1650, !tbaa !8
  %398 = sub nsw i32 %397, 1, !dbg !1651
  br label %399, !dbg !1648

399:                                              ; preds = %396, %394
  %400 = phi i32 [ %395, %394 ], [ %398, %396 ], !dbg !1648
  store i32 %400, ptr %179, align 4, !dbg !1652, !tbaa !8
  %401 = load i32, ptr %178, align 4, !dbg !1653, !tbaa !8
  %402 = load i32, ptr %147, align 4, !dbg !1654, !tbaa !8
  %403 = icmp slt i32 %401, %402, !dbg !1655
  %404 = zext i1 %403 to i64, !dbg !1656
  %405 = select i1 %403, i32 1, i32 0, !dbg !1656
  %406 = load i32, ptr %177, align 4, !dbg !1657, !tbaa !8
  %407 = sext i32 %406 to i64, !dbg !1658
  %408 = getelementptr inbounds [2 x i32], ptr %176, i64 0, i64 %407, !dbg !1658
  store i32 %405, ptr %408, align 4, !dbg !1659, !tbaa !8
  %409 = load i32, ptr %179, align 4, !dbg !1660, !tbaa !8
  %410 = mul i32 %409, 72, !dbg !1661
  %411 = load i32, ptr %148, align 4, !dbg !1662, !tbaa !8
  %412 = and i32 %411, 3, !dbg !1663
  %413 = mul i32 %412, 8, !dbg !1664
  %414 = add i32 %410, %413, !dbg !1665
  %415 = load i32, ptr %177, align 4, !dbg !1666, !tbaa !8
  %416 = sext i32 %415 to i64, !dbg !1667
  %417 = getelementptr inbounds [2 x i32], ptr %173, i64 0, i64 %416, !dbg !1667
  store i32 %414, ptr %417, align 4, !dbg !1668, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %45) #25, !dbg !1669
  %418 = load i32, ptr %151, align 4, !dbg !1670, !tbaa !8
  %419 = load i32, ptr %177, align 4, !dbg !1671, !tbaa !8
  %420 = mul nsw i32 %419, 64, !dbg !1672
  %421 = add nsw i32 %418, %420, !dbg !1673
  %422 = load i32, ptr %148, align 4, !dbg !1674, !tbaa !8
  %423 = ashr i32 %422, 2, !dbg !1675
  %424 = add nsw i32 %421, %423, !dbg !1676
  store i32 %424, ptr %180, align 4, !dbg !1677, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %46) #25, !dbg !1678
  %425 = load i32, ptr %180, align 4, !dbg !1679, !tbaa !8
  %426 = load i32, ptr %145, align 4, !dbg !1680, !tbaa !8
  %427 = icmp slt i32 %425, %426, !dbg !1681
  br i1 %427, label %428, label %430, !dbg !1682

428:                                              ; preds = %399
  %429 = load i32, ptr %180, align 4, !dbg !1683, !tbaa !8
  br label %433, !dbg !1682

430:                                              ; preds = %399
  %431 = load i32, ptr %145, align 4, !dbg !1684, !tbaa !8
  %432 = sub nsw i32 %431, 1, !dbg !1685
  br label %433, !dbg !1682

433:                                              ; preds = %430, %428
  %434 = phi i32 [ %429, %428 ], [ %432, %430 ], !dbg !1682
  store i32 %434, ptr %181, align 4, !dbg !1686, !tbaa !8
  %435 = load i32, ptr %181, align 4, !dbg !1687, !tbaa !8
  %436 = load i32, ptr %160, align 4, !dbg !1688, !tbaa !8
  %437 = mul i32 %435, %436, !dbg !1689
  %438 = mul i32 %437, 136, !dbg !1690
  %439 = load i32, ptr %148, align 4, !dbg !1691, !tbaa !8
  %440 = and i32 %439, 3, !dbg !1692
  %441 = mul i32 %440, 8, !dbg !1693
  %442 = add i32 %438, %441, !dbg !1694
  %443 = load i32, ptr %177, align 4, !dbg !1695, !tbaa !8
  %444 = sext i32 %443 to i64, !dbg !1696
  %445 = getelementptr inbounds [2 x i32], ptr %174, i64 0, i64 %444, !dbg !1696
  store i32 %442, ptr %445, align 4, !dbg !1697, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %47) #25, !dbg !1698
  %446 = load i32, ptr %177, align 4, !dbg !1699, !tbaa !8
  %447 = mul nsw i32 %446, 64, !dbg !1700
  %448 = load i32, ptr %148, align 4, !dbg !1701, !tbaa !8
  %449 = ashr i32 %448, 2, !dbg !1702
  %450 = add nsw i32 %447, %449, !dbg !1703
  store i32 %450, ptr %182, align 4, !dbg !1704, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %48) #25, !dbg !1705
  %451 = load i32, ptr %148, align 4, !dbg !1706, !tbaa !8
  %452 = and i32 %451, 3, !dbg !1707
  store i32 %452, ptr %183, align 4, !dbg !1708, !tbaa !8
  %453 = load i32, ptr %182, align 4, !dbg !1709, !tbaa !8
  %454 = lshr i32 %453, 4, !dbg !1710
  %455 = mul i32 %454, 2, !dbg !1711
  %456 = load i32, ptr %183, align 4, !dbg !1712, !tbaa !8
  %457 = lshr i32 %456, 1, !dbg !1713
  %458 = add i32 %455, %457, !dbg !1714
  %459 = mul i32 %458, 256, !dbg !1715
  %460 = load i32, ptr %182, align 4, !dbg !1716, !tbaa !8
  %461 = and i32 %460, 15, !dbg !1717
  %462 = load i32, ptr %183, align 4, !dbg !1718, !tbaa !8
  %463 = and i32 %462, 1, !dbg !1719
  %464 = mul i32 %463, 16, !dbg !1720
  %465 = add i32 %461, %464, !dbg !1721
  %466 = mul i32 %465, 8, !dbg !1722
  %467 = add i32 %459, %466, !dbg !1723
  %468 = load i32, ptr %177, align 4, !dbg !1724, !tbaa !8
  %469 = sext i32 %468 to i64, !dbg !1725
  %470 = getelementptr inbounds [2 x i32], ptr %175, i64 0, i64 %469, !dbg !1725
  store i32 %467, ptr %470, align 4, !dbg !1726, !tbaa !8
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %48) #25, !dbg !1727
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %47) #25, !dbg !1727
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %46) #25, !dbg !1727
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %45) #25, !dbg !1727
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %44) #25, !dbg !1727
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %43) #25, !dbg !1727
  br label %471, !dbg !1727

471:                                              ; preds = %433
  %472 = load i32, ptr %177, align 4, !dbg !1728, !tbaa !8
  %473 = add nsw i32 %472, 1, !dbg !1728
  store i32 %473, ptr %177, align 4, !dbg !1728, !tbaa !8
  br label %379, !dbg !1634, !llvm.loop !1729

474:                                              ; preds = %382
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %49) #25, !dbg !1730
  store <2 x i32> zeroinitializer, ptr %184, align 8, !dbg !1731, !tbaa !47
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %50) #25, !dbg !1732
  store i32 0, ptr %185, align 4, !dbg !1733, !tbaa !8
  br label %475, !dbg !1732

475:                                              ; preds = %496, %474
  %476 = load i32, ptr %185, align 4, !dbg !1734, !tbaa !8
  %477 = icmp slt i32 %476, 4, !dbg !1735
  br i1 %477, label %479, label %478, !dbg !1736

478:                                              ; preds = %475
  store i32 17, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %50) #25, !dbg !1736
  br label %499

479:                                              ; preds = %475
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %51) #25, !dbg !1737
  store i32 0, ptr %186, align 4, !dbg !1738, !tbaa !8
  br label %480, !dbg !1737

480:                                              ; preds = %492, %479
  %481 = load i32, ptr %186, align 4, !dbg !1739, !tbaa !8
  %482 = icmp slt i32 %481, 2, !dbg !1740
  br i1 %482, label %484, label %483, !dbg !1741

483:                                              ; preds = %480
  store i32 20, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %51) #25, !dbg !1741
  br label %495

484:                                              ; preds = %480
  store <8 x float> zeroinitializer, ptr addrspace(5) %52, align 32, !dbg !1742, !tbaa !47
  %485 = load <8 x float>, ptr addrspace(5) %52, align 32, !dbg !1742, !tbaa !47
  %486 = load i32, ptr %185, align 4, !dbg !1743, !tbaa !8
  %487 = sext i32 %486 to i64, !dbg !1744
  %488 = getelementptr inbounds [4 x [2 x <8 x float>]], ptr %172, i64 0, i64 %487, !dbg !1744
  %489 = load i32, ptr %186, align 4, !dbg !1745, !tbaa !8
  %490 = sext i32 %489 to i64, !dbg !1744
  %491 = getelementptr inbounds [2 x <8 x float>], ptr %488, i64 0, i64 %490, !dbg !1744
  store <8 x float> %485, ptr %491, align 32, !dbg !1746, !tbaa !47
  br label %492, !dbg !1747

492:                                              ; preds = %484
  %493 = load i32, ptr %186, align 4, !dbg !1748, !tbaa !8
  %494 = add nsw i32 %493, 1, !dbg !1748
  store i32 %494, ptr %186, align 4, !dbg !1748, !tbaa !8
  br label %480, !dbg !1741, !llvm.loop !1749

495:                                              ; preds = %483
  br label %496, !dbg !1750

496:                                              ; preds = %495
  %497 = load i32, ptr %185, align 4, !dbg !1751, !tbaa !8
  %498 = add nsw i32 %497, 1, !dbg !1751
  store i32 %498, ptr %185, align 4, !dbg !1751, !tbaa !8
  br label %475, !dbg !1736, !llvm.loop !1752

499:                                              ; preds = %478
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %53) #25, !dbg !1753
  store <8 x i32> zeroinitializer, ptr addrspace(5) %54, align 32, !dbg !1754, !tbaa !47
  %500 = load <8 x i32>, ptr addrspace(5) %54, align 32, !dbg !1754, !tbaa !47
  store <8 x i32> %500, ptr %187, align 32, !dbg !1755, !tbaa !47
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %55) #25, !dbg !1756
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %56) #25, !dbg !1757
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %57) #25, !dbg !1758
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %58) #25, !dbg !1759
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %59) #25, !dbg !1760
  store i32 0, ptr %192, align 4, !dbg !1761, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %60) #25, !dbg !1762
  store i32 0, ptr %193, align 4, !dbg !1763, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %61) #25, !dbg !1764
  %501 = load i32, ptr %148, align 4, !dbg !1765, !tbaa !8
  store i32 %501, ptr %194, align 4, !dbg !1766, !tbaa !8
  br label %502, !dbg !1764

502:                                              ; preds = %537, %499
  %503 = load i32, ptr %194, align 4, !dbg !1767, !tbaa !8
  %504 = icmp slt i32 %503, 256, !dbg !1768
  br i1 %504, label %506, label %505, !dbg !1769

505:                                              ; preds = %502
  store i32 23, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %61) #25, !dbg !1769
  br label %540

506:                                              ; preds = %502
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %62) #25, !dbg !1770
  %507 = load i32, ptr %194, align 4, !dbg !1771, !tbaa !8
  %508 = ashr i32 %507, 1, !dbg !1772
  store i32 %508, ptr %195, align 4, !dbg !1773, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %63) #25, !dbg !1774
  %509 = load i32, ptr %194, align 4, !dbg !1775, !tbaa !8
  %510 = and i32 %509, 1, !dbg !1776
  store i32 %510, ptr %196, align 4, !dbg !1777, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %64) #25, !dbg !1778
  %511 = load i32, ptr %152, align 4, !dbg !1779, !tbaa !8
  %512 = load i32, ptr %195, align 4, !dbg !1780, !tbaa !8
  %513 = add nsw i32 %511, %512, !dbg !1781
  store i32 %513, ptr %197, align 4, !dbg !1782, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %65) #25, !dbg !1783
  %514 = load i32, ptr %197, align 4, !dbg !1784, !tbaa !8
  %515 = load i32, ptr %147, align 4, !dbg !1785, !tbaa !8
  %516 = icmp slt i32 %514, %515, !dbg !1786
  br i1 %516, label %517, label %526, !dbg !1787

517:                                              ; preds = %506
  %518 = load ptr, ptr %143, align 8, !dbg !1788, !tbaa !130
  %519 = load i32, ptr %197, align 4, !dbg !1789, !tbaa !8
  %520 = sext i32 %519 to i64, !dbg !1790
  %521 = getelementptr inbounds %struct.block_i4_128, ptr %518, i64 %520, !dbg !1790
  %522 = load i32, ptr %196, align 4, !dbg !1791, !tbaa !8
  %523 = sext i32 %522 to i64, !dbg !1790
  %524 = getelementptr inbounds i32, ptr %521, i64 %523, !dbg !1790
  %525 = load i32, ptr %524, align 4, !dbg !1790, !tbaa !8
  br label %527, !dbg !1787

526:                                              ; preds = %506
  br label %527, !dbg !1787

527:                                              ; preds = %526, %517
  %528 = phi i32 [ %525, %517 ], [ 0, %526 ], !dbg !1787
  store i32 %528, ptr %198, align 4, !dbg !1792, !tbaa !8
  %529 = load i32, ptr %198, align 4, !dbg !1793, !tbaa !8
  %530 = load i32, ptr %195, align 4, !dbg !1794, !tbaa !8
  %531 = mul nsw i32 %530, 8, !dbg !1795
  %532 = sext i32 %531 to i64, !dbg !1796
  %533 = getelementptr inbounds i8, ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @LDS to ptr), i64 12288), i64 %532, !dbg !1796
  %534 = load i32, ptr %196, align 4, !dbg !1797, !tbaa !8
  %535 = sext i32 %534 to i64, !dbg !1796
  %536 = getelementptr inbounds i32, ptr %533, i64 %535, !dbg !1796
  store i32 %529, ptr %536, align 1, !dbg !1798, !tbaa !8
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %65) #25, !dbg !1799
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %64) #25, !dbg !1799
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %63) #25, !dbg !1799
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %62) #25, !dbg !1799
  br label %537, !dbg !1799

537:                                              ; preds = %527
  %538 = load i32, ptr %194, align 4, !dbg !1800, !tbaa !8
  %539 = add nsw i32 %538, 256, !dbg !1800
  store i32 %539, ptr %194, align 4, !dbg !1800, !tbaa !8
  br label %502, !dbg !1769, !llvm.loop !1801

540:                                              ; preds = %505
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %66) #25, !dbg !1802
  %541 = load i32, ptr %148, align 4, !dbg !1803, !tbaa !8
  store i32 %541, ptr %199, align 4, !dbg !1804, !tbaa !8
  br label %542, !dbg !1802

542:                                              ; preds = %616, %540
  %543 = load i32, ptr %199, align 4, !dbg !1805, !tbaa !8
  %544 = icmp slt i32 %543, 256, !dbg !1806
  br i1 %544, label %546, label %545, !dbg !1807

545:                                              ; preds = %542
  store i32 26, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %66) #25, !dbg !1807
  br label %619

546:                                              ; preds = %542
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %67) #25, !dbg !1808
  %547 = load i32, ptr %199, align 4, !dbg !1809, !tbaa !8
  %548 = ashr i32 %547, 1, !dbg !1810
  store i32 %548, ptr %200, align 4, !dbg !1811, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %68) #25, !dbg !1812
  %549 = load i32, ptr %199, align 4, !dbg !1813, !tbaa !8
  %550 = and i32 %549, 1, !dbg !1814
  store i32 %550, ptr %201, align 4, !dbg !1815, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %69) #25, !dbg !1816
  %551 = load i32, ptr %151, align 4, !dbg !1817, !tbaa !8
  %552 = load i32, ptr %200, align 4, !dbg !1818, !tbaa !8
  %553 = add nsw i32 %551, %552, !dbg !1819
  store i32 %553, ptr %202, align 4, !dbg !1820, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %70) #25, !dbg !1821
  %554 = load i32, ptr %202, align 4, !dbg !1822, !tbaa !8
  %555 = load i32, ptr %145, align 4, !dbg !1823, !tbaa !8
  %556 = icmp slt i32 %554, %555, !dbg !1824
  br i1 %556, label %557, label %559, !dbg !1825

557:                                              ; preds = %546
  %558 = load i32, ptr %202, align 4, !dbg !1826, !tbaa !8
  br label %562, !dbg !1825

559:                                              ; preds = %546
  %560 = load i32, ptr %145, align 4, !dbg !1827, !tbaa !8
  %561 = sub nsw i32 %560, 1, !dbg !1828
  br label %562, !dbg !1825

562:                                              ; preds = %559, %557
  %563 = phi i32 [ %558, %557 ], [ %561, %559 ], !dbg !1825
  store i32 %563, ptr %203, align 4, !dbg !1829, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %71) #25, !dbg !1830
  %564 = load ptr, ptr %142, align 8, !dbg !1831, !tbaa !28
  %565 = load i32, ptr %203, align 4, !dbg !1832, !tbaa !8
  %566 = sext i32 %565 to i64, !dbg !1832
  %567 = load i32, ptr %160, align 4, !dbg !1833, !tbaa !8
  %568 = sext i32 %567 to i64, !dbg !1833
  %569 = mul nsw i64 %566, %568, !dbg !1834
  %570 = mul nsw i64 %569, 136, !dbg !1835
  %571 = getelementptr inbounds i8, ptr %564, i64 %570, !dbg !1836
  %572 = load i32, ptr %571, align 4, !dbg !1836, !tbaa !8
  store i32 %572, ptr %204, align 4, !dbg !1837, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %72) #25, !dbg !1838
  %573 = load i32, ptr %201, align 4, !dbg !1839, !tbaa !8
  %574 = icmp eq i32 %573, 0, !dbg !1840
  br i1 %574, label %575, label %587, !dbg !1841

575:                                              ; preds = %562
  %576 = addrspacecast ptr %206 to ptr addrspace(5), !dbg !1842
  %577 = load i32, ptr %204, align 4, !dbg !1843, !tbaa !8
  %578 = and i32 %577, 65535, !dbg !1844
  %579 = trunc i32 %578 to i16, !dbg !1845
  %580 = call i16 @_Z16__ushort_as_halft(i16 noundef zeroext %579) #26, !dbg !1842
  %581 = getelementptr inbounds nuw %struct.__half, ptr addrspace(5) %576, i32 0, i32 0, !dbg !1842
  %582 = getelementptr inbounds nuw %union.anon.0, ptr addrspace(5) %581, i32 0, i32 0, !dbg !1842
  store i16 %580, ptr addrspace(5) %582, align 2, !dbg !1842
  %583 = getelementptr inbounds nuw %struct.__half, ptr %206, i32 0, i32 0, !dbg !1846
  %584 = getelementptr inbounds nuw %union.anon.0, ptr %583, i32 0, i32 0, !dbg !1846
  %585 = load i16, ptr %584, align 2, !dbg !1846
  %586 = call contract noundef float @_Z12__half2float6__half(i16 %585) #26, !dbg !1846
  br label %599, !dbg !1841

587:                                              ; preds = %562
  %588 = addrspacecast ptr %207 to ptr addrspace(5), !dbg !1847
  %589 = load i32, ptr %204, align 4, !dbg !1848, !tbaa !8
  %590 = lshr i32 %589, 16, !dbg !1849
  %591 = trunc i32 %590 to i16, !dbg !1850
  %592 = call i16 @_Z16__ushort_as_halft(i16 noundef zeroext %591) #26, !dbg !1847
  %593 = getelementptr inbounds nuw %struct.__half, ptr addrspace(5) %588, i32 0, i32 0, !dbg !1847
  %594 = getelementptr inbounds nuw %union.anon.0, ptr addrspace(5) %593, i32 0, i32 0, !dbg !1847
  store i16 %592, ptr addrspace(5) %594, align 2, !dbg !1847
  %595 = getelementptr inbounds nuw %struct.__half, ptr %207, i32 0, i32 0, !dbg !1851
  %596 = getelementptr inbounds nuw %union.anon.0, ptr %595, i32 0, i32 0, !dbg !1851
  %597 = load i16, ptr %596, align 2, !dbg !1851
  %598 = call contract noundef float @_Z12__half2float6__half(i16 %597) #26, !dbg !1851
  br label %599, !dbg !1841

599:                                              ; preds = %587, %575
  %600 = phi contract float [ %586, %575 ], [ %598, %587 ], !dbg !1841
  store float %600, ptr %205, align 4, !dbg !1852, !tbaa !177
  %601 = load i32, ptr %202, align 4, !dbg !1853, !tbaa !8
  %602 = load i32, ptr %145, align 4, !dbg !1854, !tbaa !8
  %603 = icmp slt i32 %601, %602, !dbg !1855
  br i1 %603, label %604, label %606, !dbg !1856

604:                                              ; preds = %599
  %605 = load float, ptr %205, align 4, !dbg !1857, !tbaa !177
  br label %607, !dbg !1856

606:                                              ; preds = %599
  br label %607, !dbg !1856

607:                                              ; preds = %606, %604
  %608 = phi contract float [ %605, %604 ], [ 0.000000e+00, %606 ], !dbg !1856
  %609 = load i32, ptr %200, align 4, !dbg !1858, !tbaa !8
  %610 = mul nsw i32 %609, 8, !dbg !1859
  %611 = sext i32 %610 to i64, !dbg !1860
  %612 = getelementptr inbounds i8, ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @LDS to ptr), i64 14336), i64 %611, !dbg !1860
  %613 = load i32, ptr %201, align 4, !dbg !1861, !tbaa !8
  %614 = sext i32 %613 to i64, !dbg !1860
  %615 = getelementptr inbounds float, ptr %612, i64 %614, !dbg !1860
  store float %608, ptr %615, align 1, !dbg !1862, !tbaa !177
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %72) #25, !dbg !1863
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %71) #25, !dbg !1863
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %70) #25, !dbg !1863
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %69) #25, !dbg !1863
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %68) #25, !dbg !1863
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %67) #25, !dbg !1863
  br label %616, !dbg !1863

616:                                              ; preds = %607
  %617 = load i32, ptr %199, align 4, !dbg !1864, !tbaa !8
  %618 = add nsw i32 %617, 256, !dbg !1864
  store i32 %618, ptr %199, align 4, !dbg !1864, !tbaa !8
  br label %542, !dbg !1807, !llvm.loop !1865

619:                                              ; preds = %545
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %75) #25, !dbg !1866
  %620 = load ptr, ptr %143, align 8, !dbg !1867, !tbaa !130
  %621 = getelementptr inbounds i8, ptr %620, i64 8, !dbg !1868
  store ptr %621, ptr %208, align 8, !dbg !1869, !tbaa !28
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %76) #25, !dbg !1870
  %622 = load ptr, ptr %142, align 8, !dbg !1871, !tbaa !28
  %623 = getelementptr inbounds i8, ptr %622, i64 8, !dbg !1872
  store ptr %623, ptr %209, align 8, !dbg !1873, !tbaa !28
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %77) #25, !dbg !1874
  store i32 0, ptr %210, align 4, !dbg !1875, !tbaa !8
  br label %624, !dbg !1874

624:                                              ; preds = %669, %619
  %625 = load i32, ptr %210, align 4, !dbg !1876, !tbaa !8
  %626 = icmp slt i32 %625, 2, !dbg !1877
  br i1 %626, label %628, label %627, !dbg !1878

627:                                              ; preds = %624
  store i32 29, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %77) #25, !dbg !1878
  br label %672

628:                                              ; preds = %624
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %78) #25, !dbg !1879
  %629 = load ptr, ptr %208, align 8, !dbg !1880, !tbaa !28
  %630 = load i32, ptr %210, align 4, !dbg !1881, !tbaa !8
  %631 = sext i32 %630 to i64, !dbg !1882
  %632 = getelementptr inbounds [2 x i32], ptr %173, i64 0, i64 %631, !dbg !1882
  %633 = load i32, ptr %632, align 4, !dbg !1882, !tbaa !8
  %634 = zext i32 %633 to i64, !dbg !1883
  %635 = getelementptr inbounds nuw i8, ptr %629, i64 %634, !dbg !1883
  %636 = load <2 x i32>, ptr %635, align 8, !dbg !1883, !tbaa !47
  store <2 x i32> %636, ptr %211, align 8, !dbg !1884, !tbaa !47
  %637 = load i32, ptr %210, align 4, !dbg !1885, !tbaa !8
  %638 = sext i32 %637 to i64, !dbg !1886
  %639 = getelementptr inbounds [2 x i32], ptr %176, i64 0, i64 %638, !dbg !1886
  %640 = load i32, ptr %639, align 4, !dbg !1886, !tbaa !8
  %641 = icmp ne i32 %640, 0, !dbg !1886
  br i1 %641, label %642, label %644, !dbg !1886

642:                                              ; preds = %628
  %643 = load <2 x i32>, ptr %211, align 8, !dbg !1887, !tbaa !47
  br label %645, !dbg !1886

644:                                              ; preds = %628
  br label %645, !dbg !1886

645:                                              ; preds = %644, %642
  %646 = phi <2 x i32> [ %643, %642 ], [ zeroinitializer, %644 ], !dbg !1886
  %647 = load ptr, ptr %161, align 8, !dbg !1888, !tbaa !28
  %648 = load i32, ptr %210, align 4, !dbg !1889, !tbaa !8
  %649 = sext i32 %648 to i64, !dbg !1890
  %650 = getelementptr inbounds [2 x i32], ptr %175, i64 0, i64 %649, !dbg !1890
  %651 = load i32, ptr %650, align 4, !dbg !1890, !tbaa !8
  %652 = zext i32 %651 to i64, !dbg !1891
  %653 = getelementptr inbounds nuw i8, ptr %647, i64 %652, !dbg !1891
  store <2 x i32> %646, ptr %653, align 8, !dbg !1892, !tbaa !47
  %654 = load ptr, ptr %209, align 8, !dbg !1893, !tbaa !28
  %655 = load i32, ptr %210, align 4, !dbg !1894, !tbaa !8
  %656 = sext i32 %655 to i64, !dbg !1895
  %657 = getelementptr inbounds [2 x i32], ptr %174, i64 0, i64 %656, !dbg !1895
  %658 = load i32, ptr %657, align 4, !dbg !1895, !tbaa !8
  %659 = zext i32 %658 to i64, !dbg !1896
  %660 = getelementptr inbounds nuw i8, ptr %654, i64 %659, !dbg !1896
  %661 = load <2 x i32>, ptr %660, align 8, !dbg !1896, !tbaa !47
  %662 = load ptr, ptr %162, align 8, !dbg !1897, !tbaa !28
  %663 = load i32, ptr %210, align 4, !dbg !1898, !tbaa !8
  %664 = sext i32 %663 to i64, !dbg !1899
  %665 = getelementptr inbounds [2 x i32], ptr %175, i64 0, i64 %664, !dbg !1899
  %666 = load i32, ptr %665, align 4, !dbg !1899, !tbaa !8
  %667 = zext i32 %666 to i64, !dbg !1900
  %668 = getelementptr inbounds nuw i8, ptr %662, i64 %667, !dbg !1900
  store <2 x i32> %661, ptr %668, align 8, !dbg !1901, !tbaa !47
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %78) #25, !dbg !1902
  br label %669, !dbg !1902

669:                                              ; preds = %645
  %670 = load i32, ptr %210, align 4, !dbg !1903, !tbaa !8
  %671 = add nsw i32 %670, 1, !dbg !1903
  store i32 %671, ptr %210, align 4, !dbg !1903, !tbaa !8
  br label %624, !dbg !1878, !llvm.loop !1904

672:                                              ; preds = %627
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %76) #25, !dbg !1905
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %75) #25, !dbg !1905
  call void @_Z13__syncthreadsv() #26, !dbg !1906
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %79) #25, !dbg !1907
  store i32 0, ptr %212, align 4, !dbg !1908, !tbaa !8
  br label %673, !dbg !1907

673:                                              ; preds = %1246, %672
  %674 = load i32, ptr %212, align 4, !dbg !1909, !tbaa !8
  %675 = load i32, ptr %160, align 4, !dbg !1910, !tbaa !8
  %676 = icmp slt i32 %674, %675, !dbg !1911
  br i1 %676, label %678, label %677, !dbg !1912

677:                                              ; preds = %673
  store i32 32, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %79) #25, !dbg !1912
  br label %1249

678:                                              ; preds = %673
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %80) #25, !dbg !1913
  store i32 0, ptr %213, align 4, !dbg !1914, !tbaa !8
  br label %679, !dbg !1913

679:                                              ; preds = %1242, %678
  %680 = load i32, ptr %213, align 4, !dbg !1915, !tbaa !8
  %681 = icmp slt i32 %680, 2, !dbg !1916
  br i1 %681, label %683, label %682, !dbg !1917

682:                                              ; preds = %679
  store i32 35, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %80) #25, !dbg !1917
  br label %1245

683:                                              ; preds = %679
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %81) #25, !dbg !1918
  %684 = load i32, ptr %212, align 4, !dbg !1919, !tbaa !8
  %685 = mul nsw i32 2, %684, !dbg !1920
  %686 = load i32, ptr %213, align 4, !dbg !1921, !tbaa !8
  %687 = add nsw i32 %685, %686, !dbg !1922
  store i32 %687, ptr %214, align 4, !dbg !1923, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %82) #25, !dbg !1924
  %688 = load i32, ptr %213, align 4, !dbg !1925, !tbaa !8
  %689 = icmp eq i32 %688, 1, !dbg !1926
  br i1 %689, label %690, label %695, !dbg !1927

690:                                              ; preds = %683
  %691 = load i32, ptr %212, align 4, !dbg !1928, !tbaa !8
  %692 = add nsw i32 %691, 1, !dbg !1929
  %693 = load i32, ptr %160, align 4, !dbg !1930, !tbaa !8
  %694 = icmp eq i32 %692, %693, !dbg !1931
  br label %695

695:                                              ; preds = %690, %683
  %696 = phi i1 [ false, %683 ], [ %694, %690 ], !dbg !1932
  %697 = zext i1 %696 to i8, !dbg !1933
  store i8 %697, ptr %215, align 1, !dbg !1933, !tbaa !880
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %83) #25, !dbg !1934
  %698 = load i32, ptr %213, align 4, !dbg !1935, !tbaa !8
  %699 = icmp eq i32 %698, 0, !dbg !1936
  %700 = zext i1 %699 to i64, !dbg !1937
  %701 = select i1 %699, i32 12288, i32 13312, !dbg !1937
  %702 = sext i32 %701 to i64, !dbg !1938
  %703 = getelementptr inbounds i8, ptr addrspacecast (ptr addrspace(3) @LDS to ptr), i64 %702, !dbg !1938
  store ptr %703, ptr %216, align 8, !dbg !1939, !tbaa !28
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %84) #25, !dbg !1940
  %704 = load i32, ptr %213, align 4, !dbg !1941, !tbaa !8
  %705 = icmp eq i32 %704, 0, !dbg !1942
  %706 = zext i1 %705 to i64, !dbg !1943
  %707 = select i1 %705, i32 14336, i32 15360, !dbg !1943
  %708 = sext i32 %707 to i64, !dbg !1944
  %709 = getelementptr inbounds i8, ptr addrspacecast (ptr addrspace(3) @LDS to ptr), i64 %708, !dbg !1944
  store ptr %709, ptr %217, align 8, !dbg !1945, !tbaa !28
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %85) #25, !dbg !1946
  store i32 0, ptr %218, align 4, !dbg !1947, !tbaa !8
  br label %710, !dbg !1946

710:                                              ; preds = %745, %695
  %711 = load i32, ptr %218, align 4, !dbg !1948, !tbaa !8
  %712 = icmp slt i32 %711, 2, !dbg !1949
  br i1 %712, label %714, label %713, !dbg !1950

713:                                              ; preds = %710
  store i32 38, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %85) #25, !dbg !1950
  br label %748

714:                                              ; preds = %710
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %86) #25, !dbg !1951
  store i32 0, ptr %219, align 4, !dbg !1952, !tbaa !8
  br label %715, !dbg !1951

715:                                              ; preds = %741, %714
  %716 = load i32, ptr %219, align 4, !dbg !1953, !tbaa !8
  %717 = icmp slt i32 %716, 8, !dbg !1954
  br i1 %717, label %719, label %718, !dbg !1955

718:                                              ; preds = %715
  store i32 41, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %86) #25, !dbg !1955
  br label %744

719:                                              ; preds = %715
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %87) #25, !dbg !1956
  %720 = load i32, ptr %158, align 4, !dbg !1957, !tbaa !8
  %721 = load i32, ptr %218, align 4, !dbg !1958, !tbaa !8
  %722 = mul nsw i32 %721, 16, !dbg !1959
  %723 = add nsw i32 %720, %722, !dbg !1960
  %724 = load i32, ptr %155, align 4, !dbg !1961, !tbaa !8
  %725 = mul nsw i32 8, %724, !dbg !1962
  %726 = add nsw i32 %723, %725, !dbg !1963
  %727 = load i32, ptr %219, align 4, !dbg !1964, !tbaa !8
  %728 = add nsw i32 %726, %727, !dbg !1965
  %729 = mul i32 %728, 8, !dbg !1966
  store i32 %729, ptr %220, align 4, !dbg !1967, !tbaa !8
  %730 = load ptr, ptr %217, align 8, !dbg !1968, !tbaa !28
  %731 = load i32, ptr %220, align 4, !dbg !1969, !tbaa !8
  %732 = zext i32 %731 to i64, !dbg !1970
  %733 = getelementptr inbounds nuw i8, ptr %730, i64 %732, !dbg !1970
  %734 = load float, ptr %733, align 4, !dbg !1970, !tbaa !177
  %735 = load i32, ptr %218, align 4, !dbg !1971, !tbaa !8
  %736 = sext i32 %735 to i64, !dbg !1972
  %737 = getelementptr inbounds [2 x <8 x float>], ptr %188, i64 0, i64 %736, !dbg !1972
  %738 = load i32, ptr %219, align 4, !dbg !1973, !tbaa !8
  %739 = load <8 x float>, ptr %737, align 32, !dbg !1974
  %740 = insertelement <8 x float> %739, float %734, i32 %738, !dbg !1974
  store <8 x float> %740, ptr %737, align 32, !dbg !1974
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %87) #25, !dbg !1975
  br label %741, !dbg !1975

741:                                              ; preds = %719
  %742 = load i32, ptr %219, align 4, !dbg !1976, !tbaa !8
  %743 = add nsw i32 %742, 1, !dbg !1976
  store i32 %743, ptr %219, align 4, !dbg !1976, !tbaa !8
  br label %715, !dbg !1955, !llvm.loop !1977

744:                                              ; preds = %718
  br label %745, !dbg !1978

745:                                              ; preds = %744
  %746 = load i32, ptr %218, align 4, !dbg !1979, !tbaa !8
  %747 = add nsw i32 %746, 1, !dbg !1979
  store i32 %747, ptr %218, align 4, !dbg !1979, !tbaa !8
  br label %710, !dbg !1950, !llvm.loop !1980

748:                                              ; preds = %713
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %88) #25, !dbg !1981
  store i32 0, ptr %221, align 4, !dbg !1982, !tbaa !8
  br label %749, !dbg !1981

749:                                              ; preds = %769, %748
  %750 = load i32, ptr %221, align 4, !dbg !1983, !tbaa !8
  %751 = icmp slt i32 %750, 4, !dbg !1984
  br i1 %751, label %753, label %752, !dbg !1985

752:                                              ; preds = %749
  store i32 44, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %88) #25, !dbg !1985
  br label %772

753:                                              ; preds = %749
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %89) #25, !dbg !1986
  %754 = load i32, ptr %159, align 4, !dbg !1987, !tbaa !8
  %755 = load i32, ptr %221, align 4, !dbg !1988, !tbaa !8
  %756 = mul nsw i32 %755, 16, !dbg !1989
  %757 = add nsw i32 %754, %756, !dbg !1990
  %758 = load i32, ptr %154, align 4, !dbg !1991, !tbaa !8
  %759 = add nsw i32 %757, %758, !dbg !1992
  %760 = mul i32 %759, 8, !dbg !1993
  store i32 %760, ptr %222, align 4, !dbg !1994, !tbaa !8
  %761 = load ptr, ptr %216, align 8, !dbg !1995, !tbaa !28
  %762 = load i32, ptr %222, align 4, !dbg !1996, !tbaa !8
  %763 = zext i32 %762 to i64, !dbg !1997
  %764 = getelementptr inbounds nuw i8, ptr %761, i64 %763, !dbg !1997
  %765 = load float, ptr %764, align 4, !dbg !1997, !tbaa !177
  %766 = load i32, ptr %221, align 4, !dbg !1998, !tbaa !8
  %767 = sext i32 %766 to i64, !dbg !1999
  %768 = getelementptr inbounds [4 x float], ptr %189, i64 0, i64 %767, !dbg !1999
  store float %765, ptr %768, align 4, !dbg !2000, !tbaa !177
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %89) #25, !dbg !2001
  br label %769, !dbg !2001

769:                                              ; preds = %753
  %770 = load i32, ptr %221, align 4, !dbg !2002, !tbaa !8
  %771 = add nsw i32 %770, 1, !dbg !2002
  store i32 %771, ptr %221, align 4, !dbg !2002, !tbaa !8
  br label %749, !dbg !1985, !llvm.loop !2003

772:                                              ; preds = %752
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %90) #25, !dbg !2004
  %773 = load ptr, ptr %143, align 8, !dbg !2005, !tbaa !130
  %774 = load i32, ptr %214, align 4, !dbg !2006, !tbaa !8
  %775 = sext i32 %774 to i64, !dbg !2006
  %776 = load i32, ptr %147, align 4, !dbg !2007, !tbaa !8
  %777 = sext i32 %776 to i64, !dbg !2007
  %778 = mul nsw i64 %775, %777, !dbg !2008
  %779 = getelementptr inbounds %struct.block_i4_128, ptr %773, i64 %778, !dbg !2009
  %780 = getelementptr inbounds i8, ptr %779, i64 8, !dbg !2010
  %781 = getelementptr inbounds i8, ptr %780, i64 32, !dbg !2011
  store ptr %781, ptr %223, align 8, !dbg !2012, !tbaa !28
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %91) #25, !dbg !2013
  %782 = load ptr, ptr %142, align 8, !dbg !2014, !tbaa !28
  %783 = load i32, ptr %212, align 4, !dbg !2015, !tbaa !8
  %784 = sext i32 %783 to i64, !dbg !2015
  %785 = mul nsw i64 %784, 136, !dbg !2016
  %786 = getelementptr inbounds i8, ptr %782, i64 %785, !dbg !2017
  %787 = getelementptr inbounds i8, ptr %786, i64 8, !dbg !2018
  %788 = load i32, ptr %213, align 4, !dbg !2019, !tbaa !8
  %789 = sext i32 %788 to i64, !dbg !2019
  %790 = mul nsw i64 %789, 64, !dbg !2020
  %791 = getelementptr inbounds i8, ptr %787, i64 %790, !dbg !2021
  %792 = getelementptr inbounds i8, ptr %791, i64 32, !dbg !2022
  store ptr %792, ptr %224, align 8, !dbg !2023, !tbaa !28
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %92) #25, !dbg !2024
  store i32 0, ptr %225, align 4, !dbg !2025, !tbaa !8
  br label %793, !dbg !2024

793:                                              ; preds = %830, %772
  %794 = load i32, ptr %225, align 4, !dbg !2026, !tbaa !8
  %795 = icmp slt i32 %794, 2, !dbg !2027
  br i1 %795, label %797, label %796, !dbg !2028

796:                                              ; preds = %793
  store i32 47, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %92) #25, !dbg !2028
  br label %833

797:                                              ; preds = %793
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %93) #25, !dbg !2029
  %798 = load ptr, ptr %223, align 8, !dbg !2030, !tbaa !28
  %799 = load i32, ptr %225, align 4, !dbg !2031, !tbaa !8
  %800 = sext i32 %799 to i64, !dbg !2032
  %801 = getelementptr inbounds [2 x i32], ptr %173, i64 0, i64 %800, !dbg !2032
  %802 = load i32, ptr %801, align 4, !dbg !2032, !tbaa !8
  %803 = zext i32 %802 to i64, !dbg !2033
  %804 = getelementptr inbounds nuw i8, ptr %798, i64 %803, !dbg !2033
  %805 = load <2 x i32>, ptr %804, align 8, !dbg !2033, !tbaa !47
  store <2 x i32> %805, ptr %226, align 8, !dbg !2034, !tbaa !47
  %806 = load i32, ptr %225, align 4, !dbg !2035, !tbaa !8
  %807 = sext i32 %806 to i64, !dbg !2036
  %808 = getelementptr inbounds [2 x i32], ptr %176, i64 0, i64 %807, !dbg !2036
  %809 = load i32, ptr %808, align 4, !dbg !2036, !tbaa !8
  %810 = icmp ne i32 %809, 0, !dbg !2036
  br i1 %810, label %811, label %813, !dbg !2036

811:                                              ; preds = %797
  %812 = load <2 x i32>, ptr %226, align 8, !dbg !2037, !tbaa !47
  br label %814, !dbg !2036

813:                                              ; preds = %797
  br label %814, !dbg !2036

814:                                              ; preds = %813, %811
  %815 = phi <2 x i32> [ %812, %811 ], [ zeroinitializer, %813 ], !dbg !2036
  %816 = load i32, ptr %225, align 4, !dbg !2038, !tbaa !8
  %817 = sext i32 %816 to i64, !dbg !2039
  %818 = getelementptr inbounds [2 x <2 x i32>], ptr %190, i64 0, i64 %817, !dbg !2039
  store <2 x i32> %815, ptr %818, align 8, !dbg !2040, !tbaa !47
  %819 = load ptr, ptr %224, align 8, !dbg !2041, !tbaa !28
  %820 = load i32, ptr %225, align 4, !dbg !2042, !tbaa !8
  %821 = sext i32 %820 to i64, !dbg !2043
  %822 = getelementptr inbounds [2 x i32], ptr %174, i64 0, i64 %821, !dbg !2043
  %823 = load i32, ptr %822, align 4, !dbg !2043, !tbaa !8
  %824 = zext i32 %823 to i64, !dbg !2044
  %825 = getelementptr inbounds nuw i8, ptr %819, i64 %824, !dbg !2044
  %826 = load <2 x i32>, ptr %825, align 8, !dbg !2044, !tbaa !47
  %827 = load i32, ptr %225, align 4, !dbg !2045, !tbaa !8
  %828 = sext i32 %827 to i64, !dbg !2046
  %829 = getelementptr inbounds [2 x <2 x i32>], ptr %191, i64 0, i64 %828, !dbg !2046
  store <2 x i32> %826, ptr %829, align 8, !dbg !2047, !tbaa !47
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %93) #25, !dbg !2048
  br label %830, !dbg !2048

830:                                              ; preds = %814
  %831 = load i32, ptr %225, align 4, !dbg !2049, !tbaa !8
  %832 = add nsw i32 %831, 1, !dbg !2049
  store i32 %832, ptr %225, align 4, !dbg !2049, !tbaa !8
  br label %793, !dbg !2028, !llvm.loop !2050

833:                                              ; preds = %796
  %834 = load ptr, ptr %162, align 8, !dbg !2051, !tbaa !28
  %835 = load ptr, ptr %161, align 8, !dbg !2052, !tbaa !28
  %836 = getelementptr inbounds [4 x [2 x <8 x float>]], ptr %172, i64 0, i64 0, !dbg !2053
  %837 = getelementptr inbounds [2 x <8 x float>], ptr %188, i64 0, i64 0, !dbg !2054
  call void @_ZL21iu4_bundle_single_accILi0EEvPKcS1_RA2_A2_KjRA4_S3_PA2_Dv8_fPKS8_RA4_KfRKDv8_i(ptr noundef %834, ptr noundef %835, ptr noundef nonnull align 4 dereferenceable(16) %164, ptr noundef nonnull align 4 dereferenceable(32) %163, ptr noundef %836, ptr noundef %837, ptr noundef nonnull align 4 dereferenceable(16) %189, ptr noundef nonnull align 32 dereferenceable(32) %187) #26, !dbg !2055
  %838 = load ptr, ptr %162, align 8, !dbg !2056, !tbaa !28
  %839 = load ptr, ptr %161, align 8, !dbg !2057, !tbaa !28
  %840 = getelementptr inbounds [4 x [2 x <8 x float>]], ptr %172, i64 0, i64 0, !dbg !2058
  %841 = getelementptr inbounds [2 x <8 x float>], ptr %188, i64 0, i64 0, !dbg !2059
  call void @_ZL21iu4_bundle_single_accILi1EEvPKcS1_RA2_A2_KjRA4_S3_PA2_Dv8_fPKS8_RA4_KfRKDv8_i(ptr noundef %838, ptr noundef %839, ptr noundef nonnull align 4 dereferenceable(16) %164, ptr noundef nonnull align 4 dereferenceable(32) %163, ptr noundef %840, ptr noundef %841, ptr noundef nonnull align 4 dereferenceable(16) %189, ptr noundef nonnull align 32 dereferenceable(32) %187) #26, !dbg !2060
  call void @_Z13__syncthreadsv() #26, !dbg !2061
  store ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @LDS to ptr), i64 8192), ptr %162, align 8, !dbg !2062, !tbaa !28
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %94) #25, !dbg !2063
  store i32 0, ptr %227, align 4, !dbg !2064, !tbaa !8
  br label %842, !dbg !2063

842:                                              ; preds = %869, %833
  %843 = load i32, ptr %227, align 4, !dbg !2065, !tbaa !8
  %844 = icmp slt i32 %843, 2, !dbg !2066
  br i1 %844, label %846, label %845, !dbg !2067

845:                                              ; preds = %842
  store i32 50, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %94) #25, !dbg !2067
  br label %872

846:                                              ; preds = %842
  %847 = load i32, ptr %227, align 4, !dbg !2068, !tbaa !8
  %848 = sext i32 %847 to i64, !dbg !2069
  %849 = getelementptr inbounds [2 x <2 x i32>], ptr %190, i64 0, i64 %848, !dbg !2069
  %850 = load <2 x i32>, ptr %849, align 8, !dbg !2069, !tbaa !47
  %851 = load ptr, ptr %161, align 8, !dbg !2070, !tbaa !28
  %852 = load i32, ptr %227, align 4, !dbg !2071, !tbaa !8
  %853 = sext i32 %852 to i64, !dbg !2072
  %854 = getelementptr inbounds [2 x i32], ptr %175, i64 0, i64 %853, !dbg !2072
  %855 = load i32, ptr %854, align 4, !dbg !2072, !tbaa !8
  %856 = zext i32 %855 to i64, !dbg !2073
  %857 = getelementptr inbounds nuw i8, ptr %851, i64 %856, !dbg !2073
  store <2 x i32> %850, ptr %857, align 8, !dbg !2074, !tbaa !47
  %858 = load i32, ptr %227, align 4, !dbg !2075, !tbaa !8
  %859 = sext i32 %858 to i64, !dbg !2076
  %860 = getelementptr inbounds [2 x <2 x i32>], ptr %191, i64 0, i64 %859, !dbg !2076
  %861 = load <2 x i32>, ptr %860, align 8, !dbg !2076, !tbaa !47
  %862 = load ptr, ptr %162, align 8, !dbg !2077, !tbaa !28
  %863 = load i32, ptr %227, align 4, !dbg !2078, !tbaa !8
  %864 = sext i32 %863 to i64, !dbg !2079
  %865 = getelementptr inbounds [2 x i32], ptr %175, i64 0, i64 %864, !dbg !2079
  %866 = load i32, ptr %865, align 4, !dbg !2079, !tbaa !8
  %867 = zext i32 %866 to i64, !dbg !2080
  %868 = getelementptr inbounds nuw i8, ptr %862, i64 %867, !dbg !2080
  store <2 x i32> %861, ptr %868, align 8, !dbg !2081, !tbaa !47
  br label %869, !dbg !2082

869:                                              ; preds = %846
  %870 = load i32, ptr %227, align 4, !dbg !2083, !tbaa !8
  %871 = add nsw i32 %870, 1, !dbg !2083
  store i32 %871, ptr %227, align 4, !dbg !2083, !tbaa !8
  br label %842, !dbg !2067, !llvm.loop !2084

872:                                              ; preds = %845
  call void @_Z13__syncthreadsv() #26, !dbg !2085
  %873 = load i8, ptr %215, align 1, !dbg !2086, !tbaa !880, !range !1035, !noundef !20
  %874 = trunc i8 %873 to i1, !dbg !2086
  br i1 %874, label %1001, label %875, !dbg !2087

875:                                              ; preds = %872
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %95) #25, !dbg !2088
  %876 = load i32, ptr %214, align 4, !dbg !2089, !tbaa !8
  %877 = add nsw i32 %876, 1, !dbg !2090
  store i32 %877, ptr %228, align 4, !dbg !2091, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %96) #25, !dbg !2092
  %878 = load ptr, ptr %143, align 8, !dbg !2093, !tbaa !130
  %879 = load i32, ptr %228, align 4, !dbg !2094, !tbaa !8
  %880 = sext i32 %879 to i64, !dbg !2094
  %881 = load i32, ptr %147, align 4, !dbg !2095, !tbaa !8
  %882 = sext i32 %881 to i64, !dbg !2095
  %883 = mul nsw i64 %880, %882, !dbg !2096
  %884 = getelementptr inbounds %struct.block_i4_128, ptr %878, i64 %883, !dbg !2097
  %885 = getelementptr inbounds i8, ptr %884, i64 8, !dbg !2098
  store ptr %885, ptr %229, align 8, !dbg !2099, !tbaa !28
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %97) #25, !dbg !2100
  store i32 0, ptr %230, align 4, !dbg !2101, !tbaa !8
  br label %886, !dbg !2100

886:                                              ; preds = %902, %875
  %887 = load i32, ptr %230, align 4, !dbg !2102, !tbaa !8
  %888 = icmp slt i32 %887, 2, !dbg !2103
  br i1 %888, label %890, label %889, !dbg !2104

889:                                              ; preds = %886
  store i32 53, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %97) #25, !dbg !2104
  br label %905

890:                                              ; preds = %886
  %891 = load ptr, ptr %229, align 8, !dbg !2105, !tbaa !28
  %892 = load i32, ptr %230, align 4, !dbg !2106, !tbaa !8
  %893 = sext i32 %892 to i64, !dbg !2107
  %894 = getelementptr inbounds [2 x i32], ptr %173, i64 0, i64 %893, !dbg !2107
  %895 = load i32, ptr %894, align 4, !dbg !2107, !tbaa !8
  %896 = zext i32 %895 to i64, !dbg !2108
  %897 = getelementptr inbounds nuw i8, ptr %891, i64 %896, !dbg !2108
  %898 = load <2 x i32>, ptr %897, align 8, !dbg !2108, !tbaa !47
  %899 = load i32, ptr %230, align 4, !dbg !2109, !tbaa !8
  %900 = sext i32 %899 to i64, !dbg !2110
  %901 = getelementptr inbounds [2 x <2 x i32>], ptr %190, i64 0, i64 %900, !dbg !2110
  store <2 x i32> %898, ptr %901, align 8, !dbg !2111, !tbaa !47
  br label %902, !dbg !2112

902:                                              ; preds = %890
  %903 = load i32, ptr %230, align 4, !dbg !2113, !tbaa !8
  %904 = add nsw i32 %903, 1, !dbg !2113
  store i32 %904, ptr %230, align 4, !dbg !2113, !tbaa !8
  br label %886, !dbg !2104, !llvm.loop !2114

905:                                              ; preds = %889
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %98) #25, !dbg !2115
  %906 = load i32, ptr %212, align 4, !dbg !2116, !tbaa !8
  %907 = load i32, ptr %213, align 4, !dbg !2117, !tbaa !8
  %908 = add nsw i32 %906, %907, !dbg !2118
  store i32 %908, ptr %231, align 4, !dbg !2119, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %99) #25, !dbg !2120
  %909 = load i32, ptr %213, align 4, !dbg !2121, !tbaa !8
  %910 = xor i32 %909, 1, !dbg !2122
  store i32 %910, ptr %232, align 4, !dbg !2123, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %100) #25, !dbg !2124
  %911 = load ptr, ptr %142, align 8, !dbg !2125, !tbaa !28
  %912 = load i32, ptr %231, align 4, !dbg !2126, !tbaa !8
  %913 = sext i32 %912 to i64, !dbg !2126
  %914 = mul nsw i64 %913, 136, !dbg !2127
  %915 = getelementptr inbounds i8, ptr %911, i64 %914, !dbg !2128
  %916 = getelementptr inbounds i8, ptr %915, i64 8, !dbg !2129
  %917 = load i32, ptr %232, align 4, !dbg !2130, !tbaa !8
  %918 = sext i32 %917 to i64, !dbg !2130
  %919 = mul nsw i64 %918, 64, !dbg !2131
  %920 = getelementptr inbounds i8, ptr %916, i64 %919, !dbg !2132
  store ptr %920, ptr %233, align 8, !dbg !2133, !tbaa !28
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %101) #25, !dbg !2134
  store i32 0, ptr %234, align 4, !dbg !2135, !tbaa !8
  br label %921, !dbg !2134

921:                                              ; preds = %937, %905
  %922 = load i32, ptr %234, align 4, !dbg !2136, !tbaa !8
  %923 = icmp slt i32 %922, 2, !dbg !2137
  br i1 %923, label %925, label %924, !dbg !2138

924:                                              ; preds = %921
  store i32 56, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %101) #25, !dbg !2138
  br label %940

925:                                              ; preds = %921
  %926 = load ptr, ptr %233, align 8, !dbg !2139, !tbaa !28
  %927 = load i32, ptr %234, align 4, !dbg !2140, !tbaa !8
  %928 = sext i32 %927 to i64, !dbg !2141
  %929 = getelementptr inbounds [2 x i32], ptr %174, i64 0, i64 %928, !dbg !2141
  %930 = load i32, ptr %929, align 4, !dbg !2141, !tbaa !8
  %931 = zext i32 %930 to i64, !dbg !2142
  %932 = getelementptr inbounds nuw i8, ptr %926, i64 %931, !dbg !2142
  %933 = load <2 x i32>, ptr %932, align 8, !dbg !2142, !tbaa !47
  %934 = load i32, ptr %234, align 4, !dbg !2143, !tbaa !8
  %935 = sext i32 %934 to i64, !dbg !2144
  %936 = getelementptr inbounds [2 x <2 x i32>], ptr %191, i64 0, i64 %935, !dbg !2144
  store <2 x i32> %933, ptr %936, align 8, !dbg !2145, !tbaa !47
  br label %937, !dbg !2146

937:                                              ; preds = %925
  %938 = load i32, ptr %234, align 4, !dbg !2147, !tbaa !8
  %939 = add nsw i32 %938, 1, !dbg !2147
  store i32 %939, ptr %234, align 4, !dbg !2147, !tbaa !8
  br label %921, !dbg !2138, !llvm.loop !2148

940:                                              ; preds = %924
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %102) #25, !dbg !2149
  %941 = load i32, ptr %152, align 4, !dbg !2150, !tbaa !8
  %942 = load i32, ptr %148, align 4, !dbg !2151, !tbaa !8
  %943 = ashr i32 %942, 1, !dbg !2152
  %944 = add nsw i32 %941, %943, !dbg !2153
  store i32 %944, ptr %235, align 4, !dbg !2154, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %103) #25, !dbg !2155
  %945 = load i32, ptr %235, align 4, !dbg !2156, !tbaa !8
  %946 = load i32, ptr %147, align 4, !dbg !2157, !tbaa !8
  %947 = icmp slt i32 %945, %946, !dbg !2158
  br i1 %947, label %948, label %950, !dbg !2159

948:                                              ; preds = %940
  %949 = load i32, ptr %235, align 4, !dbg !2160, !tbaa !8
  br label %953, !dbg !2159

950:                                              ; preds = %940
  %951 = load i32, ptr %147, align 4, !dbg !2161, !tbaa !8
  %952 = sub nsw i32 %951, 1, !dbg !2162
  br label %953, !dbg !2159

953:                                              ; preds = %950, %948
  %954 = phi i32 [ %949, %948 ], [ %952, %950 ], !dbg !2159
  store i32 %954, ptr %236, align 4, !dbg !2163, !tbaa !8
  %955 = load ptr, ptr %143, align 8, !dbg !2164, !tbaa !130
  %956 = load i32, ptr %228, align 4, !dbg !2165, !tbaa !8
  %957 = sext i32 %956 to i64, !dbg !2165
  %958 = load i32, ptr %147, align 4, !dbg !2166, !tbaa !8
  %959 = sext i32 %958 to i64, !dbg !2166
  %960 = mul nsw i64 %957, %959, !dbg !2167
  %961 = getelementptr inbounds %struct.block_i4_128, ptr %955, i64 %960, !dbg !2168
  %962 = load i32, ptr %236, align 4, !dbg !2169, !tbaa !8
  %963 = sext i32 %962 to i64, !dbg !2168
  %964 = getelementptr inbounds %struct.block_i4_128, ptr %961, i64 %963, !dbg !2168
  %965 = load i32, ptr %148, align 4, !dbg !2170, !tbaa !8
  %966 = and i32 %965, 1, !dbg !2171
  %967 = sext i32 %966 to i64, !dbg !2168
  %968 = getelementptr inbounds i32, ptr %964, i64 %967, !dbg !2168
  %969 = load i32, ptr %968, align 4, !dbg !2168, !tbaa !8
  store i32 %969, ptr %192, align 4, !dbg !2172, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %104) #25, !dbg !2173
  %970 = load i32, ptr %151, align 4, !dbg !2174, !tbaa !8
  %971 = load i32, ptr %148, align 4, !dbg !2175, !tbaa !8
  %972 = ashr i32 %971, 1, !dbg !2176
  %973 = add nsw i32 %970, %972, !dbg !2177
  store i32 %973, ptr %237, align 4, !dbg !2178, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %105) #25, !dbg !2179
  %974 = load i32, ptr %237, align 4, !dbg !2180, !tbaa !8
  %975 = load i32, ptr %145, align 4, !dbg !2181, !tbaa !8
  %976 = icmp slt i32 %974, %975, !dbg !2182
  br i1 %976, label %977, label %979, !dbg !2183

977:                                              ; preds = %953
  %978 = load i32, ptr %237, align 4, !dbg !2184, !tbaa !8
  br label %982, !dbg !2183

979:                                              ; preds = %953
  %980 = load i32, ptr %145, align 4, !dbg !2185, !tbaa !8
  %981 = sub nsw i32 %980, 1, !dbg !2186
  br label %982, !dbg !2183

982:                                              ; preds = %979, %977
  %983 = phi i32 [ %978, %977 ], [ %981, %979 ], !dbg !2183
  store i32 %983, ptr %238, align 4, !dbg !2187, !tbaa !8
  %984 = load ptr, ptr %142, align 8, !dbg !2188, !tbaa !28
  %985 = load i32, ptr %238, align 4, !dbg !2189, !tbaa !8
  %986 = sext i32 %985 to i64, !dbg !2189
  %987 = load i32, ptr %160, align 4, !dbg !2190, !tbaa !8
  %988 = sext i32 %987 to i64, !dbg !2190
  %989 = mul nsw i64 %986, %988, !dbg !2191
  %990 = mul nsw i64 %989, 136, !dbg !2192
  %991 = getelementptr inbounds i8, ptr %984, i64 %990, !dbg !2193
  %992 = load i32, ptr %231, align 4, !dbg !2194, !tbaa !8
  %993 = mul nsw i32 %992, 136, !dbg !2195
  %994 = sext i32 %993 to i64, !dbg !2193
  %995 = getelementptr inbounds i8, ptr %991, i64 %994, !dbg !2193
  %996 = load i32, ptr %232, align 4, !dbg !2196, !tbaa !8
  %997 = mul nsw i32 %996, 4, !dbg !2197
  %998 = sext i32 %997 to i64, !dbg !2193
  %999 = getelementptr inbounds i8, ptr %995, i64 %998, !dbg !2193
  %1000 = load i32, ptr %999, align 4, !dbg !2193, !tbaa !8
  store i32 %1000, ptr %193, align 4, !dbg !2198, !tbaa !8
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %105) #25, !dbg !2199
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %104) #25, !dbg !2199
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %103) #25, !dbg !2199
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %102) #25, !dbg !2199
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %100) #25, !dbg !2199
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %99) #25, !dbg !2199
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %98) #25, !dbg !2199
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %96) #25, !dbg !2199
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %95) #25, !dbg !2199
  br label %1001, !dbg !2199

1001:                                             ; preds = %982, %872
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %91) #25, !dbg !2200
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %90) #25, !dbg !2200
  %1002 = load ptr, ptr %162, align 8, !dbg !2201, !tbaa !28
  %1003 = load ptr, ptr %161, align 8, !dbg !2202, !tbaa !28
  %1004 = getelementptr inbounds [4 x [2 x <8 x float>]], ptr %172, i64 0, i64 0, !dbg !2203
  %1005 = getelementptr inbounds [2 x <8 x float>], ptr %188, i64 0, i64 0, !dbg !2204
  call void @_ZL21iu4_bundle_single_accILi0EEvPKcS1_RA2_A2_KjRA4_S3_PA2_Dv8_fPKS8_RA4_KfRKDv8_i(ptr noundef %1002, ptr noundef %1003, ptr noundef nonnull align 4 dereferenceable(16) %164, ptr noundef nonnull align 4 dereferenceable(32) %163, ptr noundef %1004, ptr noundef %1005, ptr noundef nonnull align 4 dereferenceable(16) %189, ptr noundef nonnull align 32 dereferenceable(32) %187) #26, !dbg !2205
  %1006 = load ptr, ptr %162, align 8, !dbg !2206, !tbaa !28
  %1007 = load ptr, ptr %161, align 8, !dbg !2207, !tbaa !28
  %1008 = getelementptr inbounds [4 x [2 x <8 x float>]], ptr %172, i64 0, i64 0, !dbg !2208
  %1009 = getelementptr inbounds [2 x <8 x float>], ptr %188, i64 0, i64 0, !dbg !2209
  call void @_ZL21iu4_bundle_single_accILi1EEvPKcS1_RA2_A2_KjRA4_S3_PA2_Dv8_fPKS8_RA4_KfRKDv8_i(ptr noundef %1006, ptr noundef %1007, ptr noundef nonnull align 4 dereferenceable(16) %164, ptr noundef nonnull align 4 dereferenceable(32) %163, ptr noundef %1008, ptr noundef %1009, ptr noundef nonnull align 4 dereferenceable(16) %189, ptr noundef nonnull align 32 dereferenceable(32) %187) #26, !dbg !2210
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %106) #25, !dbg !2211
  store i32 0, ptr %239, align 4, !dbg !2212, !tbaa !8
  br label %1010, !dbg !2211

1010:                                             ; preds = %1107, %1001
  %1011 = load i32, ptr %239, align 4, !dbg !2213, !tbaa !8
  %1012 = icmp slt i32 %1011, 2, !dbg !2214
  br i1 %1012, label %1014, label %1013, !dbg !2215

1013:                                             ; preds = %1010
  store i32 59, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %106) #25, !dbg !2215
  br label %1110

1014:                                             ; preds = %1010
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %107) #25, !dbg !2216
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %108) #25, !dbg !2217
  store i32 0, ptr %241, align 4, !dbg !2218, !tbaa !8
  br label %1015, !dbg !2217

1015:                                             ; preds = %1039, %1014
  %1016 = load i32, ptr %241, align 4, !dbg !2219, !tbaa !8
  %1017 = icmp slt i32 %1016, 8, !dbg !2220
  br i1 %1017, label %1019, label %1018, !dbg !2221

1018:                                             ; preds = %1015
  store i32 62, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %108) #25, !dbg !2221
  br label %1042

1019:                                             ; preds = %1015
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %109) #25, !dbg !2222
  %1020 = load i32, ptr %158, align 4, !dbg !2223, !tbaa !8
  %1021 = load i32, ptr %239, align 4, !dbg !2224, !tbaa !8
  %1022 = mul nsw i32 %1021, 16, !dbg !2225
  %1023 = add nsw i32 %1020, %1022, !dbg !2226
  %1024 = load i32, ptr %155, align 4, !dbg !2227, !tbaa !8
  %1025 = mul nsw i32 8, %1024, !dbg !2228
  %1026 = add nsw i32 %1023, %1025, !dbg !2229
  %1027 = load i32, ptr %241, align 4, !dbg !2230, !tbaa !8
  %1028 = add nsw i32 %1026, %1027, !dbg !2231
  %1029 = mul i32 %1028, 8, !dbg !2232
  store i32 %1029, ptr %242, align 4, !dbg !2233, !tbaa !8
  %1030 = load ptr, ptr %217, align 8, !dbg !2234, !tbaa !28
  %1031 = load i32, ptr %242, align 4, !dbg !2235, !tbaa !8
  %1032 = zext i32 %1031 to i64, !dbg !2236
  %1033 = getelementptr inbounds nuw i8, ptr %1030, i64 %1032, !dbg !2236
  %1034 = getelementptr inbounds nuw i8, ptr %1033, i64 4, !dbg !2236
  %1035 = load float, ptr %1034, align 4, !dbg !2236, !tbaa !177
  %1036 = load i32, ptr %241, align 4, !dbg !2237, !tbaa !8
  %1037 = sext i32 %1036 to i64, !dbg !2238
  %1038 = getelementptr inbounds [8 x float], ptr %240, i64 0, i64 %1037, !dbg !2238
  store float %1035, ptr %1038, align 4, !dbg !2239, !tbaa !177
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %109) #25, !dbg !2240
  br label %1039, !dbg !2240

1039:                                             ; preds = %1019
  %1040 = load i32, ptr %241, align 4, !dbg !2241, !tbaa !8
  %1041 = add nsw i32 %1040, 1, !dbg !2241
  store i32 %1041, ptr %241, align 4, !dbg !2241, !tbaa !8
  br label %1015, !dbg !2221, !llvm.loop !2242

1042:                                             ; preds = %1018
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %110) #25, !dbg !2243
  store i32 0, ptr %243, align 4, !dbg !2244, !tbaa !8
  br label %1043, !dbg !2243

1043:                                             ; preds = %1103, %1042
  %1044 = load i32, ptr %243, align 4, !dbg !2245, !tbaa !8
  %1045 = icmp slt i32 %1044, 4, !dbg !2246
  br i1 %1045, label %1047, label %1046, !dbg !2247

1046:                                             ; preds = %1043
  store i32 65, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %110) #25, !dbg !2247
  br label %1106

1047:                                             ; preds = %1043
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %111) #25, !dbg !2248
  %1048 = load i32, ptr %159, align 4, !dbg !2249, !tbaa !8
  %1049 = load i32, ptr %243, align 4, !dbg !2250, !tbaa !8
  %1050 = mul nsw i32 %1049, 16, !dbg !2251
  %1051 = add nsw i32 %1048, %1050, !dbg !2252
  %1052 = load i32, ptr %154, align 4, !dbg !2253, !tbaa !8
  %1053 = add nsw i32 %1051, %1052, !dbg !2254
  %1054 = mul i32 %1053, 8, !dbg !2255
  store i32 %1054, ptr %244, align 4, !dbg !2256, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %112) #25, !dbg !2257
  %1055 = load ptr, ptr %216, align 8, !dbg !2258, !tbaa !28
  %1056 = load i32, ptr %244, align 4, !dbg !2259, !tbaa !8
  %1057 = zext i32 %1056 to i64, !dbg !2260
  %1058 = getelementptr inbounds nuw i8, ptr %1055, i64 %1057, !dbg !2260
  %1059 = load float, ptr %1058, align 4, !dbg !2260, !tbaa !177
  store float %1059, ptr %245, align 4, !dbg !2261, !tbaa !177
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %113) #25, !dbg !2262
  %1060 = load ptr, ptr %216, align 8, !dbg !2263, !tbaa !28
  %1061 = load i32, ptr %244, align 4, !dbg !2264, !tbaa !8
  %1062 = zext i32 %1061 to i64, !dbg !2265
  %1063 = getelementptr inbounds nuw i8, ptr %1060, i64 %1062, !dbg !2265
  %1064 = getelementptr inbounds nuw i8, ptr %1063, i64 4, !dbg !2265
  %1065 = load i32, ptr %1064, align 4, !dbg !2265, !tbaa !8
  %1066 = sitofp i32 %1065 to float, !dbg !2265
  store float %1066, ptr %246, align 4, !dbg !2266, !tbaa !177
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %114) #25, !dbg !2267
  store i32 0, ptr %247, align 4, !dbg !2268, !tbaa !8
  br label %1067, !dbg !2267

1067:                                             ; preds = %1099, %1047
  %1068 = load i32, ptr %247, align 4, !dbg !2269, !tbaa !8
  %1069 = icmp slt i32 %1068, 8, !dbg !2270
  br i1 %1069, label %1071, label %1070, !dbg !2271

1070:                                             ; preds = %1067
  store i32 68, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %114) #25, !dbg !2271
  br label %1102

1071:                                             ; preds = %1067
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %115) #25, !dbg !2272
  %1072 = load i32, ptr %247, align 4, !dbg !2273, !tbaa !8
  %1073 = sext i32 %1072 to i64, !dbg !2274
  %1074 = getelementptr inbounds [8 x float], ptr %240, i64 0, i64 %1073, !dbg !2274
  %1075 = load float, ptr %1074, align 4, !dbg !2274, !tbaa !177
  %1076 = load float, ptr %245, align 4, !dbg !2275, !tbaa !177
  %1077 = call contract noundef float @_ZL9__fmul_rnff(float noundef %1075, float noundef %1076) #26, !dbg !2276
  store float %1077, ptr %248, align 4, !dbg !2277, !tbaa !177
  %1078 = load float, ptr %248, align 4, !dbg !2278, !tbaa !177
  %1079 = load float, ptr %246, align 4, !dbg !2279, !tbaa !177
  %1080 = load i32, ptr %243, align 4, !dbg !2280, !tbaa !8
  %1081 = sext i32 %1080 to i64, !dbg !2281
  %1082 = getelementptr inbounds [4 x [2 x <8 x float>]], ptr %172, i64 0, i64 %1081, !dbg !2281
  %1083 = load i32, ptr %239, align 4, !dbg !2282, !tbaa !8
  %1084 = sext i32 %1083 to i64, !dbg !2281
  %1085 = getelementptr inbounds [2 x <8 x float>], ptr %1082, i64 0, i64 %1084, !dbg !2281
  %1086 = load <8 x float>, ptr %1085, align 32, !dbg !2281, !tbaa !47
  %1087 = load i32, ptr %247, align 4, !dbg !2283, !tbaa !8
  %1088 = extractelement <8 x float> %1086, i32 %1087, !dbg !2281
  %1089 = call contract noundef float @_ZL9__fmaf_rnfff(float noundef %1078, float noundef %1079, float noundef %1088) #26, !dbg !2284
  %1090 = load i32, ptr %243, align 4, !dbg !2285, !tbaa !8
  %1091 = sext i32 %1090 to i64, !dbg !2286
  %1092 = getelementptr inbounds [4 x [2 x <8 x float>]], ptr %172, i64 0, i64 %1091, !dbg !2286
  %1093 = load i32, ptr %239, align 4, !dbg !2287, !tbaa !8
  %1094 = sext i32 %1093 to i64, !dbg !2286
  %1095 = getelementptr inbounds [2 x <8 x float>], ptr %1092, i64 0, i64 %1094, !dbg !2286
  %1096 = load i32, ptr %247, align 4, !dbg !2288, !tbaa !8
  %1097 = load <8 x float>, ptr %1095, align 32, !dbg !2289
  %1098 = insertelement <8 x float> %1097, float %1089, i32 %1096, !dbg !2289
  store <8 x float> %1098, ptr %1095, align 32, !dbg !2289
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %115) #25, !dbg !2290
  br label %1099, !dbg !2290

1099:                                             ; preds = %1071
  %1100 = load i32, ptr %247, align 4, !dbg !2291, !tbaa !8
  %1101 = add nsw i32 %1100, 1, !dbg !2291
  store i32 %1101, ptr %247, align 4, !dbg !2291, !tbaa !8
  br label %1067, !dbg !2271, !llvm.loop !2292

1102:                                             ; preds = %1070
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %113) #25, !dbg !2293
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %112) #25, !dbg !2293
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %111) #25, !dbg !2293
  br label %1103, !dbg !2293

1103:                                             ; preds = %1102
  %1104 = load i32, ptr %243, align 4, !dbg !2294, !tbaa !8
  %1105 = add nsw i32 %1104, 1, !dbg !2294
  store i32 %1105, ptr %243, align 4, !dbg !2294, !tbaa !8
  br label %1043, !dbg !2247, !llvm.loop !2295

1106:                                             ; preds = %1046
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %107) #25, !dbg !2296
  br label %1107, !dbg !2296

1107:                                             ; preds = %1106
  %1108 = load i32, ptr %239, align 4, !dbg !2297, !tbaa !8
  %1109 = add nsw i32 %1108, 1, !dbg !2297
  store i32 %1109, ptr %239, align 4, !dbg !2297, !tbaa !8
  br label %1010, !dbg !2215, !llvm.loop !2298

1110:                                             ; preds = %1013
  call void @_Z13__syncthreadsv() #26, !dbg !2299
  %1111 = load i8, ptr %215, align 1, !dbg !2300, !tbaa !880, !range !1035, !noundef !20
  %1112 = trunc i8 %1111 to i1, !dbg !2300
  br i1 %1112, label %1241, label %1113, !dbg !2301

1113:                                             ; preds = %1110
  store ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @LDS to ptr), i64 4096), ptr %162, align 8, !dbg !2302, !tbaa !28
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %116) #25, !dbg !2303
  store i32 0, ptr %249, align 4, !dbg !2304, !tbaa !8
  br label %1114, !dbg !2303

1114:                                             ; preds = %1150, %1113
  %1115 = load i32, ptr %249, align 4, !dbg !2305, !tbaa !8
  %1116 = icmp slt i32 %1115, 2, !dbg !2306
  br i1 %1116, label %1118, label %1117, !dbg !2307

1117:                                             ; preds = %1114
  store i32 71, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %116) #25, !dbg !2307
  br label %1153

1118:                                             ; preds = %1114
  %1119 = load i32, ptr %249, align 4, !dbg !2308, !tbaa !8
  %1120 = sext i32 %1119 to i64, !dbg !2309
  %1121 = getelementptr inbounds [2 x i32], ptr %176, i64 0, i64 %1120, !dbg !2309
  %1122 = load i32, ptr %1121, align 4, !dbg !2309, !tbaa !8
  %1123 = icmp ne i32 %1122, 0, !dbg !2309
  br i1 %1123, label %1124, label %1129, !dbg !2309

1124:                                             ; preds = %1118
  %1125 = load i32, ptr %249, align 4, !dbg !2310, !tbaa !8
  %1126 = sext i32 %1125 to i64, !dbg !2311
  %1127 = getelementptr inbounds [2 x <2 x i32>], ptr %190, i64 0, i64 %1126, !dbg !2311
  %1128 = load <2 x i32>, ptr %1127, align 8, !dbg !2311, !tbaa !47
  br label %1130, !dbg !2309

1129:                                             ; preds = %1118
  br label %1130, !dbg !2309

1130:                                             ; preds = %1129, %1124
  %1131 = phi <2 x i32> [ %1128, %1124 ], [ zeroinitializer, %1129 ], !dbg !2309
  %1132 = load ptr, ptr %161, align 8, !dbg !2312, !tbaa !28
  %1133 = load i32, ptr %249, align 4, !dbg !2313, !tbaa !8
  %1134 = sext i32 %1133 to i64, !dbg !2314
  %1135 = getelementptr inbounds [2 x i32], ptr %175, i64 0, i64 %1134, !dbg !2314
  %1136 = load i32, ptr %1135, align 4, !dbg !2314, !tbaa !8
  %1137 = zext i32 %1136 to i64, !dbg !2315
  %1138 = getelementptr inbounds nuw i8, ptr %1132, i64 %1137, !dbg !2315
  store <2 x i32> %1131, ptr %1138, align 8, !dbg !2316, !tbaa !47
  %1139 = load i32, ptr %249, align 4, !dbg !2317, !tbaa !8
  %1140 = sext i32 %1139 to i64, !dbg !2318
  %1141 = getelementptr inbounds [2 x <2 x i32>], ptr %191, i64 0, i64 %1140, !dbg !2318
  %1142 = load <2 x i32>, ptr %1141, align 8, !dbg !2318, !tbaa !47
  %1143 = load ptr, ptr %162, align 8, !dbg !2319, !tbaa !28
  %1144 = load i32, ptr %249, align 4, !dbg !2320, !tbaa !8
  %1145 = sext i32 %1144 to i64, !dbg !2321
  %1146 = getelementptr inbounds [2 x i32], ptr %175, i64 0, i64 %1145, !dbg !2321
  %1147 = load i32, ptr %1146, align 4, !dbg !2321, !tbaa !8
  %1148 = zext i32 %1147 to i64, !dbg !2322
  %1149 = getelementptr inbounds nuw i8, ptr %1143, i64 %1148, !dbg !2322
  store <2 x i32> %1142, ptr %1149, align 8, !dbg !2323, !tbaa !47
  br label %1150, !dbg !2324

1150:                                             ; preds = %1130
  %1151 = load i32, ptr %249, align 4, !dbg !2325, !tbaa !8
  %1152 = add nsw i32 %1151, 1, !dbg !2325
  store i32 %1152, ptr %249, align 4, !dbg !2325, !tbaa !8
  br label %1114, !dbg !2307, !llvm.loop !2326

1153:                                             ; preds = %1117
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %117) #25, !dbg !2327
  %1154 = load i32, ptr %213, align 4, !dbg !2328, !tbaa !8
  %1155 = xor i32 %1154, 1, !dbg !2329
  store i32 %1155, ptr %250, align 4, !dbg !2330, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %118) #25, !dbg !2331
  %1156 = load i32, ptr %250, align 4, !dbg !2332, !tbaa !8
  %1157 = icmp eq i32 %1156, 0, !dbg !2333
  %1158 = zext i1 %1157 to i64, !dbg !2334
  %1159 = select i1 %1157, i32 12288, i32 13312, !dbg !2334
  %1160 = sext i32 %1159 to i64, !dbg !2335
  %1161 = getelementptr inbounds i8, ptr addrspacecast (ptr addrspace(3) @LDS to ptr), i64 %1160, !dbg !2335
  store ptr %1161, ptr %251, align 8, !dbg !2336, !tbaa !28
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %119) #25, !dbg !2337
  %1162 = load i32, ptr %152, align 4, !dbg !2338, !tbaa !8
  %1163 = load i32, ptr %148, align 4, !dbg !2339, !tbaa !8
  %1164 = ashr i32 %1163, 1, !dbg !2340
  %1165 = add nsw i32 %1162, %1164, !dbg !2341
  store i32 %1165, ptr %252, align 4, !dbg !2342, !tbaa !8
  %1166 = load i32, ptr %252, align 4, !dbg !2343, !tbaa !8
  %1167 = load i32, ptr %147, align 4, !dbg !2344, !tbaa !8
  %1168 = icmp slt i32 %1166, %1167, !dbg !2345
  br i1 %1168, label %1169, label %1171, !dbg !2346

1169:                                             ; preds = %1153
  %1170 = load i32, ptr %192, align 4, !dbg !2347, !tbaa !8
  br label %1172, !dbg !2346

1171:                                             ; preds = %1153
  br label %1172, !dbg !2346

1172:                                             ; preds = %1171, %1169
  %1173 = phi i32 [ %1170, %1169 ], [ 0, %1171 ], !dbg !2346
  %1174 = load ptr, ptr %251, align 8, !dbg !2348, !tbaa !28
  %1175 = load i32, ptr %148, align 4, !dbg !2349, !tbaa !8
  %1176 = ashr i32 %1175, 1, !dbg !2350
  %1177 = mul nsw i32 %1176, 8, !dbg !2351
  %1178 = sext i32 %1177 to i64, !dbg !2352
  %1179 = getelementptr inbounds i8, ptr %1174, i64 %1178, !dbg !2352
  %1180 = load i32, ptr %148, align 4, !dbg !2353, !tbaa !8
  %1181 = and i32 %1180, 1, !dbg !2354
  %1182 = sext i32 %1181 to i64, !dbg !2352
  %1183 = getelementptr inbounds i32, ptr %1179, i64 %1182, !dbg !2352
  store i32 %1173, ptr %1183, align 4, !dbg !2355, !tbaa !8
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %119) #25, !dbg !2356
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %120) #25, !dbg !2357
  %1184 = load i32, ptr %250, align 4, !dbg !2358, !tbaa !8
  %1185 = icmp eq i32 %1184, 0, !dbg !2359
  %1186 = zext i1 %1185 to i64, !dbg !2360
  %1187 = select i1 %1185, i32 14336, i32 15360, !dbg !2360
  %1188 = sext i32 %1187 to i64, !dbg !2361
  %1189 = getelementptr inbounds i8, ptr addrspacecast (ptr addrspace(3) @LDS to ptr), i64 %1188, !dbg !2361
  store ptr %1189, ptr %253, align 8, !dbg !2362, !tbaa !28
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %121) #25, !dbg !2363
  %1190 = load i32, ptr %151, align 4, !dbg !2364, !tbaa !8
  %1191 = load i32, ptr %148, align 4, !dbg !2365, !tbaa !8
  %1192 = ashr i32 %1191, 1, !dbg !2366
  %1193 = add nsw i32 %1190, %1192, !dbg !2367
  store i32 %1193, ptr %254, align 4, !dbg !2368, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %122) #25, !dbg !2369
  %1194 = load i32, ptr %148, align 4, !dbg !2370, !tbaa !8
  %1195 = and i32 %1194, 1, !dbg !2371
  store i32 %1195, ptr %255, align 4, !dbg !2372, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %123) #25, !dbg !2373
  %1196 = load i32, ptr %255, align 4, !dbg !2374, !tbaa !8
  %1197 = icmp eq i32 %1196, 0, !dbg !2375
  br i1 %1197, label %1198, label %1210, !dbg !2376

1198:                                             ; preds = %1172
  %1199 = addrspacecast ptr %257 to ptr addrspace(5), !dbg !2377
  %1200 = load i32, ptr %193, align 4, !dbg !2378, !tbaa !8
  %1201 = and i32 %1200, 65535, !dbg !2379
  %1202 = trunc i32 %1201 to i16, !dbg !2380
  %1203 = call i16 @_Z16__ushort_as_halft(i16 noundef zeroext %1202) #26, !dbg !2377
  %1204 = getelementptr inbounds nuw %struct.__half, ptr addrspace(5) %1199, i32 0, i32 0, !dbg !2377
  %1205 = getelementptr inbounds nuw %union.anon.0, ptr addrspace(5) %1204, i32 0, i32 0, !dbg !2377
  store i16 %1203, ptr addrspace(5) %1205, align 2, !dbg !2377
  %1206 = getelementptr inbounds nuw %struct.__half, ptr %257, i32 0, i32 0, !dbg !2381
  %1207 = getelementptr inbounds nuw %union.anon.0, ptr %1206, i32 0, i32 0, !dbg !2381
  %1208 = load i16, ptr %1207, align 2, !dbg !2381
  %1209 = call contract noundef float @_Z12__half2float6__half(i16 %1208) #26, !dbg !2381
  br label %1222, !dbg !2376

1210:                                             ; preds = %1172
  %1211 = addrspacecast ptr %258 to ptr addrspace(5), !dbg !2382
  %1212 = load i32, ptr %193, align 4, !dbg !2383, !tbaa !8
  %1213 = lshr i32 %1212, 16, !dbg !2384
  %1214 = trunc i32 %1213 to i16, !dbg !2385
  %1215 = call i16 @_Z16__ushort_as_halft(i16 noundef zeroext %1214) #26, !dbg !2382
  %1216 = getelementptr inbounds nuw %struct.__half, ptr addrspace(5) %1211, i32 0, i32 0, !dbg !2382
  %1217 = getelementptr inbounds nuw %union.anon.0, ptr addrspace(5) %1216, i32 0, i32 0, !dbg !2382
  store i16 %1215, ptr addrspace(5) %1217, align 2, !dbg !2382
  %1218 = getelementptr inbounds nuw %struct.__half, ptr %258, i32 0, i32 0, !dbg !2386
  %1219 = getelementptr inbounds nuw %union.anon.0, ptr %1218, i32 0, i32 0, !dbg !2386
  %1220 = load i16, ptr %1219, align 2, !dbg !2386
  %1221 = call contract noundef float @_Z12__half2float6__half(i16 %1220) #26, !dbg !2386
  br label %1222, !dbg !2376

1222:                                             ; preds = %1210, %1198
  %1223 = phi contract float [ %1209, %1198 ], [ %1221, %1210 ], !dbg !2376
  store float %1223, ptr %256, align 4, !dbg !2387, !tbaa !177
  %1224 = load i32, ptr %254, align 4, !dbg !2388, !tbaa !8
  %1225 = load i32, ptr %145, align 4, !dbg !2389, !tbaa !8
  %1226 = icmp slt i32 %1224, %1225, !dbg !2390
  br i1 %1226, label %1227, label %1229, !dbg !2391

1227:                                             ; preds = %1222
  %1228 = load float, ptr %256, align 4, !dbg !2392, !tbaa !177
  br label %1230, !dbg !2391

1229:                                             ; preds = %1222
  br label %1230, !dbg !2391

1230:                                             ; preds = %1229, %1227
  %1231 = phi contract float [ %1228, %1227 ], [ 0.000000e+00, %1229 ], !dbg !2391
  %1232 = load ptr, ptr %253, align 8, !dbg !2393, !tbaa !28
  %1233 = load i32, ptr %148, align 4, !dbg !2394, !tbaa !8
  %1234 = ashr i32 %1233, 1, !dbg !2395
  %1235 = mul nsw i32 %1234, 8, !dbg !2396
  %1236 = sext i32 %1235 to i64, !dbg !2397
  %1237 = getelementptr inbounds i8, ptr %1232, i64 %1236, !dbg !2397
  %1238 = load i32, ptr %255, align 4, !dbg !2398, !tbaa !8
  %1239 = sext i32 %1238 to i64, !dbg !2397
  %1240 = getelementptr inbounds float, ptr %1237, i64 %1239, !dbg !2397
  store float %1231, ptr %1240, align 4, !dbg !2399, !tbaa !177
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %123) #25, !dbg !2400
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %122) #25, !dbg !2400
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %121) #25, !dbg !2400
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %120) #25, !dbg !2401
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %118) #25, !dbg !2401
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %117) #25, !dbg !2401
  br label %1241, !dbg !2401

1241:                                             ; preds = %1230, %1110
  call void @_Z13__syncthreadsv() #26, !dbg !2402
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %84) #25, !dbg !2403
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %83) #25, !dbg !2403
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %82) #25, !dbg !2403
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %81) #25, !dbg !2403
  br label %1242, !dbg !2403

1242:                                             ; preds = %1241
  %1243 = load i32, ptr %213, align 4, !dbg !2404, !tbaa !8
  %1244 = add nsw i32 %1243, 1, !dbg !2404
  store i32 %1244, ptr %213, align 4, !dbg !2404, !tbaa !8
  br label %679, !dbg !1917, !llvm.loop !2405

1245:                                             ; preds = %682
  br label %1246, !dbg !2406

1246:                                             ; preds = %1245
  %1247 = load i32, ptr %212, align 4, !dbg !2407, !tbaa !8
  %1248 = add nsw i32 %1247, 1, !dbg !2407
  store i32 %1248, ptr %212, align 4, !dbg !2407, !tbaa !8
  br label %673, !dbg !1912, !llvm.loop !2408

1249:                                             ; preds = %677
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %126) #25, !dbg !2409
  store ptr addrspacecast (ptr addrspace(3) @LDS to ptr), ptr %259, align 8, !dbg !2410, !tbaa !128
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %127) #25, !dbg !2411
  store i32 0, ptr %260, align 4, !dbg !2412, !tbaa !8
  br label %1250, !dbg !2411

1250:                                             ; preds = %1369, %1249
  %1251 = load i32, ptr %260, align 4, !dbg !2413, !tbaa !8
  %1252 = icmp slt i32 %1251, 2, !dbg !2414
  br i1 %1252, label %1254, label %1253, !dbg !2415

1253:                                             ; preds = %1250
  store i32 74, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %127) #25, !dbg !2415
  br label %1372

1254:                                             ; preds = %1250
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %128) #25, !dbg !2416
  store i32 0, ptr %261, align 4, !dbg !2417, !tbaa !8
  br label %1255, !dbg !2416

1255:                                             ; preds = %1365, %1254
  %1256 = load i32, ptr %261, align 4, !dbg !2418, !tbaa !8
  %1257 = icmp slt i32 %1256, 4, !dbg !2419
  br i1 %1257, label %1259, label %1258, !dbg !2420

1258:                                             ; preds = %1255
  store i32 77, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %128) #25, !dbg !2420
  br label %1368

1259:                                             ; preds = %1255
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %129) #25, !dbg !2421
  store i32 0, ptr %262, align 4, !dbg !2422, !tbaa !8
  br label %1260, !dbg !2421

1260:                                             ; preds = %1287, %1259
  %1261 = load i32, ptr %262, align 4, !dbg !2423, !tbaa !8
  %1262 = icmp slt i32 %1261, 8, !dbg !2424
  br i1 %1262, label %1264, label %1263, !dbg !2425

1263:                                             ; preds = %1260
  store i32 80, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %129) #25, !dbg !2425
  br label %1290

1264:                                             ; preds = %1260
  %1265 = load i32, ptr %261, align 4, !dbg !2426, !tbaa !8
  %1266 = sext i32 %1265 to i64, !dbg !2427
  %1267 = getelementptr inbounds [4 x [2 x <8 x float>]], ptr %172, i64 0, i64 %1266, !dbg !2427
  %1268 = load i32, ptr %260, align 4, !dbg !2428, !tbaa !8
  %1269 = sext i32 %1268 to i64, !dbg !2427
  %1270 = getelementptr inbounds [2 x <8 x float>], ptr %1267, i64 0, i64 %1269, !dbg !2427
  %1271 = load <8 x float>, ptr %1270, align 32, !dbg !2427, !tbaa !47
  %1272 = load i32, ptr %262, align 4, !dbg !2429, !tbaa !8
  %1273 = extractelement <8 x float> %1271, i32 %1272, !dbg !2427
  %1274 = load ptr, ptr %259, align 8, !dbg !2430, !tbaa !128
  %1275 = load i32, ptr %149, align 4, !dbg !2431, !tbaa !8
  %1276 = mul nsw i32 %1275, 320, !dbg !2432
  %1277 = load i32, ptr %155, align 4, !dbg !2433, !tbaa !8
  %1278 = mul nsw i32 8, %1277, !dbg !2434
  %1279 = load i32, ptr %262, align 4, !dbg !2435, !tbaa !8
  %1280 = add nsw i32 %1278, %1279, !dbg !2436
  %1281 = mul nsw i32 %1280, 20, !dbg !2437
  %1282 = add nsw i32 %1276, %1281, !dbg !2438
  %1283 = load i32, ptr %154, align 4, !dbg !2439, !tbaa !8
  %1284 = add nsw i32 %1282, %1283, !dbg !2440
  %1285 = zext i32 %1284 to i64, !dbg !2430
  %1286 = getelementptr inbounds nuw float, ptr %1274, i64 %1285, !dbg !2430
  store float %1273, ptr %1286, align 4, !dbg !2441, !tbaa !177
  br label %1287, !dbg !2442

1287:                                             ; preds = %1264
  %1288 = load i32, ptr %262, align 4, !dbg !2443, !tbaa !8
  %1289 = add nsw i32 %1288, 1, !dbg !2443
  store i32 %1289, ptr %262, align 4, !dbg !2443, !tbaa !8
  br label %1260, !dbg !2425, !llvm.loop !2444

1290:                                             ; preds = %1263
  call void @_Z13__syncthreadsv() #26, !dbg !2445
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %130) #25, !dbg !2446
  store i32 0, ptr %263, align 4, !dbg !2447, !tbaa !8
  br label %1291, !dbg !2446

1291:                                             ; preds = %1361, %1290
  %1292 = load i32, ptr %263, align 4, !dbg !2448, !tbaa !8
  %1293 = icmp slt i32 %1292, 8, !dbg !2449
  br i1 %1293, label %1295, label %1294, !dbg !2450

1294:                                             ; preds = %1291
  store i32 83, ptr %153, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %130) #25, !dbg !2450
  br label %1364

1295:                                             ; preds = %1291
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %131) #25, !dbg !2451
  %1296 = load i32, ptr %263, align 4, !dbg !2452, !tbaa !8
  %1297 = ashr i32 %1296, 1, !dbg !2453
  store i32 %1297, ptr %264, align 4, !dbg !2454, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %132) #25, !dbg !2455
  %1298 = load i32, ptr %263, align 4, !dbg !2456, !tbaa !8
  %1299 = and i32 %1298, 1, !dbg !2457
  store i32 %1299, ptr %265, align 4, !dbg !2458, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %133) #25, !dbg !2459
  %1300 = load i32, ptr %148, align 4, !dbg !2460, !tbaa !8
  %1301 = and i32 %1300, 15, !dbg !2461
  store i32 %1301, ptr %266, align 4, !dbg !2462, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %134) #25, !dbg !2463
  %1302 = load i32, ptr %148, align 4, !dbg !2464, !tbaa !8
  %1303 = ashr i32 %1302, 4, !dbg !2465
  %1304 = and i32 %1303, 15, !dbg !2466
  store i32 %1304, ptr %267, align 4, !dbg !2467, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %135) #25, !dbg !2468
  %1305 = load i32, ptr %264, align 4, !dbg !2469, !tbaa !8
  %1306 = mul nsw i32 %1305, 32, !dbg !2470
  %1307 = load i32, ptr %260, align 4, !dbg !2471, !tbaa !8
  %1308 = mul nsw i32 %1307, 16, !dbg !2472
  %1309 = add nsw i32 %1306, %1308, !dbg !2473
  %1310 = load i32, ptr %266, align 4, !dbg !2474, !tbaa !8
  %1311 = add nsw i32 %1309, %1310, !dbg !2475
  store i32 %1311, ptr %268, align 4, !dbg !2476, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %136) #25, !dbg !2477
  %1312 = load i32, ptr %265, align 4, !dbg !2478, !tbaa !8
  %1313 = mul nsw i32 %1312, 64, !dbg !2479
  %1314 = load i32, ptr %261, align 4, !dbg !2480, !tbaa !8
  %1315 = mul nsw i32 %1314, 16, !dbg !2481
  %1316 = add nsw i32 %1313, %1315, !dbg !2482
  %1317 = load i32, ptr %267, align 4, !dbg !2483, !tbaa !8
  %1318 = add nsw i32 %1316, %1317, !dbg !2484
  store i32 %1318, ptr %269, align 4, !dbg !2485, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %137) #25, !dbg !2486
  %1319 = load i32, ptr %264, align 4, !dbg !2487, !tbaa !8
  %1320 = shl i32 %1319, 1, !dbg !2488
  %1321 = load i32, ptr %265, align 4, !dbg !2489, !tbaa !8
  %1322 = or i32 %1320, %1321, !dbg !2490
  store i32 %1322, ptr %270, align 4, !dbg !2491, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %138) #25, !dbg !2492
  %1323 = load ptr, ptr %259, align 8, !dbg !2493, !tbaa !128
  %1324 = load i32, ptr %270, align 4, !dbg !2494, !tbaa !8
  %1325 = mul nsw i32 %1324, 320, !dbg !2495
  %1326 = load i32, ptr %266, align 4, !dbg !2496, !tbaa !8
  %1327 = mul nsw i32 %1326, 20, !dbg !2497
  %1328 = add nsw i32 %1325, %1327, !dbg !2498
  %1329 = load i32, ptr %267, align 4, !dbg !2499, !tbaa !8
  %1330 = add nsw i32 %1328, %1329, !dbg !2500
  %1331 = zext i32 %1330 to i64, !dbg !2493
  %1332 = getelementptr inbounds nuw float, ptr %1323, i64 %1331, !dbg !2493
  %1333 = load float, ptr %1332, align 4, !dbg !2493, !tbaa !177
  store float %1333, ptr %271, align 4, !dbg !2501, !tbaa !177
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %139) #25, !dbg !2502
  %1334 = load i32, ptr %151, align 4, !dbg !2503, !tbaa !8
  %1335 = load i32, ptr %268, align 4, !dbg !2504, !tbaa !8
  %1336 = add nsw i32 %1334, %1335, !dbg !2505
  store i32 %1336, ptr %272, align 4, !dbg !2506, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %140) #25, !dbg !2507
  %1337 = load i32, ptr %152, align 4, !dbg !2508, !tbaa !8
  %1338 = load i32, ptr %269, align 4, !dbg !2509, !tbaa !8
  %1339 = add nsw i32 %1337, %1338, !dbg !2510
  store i32 %1339, ptr %273, align 4, !dbg !2511, !tbaa !8
  %1340 = load i32, ptr %272, align 4, !dbg !2512, !tbaa !8
  %1341 = load i32, ptr %145, align 4, !dbg !2513, !tbaa !8
  %1342 = icmp slt i32 %1340, %1341, !dbg !2514
  br i1 %1342, label %1343, label %1360, !dbg !2515

1343:                                             ; preds = %1295
  %1344 = load i32, ptr %273, align 4, !dbg !2516, !tbaa !8
  %1345 = load i32, ptr %147, align 4, !dbg !2517, !tbaa !8
  %1346 = icmp slt i32 %1344, %1345, !dbg !2518
  br i1 %1346, label %1347, label %1360, !dbg !2515

1347:                                             ; preds = %1343
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %141) #25, !dbg !2519
  %1348 = load ptr, ptr %144, align 8, !dbg !2520, !tbaa !128
  %1349 = load i32, ptr %273, align 4, !dbg !2521, !tbaa !8
  %1350 = sext i32 %1349 to i64, !dbg !2521
  %1351 = load i32, ptr %145, align 4, !dbg !2522, !tbaa !8
  %1352 = sext i32 %1351 to i64, !dbg !2522
  %1353 = mul nsw i64 %1350, %1352, !dbg !2523
  %1354 = getelementptr inbounds float, ptr %1348, i64 %1353, !dbg !2524
  %1355 = load i32, ptr %272, align 4, !dbg !2525, !tbaa !8
  %1356 = sext i32 %1355 to i64, !dbg !2526
  %1357 = getelementptr inbounds float, ptr %1354, i64 %1356, !dbg !2526
  store ptr %1357, ptr %274, align 8, !dbg !2527, !tbaa !128
  %1358 = load float, ptr %271, align 4, !dbg !2528, !tbaa !177
  %1359 = load ptr, ptr %274, align 8, !dbg !2529, !tbaa !128
  store float %1358, ptr %1359, align 4, !dbg !2530, !tbaa !177
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %141) #25, !dbg !2531
  br label %1360, !dbg !2531

1360:                                             ; preds = %1347, %1343, %1295
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %140) #25, !dbg !2532
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %139) #25, !dbg !2532
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %138) #25, !dbg !2532
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %137) #25, !dbg !2532
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %136) #25, !dbg !2532
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %135) #25, !dbg !2532
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %134) #25, !dbg !2532
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %133) #25, !dbg !2532
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %132) #25, !dbg !2532
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %131) #25, !dbg !2532
  br label %1361, !dbg !2532

1361:                                             ; preds = %1360
  %1362 = load i32, ptr %263, align 4, !dbg !2533, !tbaa !8
  %1363 = add nsw i32 %1362, 1, !dbg !2533
  store i32 %1363, ptr %263, align 4, !dbg !2533, !tbaa !8
  br label %1291, !dbg !2450, !llvm.loop !2534

1364:                                             ; preds = %1294
  call void @_Z13__syncthreadsv() #26, !dbg !2535
  br label %1365, !dbg !2536

1365:                                             ; preds = %1364
  %1366 = load i32, ptr %261, align 4, !dbg !2537, !tbaa !8
  %1367 = add nsw i32 %1366, 1, !dbg !2537
  store i32 %1367, ptr %261, align 4, !dbg !2537, !tbaa !8
  br label %1255, !dbg !2420, !llvm.loop !2538

1368:                                             ; preds = %1258
  br label %1369, !dbg !2539

1369:                                             ; preds = %1368
  %1370 = load i32, ptr %260, align 4, !dbg !2540, !tbaa !8
  %1371 = add nsw i32 %1370, 1, !dbg !2540
  store i32 %1371, ptr %260, align 4, !dbg !2540, !tbaa !8
  br label %1250, !dbg !2415, !llvm.loop !2541

1372:                                             ; preds = %1253
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %126) #25, !dbg !2542
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %60) #25, !dbg !2542
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %59) #25, !dbg !2542
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %58) #25, !dbg !2542
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %57) #25, !dbg !2542
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %56) #25, !dbg !2542
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %55) #25, !dbg !2542
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %53) #25, !dbg !2542
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %49) #25, !dbg !2542
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %41) #25, !dbg !2542
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %40) #25, !dbg !2542
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %39) #25, !dbg !2542
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %38) #25, !dbg !2542
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %37) #25, !dbg !2542
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %30) #25, !dbg !2542
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %29) #25, !dbg !2542
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %28) #25, !dbg !2542
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %27) #25, !dbg !2542
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %26) #25, !dbg !2542
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %25) #25, !dbg !2542
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %24) #25, !dbg !2542
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %23) #25, !dbg !2542
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %22) #25, !dbg !2542
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %21) #25, !dbg !2542
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %20) #25, !dbg !2542
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %19) #25, !dbg !2542
  store i32 0, ptr %153, align 4, !dbg !2542
  br label %1373, !dbg !2542

1373:                                             ; preds = %1372, %291
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %17) #25, !dbg !2542
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %16) #25, !dbg !2542
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %15) #25, !dbg !2542
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %14) #25, !dbg !2542
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %13) #25, !dbg !2542
  %1374 = load i32, ptr %153, align 4
  switch i32 %1374, label %1376 [
    i32 0, label %1375
    i32 1, label %1375
  ]

1375:                                             ; preds = %1373, %1373
  ret void, !dbg !2542

1376:                                             ; preds = %1373
  unreachable
}

; Function Attrs: convergent mustprogress norecurse nounwind uwtable
define protected amdgpu_kernel void @gemm_mq4g256v2_residual_mmq_iu4_full_add(ptr addrspace(1) noalias noundef %0, ptr addrspace(1) noalias noundef %1, ptr addrspace(1) noalias noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5, i32 noundef %6) #7 !dbg !2543 {
  %8 = alloca ptr, align 8, addrspace(5)
  %9 = alloca ptr, align 8, addrspace(5)
  %10 = alloca ptr, align 8, addrspace(5)
  %11 = alloca ptr, align 8, addrspace(5)
  %12 = alloca ptr, align 8, addrspace(5)
  %13 = alloca ptr, align 8, addrspace(5)
  %14 = alloca i32, align 4, addrspace(5)
  %15 = alloca i32, align 4, addrspace(5)
  %16 = alloca i32, align 4, addrspace(5)
  %17 = alloca i32, align 4, addrspace(5)
  %18 = addrspacecast ptr addrspace(5) %11 to ptr
  %19 = addrspacecast ptr addrspace(5) %12 to ptr
  %20 = addrspacecast ptr addrspace(5) %13 to ptr
  %21 = addrspacecast ptr addrspace(5) %14 to ptr
  %22 = addrspacecast ptr addrspace(5) %15 to ptr
  %23 = addrspacecast ptr addrspace(5) %16 to ptr
  %24 = addrspacecast ptr addrspace(5) %17 to ptr
  store ptr addrspace(1) %0, ptr addrspace(5) %8, align 8
  %25 = load ptr, ptr addrspace(5) %8, align 8, !tbaa !28
  store ptr addrspace(1) %1, ptr addrspace(5) %9, align 8
  %26 = load ptr, ptr addrspace(5) %9, align 8, !tbaa !130
  store ptr addrspace(1) %2, ptr addrspace(5) %10, align 8
  %27 = load ptr, ptr addrspace(5) %10, align 8, !tbaa !128
  store ptr %25, ptr %18, align 8, !tbaa !28
  store ptr %26, ptr %19, align 8, !tbaa !130
  store ptr %27, ptr %20, align 8, !tbaa !128
  store i32 %3, ptr %21, align 4, !tbaa !8
  store i32 %4, ptr %22, align 4, !tbaa !8
  store i32 %5, ptr %23, align 4, !tbaa !8
  store i32 %6, ptr %24, align 4, !tbaa !8
  %28 = load ptr, ptr %18, align 8, !dbg !2544, !tbaa !28
  %29 = load ptr, ptr %19, align 8, !dbg !2545, !tbaa !130
  %30 = load ptr, ptr %20, align 8, !dbg !2546, !tbaa !128
  %31 = load i32, ptr %21, align 4, !dbg !2547, !tbaa !8
  %32 = load i32, ptr %22, align 4, !dbg !2548, !tbaa !8
  %33 = load i32, ptr %23, align 4, !dbg !2549, !tbaa !8
  call void @_ZL42gemm_mq4g256v2_residual_mmq_iu4_gfx12_bodyILb1EEvPKcPK12block_i4_128Pfiii(ptr noundef %28, ptr noundef %29, ptr noundef %30, i32 noundef %31, i32 noundef %32, i32 noundef %33) #26, !dbg !2550
  ret void, !dbg !2551
}

; Function Attrs: convergent mustprogress norecurse nounwind uwtable
define protected amdgpu_kernel void @gemm_mq4g256v2_residual_mmq_iu4_full_set(ptr addrspace(1) noalias noundef %0, ptr addrspace(1) noalias noundef %1, ptr addrspace(1) noalias noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5, i32 noundef %6) #7 !dbg !2552 {
  %8 = alloca ptr, align 8, addrspace(5)
  %9 = alloca ptr, align 8, addrspace(5)
  %10 = alloca ptr, align 8, addrspace(5)
  %11 = alloca ptr, align 8, addrspace(5)
  %12 = alloca ptr, align 8, addrspace(5)
  %13 = alloca ptr, align 8, addrspace(5)
  %14 = alloca i32, align 4, addrspace(5)
  %15 = alloca i32, align 4, addrspace(5)
  %16 = alloca i32, align 4, addrspace(5)
  %17 = alloca i32, align 4, addrspace(5)
  %18 = addrspacecast ptr addrspace(5) %11 to ptr
  %19 = addrspacecast ptr addrspace(5) %12 to ptr
  %20 = addrspacecast ptr addrspace(5) %13 to ptr
  %21 = addrspacecast ptr addrspace(5) %14 to ptr
  %22 = addrspacecast ptr addrspace(5) %15 to ptr
  %23 = addrspacecast ptr addrspace(5) %16 to ptr
  %24 = addrspacecast ptr addrspace(5) %17 to ptr
  store ptr addrspace(1) %0, ptr addrspace(5) %8, align 8
  %25 = load ptr, ptr addrspace(5) %8, align 8, !tbaa !28
  store ptr addrspace(1) %1, ptr addrspace(5) %9, align 8
  %26 = load ptr, ptr addrspace(5) %9, align 8, !tbaa !130
  store ptr addrspace(1) %2, ptr addrspace(5) %10, align 8
  %27 = load ptr, ptr addrspace(5) %10, align 8, !tbaa !128
  store ptr %25, ptr %18, align 8, !tbaa !28
  store ptr %26, ptr %19, align 8, !tbaa !130
  store ptr %27, ptr %20, align 8, !tbaa !128
  store i32 %3, ptr %21, align 4, !tbaa !8
  store i32 %4, ptr %22, align 4, !tbaa !8
  store i32 %5, ptr %23, align 4, !tbaa !8
  store i32 %6, ptr %24, align 4, !tbaa !8
  %28 = load ptr, ptr %18, align 8, !dbg !2553, !tbaa !28
  %29 = load ptr, ptr %19, align 8, !dbg !2554, !tbaa !130
  %30 = load ptr, ptr %20, align 8, !dbg !2555, !tbaa !128
  %31 = load i32, ptr %21, align 4, !dbg !2556, !tbaa !8
  %32 = load i32, ptr %22, align 4, !dbg !2557, !tbaa !8
  %33 = load i32, ptr %23, align 4, !dbg !2558, !tbaa !8
  call void @_ZL42gemm_mq4g256v2_residual_mmq_iu4_gfx12_bodyILb0EEvPKcPK12block_i4_128Pfiii(ptr noundef %28, ptr noundef %29, ptr noundef %30, i32 noundef %31, i32 noundef %32, i32 noundef %33) #26, !dbg !2559
  ret void, !dbg !2560
}

; Function Attrs: alwaysinline convergent mustprogress nounwind uwtable
define internal noundef i32 @_ZL21__hip_get_block_idx_yv() #5 !dbg !2561 {
  %1 = call i64 @__ockl_get_group_id(i32 noundef 1) #27, !dbg !2562
  %2 = trunc i64 %1 to i32, !dbg !2562
  ret i32 %2, !dbg !2563
}

; Function Attrs: alwaysinline convergent mustprogress nounwind uwtable
define internal noundef i32 @_ZL21__hip_get_block_idx_xv() #5 !dbg !2564 {
  %1 = call i64 @__ockl_get_group_id(i32 noundef 0) #27, !dbg !2565
  %2 = trunc i64 %1 to i32, !dbg !2565
  ret i32 %2, !dbg !2566
}

; Function Attrs: alwaysinline convergent mustprogress nounwind uwtable
define internal noundef i32 @_ZL21__hip_get_block_dim_xv() #5 !dbg !2567 {
  %1 = call i64 @__ockl_get_local_size(i32 noundef 0) #27, !dbg !2568
  %2 = trunc i64 %1 to i32, !dbg !2568
  ret i32 %2, !dbg !2569
}

; Function Attrs: alwaysinline convergent mustprogress nounwind uwtable
define internal noundef i32 @_ZL22__hip_get_thread_idx_xv() #5 !dbg !2570 {
  %1 = call i64 @__ockl_get_local_id(i32 noundef 0) #27, !dbg !2571
  %2 = trunc i64 %1 to i32, !dbg !2571
  ret i32 %2, !dbg !2572
}

; Function Attrs: convergent mustprogress nounwind uwtable
define linkonce_odr hidden void @_ZN15HIP_vector_typeIfLj4EEC2IJffffETnPN14__hip_internal9enable_ifIXaagtLj4ELi1EeqsZT_Lj4EEvE4typeELPv0EEEDpT_(ptr noundef nonnull align 16 dereferenceable(16) %0, float noundef %1, float noundef %2, float noundef %3, float noundef %4) unnamed_addr #8 comdat align 2 !dbg !2573 {
  %6 = alloca ptr, align 8, addrspace(5)
  %7 = alloca float, align 4, addrspace(5)
  %8 = alloca float, align 4, addrspace(5)
  %9 = alloca float, align 4, addrspace(5)
  %10 = alloca float, align 4, addrspace(5)
  %11 = addrspacecast ptr addrspace(5) %6 to ptr
  %12 = addrspacecast ptr addrspace(5) %7 to ptr
  %13 = addrspacecast ptr addrspace(5) %8 to ptr
  %14 = addrspacecast ptr addrspace(5) %9 to ptr
  %15 = addrspacecast ptr addrspace(5) %10 to ptr
  store ptr %0, ptr %11, align 8, !tbaa !2574
  store float %1, ptr %12, align 4, !tbaa !177
  store float %2, ptr %13, align 4, !tbaa !177
  store float %3, ptr %14, align 4, !tbaa !177
  store float %4, ptr %15, align 4, !tbaa !177
  %16 = load ptr, ptr %11, align 8
  %17 = load float, ptr %12, align 4, !dbg !2576, !tbaa !177
  %18 = load float, ptr %13, align 4, !dbg !2576, !tbaa !177
  %19 = load float, ptr %14, align 4, !dbg !2576, !tbaa !177
  %20 = load float, ptr %15, align 4, !dbg !2576, !tbaa !177
  call void @_ZN15HIP_vector_baseIfLj4EEC2Effff(ptr noundef nonnull align 16 dereferenceable(16) %16, float noundef %17, float noundef %18, float noundef %19, float noundef %20) #26, !dbg !2577
  ret void, !dbg !2578
}

; Function Attrs: convergent mustprogress nounwind uwtable
define linkonce_odr hidden void @_ZN15HIP_vector_baseIfLj4EEC2Effff(ptr noundef nonnull align 16 dereferenceable(16) %0, float noundef %1, float noundef %2, float noundef %3, float noundef %4) unnamed_addr #8 comdat align 2 !dbg !2579 {
  %6 = alloca ptr, align 8, addrspace(5)
  %7 = alloca float, align 4, addrspace(5)
  %8 = alloca float, align 4, addrspace(5)
  %9 = alloca float, align 4, addrspace(5)
  %10 = alloca float, align 4, addrspace(5)
  %11 = addrspacecast ptr addrspace(5) %6 to ptr
  %12 = addrspacecast ptr addrspace(5) %7 to ptr
  %13 = addrspacecast ptr addrspace(5) %8 to ptr
  %14 = addrspacecast ptr addrspace(5) %9 to ptr
  %15 = addrspacecast ptr addrspace(5) %10 to ptr
  store ptr %0, ptr %11, align 8, !tbaa !2580
  store float %1, ptr %12, align 4, !tbaa !177
  store float %2, ptr %13, align 4, !tbaa !177
  store float %3, ptr %14, align 4, !tbaa !177
  store float %4, ptr %15, align 4, !tbaa !177
  %16 = load ptr, ptr %11, align 8
  %17 = getelementptr inbounds nuw %struct.HIP_vector_base, ptr %16, i32 0, i32 0, !dbg !2582
  %18 = load float, ptr %12, align 4, !dbg !2583, !tbaa !177
  store float %18, ptr %17, align 16, !dbg !2582, !tbaa !173
  %19 = getelementptr inbounds nuw %struct.HIP_vector_base, ptr %16, i32 0, i32 1, !dbg !2584
  %20 = load float, ptr %13, align 4, !dbg !2585, !tbaa !177
  store float %20, ptr %19, align 4, !dbg !2584, !tbaa !179
  %21 = getelementptr inbounds nuw %struct.HIP_vector_base, ptr %16, i32 0, i32 2, !dbg !2586
  %22 = load float, ptr %14, align 4, !dbg !2587, !tbaa !177
  store float %22, ptr %21, align 8, !dbg !2586, !tbaa !181
  %23 = getelementptr inbounds nuw %struct.HIP_vector_base, ptr %16, i32 0, i32 3, !dbg !2588
  %24 = load float, ptr %15, align 4, !dbg !2589, !tbaa !177
  store float %24, ptr %23, align 4, !dbg !2588, !tbaa !183
  ret void, !dbg !2590
}

; Function Attrs: alwaysinline convergent mustprogress nounwind uwtable
define internal noundef float @_ZL5fmaxfff(float noundef %0, float noundef %1) #5 !dbg !2591 {
  %3 = alloca float, align 4, addrspace(5)
  %4 = alloca float, align 4, addrspace(5)
  %5 = addrspacecast ptr addrspace(5) %3 to ptr
  %6 = addrspacecast ptr addrspace(5) %4 to ptr
  store float %0, ptr %5, align 4, !tbaa !177
  store float %1, ptr %6, align 4, !tbaa !177
  %7 = load float, ptr %5, align 4, !dbg !2593, !tbaa !177
  %8 = load float, ptr %6, align 4, !dbg !2594, !tbaa !177
  %9 = call nsz contract float @llvm.maxnum.f32(float %7, float %8), !dbg !2595
  ret float %9, !dbg !2596
}

; Function Attrs: alwaysinline convergent mustprogress nounwind uwtable
define internal noundef float @_ZL5fabsff(float noundef %0) #5 !dbg !2597 {
  %2 = alloca float, align 4, addrspace(5)
  %3 = addrspacecast ptr addrspace(5) %2 to ptr
  store float %0, ptr %3, align 4, !tbaa !177
  %4 = load float, ptr %3, align 4, !dbg !2598, !tbaa !177
  %5 = call contract float @llvm.fabs.f32(float %4), !dbg !2599
  ret float %5, !dbg !2600
}

; Function Attrs: convergent inlinehint mustprogress nounwind uwtable
define linkonce_odr hidden noundef float @_Z10__shfl_xorfii(float noundef %0, i32 noundef %1, i32 noundef %2) #6 comdat !dbg !2601 {
  %4 = alloca float, align 4, addrspace(5)
  %5 = alloca i32, align 4, addrspace(5)
  %6 = alloca i32, align 4, addrspace(5)
  %7 = alloca %union.anon, align 4, addrspace(5)
  %8 = alloca i32, align 4, addrspace(5)
  %9 = addrspacecast ptr addrspace(5) %4 to ptr
  %10 = addrspacecast ptr addrspace(5) %5 to ptr
  %11 = addrspacecast ptr addrspace(5) %6 to ptr
  %12 = addrspacecast ptr addrspace(5) %7 to ptr
  store float %0, ptr %9, align 4, !tbaa !177
  store i32 %1, ptr %10, align 4, !tbaa !8
  store i32 %2, ptr %11, align 4, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %7) #25, !dbg !2603
  %13 = load float, ptr %9, align 4, !dbg !2604, !tbaa !177
  store float %13, ptr %12, align 4, !dbg !2605, !tbaa !47
  %14 = load i32, ptr %12, align 4, !dbg !2606, !tbaa !47
  %15 = load i32, ptr %10, align 4, !dbg !2607, !tbaa !8
  %16 = load i32, ptr %11, align 4, !dbg !2608, !tbaa !8
  %17 = freeze i32 %14, !dbg !2609
  %18 = call noundef i32 @_Z10__shfl_xoriii(i32 noundef %17, i32 noundef %15, i32 noundef %16) #26, !dbg !2609
  store i32 %18, ptr %12, align 4, !dbg !2610, !tbaa !47
  %19 = load float, ptr %12, align 4, !dbg !2611, !tbaa !47
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %7) #25, !dbg !2612
  ret float %19, !dbg !2613
}

; Function Attrs: alwaysinline convergent mustprogress nounwind uwtable
define internal noundef float @_ZL5rintff(float noundef %0) #5 !dbg !2614 {
  %2 = alloca float, align 4, addrspace(5)
  %3 = addrspacecast ptr addrspace(5) %2 to ptr
  store float %0, ptr %3, align 4, !tbaa !177
  %4 = load float, ptr %3, align 4, !dbg !2615, !tbaa !177
  %5 = call contract float @llvm.rint.f32(float %4), !dbg !2616
  ret float %5, !dbg !2617
}

; Function Attrs: alwaysinline convergent mustprogress nounwind uwtable
define internal noundef float @_ZL5fminfff(float noundef %0, float noundef %1) #5 !dbg !2618 {
  %3 = alloca float, align 4, addrspace(5)
  %4 = alloca float, align 4, addrspace(5)
  %5 = addrspacecast ptr addrspace(5) %3 to ptr
  %6 = addrspacecast ptr addrspace(5) %4 to ptr
  store float %0, ptr %5, align 4, !tbaa !177
  store float %1, ptr %6, align 4, !tbaa !177
  %7 = load float, ptr %5, align 4, !dbg !2619, !tbaa !177
  %8 = load float, ptr %6, align 4, !dbg !2620, !tbaa !177
  %9 = call nsz contract float @llvm.minnum.f32(float %7, float %8), !dbg !2621
  ret float %9, !dbg !2622
}

; Function Attrs: alwaysinline convergent mustprogress nounwind uwtable
define internal noundef float @_ZL9__fmaf_rnfff(float noundef %0, float noundef %1, float noundef %2) #5 !dbg !2623 {
  %4 = alloca float, align 4, addrspace(5)
  %5 = alloca float, align 4, addrspace(5)
  %6 = alloca float, align 4, addrspace(5)
  %7 = addrspacecast ptr addrspace(5) %4 to ptr
  %8 = addrspacecast ptr addrspace(5) %5 to ptr
  %9 = addrspacecast ptr addrspace(5) %6 to ptr
  store float %0, ptr %7, align 4, !tbaa !177
  store float %1, ptr %8, align 4, !tbaa !177
  store float %2, ptr %9, align 4, !tbaa !177
  %10 = load float, ptr %7, align 4, !dbg !2624, !tbaa !177
  %11 = load float, ptr %8, align 4, !dbg !2625, !tbaa !177
  %12 = load float, ptr %9, align 4, !dbg !2626, !tbaa !177
  %13 = call contract float @llvm.fma.f32(float %10, float %11, float %12), !dbg !2627
  ret float %13, !dbg !2628
}

; Function Attrs: convergent inlinehint mustprogress nounwind uwtable
define linkonce_odr hidden noundef i32 @_Z10__shfl_xoriii(i32 noundef %0, i32 noundef %1, i32 noundef %2) #6 comdat !dbg !2629 {
  %4 = alloca i32, align 4, addrspace(5)
  %5 = alloca i32, align 4, addrspace(5)
  %6 = alloca i32, align 4, addrspace(5)
  %7 = alloca i32, align 4, addrspace(5)
  %8 = alloca i32, align 4, addrspace(5)
  %9 = alloca i32, align 4, addrspace(5)
  %10 = addrspacecast ptr addrspace(5) %4 to ptr
  %11 = addrspacecast ptr addrspace(5) %5 to ptr
  %12 = addrspacecast ptr addrspace(5) %6 to ptr
  %13 = addrspacecast ptr addrspace(5) %7 to ptr
  %14 = addrspacecast ptr addrspace(5) %8 to ptr
  store i32 %0, ptr %10, align 4, !tbaa !8
  store i32 %1, ptr %11, align 4, !tbaa !8
  store i32 %2, ptr %12, align 4, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %7) #25, !dbg !2630
  %15 = call noundef i32 @_ZL9__lane_idv() #26, !dbg !2631
  store i32 %15, ptr %13, align 4, !dbg !2632, !tbaa !8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %8) #25, !dbg !2633
  %16 = load i32, ptr %13, align 4, !dbg !2634, !tbaa !8
  %17 = load i32, ptr %11, align 4, !dbg !2635, !tbaa !8
  %18 = xor i32 %16, %17, !dbg !2636
  store i32 %18, ptr %14, align 4, !dbg !2637, !tbaa !8
  %19 = load i32, ptr %14, align 4, !dbg !2638, !tbaa !8
  %20 = load i32, ptr %13, align 4, !dbg !2639, !tbaa !8
  %21 = load i32, ptr %12, align 4, !dbg !2640, !tbaa !8
  %22 = add nsw i32 %20, %21, !dbg !2641
  %23 = load i32, ptr %12, align 4, !dbg !2642, !tbaa !8
  %24 = sub nsw i32 %23, 1, !dbg !2643
  %25 = xor i32 %24, -1, !dbg !2644
  %26 = and i32 %22, %25, !dbg !2645
  %27 = icmp sge i32 %19, %26, !dbg !2646
  br i1 %27, label %28, label %30, !dbg !2638

28:                                               ; preds = %3
  %29 = load i32, ptr %13, align 4, !dbg !2647, !tbaa !8
  br label %32, !dbg !2638

30:                                               ; preds = %3
  %31 = load i32, ptr %14, align 4, !dbg !2648, !tbaa !8
  br label %32, !dbg !2638

32:                                               ; preds = %30, %28
  %33 = phi i32 [ %29, %28 ], [ %31, %30 ], !dbg !2638
  store i32 %33, ptr %14, align 4, !dbg !2649, !tbaa !8
  %34 = load i32, ptr %10, align 4, !dbg !2650, !tbaa !8
  %35 = load i32, ptr %14, align 4, !dbg !2651, !tbaa !8
  %36 = call i32 @llvm.amdgcn.wave.shuffle.i32(i32 %34, i32 %35), !dbg !2652
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %8) #25, !dbg !2653
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %7) #25, !dbg !2653
  ret i32 %36, !dbg !2654
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #9

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fabs.f32(float) #9

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.rint.f32(float) #9

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.minnum.f32(float, float) #9

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fma.f32(float, float, float) #9

; Function Attrs: convergent inlinehint mustprogress nounwind uwtable
define internal noundef i32 @_ZL9__lane_idv() #6 !dbg !2655 {
  %1 = alloca i32, align 4, addrspace(5)
  %2 = call noundef i32 @"_ZNK3$_0cviEv"(ptr noundef nonnull align 1 dereferenceable(1) addrspacecast (ptr addrspace(4) @warpSize to ptr)) #27, !dbg !2656
  %3 = icmp eq i32 %2, 32, !dbg !2657
  br i1 %3, label %4, label %6, !dbg !2657

4:                                                ; preds = %0
  %5 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !2658
  store i32 %5, ptr addrspace(5) %1, align 4, !dbg !2659
  br label %9, !dbg !2659

6:                                                ; preds = %0
  %7 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !2660
  %8 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %7), !dbg !2661
  store i32 %8, ptr addrspace(5) %1, align 4, !dbg !2662
  br label %9, !dbg !2662

9:                                                ; preds = %6, %4
  %10 = load i32, ptr addrspace(5) %1, align 4, !dbg !2663
  ret i32 %10, !dbg !2663
}

; Function Attrs: convergent nocallback nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.wave.shuffle.i32(i32, i32) #10

; Function Attrs: alwaysinline convergent mustprogress nounwind willreturn memory(none) uwtable
define internal noundef i32 @"_ZNK3$_0cviEv"(ptr noundef nonnull align 1 dereferenceable(1) %0) #11 align 2 !dbg !2664 {
  %2 = alloca ptr, align 8, addrspace(5)
  %3 = addrspacecast ptr addrspace(5) %2 to ptr
  store ptr %0, ptr %3, align 8, !tbaa !2665
  %4 = load ptr, ptr %3, align 8
  %5 = call i32 @llvm.amdgcn.wavefrontsize(), !dbg !2666
  ret i32 %5, !dbg !2667
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #12

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #12

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 32, 65) i32 @llvm.amdgcn.wavefrontsize() #13

; Function Attrs: convergent inlinehint mustprogress nounwind uwtable
define linkonce_odr hidden noundef float @_Z12__half2float6__half(i16 %0) #6 comdat !dbg !2668 {
  %2 = alloca %struct.__half, align 2, addrspace(5)
  %3 = alloca %struct.__half_raw, align 2, addrspace(5)
  %4 = addrspacecast ptr addrspace(5) %3 to ptr
  %5 = getelementptr inbounds nuw %struct.__half, ptr addrspace(5) %2, i32 0, i32 0
  %6 = getelementptr inbounds nuw %union.anon.0, ptr addrspace(5) %5, i32 0, i32 0
  store i16 %0, ptr addrspace(5) %6, align 2
  %7 = addrspacecast ptr addrspace(5) %2 to ptr
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %3) #25, !dbg !2670
  %8 = addrspacecast ptr %4 to ptr addrspace(5), !dbg !2671
  %9 = call i16 @_ZNK6__halfcv10__half_rawEv(ptr noundef nonnull align 2 dereferenceable(2) %7) #26, !dbg !2671
  %10 = getelementptr inbounds nuw %struct.__half_raw, ptr addrspace(5) %8, i32 0, i32 0, !dbg !2671
  %11 = getelementptr inbounds nuw %union.anon.1, ptr addrspace(5) %10, i32 0, i32 0, !dbg !2671
  store i16 %9, ptr addrspace(5) %11, align 2, !dbg !2671
  %12 = getelementptr inbounds nuw %struct.__half_raw, ptr %4, i32 0, i32 0, !dbg !2672
  %13 = load half, ptr %12, align 2, !dbg !2672, !tbaa !47
  %14 = fpext contract half %13 to float, !dbg !2670
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %3) #25, !dbg !2673
  ret float %14, !dbg !2673
}

; Function Attrs: convergent inlinehint mustprogress nounwind uwtable
define linkonce_odr hidden i16 @_Z16__ushort_as_halft(i16 noundef zeroext %0) #6 comdat !dbg !2674 {
  %2 = alloca %struct.__half, align 2, addrspace(5)
  %3 = alloca i16, align 2, addrspace(5)
  %4 = alloca %struct.__half_raw, align 2, addrspace(5)
  %5 = alloca i32, align 4, addrspace(5)
  %6 = addrspacecast ptr addrspace(5) %3 to ptr
  %7 = addrspacecast ptr addrspace(5) %4 to ptr
  store i16 %0, ptr %6, align 2, !tbaa !2675
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %4) #25, !dbg !2677
  %8 = load i16, ptr %6, align 2, !dbg !2678, !tbaa !2675
  %9 = getelementptr inbounds nuw %struct.__half_raw, ptr %7, i32 0, i32 0, !dbg !2679
  store i16 %8, ptr %9, align 2, !dbg !2680, !tbaa !47
  %10 = addrspacecast ptr addrspace(5) %2 to ptr, !dbg !2681
  call void @_ZN6__halfC2ERK10__half_raw(ptr noundef nonnull align 2 dereferenceable(2) %10, ptr noundef nonnull align 2 dereferenceable(2) %7) #26, !dbg !2681
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %4) #25, !dbg !2682
  %11 = getelementptr inbounds nuw %struct.__half, ptr addrspace(5) %2, i32 0, i32 0, !dbg !2682
  %12 = getelementptr inbounds nuw %union.anon.0, ptr addrspace(5) %11, i32 0, i32 0, !dbg !2682
  %13 = load i16, ptr addrspace(5) %12, align 2, !dbg !2682
  ret i16 %13, !dbg !2682
}

; Function Attrs: convergent inlinehint mustprogress nounwind uwtable
define linkonce_odr hidden void @_Z13__syncthreadsv() #6 comdat !dbg !2683 {
  call void @_ZL9__barrieri(i32 noundef 3) #26, !dbg !2685
  ret void, !dbg !2686
}

; Function Attrs: alwaysinline convergent mustprogress nounwind uwtable
define internal void @_ZL21iu4_bundle_single_accILi0EEvPKcS1_RA2_A2_KjRA4_S3_PA2_Dv8_fPKS8_RA4_KfRKDv8_i(ptr noalias noundef %0, ptr noalias noundef %1, ptr noundef nonnull align 4 dereferenceable(16) %2, ptr noundef nonnull align 4 dereferenceable(32) %3, ptr noundef %4, ptr noundef %5, ptr noundef nonnull align 4 dereferenceable(16) %6, ptr noundef nonnull align 32 dereferenceable(32) %7) #5 !dbg !2687 {
  %9 = alloca ptr, align 8, addrspace(5)
  %10 = alloca ptr, align 8, addrspace(5)
  %11 = alloca ptr, align 8, addrspace(5)
  %12 = alloca ptr, align 8, addrspace(5)
  %13 = alloca ptr, align 8, addrspace(5)
  %14 = alloca ptr, align 8, addrspace(5)
  %15 = alloca ptr, align 8, addrspace(5)
  %16 = alloca ptr, align 8, addrspace(5)
  %17 = alloca <2 x i32>, align 8, addrspace(5)
  %18 = alloca <2 x i32>, align 8, addrspace(5)
  %19 = alloca <2 x i32>, align 8, addrspace(5)
  %20 = alloca <2 x i32>, align 8, addrspace(5)
  %21 = alloca <2 x i32>, align 8, addrspace(5)
  %22 = alloca <2 x i32>, align 8, addrspace(5)
  %23 = alloca i32, align 4, addrspace(5)
  %24 = alloca i32, align 4, addrspace(5)
  %25 = alloca <2 x i32>, align 8, addrspace(5)
  %26 = alloca i32, align 4, addrspace(5)
  %27 = alloca <2 x i32>, align 8, addrspace(5)
  %28 = alloca <8 x i32>, align 32, addrspace(5)
  %29 = alloca <8 x float>, align 32, addrspace(5)
  %30 = alloca <8 x float>, align 32, addrspace(5)
  %31 = addrspacecast ptr addrspace(5) %9 to ptr
  %32 = addrspacecast ptr addrspace(5) %10 to ptr
  %33 = addrspacecast ptr addrspace(5) %11 to ptr
  %34 = addrspacecast ptr addrspace(5) %12 to ptr
  %35 = addrspacecast ptr addrspace(5) %13 to ptr
  %36 = addrspacecast ptr addrspace(5) %14 to ptr
  %37 = addrspacecast ptr addrspace(5) %15 to ptr
  %38 = addrspacecast ptr addrspace(5) %16 to ptr
  %39 = addrspacecast ptr addrspace(5) %17 to ptr
  %40 = addrspacecast ptr addrspace(5) %18 to ptr
  %41 = addrspacecast ptr addrspace(5) %19 to ptr
  %42 = addrspacecast ptr addrspace(5) %20 to ptr
  %43 = addrspacecast ptr addrspace(5) %21 to ptr
  %44 = addrspacecast ptr addrspace(5) %22 to ptr
  %45 = addrspacecast ptr addrspace(5) %23 to ptr
  %46 = addrspacecast ptr addrspace(5) %24 to ptr
  %47 = addrspacecast ptr addrspace(5) %25 to ptr
  %48 = addrspacecast ptr addrspace(5) %26 to ptr
  %49 = addrspacecast ptr addrspace(5) %27 to ptr
  %50 = addrspacecast ptr addrspace(5) %28 to ptr
  %51 = addrspacecast ptr addrspace(5) %29 to ptr
  %52 = addrspacecast ptr addrspace(5) %30 to ptr
  store ptr %0, ptr %31, align 8, !tbaa !28
  store ptr %1, ptr %32, align 8, !tbaa !28
  store ptr %2, ptr %33, align 8, !tbaa !2688
  store ptr %3, ptr %34, align 8, !tbaa !2688
  store ptr %4, ptr %35, align 8, !tbaa !2665
  store ptr %5, ptr %36, align 8, !tbaa !2665
  store ptr %6, ptr %37, align 8, !tbaa !128
  store ptr %7, ptr %38, align 8, !tbaa !2665
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %17) #25, !dbg !2690
  %53 = load ptr, ptr %31, align 8, !dbg !2691, !tbaa !28
  %54 = load ptr, ptr %33, align 8, !dbg !2692, !tbaa !2688, !nonnull !20, !align !2693
  %55 = getelementptr inbounds [2 x [2 x i32]], ptr %54, i64 0, i64 0, !dbg !2692
  %56 = getelementptr inbounds [2 x i32], ptr %55, i64 0, i64 0, !dbg !2692
  %57 = load i32, ptr %56, align 4, !dbg !2692, !tbaa !8
  %58 = zext i32 %57 to i64, !dbg !2694
  %59 = getelementptr inbounds nuw i8, ptr %53, i64 %58, !dbg !2694
  %60 = load <2 x i32>, ptr %59, align 8, !dbg !2694, !tbaa !47
  store <2 x i32> %60, ptr %39, align 8, !dbg !2695, !tbaa !47
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %18) #25, !dbg !2696
  %61 = load ptr, ptr %31, align 8, !dbg !2697, !tbaa !28
  %62 = load ptr, ptr %33, align 8, !dbg !2698, !tbaa !2688, !nonnull !20, !align !2693
  %63 = getelementptr inbounds [2 x [2 x i32]], ptr %62, i64 0, i64 1, !dbg !2698
  %64 = getelementptr inbounds [2 x i32], ptr %63, i64 0, i64 0, !dbg !2698
  %65 = load i32, ptr %64, align 4, !dbg !2698, !tbaa !8
  %66 = zext i32 %65 to i64, !dbg !2699
  %67 = getelementptr inbounds nuw i8, ptr %61, i64 %66, !dbg !2699
  %68 = load <2 x i32>, ptr %67, align 8, !dbg !2699, !tbaa !47
  store <2 x i32> %68, ptr %40, align 8, !dbg !2700, !tbaa !47
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %19) #25, !dbg !2701
  %69 = load ptr, ptr %32, align 8, !dbg !2702, !tbaa !28
  %70 = load ptr, ptr %34, align 8, !dbg !2703, !tbaa !2688, !nonnull !20, !align !2693
  %71 = getelementptr inbounds [4 x [2 x i32]], ptr %70, i64 0, i64 0, !dbg !2703
  %72 = getelementptr inbounds [2 x i32], ptr %71, i64 0, i64 0, !dbg !2703
  %73 = load i32, ptr %72, align 4, !dbg !2703, !tbaa !8
  %74 = zext i32 %73 to i64, !dbg !2704
  %75 = getelementptr inbounds nuw i8, ptr %69, i64 %74, !dbg !2704
  %76 = load <2 x i32>, ptr %75, align 8, !dbg !2704, !tbaa !47
  store <2 x i32> %76, ptr %41, align 8, !dbg !2705, !tbaa !47
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %20) #25, !dbg !2706
  %77 = load ptr, ptr %32, align 8, !dbg !2707, !tbaa !28
  %78 = load ptr, ptr %34, align 8, !dbg !2708, !tbaa !2688, !nonnull !20, !align !2693
  %79 = getelementptr inbounds [4 x [2 x i32]], ptr %78, i64 0, i64 1, !dbg !2708
  %80 = getelementptr inbounds [2 x i32], ptr %79, i64 0, i64 0, !dbg !2708
  %81 = load i32, ptr %80, align 4, !dbg !2708, !tbaa !8
  %82 = zext i32 %81 to i64, !dbg !2709
  %83 = getelementptr inbounds nuw i8, ptr %77, i64 %82, !dbg !2709
  %84 = load <2 x i32>, ptr %83, align 8, !dbg !2709, !tbaa !47
  store <2 x i32> %84, ptr %42, align 8, !dbg !2710, !tbaa !47
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %21) #25, !dbg !2711
  %85 = load ptr, ptr %32, align 8, !dbg !2712, !tbaa !28
  %86 = load ptr, ptr %34, align 8, !dbg !2713, !tbaa !2688, !nonnull !20, !align !2693
  %87 = getelementptr inbounds [4 x [2 x i32]], ptr %86, i64 0, i64 2, !dbg !2713
  %88 = getelementptr inbounds [2 x i32], ptr %87, i64 0, i64 0, !dbg !2713
  %89 = load i32, ptr %88, align 4, !dbg !2713, !tbaa !8
  %90 = zext i32 %89 to i64, !dbg !2714
  %91 = getelementptr inbounds nuw i8, ptr %85, i64 %90, !dbg !2714
  %92 = load <2 x i32>, ptr %91, align 8, !dbg !2714, !tbaa !47
  store <2 x i32> %92, ptr %43, align 8, !dbg !2715, !tbaa !47
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %22) #25, !dbg !2716
  %93 = load ptr, ptr %32, align 8, !dbg !2717, !tbaa !28
  %94 = load ptr, ptr %34, align 8, !dbg !2718, !tbaa !2688, !nonnull !20, !align !2693
  %95 = getelementptr inbounds [4 x [2 x i32]], ptr %94, i64 0, i64 3, !dbg !2718
  %96 = getelementptr inbounds [2 x i32], ptr %95, i64 0, i64 0, !dbg !2718
  %97 = load i32, ptr %96, align 4, !dbg !2718, !tbaa !8
  %98 = zext i32 %97 to i64, !dbg !2719
  %99 = getelementptr inbounds nuw i8, ptr %93, i64 %98, !dbg !2719
  %100 = load <2 x i32>, ptr %99, align 8, !dbg !2719, !tbaa !47
  store <2 x i32> %100, ptr %44, align 8, !dbg !2720, !tbaa !47
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %23) #25, !dbg !2721
  store i32 0, ptr %45, align 4, !dbg !2722, !tbaa !8
  br label %101, !dbg !2721

101:                                              ; preds = %183, %8
  %102 = load i32, ptr %45, align 4, !dbg !2723, !tbaa !8
  %103 = icmp slt i32 %102, 2, !dbg !2724
  br i1 %103, label %105, label %104, !dbg !2725

104:                                              ; preds = %101
  store i32 2, ptr %46, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %23) #25, !dbg !2725
  br label %186

105:                                              ; preds = %101
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %25) #25, !dbg !2726
  %106 = load i32, ptr %45, align 4, !dbg !2727, !tbaa !8
  %107 = icmp eq i32 %106, 0, !dbg !2728
  br i1 %107, label %108, label %110, !dbg !2729

108:                                              ; preds = %105
  %109 = load <2 x i32>, ptr %39, align 8, !dbg !2730, !tbaa !47
  br label %112, !dbg !2729

110:                                              ; preds = %105
  %111 = load <2 x i32>, ptr %40, align 8, !dbg !2731, !tbaa !47
  br label %112, !dbg !2729

112:                                              ; preds = %110, %108
  %113 = phi <2 x i32> [ %109, %108 ], [ %111, %110 ], !dbg !2729
  store <2 x i32> %113, ptr %47, align 8, !dbg !2732, !tbaa !47
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %26) #25, !dbg !2733
  store i32 0, ptr %48, align 4, !dbg !2734, !tbaa !8
  br label %114, !dbg !2733

114:                                              ; preds = %179, %112
  %115 = load i32, ptr %48, align 4, !dbg !2735, !tbaa !8
  %116 = icmp slt i32 %115, 4, !dbg !2736
  br i1 %116, label %118, label %117, !dbg !2737

117:                                              ; preds = %114
  store i32 5, ptr %46, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %26) #25, !dbg !2737
  br label %182

118:                                              ; preds = %114
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %27) #25, !dbg !2738
  %119 = load i32, ptr %48, align 4, !dbg !2739, !tbaa !8
  %120 = icmp eq i32 %119, 0, !dbg !2740
  br i1 %120, label %121, label %123, !dbg !2741

121:                                              ; preds = %118
  %122 = load <2 x i32>, ptr %41, align 8, !dbg !2742, !tbaa !47
  br label %139, !dbg !2741

123:                                              ; preds = %118
  %124 = load i32, ptr %48, align 4, !dbg !2743, !tbaa !8
  %125 = icmp eq i32 %124, 1, !dbg !2744
  br i1 %125, label %126, label %128, !dbg !2745

126:                                              ; preds = %123
  %127 = load <2 x i32>, ptr %42, align 8, !dbg !2746, !tbaa !47
  br label %137, !dbg !2745

128:                                              ; preds = %123
  %129 = load i32, ptr %48, align 4, !dbg !2747, !tbaa !8
  %130 = icmp eq i32 %129, 2, !dbg !2748
  br i1 %130, label %131, label %133, !dbg !2749

131:                                              ; preds = %128
  %132 = load <2 x i32>, ptr %43, align 8, !dbg !2750, !tbaa !47
  br label %135, !dbg !2749

133:                                              ; preds = %128
  %134 = load <2 x i32>, ptr %44, align 8, !dbg !2751, !tbaa !47
  br label %135, !dbg !2749

135:                                              ; preds = %133, %131
  %136 = phi <2 x i32> [ %132, %131 ], [ %134, %133 ], !dbg !2749
  br label %137, !dbg !2745

137:                                              ; preds = %135, %126
  %138 = phi <2 x i32> [ %127, %126 ], [ %136, %135 ], !dbg !2745
  br label %139, !dbg !2741

139:                                              ; preds = %137, %121
  %140 = phi <2 x i32> [ %122, %121 ], [ %138, %137 ], !dbg !2741
  store <2 x i32> %140, ptr %49, align 8, !dbg !2752, !tbaa !47
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %28) #25, !dbg !2753
  %141 = load <2 x i32>, ptr %47, align 8, !dbg !2754, !tbaa !47
  %142 = load <2 x i32>, ptr %49, align 8, !dbg !2755, !tbaa !47
  %143 = load ptr, ptr %38, align 8, !dbg !2756, !tbaa !2665, !nonnull !20, !align !2757
  %144 = load <8 x i32>, ptr %143, align 32, !dbg !2756, !tbaa !47
  %145 = call <8 x i32> @llvm.amdgcn.wmma.i32.16x16x32.iu4.v8i32.v2i32(i1 false, <2 x i32> %141, i1 true, <2 x i32> %142, <8 x i32> %144, i1 false), !dbg !2758
  store <8 x i32> %145, ptr %50, align 32, !dbg !2759, !tbaa !47
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %29) #25, !dbg !2760
  %146 = load ptr, ptr %36, align 8, !dbg !2761, !tbaa !2665
  %147 = load i32, ptr %45, align 4, !dbg !2762, !tbaa !8
  %148 = sext i32 %147 to i64, !dbg !2761
  %149 = getelementptr inbounds <8 x float>, ptr %146, i64 %148, !dbg !2761
  %150 = load <8 x float>, ptr %149, align 32, !dbg !2761, !tbaa !47
  %151 = load ptr, ptr %37, align 8, !dbg !2763, !tbaa !128, !nonnull !20, !align !2693
  %152 = load i32, ptr %48, align 4, !dbg !2764, !tbaa !8
  %153 = sext i32 %152 to i64, !dbg !2763
  %154 = getelementptr inbounds [4 x float], ptr %151, i64 0, i64 %153, !dbg !2763
  %155 = load float, ptr %154, align 4, !dbg !2763, !tbaa !177
  %156 = insertelement <8 x float> poison, float %155, i64 0, !dbg !2763
  %157 = shufflevector <8 x float> %156, <8 x float> poison, <8 x i32> zeroinitializer, !dbg !2763
  %158 = fmul contract <8 x float> %150, %157, !dbg !2765
  store <8 x float> %158, ptr %51, align 32, !dbg !2766, !tbaa !47
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %30) #25, !dbg !2767
  %159 = load <8 x i32>, ptr %50, align 32, !dbg !2768, !tbaa !47
  %160 = sitofp <8 x i32> %159 to <8 x float>, !dbg !2769
  store <8 x float> %160, ptr %52, align 32, !dbg !2770, !tbaa !47
  %161 = load <8 x float>, ptr %51, align 32, !dbg !2771, !tbaa !47
  %162 = load <8 x float>, ptr %52, align 32, !dbg !2772, !tbaa !47
  %163 = load ptr, ptr %35, align 8, !dbg !2773, !tbaa !2665
  %164 = load i32, ptr %48, align 4, !dbg !2774, !tbaa !8
  %165 = sext i32 %164 to i64, !dbg !2773
  %166 = getelementptr inbounds [2 x <8 x float>], ptr %163, i64 %165, !dbg !2773
  %167 = load i32, ptr %45, align 4, !dbg !2775, !tbaa !8
  %168 = sext i32 %167 to i64, !dbg !2773
  %169 = getelementptr inbounds [2 x <8 x float>], ptr %166, i64 0, i64 %168, !dbg !2773
  %170 = load <8 x float>, ptr %169, align 32, !dbg !2773, !tbaa !47
  %171 = call contract <8 x float> @llvm.fma.v8f32(<8 x float> %161, <8 x float> %162, <8 x float> %170), !dbg !2776
  %172 = load ptr, ptr %35, align 8, !dbg !2777, !tbaa !2665
  %173 = load i32, ptr %48, align 4, !dbg !2778, !tbaa !8
  %174 = sext i32 %173 to i64, !dbg !2777
  %175 = getelementptr inbounds [2 x <8 x float>], ptr %172, i64 %174, !dbg !2777
  %176 = load i32, ptr %45, align 4, !dbg !2779, !tbaa !8
  %177 = sext i32 %176 to i64, !dbg !2777
  %178 = getelementptr inbounds [2 x <8 x float>], ptr %175, i64 0, i64 %177, !dbg !2777
  store <8 x float> %171, ptr %178, align 32, !dbg !2780, !tbaa !47
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %30) #25, !dbg !2781
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %29) #25, !dbg !2781
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %28) #25, !dbg !2781
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %27) #25, !dbg !2781
  br label %179, !dbg !2781

179:                                              ; preds = %139
  %180 = load i32, ptr %48, align 4, !dbg !2782, !tbaa !8
  %181 = add nsw i32 %180, 1, !dbg !2782
  store i32 %181, ptr %48, align 4, !dbg !2782, !tbaa !8
  br label %114, !dbg !2737, !llvm.loop !2783

182:                                              ; preds = %117
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %25) #25, !dbg !2784
  br label %183, !dbg !2784

183:                                              ; preds = %182
  %184 = load i32, ptr %45, align 4, !dbg !2785, !tbaa !8
  %185 = add nsw i32 %184, 1, !dbg !2785
  store i32 %185, ptr %45, align 4, !dbg !2785, !tbaa !8
  br label %101, !dbg !2725, !llvm.loop !2786

186:                                              ; preds = %104
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !2787
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %22) #25, !dbg !2788
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %21) #25, !dbg !2788
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %20) #25, !dbg !2788
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %19) #25, !dbg !2788
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %18) #25, !dbg !2788
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %17) #25, !dbg !2788
  ret void, !dbg !2788
}

; Function Attrs: alwaysinline convergent mustprogress nounwind uwtable
define internal void @_ZL21iu4_bundle_single_accILi1EEvPKcS1_RA2_A2_KjRA4_S3_PA2_Dv8_fPKS8_RA4_KfRKDv8_i(ptr noalias noundef %0, ptr noalias noundef %1, ptr noundef nonnull align 4 dereferenceable(16) %2, ptr noundef nonnull align 4 dereferenceable(32) %3, ptr noundef %4, ptr noundef %5, ptr noundef nonnull align 4 dereferenceable(16) %6, ptr noundef nonnull align 32 dereferenceable(32) %7) #5 !dbg !2789 {
  %9 = alloca ptr, align 8, addrspace(5)
  %10 = alloca ptr, align 8, addrspace(5)
  %11 = alloca ptr, align 8, addrspace(5)
  %12 = alloca ptr, align 8, addrspace(5)
  %13 = alloca ptr, align 8, addrspace(5)
  %14 = alloca ptr, align 8, addrspace(5)
  %15 = alloca ptr, align 8, addrspace(5)
  %16 = alloca ptr, align 8, addrspace(5)
  %17 = alloca <2 x i32>, align 8, addrspace(5)
  %18 = alloca <2 x i32>, align 8, addrspace(5)
  %19 = alloca <2 x i32>, align 8, addrspace(5)
  %20 = alloca <2 x i32>, align 8, addrspace(5)
  %21 = alloca <2 x i32>, align 8, addrspace(5)
  %22 = alloca <2 x i32>, align 8, addrspace(5)
  %23 = alloca i32, align 4, addrspace(5)
  %24 = alloca i32, align 4, addrspace(5)
  %25 = alloca <2 x i32>, align 8, addrspace(5)
  %26 = alloca i32, align 4, addrspace(5)
  %27 = alloca <2 x i32>, align 8, addrspace(5)
  %28 = alloca <8 x i32>, align 32, addrspace(5)
  %29 = alloca <8 x float>, align 32, addrspace(5)
  %30 = alloca <8 x float>, align 32, addrspace(5)
  %31 = addrspacecast ptr addrspace(5) %9 to ptr
  %32 = addrspacecast ptr addrspace(5) %10 to ptr
  %33 = addrspacecast ptr addrspace(5) %11 to ptr
  %34 = addrspacecast ptr addrspace(5) %12 to ptr
  %35 = addrspacecast ptr addrspace(5) %13 to ptr
  %36 = addrspacecast ptr addrspace(5) %14 to ptr
  %37 = addrspacecast ptr addrspace(5) %15 to ptr
  %38 = addrspacecast ptr addrspace(5) %16 to ptr
  %39 = addrspacecast ptr addrspace(5) %17 to ptr
  %40 = addrspacecast ptr addrspace(5) %18 to ptr
  %41 = addrspacecast ptr addrspace(5) %19 to ptr
  %42 = addrspacecast ptr addrspace(5) %20 to ptr
  %43 = addrspacecast ptr addrspace(5) %21 to ptr
  %44 = addrspacecast ptr addrspace(5) %22 to ptr
  %45 = addrspacecast ptr addrspace(5) %23 to ptr
  %46 = addrspacecast ptr addrspace(5) %24 to ptr
  %47 = addrspacecast ptr addrspace(5) %25 to ptr
  %48 = addrspacecast ptr addrspace(5) %26 to ptr
  %49 = addrspacecast ptr addrspace(5) %27 to ptr
  %50 = addrspacecast ptr addrspace(5) %28 to ptr
  %51 = addrspacecast ptr addrspace(5) %29 to ptr
  %52 = addrspacecast ptr addrspace(5) %30 to ptr
  store ptr %0, ptr %31, align 8, !tbaa !28
  store ptr %1, ptr %32, align 8, !tbaa !28
  store ptr %2, ptr %33, align 8, !tbaa !2688
  store ptr %3, ptr %34, align 8, !tbaa !2688
  store ptr %4, ptr %35, align 8, !tbaa !2665
  store ptr %5, ptr %36, align 8, !tbaa !2665
  store ptr %6, ptr %37, align 8, !tbaa !128
  store ptr %7, ptr %38, align 8, !tbaa !2665
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %17) #25, !dbg !2790
  %53 = load ptr, ptr %31, align 8, !dbg !2791, !tbaa !28
  %54 = load ptr, ptr %33, align 8, !dbg !2792, !tbaa !2688, !nonnull !20, !align !2693
  %55 = getelementptr inbounds [2 x [2 x i32]], ptr %54, i64 0, i64 0, !dbg !2792
  %56 = getelementptr inbounds [2 x i32], ptr %55, i64 0, i64 1, !dbg !2792
  %57 = load i32, ptr %56, align 4, !dbg !2792, !tbaa !8
  %58 = zext i32 %57 to i64, !dbg !2793
  %59 = getelementptr inbounds nuw i8, ptr %53, i64 %58, !dbg !2793
  %60 = load <2 x i32>, ptr %59, align 8, !dbg !2793, !tbaa !47
  store <2 x i32> %60, ptr %39, align 8, !dbg !2794, !tbaa !47
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %18) #25, !dbg !2795
  %61 = load ptr, ptr %31, align 8, !dbg !2796, !tbaa !28
  %62 = load ptr, ptr %33, align 8, !dbg !2797, !tbaa !2688, !nonnull !20, !align !2693
  %63 = getelementptr inbounds [2 x [2 x i32]], ptr %62, i64 0, i64 1, !dbg !2797
  %64 = getelementptr inbounds [2 x i32], ptr %63, i64 0, i64 1, !dbg !2797
  %65 = load i32, ptr %64, align 4, !dbg !2797, !tbaa !8
  %66 = zext i32 %65 to i64, !dbg !2798
  %67 = getelementptr inbounds nuw i8, ptr %61, i64 %66, !dbg !2798
  %68 = load <2 x i32>, ptr %67, align 8, !dbg !2798, !tbaa !47
  store <2 x i32> %68, ptr %40, align 8, !dbg !2799, !tbaa !47
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %19) #25, !dbg !2800
  %69 = load ptr, ptr %32, align 8, !dbg !2801, !tbaa !28
  %70 = load ptr, ptr %34, align 8, !dbg !2802, !tbaa !2688, !nonnull !20, !align !2693
  %71 = getelementptr inbounds [4 x [2 x i32]], ptr %70, i64 0, i64 0, !dbg !2802
  %72 = getelementptr inbounds [2 x i32], ptr %71, i64 0, i64 1, !dbg !2802
  %73 = load i32, ptr %72, align 4, !dbg !2802, !tbaa !8
  %74 = zext i32 %73 to i64, !dbg !2803
  %75 = getelementptr inbounds nuw i8, ptr %69, i64 %74, !dbg !2803
  %76 = load <2 x i32>, ptr %75, align 8, !dbg !2803, !tbaa !47
  store <2 x i32> %76, ptr %41, align 8, !dbg !2804, !tbaa !47
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %20) #25, !dbg !2805
  %77 = load ptr, ptr %32, align 8, !dbg !2806, !tbaa !28
  %78 = load ptr, ptr %34, align 8, !dbg !2807, !tbaa !2688, !nonnull !20, !align !2693
  %79 = getelementptr inbounds [4 x [2 x i32]], ptr %78, i64 0, i64 1, !dbg !2807
  %80 = getelementptr inbounds [2 x i32], ptr %79, i64 0, i64 1, !dbg !2807
  %81 = load i32, ptr %80, align 4, !dbg !2807, !tbaa !8
  %82 = zext i32 %81 to i64, !dbg !2808
  %83 = getelementptr inbounds nuw i8, ptr %77, i64 %82, !dbg !2808
  %84 = load <2 x i32>, ptr %83, align 8, !dbg !2808, !tbaa !47
  store <2 x i32> %84, ptr %42, align 8, !dbg !2809, !tbaa !47
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %21) #25, !dbg !2810
  %85 = load ptr, ptr %32, align 8, !dbg !2811, !tbaa !28
  %86 = load ptr, ptr %34, align 8, !dbg !2812, !tbaa !2688, !nonnull !20, !align !2693
  %87 = getelementptr inbounds [4 x [2 x i32]], ptr %86, i64 0, i64 2, !dbg !2812
  %88 = getelementptr inbounds [2 x i32], ptr %87, i64 0, i64 1, !dbg !2812
  %89 = load i32, ptr %88, align 4, !dbg !2812, !tbaa !8
  %90 = zext i32 %89 to i64, !dbg !2813
  %91 = getelementptr inbounds nuw i8, ptr %85, i64 %90, !dbg !2813
  %92 = load <2 x i32>, ptr %91, align 8, !dbg !2813, !tbaa !47
  store <2 x i32> %92, ptr %43, align 8, !dbg !2814, !tbaa !47
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %22) #25, !dbg !2815
  %93 = load ptr, ptr %32, align 8, !dbg !2816, !tbaa !28
  %94 = load ptr, ptr %34, align 8, !dbg !2817, !tbaa !2688, !nonnull !20, !align !2693
  %95 = getelementptr inbounds [4 x [2 x i32]], ptr %94, i64 0, i64 3, !dbg !2817
  %96 = getelementptr inbounds [2 x i32], ptr %95, i64 0, i64 1, !dbg !2817
  %97 = load i32, ptr %96, align 4, !dbg !2817, !tbaa !8
  %98 = zext i32 %97 to i64, !dbg !2818
  %99 = getelementptr inbounds nuw i8, ptr %93, i64 %98, !dbg !2818
  %100 = load <2 x i32>, ptr %99, align 8, !dbg !2818, !tbaa !47
  store <2 x i32> %100, ptr %44, align 8, !dbg !2819, !tbaa !47
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %23) #25, !dbg !2820
  store i32 0, ptr %45, align 4, !dbg !2821, !tbaa !8
  br label %101, !dbg !2820

101:                                              ; preds = %183, %8
  %102 = load i32, ptr %45, align 4, !dbg !2822, !tbaa !8
  %103 = icmp slt i32 %102, 2, !dbg !2823
  br i1 %103, label %105, label %104, !dbg !2824

104:                                              ; preds = %101
  store i32 2, ptr %46, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %23) #25, !dbg !2824
  br label %186

105:                                              ; preds = %101
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %25) #25, !dbg !2825
  %106 = load i32, ptr %45, align 4, !dbg !2826, !tbaa !8
  %107 = icmp eq i32 %106, 0, !dbg !2827
  br i1 %107, label %108, label %110, !dbg !2828

108:                                              ; preds = %105
  %109 = load <2 x i32>, ptr %39, align 8, !dbg !2829, !tbaa !47
  br label %112, !dbg !2828

110:                                              ; preds = %105
  %111 = load <2 x i32>, ptr %40, align 8, !dbg !2830, !tbaa !47
  br label %112, !dbg !2828

112:                                              ; preds = %110, %108
  %113 = phi <2 x i32> [ %109, %108 ], [ %111, %110 ], !dbg !2828
  store <2 x i32> %113, ptr %47, align 8, !dbg !2831, !tbaa !47
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %26) #25, !dbg !2832
  store i32 0, ptr %48, align 4, !dbg !2833, !tbaa !8
  br label %114, !dbg !2832

114:                                              ; preds = %179, %112
  %115 = load i32, ptr %48, align 4, !dbg !2834, !tbaa !8
  %116 = icmp slt i32 %115, 4, !dbg !2835
  br i1 %116, label %118, label %117, !dbg !2836

117:                                              ; preds = %114
  store i32 5, ptr %46, align 4
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %26) #25, !dbg !2836
  br label %182

118:                                              ; preds = %114
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %27) #25, !dbg !2837
  %119 = load i32, ptr %48, align 4, !dbg !2838, !tbaa !8
  %120 = icmp eq i32 %119, 0, !dbg !2839
  br i1 %120, label %121, label %123, !dbg !2840

121:                                              ; preds = %118
  %122 = load <2 x i32>, ptr %41, align 8, !dbg !2841, !tbaa !47
  br label %139, !dbg !2840

123:                                              ; preds = %118
  %124 = load i32, ptr %48, align 4, !dbg !2842, !tbaa !8
  %125 = icmp eq i32 %124, 1, !dbg !2843
  br i1 %125, label %126, label %128, !dbg !2844

126:                                              ; preds = %123
  %127 = load <2 x i32>, ptr %42, align 8, !dbg !2845, !tbaa !47
  br label %137, !dbg !2844

128:                                              ; preds = %123
  %129 = load i32, ptr %48, align 4, !dbg !2846, !tbaa !8
  %130 = icmp eq i32 %129, 2, !dbg !2847
  br i1 %130, label %131, label %133, !dbg !2848

131:                                              ; preds = %128
  %132 = load <2 x i32>, ptr %43, align 8, !dbg !2849, !tbaa !47
  br label %135, !dbg !2848

133:                                              ; preds = %128
  %134 = load <2 x i32>, ptr %44, align 8, !dbg !2850, !tbaa !47
  br label %135, !dbg !2848

135:                                              ; preds = %133, %131
  %136 = phi <2 x i32> [ %132, %131 ], [ %134, %133 ], !dbg !2848
  br label %137, !dbg !2844

137:                                              ; preds = %135, %126
  %138 = phi <2 x i32> [ %127, %126 ], [ %136, %135 ], !dbg !2844
  br label %139, !dbg !2840

139:                                              ; preds = %137, %121
  %140 = phi <2 x i32> [ %122, %121 ], [ %138, %137 ], !dbg !2840
  store <2 x i32> %140, ptr %49, align 8, !dbg !2851, !tbaa !47
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %28) #25, !dbg !2852
  %141 = load <2 x i32>, ptr %47, align 8, !dbg !2853, !tbaa !47
  %142 = load <2 x i32>, ptr %49, align 8, !dbg !2854, !tbaa !47
  %143 = load ptr, ptr %38, align 8, !dbg !2855, !tbaa !2665, !nonnull !20, !align !2757
  %144 = load <8 x i32>, ptr %143, align 32, !dbg !2855, !tbaa !47
  %145 = call <8 x i32> @llvm.amdgcn.wmma.i32.16x16x32.iu4.v8i32.v2i32(i1 false, <2 x i32> %141, i1 true, <2 x i32> %142, <8 x i32> %144, i1 false), !dbg !2856
  store <8 x i32> %145, ptr %50, align 32, !dbg !2857, !tbaa !47
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %29) #25, !dbg !2858
  %146 = load ptr, ptr %36, align 8, !dbg !2859, !tbaa !2665
  %147 = load i32, ptr %45, align 4, !dbg !2860, !tbaa !8
  %148 = sext i32 %147 to i64, !dbg !2859
  %149 = getelementptr inbounds <8 x float>, ptr %146, i64 %148, !dbg !2859
  %150 = load <8 x float>, ptr %149, align 32, !dbg !2859, !tbaa !47
  %151 = load ptr, ptr %37, align 8, !dbg !2861, !tbaa !128, !nonnull !20, !align !2693
  %152 = load i32, ptr %48, align 4, !dbg !2862, !tbaa !8
  %153 = sext i32 %152 to i64, !dbg !2861
  %154 = getelementptr inbounds [4 x float], ptr %151, i64 0, i64 %153, !dbg !2861
  %155 = load float, ptr %154, align 4, !dbg !2861, !tbaa !177
  %156 = insertelement <8 x float> poison, float %155, i64 0, !dbg !2861
  %157 = shufflevector <8 x float> %156, <8 x float> poison, <8 x i32> zeroinitializer, !dbg !2861
  %158 = fmul contract <8 x float> %150, %157, !dbg !2863
  store <8 x float> %158, ptr %51, align 32, !dbg !2864, !tbaa !47
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %30) #25, !dbg !2865
  %159 = load <8 x i32>, ptr %50, align 32, !dbg !2866, !tbaa !47
  %160 = sitofp <8 x i32> %159 to <8 x float>, !dbg !2867
  store <8 x float> %160, ptr %52, align 32, !dbg !2868, !tbaa !47
  %161 = load <8 x float>, ptr %51, align 32, !dbg !2869, !tbaa !47
  %162 = load <8 x float>, ptr %52, align 32, !dbg !2870, !tbaa !47
  %163 = load ptr, ptr %35, align 8, !dbg !2871, !tbaa !2665
  %164 = load i32, ptr %48, align 4, !dbg !2872, !tbaa !8
  %165 = sext i32 %164 to i64, !dbg !2871
  %166 = getelementptr inbounds [2 x <8 x float>], ptr %163, i64 %165, !dbg !2871
  %167 = load i32, ptr %45, align 4, !dbg !2873, !tbaa !8
  %168 = sext i32 %167 to i64, !dbg !2871
  %169 = getelementptr inbounds [2 x <8 x float>], ptr %166, i64 0, i64 %168, !dbg !2871
  %170 = load <8 x float>, ptr %169, align 32, !dbg !2871, !tbaa !47
  %171 = call contract <8 x float> @llvm.fma.v8f32(<8 x float> %161, <8 x float> %162, <8 x float> %170), !dbg !2874
  %172 = load ptr, ptr %35, align 8, !dbg !2875, !tbaa !2665
  %173 = load i32, ptr %48, align 4, !dbg !2876, !tbaa !8
  %174 = sext i32 %173 to i64, !dbg !2875
  %175 = getelementptr inbounds [2 x <8 x float>], ptr %172, i64 %174, !dbg !2875
  %176 = load i32, ptr %45, align 4, !dbg !2877, !tbaa !8
  %177 = sext i32 %176 to i64, !dbg !2875
  %178 = getelementptr inbounds [2 x <8 x float>], ptr %175, i64 0, i64 %177, !dbg !2875
  store <8 x float> %171, ptr %178, align 32, !dbg !2878, !tbaa !47
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %30) #25, !dbg !2879
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %29) #25, !dbg !2879
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %28) #25, !dbg !2879
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %27) #25, !dbg !2879
  br label %179, !dbg !2879

179:                                              ; preds = %139
  %180 = load i32, ptr %48, align 4, !dbg !2880, !tbaa !8
  %181 = add nsw i32 %180, 1, !dbg !2880
  store i32 %181, ptr %48, align 4, !dbg !2880, !tbaa !8
  br label %114, !dbg !2836, !llvm.loop !2881

182:                                              ; preds = %117
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %25) #25, !dbg !2882
  br label %183, !dbg !2882

183:                                              ; preds = %182
  %184 = load i32, ptr %45, align 4, !dbg !2883, !tbaa !8
  %185 = add nsw i32 %184, 1, !dbg !2883
  store i32 %185, ptr %45, align 4, !dbg !2883, !tbaa !8
  br label %101, !dbg !2824, !llvm.loop !2884

186:                                              ; preds = %104
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !2885
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %22) #25, !dbg !2886
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %21) #25, !dbg !2886
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %20) #25, !dbg !2886
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %19) #25, !dbg !2886
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %18) #25, !dbg !2886
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %17) #25, !dbg !2886
  ret void, !dbg !2886
}

; Function Attrs: alwaysinline convergent mustprogress nounwind uwtable
define internal noundef float @_ZL9__fmul_rnff(float noundef %0, float noundef %1) #5 !dbg !2887 {
  %3 = alloca float, align 4, addrspace(5)
  %4 = alloca float, align 4, addrspace(5)
  %5 = addrspacecast ptr addrspace(5) %3 to ptr
  %6 = addrspacecast ptr addrspace(5) %4 to ptr
  store float %0, ptr %5, align 4, !tbaa !177
  store float %1, ptr %6, align 4, !tbaa !177
  %7 = load float, ptr %5, align 4, !dbg !2888, !tbaa !177
  %8 = load float, ptr %6, align 4, !dbg !2889, !tbaa !177
  %9 = fmul contract float %7, %8, !dbg !2890
  ret float %9, !dbg !2891
}

; Function Attrs: convergent mustprogress nounwind uwtable
define linkonce_odr hidden i16 @_ZNK6__halfcv10__half_rawEv(ptr noundef nonnull align 2 dereferenceable(2) %0) #8 comdat align 2 !dbg !2892 {
  %2 = alloca %struct.__half_raw, align 2, addrspace(5)
  %3 = alloca ptr, align 8, addrspace(5)
  %4 = addrspacecast ptr addrspace(5) %3 to ptr
  store ptr %0, ptr %4, align 8, !tbaa !2893
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr inbounds nuw %struct.__half_raw, ptr addrspace(5) %2, i32 0, i32 0, !dbg !2895
  %7 = getelementptr inbounds nuw %struct.__half, ptr %5, i32 0, i32 0, !dbg !2896
  %8 = load half, ptr %7, align 2, !dbg !2896, !tbaa !47
  store half %8, ptr addrspace(5) %6, align 2, !dbg !2896, !tbaa !47
  %9 = getelementptr inbounds nuw %struct.__half_raw, ptr addrspace(5) %2, i32 0, i32 0, !dbg !2897
  %10 = getelementptr inbounds nuw %union.anon.1, ptr addrspace(5) %9, i32 0, i32 0, !dbg !2897
  %11 = load i16, ptr addrspace(5) %10, align 2, !dbg !2897
  ret i16 %11, !dbg !2897
}

; Function Attrs: convergent mustprogress nounwind uwtable
define linkonce_odr hidden void @_ZN6__halfC2ERK10__half_raw(ptr noundef nonnull align 2 dereferenceable(2) %0, ptr noundef nonnull align 2 dereferenceable(2) %1) unnamed_addr #8 comdat align 2 !dbg !2898 {
  %3 = alloca ptr, align 8, addrspace(5)
  %4 = alloca ptr, align 8, addrspace(5)
  %5 = addrspacecast ptr addrspace(5) %3 to ptr
  %6 = addrspacecast ptr addrspace(5) %4 to ptr
  store ptr %0, ptr %5, align 8, !tbaa !2893
  store ptr %1, ptr %6, align 8, !tbaa !2899
  %7 = load ptr, ptr %5, align 8
  %8 = getelementptr inbounds nuw %struct.__half, ptr %7, i32 0, i32 0, !dbg !2901
  %9 = load ptr, ptr %6, align 8, !dbg !2902, !tbaa !2899, !nonnull !20, !align !2903
  %10 = getelementptr inbounds nuw %struct.__half_raw, ptr %9, i32 0, i32 0, !dbg !2904
  %11 = load i16, ptr %10, align 2, !dbg !2904, !tbaa !47
  store i16 %11, ptr %8, align 2, !dbg !2901, !tbaa !47
  ret void, !dbg !2905
}

; Function Attrs: convergent inlinehint mustprogress nounwind uwtable
define internal void @_ZL9__barrieri(i32 noundef %0) #6 !dbg !2906 {
  %2 = alloca i32, align 4, addrspace(5)
  %3 = addrspacecast ptr addrspace(5) %2 to ptr
  store i32 %0, ptr %3, align 4, !tbaa !8
  %4 = load i32, ptr %3, align 4, !dbg !2907, !tbaa !8
  call void @_ZL20__work_group_barrierj(i32 noundef %4) #26, !dbg !2908
  ret void, !dbg !2909
}

; Function Attrs: convergent inlinehint mustprogress nounwind uwtable
define internal void @_ZL20__work_group_barrierj(i32 noundef %0) #6 !dbg !2910 {
  %2 = alloca i32, align 4, addrspace(5)
  %3 = addrspacecast ptr addrspace(5) %2 to ptr
  store i32 %0, ptr %3, align 4, !tbaa !8
  %4 = load i32, ptr %3, align 4, !dbg !2911, !tbaa !8
  %5 = icmp eq i32 %4, 3, !dbg !2912
  br i1 %5, label %6, label %7, !dbg !2912

6:                                                ; preds = %1
  fence syncscope("workgroup") release, !dbg !2913
  call void @llvm.amdgcn.s.barrier(), !dbg !2914
  fence syncscope("workgroup") acquire, !dbg !2915
  br label %20, !dbg !2916

7:                                                ; preds = %1
  %8 = load i32, ptr %3, align 4, !dbg !2917, !tbaa !8
  %9 = and i32 %8, 2, !dbg !2918
  %10 = icmp ne i32 %9, 0, !dbg !2917
  br i1 %10, label %11, label %12, !dbg !2917

11:                                               ; preds = %7
  fence syncscope("workgroup") release, !dbg !2919, !mmra !2920
  call void @llvm.amdgcn.s.barrier(), !dbg !2921
  fence syncscope("workgroup") acquire, !dbg !2922, !mmra !2920
  br label %19, !dbg !2923

12:                                               ; preds = %7
  %13 = load i32, ptr %3, align 4, !dbg !2924, !tbaa !8
  %14 = and i32 %13, 1, !dbg !2925
  %15 = icmp ne i32 %14, 0, !dbg !2924
  br i1 %15, label %16, label %17, !dbg !2924

16:                                               ; preds = %12
  fence syncscope("workgroup") release, !dbg !2926, !mmra !2927
  call void @llvm.amdgcn.s.barrier(), !dbg !2928
  fence syncscope("workgroup") acquire, !dbg !2929, !mmra !2927
  br label %18, !dbg !2930

17:                                               ; preds = %12
  call void @llvm.amdgcn.s.barrier(), !dbg !2931
  br label %18

18:                                               ; preds = %17, %16
  br label %19

19:                                               ; preds = %18, %11
  br label %20

20:                                               ; preds = %19, %6
  ret void, !dbg !2932
}

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #14

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare <8 x i32> @llvm.amdgcn.wmma.i32.16x16x32.iu4.v8i32.v2i32(i1 immarg, <2 x i32>, i1 immarg, <2 x i32>, <8 x i32>, i1 immarg) #15

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x float> @llvm.fma.v8f32(<8 x float>, <8 x float>, <8 x float>) #9

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #14

; Function Attrs: convergent mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define internal range(i64 0, 1024) i64 @__ockl_get_local_id(i32 noundef %0) #16 {
  switch i32 %0, label %8 [
    i32 0, label %2
    i32 1, label %4
    i32 2, label %6
  ]

2:                                                ; preds = %1
  %3 = tail call i32 @llvm.amdgcn.workitem.id.x()
  br label %8

4:                                                ; preds = %1
  %5 = tail call i32 @llvm.amdgcn.workitem.id.y()
  br label %8

6:                                                ; preds = %1
  %7 = tail call i32 @llvm.amdgcn.workitem.id.z()
  br label %8

8:                                                ; preds = %6, %4, %2, %1
  %9 = phi i32 [ %7, %6 ], [ %3, %2 ], [ %5, %4 ], [ 0, %1 ]
  %10 = zext nneg i32 %9 to i64
  ret i64 %10
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #13

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.y() #13

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.z() #13

; Function Attrs: convergent norecurse nounwind uwtable
define internal i64 @__ockl_fprintf_stderr_begin() #17 {
  %1 = tail call <2 x i64> @__ockl_hostcall_preview(i32 noundef 2, i64 noundef 33, i64 noundef 1, i64 noundef 0, i64 noundef 0, i64 noundef 0, i64 noundef 0, i64 noundef 0, i64 noundef 0) #28
  %2 = extractelement <2 x i64> %1, i64 0
  ret i64 %2
}

; Function Attrs: cold convergent norecurse nounwind uwtable
define internal <2 x i64> @__ockl_hostcall_preview(i32 noundef %0, i64 noundef %1, i64 noundef %2, i64 noundef %3, i64 noundef %4, i64 noundef %5, i64 noundef %6, i64 noundef %7, i64 noundef %8) local_unnamed_addr #18 {
  %10 = load i32, ptr addrspace(4) @__oclc_ABI_version, align 4, !tbaa !12
  %11 = icmp slt i32 %10, 500
  %12 = tail call dereferenceable(256) ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %13 = select i1 %11, i64 24, i64 80
  %14 = getelementptr inbounds nuw i8, ptr addrspace(4) %12, i64 %13
  %15 = load i64, ptr addrspace(4) %14, align 8, !tbaa !2933
  %16 = inttoptr i64 %15 to ptr addrspace(1)
  %17 = addrspacecast ptr addrspace(1) %16 to ptr
  %18 = tail call <2 x i64> @__ockl_hostcall_internal(ptr noundef %17, i32 noundef %0, i64 noundef %1, i64 noundef %2, i64 noundef %3, i64 noundef %4, i64 noundef %5, i64 noundef %6, i64 noundef %7, i64 noundef %8) #29
  ret <2 x i64> %18
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef align 4 ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr() #13

; Function Attrs: convergent norecurse nounwind uwtable
define internal <2 x i64> @__ockl_hostcall_internal(ptr noundef captures(none) %0, i32 noundef %1, i64 noundef %2, i64 noundef %3, i64 noundef %4, i64 noundef %5, i64 noundef %6, i64 noundef %7, i64 noundef %8, i64 noundef %9) local_unnamed_addr #17 {
  %11 = tail call i32 @__ockl_lane_u32() #28
  %12 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %11)
  %13 = addrspacecast ptr %0 to ptr addrspace(1)
  %14 = icmp eq i32 %11, %12
  br i1 %14, label %15, label %37

15:                                               ; preds = %10
  %16 = getelementptr inbounds nuw i8, ptr addrspace(1) %13, i64 24
  %17 = load atomic i64, ptr addrspace(1) %16 syncscope("one-as") acquire, align 8
  %18 = getelementptr i8, ptr addrspace(1) %13, i64 40
  %19 = load ptr addrspace(1), ptr addrspace(1) %13, align 8, !tbaa !2935
  %20 = load i64, ptr addrspace(1) %18, align 8, !tbaa !2939
  %21 = and i64 %20, %17
  %22 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %19, i64 %21
  %23 = load atomic i64, ptr addrspace(1) %22 syncscope("one-as") monotonic, align 8
  %24 = cmpxchg ptr addrspace(1) %16, i64 %17, i64 %23 syncscope("one-as") acquire monotonic, align 8
  %25 = extractvalue { i64, i1 } %24, 1
  %26 = extractvalue { i64, i1 } %24, 0
  br i1 %25, label %37, label %27

27:                                               ; preds = %27, %15
  %28 = phi i64 [ %36, %27 ], [ %26, %15 ]
  tail call void @llvm.amdgcn.s.sleep(i32 1)
  %29 = load ptr addrspace(1), ptr addrspace(1) %13, align 8, !tbaa !2935
  %30 = load i64, ptr addrspace(1) %18, align 8, !tbaa !2939
  %31 = and i64 %30, %28
  %32 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %29, i64 %31
  %33 = load atomic i64, ptr addrspace(1) %32 syncscope("one-as") monotonic, align 8
  %34 = cmpxchg ptr addrspace(1) %16, i64 %28, i64 %33 syncscope("one-as") acquire monotonic, align 8
  %35 = extractvalue { i64, i1 } %34, 1
  %36 = extractvalue { i64, i1 } %34, 0
  br i1 %35, label %37, label %27

37:                                               ; preds = %27, %15, %10
  %38 = phi i64 [ 0, %10 ], [ %26, %15 ], [ %36, %27 ]
  %39 = tail call i64 @llvm.amdgcn.readfirstlane.i64(i64 %38)
  %40 = load ptr addrspace(1), ptr addrspace(1) %13, align 8, !tbaa !2935
  %41 = getelementptr i8, ptr addrspace(1) %13, i64 40
  %42 = load i64, ptr addrspace(1) %41, align 8, !tbaa !2939
  %43 = and i64 %42, %39
  %44 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %40, i64 %43
  %45 = getelementptr i8, ptr addrspace(1) %13, i64 8
  %46 = load ptr addrspace(1), ptr addrspace(1) %45, align 8, !tbaa !2940
  %47 = getelementptr inbounds nuw [4096 x i8], ptr addrspace(1) %46, i64 %43
  %48 = tail call i64 @llvm.amdgcn.ballot.i64(i1 true)
  br i1 %14, label %49, label %53

49:                                               ; preds = %37
  %50 = getelementptr inbounds nuw i8, ptr addrspace(1) %44, i64 16
  %51 = getelementptr inbounds nuw i8, ptr addrspace(1) %44, i64 8
  %52 = getelementptr inbounds nuw i8, ptr addrspace(1) %44, i64 20
  store i32 %1, ptr addrspace(1) %50, align 8, !tbaa !2941
  store i64 %48, ptr addrspace(1) %51, align 8, !tbaa !2943
  store i32 1, ptr addrspace(1) %52, align 4, !tbaa !2944
  br label %53

53:                                               ; preds = %49, %37
  %54 = zext i32 %11 to i64
  %55 = getelementptr inbounds nuw [64 x i8], ptr addrspace(1) %47, i64 %54
  store i64 %2, ptr addrspace(1) %55, align 8, !tbaa !2933
  %56 = getelementptr inbounds nuw i8, ptr addrspace(1) %55, i64 8
  store i64 %3, ptr addrspace(1) %56, align 8, !tbaa !2933
  %57 = getelementptr inbounds nuw i8, ptr addrspace(1) %55, i64 16
  store i64 %4, ptr addrspace(1) %57, align 8, !tbaa !2933
  %58 = getelementptr inbounds nuw i8, ptr addrspace(1) %55, i64 24
  store i64 %5, ptr addrspace(1) %58, align 8, !tbaa !2933
  %59 = getelementptr inbounds nuw i8, ptr addrspace(1) %55, i64 32
  store i64 %6, ptr addrspace(1) %59, align 8, !tbaa !2933
  %60 = getelementptr inbounds nuw i8, ptr addrspace(1) %55, i64 40
  store i64 %7, ptr addrspace(1) %60, align 8, !tbaa !2933
  %61 = getelementptr inbounds nuw i8, ptr addrspace(1) %55, i64 48
  store i64 %8, ptr addrspace(1) %61, align 8, !tbaa !2933
  %62 = getelementptr inbounds nuw i8, ptr addrspace(1) %55, i64 56
  store i64 %9, ptr addrspace(1) %62, align 8, !tbaa !2933
  br i1 %14, label %63, label %79

63:                                               ; preds = %53
  %64 = getelementptr inbounds nuw i8, ptr addrspace(1) %13, i64 32
  %65 = load atomic i64, ptr addrspace(1) %64 syncscope("one-as") monotonic, align 8
  %66 = load i64, ptr addrspace(1) %41, align 8, !tbaa !2939
  %67 = and i64 %66, %39
  %68 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %40, i64 %67
  store i64 %65, ptr addrspace(1) %68, align 8, !tbaa !2945
  %69 = cmpxchg ptr addrspace(1) %64, i64 %65, i64 %39 syncscope("one-as") release monotonic, align 8
  %70 = extractvalue { i64, i1 } %69, 1
  br i1 %70, label %76, label %71

71:                                               ; preds = %71, %63
  %72 = phi { i64, i1 } [ %74, %71 ], [ %69, %63 ]
  %73 = extractvalue { i64, i1 } %72, 0
  tail call void @llvm.amdgcn.s.sleep(i32 1)
  store i64 %73, ptr addrspace(1) %68, align 8, !tbaa !2945
  %74 = cmpxchg ptr addrspace(1) %64, i64 %73, i64 %39 syncscope("one-as") release monotonic, align 8
  %75 = extractvalue { i64, i1 } %74, 1
  br i1 %75, label %76, label %71

76:                                               ; preds = %71, %63
  %77 = getelementptr inbounds nuw i8, ptr addrspace(1) %13, i64 16
  %78 = load i64, ptr addrspace(1) %77, align 8
  tail call void @__ockl_hsa_signal_add(i64 %78, i64 noundef 1, i32 noundef 3) #28
  br label %79

79:                                               ; preds = %76, %53
  %80 = getelementptr inbounds nuw i8, ptr addrspace(1) %44, i64 20
  br label %81

81:                                               ; preds = %89, %79
  br i1 %14, label %82, label %85

82:                                               ; preds = %81
  %83 = load atomic i32, ptr addrspace(1) %80 syncscope("one-as") acquire, align 4
  %84 = and i32 %83, 1
  br label %85

85:                                               ; preds = %82, %81
  %86 = phi i32 [ %84, %82 ], [ 1, %81 ]
  %87 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %86)
  %88 = icmp eq i32 %87, 0
  br i1 %88, label %90, label %89

89:                                               ; preds = %85
  tail call void @llvm.amdgcn.s.sleep(i32 1)
  br label %81

90:                                               ; preds = %85
  %91 = load i64, ptr addrspace(1) %55, align 8, !tbaa !2933
  %92 = load i64, ptr addrspace(1) %56, align 8, !tbaa !2933
  br i1 %14, label %93, label %111

93:                                               ; preds = %90
  %94 = load i64, ptr addrspace(1) %41, align 8, !tbaa !2939
  %95 = add i64 %94, 1
  %96 = add i64 %95, %39
  %97 = icmp eq i64 %96, 0
  %98 = select i1 %97, i64 %95, i64 %96
  %99 = getelementptr inbounds nuw i8, ptr addrspace(1) %13, i64 24
  %100 = load atomic i64, ptr addrspace(1) %99 syncscope("one-as") monotonic, align 8
  %101 = load ptr addrspace(1), ptr addrspace(1) %13, align 8, !tbaa !2935
  %102 = and i64 %98, %94
  %103 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %101, i64 %102
  store i64 %100, ptr addrspace(1) %103, align 8, !tbaa !2945
  %104 = cmpxchg ptr addrspace(1) %99, i64 %100, i64 %98 syncscope("one-as") release monotonic, align 8
  %105 = extractvalue { i64, i1 } %104, 1
  br i1 %105, label %111, label %106

106:                                              ; preds = %106, %93
  %107 = phi { i64, i1 } [ %109, %106 ], [ %104, %93 ]
  %108 = extractvalue { i64, i1 } %107, 0
  tail call void @llvm.amdgcn.s.sleep(i32 1)
  store i64 %108, ptr addrspace(1) %103, align 8, !tbaa !2945
  %109 = cmpxchg ptr addrspace(1) %99, i64 %108, i64 %98 syncscope("one-as") release monotonic, align 8
  %110 = extractvalue { i64, i1 } %109, 1
  br i1 %110, label %111, label %106

111:                                              ; preds = %106, %93, %90
  %112 = insertelement <2 x i64> poison, i64 %91, i64 0
  %113 = insertelement <2 x i64> %112, i64 %92, i64 1
  ret <2 x i64> %113
}

; Function Attrs: alwaysinline convergent mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define internal range(i32 0, 65) i32 @__ockl_lane_u32() local_unnamed_addr #19 {
  %1 = tail call range(i32 0, 33) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %2 = tail call range(i32 0, 65) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1)
  ret i32 %2
}

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #15

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.amdgcn.s.sleep(i32 immarg) #20

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i64 @llvm.amdgcn.readfirstlane.i64(i64) #15

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i64 @llvm.amdgcn.ballot.i64(i1) #15

; Function Attrs: convergent mustprogress norecurse nounwind willreturn uwtable
define internal void @__ockl_hsa_signal_add(i64 %0, i64 noundef %1, i32 noundef %2) local_unnamed_addr #21 {
  %4 = inttoptr i64 %0 to ptr addrspace(1)
  %5 = getelementptr inbounds nuw i8, ptr addrspace(1) %4, i64 8
  switch i32 %2, label %6 [
    i32 1, label %8
    i32 2, label %8
    i32 3, label %10
    i32 4, label %12
    i32 5, label %14
  ]

6:                                                ; preds = %3
  %7 = atomicrmw add ptr addrspace(1) %5, i64 %1 syncscope("one-as") monotonic, align 8, !amdgpu.no.fine.grained.memory !20, !amdgpu.no.remote.memory !20
  br label %16

8:                                                ; preds = %3, %3
  %9 = atomicrmw add ptr addrspace(1) %5, i64 %1 syncscope("one-as") acquire, align 8, !amdgpu.no.fine.grained.memory !20, !amdgpu.no.remote.memory !20
  br label %16

10:                                               ; preds = %3
  %11 = atomicrmw add ptr addrspace(1) %5, i64 %1 syncscope("one-as") release, align 8, !amdgpu.no.fine.grained.memory !20, !amdgpu.no.remote.memory !20
  br label %16

12:                                               ; preds = %3
  %13 = atomicrmw add ptr addrspace(1) %5, i64 %1 syncscope("one-as") acq_rel, align 8, !amdgpu.no.fine.grained.memory !20, !amdgpu.no.remote.memory !20
  br label %16

14:                                               ; preds = %3
  %15 = atomicrmw add ptr addrspace(1) %5, i64 %1 seq_cst, align 8, !amdgpu.no.fine.grained.memory !20, !amdgpu.no.remote.memory !20
  br label %16

16:                                               ; preds = %14, %12, %10, %8, %6
  %17 = getelementptr inbounds nuw i8, ptr addrspace(1) %4, i64 16
  %18 = load i64, ptr addrspace(1) %17, align 16, !tbaa !2946
  %19 = icmp eq i64 %18, 0
  br i1 %19, label %33, label %20

20:                                               ; preds = %16
  %21 = inttoptr i64 %18 to ptr addrspace(1)
  %22 = getelementptr inbounds nuw i8, ptr addrspace(1) %4, i64 24
  %23 = load i32, ptr addrspace(1) %22, align 8, !tbaa !2948
  %24 = zext i32 %23 to i64
  store atomic i64 %24, ptr addrspace(1) %21 syncscope("one-as") release, align 8
  %25 = load i32, ptr addrspace(4) @__oclc_ISA_version, align 4, !tbaa !12
  %26 = icmp slt i32 %25, 9000
  %27 = add i32 %25, -11000
  %28 = icmp ult i32 %27, -1000
  %29 = select i1 %28, i32 16777215, i32 8388607
  %30 = select i1 %26, i32 255, i32 %29
  %31 = and i32 %30, %23
  %32 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %31)
  tail call void @llvm.amdgcn.s.sendmsg(i32 1, i32 %32)
  br label %33

33:                                               ; preds = %20, %16
  ret void
}

; Function Attrs: nocallback nounwind willreturn
declare void @llvm.amdgcn.s.sendmsg(i32 immarg, i32) #22

; Function Attrs: convergent norecurse nounwind uwtable
define internal i64 @__ockl_fprintf_append_args(i64 noundef %0, i32 noundef %1, i64 noundef %2, i64 noundef %3, i64 noundef %4, i64 noundef %5, i64 noundef %6, i64 noundef %7, i64 noundef %8, i32 noundef %9) #17 {
  %11 = icmp eq i32 %9, 0
  %12 = or i64 %0, 2
  %13 = select i1 %11, i64 %0, i64 %12
  %14 = and i64 %13, -225
  %15 = zext i32 %1 to i64
  %16 = shl nuw nsw i64 %15, 5
  %17 = or i64 %14, %16
  %18 = tail call <2 x i64> @__ockl_hostcall_preview(i32 noundef 2, i64 noundef %17, i64 noundef %2, i64 noundef %3, i64 noundef %4, i64 noundef %5, i64 noundef %6, i64 noundef %7, i64 noundef %8) #28
  %19 = extractelement <2 x i64> %18, i64 0
  ret i64 %19
}

; Function Attrs: convergent norecurse nounwind uwtable
define internal i64 @__ockl_fprintf_append_string_n(i64 noundef %0, ptr noundef readonly %1, i64 noundef %2, i32 noundef %3) #17 {
  %5 = icmp eq i32 %3, 0
  %6 = or i64 %0, 2
  %7 = select i1 %5, i64 %0, i64 %6
  %8 = icmp eq ptr %1, null
  br i1 %8, label %9, label %13

9:                                                ; preds = %4
  %10 = and i64 %7, -225
  %11 = or disjoint i64 %10, 32
  %12 = tail call <2 x i64> @__ockl_hostcall_preview(i32 noundef 2, i64 noundef %11, i64 noundef 0, i64 noundef 0, i64 noundef 0, i64 noundef 0, i64 noundef 0, i64 noundef 0, i64 noundef 0) #28
  br label %452

13:                                               ; preds = %4
  %14 = and i64 %7, 2
  %15 = and i64 %7, -3
  %16 = insertelement <2 x i64> <i64 poison, i64 0>, i64 %15, i64 0
  br label %17

17:                                               ; preds = %440, %13
  %18 = phi i64 [ %2, %13 ], [ %449, %440 ]
  %19 = phi ptr [ %1, %13 ], [ %450, %440 ]
  %20 = phi <2 x i64> [ %16, %13 ], [ %448, %440 ]
  %21 = icmp ugt i64 %18, 56
  %22 = extractelement <2 x i64> %20, i64 0
  %23 = tail call i64 @llvm.umin.i64(i64 %18, i64 56)
  %24 = trunc nuw nsw i64 %23 to i32
  %25 = select i1 %21, i64 0, i64 %14
  %26 = icmp ugt i64 %18, 7
  br i1 %26, label %29, label %27

27:                                               ; preds = %17
  %28 = icmp eq i64 %18, 0
  br i1 %28, label %82, label %69

29:                                               ; preds = %17
  %30 = load i8, ptr %19, align 1, !tbaa !2949
  %31 = zext i8 %30 to i64
  %32 = getelementptr inbounds nuw i8, ptr %19, i64 1
  %33 = load i8, ptr %32, align 1, !tbaa !2949
  %34 = zext i8 %33 to i64
  %35 = shl nuw nsw i64 %34, 8
  %36 = or disjoint i64 %35, %31
  %37 = getelementptr inbounds nuw i8, ptr %19, i64 2
  %38 = load i8, ptr %37, align 1, !tbaa !2949
  %39 = zext i8 %38 to i64
  %40 = shl nuw nsw i64 %39, 16
  %41 = or disjoint i64 %36, %40
  %42 = getelementptr inbounds nuw i8, ptr %19, i64 3
  %43 = load i8, ptr %42, align 1, !tbaa !2949
  %44 = zext i8 %43 to i64
  %45 = shl nuw nsw i64 %44, 24
  %46 = or disjoint i64 %41, %45
  %47 = getelementptr inbounds nuw i8, ptr %19, i64 4
  %48 = load i8, ptr %47, align 1, !tbaa !2949
  %49 = zext i8 %48 to i64
  %50 = shl nuw nsw i64 %49, 32
  %51 = or disjoint i64 %46, %50
  %52 = getelementptr inbounds nuw i8, ptr %19, i64 5
  %53 = load i8, ptr %52, align 1, !tbaa !2949
  %54 = zext i8 %53 to i64
  %55 = shl nuw nsw i64 %54, 40
  %56 = or i64 %51, %55
  %57 = getelementptr inbounds nuw i8, ptr %19, i64 6
  %58 = load i8, ptr %57, align 1, !tbaa !2949
  %59 = zext i8 %58 to i64
  %60 = shl nuw nsw i64 %59, 48
  %61 = or i64 %56, %60
  %62 = getelementptr inbounds nuw i8, ptr %19, i64 7
  %63 = load i8, ptr %62, align 1, !tbaa !2949
  %64 = zext i8 %63 to i64
  %65 = shl nuw i64 %64, 56
  %66 = or i64 %61, %65
  %67 = add nsw i32 %24, -8
  %68 = getelementptr inbounds nuw i8, ptr %19, i64 8
  br label %82

69:                                               ; preds = %69, %27
  %70 = phi i32 [ %80, %69 ], [ 0, %27 ]
  %71 = phi i64 [ %79, %69 ], [ 0, %27 ]
  %72 = zext nneg i32 %70 to i64
  %73 = getelementptr inbounds nuw i8, ptr %19, i64 %72
  %74 = load i8, ptr %73, align 1, !tbaa !2949
  %75 = zext i8 %74 to i64
  %76 = shl i32 %70, 3
  %77 = zext nneg i32 %76 to i64
  %78 = shl nuw i64 %75, %77
  %79 = or i64 %78, %71
  %80 = add nuw nsw i32 %70, 1
  %81 = icmp eq i32 %80, %24
  br i1 %81, label %82, label %69

82:                                               ; preds = %69, %29, %27
  %83 = phi ptr [ %68, %29 ], [ %19, %27 ], [ %19, %69 ]
  %84 = phi i32 [ %67, %29 ], [ 0, %27 ], [ 0, %69 ]
  %85 = phi i64 [ %66, %29 ], [ 0, %27 ], [ %79, %69 ]
  %86 = icmp ugt i32 %84, 7
  br i1 %86, label %89, label %87

87:                                               ; preds = %82
  %88 = icmp eq i32 %84, 0
  br i1 %88, label %142, label %129

89:                                               ; preds = %82
  %90 = load i8, ptr %83, align 1, !tbaa !2949
  %91 = zext i8 %90 to i64
  %92 = getelementptr inbounds nuw i8, ptr %83, i64 1
  %93 = load i8, ptr %92, align 1, !tbaa !2949
  %94 = zext i8 %93 to i64
  %95 = shl nuw nsw i64 %94, 8
  %96 = or disjoint i64 %95, %91
  %97 = getelementptr inbounds nuw i8, ptr %83, i64 2
  %98 = load i8, ptr %97, align 1, !tbaa !2949
  %99 = zext i8 %98 to i64
  %100 = shl nuw nsw i64 %99, 16
  %101 = or disjoint i64 %96, %100
  %102 = getelementptr inbounds nuw i8, ptr %83, i64 3
  %103 = load i8, ptr %102, align 1, !tbaa !2949
  %104 = zext i8 %103 to i64
  %105 = shl nuw nsw i64 %104, 24
  %106 = or disjoint i64 %101, %105
  %107 = getelementptr inbounds nuw i8, ptr %83, i64 4
  %108 = load i8, ptr %107, align 1, !tbaa !2949
  %109 = zext i8 %108 to i64
  %110 = shl nuw nsw i64 %109, 32
  %111 = or disjoint i64 %106, %110
  %112 = getelementptr inbounds nuw i8, ptr %83, i64 5
  %113 = load i8, ptr %112, align 1, !tbaa !2949
  %114 = zext i8 %113 to i64
  %115 = shl nuw nsw i64 %114, 40
  %116 = or i64 %111, %115
  %117 = getelementptr inbounds nuw i8, ptr %83, i64 6
  %118 = load i8, ptr %117, align 1, !tbaa !2949
  %119 = zext i8 %118 to i64
  %120 = shl nuw nsw i64 %119, 48
  %121 = or i64 %116, %120
  %122 = getelementptr inbounds nuw i8, ptr %83, i64 7
  %123 = load i8, ptr %122, align 1, !tbaa !2949
  %124 = zext i8 %123 to i64
  %125 = shl nuw i64 %124, 56
  %126 = or i64 %121, %125
  %127 = add nsw i32 %84, -8
  %128 = getelementptr inbounds nuw i8, ptr %83, i64 8
  br label %142

129:                                              ; preds = %129, %87
  %130 = phi i32 [ %140, %129 ], [ 0, %87 ]
  %131 = phi i64 [ %139, %129 ], [ 0, %87 ]
  %132 = zext nneg i32 %130 to i64
  %133 = getelementptr inbounds nuw i8, ptr %83, i64 %132
  %134 = load i8, ptr %133, align 1, !tbaa !2949
  %135 = zext i8 %134 to i64
  %136 = shl i32 %130, 3
  %137 = zext nneg i32 %136 to i64
  %138 = shl nuw i64 %135, %137
  %139 = or i64 %138, %131
  %140 = add nuw nsw i32 %130, 1
  %141 = icmp eq i32 %140, %84
  br i1 %141, label %142, label %129

142:                                              ; preds = %129, %89, %87
  %143 = phi ptr [ %128, %89 ], [ %83, %87 ], [ %83, %129 ]
  %144 = phi i32 [ %127, %89 ], [ 0, %87 ], [ 0, %129 ]
  %145 = phi i64 [ %126, %89 ], [ 0, %87 ], [ %139, %129 ]
  %146 = icmp ugt i32 %144, 7
  br i1 %146, label %149, label %147

147:                                              ; preds = %142
  %148 = icmp eq i32 %144, 0
  br i1 %148, label %202, label %189

149:                                              ; preds = %142
  %150 = load i8, ptr %143, align 1, !tbaa !2949
  %151 = zext i8 %150 to i64
  %152 = getelementptr inbounds nuw i8, ptr %143, i64 1
  %153 = load i8, ptr %152, align 1, !tbaa !2949
  %154 = zext i8 %153 to i64
  %155 = shl nuw nsw i64 %154, 8
  %156 = or disjoint i64 %155, %151
  %157 = getelementptr inbounds nuw i8, ptr %143, i64 2
  %158 = load i8, ptr %157, align 1, !tbaa !2949
  %159 = zext i8 %158 to i64
  %160 = shl nuw nsw i64 %159, 16
  %161 = or disjoint i64 %156, %160
  %162 = getelementptr inbounds nuw i8, ptr %143, i64 3
  %163 = load i8, ptr %162, align 1, !tbaa !2949
  %164 = zext i8 %163 to i64
  %165 = shl nuw nsw i64 %164, 24
  %166 = or disjoint i64 %161, %165
  %167 = getelementptr inbounds nuw i8, ptr %143, i64 4
  %168 = load i8, ptr %167, align 1, !tbaa !2949
  %169 = zext i8 %168 to i64
  %170 = shl nuw nsw i64 %169, 32
  %171 = or disjoint i64 %166, %170
  %172 = getelementptr inbounds nuw i8, ptr %143, i64 5
  %173 = load i8, ptr %172, align 1, !tbaa !2949
  %174 = zext i8 %173 to i64
  %175 = shl nuw nsw i64 %174, 40
  %176 = or i64 %171, %175
  %177 = getelementptr inbounds nuw i8, ptr %143, i64 6
  %178 = load i8, ptr %177, align 1, !tbaa !2949
  %179 = zext i8 %178 to i64
  %180 = shl nuw nsw i64 %179, 48
  %181 = or i64 %176, %180
  %182 = getelementptr inbounds nuw i8, ptr %143, i64 7
  %183 = load i8, ptr %182, align 1, !tbaa !2949
  %184 = zext i8 %183 to i64
  %185 = shl nuw i64 %184, 56
  %186 = or i64 %181, %185
  %187 = add nsw i32 %144, -8
  %188 = getelementptr inbounds nuw i8, ptr %143, i64 8
  br label %202

189:                                              ; preds = %189, %147
  %190 = phi i32 [ %200, %189 ], [ 0, %147 ]
  %191 = phi i64 [ %199, %189 ], [ 0, %147 ]
  %192 = zext nneg i32 %190 to i64
  %193 = getelementptr inbounds nuw i8, ptr %143, i64 %192
  %194 = load i8, ptr %193, align 1, !tbaa !2949
  %195 = zext i8 %194 to i64
  %196 = shl i32 %190, 3
  %197 = zext nneg i32 %196 to i64
  %198 = shl nuw i64 %195, %197
  %199 = or i64 %198, %191
  %200 = add nuw nsw i32 %190, 1
  %201 = icmp eq i32 %200, %144
  br i1 %201, label %202, label %189

202:                                              ; preds = %189, %149, %147
  %203 = phi ptr [ %188, %149 ], [ %143, %147 ], [ %143, %189 ]
  %204 = phi i32 [ %187, %149 ], [ 0, %147 ], [ 0, %189 ]
  %205 = phi i64 [ %186, %149 ], [ 0, %147 ], [ %199, %189 ]
  %206 = icmp ugt i32 %204, 7
  br i1 %206, label %209, label %207

207:                                              ; preds = %202
  %208 = icmp eq i32 %204, 0
  br i1 %208, label %262, label %249

209:                                              ; preds = %202
  %210 = load i8, ptr %203, align 1, !tbaa !2949
  %211 = zext i8 %210 to i64
  %212 = getelementptr inbounds nuw i8, ptr %203, i64 1
  %213 = load i8, ptr %212, align 1, !tbaa !2949
  %214 = zext i8 %213 to i64
  %215 = shl nuw nsw i64 %214, 8
  %216 = or disjoint i64 %215, %211
  %217 = getelementptr inbounds nuw i8, ptr %203, i64 2
  %218 = load i8, ptr %217, align 1, !tbaa !2949
  %219 = zext i8 %218 to i64
  %220 = shl nuw nsw i64 %219, 16
  %221 = or disjoint i64 %216, %220
  %222 = getelementptr inbounds nuw i8, ptr %203, i64 3
  %223 = load i8, ptr %222, align 1, !tbaa !2949
  %224 = zext i8 %223 to i64
  %225 = shl nuw nsw i64 %224, 24
  %226 = or disjoint i64 %221, %225
  %227 = getelementptr inbounds nuw i8, ptr %203, i64 4
  %228 = load i8, ptr %227, align 1, !tbaa !2949
  %229 = zext i8 %228 to i64
  %230 = shl nuw nsw i64 %229, 32
  %231 = or disjoint i64 %226, %230
  %232 = getelementptr inbounds nuw i8, ptr %203, i64 5
  %233 = load i8, ptr %232, align 1, !tbaa !2949
  %234 = zext i8 %233 to i64
  %235 = shl nuw nsw i64 %234, 40
  %236 = or i64 %231, %235
  %237 = getelementptr inbounds nuw i8, ptr %203, i64 6
  %238 = load i8, ptr %237, align 1, !tbaa !2949
  %239 = zext i8 %238 to i64
  %240 = shl nuw nsw i64 %239, 48
  %241 = or i64 %236, %240
  %242 = getelementptr inbounds nuw i8, ptr %203, i64 7
  %243 = load i8, ptr %242, align 1, !tbaa !2949
  %244 = zext i8 %243 to i64
  %245 = shl nuw i64 %244, 56
  %246 = or i64 %241, %245
  %247 = add nsw i32 %204, -8
  %248 = getelementptr inbounds nuw i8, ptr %203, i64 8
  br label %262

249:                                              ; preds = %249, %207
  %250 = phi i32 [ %260, %249 ], [ 0, %207 ]
  %251 = phi i64 [ %259, %249 ], [ 0, %207 ]
  %252 = zext nneg i32 %250 to i64
  %253 = getelementptr inbounds nuw i8, ptr %203, i64 %252
  %254 = load i8, ptr %253, align 1, !tbaa !2949
  %255 = zext i8 %254 to i64
  %256 = shl i32 %250, 3
  %257 = zext nneg i32 %256 to i64
  %258 = shl nuw i64 %255, %257
  %259 = or i64 %258, %251
  %260 = add nuw nsw i32 %250, 1
  %261 = icmp eq i32 %260, %204
  br i1 %261, label %262, label %249

262:                                              ; preds = %249, %209, %207
  %263 = phi ptr [ %248, %209 ], [ %203, %207 ], [ %203, %249 ]
  %264 = phi i32 [ %247, %209 ], [ 0, %207 ], [ 0, %249 ]
  %265 = phi i64 [ %246, %209 ], [ 0, %207 ], [ %259, %249 ]
  %266 = icmp ugt i32 %264, 7
  br i1 %266, label %269, label %267

267:                                              ; preds = %262
  %268 = icmp eq i32 %264, 0
  br i1 %268, label %322, label %309

269:                                              ; preds = %262
  %270 = load i8, ptr %263, align 1, !tbaa !2949
  %271 = zext i8 %270 to i64
  %272 = getelementptr inbounds nuw i8, ptr %263, i64 1
  %273 = load i8, ptr %272, align 1, !tbaa !2949
  %274 = zext i8 %273 to i64
  %275 = shl nuw nsw i64 %274, 8
  %276 = or disjoint i64 %275, %271
  %277 = getelementptr inbounds nuw i8, ptr %263, i64 2
  %278 = load i8, ptr %277, align 1, !tbaa !2949
  %279 = zext i8 %278 to i64
  %280 = shl nuw nsw i64 %279, 16
  %281 = or disjoint i64 %276, %280
  %282 = getelementptr inbounds nuw i8, ptr %263, i64 3
  %283 = load i8, ptr %282, align 1, !tbaa !2949
  %284 = zext i8 %283 to i64
  %285 = shl nuw nsw i64 %284, 24
  %286 = or disjoint i64 %281, %285
  %287 = getelementptr inbounds nuw i8, ptr %263, i64 4
  %288 = load i8, ptr %287, align 1, !tbaa !2949
  %289 = zext i8 %288 to i64
  %290 = shl nuw nsw i64 %289, 32
  %291 = or disjoint i64 %286, %290
  %292 = getelementptr inbounds nuw i8, ptr %263, i64 5
  %293 = load i8, ptr %292, align 1, !tbaa !2949
  %294 = zext i8 %293 to i64
  %295 = shl nuw nsw i64 %294, 40
  %296 = or i64 %291, %295
  %297 = getelementptr inbounds nuw i8, ptr %263, i64 6
  %298 = load i8, ptr %297, align 1, !tbaa !2949
  %299 = zext i8 %298 to i64
  %300 = shl nuw nsw i64 %299, 48
  %301 = or i64 %296, %300
  %302 = getelementptr inbounds nuw i8, ptr %263, i64 7
  %303 = load i8, ptr %302, align 1, !tbaa !2949
  %304 = zext i8 %303 to i64
  %305 = shl nuw i64 %304, 56
  %306 = or i64 %301, %305
  %307 = add nsw i32 %264, -8
  %308 = getelementptr inbounds nuw i8, ptr %263, i64 8
  br label %322

309:                                              ; preds = %309, %267
  %310 = phi i32 [ %320, %309 ], [ 0, %267 ]
  %311 = phi i64 [ %319, %309 ], [ 0, %267 ]
  %312 = zext nneg i32 %310 to i64
  %313 = getelementptr inbounds nuw i8, ptr %263, i64 %312
  %314 = load i8, ptr %313, align 1, !tbaa !2949
  %315 = zext i8 %314 to i64
  %316 = shl i32 %310, 3
  %317 = zext nneg i32 %316 to i64
  %318 = shl nuw i64 %315, %317
  %319 = or i64 %318, %311
  %320 = add nuw nsw i32 %310, 1
  %321 = icmp eq i32 %320, %264
  br i1 %321, label %322, label %309

322:                                              ; preds = %309, %269, %267
  %323 = phi ptr [ %308, %269 ], [ %263, %267 ], [ %263, %309 ]
  %324 = phi i32 [ %307, %269 ], [ 0, %267 ], [ 0, %309 ]
  %325 = phi i64 [ %306, %269 ], [ 0, %267 ], [ %319, %309 ]
  %326 = icmp ugt i32 %324, 7
  br i1 %326, label %329, label %327

327:                                              ; preds = %322
  %328 = icmp eq i32 %324, 0
  br i1 %328, label %382, label %369

329:                                              ; preds = %322
  %330 = load i8, ptr %323, align 1, !tbaa !2949
  %331 = zext i8 %330 to i64
  %332 = getelementptr inbounds nuw i8, ptr %323, i64 1
  %333 = load i8, ptr %332, align 1, !tbaa !2949
  %334 = zext i8 %333 to i64
  %335 = shl nuw nsw i64 %334, 8
  %336 = or disjoint i64 %335, %331
  %337 = getelementptr inbounds nuw i8, ptr %323, i64 2
  %338 = load i8, ptr %337, align 1, !tbaa !2949
  %339 = zext i8 %338 to i64
  %340 = shl nuw nsw i64 %339, 16
  %341 = or disjoint i64 %336, %340
  %342 = getelementptr inbounds nuw i8, ptr %323, i64 3
  %343 = load i8, ptr %342, align 1, !tbaa !2949
  %344 = zext i8 %343 to i64
  %345 = shl nuw nsw i64 %344, 24
  %346 = or disjoint i64 %341, %345
  %347 = getelementptr inbounds nuw i8, ptr %323, i64 4
  %348 = load i8, ptr %347, align 1, !tbaa !2949
  %349 = zext i8 %348 to i64
  %350 = shl nuw nsw i64 %349, 32
  %351 = or disjoint i64 %346, %350
  %352 = getelementptr inbounds nuw i8, ptr %323, i64 5
  %353 = load i8, ptr %352, align 1, !tbaa !2949
  %354 = zext i8 %353 to i64
  %355 = shl nuw nsw i64 %354, 40
  %356 = or i64 %351, %355
  %357 = getelementptr inbounds nuw i8, ptr %323, i64 6
  %358 = load i8, ptr %357, align 1, !tbaa !2949
  %359 = zext i8 %358 to i64
  %360 = shl nuw nsw i64 %359, 48
  %361 = or i64 %356, %360
  %362 = getelementptr inbounds nuw i8, ptr %323, i64 7
  %363 = load i8, ptr %362, align 1, !tbaa !2949
  %364 = zext i8 %363 to i64
  %365 = shl nuw i64 %364, 56
  %366 = or i64 %361, %365
  %367 = add nsw i32 %324, -8
  %368 = getelementptr inbounds nuw i8, ptr %323, i64 8
  br label %382

369:                                              ; preds = %369, %327
  %370 = phi i32 [ %380, %369 ], [ 0, %327 ]
  %371 = phi i64 [ %379, %369 ], [ 0, %327 ]
  %372 = zext nneg i32 %370 to i64
  %373 = getelementptr inbounds nuw i8, ptr %323, i64 %372
  %374 = load i8, ptr %373, align 1, !tbaa !2949
  %375 = zext i8 %374 to i64
  %376 = shl i32 %370, 3
  %377 = zext nneg i32 %376 to i64
  %378 = shl nuw i64 %375, %377
  %379 = or i64 %378, %371
  %380 = add nuw nsw i32 %370, 1
  %381 = icmp eq i32 %380, %324
  br i1 %381, label %382, label %369

382:                                              ; preds = %369, %329, %327
  %383 = phi ptr [ %368, %329 ], [ %323, %327 ], [ %323, %369 ]
  %384 = phi i32 [ %367, %329 ], [ 0, %327 ], [ 0, %369 ]
  %385 = phi i64 [ %366, %329 ], [ 0, %327 ], [ %379, %369 ]
  %386 = icmp ugt i32 %384, 7
  br i1 %386, label %389, label %387

387:                                              ; preds = %382
  %388 = icmp eq i32 %384, 0
  br i1 %388, label %440, label %427

389:                                              ; preds = %382
  %390 = load i8, ptr %383, align 1, !tbaa !2949
  %391 = zext i8 %390 to i64
  %392 = getelementptr inbounds nuw i8, ptr %383, i64 1
  %393 = load i8, ptr %392, align 1, !tbaa !2949
  %394 = zext i8 %393 to i64
  %395 = shl nuw nsw i64 %394, 8
  %396 = or disjoint i64 %395, %391
  %397 = getelementptr inbounds nuw i8, ptr %383, i64 2
  %398 = load i8, ptr %397, align 1, !tbaa !2949
  %399 = zext i8 %398 to i64
  %400 = shl nuw nsw i64 %399, 16
  %401 = or disjoint i64 %396, %400
  %402 = getelementptr inbounds nuw i8, ptr %383, i64 3
  %403 = load i8, ptr %402, align 1, !tbaa !2949
  %404 = zext i8 %403 to i64
  %405 = shl nuw nsw i64 %404, 24
  %406 = or disjoint i64 %401, %405
  %407 = getelementptr inbounds nuw i8, ptr %383, i64 4
  %408 = load i8, ptr %407, align 1, !tbaa !2949
  %409 = zext i8 %408 to i64
  %410 = shl nuw nsw i64 %409, 32
  %411 = or disjoint i64 %406, %410
  %412 = getelementptr inbounds nuw i8, ptr %383, i64 5
  %413 = load i8, ptr %412, align 1, !tbaa !2949
  %414 = zext i8 %413 to i64
  %415 = shl nuw nsw i64 %414, 40
  %416 = or i64 %411, %415
  %417 = getelementptr inbounds nuw i8, ptr %383, i64 6
  %418 = load i8, ptr %417, align 1, !tbaa !2949
  %419 = zext i8 %418 to i64
  %420 = shl nuw nsw i64 %419, 48
  %421 = or i64 %416, %420
  %422 = getelementptr inbounds nuw i8, ptr %383, i64 7
  %423 = load i8, ptr %422, align 1, !tbaa !2949
  %424 = zext i8 %423 to i64
  %425 = shl nuw i64 %424, 56
  %426 = or i64 %421, %425
  br label %440

427:                                              ; preds = %427, %387
  %428 = phi i32 [ %438, %427 ], [ 0, %387 ]
  %429 = phi i64 [ %437, %427 ], [ 0, %387 ]
  %430 = zext nneg i32 %428 to i64
  %431 = getelementptr inbounds nuw i8, ptr %383, i64 %430
  %432 = load i8, ptr %431, align 1, !tbaa !2949
  %433 = zext i8 %432 to i64
  %434 = shl i32 %428, 3
  %435 = zext nneg i32 %434 to i64
  %436 = shl nuw i64 %433, %435
  %437 = or i64 %436, %429
  %438 = add nuw nsw i32 %428, 1
  %439 = icmp eq i32 %438, %384
  br i1 %439, label %440, label %427

440:                                              ; preds = %427, %389, %387
  %441 = phi i64 [ %426, %389 ], [ 0, %387 ], [ %437, %427 ]
  %442 = shl nuw nsw i64 %23, 2
  %443 = add nuw nsw i64 %442, 28
  %444 = and i64 %443, 480
  %445 = and i64 %22, -225
  %446 = or i64 %445, %25
  %447 = or i64 %446, %444
  %448 = tail call <2 x i64> @__ockl_hostcall_preview(i32 noundef 2, i64 noundef %447, i64 noundef %85, i64 noundef %145, i64 noundef %205, i64 noundef %265, i64 noundef %325, i64 noundef %385, i64 noundef %441) #28
  %449 = sub i64 %18, %23
  %450 = getelementptr inbounds nuw i8, ptr %19, i64 %23
  %451 = icmp eq i64 %449, 0
  br i1 %451, label %452, label %17

452:                                              ; preds = %440, %9
  %453 = phi <2 x i64> [ %12, %9 ], [ %448, %440 ]
  %454 = extractelement <2 x i64> %453, i64 0
  ret i64 %454
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umin.i64(i64, i64) #9

; Function Attrs: convergent mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define internal range(i64 0, 65536) i64 @__ockl_get_local_size(i32 noundef %0) #23 {
  switch i32 %0, label %73 [
    i32 0, label %2
    i32 1, label %25
    i32 2, label %49
  ]

2:                                                ; preds = %1
  %3 = load i32, ptr addrspace(4) @__oclc_ABI_version, align 4, !tbaa !12
  %4 = icmp slt i32 %3, 500
  %5 = tail call i32 @llvm.amdgcn.workgroup.id.x()
  br i1 %4, label %6, label %17

6:                                                ; preds = %2
  %7 = tail call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %8 = getelementptr inbounds nuw i8, ptr addrspace(4) %7, i64 4
  %9 = load i16, ptr addrspace(4) %8, align 4, !tbaa !2950
  %10 = zext i16 %9 to i32
  %11 = getelementptr inbounds nuw i8, ptr addrspace(4) %7, i64 12
  %12 = load i32, ptr addrspace(4) %11, align 4, !tbaa !2953
  %13 = mul i32 %5, %10
  %14 = sub i32 %12, %13
  %15 = tail call i32 @llvm.umin.i32(i32 %14, i32 %10)
  %16 = zext nneg i32 %15 to i64
  br label %73

17:                                               ; preds = %2
  %18 = tail call noundef dereferenceable(256) ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %19 = load i32, ptr addrspace(4) %18, align 8, !tbaa !12
  %20 = icmp ult i32 %5, %19
  %21 = select i1 %20, i64 12, i64 18
  %22 = getelementptr inbounds nuw i8, ptr addrspace(4) %18, i64 %21
  %23 = load i16, ptr addrspace(4) %22, align 2, !tbaa !2954
  %24 = zext i16 %23 to i64
  br label %73

25:                                               ; preds = %1
  %26 = load i32, ptr addrspace(4) @__oclc_ABI_version, align 4, !tbaa !12
  %27 = icmp slt i32 %26, 500
  %28 = tail call i32 @llvm.amdgcn.workgroup.id.y()
  br i1 %27, label %29, label %40

29:                                               ; preds = %25
  %30 = tail call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %31 = getelementptr inbounds nuw i8, ptr addrspace(4) %30, i64 6
  %32 = load i16, ptr addrspace(4) %31, align 2, !tbaa !2955
  %33 = zext i16 %32 to i32
  %34 = getelementptr inbounds nuw i8, ptr addrspace(4) %30, i64 16
  %35 = load i32, ptr addrspace(4) %34, align 8, !tbaa !2956
  %36 = mul i32 %28, %33
  %37 = sub i32 %35, %36
  %38 = tail call i32 @llvm.umin.i32(i32 %37, i32 %33)
  %39 = zext nneg i32 %38 to i64
  br label %73

40:                                               ; preds = %25
  %41 = tail call noundef dereferenceable(256) ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %42 = getelementptr inbounds nuw i8, ptr addrspace(4) %41, i64 4
  %43 = load i32, ptr addrspace(4) %42, align 4, !tbaa !12
  %44 = icmp ult i32 %28, %43
  %45 = select i1 %44, i64 14, i64 20
  %46 = getelementptr inbounds nuw i8, ptr addrspace(4) %41, i64 %45
  %47 = load i16, ptr addrspace(4) %46, align 2, !tbaa !2954
  %48 = zext i16 %47 to i64
  br label %73

49:                                               ; preds = %1
  %50 = load i32, ptr addrspace(4) @__oclc_ABI_version, align 4, !tbaa !12
  %51 = icmp slt i32 %50, 500
  %52 = tail call i32 @llvm.amdgcn.workgroup.id.z()
  br i1 %51, label %53, label %64

53:                                               ; preds = %49
  %54 = tail call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %55 = getelementptr inbounds nuw i8, ptr addrspace(4) %54, i64 8
  %56 = load i16, ptr addrspace(4) %55, align 8, !tbaa !2957
  %57 = zext i16 %56 to i32
  %58 = getelementptr inbounds nuw i8, ptr addrspace(4) %54, i64 20
  %59 = load i32, ptr addrspace(4) %58, align 4, !tbaa !2958
  %60 = mul i32 %52, %57
  %61 = sub i32 %59, %60
  %62 = tail call i32 @llvm.umin.i32(i32 %61, i32 %57)
  %63 = zext nneg i32 %62 to i64
  br label %73

64:                                               ; preds = %49
  %65 = tail call noundef dereferenceable(256) ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %66 = getelementptr inbounds nuw i8, ptr addrspace(4) %65, i64 8
  %67 = load i32, ptr addrspace(4) %66, align 8, !tbaa !12
  %68 = icmp ult i32 %52, %67
  %69 = select i1 %68, i64 16, i64 22
  %70 = getelementptr inbounds nuw i8, ptr addrspace(4) %65, i64 %69
  %71 = load i16, ptr addrspace(4) %70, align 2, !tbaa !2954
  %72 = zext i16 %71 to i64
  br label %73

73:                                               ; preds = %64, %53, %40, %29, %17, %6, %1
  %74 = phi i64 [ %48, %40 ], [ 1, %1 ], [ %24, %17 ], [ %16, %6 ], [ %39, %29 ], [ %63, %53 ], [ %72, %64 ]
  ret i64 %74
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #13

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef nonnull align 4 dereferenceable(64) ptr addrspace(4) @llvm.amdgcn.dispatch.ptr() #13

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.umin.i32(i32, i32) #9

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #13

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.z() #13

; Function Attrs: convergent mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define internal range(i64 0, 4294967296) i64 @__ockl_get_group_id(i32 noundef %0) #24 {
  switch i32 %0, label %8 [
    i32 0, label %2
    i32 1, label %4
    i32 2, label %6
  ]

2:                                                ; preds = %1
  %3 = tail call i32 @llvm.amdgcn.workgroup.id.x()
  br label %8

4:                                                ; preds = %1
  %5 = tail call i32 @llvm.amdgcn.workgroup.id.y()
  br label %8

6:                                                ; preds = %1
  %7 = tail call i32 @llvm.amdgcn.workgroup.id.z()
  br label %8

8:                                                ; preds = %6, %4, %2, %1
  %9 = phi i32 [ %7, %6 ], [ %3, %2 ], [ %5, %4 ], [ 0, %1 ]
  %10 = zext i32 %9 to i64
  ret i64 %10
}

attributes #0 = { convergent mustprogress noreturn nounwind uwtable "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1201" "uniform-work-group-size" }
attributes #1 = { cold noreturn nounwind memory(inaccessiblemem: write) }
attributes #2 = { convergent mustprogress noinline nounwind uwtable "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1201" "uniform-work-group-size" }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { convergent mustprogress norecurse nounwind uwtable "amdgpu-flat-work-group-size"="1,1024" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1201" "uniform-work-group-size" }
attributes #5 = { alwaysinline convergent mustprogress nounwind uwtable "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1201" "uniform-work-group-size" }
attributes #6 = { convergent inlinehint mustprogress nounwind uwtable "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1201" "uniform-work-group-size" }
attributes #7 = { convergent mustprogress norecurse nounwind uwtable "amdgpu-flat-work-group-size"="1,256" "amdgpu-waves-per-eu"="2" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1201" "uniform-work-group-size" }
attributes #8 = { convergent mustprogress nounwind uwtable "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1201" "uniform-work-group-size" }
attributes #9 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #10 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
attributes #11 = { alwaysinline convergent mustprogress nounwind willreturn memory(none) uwtable "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1201" "uniform-work-group-size" }
attributes #12 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #13 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #14 = { convergent nocallback nofree nounwind willreturn }
attributes #15 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #16 = { convergent mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-wwm" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1201" "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-buffer-pk-add-bf16-inst,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-fmin-fmax-global-f32,+atomic-global-pk-add-bf16-inst,+ci-insts,+cube-insts,+cvt-pknorm-vop2-insts,+dl-insts,+dot10-insts,+dot11-insts,+dot12-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+fp8-conversion-insts,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx12-insts,+gfx8-insts,+gfx9-insts,+image-insts,+lerp-inst,+mqsad-insts,+mqsad-pk-insts,+msad-insts,+qsad-insts,+sad-insts,+swmmac-gfx1200-insts,+wavefrontsize32,+wmma-128b-insts" }
attributes #17 = { convergent norecurse nounwind uwtable "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1201" "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-buffer-pk-add-bf16-inst,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-fmin-fmax-global-f32,+atomic-global-pk-add-bf16-inst,+ci-insts,+cube-insts,+cvt-pknorm-vop2-insts,+dl-insts,+dot10-insts,+dot11-insts,+dot12-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+fp8-conversion-insts,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx12-insts,+gfx8-insts,+gfx9-insts,+image-insts,+lerp-inst,+mqsad-insts,+mqsad-pk-insts,+msad-insts,+qsad-insts,+sad-insts,+swmmac-gfx1200-insts,+wavefrontsize32,+wmma-128b-insts" }
attributes #18 = { cold convergent norecurse nounwind uwtable "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1201" "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-buffer-pk-add-bf16-inst,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-fmin-fmax-global-f32,+atomic-global-pk-add-bf16-inst,+ci-insts,+cube-insts,+cvt-pknorm-vop2-insts,+dl-insts,+dot10-insts,+dot11-insts,+dot12-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+fp8-conversion-insts,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx12-insts,+gfx8-insts,+gfx9-insts,+image-insts,+lerp-inst,+mqsad-insts,+mqsad-pk-insts,+msad-insts,+qsad-insts,+sad-insts,+swmmac-gfx1200-insts,+wavefrontsize32,+wmma-128b-insts" }
attributes #19 = { alwaysinline convergent mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-no-wwm" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1201" "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-buffer-pk-add-bf16-inst,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-fmin-fmax-global-f32,+atomic-global-pk-add-bf16-inst,+ci-insts,+cube-insts,+cvt-pknorm-vop2-insts,+dl-insts,+dot10-insts,+dot11-insts,+dot12-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+fp8-conversion-insts,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx12-insts,+gfx8-insts,+gfx9-insts,+image-insts,+lerp-inst,+mqsad-insts,+mqsad-pk-insts,+msad-insts,+qsad-insts,+sad-insts,+swmmac-gfx1200-insts,+wavefrontsize32,+wmma-128b-insts" }
attributes #20 = { nocallback nofree nosync nounwind willreturn }
attributes #21 = { convergent mustprogress norecurse nounwind willreturn uwtable "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-no-wwm" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1201" "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-buffer-pk-add-bf16-inst,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-fmin-fmax-global-f32,+atomic-global-pk-add-bf16-inst,+ci-insts,+cube-insts,+cvt-pknorm-vop2-insts,+dl-insts,+dot10-insts,+dot11-insts,+dot12-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+fp8-conversion-insts,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx12-insts,+gfx8-insts,+gfx9-insts,+image-insts,+lerp-inst,+mqsad-insts,+mqsad-pk-insts,+msad-insts,+qsad-insts,+sad-insts,+swmmac-gfx1200-insts,+wavefrontsize32,+wmma-128b-insts" }
attributes #22 = { nocallback nounwind willreturn }
attributes #23 = { convergent mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-no-wwm" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1201" "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-buffer-pk-add-bf16-inst,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-fmin-fmax-global-f32,+atomic-global-pk-add-bf16-inst,+ci-insts,+cube-insts,+cvt-pknorm-vop2-insts,+dl-insts,+dot10-insts,+dot11-insts,+dot12-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+fp8-conversion-insts,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx12-insts,+gfx8-insts,+gfx9-insts,+image-insts,+lerp-inst,+mqsad-insts,+mqsad-pk-insts,+msad-insts,+qsad-insts,+sad-insts,+swmmac-gfx1200-insts,+wavefrontsize32,+wmma-128b-insts" }
attributes #24 = { convergent mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-no-wwm" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1201" "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-buffer-pk-add-bf16-inst,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-fmin-fmax-global-f32,+atomic-global-pk-add-bf16-inst,+ci-insts,+cube-insts,+cvt-pknorm-vop2-insts,+dl-insts,+dot10-insts,+dot11-insts,+dot12-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+fp8-conversion-insts,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx12-insts,+gfx8-insts,+gfx9-insts,+image-insts,+lerp-inst,+mqsad-insts,+mqsad-pk-insts,+msad-insts,+qsad-insts,+sad-insts,+swmmac-gfx1200-insts,+wavefrontsize32,+wmma-128b-insts" }
attributes #25 = { nounwind }
attributes #26 = { convergent nounwind "uniform-work-group-size" }
attributes #27 = { convergent nounwind willreturn memory(none) "uniform-work-group-size" }
attributes #28 = { convergent nounwind }
attributes #29 = { cold convergent nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6}
!llvm.ident = !{!7}
!llvm.errno.tbaa = !{!8, !12}
!opencl.ocl.version = !{!16}

!0 = distinct !DICompileUnit(language: DW_LANG_HIP, file: !1, producer: "AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "iu4_combined.hip", directory: "/home/kaden/ClaudeCode/warpfront/wt-iu4acc/scratch-2026-09-17/Iu4Acc/single")
!2 = !{i32 1, !"amdhsa_code_object_version", i32 600}
!3 = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"uwtable", i32 2}
!7 = !{!"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"}
!8 = !{!9, !9, i64 0}
!9 = !{!"int", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C++ TBAA"}
!12 = !{!13, !13, i64 0}
!13 = !{!"int", !14, i64 0}
!14 = !{!"omnipotent char", !15, i64 0}
!15 = !{!"Simple C/C++ TBAA"}
!16 = !{i32 2, i32 0}
!17 = distinct !DISubprogram(name: "__cxa_pure_virtual", scope: !18, file: !18, line: 40, type: !19, scopeLine: 40, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!18 = !DIFile(filename: "/opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_runtime_wrapper.h", directory: "")
!19 = !DISubroutineType(types: !20)
!20 = !{}
!21 = !DILocation(line: 41, column: 5, scope: !17)
!22 = !DILocation(line: 42, column: 3, scope: !17)
!23 = distinct !DISubprogram(name: "__cxa_deleted_virtual", scope: !18, file: !18, line: 46, type: !19, scopeLine: 46, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!24 = !DILocation(line: 47, column: 5, scope: !23)
!25 = !DILocation(line: 48, column: 3, scope: !23)
!26 = distinct !DISubprogram(name: "__assert_fail", scope: !27, file: !27, line: 26, type: !19, scopeLine: 27, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!27 = !DIFile(filename: "/opt/rocm/core/include/hip/amd_detail/hip_assert.h", directory: "")
!28 = !{!29, !29, i64 0}
!29 = !{!"p1 omnipotent char", !30, i64 0}
!30 = !{!"any pointer", !10, i64 0}
!31 = !DILocation(line: 28, column: 3, scope: !26)
!32 = !DILocation(line: 28, column: 14, scope: !26)
!33 = !DILocation(line: 47, column: 3, scope: !26)
!34 = !DILocation(line: 47, column: 14, scope: !26)
!35 = !DILocation(line: 47, column: 8, scope: !26)
!36 = !{!37, !37, i64 0}
!37 = !{!"long long", !10, i64 0}
!38 = !DILocation(line: 48, column: 3, scope: !26)
!39 = !DILocation(line: 48, column: 7, scope: !26)
!40 = !DILocation(line: 49, column: 3, scope: !26)
!41 = !DILocation(line: 49, column: 8, scope: !26)
!42 = !DILocation(line: 49, column: 26, scope: !26)
!43 = !DILocation(line: 49, column: 20, scope: !26)
!44 = !DILocation(line: 49, column: 31, scope: !26)
!45 = !DILocation(line: 49, column: 42, scope: !26)
!46 = !DILocation(line: 49, column: 38, scope: !26)
!47 = !{!10, !10, i64 0}
!48 = distinct !{!48, !44, !49, !50}
!49 = !DILocation(line: 49, column: 45, scope: !26)
!50 = !{!"llvm.loop.mustprogress"}
!51 = !DILocation(line: 49, column: 53, scope: !26)
!52 = !DILocation(line: 49, column: 59, scope: !26)
!53 = !DILocation(line: 49, column: 57, scope: !26)
!54 = !DILocation(line: 49, column: 51, scope: !26)
!55 = !DILocation(line: 49, column: 64, scope: !26)
!56 = !DILocation(line: 50, column: 40, scope: !26)
!57 = !DILocation(line: 50, column: 45, scope: !26)
!58 = !DILocation(line: 50, column: 50, scope: !26)
!59 = !DILocation(line: 50, column: 9, scope: !26)
!60 = !DILocation(line: 50, column: 7, scope: !26)
!61 = !DILocation(line: 51, column: 3, scope: !26)
!62 = !DILocation(line: 51, column: 8, scope: !26)
!63 = !DILocation(line: 51, column: 26, scope: !26)
!64 = !DILocation(line: 51, column: 20, scope: !26)
!65 = !DILocation(line: 51, column: 32, scope: !26)
!66 = !DILocation(line: 51, column: 43, scope: !26)
!67 = !DILocation(line: 51, column: 39, scope: !26)
!68 = distinct !{!68, !65, !69, !50}
!69 = !DILocation(line: 51, column: 46, scope: !26)
!70 = !DILocation(line: 51, column: 54, scope: !26)
!71 = !DILocation(line: 51, column: 60, scope: !26)
!72 = !DILocation(line: 51, column: 58, scope: !26)
!73 = !DILocation(line: 51, column: 52, scope: !26)
!74 = !DILocation(line: 51, column: 66, scope: !26)
!75 = !DILocation(line: 52, column: 40, scope: !26)
!76 = !DILocation(line: 52, column: 45, scope: !26)
!77 = !DILocation(line: 52, column: 51, scope: !26)
!78 = !DILocation(line: 52, column: 9, scope: !26)
!79 = !DILocation(line: 52, column: 7, scope: !26)
!80 = !DILocation(line: 53, column: 36, scope: !26)
!81 = !DILocation(line: 53, column: 44, scope: !26)
!82 = !DILocation(line: 53, column: 9, scope: !26)
!83 = !DILocation(line: 53, column: 7, scope: !26)
!84 = !DILocation(line: 54, column: 3, scope: !26)
!85 = !DILocation(line: 54, column: 8, scope: !26)
!86 = !DILocation(line: 54, column: 26, scope: !26)
!87 = !DILocation(line: 54, column: 20, scope: !26)
!88 = !DILocation(line: 54, column: 36, scope: !26)
!89 = !DILocation(line: 54, column: 47, scope: !26)
!90 = !DILocation(line: 54, column: 43, scope: !26)
!91 = distinct !{!91, !88, !92, !50}
!92 = !DILocation(line: 54, column: 50, scope: !26)
!93 = !DILocation(line: 54, column: 58, scope: !26)
!94 = !DILocation(line: 54, column: 64, scope: !26)
!95 = !DILocation(line: 54, column: 62, scope: !26)
!96 = !DILocation(line: 54, column: 56, scope: !26)
!97 = !DILocation(line: 54, column: 74, scope: !26)
!98 = !DILocation(line: 55, column: 40, scope: !26)
!99 = !DILocation(line: 55, column: 45, scope: !26)
!100 = !DILocation(line: 55, column: 55, scope: !26)
!101 = !DILocation(line: 55, column: 9, scope: !26)
!102 = !DILocation(line: 55, column: 7, scope: !26)
!103 = !DILocation(line: 56, column: 3, scope: !26)
!104 = !DILocation(line: 56, column: 8, scope: !26)
!105 = !DILocation(line: 56, column: 26, scope: !26)
!106 = !DILocation(line: 56, column: 20, scope: !26)
!107 = !DILocation(line: 56, column: 37, scope: !26)
!108 = !DILocation(line: 56, column: 48, scope: !26)
!109 = !DILocation(line: 56, column: 44, scope: !26)
!110 = distinct !{!110, !107, !111, !50}
!111 = !DILocation(line: 56, column: 51, scope: !26)
!112 = !DILocation(line: 56, column: 59, scope: !26)
!113 = !DILocation(line: 56, column: 65, scope: !26)
!114 = !DILocation(line: 56, column: 63, scope: !26)
!115 = !DILocation(line: 56, column: 57, scope: !26)
!116 = !DILocation(line: 56, column: 76, scope: !26)
!117 = !DILocation(line: 57, column: 34, scope: !26)
!118 = !DILocation(line: 57, column: 39, scope: !26)
!119 = !DILocation(line: 57, column: 50, scope: !26)
!120 = !DILocation(line: 57, column: 3, scope: !26)
!121 = !DILocation(line: 61, column: 3, scope: !26)
!122 = !DILocation(line: 62, column: 1, scope: !26)
!123 = distinct !DISubprogram(name: "__assertfail", scope: !27, file: !27, line: 64, type: !19, scopeLine: 64, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!124 = !DILocation(line: 66, column: 3, scope: !123)
!125 = !DILocation(line: 67, column: 1, scope: !123)
!126 = distinct !DISubprogram(name: "quantize_int4_mmq_ds128", scope: !127, file: !127, line: 130, type: !19, scopeLine: 135, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!127 = !DIFile(filename: "../iu4_combined.hip", directory: "/home/kaden/ClaudeCode/warpfront/wt-iu4acc/scratch-2026-09-17/Iu4Acc/single")
!128 = !{!129, !129, i64 0}
!129 = !{!"p1 float", !30, i64 0}
!130 = !{!131, !131, i64 0}
!131 = !{!"p1 _ZTS12block_i4_128", !30, i64 0}
!132 = !DILocation(line: 136, column: 5, scope: !126)
!133 = !DILocation(line: 136, column: 23, scope: !126)
!134 = !DILocation(line: 136, column: 15, scope: !126)
!135 = !DILocation(line: 137, column: 5, scope: !126)
!136 = !DILocation(line: 137, column: 19, scope: !126)
!137 = !DILocation(line: 137, column: 32, scope: !126)
!138 = !DILocation(line: 137, column: 30, scope: !126)
!139 = !DILocation(line: 137, column: 45, scope: !126)
!140 = !DILocation(line: 137, column: 43, scope: !126)
!141 = !DILocation(line: 137, column: 15, scope: !126)
!142 = !DILocation(line: 138, column: 5, scope: !126)
!143 = !DILocation(line: 138, column: 20, scope: !126)
!144 = !DILocation(line: 138, column: 22, scope: !126)
!145 = !DILocation(line: 138, column: 15, scope: !126)
!146 = !DILocation(line: 139, column: 9, scope: !126)
!147 = !DILocation(line: 139, column: 18, scope: !126)
!148 = !DILocation(line: 139, column: 15, scope: !126)
!149 = !DILocation(line: 139, column: 20, scope: !126)
!150 = !DILocation(line: 139, column: 23, scope: !126)
!151 = !DILocation(line: 139, column: 29, scope: !126)
!152 = !DILocation(line: 139, column: 26, scope: !126)
!153 = !DILocation(line: 139, column: 32, scope: !126)
!154 = !DILocation(line: 140, column: 5, scope: !126)
!155 = !DILocation(line: 140, column: 22, scope: !126)
!156 = !DILocation(line: 140, column: 34, scope: !126)
!157 = !DILocation(line: 140, column: 15, scope: !126)
!158 = !DILocation(line: 142, column: 5, scope: !126)
!159 = !DILocation(line: 142, column: 24, scope: !126)
!160 = !DILocation(line: 142, column: 27, scope: !126)
!161 = !DILocation(line: 142, column: 33, scope: !126)
!162 = !DILocation(line: 142, column: 31, scope: !126)
!163 = !DILocation(line: 142, column: 23, scope: !126)
!164 = !DILocation(line: 143, column: 28, scope: !126)
!165 = !DILocation(line: 143, column: 43, scope: !126)
!166 = !DILocation(line: 143, column: 51, scope: !126)
!167 = !DILocation(line: 143, column: 49, scope: !126)
!168 = !DILocation(line: 143, column: 11, scope: !126)
!169 = !DILocation(line: 143, column: 55, scope: !126)
!170 = !DILocation(line: 144, column: 11, scope: !126)
!171 = !DILocation(line: 145, column: 5, scope: !126)
!172 = !DILocation(line: 145, column: 29, scope: !126)
!173 = !{!174, !175, i64 0}
!174 = !{!"_ZTS15HIP_vector_baseIfLj4EE", !175, i64 0, !175, i64 4, !175, i64 8, !175, i64 12}
!175 = !{!"float", !10, i64 0}
!176 = !DILocation(line: 145, column: 25, scope: !126)
!177 = !{!175, !175, i64 0}
!178 = !DILocation(line: 145, column: 35, scope: !126)
!179 = !{!174, !175, i64 4}
!180 = !DILocation(line: 145, column: 41, scope: !126)
!181 = !{!174, !175, i64 8}
!182 = !DILocation(line: 145, column: 47, scope: !126)
!183 = !{!174, !175, i64 12}
!184 = !DILocation(line: 147, column: 5, scope: !126)
!185 = !DILocation(line: 147, column: 24, scope: !126)
!186 = !DILocation(line: 147, column: 27, scope: !126)
!187 = !DILocation(line: 147, column: 15, scope: !126)
!188 = !DILocation(line: 148, column: 5, scope: !126)
!189 = !DILocation(line: 148, column: 25, scope: !126)
!190 = !DILocation(line: 148, column: 40, scope: !126)
!191 = !DILocation(line: 148, column: 49, scope: !126)
!192 = !DILocation(line: 148, column: 47, scope: !126)
!193 = !DILocation(line: 148, column: 27, scope: !126)
!194 = !DILocation(line: 148, column: 53, scope: !126)
!195 = !DILocation(line: 148, column: 51, scope: !126)
!196 = !DILocation(line: 148, column: 19, scope: !126)
!197 = !DILocation(line: 149, column: 32, scope: !126)
!198 = !DILocation(line: 149, column: 36, scope: !126)
!199 = !DILocation(line: 149, column: 41, scope: !126)
!200 = !DILocation(line: 149, column: 5, scope: !126)
!201 = !DILocation(line: 150, column: 1, scope: !126)
!202 = distinct !DISubprogram(name: "__get_y", scope: !203, file: !203, line: 309, type: !19, scopeLine: 309, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!203 = !DIFile(filename: "/opt/rocm/core/include/hip/amd_detail/amd_hip_runtime.h", directory: "")
!204 = !DILocation(line: 309, column: 160, scope: !202)
!205 = !DILocation(line: 309, column: 153, scope: !202)
!206 = distinct !DISubprogram(name: "__get_x", scope: !203, file: !203, line: 308, type: !19, scopeLine: 308, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!207 = !DILocation(line: 308, column: 160, scope: !206)
!208 = !DILocation(line: 308, column: 153, scope: !206)
!209 = distinct !DISubprogram(name: "__get_x", scope: !203, file: !203, line: 317, type: !19, scopeLine: 317, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!210 = !DILocation(line: 317, column: 194, scope: !209)
!211 = !DILocation(line: 317, column: 187, scope: !209)
!212 = distinct !DISubprogram(name: "__get_x", scope: !203, file: !203, line: 299, type: !19, scopeLine: 299, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!213 = !DILocation(line: 299, column: 194, scope: !212)
!214 = !DILocation(line: 299, column: 187, scope: !212)
!215 = distinct !DISubprogram(name: "make_float4", scope: !216, file: !216, line: 1563, type: !19, scopeLine: 1563, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!216 = !DIFile(filename: "/opt/rocm/core/include/hip/amd_detail/amd_hip_vector_types.h", directory: "")
!217 = !DILocation(line: 1563, column: 127, scope: !215)
!218 = !DILocation(line: 1563, column: 130, scope: !215)
!219 = !DILocation(line: 1563, column: 133, scope: !215)
!220 = !DILocation(line: 1563, column: 136, scope: !215)
!221 = !DILocation(line: 1563, column: 125, scope: !215)
!222 = !DILocation(line: 1563, column: 140, scope: !215)
!223 = distinct !DISubprogram(name: "quantize_block_i4_128_wave", scope: !127, file: !127, line: 36, type: !19, scopeLine: 40, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!224 = !DILocation(line: 41, column: 5, scope: !223)
!225 = !DILocation(line: 41, column: 36, scope: !223)
!226 = !DILocation(line: 41, column: 30, scope: !223)
!227 = !DILocation(line: 41, column: 50, scope: !223)
!228 = !DILocation(line: 41, column: 44, scope: !223)
!229 = !DILocation(line: 41, column: 24, scope: !223)
!230 = !DILocation(line: 41, column: 71, scope: !223)
!231 = !DILocation(line: 41, column: 65, scope: !223)
!232 = !DILocation(line: 41, column: 85, scope: !223)
!233 = !DILocation(line: 41, column: 79, scope: !223)
!234 = !DILocation(line: 41, column: 59, scope: !223)
!235 = !DILocation(line: 41, column: 18, scope: !223)
!236 = !DILocation(line: 41, column: 11, scope: !223)
!237 = !DILocation(line: 42, column: 10, scope: !223)
!238 = !DILocation(line: 42, column: 14, scope: !223)
!239 = !DILocation(line: 42, column: 27, scope: !223)
!240 = !DILocation(line: 42, column: 34, scope: !223)
!241 = !DILocation(line: 42, column: 5, scope: !223)
!242 = !DILocation(line: 43, column: 22, scope: !223)
!243 = !DILocation(line: 43, column: 39, scope: !223)
!244 = !DILocation(line: 43, column: 45, scope: !223)
!245 = !DILocation(line: 43, column: 28, scope: !223)
!246 = !DILocation(line: 43, column: 16, scope: !223)
!247 = !DILocation(line: 43, column: 14, scope: !223)
!248 = !DILocation(line: 44, column: 5, scope: !223)
!249 = !DILocation(line: 42, column: 46, scope: !223)
!250 = distinct !{!250, !241, !248, !50}
!251 = !DILocation(line: 46, column: 5, scope: !223)
!252 = !DILocation(line: 46, column: 11, scope: !223)
!253 = !DILocation(line: 47, column: 9, scope: !223)
!254 = !DILocation(line: 47, column: 14, scope: !223)
!255 = !DILocation(line: 48, column: 9, scope: !223)
!256 = !DILocation(line: 48, column: 15, scope: !223)
!257 = !DILocation(line: 49, column: 14, scope: !223)
!258 = !DILocation(line: 49, column: 18, scope: !223)
!259 = !DILocation(line: 49, column: 25, scope: !223)
!260 = !DILocation(line: 49, column: 27, scope: !223)
!261 = !DILocation(line: 49, column: 9, scope: !223)
!262 = !DILocation(line: 50, column: 13, scope: !223)
!263 = !DILocation(line: 50, column: 31, scope: !223)
!264 = !DILocation(line: 50, column: 36, scope: !223)
!265 = !DILocation(line: 50, column: 44, scope: !223)
!266 = !DILocation(line: 50, column: 68, scope: !223)
!267 = !DILocation(line: 50, column: 70, scope: !223)
!268 = !DILocation(line: 50, column: 59, scope: !223)
!269 = !DILocation(line: 50, column: 51, scope: !223)
!270 = !DILocation(line: 50, column: 25, scope: !223)
!271 = !DILocation(line: 51, column: 13, scope: !223)
!272 = !DILocation(line: 51, column: 19, scope: !223)
!273 = !DILocation(line: 53, column: 18, scope: !223)
!274 = !DILocation(line: 53, column: 22, scope: !223)
!275 = !DILocation(line: 53, column: 29, scope: !223)
!276 = !DILocation(line: 53, column: 31, scope: !223)
!277 = !DILocation(line: 53, column: 13, scope: !223)
!278 = !DILocation(line: 54, column: 17, scope: !223)
!279 = !DILocation(line: 54, column: 33, scope: !223)
!280 = !DILocation(line: 54, column: 36, scope: !223)
!281 = !DILocation(line: 54, column: 41, scope: !223)
!282 = !DILocation(line: 54, column: 39, scope: !223)
!283 = !DILocation(line: 54, column: 27, scope: !223)
!284 = !DILocation(line: 54, column: 23, scope: !223)
!285 = !DILocation(line: 55, column: 33, scope: !223)
!286 = !DILocation(line: 55, column: 27, scope: !223)
!287 = !DILocation(line: 55, column: 21, scope: !223)
!288 = !DILocation(line: 55, column: 19, scope: !223)
!289 = !DILocation(line: 56, column: 17, scope: !223)
!290 = !DILocation(line: 56, column: 46, scope: !223)
!291 = !DILocation(line: 56, column: 45, scope: !223)
!292 = !DILocation(line: 56, column: 49, scope: !223)
!293 = !DILocation(line: 56, column: 53, scope: !223)
!294 = !DILocation(line: 56, column: 56, scope: !223)
!295 = !DILocation(line: 56, column: 35, scope: !223)
!296 = !DILocation(line: 56, column: 29, scope: !223)
!297 = !DILocation(line: 57, column: 33, scope: !223)
!298 = !DILocation(line: 57, column: 38, scope: !223)
!299 = !DILocation(line: 57, column: 43, scope: !223)
!300 = !DILocation(line: 57, column: 23, scope: !223)
!301 = !DILocation(line: 57, column: 21, scope: !223)
!302 = !DILocation(line: 58, column: 13, scope: !223)
!303 = !DILocation(line: 53, column: 36, scope: !223)
!304 = distinct !{!304, !277, !302, !50, !305}
!305 = !{!"llvm.loop.unroll.enable"}
!306 = !DILocation(line: 59, column: 18, scope: !223)
!307 = !DILocation(line: 59, column: 22, scope: !223)
!308 = !DILocation(line: 59, column: 35, scope: !223)
!309 = !DILocation(line: 59, column: 42, scope: !223)
!310 = !DILocation(line: 59, column: 13, scope: !223)
!311 = !DILocation(line: 60, column: 35, scope: !223)
!312 = !DILocation(line: 60, column: 40, scope: !223)
!313 = !DILocation(line: 60, column: 24, scope: !223)
!314 = !DILocation(line: 60, column: 21, scope: !223)
!315 = !DILocation(line: 61, column: 13, scope: !223)
!316 = !DILocation(line: 59, column: 54, scope: !223)
!317 = distinct !{!317, !310, !315, !50}
!318 = !DILocation(line: 62, column: 17, scope: !223)
!319 = !DILocation(line: 62, column: 23, scope: !223)
!320 = !DILocation(line: 62, column: 21, scope: !223)
!321 = !DILocation(line: 63, column: 28, scope: !223)
!322 = !DILocation(line: 63, column: 26, scope: !223)
!323 = !DILocation(line: 64, column: 26, scope: !223)
!324 = !DILocation(line: 64, column: 24, scope: !223)
!325 = !DILocation(line: 65, column: 13, scope: !223)
!326 = !DILocation(line: 66, column: 9, scope: !223)
!327 = !DILocation(line: 49, column: 32, scope: !223)
!328 = distinct !{!328, !261, !326, !50}
!329 = !DILocation(line: 67, column: 5, scope: !223)
!330 = !DILocation(line: 69, column: 5, scope: !223)
!331 = !DILocation(line: 71, column: 10, scope: !223)
!332 = !DILocation(line: 71, column: 14, scope: !223)
!333 = !DILocation(line: 71, column: 21, scope: !223)
!334 = !DILocation(line: 71, column: 23, scope: !223)
!335 = !DILocation(line: 71, column: 5, scope: !223)
!336 = !DILocation(line: 72, column: 9, scope: !223)
!337 = !DILocation(line: 72, column: 20, scope: !223)
!338 = !DILocation(line: 72, column: 25, scope: !223)
!339 = !DILocation(line: 72, column: 19, scope: !223)
!340 = !DILocation(line: 72, column: 49, scope: !223)
!341 = !DILocation(line: 72, column: 52, scope: !223)
!342 = !DILocation(line: 72, column: 57, scope: !223)
!343 = !DILocation(line: 72, column: 55, scope: !223)
!344 = !DILocation(line: 72, column: 43, scope: !223)
!345 = !DILocation(line: 72, column: 15, scope: !223)
!346 = !DILocation(line: 73, column: 25, scope: !223)
!347 = !DILocation(line: 73, column: 19, scope: !223)
!348 = !DILocation(line: 73, column: 13, scope: !223)
!349 = !DILocation(line: 73, column: 11, scope: !223)
!350 = !DILocation(line: 74, column: 22, scope: !223)
!351 = !DILocation(line: 74, column: 12, scope: !223)
!352 = !DILocation(line: 74, column: 9, scope: !223)
!353 = !DILocation(line: 74, column: 15, scope: !223)
!354 = !DILocation(line: 75, column: 5, scope: !223)
!355 = !DILocation(line: 71, column: 28, scope: !223)
!356 = distinct !{!356, !335, !354, !50, !305}
!357 = !DILocation(line: 76, column: 5, scope: !223)
!358 = !DILocation(line: 76, column: 13, scope: !223)
!359 = !DILocation(line: 76, column: 21, scope: !223)
!360 = !DILocation(line: 76, column: 19, scope: !223)
!361 = !DILocation(line: 76, column: 29, scope: !223)
!362 = !DILocation(line: 76, column: 27, scope: !223)
!363 = !DILocation(line: 76, column: 37, scope: !223)
!364 = !DILocation(line: 76, column: 35, scope: !223)
!365 = !DILocation(line: 76, column: 9, scope: !223)
!366 = !DILocation(line: 77, column: 10, scope: !223)
!367 = !DILocation(line: 77, column: 14, scope: !223)
!368 = !DILocation(line: 77, column: 27, scope: !223)
!369 = !DILocation(line: 77, column: 34, scope: !223)
!370 = !DILocation(line: 77, column: 5, scope: !223)
!371 = !DILocation(line: 78, column: 25, scope: !223)
!372 = !DILocation(line: 78, column: 28, scope: !223)
!373 = !DILocation(line: 78, column: 14, scope: !223)
!374 = !DILocation(line: 78, column: 11, scope: !223)
!375 = !DILocation(line: 79, column: 5, scope: !223)
!376 = !DILocation(line: 77, column: 46, scope: !223)
!377 = distinct !{!377, !370, !375, !50}
!378 = !DILocation(line: 81, column: 5, scope: !223)
!379 = !DILocation(line: 81, column: 25, scope: !223)
!380 = !DILocation(line: 81, column: 30, scope: !223)
!381 = !DILocation(line: 81, column: 20, scope: !223)
!382 = !DILocation(line: 82, column: 41, scope: !223)
!383 = !DILocation(line: 82, column: 47, scope: !223)
!384 = !DILocation(line: 82, column: 57, scope: !223)
!385 = !DILocation(line: 82, column: 63, scope: !223)
!386 = !DILocation(line: 82, column: 69, scope: !223)
!387 = !DILocation(line: 82, column: 53, scope: !223)
!388 = !DILocation(line: 82, column: 39, scope: !223)
!389 = !DILocation(line: 82, column: 5, scope: !223)
!390 = !DILocation(line: 82, column: 8, scope: !223)
!391 = !DILocation(line: 82, column: 13, scope: !223)
!392 = !DILocation(line: 82, column: 17, scope: !223)
!393 = !DILocation(line: 82, column: 22, scope: !223)
!394 = !DILocation(line: 83, column: 41, scope: !223)
!395 = !DILocation(line: 83, column: 47, scope: !223)
!396 = !DILocation(line: 83, column: 57, scope: !223)
!397 = !DILocation(line: 83, column: 63, scope: !223)
!398 = !DILocation(line: 83, column: 69, scope: !223)
!399 = !DILocation(line: 83, column: 53, scope: !223)
!400 = !DILocation(line: 83, column: 39, scope: !223)
!401 = !DILocation(line: 83, column: 5, scope: !223)
!402 = !DILocation(line: 83, column: 8, scope: !223)
!403 = !DILocation(line: 83, column: 13, scope: !223)
!404 = !DILocation(line: 83, column: 17, scope: !223)
!405 = !DILocation(line: 83, column: 22, scope: !223)
!406 = !DILocation(line: 84, column: 9, scope: !223)
!407 = !DILocation(line: 84, column: 14, scope: !223)
!408 = !DILocation(line: 85, column: 18, scope: !223)
!409 = !DILocation(line: 85, column: 9, scope: !223)
!410 = !DILocation(line: 85, column: 14, scope: !223)
!411 = !DILocation(line: 85, column: 16, scope: !223)
!412 = !{!413, !175, i64 0}
!413 = !{!"_ZTS12block_i4_128", !175, i64 0, !9, i64 4, !10, i64 8}
!414 = !DILocation(line: 86, column: 18, scope: !223)
!415 = !DILocation(line: 86, column: 9, scope: !223)
!416 = !DILocation(line: 86, column: 14, scope: !223)
!417 = !DILocation(line: 86, column: 16, scope: !223)
!418 = !{!413, !9, i64 4}
!419 = !DILocation(line: 87, column: 5, scope: !223)
!420 = !DILocation(line: 88, column: 1, scope: !223)
!421 = distinct !DISubprogram(name: "gemm_mq4g256v2_residual_mmq_iu4", scope: !127, file: !127, line: 946, type: !19, scopeLine: 951, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!422 = !DILocation(line: 952, column: 9, scope: !421)
!423 = !DILocation(line: 953, column: 58, scope: !421)
!424 = !DILocation(line: 953, column: 61, scope: !421)
!425 = !DILocation(line: 953, column: 65, scope: !421)
!426 = !DILocation(line: 953, column: 68, scope: !421)
!427 = !DILocation(line: 953, column: 71, scope: !421)
!428 = !DILocation(line: 953, column: 74, scope: !421)
!429 = !DILocation(line: 953, column: 9, scope: !421)
!430 = !DILocation(line: 954, column: 5, scope: !421)
!431 = !DILocation(line: 955, column: 59, scope: !421)
!432 = !DILocation(line: 955, column: 62, scope: !421)
!433 = !DILocation(line: 955, column: 66, scope: !421)
!434 = !DILocation(line: 955, column: 69, scope: !421)
!435 = !DILocation(line: 955, column: 72, scope: !421)
!436 = !DILocation(line: 955, column: 75, scope: !421)
!437 = !DILocation(line: 955, column: 9, scope: !421)
!438 = !DILocation(line: 957, column: 1, scope: !421)
!439 = distinct !DISubprogram(name: "gemm_mq4g256v2_residual_mmq_iu4_gfx12_body<true>", scope: !127, file: !127, line: 488, type: !19, scopeLine: 493, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!440 = !DILocation(line: 494, column: 5, scope: !439)
!441 = !DILocation(line: 494, column: 21, scope: !439)
!442 = !DILocation(line: 494, column: 15, scope: !439)
!443 = !DILocation(line: 495, column: 5, scope: !439)
!444 = !DILocation(line: 495, column: 22, scope: !439)
!445 = !DILocation(line: 495, column: 26, scope: !439)
!446 = !DILocation(line: 495, column: 15, scope: !439)
!447 = !DILocation(line: 496, column: 5, scope: !439)
!448 = !DILocation(line: 496, column: 22, scope: !439)
!449 = !DILocation(line: 496, column: 26, scope: !439)
!450 = !DILocation(line: 496, column: 15, scope: !439)
!451 = !DILocation(line: 497, column: 5, scope: !439)
!452 = !DILocation(line: 497, column: 20, scope: !439)
!453 = !DILocation(line: 497, column: 31, scope: !439)
!454 = !DILocation(line: 497, column: 15, scope: !439)
!455 = !DILocation(line: 498, column: 5, scope: !439)
!456 = !DILocation(line: 498, column: 20, scope: !439)
!457 = !DILocation(line: 498, column: 31, scope: !439)
!458 = !DILocation(line: 498, column: 15, scope: !439)
!459 = !DILocation(line: 500, column: 9, scope: !439)
!460 = !DILocation(line: 500, column: 15, scope: !439)
!461 = !DILocation(line: 500, column: 12, scope: !439)
!462 = !DILocation(line: 500, column: 17, scope: !439)
!463 = !DILocation(line: 500, column: 20, scope: !439)
!464 = !DILocation(line: 500, column: 26, scope: !439)
!465 = !DILocation(line: 500, column: 23, scope: !439)
!466 = !DILocation(line: 500, column: 29, scope: !439)
!467 = !DILocation(line: 502, column: 5, scope: !439)
!468 = !DILocation(line: 502, column: 24, scope: !439)
!469 = !DILocation(line: 502, column: 29, scope: !439)
!470 = !DILocation(line: 502, column: 15, scope: !439)
!471 = !DILocation(line: 503, column: 5, scope: !439)
!472 = !DILocation(line: 503, column: 23, scope: !439)
!473 = !DILocation(line: 503, column: 28, scope: !439)
!474 = !DILocation(line: 503, column: 15, scope: !439)
!475 = !DILocation(line: 504, column: 5, scope: !439)
!476 = !DILocation(line: 504, column: 25, scope: !439)
!477 = !DILocation(line: 504, column: 30, scope: !439)
!478 = !DILocation(line: 504, column: 15, scope: !439)
!479 = !DILocation(line: 505, column: 5, scope: !439)
!480 = !DILocation(line: 505, column: 26, scope: !439)
!481 = !DILocation(line: 505, column: 31, scope: !439)
!482 = !DILocation(line: 505, column: 15, scope: !439)
!483 = !DILocation(line: 506, column: 5, scope: !439)
!484 = !DILocation(line: 506, column: 27, scope: !439)
!485 = !DILocation(line: 506, column: 35, scope: !439)
!486 = !DILocation(line: 506, column: 15, scope: !439)
!487 = !DILocation(line: 507, column: 5, scope: !439)
!488 = !DILocation(line: 507, column: 27, scope: !439)
!489 = !DILocation(line: 507, column: 36, scope: !439)
!490 = !DILocation(line: 507, column: 15, scope: !439)
!491 = !DILocation(line: 509, column: 5, scope: !439)
!492 = !DILocation(line: 509, column: 32, scope: !439)
!493 = !DILocation(line: 509, column: 34, scope: !439)
!494 = !DILocation(line: 509, column: 15, scope: !439)
!495 = !DILocation(line: 512, column: 5, scope: !439)
!496 = !DILocation(line: 512, column: 24, scope: !439)
!497 = !DILocation(line: 513, column: 5, scope: !439)
!498 = !DILocation(line: 513, column: 24, scope: !439)
!499 = !DILocation(line: 518, column: 5, scope: !439)
!500 = !DILocation(line: 519, column: 5, scope: !439)
!501 = !DILocation(line: 524, column: 5, scope: !439)
!502 = !DILocation(line: 524, column: 38, scope: !439)
!503 = !DILocation(line: 524, column: 43, scope: !439)
!504 = !DILocation(line: 524, column: 20, scope: !439)
!505 = !DILocation(line: 526, column: 10, scope: !439)
!506 = !DILocation(line: 526, column: 14, scope: !439)
!507 = !DILocation(line: 526, column: 22, scope: !439)
!508 = !DILocation(line: 526, column: 25, scope: !439)
!509 = !DILocation(line: 526, column: 5, scope: !439)
!510 = !DILocation(line: 527, column: 9, scope: !439)
!511 = !DILocation(line: 527, column: 41, scope: !439)
!512 = !DILocation(line: 527, column: 50, scope: !439)
!513 = !DILocation(line: 527, column: 56, scope: !439)
!514 = !DILocation(line: 527, column: 54, scope: !439)
!515 = !DILocation(line: 527, column: 24, scope: !439)
!516 = !DILocation(line: 529, column: 14, scope: !439)
!517 = !DILocation(line: 529, column: 18, scope: !439)
!518 = !DILocation(line: 529, column: 26, scope: !439)
!519 = !DILocation(line: 529, column: 29, scope: !439)
!520 = !DILocation(line: 529, column: 9, scope: !439)
!521 = !DILocation(line: 530, column: 28, scope: !439)
!522 = !DILocation(line: 530, column: 32, scope: !439)
!523 = !DILocation(line: 530, column: 49, scope: !439)
!524 = !DILocation(line: 530, column: 37, scope: !439)
!525 = !DILocation(line: 530, column: 53, scope: !439)
!526 = !DILocation(line: 530, column: 71, scope: !439)
!527 = !DILocation(line: 530, column: 69, scope: !439)
!528 = !DILocation(line: 530, column: 17, scope: !439)
!529 = !DILocation(line: 530, column: 13, scope: !439)
!530 = !DILocation(line: 530, column: 21, scope: !439)
!531 = !DILocation(line: 530, column: 25, scope: !439)
!532 = !DILocation(line: 531, column: 9, scope: !439)
!533 = !DILocation(line: 529, column: 36, scope: !439)
!534 = distinct !{!534, !520, !532, !50, !305}
!535 = !DILocation(line: 532, column: 5, scope: !439)
!536 = !DILocation(line: 526, column: 32, scope: !439)
!537 = distinct !{!537, !509, !535, !50, !305}
!538 = !DILocation(line: 534, column: 10, scope: !439)
!539 = !DILocation(line: 534, column: 14, scope: !439)
!540 = !DILocation(line: 534, column: 22, scope: !439)
!541 = !DILocation(line: 534, column: 25, scope: !439)
!542 = !DILocation(line: 534, column: 5, scope: !439)
!543 = !DILocation(line: 535, column: 9, scope: !439)
!544 = !DILocation(line: 535, column: 41, scope: !439)
!545 = !DILocation(line: 535, column: 49, scope: !439)
!546 = !DILocation(line: 535, column: 55, scope: !439)
!547 = !DILocation(line: 535, column: 53, scope: !439)
!548 = !DILocation(line: 535, column: 24, scope: !439)
!549 = !DILocation(line: 537, column: 14, scope: !439)
!550 = !DILocation(line: 537, column: 18, scope: !439)
!551 = !DILocation(line: 537, column: 26, scope: !439)
!552 = !DILocation(line: 537, column: 29, scope: !439)
!553 = !DILocation(line: 537, column: 9, scope: !439)
!554 = !DILocation(line: 538, column: 28, scope: !439)
!555 = !DILocation(line: 538, column: 32, scope: !439)
!556 = !DILocation(line: 538, column: 49, scope: !439)
!557 = !DILocation(line: 538, column: 37, scope: !439)
!558 = !DILocation(line: 538, column: 53, scope: !439)
!559 = !DILocation(line: 538, column: 71, scope: !439)
!560 = !DILocation(line: 538, column: 69, scope: !439)
!561 = !DILocation(line: 538, column: 17, scope: !439)
!562 = !DILocation(line: 538, column: 13, scope: !439)
!563 = !DILocation(line: 538, column: 21, scope: !439)
!564 = !DILocation(line: 538, column: 25, scope: !439)
!565 = !DILocation(line: 539, column: 9, scope: !439)
!566 = !DILocation(line: 537, column: 36, scope: !439)
!567 = distinct !{!567, !553, !565, !50, !305}
!568 = !DILocation(line: 540, column: 5, scope: !439)
!569 = !DILocation(line: 534, column: 32, scope: !439)
!570 = distinct !{!570, !542, !568, !50, !305}
!571 = !DILocation(line: 544, column: 5, scope: !439)
!572 = !DILocation(line: 564, column: 5, scope: !439)
!573 = !DILocation(line: 565, column: 5, scope: !439)
!574 = !DILocation(line: 566, column: 5, scope: !439)
!575 = !DILocation(line: 567, column: 5, scope: !439)
!576 = !DILocation(line: 569, column: 10, scope: !439)
!577 = !DILocation(line: 569, column: 14, scope: !439)
!578 = !DILocation(line: 569, column: 21, scope: !439)
!579 = !DILocation(line: 569, column: 23, scope: !439)
!580 = !DILocation(line: 569, column: 5, scope: !439)
!581 = !DILocation(line: 570, column: 9, scope: !439)
!582 = !DILocation(line: 570, column: 27, scope: !439)
!583 = !DILocation(line: 570, column: 32, scope: !439)
!584 = !DILocation(line: 570, column: 34, scope: !439)
!585 = !DILocation(line: 570, column: 30, scope: !439)
!586 = !DILocation(line: 570, column: 42, scope: !439)
!587 = !DILocation(line: 570, column: 46, scope: !439)
!588 = !DILocation(line: 570, column: 39, scope: !439)
!589 = !DILocation(line: 570, column: 19, scope: !439)
!590 = !DILocation(line: 571, column: 9, scope: !439)
!591 = !DILocation(line: 571, column: 28, scope: !439)
!592 = !DILocation(line: 571, column: 36, scope: !439)
!593 = !DILocation(line: 571, column: 34, scope: !439)
!594 = !DILocation(line: 571, column: 27, scope: !439)
!595 = !DILocation(line: 571, column: 41, scope: !439)
!596 = !DILocation(line: 571, column: 50, scope: !439)
!597 = !DILocation(line: 571, column: 52, scope: !439)
!598 = !DILocation(line: 571, column: 19, scope: !439)
!599 = !DILocation(line: 572, column: 22, scope: !439)
!600 = !DILocation(line: 572, column: 30, scope: !439)
!601 = !DILocation(line: 572, column: 28, scope: !439)
!602 = !DILocation(line: 572, column: 21, scope: !439)
!603 = !DILocation(line: 572, column: 16, scope: !439)
!604 = !DILocation(line: 572, column: 9, scope: !439)
!605 = !DILocation(line: 572, column: 19, scope: !439)
!606 = !DILocation(line: 573, column: 33, scope: !439)
!607 = !DILocation(line: 573, column: 39, scope: !439)
!608 = !DILocation(line: 574, column: 26, scope: !439)
!609 = !DILocation(line: 574, column: 30, scope: !439)
!610 = !DILocation(line: 574, column: 35, scope: !439)
!611 = !DILocation(line: 574, column: 13, scope: !439)
!612 = !DILocation(line: 573, column: 18, scope: !439)
!613 = !DILocation(line: 573, column: 9, scope: !439)
!614 = !DILocation(line: 573, column: 21, scope: !439)
!615 = !DILocation(line: 575, column: 9, scope: !439)
!616 = !DILocation(line: 575, column: 28, scope: !439)
!617 = !DILocation(line: 575, column: 33, scope: !439)
!618 = !DILocation(line: 575, column: 35, scope: !439)
!619 = !DILocation(line: 575, column: 31, scope: !439)
!620 = !DILocation(line: 575, column: 43, scope: !439)
!621 = !DILocation(line: 575, column: 47, scope: !439)
!622 = !DILocation(line: 575, column: 40, scope: !439)
!623 = !DILocation(line: 575, column: 19, scope: !439)
!624 = !DILocation(line: 576, column: 9, scope: !439)
!625 = !DILocation(line: 576, column: 28, scope: !439)
!626 = !DILocation(line: 576, column: 37, scope: !439)
!627 = !DILocation(line: 576, column: 35, scope: !439)
!628 = !DILocation(line: 576, column: 27, scope: !439)
!629 = !DILocation(line: 576, column: 42, scope: !439)
!630 = !DILocation(line: 576, column: 52, scope: !439)
!631 = !DILocation(line: 576, column: 54, scope: !439)
!632 = !DILocation(line: 576, column: 19, scope: !439)
!633 = !DILocation(line: 577, column: 33, scope: !439)
!634 = !DILocation(line: 577, column: 51, scope: !439)
!635 = !DILocation(line: 577, column: 39, scope: !439)
!636 = !DILocation(line: 577, column: 66, scope: !439)
!637 = !DILocation(line: 578, column: 26, scope: !439)
!638 = !DILocation(line: 578, column: 30, scope: !439)
!639 = !DILocation(line: 578, column: 35, scope: !439)
!640 = !DILocation(line: 578, column: 13, scope: !439)
!641 = !DILocation(line: 577, column: 18, scope: !439)
!642 = !DILocation(line: 577, column: 9, scope: !439)
!643 = !DILocation(line: 577, column: 21, scope: !439)
!644 = !DILocation(line: 579, column: 9, scope: !439)
!645 = !DILocation(line: 579, column: 39, scope: !439)
!646 = !DILocation(line: 579, column: 41, scope: !439)
!647 = !DILocation(line: 579, column: 49, scope: !439)
!648 = !DILocation(line: 579, column: 53, scope: !439)
!649 = !DILocation(line: 579, column: 46, scope: !439)
!650 = !DILocation(line: 579, column: 24, scope: !439)
!651 = !DILocation(line: 580, column: 9, scope: !439)
!652 = !DILocation(line: 580, column: 39, scope: !439)
!653 = !DILocation(line: 580, column: 43, scope: !439)
!654 = !DILocation(line: 580, column: 24, scope: !439)
!655 = !DILocation(line: 581, column: 26, scope: !439)
!656 = !DILocation(line: 581, column: 28, scope: !439)
!657 = !DILocation(line: 581, column: 34, scope: !439)
!658 = !DILocation(line: 581, column: 42, scope: !439)
!659 = !DILocation(line: 581, column: 44, scope: !439)
!660 = !DILocation(line: 581, column: 39, scope: !439)
!661 = !DILocation(line: 581, column: 51, scope: !439)
!662 = !DILocation(line: 582, column: 17, scope: !439)
!663 = !DILocation(line: 582, column: 19, scope: !439)
!664 = !DILocation(line: 582, column: 29, scope: !439)
!665 = !DILocation(line: 582, column: 31, scope: !439)
!666 = !DILocation(line: 582, column: 37, scope: !439)
!667 = !DILocation(line: 582, column: 26, scope: !439)
!668 = !DILocation(line: 582, column: 44, scope: !439)
!669 = !DILocation(line: 582, column: 13, scope: !439)
!670 = !DILocation(line: 581, column: 19, scope: !439)
!671 = !DILocation(line: 581, column: 9, scope: !439)
!672 = !DILocation(line: 581, column: 22, scope: !439)
!673 = !DILocation(line: 583, column: 5, scope: !439)
!674 = !DILocation(line: 569, column: 29, scope: !439)
!675 = distinct !{!675, !580, !673, !50, !305}
!676 = !DILocation(line: 584, column: 5, scope: !439)
!677 = !DILocation(line: 584, column: 19, scope: !439)
!678 = !DILocation(line: 585, column: 10, scope: !439)
!679 = !DILocation(line: 585, column: 14, scope: !439)
!680 = !DILocation(line: 585, column: 22, scope: !439)
!681 = !DILocation(line: 585, column: 25, scope: !439)
!682 = !DILocation(line: 585, column: 5, scope: !439)
!683 = !DILocation(line: 587, column: 14, scope: !439)
!684 = !DILocation(line: 587, column: 18, scope: !439)
!685 = !DILocation(line: 587, column: 26, scope: !439)
!686 = !DILocation(line: 587, column: 29, scope: !439)
!687 = !DILocation(line: 587, column: 9, scope: !439)
!688 = !DILocation(line: 588, column: 27, scope: !439)
!689 = !DILocation(line: 588, column: 17, scope: !439)
!690 = !DILocation(line: 588, column: 13, scope: !439)
!691 = !DILocation(line: 588, column: 21, scope: !439)
!692 = !DILocation(line: 588, column: 25, scope: !439)
!693 = !DILocation(line: 589, column: 9, scope: !439)
!694 = !DILocation(line: 587, column: 36, scope: !439)
!695 = distinct !{!695, !687, !693, !50, !305}
!696 = !DILocation(line: 590, column: 5, scope: !439)
!697 = !DILocation(line: 585, column: 32, scope: !439)
!698 = distinct !{!698, !682, !696, !50}
!699 = !DILocation(line: 593, column: 5, scope: !439)
!700 = !DILocation(line: 593, column: 26, scope: !439)
!701 = !DILocation(line: 593, column: 21, scope: !439)
!702 = !DILocation(line: 595, column: 5, scope: !439)
!703 = !DILocation(line: 596, column: 5, scope: !439)
!704 = !DILocation(line: 602, column: 5, scope: !439)
!705 = !DILocation(line: 603, column: 5, scope: !439)
!706 = !DILocation(line: 607, column: 5, scope: !439)
!707 = !DILocation(line: 607, column: 14, scope: !439)
!708 = !DILocation(line: 608, column: 5, scope: !439)
!709 = !DILocation(line: 608, column: 14, scope: !439)
!710 = !DILocation(line: 616, column: 10, scope: !439)
!711 = !DILocation(line: 616, column: 18, scope: !439)
!712 = !DILocation(line: 616, column: 14, scope: !439)
!713 = !DILocation(line: 616, column: 23, scope: !439)
!714 = !DILocation(line: 616, column: 25, scope: !439)
!715 = !DILocation(line: 616, column: 5, scope: !439)
!716 = !DILocation(line: 617, column: 9, scope: !439)
!717 = !DILocation(line: 617, column: 29, scope: !439)
!718 = !DILocation(line: 617, column: 31, scope: !439)
!719 = !DILocation(line: 617, column: 19, scope: !439)
!720 = !DILocation(line: 618, column: 9, scope: !439)
!721 = !DILocation(line: 618, column: 23, scope: !439)
!722 = !DILocation(line: 618, column: 25, scope: !439)
!723 = !DILocation(line: 618, column: 19, scope: !439)
!724 = !DILocation(line: 619, column: 9, scope: !439)
!725 = !DILocation(line: 619, column: 25, scope: !439)
!726 = !DILocation(line: 619, column: 30, scope: !439)
!727 = !DILocation(line: 619, column: 28, scope: !439)
!728 = !DILocation(line: 619, column: 19, scope: !439)
!729 = !DILocation(line: 620, column: 9, scope: !439)
!730 = !DILocation(line: 620, column: 30, scope: !439)
!731 = !DILocation(line: 620, column: 36, scope: !439)
!732 = !DILocation(line: 620, column: 34, scope: !439)
!733 = !DILocation(line: 620, column: 29, scope: !439)
!734 = !DILocation(line: 621, column: 34, scope: !439)
!735 = !DILocation(line: 621, column: 39, scope: !439)
!736 = !DILocation(line: 621, column: 15, scope: !439)
!737 = !DILocation(line: 621, column: 45, scope: !439)
!738 = !DILocation(line: 620, column: 24, scope: !439)
!739 = !DILocation(line: 623, column: 162, scope: !439)
!740 = !DILocation(line: 623, column: 143, scope: !439)
!741 = !DILocation(line: 623, column: 151, scope: !439)
!742 = !DILocation(line: 623, column: 9, scope: !439)
!743 = !DILocation(line: 623, column: 157, scope: !439)
!744 = !DILocation(line: 623, column: 160, scope: !439)
!745 = !DILocation(line: 624, column: 5, scope: !439)
!746 = !DILocation(line: 616, column: 38, scope: !439)
!747 = distinct !{!747, !715, !745, !50}
!748 = !DILocation(line: 625, column: 10, scope: !439)
!749 = !DILocation(line: 625, column: 18, scope: !439)
!750 = !DILocation(line: 625, column: 14, scope: !439)
!751 = !DILocation(line: 625, column: 23, scope: !439)
!752 = !DILocation(line: 625, column: 25, scope: !439)
!753 = !DILocation(line: 625, column: 5, scope: !439)
!754 = !DILocation(line: 626, column: 9, scope: !439)
!755 = !DILocation(line: 626, column: 29, scope: !439)
!756 = !DILocation(line: 626, column: 31, scope: !439)
!757 = !DILocation(line: 626, column: 19, scope: !439)
!758 = !DILocation(line: 627, column: 9, scope: !439)
!759 = !DILocation(line: 627, column: 23, scope: !439)
!760 = !DILocation(line: 627, column: 25, scope: !439)
!761 = !DILocation(line: 627, column: 19, scope: !439)
!762 = !DILocation(line: 628, column: 9, scope: !439)
!763 = !DILocation(line: 628, column: 26, scope: !439)
!764 = !DILocation(line: 628, column: 31, scope: !439)
!765 = !DILocation(line: 628, column: 29, scope: !439)
!766 = !DILocation(line: 628, column: 19, scope: !439)
!767 = !DILocation(line: 629, column: 9, scope: !439)
!768 = !DILocation(line: 629, column: 30, scope: !439)
!769 = !DILocation(line: 629, column: 37, scope: !439)
!770 = !DILocation(line: 629, column: 35, scope: !439)
!771 = !DILocation(line: 629, column: 29, scope: !439)
!772 = !DILocation(line: 629, column: 42, scope: !439)
!773 = !DILocation(line: 629, column: 50, scope: !439)
!774 = !DILocation(line: 629, column: 52, scope: !439)
!775 = !DILocation(line: 629, column: 19, scope: !439)
!776 = !DILocation(line: 630, column: 9, scope: !439)
!777 = !DILocation(line: 630, column: 48, scope: !439)
!778 = !DILocation(line: 630, column: 63, scope: !439)
!779 = !DILocation(line: 630, column: 73, scope: !439)
!780 = !DILocation(line: 630, column: 71, scope: !439)
!781 = !DILocation(line: 630, column: 88, scope: !439)
!782 = !DILocation(line: 630, column: 29, scope: !439)
!783 = !DILocation(line: 630, column: 24, scope: !439)
!784 = !DILocation(line: 631, column: 9, scope: !439)
!785 = !DILocation(line: 631, column: 26, scope: !439)
!786 = !DILocation(line: 631, column: 28, scope: !439)
!787 = !DILocation(line: 631, column: 25, scope: !439)
!788 = !DILocation(line: 632, column: 28, scope: !439)
!789 = !DILocation(line: 632, column: 62, scope: !439)
!790 = !DILocation(line: 632, column: 65, scope: !439)
!791 = !DILocation(line: 632, column: 61, scope: !439)
!792 = !DILocation(line: 632, column: 15, scope: !439)
!793 = !DILocation(line: 633, column: 28, scope: !439)
!794 = !DILocation(line: 633, column: 62, scope: !439)
!795 = !DILocation(line: 633, column: 65, scope: !439)
!796 = !DILocation(line: 633, column: 61, scope: !439)
!797 = !DILocation(line: 633, column: 15, scope: !439)
!798 = !DILocation(line: 631, column: 21, scope: !439)
!799 = !DILocation(line: 634, column: 188, scope: !439)
!800 = !DILocation(line: 634, column: 195, scope: !439)
!801 = !DILocation(line: 634, column: 193, scope: !439)
!802 = !DILocation(line: 634, column: 187, scope: !439)
!803 = !DILocation(line: 634, column: 200, scope: !439)
!804 = !DILocation(line: 634, column: 168, scope: !439)
!805 = !DILocation(line: 634, column: 176, scope: !439)
!806 = !DILocation(line: 634, column: 9, scope: !439)
!807 = !DILocation(line: 634, column: 182, scope: !439)
!808 = !DILocation(line: 634, column: 185, scope: !439)
!809 = !DILocation(line: 635, column: 5, scope: !439)
!810 = !DILocation(line: 625, column: 38, scope: !439)
!811 = distinct !{!811, !753, !809, !50}
!812 = !DILocation(line: 638, column: 9, scope: !439)
!813 = !DILocation(line: 638, column: 54, scope: !439)
!814 = !DILocation(line: 638, column: 57, scope: !439)
!815 = !DILocation(line: 638, column: 34, scope: !439)
!816 = !DILocation(line: 639, column: 9, scope: !439)
!817 = !DILocation(line: 639, column: 40, scope: !439)
!818 = !DILocation(line: 639, column: 42, scope: !439)
!819 = !DILocation(line: 639, column: 34, scope: !439)
!820 = !DILocation(line: 641, column: 14, scope: !439)
!821 = !DILocation(line: 641, column: 18, scope: !439)
!822 = !DILocation(line: 641, column: 25, scope: !439)
!823 = !DILocation(line: 641, column: 27, scope: !439)
!824 = !DILocation(line: 641, column: 9, scope: !439)
!825 = !DILocation(line: 642, column: 13, scope: !439)
!826 = !DILocation(line: 642, column: 50, scope: !439)
!827 = !DILocation(line: 642, column: 66, scope: !439)
!828 = !DILocation(line: 642, column: 57, scope: !439)
!829 = !DILocation(line: 642, column: 32, scope: !439)
!830 = !DILocation(line: 642, column: 27, scope: !439)
!831 = !DILocation(line: 643, column: 56, scope: !439)
!832 = !DILocation(line: 643, column: 49, scope: !439)
!833 = !DILocation(line: 643, column: 61, scope: !439)
!834 = !DILocation(line: 643, column: 25, scope: !439)
!835 = !DILocation(line: 643, column: 43, scope: !439)
!836 = !DILocation(line: 643, column: 33, scope: !439)
!837 = !DILocation(line: 643, column: 13, scope: !439)
!838 = !DILocation(line: 643, column: 47, scope: !439)
!839 = !DILocation(line: 644, column: 67, scope: !439)
!840 = !DILocation(line: 644, column: 82, scope: !439)
!841 = !DILocation(line: 644, column: 73, scope: !439)
!842 = !DILocation(line: 644, column: 49, scope: !439)
!843 = !DILocation(line: 644, column: 25, scope: !439)
!844 = !DILocation(line: 644, column: 43, scope: !439)
!845 = !DILocation(line: 644, column: 33, scope: !439)
!846 = !DILocation(line: 644, column: 13, scope: !439)
!847 = !DILocation(line: 644, column: 47, scope: !439)
!848 = !DILocation(line: 645, column: 9, scope: !439)
!849 = !DILocation(line: 641, column: 33, scope: !439)
!850 = distinct !{!850, !824, !848, !50, !305}
!851 = !DILocation(line: 646, column: 5, scope: !439)
!852 = !DILocation(line: 647, column: 5, scope: !439)
!853 = !DILocation(line: 649, column: 10, scope: !439)
!854 = !DILocation(line: 649, column: 14, scope: !439)
!855 = !DILocation(line: 649, column: 21, scope: !439)
!856 = !DILocation(line: 649, column: 25, scope: !439)
!857 = !DILocation(line: 649, column: 23, scope: !439)
!858 = !DILocation(line: 649, column: 5, scope: !439)
!859 = !DILocation(line: 655, column: 14, scope: !439)
!860 = !DILocation(line: 655, column: 18, scope: !439)
!861 = !DILocation(line: 655, column: 25, scope: !439)
!862 = !DILocation(line: 655, column: 27, scope: !439)
!863 = !DILocation(line: 655, column: 9, scope: !439)
!864 = !DILocation(line: 656, column: 13, scope: !439)
!865 = !DILocation(line: 656, column: 32, scope: !439)
!866 = !DILocation(line: 656, column: 30, scope: !439)
!867 = !DILocation(line: 656, column: 36, scope: !439)
!868 = !DILocation(line: 656, column: 34, scope: !439)
!869 = !DILocation(line: 656, column: 23, scope: !439)
!870 = !DILocation(line: 657, column: 13, scope: !439)
!871 = !DILocation(line: 657, column: 32, scope: !439)
!872 = !DILocation(line: 657, column: 34, scope: !439)
!873 = !DILocation(line: 657, column: 40, scope: !439)
!874 = !DILocation(line: 657, column: 44, scope: !439)
!875 = !DILocation(line: 657, column: 46, scope: !439)
!876 = !DILocation(line: 657, column: 53, scope: !439)
!877 = !DILocation(line: 657, column: 50, scope: !439)
!878 = !DILocation(line: 0, scope: !439)
!879 = !DILocation(line: 657, column: 24, scope: !439)
!880 = !{!881, !881, i64 0}
!881 = !{!"bool", !10, i64 0}
!882 = !DILocation(line: 661, column: 13, scope: !439)
!883 = !DILocation(line: 661, column: 49, scope: !439)
!884 = !DILocation(line: 661, column: 51, scope: !439)
!885 = !DILocation(line: 661, column: 48, scope: !439)
!886 = !DILocation(line: 661, column: 45, scope: !439)
!887 = !DILocation(line: 661, column: 32, scope: !439)
!888 = !DILocation(line: 662, column: 13, scope: !439)
!889 = !DILocation(line: 662, column: 49, scope: !439)
!890 = !DILocation(line: 662, column: 51, scope: !439)
!891 = !DILocation(line: 662, column: 48, scope: !439)
!892 = !DILocation(line: 662, column: 45, scope: !439)
!893 = !DILocation(line: 662, column: 32, scope: !439)
!894 = !DILocation(line: 671, column: 18, scope: !439)
!895 = !DILocation(line: 671, column: 22, scope: !439)
!896 = !DILocation(line: 671, column: 30, scope: !439)
!897 = !DILocation(line: 671, column: 33, scope: !439)
!898 = !DILocation(line: 671, column: 13, scope: !439)
!899 = !DILocation(line: 673, column: 22, scope: !439)
!900 = !DILocation(line: 673, column: 26, scope: !439)
!901 = !DILocation(line: 673, column: 33, scope: !439)
!902 = !DILocation(line: 673, column: 35, scope: !439)
!903 = !DILocation(line: 673, column: 17, scope: !439)
!904 = !DILocation(line: 674, column: 21, scope: !439)
!905 = !DILocation(line: 675, column: 36, scope: !439)
!906 = !DILocation(line: 675, column: 48, scope: !439)
!907 = !DILocation(line: 675, column: 51, scope: !439)
!908 = !DILocation(line: 675, column: 46, scope: !439)
!909 = !DILocation(line: 675, column: 62, scope: !439)
!910 = !DILocation(line: 675, column: 60, scope: !439)
!911 = !DILocation(line: 675, column: 56, scope: !439)
!912 = !DILocation(line: 675, column: 70, scope: !439)
!913 = !DILocation(line: 675, column: 68, scope: !439)
!914 = !DILocation(line: 675, column: 73, scope: !439)
!915 = !DILocation(line: 674, column: 36, scope: !439)
!916 = !DILocation(line: 676, column: 59, scope: !439)
!917 = !DILocation(line: 676, column: 68, scope: !439)
!918 = !DILocation(line: 676, column: 43, scope: !439)
!919 = !DILocation(line: 676, column: 34, scope: !439)
!920 = !DILocation(line: 676, column: 21, scope: !439)
!921 = !DILocation(line: 676, column: 38, scope: !439)
!922 = !DILocation(line: 676, column: 41, scope: !439)
!923 = !DILocation(line: 677, column: 17, scope: !439)
!924 = !DILocation(line: 673, column: 41, scope: !439)
!925 = distinct !{!925, !903, !923, !50, !305}
!926 = !DILocation(line: 678, column: 13, scope: !439)
!927 = !DILocation(line: 671, column: 40, scope: !439)
!928 = distinct !{!928, !898, !926, !50, !305}
!929 = !DILocation(line: 680, column: 18, scope: !439)
!930 = !DILocation(line: 680, column: 22, scope: !439)
!931 = !DILocation(line: 680, column: 30, scope: !439)
!932 = !DILocation(line: 680, column: 33, scope: !439)
!933 = !DILocation(line: 680, column: 13, scope: !439)
!934 = !DILocation(line: 681, column: 17, scope: !439)
!935 = !DILocation(line: 682, column: 32, scope: !439)
!936 = !DILocation(line: 682, column: 44, scope: !439)
!937 = !DILocation(line: 682, column: 47, scope: !439)
!938 = !DILocation(line: 682, column: 42, scope: !439)
!939 = !DILocation(line: 682, column: 54, scope: !439)
!940 = !DILocation(line: 682, column: 52, scope: !439)
!941 = !DILocation(line: 682, column: 62, scope: !439)
!942 = !DILocation(line: 681, column: 32, scope: !439)
!943 = !DILocation(line: 683, column: 52, scope: !439)
!944 = !DILocation(line: 683, column: 61, scope: !439)
!945 = !DILocation(line: 683, column: 36, scope: !439)
!946 = !DILocation(line: 683, column: 30, scope: !439)
!947 = !DILocation(line: 683, column: 17, scope: !439)
!948 = !DILocation(line: 683, column: 34, scope: !439)
!949 = !DILocation(line: 684, column: 13, scope: !439)
!950 = !DILocation(line: 680, column: 40, scope: !439)
!951 = distinct !{!951, !933, !949, !50, !305}
!952 = !DILocation(line: 690, column: 17, scope: !439)
!953 = !DILocation(line: 690, column: 63, scope: !439)
!954 = !DILocation(line: 690, column: 79, scope: !439)
!955 = !DILocation(line: 690, column: 84, scope: !439)
!956 = !DILocation(line: 690, column: 82, scope: !439)
!957 = !DILocation(line: 690, column: 66, scope: !439)
!958 = !DILocation(line: 690, column: 87, scope: !439)
!959 = !DILocation(line: 690, column: 91, scope: !439)
!960 = !DILocation(line: 690, column: 42, scope: !439)
!961 = !DILocation(line: 691, column: 17, scope: !439)
!962 = !DILocation(line: 691, column: 48, scope: !439)
!963 = !DILocation(line: 691, column: 63, scope: !439)
!964 = !DILocation(line: 691, column: 65, scope: !439)
!965 = !DILocation(line: 691, column: 50, scope: !439)
!966 = !DILocation(line: 691, column: 71, scope: !439)
!967 = !DILocation(line: 691, column: 88, scope: !439)
!968 = !DILocation(line: 691, column: 90, scope: !439)
!969 = !DILocation(line: 691, column: 75, scope: !439)
!970 = !DILocation(line: 691, column: 95, scope: !439)
!971 = !DILocation(line: 691, column: 42, scope: !439)
!972 = !DILocation(line: 693, column: 22, scope: !439)
!973 = !DILocation(line: 693, column: 26, scope: !439)
!974 = !DILocation(line: 693, column: 33, scope: !439)
!975 = !DILocation(line: 693, column: 35, scope: !439)
!976 = !DILocation(line: 693, column: 17, scope: !439)
!977 = !DILocation(line: 694, column: 21, scope: !439)
!978 = !DILocation(line: 694, column: 58, scope: !439)
!979 = !DILocation(line: 694, column: 74, scope: !439)
!980 = !DILocation(line: 694, column: 65, scope: !439)
!981 = !DILocation(line: 694, column: 40, scope: !439)
!982 = !DILocation(line: 694, column: 35, scope: !439)
!983 = !DILocation(line: 695, column: 38, scope: !439)
!984 = !DILocation(line: 695, column: 31, scope: !439)
!985 = !DILocation(line: 695, column: 43, scope: !439)
!986 = !DILocation(line: 695, column: 26, scope: !439)
!987 = !DILocation(line: 695, column: 21, scope: !439)
!988 = !DILocation(line: 695, column: 29, scope: !439)
!989 = !DILocation(line: 696, column: 49, scope: !439)
!990 = !DILocation(line: 696, column: 64, scope: !439)
!991 = !DILocation(line: 696, column: 55, scope: !439)
!992 = !DILocation(line: 696, column: 31, scope: !439)
!993 = !DILocation(line: 696, column: 26, scope: !439)
!994 = !DILocation(line: 696, column: 21, scope: !439)
!995 = !DILocation(line: 696, column: 29, scope: !439)
!996 = !DILocation(line: 697, column: 17, scope: !439)
!997 = !DILocation(line: 693, column: 41, scope: !439)
!998 = distinct !{!998, !976, !996, !50, !305}
!999 = !DILocation(line: 710, column: 21, scope: !439)
!1000 = !DILocation(line: 710, column: 28, scope: !439)
!1001 = !DILocation(line: 710, column: 45, scope: !439)
!1002 = !DILocation(line: 711, column: 21, scope: !439)
!1003 = !DILocation(line: 709, column: 17, scope: !439)
!1004 = !DILocation(line: 713, column: 21, scope: !439)
!1005 = !DILocation(line: 713, column: 28, scope: !439)
!1006 = !DILocation(line: 713, column: 45, scope: !439)
!1007 = !DILocation(line: 714, column: 21, scope: !439)
!1008 = !DILocation(line: 712, column: 17, scope: !439)
!1009 = !DILocation(line: 722, column: 17, scope: !439)
!1010 = !DILocation(line: 726, column: 23, scope: !439)
!1011 = !DILocation(line: 728, column: 22, scope: !439)
!1012 = !DILocation(line: 728, column: 26, scope: !439)
!1013 = !DILocation(line: 728, column: 33, scope: !439)
!1014 = !DILocation(line: 728, column: 35, scope: !439)
!1015 = !DILocation(line: 728, column: 17, scope: !439)
!1016 = !DILocation(line: 729, column: 62, scope: !439)
!1017 = !DILocation(line: 729, column: 57, scope: !439)
!1018 = !DILocation(line: 729, column: 33, scope: !439)
!1019 = !DILocation(line: 729, column: 51, scope: !439)
!1020 = !DILocation(line: 729, column: 41, scope: !439)
!1021 = !DILocation(line: 729, column: 21, scope: !439)
!1022 = !DILocation(line: 729, column: 55, scope: !439)
!1023 = !DILocation(line: 730, column: 62, scope: !439)
!1024 = !DILocation(line: 730, column: 57, scope: !439)
!1025 = !DILocation(line: 730, column: 33, scope: !439)
!1026 = !DILocation(line: 730, column: 51, scope: !439)
!1027 = !DILocation(line: 730, column: 41, scope: !439)
!1028 = !DILocation(line: 730, column: 21, scope: !439)
!1029 = !DILocation(line: 730, column: 55, scope: !439)
!1030 = !DILocation(line: 731, column: 17, scope: !439)
!1031 = !DILocation(line: 728, column: 41, scope: !439)
!1032 = distinct !{!1032, !1015, !1030, !50, !305}
!1033 = !DILocation(line: 733, column: 17, scope: !439)
!1034 = !DILocation(line: 734, column: 22, scope: !439)
!1035 = !{i8 0, i8 2}
!1036 = !DILocation(line: 734, column: 21, scope: !439)
!1037 = !DILocation(line: 744, column: 21, scope: !439)
!1038 = !DILocation(line: 744, column: 39, scope: !439)
!1039 = !DILocation(line: 744, column: 42, scope: !439)
!1040 = !DILocation(line: 744, column: 31, scope: !439)
!1041 = !DILocation(line: 745, column: 21, scope: !439)
!1042 = !DILocation(line: 745, column: 68, scope: !439)
!1043 = !DILocation(line: 745, column: 84, scope: !439)
!1044 = !DILocation(line: 745, column: 92, scope: !439)
!1045 = !DILocation(line: 745, column: 90, scope: !439)
!1046 = !DILocation(line: 745, column: 71, scope: !439)
!1047 = !DILocation(line: 745, column: 95, scope: !439)
!1048 = !DILocation(line: 745, column: 46, scope: !439)
!1049 = !DILocation(line: 747, column: 26, scope: !439)
!1050 = !DILocation(line: 747, column: 30, scope: !439)
!1051 = !DILocation(line: 747, column: 37, scope: !439)
!1052 = !DILocation(line: 747, column: 39, scope: !439)
!1053 = !DILocation(line: 747, column: 21, scope: !439)
!1054 = !DILocation(line: 748, column: 53, scope: !439)
!1055 = !DILocation(line: 748, column: 70, scope: !439)
!1056 = !DILocation(line: 748, column: 61, scope: !439)
!1057 = !DILocation(line: 748, column: 35, scope: !439)
!1058 = !DILocation(line: 748, column: 30, scope: !439)
!1059 = !DILocation(line: 748, column: 25, scope: !439)
!1060 = !DILocation(line: 748, column: 33, scope: !439)
!1061 = !DILocation(line: 749, column: 21, scope: !439)
!1062 = !DILocation(line: 747, column: 45, scope: !439)
!1063 = distinct !{!1063, !1053, !1061, !50, !305}
!1064 = !DILocation(line: 750, column: 21, scope: !439)
!1065 = !DILocation(line: 750, column: 38, scope: !439)
!1066 = !DILocation(line: 750, column: 42, scope: !439)
!1067 = !DILocation(line: 750, column: 40, scope: !439)
!1068 = !DILocation(line: 750, column: 31, scope: !439)
!1069 = !DILocation(line: 751, column: 21, scope: !439)
!1070 = !DILocation(line: 751, column: 38, scope: !439)
!1071 = !DILocation(line: 751, column: 40, scope: !439)
!1072 = !DILocation(line: 751, column: 31, scope: !439)
!1073 = !DILocation(line: 752, column: 21, scope: !439)
!1074 = !DILocation(line: 752, column: 53, scope: !439)
!1075 = !DILocation(line: 752, column: 68, scope: !439)
!1076 = !DILocation(line: 752, column: 73, scope: !439)
!1077 = !DILocation(line: 752, column: 55, scope: !439)
!1078 = !DILocation(line: 752, column: 79, scope: !439)
!1079 = !DILocation(line: 752, column: 96, scope: !439)
!1080 = !DILocation(line: 752, column: 101, scope: !439)
!1081 = !DILocation(line: 752, column: 83, scope: !439)
!1082 = !DILocation(line: 752, column: 46, scope: !439)
!1083 = !DILocation(line: 754, column: 26, scope: !439)
!1084 = !DILocation(line: 754, column: 30, scope: !439)
!1085 = !DILocation(line: 754, column: 37, scope: !439)
!1086 = !DILocation(line: 754, column: 39, scope: !439)
!1087 = !DILocation(line: 754, column: 21, scope: !439)
!1088 = !DILocation(line: 755, column: 53, scope: !439)
!1089 = !DILocation(line: 755, column: 69, scope: !439)
!1090 = !DILocation(line: 755, column: 60, scope: !439)
!1091 = !DILocation(line: 755, column: 35, scope: !439)
!1092 = !DILocation(line: 755, column: 30, scope: !439)
!1093 = !DILocation(line: 755, column: 25, scope: !439)
!1094 = !DILocation(line: 755, column: 33, scope: !439)
!1095 = !DILocation(line: 756, column: 21, scope: !439)
!1096 = !DILocation(line: 754, column: 45, scope: !439)
!1097 = distinct !{!1097, !1087, !1095, !50, !305}
!1098 = !DILocation(line: 757, column: 21, scope: !439)
!1099 = !DILocation(line: 757, column: 40, scope: !439)
!1100 = !DILocation(line: 757, column: 46, scope: !439)
!1101 = !DILocation(line: 757, column: 50, scope: !439)
!1102 = !DILocation(line: 757, column: 43, scope: !439)
!1103 = !DILocation(line: 757, column: 31, scope: !439)
!1104 = !DILocation(line: 758, column: 21, scope: !439)
!1105 = !DILocation(line: 758, column: 42, scope: !439)
!1106 = !DILocation(line: 758, column: 51, scope: !439)
!1107 = !DILocation(line: 758, column: 49, scope: !439)
!1108 = !DILocation(line: 758, column: 41, scope: !439)
!1109 = !DILocation(line: 758, column: 56, scope: !439)
!1110 = !DILocation(line: 758, column: 66, scope: !439)
!1111 = !DILocation(line: 758, column: 68, scope: !439)
!1112 = !DILocation(line: 758, column: 31, scope: !439)
!1113 = !DILocation(line: 759, column: 48, scope: !439)
!1114 = !DILocation(line: 759, column: 64, scope: !439)
!1115 = !DILocation(line: 759, column: 72, scope: !439)
!1116 = !DILocation(line: 759, column: 70, scope: !439)
!1117 = !DILocation(line: 759, column: 29, scope: !439)
!1118 = !DILocation(line: 759, column: 76, scope: !439)
!1119 = !DILocation(line: 759, column: 86, scope: !439)
!1120 = !DILocation(line: 759, column: 90, scope: !439)
!1121 = !DILocation(line: 759, column: 27, scope: !439)
!1122 = !DILocation(line: 760, column: 21, scope: !439)
!1123 = !DILocation(line: 760, column: 40, scope: !439)
!1124 = !DILocation(line: 760, column: 46, scope: !439)
!1125 = !DILocation(line: 760, column: 50, scope: !439)
!1126 = !DILocation(line: 760, column: 43, scope: !439)
!1127 = !DILocation(line: 760, column: 31, scope: !439)
!1128 = !DILocation(line: 761, column: 21, scope: !439)
!1129 = !DILocation(line: 761, column: 41, scope: !439)
!1130 = !DILocation(line: 761, column: 50, scope: !439)
!1131 = !DILocation(line: 761, column: 48, scope: !439)
!1132 = !DILocation(line: 761, column: 40, scope: !439)
!1133 = !DILocation(line: 761, column: 55, scope: !439)
!1134 = !DILocation(line: 761, column: 65, scope: !439)
!1135 = !DILocation(line: 761, column: 67, scope: !439)
!1136 = !DILocation(line: 761, column: 31, scope: !439)
!1137 = !DILocation(line: 762, column: 48, scope: !439)
!1138 = !DILocation(line: 762, column: 63, scope: !439)
!1139 = !DILocation(line: 762, column: 72, scope: !439)
!1140 = !DILocation(line: 762, column: 70, scope: !439)
!1141 = !DILocation(line: 762, column: 87, scope: !439)
!1142 = !DILocation(line: 762, column: 29, scope: !439)
!1143 = !DILocation(line: 763, column: 27, scope: !439)
!1144 = !DILocation(line: 763, column: 32, scope: !439)
!1145 = !DILocation(line: 763, column: 40, scope: !439)
!1146 = !DILocation(line: 763, column: 45, scope: !439)
!1147 = !DILocation(line: 762, column: 27, scope: !439)
!1148 = !DILocation(line: 764, column: 17, scope: !439)
!1149 = !DILocation(line: 765, column: 13, scope: !439)
!1150 = !DILocation(line: 770, column: 21, scope: !439)
!1151 = !DILocation(line: 770, column: 28, scope: !439)
!1152 = !DILocation(line: 770, column: 45, scope: !439)
!1153 = !DILocation(line: 771, column: 21, scope: !439)
!1154 = !DILocation(line: 769, column: 17, scope: !439)
!1155 = !DILocation(line: 773, column: 21, scope: !439)
!1156 = !DILocation(line: 773, column: 28, scope: !439)
!1157 = !DILocation(line: 773, column: 45, scope: !439)
!1158 = !DILocation(line: 774, column: 21, scope: !439)
!1159 = !DILocation(line: 772, column: 17, scope: !439)
!1160 = !DILocation(line: 784, column: 22, scope: !439)
!1161 = !DILocation(line: 784, column: 26, scope: !439)
!1162 = !DILocation(line: 784, column: 34, scope: !439)
!1163 = !DILocation(line: 784, column: 37, scope: !439)
!1164 = !DILocation(line: 784, column: 17, scope: !439)
!1165 = !DILocation(line: 785, column: 21, scope: !439)
!1166 = !DILocation(line: 787, column: 26, scope: !439)
!1167 = !DILocation(line: 787, column: 30, scope: !439)
!1168 = !DILocation(line: 787, column: 37, scope: !439)
!1169 = !DILocation(line: 787, column: 39, scope: !439)
!1170 = !DILocation(line: 787, column: 21, scope: !439)
!1171 = !DILocation(line: 788, column: 25, scope: !439)
!1172 = !DILocation(line: 789, column: 40, scope: !439)
!1173 = !DILocation(line: 789, column: 52, scope: !439)
!1174 = !DILocation(line: 789, column: 55, scope: !439)
!1175 = !DILocation(line: 789, column: 50, scope: !439)
!1176 = !DILocation(line: 789, column: 66, scope: !439)
!1177 = !DILocation(line: 789, column: 64, scope: !439)
!1178 = !DILocation(line: 789, column: 60, scope: !439)
!1179 = !DILocation(line: 789, column: 74, scope: !439)
!1180 = !DILocation(line: 789, column: 72, scope: !439)
!1181 = !DILocation(line: 789, column: 77, scope: !439)
!1182 = !DILocation(line: 788, column: 40, scope: !439)
!1183 = !DILocation(line: 790, column: 53, scope: !439)
!1184 = !DILocation(line: 790, column: 62, scope: !439)
!1185 = !DILocation(line: 790, column: 37, scope: !439)
!1186 = !DILocation(line: 790, column: 32, scope: !439)
!1187 = !DILocation(line: 790, column: 25, scope: !439)
!1188 = !DILocation(line: 790, column: 35, scope: !439)
!1189 = !DILocation(line: 791, column: 21, scope: !439)
!1190 = !DILocation(line: 787, column: 45, scope: !439)
!1191 = distinct !{!1191, !1170, !1189, !50, !305}
!1192 = !DILocation(line: 793, column: 26, scope: !439)
!1193 = !DILocation(line: 793, column: 30, scope: !439)
!1194 = !DILocation(line: 793, column: 38, scope: !439)
!1195 = !DILocation(line: 793, column: 41, scope: !439)
!1196 = !DILocation(line: 793, column: 21, scope: !439)
!1197 = !DILocation(line: 794, column: 25, scope: !439)
!1198 = !DILocation(line: 795, column: 40, scope: !439)
!1199 = !DILocation(line: 795, column: 52, scope: !439)
!1200 = !DILocation(line: 795, column: 55, scope: !439)
!1201 = !DILocation(line: 795, column: 50, scope: !439)
!1202 = !DILocation(line: 795, column: 62, scope: !439)
!1203 = !DILocation(line: 795, column: 60, scope: !439)
!1204 = !DILocation(line: 795, column: 70, scope: !439)
!1205 = !DILocation(line: 794, column: 40, scope: !439)
!1206 = !DILocation(line: 796, column: 25, scope: !439)
!1207 = !DILocation(line: 796, column: 59, scope: !439)
!1208 = !DILocation(line: 796, column: 68, scope: !439)
!1209 = !DILocation(line: 796, column: 43, scope: !439)
!1210 = !DILocation(line: 796, column: 37, scope: !439)
!1211 = !DILocation(line: 797, column: 25, scope: !439)
!1212 = !DILocation(line: 798, column: 50, scope: !439)
!1213 = !DILocation(line: 798, column: 59, scope: !439)
!1214 = !DILocation(line: 798, column: 36, scope: !439)
!1215 = !DILocation(line: 797, column: 37, scope: !439)
!1216 = !DILocation(line: 800, column: 30, scope: !439)
!1217 = !DILocation(line: 800, column: 34, scope: !439)
!1218 = !DILocation(line: 800, column: 41, scope: !439)
!1219 = !DILocation(line: 800, column: 43, scope: !439)
!1220 = !DILocation(line: 800, column: 25, scope: !439)
!1221 = !DILocation(line: 801, column: 29, scope: !439)
!1222 = !DILocation(line: 801, column: 62, scope: !439)
!1223 = !DILocation(line: 801, column: 55, scope: !439)
!1224 = !DILocation(line: 801, column: 66, scope: !439)
!1225 = !DILocation(line: 801, column: 45, scope: !439)
!1226 = !DILocation(line: 801, column: 41, scope: !439)
!1227 = !DILocation(line: 803, column: 43, scope: !439)
!1228 = !DILocation(line: 803, column: 46, scope: !439)
!1229 = !DILocation(line: 803, column: 56, scope: !439)
!1230 = !DILocation(line: 803, column: 52, scope: !439)
!1231 = !DILocation(line: 803, column: 60, scope: !439)
!1232 = !DILocation(line: 803, column: 64, scope: !439)
!1233 = !DILocation(line: 803, column: 33, scope: !439)
!1234 = !DILocation(line: 802, column: 33, scope: !439)
!1235 = !DILocation(line: 802, column: 29, scope: !439)
!1236 = !DILocation(line: 802, column: 37, scope: !439)
!1237 = !DILocation(line: 802, column: 41, scope: !439)
!1238 = !DILocation(line: 802, column: 44, scope: !439)
!1239 = !DILocation(line: 804, column: 25, scope: !439)
!1240 = !DILocation(line: 800, column: 49, scope: !439)
!1241 = distinct !{!1241, !1220, !1239, !50, !305}
!1242 = !DILocation(line: 805, column: 21, scope: !439)
!1243 = !DILocation(line: 793, column: 48, scope: !439)
!1244 = distinct !{!1244, !1196, !1242, !50, !305}
!1245 = !DILocation(line: 806, column: 17, scope: !439)
!1246 = !DILocation(line: 784, column: 44, scope: !439)
!1247 = distinct !{!1247, !1164, !1245, !50, !305}
!1248 = !DILocation(line: 863, column: 17, scope: !439)
!1249 = !DILocation(line: 864, column: 22, scope: !439)
!1250 = !DILocation(line: 864, column: 21, scope: !439)
!1251 = !DILocation(line: 869, column: 27, scope: !439)
!1252 = !DILocation(line: 871, column: 26, scope: !439)
!1253 = !DILocation(line: 871, column: 30, scope: !439)
!1254 = !DILocation(line: 871, column: 37, scope: !439)
!1255 = !DILocation(line: 871, column: 39, scope: !439)
!1256 = !DILocation(line: 871, column: 21, scope: !439)
!1257 = !DILocation(line: 872, column: 68, scope: !439)
!1258 = !DILocation(line: 872, column: 61, scope: !439)
!1259 = !DILocation(line: 872, column: 78, scope: !439)
!1260 = !DILocation(line: 872, column: 73, scope: !439)
!1261 = !DILocation(line: 872, column: 37, scope: !439)
!1262 = !DILocation(line: 872, column: 55, scope: !439)
!1263 = !DILocation(line: 872, column: 45, scope: !439)
!1264 = !DILocation(line: 872, column: 25, scope: !439)
!1265 = !DILocation(line: 872, column: 59, scope: !439)
!1266 = !DILocation(line: 873, column: 66, scope: !439)
!1267 = !DILocation(line: 873, column: 61, scope: !439)
!1268 = !DILocation(line: 873, column: 37, scope: !439)
!1269 = !DILocation(line: 873, column: 55, scope: !439)
!1270 = !DILocation(line: 873, column: 45, scope: !439)
!1271 = !DILocation(line: 873, column: 25, scope: !439)
!1272 = !DILocation(line: 873, column: 59, scope: !439)
!1273 = !DILocation(line: 874, column: 21, scope: !439)
!1274 = !DILocation(line: 871, column: 45, scope: !439)
!1275 = distinct !{!1275, !1256, !1273, !50, !305}
!1276 = !DILocation(line: 875, column: 21, scope: !439)
!1277 = !DILocation(line: 875, column: 38, scope: !439)
!1278 = !DILocation(line: 875, column: 40, scope: !439)
!1279 = !DILocation(line: 875, column: 31, scope: !439)
!1280 = !DILocation(line: 876, column: 21, scope: !439)
!1281 = !DILocation(line: 876, column: 55, scope: !439)
!1282 = !DILocation(line: 876, column: 60, scope: !439)
!1283 = !DILocation(line: 876, column: 54, scope: !439)
!1284 = !DILocation(line: 876, column: 51, scope: !439)
!1285 = !DILocation(line: 876, column: 40, scope: !439)
!1286 = !DILocation(line: 878, column: 25, scope: !439)
!1287 = !DILocation(line: 878, column: 45, scope: !439)
!1288 = !DILocation(line: 878, column: 51, scope: !439)
!1289 = !DILocation(line: 878, column: 55, scope: !439)
!1290 = !DILocation(line: 878, column: 48, scope: !439)
!1291 = !DILocation(line: 878, column: 35, scope: !439)
!1292 = !DILocation(line: 879, column: 74, scope: !439)
!1293 = !DILocation(line: 879, column: 84, scope: !439)
!1294 = !DILocation(line: 879, column: 82, scope: !439)
!1295 = !DILocation(line: 879, column: 73, scope: !439)
!1296 = !DILocation(line: 879, column: 89, scope: !439)
!1297 = !DILocation(line: 879, column: 38, scope: !439)
!1298 = !DILocation(line: 879, column: 46, scope: !439)
!1299 = !DILocation(line: 879, column: 50, scope: !439)
!1300 = !DILocation(line: 879, column: 56, scope: !439)
!1301 = !DILocation(line: 879, column: 25, scope: !439)
!1302 = !DILocation(line: 879, column: 62, scope: !439)
!1303 = !DILocation(line: 879, column: 66, scope: !439)
!1304 = !DILocation(line: 879, column: 71, scope: !439)
!1305 = !DILocation(line: 880, column: 21, scope: !439)
!1306 = !DILocation(line: 881, column: 21, scope: !439)
!1307 = !DILocation(line: 881, column: 55, scope: !439)
!1308 = !DILocation(line: 881, column: 60, scope: !439)
!1309 = !DILocation(line: 881, column: 54, scope: !439)
!1310 = !DILocation(line: 881, column: 51, scope: !439)
!1311 = !DILocation(line: 881, column: 40, scope: !439)
!1312 = !DILocation(line: 883, column: 25, scope: !439)
!1313 = !DILocation(line: 883, column: 45, scope: !439)
!1314 = !DILocation(line: 883, column: 51, scope: !439)
!1315 = !DILocation(line: 883, column: 55, scope: !439)
!1316 = !DILocation(line: 883, column: 48, scope: !439)
!1317 = !DILocation(line: 883, column: 35, scope: !439)
!1318 = !DILocation(line: 884, column: 25, scope: !439)
!1319 = !DILocation(line: 884, column: 43, scope: !439)
!1320 = !DILocation(line: 884, column: 47, scope: !439)
!1321 = !DILocation(line: 884, column: 35, scope: !439)
!1322 = !DILocation(line: 885, column: 25, scope: !439)
!1323 = !DILocation(line: 885, column: 42, scope: !439)
!1324 = !DILocation(line: 885, column: 48, scope: !439)
!1325 = !DILocation(line: 885, column: 41, scope: !439)
!1326 = !DILocation(line: 886, column: 44, scope: !439)
!1327 = !DILocation(line: 886, column: 78, scope: !439)
!1328 = !DILocation(line: 886, column: 84, scope: !439)
!1329 = !DILocation(line: 886, column: 77, scope: !439)
!1330 = !DILocation(line: 886, column: 31, scope: !439)
!1331 = !DILocation(line: 887, column: 44, scope: !439)
!1332 = !DILocation(line: 887, column: 78, scope: !439)
!1333 = !DILocation(line: 887, column: 84, scope: !439)
!1334 = !DILocation(line: 887, column: 77, scope: !439)
!1335 = !DILocation(line: 887, column: 31, scope: !439)
!1336 = !DILocation(line: 885, column: 37, scope: !439)
!1337 = !DILocation(line: 888, column: 69, scope: !439)
!1338 = !DILocation(line: 888, column: 79, scope: !439)
!1339 = !DILocation(line: 888, column: 77, scope: !439)
!1340 = !DILocation(line: 888, column: 68, scope: !439)
!1341 = !DILocation(line: 888, column: 84, scope: !439)
!1342 = !DILocation(line: 888, column: 35, scope: !439)
!1343 = !DILocation(line: 888, column: 43, scope: !439)
!1344 = !DILocation(line: 888, column: 47, scope: !439)
!1345 = !DILocation(line: 888, column: 53, scope: !439)
!1346 = !DILocation(line: 888, column: 25, scope: !439)
!1347 = !DILocation(line: 888, column: 59, scope: !439)
!1348 = !DILocation(line: 888, column: 66, scope: !439)
!1349 = !DILocation(line: 889, column: 21, scope: !439)
!1350 = !DILocation(line: 890, column: 17, scope: !439)
!1351 = !DILocation(line: 894, column: 17, scope: !439)
!1352 = !DILocation(line: 896, column: 9, scope: !439)
!1353 = !DILocation(line: 655, column: 33, scope: !439)
!1354 = distinct !{!1354, !863, !1352, !50, !1355}
!1355 = !{!"llvm.loop.unroll.count", i32 1}
!1356 = !DILocation(line: 897, column: 5, scope: !439)
!1357 = !DILocation(line: 649, column: 42, scope: !439)
!1358 = distinct !{!1358, !858, !1356, !50}
!1359 = !DILocation(line: 908, column: 5, scope: !439)
!1360 = !DILocation(line: 908, column: 12, scope: !439)
!1361 = !DILocation(line: 910, column: 10, scope: !439)
!1362 = !DILocation(line: 910, column: 14, scope: !439)
!1363 = !DILocation(line: 910, column: 22, scope: !439)
!1364 = !DILocation(line: 910, column: 25, scope: !439)
!1365 = !DILocation(line: 910, column: 5, scope: !439)
!1366 = !DILocation(line: 912, column: 14, scope: !439)
!1367 = !DILocation(line: 912, column: 18, scope: !439)
!1368 = !DILocation(line: 912, column: 26, scope: !439)
!1369 = !DILocation(line: 912, column: 29, scope: !439)
!1370 = !DILocation(line: 912, column: 9, scope: !439)
!1371 = !DILocation(line: 914, column: 18, scope: !439)
!1372 = !DILocation(line: 914, column: 22, scope: !439)
!1373 = !DILocation(line: 914, column: 29, scope: !439)
!1374 = !DILocation(line: 914, column: 31, scope: !439)
!1375 = !DILocation(line: 914, column: 13, scope: !439)
!1376 = !DILocation(line: 915, column: 85, scope: !439)
!1377 = !DILocation(line: 915, column: 81, scope: !439)
!1378 = !DILocation(line: 915, column: 89, scope: !439)
!1379 = !DILocation(line: 915, column: 93, scope: !439)
!1380 = !DILocation(line: 915, column: 17, scope: !439)
!1381 = !DILocation(line: 915, column: 34, scope: !439)
!1382 = !DILocation(line: 915, column: 39, scope: !439)
!1383 = !DILocation(line: 915, column: 52, scope: !439)
!1384 = !DILocation(line: 915, column: 50, scope: !439)
!1385 = !DILocation(line: 915, column: 60, scope: !439)
!1386 = !DILocation(line: 915, column: 58, scope: !439)
!1387 = !DILocation(line: 915, column: 63, scope: !439)
!1388 = !DILocation(line: 915, column: 45, scope: !439)
!1389 = !DILocation(line: 915, column: 70, scope: !439)
!1390 = !DILocation(line: 915, column: 68, scope: !439)
!1391 = !DILocation(line: 915, column: 79, scope: !439)
!1392 = !DILocation(line: 916, column: 13, scope: !439)
!1393 = !DILocation(line: 914, column: 37, scope: !439)
!1394 = distinct !{!1394, !1375, !1392, !50, !305}
!1395 = !DILocation(line: 917, column: 13, scope: !439)
!1396 = !DILocation(line: 919, column: 18, scope: !439)
!1397 = !DILocation(line: 919, column: 22, scope: !439)
!1398 = !DILocation(line: 919, column: 29, scope: !439)
!1399 = !DILocation(line: 919, column: 31, scope: !439)
!1400 = !DILocation(line: 919, column: 13, scope: !439)
!1401 = !DILocation(line: 920, column: 17, scope: !439)
!1402 = !DILocation(line: 920, column: 31, scope: !439)
!1403 = !DILocation(line: 920, column: 33, scope: !439)
!1404 = !DILocation(line: 920, column: 27, scope: !439)
!1405 = !DILocation(line: 921, column: 17, scope: !439)
!1406 = !DILocation(line: 921, column: 31, scope: !439)
!1407 = !DILocation(line: 921, column: 33, scope: !439)
!1408 = !DILocation(line: 921, column: 27, scope: !439)
!1409 = !DILocation(line: 922, column: 17, scope: !439)
!1410 = !DILocation(line: 922, column: 31, scope: !439)
!1411 = !DILocation(line: 922, column: 35, scope: !439)
!1412 = !DILocation(line: 922, column: 27, scope: !439)
!1413 = !DILocation(line: 923, column: 17, scope: !439)
!1414 = !DILocation(line: 923, column: 32, scope: !439)
!1415 = !DILocation(line: 923, column: 36, scope: !439)
!1416 = !DILocation(line: 923, column: 42, scope: !439)
!1417 = !DILocation(line: 923, column: 27, scope: !439)
!1418 = !DILocation(line: 924, column: 17, scope: !439)
!1419 = !DILocation(line: 924, column: 37, scope: !439)
!1420 = !DILocation(line: 924, column: 39, scope: !439)
!1421 = !DILocation(line: 924, column: 46, scope: !439)
!1422 = !DILocation(line: 924, column: 49, scope: !439)
!1423 = !DILocation(line: 924, column: 44, scope: !439)
!1424 = !DILocation(line: 924, column: 56, scope: !439)
!1425 = !DILocation(line: 924, column: 54, scope: !439)
!1426 = !DILocation(line: 924, column: 27, scope: !439)
!1427 = !DILocation(line: 925, column: 17, scope: !439)
!1428 = !DILocation(line: 925, column: 37, scope: !439)
!1429 = !DILocation(line: 925, column: 39, scope: !439)
!1430 = !DILocation(line: 925, column: 46, scope: !439)
!1431 = !DILocation(line: 925, column: 49, scope: !439)
!1432 = !DILocation(line: 925, column: 44, scope: !439)
!1433 = !DILocation(line: 925, column: 56, scope: !439)
!1434 = !DILocation(line: 925, column: 54, scope: !439)
!1435 = !DILocation(line: 925, column: 27, scope: !439)
!1436 = !DILocation(line: 926, column: 17, scope: !439)
!1437 = !DILocation(line: 926, column: 36, scope: !439)
!1438 = !DILocation(line: 926, column: 38, scope: !439)
!1439 = !DILocation(line: 926, column: 46, scope: !439)
!1440 = !DILocation(line: 926, column: 44, scope: !439)
!1441 = !DILocation(line: 926, column: 27, scope: !439)
!1442 = !DILocation(line: 927, column: 17, scope: !439)
!1443 = !DILocation(line: 927, column: 33, scope: !439)
!1444 = !DILocation(line: 927, column: 50, scope: !439)
!1445 = !DILocation(line: 927, column: 56, scope: !439)
!1446 = !DILocation(line: 927, column: 64, scope: !439)
!1447 = !DILocation(line: 927, column: 66, scope: !439)
!1448 = !DILocation(line: 927, column: 62, scope: !439)
!1449 = !DILocation(line: 927, column: 73, scope: !439)
!1450 = !DILocation(line: 927, column: 71, scope: !439)
!1451 = !DILocation(line: 927, column: 29, scope: !439)
!1452 = !DILocation(line: 928, column: 17, scope: !439)
!1453 = !DILocation(line: 928, column: 33, scope: !439)
!1454 = !DILocation(line: 928, column: 38, scope: !439)
!1455 = !DILocation(line: 928, column: 36, scope: !439)
!1456 = !DILocation(line: 928, column: 27, scope: !439)
!1457 = !DILocation(line: 929, column: 17, scope: !439)
!1458 = !DILocation(line: 929, column: 33, scope: !439)
!1459 = !DILocation(line: 929, column: 38, scope: !439)
!1460 = !DILocation(line: 929, column: 36, scope: !439)
!1461 = !DILocation(line: 929, column: 27, scope: !439)
!1462 = !DILocation(line: 930, column: 21, scope: !439)
!1463 = !DILocation(line: 930, column: 27, scope: !439)
!1464 = !DILocation(line: 930, column: 25, scope: !439)
!1465 = !DILocation(line: 930, column: 29, scope: !439)
!1466 = !DILocation(line: 930, column: 32, scope: !439)
!1467 = !DILocation(line: 930, column: 38, scope: !439)
!1468 = !DILocation(line: 930, column: 36, scope: !439)
!1469 = !DILocation(line: 931, column: 21, scope: !439)
!1470 = !DILocation(line: 931, column: 33, scope: !439)
!1471 = !DILocation(line: 931, column: 48, scope: !439)
!1472 = !DILocation(line: 931, column: 54, scope: !439)
!1473 = !DILocation(line: 931, column: 52, scope: !439)
!1474 = !DILocation(line: 931, column: 35, scope: !439)
!1475 = !DILocation(line: 931, column: 58, scope: !439)
!1476 = !DILocation(line: 931, column: 56, scope: !439)
!1477 = !DILocation(line: 931, column: 28, scope: !439)
!1478 = !DILocation(line: 933, column: 32, scope: !439)
!1479 = !DILocation(line: 933, column: 26, scope: !439)
!1480 = !DILocation(line: 933, column: 29, scope: !439)
!1481 = !DILocation(line: 937, column: 17, scope: !439)
!1482 = !DILocation(line: 938, column: 13, scope: !439)
!1483 = !DILocation(line: 919, column: 37, scope: !439)
!1484 = distinct !{!1484, !1400, !1482, !50, !305}
!1485 = !DILocation(line: 939, column: 13, scope: !439)
!1486 = !DILocation(line: 940, column: 9, scope: !439)
!1487 = !DILocation(line: 912, column: 36, scope: !439)
!1488 = distinct !{!1488, !1370, !1486, !50, !305}
!1489 = !DILocation(line: 941, column: 5, scope: !439)
!1490 = !DILocation(line: 910, column: 32, scope: !439)
!1491 = distinct !{!1491, !1365, !1489, !50, !305}
!1492 = !DILocation(line: 942, column: 1, scope: !439)
!1493 = distinct !DISubprogram(name: "gemm_mq4g256v2_residual_mmq_iu4_gfx12_body<false>", scope: !127, file: !127, line: 488, type: !19, scopeLine: 493, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!1494 = !DILocation(line: 494, column: 5, scope: !1493)
!1495 = !DILocation(line: 494, column: 21, scope: !1493)
!1496 = !DILocation(line: 494, column: 15, scope: !1493)
!1497 = !DILocation(line: 495, column: 5, scope: !1493)
!1498 = !DILocation(line: 495, column: 22, scope: !1493)
!1499 = !DILocation(line: 495, column: 26, scope: !1493)
!1500 = !DILocation(line: 495, column: 15, scope: !1493)
!1501 = !DILocation(line: 496, column: 5, scope: !1493)
!1502 = !DILocation(line: 496, column: 22, scope: !1493)
!1503 = !DILocation(line: 496, column: 26, scope: !1493)
!1504 = !DILocation(line: 496, column: 15, scope: !1493)
!1505 = !DILocation(line: 497, column: 5, scope: !1493)
!1506 = !DILocation(line: 497, column: 20, scope: !1493)
!1507 = !DILocation(line: 497, column: 31, scope: !1493)
!1508 = !DILocation(line: 497, column: 15, scope: !1493)
!1509 = !DILocation(line: 498, column: 5, scope: !1493)
!1510 = !DILocation(line: 498, column: 20, scope: !1493)
!1511 = !DILocation(line: 498, column: 31, scope: !1493)
!1512 = !DILocation(line: 498, column: 15, scope: !1493)
!1513 = !DILocation(line: 500, column: 9, scope: !1493)
!1514 = !DILocation(line: 500, column: 15, scope: !1493)
!1515 = !DILocation(line: 500, column: 12, scope: !1493)
!1516 = !DILocation(line: 500, column: 17, scope: !1493)
!1517 = !DILocation(line: 500, column: 20, scope: !1493)
!1518 = !DILocation(line: 500, column: 26, scope: !1493)
!1519 = !DILocation(line: 500, column: 23, scope: !1493)
!1520 = !DILocation(line: 500, column: 29, scope: !1493)
!1521 = !DILocation(line: 502, column: 5, scope: !1493)
!1522 = !DILocation(line: 502, column: 24, scope: !1493)
!1523 = !DILocation(line: 502, column: 29, scope: !1493)
!1524 = !DILocation(line: 502, column: 15, scope: !1493)
!1525 = !DILocation(line: 503, column: 5, scope: !1493)
!1526 = !DILocation(line: 503, column: 23, scope: !1493)
!1527 = !DILocation(line: 503, column: 28, scope: !1493)
!1528 = !DILocation(line: 503, column: 15, scope: !1493)
!1529 = !DILocation(line: 504, column: 5, scope: !1493)
!1530 = !DILocation(line: 504, column: 25, scope: !1493)
!1531 = !DILocation(line: 504, column: 30, scope: !1493)
!1532 = !DILocation(line: 504, column: 15, scope: !1493)
!1533 = !DILocation(line: 505, column: 5, scope: !1493)
!1534 = !DILocation(line: 505, column: 26, scope: !1493)
!1535 = !DILocation(line: 505, column: 31, scope: !1493)
!1536 = !DILocation(line: 505, column: 15, scope: !1493)
!1537 = !DILocation(line: 506, column: 5, scope: !1493)
!1538 = !DILocation(line: 506, column: 27, scope: !1493)
!1539 = !DILocation(line: 506, column: 35, scope: !1493)
!1540 = !DILocation(line: 506, column: 15, scope: !1493)
!1541 = !DILocation(line: 507, column: 5, scope: !1493)
!1542 = !DILocation(line: 507, column: 27, scope: !1493)
!1543 = !DILocation(line: 507, column: 36, scope: !1493)
!1544 = !DILocation(line: 507, column: 15, scope: !1493)
!1545 = !DILocation(line: 509, column: 5, scope: !1493)
!1546 = !DILocation(line: 509, column: 32, scope: !1493)
!1547 = !DILocation(line: 509, column: 34, scope: !1493)
!1548 = !DILocation(line: 509, column: 15, scope: !1493)
!1549 = !DILocation(line: 512, column: 5, scope: !1493)
!1550 = !DILocation(line: 512, column: 24, scope: !1493)
!1551 = !DILocation(line: 513, column: 5, scope: !1493)
!1552 = !DILocation(line: 513, column: 24, scope: !1493)
!1553 = !DILocation(line: 518, column: 5, scope: !1493)
!1554 = !DILocation(line: 519, column: 5, scope: !1493)
!1555 = !DILocation(line: 524, column: 5, scope: !1493)
!1556 = !DILocation(line: 524, column: 38, scope: !1493)
!1557 = !DILocation(line: 524, column: 43, scope: !1493)
!1558 = !DILocation(line: 524, column: 20, scope: !1493)
!1559 = !DILocation(line: 526, column: 10, scope: !1493)
!1560 = !DILocation(line: 526, column: 14, scope: !1493)
!1561 = !DILocation(line: 526, column: 22, scope: !1493)
!1562 = !DILocation(line: 526, column: 25, scope: !1493)
!1563 = !DILocation(line: 526, column: 5, scope: !1493)
!1564 = !DILocation(line: 527, column: 9, scope: !1493)
!1565 = !DILocation(line: 527, column: 41, scope: !1493)
!1566 = !DILocation(line: 527, column: 50, scope: !1493)
!1567 = !DILocation(line: 527, column: 56, scope: !1493)
!1568 = !DILocation(line: 527, column: 54, scope: !1493)
!1569 = !DILocation(line: 527, column: 24, scope: !1493)
!1570 = !DILocation(line: 529, column: 14, scope: !1493)
!1571 = !DILocation(line: 529, column: 18, scope: !1493)
!1572 = !DILocation(line: 529, column: 26, scope: !1493)
!1573 = !DILocation(line: 529, column: 29, scope: !1493)
!1574 = !DILocation(line: 529, column: 9, scope: !1493)
!1575 = !DILocation(line: 530, column: 28, scope: !1493)
!1576 = !DILocation(line: 530, column: 32, scope: !1493)
!1577 = !DILocation(line: 530, column: 49, scope: !1493)
!1578 = !DILocation(line: 530, column: 37, scope: !1493)
!1579 = !DILocation(line: 530, column: 53, scope: !1493)
!1580 = !DILocation(line: 530, column: 71, scope: !1493)
!1581 = !DILocation(line: 530, column: 69, scope: !1493)
!1582 = !DILocation(line: 530, column: 17, scope: !1493)
!1583 = !DILocation(line: 530, column: 13, scope: !1493)
!1584 = !DILocation(line: 530, column: 21, scope: !1493)
!1585 = !DILocation(line: 530, column: 25, scope: !1493)
!1586 = !DILocation(line: 531, column: 9, scope: !1493)
!1587 = !DILocation(line: 529, column: 36, scope: !1493)
!1588 = distinct !{!1588, !1574, !1586, !50, !305}
!1589 = !DILocation(line: 532, column: 5, scope: !1493)
!1590 = !DILocation(line: 526, column: 32, scope: !1493)
!1591 = distinct !{!1591, !1563, !1589, !50, !305}
!1592 = !DILocation(line: 534, column: 10, scope: !1493)
!1593 = !DILocation(line: 534, column: 14, scope: !1493)
!1594 = !DILocation(line: 534, column: 22, scope: !1493)
!1595 = !DILocation(line: 534, column: 25, scope: !1493)
!1596 = !DILocation(line: 534, column: 5, scope: !1493)
!1597 = !DILocation(line: 535, column: 9, scope: !1493)
!1598 = !DILocation(line: 535, column: 41, scope: !1493)
!1599 = !DILocation(line: 535, column: 49, scope: !1493)
!1600 = !DILocation(line: 535, column: 55, scope: !1493)
!1601 = !DILocation(line: 535, column: 53, scope: !1493)
!1602 = !DILocation(line: 535, column: 24, scope: !1493)
!1603 = !DILocation(line: 537, column: 14, scope: !1493)
!1604 = !DILocation(line: 537, column: 18, scope: !1493)
!1605 = !DILocation(line: 537, column: 26, scope: !1493)
!1606 = !DILocation(line: 537, column: 29, scope: !1493)
!1607 = !DILocation(line: 537, column: 9, scope: !1493)
!1608 = !DILocation(line: 538, column: 28, scope: !1493)
!1609 = !DILocation(line: 538, column: 32, scope: !1493)
!1610 = !DILocation(line: 538, column: 49, scope: !1493)
!1611 = !DILocation(line: 538, column: 37, scope: !1493)
!1612 = !DILocation(line: 538, column: 53, scope: !1493)
!1613 = !DILocation(line: 538, column: 71, scope: !1493)
!1614 = !DILocation(line: 538, column: 69, scope: !1493)
!1615 = !DILocation(line: 538, column: 17, scope: !1493)
!1616 = !DILocation(line: 538, column: 13, scope: !1493)
!1617 = !DILocation(line: 538, column: 21, scope: !1493)
!1618 = !DILocation(line: 538, column: 25, scope: !1493)
!1619 = !DILocation(line: 539, column: 9, scope: !1493)
!1620 = !DILocation(line: 537, column: 36, scope: !1493)
!1621 = distinct !{!1621, !1607, !1619, !50, !305}
!1622 = !DILocation(line: 540, column: 5, scope: !1493)
!1623 = !DILocation(line: 534, column: 32, scope: !1493)
!1624 = distinct !{!1624, !1596, !1622, !50, !305}
!1625 = !DILocation(line: 544, column: 5, scope: !1493)
!1626 = !DILocation(line: 564, column: 5, scope: !1493)
!1627 = !DILocation(line: 565, column: 5, scope: !1493)
!1628 = !DILocation(line: 566, column: 5, scope: !1493)
!1629 = !DILocation(line: 567, column: 5, scope: !1493)
!1630 = !DILocation(line: 569, column: 10, scope: !1493)
!1631 = !DILocation(line: 569, column: 14, scope: !1493)
!1632 = !DILocation(line: 569, column: 21, scope: !1493)
!1633 = !DILocation(line: 569, column: 23, scope: !1493)
!1634 = !DILocation(line: 569, column: 5, scope: !1493)
!1635 = !DILocation(line: 570, column: 9, scope: !1493)
!1636 = !DILocation(line: 570, column: 27, scope: !1493)
!1637 = !DILocation(line: 570, column: 32, scope: !1493)
!1638 = !DILocation(line: 570, column: 34, scope: !1493)
!1639 = !DILocation(line: 570, column: 30, scope: !1493)
!1640 = !DILocation(line: 570, column: 42, scope: !1493)
!1641 = !DILocation(line: 570, column: 46, scope: !1493)
!1642 = !DILocation(line: 570, column: 39, scope: !1493)
!1643 = !DILocation(line: 570, column: 19, scope: !1493)
!1644 = !DILocation(line: 571, column: 9, scope: !1493)
!1645 = !DILocation(line: 571, column: 28, scope: !1493)
!1646 = !DILocation(line: 571, column: 36, scope: !1493)
!1647 = !DILocation(line: 571, column: 34, scope: !1493)
!1648 = !DILocation(line: 571, column: 27, scope: !1493)
!1649 = !DILocation(line: 571, column: 41, scope: !1493)
!1650 = !DILocation(line: 571, column: 50, scope: !1493)
!1651 = !DILocation(line: 571, column: 52, scope: !1493)
!1652 = !DILocation(line: 571, column: 19, scope: !1493)
!1653 = !DILocation(line: 572, column: 22, scope: !1493)
!1654 = !DILocation(line: 572, column: 30, scope: !1493)
!1655 = !DILocation(line: 572, column: 28, scope: !1493)
!1656 = !DILocation(line: 572, column: 21, scope: !1493)
!1657 = !DILocation(line: 572, column: 16, scope: !1493)
!1658 = !DILocation(line: 572, column: 9, scope: !1493)
!1659 = !DILocation(line: 572, column: 19, scope: !1493)
!1660 = !DILocation(line: 573, column: 33, scope: !1493)
!1661 = !DILocation(line: 573, column: 39, scope: !1493)
!1662 = !DILocation(line: 574, column: 26, scope: !1493)
!1663 = !DILocation(line: 574, column: 30, scope: !1493)
!1664 = !DILocation(line: 574, column: 35, scope: !1493)
!1665 = !DILocation(line: 574, column: 13, scope: !1493)
!1666 = !DILocation(line: 573, column: 18, scope: !1493)
!1667 = !DILocation(line: 573, column: 9, scope: !1493)
!1668 = !DILocation(line: 573, column: 21, scope: !1493)
!1669 = !DILocation(line: 575, column: 9, scope: !1493)
!1670 = !DILocation(line: 575, column: 28, scope: !1493)
!1671 = !DILocation(line: 575, column: 33, scope: !1493)
!1672 = !DILocation(line: 575, column: 35, scope: !1493)
!1673 = !DILocation(line: 575, column: 31, scope: !1493)
!1674 = !DILocation(line: 575, column: 43, scope: !1493)
!1675 = !DILocation(line: 575, column: 47, scope: !1493)
!1676 = !DILocation(line: 575, column: 40, scope: !1493)
!1677 = !DILocation(line: 575, column: 19, scope: !1493)
!1678 = !DILocation(line: 576, column: 9, scope: !1493)
!1679 = !DILocation(line: 576, column: 28, scope: !1493)
!1680 = !DILocation(line: 576, column: 37, scope: !1493)
!1681 = !DILocation(line: 576, column: 35, scope: !1493)
!1682 = !DILocation(line: 576, column: 27, scope: !1493)
!1683 = !DILocation(line: 576, column: 42, scope: !1493)
!1684 = !DILocation(line: 576, column: 52, scope: !1493)
!1685 = !DILocation(line: 576, column: 54, scope: !1493)
!1686 = !DILocation(line: 576, column: 19, scope: !1493)
!1687 = !DILocation(line: 577, column: 33, scope: !1493)
!1688 = !DILocation(line: 577, column: 51, scope: !1493)
!1689 = !DILocation(line: 577, column: 39, scope: !1493)
!1690 = !DILocation(line: 577, column: 66, scope: !1493)
!1691 = !DILocation(line: 578, column: 26, scope: !1493)
!1692 = !DILocation(line: 578, column: 30, scope: !1493)
!1693 = !DILocation(line: 578, column: 35, scope: !1493)
!1694 = !DILocation(line: 578, column: 13, scope: !1493)
!1695 = !DILocation(line: 577, column: 18, scope: !1493)
!1696 = !DILocation(line: 577, column: 9, scope: !1493)
!1697 = !DILocation(line: 577, column: 21, scope: !1493)
!1698 = !DILocation(line: 579, column: 9, scope: !1493)
!1699 = !DILocation(line: 579, column: 39, scope: !1493)
!1700 = !DILocation(line: 579, column: 41, scope: !1493)
!1701 = !DILocation(line: 579, column: 49, scope: !1493)
!1702 = !DILocation(line: 579, column: 53, scope: !1493)
!1703 = !DILocation(line: 579, column: 46, scope: !1493)
!1704 = !DILocation(line: 579, column: 24, scope: !1493)
!1705 = !DILocation(line: 580, column: 9, scope: !1493)
!1706 = !DILocation(line: 580, column: 39, scope: !1493)
!1707 = !DILocation(line: 580, column: 43, scope: !1493)
!1708 = !DILocation(line: 580, column: 24, scope: !1493)
!1709 = !DILocation(line: 581, column: 26, scope: !1493)
!1710 = !DILocation(line: 581, column: 28, scope: !1493)
!1711 = !DILocation(line: 581, column: 34, scope: !1493)
!1712 = !DILocation(line: 581, column: 42, scope: !1493)
!1713 = !DILocation(line: 581, column: 44, scope: !1493)
!1714 = !DILocation(line: 581, column: 39, scope: !1493)
!1715 = !DILocation(line: 581, column: 51, scope: !1493)
!1716 = !DILocation(line: 582, column: 17, scope: !1493)
!1717 = !DILocation(line: 582, column: 19, scope: !1493)
!1718 = !DILocation(line: 582, column: 29, scope: !1493)
!1719 = !DILocation(line: 582, column: 31, scope: !1493)
!1720 = !DILocation(line: 582, column: 37, scope: !1493)
!1721 = !DILocation(line: 582, column: 26, scope: !1493)
!1722 = !DILocation(line: 582, column: 44, scope: !1493)
!1723 = !DILocation(line: 582, column: 13, scope: !1493)
!1724 = !DILocation(line: 581, column: 19, scope: !1493)
!1725 = !DILocation(line: 581, column: 9, scope: !1493)
!1726 = !DILocation(line: 581, column: 22, scope: !1493)
!1727 = !DILocation(line: 583, column: 5, scope: !1493)
!1728 = !DILocation(line: 569, column: 29, scope: !1493)
!1729 = distinct !{!1729, !1634, !1727, !50, !305}
!1730 = !DILocation(line: 584, column: 5, scope: !1493)
!1731 = !DILocation(line: 584, column: 19, scope: !1493)
!1732 = !DILocation(line: 585, column: 10, scope: !1493)
!1733 = !DILocation(line: 585, column: 14, scope: !1493)
!1734 = !DILocation(line: 585, column: 22, scope: !1493)
!1735 = !DILocation(line: 585, column: 25, scope: !1493)
!1736 = !DILocation(line: 585, column: 5, scope: !1493)
!1737 = !DILocation(line: 587, column: 14, scope: !1493)
!1738 = !DILocation(line: 587, column: 18, scope: !1493)
!1739 = !DILocation(line: 587, column: 26, scope: !1493)
!1740 = !DILocation(line: 587, column: 29, scope: !1493)
!1741 = !DILocation(line: 587, column: 9, scope: !1493)
!1742 = !DILocation(line: 588, column: 27, scope: !1493)
!1743 = !DILocation(line: 588, column: 17, scope: !1493)
!1744 = !DILocation(line: 588, column: 13, scope: !1493)
!1745 = !DILocation(line: 588, column: 21, scope: !1493)
!1746 = !DILocation(line: 588, column: 25, scope: !1493)
!1747 = !DILocation(line: 589, column: 9, scope: !1493)
!1748 = !DILocation(line: 587, column: 36, scope: !1493)
!1749 = distinct !{!1749, !1741, !1747, !50, !305}
!1750 = !DILocation(line: 590, column: 5, scope: !1493)
!1751 = !DILocation(line: 585, column: 32, scope: !1493)
!1752 = distinct !{!1752, !1736, !1750, !50}
!1753 = !DILocation(line: 593, column: 5, scope: !1493)
!1754 = !DILocation(line: 593, column: 26, scope: !1493)
!1755 = !DILocation(line: 593, column: 21, scope: !1493)
!1756 = !DILocation(line: 595, column: 5, scope: !1493)
!1757 = !DILocation(line: 596, column: 5, scope: !1493)
!1758 = !DILocation(line: 602, column: 5, scope: !1493)
!1759 = !DILocation(line: 603, column: 5, scope: !1493)
!1760 = !DILocation(line: 607, column: 5, scope: !1493)
!1761 = !DILocation(line: 607, column: 14, scope: !1493)
!1762 = !DILocation(line: 608, column: 5, scope: !1493)
!1763 = !DILocation(line: 608, column: 14, scope: !1493)
!1764 = !DILocation(line: 616, column: 10, scope: !1493)
!1765 = !DILocation(line: 616, column: 18, scope: !1493)
!1766 = !DILocation(line: 616, column: 14, scope: !1493)
!1767 = !DILocation(line: 616, column: 23, scope: !1493)
!1768 = !DILocation(line: 616, column: 25, scope: !1493)
!1769 = !DILocation(line: 616, column: 5, scope: !1493)
!1770 = !DILocation(line: 617, column: 9, scope: !1493)
!1771 = !DILocation(line: 617, column: 29, scope: !1493)
!1772 = !DILocation(line: 617, column: 31, scope: !1493)
!1773 = !DILocation(line: 617, column: 19, scope: !1493)
!1774 = !DILocation(line: 618, column: 9, scope: !1493)
!1775 = !DILocation(line: 618, column: 23, scope: !1493)
!1776 = !DILocation(line: 618, column: 25, scope: !1493)
!1777 = !DILocation(line: 618, column: 19, scope: !1493)
!1778 = !DILocation(line: 619, column: 9, scope: !1493)
!1779 = !DILocation(line: 619, column: 25, scope: !1493)
!1780 = !DILocation(line: 619, column: 30, scope: !1493)
!1781 = !DILocation(line: 619, column: 28, scope: !1493)
!1782 = !DILocation(line: 619, column: 19, scope: !1493)
!1783 = !DILocation(line: 620, column: 9, scope: !1493)
!1784 = !DILocation(line: 620, column: 30, scope: !1493)
!1785 = !DILocation(line: 620, column: 36, scope: !1493)
!1786 = !DILocation(line: 620, column: 34, scope: !1493)
!1787 = !DILocation(line: 620, column: 29, scope: !1493)
!1788 = !DILocation(line: 621, column: 34, scope: !1493)
!1789 = !DILocation(line: 621, column: 39, scope: !1493)
!1790 = !DILocation(line: 621, column: 15, scope: !1493)
!1791 = !DILocation(line: 621, column: 45, scope: !1493)
!1792 = !DILocation(line: 620, column: 24, scope: !1493)
!1793 = !DILocation(line: 623, column: 162, scope: !1493)
!1794 = !DILocation(line: 623, column: 143, scope: !1493)
!1795 = !DILocation(line: 623, column: 151, scope: !1493)
!1796 = !DILocation(line: 623, column: 9, scope: !1493)
!1797 = !DILocation(line: 623, column: 157, scope: !1493)
!1798 = !DILocation(line: 623, column: 160, scope: !1493)
!1799 = !DILocation(line: 624, column: 5, scope: !1493)
!1800 = !DILocation(line: 616, column: 38, scope: !1493)
!1801 = distinct !{!1801, !1769, !1799, !50}
!1802 = !DILocation(line: 625, column: 10, scope: !1493)
!1803 = !DILocation(line: 625, column: 18, scope: !1493)
!1804 = !DILocation(line: 625, column: 14, scope: !1493)
!1805 = !DILocation(line: 625, column: 23, scope: !1493)
!1806 = !DILocation(line: 625, column: 25, scope: !1493)
!1807 = !DILocation(line: 625, column: 5, scope: !1493)
!1808 = !DILocation(line: 626, column: 9, scope: !1493)
!1809 = !DILocation(line: 626, column: 29, scope: !1493)
!1810 = !DILocation(line: 626, column: 31, scope: !1493)
!1811 = !DILocation(line: 626, column: 19, scope: !1493)
!1812 = !DILocation(line: 627, column: 9, scope: !1493)
!1813 = !DILocation(line: 627, column: 23, scope: !1493)
!1814 = !DILocation(line: 627, column: 25, scope: !1493)
!1815 = !DILocation(line: 627, column: 19, scope: !1493)
!1816 = !DILocation(line: 628, column: 9, scope: !1493)
!1817 = !DILocation(line: 628, column: 26, scope: !1493)
!1818 = !DILocation(line: 628, column: 31, scope: !1493)
!1819 = !DILocation(line: 628, column: 29, scope: !1493)
!1820 = !DILocation(line: 628, column: 19, scope: !1493)
!1821 = !DILocation(line: 629, column: 9, scope: !1493)
!1822 = !DILocation(line: 629, column: 30, scope: !1493)
!1823 = !DILocation(line: 629, column: 37, scope: !1493)
!1824 = !DILocation(line: 629, column: 35, scope: !1493)
!1825 = !DILocation(line: 629, column: 29, scope: !1493)
!1826 = !DILocation(line: 629, column: 42, scope: !1493)
!1827 = !DILocation(line: 629, column: 50, scope: !1493)
!1828 = !DILocation(line: 629, column: 52, scope: !1493)
!1829 = !DILocation(line: 629, column: 19, scope: !1493)
!1830 = !DILocation(line: 630, column: 9, scope: !1493)
!1831 = !DILocation(line: 630, column: 48, scope: !1493)
!1832 = !DILocation(line: 630, column: 63, scope: !1493)
!1833 = !DILocation(line: 630, column: 73, scope: !1493)
!1834 = !DILocation(line: 630, column: 71, scope: !1493)
!1835 = !DILocation(line: 630, column: 88, scope: !1493)
!1836 = !DILocation(line: 630, column: 29, scope: !1493)
!1837 = !DILocation(line: 630, column: 24, scope: !1493)
!1838 = !DILocation(line: 631, column: 9, scope: !1493)
!1839 = !DILocation(line: 631, column: 26, scope: !1493)
!1840 = !DILocation(line: 631, column: 28, scope: !1493)
!1841 = !DILocation(line: 631, column: 25, scope: !1493)
!1842 = !DILocation(line: 632, column: 28, scope: !1493)
!1843 = !DILocation(line: 632, column: 62, scope: !1493)
!1844 = !DILocation(line: 632, column: 65, scope: !1493)
!1845 = !DILocation(line: 632, column: 61, scope: !1493)
!1846 = !DILocation(line: 632, column: 15, scope: !1493)
!1847 = !DILocation(line: 633, column: 28, scope: !1493)
!1848 = !DILocation(line: 633, column: 62, scope: !1493)
!1849 = !DILocation(line: 633, column: 65, scope: !1493)
!1850 = !DILocation(line: 633, column: 61, scope: !1493)
!1851 = !DILocation(line: 633, column: 15, scope: !1493)
!1852 = !DILocation(line: 631, column: 21, scope: !1493)
!1853 = !DILocation(line: 634, column: 188, scope: !1493)
!1854 = !DILocation(line: 634, column: 195, scope: !1493)
!1855 = !DILocation(line: 634, column: 193, scope: !1493)
!1856 = !DILocation(line: 634, column: 187, scope: !1493)
!1857 = !DILocation(line: 634, column: 200, scope: !1493)
!1858 = !DILocation(line: 634, column: 168, scope: !1493)
!1859 = !DILocation(line: 634, column: 176, scope: !1493)
!1860 = !DILocation(line: 634, column: 9, scope: !1493)
!1861 = !DILocation(line: 634, column: 182, scope: !1493)
!1862 = !DILocation(line: 634, column: 185, scope: !1493)
!1863 = !DILocation(line: 635, column: 5, scope: !1493)
!1864 = !DILocation(line: 625, column: 38, scope: !1493)
!1865 = distinct !{!1865, !1807, !1863, !50}
!1866 = !DILocation(line: 638, column: 9, scope: !1493)
!1867 = !DILocation(line: 638, column: 54, scope: !1493)
!1868 = !DILocation(line: 638, column: 57, scope: !1493)
!1869 = !DILocation(line: 638, column: 34, scope: !1493)
!1870 = !DILocation(line: 639, column: 9, scope: !1493)
!1871 = !DILocation(line: 639, column: 40, scope: !1493)
!1872 = !DILocation(line: 639, column: 42, scope: !1493)
!1873 = !DILocation(line: 639, column: 34, scope: !1493)
!1874 = !DILocation(line: 641, column: 14, scope: !1493)
!1875 = !DILocation(line: 641, column: 18, scope: !1493)
!1876 = !DILocation(line: 641, column: 25, scope: !1493)
!1877 = !DILocation(line: 641, column: 27, scope: !1493)
!1878 = !DILocation(line: 641, column: 9, scope: !1493)
!1879 = !DILocation(line: 642, column: 13, scope: !1493)
!1880 = !DILocation(line: 642, column: 50, scope: !1493)
!1881 = !DILocation(line: 642, column: 66, scope: !1493)
!1882 = !DILocation(line: 642, column: 57, scope: !1493)
!1883 = !DILocation(line: 642, column: 32, scope: !1493)
!1884 = !DILocation(line: 642, column: 27, scope: !1493)
!1885 = !DILocation(line: 643, column: 56, scope: !1493)
!1886 = !DILocation(line: 643, column: 49, scope: !1493)
!1887 = !DILocation(line: 643, column: 61, scope: !1493)
!1888 = !DILocation(line: 643, column: 25, scope: !1493)
!1889 = !DILocation(line: 643, column: 43, scope: !1493)
!1890 = !DILocation(line: 643, column: 33, scope: !1493)
!1891 = !DILocation(line: 643, column: 13, scope: !1493)
!1892 = !DILocation(line: 643, column: 47, scope: !1493)
!1893 = !DILocation(line: 644, column: 67, scope: !1493)
!1894 = !DILocation(line: 644, column: 82, scope: !1493)
!1895 = !DILocation(line: 644, column: 73, scope: !1493)
!1896 = !DILocation(line: 644, column: 49, scope: !1493)
!1897 = !DILocation(line: 644, column: 25, scope: !1493)
!1898 = !DILocation(line: 644, column: 43, scope: !1493)
!1899 = !DILocation(line: 644, column: 33, scope: !1493)
!1900 = !DILocation(line: 644, column: 13, scope: !1493)
!1901 = !DILocation(line: 644, column: 47, scope: !1493)
!1902 = !DILocation(line: 645, column: 9, scope: !1493)
!1903 = !DILocation(line: 641, column: 33, scope: !1493)
!1904 = distinct !{!1904, !1878, !1902, !50, !305}
!1905 = !DILocation(line: 646, column: 5, scope: !1493)
!1906 = !DILocation(line: 647, column: 5, scope: !1493)
!1907 = !DILocation(line: 649, column: 10, scope: !1493)
!1908 = !DILocation(line: 649, column: 14, scope: !1493)
!1909 = !DILocation(line: 649, column: 21, scope: !1493)
!1910 = !DILocation(line: 649, column: 25, scope: !1493)
!1911 = !DILocation(line: 649, column: 23, scope: !1493)
!1912 = !DILocation(line: 649, column: 5, scope: !1493)
!1913 = !DILocation(line: 655, column: 14, scope: !1493)
!1914 = !DILocation(line: 655, column: 18, scope: !1493)
!1915 = !DILocation(line: 655, column: 25, scope: !1493)
!1916 = !DILocation(line: 655, column: 27, scope: !1493)
!1917 = !DILocation(line: 655, column: 9, scope: !1493)
!1918 = !DILocation(line: 656, column: 13, scope: !1493)
!1919 = !DILocation(line: 656, column: 32, scope: !1493)
!1920 = !DILocation(line: 656, column: 30, scope: !1493)
!1921 = !DILocation(line: 656, column: 36, scope: !1493)
!1922 = !DILocation(line: 656, column: 34, scope: !1493)
!1923 = !DILocation(line: 656, column: 23, scope: !1493)
!1924 = !DILocation(line: 657, column: 13, scope: !1493)
!1925 = !DILocation(line: 657, column: 32, scope: !1493)
!1926 = !DILocation(line: 657, column: 34, scope: !1493)
!1927 = !DILocation(line: 657, column: 40, scope: !1493)
!1928 = !DILocation(line: 657, column: 44, scope: !1493)
!1929 = !DILocation(line: 657, column: 46, scope: !1493)
!1930 = !DILocation(line: 657, column: 53, scope: !1493)
!1931 = !DILocation(line: 657, column: 50, scope: !1493)
!1932 = !DILocation(line: 0, scope: !1493)
!1933 = !DILocation(line: 657, column: 24, scope: !1493)
!1934 = !DILocation(line: 661, column: 13, scope: !1493)
!1935 = !DILocation(line: 661, column: 49, scope: !1493)
!1936 = !DILocation(line: 661, column: 51, scope: !1493)
!1937 = !DILocation(line: 661, column: 48, scope: !1493)
!1938 = !DILocation(line: 661, column: 45, scope: !1493)
!1939 = !DILocation(line: 661, column: 32, scope: !1493)
!1940 = !DILocation(line: 662, column: 13, scope: !1493)
!1941 = !DILocation(line: 662, column: 49, scope: !1493)
!1942 = !DILocation(line: 662, column: 51, scope: !1493)
!1943 = !DILocation(line: 662, column: 48, scope: !1493)
!1944 = !DILocation(line: 662, column: 45, scope: !1493)
!1945 = !DILocation(line: 662, column: 32, scope: !1493)
!1946 = !DILocation(line: 671, column: 18, scope: !1493)
!1947 = !DILocation(line: 671, column: 22, scope: !1493)
!1948 = !DILocation(line: 671, column: 30, scope: !1493)
!1949 = !DILocation(line: 671, column: 33, scope: !1493)
!1950 = !DILocation(line: 671, column: 13, scope: !1493)
!1951 = !DILocation(line: 673, column: 22, scope: !1493)
!1952 = !DILocation(line: 673, column: 26, scope: !1493)
!1953 = !DILocation(line: 673, column: 33, scope: !1493)
!1954 = !DILocation(line: 673, column: 35, scope: !1493)
!1955 = !DILocation(line: 673, column: 17, scope: !1493)
!1956 = !DILocation(line: 674, column: 21, scope: !1493)
!1957 = !DILocation(line: 675, column: 36, scope: !1493)
!1958 = !DILocation(line: 675, column: 48, scope: !1493)
!1959 = !DILocation(line: 675, column: 51, scope: !1493)
!1960 = !DILocation(line: 675, column: 46, scope: !1493)
!1961 = !DILocation(line: 675, column: 62, scope: !1493)
!1962 = !DILocation(line: 675, column: 60, scope: !1493)
!1963 = !DILocation(line: 675, column: 56, scope: !1493)
!1964 = !DILocation(line: 675, column: 70, scope: !1493)
!1965 = !DILocation(line: 675, column: 68, scope: !1493)
!1966 = !DILocation(line: 675, column: 73, scope: !1493)
!1967 = !DILocation(line: 674, column: 36, scope: !1493)
!1968 = !DILocation(line: 676, column: 59, scope: !1493)
!1969 = !DILocation(line: 676, column: 68, scope: !1493)
!1970 = !DILocation(line: 676, column: 43, scope: !1493)
!1971 = !DILocation(line: 676, column: 34, scope: !1493)
!1972 = !DILocation(line: 676, column: 21, scope: !1493)
!1973 = !DILocation(line: 676, column: 38, scope: !1493)
!1974 = !DILocation(line: 676, column: 41, scope: !1493)
!1975 = !DILocation(line: 677, column: 17, scope: !1493)
!1976 = !DILocation(line: 673, column: 41, scope: !1493)
!1977 = distinct !{!1977, !1955, !1975, !50, !305}
!1978 = !DILocation(line: 678, column: 13, scope: !1493)
!1979 = !DILocation(line: 671, column: 40, scope: !1493)
!1980 = distinct !{!1980, !1950, !1978, !50, !305}
!1981 = !DILocation(line: 680, column: 18, scope: !1493)
!1982 = !DILocation(line: 680, column: 22, scope: !1493)
!1983 = !DILocation(line: 680, column: 30, scope: !1493)
!1984 = !DILocation(line: 680, column: 33, scope: !1493)
!1985 = !DILocation(line: 680, column: 13, scope: !1493)
!1986 = !DILocation(line: 681, column: 17, scope: !1493)
!1987 = !DILocation(line: 682, column: 32, scope: !1493)
!1988 = !DILocation(line: 682, column: 44, scope: !1493)
!1989 = !DILocation(line: 682, column: 47, scope: !1493)
!1990 = !DILocation(line: 682, column: 42, scope: !1493)
!1991 = !DILocation(line: 682, column: 54, scope: !1493)
!1992 = !DILocation(line: 682, column: 52, scope: !1493)
!1993 = !DILocation(line: 682, column: 62, scope: !1493)
!1994 = !DILocation(line: 681, column: 32, scope: !1493)
!1995 = !DILocation(line: 683, column: 52, scope: !1493)
!1996 = !DILocation(line: 683, column: 61, scope: !1493)
!1997 = !DILocation(line: 683, column: 36, scope: !1493)
!1998 = !DILocation(line: 683, column: 30, scope: !1493)
!1999 = !DILocation(line: 683, column: 17, scope: !1493)
!2000 = !DILocation(line: 683, column: 34, scope: !1493)
!2001 = !DILocation(line: 684, column: 13, scope: !1493)
!2002 = !DILocation(line: 680, column: 40, scope: !1493)
!2003 = distinct !{!2003, !1985, !2001, !50, !305}
!2004 = !DILocation(line: 690, column: 17, scope: !1493)
!2005 = !DILocation(line: 690, column: 63, scope: !1493)
!2006 = !DILocation(line: 690, column: 79, scope: !1493)
!2007 = !DILocation(line: 690, column: 84, scope: !1493)
!2008 = !DILocation(line: 690, column: 82, scope: !1493)
!2009 = !DILocation(line: 690, column: 66, scope: !1493)
!2010 = !DILocation(line: 690, column: 87, scope: !1493)
!2011 = !DILocation(line: 690, column: 91, scope: !1493)
!2012 = !DILocation(line: 690, column: 42, scope: !1493)
!2013 = !DILocation(line: 691, column: 17, scope: !1493)
!2014 = !DILocation(line: 691, column: 48, scope: !1493)
!2015 = !DILocation(line: 691, column: 63, scope: !1493)
!2016 = !DILocation(line: 691, column: 65, scope: !1493)
!2017 = !DILocation(line: 691, column: 50, scope: !1493)
!2018 = !DILocation(line: 691, column: 71, scope: !1493)
!2019 = !DILocation(line: 691, column: 88, scope: !1493)
!2020 = !DILocation(line: 691, column: 90, scope: !1493)
!2021 = !DILocation(line: 691, column: 75, scope: !1493)
!2022 = !DILocation(line: 691, column: 95, scope: !1493)
!2023 = !DILocation(line: 691, column: 42, scope: !1493)
!2024 = !DILocation(line: 693, column: 22, scope: !1493)
!2025 = !DILocation(line: 693, column: 26, scope: !1493)
!2026 = !DILocation(line: 693, column: 33, scope: !1493)
!2027 = !DILocation(line: 693, column: 35, scope: !1493)
!2028 = !DILocation(line: 693, column: 17, scope: !1493)
!2029 = !DILocation(line: 694, column: 21, scope: !1493)
!2030 = !DILocation(line: 694, column: 58, scope: !1493)
!2031 = !DILocation(line: 694, column: 74, scope: !1493)
!2032 = !DILocation(line: 694, column: 65, scope: !1493)
!2033 = !DILocation(line: 694, column: 40, scope: !1493)
!2034 = !DILocation(line: 694, column: 35, scope: !1493)
!2035 = !DILocation(line: 695, column: 38, scope: !1493)
!2036 = !DILocation(line: 695, column: 31, scope: !1493)
!2037 = !DILocation(line: 695, column: 43, scope: !1493)
!2038 = !DILocation(line: 695, column: 26, scope: !1493)
!2039 = !DILocation(line: 695, column: 21, scope: !1493)
!2040 = !DILocation(line: 695, column: 29, scope: !1493)
!2041 = !DILocation(line: 696, column: 49, scope: !1493)
!2042 = !DILocation(line: 696, column: 64, scope: !1493)
!2043 = !DILocation(line: 696, column: 55, scope: !1493)
!2044 = !DILocation(line: 696, column: 31, scope: !1493)
!2045 = !DILocation(line: 696, column: 26, scope: !1493)
!2046 = !DILocation(line: 696, column: 21, scope: !1493)
!2047 = !DILocation(line: 696, column: 29, scope: !1493)
!2048 = !DILocation(line: 697, column: 17, scope: !1493)
!2049 = !DILocation(line: 693, column: 41, scope: !1493)
!2050 = distinct !{!2050, !2028, !2048, !50, !305}
!2051 = !DILocation(line: 710, column: 21, scope: !1493)
!2052 = !DILocation(line: 710, column: 28, scope: !1493)
!2053 = !DILocation(line: 710, column: 45, scope: !1493)
!2054 = !DILocation(line: 711, column: 21, scope: !1493)
!2055 = !DILocation(line: 709, column: 17, scope: !1493)
!2056 = !DILocation(line: 713, column: 21, scope: !1493)
!2057 = !DILocation(line: 713, column: 28, scope: !1493)
!2058 = !DILocation(line: 713, column: 45, scope: !1493)
!2059 = !DILocation(line: 714, column: 21, scope: !1493)
!2060 = !DILocation(line: 712, column: 17, scope: !1493)
!2061 = !DILocation(line: 722, column: 17, scope: !1493)
!2062 = !DILocation(line: 726, column: 23, scope: !1493)
!2063 = !DILocation(line: 728, column: 22, scope: !1493)
!2064 = !DILocation(line: 728, column: 26, scope: !1493)
!2065 = !DILocation(line: 728, column: 33, scope: !1493)
!2066 = !DILocation(line: 728, column: 35, scope: !1493)
!2067 = !DILocation(line: 728, column: 17, scope: !1493)
!2068 = !DILocation(line: 729, column: 62, scope: !1493)
!2069 = !DILocation(line: 729, column: 57, scope: !1493)
!2070 = !DILocation(line: 729, column: 33, scope: !1493)
!2071 = !DILocation(line: 729, column: 51, scope: !1493)
!2072 = !DILocation(line: 729, column: 41, scope: !1493)
!2073 = !DILocation(line: 729, column: 21, scope: !1493)
!2074 = !DILocation(line: 729, column: 55, scope: !1493)
!2075 = !DILocation(line: 730, column: 62, scope: !1493)
!2076 = !DILocation(line: 730, column: 57, scope: !1493)
!2077 = !DILocation(line: 730, column: 33, scope: !1493)
!2078 = !DILocation(line: 730, column: 51, scope: !1493)
!2079 = !DILocation(line: 730, column: 41, scope: !1493)
!2080 = !DILocation(line: 730, column: 21, scope: !1493)
!2081 = !DILocation(line: 730, column: 55, scope: !1493)
!2082 = !DILocation(line: 731, column: 17, scope: !1493)
!2083 = !DILocation(line: 728, column: 41, scope: !1493)
!2084 = distinct !{!2084, !2067, !2082, !50, !305}
!2085 = !DILocation(line: 733, column: 17, scope: !1493)
!2086 = !DILocation(line: 734, column: 22, scope: !1493)
!2087 = !DILocation(line: 734, column: 21, scope: !1493)
!2088 = !DILocation(line: 744, column: 21, scope: !1493)
!2089 = !DILocation(line: 744, column: 39, scope: !1493)
!2090 = !DILocation(line: 744, column: 42, scope: !1493)
!2091 = !DILocation(line: 744, column: 31, scope: !1493)
!2092 = !DILocation(line: 745, column: 21, scope: !1493)
!2093 = !DILocation(line: 745, column: 68, scope: !1493)
!2094 = !DILocation(line: 745, column: 84, scope: !1493)
!2095 = !DILocation(line: 745, column: 92, scope: !1493)
!2096 = !DILocation(line: 745, column: 90, scope: !1493)
!2097 = !DILocation(line: 745, column: 71, scope: !1493)
!2098 = !DILocation(line: 745, column: 95, scope: !1493)
!2099 = !DILocation(line: 745, column: 46, scope: !1493)
!2100 = !DILocation(line: 747, column: 26, scope: !1493)
!2101 = !DILocation(line: 747, column: 30, scope: !1493)
!2102 = !DILocation(line: 747, column: 37, scope: !1493)
!2103 = !DILocation(line: 747, column: 39, scope: !1493)
!2104 = !DILocation(line: 747, column: 21, scope: !1493)
!2105 = !DILocation(line: 748, column: 53, scope: !1493)
!2106 = !DILocation(line: 748, column: 70, scope: !1493)
!2107 = !DILocation(line: 748, column: 61, scope: !1493)
!2108 = !DILocation(line: 748, column: 35, scope: !1493)
!2109 = !DILocation(line: 748, column: 30, scope: !1493)
!2110 = !DILocation(line: 748, column: 25, scope: !1493)
!2111 = !DILocation(line: 748, column: 33, scope: !1493)
!2112 = !DILocation(line: 749, column: 21, scope: !1493)
!2113 = !DILocation(line: 747, column: 45, scope: !1493)
!2114 = distinct !{!2114, !2104, !2112, !50, !305}
!2115 = !DILocation(line: 750, column: 21, scope: !1493)
!2116 = !DILocation(line: 750, column: 38, scope: !1493)
!2117 = !DILocation(line: 750, column: 42, scope: !1493)
!2118 = !DILocation(line: 750, column: 40, scope: !1493)
!2119 = !DILocation(line: 750, column: 31, scope: !1493)
!2120 = !DILocation(line: 751, column: 21, scope: !1493)
!2121 = !DILocation(line: 751, column: 38, scope: !1493)
!2122 = !DILocation(line: 751, column: 40, scope: !1493)
!2123 = !DILocation(line: 751, column: 31, scope: !1493)
!2124 = !DILocation(line: 752, column: 21, scope: !1493)
!2125 = !DILocation(line: 752, column: 53, scope: !1493)
!2126 = !DILocation(line: 752, column: 68, scope: !1493)
!2127 = !DILocation(line: 752, column: 73, scope: !1493)
!2128 = !DILocation(line: 752, column: 55, scope: !1493)
!2129 = !DILocation(line: 752, column: 79, scope: !1493)
!2130 = !DILocation(line: 752, column: 96, scope: !1493)
!2131 = !DILocation(line: 752, column: 101, scope: !1493)
!2132 = !DILocation(line: 752, column: 83, scope: !1493)
!2133 = !DILocation(line: 752, column: 46, scope: !1493)
!2134 = !DILocation(line: 754, column: 26, scope: !1493)
!2135 = !DILocation(line: 754, column: 30, scope: !1493)
!2136 = !DILocation(line: 754, column: 37, scope: !1493)
!2137 = !DILocation(line: 754, column: 39, scope: !1493)
!2138 = !DILocation(line: 754, column: 21, scope: !1493)
!2139 = !DILocation(line: 755, column: 53, scope: !1493)
!2140 = !DILocation(line: 755, column: 69, scope: !1493)
!2141 = !DILocation(line: 755, column: 60, scope: !1493)
!2142 = !DILocation(line: 755, column: 35, scope: !1493)
!2143 = !DILocation(line: 755, column: 30, scope: !1493)
!2144 = !DILocation(line: 755, column: 25, scope: !1493)
!2145 = !DILocation(line: 755, column: 33, scope: !1493)
!2146 = !DILocation(line: 756, column: 21, scope: !1493)
!2147 = !DILocation(line: 754, column: 45, scope: !1493)
!2148 = distinct !{!2148, !2138, !2146, !50, !305}
!2149 = !DILocation(line: 757, column: 21, scope: !1493)
!2150 = !DILocation(line: 757, column: 40, scope: !1493)
!2151 = !DILocation(line: 757, column: 46, scope: !1493)
!2152 = !DILocation(line: 757, column: 50, scope: !1493)
!2153 = !DILocation(line: 757, column: 43, scope: !1493)
!2154 = !DILocation(line: 757, column: 31, scope: !1493)
!2155 = !DILocation(line: 758, column: 21, scope: !1493)
!2156 = !DILocation(line: 758, column: 42, scope: !1493)
!2157 = !DILocation(line: 758, column: 51, scope: !1493)
!2158 = !DILocation(line: 758, column: 49, scope: !1493)
!2159 = !DILocation(line: 758, column: 41, scope: !1493)
!2160 = !DILocation(line: 758, column: 56, scope: !1493)
!2161 = !DILocation(line: 758, column: 66, scope: !1493)
!2162 = !DILocation(line: 758, column: 68, scope: !1493)
!2163 = !DILocation(line: 758, column: 31, scope: !1493)
!2164 = !DILocation(line: 759, column: 48, scope: !1493)
!2165 = !DILocation(line: 759, column: 64, scope: !1493)
!2166 = !DILocation(line: 759, column: 72, scope: !1493)
!2167 = !DILocation(line: 759, column: 70, scope: !1493)
!2168 = !DILocation(line: 759, column: 29, scope: !1493)
!2169 = !DILocation(line: 759, column: 76, scope: !1493)
!2170 = !DILocation(line: 759, column: 86, scope: !1493)
!2171 = !DILocation(line: 759, column: 90, scope: !1493)
!2172 = !DILocation(line: 759, column: 27, scope: !1493)
!2173 = !DILocation(line: 760, column: 21, scope: !1493)
!2174 = !DILocation(line: 760, column: 40, scope: !1493)
!2175 = !DILocation(line: 760, column: 46, scope: !1493)
!2176 = !DILocation(line: 760, column: 50, scope: !1493)
!2177 = !DILocation(line: 760, column: 43, scope: !1493)
!2178 = !DILocation(line: 760, column: 31, scope: !1493)
!2179 = !DILocation(line: 761, column: 21, scope: !1493)
!2180 = !DILocation(line: 761, column: 41, scope: !1493)
!2181 = !DILocation(line: 761, column: 50, scope: !1493)
!2182 = !DILocation(line: 761, column: 48, scope: !1493)
!2183 = !DILocation(line: 761, column: 40, scope: !1493)
!2184 = !DILocation(line: 761, column: 55, scope: !1493)
!2185 = !DILocation(line: 761, column: 65, scope: !1493)
!2186 = !DILocation(line: 761, column: 67, scope: !1493)
!2187 = !DILocation(line: 761, column: 31, scope: !1493)
!2188 = !DILocation(line: 762, column: 48, scope: !1493)
!2189 = !DILocation(line: 762, column: 63, scope: !1493)
!2190 = !DILocation(line: 762, column: 72, scope: !1493)
!2191 = !DILocation(line: 762, column: 70, scope: !1493)
!2192 = !DILocation(line: 762, column: 87, scope: !1493)
!2193 = !DILocation(line: 762, column: 29, scope: !1493)
!2194 = !DILocation(line: 763, column: 27, scope: !1493)
!2195 = !DILocation(line: 763, column: 32, scope: !1493)
!2196 = !DILocation(line: 763, column: 40, scope: !1493)
!2197 = !DILocation(line: 763, column: 45, scope: !1493)
!2198 = !DILocation(line: 762, column: 27, scope: !1493)
!2199 = !DILocation(line: 764, column: 17, scope: !1493)
!2200 = !DILocation(line: 765, column: 13, scope: !1493)
!2201 = !DILocation(line: 770, column: 21, scope: !1493)
!2202 = !DILocation(line: 770, column: 28, scope: !1493)
!2203 = !DILocation(line: 770, column: 45, scope: !1493)
!2204 = !DILocation(line: 771, column: 21, scope: !1493)
!2205 = !DILocation(line: 769, column: 17, scope: !1493)
!2206 = !DILocation(line: 773, column: 21, scope: !1493)
!2207 = !DILocation(line: 773, column: 28, scope: !1493)
!2208 = !DILocation(line: 773, column: 45, scope: !1493)
!2209 = !DILocation(line: 774, column: 21, scope: !1493)
!2210 = !DILocation(line: 772, column: 17, scope: !1493)
!2211 = !DILocation(line: 784, column: 22, scope: !1493)
!2212 = !DILocation(line: 784, column: 26, scope: !1493)
!2213 = !DILocation(line: 784, column: 34, scope: !1493)
!2214 = !DILocation(line: 784, column: 37, scope: !1493)
!2215 = !DILocation(line: 784, column: 17, scope: !1493)
!2216 = !DILocation(line: 785, column: 21, scope: !1493)
!2217 = !DILocation(line: 787, column: 26, scope: !1493)
!2218 = !DILocation(line: 787, column: 30, scope: !1493)
!2219 = !DILocation(line: 787, column: 37, scope: !1493)
!2220 = !DILocation(line: 787, column: 39, scope: !1493)
!2221 = !DILocation(line: 787, column: 21, scope: !1493)
!2222 = !DILocation(line: 788, column: 25, scope: !1493)
!2223 = !DILocation(line: 789, column: 40, scope: !1493)
!2224 = !DILocation(line: 789, column: 52, scope: !1493)
!2225 = !DILocation(line: 789, column: 55, scope: !1493)
!2226 = !DILocation(line: 789, column: 50, scope: !1493)
!2227 = !DILocation(line: 789, column: 66, scope: !1493)
!2228 = !DILocation(line: 789, column: 64, scope: !1493)
!2229 = !DILocation(line: 789, column: 60, scope: !1493)
!2230 = !DILocation(line: 789, column: 74, scope: !1493)
!2231 = !DILocation(line: 789, column: 72, scope: !1493)
!2232 = !DILocation(line: 789, column: 77, scope: !1493)
!2233 = !DILocation(line: 788, column: 40, scope: !1493)
!2234 = !DILocation(line: 790, column: 53, scope: !1493)
!2235 = !DILocation(line: 790, column: 62, scope: !1493)
!2236 = !DILocation(line: 790, column: 37, scope: !1493)
!2237 = !DILocation(line: 790, column: 32, scope: !1493)
!2238 = !DILocation(line: 790, column: 25, scope: !1493)
!2239 = !DILocation(line: 790, column: 35, scope: !1493)
!2240 = !DILocation(line: 791, column: 21, scope: !1493)
!2241 = !DILocation(line: 787, column: 45, scope: !1493)
!2242 = distinct !{!2242, !2221, !2240, !50, !305}
!2243 = !DILocation(line: 793, column: 26, scope: !1493)
!2244 = !DILocation(line: 793, column: 30, scope: !1493)
!2245 = !DILocation(line: 793, column: 38, scope: !1493)
!2246 = !DILocation(line: 793, column: 41, scope: !1493)
!2247 = !DILocation(line: 793, column: 21, scope: !1493)
!2248 = !DILocation(line: 794, column: 25, scope: !1493)
!2249 = !DILocation(line: 795, column: 40, scope: !1493)
!2250 = !DILocation(line: 795, column: 52, scope: !1493)
!2251 = !DILocation(line: 795, column: 55, scope: !1493)
!2252 = !DILocation(line: 795, column: 50, scope: !1493)
!2253 = !DILocation(line: 795, column: 62, scope: !1493)
!2254 = !DILocation(line: 795, column: 60, scope: !1493)
!2255 = !DILocation(line: 795, column: 70, scope: !1493)
!2256 = !DILocation(line: 794, column: 40, scope: !1493)
!2257 = !DILocation(line: 796, column: 25, scope: !1493)
!2258 = !DILocation(line: 796, column: 59, scope: !1493)
!2259 = !DILocation(line: 796, column: 68, scope: !1493)
!2260 = !DILocation(line: 796, column: 43, scope: !1493)
!2261 = !DILocation(line: 796, column: 37, scope: !1493)
!2262 = !DILocation(line: 797, column: 25, scope: !1493)
!2263 = !DILocation(line: 798, column: 50, scope: !1493)
!2264 = !DILocation(line: 798, column: 59, scope: !1493)
!2265 = !DILocation(line: 798, column: 36, scope: !1493)
!2266 = !DILocation(line: 797, column: 37, scope: !1493)
!2267 = !DILocation(line: 800, column: 30, scope: !1493)
!2268 = !DILocation(line: 800, column: 34, scope: !1493)
!2269 = !DILocation(line: 800, column: 41, scope: !1493)
!2270 = !DILocation(line: 800, column: 43, scope: !1493)
!2271 = !DILocation(line: 800, column: 25, scope: !1493)
!2272 = !DILocation(line: 801, column: 29, scope: !1493)
!2273 = !DILocation(line: 801, column: 62, scope: !1493)
!2274 = !DILocation(line: 801, column: 55, scope: !1493)
!2275 = !DILocation(line: 801, column: 66, scope: !1493)
!2276 = !DILocation(line: 801, column: 45, scope: !1493)
!2277 = !DILocation(line: 801, column: 41, scope: !1493)
!2278 = !DILocation(line: 803, column: 43, scope: !1493)
!2279 = !DILocation(line: 803, column: 46, scope: !1493)
!2280 = !DILocation(line: 803, column: 56, scope: !1493)
!2281 = !DILocation(line: 803, column: 52, scope: !1493)
!2282 = !DILocation(line: 803, column: 60, scope: !1493)
!2283 = !DILocation(line: 803, column: 64, scope: !1493)
!2284 = !DILocation(line: 803, column: 33, scope: !1493)
!2285 = !DILocation(line: 802, column: 33, scope: !1493)
!2286 = !DILocation(line: 802, column: 29, scope: !1493)
!2287 = !DILocation(line: 802, column: 37, scope: !1493)
!2288 = !DILocation(line: 802, column: 41, scope: !1493)
!2289 = !DILocation(line: 802, column: 44, scope: !1493)
!2290 = !DILocation(line: 804, column: 25, scope: !1493)
!2291 = !DILocation(line: 800, column: 49, scope: !1493)
!2292 = distinct !{!2292, !2271, !2290, !50, !305}
!2293 = !DILocation(line: 805, column: 21, scope: !1493)
!2294 = !DILocation(line: 793, column: 48, scope: !1493)
!2295 = distinct !{!2295, !2247, !2293, !50, !305}
!2296 = !DILocation(line: 806, column: 17, scope: !1493)
!2297 = !DILocation(line: 784, column: 44, scope: !1493)
!2298 = distinct !{!2298, !2215, !2296, !50, !305}
!2299 = !DILocation(line: 863, column: 17, scope: !1493)
!2300 = !DILocation(line: 864, column: 22, scope: !1493)
!2301 = !DILocation(line: 864, column: 21, scope: !1493)
!2302 = !DILocation(line: 869, column: 27, scope: !1493)
!2303 = !DILocation(line: 871, column: 26, scope: !1493)
!2304 = !DILocation(line: 871, column: 30, scope: !1493)
!2305 = !DILocation(line: 871, column: 37, scope: !1493)
!2306 = !DILocation(line: 871, column: 39, scope: !1493)
!2307 = !DILocation(line: 871, column: 21, scope: !1493)
!2308 = !DILocation(line: 872, column: 68, scope: !1493)
!2309 = !DILocation(line: 872, column: 61, scope: !1493)
!2310 = !DILocation(line: 872, column: 78, scope: !1493)
!2311 = !DILocation(line: 872, column: 73, scope: !1493)
!2312 = !DILocation(line: 872, column: 37, scope: !1493)
!2313 = !DILocation(line: 872, column: 55, scope: !1493)
!2314 = !DILocation(line: 872, column: 45, scope: !1493)
!2315 = !DILocation(line: 872, column: 25, scope: !1493)
!2316 = !DILocation(line: 872, column: 59, scope: !1493)
!2317 = !DILocation(line: 873, column: 66, scope: !1493)
!2318 = !DILocation(line: 873, column: 61, scope: !1493)
!2319 = !DILocation(line: 873, column: 37, scope: !1493)
!2320 = !DILocation(line: 873, column: 55, scope: !1493)
!2321 = !DILocation(line: 873, column: 45, scope: !1493)
!2322 = !DILocation(line: 873, column: 25, scope: !1493)
!2323 = !DILocation(line: 873, column: 59, scope: !1493)
!2324 = !DILocation(line: 874, column: 21, scope: !1493)
!2325 = !DILocation(line: 871, column: 45, scope: !1493)
!2326 = distinct !{!2326, !2307, !2324, !50, !305}
!2327 = !DILocation(line: 875, column: 21, scope: !1493)
!2328 = !DILocation(line: 875, column: 38, scope: !1493)
!2329 = !DILocation(line: 875, column: 40, scope: !1493)
!2330 = !DILocation(line: 875, column: 31, scope: !1493)
!2331 = !DILocation(line: 876, column: 21, scope: !1493)
!2332 = !DILocation(line: 876, column: 55, scope: !1493)
!2333 = !DILocation(line: 876, column: 60, scope: !1493)
!2334 = !DILocation(line: 876, column: 54, scope: !1493)
!2335 = !DILocation(line: 876, column: 51, scope: !1493)
!2336 = !DILocation(line: 876, column: 40, scope: !1493)
!2337 = !DILocation(line: 878, column: 25, scope: !1493)
!2338 = !DILocation(line: 878, column: 45, scope: !1493)
!2339 = !DILocation(line: 878, column: 51, scope: !1493)
!2340 = !DILocation(line: 878, column: 55, scope: !1493)
!2341 = !DILocation(line: 878, column: 48, scope: !1493)
!2342 = !DILocation(line: 878, column: 35, scope: !1493)
!2343 = !DILocation(line: 879, column: 74, scope: !1493)
!2344 = !DILocation(line: 879, column: 84, scope: !1493)
!2345 = !DILocation(line: 879, column: 82, scope: !1493)
!2346 = !DILocation(line: 879, column: 73, scope: !1493)
!2347 = !DILocation(line: 879, column: 89, scope: !1493)
!2348 = !DILocation(line: 879, column: 38, scope: !1493)
!2349 = !DILocation(line: 879, column: 46, scope: !1493)
!2350 = !DILocation(line: 879, column: 50, scope: !1493)
!2351 = !DILocation(line: 879, column: 56, scope: !1493)
!2352 = !DILocation(line: 879, column: 25, scope: !1493)
!2353 = !DILocation(line: 879, column: 62, scope: !1493)
!2354 = !DILocation(line: 879, column: 66, scope: !1493)
!2355 = !DILocation(line: 879, column: 71, scope: !1493)
!2356 = !DILocation(line: 880, column: 21, scope: !1493)
!2357 = !DILocation(line: 881, column: 21, scope: !1493)
!2358 = !DILocation(line: 881, column: 55, scope: !1493)
!2359 = !DILocation(line: 881, column: 60, scope: !1493)
!2360 = !DILocation(line: 881, column: 54, scope: !1493)
!2361 = !DILocation(line: 881, column: 51, scope: !1493)
!2362 = !DILocation(line: 881, column: 40, scope: !1493)
!2363 = !DILocation(line: 883, column: 25, scope: !1493)
!2364 = !DILocation(line: 883, column: 45, scope: !1493)
!2365 = !DILocation(line: 883, column: 51, scope: !1493)
!2366 = !DILocation(line: 883, column: 55, scope: !1493)
!2367 = !DILocation(line: 883, column: 48, scope: !1493)
!2368 = !DILocation(line: 883, column: 35, scope: !1493)
!2369 = !DILocation(line: 884, column: 25, scope: !1493)
!2370 = !DILocation(line: 884, column: 43, scope: !1493)
!2371 = !DILocation(line: 884, column: 47, scope: !1493)
!2372 = !DILocation(line: 884, column: 35, scope: !1493)
!2373 = !DILocation(line: 885, column: 25, scope: !1493)
!2374 = !DILocation(line: 885, column: 42, scope: !1493)
!2375 = !DILocation(line: 885, column: 48, scope: !1493)
!2376 = !DILocation(line: 885, column: 41, scope: !1493)
!2377 = !DILocation(line: 886, column: 44, scope: !1493)
!2378 = !DILocation(line: 886, column: 78, scope: !1493)
!2379 = !DILocation(line: 886, column: 84, scope: !1493)
!2380 = !DILocation(line: 886, column: 77, scope: !1493)
!2381 = !DILocation(line: 886, column: 31, scope: !1493)
!2382 = !DILocation(line: 887, column: 44, scope: !1493)
!2383 = !DILocation(line: 887, column: 78, scope: !1493)
!2384 = !DILocation(line: 887, column: 84, scope: !1493)
!2385 = !DILocation(line: 887, column: 77, scope: !1493)
!2386 = !DILocation(line: 887, column: 31, scope: !1493)
!2387 = !DILocation(line: 885, column: 37, scope: !1493)
!2388 = !DILocation(line: 888, column: 69, scope: !1493)
!2389 = !DILocation(line: 888, column: 79, scope: !1493)
!2390 = !DILocation(line: 888, column: 77, scope: !1493)
!2391 = !DILocation(line: 888, column: 68, scope: !1493)
!2392 = !DILocation(line: 888, column: 84, scope: !1493)
!2393 = !DILocation(line: 888, column: 35, scope: !1493)
!2394 = !DILocation(line: 888, column: 43, scope: !1493)
!2395 = !DILocation(line: 888, column: 47, scope: !1493)
!2396 = !DILocation(line: 888, column: 53, scope: !1493)
!2397 = !DILocation(line: 888, column: 25, scope: !1493)
!2398 = !DILocation(line: 888, column: 59, scope: !1493)
!2399 = !DILocation(line: 888, column: 66, scope: !1493)
!2400 = !DILocation(line: 889, column: 21, scope: !1493)
!2401 = !DILocation(line: 890, column: 17, scope: !1493)
!2402 = !DILocation(line: 894, column: 17, scope: !1493)
!2403 = !DILocation(line: 896, column: 9, scope: !1493)
!2404 = !DILocation(line: 655, column: 33, scope: !1493)
!2405 = distinct !{!2405, !1917, !2403, !50, !1355}
!2406 = !DILocation(line: 897, column: 5, scope: !1493)
!2407 = !DILocation(line: 649, column: 42, scope: !1493)
!2408 = distinct !{!2408, !1912, !2406, !50}
!2409 = !DILocation(line: 908, column: 5, scope: !1493)
!2410 = !DILocation(line: 908, column: 12, scope: !1493)
!2411 = !DILocation(line: 910, column: 10, scope: !1493)
!2412 = !DILocation(line: 910, column: 14, scope: !1493)
!2413 = !DILocation(line: 910, column: 22, scope: !1493)
!2414 = !DILocation(line: 910, column: 25, scope: !1493)
!2415 = !DILocation(line: 910, column: 5, scope: !1493)
!2416 = !DILocation(line: 912, column: 14, scope: !1493)
!2417 = !DILocation(line: 912, column: 18, scope: !1493)
!2418 = !DILocation(line: 912, column: 26, scope: !1493)
!2419 = !DILocation(line: 912, column: 29, scope: !1493)
!2420 = !DILocation(line: 912, column: 9, scope: !1493)
!2421 = !DILocation(line: 914, column: 18, scope: !1493)
!2422 = !DILocation(line: 914, column: 22, scope: !1493)
!2423 = !DILocation(line: 914, column: 29, scope: !1493)
!2424 = !DILocation(line: 914, column: 31, scope: !1493)
!2425 = !DILocation(line: 914, column: 13, scope: !1493)
!2426 = !DILocation(line: 915, column: 85, scope: !1493)
!2427 = !DILocation(line: 915, column: 81, scope: !1493)
!2428 = !DILocation(line: 915, column: 89, scope: !1493)
!2429 = !DILocation(line: 915, column: 93, scope: !1493)
!2430 = !DILocation(line: 915, column: 17, scope: !1493)
!2431 = !DILocation(line: 915, column: 34, scope: !1493)
!2432 = !DILocation(line: 915, column: 39, scope: !1493)
!2433 = !DILocation(line: 915, column: 52, scope: !1493)
!2434 = !DILocation(line: 915, column: 50, scope: !1493)
!2435 = !DILocation(line: 915, column: 60, scope: !1493)
!2436 = !DILocation(line: 915, column: 58, scope: !1493)
!2437 = !DILocation(line: 915, column: 63, scope: !1493)
!2438 = !DILocation(line: 915, column: 45, scope: !1493)
!2439 = !DILocation(line: 915, column: 70, scope: !1493)
!2440 = !DILocation(line: 915, column: 68, scope: !1493)
!2441 = !DILocation(line: 915, column: 79, scope: !1493)
!2442 = !DILocation(line: 916, column: 13, scope: !1493)
!2443 = !DILocation(line: 914, column: 37, scope: !1493)
!2444 = distinct !{!2444, !2425, !2442, !50, !305}
!2445 = !DILocation(line: 917, column: 13, scope: !1493)
!2446 = !DILocation(line: 919, column: 18, scope: !1493)
!2447 = !DILocation(line: 919, column: 22, scope: !1493)
!2448 = !DILocation(line: 919, column: 29, scope: !1493)
!2449 = !DILocation(line: 919, column: 31, scope: !1493)
!2450 = !DILocation(line: 919, column: 13, scope: !1493)
!2451 = !DILocation(line: 920, column: 17, scope: !1493)
!2452 = !DILocation(line: 920, column: 31, scope: !1493)
!2453 = !DILocation(line: 920, column: 33, scope: !1493)
!2454 = !DILocation(line: 920, column: 27, scope: !1493)
!2455 = !DILocation(line: 921, column: 17, scope: !1493)
!2456 = !DILocation(line: 921, column: 31, scope: !1493)
!2457 = !DILocation(line: 921, column: 33, scope: !1493)
!2458 = !DILocation(line: 921, column: 27, scope: !1493)
!2459 = !DILocation(line: 922, column: 17, scope: !1493)
!2460 = !DILocation(line: 922, column: 31, scope: !1493)
!2461 = !DILocation(line: 922, column: 35, scope: !1493)
!2462 = !DILocation(line: 922, column: 27, scope: !1493)
!2463 = !DILocation(line: 923, column: 17, scope: !1493)
!2464 = !DILocation(line: 923, column: 32, scope: !1493)
!2465 = !DILocation(line: 923, column: 36, scope: !1493)
!2466 = !DILocation(line: 923, column: 42, scope: !1493)
!2467 = !DILocation(line: 923, column: 27, scope: !1493)
!2468 = !DILocation(line: 924, column: 17, scope: !1493)
!2469 = !DILocation(line: 924, column: 37, scope: !1493)
!2470 = !DILocation(line: 924, column: 39, scope: !1493)
!2471 = !DILocation(line: 924, column: 46, scope: !1493)
!2472 = !DILocation(line: 924, column: 49, scope: !1493)
!2473 = !DILocation(line: 924, column: 44, scope: !1493)
!2474 = !DILocation(line: 924, column: 56, scope: !1493)
!2475 = !DILocation(line: 924, column: 54, scope: !1493)
!2476 = !DILocation(line: 924, column: 27, scope: !1493)
!2477 = !DILocation(line: 925, column: 17, scope: !1493)
!2478 = !DILocation(line: 925, column: 37, scope: !1493)
!2479 = !DILocation(line: 925, column: 39, scope: !1493)
!2480 = !DILocation(line: 925, column: 46, scope: !1493)
!2481 = !DILocation(line: 925, column: 49, scope: !1493)
!2482 = !DILocation(line: 925, column: 44, scope: !1493)
!2483 = !DILocation(line: 925, column: 56, scope: !1493)
!2484 = !DILocation(line: 925, column: 54, scope: !1493)
!2485 = !DILocation(line: 925, column: 27, scope: !1493)
!2486 = !DILocation(line: 926, column: 17, scope: !1493)
!2487 = !DILocation(line: 926, column: 36, scope: !1493)
!2488 = !DILocation(line: 926, column: 38, scope: !1493)
!2489 = !DILocation(line: 926, column: 46, scope: !1493)
!2490 = !DILocation(line: 926, column: 44, scope: !1493)
!2491 = !DILocation(line: 926, column: 27, scope: !1493)
!2492 = !DILocation(line: 927, column: 17, scope: !1493)
!2493 = !DILocation(line: 927, column: 33, scope: !1493)
!2494 = !DILocation(line: 927, column: 50, scope: !1493)
!2495 = !DILocation(line: 927, column: 56, scope: !1493)
!2496 = !DILocation(line: 927, column: 64, scope: !1493)
!2497 = !DILocation(line: 927, column: 66, scope: !1493)
!2498 = !DILocation(line: 927, column: 62, scope: !1493)
!2499 = !DILocation(line: 927, column: 73, scope: !1493)
!2500 = !DILocation(line: 927, column: 71, scope: !1493)
!2501 = !DILocation(line: 927, column: 29, scope: !1493)
!2502 = !DILocation(line: 928, column: 17, scope: !1493)
!2503 = !DILocation(line: 928, column: 33, scope: !1493)
!2504 = !DILocation(line: 928, column: 38, scope: !1493)
!2505 = !DILocation(line: 928, column: 36, scope: !1493)
!2506 = !DILocation(line: 928, column: 27, scope: !1493)
!2507 = !DILocation(line: 929, column: 17, scope: !1493)
!2508 = !DILocation(line: 929, column: 33, scope: !1493)
!2509 = !DILocation(line: 929, column: 38, scope: !1493)
!2510 = !DILocation(line: 929, column: 36, scope: !1493)
!2511 = !DILocation(line: 929, column: 27, scope: !1493)
!2512 = !DILocation(line: 930, column: 21, scope: !1493)
!2513 = !DILocation(line: 930, column: 27, scope: !1493)
!2514 = !DILocation(line: 930, column: 25, scope: !1493)
!2515 = !DILocation(line: 930, column: 29, scope: !1493)
!2516 = !DILocation(line: 930, column: 32, scope: !1493)
!2517 = !DILocation(line: 930, column: 38, scope: !1493)
!2518 = !DILocation(line: 930, column: 36, scope: !1493)
!2519 = !DILocation(line: 931, column: 21, scope: !1493)
!2520 = !DILocation(line: 931, column: 33, scope: !1493)
!2521 = !DILocation(line: 931, column: 48, scope: !1493)
!2522 = !DILocation(line: 931, column: 54, scope: !1493)
!2523 = !DILocation(line: 931, column: 52, scope: !1493)
!2524 = !DILocation(line: 931, column: 35, scope: !1493)
!2525 = !DILocation(line: 931, column: 58, scope: !1493)
!2526 = !DILocation(line: 931, column: 56, scope: !1493)
!2527 = !DILocation(line: 931, column: 28, scope: !1493)
!2528 = !DILocation(line: 935, column: 31, scope: !1493)
!2529 = !DILocation(line: 935, column: 26, scope: !1493)
!2530 = !DILocation(line: 935, column: 29, scope: !1493)
!2531 = !DILocation(line: 937, column: 17, scope: !1493)
!2532 = !DILocation(line: 938, column: 13, scope: !1493)
!2533 = !DILocation(line: 919, column: 37, scope: !1493)
!2534 = distinct !{!2534, !2450, !2532, !50, !305}
!2535 = !DILocation(line: 939, column: 13, scope: !1493)
!2536 = !DILocation(line: 940, column: 9, scope: !1493)
!2537 = !DILocation(line: 912, column: 36, scope: !1493)
!2538 = distinct !{!2538, !2420, !2536, !50, !305}
!2539 = !DILocation(line: 941, column: 5, scope: !1493)
!2540 = !DILocation(line: 910, column: 32, scope: !1493)
!2541 = distinct !{!2541, !2415, !2539, !50, !305}
!2542 = !DILocation(line: 942, column: 1, scope: !1493)
!2543 = distinct !DISubprogram(name: "gemm_mq4g256v2_residual_mmq_iu4_full_add", scope: !127, file: !127, line: 960, type: !19, scopeLine: 965, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!2544 = !DILocation(line: 966, column: 54, scope: !2543)
!2545 = !DILocation(line: 966, column: 57, scope: !2543)
!2546 = !DILocation(line: 966, column: 61, scope: !2543)
!2547 = !DILocation(line: 966, column: 64, scope: !2543)
!2548 = !DILocation(line: 966, column: 67, scope: !2543)
!2549 = !DILocation(line: 966, column: 70, scope: !2543)
!2550 = !DILocation(line: 966, column: 5, scope: !2543)
!2551 = !DILocation(line: 967, column: 1, scope: !2543)
!2552 = distinct !DISubprogram(name: "gemm_mq4g256v2_residual_mmq_iu4_full_set", scope: !127, file: !127, line: 970, type: !19, scopeLine: 975, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!2553 = !DILocation(line: 976, column: 55, scope: !2552)
!2554 = !DILocation(line: 976, column: 58, scope: !2552)
!2555 = !DILocation(line: 976, column: 62, scope: !2552)
!2556 = !DILocation(line: 976, column: 65, scope: !2552)
!2557 = !DILocation(line: 976, column: 68, scope: !2552)
!2558 = !DILocation(line: 976, column: 71, scope: !2552)
!2559 = !DILocation(line: 976, column: 5, scope: !2552)
!2560 = !DILocation(line: 977, column: 1, scope: !2552)
!2561 = distinct !DISubprogram(name: "__hip_get_block_idx_y", scope: !203, file: !203, line: 254, type: !19, scopeLine: 254, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!2562 = !DILocation(line: 254, column: 116, scope: !2561)
!2563 = !DILocation(line: 254, column: 109, scope: !2561)
!2564 = distinct !DISubprogram(name: "__hip_get_block_idx_x", scope: !203, file: !203, line: 253, type: !19, scopeLine: 253, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!2565 = !DILocation(line: 253, column: 116, scope: !2564)
!2566 = !DILocation(line: 253, column: 109, scope: !2564)
!2567 = distinct !DISubprogram(name: "__hip_get_block_dim_x", scope: !203, file: !203, line: 258, type: !19, scopeLine: 258, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!2568 = !DILocation(line: 258, column: 116, scope: !2567)
!2569 = !DILocation(line: 258, column: 109, scope: !2567)
!2570 = distinct !DISubprogram(name: "__hip_get_thread_idx_x", scope: !203, file: !203, line: 248, type: !19, scopeLine: 248, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!2571 = !DILocation(line: 248, column: 117, scope: !2570)
!2572 = !DILocation(line: 248, column: 110, scope: !2570)
!2573 = distinct !DISubprogram(name: "HIP_vector_type<float, float, float, float, nullptr>", scope: !216, file: !216, line: 300, type: !19, scopeLine: 301, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!2574 = !{!2575, !2575, i64 0}
!2575 = !{!"p1 _ZTS15HIP_vector_typeIfLj4EE", !30, i64 0}
!2576 = !DILocation(line: 301, column: 49, scope: !2573)
!2577 = !DILocation(line: 301, column: 9, scope: !2573)
!2578 = !DILocation(line: 301, column: 58, scope: !2573)
!2579 = distinct !DISubprogram(name: "HIP_vector_base", scope: !216, file: !216, line: 265, type: !19, scopeLine: 266, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!2580 = !{!2581, !2581, i64 0}
!2581 = !{!"p1 _ZTS15HIP_vector_baseIfLj4EE", !30, i64 0}
!2582 = !DILocation(line: 266, column: 9, scope: !2579)
!2583 = !DILocation(line: 266, column: 11, scope: !2579)
!2584 = !DILocation(line: 266, column: 16, scope: !2579)
!2585 = !DILocation(line: 266, column: 18, scope: !2579)
!2586 = !DILocation(line: 266, column: 23, scope: !2579)
!2587 = !DILocation(line: 266, column: 25, scope: !2579)
!2588 = !DILocation(line: 266, column: 30, scope: !2579)
!2589 = !DILocation(line: 266, column: 32, scope: !2579)
!2590 = !DILocation(line: 266, column: 37, scope: !2579)
!2591 = distinct !DISubprogram(name: "fmaxf", scope: !2592, file: !2592, line: 454, type: !19, scopeLine: 454, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!2592 = !DIFile(filename: "/opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h", directory: "")
!2593 = !DILocation(line: 454, column: 60, scope: !2591)
!2594 = !DILocation(line: 454, column: 65, scope: !2591)
!2595 = !DILocation(line: 454, column: 44, scope: !2591)
!2596 = !DILocation(line: 454, column: 37, scope: !2591)
!2597 = distinct !DISubprogram(name: "fabsf", scope: !2592, file: !2592, line: 437, type: !19, scopeLine: 437, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!2598 = !DILocation(line: 437, column: 49, scope: !2597)
!2599 = !DILocation(line: 437, column: 33, scope: !2597)
!2600 = !DILocation(line: 437, column: 26, scope: !2597)
!2601 = distinct !DISubprogram(name: "__shfl_xor", scope: !2602, file: !2602, line: 545, type: !19, scopeLine: 545, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!2602 = !DIFile(filename: "/opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h", directory: "")
!2603 = !DILocation(line: 546, column: 3, scope: !2601)
!2604 = !DILocation(line: 551, column: 11, scope: !2601)
!2605 = !DILocation(line: 551, column: 9, scope: !2601)
!2606 = !DILocation(line: 552, column: 26, scope: !2601)
!2607 = !DILocation(line: 552, column: 29, scope: !2601)
!2608 = !DILocation(line: 552, column: 40, scope: !2601)
!2609 = !DILocation(line: 552, column: 11, scope: !2601)
!2610 = !DILocation(line: 552, column: 9, scope: !2601)
!2611 = !DILocation(line: 553, column: 14, scope: !2601)
!2612 = !DILocation(line: 554, column: 1, scope: !2601)
!2613 = !DILocation(line: 553, column: 3, scope: !2601)
!2614 = distinct !DISubprogram(name: "rintf", scope: !2592, file: !2592, line: 643, type: !19, scopeLine: 643, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!2615 = !DILocation(line: 643, column: 49, scope: !2614)
!2616 = !DILocation(line: 643, column: 33, scope: !2614)
!2617 = !DILocation(line: 643, column: 26, scope: !2614)
!2618 = distinct !DISubprogram(name: "fminf", scope: !2592, file: !2592, line: 457, type: !19, scopeLine: 457, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!2619 = !DILocation(line: 457, column: 60, scope: !2618)
!2620 = !DILocation(line: 457, column: 65, scope: !2618)
!2621 = !DILocation(line: 457, column: 44, scope: !2618)
!2622 = !DILocation(line: 457, column: 37, scope: !2618)
!2623 = distinct !DISubprogram(name: "__fmaf_rn", scope: !2592, file: !2592, line: 255, type: !19, scopeLine: 255, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!2624 = !DILocation(line: 256, column: 25, scope: !2623)
!2625 = !DILocation(line: 256, column: 30, scope: !2623)
!2626 = !DILocation(line: 256, column: 35, scope: !2623)
!2627 = !DILocation(line: 256, column: 10, scope: !2623)
!2628 = !DILocation(line: 256, column: 3, scope: !2623)
!2629 = distinct !DISubprogram(name: "__shfl_xor", scope: !2602, file: !2602, line: 521, type: !19, scopeLine: 521, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!2630 = !DILocation(line: 522, column: 3, scope: !2629)
!2631 = !DILocation(line: 522, column: 14, scope: !2629)
!2632 = !DILocation(line: 522, column: 7, scope: !2629)
!2633 = !DILocation(line: 523, column: 3, scope: !2629)
!2634 = !DILocation(line: 523, column: 15, scope: !2629)
!2635 = !DILocation(line: 523, column: 22, scope: !2629)
!2636 = !DILocation(line: 523, column: 20, scope: !2629)
!2637 = !DILocation(line: 523, column: 7, scope: !2629)
!2638 = !DILocation(line: 524, column: 11, scope: !2629)
!2639 = !DILocation(line: 524, column: 22, scope: !2629)
!2640 = !DILocation(line: 524, column: 29, scope: !2629)
!2641 = !DILocation(line: 524, column: 27, scope: !2629)
!2642 = !DILocation(line: 524, column: 40, scope: !2629)
!2643 = !DILocation(line: 524, column: 46, scope: !2629)
!2644 = !DILocation(line: 524, column: 38, scope: !2629)
!2645 = !DILocation(line: 524, column: 36, scope: !2629)
!2646 = !DILocation(line: 524, column: 17, scope: !2629)
!2647 = !DILocation(line: 524, column: 54, scope: !2629)
!2648 = !DILocation(line: 524, column: 61, scope: !2629)
!2649 = !DILocation(line: 524, column: 9, scope: !2629)
!2650 = !DILocation(line: 527, column: 42, scope: !2629)
!2651 = !DILocation(line: 527, column: 47, scope: !2629)
!2652 = !DILocation(line: 527, column: 12, scope: !2629)
!2653 = !DILocation(line: 533, column: 1, scope: !2629)
!2654 = !DILocation(line: 527, column: 5, scope: !2629)
!2655 = distinct !DISubprogram(name: "__lane_id", scope: !2602, file: !2602, line: 139, type: !19, scopeLine: 139, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!2656 = !DILocation(line: 140, column: 24, scope: !2655)
!2657 = !DILocation(line: 140, column: 34, scope: !2655)
!2658 = !DILocation(line: 142, column: 14, scope: !2655)
!2659 = !DILocation(line: 142, column: 7, scope: !2655)
!2660 = !DILocation(line: 147, column: 44, scope: !2655)
!2661 = !DILocation(line: 147, column: 14, scope: !2655)
!2662 = !DILocation(line: 147, column: 7, scope: !2655)
!2663 = !DILocation(line: 149, column: 1, scope: !2655)
!2664 = distinct !DISubprogram(name: "operator int", scope: !2602, file: !2602, line: 112, type: !19, scopeLine: 112, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!2665 = !{!30, !30, i64 0}
!2666 = !DILocation(line: 114, column: 14, scope: !2664)
!2667 = !DILocation(line: 114, column: 7, scope: !2664)
!2668 = distinct !DISubprogram(name: "__half2float", scope: !2669, file: !2669, line: 485, type: !19, scopeLine: 485, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!2669 = !DIFile(filename: "/opt/rocm/core/include/hip/amd_detail/amd_hip_fp16.h", directory: "")
!2670 = !DILocation(line: 485, column: 92, scope: !2668)
!2671 = !DILocation(line: 485, column: 116, scope: !2668)
!2672 = !DILocation(line: 485, column: 119, scope: !2668)
!2673 = !DILocation(line: 485, column: 85, scope: !2668)
!2674 = distinct !DISubprogram(name: "__ushort_as_half", scope: !2669, file: !2669, line: 456, type: !19, scopeLine: 456, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!2675 = !{!2676, !2676, i64 0}
!2676 = !{!"short", !10, i64 0}
!2677 = !DILocation(line: 457, column: 3, scope: !2674)
!2678 = !DILocation(line: 458, column: 9, scope: !2674)
!2679 = !DILocation(line: 458, column: 5, scope: !2674)
!2680 = !DILocation(line: 458, column: 7, scope: !2674)
!2681 = !DILocation(line: 459, column: 10, scope: !2674)
!2682 = !DILocation(line: 460, column: 1, scope: !2674)
!2683 = distinct !DISubprogram(name: "__syncthreads", scope: !2684, file: !2684, line: 720, type: !19, scopeLine: 720, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!2684 = !DIFile(filename: "/opt/rocm/core/include/hip/amd_detail/amd_device_functions.h", directory: "")
!2685 = !DILocation(line: 721, column: 3, scope: !2683)
!2686 = !DILocation(line: 722, column: 1, scope: !2683)
!2687 = distinct !DISubprogram(name: "iu4_bundle_single_acc<0>", scope: !127, file: !127, line: 449, type: !19, scopeLine: 458, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!2688 = !{!2689, !2689, i64 0}
!2689 = !{!"p1 int", !30, i64 0}
!2690 = !DILocation(line: 459, column: 5, scope: !2687)
!2691 = !DILocation(line: 459, column: 47, scope: !2687)
!2692 = !DILocation(line: 459, column: 55, scope: !2687)
!2693 = !{i64 4}
!2694 = !DILocation(line: 459, column: 27, scope: !2687)
!2695 = !DILocation(line: 459, column: 21, scope: !2687)
!2696 = !DILocation(line: 460, column: 5, scope: !2687)
!2697 = !DILocation(line: 460, column: 47, scope: !2687)
!2698 = !DILocation(line: 460, column: 55, scope: !2687)
!2699 = !DILocation(line: 460, column: 27, scope: !2687)
!2700 = !DILocation(line: 460, column: 21, scope: !2687)
!2701 = !DILocation(line: 461, column: 5, scope: !2687)
!2702 = !DILocation(line: 461, column: 47, scope: !2687)
!2703 = !DILocation(line: 461, column: 55, scope: !2687)
!2704 = !DILocation(line: 461, column: 27, scope: !2687)
!2705 = !DILocation(line: 461, column: 21, scope: !2687)
!2706 = !DILocation(line: 462, column: 5, scope: !2687)
!2707 = !DILocation(line: 462, column: 47, scope: !2687)
!2708 = !DILocation(line: 462, column: 55, scope: !2687)
!2709 = !DILocation(line: 462, column: 27, scope: !2687)
!2710 = !DILocation(line: 462, column: 21, scope: !2687)
!2711 = !DILocation(line: 463, column: 5, scope: !2687)
!2712 = !DILocation(line: 463, column: 47, scope: !2687)
!2713 = !DILocation(line: 463, column: 55, scope: !2687)
!2714 = !DILocation(line: 463, column: 27, scope: !2687)
!2715 = !DILocation(line: 463, column: 21, scope: !2687)
!2716 = !DILocation(line: 464, column: 5, scope: !2687)
!2717 = !DILocation(line: 464, column: 47, scope: !2687)
!2718 = !DILocation(line: 464, column: 55, scope: !2687)
!2719 = !DILocation(line: 464, column: 27, scope: !2687)
!2720 = !DILocation(line: 464, column: 21, scope: !2687)
!2721 = !DILocation(line: 466, column: 10, scope: !2687)
!2722 = !DILocation(line: 466, column: 14, scope: !2687)
!2723 = !DILocation(line: 466, column: 22, scope: !2687)
!2724 = !DILocation(line: 466, column: 25, scope: !2687)
!2725 = !DILocation(line: 466, column: 5, scope: !2687)
!2726 = !DILocation(line: 467, column: 9, scope: !2687)
!2727 = !DILocation(line: 467, column: 30, scope: !2687)
!2728 = !DILocation(line: 467, column: 33, scope: !2687)
!2729 = !DILocation(line: 467, column: 29, scope: !2687)
!2730 = !DILocation(line: 467, column: 41, scope: !2687)
!2731 = !DILocation(line: 467, column: 47, scope: !2687)
!2732 = !DILocation(line: 467, column: 25, scope: !2687)
!2733 = !DILocation(line: 469, column: 14, scope: !2687)
!2734 = !DILocation(line: 469, column: 18, scope: !2687)
!2735 = !DILocation(line: 469, column: 26, scope: !2687)
!2736 = !DILocation(line: 469, column: 29, scope: !2687)
!2737 = !DILocation(line: 469, column: 9, scope: !2687)
!2738 = !DILocation(line: 470, column: 13, scope: !2687)
!2739 = !DILocation(line: 470, column: 34, scope: !2687)
!2740 = !DILocation(line: 470, column: 37, scope: !2687)
!2741 = !DILocation(line: 470, column: 33, scope: !2687)
!2742 = !DILocation(line: 470, column: 45, scope: !2687)
!2743 = !DILocation(line: 470, column: 52, scope: !2687)
!2744 = !DILocation(line: 470, column: 55, scope: !2687)
!2745 = !DILocation(line: 470, column: 51, scope: !2687)
!2746 = !DILocation(line: 470, column: 63, scope: !2687)
!2747 = !DILocation(line: 470, column: 70, scope: !2687)
!2748 = !DILocation(line: 470, column: 73, scope: !2687)
!2749 = !DILocation(line: 470, column: 69, scope: !2687)
!2750 = !DILocation(line: 470, column: 81, scope: !2687)
!2751 = !DILocation(line: 470, column: 87, scope: !2687)
!2752 = !DILocation(line: 470, column: 29, scope: !2687)
!2753 = !DILocation(line: 471, column: 13, scope: !2687)
!2754 = !DILocation(line: 473, column: 28, scope: !2687)
!2755 = !DILocation(line: 473, column: 37, scope: !2687)
!2756 = !DILocation(line: 473, column: 40, scope: !2687)
!2757 = !{i64 32}
!2758 = !DILocation(line: 472, column: 17, scope: !2687)
!2759 = !DILocation(line: 471, column: 29, scope: !2687)
!2760 = !DILocation(line: 474, column: 13, scope: !2687)
!2761 = !DILocation(line: 474, column: 38, scope: !2687)
!2762 = !DILocation(line: 474, column: 46, scope: !2687)
!2763 = !DILocation(line: 474, column: 52, scope: !2687)
!2764 = !DILocation(line: 474, column: 60, scope: !2687)
!2765 = !DILocation(line: 474, column: 50, scope: !2687)
!2766 = !DILocation(line: 474, column: 28, scope: !2687)
!2767 = !DILocation(line: 475, column: 13, scope: !2687)
!2768 = !DILocation(line: 476, column: 41, scope: !2687)
!2769 = !DILocation(line: 476, column: 17, scope: !2687)
!2770 = !DILocation(line: 475, column: 28, scope: !2687)
!2771 = !DILocation(line: 478, column: 43, scope: !2687)
!2772 = !DILocation(line: 478, column: 52, scope: !2687)
!2773 = !DILocation(line: 478, column: 63, scope: !2687)
!2774 = !DILocation(line: 478, column: 67, scope: !2687)
!2775 = !DILocation(line: 478, column: 71, scope: !2687)
!2776 = !DILocation(line: 478, column: 17, scope: !2687)
!2777 = !DILocation(line: 477, column: 13, scope: !2687)
!2778 = !DILocation(line: 477, column: 17, scope: !2687)
!2779 = !DILocation(line: 477, column: 21, scope: !2687)
!2780 = !DILocation(line: 477, column: 25, scope: !2687)
!2781 = !DILocation(line: 479, column: 9, scope: !2687)
!2782 = !DILocation(line: 469, column: 36, scope: !2687)
!2783 = distinct !{!2783, !2737, !2781, !50, !305}
!2784 = !DILocation(line: 480, column: 5, scope: !2687)
!2785 = !DILocation(line: 466, column: 32, scope: !2687)
!2786 = distinct !{!2786, !2725, !2784, !50, !305}
!2787 = !DILocation(line: 481, column: 5, scope: !2687)
!2788 = !DILocation(line: 482, column: 1, scope: !2687)
!2789 = distinct !DISubprogram(name: "iu4_bundle_single_acc<1>", scope: !127, file: !127, line: 449, type: !19, scopeLine: 458, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!2790 = !DILocation(line: 459, column: 5, scope: !2789)
!2791 = !DILocation(line: 459, column: 47, scope: !2789)
!2792 = !DILocation(line: 459, column: 55, scope: !2789)
!2793 = !DILocation(line: 459, column: 27, scope: !2789)
!2794 = !DILocation(line: 459, column: 21, scope: !2789)
!2795 = !DILocation(line: 460, column: 5, scope: !2789)
!2796 = !DILocation(line: 460, column: 47, scope: !2789)
!2797 = !DILocation(line: 460, column: 55, scope: !2789)
!2798 = !DILocation(line: 460, column: 27, scope: !2789)
!2799 = !DILocation(line: 460, column: 21, scope: !2789)
!2800 = !DILocation(line: 461, column: 5, scope: !2789)
!2801 = !DILocation(line: 461, column: 47, scope: !2789)
!2802 = !DILocation(line: 461, column: 55, scope: !2789)
!2803 = !DILocation(line: 461, column: 27, scope: !2789)
!2804 = !DILocation(line: 461, column: 21, scope: !2789)
!2805 = !DILocation(line: 462, column: 5, scope: !2789)
!2806 = !DILocation(line: 462, column: 47, scope: !2789)
!2807 = !DILocation(line: 462, column: 55, scope: !2789)
!2808 = !DILocation(line: 462, column: 27, scope: !2789)
!2809 = !DILocation(line: 462, column: 21, scope: !2789)
!2810 = !DILocation(line: 463, column: 5, scope: !2789)
!2811 = !DILocation(line: 463, column: 47, scope: !2789)
!2812 = !DILocation(line: 463, column: 55, scope: !2789)
!2813 = !DILocation(line: 463, column: 27, scope: !2789)
!2814 = !DILocation(line: 463, column: 21, scope: !2789)
!2815 = !DILocation(line: 464, column: 5, scope: !2789)
!2816 = !DILocation(line: 464, column: 47, scope: !2789)
!2817 = !DILocation(line: 464, column: 55, scope: !2789)
!2818 = !DILocation(line: 464, column: 27, scope: !2789)
!2819 = !DILocation(line: 464, column: 21, scope: !2789)
!2820 = !DILocation(line: 466, column: 10, scope: !2789)
!2821 = !DILocation(line: 466, column: 14, scope: !2789)
!2822 = !DILocation(line: 466, column: 22, scope: !2789)
!2823 = !DILocation(line: 466, column: 25, scope: !2789)
!2824 = !DILocation(line: 466, column: 5, scope: !2789)
!2825 = !DILocation(line: 467, column: 9, scope: !2789)
!2826 = !DILocation(line: 467, column: 30, scope: !2789)
!2827 = !DILocation(line: 467, column: 33, scope: !2789)
!2828 = !DILocation(line: 467, column: 29, scope: !2789)
!2829 = !DILocation(line: 467, column: 41, scope: !2789)
!2830 = !DILocation(line: 467, column: 47, scope: !2789)
!2831 = !DILocation(line: 467, column: 25, scope: !2789)
!2832 = !DILocation(line: 469, column: 14, scope: !2789)
!2833 = !DILocation(line: 469, column: 18, scope: !2789)
!2834 = !DILocation(line: 469, column: 26, scope: !2789)
!2835 = !DILocation(line: 469, column: 29, scope: !2789)
!2836 = !DILocation(line: 469, column: 9, scope: !2789)
!2837 = !DILocation(line: 470, column: 13, scope: !2789)
!2838 = !DILocation(line: 470, column: 34, scope: !2789)
!2839 = !DILocation(line: 470, column: 37, scope: !2789)
!2840 = !DILocation(line: 470, column: 33, scope: !2789)
!2841 = !DILocation(line: 470, column: 45, scope: !2789)
!2842 = !DILocation(line: 470, column: 52, scope: !2789)
!2843 = !DILocation(line: 470, column: 55, scope: !2789)
!2844 = !DILocation(line: 470, column: 51, scope: !2789)
!2845 = !DILocation(line: 470, column: 63, scope: !2789)
!2846 = !DILocation(line: 470, column: 70, scope: !2789)
!2847 = !DILocation(line: 470, column: 73, scope: !2789)
!2848 = !DILocation(line: 470, column: 69, scope: !2789)
!2849 = !DILocation(line: 470, column: 81, scope: !2789)
!2850 = !DILocation(line: 470, column: 87, scope: !2789)
!2851 = !DILocation(line: 470, column: 29, scope: !2789)
!2852 = !DILocation(line: 471, column: 13, scope: !2789)
!2853 = !DILocation(line: 473, column: 28, scope: !2789)
!2854 = !DILocation(line: 473, column: 37, scope: !2789)
!2855 = !DILocation(line: 473, column: 40, scope: !2789)
!2856 = !DILocation(line: 472, column: 17, scope: !2789)
!2857 = !DILocation(line: 471, column: 29, scope: !2789)
!2858 = !DILocation(line: 474, column: 13, scope: !2789)
!2859 = !DILocation(line: 474, column: 38, scope: !2789)
!2860 = !DILocation(line: 474, column: 46, scope: !2789)
!2861 = !DILocation(line: 474, column: 52, scope: !2789)
!2862 = !DILocation(line: 474, column: 60, scope: !2789)
!2863 = !DILocation(line: 474, column: 50, scope: !2789)
!2864 = !DILocation(line: 474, column: 28, scope: !2789)
!2865 = !DILocation(line: 475, column: 13, scope: !2789)
!2866 = !DILocation(line: 476, column: 41, scope: !2789)
!2867 = !DILocation(line: 476, column: 17, scope: !2789)
!2868 = !DILocation(line: 475, column: 28, scope: !2789)
!2869 = !DILocation(line: 478, column: 43, scope: !2789)
!2870 = !DILocation(line: 478, column: 52, scope: !2789)
!2871 = !DILocation(line: 478, column: 63, scope: !2789)
!2872 = !DILocation(line: 478, column: 67, scope: !2789)
!2873 = !DILocation(line: 478, column: 71, scope: !2789)
!2874 = !DILocation(line: 478, column: 17, scope: !2789)
!2875 = !DILocation(line: 477, column: 13, scope: !2789)
!2876 = !DILocation(line: 477, column: 17, scope: !2789)
!2877 = !DILocation(line: 477, column: 21, scope: !2789)
!2878 = !DILocation(line: 477, column: 25, scope: !2789)
!2879 = !DILocation(line: 479, column: 9, scope: !2789)
!2880 = !DILocation(line: 469, column: 36, scope: !2789)
!2881 = distinct !{!2881, !2836, !2879, !50, !305}
!2882 = !DILocation(line: 480, column: 5, scope: !2789)
!2883 = !DILocation(line: 466, column: 32, scope: !2789)
!2884 = distinct !{!2884, !2824, !2882, !50, !305}
!2885 = !DILocation(line: 481, column: 5, scope: !2789)
!2886 = !DILocation(line: 482, column: 1, scope: !2789)
!2887 = distinct !DISubprogram(name: "__fmul_rn", scope: !2592, file: !2592, line: 271, type: !19, scopeLine: 271, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!2888 = !DILocation(line: 271, column: 48, scope: !2887)
!2889 = !DILocation(line: 271, column: 54, scope: !2887)
!2890 = !DILocation(line: 271, column: 52, scope: !2887)
!2891 = !DILocation(line: 271, column: 41, scope: !2887)
!2892 = distinct !DISubprogram(name: "operator __half_raw", scope: !2669, file: !2669, line: 207, type: !19, scopeLine: 207, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!2893 = !{!2894, !2894, i64 0}
!2894 = !{!"p1 _ZTS6__half", !30, i64 0}
!2895 = !DILocation(line: 207, column: 50, scope: !2892)
!2896 = !DILocation(line: 207, column: 51, scope: !2892)
!2897 = !DILocation(line: 207, column: 33, scope: !2892)
!2898 = distinct !DISubprogram(name: "__half", scope: !2669, file: !2669, line: 86, type: !19, scopeLine: 86, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!2899 = !{!2900, !2900, i64 0}
!2900 = !{!"p1 _ZTS10__half_raw", !30, i64 0}
!2901 = !DILocation(line: 86, column: 89, scope: !2898)
!2902 = !DILocation(line: 86, column: 93, scope: !2898)
!2903 = !{i64 2}
!2904 = !DILocation(line: 86, column: 95, scope: !2898)
!2905 = !DILocation(line: 86, column: 99, scope: !2898)
!2906 = distinct !DISubprogram(name: "__barrier", scope: !2684, file: !2684, line: 718, type: !19, scopeLine: 718, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!2907 = !DILocation(line: 718, column: 106, scope: !2906)
!2908 = !DILocation(line: 718, column: 63, scope: !2906)
!2909 = !DILocation(line: 718, column: 110, scope: !2906)
!2910 = distinct !DISubprogram(name: "__work_group_barrier", scope: !2684, file: !2684, line: 697, type: !19, scopeLine: 697, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!2911 = !DILocation(line: 700, column: 9, scope: !2910)
!2912 = !DILocation(line: 700, column: 15, scope: !2910)
!2913 = !DILocation(line: 701, column: 7, scope: !2910)
!2914 = !DILocation(line: 702, column: 7, scope: !2910)
!2915 = !DILocation(line: 703, column: 7, scope: !2910)
!2916 = !DILocation(line: 704, column: 5, scope: !2910)
!2917 = !DILocation(line: 704, column: 16, scope: !2910)
!2918 = !DILocation(line: 704, column: 22, scope: !2910)
!2919 = !DILocation(line: 705, column: 7, scope: !2910)
!2920 = !{!"amdgpu-synchronize-as", !"global"}
!2921 = !DILocation(line: 706, column: 7, scope: !2910)
!2922 = !DILocation(line: 707, column: 7, scope: !2910)
!2923 = !DILocation(line: 708, column: 5, scope: !2910)
!2924 = !DILocation(line: 708, column: 16, scope: !2910)
!2925 = !DILocation(line: 708, column: 22, scope: !2910)
!2926 = !DILocation(line: 709, column: 7, scope: !2910)
!2927 = !{!"amdgpu-synchronize-as", !"local"}
!2928 = !DILocation(line: 710, column: 7, scope: !2910)
!2929 = !DILocation(line: 711, column: 7, scope: !2910)
!2930 = !DILocation(line: 712, column: 5, scope: !2910)
!2931 = !DILocation(line: 713, column: 7, scope: !2910)
!2932 = !DILocation(line: 716, column: 1, scope: !2910)
!2933 = !{!2934, !2934, i64 0}
!2934 = !{!"long", !14, i64 0}
!2935 = !{!2936, !2937, i64 0}
!2936 = !{!"", !2937, i64 0, !2937, i64 8, !2938, i64 16, !2934, i64 24, !2934, i64 32, !2934, i64 40}
!2937 = !{!"any pointer", !14, i64 0}
!2938 = !{!"hsa_signal_s", !2934, i64 0}
!2939 = !{!2936, !2934, i64 40}
!2940 = !{!2936, !2937, i64 8}
!2941 = !{!2942, !13, i64 16}
!2942 = !{!"", !2934, i64 0, !2934, i64 8, !13, i64 16, !13, i64 20}
!2943 = !{!2942, !2934, i64 8}
!2944 = !{!2942, !13, i64 20}
!2945 = !{!2942, !2934, i64 0}
!2946 = !{!2947, !2934, i64 16}
!2947 = !{!"amd_signal_s", !2934, i64 0, !14, i64 8, !2934, i64 16, !13, i64 24, !13, i64 28, !2934, i64 32, !2934, i64 40, !14, i64 48, !14, i64 56}
!2948 = !{!2947, !13, i64 24}
!2949 = !{!14, !14, i64 0}
!2950 = !{!2951, !2952, i64 4}
!2951 = !{!"hsa_kernel_dispatch_packet_s", !2952, i64 0, !2952, i64 2, !2952, i64 4, !2952, i64 6, !2952, i64 8, !2952, i64 10, !13, i64 12, !13, i64 16, !13, i64 20, !13, i64 24, !13, i64 28, !14, i64 32, !2937, i64 40, !2934, i64 48, !2938, i64 56}
!2952 = !{!"short", !14, i64 0}
!2953 = !{!2951, !13, i64 12}
!2954 = !{!2952, !2952, i64 0}
!2955 = !{!2951, !2952, i64 6}
!2956 = !{!2951, !13, i64 16}
!2957 = !{!2951, !2952, i64 8}
!2958 = !{!2951, !13, i64 20}
