// Standalone carry-unit test: Kt panel + D + S update only, one-hot K.
// Proves/disproves the carry WMMA mapping in isolation.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#define CK(x) do { auto e=(x); if(e!=hipSuccess){fprintf(stderr,"HIP: %s\n",hipGetErrorString(e));exit(1);} }while(0)
typedef _Float16 __attribute__((ext_vector_type(8))) h8;
typedef float __attribute__((ext_vector_type(8))) f8;
__device__ __forceinline__ h8 halfv(f8 x){return __builtin_convertvector(x,h8);}
__device__ __forceinline__ f8 floatv(h8 x){return __builtin_convertvector(x,f8);}
__device__ __forceinline__ f8 mma(h8 a,h8 b,f8 c){return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a,b,c);}

__global__ void carry_unit(const float* K, float* Sout, int kh) {
    __shared__ _Float16 kt[4096];
    const int tid = threadIdx.x, lane = tid & 31;
    const int ml = lane & 15, kg = lane >> 4;
    // Kt fill: kt[r*64+tc] = K[tc][kh*64+r]
    for (int u = 0; u < 16; u++) {
        int x = tid + u * 256, r = x / 64, tc = x % 64;
        kt[x] = (__half)K[tc * 128 + kh * 64 + r];
    }
    __syncthreads();
    // S fragments (zero init) + D small ints: D[s][j] = (s*8+j+1)*0.5
    h8 S[4]; for (int b = 0; b < 4; b++) for (int j = 0; j < 8; j++) S[b][j] = 0;
    f8 D[4]; for (int s = 0; s < 4; s++) for (int j = 0; j < 8; j++) D[s][j] = (s * 8 + j + 1) * 0.5f;
    for (int kb = 0; kb < 4; kb++) {
        f8 acc = floatv((kb==0)?S[0]:(kb==1)?S[1]:(kb==2)?S[2]:S[3]);
        for (int s = 0; s < 4; s++) {
            h8 A, B;
            f8 Ds = (s==0)?D[0]:(s==1)?D[1]:(s==2)?D[2]:D[3];
            for (int j = 0; j < 8; j++) {
                int stok = s * 16 + kg * 8 + j;
                A[j] = kt[(kb * 16 + ml) * 64 + stok];
                B[j] = (__half)(1.0f * Ds[j]);
            }
            acc = mma(A, B, acc);
        }
        h8 r = halfv(acc);
        if (kb==0) S[0]=r; else if (kb==1) S[1]=r; else if (kb==2) S[2]=r; else S[3]=r;
    }
    __syncthreads();
    // Sout[value ml][key kh*64+b*16+kg*8+j] for value group 0 (this block = wave 0/1 only test: use wave in tid)
    int wave = tid >> 5, g = wave >> 1, khw = wave & 1;
    if (khw != kh) return;
    int vrow = g * 16 + ml;
    for (int b = 0; b < 4; b++) for (int j = 0; j < 8; j++)
        Sout[vrow * 128 + kh * 64 + b * 16 + kg * 8 + j] = (float)S[b][j];
}

int main() {
    // K one-hot 64x128 (rows 0..63 = e_0..e_63)
    std::vector<float> K(64 * 128, 0);
    for (int i = 0; i < 64; i++) K[i * 128 + i] = 1.0f;
    float *dK, *dS; CK(hipMalloc(&dK, K.size()*4)); CK(hipMalloc(&dS, 64*128*4));
    CK(hipMemcpy(dK, K.data(), K.size()*4, hipMemcpyHostToDevice));
    double worst = 0;
    for (int kh = 0; kh < 2; kh++) {
        CK(hipMemset(dS, 0, 64*128*4));
        hipLaunchKernelGGL(carry_unit, dim3(1), dim3(256), 0, 0, dK, dS, kh);
        CK(hipDeviceSynchronize());
        std::vector<float> S(64*128); CK(hipMemcpy(S.data(), dS, S.size()*4, hipMemcpyDeviceToHost));
        // Expected: S_out[v][c] = sum_j D_j[v]*K[j][c], D_j[v] = (tok grouped)... D[s][j] <-> token s*16+kg*8+j for value row g*16+ml.
        // K one-hot: K[j][c] = (j==c). S_out[v][c] = D_c[v] where D_c = delta of token c for value v.
        // D[s][j] = (s*8+j+1)*0.5 for token s*16+kg*8+j... per lane (ml,kg): token->value: (s*8+j+1)*0.5.
        for (int v = 0; v < 64; v++) for (int cc = 0; cc < 64; cc++) {
            int c = kh * 64 + cc;
            // token c: s = (c%16)/8... token = s*16+kg*8+j -> for token c: kg=(c>>3)&1, j=c&7, s=c>>4
            int s = c >> 4, j = c & 7;
            float expect = (c < 64) ? ((s * 8 + j + 1) * 0.5f) : 0.0f;
            // note: only tokens 0..63 exist; c in kh-half: kh=0 -> c 0..63 exact; kh=1 -> c 64..127 -> K col 64+ has no one-hot (K rows only 0..63 have e_0..63; K[j][64+cc]=0) -> expect 0
            float got = S[v * 128 + c];
            worst = std::max(worst, (double)fabs(got - expect));
        }
    }
    printf("carry_unit maxerr=%.4g %s\n", worst, worst < 1e-5 ? "PASS" : "FAIL");
    return worst < 1e-5 ? 0 : 1;
}
