// Direct emit kernel for the quant oracle: one wave per (group, token) calls
// emit_iu4_sidecar_from_producer8 on 256 floats in producer layout
// (lane p holds in[w*256 + 8p .. 8p+7]). Build with -DQFILE=\"<quant.hip>\".
#define HIPFIRE_BLOCK_I4_128_QUANT_NO_STANDALONE 1
#include QFILE

extern "C" __global__ void qemit(const float* __restrict__ in, block_i4_128* __restrict__ out,
                                 int N, int G) {
    const int w = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = threadIdx.x & 31;
    if (w >= N * G) return;
    const int token = w % N, group = w / N;
    const float* p = in + (long long)w * 256 + lane * 8;
    emit_iu4_sidecar_from_producer8(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7],
                                    out, group, token, N, lane);
}
