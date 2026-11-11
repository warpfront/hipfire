// R9700 (gfx1201) <-> Strix Halo (gfx1151) round-trip latency over the USB4 link,
// for the Qwen4 heterogeneous expert server (see tools/hetero/README.md).
//
// Device A (gfx1201, the primary) posts a payload and a flag into pinned, coherent
// host memory; device B (gfx1151, the expert server) answers with a payload and a
// flag; A waits and reads the answer. Modes:
//   A  GPU<->GPU: persistent kernels on both devices, no CPU involvement.
//   B  host-mediated: kernel + hipMemcpy + sync on each side per round trip.
//   C  CPU-driven responder: a host thread on B's side spins on A's flag (CPU-
//      coherent memory) and launches one captured hipGraph per request whose
//      kernel answers; A's side is the persistent kernel of mode A.
// Every spin is bounded and watches a shared abort word, so a missed flag can never
// hang a GPU. Shared words use plain loads/stores only (the link has no PCIe atomics).
//
// Build: hipcc -O2 -std=c++20 --offload-arch=gfx1151 --offload-arch=gfx1201 pingpong.hip -o pingpong
// Run:   ./pingpong [rounds=20000] [payload_bytes=5120] [modes=ABC]
#include <hip/hip_runtime.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("%s:%d %s -> %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e)); exit(1); } } while (0)

constexpr int MAX_WORDS = 5120 / 4;        // one bf16 hidden state of Flash-Next
constexpr long long MAX_POLLS = 20000000;  // per device wait; bail out instead of hanging
constexpr long long MAX_HOST_POLLS = 200000000;
constexpr int THREADS = 1024;              // spread the cross-link readback

struct Shared {
    int flag_a;  int pad0[31];             // A: "input for round i ready"
    int flag_b;  int pad1[31];             // B: "output for round i ready"
    int abort_;  int pad2[31];             // either side gave up
    int cur;     int pad3[31];             // mode C: round the graph kernel answers
    unsigned to_b[MAX_WORDS];
    unsigned to_a[MAX_WORDS];
};

__device__ inline int ld(int* p) {
    return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ inline void st(int* p, int v) {
    __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
static inline int host_ld(int* p) { return std::atomic_ref<int>(*p).load(std::memory_order_acquire); }
static inline void host_st(int* p, int v) { std::atomic_ref<int>(*p).store(v, std::memory_order_release); }

__device__ unsigned in_word(int i, int w)  { return i * 7919u + w; }
__device__ unsigned out_word(int i, int w) { return i * 104729u + w; }

// Primary: post round i, wait for the answer, check it.
__global__ void side_a(Shared* s, int rounds, int words, int* bad) {
    __shared__ int ok;
    for (int i = 1; i <= rounds; i++) {
        for (int w = threadIdx.x; w < words; w += blockDim.x) s->to_b[w] = in_word(i, w);
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence_system();
            st(&s->flag_a, i);
            long long n = 0;
            while (ld(&s->flag_b) != i && !ld(&s->abort_) && ++n < MAX_POLLS) {}
            ok = (n < MAX_POLLS) && !ld(&s->abort_);
            if (!ok) st(&s->abort_, 1);
        }
        __syncthreads();
        if (!ok) { if (threadIdx.x == 0) *bad = -i; return; }
        for (int w = threadIdx.x; w < words; w += blockDim.x)
            if (s->to_a[w] != out_word(i, w)) atomicAdd(bad, 1);
    }
}

// Expert server's answer for round i: check the input, write the output, flag it.
__device__ void answer(Shared* s, int i, int words, int* bad) {
    for (int w = threadIdx.x; w < words; w += blockDim.x) {
        if (s->to_b[w] != in_word(i, w)) atomicAdd(bad, 1);
        s->to_a[w] = out_word(i, w);
    }
    __syncthreads();
    if (threadIdx.x == 0) { __threadfence_system(); st(&s->flag_b, i); }
}

// Mode A expert server: persistent, polls on the device.
__global__ void side_b(Shared* s, int rounds, int words, int* bad) {
    __shared__ int ok;
    for (int i = 1; i <= rounds; i++) {
        if (threadIdx.x == 0) {
            long long n = 0;
            while (ld(&s->flag_a) != i && !ld(&s->abort_) && ++n < MAX_POLLS) {}
            ok = (n < MAX_POLLS) && !ld(&s->abort_);
            if (!ok) st(&s->abort_, 1);
        }
        __syncthreads();
        if (!ok) { if (threadIdx.x == 0) *bad = -i; return; }
        answer(s, i, words, bad);
    }
}

// Mode C expert server: one launch per request, round taken from s->cur.
__global__ void side_b_once(Shared* s, int words, int* bad) {
    __shared__ int i;
    if (threadIdx.x == 0) i = ld(&s->cur);
    __syncthreads();
    answer(s, i, words, bad);
}

__global__ void touch(unsigned* p, int n, unsigned v) {
    for (int w = threadIdx.x; w < n; w += blockDim.x) p[w] += v;
}

static int find(const char* arch) {
    int n; CK(hipGetDeviceCount(&n));
    for (int d = 0; d < n; d++) {
        hipDeviceProp_t p; CK(hipGetDeviceProperties(&p, d));
        if (std::string(p.gcnArchName).rfind(arch, 0) == 0) return d;
    }
    printf("no %s device\n", arch); exit(1);
}

static double now_us() {
    return std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

template <typename T> static T* pinned(size_t n) {
    T* p;
    CK(hipHostMalloc((void**)&p, n * sizeof(T),
                     hipHostMallocCoherent | hipHostMallocMapped | hipHostMallocPortable));
    memset((void*)p, 0, n * sizeof(T));
    return p;
}

static void report(const char* mode, int r, double dt, int bad_a, int bad_b) {
    if (bad_a < 0 || bad_b < 0)
        printf("%s: ABORTED (a=%d b=%d), a flag was never seen\n", mode, bad_a, bad_b);
    else
        printf("%s: %d rounds, %.2f us/round trip, payload errors a=%d b=%d\n",
               mode, r, dt / r, bad_a, bad_b);
}

// Mode A or C; returns false if it aborted.
static bool run_flags(char mode, int A, int B, Shared* s, int rounds, int words) {
    int* bad_a = pinned<int>(1);
    int* bad_b = pinned<int>(1);
    hipStream_t sa, sb;
    CK(hipSetDevice(B)); CK(hipStreamCreateWithFlags(&sb, hipStreamNonBlocking));
    hipGraphExec_t once = nullptr;
    if (mode == 'C') {
        hipGraph_t g;
        CK(hipStreamBeginCapture(sb, hipStreamCaptureModeGlobal));
        hipLaunchKernelGGL(side_b_once, 1, THREADS, 0, sb, s, words, bad_b);
        CK(hipStreamEndCapture(sb, &g));
        CK(hipGraphInstantiate(&once, g, nullptr, nullptr, 0));
        CK(hipGraphDestroy(g));
    }
    CK(hipSetDevice(A)); CK(hipStreamCreateWithFlags(&sa, hipStreamNonBlocking));
    bool ok = true;
    for (int pass = 0; pass < 2 && ok; pass++) {  // pass 0 = warm-up
        int r = pass ? rounds : 200;
        memset((void*)s, 0, sizeof(Shared)); *bad_a = *bad_b = 0;
        double t0 = now_us();
        std::thread responder;
        if (mode == 'A') {
            CK(hipSetDevice(B));
            hipLaunchKernelGGL(side_b, 1, THREADS, 0, sb, s, r, words, bad_b);
        } else {
            responder = std::thread([&] {
                CK(hipSetDevice(B));
                for (int i = 1; i <= r; i++) {
                    long long n = 0;
                    while (host_ld(&s->flag_a) != i && !host_ld(&s->abort_) && ++n < MAX_HOST_POLLS) {}
                    if (host_ld(&s->abort_) || n >= MAX_HOST_POLLS) {
                        host_st(&s->abort_, 1); *bad_b = -i; return;
                    }
                    host_st(&s->cur, i);
                    CK(hipGraphLaunch(once, sb));
                    // The next request cannot arrive before this answer; waiting for the
                    // launch keeps `cur` from moving under a queued kernel.
                    CK(hipStreamSynchronize(sb));
                }
            });
        }
        CK(hipSetDevice(A));
        hipLaunchKernelGGL(side_a, 1, THREADS, 0, sa, s, r, words, bad_a);
        CK(hipStreamSynchronize(sa));
        if (responder.joinable()) responder.join();
        CK(hipSetDevice(B)); CK(hipStreamSynchronize(sb));
        double dt = now_us() - t0;
        if (pass) report(mode == 'A' ? "A) GPU<->GPU flags" : "C) CPU-driven responder", r, dt, *bad_a, *bad_b);
        ok = *bad_a >= 0 && *bad_b >= 0;
    }
    if (once) CK(hipGraphExecDestroy(once));
    CK(hipStreamDestroy(sa)); CK(hipStreamDestroy(sb));
    return ok;
}

static void run_host(int A, int B, int rounds, int bytes) {
    unsigned *da, *db, *ha;
    int words = bytes / 4;
    size_t alloc = bytes > 0 ? bytes : 4;
    CK(hipSetDevice(A)); CK(hipMalloc(&da, alloc));
    CK(hipSetDevice(B)); CK(hipMalloc(&db, alloc));
    CK(hipHostMalloc((void**)&ha, alloc, hipHostMallocPortable));
    for (int pass = 0; pass < 2; pass++) {
        int r = pass ? rounds : 100;
        double t0 = now_us();
        for (int i = 0; i < r; i++) {
            CK(hipSetDevice(A));
            hipLaunchKernelGGL(touch, 1, 256, 0, 0, da, words, 1u);
            CK(hipMemcpy(ha, da, bytes, hipMemcpyDeviceToHost));
            CK(hipSetDevice(B));
            CK(hipMemcpy(db, ha, bytes, hipMemcpyHostToDevice));
            hipLaunchKernelGGL(touch, 1, 256, 0, 0, db, words, 1u);
            CK(hipMemcpy(ha, db, bytes, hipMemcpyDeviceToHost));
            CK(hipSetDevice(A));
            CK(hipMemcpy(da, ha, bytes, hipMemcpyHostToDevice));
        }
        if (pass) report("B) host-mediated", r, now_us() - t0, 0, 0);
    }
}

int main(int argc, char** argv) {
    int rounds = argc > 1 ? atoi(argv[1]) : 20000;
    int bytes = argc > 2 ? atoi(argv[2]) : 5120;
    std::string modes = argc > 3 ? argv[3] : "ABC";
    if (bytes < 0 || bytes > MAX_WORDS * 4 || bytes % 4) {
        printf("payload must be a multiple of 4 in 0..%d\n", MAX_WORDS * 4);
        return 2;
    }
    int A = find("gfx1201"), B = find("gfx1151");
    printf("A=gfx1201 dev%d  B=gfx1151 dev%d  payload=%d B each way\n", A, B, bytes);
    Shared* s = pinned<Shared>(1);
    for (char m : modes) {
        if (m == 'A' || m == 'C') {
            if (!run_flags(m, A, B, s, rounds, bytes / 4)) return 1;
        } else if (m == 'B') {
            run_host(A, B, rounds / 10, bytes);
        }
    }
    return 0;
}
