#include <hip/hip_runtime_api.h>
#include <cstdlib>
#include <iostream>

static void check(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        std::cerr << what << ": " << hipGetErrorString(e) << "\n";
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: occupancy_probe HSACO SYMBOL BLOCK LDS\n";
        return 2;
    }
    check(hipSetDevice(0), "hipSetDevice");
    hipModule_t module = nullptr;
    hipFunction_t function = nullptr;
    check(hipModuleLoad(&module, argv[1]), "hipModuleLoad");
    check(hipModuleGetFunction(&function, module, argv[2]), "hipModuleGetFunction");
    const int block = std::atoi(argv[3]);
    const size_t lds = std::strtoull(argv[4], nullptr, 10);
    int blocks = 0;
    check(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
              &blocks, function, block, lds),
          "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor");
    std::cout << "hsaco=" << argv[1] << "\n"
              << "symbol=" << argv[2] << "\n"
              << "block=" << block << " lds=" << lds << "\n"
              << "max_blocks_per_cu=" << blocks << "\n"
              << "resident_waves_per_cu=" << blocks * block / 32 << "\n"
              << "average_waves_per_simd=" << blocks * block / 128.0 << "\n";
    check(hipModuleUnload(module), "hipModuleUnload");
    return 0;
}
