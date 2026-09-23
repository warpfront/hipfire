FROM rocm/dev-ubuntu-24.04:7.0-complete

# hipfire kernel build container — ROCm 7.0 for gfx12 RDNA4 optimized kernels
# Usage:
#   docker build -f docker/rocm7-builder.Dockerfile -t hipfire-builder .
#   docker run --rm -v $(pwd):/hipfire hipfire-builder bash /hipfire/scripts/compile-kernels.sh gfx1200 gfx1201

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /hipfire
