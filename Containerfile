# hipfire: ROCm SDK -> builder -> full / gate-runner / runtime (default).
# ROCm 10.1 on Ubuntu 26.04 matches the release kernel-pack toolchain.
# runtime ships indexed packs and runtime libraries only; full also permits JIT.
# The entrypoint starts as root only to grant device groups, then drops to uid 1000.
# Models are downloaded at run time and are never baked into either image.
#   docker build -f Containerfile --target runtime -t hipfire .
#   docker build -f Containerfile --target full -t hipfire:full .
#   podman build -f Containerfile --target runtime -t hipfire .
#   podman build -f Containerfile --target full -t hipfire:full .

ARG ROCM_PKG_VERSION=10.1.0-3
ARG HIPFIRE_GPU_ARCHES="gfx1100 gfx1151 gfx1201"
ARG HIPFIRE_VERSION=0.4.5
ARG HIPFIRE_REVISION=unknown

FROM docker.io/library/ubuntu:26.04 AS rocm-sdk
ARG ROCM_PKG_VERSION
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl gpg \
    && install -d -m 0755 /etc/apt/keyrings \
    && curl -fsSL https://stable.repo.amd.com/rocm/gpg/packages.gpg -o /etc/apt/keyrings/amdrocm.asc \
    && fingerprint="$(gpg --batch --with-colons --show-keys /etc/apt/keyrings/amdrocm.asc | awk -F: '$1 == "fpr" { print $10; exit }')" \
    && test "$fingerprint" = D0F004A0025A1145C7807FCD0701EAC4D5E02107 \
    && chmod 0644 /etc/apt/keyrings/amdrocm.asc \
    && printf '%s\n' 'deb [arch=amd64 signed-by=/etc/apt/keyrings/amdrocm.asc] https://stable.repo.amd.com/rocm/core/packages/ubuntu2604 stable main' > /etc/apt/sources.list.d/amdrocm.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        amdrocm-runtime10.1="${ROCM_PKG_VERSION}" \
        amdrocm-runtime-dev10.1="${ROCM_PKG_VERSION}" \
    && apt-get purge -y --auto-remove gpg \
    && rm -rf /var/lib/apt/lists/* /root/.gnupg
# The packages install only core-10.1; expose the host's active-root layout.
RUN for name in bin lib include llvm amdgcn share libexec .info; do \
        ln -s "core-10.1/$name" "/opt/rocm/$name"; \
    done
ENV ROCM_PATH=/opt/rocm \
    PATH="/opt/rocm/bin:${PATH}"

FROM rocm-sdk AS builder
ARG HIPFIRE_GPU_ARCHES
ARG HIPFIRE_REVISION
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git pkg-config python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}" \
    CARGO_TARGET_DIR=/hipfire/target
WORKDIR /hipfire
COPY . .
# Plain RUN (no cache mounts) so the classic docker builder, BuildKit and
# buildah all accept this file. The absolute target dir lets compile-kernels.sh
# find its debug registry tools.
RUN export ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 \
        HIPFIRE_BUILD_COMMIT_OVERRIDE="${HIPFIRE_REVISION}" \
    && cargo build --release --locked -p hipfire-daemon -p hipfire-cli \
    && bash scripts/compile-kernels.sh ${HIPFIRE_GPU_ARCHES} \
    && mkdir -p /out/kernels \
    && cp target/release/daemon /out/daemon \
    && cp target/release/hipfire /out/hipfire \
    && cp -a kernels/compiled /out/kernels/compiled

FROM rocm-sdk AS full
ARG ROCM_PKG_VERSION
ARG HIPFIRE_GPU_ARCHES
ARG HIPFIRE_VERSION
ARG HIPFIRE_REVISION
RUN userdel -r ubuntu \
    && groupadd --gid 1000 hipfire \
    && useradd --uid 1000 --gid hipfire --create-home --shell /bin/sh hipfire \
    && groupadd -f video && groupadd -f render \
    && usermod -aG video,render hipfire \
    && command -v setpriv \
    && install -d -o hipfire -g hipfire /home/hipfire/.hipfire \
        /home/hipfire/.hipfire/models /var/cache/hipfire
COPY --from=builder /out/daemon /opt/hipfire/bin/daemon
COPY --from=builder /out/kernels/compiled /opt/hipfire/bin/kernels/compiled
COPY --from=builder /out/hipfire /usr/local/bin/hipfire
COPY docker/hipfire-entrypoint.sh /usr/local/bin/hipfire-entrypoint
COPY docker/hipfire-halo-tune.sh /usr/local/bin/hipfire-halo-tune
RUN chmod 0755 /usr/local/bin/hipfire-entrypoint /usr/local/bin/hipfire-halo-tune
ENV HOME=/home/hipfire \
    HIPFIRE_DAEMON_BIN=/opt/hipfire/bin/daemon \
    HIPFIRE_HOME=/home/hipfire/.hipfire \
    HIPFIRE_KERNEL_CACHE=/var/cache/hipfire \
    HIPFIRE_HOST=0.0.0.0 \
    HIPFIRE_PORT=11435 \
    ROCM_PATH=/opt/rocm
LABEL org.opencontainers.image.title="hipfire" \
    org.opencontainers.image.description="AMD GPU inference with the ROCm device compiler" \
    org.opencontainers.image.source="https://github.com/warpfront/hipfire" \
    org.opencontainers.image.url="https://github.com/warpfront/hipfire" \
    org.opencontainers.image.documentation="https://github.com/warpfront/hipfire/blob/master/docs/CONTAINER.md" \
    org.opencontainers.image.licenses="Apache-2.0" \
    org.opencontainers.image.version="${HIPFIRE_VERSION}" \
    org.opencontainers.image.revision="${HIPFIRE_REVISION}" \
    org.opencontainers.image.vendor="hipfire" \
    org.opencontainers.image.base.name="docker.io/library/ubuntu:26.04" \
    dev.hipfire.gpu-arches="${HIPFIRE_GPU_ARCHES}" \
    dev.hipfire.rocm="${ROCM_PKG_VERSION}" \
    dev.hipfire.variant="full"
EXPOSE 11435
# No USER: the entrypoint grants host device GIDs before dropping privileges.
ENTRYPOINT ["/usr/local/bin/hipfire-entrypoint"]
CMD ["serve"]

FROM builder AS gate-runner
# Python-backed detector arms; mounted models retain the established root path.
RUN pip install --no-cache-dir --break-system-packages pytest numpy
COPY --from=builder /out/hipfire /usr/local/bin/hipfire
ENV HIPFIRE_DAEMON_BIN=/out/daemon \
    HIPFIRE_HOME=/root/.hipfire \
    HIPFIRE_KERNEL_CACHE=/var/cache/hipfire \
    MODELS_DIR=/root/.hipfire/models
WORKDIR /hipfire
# No default gate: there is no universal correctness battery. Select a route
# from docs/VALIDATION.md and pass the command explicitly.
ENTRYPOINT []
CMD ["/bin/bash", "-c", "echo '[hipfire-gate] no gate command given.' >&2; echo 'There is no universal gate. Select a validation route from docs/VALIDATION.md and pass it explicitly, e.g.:' >&2; echo '  python3 scripts/serve_harness.py battery --model <model>' >&2; echo '  python3 scripts/redline_daemon_harness.py   # kernel/dispatch/graph/Redline changes' >&2; exit 2"]

# The HIP runtime's shared-library closure (ldd of libamdhip64 and
# libhsa-runtime64, ROCm-owned entries only), staged with its SONAME links so
# one directory COPY keeps them as links: a per-file COPY dereferences each
# link into another full copy of the library.
FROM rocm-sdk AS rocm-runtime-libs
RUN set -eu; \
    lib=/opt/rocm/core-10.1/lib; \
    mkdir -p /stage; \
    for root in "$lib/libamdhip64.so.7" "$lib/libhsa-runtime64.so.1"; do \
        echo "$root"; ldd "$root" | awk '$3 ~ "^/opt/rocm/" { print $3 }'; \
    done | sort -u > /tmp/closure; \
    while read -r path; do \
        real=$(readlink -f "$path"); \
        link="$(dirname "$real")/$(basename "$path")"; \
        cp -a --parents "$real" /stage; \
        [ "$link" = "$real" ] || cp -a --parents "$link" /stage; \
    done < /tmp/closure; \
    cp -a --parents "$lib/libamdhip64.so" "$lib/libhsa-runtime64.so" \
        /opt/rocm/core-10.1/.info/version /stage; \
    find /stage | sort; du -sh /stage

# Last stage is the slim default: no compiler, headers, bitcode, or source tree.
FROM docker.io/library/ubuntu:26.04 AS runtime
ARG ROCM_PKG_VERSION
ARG HIPFIRE_GPU_ARCHES
ARG HIPFIRE_VERSION
ARG HIPFIRE_REVISION
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=rocm-runtime-libs /stage/ /
# Runtime-only roots are accepted via the resolver's first-existing-root path.
RUN ln -s core-10.1/lib /opt/rocm/lib \
    && ln -s core-10.1/.info /opt/rocm/.info \
    && printf '%s\n' /opt/rocm/core-10.1/lib \
        /opt/rocm/core-10.1/lib/rocm_sysdeps/lib \
        /opt/rocm/core-10.1/lib/llvm/lib > /etc/ld.so.conf.d/hipfire-rocm.conf \
    && ldconfig \
    && userdel -r ubuntu \
    && groupadd --gid 1000 hipfire \
    && useradd --uid 1000 --gid hipfire --create-home --shell /bin/sh hipfire \
    && groupadd -f video && groupadd -f render \
    && usermod -aG video,render hipfire \
    && command -v setpriv \
    && install -d -o hipfire -g hipfire /home/hipfire/.hipfire \
        /home/hipfire/.hipfire/models /var/cache/hipfire
COPY --from=builder /out/daemon /opt/hipfire/bin/daemon
COPY --from=builder /out/kernels/compiled /opt/hipfire/bin/kernels/compiled
COPY --from=builder /out/hipfire /usr/local/bin/hipfire
COPY docker/hipfire-entrypoint.sh /usr/local/bin/hipfire-entrypoint
COPY docker/hipfire-halo-tune.sh /usr/local/bin/hipfire-halo-tune
RUN chmod 0755 /usr/local/bin/hipfire-entrypoint /usr/local/bin/hipfire-halo-tune
ENV HOME=/home/hipfire \
    HIPFIRE_DAEMON_BIN=/opt/hipfire/bin/daemon \
    HIPFIRE_HOME=/home/hipfire/.hipfire \
    HIPFIRE_KERNEL_CACHE=/var/cache/hipfire \
    HIPFIRE_HOST=0.0.0.0 \
    HIPFIRE_PORT=11435 \
    ROCM_PATH=/opt/rocm \
    PATH="/opt/rocm/bin:${PATH}"
LABEL org.opencontainers.image.title="hipfire" \
    org.opencontainers.image.description="AMD GPU inference with indexed kernel packs and a slim ROCm runtime" \
    org.opencontainers.image.source="https://github.com/warpfront/hipfire" \
    org.opencontainers.image.url="https://github.com/warpfront/hipfire" \
    org.opencontainers.image.documentation="https://github.com/warpfront/hipfire/blob/master/docs/CONTAINER.md" \
    org.opencontainers.image.licenses="Apache-2.0" \
    org.opencontainers.image.version="${HIPFIRE_VERSION}" \
    org.opencontainers.image.revision="${HIPFIRE_REVISION}" \
    org.opencontainers.image.vendor="hipfire" \
    org.opencontainers.image.base.name="docker.io/library/ubuntu:26.04" \
    dev.hipfire.gpu-arches="${HIPFIRE_GPU_ARCHES}" \
    dev.hipfire.rocm="${ROCM_PKG_VERSION}" \
    dev.hipfire.variant="runtime"
EXPOSE 11435
# No USER: the entrypoint grants host device GIDs before dropping privileges.
ENTRYPOINT ["/usr/local/bin/hipfire-entrypoint"]
CMD ["serve"]
