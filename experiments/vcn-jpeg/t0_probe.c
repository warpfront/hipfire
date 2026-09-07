// SPDX-License-Identifier: MIT OR Apache-2.0
// T0 kill-gate probe for experiment/vcn-jpeg.
// Throwaway: dlopens libva + libva-drm + libamdhip64, no link-time deps.
//
// Gate (a): vaInitialize + vaCreateConfig(VAProfileJPEGBaseline=12, VAEntrypointVLD=1)
// Gate (b): vaCreateSurfaces(YUV420) + vaExportSurfaceHandle(DRM_PRIME_2) gives a
//           valid dmabuf fd, AND hipImportExternalMemory(OpaqueFd) accepts it.
//
// Enum/struct values transcribed from libva master (2.x ABI, stable):
//   va/va.h, va/va_dec_jpeg.h, va/va_drmcommon.h, va/va_drm.h
// HIP structs from /opt/rocm/include/hip/hip_runtime_api.h (ROCm 7.x, local).
// libva on host is 2.2300.0 (/usr/lib/x86_64-linux-gnu/libva.so.2).
//
// Build: gcc -O2 -o t0_probe t0_probe.c -ldl
// Run (device 0): source scripts/gpu-lock.sh && gpu_acquire vcn-jpeg && ./t0_probe; gpu_release
#include <dlfcn.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

// ---- libva core (va/va.h) ----
typedef void *VADisplay;
typedef int VAStatus;
typedef unsigned int VAGenericID;
typedef VAGenericID VAConfigID, VAContextID, VASurfaceID, VABufferID;
#define VA_STATUS_SUCCESS 0
#define VAProfileJPEGBaseline 12
#define VAEntrypointVLD 1
#define VA_RT_FORMAT_YUV420 0x1

// ---- export (va/va.h + va/va_drmcommon.h) ----
#define VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2 0x40000000u
#define VA_EXPORT_SURFACE_READ_ONLY 0x0001u
#define VA_EXPORT_SURFACE_SEPARATE_LAYERS 0x0004u
#define VA_FOURCC_NV12 0x3231564Eu

typedef struct {
    uint32_t fourcc, width, height, num_objects;
    struct {
        int fd;
        uint32_t size;
        uint64_t drm_format_modifier;
    } objects[4];
    uint32_t num_layers;
    struct {
        uint32_t drm_format, num_planes;
        uint32_t object_index[4], offset[4], pitch[4];
    } layers[4];
} VADRMPRIMESurfaceDescriptor;

// ---- HIP external memory (hip_runtime_api.h, ROCm local) ----
typedef enum {
    hipExternalMemoryHandleTypeOpaqueFd = 1,
} HipExtMemHandleType;
typedef void *HipExternalMemory;
typedef struct {
    HipExtMemHandleType type;
    union {
        int fd;
        struct {
            void *handle;
            const void *name;
        } win32;
        const void *nvSciBufObject;
    } handle;
    unsigned long long size;
    unsigned int flags;
    unsigned int reserved[16];
} HipExtMemHandleDesc;
typedef struct {
    unsigned long long offset, size;
    unsigned int flags;
    unsigned int reserved[16];
} HipExtMemBufferDesc;

int main(void) {
    int fails = 0;
    void *va = dlopen("libva.so.2", RTLD_NOW);
    void *vadrm = dlopen("libva-drm.so.2", RTLD_NOW);
    void *hip = dlopen("libamdhip64.so.7", RTLD_NOW | RTLD_GLOBAL);
    if (!va || !vadrm) {
        printf("T0 dlopen libva FAIL va=%p vadrm=%p\n", va, vadrm);
        return 2;
    }
    printf("T0 dlopen libva.so.2 + libva-drm.so.2 OK, libamdhip64=%p\n", hip);

    VADisplay (*getDisplayDRM)(int) = dlsym(vadrm, "vaGetDisplayDRM");
    VAStatus (*vaInit)(VADisplay, int *, int *) = dlsym(va, "vaInitialize");
    VAStatus (*vaTerm)(VADisplay) = dlsym(va, "vaTerminate");
    const char *(*vaErr)(VAStatus) = dlsym(va, "vaErrorStr");
    int (*maxProfiles)(VADisplay) = dlsym(va, "vaMaxNumProfiles");
    VAStatus (*queryProfiles)(VADisplay, int *, int *) = dlsym(va, "vaQueryConfigProfiles");
    VAStatus (*createConfig)(VADisplay, int, int, void *, int, VAConfigID *) = dlsym(va, "vaCreateConfig");
    VAStatus (*destroyConfig)(VADisplay, VAConfigID) = dlsym(va, "vaDestroyConfig");
    VAStatus (*createSurfaces)(VADisplay, unsigned, unsigned, unsigned, VASurfaceID *, unsigned, void *, unsigned) =
        dlsym(va, "vaCreateSurfaces");
    VAStatus (*destroySurfaces)(VADisplay, VASurfaceID *, int) = dlsym(va, "vaDestroySurfaces");
    VAStatus (*createContext)(VADisplay, VAConfigID, int, int, int, VASurfaceID *, int, VAContextID *) =
        dlsym(va, "vaCreateContext");
    VAStatus (*destroyContext)(VADisplay, VAContextID) = dlsym(va, "vaDestroyContext");
    VAStatus (*exportSurf)(VADisplay, VASurfaceID, uint32_t, uint32_t, void *) = dlsym(va, "vaExportSurfaceHandle");
    VAStatus (*syncSurf)(VADisplay, VASurfaceID) = dlsym(va, "vaSyncSurface");
    if (!getDisplayDRM || !vaInit || !createConfig || !exportSurf || !createSurfaces || !createContext) {
        printf("T0 dlsym FAIL\n");
        return 2;
    }

    const char *nodes[] = {"/dev/dri/renderD128", "/dev/dri/renderD129", "/dev/dri/renderD130",
                           "/dev/dri/renderD131"};
    for (int n = 0; n < 4; n++) {
        int fd = open(nodes[n], O_RDWR);
        if (fd < 0) {
            printf("T0 %s: no open (skip)\n", nodes[n]);
            continue;
        }
        VADisplay dpy = getDisplayDRM(fd);
        if (!dpy) {
            printf("T0 %s: vaGetDisplayDRM NULL\n", nodes[n]);
            close(fd);
            continue;
        }
        int major = 0, minor = 0;
        VAStatus st = vaInit(dpy, &major, &minor);
        printf("T0 %s: vaInitialize -> 0x%x (%s) ver=%d.%d\n", nodes[n], st,
               vaErr ? vaErr(st) : "?", major, minor);
        if (st != VA_STATUS_SUCCESS) {
            close(fd);
            fails++;
            continue;
        }
        // vendor string
        const char *(*queryVendor)(VADisplay) = dlsym(va, "vaQueryVendorString");
        if (queryVendor) printf("T0 %s: vendor='%s'\n", nodes[n], queryVendor(dpy));

        // profiles: is JPEGBaseline(12) supported?
        int np = maxProfiles ? maxProfiles(dpy) : 64;
        int plist[128];
        int nprof = np < 128 ? np : 128;
        st = queryProfiles(dpy, plist, &nprof);
        int has_jpeg = 0;
        printf("T0 %s: profiles(%d):", nodes[n], nprof);
        for (int i = 0; i < nprof; i++) {
            printf(" %d", plist[i]);
            if (plist[i] == VAProfileJPEGBaseline) has_jpeg = 1;
        }
        printf(" jpeg_baseline=%s\n", has_jpeg ? "YES" : "no");

        // Gate (a): create JPEG VLD config
        VAConfigID cfg = 0;
        st = createConfig(dpy, VAProfileJPEGBaseline, VAEntrypointVLD, NULL, 0, &cfg);
        printf("T0 %s: GATE-A vaCreateConfig(JPEG=12,VLD=1) -> 0x%x (%s) cfg=%u\n", nodes[n], st,
               vaErr ? vaErr(st) : "?", cfg);
        if (st != VA_STATUS_SUCCESS) {
            vaTerm(dpy);
            close(fd);
            fails++;
            continue;
        }

        // surface + context + export (gate b, part 1)
        VASurfaceID surf = 0;
        st = createSurfaces(dpy, VA_RT_FORMAT_YUV420, 946, 1024, &surf, 1, NULL, 0);
        printf("T0 %s: vaCreateSurfaces(YUV420,946x1024) -> 0x%x surf=%u\n", nodes[n], st, surf);
        VAContextID ctx = 0;
        if (st == VA_STATUS_SUCCESS)
            st = createContext(dpy, cfg, 946, 1024, 0, &surf, 1, &ctx);
        printf("T0 %s: vaCreateContext -> 0x%x ctx=%u\n", nodes[n], st, ctx);

        VADRMPRIMESurfaceDescriptor desc;
        memset(&desc, 0, sizeof desc);
        st = exportSurf(dpy, surf, VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2, VA_EXPORT_SURFACE_READ_ONLY,
                        &desc);
        printf("T0 %s: GATE-B1 vaExportSurfaceHandle(PRIME_2) -> 0x%x (%s)\n", nodes[n], st,
               vaErr ? vaErr(st) : "?");
        if (st == VA_STATUS_SUCCESS) {
            printf("T0 %s: export fourcc=0x%08x %ux%u objects=%u layers=%u\n", nodes[n], desc.fourcc,
                   desc.width, desc.height, desc.num_objects, desc.num_layers);
            for (uint32_t i = 0; i < desc.num_objects && i < 4; i++)
                printf("T0 %s:   obj%u fd=%d size=%u mod=0x%llx\n", nodes[n], i, desc.objects[i].fd,
                       desc.objects[i].size, (unsigned long long)desc.objects[i].drm_format_modifier);
            for (uint32_t i = 0; i < desc.num_layers && i < 4; i++)
                printf("T0 %s:   layer%u fmt=0x%08x planes=%u pitch0=%u off0=%u\n", nodes[n], i,
                       desc.layers[i].drm_format, desc.layers[i].num_planes, desc.layers[i].pitch[0],
                       desc.layers[i].offset[0]);

            // Gate (b) part 2: HIP import of the dmabuf as OpaqueFd
            if (hip && desc.num_objects > 0 && desc.objects[0].fd >= 0) {
                int (*hipSetDevice)(int) = dlsym(hip, "hipSetDevice");
                int (*hipImport)(HipExternalMemory *, const HipExtMemHandleDesc *) =
                    dlsym(hip, "hipImportExternalMemory");
                int (*hipGetMapped)(void **, HipExternalMemory, const HipExtMemBufferDesc *) =
                    dlsym(hip, "hipExternalMemoryGetMappedBuffer");
                int (*hipDestroy)(HipExternalMemory) = dlsym(hip, "hipDestroyExternalMemory");
                if (hipSetDevice && hipImport && hipGetMapped) {
                    int hr = hipSetDevice(0);
                    printf("T0 %s: hipSetDevice(0) -> %d\n", nodes[n], hr);
                    HipExtMemHandleDesc hd;
                    memset(&hd, 0, sizeof hd);
                    hd.type = hipExternalMemoryHandleTypeOpaqueFd;
                    hd.handle.fd = desc.objects[0].fd;
                    hd.size = desc.objects[0].size;
                    HipExternalMemory em = NULL;
                    hr = hipImport(&em, &hd);
                    printf("T0 %s: GATE-B2 hipImportExternalMemory(opaqueFd,size=%llu) -> %d em=%p\n",
                           nodes[n], (unsigned long long)hd.size, hr, em);
                    if (hr == 0 && em) {
                        HipExtMemBufferDesc bd;
                        memset(&bd, 0, sizeof bd);
                        bd.size = hd.size;
                        void *ptr = NULL;
                        hr = hipGetMapped(&ptr, em, &bd);
                        printf("T0 %s: hipExternalMemoryGetMappedBuffer -> %d ptr=%p\n", nodes[n],
                               hr, ptr);
                        if (hipDestroy) hipDestroy(em);
                    } else {
                        fails++;
                    }
                } else {
                    printf("T0 %s: HIP import symbols missing (skip B2)\n", nodes[n]);
                }
            }
            for (uint32_t i = 0; i < desc.num_objects && i < 4; i++)
                if (desc.objects[i].fd >= 0) close(desc.objects[i].fd);
        } else {
            fails++;
        }
        if (ctx) destroyContext(dpy, ctx);
        destroySurfaces(dpy, &surf, 1);
        destroyConfig(dpy, cfg);
        vaTerm(dpy);
        close(fd);
    }
    printf("T0 verdict: %s\n", fails ? "KILL (see failures above)" : "GO on gates A+B1+B2");
    return fails ? 1 : 0;
}
