/* SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 Kaden Schutt
 * hipfire — see LICENSE and NOTICE in the project root.
 *
 * Offline developer Vulkan host for the PM-ACO producer (tools/aco-export). NOT linked into any
 * shipping crate and has no Mesa dependency: it speaks plain Vulkan 1.3 through the system loader
 * (or through VK_DRIVER_FILES pointing at the patched pinned RADV ICD).
 *
 * Modes
 *   compile   Create ONE compute pipeline from a SPIR-V file. This triggers the real RADV
 *             SPIR-V -> NIR -> ACO compile; the patched RADV writes the raw capture into
 *             $RADV_ACO_EXPORT_DIR. Also writes the VK_KHR_pipeline_executable_properties statistics
 *             in RADV's "*** SHADER STATS ***" text format and the device/driver identity JSON.
 *   dispatch  compile + execute: one VkBuffer per --buffer (with BDA), bytes uploaded from a file,
 *             inline push constants assembled from --push, dispatch --groups, then every buffer's
 *             full post-state is written to --out-dir/<name>.post.bin.
 *
 * Usage
 *   vk_aco_host compile  --spv F --push-size N --local X,Y,Z --subgroup 32 --stats-out F --identity-out F
 *                        (--pci-bdf DDDD:BB:DD.F | --offline-shim) [--entry main]
 *   vk_aco_host dispatch (same) --groups X,Y,Z --out-dir D
 *                        --buffer NAME=FILE:PTR_OFFSET ...        (repeatable; FILE = framed pre bytes)
 *                        --push 'u64:buf:A,u64:buf:x,u64:buf:y,i32:5120,i32:6144'
 * Push items: u64:buf:NAME (device address of buffer NAME + its PTR_OFFSET), u64:VALUE, i32:VALUE,
 * u32:VALUE, f32:VALUE (little-endian, tightly packed in the listed order, total must equal --push-size).
 */
#define _GNU_SOURCE
#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define CHECK(x)                                                                                                       \
   do {                                                                                                                \
      VkResult r_ = (x);                                                                                               \
      if (r_ != VK_SUCCESS) {                                                                                          \
         fprintf(stderr, "vk_aco_host: %s failed: %d (%s:%d)\n", #x, (int)r_, __FILE__, __LINE__);                    \
         exit(2);                                                                                                      \
      }                                                                                                                \
   } while (0)

#define DIE(...)                                                                                                       \
   do {                                                                                                                \
      fprintf(stderr, "vk_aco_host: " __VA_ARGS__);                                                                    \
      fputc('\n', stderr);                                                                                             \
      exit(2);                                                                                                         \
   } while (0)

struct buf {
   char name[32];
   char path[4096];
   uint64_t ptr_offset;
   uint64_t size;
   VkBuffer buffer;
   VkDeviceMemory memory;
   VkDeviceAddress address;
   void *map;
};

static void *
read_file(const char *path, size_t *size)
{
   FILE *f = fopen(path, "rb");
   if (!f)
      DIE("cannot open %s", path);
   fseek(f, 0, SEEK_END);
   long n = ftell(f);
   fseek(f, 0, SEEK_SET);
   void *data = malloc(n ? (size_t)n : 1);
   if (n && fread(data, 1, (size_t)n, f) != (size_t)n)
      DIE("short read %s", path);
   fclose(f);
   *size = (size_t)n;
   return data;
}

static void
write_file(const char *path, const void *data, size_t size)
{
   FILE *f = fopen(path, "wb");
   if (!f)
      DIE("cannot write %s", path);
   if (size && fwrite(data, 1, size, f) != size)
      DIE("short write %s", path);
   fclose(f);
}

static void
parse3(const char *s, uint32_t out[3])
{
   if (sscanf(s, "%u,%u,%u", &out[0], &out[1], &out[2]) != 3)
      DIE("expected X,Y,Z: %s", s);
}

static uint32_t
find_memory_type(const VkPhysicalDeviceMemoryProperties *mp, uint32_t bits, VkMemoryPropertyFlags want)
{
   for (uint32_t i = 0; i < mp->memoryTypeCount; i++)
      if ((bits & (1u << i)) && (mp->memoryTypes[i].propertyFlags & want) == want)
         return i;
   DIE("no host-visible coherent memory type");
   return 0;
}

static void
json_str(FILE *f, const char *s)
{
   fputc('"', f);
   for (; *s; s++) {
      unsigned char c = (unsigned char)*s;
      if (c == '"' || c == '\\')
         fprintf(f, "\\%c", c);
      else if (c < 0x20)
         fprintf(f, "\\u%04x", c);
      else
         fputc(c, f);
   }
   fputc('"', f);
}

static const char *
next_arg(int argc, char **argv, int *i, const char *name)
{
   if (*i + 1 >= argc)
      DIE("missing value for %s", name);
   return argv[++*i];
}

int
main(int argc, char **argv)
{
   if (argc < 2)
      DIE("mode required: compile|dispatch");
   const bool dispatch = !strcmp(argv[1], "dispatch");
   if (!dispatch && strcmp(argv[1], "compile"))
      DIE("unknown mode %s", argv[1]);

   const char *spv_path = NULL, *stats_out = NULL, *identity_out = NULL, *out_dir = NULL, *push_spec = NULL;
   const char *entry = "main";
   uint32_t push_size = 0, subgroup = 0, local[3] = {0, 0, 0}, groups[3] = {0, 0, 0};
   const char *want_bdf = NULL;
   bool offline_shim = false;
   struct buf bufs[16];
   unsigned nbuf = 0;

   for (int i = 2; i < argc; i++) {
      const char *a = argv[i];
#define NEXT() next_arg(argc, argv, &i, a)
      if (!strcmp(a, "--spv")) spv_path = NEXT();
      else if (!strcmp(a, "--push-size")) push_size = (uint32_t)strtoul(NEXT(), NULL, 10);
      else if (!strcmp(a, "--local")) parse3(NEXT(), local);
      else if (!strcmp(a, "--groups")) parse3(NEXT(), groups);
      else if (!strcmp(a, "--subgroup")) subgroup = (uint32_t)strtoul(NEXT(), NULL, 10);
      else if (!strcmp(a, "--stats-out")) stats_out = NEXT();
      else if (!strcmp(a, "--identity-out")) identity_out = NEXT();
      else if (!strcmp(a, "--out-dir")) out_dir = NEXT();
      else if (!strcmp(a, "--entry")) entry = NEXT();
      else if (!strcmp(a, "--pci-bdf")) want_bdf = NEXT();
      else if (!strcmp(a, "--offline-shim")) offline_shim = true;
      else if (!strcmp(a, "--push")) push_spec = NEXT();
      else if (!strcmp(a, "--buffer")) {
         if (nbuf == 16) DIE("too many buffers");
         const char *v = NEXT();
         struct buf *b = &bufs[nbuf++];
         memset(b, 0, sizeof(*b));
         char *eq = strchr(v, '=');
         char *colon = eq ? strrchr(eq, ':') : NULL;
         if (!eq || !colon) DIE("bad --buffer %s", v);
         snprintf(b->name, sizeof(b->name), "%.*s", (int)(eq - v), v);
         snprintf(b->path, sizeof(b->path), "%.*s", (int)(colon - eq - 1), eq + 1);
         b->ptr_offset = strtoull(colon + 1, NULL, 10);
      } else DIE("unknown argument %s", a);
   }
   if (!spv_path || !push_size || !local[0] || !subgroup || !stats_out || !identity_out)
      DIE("--spv --push-size --local --subgroup --stats-out --identity-out are required");
   if (dispatch && (!groups[0] || !out_dir || !push_spec))
      DIE("dispatch needs --groups --out-dir --push");
   if (dispatch && !want_bdf)
      DIE("dispatch (hardware reference) requires --pci-bdf DDDD:BB:DD.F; no first-device fallback");
   if (!want_bdf && !offline_shim)
      DIE("compile requires --pci-bdf BDF (hardware) or --offline-shim (drm-shim, labelled offline)");
   if (want_bdf && offline_shim)
      DIE("--pci-bdf and --offline-shim are exclusive");

   size_t spv_size;
   uint32_t *spv = read_file(spv_path, &spv_size);

   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                            .pApplicationName = "hipfire-aco-export",
                            .apiVersion = VK_API_VERSION_1_3};
   VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
   VkInstance instance;
   CHECK(vkCreateInstance(&ici, NULL, &instance));

   uint32_t ndev = 0;
   CHECK(vkEnumeratePhysicalDevices(instance, &ndev, NULL));
   VkPhysicalDevice *devs = calloc(ndev ? ndev : 1, sizeof(*devs));
   CHECK(vkEnumeratePhysicalDevices(instance, &ndev, devs));
   VkPhysicalDevice pd = VK_NULL_HANDLE;
   VkPhysicalDeviceDriverProperties drv = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
   VkPhysicalDevicePCIBusInfoPropertiesEXT pci = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT};
   VkPhysicalDeviceIDProperties idp = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
   VkPhysicalDeviceProperties2 props = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
   unsigned matches = 0;
   for (uint32_t i = 0; i < ndev; i++) {
      VkPhysicalDeviceDriverProperties d = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
      VkPhysicalDevicePCIBusInfoPropertiesEXT pb = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT};
      VkPhysicalDeviceIDProperties id = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
      VkPhysicalDeviceProperties2 p = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &d};
      d.pNext = &pb;
      pb.pNext = &id;
      vkGetPhysicalDeviceProperties2(devs[i], &p);
      char bdf[32];
      snprintf(bdf, sizeof(bdf), "%04x:%02x:%02x.%x", pb.pciDomain, pb.pciBus, pb.pciDevice, pb.pciFunction);
      fprintf(stderr, "vk_aco_host: physical device %u: %s driverID=%d pci=%s\n", i, p.properties.deviceName,
              (int)d.driverID, bdf);
      /* Hardware reference: ONLY the exact PCI BDF is acceptable (Vulkan ignores ROCR_VISIBLE_DEVICES and all
       * gfx1201 cards share the same name). Offline compile: the single RADV device the shim exposes. */
      bool pick = want_bdf ? (d.driverID == VK_DRIVER_ID_MESA_RADV && !strcasecmp(bdf, want_bdf))
                           : (d.driverID == VK_DRIVER_ID_MESA_RADV);
      if (pick) {
         matches++;
         if (pd == VK_NULL_HANDLE) {
            pd = devs[i];
            drv = d; drv.pNext = NULL;
            pci = pb; pci.pNext = NULL;
            idp = id; idp.pNext = NULL;
            props = p;
            props.pNext = NULL;
         }
      }
   }
   if (matches != 1)
      DIE("%u RADV physical devices match %s among %u (VK_DRIVER_FILES must name the patched RADV ICD; hardware "
          "references need --pci-bdf)", matches, want_bdf ? want_bdf : "offline selection", ndev);

   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);

   /* Feature chain. */
   VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pexf = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR,
      .pipelineExecutableInfo = VK_TRUE};
   VkPhysicalDeviceVulkan13Features f13 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
                                           .pNext = &pexf,
                                           .subgroupSizeControl = VK_TRUE,
                                           .computeFullSubgroups = VK_TRUE};
   VkPhysicalDeviceVulkan12Features f12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                                           .pNext = &f13,
                                           .bufferDeviceAddress = VK_TRUE};
   VkPhysicalDeviceFeatures2 f2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f12};
   f2.features.shaderInt64 = VK_TRUE;

   VkPhysicalDeviceVulkan13Properties p13 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
   VkPhysicalDeviceProperties2 pq = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &p13};
   vkGetPhysicalDeviceProperties2(pd, &pq);
   if (subgroup < p13.minSubgroupSize || subgroup > p13.maxSubgroupSize)
      DIE("required subgroup size %u outside device range [%u,%u]", subgroup, p13.minSubgroupSize, p13.maxSubgroupSize);

   float prio = 1.0f;
   uint32_t nq = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, NULL);
   VkQueueFamilyProperties *qp = calloc(nq, sizeof(*qp));
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qp);
   uint32_t qfam = UINT32_MAX;
   for (uint32_t i = 0; i < nq; i++)
      if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfam = i; break; }
   if (qfam == UINT32_MAX) DIE("no compute queue");
   VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                  .queueFamilyIndex = qfam, .queueCount = 1, .pQueuePriorities = &prio};
   const char *dev_exts[] = {VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME};
   VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &f2, .queueCreateInfoCount = 1,
                             .pQueueCreateInfos = &qci, .enabledExtensionCount = 1, .ppEnabledExtensionNames = dev_exts};
   VkDevice dev;
   CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, qfam, 0, &queue);

   /* Identity JSON. */
   {
      FILE *f = fopen(identity_out, "wb");
      if (!f) DIE("cannot write %s", identity_out);
      fprintf(f, "{\"selection\":\"%s\",\"device_name\":", want_bdf ? "pci_bdf" : "offline_shim");
      json_str(f, props.properties.deviceName);
      fprintf(f, ",\"vendor_id\":%u,\"device_id\":%u,\"driver_id\":%d,\"driver_name\":", props.properties.vendorID,
              props.properties.deviceID, (int)drv.driverID);
      json_str(f, drv.driverName);
      fprintf(f, ",\"driver_info\":");
      json_str(f, drv.driverInfo);
      fprintf(f, ",\"pci_bdf\":\"%04x:%02x:%02x.%x\",\"device_uuid\":\"", pci.pciDomain, pci.pciBus, pci.pciDevice,
              pci.pciFunction);
      for (int i = 0; i < VK_UUID_SIZE; i++) fprintf(f, "%02x", idp.deviceUUID[i]);
      fprintf(f, "\",\"driver_uuid\":\"");
      for (int i = 0; i < VK_UUID_SIZE; i++) fprintf(f, "%02x", idp.driverUUID[i]);
      fprintf(f, "\",\"api_version\":\"%u.%u.%u\",\"driver_version\":%u,\"conformance_version\":\"%u.%u.%u.%u\","
                 "\"min_subgroup_size\":%u,\"max_subgroup_size\":%u}\n",
              VK_API_VERSION_MAJOR(props.properties.apiVersion), VK_API_VERSION_MINOR(props.properties.apiVersion),
              VK_API_VERSION_PATCH(props.properties.apiVersion), props.properties.driverVersion,
              drv.conformanceVersion.major, drv.conformanceVersion.minor, drv.conformanceVersion.subminor,
              drv.conformanceVersion.patch, p13.minSubgroupSize, p13.maxSubgroupSize);
      fclose(f);
   }

   /* Pipeline. */
   VkShaderModuleCreateInfo smci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = spv_size,
                                    .pCode = spv};
   VkShaderModule module;
   CHECK(vkCreateShaderModule(dev, &smci, NULL, &module));
   VkPushConstantRange pcr = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = push_size};
   VkPipelineLayoutCreateInfo plci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                      .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr};
   VkPipelineLayout layout;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &layout));
   VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rss = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
      .requiredSubgroupSize = subgroup};
   VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .flags = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR | VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR,
      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .pNext = &rss,
                .flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = module, .pName = entry},
      .layout = layout};
   VkPipeline pipeline;
   CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipeline));

   /* Statistics in RADV's own dump text format (radv_dump_shader_stats). */
   {
      PFN_vkGetPipelineExecutablePropertiesKHR getProps =
         (PFN_vkGetPipelineExecutablePropertiesKHR)vkGetDeviceProcAddr(dev, "vkGetPipelineExecutablePropertiesKHR");
      PFN_vkGetPipelineExecutableStatisticsKHR getStats =
         (PFN_vkGetPipelineExecutableStatisticsKHR)vkGetDeviceProcAddr(dev, "vkGetPipelineExecutableStatisticsKHR");
      if (!getProps || !getStats) DIE("pipeline executable properties entry points missing");
      VkPipelineInfoKHR pinfo = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR, .pipeline = pipeline};
      uint32_t nexec = 0;
      CHECK(getProps(dev, &pinfo, &nexec, NULL));
      if (nexec != 1) DIE("expected exactly one pipeline executable, got %u", nexec);
      VkPipelineExecutablePropertiesKHR ep = {.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR};
      uint32_t one = 1;
      CHECK(getProps(dev, &pinfo, &one, &ep));
      VkPipelineExecutableInfoKHR einfo = {.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR,
                                           .pipeline = pipeline, .executableIndex = 0};
      uint32_t ns = 0;
      CHECK(getStats(dev, &einfo, &ns, NULL));
      VkPipelineExecutableStatisticKHR *st = calloc(ns, sizeof(*st));
      for (uint32_t i = 0; i < ns; i++) st[i].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;
      CHECK(getStats(dev, &einfo, &ns, st));
      FILE *f = fopen(stats_out, "wb");
      if (!f) DIE("cannot write %s", stats_out);
      fprintf(f, "Compute Shader:\n*** SHADER STATS ***\n");
      for (uint32_t i = 0; i < ns; i++) {
         fprintf(f, "%s: ", st[i].name);
         switch (st[i].format) {
         case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:
            fprintf(f, "%s", st[i].value.b32 == VK_TRUE ? "true" : "false"); break;
         case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR: fprintf(f, "%" PRIi64, st[i].value.i64); break;
         case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR: fprintf(f, "%" PRIu64, st[i].value.u64); break;
         case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR: fprintf(f, "%f", st[i].value.f64); break;
         default: DIE("unknown statistic format");
         }
         fputc('\n', f);
      }
      fprintf(f, "********************\n\n\n");
      fclose(f);
      /* The executable's reported subgroup size is the wave size actually compiled. */
      fprintf(stderr, "vk_aco_host: executable '%s' subgroupSize=%u\n", ep.name, ep.subgroupSize);
      if (ep.subgroupSize != subgroup)
         DIE("executable subgroup size %u != required %u", ep.subgroupSize, subgroup);
   }

   if (!dispatch) {
      printf("compiled\n");
      return 0;
   }

   /* Buffers: framed pre bytes uploaded verbatim. */
   for (unsigned i = 0; i < nbuf; i++) {
      struct buf *b = &bufs[i];
      size_t sz;
      void *data = read_file(b->path, &sz);
      b->size = sz;
      VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = sz,
                                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
      CHECK(vkCreateBuffer(dev, &bci, NULL, &b->buffer));
      VkMemoryRequirements mr;
      vkGetBufferMemoryRequirements(dev, b->buffer, &mr);
      VkMemoryAllocateFlagsInfo mafi = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
                                        .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
      VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &mafi,
                                  .allocationSize = mr.size,
                                  .memoryTypeIndex = find_memory_type(&mp, mr.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
      CHECK(vkAllocateMemory(dev, &mai, NULL, &b->memory));
      CHECK(vkBindBufferMemory(dev, b->buffer, b->memory, 0));
      CHECK(vkMapMemory(dev, b->memory, 0, VK_WHOLE_SIZE, 0, &b->map));
      memcpy(b->map, data, sz);
      free(data);
      VkBufferDeviceAddressInfo bdai = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = b->buffer};
      b->address = vkGetBufferDeviceAddress(dev, &bdai);
      fprintf(stderr, "vk_aco_host: buffer %s size=%" PRIu64 " address=0x%016" PRIx64 " ptr_offset=%" PRIu64 "\n",
              b->name, b->size, (uint64_t)b->address, b->ptr_offset);
   }

   /* Push constants. */
   uint8_t *push = calloc(push_size, 1);
   {
      char *spec = strdup(push_spec);
      uint32_t off = 0;
      for (char *tok = strtok(spec, ","); tok; tok = strtok(NULL, ",")) {
         if (!strncmp(tok, "u64:buf:", 8)) {
            struct buf *b = NULL;
            for (unsigned i = 0; i < nbuf; i++) if (!strcmp(bufs[i].name, tok + 8)) b = &bufs[i];
            if (!b) DIE("push references unknown buffer %s", tok + 8);
            uint64_t v = (uint64_t)b->address + b->ptr_offset;
            if (off + 8 > push_size) DIE("push overflow");
            memcpy(push + off, &v, 8); off += 8;
         } else if (!strncmp(tok, "u64:", 4)) {
            uint64_t v = strtoull(tok + 4, NULL, 0);
            if (off + 8 > push_size) DIE("push overflow");
            memcpy(push + off, &v, 8); off += 8;
         } else if (!strncmp(tok, "i32:", 4) || !strncmp(tok, "u32:", 4)) {
            uint32_t v = (uint32_t)strtoll(tok + 4, NULL, 0);
            if (off + 4 > push_size) DIE("push overflow");
            memcpy(push + off, &v, 4); off += 4;
         } else if (!strncmp(tok, "f32:", 4)) {
            float v = strtof(tok + 4, NULL);
            if (off + 4 > push_size) DIE("push overflow");
            memcpy(push + off, &v, 4); off += 4;
         } else DIE("bad push item %s", tok);
      }
      if (off != push_size) DIE("push spec packs %u bytes, --push-size is %u", off, push_size);
      free(spec);
   }

   VkCommandPoolCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = qfam};
   VkCommandPool pool;
   CHECK(vkCreateCommandPool(dev, &cpi, NULL, &pool));
   VkCommandBufferAllocateInfo cbai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool,
                                       .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
   VkCommandBuffer cb;
   CHECK(vkAllocateCommandBuffers(dev, &cbai, &cb));
   VkCommandBufferBeginInfo cbbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
   CHECK(vkBeginCommandBuffer(cb, &cbbi));
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
   vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size, push);
   vkCmdDispatch(cb, groups[0], groups[1] ? groups[1] : 1, groups[2] ? groups[2] : 1);
   VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                         .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
   vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0,
                        NULL);
   CHECK(vkEndCommandBuffer(cb));
   VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb};
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   for (unsigned i = 0; i < nbuf; i++) {
      char path[4200];
      snprintf(path, sizeof(path), "%s/%s.post.bin", out_dir, bufs[i].name);
      write_file(path, bufs[i].map, bufs[i].size);
   }
   printf("dispatched\n");
   return 0;
}
