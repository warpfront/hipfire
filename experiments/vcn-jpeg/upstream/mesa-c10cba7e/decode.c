/*
 * Copyright © 2025 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 */

#include "pipe/p_video_codec.h"

#include "util/vl_vlc.h"
#include "util/u_video.h"

#include "va_private.h"

static VAStatus
handlePictureParameterBuffer(vlVaDriver *drv, vlVaContext *context, vlVaBuffer *buf)
{
   VAStatus vaStatus = VA_STATUS_SUCCESS;
   enum pipe_video_format format =
      u_reduce_video_profile(context->templat.profile);

   switch (format) {
   case PIPE_VIDEO_FORMAT_MPEG12:
      vlVaHandlePictureParameterBufferMPEG12(drv, context, buf);
      break;

   case PIPE_VIDEO_FORMAT_MPEG4_AVC:
      vlVaHandlePictureParameterBufferH264(drv, context, buf);
      break;

   case PIPE_VIDEO_FORMAT_VC1:
      vlVaHandlePictureParameterBufferVC1(drv, context, buf);
      break;

   case PIPE_VIDEO_FORMAT_HEVC:
      vlVaHandlePictureParameterBufferHEVC(drv, context, buf);
      break;

   case PIPE_VIDEO_FORMAT_JPEG:
      vlVaHandlePictureParameterBufferMJPEG(drv, context, buf);
      break;

   case PIPE_VIDEO_FORMAT_VP9:
      vlVaHandlePictureParameterBufferVP9(drv, context, buf);
      break;

   case PIPE_VIDEO_FORMAT_AV1:
      vaStatus = vlVaHandlePictureParameterBufferAV1(drv, context, buf);
      break;

   default:
      break;
   }

   /* Create the decoder once max_references is known. */
   if (!context->decoder) {
      if (!context->target)
         return VA_STATUS_ERROR_INVALID_CONTEXT;

      mtx_lock(&context->mutex);

      if (format == PIPE_VIDEO_FORMAT_MPEG4_AVC)
         context->templat.level = u_get_h264_level(context->templat.width,
            context->templat.height, &context->templat.max_references);

      context->decoder = drv->pipe->create_video_codec(drv->pipe,
         &context->templat);

      mtx_unlock(&context->mutex);

      if (!context->decoder)
         return VA_STATUS_ERROR_ALLOCATION_FAILED;

      context->needs_begin_frame = true;
   }

   if (format == PIPE_VIDEO_FORMAT_VP9) {
      context->decoder->width =
         context->desc.vp9.picture_parameter.frame_width;
      context->decoder->height =
         context->desc.vp9.picture_parameter.frame_height;
   }

   return vaStatus;
}

static void
handleIQMatrixBuffer(vlVaContext *context, vlVaBuffer *buf)
{
   switch (u_reduce_video_profile(context->templat.profile)) {
   case PIPE_VIDEO_FORMAT_MPEG12:
      vlVaHandleIQMatrixBufferMPEG12(context, buf);
      break;

   case PIPE_VIDEO_FORMAT_MPEG4_AVC:
      vlVaHandleIQMatrixBufferH264(context, buf);
      break;

   case PIPE_VIDEO_FORMAT_HEVC:
      vlVaHandleIQMatrixBufferHEVC(context, buf);
      break;

   case PIPE_VIDEO_FORMAT_JPEG:
      vlVaHandleIQMatrixBufferMJPEG(context, buf);
      break;

   default:
      break;
   }
}

static void
handleSliceParameterBuffer(vlVaContext *context, vlVaBuffer *buf)
{
   switch (u_reduce_video_profile(context->templat.profile)) {
   case PIPE_VIDEO_FORMAT_MPEG12:
      vlVaHandleSliceParameterBufferMPEG12(context, buf);
      break;

   case PIPE_VIDEO_FORMAT_VC1:
      vlVaHandleSliceParameterBufferVC1(context, buf);
      break;

   case PIPE_VIDEO_FORMAT_MPEG4_AVC:
      vlVaHandleSliceParameterBufferH264(context, buf);
      break;

   case PIPE_VIDEO_FORMAT_HEVC:
      vlVaHandleSliceParameterBufferHEVC(context, buf);
      break;

   case PIPE_VIDEO_FORMAT_JPEG:
      vlVaHandleSliceParameterBufferMJPEG(context, buf);
      break;

   case PIPE_VIDEO_FORMAT_VP9:
      vlVaHandleSliceParameterBufferVP9(context, buf);
      break;

   case PIPE_VIDEO_FORMAT_AV1:
      vlVaHandleSliceParameterBufferAV1(context, buf);
      break;

   default:
      break;
   }
}

static unsigned int
bufHasStartcode(vlVaBuffer *buf, unsigned int code, unsigned int bits)
{
   struct vl_vlc vlc = {0};
   int i;

   /* search the first 64 bytes for a startcode */
   vl_vlc_init(&vlc, 1, (const void * const*)&buf->data, &buf->size);
   for (i = 0; i < 64 && vl_vlc_bits_left(&vlc) >= bits; ++i) {
      if (vl_vlc_peekbits(&vlc, bits) == code)
         return 1;
      vl_vlc_eatbits(&vlc, 8);
      vl_vlc_fillbits(&vlc);
   }

   return 0;
}

static VAStatus
handleVAProtectedSliceDataBufferType(vlVaContext *context, vlVaBuffer *buf)
{
   char cookie[] = {'w', 'v', 'c', 'e', 'n', 'c', 's', 'b'};
   uint8_t *encrypted_data = (uint8_t*)buf->data;
   uint8_t *drm_key;
   unsigned int drm_key_size = buf->size;

   if (!context->desc.base.protected_playback)
      return VA_STATUS_ERROR_INVALID_CONTEXT;

   drm_key = REALLOC(context->desc.base.decrypt_key,
         context->desc.base.key_size, drm_key_size);
   if (!drm_key)
      return VA_STATUS_ERROR_ALLOCATION_FAILED;
   context->desc.base.decrypt_key = drm_key;
   memcpy(context->desc.base.decrypt_key, encrypted_data, drm_key_size);
   context->desc.base.key_size = drm_key_size;
   /* context->desc.base.cenc defines the type of secure decode being used.
    * true: Native CENC Secure Decode
    * false: Legacy Secure Decode
    */
   if (memcmp(encrypted_data, cookie, sizeof(cookie)) == 0)
      context->desc.base.cenc = true;

   return VA_STATUS_SUCCESS;
}

static void
vlVaAddSliceDataBuffer(vlVaContext *context, const void *data, unsigned size)
{
   util_dynarray_append(&context->bs.buffers, data);
   util_dynarray_append(&context->bs.sizes, size);
}

static VAStatus
handleVASliceDataBufferType(vlVaContext *context, vlVaBuffer *buf)
{
   enum pipe_video_format format = u_reduce_video_profile(context->templat.profile);
   static const uint8_t start_code_h264[] = { 0x00, 0x00, 0x01 };
   static const uint8_t start_code_h265[] = { 0x00, 0x00, 0x01 };
   static const uint8_t start_code_vc1_frame[] = { 0x00, 0x00, 0x01, 0x0d };
   static const uint8_t start_code_vc1_field[] = { 0x00, 0x00, 0x01, 0x0c };
   static const uint8_t start_code_vc1_slice[] = { 0x00, 0x00, 0x01, 0x0b };
   static const uint8_t eoi_jpeg[] = { 0xff, 0xd9 };

   if (!context->decoder)
      return VA_STATUS_ERROR_INVALID_CONTEXT;

   format = u_reduce_video_profile(context->templat.profile);
   if (!context->desc.base.protected_playback) {
      switch (format) {
      case PIPE_VIDEO_FORMAT_MPEG4_AVC:
         if (bufHasStartcode(buf, 0x000001, 24))
            break;

         vlVaAddSliceDataBuffer(context, start_code_h264, sizeof(start_code_h264));
         break;
      case PIPE_VIDEO_FORMAT_HEVC:
         if (bufHasStartcode(buf, 0x000001, 24))
            break;

         vlVaAddSliceDataBuffer(context, start_code_h265, sizeof(start_code_h265));
         vlVaDecoderHEVCBitstreamHeader(context, buf);
         break;
      case PIPE_VIDEO_FORMAT_VC1:
         if (bufHasStartcode(buf, 0x000001, 24))
            break;

         if (context->decoder->profile == PIPE_VIDEO_PROFILE_VC1_ADVANCED) {
            const uint8_t *start_code;
            if (context->slice_data_offset)
               start_code = start_code_vc1_slice;
            else if (context->desc.vc1.is_first_field)
               start_code = start_code_vc1_frame;
            else
               start_code = start_code_vc1_field;
            vlVaAddSliceDataBuffer(context, start_code, sizeof(start_code_vc1_frame));
         }
         break;
      case PIPE_VIDEO_FORMAT_JPEG:
         if (bufHasStartcode(buf, 0xffd8ffdb, 32))
            break;

         vlVaGetJpegSliceHeader(context);
         vlVaAddSliceDataBuffer(context, context->mjpeg.slice_header, context->mjpeg.slice_header_size);
         break;
      case PIPE_VIDEO_FORMAT_VP9:
         vlVaDecoderVP9BitstreamHeader(context, buf);
         break;
      case PIPE_VIDEO_FORMAT_AV1:
         break;
      default:
         break;
      }
   }

   vlVaAddSliceDataBuffer(context, buf->data, buf->size);

   if (format == PIPE_VIDEO_FORMAT_JPEG)
      vlVaAddSliceDataBuffer(context, eoi_jpeg, sizeof(eoi_jpeg));

   if (context->needs_begin_frame) {
      context->decoder->begin_frame(context->decoder, context->target,
         &context->desc.base);
      context->needs_begin_frame = false;
   }
   return VA_STATUS_SUCCESS;
}

VAStatus
vlVaHandleDecBufferType(vlVaDriver *drv, vlVaContext *context, vlVaBuffer *buf)
{
   VAStatus vaStatus = VA_STATUS_SUCCESS;

   switch (buf->type) {
   case VAPictureParameterBufferType:
      vaStatus = handlePictureParameterBuffer(drv, context, buf);
      break;

   case VAIQMatrixBufferType:
      handleIQMatrixBuffer(context, buf);
      break;

   case VASliceParameterBufferType:
      handleSliceParameterBuffer(context, buf);
      context->have_slice_params = true;
      break;

   case VASliceDataBufferType:
      vaStatus = handleVASliceDataBufferType(context, buf);
      /* Workaround for apps sending single slice data buffer followed
       * by multiple slice parameter buffers. */
      if (context->have_slice_params)
         context->slice_data_offset += buf->size;
      break;

   case VAHuffmanTableBufferType:
      vlVaHandleHuffmanTableBufferType(context, buf);
      break;

   case VAProtectedSliceDataBufferType:
      vaStatus = handleVAProtectedSliceDataBufferType(context, buf);
      break;

   default:
      break;
   }

   return vaStatus;
}
