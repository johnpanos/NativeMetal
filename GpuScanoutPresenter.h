// SPDX-License-Identifier: MIT
// GPU presentation: the GPU copies a finished BGRA frame (the compositor's IOSurface pages) into the firmware scanout memory. Both are
// imported without a copy (newBufferWithBytesNoCopy: pinned by the kernel and mapped into the GPU's 4 KiB context), the copy is a blit on
// the GPU's queue, and the call returns after the GPU retired it. No CPU touches a pixel; there is no staging copy and no software fallback
// here. Used through the public Metal API, on any device (the native frontend on the tablet, the Mac's device for the host check).
// A frame of another size (a logical display, fajita-display-logical) is scaled into the fitted, centered rectangle by a compute kernel
// (shaders/scanout_scale.metal, carried as a metallib in this image's __DATA,__scalelib section) with the CPU copy's taps
// (NativeMobileDisplay/NativeLegacyScale.h); the letterbox is cleared once by the GPU. Without that section such frames are refused.
#pragma once
#import <Metal/Metal.h>
#include <dlfcn.h>
#include <mach-o/getsect.h>
#include <mach-o/loader.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>

// The native frontend's opt-out from its commit-time cleaning of imports (NativeBridgeMetalFrontend.mm): the frame is CoreAnimation's GPU
// output, which no CPU wrote. A no-op elsewhere (the Mac's device for the host check); the frontend's definition replaces this one.
extern "C" __attribute__((weak)) void NBImportNoFlush(id buffer) {(void)buffer;}

namespace gpupresent {

struct Scanout {
    void *base;       // writable mapping of the firmware framebuffer (page aligned)
    size_t bytes;     // whole extent, a multiple of the page size
    size_t stride;    // bytes per row
    uint32_t width, height;
};

struct Frame {
    const void *base; // page-aligned mapping of the surface pages
    size_t bytes;     // allocation size, a multiple of the page size
    size_t stride;
    uint32_t width, height;
};

enum Result { kOk = 0, kBadFrame, kBadScanout, kNoResource, kGpuFailed };

class Presenter {
public:
    Presenter(id<MTLDevice> device, const Scanout &scanout) : device_(device), scanout_(scanout) {
        queue_ = [device newCommandQueue];
        if (!validScanout() || !queue_) return;
        target_ = [device newBufferWithBytesNoCopy:scanout.base length:scanout.bytes options:MTLResourceStorageModeShared deallocator:nil];
    }
    bool ready() const { return target_ != nil; }
    // Last scaled-path failure (0 none): 1 no metallib section, 2 library, 3 function, 4 pipeline, 5 geometry, 6 taps.
    int scaleError() const { return scaleError_; }
    // One frame: encodes the GPU copy, commits, waits for the GPU. Rows are copied as whole rows of width*4 bytes.
    Result present(const Frame &frame) {
        if (!ready()) return kBadScanout;
        const size_t page = (size_t)getpagesize(), row = (size_t)frame.width * 4;
        if (!frame.base || ((uintptr_t)frame.base & (page - 1)) || !frame.bytes || (frame.bytes & (page - 1)) ||
            !frame.width || !frame.height || frame.stride < row || (frame.stride & 3) ||
            (size_t)frame.height > frame.bytes / frame.stride || frame.stride * (frame.height - 1) + row > frame.bytes) return kBadFrame;
        const bool scaled = frame.width != scanout_.width || frame.height != scanout_.height;
        if (scaled && !prepareScale(frame.width, frame.height)) return kBadFrame;
        id<MTLBuffer> source = [device_ newBufferWithBytesNoCopy:(void *)frame.base length:frame.bytes options:MTLResourceStorageModeShared deallocator:nil];
        if (!source) return kNoResource;
        NBImportNoFlush(source);
        if (scaled) return presentScaled(source, frame);
        id<MTLCommandBuffer> commands = [queue_ commandBuffer];
        id<MTLBlitCommandEncoder> blit = commands ? [commands blitCommandEncoder] : nil;
        if (!blit) return kNoResource;
        if (frame.stride == scanout_.stride) {
            // Equal pitch: the rows are one contiguous run.
            [blit copyFromBuffer:source sourceOffset:0 toBuffer:target_ destinationOffset:0 size:frame.stride * (frame.height - 1) + row];
        } else {
            // A linear texture over the surface pages reads them with the surface's pitch and writes the scanout's pitch.
            MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:frame.width
                                                                                       height:frame.height mipmapped:NO];
            d.storageMode = MTLStorageModeShared;
            d.usage = MTLTextureUsageShaderRead;
            if (frame.stride % [device_ minimumLinearTextureAlignmentForPixelFormat:MTLPixelFormatBGRA8Unorm]) {
                for (uint32_t y = 0; y < frame.height; ++y)
                    [blit copyFromBuffer:source sourceOffset:y * frame.stride toBuffer:target_ destinationOffset:y * scanout_.stride size:row];
            } else {
                id<MTLTexture> texture = [source newTextureWithDescriptor:d offset:0 bytesPerRow:frame.stride];
                if (!texture) {[blit endEncoding]; return kNoResource;}
                [blit copyFromTexture:texture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
                           sourceSize:MTLSizeMake(frame.width, frame.height, 1) toBuffer:target_ destinationOffset:0
                    destinationBytesPerRow:scanout_.stride destinationBytesPerImage:scanout_.stride * frame.height];
            }
        }
        [blit endEncoding];
        [commands commit];
        [commands waitUntilCompleted];
        return commands.status == MTLCommandBufferStatusCompleted ? kOk : kGpuFailed;
    }
private:
    struct ScaleParams { uint32_t ow, oh, ox, oy, srcWords, dstWords; };
    // The pipeline once, and the fit and taps whenever the logical size changes (native_legacy_scale_fit / _taps, same rounding).
    bool prepareScale(uint32_t lw, uint32_t lh) {
        if (!pipeline_) {
            if (scaleError_ && scaleError_ < 5) return false;   // a missing or broken library does not get better
            Dl_info info = {};
            unsigned long size = 0;
            const uint8_t *bytes = nullptr;
            if (dladdr((const void *)&scanoutScaleAnchor, &info) && info.dli_fbase)
                bytes = getsectiondata((const struct mach_header_64 *)info.dli_fbase, "__DATA", "__scalelib", &size);
            if (!bytes || !size) return scaleFail(1);
            dispatch_data_t data = dispatch_data_create(bytes, size, nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
            NSError *error = nil;
            id<MTLLibrary> library = [device_ newLibraryWithData:data error:&error];
            if (!library) return scaleFail(2);
            id<MTLFunction> function = [library newFunctionWithName:@"scanout_scale"];
            if (!function) return scaleFail(3);
            pipeline_ = [device_ newComputePipelineStateWithFunction:function error:&error];
            if (!pipeline_) return scaleFail(4);
        }
        if (lw == mapW_ && lh == mapH_ && taps_) return true;
        taps_ = nil; mapW_ = mapH_ = 0; letterboxed_ = false;
        const uint32_t pw = scanout_.width, ph = scanout_.height;
        if (lw > 16384 || lh > 16384 || lw < 2 || lh < 2) return scaleFail(5);
        uint64_t w, h;
        if ((uint64_t)pw * lh <= (uint64_t)ph * lw) { w = pw; h = ((uint64_t)lh * pw * 2 + lw) / ((uint64_t)lw * 2); }
        else { h = ph; w = ((uint64_t)lw * ph * 2 + lh) / ((uint64_t)lh * 2); }
        if (w > pw) w = pw;
        if (h > ph) h = ph;
        w &= ~1ull;
        if (!w || !h) return scaleFail(5);
        params_ = {(uint32_t)w, (uint32_t)h, ((pw - (uint32_t)w) / 2) & ~1u, (ph - (uint32_t)h) / 2, 0, (uint32_t)(scanout_.stride / 4)};
        taps_ = [device_ newBufferWithLength:(NSUInteger)(2 * w + 2 * h) * 4 options:MTLResourceStorageModeShared];
        if (!taps_) return scaleFail(6);
        uint32_t *t = (uint32_t *)taps_.contents;
        fillTaps(t, t + w, (uint32_t)w, lw);
        fillTaps(t + 2 * w, t + 2 * w + h, (uint32_t)h, lh);
        mapW_ = lw; mapH_ = lh;
        return true;
    }
    static void fillTaps(uint32_t *a, uint32_t *b, uint32_t out, uint32_t in) {
        for (uint32_t i = 0; i < out; ++i) {
            const uint64_t twice = (uint64_t)(2 * i + 1) * in;
            uint32_t first;
            if (in > out) {
                uint64_t left = twice > out ? (twice - out) / (2ull * out) : 0;
                if (left > in - 2) left = in - 2;
                first = (uint32_t)left; a[i] = first; b[i] = first + 1;
            } else {
                uint64_t nearest = twice / (2ull * out);
                if (nearest > in - 1) nearest = in - 1;
                a[i] = b[i] = (uint32_t)nearest;
            }
        }
    }
    Result presentScaled(id<MTLBuffer> source, const Frame &frame) {
        id<MTLCommandBuffer> commands = [queue_ commandBuffer];
        if (!commands) return kNoResource;
        if (!letterboxed_ && (params_.ow != scanout_.width || params_.oh != scanout_.height)) {
            id<MTLBlitCommandEncoder> blit = [commands blitCommandEncoder];
            if (!blit) return kNoResource;
            [blit fillBuffer:target_ range:NSMakeRange(0, scanout_.stride * scanout_.height) value:0];
            [blit endEncoding];
        }
        id<MTLComputeCommandEncoder> compute = [commands computeCommandEncoder];
        if (!compute) return kNoResource;
        ScaleParams p = params_;
        p.srcWords = (uint32_t)(frame.stride / 4);
        [compute setComputePipelineState:pipeline_];
        [compute setBuffer:source offset:0 atIndex:0];
        [compute setBuffer:target_ offset:0 atIndex:1];
        [compute setBuffer:taps_ offset:0 atIndex:2];
        [compute setBytes:&p length:sizeof p atIndex:3];
        const MTLSize group = MTLSizeMake(16, 8, 1);
        [compute dispatchThreadgroups:MTLSizeMake((p.ow + 15) / 16, (p.oh + 7) / 8, 1) threadsPerThreadgroup:group];
        [compute endEncoding];
        [commands commit];
        [commands waitUntilCompleted];
        if (commands.status != MTLCommandBufferStatusCompleted) return kGpuFailed;
        letterboxed_ = true;
        return kOk;
    }
    bool scaleFail(int error) { scaleError_ = error; return false; }
    static void scanoutScaleAnchor() {}
    bool validScanout() const {
        const size_t page = (size_t)getpagesize();
        return scanout_.base && !((uintptr_t)scanout_.base & (page - 1)) && scanout_.bytes && !(scanout_.bytes & (page - 1)) &&
               scanout_.width && scanout_.height && scanout_.stride >= (size_t)scanout_.width * 4 &&
               scanout_.stride * scanout_.height <= scanout_.bytes;
    }
    id<MTLDevice> device_;
    Scanout scanout_;
    id<MTLCommandQueue> queue_;
    id<MTLBuffer> target_;
    id<MTLComputePipelineState> pipeline_ = nil;
    id<MTLBuffer> taps_ = nil;
    ScaleParams params_ = {};
    uint32_t mapW_ = 0, mapH_ = 0;
    bool letterboxed_ = false;
    int scaleError_ = 0;
};

}  // namespace gpupresent
