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
    uint32_t mapId;   // NGPSwap.reserved: nonzero when the kernel keeps this mapping across swaps, so its import can be kept too
};

enum Result { kOk = 0, kBadFrame, kBadScanout, kNoResource, kGpuFailed };

class Presenter {
public:
    Presenter(id<MTLDevice> device, const Scanout &scanout) : device_(device), scanout_(scanout) {
        queue_ = [device newCommandQueue];
        queue_.label = @"ngpu.present";
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
        id<MTLBuffer> source = import(frame);
        if (!source) return kNoResource;
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
        return finish(commands, frame);
    }
    uint64_t importsKept() const { return kept_; }
    uint64_t pipelined() const { return pipelined_; }
    uint64_t importsMade() const { return made_; }
private:
    // The frame's pages as a no-copy buffer. A mapping the kernel keeps (mapId) keeps its import: the kernel's cache holds as many entries and
    // evicts its least recently used one as this does, on the same sequence of frames, so an id dropped here is one the kernel released.
    // Pipelined presentation (a kept import only; NGPU_PRESENT_PIPELINE=0 turns it off): the frame's copy is committed and the call returns
    // once the PREVIOUS frame's copy has completed, so the swap is completed while the GPU still copies this frame. The compositor reuses a
    // surface only after the swap that replaced it on screen completes, and by then its copy has retired (this wait), so no surface is
    // rewritten under a copy; copies run in order on this queue, so the scanout still receives the frames in order.
    id<MTLCommandBuffer> pending_ = nil;
    uint64_t pipelined_ = 0;
    static bool pipelineAllowed() {
        static const bool on = [] {const char *v = getenv("NGPU_PRESENT_PIPELINE"); return !(v && v[0] == '0');}();
        return on;
    }
    Result finish(id<MTLCommandBuffer> commands, const Frame &frame) {
        id<MTLCommandBuffer> previous = pending_;
        pending_ = nil;
        if (previous) {
            [previous waitUntilCompleted];
            if (previous.status != MTLCommandBufferStatusCompleted) {[commands waitUntilCompleted]; return kGpuFailed;}
        }
        if (frame.mapId && pipelineAllowed()) {pending_ = commands; ++pipelined_; return kOk;}
        [commands waitUntilCompleted];
        return commands.status == MTLCommandBufferStatusCompleted ? kOk : kGpuFailed;
    }
    static const unsigned kImports = 4;
    struct Import { uint32_t mapId; const void *base; size_t bytes; uint64_t used; id<MTLBuffer> buffer; };
    Import imports_[kImports] = {};
    uint64_t clock_ = 0, kept_ = 0, made_ = 0;
    id<MTLBuffer> import(const Frame &frame) {
        if (frame.mapId)
            for (Import &e : imports_)
                if (e.buffer && e.mapId == frame.mapId && e.base == frame.base && e.bytes == frame.bytes) {e.used = ++clock_; ++kept_; return e.buffer;}
        id<MTLBuffer> buffer = [device_ newBufferWithBytesNoCopy:(void *)frame.base length:frame.bytes options:MTLResourceStorageModeShared deallocator:nil];
        if (!buffer) return nil;
        NBImportNoFlush(buffer);
        ++made_;
        if (frame.mapId) {
            Import *victim = &imports_[0];
            for (Import &e : imports_) {
                if (!e.buffer) {victim = &e; break;}
                if (e.used < victim->used) victim = &e;
            }
            victim->buffer = nil;
            *victim = {frame.mapId, frame.base, frame.bytes, ++clock_, buffer};
        }
        return buffer;
    }
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
            // The 16-byte-store variant (NGPU_SCALE4=0 keeps the scalar kernel); an older metallib without it keeps the scalar one too.
            const char *v4 = getenv("NGPU_SCALE4");
            id<MTLFunction> function4 = v4 && v4[0] == '0' ? nil : [library newFunctionWithName:@"scanout_scale4"];
            if (function4) pipeline4_ = [device_ newComputePipelineStateWithFunction:function4 error:&error];
            // Its vector-load form (NGPU_SCALE4V=0 keeps scanout_scale4); an older metallib without it keeps scanout_scale4 too.
            const char *v4v = getenv("NGPU_SCALE4V");
            id<MTLFunction> function4v = !function4 || (v4v && v4v[0] == '0') ? nil : [library newFunctionWithName:@"scanout_scale4v"];
            if (function4v) pipeline4v_ = [device_ newComputePipelineStateWithFunction:function4v error:&error];
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
        // scanout_scale4v reads each group's taps from at most two aligned 16-byte words of a source row.
        span2_ = !(w % 4);
        for (uint64_t x = 0; span2_ && x < w; x += 4) {
            const uint32_t lo = (t[x] < t[w + x] ? t[x] : t[w + x]) / 4, hi = (t[x + 3] > t[w + x + 3] ? t[x + 3] : t[w + x + 3]) / 4;
            if (hi - lo > 1) span2_ = false;
        }
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
        const bool wide = pipeline4_ && !(p.ow % 4) && !(p.ox % 4) && !(p.dstWords % 4);
        const bool vector = wide && pipeline4v_ && span2_ && !(p.srcWords % 4);
        [compute setComputePipelineState:vector ? pipeline4v_ : wide ? pipeline4_ : pipeline_];
        [compute setBuffer:source offset:0 atIndex:0];
        [compute setBuffer:target_ offset:0 atIndex:1];
        [compute setBuffer:taps_ offset:0 atIndex:2];
        [compute setBytes:&p length:sizeof p atIndex:3];
        const MTLSize group = MTLSizeMake(16, 8, 1);
        const uint32_t cols = wide ? p.ow / 4 : p.ow;
        [compute dispatchThreadgroups:MTLSizeMake((cols + 15) / 16, (p.oh + 7) / 8, 1) threadsPerThreadgroup:group];
        [compute endEncoding];
        [commands commit];
        const Result r = finish(commands, frame);
        if (r != kOk) return r;
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
    id<MTLComputePipelineState> pipeline_ = nil, pipeline4_ = nil, pipeline4v_ = nil;
    bool span2_ = false;
    id<MTLBuffer> taps_ = nil;
    ScaleParams params_ = {};
    uint32_t mapW_ = 0, mapH_ = 0;
    bool letterboxed_ = false;
    int scaleError_ = 0;
};

}  // namespace gpupresent
