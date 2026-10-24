// SPDX-License-Identifier: MIT
// GPU hand-off for a logical display (fajita-display-logical): one logical BGRA frame scaled into the fitted image rectangle of the
// scanout, with the CPU copy's taps (NativeMobileDisplay/NativeLegacyScale.h): per image column and row two source indices, the 2x2 block
// averaged byte-wise with rounding down. The tap tables are built on the CPU once (GpuScanoutPresenter.h); buffers only, no textures.
#include <metal_stdlib>
using namespace metal;

struct ScaleParams {
    uint ow, oh;          // image rectangle size
    uint ox, oy;          // its origin on the scanout
    uint srcWords;        // source stride in 32-bit words
    uint dstWords;        // scanout stride in 32-bit words
};

static inline uint avg2(uint a, uint b) { return (a & b) + (((a ^ b) & 0xfefefefeu) >> 1); }

// taps: col0[ow], col1[ow], row0[oh], row1[oh] (pixel and row indices into the source).
kernel void scanout_scale(device const uint *src [[buffer(0)]], device uint *dst [[buffer(1)]], device const uint *taps [[buffer(2)]],
                          constant ScaleParams &p [[buffer(3)]], uint2 g [[thread_position_in_grid]]) {
    if (g.x >= p.ow || g.y >= p.oh) return;
    const uint c0 = taps[g.x], c1 = taps[p.ow + g.x];
    const uint r0 = taps[2 * p.ow + g.y] * p.srcWords, r1 = taps[2 * p.ow + p.oh + g.y] * p.srcWords;
    const uint v = avg2(avg2(src[r0 + c0], src[r0 + c1]), avg2(src[r1 + c0], src[r1 + c1]));
    dst[(p.oy + g.y) * p.dstWords + p.ox + g.x] = v;
}

// Four image columns per thread, written with one 16-byte store: the scanout is device memory to the GPU, where every store is its own
// transaction, so the scalar kernel's 4-byte stores cost four times the transactions. Same taps and rounding as scanout_scale. Used when
// the rectangle's origin, width and the scanout stride are multiples of four pixels (GpuScanoutPresenter.h); grid x is ow / 4.
kernel void scanout_scale4(device const uint *src [[buffer(0)]], device uint4 *dst [[buffer(1)]], device const uint *taps [[buffer(2)]],
                           constant ScaleParams &p [[buffer(3)]], uint2 g [[thread_position_in_grid]]) {
    const uint x0 = g.x * 4;
    if (x0 >= p.ow || g.y >= p.oh) return;
    const uint r0 = taps[2 * p.ow + g.y] * p.srcWords, r1 = taps[2 * p.ow + p.oh + g.y] * p.srcWords;
    uint4 v;
    for (uint i = 0; i < 4; ++i) {
        const uint c0 = taps[x0 + i], c1 = taps[p.ow + x0 + i];
        v[i] = avg2(avg2(src[r0 + c0], src[r0 + c1]), avg2(src[r1 + c0], src[r1 + c1]));
    }
    dst[((p.oy + g.y) * p.dstWords + p.ox + x0) / 4] = v;
}

// scanout_scale4 with vector loads: the source rows are read as two aligned 16-byte loads per row (the 4 image columns' taps span at most
// 8 source pixels from their aligned start when the scale is under 4/3) instead of sixteen 4-byte loads, and the taps as two 16-byte loads. The A630 ran the scalar
// loads at the texture pipe's request rate (7.9 ms a 1125x2436 -> 1080x2340 frame, fajita u_trace). Same taps and rounding as
// scanout_scale. Used when scanout_scale4 is, the source stride is a multiple of 16 bytes and every group's taps span at most three
// aligned 16-byte words of a row (GpuScanoutPresenter.h checks it when it builds the taps).
kernel void scanout_scale4v(device const uint4 *src [[buffer(0)]], device uint4 *dst [[buffer(1)]], device const uint4 *taps4 [[buffer(2)]],
                            constant ScaleParams &p [[buffer(3)]], uint2 g [[thread_position_in_grid]]) {
    const uint x0 = g.x * 4;
    if (x0 >= p.ow || g.y >= p.oh) return;
    device const uint *taps = (device const uint *)taps4;
    const uint rowQuads = p.srcWords / 4;
    const uint q0 = taps[2 * p.ow + g.y] * rowQuads, q1 = taps[2 * p.ow + p.oh + g.y] * rowQuads;
    const uint4 c0 = taps4[x0 / 4], c1 = taps4[(p.ow + x0) / 4];
    const uint lo = min(c0.x, c1.x) / 4, last = max(c0.w, c1.w) / 4 - lo;
    uint a[8], b[8];
    for (uint j = 0; j < 2; ++j) {
        const uint k = lo + min(j, last);
        const uint4 u = src[q0 + k], w = src[q1 + k];
        a[4 * j] = u.x; a[4 * j + 1] = u.y; a[4 * j + 2] = u.z; a[4 * j + 3] = u.w;
        b[4 * j] = w.x; b[4 * j + 1] = w.y; b[4 * j + 2] = w.z; b[4 * j + 3] = w.w;
    }
    const uint base = lo * 4;
    uint4 v;
    for (uint i = 0; i < 4; ++i) {
        const uint i0 = c0[i] - base, i1 = c1[i] - base;
        v[i] = avg2(avg2(a[i0], a[i1]), avg2(b[i0], b[i1]));
    }
    dst[((p.oy + g.y) * p.dstWords + p.ox + x0) / 4] = v;
}
