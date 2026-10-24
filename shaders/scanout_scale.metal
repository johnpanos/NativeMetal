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
