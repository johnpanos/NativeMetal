// SPDX-License-Identifier: MIT
// The running OS build, for the few places that depend on one exact target image: the iPadOS 17.7.11 (21H461) diagnostics that read fixed
// QuartzCore addresses, and the original _MTLDevice instance size (536 bytes on 21H461, 528 on iOS 16.7.16 20H392).
#pragma once
#include <string.h>
#include <stdlib.h>
#include <sys/sysctl.h>

namespace targetbuild {
inline const char *osBuild() {
    static char build[32];
    static bool read;
    if (!read) {
        size_t size = sizeof build;
        if (sysctlbyname("kern.osversion", build, &size, nullptr, 0) != 0) build[0] = 0;
        read = true;
    }
    return build;
}
inline int osMajor() {
    static int major = -1;
    if (major < 0) {
        char version[32] = {};
        size_t size = sizeof version;
        major = sysctlbyname("kern.osproductversion", version, &size, nullptr, 0) == 0 ? atoi(version) : 0;
    }
    return major;
}
// The image whose QuartzCore addresses the display and pipeline-caller diagnostics were derived from.
inline bool is21H461() { return !strcmp(osBuild(), "21H461"); }
// The original _MTLDevice instance size of the running target (the dynamic device subclass adds no instance variables).
inline size_t mtlDeviceBytes() { return osMajor() == 16 ? 528 : 536; }
}  // namespace targetbuild
