// SPDX-License-Identifier: MIT
// The frame trace (diagnostics, off by default): with NGPU_FRAME_TRACE=1 in the plugin's gpu-env, the frontend writes one line per
// command buffer and one per scanout swap to frametrace.csv in the runtime's cache directory (NGPU_FRAME_TRACE=/abs/dir: that directory),
// as <progname>-<pid>.csv, so the host (tools/gpu/frametrace.py in the fajita board) can rank frames by time and say where each worst
// frame's time went. Off, it costs one predictable branch per command buffer.
//
// Times are microseconds of CLOCK_UPTIME_RAW (mach_absolute_time's clock, the kernel log's and the translator's GPU times' clock).
//   C,seq,queue,created,commit,committed,kstart,kend,gstart,gend,done,handled,passes,draws,dispatches,blits,pixels,pipeUs,pipeN,
//     waitUs,flushUs,flushKiB,label
//       created: the command buffer was made; commit: commit called (the encode ends); committed: commit returned (imports flushed,
//       translated, submitted); kstart/kend: the translator's scheduler submitting it to Vulkan; gstart/gend: its GPU work (the
//       translator's timestamps; uncalibrated, gend is the completion seen and the duration exact); done: the completion callback;
//       handled: its completed handlers ran. pipeUs/pipeN: pipeline, library and function creation on any thread while it was encoded;
//       waitUs: waits for earlier GPU work (resource waits, waitUntilCompleted/Scheduled) while it was encoded; flushUs/flushKiB:
//       the CPU cache maintenance of imports and surfaces at its commit.
//   W,start,us,kind,seq     a waitUntilCompleted (kind 0) or waitUntilScheduled (1) by the caller; seq the command buffer waited on
//   P,start,us,kind,label   a render (0) or compute (1) pipeline, a library (2) or a function (3) made
//   S,seq,wait,offer,presented,completed,width,height,mapId,result
//       the scanout client: began waiting for a swap, was offered one, its GPU copy returned (pipelined: the previous copy completed),
//       the swap was completed
// Lines are buffered and appended from a utility queue in 64 KiB chunks, and at least once a second while swaps arrive; the trace stops
// at NGPU_FRAME_TRACE_MAX bytes (default 16 MiB).
#ifndef NATIVE_METAL_FRAME_TRACE_H
#define NATIVE_METAL_FRAME_TRACE_H
#include <dispatch/dispatch.h>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

namespace ftrace {
inline bool &enabled() {static bool on=false;return on;}
inline bool on() {return __builtin_expect(enabled(),0);}
inline std::string &path() {static std::string p;return p;}
inline std::mutex &lock() {static std::mutex m;return m;}
inline std::string &pending() {static std::string s;return s;}
inline uint64_t &written() {static uint64_t n=0;return n;}
inline uint64_t &lastFlush() {static uint64_t t=0;return t;}
inline uint64_t limit() {static const uint64_t n=[]{const char *v=getenv("NGPU_FRAME_TRACE_MAX");return v?(uint64_t)atoll(v):(uint64_t)16<<20;}();return n;}
inline uint64_t now() {return clock_gettime_nsec_np(CLOCK_UPTIME_RAW)/1000;}
inline dispatch_queue_t writer() {
 static dispatch_queue_t q=dispatch_queue_create("ngpu.frametrace",dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL,QOS_CLASS_UTILITY,0));
 return q;
}
// Called once with the runtime's cache directory (may be null).
inline void begin(const char *cache) {
 const char *v=getenv("NGPU_FRAME_TRACE");
 if(!v || !v[0] || !strcmp(v,"0"))return;
 std::string base=v[0]=='/'?std::string(v):cache?std::string(cache)+"/frametrace":std::string();
 if(base.empty())return;
 mkdir(base.c_str(),0755);
 char leaf[96];snprintf(leaf,sizeof leaf,"/%s-%d.csv",getprogname(),(int)getpid());
 path()=base+leaf;
 if(FILE *f=fopen(path().c_str(),"w")){fprintf(f,"# frametrace v1 %s pid %d t0 %llu\n",getprogname(),(int)getpid(),(unsigned long long)now());fclose(f);}
 else {path().clear();return;}
 enabled()=true;
}
// Hands the buffered lines to the writer queue (the caller holds lock()).
inline void flushLocked() {
 if(pending().empty())return;
 std::string *chunk=new std::string;chunk->swap(pending());
 lastFlush()=now();
 const std::string file=path();
 dispatch_async(writer(),^{
  if(FILE *f=fopen(file.c_str(),"a")){fwrite(chunk->data(),1,chunk->size(),f);fclose(f);}
  delete chunk;
 });
}
inline void line(const char *text,size_t n) {
 std::lock_guard<std::mutex> guard(lock());
 if(!enabled())return;
 if(written()+n>limit()){flushLocked();enabled()=false;return;}
 written()+=n;
 pending().append(text,n);
 if(pending().size()>=(64u<<10) || now()-lastFlush()>1000000)flushLocked();
}
inline void linef(const char *format,...) __attribute__((format(printf,1,2)));
inline void linef(const char *format,...) {
 if(!on())return;
 char text[512];
 va_list ap;va_start(ap,format);int n=vsnprintf(text,sizeof text-1,format,ap);va_end(ap);
 if(n<0)return;
 if(n>(int)sizeof text-2)n=(int)sizeof text-2;
 text[n++]='\n';
 line(text,(size_t)n);
}
// A label as one CSV field: commas and line breaks become spaces.
inline std::string field(const char *s) {
 std::string out=s?s:"";
 for(char &c:out)if(c==','||c=='\n'||c=='\r')c=' ';
 if(out.size()>60)out.resize(60);
 return out;
}
}  // namespace ftrace
#endif
