// SPDX-License-Identifier: MIT
// The CPU profile (diagnostics, off by default): a sampling profiler of the threads that make Metal command buffers, for finding where
// the render server's encode time goes (frametrace.py's cpu_gap / encode columns).
// With NGPU_CPU_PROFILE=1 (every Metal process) or NGPU_CPU_PROFILE=NAME (that process only) in the plugin's gpu-env, every thread that
// makes a command buffer is registered (note()), and a sampler thread interrupts each registered thread every NGPU_CPU_PROFILE_US
// (default 1000) microseconds: thread_suspend, its pc/lr/fp, a frame-pointer walk of at most 24 frames inside its own stack, thread_resume.
// Nothing is symbolized while a thread is stopped (dladdr takes dyld's lock); the summary does it afterwards, per distinct address.
// Each sample is tagged "enc" when a command buffer is open on any registered thread (made, not yet committed: the frame trace's
// encode window), "idle" when the thread is blocked in the kernel (pc in a mach trap / the kernel's wait: libsystem_kernel), else "cpu".
// Every 2 s the summary is rewritten (tmp + rename) as <cache>/frametrace/<progname>-cpuprof.txt: every distinct stack once with its
// counts (enc: running while a command buffer is open on a registered thread; cpu: running, none open; blocked_enc / blocked_other: in
// the kernel, with / without one open), innermost frame first.
// tools/gpu/cpuprof.py (fajita board) symbolizes the plugin's frames and ranks self / inclusive / blocked time.
// A file cpuprof-reset in the same directory clears it at the next summary and is removed (scripts/fajita/test/passtrace.sh reset does).
#ifndef NATIVE_METAL_CPU_PROFILE_H
#define NATIVE_METAL_CPU_PROFILE_H
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <map>
#include <mach/mach.h>
#include <mutex>
#include <pthread.h>
#include <string>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace cpuprof {
constexpr int kDepth = 24;
struct Thread {thread_act_t port;uintptr_t lo,hi;};
struct Sample {uintptr_t pc[kDepth];int n;int tag;int thread;};  // tag 0 enc, 1 cpu, 2 idle
inline bool &armed() {static bool on=false;return on;}
inline bool on() {return __builtin_expect(armed(),0);}
inline std::atomic<int> &open() {static std::atomic<int> n{0};return n;}
inline std::mutex &lock() {static std::mutex m;return m;}
inline std::vector<Thread> &threads() {static std::vector<Thread> t;return t;}
inline std::string &dir() {static std::string d;return d;}
inline uintptr_t strip(uintptr_t a) {return a&0x0000000fffffffffull;}

// A thread that makes a command buffer (called at each one's creation; cheap after the first time on a thread).
inline void note() {
 if(!on())return;
 static thread_local bool noted=false;
 if(noted)return;
 noted=true;
 pthread_t self=pthread_self();
 uintptr_t hi=(uintptr_t)pthread_get_stackaddr_np(self);
 uintptr_t lo=hi-pthread_get_stacksize_np(self);
 std::lock_guard<std::mutex> g(lock());
 threads().push_back({pthread_mach_thread_np(self),lo,hi});
}
inline void opened() {if(on())open().fetch_add(1,std::memory_order_relaxed);}
inline void closed() {if(on())open().fetch_sub(1,std::memory_order_relaxed);}

struct Sym {std::string name;bool kernel;};
// A frame as the summary writes it: image`symbol for the system's libraries (dladdr; the shared cache's locals read <redacted>), and
// image+0xOFFSET for the plugin's own images (NativeAdrenoMetal, libvulkan_freedreno), which are stripped of local symbols on the
// device: tools/gpu/cpuprof.py resolves those offsets against the unstripped copies the build keeps.
inline const Sym &symbol(std::unordered_map<uintptr_t,Sym> &cache,uintptr_t pc) {
 auto it=cache.find(pc);
 if(it!=cache.end())return it->second;
 Dl_info info{};
 Sym s;
 if(dladdr((void *)pc,&info)&&info.dli_fname){
  const char *img=strrchr(info.dli_fname,'/');img=img?img+1:info.dli_fname;
  s.kernel=!strcmp(img,"libsystem_kernel.dylib");
  if(strstr(info.dli_fname,"NativeAdrenoMetal.bundle")||strstr(img,"freedreno")){
   char b[160];snprintf(b,sizeof b,"%s+0x%lx",img,(unsigned long)(pc-(uintptr_t)info.dli_fbase));s.name=b;
  } else s.name=std::string(img)+"`"+(info.dli_sname?info.dli_sname:"?");
 } else {char b[32];snprintf(b,sizeof b,"0x%lx",(unsigned long)pc);s.name=b;s.kernel=false;}
 return cache.emplace(pc,s).first->second;
}

inline void *sampler(void *) {
 const char *pv=getenv("NGPU_CPU_PROFILE_US");
 const useconds_t period=pv?(useconds_t)atoi(pv):1000;
 std::vector<Sample> samples;samples.reserve(1<<16);
 std::unordered_map<uintptr_t,Sym> syms;
 uint64_t last=0,counts[3]={0,0,0};
 auto summary=[&]{
  std::string flag=dir()+"/cpuprof-reset";
  if(access(flag.c_str(),F_OK)==0){unlink(flag.c_str());samples.clear();counts[0]=counts[1]=counts[2]=0;}
  // Each distinct stack once, with its sample counts: enc (a command buffer open), cpu (none open), blocked (in the kernel), innermost first.
  std::unordered_map<std::string,uint64_t[4]> stacks;
  uint64_t blocked=0;
  for(Sample &s:samples){
   if(s.n==0)continue;
   std::string st="t"+std::to_string(s.thread)+";";  // the registered thread, in registration order
   for(int i=0;i<s.n;++i){st+=(i?";":"");st+=symbol(syms,s.pc[i]).name;}
   const bool kernel=symbol(syms,s.pc[0]).kernel;
   if(kernel)blocked++;
   stacks[st][(kernel?2:0)+(s.tag==0?0:1)]++;
  }
  std::string file=dir()+"/"+getprogname()+"-cpuprof.txt",tmp=file+".tmp";
  FILE *f=fopen(tmp.c_str(),"w");
  if(!f)return;
  fprintf(f,"# cpuprof v2 %s pid %d samples %zu enc %llu cpu %llu blocked %llu period %u threads %zu\n",getprogname(),(int)getpid(),
          samples.size(),(unsigned long long)counts[0],(unsigned long long)counts[1],(unsigned long long)blocked,(unsigned)period,threads().size());
  fprintf(f,"# enc,cpu,blocked_enc,blocked_other,stack (innermost first, ';'-separated)\n");
  for(auto &kv:stacks)fprintf(f,"%llu,%llu,%llu,%llu,%s\n",(unsigned long long)kv.second[0],(unsigned long long)kv.second[1],(unsigned long long)kv.second[2],(unsigned long long)kv.second[3],kv.first.c_str());
  fclose(f);
  rename(tmp.c_str(),file.c_str());
 };
 for(;;){
  usleep(period);
  std::vector<Thread> ts;
  {std::lock_guard<std::mutex> g(lock());ts=threads();}
  const int tag=open().load(std::memory_order_relaxed)>0?0:1;
  for(size_t ti=0;ti<ts.size();++ti){
   Thread &t=ts[ti];
   if(samples.size()>=(1u<<20))break;
   Sample s{};s.tag=tag;s.thread=(int)ti;
   if(thread_suspend(t.port)!=KERN_SUCCESS)continue;
   arm_thread_state64_t st;mach_msg_type_number_t count=ARM_THREAD_STATE64_COUNT;
   if(thread_get_state(t.port,ARM_THREAD_STATE64,(thread_state_t)&st,&count)==KERN_SUCCESS){
    s.pc[s.n++]=strip((uintptr_t)arm_thread_state64_get_pc(st));
    uintptr_t lr=strip((uintptr_t)arm_thread_state64_get_lr(st));
    if(lr)s.pc[s.n++]=lr;
    uintptr_t fp=(uintptr_t)arm_thread_state64_get_fp(st);
    while(s.n<kDepth&&fp>=t.lo&&fp+16<=t.hi&&!(fp&7)){
     uintptr_t next=((uintptr_t *)fp)[0],ret=strip(((uintptr_t *)fp)[1]);
     if(!ret)break;
     if(s.n!=2||ret!=s.pc[1])s.pc[s.n++]=ret;  // the first frame's return is often the lr already taken
     if(next<=fp)break;
     fp=next;
    }
   }
   thread_resume(t.port);
   counts[tag]++;
   samples.push_back(s);
  }
  const uint64_t now=clock_gettime_nsec_np(CLOCK_UPTIME_RAW)/1000000;
  if(now-last>=2000){last=now;summary();}
 }
 return nullptr;
}

inline void arm(const char *cache) {
 const char *v=getenv("NGPU_CPU_PROFILE");
 if(!v||!v[0]||!strcmp(v,"0")||!cache||armed())return;
 if(strcmp(v,"1")&&strcmp(v,getprogname()))return;
 dir()=std::string(cache)+"/frametrace";
 mkdir(dir().c_str(),0755);
 armed()=true;
 pthread_t t;
 if(pthread_create(&t,nullptr,sampler,nullptr)!=0){armed()=false;return;}
 pthread_detach(t);
}
}  // namespace cpuprof
#endif
