// SPDX-License-Identifier: MIT
// The pass trace (diagnostics, off by default): per-pass GPU time of this process, aggregated on the device into one small file.
// With NGPU_PASS_TRACE=1 (every Metal process) or NGPU_PASS_TRACE=NAME (the process NAME only, e.g. backboardd) in the plugin's gpu-env, native_bridge_metal_open points Turnip's u_trace (MESA_GPU_TRACES=print_csv) at a pipe
// before the device opens, and a reader thread pairs each start_X/end_X event of a batch (one Vulkan submission) into a pass and adds its
// GPU time to the pass's key. Turnip's CSV never reaches the file system, so its size and the u_trace caveat of NOTES-gpu-userland.md
// (every Metal process writing one growing file) do not apply.
//   key   render passes: rp WxH aN clC/ldL/stS gmem|sysmem (attachment size, count, the clear/load/store bytes per pixel Turnip states), the draw-count bucket;
//         draws (only with TU_GPU_TRACEPOINT=+draw in gpu-env; each draw then writes two timestamps): draw fs=<blake3> vs=<blake3>;
//         nested pass steps (sysmem_clear, gmem_load, ...) and transfers (copy_buffer_to_image, blit, ...): their kind and the size the
//         event states; compute: cs and its group counts
// Every second while passes arrive, the summary is rewritten (tmp + rename) as <cache>/frametrace/<progname>-passes.csv (a fixed name,
// so the host pulls it without knowing the pid; the header names it):
//   # passtrace v1 PROG pid PID batches B swaps S t0 US t1 US events E dropped D
//   key,count,total_ms,median_ms,p95_ms,max_ms,draws_median,draws_max
// swaps counts the scanout swaps of the frame trace (NBTraceSwap), so total_ms/swaps is the pass's GPU time per presented frame.
// A file named passtrace-reset in the same directory (any content) clears the aggregate at the next second and is removed: push it right
// before a measurement (scripts/fajita/test/passtrace.sh does), so the summary covers that measurement only.
// Analysis: tools/gpu/frametrace.py --passes FILE in the fajita board.
#ifndef NATIVE_METAL_PASS_TRACE_H
#define NATIVE_METAL_PASS_TRACE_H
#include <algorithm>
#include <cctype>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <pthread.h>
#include <string>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <vector>

namespace passtrace {
struct Agg {std::vector<float> ms;std::vector<uint32_t> draws;double total=0;};
struct Open {uint64_t ns;std::string key;};
inline std::atomic<uint64_t> &swaps() {static std::atomic<uint64_t> n{0};return n;}
inline bool &armed() {static bool on=false;return on;}
inline uint64_t nowUs() {return clock_gettime_nsec_np(CLOCK_UPTIME_RAW)/1000;}

struct State {
 std::string dir,file;
 std::map<std::string,Agg> agg;
 std::map<std::string,std::vector<Open>> open;  // per batch + kind: started, not yet ended
 uint64_t events=0,dropped=0,t0=0,t1=0,swaps0=0,lastWrite=0;
 std::string lastBatch;uint64_t batches=0;
};
inline State &state() {static State s;return s;}

// "a=1, b=x, " -> the value of name (empty if absent).
inline std::string arg(const char *args,const char *name) {
 const size_t n=strlen(name);
 for(const char *p=args;(p=strstr(p,name));++p){
  if((p==args||p[-1]==' '||p[-1]==',')&&p[n]=='='){const char *v=p+n+1;size_t len=strcspn(v,", \n");return std::string(v,len);}
 }
 return std::string();
}

inline std::string keyFor(const std::string &kind,const char *args) {
 if(kind=="render_pass")
  return "rp "+arg(args,"width")+"x"+arg(args,"height")+" a"+arg(args,"attachment_count")+" cl"+arg(args,"clearCPP")+"/ld"+arg(args,"loadCPP")+
         "/st"+arg(args,"storeCPP");
 if(kind=="compute"||kind=="compute_indirect"){
  std::string k="cs";
  for(const char *f:{"num_groups_x","num_groups_y","num_groups_z"}){std::string v=arg(args,f);if(!v.empty())k+=" "+v;}
  return k;
 }
 if(kind=="draw")  // TU_GPU_TRACEPOINT=+draw: per draw, by its shaders (ir3's binary blake3, as IR3_SHADER_DEBUG's "Native code" lines say)
  return "draw fs="+arg(args,"fs_sha1").substr(0,16)+" vs="+arg(args,"vs_sha1").substr(0,16);
 std::string k=kind;
 std::string w=arg(args,"width"),h=arg(args,"height");
 if(!w.empty())k+=" "+w+"x"+h;
 std::string format=arg(args,"format");  // transfers: copy_buffer_to_image format=b8g8r8a8_unorm, ...
 if(!format.empty())k+=" "+format;
 return k;
}

inline void reset(State &s) {s.agg.clear();s.events=s.dropped=s.batches=0;s.t0=s.t1=0;s.swaps0=swaps().load();}

inline void write(State &s) {
 if(s.file.empty())return;
 std::string tmp=s.file+".tmp";
 FILE *f=fopen(tmp.c_str(),"w");
 if(!f)return;
 fprintf(f,"# passtrace v1 %s pid %d batches %llu swaps %llu t0 %llu t1 %llu events %llu dropped %llu\n",getprogname(),(int)getpid(),
         (unsigned long long)s.batches,(unsigned long long)(swaps().load()-s.swaps0),(unsigned long long)s.t0,(unsigned long long)s.t1,
         (unsigned long long)s.events,(unsigned long long)s.dropped);
 fprintf(f,"key,count,total_ms,median_ms,p95_ms,max_ms,draws_median,draws_max\n");
 std::vector<std::pair<double,const std::string *>> order;
 for(auto &kv:s.agg)order.push_back({kv.second.total,&kv.first});
 std::sort(order.begin(),order.end(),[](auto &a,auto &b){return a.first>b.first;});
 for(auto &o:order){
  Agg &a=s.agg[*o.second];
  std::vector<float> v=a.ms;std::sort(v.begin(),v.end());
  std::vector<uint32_t> d=a.draws;std::sort(d.begin(),d.end());
  auto at=[&](double q){return v.empty()?0.0:(double)v[std::min(v.size()-1,(size_t)(q*(double)v.size()))];};
  fprintf(f,"%s,%zu,%.3f,%.3f,%.3f,%.3f,%u,%u\n",o.second->c_str(),v.size(),a.total,at(0.5),at(0.95),v.empty()?0.0:(double)v.back(),
          d.empty()?0u:d[d.size()/2],d.empty()?0u:d.back());
 }
 fclose(f);
 rename(tmp.c_str(),s.file.c_str());
}

// One CSV line of Turnip's print_csv: frame,batch,time_ns,event,args...
inline void take(State &s,char *line) {
 char *fields[4];char *p=line;
 for(int i=0;i<4;++i){fields[i]=p;p=strchr(p,',');if(!p)return;*p++=0;}
 const char *batch=fields[1],*event=fields[3],*args=p;
 if(!isdigit((unsigned char)fields[2][0]))return;
 const uint64_t ns=strtoull(fields[2],nullptr,10);
 bool start=!strncmp(event,"start_",6),end=!strncmp(event,"end_",4);
 if(!start&&!end)return;
 const std::string kind=event+(start?6:4);
 if(kind=="cmd_buffer"||kind=="draw_ib_sysmem"||kind=="draw_ib_gmem")return;
 if(s.lastBatch!=batch){s.lastBatch=batch;++s.batches;}
 std::string slot=std::string(batch)+"/"+kind;
 ++s.events;
 if(start){s.open[slot].push_back({ns,keyFor(kind,args)});return;}
 auto it=s.open.find(slot);
 if(it==s.open.end()||it->second.empty()){++s.dropped;return;}
 Open o=it->second.back();it->second.pop_back();
 if(it->second.empty())s.open.erase(it);
 if(ns<o.ns||ns-o.ns>2000000000ull){++s.dropped;return;}
 std::string key=o.key;
 uint32_t draws=0;
 if(kind=="render_pass"){
  key+=arg(args,"tiledRender")=="true"?" gmem":" sysmem";
  draws=(uint32_t)strtoul(arg(args,"drawCount").c_str(),nullptr,10);
  key+=draws<=1?" d0-1":draws<10?" d2-9":draws<100?" d10-99":" d100+";  // a full-screen composite apart from a small partial update
 }
 Agg &a=s.agg[key];
 const float ms=(float)((double)(ns-o.ns)/1e6);
 if(a.ms.size()<200000){a.ms.push_back(ms);a.draws.push_back(draws);}
 a.total+=ms;
 const uint64_t us=nowUs();
 if(!s.t0)s.t0=us;
 s.t1=us;
}

inline void *reader(void *arg) {
 const int fd=(int)(intptr_t)arg;
 State &s=state();
 std::string buf;char chunk[16384];
 for(;;){
  ssize_t n=read(fd,chunk,sizeof chunk);
  if(n<=0)break;
  buf.append(chunk,(size_t)n);
  size_t from=0;
  for(size_t nl;(nl=buf.find('\n',from))!=std::string::npos;from=nl+1){buf[nl]=0;take(s,&buf[from]);}
  buf.erase(0,from);
  if(s.open.size()>4096)s.open.clear();  // events whose end never came (a lost batch): do not grow
  const uint64_t us=nowUs();
  if(us-s.lastWrite>1000000){
   s.lastWrite=us;
   std::string flag=s.dir+"/passtrace-reset";
   if(access(flag.c_str(),F_OK)==0){unlink(flag.c_str());reset(s);}
   write(s);
  }
 }
 return nullptr;
}

// Before the device opens (Turnip reads MESA_GPU_TRACES and MESA_GPU_TRACEFILE once, at its first u_trace context).
inline void arm(const char *cache) {
 const char *v=getenv("NGPU_PASS_TRACE");
 if(!v||!v[0]||!strcmp(v,"0")||!cache||armed())return;
 if(strcmp(v,"1")&&strcmp(v,getprogname()))return;  // NGPU_PASS_TRACE=NAME: that process only
 State &s=state();
 s.dir=std::string(cache)+"/frametrace";
 mkdir(s.dir.c_str(),0755);
 char leaf[96];snprintf(leaf,sizeof leaf,"/%s-passes.csv",getprogname());
 s.file=s.dir+leaf;
 int fds[2];
 if(pipe(fds)!=0)return;
 char path[32];snprintf(path,sizeof path,"/dev/fd/%d",fds[1]);
 setenv("MESA_GPU_TRACES","print_csv",1);
 setenv("MESA_GPU_TRACEFILE",path,1);
 pthread_t t;
 if(pthread_create(&t,nullptr,reader,(void *)(intptr_t)fds[0])!=0){close(fds[0]);close(fds[1]);unsetenv("MESA_GPU_TRACES");return;}
 pthread_detach(t);
 armed()=true;
}
}  // namespace passtrace
#endif
