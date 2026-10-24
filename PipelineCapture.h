// SPDX-License-Identifier: MIT
// The pipeline capture (diagnostics, off by default): with NGPU_CAPTURE in the plugin's gpu-env, the frontend records every library,
// function and pipeline state the process makes, so metal2vulkan's pipeline-census can make the same states on a host
// (metal2vulkan scripts/pipeline-census/capture_manifest.py turns the capture into its manifest).
//
// NGPU_CAPTURE=1 writes under the runtime's cache directory, NGPU_CAPTURE=/abs/dir under that directory, each in a directory of
// its own per process, <progname>-<pid>: capture.jsonl (one JSON object per line) and lib-<sha>.metallib, each library's bytes
// once. Records:
//   {"library":"<sha16>","bytes":N,"file":"lib-<sha16>.metallib"}
//   {"function":"<id>","library":"<sha16>","name":"...","constants":[{"index":I|"name":"...","type":T,"hex":"..."}]}
//   {"render":N,"ok":0|1,"label":"...","vertex":"<id>","fragment":"<id>"|null,"colors":[...],"depth":F,"stencil":F,"samples":S,
//    "topology":T,"rasterization":0|1,"alphaToCoverage":0|1,"vertexDescriptor":{...}}   (command-replay's manifest keys)
//   {"compute":N,"ok":0|1,"label":"...","function":"<id>"}
// A function's id digests its library, name and constants, so every process names a function alike. The libraries are the
// caller's (Apple's own shaders for the render server): pull them into a gitignored directory only, never commit them.
#ifndef NATIVE_METAL_PIPELINE_CAPTURE_H
#define NATIVE_METAL_PIPELINE_CAPTURE_H
#include <CommonCrypto/CommonDigest.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include "bridge/NativeMetalObjects.h"

namespace pcap {
inline std::mutex &lock() {static std::mutex m;return m;}
inline std::string &dir() {static std::string d;return d;}
inline bool on() {return !dir().empty();}
// Called once with the runtime's cache directory (may be null).
inline void begin(const char *cache) {
 const char *v=getenv("NGPU_CAPTURE");
 if(!v || !v[0] || !strcmp(v,"0"))return;
 std::string base=v[0]=='/'?std::string(v):cache?std::string(cache)+"/capture":std::string();
 if(base.empty())return;
 mkdir(base.c_str(),0755);
 char leaf[96];snprintf(leaf,sizeof leaf,"/%s-%d",getprogname(),(int)getpid());
 std::string d=base+leaf;mkdir(d.c_str(),0755);
 if(access(d.c_str(),W_OK)!=0)return;
 dir()=d;
}
inline std::string hexOf(const void *bytes,size_t n) {
 static const char digits[]="0123456789abcdef";std::string s;s.reserve(n*2);
 for(size_t i=0;i<n;i++){const uint8_t b=((const uint8_t *)bytes)[i];s+=digits[b>>4];s+=digits[b&15];}
 return s;
}
inline std::string digest(const void *bytes,size_t n) {
 uint8_t sha[CC_SHA256_DIGEST_LENGTH];CC_SHA256(bytes,(CC_LONG)n,sha);return hexOf(sha,8);
}
inline std::string quote(const char *s) {
 if(!s)return "null";
 std::string out="\"";
 for(const char *p=s;*p;p++) {
  const unsigned char c=(unsigned char)*p;
  if(c=='"'||c=='\\'){out+='\\';out+=(char)c;}
  else if(c<0x20){char e[8];snprintf(e,sizeof e,"\\u%04x",c);out+=e;}
  else out+=(char)c;
 }
 return out+"\"";
}
inline void record(const std::string &line) {
 std::lock_guard<std::mutex> guard(lock());
 if(FILE *f=fopen((dir()+"/capture.jsonl").c_str(),"a")){fwrite(line.data(),1,line.size(),f);fputc('\n',f);fclose(f);}
}
// A library's id; its bytes are written once.
inline std::string library(const void *bytes,size_t n) {
 if(!on() || !bytes || !n)return std::string();
 const std::string id=digest(bytes,n);
 static std::set<std::string> seen;
 {std::lock_guard<std::mutex> guard(lock());if(!seen.insert(id).second)return id;}
 const std::string file="lib-"+id+".metallib";
 if(FILE *f=fopen((dir()+"/"+file).c_str(),"wb")){fwrite(bytes,1,n,f);fclose(f);}
 record("{\"library\":\""+id+"\",\"bytes\":"+std::to_string(n)+",\"file\":\""+file+"\"}");
 return id;
}
// A function's id; its record is written once.
inline std::string function(const std::string &libraryId,const char *name,const NmbConstant *constants,size_t count) {
 if(!on())return std::string();
 std::string text="{\"library\":\""+libraryId+"\",\"name\":"+quote(name)+",\"constants\":[";
 for(size_t i=0;i<count;i++) {
  const NmbConstant &c=constants[i];
  if(i)text+=",";
  text+=c.name?"{\"name\":"+quote(c.name):"{\"index\":"+std::to_string(c.index);
  text+=",\"type\":"+std::to_string(c.type)+",\"hex\":\""+hexOf(c.bytes,(size_t)c.length)+"\"}";
 }
 text+="]}";
 const std::string id="f"+digest(text.data(),text.size());
 static std::set<std::string> seen;
 {std::lock_guard<std::mutex> guard(lock());if(!seen.insert(id).second)return id;}
 record("{\"function\":\""+id+"\","+text.substr(1));
 return id;
}
inline unsigned long next() {static std::atomic<unsigned long> n{0};return n++;}
inline long render(const NmbRenderPipelineDescriptor *p,const std::string &vertex,const std::string &fragment,const char *label,
                   bool alphaToCoverage,bool ok) {
 if(!on())return -1;
 const long number=(long)next();
 std::string s="{\"render\":"+std::to_string(number)+",\"ok\":"+(ok?"1":"0")+",\"label\":"+quote(label)+",\"vertex\":\""+vertex+"\",\"fragment\":";
 s+=fragment.empty()?std::string("null"):"\""+fragment+"\"";
 s+=",\"colors\":[";
 int last=-1;for(int i=0;i<8;i++)if(p->colors[i].format)last=i;
 for(int i=0;i<=last;i++) {
  const NmbColorAttachment &c=p->colors[i];
  if(i)s+=",";
  if(!c.format){s+="null";continue;}
  char e[256];
  snprintf(e,sizeof e,"{\"format\":%llu,\"mask\":%llu,\"blending\":%s,\"srcRGB\":%llu,\"dstRGB\":%llu,\"rgbOp\":%llu,\"srcA\":%llu,\"dstA\":%llu,\"alphaOp\":%llu}",
           (unsigned long long)c.format,(unsigned long long)c.write_mask,c.blending?"true":"false",(unsigned long long)c.source_rgb,
           (unsigned long long)c.destination_rgb,(unsigned long long)c.rgb_operation,(unsigned long long)c.source_alpha,
           (unsigned long long)c.destination_alpha,(unsigned long long)c.alpha_operation);
  s+=e;
 }
 char e[256];
 snprintf(e,sizeof e,"],\"depth\":%llu,\"stencil\":%llu,\"samples\":%llu,\"topology\":%llu,\"rasterization\":%s,\"alphaToCoverage\":%s",
          (unsigned long long)p->depth_format,(unsigned long long)p->stencil_format,(unsigned long long)p->samples,(unsigned long long)p->topology,
          p->rasterization?"true":"false",alphaToCoverage?"true":"false");
 s+=e;
 if(p->vertex_inputs) {
  s+=",\"vertexDescriptor\":{\"attributes\":[";
  bool first=true;std::set<uint64_t> buffers;
  for(int i=0;i<31;i++) {
   const NmbVertexAttribute &a=p->attributes[i];if(!a.format)continue;
   snprintf(e,sizeof e,"%s{\"index\":%d,\"format\":%llu,\"offset\":%llu,\"buffer\":%llu}",first?"":",",i,(unsigned long long)a.format,
            (unsigned long long)a.offset,(unsigned long long)a.buffer);
   s+=e;first=false;buffers.insert(a.buffer);
  }
  s+="],\"layouts\":[";first=true;
  for(uint64_t b:buffers) {
   if(b>=31)continue;
   const NmbVertexLayout &l=p->layouts[b];
   snprintf(e,sizeof e,"%s{\"index\":%llu,\"stride\":%llu,\"step\":%llu,\"rate\":%llu}",first?"":",",(unsigned long long)b,
            (unsigned long long)l.stride,(unsigned long long)l.step_function,(unsigned long long)l.step_rate);
   s+=e;first=false;
  }
  s+="]}";
 }
 record(s+"}");
 return number;
}
inline void compute(const std::string &function,const char *label,bool ok) {
 if(!on())return;
 record("{\"compute\":"+std::to_string(next())+",\"ok\":"+(ok?"1":"0")+",\"label\":"+quote(label)+",\"function\":\""+function+"\"}");
}
// A render encoder's state as its draws key their pipelines: the pass, each stage's bindings by slot ("v"/"f" + "b"/"t"/"s" + index),
// the depth/stencil state and the raster state. A draw's record is written once per distinct state (buffer offsets modulo 256).
struct Bindings {
 std::string pass,depthStencil="null",raster="[0,0,0]";
 long pipeline=-1;
 std::map<std::string,std::string> slots;
};
inline void draw(const Bindings &b,uint64_t primitive,long indexType) {
 if(!on() || b.pipeline<0)return;
 std::string s="\"render\":"+std::to_string(b.pipeline)+",\"primitive\":"+std::to_string(primitive)+",\"indexType\":"+
     (indexType<0?std::string("null"):std::to_string(indexType))+",\"pass\":"+b.pass+",\"depthStencil\":"+b.depthStencil+",\"raster\":"+b.raster+",\"slots\":{";
 bool first=true;
 for(const auto &[k,v]:b.slots){if(v.empty())continue;s+=(first?"\"":",\"")+k+"\":"+v;first=false;}
 s+="}";
 static std::set<std::string> seen;
 const std::string key=digest(s.data(),s.size());
 {std::lock_guard<std::mutex> guard(lock());if(!seen.insert(key).second)return;}
 record("{\"draw\":\""+key+"\","+s+"}");
}
inline std::string textureText(const NmbTextureDescriptor &d,const char *over,uint64_t offset,uint64_t row,bool view) {
 char e[512];
 snprintf(e,sizeof e,"{\"type\":%llu,\"format\":%llu,\"width\":%llu,\"height\":%llu,\"depth\":%llu,\"levels\":%llu,\"samples\":%llu,"
          "\"array\":%llu,\"usage\":%llu,\"options\":%llu,\"over\":%s%s%s,\"offset\":%llu,\"row\":%llu,\"view\":%s}",
          (unsigned long long)d.type,(unsigned long long)d.format,(unsigned long long)d.width,(unsigned long long)d.height,(unsigned long long)d.depth,
          (unsigned long long)d.levels,(unsigned long long)d.samples,(unsigned long long)d.array_length,(unsigned long long)d.usage,
          (unsigned long long)d.options,over?"\"":"",over?over:"null",over?"\"":"",(unsigned long long)offset,(unsigned long long)row,view?"true":"false");
 return e;
}
}  // namespace pcap
#endif
