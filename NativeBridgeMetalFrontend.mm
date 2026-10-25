// SPDX-License-Identifier: MIT
// Native Metal API frontend over the bridge (bridge/NativeMetalObjects.h): the metal2vulkan runtime on Turnip, on the Adreno 740 through
// NativeAdreno. Same shape as the offline frontend: the device is a dynamically allocated subclass of the target's own _MTLDevice that
// only adds this file's implementations; every other object is an own class answering the Metal protocol methods it implements.
// Every result comes from the runtime and the GPU: a refused call returns nil/NO with the runtime's reason, an encoder call the runtime
// refuses fails its command buffer (status Error, handlers run) instead of being dropped. A selector this frontend does not implement is
// reported through the missing-selector hook before the usual exception, so a hardware run names the gap.
// Shared events work within this process (no cross-process handle yet). Not yet implemented here: heaps, argument buffers, indirect commands, drawables/presentation.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <IOSurface/IOSurfaceRef.h>
#import <objc/runtime.h>
#include "TargetBuild.h"
#include <algorithm>
#include <atomic>
#include <mutex>
#include <pthread.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <sys/mman.h>
#include <CommonCrypto/CommonDigest.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <mach/mach.h>
#include "bridge/NativeMetalObjects.h"
#include "NativeBridgeMetalFrontend.h"
#include "PipelineCapture.h"
#include "FrameTrace.h"
#include "PassTrace.h"
#include "CpuProfile.h"

extern "C" void sys_dcache_flush(void *start, size_t length);

static NmbDevice *gDevice;
static id gDeviceObject;
static uint64_t gRegistryID;
static NSString *gName;
static void (*gMissing)(const char *, const char *);

// What this device has been asked and refused, for the plugin's status line (native_bridge_metal_status): commits, render/compute pipelines made,
// refusals, and the last two refusals' texts. Counters only; nothing here changes a result.
static std::atomic<uint64_t> gCommits,gCommitFailures,gPipelines,gProblems,gCompleted;
// What the render server actually asks of the GPU (int-141: 59 commits in 200 s, and a desktop that behaves like CPU compositing): render passes,
// draws, the pixels of their first colour attachment, compute dispatches, blits and the largest pass (pixels).
static std::atomic<uint64_t> gPasses,gDraws,gPassPixels,gDispatches,gBlits,gBiggestPass,gFullScreenPasses;
// Where the time goes on the CPU side of the bridge: pipeline, library and function creation (count, total and slowest, in microseconds), and the
// last commits with what each held and how long the GPU took (commit to completion).
static uint64_t nowUs() {return clock_gettime_nsec_np(CLOCK_UPTIME_RAW)/1000;}
// traceKind >= 0: each one is also a frame-trace P line (FrameTrace.h).
struct Timed {std::atomic<uint64_t> count{0},total{0},slowest{0};int traceKind;
 explicit Timed(int kind=-1):traceKind(kind){}
 void add(uint64_t began){const uint64_t took=nowUs()-began;count++;total+=took;uint64_t seen=slowest.load();while(took>seen && !slowest.compare_exchange_weak(seen,took)){}
  if(ftrace::on() && traceKind>=0)ftrace::linef("P,%llu,%llu,%d",(unsigned long long)began,(unsigned long long)took,traceKind);}};
static Timed gTimePipeline(0),gTimeLibrary(2),gTimeFunction(3),gTimeEncodeToCommit;
static Timed gTimeWaitScheduled,gTimeWaitCompleted;
static Timed gTimeResourceWait;
static std::atomic<uint64_t> gResourceChecks,gResourceSkipped;
static Timed gTimeSurfaceFlush;
static std::atomic<uint64_t> gSurfaceFlushBytes,gSurfaceFlushDuplicates;
struct EncodedWork {uint32_t passes,draws,dispatches,blits;uint64_t pixels;};
struct CommitRecord {uint64_t committedUs=0,callbackUs=0,ageUs=0;int64_t scheduledUs=-1;uint32_t index=0;EncodedWork work={};};
static CommitRecord gCommitRing[32];
static std::mutex gCommitLock;
static std::mutex gProblemLock;
static char gLastProblems[2][200];
static void journal(const std::string &line);
// Every problem also goes to the pipeline journal as a P line (the render server can die before the status line streams it).
static void noteProblem(NSString *what) {
 gProblems++;
 journal(std::string("P ")+(what.UTF8String?:"?"));
 std::lock_guard<std::mutex> lock(gProblemLock);
 memcpy(gLastProblems[1],gLastProblems[0],sizeof gLastProblems[0]);
 strlcpy(gLastProblems[0],what.UTF8String?:"?",sizeof gLastProblems[0]);
}
#define NBLOG(...) do{NSString *nbText=[NSString stringWithFormat:__VA_ARGS__];NSLog(@"%@",nbText);noteProblem(nbText);}while(0)
// The pixel formats textures were asked for (first eight distinct), with how many were refused: native_bridge_metal_formats.
static uint32_t gFormats[8];static unsigned gFormatCount;static std::atomic<uint64_t> gTextureRefusals;
// Textures over IOSurfaces: made, refused, and how many of the requests named a planar surface.
static std::atomic<uint64_t> gSurfaceShared{0};
static std::atomic<uint64_t> gSurfaceTextures,gSurfaceRefusals,gSurfacePlanar;
// The last refused surface texture: the check that refused it (1 plane, 2 base, 3 plane range, 4 import, 5 texture), the surface's format,
// planes, requested plane, plane offset, plane row bytes and allocation, and the descriptor's size and pixel format.
struct SurfaceRefusal {uint32_t step,format,planes,plane,pixelFormat;uint64_t offset,row,allocation,width,height;};
static SurfaceRefusal gSurfaceLastRefusal;
static void refuseSurface(uint32_t step,IOSurfaceRef surface,MTLTextureDescriptor *d,NSUInteger plane,uint64_t offset,uint64_t row) {
 gSurfaceRefusals++;
 SurfaceRefusal r={step,surface?IOSurfaceGetPixelFormat(surface):0,surface?(uint32_t)IOSurfaceGetPlaneCount(surface):0,(uint32_t)plane,
                   d?(uint32_t)d.pixelFormat:0,offset,row,surface?IOSurfaceGetAllocSize(surface):0,d?d.width:0,d?d.height:0};
 std::lock_guard<std::mutex> lock(gProblemLock);gSurfaceLastRefusal=r;
}
static void noteFormat(uint32_t format,bool made) {
 std::lock_guard<std::mutex> lock(gProblemLock);
 if(!made)gTextureRefusals++;
 for(unsigned i=0;i<gFormatCount;i++)if(gFormats[i]==format)return;
 if(gFormatCount<8)gFormats[gFormatCount++]=format;
}
static NSError *nbError(NSString *why) {
 return [NSError errorWithDomain:@"org.blu-sharky.native-metal" code:-1 userInfo:@{NSLocalizedDescriptionKey:why?:@"refused"}];
}
static NSString *lastError() {return [NSString stringWithUTF8String:nmb_last_error()];}
// The render pipeline journal: one line per render pipeline creation (when, on which thread, how long, the outcome, both functions with
// every constant value, the attachments, blending and vertex descriptor) appended to pipelines.log in the runtime's cache directory, which
// the board log exports with the caches (CacheExport.h): every pipeline a boot made, or was refused, can be made again on the host.
static std::mutex gJournalLock;
static std::string gJournalPath;
static uint64_t gJournalBegan;
static void journal(const std::string &line) {
 if(gJournalPath.empty())return;
 std::lock_guard<std::mutex> lock(gJournalLock);
 if(FILE *f=fopen(gJournalPath.c_str(),"a")){fwrite(line.data(),1,line.size(),f);fputc('\n',f);fclose(f);}
}

@interface NBObject : NSObject
@property(atomic,copy) NSString *label;
@end
@implementation NBObject
- (id<MTLDevice>)device {return gDeviceObject;}
- (void)doesNotRecognizeSelector:(SEL)selector {
 journal(std::string("M ")+class_getName(object_getClass(self))+" "+sel_getName(selector));
 if(gMissing)gMissing(class_getName(object_getClass(self)),sel_getName(selector));
 noteProblem([NSString stringWithFormat:@"missing %s %s",class_getName(object_getClass(self)),sel_getName(selector)]);
 [super doesNotRecognizeSelector:selector];
}
@end

// ---- resources ----
// The wait takes a strong snapshot of already-committed buffers; new commits require the
// caller's usual resource synchronization. Storage dependency checks are in the runtime.
static NSMutableSet *gInFlight;
static void waitForResource(id resource);
// Resource SPI the runtime has no state for: no protected content (Metal's default options), and no owner transfer
// (memory attribution is the kernel driver's, by the importing task), refused as not supported.
#define NB_RESOURCE_SPI \
- (uint64_t)protectionOptions {return 0;} \
- (id<MTLHeap>)heap {return _parentHeap;} \
- (BOOL)isAliasable {return NO;} \
- (int)setOwnerWithIdentity:(unsigned)identity {(void)identity;return 46; /* KERN_NOT_SUPPORTED */} \
- (void)waitUntilComplete {waitForResource(self);}
@interface NBBuffer : NBObject
@property(nonatomic) NmbBuffer *handle;
@property(nonatomic) MTLResourceOptions options;
@property(nonatomic) void *wrapped;
@property(nonatomic,strong) id<MTLHeap> parentHeap;   // the heap that made the buffer, nil otherwise
@property(nonatomic) NSUInteger heapOffset;
// Imports (no-copy buffers and IOSurface pages): the CPU range the commit-time cleaning covers (flushImports), the surface whose seed
// says whether its pixels changed since the last clean, and the presenter's opt-out for frames only the GPU wrote.
@property(nonatomic) IOSurfaceRef importSurface;
@property(nonatomic) NSUInteger importBytes;
@property(nonatomic) BOOL importNoFlush,importFlushed;
@property(nonatomic) uint32_t importSeed;
@property(nonatomic) BOOL hostCounted;   // counted in gHostLive (noteHostRange)
// NGPU_HOST_COPY=1: a no-copy buffer stood in by a GPU-owned copy of the caller's bytes, refreshed at every commit; the caller's
// deallocator runs when the buffer goes.
@property(nonatomic) void *copySource;
@property(atomic) BOOL contentsTouched;   // contents was asked for: the CPU may write the bytes the runtime holds
@property(nonatomic) NSUInteger copyBytes;
@property(nonatomic,copy) void (^copyDeallocator)(void *,NSUInteger);
@end
@interface NBTexture : NBObject
@property(nonatomic) NmbTexture *handle;
@property(nonatomic) NmbTextureDescriptor descriptor;
@property(nonatomic,strong) NBBuffer *buffer;
@property(nonatomic) IOSurfaceRef surface;
@property(nonatomic) NSUInteger offset,rowBytes;
@property(nonatomic,strong) id<MTLTexture> parent;
@property(nonatomic,strong) id<MTLHeap> parentHeap;
@property(nonatomic) NSUInteger heapOffset;
- (id<MTLTexture>)view:(NmbTexture *)handle;
@property(nonatomic) uint64_t cpuGeneration,cpuCheckedIn;   // CPU-written texture over a buffer: its stated generation, last command buffer checked
@end
// A no-copy import keeps its pages until the runtime frees it, which can be after the buffer object goes (a command buffer still using
// it). The owner's release (Metal's deallocator, an IOSurface's mapping) waits for that, as Metal's does: run earlier, the owner could
// unmap or reuse pages the GPU still reads or writes.
static std::atomic<uint64_t> gNoCopyAll,gNoCopyMade,gNoCopyReleased,gSurfaceImportsMade,gSurfaceImportsReleased;
static void releaseBlock(void *context,void *pointer,uint64_t length) {
 gNoCopyReleased++;
 void (^deallocator)(void *,NSUInteger)=(__bridge_transfer void (^)(void *,NSUInteger))context;deallocator(pointer,(NSUInteger)length);
}
static void releaseSurface(void *context,void *,uint64_t) {gSurfaceImportsReleased++;CFRelease((IOSurfaceRef)context);}
static void hostBufferGone(void *base);
// Live objects (leakwatch.sh): buffers, the copy-mode copies among them, textures.
static std::atomic<int64_t> gLiveBuffers,gLiveCopies,gLiveTextures;
@implementation NBBuffer
- (instancetype)init {if((self=[super init]))gLiveBuffers++;return self;}
- (void)dealloc {
 gLiveBuffers--;if(_copySource)gLiveCopies--;
 nmb_buffer_release(_handle);
 if(_hostCounted)hostBufferGone(_wrapped);
 if(_copyDeallocator)_copyDeallocator(_copySource,_copyBytes);
}
- (NSUInteger)length {return nmb_buffer_length(_handle);}
- (void *)contents {_contentsTouched=YES;return nmb_buffer_contents(_handle);}
- (uint64_t)gpuAddress {return nmb_buffer_gpu_address(_handle);}
- (MTLResourceOptions)resourceOptions {return _options;}
- (MTLStorageMode)storageMode {return (MTLStorageMode)((_options&MTLResourceStorageModeMask)>>MTLResourceStorageModeShift);}
- (MTLCPUCacheMode)cpuCacheMode {return (MTLCPUCacheMode)((_options&MTLResourceCPUCacheModeMask)>>MTLResourceCPUCacheModeShift);}
- (MTLHazardTrackingMode)hazardTrackingMode {return (MTLHazardTrackingMode)((_options&MTLResourceHazardTrackingModeMask)>>MTLResourceHazardTrackingModeShift);}
- (void)didModifyRange:(NSRange)range {(void)range;}
- (MTLPurgeableState)setPurgeableState:(MTLPurgeableState)state {
 const uint64_t answer=nmb_buffer_set_purgeable_state(_handle,state);
 if(!answer)NBLOG(@"native-metal: buffer setPurgeableState refused: %@",lastError());
 return (MTLPurgeableState)answer;
}
- (NSUInteger)allocatedSize {return nmb_buffer_length(_handle);}
- (IOSurfaceRef)iosurface {return nullptr;}   // a buffer is never the backing of a surface here (surfaces back textures)
NB_RESOURCE_SPI
@end
// Reuse only an import still held by a texture/view or pending command. Weak
// values never extend a GPU allocation's lifetime. The strong buffer snapshot
// and its retained surface prevent object/address reuse during a hit. A new
// mapping, allocation identity or size needs a separate import; ordinary
// caller-owned no-copy buffers do not enter this cache.
static NSMapTable<NSValue *,NBBuffer *> *gSurfaceImports;
static std::mutex gSurfaceImportLock;
// Every live import of this process, for the commit-time cleaning (flushImports). Weak: an import leaves when its buffer object goes.
static std::mutex gImportLock;
static NSHashTable *gImports;
static std::atomic<uint64_t> gImportFlushes,gImportFlushBytes,gImportSeedSkips;
static Timed gTimeImportFlush;
static void registerImport(NBBuffer *buffer,IOSurfaceRef surface,void *base,size_t bytes) {
 buffer.wrapped=base;buffer.importSurface=surface;buffer.importBytes=bytes;
 std::lock_guard<std::mutex> lock(gImportLock);
 if(!gImports)gImports=[NSHashTable weakObjectsHashTable];
 [gImports addObject:buffer];
}
// The kernel does not maintain read-only imports per submission on a board that sets fajita-gpu-ro-skip (the ABI's contract: the
// importer cleans its caches before GPU use). Before a command runs, every live import of this process is cleaned once its CPU range may
// hold new data: a surface when its seed moved (CPU writers lock and unlock it, which advances the seed), a plain no-copy buffer always,
// unless the presenter marked it as written only by the GPU. Cleaning a range nothing dirtied costs as much as a dirty one, so the seed
// is what saves the work (a static wallpaper is 11 MiB).
static std::atomic<uint64_t> gHostCopies,gHostCopyBytes;
static void flushImports() {
 NSArray *imports;
 {std::lock_guard<std::mutex> lock(gImportLock);imports=gImports.allObjects;}
 for(NBBuffer *buffer in imports) {
  if(buffer.copySource) {   // NGPU_HOST_COPY=2: the caller's current bytes into the GPU's copy at every commit (=1: at creation only)
   static const bool everyCommit=[]{const char *v=getenv("NGPU_HOST_COPY");return v && v[0]=='2';}();
   if(everyCommit)
    if(void *into=nmb_buffer_contents(buffer.handle)){const uint64_t began=nowUs();memcpy(into,buffer.copySource,buffer.copyBytes);gTimeImportFlush.add(began);gHostCopies++;gHostCopyBytes+=buffer.copyBytes;}
   continue;
  }
  if(buffer.importNoFlush || !buffer.wrapped || !buffer.importBytes)continue;
  uint32_t seed=0;
  if(buffer.importSurface) {
   seed=IOSurfaceGetSeed(buffer.importSurface);
   if(buffer.importFlushed && seed==buffer.importSeed){gImportSeedSkips++;continue;}
  }
  const uint64_t began=nowUs();
  sys_dcache_flush(buffer.wrapped,buffer.importBytes);
  gTimeImportFlush.add(began);gImportFlushes++;gImportFlushBytes+=buffer.importBytes;
  buffer.importSeed=seed;buffer.importFlushed=YES;
 }
}
// The presenter's frame source: CoreAnimation's GPU-rendered surface, which no CPU writes; cleaning it would cost a frame's size each time.
extern "C" void NBImportNoFlush(id buffer) {
 if([(id)buffer isKindOfClass:[NBBuffer class]])((NBBuffer *)buffer).importNoFlush=YES;
}

// Same address, new pages: Core Animation frees a backing store (vm_deallocate) once its no-copy buffer is released while the runtime may
// still hold that buffer's import, and maps a new one at the same address later. The runtime joins a new no-copy buffer over a live
// import of the same bytes, which would leave it over the old pages (fajita: wrong icons and labels while Settings scrolled; the kernel's
// import check counted pages REPLACED under live imports). A new buffer over a range seen before, with no buffer over it alive now,
// reports the range remapped first, so it imports the pages mapped there now; while one is alive (two textures of one IOSurface, a
// buffer the owner still holds) it keeps the runtime's join. NGPU_HOST_REMAP=0 turns this off.
static std::mutex gHostRangeLock;
static std::unordered_map<uintptr_t,std::pair<uint64_t,uint32_t>> gHostRanges;   // base -> (length, VM object id)
static std::atomic<uint64_t> gHostRemaps;
static uint32_t hostObject(void *base) {
 vm_address_t address=(vm_address_t)base;vm_size_t size=0;natural_t depth=64;
 vm_region_submap_info_data_64_t info={};mach_msg_type_number_t count=VM_REGION_SUBMAP_INFO_COUNT_64;
 if(vm_region_recurse_64(mach_task_self(),&address,&size,&depth,(vm_region_recurse_info_t)&info,&count)!=KERN_SUCCESS)return 0;
 if(address>(vm_address_t)base)return 0;   // nothing mapped at base
 return info.object_id;
}
// Live no-copy buffers (plain and IOSurface) per base address: a range may join an earlier import only while a buffer over it is still
// alive (its owner cannot have freed the memory then); once every buffer over it is gone, the owner may have unmapped and remapped it.
static std::unordered_map<uintptr_t,uint32_t> gHostLive;
static void hostBufferGone(void *base) {
 std::lock_guard<std::mutex> lock(gHostRangeLock);
 auto it=gHostLive.find((uintptr_t)base);
 if(it!=gHostLive.end() && it->second && !--it->second)gHostLive.erase(it);
}
static void noteHostRange(void *base,size_t length) {
 static const bool off=[]{const char *v=getenv("NGPU_HOST_REMAP");return v && v[0]=='0';}();
 if(off || !base || !length)return;
 const uint32_t object=hostObject(base);
 bool remapped=false;
 {
  std::lock_guard<std::mutex> lock(gHostRangeLock);
  auto it=gHostRanges.find((uintptr_t)base);
  const bool live=gHostLive.count((uintptr_t)base)!=0;
  // Seen before, and either no buffer over it is alive now, or the VM object there changed (the owner unmapped it and mapped other
  // memory: its old pages are gone for every buffer over it).
  if(it!=gHostRanges.end() && (!live || it->second.second!=object || it->second.first!=length))remapped=true;
  gHostRanges[(uintptr_t)base]={length,object};
 }
 if(remapped){gHostRemaps++;nmb_device_host_range_remapped(gDevice,base,length);}
}
static void countHostBuffer(NBBuffer *buffer,void *base) {
 std::lock_guard<std::mutex> lock(gHostRangeLock);
 ++gHostLive[(uintptr_t)base];buffer.hostCounted=YES;
}
static NBBuffer *surfaceBuffer(IOSurfaceRef surface,void *base,size_t bytes) {
 struct Key {uint64_t identity,address,bytes;};
 const Key value={IOSurfaceGetID(surface),(uintptr_t)base,bytes};
 NSValue *key=value.identity?[NSValue valueWithBytes:&value objCType:@encode(Key)]:nil;
 std::lock_guard<std::mutex> lock(gSurfaceImportLock);
 if(!gSurfaceImports)gSurfaceImports=[NSMapTable strongToWeakObjectsMapTable];
 // NGPU_SURFACE_CACHE=0 (the plugin's gpu-env, diagnostics): every surface texture imports its pages anew.
 static const bool noCache=[]{const char *v=getenv("NGPU_SURFACE_CACHE");return v && v[0]=='0';}();
 NBBuffer *buffer=key && !noCache?[gSurfaceImports objectForKey:key]:nil;
 if(buffer)return buffer;
 // The surface keeps its CPU mapping (the imported pages) until the import is freed.
 noteHostRange(base,bytes);
 NmbBuffer *handle=nmb_buffer_new_no_copy_releasing(gDevice,base,bytes,MTLResourceStorageModeShared,releaseSurface,(void *)CFRetain(surface));
 if(!handle){CFRelease(surface);NBLOG(@"native-metal: surface import refused: %@",lastError());return nil;}
 gSurfaceImportsMade++;buffer=[NBBuffer new];buffer.handle=handle;buffer.options=MTLResourceStorageModeShared;
 registerImport(buffer,surface,base,bytes);
 countHostBuffer(buffer,base);
 if(key) {
  // Bound metadata without retaining old frames. Evicting an association is
  // harmless: its owners still retain the actual import until their own end.
  if(gSurfaceImports.count>=256) {
   for(NSValue *old in gSurfaceImports.keyEnumerator.allObjects)
    if(![gSurfaceImports objectForKey:old])[gSurfaceImports removeObjectForKey:old];
   if(gSurfaceImports.count>=256)[gSurfaceImports removeObjectForKey:gSurfaceImports.keyEnumerator.nextObject];
  }
  [gSurfaceImports setObject:buffer forKey:key];
 }
 return buffer;
}
// Extended-range 10-bit formats (BGRA10_XR, BGR10_XR and their sRGB forms, 550-555) on the render server's own textures and pipelines
// are RGBA16Float: the translator stores XR as R32G32_UINT and packs, blends and unpacks it in each fragment function (fajita census:
// fixed_frag_lph_cph at 1,062-1,184 IR3 instructions); RGBA16Float holds the same extended range and the A630 blends it in hardware.
// Textures over IOSurfaces or buffers keep their format (their bytes are XR). NGPU_XR_AS_F16=0 keeps XR everywhere.
static uint64_t xrFormat(uint64_t format) {
 static const bool on=[]{const char *v=getenv("NGPU_XR_AS_F16");return !(v && v[0]=='0');}();
 return on && format>=550 && format<=555 ? (uint64_t)MTLPixelFormatRGBA16Float : format;
}
static std::atomic<uint64_t> gXrTextures,gXrPipelines;
static bool textureDescriptor(MTLTextureDescriptor *d,NmbTextureDescriptor *out) {
 if(!d)return false;
 *out={(uint64_t)d.textureType,(uint64_t)d.pixelFormat,d.width,d.height,d.depth,d.mipmapLevelCount,d.sampleCount,d.arrayLength,(uint64_t)d.usage,(uint64_t)d.resourceOptions};
 return true;
}
@implementation NBTexture
- (instancetype)init {if((self=[super init]))gLiveTextures++;return self;}
- (void)dealloc {gLiveTextures--;nmb_texture_release(_handle);if(_surface)CFRelease(_surface);}
- (MTLTextureType)textureType {return (MTLTextureType)_descriptor.type;}
- (MTLPixelFormat)pixelFormat {return (MTLPixelFormat)_descriptor.format;}
- (NSUInteger)width {return _descriptor.width;}
- (NSUInteger)height {return _descriptor.height;}
- (NSUInteger)depth {return _descriptor.depth;}
- (NSUInteger)mipmapLevelCount {return _descriptor.levels;}
- (NSUInteger)sampleCount {return _descriptor.samples;}
- (NSUInteger)arrayLength {return _descriptor.array_length;}
- (MTLTextureUsage)usage {return (MTLTextureUsage)_descriptor.usage;}
- (MTLResourceOptions)resourceOptions {return (MTLResourceOptions)_descriptor.options;}
- (MTLStorageMode)storageMode {return (MTLStorageMode)((_descriptor.options&MTLResourceStorageModeMask)>>MTLResourceStorageModeShift);}
- (MTLCPUCacheMode)cpuCacheMode {return (MTLCPUCacheMode)((_descriptor.options&MTLResourceCPUCacheModeMask)>>MTLResourceCPUCacheModeShift);}
- (BOOL)isFramebufferOnly {return NO;}
- (IOSurfaceRef)iosurface {return _surface;}
- (NSUInteger)iosurfacePlane {return 0;}
- (id<MTLBuffer>)buffer {return _surface?nil:(id<MTLBuffer>)_buffer;}
- (NSUInteger)bufferOffset {return _offset;}
- (NSUInteger)bufferBytesPerRow {return _rowBytes;}
- (id<MTLTexture>)parentTexture {return _parent;}
- (NSUInteger)rotation {return 0;}
- (MTLPurgeableState)setPurgeableState:(MTLPurgeableState)state {
 const uint64_t answer=nmb_texture_set_purgeable_state(_handle,state);
 if(!answer)NBLOG(@"native-metal: texture setPurgeableState refused: %@",lastError());
 return (MTLPurgeableState)answer;
}
NB_RESOURCE_SPI
- (id<MTLTexture>)view:(NmbTexture *)handle {
 if(!handle){NBLOG(@"native-metal: texture view refused: %@",lastError());return nil;}
 NBTexture *view=[NBTexture new];view.handle=handle;NmbTextureDescriptor descriptor;nmb_texture_describe(handle,&descriptor);view.descriptor=descriptor;
 view.parent=(id<MTLTexture>)self;view.buffer=_buffer;view.offset=_offset;view.rowBytes=_rowBytes;
 if(_surface)view.surface=(IOSurfaceRef)CFRetain(_surface);
 return (id<MTLTexture>)view;
}
- (id<MTLTexture>)newTextureViewWithPixelFormat:(MTLPixelFormat)format {
 return [self view:nmb_texture_new_view(_handle,format,0,nullptr,nullptr,true)];
}
- (id<MTLTexture>)newTextureViewWithPixelFormat:(MTLPixelFormat)format textureType:(MTLTextureType)type levels:(NSRange)levels slices:(NSRange)slices {
 const uint64_t l[2]={levels.location,levels.length},s[2]={slices.location,slices.length};
 return [self view:nmb_texture_new_view(_handle,format,type,l,s,false)];
}
- (void)replaceRegion:(MTLRegion)r mipmapLevel:(NSUInteger)level slice:(NSUInteger)slice withBytes:(const void *)bytes bytesPerRow:(NSUInteger)row bytesPerImage:(NSUInteger)image {
 const uint64_t region[6]={r.origin.x,r.origin.y,r.origin.z,r.size.width,r.size.height,r.size.depth};
 const uint64_t length=(r.size.depth>1?image*(r.size.depth-1):0)+row*r.size.height;
 if(!nmb_texture_replace(_handle,region,level,slice,bytes,row,image,length))NBLOG(@"native-metal: replaceRegion refused: %@",lastError());
}
- (void)replaceRegion:(MTLRegion)r mipmapLevel:(NSUInteger)level withBytes:(const void *)bytes bytesPerRow:(NSUInteger)row {
 [self replaceRegion:r mipmapLevel:level slice:0 withBytes:bytes bytesPerRow:row bytesPerImage:row*r.size.height];
}
- (void)getBytes:(void *)bytes bytesPerRow:(NSUInteger)row bytesPerImage:(NSUInteger)image fromRegion:(MTLRegion)r mipmapLevel:(NSUInteger)level slice:(NSUInteger)slice {
 const uint64_t region[6]={r.origin.x,r.origin.y,r.origin.z,r.size.width,r.size.height,r.size.depth};
 const uint64_t length=(r.size.depth>1?image*(r.size.depth-1):0)+row*r.size.height;
 if(!nmb_texture_get_bytes(_handle,bytes,row,image,length,region,level,slice))NBLOG(@"native-metal: getBytes refused: %@",lastError());
}
- (void)getBytes:(void *)bytes bytesPerRow:(NSUInteger)row fromRegion:(MTLRegion)r mipmapLevel:(NSUInteger)level {
 [self getBytes:bytes bytesPerRow:row bytesPerImage:row*r.size.height fromRegion:r mipmapLevel:level slice:0];
}
@end
// A command can use many independent imports/views of one IOSurface. Their
// owners keep the mapping live through this phase. Maintain each exact CPU
// range once; no address identity or decision is carried into another phase
// or command, and distinct/partially overlapping ranges retain their calls.
static void flushSurfaces(NSMutableSet *surfaces) {
 if(!surfaces.count)return;
 std::vector<std::pair<uintptr_t,size_t>> ranges;
 ranges.reserve(surfaces.count);
 for(NBTexture *texture in surfaces)if(texture.surface)
  ranges.emplace_back(reinterpret_cast<uintptr_t>(IOSurfaceGetBaseAddress(texture.surface)),IOSurfaceGetAllocSize(texture.surface));
 std::sort(ranges.begin(),ranges.end());
 const auto last=std::unique(ranges.begin(),ranges.end());
 gSurfaceFlushDuplicates+=(uint64_t)(ranges.end()-last);
 for(auto it=ranges.begin();it!=last;++it) {
  const uint64_t began=nowUs();
  sys_dcache_flush(reinterpret_cast<void *>(it->first),it->second);
  gSurfaceFlushBytes+=it->second;gTimeSurfaceFlush.add(began);
 }
}
// NGPU_TEXTURE_LOG=1 (gpu-env, diagnostics): a "T" journal line (pipelines.log) per texture over a heap's or a buffer's bytes, with its
// size, format, usage, the bytes' storage mode and row pitch: what M2V_SHADOW_LOG's shadows (logged by extent) are.
static void noteTextureMade(const char *kind,MTLTextureDescriptor *d,unsigned long storage,unsigned long row) {
 static const bool on=[]{const char *v=getenv("NGPU_TEXTURE_LOG");return v && v[0]=='1';}();
 if(!on || !d)return;
 char line[160];snprintf(line,sizeof line,"T %s %lux%lu format %lu usage %lx storage %lu row %lu",kind,(unsigned long)d.width,(unsigned long)d.height,
  (unsigned long)d.pixelFormat,(unsigned long)d.usage,storage,row);journal(line);
}
static NBTexture *textureOnBuffer(NBBuffer *buffer,MTLTextureDescriptor *d,NSUInteger offset,NSUInteger row,bool surface) {
 NmbTextureDescriptor descriptor;if(!buffer || !textureDescriptor(d,&descriptor))return nil;
 noteTextureMade(surface?"surface":"buffer",d,(unsigned long)((buffer.options&MTLResourceStorageModeMask)>>MTLResourceStorageModeShift),row);
 NmbTexture *handle=nmb_texture_new_on_buffer(buffer.handle,&descriptor,offset,row,surface);
 if(!handle){NBLOG(@"native-metal: texture over buffer refused: %@",lastError());return nil;}
 NBTexture *texture=[NBTexture new];texture.handle=handle;nmb_texture_describe(handle,&descriptor);texture.descriptor=descriptor;
 texture.buffer=buffer;texture.offset=offset;texture.rowBytes=row;return texture;
}
// A texture over a two-plane surface: the runtime reads plane 1 from the surface's own layout (two-plane RGB and YCbCr formats).
static NBTexture *textureOnSurface(NBBuffer *buffer,MTLTextureDescriptor *d,IOSurfaceRef surface,NSUInteger plane,NSUInteger offset,NSUInteger row) {
 NmbTextureDescriptor descriptor;if(!buffer || !textureDescriptor(d,&descriptor))return nil;
 const uint64_t base=(uintptr_t)IOSurfaceGetBaseAddress(surface),chroma=(uintptr_t)IOSurfaceGetBaseAddressOfPlane(surface,1);
 noteTextureMade("surface2",d,0,row);
 NmbTexture *handle=nmb_texture_new_on_surface(buffer.handle,&descriptor,offset,row,IOSurfaceGetPixelFormat(surface),plane,
  IOSurfaceGetWidth(surface),IOSurfaceGetHeight(surface),chroma>=base?chroma-base:0,IOSurfaceGetBytesPerRowOfPlane(surface,1));
 if(!handle){
  const uint32_t code=IOSurfaceGetPixelFormat(surface);const char fourcc[5]={char(code>>24),char(code>>16),char(code>>8),char(code),0};
  NBLOG(@"native-metal: texture over surface refused (format %lu %lux%lu usage %lx surface '%s' plane %lu row %lu): %@",(unsigned long)d.pixelFormat,
   (unsigned long)d.width,(unsigned long)d.height,(unsigned long)d.usage,fourcc,(unsigned long)plane,(unsigned long)row,lastError());return nil;
 }
 NBTexture *texture=[NBTexture new];texture.handle=handle;nmb_texture_describe(handle,&descriptor);texture.descriptor=descriptor;
 texture.buffer=buffer;texture.offset=offset;texture.rowBytes=row;return texture;
}
@implementation NBBuffer (Textures)
- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor *)d offset:(NSUInteger)offset bytesPerRow:(NSUInteger)row {
 return (id<MTLTexture>)textureOnBuffer(self,d,offset,row,false);
}
- (id<MTLTexture>)newLinearTextureWithDescriptor:(MTLTextureDescriptor *)d offset:(NSUInteger)offset bytesPerRow:(NSUInteger)row bytesPerImage:(NSUInteger)image {
 NmbTextureDescriptor descriptor;if(!textureDescriptor(d,&descriptor))return nil;
 noteTextureMade("linear",d,(unsigned long)((self.options&MTLResourceStorageModeMask)>>MTLResourceStorageModeShift),row);
 NmbTexture *handle=nmb_texture_new_linear(self.handle,&descriptor,offset,row,image);
 if(!handle){NBLOG(@"native-metal: linear texture refused: %@",lastError());return nil;}
 NBTexture *texture=[NBTexture new];texture.handle=handle;nmb_texture_describe(handle,&descriptor);texture.descriptor=descriptor;
 texture.buffer=self;texture.offset=offset;texture.rowBytes=row;return (id<MTLTexture>)texture;
}
@end

// ---- libraries, functions, states ----
@interface NBLibrary : NBObject
@property(nonatomic) NmbLibrary *handle;
@property(nonatomic) std::string captureId;   // the pipeline capture's library id (PipelineCapture.h), empty when it is off
@end
@interface NBFunction : NBObject
@property(nonatomic) NmbFunction *handle;
@property(nonatomic,copy) NSString *name;
@property(nonatomic) MTLFunctionType functionType;
@property(nonatomic) std::string constantsText;   // name:type:hex of each constant value it was specialized with (the journal's)
@property(nonatomic) std::string captureId;       // the pipeline capture's function id, empty when it is off
@end
@implementation NBFunction
- (void)dealloc {nmb_function_release(_handle);}
- (MTLFunctionOptions)options {return MTLFunctionOptionNone;}
@end
// The target's own accessors of MTLFunctionConstantValues (as the offline frontend reads them); their ABI is checked before any call.
@interface MTLFunctionConstantValues (NBOriginalConstantAccess)
- (NSArray *)newIndexedConstantArray;
- (NSArray *)newNamedConstantArray;
@end
@protocol NBOriginalConstantValue
- (NSUInteger)dataType;
- (void *)data;
- (NSUInteger)index;
- (NSString *)name;
@end
static bool methodABI(id object,SEL selector,const char *encoding) {
 Method method=class_getInstanceMethod(object_getClass(object),selector);
 return method && !strcmp(method_getTypeEncoding(method),encoding);
}
// Bytes of a function constant of MTLDataType `type`. 3-wide vectors are their packed elements: iOS 16 QuartzCore specializes its uber
// shaders with uchar3 constants (MTLDataTypeUChar3, 51), which iPadOS 17's never used.
static uint64_t constantBytes(uint64_t type) {
 switch(type) {
  case 5:case 31:case 35:return 12;     // float3, int3, uint3
  case 18:case 39:case 43:return 6;     // half3, short3, ushort3
  case 47:case 51:case 55:return 3;     // char3, uchar3, bool3
  case 3:case 29:case 33:return 4;
  case 4:case 30:case 34:return 8;
  case 6:case 32:case 36:return 16;
  case 16:case 37:case 41:return 2;
  case 17:case 38:case 42:return 4;
  case 19:case 40:case 44:return 8;
  case 45:case 49:case 53:return 1;
  case 46:case 50:case 54:return 2;
  case 48:case 52:case 56:return 4;
  case 81:case 85:return 8;
  case 82:case 86:return 16;
  case 84:case 88:return 32;
  default:return 0;
 }
}
static std::string quartzCoreCallers();
@implementation NBLibrary
- (void)dealloc {nmb_library_release(_handle);}
- (id<MTLFunction>)newFunctionWithName:(NSString *)name constantValues:(MTLFunctionConstantValues *)values error:(NSError **)error {
 if(!name || !values || !methodABI(values,@selector(newIndexedConstantArray),"@16@0:8") || !methodABI(values,@selector(newNamedConstantArray),"@16@0:8")) {
  journal(std::string("C ")+(name.UTF8String?:"(nil)")+" unqualified constant values "+(values?class_getName(object_getClass(values)):"(nil)"));
  if(error)*error=nbError(@"unqualified function constant values");return nil;
 }
 MTLFunctionConstantValues *snapshot=[values copy];
 NSArray *indexed=[snapshot newIndexedConstantArray],*named=[snapshot newNamedConstantArray];
 std::vector<NmbConstant> constants;
 for(unsigned group=0;group<2;group++)for(id value in (group?named:indexed)) {
  if(!methodABI(value,@selector(dataType),"Q16@0:8") || !methodABI(value,@selector(data),"^v16@0:8") ||
     !methodABI(value,group?@selector(name):@selector(index),group?"@16@0:8":"Q16@0:8")) {
   journal(std::string("C ")+name.UTF8String+" unqualified constant value "+class_getName(object_getClass(value)));
   if(error)*error=nbError(@"unqualified constant value ABI");return nil;
  }
  id<NBOriginalConstantValue> v=value;
  const uint64_t bytes=constantBytes(v.dataType);
  if(!bytes || !v.data){journal(std::string("C ")+name.UTF8String+" constant type "+std::to_string(v.dataType)+(group?std::string(" ")+(v.name.UTF8String?:"?"):std::string()));if(error)*error=nbError([NSString stringWithFormat:@"constant type %lu is not qualified",(unsigned long)v.dataType]);return nil;}
  constants.push_back({v.dataType,group?0:v.index,group?v.name.UTF8String:nullptr,v.data,bytes});
 }
 std::string text;
 for(const NmbConstant &c:constants) {
  char head[32];if(!c.name)snprintf(head,sizeof head,"#%llu",(unsigned long long)c.index);
  text+=text.empty()?"":" ";text+=c.name?c.name:head;snprintf(head,sizeof head,":%llu:",(unsigned long long)c.type);text+=head;
  for(uint64_t i=0;i<c.length;i++){snprintf(head,sizeof head,"%02x",((const uint8_t *)c.bytes)[i]);text+=head;}
 }
 const uint64_t functionBegan=nowUs();
 NmbFunction *handle=nmb_function_new_specialized(_handle,name.UTF8String,constants.data(),(uint32_t)constants.size());gTimeFunction.add(functionBegan);
 // One S line per specialization (int-155: every QuartzCore pipeline was made on its draw, and a refused generic function would leave its
 // generic pipeline unmade without a word): the function, its constants, the time, the refusal and QuartzCore's callers.
 if(!gJournalPath.empty()) {
  char head[96];snprintf(head,sizeof head,"S t=%llu us=%llu ok=%d name=",(unsigned long long)((functionBegan-gJournalBegan)/1000),
      (unsigned long long)(nowUs()-functionBegan),handle?1:0);
  journal(head+std::string(name.UTF8String?:"?")+" k="+text+(handle?std::string():" err="+std::string(lastError().UTF8String?:"?"))+" | "+quartzCoreCallers());
 }
 if(!handle){
  noteProblem([NSString stringWithFormat:@"specialized function %@: %@",name,lastError()]);
  if(error)*error=nbError([NSString stringWithFormat:@"specialized function: %@",lastError()]);return nil;
 }
 NBFunction *function=[NBFunction new];function.handle=handle;function.name=name;function.functionType=(MTLFunctionType)nmb_function_type(handle);
 function.constantsText=text;function.captureId=pcap::function(_captureId,name.UTF8String,constants.data(),constants.size());
 if(error)*error=nil;return (id<MTLFunction>)function;
}
- (void)newFunctionWithName:(NSString *)name constantValues:(MTLFunctionConstantValues *)values completionHandler:(void (^)(id<MTLFunction>,NSError *))handler {
 NSError *error=nil;id<MTLFunction> function=[self newFunctionWithName:name constantValues:values error:&error];
 handler(function,error);
}
- (MTLLibraryType)type {return MTLLibraryTypeExecutable;}
// The descriptor's name and constants; its specialized name only labels Metal's function and its options ask for compile-time work the
// runtime does when the function is made.
- (id<MTLFunction>)newFunctionWithDescriptor:(MTLFunctionDescriptor *)d error:(NSError **)error {
 journal(std::string("D ")+(d.name.UTF8String?:"(nil)")+(d.constantValues?" constants":"")+(d.specializedName?std::string(" as ")+d.specializedName.UTF8String:std::string()));
 if(!d.name){if(error)*error=nbError(@"a function name is required");return nil;}
 if(d.constantValues)return [self newFunctionWithName:d.name constantValues:d.constantValues error:error];
 id<MTLFunction> function=[self newFunctionWithName:d.name];
 if(!function){if(error)*error=nbError([NSString stringWithFormat:@"function %@: %@",d.name,lastError()]);return nil;}
 if(error)*error=nil;return function;
}
- (NSArray<NSString *> *)functionNames {
 NSMutableArray *names=[NSMutableArray array];
 for(uint32_t i=0,n=nmb_library_function_count(_handle);i<n;i++) {
  char name[256];nmb_library_function_name(_handle,i,name,sizeof name);[names addObject:[NSString stringWithUTF8String:name]];
 }
 return names;
}
- (id<MTLFunction>)newFunctionWithName:(NSString *)name {
 const uint64_t functionBegan=nowUs();
 NmbFunction *handle=name?nmb_function_new(_handle,name.UTF8String):nullptr;gTimeFunction.add(functionBegan);
 if(!handle){journal(std::string("N ")+(name.UTF8String?:"(nil)")+" err="+(lastError().UTF8String?:"?"));return nil;}
 NBFunction *function=[NBFunction new];function.handle=handle;function.name=name;function.functionType=(MTLFunctionType)nmb_function_type(handle);
 function.captureId=pcap::function(_captureId,name.UTF8String,nullptr,0);
 return (id<MTLFunction>)function;
}
@end
@interface NBComputePipeline : NBObject
@property(nonatomic) NmbComputePipeline *handle;
@property(nonatomic) long captureId;   // the pipeline capture's compute number, -1 when it is off
@end
@implementation NBComputePipeline
- (void)dealloc {nmb_compute_pipeline_release(_handle);}
- (NSUInteger)maxTotalThreadsPerThreadgroup {return nmb_compute_pipeline_max_threads(_handle);}
- (NSUInteger)threadExecutionWidth {return nmb_compute_pipeline_execution_width(_handle);}
@end
@interface NBRenderPipeline : NBObject {
@public
 // NGPU_XR_AS_F16: the color attachments whose BGRA10_XR-family format the pipeline takes as RGBA16Float (bit i), the descriptor with the
 // caller's formats and its functions, and the pipeline made from it once a pass of XR attachments (textures over IOSurfaces or buffers,
 // which keep XR) sets it. Without it such a pass refused the command buffer: setRenderPipelineState AttachmentFormat(Color(0)),
 // the App Library's backdrop black (session-163610).
 uint32_t xrMask;
 std::shared_ptr<NmbRenderPipelineDescriptor> xrDescriptor;
 std::map<uint32_t,NmbRenderPipeline *> xrHandles;   // by the remapped attachments the pass keeps XR on; null: refused
}
@property(nonatomic) NmbRenderPipeline *handle;
@property(nonatomic) long captureId;   // the pipeline capture's render number, -1 when it is off
@property(nonatomic,strong) id xrFunctions;   // keeps the descriptor's functions alive for the XR variant
- (NmbRenderPipeline *)handleForXr:(uint32_t)passXr;
@end
static std::atomic<uint64_t> gXrVariants;
@implementation NBRenderPipeline
- (void)dealloc {nmb_render_pipeline_release(_handle);for(auto &[k,h]:xrHandles)if(h)nmb_render_pipeline_release(h);}
// The pipeline for a pass whose XR color attachments are passXr: each attachment this pipeline remapped takes the caller's XR format
// where the pass keeps XR on it, RGBA16Float elsewhere (a variant per combination, made at its first use).
- (NmbRenderPipeline *)handleForXr:(uint32_t)passXr {
 const uint32_t key=xrMask & passXr;
 if(!key || !xrDescriptor)return _handle;
 @synchronized(self) {
  auto at=xrHandles.find(key);
  if(at==xrHandles.end()) {
   NmbRenderPipelineDescriptor variant=*xrDescriptor;
   for(unsigned i=0;i<8;i++)if((xrMask>>i&1) && !(key>>i&1))variant.colors[i].format=xrFormat(variant.colors[i].format);
   const NmbPipelineOptions creation={0,nullptr,0};
   const uint64_t began=nowUs();
   NmbRenderPipeline *h=nmb_render_pipeline_new_with_options(gDevice,&variant,&creation);gTimePipeline.add(began);
   if(h)gXrVariants++;else noteProblem([NSString stringWithFormat:@"render pipeline (XR variant %x): %s",key,nmb_last_error()]);
   at=xrHandles.emplace(key,h).first;
  }
  return at->second?:_handle;
 }
}
// The threadgroup size a tile dispatch of this pipeline could use: the device's limit (no tile shaders are offered).
- (NSUInteger)maxTotalThreadsPerThreadgroup {
 uint64_t s[3]={};if(!nmb_device_max_threads_per_threadgroup(gDevice,s))return 0;
 return s[0];
}
@end
@interface NBSampler : NBObject
@property(nonatomic) NmbSampler *handle;
@property(nonatomic) std::string captureText;   // the pipeline capture's descriptor (PipelineCapture.h)
@end
@implementation NBSampler
- (void)dealloc {nmb_sampler_release(_handle);}
- (NSUInteger)pixelFormat {return MTLPixelFormatInvalid;}   // SPI: a sampler built from a plain descriptor names no format
@end
@interface NBDepthStencil : NBObject
@property(nonatomic) NmbDepthStencil *handle;
@property(nonatomic) std::string captureText;   // the pipeline capture's descriptor
@end
@implementation NBDepthStencil
- (void)dealloc {nmb_depth_stencil_release(_handle);}
@end

// ---- events and fences ----
@interface NBEvent : NBObject
@property(nonatomic) NmbEvent *handle;
@end
@implementation NBEvent
- (void)dealloc {nmb_event_release(_handle);}
@end
@interface NBSharedEvent : NBEvent
@end
static void notified(void *context,uint64_t value);
@implementation NBSharedEvent
- (uint64_t)signaledValue {return nmb_shared_event_value(self.handle);}
- (void)setSignaledValue:(uint64_t)value {nmb_shared_event_set_value(self.handle,value);}
- (BOOL)waitUntilSignaledValue:(uint64_t)value timeoutMS:(uint64_t)milliseconds {return nmb_shared_event_wait(self.handle,value,milliseconds);}
// The block runs on the listener's dispatch queue (Metal's contract); the bridge's listener thread only forwards it there.
- (void)notifyListener:(MTLSharedEventListener *)listener atValue:(uint64_t)value block:(MTLSharedEventNotificationBlock)block {
 if(!block)return;
 dispatch_queue_t queue=listener.dispatchQueue?:dispatch_get_global_queue(QOS_CLASS_DEFAULT,0);
 NSArray *job=@[self,queue,[block copy]];
 if(!nmb_shared_event_notify(self.handle,value,notified,(__bridge_retained void *)job)) {
  (void)(__bridge_transfer NSArray *)(__bridge void *)job;
  NBLOG(@"native-metal: notifyListener refused: %@",lastError());
 }
}
// A handle for another process needs a kernel-backed event this bridge does not have yet: refused, never a handle that could not work.
- (MTLSharedEventHandle *)newSharedEventHandle {
 if(gMissing)gMissing("NBSharedEvent","newSharedEventHandle (cross-process)");
 return nil;
}
@end
static void notified(void *context,uint64_t value) {
 NSArray *job=(__bridge_transfer NSArray *)context;
 id<MTLSharedEvent> event=job[0];MTLSharedEventNotificationBlock block=job[2];
 dispatch_async(job[1],^{block(event,value);});
}
@interface NBFence : NBObject
@property(nonatomic) NmbFence *handle;
@end
@implementation NBFence
- (void)dealloc {nmb_fence_release(_handle);}
@end

// ---- command buffers and encoders ----
@interface NBQueue : NBObject
@property(nonatomic) NmbQueue *handle;
@property(nonatomic) uint32_t traceSerial;   // the frame trace's queue number
@property(atomic,strong) dispatch_queue_t completionQueue;   // SPI: where command buffers' handlers run (nil: the runtime's thread)
@end
@interface NBCommandBuffer : NBObject {
@public
 uint64_t serial;   // nonzero, unique: a texture over CPU-written bytes is checked once per command buffer (noteCpuContents)
 EncodedWork encodedWork;   // the command buffer's single encoder thread, not global counter deltas
 uint64_t createdUs;
 bool cpuOpen;  // counted open in the CPU profile (CpuProfile.h)
 // The frame trace (FrameTrace.h), when it is on: the commit's return, and the frontend's totals when the command buffer was made.
 uint64_t traceCommittedUs,tracePipeUs,tracePipeN,traceWaitUs,traceFlushUs,traceFlushBytes,traceDoneUs,traceHandledUs;
 std::atomic<int> traceJoin;   // the commit's return and the completion: the second writes the C line
}
@property(nonatomic) NmbCommandBuffer *handle;
@property(nonatomic,strong) NBQueue *queue;
@property(atomic,copy) NSString *encodeFailure;
@property(atomic) BOOL committed,failedLocally;
@property(nonatomic) uint64_t committedUs;
@property(nonatomic) uint32_t commitIndex;
@property(nonatomic,strong) NSMutableArray *scheduledHandlers,*completedHandlers;
@property(nonatomic,strong) NSMutableSet *surfaces;
// Runtime handles retain GPU storage, but only the Objective-C owner retains
// a no-copy deallocator, its memory owner and texture/heap ancestry.
@property(nonatomic,strong) NSMutableSet *resourceOwners;
@property(nonatomic,strong) dispatch_group_t handlers;
@property(nonatomic,strong) dispatch_group_t schedulingHandlers;
@property(nonatomic) BOOL schedulingDelivered;
- (void)refuse:(NSString *)what;
@end

// ---- frame capture (NGPU_CAPTURE with NGPU_CAPTURE_FRAMES=N, diagnostics) ----
// N command buffers after the first NGPU_CAPTURE_SKIP (default 0) the process makes, written as metal2vulkan command-replay records into the
// pipeline capture's directory (frames.jsonl, blob-<hash>.bin): each resource at its first use in the window (a texture's level 0 of each
// slice where its contents are read before they are written, a buffer's bytes at each commit that finds them changed), samplers and
// depth/stencil states, and every encoder's commands in order. Resources are made alone (heap placement is not kept); a view is a view of
// its parent's first level and slice. metal2vulkan scripts/pipeline-census/frames_manifest.py builds the manifest.
namespace fcap {
static std::recursive_mutex gLock;
static long gSkip=0,gCount=0,gEnd=600;static std::atomic<long> gMade{0},gCommitted{0},gWindowCommitted{0};static bool gArmed=false;
// A resource's id is an associated object of the resource itself: the capture never extends a resource's life (session-085521 and
// -093006: backboardd SIGBUS while the window held CA's no-copy buffers past their owners' release, the owners' pages unmapped).
static char gIdKey;
static std::unordered_map<std::string,uint64_t> gHashes;      // buffer id -> hash of its last recorded bytes
static std::set<uint64_t> gBlobs;
static unsigned gNext=0;
static thread_local bool gInternal=false;                     // a command buffer the capture makes itself
static void arm() {
 const char *n=getenv("NGPU_CAPTURE_FRAMES"),*s=getenv("NGPU_CAPTURE_SKIP");
 const char *e=getenv("NGPU_CAPTURE_END");
 gCount=n?atol(n):0;gSkip=s?atol(s):0;gArmed=pcap::on() && gCount>0;
 gEnd=e?atol(e):gArmed?0:600;   // a window ends the capture itself; NGPU_CAPTURE_END set ends it there in any case
 if(gArmed){journal("X frames "+std::to_string(gCount)+" after "+std::to_string(gSkip));}
}
// The command buffer's index in the window, or -1.
static long begin() {
 if(!gArmed || gInternal)return -1;
 const long n=gMade.fetch_add(1);
 return n>=gSkip && n<gSkip+gCount?n-gSkip:-1;
}
static void record(const std::string &line) {
 std::lock_guard<std::recursive_mutex> guard(gLock);
 if(pcap::stopped())return;
 if(FILE *f=fopen((pcap::dir()+"/frames.jsonl").c_str(),"a")){fwrite(line.data(),1,line.size(),f);fputc('\n',f);fclose(f);}
}
static uint64_t hash(const void *bytes,size_t n) {
 uint64_t h=0xcbf29ce484222325ull;const uint8_t *p=(const uint8_t *)bytes;size_t i=0;
 for(;i+8<=n;i+=8){uint64_t w;memcpy(&w,p+i,8);h=(h^w)*0x100000001b3ull;h^=h>>29;}
 for(;i<n;i++)h=(h^p[i])*0x100000001b3ull;
 return h^n;
}
// The bytes as a blob file, written once per content; its name.
static std::string blob(const void *bytes,size_t n) {
 const uint64_t h=hash(bytes,n);char name[48];snprintf(name,sizeof name,"blob-%016llx.bin",(unsigned long long)h);
 std::lock_guard<std::recursive_mutex> guard(gLock);
 if(gBlobs.insert(h).second)if(FILE *f=fopen((pcap::dir()+"/"+name).c_str(),"wb")){fwrite(bytes,1,n,f);fclose(f);}
 return name;
}
static std::string named(id object,const char *prefix,bool *fresh) {
 std::lock_guard<std::recursive_mutex> guard(gLock);
 if(NSString *known=objc_getAssociatedObject(object,&gIdKey)){if(fresh)*fresh=false;return known.UTF8String;}
 const std::string key=prefix+std::to_string(gNext++);
 objc_setAssociatedObject(object,&gIdKey,[NSString stringWithUTF8String:key.c_str()],OBJC_ASSOCIATION_RETAIN_NONATOMIC);
 if(fresh)*fresh=true;
 return key;
}
}  // namespace fcap
// One command buffer's records in the window, and one encoder's.
struct FrameCb {long index;std::vector<std::string> encoders;NSMutableArray *buffers=[NSMutableArray array];};
struct FrameEncoder {std::string kind,head;std::vector<std::string> commands;std::map<std::string,std::pair<std::string,uint64_t>> slots;};
@interface NBEncoder : NBObject
@property(nonatomic) NmbEncoder *handle;
@property(nonatomic) std::shared_ptr<pcap::Bindings> capture;   // a render encoder's state for the pipeline capture, null when it is off
@property(nonatomic,strong) NBCommandBuffer *commandBuffer;
@property(nonatomic) std::shared_ptr<FrameEncoder> frame;   // the frame capture's record of this encoder, null outside the window
@end
@interface NBCommandBuffer ()
@property(nonatomic) std::shared_ptr<FrameCb> frame;   // the frame capture's record of this command buffer, null outside the window
@end
namespace fcap {
static std::string buffer(NBBuffer *b,FrameCb *cb) {
 if(!b)return "null";
 bool fresh=false;const std::string key=named(b,"b",&fresh);
 if(fresh)record("{\"buffer\":\""+key+"\",\"length\":"+std::to_string((unsigned long long)b.length)+",\"options\":"+std::to_string((unsigned long long)b.options)+"}");
 if(cb)[cb->buffers addObject:b];
 return key;
}
// Level 0 of each slice, rows packed, read back for a texture whose contents the window reads before it writes them; empty when it cannot.
static std::string contents(NBTexture *t) {
 const NmbTextureDescriptor d=t.descriptor;
 const uint64_t texel=nmb_device_linear_bytes_per_row(gDevice,d.format,256)/256;
 if(!texel || d.samples>1 || !(d.type==2 || d.type==3 || d.type==5 || d.type==1))return std::string();
 const uint64_t row=texel*d.width,image=row*d.height,slices=std::max<uint64_t>(d.array_length,1)*(d.type==5?6:1);
 std::vector<uint8_t> bytes(image*slices);
 waitForResource(t);
 {
  // Copied through a blit on a queue of the capture's own (not captured), never read by the CPU: getBytes on a texture over a host
  // range its owner had unmapped faulted the render server (session-093006: SIGBUS in Texture::get_bytes under this function).
  static id<MTLCommandQueue> queue;static std::once_flag once;
  std::call_once(once,[]{queue=[(id<MTLDevice>)gDeviceObject newCommandQueue];});
  gInternal=true;
  id<MTLBuffer> staging=[(id<MTLDevice>)gDeviceObject newBufferWithLength:bytes.size() options:MTLResourceStorageModeShared];
  id<MTLCommandBuffer> cb=[queue commandBuffer];id<MTLBlitCommandEncoder> blit=[cb blitCommandEncoder];
  for(uint64_t s=0;s<slices;s++)
   [blit copyFromTexture:(id<MTLTexture>)t sourceSlice:s sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(d.width,d.height,1)
                toBuffer:staging destinationOffset:s*image destinationBytesPerRow:row destinationBytesPerImage:image];
  [blit endEncoding];[cb commit];[cb waitUntilCompleted];
  gInternal=false;
  if(!staging.contents || cb.status!=MTLCommandBufferStatusCompleted)return std::string();
  memcpy(bytes.data(),staging.contents,bytes.size());
 }
 return blob(bytes.data(),bytes.size());
}
// A texture's key; its record (and its parent's, its buffer's) at its first use. `read`: the use reads the contents it starts with.
static std::string texture(NBTexture *t,bool read,FrameCb *cb) {
 if(!t)return "null";
 if(t.buffer)buffer(t.buffer,cb);
 bool fresh=false;const std::string key=named(t,"t",&fresh);
 if(!fresh)return key;
 const NmbTextureDescriptor d=t.descriptor;
 if([(id)t.parent isKindOfClass:[NBTexture class]]) {
  const std::string of=texture((NBTexture *)t.parent,read,cb);
  record("{\"view\":\""+key+"\",\"of\":\""+of+"\",\"format\":"+std::to_string(d.format)+",\"type\":"+std::to_string(d.type)+",\"levels\":[0,"+
         std::to_string(std::max<uint64_t>(d.levels,1))+"],\"slices\":[0,"+std::to_string(std::max<uint64_t>(d.array_length,1)*(d.type==5?6:1))+"]}");
  return key;
 }
 std::string s="{\"texture\":\""+key+"\",\"type\":"+std::to_string(d.type)+",\"format\":"+std::to_string(d.format)+",\"width\":"+std::to_string(d.width)+
     ",\"height\":"+std::to_string(d.height)+",\"depth\":"+std::to_string(std::max<uint64_t>(d.depth,1))+",\"levels\":"+std::to_string(std::max<uint64_t>(d.levels,1))+
     ",\"samples\":"+std::to_string(std::max<uint64_t>(d.samples,1))+",\"array\":"+std::to_string(std::max<uint64_t>(d.array_length,1))+",\"usage\":"+
     std::to_string(d.usage)+",\"options\":"+std::to_string(d.options);
 if(t.buffer) {
  s+=",\"buffer\":\""+named(t.buffer,"b",nullptr)+"\",\"offset\":"+std::to_string((unsigned long long)t.offset)+",\"row\":"+std::to_string((unsigned long long)t.rowBytes);
  if(t.surface)s+=",\"surface\":true";
 } else if(read) {
  const std::string file=contents(t);
  if(!file.empty())s+=",\"file\":\""+file+"\"";else s+=",\"contents\":\"unread\"";
 }
 record(s+"}");
 return key;
}
static std::string sampler(NBSampler *s) {
 if(!s)return "null";
 bool fresh=false;const std::string key=named(s,"s",&fresh);
 if(fresh)record("{\"sampler\":\""+key+"\",\"descriptor\":"+(s.captureText.empty()?std::string("{}"):s.captureText)+"}");
 return key;
}
static std::string depthStencil(NBDepthStencil *d) {
 if(!d)return "null";
 bool fresh=false;const std::string key=named(d,"d",&fresh);
 if(fresh)record("{\"depthStencil\":\""+key+"\",\"descriptor\":"+(d.captureText.empty()?std::string("{}"):d.captureText)+"}");
 return key;
}
// A buffer's bytes read through the VM (vm_read_overwrite), so a no-copy range its owner unmapped answers false instead of faulting
// the process (session-085521: backboardd SIGBUS while the capture read buffers whose imports the owner had unmapped).
static bool copyOut(const void *bytes,size_t n,std::vector<uint8_t> &out) {
 out.resize(n);vm_size_t got=0;
 return n && vm_read_overwrite(mach_task_self(),(vm_address_t)bytes,n,(vm_address_t)out.data(),&got)==KERN_SUCCESS && got==n;
}
// At commit: the bytes of every buffer the command buffer names that changed since they were last recorded, then its encoders.
static void commit(FrameCb *cb) {
 std::string uploads;
 std::vector<uint8_t> copy;
 for(NBBuffer *b in cb->buffers) {
  const void *mapped=b.contents;if(!mapped)continue;
  const std::string key=named(b,"b",nullptr);const size_t n=b.length;
  if(!copyOut(mapped,n,copy)){uploads+=(uploads.empty()?"":",")+std::string("{\"buffer\":\"")+key+"\",\"unreadable\":true}";continue;}
  const void *bytes=copy.data();
  const uint64_t h=hash(bytes,n);
  {std::lock_guard<std::recursive_mutex> guard(gLock);auto at=gHashes.find(key);if(at!=gHashes.end() && at->second==h)continue;gHashes[key]=h;}
  uploads+=(uploads.empty()?"":",")+std::string("{\"buffer\":\"")+key+"\",\"file\":\""+blob(bytes,n)+"\"}";
 }
 std::string s="{\"commandBuffer\":"+std::to_string(cb->index)+",\"uploads\":["+uploads+"],\"encoders\":[";
 for(size_t i=0;i<cb->encoders.size();i++)s+=(i?",":"")+cb->encoders[i];
 record(s+"]}");
 gWindowCommitted++;
}
// Each commit while the capture is on: the capture ends once the window's command buffers committed (or 64 command buffers were made
// past the window, one of them never committed), or at NGPU_CAPTURE_END committed command buffers (default 600 without a window).
// session-091248: ending at SKIP+FRAMES commits ended it before the window's own command buffers, made then, had committed.
static void committed() {
 if(!pcap::on())return;
 const long n=++gCommitted;
 const bool windowDone=gArmed && (gWindowCommitted>=gCount || gMade>gSkip+gCount+64);
 if((gArmed && windowDone) || (gEnd>0 && n>=gEnd)) {
  std::lock_guard<std::recursive_mutex> guard(gLock);
  pcap::finish("{\"commandBuffers\":"+std::to_string(n)+",\"frames\":"+std::to_string((long)gWindowCommitted)+",\"resources\":"+std::to_string(gNext)+"}");
  journal("X capture done after "+std::to_string(n)+" command buffers, "+std::to_string((long)gWindowCommitted)+" captured");
 }
}
static const char *stageName(uint32_t stage) {return stage==1?"vertex":stage==2?"fragment":"kernel";}
static std::string num(double v) {char e[40];snprintf(e,sizeof e,"%.9g",v);return e;}
static std::string triple(uint64_t a,uint64_t b,uint64_t c) {return "["+std::to_string(a)+","+std::to_string(b)+","+std::to_string(c)+"]";}
}  // namespace fcap
static void waitForResource(id resource) {
 const uint64_t began=nowUs();
 NSArray *pending;@synchronized(gInFlight){pending=gInFlight.allObjects;}
 for(NBCommandBuffer *commandBuffer in pending) {
  int dependency=-1;
  if([resource isKindOfClass:[NBBuffer class]])dependency=nmb_command_buffer_uses_buffer(commandBuffer.handle,[(NBBuffer *)resource handle]);
  else if([resource isKindOfClass:[NBTexture class]])dependency=nmb_command_buffer_uses_texture(commandBuffer.handle,[(NBTexture *)resource handle]);
  gResourceChecks++;
  // An unknown result retains the old conservative wait. False means no matching storage
  // (or GPU work already completed), not a fabricated completion for a pending dependency.
  if(dependency!=0)[(id<MTLCommandBuffer>)commandBuffer waitUntilCompleted];else gResourceSkipped++;
 }
 gTimeResourceWait.add(began);
}
// A bound texture as the pipeline capture records it.
static std::string captureTexture(NBTexture *texture) {
 const char *over=texture.surface?"surface":texture.buffer?"buffer":nullptr;
 return pcap::textureText(texture.descriptor,over,texture.offset,texture.rowBytes,texture.parent!=nil);
}
static const char *captureStage(uint32_t stage) {return stage==1?"v":stage==2?"f":"k";}
// A texture over an IOSurface states the surface's contents to the runtime as its seed (plus one: 0 states nothing), so a
// two-plane YCbCr texture converts its planes only when the seed changed or a command wrote them, not at every encoder
// (session-103156: a 1105x1160 YCBCR8_420_2P surface converted every frame, 17.5 ms, 36% of GPU time). NGPU_YCBCR_SEED=0 (the
// plugin's gpu-env) states nothing: every encoder converts, as before.
static std::atomic<uint64_t> gSurfaceGenerations;
static void noteSurfaceContents(NBTexture *texture) {
 static const bool on=[]{const char *v=getenv("NGPU_YCBCR_SEED");return !(v && v[0]=='0');}();
 if(!on || !texture.surface)return;
 // A view states its parents' too: a view of another component layout is a shadow of its parent's image, and the copy in from the
 // surface's bytes is the parent's (fajita shadow log: 1271x1823 and 1326x1326 surfaces copied in at every use, never stated).
 const uint64_t generation=(uint64_t)IOSurfaceGetSeed(texture.surface)+1;
 for(NBTexture *t=texture;t;t=[t.parent isKindOfClass:[NBTexture class]]?(NBTexture *)t.parent:nil)
  nmb_texture_set_contents_generation(t.handle,generation);
 gSurfaceGenerations++;
}
// A texture over a buffer's bytes whose generation the frontend can state (NGPU_CPU_HELD=0: none, every use copies in as before):
//  - a no-copy buffer's GPU-owned copy (NGPU_HOST_COPY's default, the bytes copied once at creation): only a writer through `contents`
//    changes them, so its texture states a constant generation until the first `contents` call;
//  - a no-copy import of the caller's pages (above 4 MiB: the wallpaper's images): the kernel says whether the CPU wrote them since the
//    last check (NativeAdreno HostDirty, through Turnip's tu_darwin_host_dirty), once per command buffer, and a write moves the generation.
// The runtime then copies the texture's bytes into its shadow image only when they changed (fajita shadow log: 1271x1823 and 1326x1326
// textures over the wallpaper's pages copied in at every use, 19 GB in 86 s; 180x180 icons and labels over copies, about 50,000).
static int (*gHostDirty)(const uint64_t *,uint32_t,uint32_t *);
static std::string gLoaderPath;
static void noteCpuContents(NBTexture *texture,NBCommandBuffer *commands) {
 static const bool on=[]{const char *v=getenv("NGPU_CPU_HELD");return !(v && v[0]=='0');}();
 NBBuffer *buffer=texture.buffer;
 if(!on || texture.surface || !buffer || buffer.importSurface || !commands)return;
 if(buffer.copySource) {
  if(buffer.contentsTouched)return;
  static const bool everyCommit=[]{const char *v=getenv("NGPU_HOST_COPY");return v && v[0]=='2';}();
  if(everyCommit)return;
  if(!texture.cpuGeneration){texture.cpuGeneration=1;nmb_texture_set_contents_generation(texture.handle,1);}
  return;
 }
 if(!buffer.wrapped || texture.cpuCheckedIn==commands->serial)return;
 static dispatch_once_t once;
 dispatch_once(&once,^{
  if(void *lib=gLoaderPath.empty()?nullptr:dlopen(gLoaderPath.c_str(),RTLD_NOLOAD|RTLD_LAZY))
   gHostDirty=(int (*)(const uint64_t *,uint32_t,uint32_t *))dlsym(lib,"tu_darwin_host_dirty");
 });
 if(!gHostDirty)return;
 texture.cpuCheckedIn=commands->serial;
 const NmbTextureDescriptor &d=texture.descriptor;
 const uint64_t range[2]={(uint64_t)(uintptr_t)buffer.wrapped+texture.offset,(uint64_t)texture.rowBytes*(d.height?d.height:1)};
 uint32_t dirty=~0u;
 if(gHostDirty(range,1,&dirty)!=0)return;   // an older kernel: no statement
 if((dirty&1) || !texture.cpuGeneration)texture.cpuGeneration++;
 nmb_texture_set_contents_generation(texture.handle,texture.cpuGeneration);
}
@implementation NBEncoder
- (void)dealloc {if(_handle)nmb_encoder_discard(_handle);}
- (bool)check:(bool)ok what:(const char *)what {
 if(!ok)[_commandBuffer refuse:[NSString stringWithFormat:@"%s: %@",what,lastError()]];
 return ok;
}
- (void)endEncoding {
 NmbEncoder *handle=_handle;_handle=nullptr;
 if(!handle){[_commandBuffer refuse:@"endEncoding twice"];return;}
 [self check:nmb_encoder_end(handle) what:"endEncoding"];
 if(_frame && _commandBuffer.frame) {
  std::string s="{\""+_frame->kind+"\":{"+_frame->head+(_frame->head.empty()?"":",")+"\"commands\":[";
  for(size_t i=0;i<_frame->commands.size();i++)s+=(i?",":"")+_frame->commands[i];
  _commandBuffer.frame->encoders.push_back(s+"]}}");_frame=nullptr;
 }
}
- (void)frameCommand:(const std::string &)command {if(_frame)_frame->commands.push_back(command);}
- (void)insertDebugSignpost:(NSString *)s {(void)s;}
- (void)pushDebugGroup:(NSString *)s {(void)s;}
- (void)popDebugGroup {}
// Stage 0 compute, 1 vertex, 2 fragment (the bridge's numbering).
- (void)buffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset index:(NSUInteger)index stage:(uint32_t)stage {
 if(buffer && ![(id)buffer isKindOfClass:[NBBuffer class]]){[_commandBuffer refuse:@"foreign buffer"];return;}
 if(buffer)[_commandBuffer.resourceOwners addObject:buffer];
 [self check:nmb_encoder_set_buffer(_handle,stage,buffer?[(NBBuffer *)buffer handle]:nullptr,offset,(uint32_t)index) what:"setBuffer"];
 if(_frame && buffer) {
  const std::string id=fcap::buffer((NBBuffer *)buffer,_commandBuffer.frame.get());
  _frame->slots[std::to_string(stage)+":"+std::to_string(index)]={id,offset};
  _frame->commands.push_back(std::string("{\"buffer\":{\"stage\":\"")+fcap::stageName(stage)+"\",\"index\":"+std::to_string(index)+",\"buffer\":\""+id+"\",\"offset\":"+std::to_string(offset)+"}}");
 }
 if(_capture)_capture->slots[std::string(captureStage(stage))+"b"+std::to_string(index)]=buffer?"{\"buffer\":"+std::to_string(buffer.length)+",\"offset\":"+std::to_string(offset&255)+"}":std::string();
}
- (void)texture:(id<MTLTexture>)texture index:(NSUInteger)index stage:(uint32_t)stage {
 if(texture && ![(id)texture isKindOfClass:[NBTexture class]]){[_commandBuffer refuse:@"foreign texture"];return;}
 if(texture)[_commandBuffer.resourceOwners addObject:texture];
 if(texture && [(NBTexture *)texture surface]) {
  [_commandBuffer.surfaces addObject:texture];
  noteSurfaceContents((NBTexture *)texture);
 }
 if(texture)noteCpuContents((NBTexture *)texture,_commandBuffer);
 [self check:nmb_encoder_set_texture(_handle,stage,texture?[(NBTexture *)texture handle]:nullptr,(uint32_t)index) what:"setTexture"];
 if(_frame && texture)_frame->commands.push_back(std::string("{\"texture\":{\"stage\":\"")+fcap::stageName(stage)+"\",\"index\":"+std::to_string(index)+",\"texture\":\""+
     fcap::texture((NBTexture *)texture,true,_commandBuffer.frame.get())+"\"}}");
 if(_capture)_capture->slots[std::string(captureStage(stage))+"t"+std::to_string(index)]=texture?captureTexture((NBTexture *)texture):std::string();
}
- (void)sampler:(id<MTLSamplerState>)sampler index:(NSUInteger)index stage:(uint32_t)stage {
 if(sampler && ![(id)sampler isKindOfClass:[NBSampler class]]){[_commandBuffer refuse:@"foreign sampler"];return;}
 [self check:nmb_encoder_set_sampler(_handle,stage,sampler?[(NBSampler *)sampler handle]:nullptr,(uint32_t)index) what:"setSamplerState"];
 if(_frame && sampler)_frame->commands.push_back(std::string("{\"sampler\":{\"stage\":\"")+fcap::stageName(stage)+"\",\"index\":"+std::to_string(index)+",\"sampler\":\""+
     fcap::sampler((NBSampler *)sampler)+"\"}}");
 if(_capture)_capture->slots[std::string(captureStage(stage))+"s"+std::to_string(index)]=sampler?[(NBSampler *)sampler captureText]:std::string();
}
- (void)bytes:(const void *)bytes length:(NSUInteger)length index:(NSUInteger)index stage:(uint32_t)stage {
 [self check:nmb_encoder_set_bytes(_handle,stage,bytes,length,(uint32_t)index) what:"setBytes"];
 if(_frame)_frame->commands.push_back(std::string("{\"bytes\":{\"stage\":\"")+fcap::stageName(stage)+"\",\"index\":"+std::to_string(index)+",\"hex\":\""+
     pcap::hexOf(bytes,length)+"\"}}");
 if(_capture)_capture->slots[std::string(captureStage(stage))+"b"+std::to_string(index)]="{\"bytes\":"+std::to_string(length)+"}";
}
- (void)fence:(id<MTLFence>)fence stages:(uint64_t)stages update:(bool)update {
 if(![(id)fence isKindOfClass:[NBFence class]]){[_commandBuffer refuse:@"foreign fence"];return;}
 NmbFence *handle=[(NBFence *)fence handle];
 [self check:update?nmb_encoder_update_fence(_handle,handle,stages):nmb_encoder_wait_fence(_handle,handle,stages) what:update?"updateFence":"waitForFence"];
}
- (void)updateFence:(id<MTLFence>)fence {[self fence:fence stages:0 update:true];}
- (void)waitForFence:(id<MTLFence>)fence {[self fence:fence stages:0 update:false];}
- (void)offset:(NSUInteger)offset index:(NSUInteger)index stage:(uint32_t)stage {
 [self check:nmb_encoder_set_buffer_offset(_handle,stage,offset,(uint32_t)index) what:"setBufferOffset"];
 if(_frame) {
  auto at=_frame->slots.find(std::to_string(stage)+":"+std::to_string(index));
  if(at!=_frame->slots.end()){at->second.second=offset;
   _frame->commands.push_back(std::string("{\"buffer\":{\"stage\":\"")+fcap::stageName(stage)+"\",\"index\":"+std::to_string(index)+",\"buffer\":\""+at->second.first+"\",\"offset\":"+std::to_string(offset)+"}}");}
 }
 if(_capture) {
  std::string &slot=_capture->slots[std::string(captureStage(stage))+"b"+std::to_string(index)];
  const size_t at=slot.find(",\"offset\":");
  if(at!=std::string::npos)slot=slot.substr(0,at)+",\"offset\":"+std::to_string(offset&255)+"}";
 }
}
@end

@interface NBComputeEncoder : NBEncoder
@end
@implementation NBComputeEncoder
- (MTLDispatchType)dispatchType {return MTLDispatchTypeSerial;}
- (void)setComputePipelineState:(id<MTLComputePipelineState>)state {
 if(![(id)state isKindOfClass:[NBComputePipeline class]]){[self.commandBuffer refuse:@"foreign compute pipeline"];return;}
 [self check:nmb_compute_set_pipeline(self.handle,[(NBComputePipeline *)state handle]) what:"setComputePipelineState"];
 if(self.frame)[self frameCommand:"{\"pipeline\":\"c"+std::to_string([(NBComputePipeline *)state captureId])+"\"}"];
}
- (void)setBuffer:(id<MTLBuffer>)b offset:(NSUInteger)o atIndex:(NSUInteger)i {[self buffer:b offset:o index:i stage:0];}
- (void)setBufferOffset:(NSUInteger)o atIndex:(NSUInteger)i {[self offset:o index:i stage:0];}
- (void)setBuffers:(const id<MTLBuffer> __unsafe_unretained [])b offsets:(const NSUInteger [])o withRange:(NSRange)r {
 for(NSUInteger k=0;k<r.length;k++)[self buffer:b[k] offset:o[k] index:r.location+k stage:0];
}
- (void)setBytes:(const void *)p length:(NSUInteger)n atIndex:(NSUInteger)i {[self bytes:p length:n index:i stage:0];}
- (void)setTexture:(id<MTLTexture>)t atIndex:(NSUInteger)i {[self texture:t index:i stage:0];}
- (void)setTextures:(const id<MTLTexture> __unsafe_unretained [])t withRange:(NSRange)r {
 for(NSUInteger k=0;k<r.length;k++)[self texture:t[k] index:r.location+k stage:0];
}
- (void)setSamplerState:(id<MTLSamplerState>)s atIndex:(NSUInteger)i {[self sampler:s index:i stage:0];}
- (void)setThreadgroupMemoryLength:(NSUInteger)n atIndex:(NSUInteger)i {
 [self check:nmb_compute_set_threadgroup_memory(self.handle,(uint32_t)n,(uint32_t)i) what:"setThreadgroupMemoryLength"];
 if(self.frame)[self frameCommand:"{\"threadgroupMemory\":{\"index\":"+std::to_string((unsigned long long)(i))+",\"length\":"+std::to_string((unsigned long long)(n))+"}}"];
}
- (void)dispatch:(MTLSize)size group:(MTLSize)group threads:(bool)threads {
 gDispatches++;self.commandBuffer->encodedWork.dispatches++;
 const uint64_t s[3]={size.width,size.height,size.depth},g[3]={group.width,group.height,group.depth};
 [self check:nmb_compute_dispatch(self.handle,s,g,threads) what:"dispatch"];
 if(self.frame)[self frameCommand:threads?"{\"dispatchThreads\":{\"grid\":"+fcap::triple(s[0],s[1],s[2])+",\"threads\":"+fcap::triple(g[0],g[1],g[2])+"}}":
     "{\"dispatchThreadgroups\":{\"groups\":"+fcap::triple(s[0],s[1],s[2])+",\"threads\":"+fcap::triple(g[0],g[1],g[2])+"}}"];
}
- (void)dispatchThreads:(MTLSize)t threadsPerThreadgroup:(MTLSize)g {[self dispatch:t group:g threads:true];}
- (void)dispatchThreadgroups:(MTLSize)t threadsPerThreadgroup:(MTLSize)g {[self dispatch:t group:g threads:false];}
- (void)useResource:(id<MTLResource>)r usage:(MTLResourceUsage)u {(void)r;(void)u;}
- (void)memoryBarrierWithScope:(MTLBarrierScope)s {(void)s;}
@end

@interface NBRenderEncoder : NBEncoder
@property(nonatomic) uint32_t passXr;   // the pass's color attachments that keep a BGRA10_XR-family format (bit i)
@end
@implementation NBRenderEncoder
- (void)setRenderPipelineState:(id<MTLRenderPipelineState>)state {
 if(![(id)state isKindOfClass:[NBRenderPipeline class]]){[self.commandBuffer refuse:@"foreign render pipeline"];return;}
 [self check:nmb_render_set_pipeline(self.handle,[(NBRenderPipeline *)state handleForXr:_passXr]) what:"setRenderPipelineState"];
 if(self.capture)self.capture->pipeline=[(NBRenderPipeline *)state captureId];
 if(self.frame)[self frameCommand:"{\"pipeline\":\"r"+std::to_string([(NBRenderPipeline *)state captureId])+"\"}"];
}
- (void)setVertexBuffer:(id<MTLBuffer>)b offset:(NSUInteger)o atIndex:(NSUInteger)i {[self buffer:b offset:o index:i stage:1];}
- (void)setFragmentBuffer:(id<MTLBuffer>)b offset:(NSUInteger)o atIndex:(NSUInteger)i {[self buffer:b offset:o index:i stage:2];}
- (void)setVertexBufferOffset:(NSUInteger)o atIndex:(NSUInteger)i {[self offset:o index:i stage:1];}
- (void)setFragmentBufferOffset:(NSUInteger)o atIndex:(NSUInteger)i {[self offset:o index:i stage:2];}
- (void)setVertexBuffers:(const id<MTLBuffer> __unsafe_unretained [])b offsets:(const NSUInteger [])o withRange:(NSRange)r {
 for(NSUInteger k=0;k<r.length;k++)[self buffer:b[k] offset:o[k] index:r.location+k stage:1];
}
- (void)setFragmentBuffers:(const id<MTLBuffer> __unsafe_unretained [])b offsets:(const NSUInteger [])o withRange:(NSRange)r {
 for(NSUInteger k=0;k<r.length;k++)[self buffer:b[k] offset:o[k] index:r.location+k stage:2];
}
- (void)setVertexBytes:(const void *)p length:(NSUInteger)n atIndex:(NSUInteger)i {[self bytes:p length:n index:i stage:1];}
- (void)setFragmentBytes:(const void *)p length:(NSUInteger)n atIndex:(NSUInteger)i {[self bytes:p length:n index:i stage:2];}
- (void)setVertexTexture:(id<MTLTexture>)t atIndex:(NSUInteger)i {[self texture:t index:i stage:1];}
- (void)setFragmentTexture:(id<MTLTexture>)t atIndex:(NSUInteger)i {[self texture:t index:i stage:2];}
- (void)setFragmentTextures:(const id<MTLTexture> __unsafe_unretained [])t withRange:(NSRange)r {
 for(NSUInteger k=0;k<r.length;k++)[self texture:t[k] index:r.location+k stage:2];
}
- (void)setVertexSamplerState:(id<MTLSamplerState>)s atIndex:(NSUInteger)i {[self sampler:s index:i stage:1];}
- (void)setFragmentSamplerState:(id<MTLSamplerState>)s atIndex:(NSUInteger)i {[self sampler:s index:i stage:2];}
- (void)setDepthStencilState:(id<MTLDepthStencilState>)state {
 if(state && ![(id)state isKindOfClass:[NBDepthStencil class]]){[self.commandBuffer refuse:@"foreign depth/stencil state"];return;}
 [self check:nmb_render_set_depth_stencil(self.handle,state?[(NBDepthStencil *)state handle]:nullptr) what:"setDepthStencilState"];
 if(self.frame && state)if(self.frame)[self frameCommand:"{\"depthStencil\":\""+fcap::depthStencil((NBDepthStencil *)state)+"\"}"];
 if(self.capture)self.capture->depthStencil=state?[(NBDepthStencil *)state captureText]:std::string("null");
}
- (void)setViewport:(MTLViewport)v {
 const double values[6]={v.originX,v.originY,v.width,v.height,v.znear,v.zfar};
 [self check:nmb_render_set_viewport(self.handle,values) what:"setViewport"];
 if(self.frame){std::string s="{\"viewport\":[";for(int i=0;i<6;i++)s+=(i?",":"")+fcap::num(values[i]);if(self.frame)[self frameCommand:s+"]}"];}
}
- (void)setScissorRect:(MTLScissorRect)r {
 const uint64_t values[4]={r.x,r.y,r.width,r.height};
 [self check:nmb_render_set_scissor(self.handle,values) what:"setScissorRect"];
 if(self.frame)[self frameCommand:"{\"scissor\":["+std::to_string(values[0])+","+std::to_string(values[1])+","+std::to_string(values[2])+","+std::to_string(values[3])+"]}"];
}
// The runtime takes cull mode, winding and fill mode together; this encoder keeps Metal's defaults until each is set.
- (void)raster {
 NSNumber *cull=objc_getAssociatedObject(self,"cull"),*winding=objc_getAssociatedObject(self,"winding"),*fill=objc_getAssociatedObject(self,"fill");
 [self check:nmb_render_set_raster(self.handle,cull.unsignedLongLongValue,winding.unsignedLongLongValue,fill.unsignedLongLongValue) what:"raster state"];
 if(self.capture)self.capture->raster="["+std::to_string(cull.unsignedLongLongValue)+","+std::to_string(winding.unsignedLongLongValue)+","+std::to_string(fill.unsignedLongLongValue)+"]";
}
- (void)setCullMode:(MTLCullMode)m {objc_setAssociatedObject(self,"cull",@(m),OBJC_ASSOCIATION_RETAIN);[self raster];if(self.frame)[self frameCommand:"{\"cull\":"+std::to_string((unsigned long long)(m))+"}"];}
- (void)setFrontFacingWinding:(MTLWinding)w {objc_setAssociatedObject(self,"winding",@(w),OBJC_ASSOCIATION_RETAIN);[self raster];if(self.frame)[self frameCommand:"{\"winding\":"+std::to_string((unsigned long long)(w))+"}"];}
- (void)setTriangleFillMode:(MTLTriangleFillMode)f {objc_setAssociatedObject(self,"fill",@(f),OBJC_ASSOCIATION_RETAIN);[self raster];if(self.frame)[self frameCommand:"{\"fill\":"+std::to_string((unsigned long long)(f))+"}"];}
- (void)setDepthBias:(float)bias slopeScale:(float)slope clamp:(float)clamp {
 [self check:nmb_render_set_depth_bias(self.handle,bias,slope,clamp) what:"setDepthBias"];
 if(self.frame)[self frameCommand:"{\"depthBias\":["+fcap::num(bias)+","+fcap::num(slope)+","+fcap::num(clamp)+"]}"];
}
- (void)setLineWidth:(float)width {[self check:nmb_render_set_line_width(self.handle,width) what:"setLineWidth"];}
- (void)setVisibilityResultMode:(MTLVisibilityResultMode)mode offset:(NSUInteger)offset {
 [self check:nmb_render_set_visibility_mode(self.handle,mode,offset) what:"setVisibilityResultMode"];
}
- (void)setFragmentTexture:(id<MTLTexture>)t atTextureIndex:(NSUInteger)ti samplerState:(id<MTLSamplerState>)s atSamplerIndex:(NSUInteger)si {
 [self texture:t index:ti stage:2];[self sampler:s index:si stage:2];
}
// Tile shaders and imageblock tile memory (Apple GPU families 4 and up) are not offered: supportsTileShaders answers no, and a call that
// reaches one anyway fails its command buffer.
- (NSUInteger)tileWidth {return 0;}
- (NSUInteger)tileHeight {return 0;}
- (void)setTileTexture:(id<MTLTexture>)t atIndex:(NSUInteger)i {(void)t;(void)i;[self.commandBuffer refuse:@"tile shaders are not supported"];}
- (void)setTileBytes:(const void *)p length:(NSUInteger)n atIndex:(NSUInteger)i {(void)p;(void)n;(void)i;[self.commandBuffer refuse:@"tile shaders are not supported"];}
- (void)dispatchThreadsPerTile:(MTLSize)size {(void)size;[self.commandBuffer refuse:@"tile shaders are not supported"];}
- (void)dispatchThreadsPerTile:(MTLSize)size inRegion:(MTLRegion)region {(void)size;(void)region;[self.commandBuffer refuse:@"tile shaders are not supported"];}
- (void)setStencilReferenceValue:(uint32_t)v {[self check:nmb_render_set_stencil_reference(self.handle,v,v) what:"setStencilReferenceValue"];if(self.frame)[self frameCommand:"{\"stencilRef\":"+std::to_string(v)+"}"];}
- (void)setStencilFrontReferenceValue:(uint32_t)f backReferenceValue:(uint32_t)b {
 [self check:nmb_render_set_stencil_reference(self.handle,f,b) what:"setStencilReferenceValues"];
 if(self.frame)[self frameCommand:"{\"stencilRef\":"+std::to_string(f)+"}"];
}
- (void)setBlendColorRed:(float)r green:(float)g blue:(float)b alpha:(float)a {
 const float c[4]={r,g,b,a};[self check:nmb_render_set_blend_color(self.handle,c) what:"setBlendColor"];
 if(self.frame)[self frameCommand:"{\"blendColor\":["+fcap::num(r)+","+fcap::num(g)+","+fcap::num(b)+","+fcap::num(a)+"]}"];
}
- (void)drawPrimitives:(MTLPrimitiveType)t vertexStart:(NSUInteger)s vertexCount:(NSUInteger)n instanceCount:(NSUInteger)i baseInstance:(NSUInteger)b {
 gDraws++;self.commandBuffer->encodedWork.draws++;
 [self check:nmb_render_draw(self.handle,t,(uint32_t)s,(uint32_t)n,(uint32_t)i,(uint32_t)b) what:"drawPrimitives"];
 if(self.capture)pcap::draw(*self.capture,t,-1);
 if(self.frame)[self frameCommand:"{\"draw\":{\"primitive\":"+std::to_string((unsigned long long)(t))+",\"start\":"+std::to_string((unsigned long long)(s))+",\"count\":"+std::to_string((unsigned long long)(n))+",\"instances\":"+std::to_string((unsigned long long)(i))+",\"baseInstance\":"+std::to_string((unsigned long long)(b))+"}}"];
}
- (void)drawPrimitives:(MTLPrimitiveType)t vertexStart:(NSUInteger)s vertexCount:(NSUInteger)n instanceCount:(NSUInteger)i {
 [self drawPrimitives:t vertexStart:s vertexCount:n instanceCount:i baseInstance:0];
}
- (void)drawPrimitives:(MTLPrimitiveType)t vertexStart:(NSUInteger)s vertexCount:(NSUInteger)n {
 [self drawPrimitives:t vertexStart:s vertexCount:n instanceCount:1 baseInstance:0];
}
- (void)drawIndexedPrimitives:(MTLPrimitiveType)t indexCount:(NSUInteger)n indexType:(MTLIndexType)k indexBuffer:(id<MTLBuffer>)b indexBufferOffset:(NSUInteger)o
                instanceCount:(NSUInteger)i baseVertex:(NSInteger)v baseInstance:(NSUInteger)bi {
 if(![(id)b isKindOfClass:[NBBuffer class]]){[self.commandBuffer refuse:@"foreign index buffer"];return;}
 [self.commandBuffer.resourceOwners addObject:b];
 gDraws++;self.commandBuffer->encodedWork.draws++;
 [self check:nmb_render_draw_indexed(self.handle,t,(uint32_t)n,k,[(NBBuffer *)b handle],o,(uint32_t)i,(int32_t)v,(uint32_t)bi) what:"drawIndexedPrimitives"];
 if(self.capture)pcap::draw(*self.capture,t,(long)k);
 if(self.frame)[self frameCommand:"{\"drawIndexed\":{\"primitive\":"+std::to_string((unsigned long long)(t))+",\"count\":"+std::to_string((unsigned long long)(n))+",\"indexType\":"+std::to_string((unsigned long long)(k))+",\"buffer\":\""+
     fcap::buffer((NBBuffer *)b,self.commandBuffer.frame.get())+"\",\"offset\":"+std::to_string((unsigned long long)(o))+",\"instances\":"+std::to_string((unsigned long long)(i))+",\"baseVertex\":"+std::to_string((long long)v)+",\"baseInstance\":"+std::to_string((unsigned long long)(bi))+"}}"];
}
- (void)drawIndexedPrimitives:(MTLPrimitiveType)t indexCount:(NSUInteger)n indexType:(MTLIndexType)k indexBuffer:(id<MTLBuffer>)b indexBufferOffset:(NSUInteger)o
                instanceCount:(NSUInteger)i {
 [self drawIndexedPrimitives:t indexCount:n indexType:k indexBuffer:b indexBufferOffset:o instanceCount:i baseVertex:0 baseInstance:0];
}
- (void)drawIndexedPrimitives:(MTLPrimitiveType)t indexCount:(NSUInteger)n indexType:(MTLIndexType)k indexBuffer:(id<MTLBuffer>)b indexBufferOffset:(NSUInteger)o {
 [self drawIndexedPrimitives:t indexCount:n indexType:k indexBuffer:b indexBufferOffset:o instanceCount:1 baseVertex:0 baseInstance:0];
}
- (void)useResource:(id<MTLResource>)r usage:(MTLResourceUsage)u stages:(MTLRenderStages)s {(void)r;(void)u;(void)s;}
- (void)updateFence:(id<MTLFence>)fence afterStages:(MTLRenderStages)stages {[self fence:fence stages:stages update:true];}
- (void)waitForFence:(id<MTLFence>)fence beforeStages:(MTLRenderStages)stages {[self fence:fence stages:stages update:false];}
@end

@interface NBBlitEncoder : NBEncoder
@end
@implementation NBBlitEncoder
// The arguments are passed as a plain array: an NSArray literal raises on a nil element, and a nil resource (a refused allocation) must fail the
// command buffer like any invalid argument, not abort the host process.
#define NB_OBJECTS(...) ((__unsafe_unretained id[]){__VA_ARGS__})
- (bool)buffers:(__unsafe_unretained id *)objects count:(size_t)count {
 for(size_t i=0;i<count;i++)if(![objects[i] isKindOfClass:[NBBuffer class]]){[self.commandBuffer refuse:@"missing or foreign blit buffer"];return false;}
 for(size_t i=0;i<count;i++)[self.commandBuffer.resourceOwners addObject:objects[i]];
 return true;
}
- (bool)textures:(__unsafe_unretained id *)objects count:(size_t)count {
 for(size_t i=0;i<count;i++) {
  id o=objects[i];
  if(![o isKindOfClass:[NBTexture class]]){[self.commandBuffer refuse:@"missing or foreign blit texture"];return false;}
  [self.commandBuffer.resourceOwners addObject:o];
  if([(NBTexture *)o surface]){[self.commandBuffer.surfaces addObject:o];noteSurfaceContents((NBTexture *)o);}
 }
 return true;
}
- (void)copyFromBuffer:(id<MTLBuffer>)s sourceOffset:(NSUInteger)so toBuffer:(id<MTLBuffer>)d destinationOffset:(NSUInteger)dO size:(NSUInteger)n {
 if(![self buffers:NB_OBJECTS(s,d) count:2])return;
 [self check:nmb_blit_copy_buffer(self.handle,[(NBBuffer *)s handle],so,[(NBBuffer *)d handle],dO,n) what:"copyFromBuffer:toBuffer:"];
 if(self.frame)[self frameCommand:"{\"copyBuffer\":{\"from\":\""+fcap::buffer((NBBuffer *)s,self.commandBuffer.frame.get())+"\",\"fromOffset\":"+std::to_string((unsigned long long)(so))+",\"to\":\""+fcap::buffer((NBBuffer *)d,self.commandBuffer.frame.get())+"\",\"toOffset\":"+std::to_string((unsigned long long)(dO))+",\"size\":"+std::to_string((unsigned long long)(n))+"}}"];
}
- (void)fillBuffer:(id<MTLBuffer>)b range:(NSRange)r value:(uint8_t)v {
 if(![self buffers:NB_OBJECTS(b) count:1])return;
 [self check:nmb_blit_fill_buffer(self.handle,[(NBBuffer *)b handle],r.location,r.location+r.length,v) what:"fillBuffer"];
 if(self.frame)[self frameCommand:"{\"fillBuffer\":{\"buffer\":\""+fcap::buffer((NBBuffer *)b,self.commandBuffer.frame.get())+"\",\"offset\":"+std::to_string((unsigned long long)(r.location))+",\"length\":"+std::to_string((unsigned long long)(r.length))+",\"value\":"+std::to_string((unsigned long long)(v))+"}}"];
}
- (void)copyFromTexture:(id<MTLTexture>)s sourceSlice:(NSUInteger)ss sourceLevel:(NSUInteger)sl sourceOrigin:(MTLOrigin)so sourceSize:(MTLSize)z
              toTexture:(id<MTLTexture>)d destinationSlice:(NSUInteger)ds destinationLevel:(NSUInteger)dl destinationOrigin:(MTLOrigin)dO {
 if(![self textures:NB_OBJECTS(s,d) count:2])return;
 const uint64_t a[3]={so.x,so.y,so.z},size[3]={z.width,z.height,z.depth},b[3]={dO.x,dO.y,dO.z};
 [self check:nmb_blit_copy_texture(self.handle,[(NBTexture *)s handle],ss,sl,a,size,[(NBTexture *)d handle],ds,dl,b) what:"copyFromTexture:toTexture:"];
 if(self.frame)[self frameCommand:"{\"copyTexture\":{\"from\":\""+fcap::texture((NBTexture *)s,true,self.commandBuffer.frame.get())+"\",\"fromSlice\":"+std::to_string((unsigned long long)(ss))+",\"fromLevel\":"+std::to_string((unsigned long long)(sl))+",\"fromOrigin\":"+fcap::triple(a[0],a[1],a[2])+
     ",\"size\":"+fcap::triple(size[0],size[1],size[2])+",\"to\":\""+fcap::texture((NBTexture *)d,false,self.commandBuffer.frame.get())+"\",\"toSlice\":"+std::to_string((unsigned long long)(ds))+",\"toLevel\":"+std::to_string((unsigned long long)(dl))+",\"toOrigin\":"+fcap::triple(b[0],b[1],b[2])+"}}"];
}
- (void)copyFromTexture:(id<MTLTexture>)s toTexture:(id<MTLTexture>)d {
 if(![self textures:NB_OBJECTS(s,d) count:2])return;
 const NSUInteger levels=MIN(s.mipmapLevelCount,d.mipmapLevelCount),slices=MIN(s.arrayLength,d.arrayLength);
 for(NSUInteger slice=0;slice<slices;slice++)for(NSUInteger level=0;level<levels;level++) {
  MTLSize size=MTLSizeMake(MAX(s.width>>level,(NSUInteger)1),MAX(s.height>>level,(NSUInteger)1),MAX(s.depth>>level,(NSUInteger)1));
  [self copyFromTexture:s sourceSlice:slice sourceLevel:level sourceOrigin:MTLOriginMake(0,0,0) sourceSize:size toTexture:d destinationSlice:slice
       destinationLevel:level destinationOrigin:MTLOriginMake(0,0,0)];
 }
}
- (void)copyFromBuffer:(id<MTLBuffer>)s sourceOffset:(NSUInteger)o sourceBytesPerRow:(NSUInteger)row sourceBytesPerImage:(NSUInteger)image
            sourceSize:(MTLSize)z toTexture:(id<MTLTexture>)d destinationSlice:(NSUInteger)slice destinationLevel:(NSUInteger)level
     destinationOrigin:(MTLOrigin)dO {
 if(![self buffers:NB_OBJECTS(s) count:1] || ![self textures:NB_OBJECTS(d) count:1])return;
 const uint64_t size[3]={z.width,z.height,z.depth},b[3]={dO.x,dO.y,dO.z};
 [self check:nmb_blit_buffer_to_texture(self.handle,[(NBBuffer *)s handle],o,row,image,size,[(NBTexture *)d handle],slice,level,b) what:"copyFromBuffer:toTexture:"];
 if(self.frame)[self frameCommand:"{\"bufferToTexture\":{\"buffer\":\""+fcap::buffer((NBBuffer *)s,self.commandBuffer.frame.get())+"\",\"offset\":"+std::to_string((unsigned long long)(o))+",\"row\":"+std::to_string((unsigned long long)(row))+",\"image\":"+std::to_string((unsigned long long)(image))+
     ",\"size\":"+fcap::triple(size[0],size[1],size[2])+",\"texture\":\""+fcap::texture((NBTexture *)d,false,self.commandBuffer.frame.get())+"\",\"slice\":"+std::to_string((unsigned long long)(slice))+",\"level\":"+std::to_string((unsigned long long)(level))+",\"origin\":"+fcap::triple(b[0],b[1],b[2])+"}}"];
}
- (void)copyFromTexture:(id<MTLTexture>)s sourceSlice:(NSUInteger)slice sourceLevel:(NSUInteger)level sourceOrigin:(MTLOrigin)so sourceSize:(MTLSize)z
               toBuffer:(id<MTLBuffer>)d destinationOffset:(NSUInteger)o destinationBytesPerRow:(NSUInteger)row destinationBytesPerImage:(NSUInteger)image {
 if(![self textures:NB_OBJECTS(s) count:1] || ![self buffers:NB_OBJECTS(d) count:1])return;
 const uint64_t a[3]={so.x,so.y,so.z},size[3]={z.width,z.height,z.depth};
 [self check:nmb_blit_texture_to_buffer(self.handle,[(NBTexture *)s handle],slice,level,a,size,[(NBBuffer *)d handle],o,row,image) what:"copyFromTexture:toBuffer:"];
 if(self.frame)[self frameCommand:"{\"textureToBuffer\":{\"texture\":\""+fcap::texture((NBTexture *)s,true,self.commandBuffer.frame.get())+"\",\"slice\":"+std::to_string((unsigned long long)(slice))+",\"level\":"+std::to_string((unsigned long long)(level))+",\"origin\":"+fcap::triple(a[0],a[1],a[2])+
     ",\"size\":"+fcap::triple(size[0],size[1],size[2])+",\"buffer\":\""+fcap::buffer((NBBuffer *)d,self.commandBuffer.frame.get())+"\",\"offset\":"+std::to_string((unsigned long long)(o))+",\"row\":"+std::to_string((unsigned long long)(row))+",\"image\":"+std::to_string((unsigned long long)(image))+"}}"];
}
- (void)generateMipmapsForTexture:(id<MTLTexture>)t {
 if(![self textures:NB_OBJECTS(t) count:1])return;
 [self check:nmb_blit_generate_mipmaps(self.handle,[(NBTexture *)t handle]) what:"generateMipmapsForTexture"];
 if(self.frame)[self frameCommand:"{\"generateMipmaps\":\""+fcap::texture((NBTexture *)t,true,self.commandBuffer.frame.get())+"\"}"];
}
- (void)synchronizeResource:(id<MTLResource>)r {(void)r;}
@end

static void completed(void *context);
static void scheduled(void *context);
// The frame trace's view of the frontend's totals (FrameTrace.h): pipeline/library/function creation, waits for GPU work, CPU cache
// maintenance and host copies of imports and surfaces.
static void traceTotals(uint64_t &pipeUs,uint64_t &pipeN,uint64_t &waitUs,uint64_t &flushUs,uint64_t &flushBytes) {
 pipeUs=gTimePipeline.total+gTimeLibrary.total+gTimeFunction.total;pipeN=gTimePipeline.count+gTimeLibrary.count+gTimeFunction.count;
 waitUs=gTimeResourceWait.total+gTimeWaitCompleted.total+gTimeWaitScheduled.total;
 flushUs=gTimeImportFlush.total+gTimeSurfaceFlush.total;flushBytes=gImportFlushBytes+gSurfaceFlushBytes+gHostCopyBytes;
}
@implementation NBCommandBuffer
- (instancetype)init {
 if((self=[super init])){_scheduledHandlers=[NSMutableArray array];_completedHandlers=[NSMutableArray array];_surfaces=[NSMutableSet set];_resourceOwners=[NSMutableSet set];_handlers=dispatch_group_create();_schedulingHandlers=dispatch_group_create();}
 if(self){createdUs=nowUs();if(cpuprof::on()){cpuprof::note();cpuprof::opened();cpuOpen=true;}if(ftrace::on())traceTotals(tracePipeUs,tracePipeN,traceWaitUs,traceFlushUs,traceFlushBytes);}
 return self;
}
- (void)dealloc {if(cpuOpen)cpuprof::closed();nmb_command_buffer_release(_handle);}
- (id<MTLCommandQueue>)commandQueue {return (id<MTLCommandQueue>)_queue;}
- (BOOL)retainedReferences {return YES;}
// Protected content is not supported (the runtime's AppleOnlyCall::SetProtectionOptions takes Metal's default only): any other option fails
// the command buffer.
- (uint64_t)protectionOptions {return 0;}
- (void)setProtectionOptions:(uint64_t)options {if(options)[self refuse:[NSString stringWithFormat:@"protection options 0x%llx",options]];}
- (void)refuse:(NSString *)what {
 if(!self.encodeFailure){self.encodeFailure=what;NBLOG(@"native-metal: command buffer refused: %@",what);}
}
- (MTLCommandBufferStatus)status {
 if(self.failedLocally)return MTLCommandBufferStatusError;
 return (MTLCommandBufferStatus)nmb_command_buffer_status(_handle);
}
- (NSError *)error {
 if(self.failedLocally)return nbError(self.encodeFailure);
 if(self.status!=MTLCommandBufferStatusError)return nil;
 char text[512]={};nmb_command_buffer_error(_handle,text,sizeof text);
 return nbError([NSString stringWithUTF8String:text]);
}
// Debug groups and cache hints name work for tools and the driver's caches; they change nothing the GPU computes here.
- (void)pushDebugGroup:(NSString *)name {(void)name;}
- (void)popDebugGroup {}
- (void)encodeCacheHintTag:(uint64_t)tag resourceGroups:(const id __unsafe_unretained [])groups count:(NSUInteger)count {(void)tag;(void)groups;(void)count;}
- (void)encodeCacheHintFinalize:(uint64_t)tag resourceGroups:(const id __unsafe_unretained [])groups count:(NSUInteger)count {(void)tag;(void)groups;(void)count;}
// The kernel driver's own submission times are not recorded: 0 is Metal's answer where they are unavailable.
- (CFTimeInterval)kernelStartTime {return 0;}
- (CFTimeInterval)kernelEndTime {return 0;}
- (BOOL)commitAndWaitUntilSubmitted {[(id<MTLCommandBuffer>)self commit];[self waitUntilScheduled];return !self.failedLocally;}
- (CFTimeInterval)GPUStartTime {return nmb_command_buffer_gpu_time(_handle,false);}
- (CFTimeInterval)GPUEndTime {return nmb_command_buffer_gpu_time(_handle,true);}
- (id)encoder:(NBEncoder *)encoder handle:(NmbEncoder *)handle what:(NSString *)what {
 if(self.committed){[self refuse:[what stringByAppendingString:@" after commit"]];return nil;}
 if(!handle){[self refuse:[NSString stringWithFormat:@"%@: %@",what,lastError()]];return nil;}
 encoder.handle=handle;encoder.commandBuffer=self;
 if(_frame){auto f=std::make_shared<FrameEncoder>();f->kind=[encoder isKindOfClass:[NBRenderEncoder class]]?"render":[encoder isKindOfClass:[NBComputeEncoder class]]?"compute":"blit";encoder.frame=f;}
 return encoder;
}
- (id<MTLComputeCommandEncoder>)computeCommandEncoder {
 return [self encoder:[NBComputeEncoder new] handle:self.committed?nullptr:nmb_compute_encoder_new(_handle) what:@"computeCommandEncoder"];
}
- (id<MTLComputeCommandEncoder>)computeCommandEncoderWithDispatchType:(MTLDispatchType)type {
 if(type!=MTLDispatchTypeSerial){[self refuse:@"concurrent compute dispatch"];return nil;}
 return [self computeCommandEncoder];
}
- (id<MTLBlitCommandEncoder>)blitCommandEncoder {
 gBlits++;encodedWork.blits++;
 return [self encoder:[NBBlitEncoder new] handle:self.committed?nullptr:nmb_blit_encoder_new(_handle) what:@"blitCommandEncoder"];
}
- (id<MTLRenderCommandEncoder>)renderCommandEncoderWithDescriptor:(MTLRenderPassDescriptor *)d {
 NmbRenderPassDescriptor pass={};
 {
  id<MTLTexture> first=d.colorAttachments[0].texture;
  if(first) {
   const uint64_t pixels=(uint64_t)first.width*first.height;
   gPasses++;gPassPixels+=pixels;encodedWork.passes++;encodedWork.pixels+=pixels;
   uint64_t seen=gBiggestPass.load();while(pixels>seen && !gBiggestPass.compare_exchange_weak(seen,pixels)){}
   if(pixels>=2000000)gFullScreenPasses++;
  }
 }
 auto attachment=[&](MTLRenderPassAttachmentDescriptor *a,NmbPassAttachment &out)->bool {
  if(!a.texture)return true;
  if(![(id)a.texture isKindOfClass:[NBTexture class]])return false;
  NBTexture *texture=(NBTexture *)a.texture;if(texture.surface){[_surfaces addObject:texture];noteSurfaceContents(texture);}
  [_resourceOwners addObject:texture];
  out.texture=texture.handle;out.level=a.level;out.slice=a.slice;out.depth_plane=a.depthPlane;out.load=a.loadAction;out.store=a.storeAction;
  return true;
 };
 for(unsigned i=0;i<8;i++) {
  MTLRenderPassColorAttachmentDescriptor *c=d.colorAttachments[i];
  if(!attachment(c,pass.colors[i])){[self refuse:@"foreign color attachment"];return nil;}
  pass.colors[i].clear[0]=c.clearColor.red;pass.colors[i].clear[1]=c.clearColor.green;pass.colors[i].clear[2]=c.clearColor.blue;pass.colors[i].clear[3]=c.clearColor.alpha;
 }
 if(!attachment(d.depthAttachment,pass.depth) || !attachment(d.stencilAttachment,pass.stencil)){[self refuse:@"foreign depth/stencil attachment"];return nil;}
 pass.depth.clear[0]=d.depthAttachment.clearDepth;pass.stencil.clear[0]=d.stencilAttachment.clearStencil;
 pass.width=d.renderTargetWidth;pass.height=d.renderTargetHeight;pass.array_length=d.renderTargetArrayLength;pass.samples=d.defaultRasterSampleCount;
 if(d.visibilityResultBuffer) {
  if(![(id)d.visibilityResultBuffer isKindOfClass:[NBBuffer class]]){[self refuse:@"foreign visibility result buffer"];return nil;}
  [_resourceOwners addObject:d.visibilityResultBuffer];
  pass.visibility=[(NBBuffer *)d.visibilityResultBuffer handle];
 }
 NBRenderEncoder *encoder=[self encoder:[NBRenderEncoder new] handle:self.committed?nullptr:nmb_render_encoder_new(_handle,&pass) what:@"renderCommandEncoder"];
 if(encoder)for(unsigned i=0;i<8;i++) {
  id<MTLTexture> t=d.colorAttachments[i].texture;
  if([(id)t isKindOfClass:[NBTexture class]]) {const uint64_t f=[(NBTexture *)t descriptor].format;if(f>=550 && f<=555)encoder.passXr|=1u<<i;}
 }
 if(encoder && encoder.frame) {
  FrameCb *cb=_frame.get();
  auto attachment=[&](MTLRenderPassAttachmentDescriptor *a,const std::string &clear)->std::string {
   if(![(id)a.texture isKindOfClass:[NBTexture class]])return "null";
   return "{\"texture\":\""+fcap::texture((NBTexture *)a.texture,a.loadAction==MTLLoadActionLoad,cb)+"\",\"load\":"+std::to_string((unsigned long)a.loadAction)+
          ",\"store\":"+std::to_string((unsigned long)a.storeAction)+",\"level\":"+std::to_string((unsigned long)a.level)+",\"slice\":"+std::to_string((unsigned long)a.slice)+",\"clear\":"+clear+"}";
  };
  std::string head="\"colors\":[";int last=-1;for(int i=0;i<8;i++)if(d.colorAttachments[i].texture)last=i;
  for(int i=0;i<=last;i++) {
   MTLClearColor c=d.colorAttachments[i].clearColor;
   head+=(i?",":"")+attachment(d.colorAttachments[i],"["+fcap::num(c.red)+","+fcap::num(c.green)+","+fcap::num(c.blue)+","+fcap::num(c.alpha)+"]");
  }
  head+="]";
  if(d.depthAttachment.texture)head+=",\"depth\":"+attachment(d.depthAttachment,fcap::num(d.depthAttachment.clearDepth));
  if(d.stencilAttachment.texture)head+=",\"stencil\":"+attachment(d.stencilAttachment,std::to_string((unsigned long)d.stencilAttachment.clearStencil));
  head+=",\"width\":"+std::to_string((unsigned long)d.renderTargetWidth)+",\"height\":"+std::to_string((unsigned long)d.renderTargetHeight);
  encoder.frame->head=head;
 }
 if(encoder && pcap::on()) {
  // The pass as the pipeline capture records it: each attachment's texture format and samples, the render target's size and samples.
  auto format=[](MTLRenderPassAttachmentDescriptor *a)->std::string {
   if(!a.texture)return "null";
   return "{\"format\":"+std::to_string((unsigned long)a.texture.pixelFormat)+",\"samples\":"+std::to_string((unsigned long)a.texture.sampleCount)+
          ",\"type\":"+std::to_string((unsigned long)a.texture.textureType)+",\"load\":"+std::to_string((unsigned long)a.loadAction)+
          ",\"store\":"+std::to_string((unsigned long)a.storeAction)+"}";
  };
  auto bindings=std::make_shared<pcap::Bindings>();
  std::string text="{\"colors\":[";
  int last=-1;for(int i=0;i<8;i++)if(d.colorAttachments[i].texture)last=i;
  for(int i=0;i<=last;i++){if(i)text+=",";text+=format(d.colorAttachments[i]);}
  text+="],\"depth\":"+format(d.depthAttachment)+",\"stencil\":"+format(d.stencilAttachment)+",\"width\":"+std::to_string((unsigned long)d.renderTargetWidth)+
        ",\"height\":"+std::to_string((unsigned long)d.renderTargetHeight)+",\"samples\":"+std::to_string((unsigned long)d.defaultRasterSampleCount)+"}";
  bindings->pass=text;encoder.capture=bindings;
 }
 return (id<MTLRenderCommandEncoder>)encoder;
}
- (void)addScheduledHandler:(MTLCommandBufferHandler)block {if(block)@synchronized(self){[_scheduledHandlers addObject:[block copy]];}}
- (void)addCompletedHandler:(MTLCommandBufferHandler)block {if(block)@synchronized(self){[_completedHandlers addObject:[block copy]];}}
// A drawable (RenderBox's CAMetalLayer path) is presented once the GPU has written it: this runtime's scheduling comes before completion.
- (void)presentDrawable:(id<MTLDrawable>)drawable {[self addCompletedHandler:^(id<MTLCommandBuffer>){[drawable present];}];}
- (void)presentDrawable:(id<MTLDrawable>)drawable atTime:(CFTimeInterval)time {[self addCompletedHandler:^(id<MTLCommandBuffer>){[drawable presentAtTime:time];}];}
- (void)enqueue {}
- (void)event:(id<MTLEvent>)event value:(uint64_t)value signal:(bool)signal {
 if(![(id)event isKindOfClass:[NBEvent class]]){[self refuse:@"foreign event"];return;}
 if(self.committed){[self refuse:@"event encoded after commit"];return;}
 NmbEvent *handle=[(NBEvent *)event handle];
 if(!(signal?nmb_command_buffer_signal_event(_handle,handle,value):nmb_command_buffer_wait_event(_handle,handle,value)))
  [self refuse:[NSString stringWithFormat:@"%s: %@",signal?"encodeSignalEvent":"encodeWaitForEvent",lastError()]];
}
- (void)encodeSignalEvent:(id<MTLEvent>)event value:(uint64_t)value {[self event:event value:value signal:true];}
- (void)encodeWaitForEvent:(id<MTLEvent>)event value:(uint64_t)value {[self event:event value:value signal:false];}
// The runtime has submitted the work. Deliver scheduling without waiting for GPU completion.
- (void)didSchedule {
 {
  const uint64_t took=nowUs()-self.committedUs;
  std::lock_guard<std::mutex> lock(gCommitLock);
  CommitRecord &r=gCommitRing[self.commitIndex%32];
  if(r.index==self.commitIndex)r.scheduledUs=took;
 }
 NSArray *handlers;
 @synchronized(self){
  _schedulingDelivered=YES;handlers=[_scheduledHandlers copy];[_scheduledHandlers removeAllObjects];
 }
 void (^run)(void)=^{
  for(MTLCommandBufferHandler block in handlers)block((id<MTLCommandBuffer>)self);
  dispatch_group_leave(self->_schedulingHandlers);
 };
 // Run on the runtime's scheduled-handler thread, preserving its ordering and own-wait
 // deadlock checks. The queue's completionQueue routes completed handlers only.
 run();
}
// After completion (or a refusal): invalidate the CPU's view of imported surfaces, then completed handlers, once.
- (void)finish {
 flushSurfaces(_surfaces);
 // Actual GPU completion (or a local refusal that never submitted) is the
 // release point. A retained completed command must not pin every old frame.
 [_surfaces removeAllObjects];[_resourceOwners removeAllObjects];
 @synchronized(gInFlight){[gInFlight removeObject:self];}
 NSArray *completedHandlers;
 @synchronized(self){
  // A frontend refusal or a failed submission never scheduled. Unblock its waiter without inventing a scheduled event.
  if(!_schedulingDelivered){_schedulingDelivered=YES;[_scheduledHandlers removeAllObjects];dispatch_group_leave(_schedulingHandlers);}
  completedHandlers=[_completedHandlers copy];[_completedHandlers removeAllObjects];
 }
 void (^run)(void)=^{
  for(MTLCommandBufferHandler block in completedHandlers)block((id<MTLCommandBuffer>)self);
  dispatch_group_leave(self->_handlers);
 };
 dispatch_queue_t queue=_queue.completionQueue;
 if(queue)dispatch_async(queue,run);else run();
}
- (void)commit {
 if(self.committed){NBLOG(@"native-metal: commit twice");return;}
 self.committed=YES;
 if(cpuOpen){cpuOpen=false;cpuprof::closed();}
 if(_frame){fcap::commit(_frame.get());_frame=nullptr;}
 fcap::committed();
 self.committedUs=nowUs();
 {
  std::lock_guard<std::mutex> lock(gCommitLock);
  // Assign once under the same lock as publication. A concurrent commit cannot reuse this index.
  const uint32_t index=(uint32_t)(gCommits.fetch_add(1)+1);self.commitIndex=index;
  CommitRecord &r=gCommitRing[index%32];
  r={};r.index=index;r.committedUs=self.committedUs;r.ageUs=self.committedUs-createdUs;r.work=encodedWork;
 }
 dispatch_group_enter(_handlers);
 dispatch_group_enter(_schedulingHandlers);
 @synchronized(gInFlight){[gInFlight addObject:self];}
 if(!self.encodeFailure) {
  flushImports();
  flushSurfaces(_surfaces);
  // Completion owns this reference; the runtime orders the non-owning scheduled callback before it.
  void *context=(__bridge_retained void *)self;
  if(ftrace::on())[self traceTotalsAtCommit];
  if(nmb_command_buffer_on_completed(_handle,completed,context) &&
     nmb_command_buffer_on_scheduled(_handle,scheduled,context) && nmb_command_buffer_commit(_handle)) {
   if(ftrace::on())[self traceCommitted];
   return;
  }
  (void)CFBridgingRelease(context);   // refused registration/commit cannot call the registered handlers
  [self refuse:[@"commit: " stringByAppendingString:lastError()]];
 }
 self.failedLocally=YES;gCommitFailures++;
 dispatch_async(dispatch_get_global_queue(QOS_CLASS_DEFAULT,0),^{[self finish];});
}
- (void)waitUntilCompleted {
 if(!self.committed)return;
 const uint64_t began=nowUs();
 if(!self.failedLocally)nmb_command_buffer_wait(_handle,0);
 dispatch_group_wait(_handlers,DISPATCH_TIME_FOREVER);
 gTimeWaitCompleted.add(began);
 if(ftrace::on())ftrace::linef("W,%llu,%llu,0,%u",(unsigned long long)began,(unsigned long long)(nowUs()-began),self.commitIndex);
}
// The frame trace: the totals' growth while the command buffer was encoded and its imports flushed replaces their snapshot.
- (void)traceTotalsAtCommit {
 uint64_t pipeUs,pipeN,waitUs,flushUs,flushBytes;traceTotals(pipeUs,pipeN,waitUs,flushUs,flushBytes);
 tracePipeUs=pipeUs-tracePipeUs;tracePipeN=pipeN-tracePipeN;traceWaitUs=waitUs-traceWaitUs;traceFlushUs=flushUs-traceFlushUs;traceFlushBytes=flushBytes-traceFlushBytes;
}
- (void)traceCommitted {traceCommittedUs=nowUs();if(++traceJoin==2)[self traceLine];}
- (void)traceCompleted:(uint64_t)doneUs {traceDoneUs=doneUs;traceHandledUs=nowUs();if(++traceJoin==2)[self traceLine];}
// The frame trace's C line (FrameTrace.h).
- (void)traceLine {
 const uint64_t doneUs=traceDoneUs,handled=traceHandledUs;
 const double ks=nmb_command_buffer_kernel_time(_handle,false),ke=nmb_command_buffer_kernel_time(_handle,true);
 const double gs=nmb_command_buffer_gpu_time(_handle,false),ge=nmb_command_buffer_gpu_time(_handle,true);
 auto us=[](double seconds)->unsigned long long {return seconds>0?(unsigned long long)(seconds*1e6+0.5):0;};
 const EncodedWork &w=encodedWork;
 ftrace::linef("C,%u,%u,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%u,%u,%u,%u,%llu,%llu,%llu,%llu,%llu,%llu,%s",self.commitIndex,_queue.traceSerial,
  (unsigned long long)createdUs,(unsigned long long)self.committedUs,(unsigned long long)traceCommittedUs,us(ks),us(ke),us(gs),us(ge),
  (unsigned long long)doneUs,(unsigned long long)handled,w.passes,w.draws,w.dispatches,w.blits,(unsigned long long)w.pixels,
  (unsigned long long)tracePipeUs,(unsigned long long)tracePipeN,(unsigned long long)traceWaitUs,(unsigned long long)traceFlushUs,
  (unsigned long long)(traceFlushBytes>>10),ftrace::field((self.label?:_queue.label).UTF8String).c_str());
}
- (void)waitUntilScheduled {
 if(!self.committed)return;
 const uint64_t began=nowUs();
 if(!self.failedLocally && !nmb_command_buffer_wait_scheduled(_handle)){
  NBLOG(@"native-metal: waitUntilScheduled refused: %@",lastError());return;
 }
 dispatch_group_wait(_schedulingHandlers,DISPATCH_TIME_FOREVER);
 gTimeWaitScheduled.add(began);
 if(ftrace::on())ftrace::linef("W,%llu,%llu,1,%u",(unsigned long long)began,(unsigned long long)(nowUs()-began),self.commitIndex);
}
@end
static void scheduled(void *context) {[(__bridge NBCommandBuffer *)context didSchedule];}
// The runtime's completion: the command buffer's own reference, taken at commit, is given back here.
static void completed(void *context) {
 gCompleted++;
 NBCommandBuffer *commandBuffer=(__bridge_transfer NBCommandBuffer *)context;
 {
  const uint64_t took=nowUs()-commandBuffer.committedUs;
  gTimeEncodeToCommit.count++;gTimeEncodeToCommit.total+=took;
  uint64_t seen=gTimeEncodeToCommit.slowest.load();while(took>seen && !gTimeEncodeToCommit.slowest.compare_exchange_weak(seen,took)){}
  std::lock_guard<std::mutex> lock(gCommitLock);
  CommitRecord &r=gCommitRing[commandBuffer.commitIndex%32];
  if(r.index==commandBuffer.commitIndex)r.callbackUs=took;
 }
 const uint64_t doneUs=ftrace::on()?nowUs():0;
 [commandBuffer finish];
 if(doneUs)[commandBuffer traceCompleted:doneUs];
}

@implementation NBQueue
- (void)dealloc {nmb_queue_release(_handle);}
- (id<MTLCommandBuffer>)commandBuffer {
 NmbCommandBuffer *handle=nmb_command_buffer_new(_handle);
 if(!handle){NBLOG(@"native-metal: commandBuffer refused: %@",lastError());return nil;}
 NBCommandBuffer *commandBuffer=[NBCommandBuffer new];commandBuffer.handle=handle;commandBuffer.queue=self;
 {static std::atomic<uint64_t> serials{0};commandBuffer->serial=++serials;}
 if(const long frame=fcap::begin();frame>=0){auto record=std::make_shared<FrameCb>();record->index=frame;commandBuffer.frame=record;}
 return (id<MTLCommandBuffer>)commandBuffer;
}
- (id<MTLCommandBuffer>)commandBufferWithUnretainedReferences {return [self commandBuffer];}
- (void)setSubmissionQueue:(dispatch_queue_t)queue {(void)queue;}   // submission happens on the caller's thread here
// setCompletionQueue: is the property's setter.
// GPU priorities are not applied (one hardware queue): reported as not applied.
- (BOOL)setGPUPriority:(NSUInteger)priority {(void)priority;return NO;}
- (BOOL)setBackgroundGPUPriority:(NSUInteger)priority {(void)priority;return NO;}
// What a new Metal queue reports (MTLIOAccelCommandQueue: normal 1, background 3). RenderBox and Core Image read both before any frame.
- (NSUInteger)getGPUPriority {return 1;}
- (NSUInteger)getBackgroundGPUPriority {return 3;}
- (id<MTLCommandBuffer>)commandBufferWithDescriptor:(MTLCommandBufferDescriptor *)d {(void)d;return [self commandBuffer];}
@end

// ---- the device ----
template <typename T> static T failWith(NSError **error,NSString *what,T value) {
 noteProblem([NSString stringWithFormat:@"%@: %@",what,lastError()]);
 if(error)*error=nbError([NSString stringWithFormat:@"%@: %@",what,lastError()]);
 return value;
}
// Which of QuartzCore's paths asked for a pipeline (int-153: every one was made synchronously on its render thread and no generic pipeline
// was ever asked for): its return addresses as offsets from its image header (header 0x189429000 in the 21H461 cache, so offset +
// 0x189429000 is the address in that image's disassembly), the global feature flags it ORs into every context's flags, and its
// CA_DISABLE_GENERIC_SHADERS, CA_FORCE_GENERIC_SHADERS and CA_DISABLE_PRECOMPILED_PIPELINES debug options (ca_debug_options +0x55..0x57,
// read only where dladdr places that address in QuartzCore).
static std::string quartzCoreCallers() {
 void *frames[24];const int n=backtrace(frames,24);
 std::string out="qc";char item[32];unsigned kept=0;uintptr_t header=0;
 for(int i=0;i<n && kept<8;i++) {
  Dl_info info{};
  if(!dladdr(frames[i],&info) || !info.dli_fname || !strstr(info.dli_fname,"/QuartzCore"))continue;
  header=(uintptr_t)info.dli_fbase;
  snprintf(item,sizeof item," %lx",(unsigned long)((uintptr_t)frames[i]-header));out+=item;++kept;
 }
 // The context's own flags (int-154: the draws never took the generic path, which only bit 18 of MetalContext +0x1a0 or the debug byte
 // can cause, and the debug byte was 0). MetalContext::create_pipeline_state (header +0x143000) saves its caller's x19 at fp-8, and its
 // caller MetalContext::draw keeps `this` in x19: the frame record whose return address is +0x143214 links to that frame. Read with
 // vm_read so a different stack layout reports ctx=? instead of faulting.
 // The context and debug-option reads below use 21H461 QuartzCore offsets; another image reports only the return addresses.
 if(header && !targetbuild::is21H461()) {out+=" ctx=- dbg=-";header=0;}
 if(header) {
  uintptr_t fp=(uintptr_t)__builtin_frame_address(0),ctx=0;
  for(unsigned depth=0;fp && depth<24;++depth) {
   uintptr_t record[2]={};vm_size_t got=0;
   if(vm_read_overwrite(mach_task_self(),fp,sizeof record,(vm_address_t)record,&got)!=KERN_SUCCESS || got!=sizeof record)break;
   if(record[1]-header==0x143214) {
    uintptr_t saved=0;
    if(record[0] && vm_read_overwrite(mach_task_self(),record[0]-8,sizeof saved,(vm_address_t)&saved,&got)==KERN_SUCCESS)ctx=saved;
    break;
   }
   if(record[0]<=fp)break;
   fp=record[0];
  }
  uint32_t flags[2]={};vm_size_t got=0;
  if(ctx && vm_read_overwrite(mach_task_self(),ctx+0x1a0,4,(vm_address_t)&flags[0],&got)==KERN_SUCCESS &&
     vm_read_overwrite(mach_task_self(),ctx+0xb3c,4,(vm_address_t)&flags[1],&got)==KERN_SUCCESS) {
   snprintf(item,sizeof item," ctx=%x/%x",flags[0],flags[1]);out+=item;
  } else out+=" ctx=?";
 }
 static auto global=(uint32_t (*)(void))dlsym(RTLD_DEFAULT,"CARenderGetGlobalFeatureFlags");
 snprintf(item,sizeof item," global=%x",global?global():0xffffffffu);out+=item;
 if(header) {
  const uintptr_t options=header+(0x1f491b3c5ull-0x189429000ull);
  Dl_info info{};
  if(dladdr((void *)options,&info) && info.dli_fname && strstr(info.dli_fname,"/QuartzCore") && (uintptr_t)info.dli_fbase==header) {
   const volatile uint8_t *b=(const volatile uint8_t *)options;
   snprintf(item,sizeof item," dbg=%u%u%u",b[0],b[1],b[2]);out+=item;
  } else out+=" dbg=?";
 }
 return out;
}
// What a creation cost beyond its wall time (int-156: a translation the host does in 6 ms took 665 ms on the tablet, whose single core is
// as fast): this thread's CPU time (user and system, microseconds) and the task's page faults, page-ins and context switches. A creation
// whose CPU time is near its wall time computed; one whose faults are many paid the 4 KiB fault path; one far below it waited.
struct NBCost {
 uint64_t user=0,system=0;int64_t faults=0,pageins=0,switches=0;
 static NBCost now() {
  NBCost c;thread_basic_info_data_t t{};mach_msg_type_number_t n=THREAD_BASIC_INFO_COUNT;mach_port_t self=mach_thread_self();
  if(thread_info(self,THREAD_BASIC_INFO,(thread_info_t)&t,&n)==KERN_SUCCESS) {
   c.user=(uint64_t)t.user_time.seconds*1000000+t.user_time.microseconds;c.system=(uint64_t)t.system_time.seconds*1000000+t.system_time.microseconds;
  }
  mach_port_deallocate(mach_task_self(),self);
  task_events_info_data_t e{};n=TASK_EVENTS_INFO_COUNT;
  if(task_info(mach_task_self(),TASK_EVENTS_INFO,(task_info_t)&e,&n)==KERN_SUCCESS){c.faults=e.faults;c.pageins=e.pageins;c.switches=e.csw;}
  return c;
 }
 std::string since(const NBCost &b) const {
  char text[128];snprintf(text,sizeof text," cpu=%llu sys=%llu flt=%lld pin=%lld csw=%lld",(unsigned long long)(user+system-b.user-b.system),
      (unsigned long long)(system-b.system),(long long)(faults-b.faults),(long long)(pageins-b.pageins),(long long)(switches-b.switches));
  return text;
 }
};
// One journal line of a render pipeline creation begun at `began` (microseconds), `refused` its refusal or null. The label is the
// caller's name for the pipeline (QuartzCore's: its spec's encoded name).
static void journalRenderPipeline(MTLRenderPipelineDescriptor *d,uint64_t began,const char *refused,const std::string &cost) {
 if(gJournalPath.empty())return;
 const uint64_t ended=nowUs();uint64_t tid=0;pthread_threadid_np(nullptr,&tid);
 char head[256];
 snprintf(head,sizeof head,"R t=%llu tid=%llu main=%d us=%llu ok=%d samples=%lu topo=%lu ds=%lu/%lu",(unsigned long long)((began-gJournalBegan)/1000),
     (unsigned long long)tid,pthread_main_np(),(unsigned long long)(ended-began),refused?0:1,(unsigned long)d.rasterSampleCount,
     (unsigned long)d.inputPrimitiveTopology,(unsigned long)d.depthAttachmentPixelFormat,(unsigned long)d.stencilAttachmentPixelFormat);
 std::string line=head;
 line+=cost;
 line+=" label=";line+=d.label.length?d.label.UTF8String:"-";
 for(unsigned i=0;i<8;i++) {
  MTLRenderPipelineColorAttachmentDescriptor *a=d.colorAttachments[i];if(a.pixelFormat==MTLPixelFormatInvalid)continue;
  snprintf(head,sizeof head," c%u=%lu:%lu:%d:%lu:%lu:%lu:%lu:%lu:%lu",i,(unsigned long)a.pixelFormat,(unsigned long)a.writeMask,(int)a.blendingEnabled,
      (unsigned long)a.sourceRGBBlendFactor,(unsigned long)a.destinationRGBBlendFactor,(unsigned long)a.rgbBlendOperation,(unsigned long)a.sourceAlphaBlendFactor,
      (unsigned long)a.destinationAlphaBlendFactor,(unsigned long)a.alphaBlendOperation);
  line+=head;
 }
 if(MTLVertexDescriptor *v=d.vertexDescriptor)for(unsigned i=0;i<31;i++) {
  MTLVertexAttributeDescriptor *a=v.attributes[i];if(a.format==MTLVertexFormatInvalid)continue;
  MTLVertexBufferLayoutDescriptor *l=v.layouts[a.bufferIndex<31?a.bufferIndex:0];
  snprintf(head,sizeof head," a%u=%lu:%lu:%lu:%lu:%lu:%lu",i,(unsigned long)a.format,(unsigned long)a.offset,(unsigned long)a.bufferIndex,(unsigned long)l.stride,
      (unsigned long)l.stepFunction,(unsigned long)l.stepRate);
  line+=head;
 }
 NBFunction *vf=[(id)d.vertexFunction isKindOfClass:[NBFunction class]]?(NBFunction *)d.vertexFunction:nil;
 NBFunction *ff=[(id)d.fragmentFunction isKindOfClass:[NBFunction class]]?(NBFunction *)d.fragmentFunction:nil;
 line+=" | v ";line+=vf?vf.name.UTF8String:"-";line+=" ";line+=vf?vf.constantsText:"";
 line+=" | f ";line+=ff?ff.name.UTF8String:"-";line+=" ";line+=ff?ff.constantsText:"";
 if(refused){line+=" | refused ";line+=refused;}
 line+=" | ";line+=quartzCoreCallers();
 for(char &c:line)if(c=='\n')c=' ';
 journal(line);
}
// A render pipeline descriptor in the bridge's form; false, with the error, for a function of another device.
static bool renderDescriptor(MTLRenderPipelineDescriptor *d,NmbRenderPipelineDescriptor *p,NSError **error) {
 if(![(id)d.vertexFunction isKindOfClass:[NBFunction class]] || (d.fragmentFunction && ![(id)d.fragmentFunction isKindOfClass:[NBFunction class]])) {
  if(error)*error=nbError(@"foreign function");return false;
 }
 p->vertex=[(NBFunction *)d.vertexFunction handle];p->fragment=d.fragmentFunction?[(NBFunction *)d.fragmentFunction handle]:nullptr;
 for(unsigned i=0;i<8;i++) {
  MTLRenderPipelineColorAttachmentDescriptor *a=d.colorAttachments[i];
  if(xrFormat((uint64_t)a.pixelFormat)!=(uint64_t)a.pixelFormat)gXrPipelines++;
  p->colors[i]={xrFormat((uint64_t)a.pixelFormat),(uint64_t)a.writeMask,(uint64_t)a.blendingEnabled,(uint64_t)a.sourceRGBBlendFactor,(uint64_t)a.destinationRGBBlendFactor,
      (uint64_t)a.rgbBlendOperation,(uint64_t)a.sourceAlphaBlendFactor,(uint64_t)a.destinationAlphaBlendFactor,(uint64_t)a.alphaBlendOperation};
 }
 p->depth_format=d.depthAttachmentPixelFormat;p->stencil_format=d.stencilAttachmentPixelFormat;p->samples=d.rasterSampleCount;
 p->topology=d.inputPrimitiveTopology;p->rasterization=d.rasterizationEnabled;
 if(MTLVertexDescriptor *v=d.vertexDescriptor) {
  for(unsigned i=0;i<31;i++) {
   MTLVertexAttributeDescriptor *a=v.attributes[i];if(a.format==MTLVertexFormatInvalid)continue;
   p->vertex_inputs=1;p->attributes[i]={(uint64_t)a.format,a.offset,a.bufferIndex};
   if(a.bufferIndex<31){MTLVertexBufferLayoutDescriptor *l=v.layouts[a.bufferIndex];p->layouts[a.bufferIndex]={l.stride,(uint64_t)l.stepFunction,l.stepRate};}
  }
 }
 return true;
}

// ---- heaps ----
// Metal heaps over the runtime's: an automatic heap places each resource, a placement heap takes an offset. Resources made from one keep it
// alive (their `heap`); a heap's resources share its storage mode, CPU cache mode and (here) hazard tracking mode.
@interface NBHeap : NBObject
@property(nonatomic) NmbHeap *handle;
@property(nonatomic) MTLResourceOptions options;
@property(nonatomic) MTLHeapType heapType;
@end
@implementation NBHeap
- (void)dealloc {nmb_heap_release(_handle);}
- (NSUInteger)size {return nmb_heap_size(_handle);}
- (NSUInteger)usedSize {return nmb_heap_used_size(_handle);}
- (NSUInteger)currentAllocatedSize {return nmb_heap_allocated_size(_handle);}
- (NSUInteger)maxAvailableSizeWithAlignment:(NSUInteger)alignment {return nmb_heap_max_available_size(_handle,alignment);}
- (MTLResourceOptions)resourceOptions {return _options;}
- (MTLStorageMode)storageMode {return (MTLStorageMode)((_options&MTLResourceStorageModeMask)>>MTLResourceStorageModeShift);}
- (MTLCPUCacheMode)cpuCacheMode {return (MTLCPUCacheMode)((_options&MTLResourceCPUCacheModeMask)>>MTLResourceCPUCacheModeShift);}
- (MTLHazardTrackingMode)hazardTrackingMode {return (MTLHazardTrackingMode)((_options&MTLResourceHazardTrackingModeMask)>>MTLResourceHazardTrackingModeShift);}
- (MTLHeapType)type {return _heapType;}
- (uint64_t)protectionOptions {return 0;}
- (MTLPurgeableState)setPurgeableState:(MTLPurgeableState)state {
 const uint64_t answer=nmb_heap_set_purgeable_state(_handle,state);
 if(!answer)NBLOG(@"native-metal: heap setPurgeableState refused: %@",lastError());
 return (MTLPurgeableState)answer;
}
- (id<MTLBuffer>)bufferWithLength:(NSUInteger)length options:(MTLResourceOptions)options offset:(uint64_t)offset {
 NmbBuffer *handle=nmb_heap_new_buffer(_handle,length,options,offset);
 if(!handle){NBLOG(@"native-metal: heap buffer refused: %@",lastError());return nil;}
 NBBuffer *buffer=[NBBuffer new];buffer.handle=handle;buffer.options=options;buffer.parentHeap=(id<MTLHeap>)self;
 buffer.heapOffset=offset==UINT64_MAX?0:offset;return (id<MTLBuffer>)buffer;
}
- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)length options:(MTLResourceOptions)options {return [self bufferWithLength:length options:options offset:UINT64_MAX];}
- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)length options:(MTLResourceOptions)options offset:(NSUInteger)offset {
 return [self bufferWithLength:length options:options offset:offset];
}
- (id<MTLTexture>)textureWithDescriptor:(MTLTextureDescriptor *)d offset:(uint64_t)offset {
 NmbTextureDescriptor descriptor;if(!textureDescriptor(d,&descriptor))return nil;
 if(xrFormat(descriptor.format)!=descriptor.format){descriptor.format=xrFormat(descriptor.format);gXrTextures++;}
 noteTextureMade("heap",d,(unsigned long)((_options&MTLResourceStorageModeMask)>>MTLResourceStorageModeShift),0);
 NmbTexture *handle=nmb_heap_new_texture(_handle,&descriptor,offset);
 if(!handle){NBLOG(@"native-metal: heap texture refused: %@",lastError());return nil;}
 NBTexture *texture=[NBTexture new];texture.handle=handle;nmb_texture_describe(handle,&descriptor);texture.descriptor=descriptor;
 texture.parentHeap=(id<MTLHeap>)self;texture.heapOffset=offset==UINT64_MAX?0:offset;return (id<MTLTexture>)texture;
}
- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor *)d {return [self textureWithDescriptor:d offset:UINT64_MAX];}
- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor *)d offset:(NSUInteger)offset {return [self textureWithDescriptor:d offset:offset];}
@end
// Apple's SPI resource groups name resources for cache hints (encodeCacheHintTag:/Finalize:); a group only keeps its resources alive.
@interface NBResourceGroup : NBObject
@property(nonatomic,strong) NSArray *resources;
@end
@implementation NBResourceGroup
@end

// Creation and archive population must use the same complete supported descriptor.
static bool computeDescriptor(MTLComputePipelineDescriptor *d,NmbComputePipelineDescriptor *p,NSError **error) {
 if(!d || ![(id)d.computeFunction isKindOfClass:[NBFunction class]]){if(error)*error=nbError(@"foreign compute function");return false;}
 // Metal may materialize empty stage/linked descriptors on access. Preserve the
 // default meaning while refusing state this frontend cannot yet represent.
 MTLStageInputOutputDescriptor *stage=d.stageInputDescriptor;
 bool empty=!stage || (stage.indexType==MTLIndexTypeUInt16 && stage.indexBufferIndex==0);
 for(NSUInteger index=0;stage && index<31;++index) {
  MTLAttributeDescriptor *a=stage.attributes[index];MTLBufferLayoutDescriptor *b=stage.layouts[index];
  empty=empty && a.format==MTLAttributeFormatInvalid && !a.offset && !a.bufferIndex && !b.stride &&
        b.stepFunction==MTLStepFunctionPerVertex && b.stepRate==1;
 }
 if(!empty){if(error)*error=nbError(@"compute stage inputs are not implemented");return false;}
 MTLLinkedFunctions *linked=d.linkedFunctions;
 if(linked.functions.count || linked.binaryFunctions.count || linked.privateFunctions.count || linked.groups.count ||
    d.preloadedLibraries.count || d.insertLibraries.count) {
  if(error)*error=nbError(@"compute linked/dynamic libraries are not implemented");return false;
 }
 p->function=[(NBFunction *)d.computeFunction handle];p->max_threads=d.maxTotalThreadsPerThreadgroup;
 p->multiple_of_execution_width=d.threadGroupSizeIsMultipleOfThreadExecutionWidth;p->indirect=d.supportIndirectCommandBuffers;
 p->max_call_stack_depth=d.maxCallStackDepth;p->adding_binary_functions=d.supportAddingBinaryFunctions;
 for(NSUInteger index=0;index<31;++index)p->mutability[index]=d.buffers[index].mutability;
 return true;
}

// ---- binary archives ----
@interface NBBinaryArchive : NBObject
@property(nonatomic) NmbBinaryArchive *handle;
@end
@implementation NBBinaryArchive
- (void)dealloc {nmb_binary_archive_release(_handle);}
- (NSUInteger)options {return 0;}
- (BOOL)addRenderPipelineFunctionsWithDescriptor:(MTLRenderPipelineDescriptor *)d error:(NSError **)error {
 auto *p=new NmbRenderPipelineDescriptor();
 const bool ok=renderDescriptor(d,p,error) && nmb_binary_archive_add_render(_handle,p);delete p;
 if(!ok)return failWith<BOOL>(error,@"archive render pipeline",NO);
 if(error)*error=nil;return YES;
}
- (BOOL)addComputePipelineFunctionsWithDescriptor:(MTLComputePipelineDescriptor *)d error:(NSError **)error {
 NmbComputePipelineDescriptor p={};if(!computeDescriptor(d,&p,error))return NO;
 if(!nmb_binary_archive_add_compute_descriptor(_handle,&p))return failWith<BOOL>(error,@"archive compute pipeline",NO);
 if(error)*error=nil;return YES;
}
- (BOOL)serializeToURL:(NSURL *)url error:(NSError **)error {
 if(!url.isFileURL){if(error)*error=nbError(@"a file URL is required");return NO;}
 if(!nmb_binary_archive_serialize(_handle,url.fileSystemRepresentation))return failWith<BOOL>(error,@"archive serialize",NO);
 if(error)*error=nil;return YES;
}
@end

static bool pipelineArchives(NSArray *list,std::vector<NmbBinaryArchive *> &archives,NSError **error) {
 for(id archive in list) {
  if(![archive isKindOfClass:[NBBinaryArchive class]]){if(error)*error=nbError(@"foreign binary archive");return false;}
  archives.push_back([(NBBinaryArchive *)archive handle]);
 }
 return true;
}

@interface NBDeviceMethods : NSObject
@end
@implementation NBDeviceMethods
- (NSString *)name {return gName;}
- (uint64_t)registryID {return gRegistryID;}
- (NSUInteger)maxBufferLength {return nmb_device_max_buffer_length(gDevice);}
- (BOOL)hasUnifiedMemory {return YES;}
// Core Image's CIMetalDeviceGetLimits reads both (Apple's GPU devices answer them; _MTLDevice does not): without them every CIContext
// creation threw and iconservicesagent aborted while rasterizing symbol glyphs. The GPU has no memory of its own; it shares the system's.
- (uint64_t)dedicatedMemorySize {return 0;}
- (uint64_t)sharedMemorySize {return [NSProcessInfo processInfo].physicalMemory;}
- (BOOL)isLowPower {return NO;}
- (BOOL)isRemovable {return NO;}
- (BOOL)isHeadless {return NO;}
// supportsFamily: and the other queries the runtime answers are installed from it (installAnswers); supportsFamily: also answers the target
// hardware's families (the bridge's TARGET_FAMILIES, journaled with their waived rows as F lines). No legacy feature set is claimed: the
// base implementation reads Apple tables this device does not have.
- (BOOL)supportsFeatureSet:(MTLFeatureSet)set {(void)set;return NO;}
// The original discovery (_MTLAddDevice) runs these four on every device it registers. The target's own versions fill the base class's limits
// and feature tables from a hardware profile (`featureProfile`, `indirectArgumentBufferCapabilities`, a table of Apple GPUs) that this device
// has not got: running them would claim an Apple GPU's capabilities. They are empty here, so a limit or feature the runtime measured is answered
// by this file's own answers (installAnswers) and any other stays the base class's neutral 0/NO. Nothing is claimed that nothing measured.
- (void)initLimits {}
- (void)initFeatureQueries {}
- (void)initGPUFamilySupport {}
- (void)initWorkarounds {}
- (NSString *)productName {return gName;}
- (NSString *)vendorName {char name[64]={};nmb_device_vendor_name(gDevice,name,sizeof name);return [NSString stringWithUTF8String:name];}
- (MTLSize)maxThreadsPerThreadgroup {
 uint64_t s[3]={};if(!nmb_device_max_threads_per_threadgroup(gDevice,s))NBLOG(@"native-metal: maxThreadsPerThreadgroup: %@",lastError());
 return MTLSizeMake(s[0],s[1],s[2]);
}
- (void)doesNotRecognizeSelector:(SEL)selector {
 journal(std::string("M ")+class_getName(object_getClass(self))+" "+sel_getName(selector));
 if(gMissing)gMissing("NativeBridgeMetalDevice",sel_getName(selector));
 noteProblem([NSString stringWithFormat:@"missing device %s",sel_getName(selector)]);
 [super doesNotRecognizeSelector:selector];
}
- (id<MTLEvent>)newEvent {
 NmbEvent *handle=nmb_event_new(gDevice,false);if(!handle)return nil;
 NBEvent *event=[NBEvent new];event.handle=handle;return (id<MTLEvent>)event;
}
- (id<MTLSharedEvent>)newSharedEvent {
 NmbEvent *handle=nmb_event_new(gDevice,true);if(!handle)return nil;
 NBSharedEvent *event=[NBSharedEvent new];event.handle=handle;return (id<MTLSharedEvent>)event;
}
- (id<MTLFence>)newFence {
 NmbFence *handle=nmb_fence_new(gDevice);if(!handle)return nil;
 NBFence *fence=[NBFence new];fence.handle=handle;return (id<MTLFence>)fence;
}
- (id<MTLCommandQueue>)newCommandQueue {
 NmbQueue *handle=nmb_queue_new(gDevice);if(!handle)return nil;
 NBQueue *queue=[NBQueue new];queue.handle=handle;
 static std::atomic<uint32_t> queues{0};queue.traceSerial=++queues;
 return (id<MTLCommandQueue>)queue;
}
- (id<MTLHeap>)newHeapWithDescriptor:(MTLHeapDescriptor *)d {
 if(!d)return nil;
 NmbHeap *handle=nmb_heap_new(gDevice,d.type,d.size,(uint64_t)d.resourceOptions);
 if(!handle){NBLOG(@"native-metal: heap refused: %@",lastError());return nil;}
 NBHeap *heap=[NBHeap new];heap.handle=handle;heap.options=d.resourceOptions;heap.heapType=d.type;return (id<MTLHeap>)heap;
}
- (id)newResourceGroupFromResources:(const id __unsafe_unretained [])resources count:(NSUInteger)count {
 NBResourceGroup *group=[NBResourceGroup new];
 NSMutableArray *held=[NSMutableArray arrayWithCapacity:count];
 for(NSUInteger i=0;i<count;i++)if(resources[i])[held addObject:resources[i]];
 group.resources=held;return group;
}
// Not made here: a texture over raw memory in Apple's layout, and Apple's texture layout object. Both are refused, with the reason, rather than
// answered with a guessed shape.
- (id<MTLTexture>)newTextureWithBytesNoCopy:(void *)bytes length:(NSUInteger)length descriptor:(MTLTextureDescriptor *)d deallocator:(void (^)(void *,NSUInteger))deallocator {
 (void)bytes;(void)length;(void)d;(void)deallocator;
 NBLOG(@"native-metal: newTextureWithBytesNoCopy:length:descriptor:deallocator: is not implemented (Apple's texture layout of raw memory is not known)");
 return nil;
}
- (id)newTextureLayoutWithDescriptor:(MTLTextureDescriptor *)d isHeapOrBufferBacked:(BOOL)backed {
 (void)d;(void)backed;
 NBLOG(@"native-metal: newTextureLayoutWithDescriptor:isHeapOrBufferBacked: is not implemented (Apple's layout object is not known)");
 return nil;
}
- (id<MTLCommandQueue>)newCommandQueueWithMaxCommandBufferCount:(NSUInteger)count {(void)count;return [(id<MTLDevice>)self newCommandQueue];}
- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)length options:(MTLResourceOptions)options {
 NmbBuffer *handle=nmb_buffer_new(gDevice,length,options,nullptr);
 if(!handle){NBLOG(@"native-metal: buffer refused: %@",lastError());return nil;}
 NBBuffer *buffer=[NBBuffer new];buffer.handle=handle;buffer.options=options;return (id<MTLBuffer>)buffer;
}
- (id<MTLBuffer>)newBufferWithBytes:(const void *)bytes length:(NSUInteger)length options:(MTLResourceOptions)options {
 if(!bytes)return nil;
 NmbBuffer *handle=nmb_buffer_new(gDevice,length,options,bytes);
 if(!handle){NBLOG(@"native-metal: buffer refused: %@",lastError());return nil;}
 NBBuffer *buffer=[NBBuffer new];buffer.handle=handle;buffer.options=options;return (id<MTLBuffer>)buffer;
}
- (id<MTLBuffer>)newBufferWithBytesNoCopy:(void *)pointer length:(NSUInteger)length options:(MTLResourceOptions)options deallocator:(void (^)(void *,NSUInteger))deallocator {
 // Default (NGPU_HOST_COPY unset or 1): the runtime's host imports outlived their owners' memory and showed other icons and labels
// (fajita Settings scroll); a GPU-owned copy made at creation is correct. NGPU_HOST_COPY=0: the no-copy import; 2: copy at every commit.
static const bool copyMode=[]{const char *v=getenv("NGPU_HOST_COPY");return !(v && v[0]=='0');}();
 // Up to 4 MiB (backing stores, images); a larger range (a framebuffer the presenter reads) stays a no-copy import.
 if(copyMode && !(options&MTLResourceStorageModePrivate) && length<=(4u<<20)) {
  NmbBuffer *owned=nmb_buffer_new(gDevice,length,options&~MTLResourceStorageModeMask,pointer);
  if(!owned){NBLOG(@"native-metal: host copy refused: %@",lastError());return nil;}
  gNoCopyAll++;gHostCopies++;gHostCopyBytes+=length;gLiveCopies++;
  NBBuffer *buffer=[NBBuffer new];buffer.handle=owned;buffer.options=options;
  buffer.copySource=pointer;buffer.copyBytes=length;buffer.copyDeallocator=deallocator;
  {std::lock_guard<std::mutex> lock(gImportLock);if(!gImports)gImports=[NSHashTable weakObjectsHashTable];[gImports addObject:buffer];}
  return (id<MTLBuffer>)buffer;
 }
 void *context=deallocator?(__bridge_retained void *)[deallocator copy]:nullptr;
 // Joins an earlier import of its range only while a buffer over it is alive (noteHostRange).
 noteHostRange(pointer,length);
 NmbBuffer *handle=nmb_buffer_new_no_copy_releasing(gDevice,pointer,length,options,context?releaseBlock:nullptr,context);
 if(!handle) {
  if(context)(void)(__bridge_transfer id)context;
  NBLOG(@"native-metal: no-copy buffer refused: %@",lastError());return nil;
 }
 gNoCopyAll++;if(context)gNoCopyMade++;
 NBBuffer *buffer=[NBBuffer new];buffer.handle=handle;buffer.options=options;
 registerImport(buffer,nullptr,pointer,length);
 countHostBuffer(buffer,pointer);
 return (id<MTLBuffer>)buffer;
}
- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor *)d {
 NmbTextureDescriptor descriptor;if(!textureDescriptor(d,&descriptor))return nil;
 if(xrFormat(descriptor.format)!=descriptor.format){descriptor.format=xrFormat(descriptor.format);gXrTextures++;}
 noteTextureMade("device",d,(unsigned long)d.storageMode,0);
 NmbTexture *handle=nmb_texture_new(gDevice,&descriptor);noteFormat((uint32_t)descriptor.format,handle!=nullptr);
 if(!handle){NBLOG(@"native-metal: texture refused (format %lu %lux%lux%lu type %lu usage %lx storage %lu samples %lu): %@",(unsigned long)d.pixelFormat,
  (unsigned long)d.width,(unsigned long)d.height,(unsigned long)d.depth,(unsigned long)d.textureType,(unsigned long)d.usage,(unsigned long)d.storageMode,
  (unsigned long)d.sampleCount,lastError());return nil;}
 NBTexture *texture=[NBTexture new];texture.handle=handle;nmb_texture_describe(handle,&descriptor);texture.descriptor=descriptor;return (id<MTLTexture>)texture;
}
// One texture per surface plane and descriptor (NGPU_SURFACE_SHARE=0 makes every call import its own): a later texture over the same
// surface plane with the same descriptor is a view of the first live one, so the two share its shadow image. CoreAnimation wraps
// its frame surfaces in several MTLTextures; each texture's own shadow forgot the other's at every copy back, so the full-screen
// frame was copied in again every frame (fajita shadow log: 1125x2436 "forgotten" why=1, 32 GB in 86 s).
static NSMapTable<NSString *,NBTexture *> *gSurfaceTexturesBySurface;
static NSString *surfaceTextureKey(IOSurfaceRef surface,NSUInteger plane,MTLTextureDescriptor *d) {
 return [NSString stringWithFormat:@"%u/%lu/%lu/%lux%lux%lu/%lu/%lu/%lu/%lu/%lu",IOSurfaceGetID(surface),(unsigned long)plane,(unsigned long)d.pixelFormat,
  (unsigned long)d.width,(unsigned long)d.height,(unsigned long)d.depth,(unsigned long)d.textureType,(unsigned long)d.mipmapLevelCount,
  (unsigned long)d.sampleCount,(unsigned long)d.arrayLength,(unsigned long)d.usage];
}
// The surface's pages are the texels: its mapping in this process is imported without a copy (pinned by the kernel, mapped into the GPU's
// 4 KiB context by Turnip), so every process that imports the surface renders and samples the same memory.
- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor *)d iosurface:(IOSurfaceRef)surface plane:(NSUInteger)plane {
 // A planar surface (Apple's lossless-compressed BGRA spelling has one plane, video formats two) is imported whole; the texture starts at
 // the plane's own address and rows. Only a plain surface has no planes, and then only plane 0 exists.
 const size_t planes=surface?IOSurfaceGetPlaneCount(surface):0;
 if(!surface || !d || (planes?plane>=planes:plane!=0)){refuseSurface(1,surface,d,plane,0,0);return nil;}
 static const bool share=[]{const char *v=getenv("NGPU_SURFACE_SHARE");return !(v && v[0]=='0');}();
 static dispatch_once_t made;dispatch_once(&made,^{gSurfaceTexturesBySurface=[NSMapTable strongToWeakObjectsMapTable];});
 NSString *key=share?surfaceTextureKey(surface,plane,d):nil;
 if(share) {
  NBTexture *first=nil;@synchronized(gSurfaceTexturesBySurface){first=[gSurfaceTexturesBySurface objectForKey:key];}
  if(first) {
   id<MTLTexture> view=[first view:nmb_texture_new_view(first.handle,(uint64_t)d.pixelFormat,0,nullptr,nullptr,true)];
   if(view){gSurfaceShared++;gSurfaceTextures++;return view;}
  }
 }
 if(planes)gSurfacePlanar++;
 void *base=IOSurfaceGetBaseAddress(surface);const size_t page=(size_t)getpagesize(),allocation=IOSurfaceGetAllocSize(surface);
 if(!page || (page&(page-1)) || !base || ((uintptr_t)base&(page-1)) || !allocation || allocation>SIZE_MAX-(page-1)){refuseSurface(2,surface,d,plane,(uintptr_t)base,0);return nil;}
 const uintptr_t start=planes?(uintptr_t)IOSurfaceGetBaseAddressOfPlane(surface,plane):(uintptr_t)base;
 const size_t row=planes?IOSurfaceGetBytesPerRowOfPlane(surface,plane):IOSurfaceGetBytesPerRow(surface);
 if(start<(uintptr_t)base || start-(uintptr_t)base>=allocation || !row){refuseSurface(3,surface,d,plane,start-(uintptr_t)base,row);return nil;}
 const size_t bytes=(allocation+page-1)&~(page-1);
 NBBuffer *buffer=surfaceBuffer(surface,base,bytes);if(!buffer){refuseSurface(4,surface,d,plane,start-(uintptr_t)base,row);return nil;}
 NBTexture *texture=planes>=2?textureOnSurface(buffer,d,surface,plane,start-(uintptr_t)base,row):textureOnBuffer(buffer,d,start-(uintptr_t)base,row,true);
 if(!texture){refuseSurface(5,surface,d,plane,start-(uintptr_t)base,row);return nil;}
 gSurfaceTextures++;texture.surface=(IOSurfaceRef)CFRetain(surface);
 if(share){@synchronized(gSurfaceTexturesBySurface){[gSurfaceTexturesBySurface setObject:texture forKey:key];}}
 return (id<MTLTexture>)texture;
}
- (id<MTLLibrary>)newLibraryWithData:(dispatch_data_t)data error:(NSError **)error {
 const void *bytes=nullptr;size_t size=0;dispatch_data_t mapped=dispatch_data_create_map(data,&bytes,&size);
 const uint64_t libraryBegan=nowUs();
 NmbLibrary *handle=nmb_library_new(gDevice,bytes,size,false);(void)mapped;gTimeLibrary.add(libraryBegan);
 if(!handle)return failWith<id<MTLLibrary>>(error,@"library",nil);
 NBLibrary *library=[NBLibrary new];library.handle=handle;library.captureId=pcap::library(bytes,size);if(error)*error=nil;return (id<MTLLibrary>)library;
}
- (id<MTLLibrary>)newLibraryWithFile:(NSString *)path error:(NSError **)error {
 NSData *bytes=path?[NSData dataWithContentsOfFile:path options:NSDataReadingMappedIfSafe error:error]:nil;if(!bytes)return nil;
 const uint64_t libraryBegan=nowUs();
 NmbLibrary *handle=nmb_library_new(gDevice,bytes.bytes,bytes.length,false);gTimeLibrary.add(libraryBegan);
 if(!handle)return failWith<id<MTLLibrary>>(error,@"library",nil);
 NBLibrary *library=[NBLibrary new];library.handle=handle;library.captureId=pcap::library(bytes.bytes,bytes.length);if(error)*error=nil;return (id<MTLLibrary>)library;
}
- (id<MTLLibrary>)newLibraryWithURL:(NSURL *)url error:(NSError **)error {
 if(!url.isFileURL){if(error)*error=nbError(@"a file URL is required");return nil;}
 return [(id<MTLDevice>)self newLibraryWithFile:url.path error:error];
}
- (id<MTLLibrary>)newDefaultLibraryWithBundle:(NSBundle *)bundle error:(NSError **)error {
 NSURL *url=[bundle URLForResource:@"default" withExtension:@"metallib"];
 if(!url){if(error)*error=nbError(@"the bundle has no default.metallib");return nil;}
 return [(id<MTLDevice>)self newLibraryWithURL:url error:error];
}
- (id<MTLLibrary>)newDefaultLibrary {return [(id<MTLDevice>)self newDefaultLibraryWithBundle:[NSBundle mainBundle] error:nil];}
- (id<MTLComputePipelineState>)newComputePipelineStateWithFunction:(id<MTLFunction>)function error:(NSError **)error {
 return [(id<MTLDevice>)self newComputePipelineStateWithFunction:function options:MTLPipelineOptionNone reflection:nil error:error];
}
- (id<MTLComputePipelineState>)newComputePipelineStateWithFunction:(id<MTLFunction>)function options:(MTLPipelineOption)options
                                                         reflection:(MTLAutoreleasedComputePipelineReflection *)reflection error:(NSError **)error {
 if(reflection)*reflection=nil;
 if(![(id)function isKindOfClass:[NBFunction class]]){if(error)*error=nbError(@"foreign function");return nil;}
 const NmbPipelineOptions creation={(uint64_t)options,nullptr,0};
 const uint64_t pipelineBegan=nowUs();
 NmbComputePipeline *handle=nmb_compute_pipeline_new_with_options(gDevice,[(NBFunction *)function handle],&creation);gTimePipeline.add(pipelineBegan);
 const long captureNumber=pcap::compute([(NBFunction *)function captureId],nullptr,handle!=nullptr);
 if(!handle)return failWith<id<MTLComputePipelineState>>(error,@"compute pipeline",nil);
 NBComputePipeline *state=[NBComputePipeline new];state.handle=handle;state.captureId=captureNumber;gPipelines++;if(error)*error=nil;return (id<MTLComputePipelineState>)state;
}
- (id<MTLComputePipelineState>)newComputePipelineStateWithDescriptor:(MTLComputePipelineDescriptor *)d options:(MTLPipelineOption)options
                                                       reflection:(MTLAutoreleasedComputePipelineReflection *)reflection error:(NSError **)error {
 if(reflection)*reflection=nil;
 NmbComputePipelineDescriptor p={};if(!computeDescriptor(d,&p,error))return nil;
 std::vector<NmbBinaryArchive *> archives;
 if(!pipelineArchives(d.binaryArchives,archives,error))return nil;
 const NmbPipelineOptions creation={(uint64_t)options,archives.data(),archives.size()};
 const uint64_t pipelineBegan=nowUs();
 NmbComputePipeline *handle=nmb_compute_pipeline_new_with_descriptor(gDevice,&p,&creation);gTimePipeline.add(pipelineBegan);
 const long captureNumber=pcap::compute([(NBFunction *)d.computeFunction captureId],d.label.UTF8String,handle!=nullptr);
 if(!handle)return failWith<id<MTLComputePipelineState>>(error,@"compute pipeline",nil);
 NBComputePipeline *state=[NBComputePipeline new];state.handle=handle;state.label=d.label;state.captureId=captureNumber;gPipelines++;
 if(error)*error=nil;return (id<MTLComputePipelineState>)state;
}
- (id<MTLRenderPipelineState>)newRenderPipelineStateWithDescriptor:(MTLRenderPipelineDescriptor *)d error:(NSError **)error {
 return [(id<MTLDevice>)self newRenderPipelineStateWithDescriptor:d options:MTLPipelineOptionNone reflection:nil error:error];
}
- (id<MTLRenderPipelineState>)newRenderPipelineStateWithDescriptor:(MTLRenderPipelineDescriptor *)d options:(MTLPipelineOption)options
                                                        reflection:(MTLAutoreleasedRenderPipelineReflection *)reflection error:(NSError **)error {
 if(reflection)*reflection=nil;
 auto *p=new NmbRenderPipelineDescriptor();
 if(!renderDescriptor(d,p,error)){delete p;return nil;}
 std::vector<NmbBinaryArchive *> archives;
 if(!pipelineArchives(d.binaryArchives,archives,error)){delete p;return nil;}
 const NmbPipelineOptions creation={(uint64_t)options,archives.data(),archives.size()};
 const uint64_t pipelineBegan=nowUs();const NBCost costBegan=NBCost::now();
 NmbRenderPipeline *handle=nmb_render_pipeline_new_with_options(gDevice,p,&creation);gTimePipeline.add(pipelineBegan);
 const long captureNumber=pcap::render(p,[(NBFunction *)d.vertexFunction captureId],d.fragmentFunction?[(NBFunction *)d.fragmentFunction captureId]:std::string(),
                            d.label.UTF8String,d.alphaToCoverageEnabled,handle!=nullptr);
 NmbRenderPipelineDescriptor *xp=p;   // kept for the XR variant below
 journalRenderPipeline(d,pipelineBegan,handle?nullptr:nmb_last_error(),gJournalPath.empty()?std::string():NBCost::now().since(costBegan));
 if(!handle){delete xp;return failWith<id<MTLRenderPipelineState>>(error,@"render pipeline",nil);}
 NBRenderPipeline *state=[NBRenderPipeline new];state.handle=handle;state.captureId=captureNumber;gPipelines++;
 for(unsigned i=0;i<8;i++) {
  const uint64_t asked=(uint64_t)d.colorAttachments[i].pixelFormat;
  if(xrFormat(asked)!=asked) {
   if(!state->xrDescriptor)state->xrDescriptor=std::make_shared<NmbRenderPipelineDescriptor>(*xp);
   state->xrDescriptor->colors[i].format=asked;state->xrMask|=1u<<i;
  }
 }
 if(state->xrMask)state.xrFunctions=@[d.vertexFunction?:(id)[NSNull null],d.fragmentFunction?:(id)[NSNull null]];
 delete xp;
 if(error)*error=nil;return (id<MTLRenderPipelineState>)state;
}
// The asynchronous forms make the pipeline on the caller's thread and call the handler before returning, as the library's
// newFunctionWithName:constantValues:completionHandler: does (RenderBox's pipeline loader, Core Image's bilateral solver).
- (void)newRenderPipelineStateWithDescriptor:(MTLRenderPipelineDescriptor *)d completionHandler:(MTLNewRenderPipelineStateCompletionHandler)handler {
 NSError *error=nil;id<MTLRenderPipelineState> state=[(id<MTLDevice>)self newRenderPipelineStateWithDescriptor:d error:&error];
 handler(state,error);
}
- (void)newRenderPipelineStateWithDescriptor:(MTLRenderPipelineDescriptor *)d options:(MTLPipelineOption)options
                           completionHandler:(MTLNewRenderPipelineStateWithReflectionCompletionHandler)handler {
 NSError *error=nil;id<MTLRenderPipelineState> state=[(id<MTLDevice>)self newRenderPipelineStateWithDescriptor:d options:options reflection:nil error:&error];
 handler(state,nil,error);
}
- (void)newComputePipelineStateWithFunction:(id<MTLFunction>)function completionHandler:(MTLNewComputePipelineStateCompletionHandler)handler {
 NSError *error=nil;id<MTLComputePipelineState> state=[(id<MTLDevice>)self newComputePipelineStateWithFunction:function error:&error];
 handler(state,error);
}
- (void)newComputePipelineStateWithFunction:(id<MTLFunction>)function options:(MTLPipelineOption)options
                          completionHandler:(MTLNewComputePipelineStateWithReflectionCompletionHandler)handler {
 NSError *error=nil;id<MTLComputePipelineState> state=[(id<MTLDevice>)self newComputePipelineStateWithFunction:function options:options reflection:nil error:&error];
 handler(state,nil,error);
}
// SPI: a precompiled pipeline library (RenderBox.framework/default.pipelinelib) holds Apple GPU binaries. None is opened; the caller logs the
// error and builds its pipelines from the shader library instead (RB::Device::render_pipeline_state).
- (id)newPipelineLibraryWithFilePath:(NSString *)path error:(NSError **)error {
 if(error)*error=nbError([NSString stringWithFormat:@"no pipeline library on this GPU: %@",path.lastPathComponent]);
 return nil;
}
- (id<MTLBinaryArchive>)newBinaryArchiveWithDescriptor:(MTLBinaryArchiveDescriptor *)d error:(NSError **)error {
 if(d.url && !d.url.isFileURL){if(error)*error=nbError(@"a file URL is required");return nil;}
 NmbBinaryArchive *handle=nmb_binary_archive_new(gDevice,d.url?d.url.fileSystemRepresentation:nullptr);
 if(!handle)return failWith<id<MTLBinaryArchive>>(error,@"binary archive",nil);
 NBBinaryArchive *archive=[NBBinaryArchive new];archive.handle=handle;if(error)*error=nil;return (id<MTLBinaryArchive>)archive;
}
- (NSUInteger)requiredLinearTextureBytesPerRowForDescriptor:(MTLTextureDescriptor *)d {
 const uint64_t row=nmb_device_linear_bytes_per_row(gDevice,d.pixelFormat,d.width);
 if(!row)NBLOG(@"native-metal: requiredLinearTextureBytesPerRow: %@",lastError());
 return row;
}
- (id<MTLSamplerState>)newSamplerStateWithDescriptor:(MTLSamplerDescriptor *)d {
 NmbSamplerDescriptor s={(uint64_t)d.minFilter,(uint64_t)d.magFilter,(uint64_t)d.mipFilter,(uint64_t)d.sAddressMode,(uint64_t)d.tAddressMode,(uint64_t)d.rAddressMode,
     (uint64_t)d.borderColor,(uint64_t)d.normalizedCoordinates,d.maxAnisotropy,(uint64_t)d.compareFunction,(uint64_t)d.lodAverage,d.lodMinClamp,d.lodMaxClamp};
 NmbSampler *handle=nmb_sampler_new(gDevice,&s);
 if(!handle){NBLOG(@"native-metal: sampler refused: %@",lastError());return nil;}
 NBSampler *sampler=[NBSampler new];sampler.handle=handle;
 if(pcap::on()) {
  char e[512];
  snprintf(e,sizeof e,"{\"min\":%llu,\"mag\":%llu,\"mip\":%llu,\"s\":%llu,\"t\":%llu,\"r\":%llu,\"border\":%llu,\"normalized\":%s,\"maxAniso\":%llu,"
           "\"compare\":%llu,\"lodAverage\":%s,\"lodMin\":%g,\"lodMax\":%g}",(unsigned long long)s.min_filter,(unsigned long long)s.mag_filter,
           (unsigned long long)s.mip_filter,(unsigned long long)s.s_address,(unsigned long long)s.t_address,(unsigned long long)s.r_address,
           (unsigned long long)s.border_color,s.normalized?"true":"false",(unsigned long long)s.max_anisotropy,(unsigned long long)s.compare,
           s.lod_average?"true":"false",(double)s.lod_min,(double)s.lod_max);
  sampler.captureText=e;
 }
 return (id<MTLSamplerState>)sampler;
}
- (id<MTLDepthStencilState>)newDepthStencilStateWithDescriptor:(MTLDepthStencilDescriptor *)d {
 auto face=[](MTLStencilDescriptor *s)->NmbStencil {
  return {(uint64_t)s.stencilCompareFunction,(uint64_t)s.stencilFailureOperation,(uint64_t)s.depthFailureOperation,(uint64_t)s.depthStencilPassOperation,s.readMask,s.writeMask};
 };
 NmbDepthStencilDescriptor s={(uint64_t)d.depthCompareFunction,(uint64_t)d.depthWriteEnabled,face(d.frontFaceStencil),face(d.backFaceStencil)};
 NmbDepthStencil *handle=nmb_depth_stencil_new(gDevice,&s);
 if(!handle){NBLOG(@"native-metal: depth/stencil state refused: %@",lastError());return nil;}
 NBDepthStencil *state=[NBDepthStencil new];state.handle=handle;
 if(pcap::on()) {
  char e[512];
  snprintf(e,sizeof e,"{\"compare\":%llu,\"write\":%s,\"stencil\":{\"compare\":%llu,\"fail\":%llu,\"depthFail\":%llu,\"pass\":%llu,\"read\":%llu,\"write\":%llu}}",
           (unsigned long long)s.compare,s.write?"true":"false",(unsigned long long)s.front.compare,(unsigned long long)s.front.failure,
           (unsigned long long)s.front.depth_failure,(unsigned long long)s.front.pass,(unsigned long long)s.front.read_mask,(unsigned long long)s.front.write_mask);
  state.captureText=e;
 }
 return (id<MTLDepthStencilState>)state;
}
@end

// The device queries the runtime answers (nmb_device_answer_selector), as methods of the device class. One without an argument is read once
// and installed only where the device has an answer of its own; one with an argument asks the runtime each call, and a refusal answers 0/NO
// with the reason logged.
static std::unordered_map<SEL,uint64_t> gAnswers;
static NSUInteger countAnswer(id,SEL cmd) {return (NSUInteger)gAnswers.at(cmd);}
static BOOL flagAnswer(id,SEL cmd) {return gAnswers.at(cmd)!=0;}
static uint64_t askAnswer(SEL cmd,NSUInteger arg) {
 uint64_t value=0;
 if(!nmb_device_answer(gDevice,sel_getName(cmd),arg,&value)){NBLOG(@"native-metal: %s(%lu): %@",sel_getName(cmd),(unsigned long)arg,lastError());return 0;}
 return value;
}
static NSUInteger countArgAnswer(id,SEL cmd,NSUInteger arg) {return (NSUInteger)askAnswer(cmd,arg);}
static BOOL flagArgAnswer(id,SEL cmd,NSUInteger arg) {return askAnswer(cmd,arg)!=0;}
static void installAnswers(Class cls) {
 uint32_t kind=0;
 for(size_t i=0;const char *name=nmb_device_answer_selector(i,&kind);i++) {
  SEL sel=sel_registerName(name);uint64_t value=0;
  switch(kind) {
  case NMB_ANSWER_COUNT: case NMB_ANSWER_FLAG:
   if(!nmb_device_answer(gDevice,name,0,&value)){NBLOG(@"native-metal: no device answer: %@",lastError());continue;}
   // Extended-range 10-bit (XR10) framebuffers are withheld: this display chain presents 8-bit BGRX scanout only. The render server picks its
   // wide-gamut 64-bit (and Apple-compressed) framebuffer when a device offers them, and the display driver could not show it (int-089: the
   // swap's surface was '&w4a', 8 bytes per pixel, and every swap was refused).
   if(!strcmp(name,"supportsExtendedXR10Formats"))value=0;
   gAnswers[sel]=value;
   class_addMethod(cls,sel,kind==NMB_ANSWER_COUNT?(IMP)countAnswer:(IMP)flagAnswer,kind==NMB_ANSWER_COUNT?"Q16@0:8":"B16@0:8");
   break;
  case NMB_ANSWER_COUNT_ARG: class_addMethod(cls,sel,(IMP)countArgAnswer,"Q24@0:8Q16");break;
  case NMB_ANSWER_FLAG_ARG: class_addMethod(cls,sel,(IMP)flagArgAnswer,"B24@0:8Q16");break;
  }
 }
}

extern "C" id native_bridge_metal_open(const char *loader,const char *cache,uint64_t registryID,void (*missing)(const char *,const char *)) {
 if(gDeviceObject)return gDeviceObject;
 gMissing=missing;gInFlight=[NSMutableSet set];
 passtrace::arm(cache);cpuprof::arm(cache);  // before the device opens: Turnip reads its u_trace settings once (PassTrace.h)
 gDevice=nmb_device_open(loader,cache);
 if(loader)gLoaderPath=loader;
 if(!gDevice){NBLOG(@"native-metal: device refused: %s",nmb_last_error());return nil;}
 if(cache){gJournalPath=std::string(cache)+"/pipelines.log";gJournalBegan=nowUs();}
 pcap::begin(cache);if(pcap::on())journal("X capture "+pcap::dir());
 ftrace::begin(cache);if(ftrace::on())journal("X frametrace "+ftrace::path());
 fcap::arm();
 char name[128]={};nmb_device_name(gDevice,name,sizeof name);gName=[NSString stringWithUTF8String:name];gRegistryID=registryID;
 // The target families supportsFamily: answers true, with the rows of Apple's tables the device does not meet in them (nmb_device_family_waivers).
 {char waived[4096]={};nmb_device_family_waivers(gDevice,waived,sizeof waived);
  for(char *line=waived,*next;*line;line=next){next=strchr(line,'\n');if(next)*next++=0;else next=line+strlen(line);journal(std::string("F ")+line);}}
 Class base=objc_getClass("_MTLDevice");
#if !TARGET_OS_IPHONE
 // Host instrument: on NSObject every selector a client sends that this file lacks reaches the missing hook, instead of the host's
 // own _MTLDevice implementation.
 if(getenv("NMB_DEVICE_BASE_NSOBJECT"))base=[NSObject class];
#endif
#if TARGET_OS_IPHONE
 // The target's base class ABI (536 bytes on iPadOS 17.7.11, 528 on iOS 16.7.16): the dynamic subclass adds no instance variables.
 if(!base || class_getInstanceSize(base)!=targetbuild::mtlDeviceBytes()){NBLOG(@"native-metal: unexpected _MTLDevice ABI (%zu bytes)",base?class_getInstanceSize(base):0);return nil;}
#endif
 if(!base)base=[NSObject class];
 Class cls=objc_allocateClassPair(base,"NativeBridgeMetalDevice",0);if(!cls)return nil;
 unsigned count=0;Method *methods=class_copyMethodList([NBDeviceMethods class],&count);
 for(unsigned i=0;i<count;i++)class_addMethod(cls,method_getName(methods[i]),method_getImplementation(methods[i]),method_getTypeEncoding(methods[i]));
 free(methods);installAnswers(cls);class_addProtocol(cls,@protocol(MTLDevice));
 // The original registration (_MTLAddDevice) asserts that a plugin's device conforms to Metal's private device protocol as well; the target's own
 // drivers get it from their IOGPU base class. Conformance only answers conformsToProtocol:, the methods are still looked up one by one.
 if(Protocol *spi=objc_getProtocol("MTLDeviceSPI"))class_addProtocol(cls,spi);
 objc_registerClassPair(cls);
 gDeviceObject=[cls new];
 return gDeviceObject;
}

// Bare AIR bitcode (no metallib wrapper), which Metal itself refuses: for the bridge's own fixtures only.
extern "C" id native_bridge_metal_library_with_air(const void *bytes,size_t length) {
 NmbLibrary *handle=gDevice?nmb_library_new(gDevice,bytes,length,true):nullptr;if(!handle)return nil;
 NBLibrary *library=[NBLibrary new];library.handle=handle;return library;
}
extern "C" bool native_bridge_metal_shader_cache_stats(NmbShaderCacheStats *out) {
 return gDevice && nmb_device_shader_cache_stats(gDevice,out);
}
extern "C" bool native_bridge_metal_work_stats(uint64_t out[10]) {
 return gDevice && nmb_device_work_stats(gDevice,out);
}

extern "C" size_t native_bridge_metal_status(char *out,size_t size) {
 std::lock_guard<std::mutex> lock(gProblemLock);
 return (size_t)snprintf(out,size,"c%llu/%llu d%llu p%llu x%llu R%llu D%llu F%llu M%llu C%llu B%llu %s | %s",(unsigned long long)gCommits.load(),(unsigned long long)gCommitFailures.load(),(unsigned long long)gCompleted.load(),
  (unsigned long long)gPipelines.load(),(unsigned long long)gProblems.load(),(unsigned long long)gPasses.load(),(unsigned long long)gDraws.load(),(unsigned long long)gFullScreenPasses.load(),(unsigned long long)(gBiggestPass.load()>>10),(unsigned long long)gDispatches.load(),(unsigned long long)gBlits.load(),gLastProblems[0],gLastProblems[1]);
}

static size_t native_bridge_metal_formats_locked(char *out,size_t size) {
 int n=snprintf(out,size,"tex x%llu s%llu/%llu p%llu f",(unsigned long long)gTextureRefusals.load(),(unsigned long long)gSurfaceTextures.load(),
              (unsigned long long)gSurfaceRefusals.load(),(unsigned long long)gSurfacePlanar.load());
 for(unsigned i=0;i<gFormatCount && n>0 && (size_t)n<size;i++)n+=snprintf(out+n,size-(size_t)n,"%u ",gFormats[i]);
 return n>0?(size_t)n:0;
}
// The benchmark's rows (native_bridge_metal_benchmark), kept for the telemetry table.
static std::mutex gBenchLock;
static char gBench[4][96];
static std::atomic<unsigned> gBenchRuns;
extern "C" size_t native_bridge_metal_benchmark(id device,const char *metallib,unsigned pairs,char *out,size_t size) {
 auto now=[]{return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);};
 // the first run fills rows 0-1 (Y, Y1), a later one rows 2-3 (Z, Z1), so a busy and an idle run are both kept
 const unsigned base=gBenchRuns++?2u:0u;
 auto put=[&](unsigned row,const char *format,auto... args){std::lock_guard<std::mutex> lock(gBenchLock);snprintf(gBench[base+row],sizeof gBench[base+row],format,args...);
  if(base)gBench[base+row][0]='Z';};
 // compute: a dependent integer chain (no memory), 100M steps
 uint64_t t=now(),x=0x9e3779b97f4a7c15ull;
 for(uint32_t i=0;i<100000000u;i++){x^=x<<13;x^=x>>7;x^=x<<17;}
 const uint64_t computeUs=(now()-t)/1000;
 // first touch: 32 MiB of fresh anonymous memory (the tablet's free memory is small), one store per 4 KiB page (the kernel's fault and zero-fill), then four memsets of it
 const size_t bytes=32u<<20;
 uint64_t faultUs=0,fillUs=0,shaUs=0;
 void *m=mmap(nullptr,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
 if(m!=MAP_FAILED) {
  volatile uint8_t *b=(volatile uint8_t *)m;
  t=now();for(size_t at=0;at<bytes;at+=4096)b[at]=1;faultUs=(now()-t)/1000;
  t=now();for(int k=0;k<4;k++)memset(m,k,bytes);fillUs=(now()-t)/1000;
  unsigned char digest[CC_SHA256_DIGEST_LENGTH];
  t=now();CC_SHA256(m,(CC_LONG)(16u<<20),digest);shaUs=(now()-t)/1000;
  x^=digest[0];
  munmap(m,bytes);
 }
 put(0,"Y cpu=%llums(100M) fault=%lluus/32MiB=%lluns/pg fill=%lluMB/s sha=%lluMB/s %llx",(unsigned long long)(computeUs/1000),
  (unsigned long long)faultUs,(unsigned long long)(faultUs*1000/8192),(unsigned long long)(fillUs?4ull*32*1000000/fillUs:0),
  (unsigned long long)(shaUs?16ull*1000000/shaUs:0),(unsigned long long)(x&0xf));
 // pipelines: the host tool's pairs, in its order
 uint64_t made=0,refused=0,createUs=0,createMax=0,drawUs=0,drawMax=0,libraryUs=0;
 @autoreleasepool {
  id<MTLDevice> dev=(id<MTLDevice>)device;NSError *error=nil;
  t=now();
  id<MTLLibrary> qc=metallib?[dev newLibraryWithURL:[NSURL fileURLWithPath:@(metallib)] error:&error]:nil;
  libraryUs=(now()-t)/1000;
  if(qc) {
   NSArray<NSString *> *names=qc.functionNames;
   id<MTLCommandQueue> queue=[dev newCommandQueue];
   MTLTextureDescriptor *td=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:64 height:64 mipmapped:NO];
   td.usage=MTLTextureUsageRenderTarget;
   id<MTLTexture> target=[dev newTextureWithDescriptor:td];
   id<MTLBuffer> zeros=[dev newBufferWithLength:65536 options:MTLResourceStorageModeShared];
   unsigned n=0;
   for(NSString *name in names) {
    if(n>=pairs)break;
    id<MTLFunction> f=[qc newFunctionWithName:name];
    if(f.functionType!=MTLFunctionTypeVertex)continue;
    NSString *fragmentName=[name stringByReplacingOccurrencesOfString:@"_vert_" withString:@"_frag_"];
    if([fragmentName isEqualToString:name] || ![names containsObject:fragmentName])continue;
    ++n;
    MTLRenderPipelineDescriptor *d=[MTLRenderPipelineDescriptor new];
    d.vertexFunction=f;d.fragmentFunction=[qc newFunctionWithName:fragmentName];
    d.colorAttachments[0].pixelFormat=MTLPixelFormatBGRA8Unorm;
    t=now();
    id<MTLRenderPipelineState> state=[dev newRenderPipelineStateWithDescriptor:d error:&error];
    const uint64_t took=(now()-t)/1000;
    if(!state){++refused;continue;}
    ++made;createUs+=took;createMax=std::max(createMax,took);
    t=now();
    MTLRenderPassDescriptor *pass=[MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture=target;pass.colorAttachments[0].loadAction=MTLLoadActionClear;pass.colorAttachments[0].storeAction=MTLStoreActionStore;
    id<MTLCommandBuffer> commands=[queue commandBuffer];
    id<MTLRenderCommandEncoder> encoder=[commands renderCommandEncoderWithDescriptor:pass];
    [encoder setRenderPipelineState:state];
    for(NSUInteger i=0;i<8;++i){[encoder setVertexBuffer:zeros offset:0 atIndex:i];[encoder setFragmentBuffer:zeros offset:0 atIndex:i];}
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [encoder endEncoding];[commands commit];[commands waitUntilCompleted];
    const uint64_t drew=(now()-t)/1000;
    drawUs+=drew;drawMax=std::max(drawMax,drew);
   }
  }
 }
 put(1,"Y1 qc lib=%llums made=%llu refused=%llu create=%llums max=%llums draw=%llums max=%llums",(unsigned long long)(libraryUs/1000),
  (unsigned long long)made,(unsigned long long)refused,(unsigned long long)(createUs/1000),(unsigned long long)(createMax/1000),
  (unsigned long long)(drawUs/1000),(unsigned long long)(drawMax/1000));
 size_t at=0;
 if(out && size) {
  std::lock_guard<std::mutex> lock(gBenchLock);
  out[0]=0;for(unsigned r=base;r<base+2;r++)at+=(size_t)snprintf(out+at,size>at?size-at:0,"%s\n",gBench[r]);
 }
 return at;
}
// The telemetry table the plugin streams to the kernel's log: newline-separated rows of at most 90 characters.
extern "C" size_t native_bridge_metal_telemetry(char *out,size_t size) {
 size_t at=0;out[0]=0;
 auto row=[&](const char *format,auto... args){
  char text[200];int n=snprintf(text,sizeof text,format,args...);if(n<=0)return;if(n>90)n=90;
  if(at+(size_t)n+2>=size)return;memcpy(out+at,text,(size_t)n);at+=(size_t)n;out[at++]='\n';out[at]=0;};
 auto ms=[](uint64_t us){return (unsigned long long)(us/1000);};
 {std::lock_guard<std::mutex> lock(gBenchLock);for(unsigned r=0;r<4;r++)if(gBench[r][0])row("%.90s",gBench[r]);}
 row("S c%llu/%llu d%llu p%llu x%llu R%llu D%llu F%llu M%lluk C%llu B%llu",(unsigned long long)gCommits.load(),(unsigned long long)gCommitFailures.load(),
  (unsigned long long)gCompleted.load(),(unsigned long long)gPipelines.load(),(unsigned long long)gProblems.load(),(unsigned long long)gPasses.load(),
  (unsigned long long)gDraws.load(),(unsigned long long)gFullScreenPasses.load(),(unsigned long long)(gBiggestPass.load()>>10),
  (unsigned long long)gDispatches.load(),(unsigned long long)gBlits.load());
 row("T pipeline n=%llu total=%llums max=%llums",(unsigned long long)gTimePipeline.count.load(),ms(gTimePipeline.total.load()),ms(gTimePipeline.slowest.load()));
 row("T library n=%llu total=%llums max=%llums",(unsigned long long)gTimeLibrary.count.load(),ms(gTimeLibrary.total.load()),ms(gTimeLibrary.slowest.load()));
 row("T function n=%llu total=%llums max=%llums",
  (unsigned long long)gTimeFunction.count.load(),ms(gTimeFunction.total.load()),ms(gTimeFunction.slowest.load()));
 row("G commit-to-complete n=%llu total=%llums max=%llums mean=%lluus",(unsigned long long)gTimeEncodeToCommit.count.load(),ms(gTimeEncodeToCommit.total.load()),
  ms(gTimeEncodeToCommit.slowest.load()),(unsigned long long)(gTimeEncodeToCommit.count.load()?gTimeEncodeToCommit.total.load()/gTimeEncodeToCommit.count.load():0));
 row("J scheduled wait n=%llu total=%llums max=%llums",(unsigned long long)gTimeWaitScheduled.count.load(),ms(gTimeWaitScheduled.total.load()),ms(gTimeWaitScheduled.slowest.load()));
 row("J completed wait n=%llu total=%llums max=%llums",(unsigned long long)gTimeWaitCompleted.count.load(),ms(gTimeWaitCompleted.total.load()),ms(gTimeWaitCompleted.slowest.load()));
 row("J resource wait n=%llu total=%llums max=%llums",(unsigned long long)gTimeResourceWait.count.load(),ms(gTimeResourceWait.total.load()),ms(gTimeResourceWait.slowest.load()));
 row("J resource checks=%llu skipped=%llu",(unsigned long long)gResourceChecks.load(),(unsigned long long)gResourceSkipped.load());
 row("J import clean n=%llu bytes=%llu seed-skips=%llu us=%llu max=%llu",(unsigned long long)gImportFlushes.load(),
  (unsigned long long)gImportFlushBytes.load(),(unsigned long long)gImportSeedSkips.load(),(unsigned long long)gTimeImportFlush.total.load(),
  (unsigned long long)gTimeImportFlush.slowest.load());
 row("J surface cache n=%llu bytes=%llu dup=%llu us=%llu max=%llu",(unsigned long long)gTimeSurfaceFlush.count.load(),
  (unsigned long long)gSurfaceFlushBytes.load(),(unsigned long long)gSurfaceFlushDuplicates.load(),
  (unsigned long long)gTimeSurfaceFlush.total.load(),(unsigned long long)gTimeSurfaceFlush.slowest.load());
 row("N live buffers %lld copies %lld textures %lld",(long long)gLiveBuffers.load(),(long long)gLiveCopies.load(),(long long)gLiveTextures.load());
 row("N no-copy %llu dealloc %llu/%llu surface %llu/%llu",(unsigned long long)gNoCopyAll.load(),(unsigned long long)gNoCopyReleased.load(),
  (unsigned long long)gNoCopyMade.load(),
  (unsigned long long)gSurfaceImportsReleased.load(),(unsigned long long)gSurfaceImportsMade.load());
 row("N xr as f16: textures %llu pipelines %llu xr variants %llu",(unsigned long long)gXrTextures.load(),(unsigned long long)gXrPipelines.load(),(unsigned long long)gXrVariants.load());
 row("N surface generations stated %llu",(unsigned long long)gSurfaceGenerations.load());
 row("N host copies %llu bytes %llu",(unsigned long long)gHostCopies.load(),(unsigned long long)gHostCopyBytes.load());
 row("N host remaps %llu ranges %llu",(unsigned long long)gHostRemaps.load(),(unsigned long long)[]{std::lock_guard<std::mutex> l(gHostRangeLock);return gHostRanges.size();}());
 // The runtime's VkPipeline makings (the creation log, METAL2VULKAN_LOG_PIPELINES; they happen at draws, not at the state's creation) and
 // its shader cache: where the driver's compile time goes.
 uint64_t work[10]={};
 if(gDevice && nmb_device_work_stats(gDevice,work)) {
  row("V vkpipe n=%llu total=%llums max=%llums waited=%llums failed=%llu",
   (unsigned long long)work[0],ms(work[1]),ms(work[2]),ms(work[3]),(unsigned long long)work[4]);
  row("V cache hit%llu miss%llu wr%llu | msg i%llu w%llu",(unsigned long long)work[7],(unsigned long long)work[8],(unsigned long long)work[9],(unsigned long long)work[5],
   (unsigned long long)work[6]);
  char warning[200];size_t n=nmb_last_warning(warning,sizeof warning);
  if(n){row("W %.86s",warning);if(n>86)row("Wb %.86s",warning+86);}
 }
 {
  std::lock_guard<std::mutex> lock(gProblemLock);
  char formats[160];native_bridge_metal_formats_locked(formats,sizeof formats);row("F %.86s",formats);
  const SurfaceRefusal &r=gSurfaceLastRefusal;
  if(r.step)row("SR %u %08x %c%c%c%c pl%u/%u o%llx rb%llu a%llx %llux%llu pf%u",r.step,r.format,(char)(r.format>>24),(char)(r.format>>16),(char)(r.format>>8),
   (char)r.format,r.plane,r.planes,(unsigned long long)r.offset,(unsigned long long)r.row,(unsigned long long)r.allocation,(unsigned long long)r.width,
   (unsigned long long)r.height,r.pixelFormat);
  row("P0 %.86s",gLastProblems[0]);row("P0b %.86s",strlen(gLastProblems[0])>86?gLastProblems[0]+86:"");
  row("P1 %.86s",gLastProblems[1]);row("P1b %.86s",strlen(gLastProblems[1])>86?gLastProblems[1]+86:"");
 }
 {
  std::lock_guard<std::mutex> lock(gCommitLock);
  const uint32_t newest=(uint32_t)gCommits.load();
  row("%s","Q timing v2: per-buffer API counts; ctc=commit-to-callback; age includes caller idle");
  for(uint32_t k=0;k<20 && k<newest;k++) {
   const CommitRecord &r=gCommitRing[(newest-k)%32];
   if(r.index!=newest-k)continue;
   row("K%u t=%llu.%03llus rp%u dr%u kpx%llu cs%u bl%u ctc=%lluus",r.index,(unsigned long long)(r.committedUs/1000000),(unsigned long long)((r.committedUs/1000)%1000),
    r.work.passes,r.work.draws,(unsigned long long)(r.work.pixels>>10),r.work.dispatches,r.work.blits,(unsigned long long)r.callbackUs);
   if(k<4)row("E%u age=%lluus scheduled=%lldus",r.index,(unsigned long long)r.ageUs,(long long)r.scheduledUs);
  }
 }
 return at;
}

extern "C" size_t native_bridge_metal_formats(char *out,size_t size) {
 std::lock_guard<std::mutex> lock(gProblemLock);
 return native_bridge_metal_formats_locked(out,size);
}

// The scanout client's swaps for the frame trace (GpuScanoutClient.h declares it weak).
extern "C" void NBTraceSwap(uint64_t sequence,uint64_t waitUs,uint64_t offerUs,uint64_t presentedUs,uint64_t completedUs,uint32_t width,uint32_t height,
                            uint32_t mapId,int result) {
 passtrace::swaps().fetch_add(1,std::memory_order_relaxed);
 if(!ftrace::on())return;
 ftrace::linef("S,%llu,%llu,%llu,%llu,%llu,%u,%u,%u,%d",(unsigned long long)sequence,(unsigned long long)waitUs,(unsigned long long)offerUs,
               (unsigned long long)presentedUs,(unsigned long long)completedUs,width,height,mapId,result);
}
