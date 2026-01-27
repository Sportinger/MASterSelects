# Shared Decoder Architecture - Export System V2

## Problem Statement

The current parallel decode system has fundamental scalability issues:

**Current Issues:**
- One VideoDecoder instance per clip instance (not per unique file)
- Same video file used 2x (regular + nested) → 2 separate decoders
- Decoders compete for different positions → constant resets/seeks
- With 10+ nested compositions → 20+ decoders → exponential slowdown
- Buffer misalignment: Target at 4.7s, buffer at 8-10s → constant seeks

**Example Failure:**
```
Timeline time: 4.033s → Source time: 2.616s
Buffer range: [8.120s-10.540s]
Result: Seek required → Buffer cleared → Infinite loop
```

## Design Goals

1. **Scale to Complex Projects**: Handle 10+ nested comps, triple-nested, 50+ unique videos
2. **Predictable Performance**: Linear scaling relative to complexity, no exponential slowdowns
3. **Memory Efficient**: Reuse decoded frames, shared decoder instances
4. **Smart Pre-fetching**: Decode frames in optimal order based on export timeline
5. **Resilient**: Graceful degradation, fallback to HTMLVideoElement if needed

## Research Findings

Based on web research ([WebCodecs Best Practices](https://developer.chrome.com/docs/web-platform/best-practices/webcodecs), [Remotion WebCodecs](https://www.remotion.dev/docs/media-parser/webcodecs), [W3C WebCodecs Issues](https://github.com/w3c/webcodecs/issues/424)):

### Key Insights

1. **Decoder Reuse**: VideoDecoder can be reused via `reset()` + `configure()` pattern
2. **Hardware Limits**: Very limited memory buffer for hardware decoders - pause until frames freed
3. **Queue Management Critical**: Monitor `decodeQueueSize` to prevent memory buildup
4. **Workers for Parallelism**: Move decoding to workers for true parallelism
5. **Frame Cache**: Professional editors use proprietary formats (ProRes, DNxHR) for frame cache
6. **Nested Comp Optimization**: Pre-render and cache nested compositions

### Professional Editor Patterns

From [DaVinci Resolve Render Cache](https://creativevideotips.com/tutorials/davinci-resolve-render-cache-essentials):
- Timeline rendered at timeline resolution in proprietary format
- Smart cache invalidation (RED for RAW, h265, effects, transitions)
- Pre-render when machine idle for 5 seconds

From [After Effects Optimization](https://pixflow.net/blog/the-ultimate-guide-to-after-effects-optimization/):
- Avoid unnecessary nested comps inside pre-comps
- Render complex comps to flattened files

## Architecture Design

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Export Orchestrator                      │
│  - Analyzes timeline                                        │
│  - Creates export plan                                      │
│  - Coordinates all subsystems                               │
└──────────────────┬──────────────────────────────────────────┘
                   │
      ┌────────────┼────────────┐
      │            │            │
      ▼            ▼            ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│  Shared  │ │  Frame   │ │  Nested  │
│ Decoder  │ │  Cache   │ │   Comp   │
│   Pool   │ │  Manager │ │ Renderer │
└──────────┘ └──────────┘ └──────────┘
      │            │            │
      └────────────┼────────────┘
                   │
                   ▼
            ┌──────────────┐
            │ Video Encoder│
            └──────────────┘
```

### 1. Shared Decoder Pool

**Purpose**: One VideoDecoder instance per unique video file (not per clip instance)

**Key Features:**
- File-based decoder mapping: `Map<fileHash, DecoderInstance>`
- Decoder reuse via `reset()` + `configure()` when switching between clips
- Worker-based for true parallelism (one worker per decoder)
- Smart position tracking to minimize seeks

**API Design:**
```typescript
class SharedDecoderPool {
  // Get decoder for a file, creates if doesn't exist
  async getDecoder(fileHash: string, fileData: ArrayBuffer): Promise<SharedDecoder>

  // Request frame from any clip using this file
  async requestFrame(
    fileHash: string,
    sourceTime: number,
    priority: number
  ): Promise<VideoFrame>

  // Bulk request for export planning
  async requestFrameBatch(
    requests: FrameRequest[]
  ): Promise<Map<string, VideoFrame>>

  // Cleanup
  dispose(): void
}

interface SharedDecoder {
  fileHash: string
  worker: Worker
  currentPosition: number  // Current decode position in seconds
  buffer: FrameBuffer      // LRU cache of decoded frames

  // Seek to position and decode
  seekAndDecode(targetTime: number, frameCount: number): Promise<void>

  // Get frame from buffer or trigger decode
  getFrame(time: number): Promise<VideoFrame>
}
```

**Smart Seeking:**
- Track current decoder position
- If target is within 2 seconds forward → sequential decode
- If target > 2 seconds away → seek to nearest keyframe
- Minimize seeks by planning decode order

### 2. Frame Cache Manager

**Purpose**: LRU cache for decoded frames with intelligent eviction

**Key Features:**
- Per-file frame buffers with configurable size (default: 120 frames per file)
- LRU eviction when cache full
- Cache statistics for monitoring
- Optional disk cache for very large projects

**API Design:**
```typescript
class FrameCacheManager {
  private caches: Map<fileHash, LRUCache<timestamp, VideoFrame>>

  // Store frame in cache
  put(fileHash: string, timestamp: number, frame: VideoFrame): void

  // Get frame from cache
  get(fileHash: string, timestamp: number, tolerance: number): VideoFrame | null

  // Check if frame exists
  has(fileHash: string, timestamp: number, tolerance: number): boolean

  // Pre-warm cache for upcoming frames
  async prewarm(requests: FrameRequest[]): Promise<void>

  // Get cache statistics
  getStats(): CacheStats

  // Clear cache for file or all
  clear(fileHash?: string): void
}

interface CacheStats {
  totalFrames: number
  totalMemoryMB: number
  hitRate: number
  perFileStats: Map<fileHash, { frames: number, memoryMB: number }>
}
```

**Memory Management:**
- Monitor total memory usage
- Automatic eviction when > 500MB cached
- Close VideoFrames immediately when evicted
- Configurable cache size per project complexity

### 3. Export Planner

**Purpose**: Analyze timeline and optimize decode scheduling (no pre-rendering!)

**Key Features:**
- Analyzes full export range to understand file usage patterns
- Groups clips by file to minimize decoder switches
- Pre-calculates decode positions to minimize seeks
- Plans ahead 2-3 seconds for smooth pipeline
- Adaptive planning based on decoder performance

**API Design:**
```typescript
class ExportPlanner {
  // Analyze timeline and create decode schedule
  async createSchedule(
    startTime: number,
    endTime: number,
    fps: number
  ): Promise<DecodeSchedule>

  // Get next batch of frames to decode (called every frame)
  getNextDecodeBatch(currentTime: number): FrameRequest[]
}

interface DecodeSchedule {
  fileUsage: Map<fileHash, UsagePattern>
  totalFrames: number
  estimatedTime: number
}

interface UsagePattern {
  fileHash: string
  clipIds: string[]          // All clips using this file
  timeRanges: TimeRange[]    // When file is needed
  totalFrames: number        // Total frames needed from file
  isHeavyUsage: boolean      // Used frequently? (increase cache)
}

interface FrameRequest {
  fileHash: string
  clipId: string
  sourceTime: number
  priority: number           // Higher = needed sooner
  isNestedComp: boolean
  nestedDepth: number        // 0 = main timeline, 1+ = nested
}
```

**Planning Algorithm:**

**Phase 1: Analyze File Usage**
```
1. Walk through entire export range
2. For each frame:
   - Find active clips (including nested)
   - Record: fileHash, time range, nesting depth
3. Group by fileHash:
   - Merge overlapping/close time ranges
   - Calculate total frame count needed
   - Mark files with heavy usage (> 20% of export)
```

**Phase 2: Optimize Decode Order**
```
1. For files with heavy usage:
   - Increase cache size (150 frames vs 60)
   - Mark for aggressive pre-fetching
2. For files with scattered usage:
   - Lower cache size to save memory
   - Decode just-in-time
```

**Phase 3: Runtime Decode Scheduling**
```
While exporting frame N:
  1. Get frames needed at N (current)
  2. Get frames needed at N+60 to N+90 (look-ahead)
  3. Group by fileHash
  4. Sort by:
     - Priority: current > near future > far future
     - Position: minimize seeks within file
  5. Submit to SharedDecoderPool
```

**Example Schedule Output:**
```typescript
{
  fileUsage: {
    'abc123': {
      fileHash: 'abc123',
      clipIds: ['clip1', 'clip3_nested'],
      timeRanges: [[0.0, 15.0], [20.0, 30.0]],
      totalFrames: 750,
      isHeavyUsage: true
    },
    'def456': {
      fileHash: 'def456',
      clipIds: ['clip2'],
      timeRanges: [[5.0, 10.0]],
      totalFrames: 150,
      isHeavyUsage: false
    }
  },
  totalFrames: 900,
  estimatedTime: 300  // 5 minutes
}
```

### 4. Nested Composition Renderer (Just-In-Time)

**Purpose**: Render nested compositions on-demand during export (After Effects style)

**Key Strategy:**
- **NO pre-rendering** - render nested comps frame-by-frame as needed
- Recursively resolve from deepest to shallowest for each frame
- Optional single-frame cache for repeated access within same frame
- Minimal memory footprint

**API Design:**
```typescript
class NestedCompRenderer {
  // Render single frame of composition at specific time
  async renderFrame(
    compId: string,
    time: number,
    width: number,
    height: number
  ): Promise<ImageBitmap>

  // Recursively build layers for nested comp at time
  private async buildNestedLayers(
    comp: Composition,
    time: number
  ): Promise<Layer[]>

  // Single-frame cache (optional optimization)
  private frameCache: Map<compId_timestamp, ImageBitmap>
}
```

**Rendering Flow (per export frame):**
```
For each frame in export:
  1. Get clips at current time
  2. For each clip:
     - If regular video → SharedDecoderPool.getFrame()
     - If nested comp → NestedCompRenderer.renderFrame()
       → Recursively resolve nested clips
       → If nested clip is video → SharedDecoderPool.getFrame()
       → If nested clip is nested comp → recurse deeper
       → Composite layers on GPU
  3. Composite main timeline layers
  4. Encode frame
```

**Why This is Better:**
- **No upfront wait**: Export starts immediately
- **Lower memory**: Only one frame at a time in memory
- **Simpler code**: No cache invalidation logic needed
- **More flexible**: Easy to handle changes during export (not that we need it)
- **After Effects proven**: This is how professional tools do it

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)
- [ ] Implement `SharedDecoderPool` with worker-based decoders
- [ ] Implement file-hash based decoder mapping
- [ ] Implement `FrameCacheManager` with LRU cache
- [ ] Add decoder reuse via reset() + configure()
- [ ] Add tests for decoder lifecycle
- [ ] Benchmark single file export vs current system

### Phase 2: Export Planner & Scheduling (Week 2)
- [ ] Implement `ExportPlanner` timeline analysis
- [ ] Implement file usage pattern detection
- [ ] Add runtime decode scheduling (look-ahead)
- [ ] Implement priority-based batch requests
- [ ] Test with complex timelines (10+ clips)

### Phase 3: Nested Comp Just-In-Time Rendering (Week 3)
- [ ] Implement `NestedCompRenderer` for on-demand rendering
- [ ] Add recursive layer building
- [ ] Integrate with SharedDecoderPool for nested clips
- [ ] Test with double/triple nested comps
- [ ] Verify no memory leaks with deep nesting

### Phase 4: Integration & Main Export Loop (Week 4)
- [ ] Integrate all components into `FrameExporter`
- [ ] Implement new export loop with just-in-time nested rendering
- [ ] Add progress reporting (accurate for nested comps)
- [ ] Implement fallback to HTMLVideoElement on errors
- [ ] Performance testing with 10+ nested compositions
- [ ] Memory profiling and leak detection

### Phase 5: Polish & Production Ready (Week 5)
- [ ] Add export settings UI (cache size: 200MB/500MB/2GB)
- [ ] Optimize worker communication (SharedArrayBuffer?)
- [ ] Add detailed logging for debugging complex exports
- [ ] Write user documentation
- [ ] Create example complex projects for testing
- [ ] Final performance validation

## Migration Strategy

**Deployment Plan:**
- Release V2 as **default for all projects immediately** (per user request)
- Keep V1 as fallback via export setting: "Use Legacy Export (if issues)"
- Automatic fallback to V1 on critical errors with user notification
- Deprecate V1 after 4 weeks if no major issues reported

**Rollout:**
1. Deploy V2 as default
2. Monitor error rates and user feedback first week
3. Quick fixes for critical bugs
4. Deprecate V1 after 4 weeks of stability

## Performance Targets

**Current System (V1):**
- Simple project (3 clips): ~2x realtime
- Complex project (10+ clips): ~0.1x realtime (FAILS)

**Target System (V2):**
- Simple project (3 clips): ~3x realtime (50% faster)
- Medium project (10 clips): ~2x realtime
- Complex project (20 clips, 5 nested): ~1.5x realtime
- Triple-nested (10 levels): ~1x realtime

**Memory Targets:**
- Cache < 500MB for typical projects
- Peak memory < 2GB for complex projects
- No memory leaks over long exports

## Risk Mitigation

**Risks:**
1. **Decoder reuse bugs**: VideoDecoder may have issues with reset/configure
   - Mitigation: Extensive testing, fallback to new decoder instance on errors
   - Monitoring: Log all reset/configure operations

2. **Cache thrashing**: LRU may evict needed frames causing re-decode
   - Mitigation: Adaptive cache sizing based on file usage patterns
   - Monitoring: Track cache hit rate, warn if < 80%

3. **Just-in-time rendering overhead**: Rendering nested comps per-frame may be slow
   - Mitigation: GPU acceleration for all compositing, single-frame cache
   - Fallback: Optional pre-render mode for very slow systems (can add later)

4. **Worker communication overhead**: Transferring VideoFrames between workers
   - Mitigation: Use transferable objects, batch transfers where possible
   - Alternative: Consider SharedArrayBuffer for zero-copy (if supported)

5. **Complex nested structures**: Triple-nested comps with many layers
   - Mitigation: Flatten layers where possible, optimize GPU compositing
   - Limit: Warn user if nesting depth > 5 levels

## Open Questions

1. **Cache persistence**: Should we save cache to disk between sessions?
2. **Decoder count**: How many decoders to run in parallel? (CPU core count?)
3. **Frame format**: Store as VideoFrame, ImageBitmap, or raw RGBA?
4. **Progress reporting**: How granular should progress be for nested pre-render?
5. **Memory limits**: Hard limit or soft limit with user warning?

## References

- [Chrome WebCodecs Best Practices](https://developer.chrome.com/docs/web-platform/best-practices/webcodecs)
- [Remotion WebCodecs Guide](https://www.remotion.dev/docs/media-parser/webcodecs)
- [W3C WebCodecs Explainer](https://github.com/w3c/webcodecs/blob/main/explainer.md)
- [DaVinci Resolve Render Cache](https://creativevideotips.com/tutorials/davinci-resolve-render-cache-essentials)
- [WebCodecs Issues: Decoder Reuse](https://github.com/w3c/webcodecs/issues/424)
- [Video Frame Processing Performance](https://webrtchacks.com/video-frame-processing-on-the-web-webassembly-webgpu-webgl-webcodecs-webnn-and-webtransport/)

---

**Document Version**: 1.0
**Author**: Claude (AI Assistant)
**Date**: 2026-01-27
**Status**: Design Proposal - Pending User Questions
