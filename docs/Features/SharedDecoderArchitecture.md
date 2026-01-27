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

**Purpose**: Analyze timeline and create optimal frame decode order

**Key Features:**
- Analyzes full export range before starting
- Groups clips by file to minimize decoder switches
- Sorts by timeline order to minimize seeks
- Pre-renders nested compositions
- Generates decode plan with priorities

**Algorithm:**
```typescript
class ExportPlanner {
  // Analyze timeline and create plan
  async createPlan(
    startTime: number,
    endTime: number,
    fps: number
  ): Promise<ExportPlan>

  // Plan structure
  interface ExportPlan {
    phases: ExportPhase[]      // Sequential phases
    totalFrames: number
    estimatedTime: number      // Based on benchmarks
  }

  interface ExportPhase {
    type: 'prerender' | 'decode' | 'render'
    description: string
    tasks: Task[]
  }
}
```

**Planning Algorithm:**

**Phase 1: Pre-render Nested Compositions**
```
For each nested composition:
  1. Recursively pre-render from deepest to shallowest
  2. Cache result as single video/image sequence
  3. Replace nested comp with cached version
```

**Phase 2: Group and Sort Clips**
```
1. Group all clips by fileHash
2. For each file group:
   - Sort clips by timeline startTime
   - Calculate required time ranges
   - Merge overlapping/adjacent ranges
3. Sort groups by timeline coverage (most used first)
```

**Phase 3: Generate Decode Schedule**
```
For each frame in export range:
  1. Determine active clips at this time
  2. For each clip:
     - Check if frame in cache
     - If not, add to decode batch
  3. Group decode requests by fileHash
  4. Sort by timeline order within group
```

**Example Plan Output:**
```typescript
{
  phases: [
    {
      type: 'prerender',
      description: 'Pre-render nested composition "Comp 2"',
      tasks: [
        { action: 'render', compId: 'comp2', frames: 180 }
      ]
    },
    {
      type: 'decode',
      description: 'Pre-decode frames for export',
      tasks: [
        {
          action: 'decode',
          fileHash: 'abc123',
          ranges: [[0, 5.0], [10.0, 15.0]],
          frames: 300
        }
      ]
    },
    {
      type: 'render',
      description: 'Render export frames',
      tasks: [
        { action: 'render', frameStart: 0, frameEnd: 600 }
      ]
    }
  ]
}
```

### 4. Nested Composition Renderer

**Purpose**: Pre-render nested compositions to avoid real-time nested rendering

**Key Features:**
- Recursively render from deepest to shallowest
- Cache result as ImageBitmap sequence or OffscreenCanvas
- Invalidate cache only when composition changes
- Support for double/triple nesting

**API Design:**
```typescript
class NestedCompRenderer {
  // Pre-render composition to cache
  async prerenderComposition(
    compId: string,
    startTime: number,
    endTime: number,
    fps: number
  ): Promise<CachedComposition>

  // Get cached frame
  getFrame(compId: string, time: number): ImageBitmap | null

  // Check if composition is cached
  isCached(compId: string): boolean

  // Clear cache
  clearCache(compId?: string): void
}

interface CachedComposition {
  compId: string
  frames: Map<timestamp, ImageBitmap>
  width: number
  height: number
  duration: number
  cacheSize: number  // Memory in bytes
}
```

**Rendering Strategy:**
1. Detect all nested compositions in export range
2. Sort by nesting depth (deepest first)
3. For each composition:
   - Render all frames to ImageBitmap
   - Store in cache
   - Replace in timeline with cached version
4. Main timeline render uses cached compositions

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)
- [ ] Implement `SharedDecoderPool` with single decoder
- [ ] Implement `FrameCacheManager` with LRU cache
- [ ] Add tests for decoder reuse patterns
- [ ] Benchmark single file export vs current system

### Phase 2: Export Planner (Week 2)
- [ ] Implement `ExportPlanner` timeline analysis
- [ ] Implement grouping and sorting algorithms
- [ ] Add decode schedule generation
- [ ] Test with complex timelines (10+ clips)

### Phase 3: Nested Comp Rendering (Week 3)
- [ ] Implement `NestedCompRenderer`
- [ ] Add recursive pre-rendering
- [ ] Integrate with export planner
- [ ] Test with double/triple nested comps

### Phase 4: Integration & Optimization (Week 4)
- [ ] Integrate all components into `FrameExporter`
- [ ] Add progress reporting for all phases
- [ ] Implement fallback to HTMLVideoElement on errors
- [ ] Performance testing and optimization
- [ ] Memory profiling and leak detection

### Phase 5: Polish & Documentation (Week 5)
- [ ] Add export settings UI (cache size, decoder count)
- [ ] Write user documentation
- [ ] Add developer documentation
- [ ] Create example projects for testing

## Migration Strategy

**Backward Compatibility:**
- Keep existing `ParallelDecodeManager` as fallback
- New system opt-in via export setting: "Use Shared Decoders (Beta)"
- Automatic fallback to old system on errors
- Feature flag for gradual rollout

**Gradual Migration:**
1. Release V2 as beta feature
2. Collect feedback and fix bugs
3. Enable by default for simple projects (< 5 clips)
4. Enable by default for complex projects
5. Deprecate V1 system

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
   - Mitigation: Extensive testing, fallback to new decoder on errors

2. **Cache thrashing**: LRU may evict needed frames
   - Mitigation: Smart pre-warming, configurable cache size

3. **Nested comp memory**: Pre-rendering all nested comps uses memory
   - Mitigation: Render on-demand for very large comps, disk cache option

4. **Worker overhead**: Communication between workers may be slow
   - Mitigation: Batch frame transfers, use SharedArrayBuffer where possible

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
