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
6. **Hybrid Approach**: Use best system for each project complexity level

## Why Hybrid Approach?

**Benefits:**
- ✅ **Simple projects keep working** with proven V1 system (no regression risk)
- ✅ **Complex projects get V2** where it's actually needed
- ✅ **Gradual rollout**: V2 bugs only affect complex projects
- ✅ **Lower risk**: Don't break what's working for 80% of users
- ✅ **Better testing**: Can compare V1 vs V2 on same project
- ✅ **User trust**: Always have fallback that works

**Risk Mitigation:**
- V1 continues to work → **no breaking changes** for existing workflows
- V2 only activates when needed → **contained blast radius**
- Manual override → **user can force V1** if V2 has issues
- Clear UI indication → **user knows which system is active**

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
- [ ] **Implement comprehensive error handling (NO auto-fallbacks)**
- [ ] Add clear error messages with actionable suggestions
- [ ] Performance testing with 10+ nested compositions
- [ ] Memory profiling and leak detection
- [ ] Test error scenarios (decoder fails, memory exhaustion, etc.)

### Phase 5: Polish & Production Ready (Week 5)
- [ ] Add export settings UI (cache size: 200MB/500MB/2GB)
- [ ] Optimize worker communication (SharedArrayBuffer?)
- [ ] Add detailed logging for debugging complex exports
- [ ] Write user documentation
- [ ] Create example complex projects for testing
- [ ] Final performance validation

## Migration Strategy - HYBRID APPROACH

**Smart Auto-Selection:**
Automatically choose the best system based on project complexity:

```typescript
function selectExportSystem(clips, tracks, compositions): 'V1' | 'V2' {
  const videoClips = clips.filter(c => c.source?.type === 'video')
  const uniqueFiles = new Set(videoClips.map(c => c.mediaFileId)).size
  const hasNestedComps = compositions.some(c => c.isNested)
  const nestedClipCount = countNestedClips(compositions)

  // Simple project: Use V1 (current system - proven, stable)
  if (uniqueFiles <= 3 && !hasNestedComps) {
    return 'V1'
  }

  // Medium complexity: Use V1 if < 8 files, otherwise V2
  if (uniqueFiles <= 8 && nestedClipCount <= 5) {
    return 'V1'
  }

  // Complex project: Always use V2 (shared decoders needed)
  return 'V2'
}
```

**Decision Matrix:**

| Project Type | Clips | Files | Nested | System | Reason |
|--------------|-------|-------|--------|--------|--------|
| Simple | 3 | 3 | No | **V1** | Proven, less overhead |
| Medium | 10 | 5 | No | **V1** | Current system handles well |
| Medium-Complex | 10 | 8 | 2 comps | **V1** | Still manageable |
| Complex | 15 | 10 | 3+ comps | **V2** | Shared decoders critical |
| Very Complex | 30 | 20 | 10+ comps | **V2** | Only V2 can handle |

**Manual Override:**
Export settings panel:
```
Export System:
  ( ) Automatic (Recommended) - Smart selection based on complexity
  ( ) Legacy System (V1) - For simple projects or if V2 has issues
  ( ) Shared Decoders (V2) - Force V2 for testing
```

**Error Handling Philosophy:**
```
NO HIDDEN FALLBACKS!

If V2 selected (auto or manual):
  → V2 must work or throw clear error
  → NO automatic fallback to V1
  → Show detailed error message with:
    - What failed (decoder, cache, worker)
    - Which file/clip caused issue
    - Suggestion: "Try Legacy System (V1) in Export Settings"

User can MANUALLY switch to V1 if needed.
```

**Why No Auto-Fallback:**
- ✅ **Clear feedback**: User knows exactly what's happening
- ✅ **Forces quality**: We must make V2 robust, no shortcuts
- ✅ **Better debugging**: Errors surface immediately, not hidden
- ✅ **User control**: Explicit choice, no surprises
- ✅ **Simpler code**: No complex fallback logic

**Rollout:**
1. **Week 1-2**: Implement V2 core (SharedDecoderPool, FrameCache)
2. **Week 3**: Add auto-selection logic, extensive error handling
3. **Week 4**: Test with complex projects, fix all errors properly
4. **Week 5**: Deploy with Automatic mode as default
5. **Week 6+**: Monitor error rates, optimize thresholds

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

## Error Handling & Risk Mitigation

### Error Handling Strategy

**Clear, Actionable Errors - No Hidden Fallbacks:**

```typescript
class ExportError extends Error {
  component: 'SharedDecoder' | 'FrameCache' | 'Worker' | 'NestedRenderer'
  clipName?: string
  fileHash?: string
  detailedMessage: string
  suggestedAction: string
}

// Example error:
throw new ExportError({
  component: 'SharedDecoder',
  clipName: 'video.mp4',
  fileHash: 'abc123',
  detailedMessage: 'Decoder failed after reset() - codec may not support reuse',
  suggestedAction: 'Switch to Legacy System (V1) in Export Settings'
})
```

**Error Display:**
```
Export Failed ❌

Component: Shared Decoder System
File: "Sunlight Forest.mp4"
Issue: Decoder reset failed after seeking

This file may use a codec that doesn't support decoder reuse.

Suggestion: Use "Legacy System (V1)" in Export Settings
```

### Risk Mitigation

**Risks:**
1. **Decoder reuse bugs**: VideoDecoder may have issues with reset/configure
   - Mitigation: Extensive testing with all codecs (H.264, H.265, VP9, AV1)
   - On error: Destroy decoder and create NEW instance (not reset)
   - If repeated failures: Throw clear error, suggest V1
   - NO silent fallback to V1

2. **Cache thrashing**: LRU may evict needed frames causing re-decode
   - Mitigation: Adaptive cache sizing based on file usage patterns
   - Monitoring: Track cache hit rate, log if < 80%
   - On thrashing: Increase cache size automatically (up to memory limit)
   - If still thrashing: Throw error with suggestion to increase cache size

3. **Just-in-time rendering overhead**: Rendering nested comps per-frame may be slow
   - Mitigation: GPU acceleration for all compositing, single-frame cache
   - Monitoring: Track frame render time, warn if > 100ms
   - If too slow: Throw error with details on which comp is slow
   - User action: Simplify composition or use V1

4. **Worker communication overhead**: Transferring VideoFrames between workers
   - Mitigation: Use transferable objects, batch transfers where possible
   - Monitoring: Track transfer time
   - If overhead > 20%: Log warning, continue (this is acceptable)

5. **Complex nested structures**: Triple-nested comps with many layers
   - Mitigation: Flatten layers where possible, optimize GPU compositing
   - Limit: Detect depth > 5, warn user before export starts
   - If render fails: Clear error about which nested comp failed

6. **Memory exhaustion**: Cache grows too large
   - Mitigation: Hard memory limit (configurable, default 1GB)
   - On approaching limit: Increase eviction rate
   - On limit exceeded: Throw clear error with suggestion to reduce cache size
   - NO silent degradation

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
