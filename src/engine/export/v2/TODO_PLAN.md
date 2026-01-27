# V2ExportBridge - TODO Fix Plan

## Current Status

V2ExportBridge has skeleton implementation but several TODOs that prevent it from working:

### Critical TODOs (Blocking)
1. ✅ Clip → FileHash mapping
2. ✅ Source time calculation from timeline time
3. ✅ MP4 parsing completion logic
4. ✅ Clip metadata storage

### Nice-to-Have TODOs (Non-blocking)
5. ✅ Store actual filename (in ClipMetadata.fileName)
6. ✅ Optimize findClip with Map instead of linear search

---

## Problem Analysis

### Problem 1: Clip → FileHash Mapping

**Current Code:**
```typescript
async getFrame(clipId: string, timelineTime: number): Promise<VideoFrame> {
  const fileHash = 'TODO' // Need to map clipId -> fileHash
  // ...
}
```

**What We Need:**
- Map clipId → clip data
- Map clip → mediaFileId (fileHash for now)
- Fast O(1) lookup

**Solution:**
```typescript
class V2ExportBridge {
  private clipMetadata: Map<string, {
    clip: TimelineClip
    fileHash: string
    fileName: string
    isNested: boolean
    parentClipId?: string
  }>

  async initialize(...) {
    // Build clipMetadata during init
    for (const pattern of schedule.fileUsage.values()) {
      for (const clipId of pattern.clipIds) {
        const clip = this.findClipInTimeline(clipId)
        this.clipMetadata.set(clipId, {
          clip,
          fileHash: pattern.fileHash,
          fileName: clip.name,
          isNested: false
        })
      }
    }
  }

  async getFrame(clipId: string, timelineTime: number) {
    const metadata = this.clipMetadata.get(clipId)
    if (!metadata) throw error
    const fileHash = metadata.fileHash
    // ...
  }
}
```

---

### Problem 2: Source Time Calculation

**Current Code:**
```typescript
const request: FrameRequest = {
  sourceTime: timelineTime, // TODO: Calculate actual source time
  // ...
}
```

**What We Need:**
- Timeline time → Clip local time
- Clip local time → Source time (handles reverse, in/out points)
- Handle nested clips (timeline time → comp time → clip local time)

**Solution:**
```typescript
private calculateSourceTime(clip: TimelineClip, timelineTime: number): number {
  // Convert timeline time to clip local time
  const clipLocalTime = timelineTime - clip.startTime

  // Handle reversed clips
  if (clip.reversed) {
    return clip.outPoint - clipLocalTime
  }

  // Normal forward playback
  return clip.inPoint + clipLocalTime
}

// For nested clips (later):
private calculateNestedSourceTime(
  clip: TimelineClip,
  timelineTime: number,
  parentClip?: TimelineClip
): number {
  if (!parentClip) {
    return this.calculateSourceTime(clip, timelineTime)
  }

  // Timeline time → Parent comp time
  const compTime = timelineTime - parentClip.startTime - (parentClip.inPoint || 0)

  // Comp time → Nested clip local time
  const clipLocalTime = compTime - clip.startTime

  // Handle reversed
  if (clip.reversed) {
    return clip.outPoint - clipLocalTime
  }

  return clip.inPoint + clipLocalTime
}
```

---

### Problem 3: MP4 Parsing Completion

**Current Code:**
```typescript
mp4File.onSamples = (_trackId, _ref, newSamples) => {
  samples.push(...newSamples)

  // PROBLEM: Resolves after 100 samples, but might need more!
  if (samples.length >= 100) {
    clearTimeout(timeout)
    resolve({ /* ... */ })
  }
}
```

**What We Need:**
- Wait for ALL samples to be extracted
- Or at least enough samples for export range
- Don't resolve too early

**Solution:**
```typescript
mp4File.onReady = (info) => {
  videoTrack = info.videoTracks[0]
  const totalSamples = videoTrack.nb_samples

  mp4File.setExtractionOptions(videoTrack.id, null, { nbSamples: Infinity })
  mp4File.start()
}

mp4File.onSamples = (_trackId, _ref, newSamples) => {
  samples.push(...newSamples)

  // Strategy 1: Wait for all samples
  if (samples.length >= videoTrack.nb_samples) {
    clearTimeout(timeout)
    resolve({ fileHash, codecConfig, videoTrack, samples, fileData })
  }

  // Strategy 2: Wait for enough samples for export range (more complex)
  // Calculate how many samples we need based on export time range
  // const samplesNeeded = calculateSamplesForTimeRange(...)
  // if (samples.length >= samplesNeeded) { resolve() }
}

// Add onFlush callback to know when extraction is complete
let extractionComplete = false
mp4File.onFlush = () => {
  extractionComplete = true
  if (samples.length > 0) {
    clearTimeout(timeout)
    resolve({ /* ... */ })
  }
}
```

---

### Problem 4: Clip Metadata Storage

**Current Code:**
```typescript
private findClip(clipId: string): TimelineClip | null {
  // TODO: Store clips in a map during initialization
  return null
}
```

**What We Need:**
- Store all clips (main timeline + nested) during initialization
- Fast lookup by clipId
- Include metadata (fileHash, fileName, isNested, parent)

**Solution:**
```typescript
class V2ExportBridge {
  private clipMetadata: Map<string, ClipMetadata>
  private allClips: TimelineClip[] = []

  async initialize(clips, tracks, compositions, ...) {
    // Step 0: Store clips for later lookup
    this.allClips = clips
    this.clipMetadata = new Map()

    // ... existing code ...

    // After creating schedule:
    this.buildClipMetadata(schedule)
  }

  private buildClipMetadata(schedule: DecodeSchedule): void {
    // For each file in schedule
    for (const [fileHash, pattern] of schedule.fileUsage) {
      // For each clip using this file
      for (const clipId of pattern.clipIds) {
        const clip = this.allClips.find(c => c.id === clipId)
        if (!clip) {
          log.warn(`Clip ${clipId} not found in timeline`)
          continue
        }

        this.clipMetadata.set(clipId, {
          clip,
          fileHash,
          fileName: clip.name,
          mediaFileId: clip.source?.mediaFileId || null,
          isNested: false, // TODO: Detect from composition structure
          parentClipId: null
        })
      }
    }

    log.info(`Built metadata for ${this.clipMetadata.size} clips`)
  }

  private findClip(clipId: string): TimelineClip | null {
    return this.clipMetadata.get(clipId)?.clip || null
  }
}

interface ClipMetadata {
  clip: TimelineClip
  fileHash: string
  fileName: string
  mediaFileId: string | null
  isNested: boolean
  parentClipId: string | null
}
```

---

## Implementation Plan

### Step 1: Add ClipMetadata Type and Storage (15 min)
- [x] Add `ClipMetadata` interface to types.ts
- [x] Add `clipMetadata` Map to V2ExportBridge
- [x] Add `buildClipMetadata()` method
- [x] Call during initialization
- [x] Update `findClip()` to use Map

### Step 2: Fix getFrame() Mapping (10 min)
- [x] Replace `fileHash = 'TODO'` with `clipMetadata.get(clipId).fileHash`
- [x] Add null check and error handling
- [x] Test that lookup works

### Step 3: Add Source Time Calculation (15 min)
- [x] Add `calculateSourceTime()` method
- [x] Handle reversed clips
- [x] Handle in/out points
- [x] Handle speed adjustments
- [x] Replace `sourceTime: timelineTime` with proper calculation

### Step 4: Fix MP4 Parsing (20 min)
- [x] Add `onFlush` callback handling
- [x] Wait for all samples (not just 100)
- [x] Store total samples count from videoTrack
- [x] Add better logging for parsing progress
- [x] Handle edge case: very small videos (<100 samples)

### Step 5: Add Filename Storage (5 min)
- [x] Store actual clip.name in ClipMetadata
- [x] Use in SharedDecoder for better logging
- [x] Update error messages to use real filename

### Step 6: Testing & Validation (15 min)
- [ ] Test with single clip export
- [ ] Verify fileHash lookup works
- [ ] Verify source time calculation correct
- [ ] Verify MP4 parsing completes
- [ ] Check memory usage

**Total Estimated Time: ~80 minutes**

---

## Success Criteria

After fixing all TODOs, V2ExportBridge should:

✅ Successfully map clipId → fileHash
✅ Calculate correct source time for any clip
✅ Parse MP4 files completely (all samples)
✅ Store and retrieve clip metadata efficiently
✅ Provide clear error messages with real filenames
✅ Be ready for integration testing

---

## Next Steps After TODOs Fixed

1. Integration into FrameExporter
2. Export Settings UI
3. End-to-end testing with real projects
4. Performance profiling
5. Bug fixes and optimization

---

## Notes

- Keep error handling clear (no auto-fallbacks!)
- Log progress for debugging
- Prioritize correctness over performance initially
- Optimize after it works
