# V2 Export System Integration Guide

## Overview

This guide shows how to integrate the V2 Shared Decoder Export System into `FrameExporter.ts`.

## Integration Steps

### 1. Add System Selection at Export Start

```typescript
import { SystemSelector, V2ExportBridge } from './v2'

class FrameExporter {
  async export(onProgress: (progress: ExportProgress) => void): Promise<Blob | null> {
    const { clips, tracks } = useTimelineStore.getState()
    const compositions = useMediaStore.getState().compositions

    // Step 1: Select system (V1 or V2)
    const selection = SystemSelector.selectSystem({
      clips,
      tracks,
      compositions,
      startTime: this.settings.startTime,
      endTime: this.settings.endTime
    })

    console.log(`Using ${selection.version.toUpperCase()}: ${selection.reason}`)

    // Step 2: Route to appropriate system
    if (selection.version === 'v2') {
      return await this.exportWithV2(onProgress, selection)
    } else {
      return await this.exportWithV1(onProgress)
    }
  }

  private async exportWithV1(onProgress: (progress: ExportProgress) => void): Promise<Blob | null> {
    // Existing V1 implementation (current code)
    // ... current export logic using ParallelDecodeManager ...
  }

  private async exportWithV2(
    onProgress: (progress: ExportProgress) => void,
    selection: SystemSelectionResult
  ): Promise<Blob | null> {
    // V2 implementation
    const bridge = new V2ExportBridge({
      maxCacheMemoryMB: 1000, // Configurable via export settings
      defaultMaxFramesPerFile: 60
    })

    try {
      // Initialize V2 system
      await bridge.initialize(
        clips,
        tracks,
        compositions,
        this.settings.startTime,
        this.settings.endTime,
        this.settings.fps
      )

      // Get schedule for progress estimation
      const schedule = bridge.getSchedule()

      // Export loop - similar to V1 but using V2 bridge
      for (let frame = 0; frame < totalFrames; frame++) {
        const time = startTime + frame * frameDuration

        // Prefetch upcoming frames (look-ahead)
        await bridge.prefetchFrames(time)

        // Get frames for current time
        // ... build layers using bridge.getFrame() ...

        // Render and encode
        // ... existing render/encode logic ...

        // Update progress
        onProgress({ /* ... */ })
      }

      // Get final cache stats
      const cacheStats = bridge.getCacheStats()
      console.log('Cache stats:', cacheStats)

      return blob
    } catch (error) {
      // V2 errors are clear and actionable (no auto-fallback!)
      if (error instanceof ExportError) {
        console.error(`Export failed: ${error.detailedMessage}`)
        console.error(`Suggestion: ${error.suggestedAction}`)
      }
      throw error
    } finally {
      bridge.cleanup()
    }
  }
}
```

### 2. Add Export Settings UI

In `ExportPanel.tsx`:

```typescript
// Add system selection setting
const [exportSystem, setExportSystem] = useState<'auto' | 'v1' | 'v2'>('auto')

// Show recommendation
const recommendation = SystemSelector.getRecommendation({
  clips: useTimelineStore.getState().clips,
  tracks: useTimelineStore.getState().tracks,
  compositions: useMediaStore.getState().compositions,
  startTime,
  endTime
})

// UI
<div className="export-system-selector">
  <label>Export System:</label>
  <select value={exportSystem} onChange={(e) => setExportSystem(e.target.value)}>
    <option value="auto">Automatic (Recommended)</option>
    <option value="v1">Legacy System (V1)</option>
    <option value="v2">Shared Decoders (V2)</option>
  </select>
  <p className="recommendation">{recommendation}</p>
</div>
```

### 3. Manual Override

Allow user to force V1 or V2:

```typescript
// In FrameExporter
const selection = userPreference === 'auto'
  ? SystemSelector.selectSystem(criteria)
  : { version: userPreference, reason: 'User manual override' }

// Warn if user forces V1 but V2 is required
if (userPreference === 'v1' && SystemSelector.isV2Required(criteria)) {
  console.warn('V2 is recommended for this project, but V1 was manually selected')
  // Show warning in UI
}
```

### 4. Error Handling

V2 errors are detailed and actionable:

```typescript
try {
  await exportWithV2(onProgress)
} catch (error) {
  if (error instanceof ExportError) {
    // Show detailed error to user
    showErrorDialog({
      title: 'Export Failed',
      message: error.message,
      details: error.detailedMessage,
      component: error.component,
      clipName: error.clipName,
      suggestion: error.suggestedAction
    })

    // Example UI:
    // ┌─────────────────────────────────────────┐
    // │ Export Failed ❌                         │
    // │                                         │
    // │ Component: Shared Decoder System        │
    // │ File: "video.mp4"                       │
    // │                                         │
    // │ Details:                                │
    // │ Decoder reset failed during seek.       │
    // │ This codec may not support reuse.       │
    // │                                         │
    // │ Suggestion:                             │
    // │ Try Legacy System (V1) in Export        │
    // │ Settings                                │
    // │                                         │
    // │ [Switch to V1]  [Report Bug]           │
    // └─────────────────────────────────────────┘
  }
}
```

### 5. Progress Reporting

V2 has more accurate progress due to scheduling:

```typescript
const schedule = bridge.getSchedule()

// Show estimated time from schedule
console.log(`Estimated export time: ${Math.ceil(schedule.estimatedTime / 60)} minutes`)

// During export, show cache efficiency
const cacheStats = bridge.getCacheStats()
if (cacheStats.hitRate < 0.8) {
  console.warn(`Low cache hit rate: ${(cacheStats.hitRate * 100).toFixed(1)}%`)
}
```

## Testing Strategy

### Phase 1: Simple Projects
- Test with 1-3 clips, no nesting
- Should auto-select V1
- Verify no regression from current behavior

### Phase 2: Medium Projects
- Test with 5-8 clips, light nesting
- Should auto-select V1
- Verify performance matches current system

### Phase 3: Complex Projects
- Test with 10+ clips, deep nesting
- Should auto-select V2
- Verify V2 handles without errors

### Phase 4: Manual Override
- Force V2 on simple project → should work
- Force V1 on complex project → should show warning

### Phase 5: Error Scenarios
- Corrupted file → clear error message
- Memory limit → clear error with suggestion
- Unsupported codec → clear error with fallback suggestion

## Performance Targets

| Project Type | V1 Performance | V2 Target | V2 Improvement |
|--------------|----------------|-----------|----------------|
| Simple (3 clips) | 2x realtime | 3x realtime | 50% faster |
| Medium (8 clips) | 1x realtime | 2x realtime | 2x faster |
| Complex (15 clips) | 0.1x (FAILS) | 1.5x realtime | 15x faster |
| Triple-nested | N/A (FAILS) | 1x realtime | ∞ (enables) |

## Next Steps

1. ✅ Implement core V2 components (SharedDecoderPool, FrameCache, ExportPlanner)
2. ✅ Implement SystemSelector
3. ✅ Create V2ExportBridge
4. 🔜 Integrate into FrameExporter
5. 🔜 Add Export Settings UI
6. 🔜 Test with real projects
7. 🔜 Optimize and polish
