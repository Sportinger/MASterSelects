# Preview & Playback

[← Back to Index](./README.md)

WebGPU preview with RAM caching, multiple panels, edit mode, and multi-output management.

---

## Table of Contents

- [Preview Panel](#preview-panel)
- [Playback Controls](#playback-controls)
- [Preview Quality](#preview-quality)
- [RAM Preview](#ram-preview)
- [Multiple Previews](#multiple-previews)
- [Edit Mode](#edit-mode)
- [Statistics Overlay](#statistics-overlay)
- [Unified RenderTarget System](#unified-rendertarget-system)
- [RenderScheduler](#renderscheduler)
- [Output Manager](#output-manager)
- [Slice & Warp System](#slice--warp-system)
- [Output Window Management](#output-window-management)
- [Output Manager Persistence](#output-manager-persistence)

---

## Preview Panel

### Features
- **Real-time GPU rendering** via WebGPU
- **Aspect ratio preserved** automatically
- **Close button** to hide panel
- **Composition selector** dropdown
- **Edit mode** toggle

### Canvas Registration
All preview canvases register through the unified RenderTarget system (see [Unified RenderTarget System](#unified-rendertarget-system)):
1. Engine assigns a WebGPU context to the canvas via `registerTargetCanvas()`
2. A `RenderTarget` entry is created in `renderTargetStore` with source and destination metadata
3. If the source is independent (not the active composition), the `RenderScheduler` manages its render loop

---

## Playback Controls

### Timeline Toolbar Controls

| Control | Shortcut | Function |
|---------|----------|----------|
| **Stop** | - | Return to time 0 |
| **Play/Pause** | `Space` | Toggle playback |
| **Loop** | `L` | Toggle loop mode |

### In/Out Points

| Shortcut | Action |
|----------|--------|
| `I` | Set In point at playhead |
| `O` | Set Out point at playhead |
| `X` | Clear In/Out points |

### Implementation
```typescript
setInPoint(time)         // Validates against outPoint
setOutPoint(time)        // Validates against inPoint
clearInOut()             // Clear both markers
setInPointAtPlayhead()   // Convenience method
setOutPointAtPlayhead()  // Convenience method
```

---

## Preview Quality

Scale the internal render resolution for better performance on complex compositions or slower hardware.

### Location
View menu → Preview Quality

### Options

| Setting | Render Resolution | Performance | Memory |
|---------|-------------------|-------------|--------|
| **Full (100%)** | 1920×1080 | Baseline | 100% |
| **Half (50%)** | 960×540 | 4× faster | 25% |
| **Quarter (25%)** | 480×270 | 16× faster | 6% |

### What Gets Scaled
- Ping-pong composite buffers
- RAM Preview cache frames
- Scrubbing cache frames
- All GPU shader operations

### What Stays the Same
- Output/export resolution (always full)
- Aspect ratio
- UI element sizes

### Memory Savings at Half Resolution
| Resource | Full (1080p) | Half (540p) | Savings |
|----------|--------------|-------------|---------|
| Scrubbing cache (300 frames) | ~2.4 GB | ~600 MB | 75% |
| RAM Preview (900 frames) | ~7.2 GB | ~1.8 GB | 75% |
| GPU frame cache (60 frames) | ~500 MB | ~125 MB | 75% |

### When to Use Lower Quality
- Complex compositions with many layers
- Real-time effect adjustments
- Slower hardware or integrated GPU
- Large 4K source files
- Timeline scrubbing responsiveness

### Implementation
```typescript
// In settingsStore
setPreviewQuality(quality: 1 | 0.5 | 0.25)

// Applied in useEngine hook
const scaledWidth = Math.round(outputResolution.width * previewQuality);
const scaledHeight = Math.round(outputResolution.height * previewQuality);
engine.setResolution(scaledWidth, scaledHeight);

// Caches cleared automatically on quality change
```

---

## RAM Preview

After Effects-style cached preview for smooth playback.

### Configuration
```typescript
RAM_PREVIEW_FPS = 30        // Target frame rate
FRAME_TOLERANCE = 0.04      // 40ms tolerance for seeks
```

### Cache Limits
| Cache Type | Max Frames | Purpose |
|------------|------------|---------|
| Scrubbing | 300 | Individual video frames |
| Composite | 900 | Fully-rendered frames |
| GPU | 60 | High-speed playback |

### Algorithm
1. Enable via "RAM ON/OFF" button
2. Frames render **outward from playhead**
3. Only caches frames where clips exist
4. Skips empty areas
5. 3-retry seeking with verification

### Smart Seeking
```typescript
// Robust video seeking
- Retries up to 3 times
- Verifies position within FRAME_TOLERANCE
- Handles reversed clips properly
```

### Cache Management
```typescript
toggleRamPreviewEnabled()  // Enable/disable
startRamPreview()          // Begin caching
cancelRamPreview()         // Stop caching
clearRamPreview()          // Clear cache
invalidateCache()          // On content change
getCachedRanges()          // For green indicator
```

### Visual Indicator
- Green bar on timeline shows cached ranges
- **Yellow indicator** on ruler shows proxy cache frames
- Progress indicator during caching
- 2-frame gap tolerance for ranges

### Video Warmup Button
- Cache button for preloading proxy frames before playback
- Ensures smoother initial playback of proxy content
- Shows progress during preload

---

## Multiple Previews

### Adding Preview Panels
1. View menu → Panel visibility
2. Or use "+" button in preview panel

### Composition Selection
Each preview can show different composition:
- Dropdown selector in panel
- "Active" follows current composition
- Or select specific saved composition

### Independent Rendering
Each additional preview panel participates in the unified RenderTarget system:
- Own canvas registered as a `RenderTarget`
- Independent sources rendered by the `RenderScheduler`
- Shared independent ping-pong buffers on the GPU
- Composition evaluation via `compositionRenderer`

### Layout
- Panels appear side-by-side
- Drag to rearrange in dock
- Layout persists on save

---

## Edit Mode

### Enabling Edit Mode
- Click "Edit" button in preview panel
- Or press `Tab` to toggle edit mode on/off

### Layer Selection
- Click layer to select
- Bounding box appears with corner and edge handles
- Handles visible on hover

### Transform Handles
| Handle | Action | Effect |
|--------|--------|--------|
| Corner | Drag | Scale from corner |
| Edge | Drag | Scale from edge |
| Center | Drag | Move layer position |
| Corner + `Shift` | Drag | Scale with locked aspect ratio |

### Drag Operations
| Action | Effect |
|--------|--------|
| Drag center | Move layer position |
| Drag corner handle | Scale layer from corner |
| Drag edge handle | Scale layer from edge |
| `Shift` + drag corner | Lock aspect ratio during scale |

### Bounding Box
```typescript
calculateLayerBounds()
- Accounts for transforms
- Correct aspect ratio
- Matches shader positioning
```

### Zoom & Pan
| Action | Method |
|--------|--------|
| Zoom | `Shift + Scroll` |
| Pan | `Alt + Drag` |
| Reset | Reset button |

---

## Statistics Overlay

### Compact Mode
- FPS (color-coded: green ≥55, yellow ≥30, red <30)
- Decoder type (WebCodecs/HTMLVideo)
- Frame drops this second
- Output resolution

### Expanded Mode (click to expand)
- FPS / target FPS
- Frame gap (RAF timing)
- Render total time
- Pipeline breakdown bars
- Layer count
- Decoder type
- Drops (last second + total)
- Last drop reason
- Bottleneck identification

### Bottleneck Detection
```
Video Import - GPU texture upload slow
GPU Render - Compositing slow
GPU Submit - Command submission slow
```

---

## Frame Caching

### ScrubbingCache Class

#### Tier 1: Scrubbing Frame Cache
```typescript
cacheFrameAtTime(video, time)  // Cache single frame
getCachedFrame(videoSrc, time) // Retrieve cached
// LRU eviction, max 300 frames
```

#### Tier 2: Last Frame Cache
```typescript
captureVideoFrame(video)       // Persistent frame
getLastFrame(video)            // During seeks
// One per video element
```

#### Tier 3: RAM Preview Composite Cache
```typescript
cacheCompositeFrame(time, imageData)  // Composited frame
getCachedCompositeFrame(time)         // Instant retrieval
hasCompositeCacheFrame(time)          // Existence check
// Max 900 frames, stored as ImageData
```

---

## Composition Rendering

### Service Methods
```typescript
prepareComposition(compositionId)
- Loads all video/image sources
- Waits for canplaythrough
- Handles both active and saved compositions

evaluateAtTime(compositionId, time)
- Returns layers ready for rendering
- Handles clip trimming
- Handles reversed clips
- Builds layer transforms
- Automatic video seeking
```

---

## Performance

### Frame Rate
- 60fps target for preview
- 30fps limit when video playing
- Frame drop detection (1.5x target)

### Optimization
- Skip caching during playhead drag
- Reuse already-cached frames
- Video paused during RAM Preview generation

---

## Unified RenderTarget System

All preview outputs (main preview, additional preview panels, output windows) use a unified RenderTarget system for rendering.

### RenderTarget

Each output is a `RenderTarget` with a source and a destination:

| Property | Description |
|----------|-------------|
| **Source** | What to display: active composition, specific composition, layer, slot, or program mix |
| **Destination** | Where to display: canvas element, popup window, or browser tab |
| **Enabled** | Toggle rendering on/off per target |
| **Fullscreen** | Toggle fullscreen mode per window |

### Source Types

| Source Type | Description |
|-------------|-------------|
| **Active Comp** | Follows whichever composition is currently open in the Timeline editor |
| **Composition** | Renders a specific composition by ID (independent of editor) |
| **Layer** | Renders specific layers from the active layer slots |
| **Slot** | Renders a slot from the multi-layer slot grid |
| **Program** | Main mix output (all layers composited) |

### Registration Flow
1. Canvas element registers via `registerTargetCanvas()`
2. Engine assigns a WebGPU context to the canvas
3. Target entry created in `renderTargetStore` with source/destination metadata
4. If source is independent (not active comp), `RenderScheduler` manages its render loop

---

## RenderScheduler

The RenderScheduler service manages independent render loops for targets that don't follow the active composition.

| Feature | Description |
|---------|-------------|
| **Independent RAF loops** | Each non-active-comp target gets its own `requestAnimationFrame` loop |
| **Composition evaluation** | Evaluates layers at the correct time for each target's source |
| **Automatic registration** | Targets with independent sources auto-register on creation |
| **Cleanup** | Loops stop when targets are removed or disabled |

---

## Output Manager

The Output Manager is a dedicated interface for managing multiple output targets, applying corner-pin warping (slices), and routing sources to different displays. Useful for projection mapping, multi-screen setups, and VJ performances.

### Opening the Output Manager
- Menu: **Output → Output Manager**
- Opens in a new browser popup window

### Layout

| Area | Description |
|------|-------------|
| **Sidebar (left)** | Target list with nested slices, source selectors, controls |
| **Preview (center)** | Live preview canvas showing the selected target with slices applied |
| **Tab Bar (top)** | Switch between Input and Output views |

### Target Management

| Action | How |
|--------|-----|
| **Add Output Window** | Click "+" button → opens new popup window |
| **Select Source** | Dropdown per target: Active Comp, specific composition, slot, etc. |
| **Rename** | Double-click the target name to edit inline |
| **Enable/Disable** | Toggle switch per target |
| **Close Window** | Close button (window becomes grayed out with Restore option) |

### Save & Exit
- **Save & Exit** button saves all configurations and closes the Output Manager
- Configurations persist per-project in localStorage

---

## Slice & Warp System

Slices map a rectangular input region to a quadrilateral output area via corner-pin warping. Each output target can have multiple slices and mask layers.

### How Slices Work

Each slice has two sets of four corner points in normalized (0–1) coordinates:

| Side | Description |
|------|-------------|
| **Input Corners** | Define which rectangular region of the source to display (clamped to 0–1) |
| **Output Corners** | Define where that region appears in the output (unclamped, can exceed bounds for warping) |

### Input Tab
- Shows the source content with draggable corner points
- Drag corners to select a sub-region of the source
- Supports zoom (Shift+Scroll) and pan (Alt+Drag)
- Right-click context menu: "Match Input to Output Shape"

### Output Tab
- Shows the output canvas with draggable corner points
- Drag corners to warp/stretch the slice into any quadrilateral shape
- Outlines and vertices visible even outside the canvas bounds
- Right-click context menu: "Match Output to Input Shape"

### Slice Controls

| Action | How |
|--------|-----|
| **Add Slice** | "Add Slice" button in sidebar |
| **Add Mask** | "Add Mask" button in sidebar |
| **Rename** | Double-click the slice name |
| **Enable/Disable** | Toggle switch per slice |
| **Reorder** | Drag handle for drag-and-drop reordering |
| **Reset** | Reset corners to default positions |
| **Delete** | Delete button per slice |

### Mask Layers

Mask layers are slices with `type: 'mask'` that control pixel visibility:

| Property | Description |
|----------|-------------|
| **Normal mode** | Pixels outside the mask quad are transparent |
| **Inverted mode** | Pixels inside the mask quad are transparent |
| **Visual style** | Displayed as dashed red outlines in both Input/Output views |
| **Non-interactive on Input** | Mask corners are view-only in Input tab, editable in Output tab |

---

## Output Window Management

### Creating Output Windows
- Click "+" in Output Manager sidebar to open a new popup window
- Each window is a full render target with its own source routing

### Window Restore-on-Close
| State | Appearance |
|-------|------------|
| **Open** | Active window with live rendering |
| **Closed** | Grayed-out entry in sidebar with "Restore" button |
| **Restored** | Re-opens at previous position and size |

Window geometry (position, size) is preserved even after closing, so restored windows reappear in the same screen location.

### Window Reconnection
- On page refresh, output windows attempt to reconnect via sessionStorage flag
- Named popup windows allow the browser to find existing windows
- Prevents duplicate windows from spawning on refresh

---

## Output Manager Persistence

### Auto-Save
- Slice configurations auto-save on every change (debounced 500ms)
- Saved per-project using localStorage key: `Outputmanager_{ProjectName}`
- Window geometry included in saved metadata

### What Gets Saved
| Data | Storage |
|------|---------|
| Slice configurations (corners, warp, masks) | localStorage per project |
| Target metadata (name, source, window geometry) | localStorage per project |
| Selected slice state | Transient (not persisted) |

### Load on Boot
1. Output Manager mounts and loads saved config from localStorage
2. Closed targets restore as grayed-out entries
3. Window geometry preserved for restoration
4. Slice configs applied immediately to render pipeline

---

## Related Features

- [Timeline](./Timeline.md) - Main editing interface
- [Export](./Export.md) - Render to file
- [GPU Engine](./GPU-Engine.md) - Rendering details
- [Keyboard Shortcuts](./Keyboard-Shortcuts.md)

---

*Source: `src/components/preview/Preview.tsx`, `src/components/outputManager/`, `src/stores/renderTargetStore.ts`, `src/stores/sliceStore.ts`, `src/services/renderScheduler.ts`*
