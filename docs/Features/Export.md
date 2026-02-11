# Export

[← Back to Index](./README.md)

Frame-by-frame video export with H.264/VP9 encoding and four export modes.

---

## Table of Contents

- [Export Modes](#export-modes)
- [Export Panel](#export-panel)
- [Export Settings](#export-settings)
- [Audio Settings](#audio-settings)
- [Export Process](#export-process)
- [Frame Export](#frame-export)
- [FFmpeg Export](#ffmpeg-export)

---

## Export Modes

MASterSelects offers four export modes optimized for different use cases:

### WebCodecs Fast Mode

**Best for: Simple timelines, maximum speed**

- Uses sequential decoding with MP4Box parsing
- Creates dedicated WebCodecs players per clip
- Parallel decoding for multi-clip exports
- Auto-extracts avcC/hvcC descriptions for H.264/H.265
- Falls back to Precise mode if codec unsupported (e.g., AV1)

```
Pipeline: MP4Box → WebCodecs Decoder → GPU Compositor → VideoEncoder
```

### HTMLVideo Precise Mode

**Best for: Complex timelines, nested compositions**

- Uses HTMLVideoElement seeking (frame-accurate)
- Handles all codec types the browser supports
- Better for clips with complex timing
- Slower but more reliable for edge cases

```
Pipeline: HTMLVideoElement → requestVideoFrameCallback → GPU Compositor → VideoEncoder
```

### FFmpeg WASM Export

**Best for: Professional codecs (ProRes, DNxHR, HAP)**

- Loads FFmpeg WASM on-demand (~20MB)
- Supports broadcast-quality codecs
- Requires SharedArrayBuffer headers
- See [FFmpeg Export](#ffmpeg-export) section below

### FCP XML Export

**Best for: Interchange with other NLEs**

- Exports timeline to Final Cut Pro XML format
- Compatible with Premiere Pro, DaVinci Resolve, and other NLEs
- Preserves clip positions, durations, and track layout
- Useful for roundtripping between applications

---

## Export Panel

### Location
- View menu → Export Panel
- Or dock panel tabs

### Panel Contents
- Resolution presets
- Frame rate options
- Quality/bitrate selection
- Time range (In/Out)
- Progress indicator
- Export/Cancel buttons

---

## Export Settings

### Resolution Presets
| Preset | Resolution |
|--------|------------|
| 4K | 3840×2160 |
| 1080p | 1920×1080 |
| 720p | 1280×720 |
| 480p | 854×480 |
| Custom | User-defined |

### Frame Rate
| Rate | Use Case |
|------|----------|
| 60fps | High motion |
| 30fps | Standard |
| 25fps | PAL |
| 24fps | Film |

### Codec Options
| Codec | Container | ID |
|-------|-----------|-----|
| H.264 | MP4 | avc1.640028 |
| VP9 | WebM | vp09.00.10.08 |

### Quality Presets
| Quality | Bitrate |
|---------|---------|
| Low | 5 Mbps |
| Medium | 15 Mbps |
| High | 25 Mbps |
| Maximum | 35 Mbps |

---

## Audio Settings

### Include Audio
Checkbox to enable audio export (default: enabled).

### Sample Rate
| Rate | Description |
|------|-------------|
| 48 kHz | Video standard (recommended) |
| 44.1 kHz | CD quality |

### Audio Quality (Bitrate)
| Quality | Bitrate |
|---------|---------|
| Good | 128 kbps |
| Better | 192 kbps |
| High Quality | 256 kbps |
| Maximum | 320 kbps |

### Normalize
Peak normalize to prevent clipping. Reduces gain if mixed audio exceeds 0dB.

### Audio Processing
When audio is exported:
1. **Extraction**: Audio decoded from source files
2. **Speed/Pitch**: SoundTouchJS applies tempo changes with pitch preservation
3. **Effects**: EQ and volume rendered with keyframe automation
4. **Mixing**: All tracks mixed, respecting mute/solo
5. **Encoding**: AAC-LC via WebCodecs

### Codec
| Codec | Container | Description |
|-------|-----------|-------------|
| AAC-LC | MP4 | mp4a.40.2 - Universal compatibility |

---

## Export Process

### Pipeline
```
Video Phase (95% of progress):
1. Prepare clips (load MP4Box players for Fast mode)
2. Parallel decode multiple clips simultaneously
3. Build layer composition
4. Render via GPU engine
5. Read pixels (staging buffer)
6. Create VideoFrame
7. Encode frame
8. Write to muxer
9. Repeat for all frames

Audio Phase (5% of progress):
1. Extract audio from all clips
2. Apply speed/pitch processing
3. Render EQ and volume effects
4. Mix all tracks
5. Encode to AAC/Opus
6. Add audio chunks to muxer
```

### Parallel Decoding

For multi-clip exports, ParallelDecodeManager handles:

- Concurrent decoding of multiple clips
- 60-frame buffer per clip
- Batch decode operations
- Smart flush timing
- Timestamp-based frame tracking

```typescript
// ParallelDecodeManager.ts
- Creates dedicated decoder per clip
- Batch decodes frames ahead of export
- Frame buffer prevents export stalls
```

### Progress Tracking
- Timeline overlay progress bar
- Frame counter: `X / Total`
- Percentage complete
- Cancel button in overlay

### Video Seeking
```typescript
// Per-clip seeking with timeout
- 1 second timeout per clip
- Handles reversed clips
- Respects track visibility
- Respects solo settings
```

### Key Frame Insertion
Every 30 frames (configurable).

### Audio Codec Detection
```typescript
// Auto-detects browser support
- AAC-LC (mp4a.40.2) - preferred
- Opus - fallback for Linux/WebM
```

---

## Frame Export

### Single Frame Export
Export current frame as PNG:
1. Position playhead
2. Click "Render Frame"
3. Downloads PNG file

### Technical Details
```typescript
// FrameExporter.ts
1. Call engine.render() at time
2. Create staging buffer
3. Copy texture to buffer
4. Map buffer for read
5. Create PNG blob
6. Trigger download
```

---

## Time Range

### Full Export
Exports entire composition duration.

### In/Out Export
Uses In/Out markers if set:
```typescript
startTime = inPoint ?? 0
endTime = outPoint ?? duration
```

### Setting In/Out
| Shortcut | Action |
|----------|--------|
| `I` | Set In point |
| `O` | Set Out point |
| `X` | Clear both |

---

## Output

### File Generation
- MP4 container for H.264
- WebM container for VP9
- Uses `mp4-muxer` library

### Download
Automatic browser download when complete:
```typescript
const blob = muxer.finalize();
const url = URL.createObjectURL(blob);
// Trigger download
```

---

## Estimated File Size

Panel shows estimated output size:
```
duration × frameRate × bitrate / 8
```

Example: 60s × 30fps × 15Mbps = ~112MB

---

## Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| Black frames | Check layer visibility |
| Slow export | Reduce resolution |
| Export fails | Check codec support |
| Large file | Reduce bitrate |

### Browser Compatibility
- Requires WebCodecs API
- Chrome/Edge recommended
- Falls back gracefully

---

## FFmpeg Export

### Overview
FFmpeg WASM integration provides professional codec support for broadcast and VJ workflows. Loads on-demand (~20MB).

### Professional Codecs

| Codec | Category | Description |
|-------|----------|-------------|
| **ProRes** | Professional | Apple ProRes (Proxy, LT, 422, HQ, 4444, 4444 XQ) |
| **DNxHR** | Professional | Avid DNxHR (LB, SQ, HQ, HQX, 444) |
| **HAP** | Real-time | GPU-accelerated VJ codec (HAP, HAP Alpha, HAP Q) |
| **FFV1** | Lossless | Open archival codec |
| **Ut Video** | Lossless | Fast lossless with alpha |

### Delivery Codecs

| Codec | Features |
|-------|----------|
| H.264 (x264) | Universal compatibility |
| H.265 (x265) | HDR support, smaller files |
| VP9 | Alpha channel support |
| AV1 (SVT) | Next-gen, best compression |

### Container Formats

| Format | Use Case |
|--------|----------|
| MOV | Apple/Pro workflows (ProRes) |
| MP4 | Universal delivery |
| MKV | Open format, all codecs |
| WebM | Web optimized |
| MXF | Broadcast (DNxHR) |

### Platform Presets

| Preset | Codec | Container |
|--------|-------|-----------|
| YouTube | H.264 | MP4 |
| YouTube HDR | H.265 | MP4 |
| Vimeo | H.264 | MP4 |
| Instagram | H.264 | MP4 |
| TikTok | H.264 | MP4 |
| Adobe Premiere | ProRes HQ | MOV |
| Final Cut Pro | ProRes HQ | MOV |
| DaVinci Resolve | DNxHR HQ | MXF |
| Avid | DNxHR HQ | MXF |
| VJ / Media Server | HAP Q | MOV |
| Archive | FFV1 | MKV |

### Loading FFmpeg
FFmpeg WASM is loaded on-demand when first used:
1. Click "Load FFmpeg" button
2. Downloads from CDN (~20MB)
3. Ready indicator shows when loaded

### Technical Notes
- Requires SharedArrayBuffer (COOP/COEP headers)
- Uses @ffmpeg/ffmpeg from npm
- Frames rendered via GPU, then encoded by FFmpeg
- Professional codecs (ProRes, HAP, DNxHR) require custom WASM build

### Source Files
- `src/engine/ffmpeg/FFmpegBridge.ts` - Core bridge
- `src/engine/ffmpeg/codecs.ts` - Codec definitions
- `src/components/export/FFmpegExportSection.tsx` - UI

---

## Export System V2 (Shared Decoder Pool)

### Overview
Export V2 introduces a shared decoder architecture for more efficient multi-clip exports. Instead of creating separate decoders per clip instance, V2 shares decoder instances per unique file.

### Components

| Component | Purpose |
|-----------|---------|
| **SharedDecoderPool** | Manages one VideoDecoder per unique file, with reuse via reset + configure |
| **ExportPlanner** | Analyzes timeline to optimize decode scheduling and cache allocation |
| **FrameCacheManager** | Integrated frame cache for decoded frames |
| **SystemSelector** | Selects between V1 and V2 based on timeline complexity |
| **V2ExportBridge** | Bridges V2 system into existing export pipeline |

### ExportPlanner Features
- Analyzes full export range to understand file usage patterns
- Detects heavy-usage files (>20% of export) for larger cache allocation
- Plans look-ahead decode scheduling (2-3 seconds ahead)
- Minimizes decoder switches and seeks
- Groups clips by file for efficient decoding

### SharedDecoderPool Features
- One decoder instance per unique file (not per clip instance)
- Decoder reuse via `reset()` + `configure()`
- Smart position tracking to minimize seeks
- 30-frame buffer ahead per decoder
- Statistics tracking (total decoded, seeks, resets)

### Source Files
- `src/engine/export/v2/SharedDecoderPool.ts`
- `src/engine/export/v2/ExportPlanner.ts`
- `src/engine/export/v2/FrameCacheManager.ts`
- `src/engine/export/v2/V2ExportBridge.ts`
- `src/engine/export/v2/SystemSelector.ts`

---

## Not Implemented

- Multi-pass encoding
- Background export
- Opus/FLAC audio codecs

---

## Related Features

- [Preview](./Preview.md) - Preview before export
- [Timeline](./Timeline.md) - Set In/Out points
- [GPU Engine](./GPU-Engine.md) - Rendering details

---

*Source: `src/engine/export/`, `src/engine/ParallelDecodeManager.ts`, `src/engine/audio/`, `src/components/export/ExportPanel.tsx`*
