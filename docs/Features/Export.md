# Export

[← Back to Index](./README.md)

Render compositions to video files or image sequences.

---

## Table of Contents

- [Export Panel](#export-panel)
- [Export Options](#export-options)
- [Frame Export](#frame-export)
- [Export Process](#export-process)

---

## Export Panel

### Opening Export Panel
- View menu → Export Panel
- Or dock panel tabs

### Panel Contents
- Format selection
- Resolution settings
- In/Out range
- Export button

---

## Export Options

### Resolution

| Option | Description |
|--------|-------------|
| **Composition** | Use composition resolution |
| **Custom** | Set custom dimensions |
| **Presets** | 720p, 1080p, 4K |

### Range

| Option | Description |
|--------|-------------|
| **Full** | Entire composition |
| **In/Out** | Between markers |
| **Custom** | Manual time range |

### Format
- WebM (VP9)
- MP4 (if supported)
- Image sequence

---

## Frame Export

### Render Current Frame
Export single frame as image:
1. Position playhead
2. Click "Render Frame"
3. Download PNG

### Technical Details
- Reads directly from GPU
- Full resolution output
- Proper GPU buffer handling

---

## Export Process

### Starting Export
1. Configure settings
2. Click "Export"
3. Progress shown in panel

### Progress Indication
- Frame counter
- Percentage complete
- Estimated time remaining

### Export Pipeline
1. Each frame rendered via GPU
2. Encoded to video format
3. Buffered for output
4. Final file assembled

### Cancellation
- Cancel button available
- Stops at current frame
- Partial file may exist

---

## In/Out Points

### Setting Points
| Key | Action |
|-----|--------|
| `I` | Set In point |
| `O` | Set Out point |

### Visual Markers
- In/Out shown on ruler
- Highlighted region
- Used for playback loop

### Export Range
- Export respects In/Out points
- Or override with custom range
- Clear to export full composition

---

## Quality Settings

### Video Quality
- Bitrate control
- Quality presets
- Format-specific options

### Encoding
- Uses browser codecs
- WebCodecs when available
- Fallback to MediaRecorder

---

## Export Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| Black frames | Check layer visibility |
| No audio | Audio export not yet implemented |
| Slow export | Reduce resolution or effects |
| Export fails | Check browser codec support |

### GPU Considerations
- Export uses WebGPU rendering
- Same quality as preview
- Texture buffers properly tracked

---

## Related Features

- [Preview](./Preview.md) - Preview before export
- [Timeline](./Timeline.md) - Set In/Out points
- [GPU Engine](./GPU-Engine.md) - Rendering details
- [Effects](./Effects.md) - Effect rendering

---

*Commits: fa36b80 through d63e381*
