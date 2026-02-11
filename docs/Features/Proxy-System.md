# Proxy System

[← Back to Index](./README.md)

GPU-accelerated proxy generation for smooth editing of large video files.

---

## Table of Contents

- [Overview](#overview)
- [Proxy Generation](#proxy-generation)
- [Cross-Platform Support](#cross-platform-support)
- [Proxy Playback](#proxy-playback)
- [Storage](#storage)
- [Configuration](#configuration)

---

## Overview

### Purpose
Large video files (4K, high bitrate) can be slow to scrub. Proxies provide:
- Smaller, faster decode files
- Smooth timeline scrubbing
- Full quality on export

### How It Works
1. Generate low-res proxy of video
2. Edit using proxy files
3. Final export uses original media

---

## Proxy Generation

### Starting Generation
1. Right-click video in Media Panel
2. Select "Generate Proxy"
3. Choose storage folder (first time)
4. Generation starts in background

### Generation Process (GPU-Accelerated)
The proxy generator uses a rewritten pipeline for maximum speed:

1. **Video Decoding**: WebCodecs VideoDecoder with hardware acceleration
2. **GPU Batch Resize**: Frames rendered to texture atlas per batch
3. **Single Buffer Readback**: GPU→CPU transfer per batch
4. **Parallel JPEG Encoding**: Worker pool encodes frames simultaneously

**Performance**: 4-10x faster than CPU-only processing

### Resume from Disk
- Proxy generation can be interrupted and **resumed from disk**
- If generation is interrupted (browser close, crash), it picks up where it left off
- Already-generated frames on disk are skipped automatically
- No need to start over from scratch

### Technical Details
- **Max Resolution**: 1280px width (configurable)
- **Batch Size**: 16 frames per GPU pass
- **Output Format**: WebP at 92% quality
- **Frame Rate**: 30 fps proxy

### Automatic Project Folder Storage
Proxies are automatically stored in your project folder:
```
MyProject/
└── Proxy/
    └── {mediaHash}/
        └── frames/
            ├── 000000.webp
            ├── 000001.webp
            └── ...
```

No folder picker needed - proxies go directly to project folder.

### Partial Proxies
- Can use proxy while generating
- Frames available immediately
- Falls back to original for missing frames

---

## Cross-Platform Support

### Windows (NVIDIA)
Streaming decode mode for Windows NVIDIA GPUs:
- Processes frames during decoding (not after)
- Releases decoder buffer memory as frames complete
- Active wait loop handles slow hardware decoders
- Prevents stalling on limited DPB (Decoded Picture Buffer)

### Linux
Standard high-performance decode:
- Hardware-accelerated via VA-API/VDPAU
- Batch processing for maximum throughput

### macOS
- VideoToolbox hardware decoding
- Same streaming approach as Windows

### Performance Tips
| Issue | Solution |
|-------|----------|
| Slow on Windows | Uses streaming decode automatically |
| Stalls at 0% | Check GPU drivers, try different video |
| Black frames | Verify WebCodecs support |

---

## Proxy Playback

### Automatic Switching
Editor automatically uses:
- Proxy frames when available
- Original video when proxy missing
- Seamless transition between

### Timeline Integration
- Proxy frames display in preview
- Scrubbing uses proxy cache
- Playback synced with timeline
- **Yellow indicator** on timeline ruler shows cached proxy frames (proxy cache indicator)
- **Warmup button**: Preload proxy frames into cache before playback for smoother start

### Preview Quality
- Proxies shown during editing
- Clear enough for decision-making
- Full quality visible in export preview

---

## Storage

### Project Folder Storage
Proxies stored in your project folder:
- No separate folder selection needed
- Files persist with project
- Hash-based deduplication

### File Organization
```
ProjectFolder/Proxy/{mediaHash}/frames/
```
- `{mediaHash}` = SHA-256 of file content
- Same file imported twice shares proxies
- Portable with project folder

### Storage Requirements
- ~10-20% of original video size
- Depends on proxy resolution (1280px max)
- Delete `Proxy/` folder to reclaim space

### Deduplication
Files are identified by content hash:
- Same video = same proxies
- Re-import doesn't regenerate
- Thumbnails also deduplicated

---

## Configuration

### Default Settings
- Proxy generation disabled by default
- Enable in settings
- Or generate manually per-file

### Toggle Proxy Mode
When proxies exist:
- Preview uses proxy
- Toggle to show original
- Useful for quality check

### Proxy Resolution
- Lower resolution than original
- Typically 1/4 or 1/2 size
- Configurable in settings

---

## Background Processing

### Progress Indication
- Shows in background tasks
- Frame count progress
- Cancelable

### Resource Usage
- GPU accelerated
- Doesn't block UI
- Can edit while generating

### Logging
Background process logging shows:
- Generation progress
- Frame timing
- Completion status

---

## Troubleshooting

### Proxy Not Used
- Check if proxy exists
- Verify folder access
- Check file permissions

### Slow Generation
- GPU acceleration required
- Check chrome://gpu
- Large files take time

### Storage Full
- Delete old proxies
- Choose different folder
- Check disk space

---

## Related Features

- [Media Panel](./Media-Panel.md) - Proxy controls
- [GPU Engine](./GPU-Engine.md) - GPU acceleration
- [Preview](./Preview.md) - Proxy playback
- [Project Persistence](./Project-Persistence.md) - Proxy paths

---

## Tests

No dedicated unit tests — this feature requires hardware-dependent APIs (WebCodecs, GPU batch resize) that cannot be easily mocked.

---

*Commits: 82db433 through d63e381*
