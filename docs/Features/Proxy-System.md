# Proxy System

[← Back to Index](./README.md)

GPU-accelerated proxy generation for smooth editing of large video files.

---

## Table of Contents

- [Overview](#overview)
- [Proxy Generation](#proxy-generation)
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

### Generation Process
- Uses GPU acceleration
- Multi-core processing
- Progress shown in background tasks

### First-Time Setup
First proxy generation prompts for folder:
1. Folder picker dialog appears
2. Select/create proxy folder
3. Folder remembered for future

### Partial Proxies
- Can use proxy while generating
- Frames available immediately
- Falls back to original for missing frames

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

### Preview Quality
- Proxies shown during editing
- Clear enough for decision-making
- Full quality visible in export preview

---

## Storage

### File System Access API
Proxies stored externally using browser File System API:
- User selects storage folder
- Files persist on disk
- Access permission remembered

### Proxy Folder
- All proxies in single folder
- Named by source file hash
- Reusable across sessions

### Storage Requirements
- ~10-20% of original size
- Depends on proxy resolution
- Can be deleted to reclaim space

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

*Commits: 82db433 through d63e381*
