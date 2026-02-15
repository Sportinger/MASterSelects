# Timeline.tsx Refactoring Plan

> **Target Audience:** AI Agent performing the refactor
> **Source File:** `src/components/timeline/Timeline.tsx` (2109 LOC)
> **Goal:** Split into focused modules, remove duplicates, target ~900 LOC main file

---

## Quick Reference

| New File | LOC | Extract From (Lines) | Purpose |
|----------|-----|---------------------|---------|
| `utils/fileTypeHelpers.ts` | ~60 | 817-877 | File type detection |
| `hooks/useExternalDrop.ts` | ~400 | 879-1335 | External drag & drop |
| `hooks/usePlaybackLoop.ts` | ~110 | 600-708 | Audio master clock |
| ~~`hooks/useVideoPreload.ts`~~ | ~~90~~ | ~~509-598~~ | ~~REMOVED: Conflicted with VideoSyncManager.preloadUpcomingClips, caused stutter at cuts~~ |
| `hooks/useAutoFeatures.ts` | ~80 | 375-481 | RAM preview & proxy auto-start |
| `components/TimelineOverlays.tsx` | ~150 | 1812-1916 + 2039-2084 | All overlay elements |
| `components/NewTrackDropZone.tsx` | ~50 | 1687-1712, 1784-1810 | Drop zone component |
| `Timeline.tsx` | ~900 | Remainder | Main component |

---

## Pre-Refactor Checklist

```bash
# 1. Verify app runs
npm run dev

# 2. Create backup branch
git checkout -b refactor/timeline-backup
git checkout -b refactor/timeline-split
```

---

## File Structure After Refactor

```
src/components/timeline/
├── Timeline.tsx                    # MODIFY: Main component (~900 LOC)
├── hooks/
│   ├── useTimelineKeyboard.ts      # EXISTING
│   ├── useTimelineZoom.ts          # EXISTING
│   ├── usePlayheadDrag.ts          # EXISTING
│   ├── useMarqueeSelection.ts      # EXISTING
│   ├── useClipTrim.ts              # EXISTING
│   ├── useClipDrag.ts              # EXISTING
│   ├── useLayerSync.ts             # EXISTING
│   ├── useExternalDrop.ts          # NEW
│   ├── usePlaybackLoop.ts          # NEW
│   # useVideoPreload.ts REMOVED — caused stutter at cuts (VideoSyncManager handles preloading)
│   └── useAutoFeatures.ts          # NEW
├── components/
│   ├── TimelineOverlays.tsx        # NEW
│   └── NewTrackDropZone.tsx        # NEW
└── utils/
    └── fileTypeHelpers.ts          # NEW
```

---

## Step-by-Step Extraction

### Step 1: Create `utils/fileTypeHelpers.ts`

**File:** `src/components/timeline/utils/fileTypeHelpers.ts`

**Action:** Create new file with file type detection utilities.

```typescript
// File type detection utilities for timeline drag & drop

const VIDEO_EXTENSIONS = [
  'mov', 'mp4', 'm4v', 'mxf', 'avi', 'mkv', 'webm',  // Common
  'ts', 'mts', 'm2ts',                               // Transport streams
  'wmv', 'asf', 'flv', 'f4v',                        // Windows/Flash
  '3gp', '3g2', 'ogv', 'vob', 'mpg', 'mpeg',         // Other
];

const AUDIO_EXTENSIONS = [
  'mp3', 'wav', 'ogg', 'aac', 'm4a', 'flac', 'wma', 'aiff', 'alac', 'opus',
];

const IMAGE_EXTENSIONS = [
  'jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp', 'tiff', 'tif', 'heic', 'heif',
];

/**
 * Check if file is a video by MIME type or extension
 */
export function isVideoFile(file: File): boolean {
  if (file.type.startsWith('video/')) return true;
  const ext = file.name.split('.').pop()?.toLowerCase() || '';
  return VIDEO_EXTENSIONS.includes(ext);
}

/**
 * Check if file is an audio file by MIME type or extension
 */
export function isAudioFile(file: File): boolean {
  if (file.type.startsWith('audio/')) return true;
  const ext = file.name.split('.').pop()?.toLowerCase() || '';
  return AUDIO_EXTENSIONS.includes(ext);
}

/**
 * Check if file is any media type (video/audio/image)
 */
export function isMediaFile(file: File): boolean {
  if (
    file.type.startsWith('video/') ||
    file.type.startsWith('audio/') ||
    file.type.startsWith('image/')
  ) {
    return true;
  }
  const ext = file.name.split('.').pop()?.toLowerCase() || '';
  return (
    VIDEO_EXTENSIONS.includes(ext) ||
    AUDIO_EXTENSIONS.includes(ext) ||
    IMAGE_EXTENSIONS.includes(ext)
  );
}

/**
 * Quick duration check for dragged video files
 * Returns null if not a video or duration cannot be determined
 */
export async function getVideoDurationQuick(
  file: File,
  timeoutMs = 3000
): Promise<number | null> {
  if (!isVideoFile(file)) return null;

  return new Promise((resolve) => {
    const video = document.createElement('video');
    video.preload = 'metadata';

    const cleanup = () => {
      URL.revokeObjectURL(video.src);
      video.remove();
    };

    video.onloadedmetadata = () => {
      const dur = video.duration;
      cleanup();
      resolve(isFinite(dur) ? dur : null);
    };

    video.onerror = () => {
      cleanup();
      resolve(null);
    };

    const timeoutId = setTimeout(() => {
      cleanup();
      resolve(null);
    }, timeoutMs);

    video.onloadedmetadata = () => {
      clearTimeout(timeoutId);
      const dur = video.duration;
      cleanup();
      resolve(isFinite(dur) ? dur : null);
    };

    video.src = URL.createObjectURL(file);
  });
}
```

**Validation:** `npx tsc src/components/timeline/utils/fileTypeHelpers.ts --noEmit --skipLibCheck`

---

### Step 2: Create `hooks/usePlaybackLoop.ts`

**File:** `src/components/timeline/hooks/usePlaybackLoop.ts`

**Action:** Extract playback loop (audio master clock) from lines 600-708.

```typescript
// Playback loop with audio master clock synchronization

import { useEffect } from 'react';
import { useTimelineStore } from '../../../stores/timeline';
import { playheadState } from '../../../services/layerBuilder';

interface UsePlaybackLoopProps {
  isPlaying: boolean;
}

/**
 * Audio Master Clock playback loop
 * Audio runs freely without correction, playhead follows audio time
 * This eliminates audio drift and clicking from constant seeks
 */
export function usePlaybackLoop({ isPlaying }: UsePlaybackLoopProps) {
  useEffect(() => {
    if (!isPlaying) {
      // Disable internal position tracking when not playing
      playheadState.isUsingInternalPosition = false;
      playheadState.hasMasterAudio = false;
      playheadState.masterAudioElement = null;
      return;
    }

    let rafId: number;
    let lastTime = performance.now();
    let lastStateUpdate = 0;
    const STATE_UPDATE_INTERVAL = 33; // Update store every 33ms (~30fps for UI/subscribers)

    // Initialize internal position from store and enable high-frequency mode
    playheadState.position = useTimelineStore.getState().playheadPosition;
    playheadState.isUsingInternalPosition = true;
    playheadState.playbackJustStarted = true; // Signal for initial audio sync

    const updatePlayhead = (currentTime: number) => {
      const state = useTimelineStore.getState();
      const {
        duration: dur,
        inPoint: ip,
        outPoint: op,
        loopPlayback: lp,
        pause: ps,
        clips,
      } = state;
      const effectiveEnd = op !== null ? op : dur;
      const effectiveStart = ip !== null ? ip : 0;

      let newPosition: number;

      // AUDIO MASTER CLOCK: If we have an active audio element, derive playhead from its time
      if (playheadState.hasMasterAudio && playheadState.masterAudioElement) {
        const audio = playheadState.masterAudioElement;
        if (!audio.paused && audio.readyState >= 2) {
          // Calculate timeline position from audio's current time
          // audioTime = clipInPoint + (timelinePosition - clipStartTime) * speed
          // So: timelinePosition = clipStartTime + (audioTime - clipInPoint) / speed
          const audioTime = audio.currentTime;
          const speed = playheadState.masterClipSpeed || 1;
          newPosition =
            playheadState.masterClipStartTime +
            (audioTime - playheadState.masterClipInPoint) / speed;
        } else {
          // Audio paused or not ready, fall back to system time
          const deltaTime = (currentTime - lastTime) / 1000;
          const cappedDelta = Math.min(deltaTime, 0.1);
          newPosition = playheadState.position + cappedDelta;
        }
      } else {
        // No audio master - use system time (fallback for video-only or image clips)
        const deltaTime = (currentTime - lastTime) / 1000;
        const cappedDelta = Math.min(deltaTime, 0.1);
        newPosition = playheadState.position + cappedDelta;
      }
      lastTime = currentTime;

      // Handle end of timeline / looping
      if (newPosition >= effectiveEnd) {
        if (lp) {
          newPosition = effectiveStart;
          // Reset audio master - will be re-established by syncAudioElements
          playheadState.hasMasterAudio = false;
          playheadState.masterAudioElement = null;
          // Seek all audio/video to start
          clips.forEach((clip) => {
            if (clip.source?.audioElement) {
              clip.source.audioElement.currentTime = clip.inPoint;
            }
            if (clip.source?.videoElement) {
              clip.source.videoElement.currentTime = clip.reversed
                ? clip.outPoint
                : clip.inPoint;
            }
          });
        } else {
          newPosition = effectiveEnd;
          ps();
          playheadState.position = newPosition;
          playheadState.isUsingInternalPosition = false;
          playheadState.hasMasterAudio = false;
          playheadState.masterAudioElement = null;
          useTimelineStore.setState({ playheadPosition: newPosition });
          return;
        }
      }

      // Clamp to start
      if (newPosition < effectiveStart) {
        newPosition = effectiveStart;
      }

      // Update high-frequency position for render loop to read
      playheadState.position = newPosition;

      // PERFORMANCE: Only update store at throttled interval
      if (currentTime - lastStateUpdate >= STATE_UPDATE_INTERVAL) {
        useTimelineStore.setState({ playheadPosition: newPosition });
        lastStateUpdate = currentTime;
      }

      rafId = requestAnimationFrame(updatePlayhead);
    };

    rafId = requestAnimationFrame(updatePlayhead);

    return () => {
      cancelAnimationFrame(rafId);
      playheadState.isUsingInternalPosition = false;
      playheadState.hasMasterAudio = false;
      playheadState.masterAudioElement = null;
    };
  }, [isPlaying]);
}
```

---

### ~~Step 3: Create `hooks/useVideoPreload.ts`~~ — REMOVED

> **REMOVED:** This hook was deleted because it conflicted with `VideoSyncManager.preloadUpcomingClips()`.
> It seeked shared video elements (split clips) to the upcoming clip's `inPoint`, disrupting
> the currently-playing clip. VideoSyncManager already handles all preloading correctly with
> shared-element detection and two-phase preroll. See commit `80287898`.

~~**File:** `src/components/timeline/hooks/useVideoPreload.ts`~~

~~**Action:** Extract video preloading from lines 509-598.~~

```typescript
// DELETED — Video preloading is handled by VideoSyncManager.preloadUpcomingClips()
// which correctly skips shared video elements and uses smart two-phase preroll.
// The old useVideoPreload hook caused stutter at cut points by seeking shared
// HTMLVideoElements away from their current playback position.
```

Original code for reference (DO NOT RE-IMPLEMENT):
```typescript
const currentPosition = playheadState.isUsingInternalPosition
      ? playheadState.position
      : playheadPosition;
    const lookaheadPosition = currentPosition + LOOKAHEAD_TIME;

    // Helper to preload a video element - seeks and forces buffering
    const preloadVideo = (
      video: HTMLVideoElement,
      targetTime: number,
      _clipName: string
    ) => {
      const timeDiff = Math.abs(video.currentTime - targetTime);

      // Only preload if significantly different (avoid repeated preloading)
      if (timeDiff > 0.1) {
        video.currentTime = Math.max(0, targetTime);

        // Force buffer by briefly playing then pausing
        // This triggers the browser to actually fetch the video data
        const wasPlaying = !video.paused;
        if (!wasPlaying) {
          video
            .play()
            .then(() => {
              // Immediately pause after play starts buffering
              setTimeout(() => {
                if (!wasPlaying) video.pause();
              }, 50);
            })
            .catch(() => {
              // Ignore play errors (e.g., autoplay policy)
            });
        }
      }
    };

    // Find clips that will start playing soon (not currently playing, but will be soon)
    const upcomingClips = clips.filter((clip) => {
      // Clip starts after current position but within lookahead window
      const startsInLookahead =
        clip.startTime > currentPosition && clip.startTime <= lookaheadPosition;
      // Has a video element to preload
      const hasVideo = clip.source?.videoElement;
      return startsInLookahead && hasVideo;
    });

    // Pre-buffer upcoming regular clips
    for (const clip of upcomingClips) {
      if (clip.source?.videoElement) {
        preloadVideo(clip.source.videoElement, clip.inPoint, clip.name);
      }
    }

    // Also preload nested composition clips
    const upcomingNestedClips = clips.filter((clip) => {
      const startsInLookahead =
        clip.startTime > currentPosition && clip.startTime <= lookaheadPosition;
      const hasNestedClips =
        clip.isComposition && clip.nestedClips && clip.nestedClips.length > 0;
      return startsInLookahead && hasNestedClips;
    });

    for (const compClip of upcomingNestedClips) {
      if (!compClip.nestedClips) continue;

      // Find the nested video clip that would play at the start of this comp clip
      const compStartTime = compClip.inPoint; // Time within the composition

      for (const nestedClip of compClip.nestedClips) {
        if (!nestedClip.source?.videoElement) continue;

        // Check if this nested clip would be playing at comp start
        if (
          compStartTime >= nestedClip.startTime &&
          compStartTime < nestedClip.startTime + nestedClip.duration
        ) {
          const nestedLocalTime = compStartTime - nestedClip.startTime;
          const targetTime = nestedClip.reversed
            ? nestedClip.outPoint - nestedLocalTime
            : nestedLocalTime + nestedClip.inPoint;

          preloadVideo(nestedClip.source.videoElement, targetTime, nestedClip.name);
        }
      }
    }
  }, [isPlaying, isDraggingPlayhead, playheadPosition, clips]);
}
```

---

### Step 4: Create `hooks/useAutoFeatures.ts`

**File:** `src/components/timeline/hooks/useAutoFeatures.ts`

**Action:** Extract RAM preview and proxy auto-start from lines 375-481.

```typescript
// Auto-start features: RAM preview and proxy generation after idle

import { useEffect, useRef } from 'react';
import { useTimelineStore } from '../../../stores/timeline';
import { useMediaStore } from '../../../stores/mediaStore';
import {
  RAM_PREVIEW_IDLE_DELAY,
  PROXY_IDLE_DELAY,
} from '../constants';
import type { TimelineClip } from '../../../types';

interface UseAutoFeaturesProps {
  ramPreviewEnabled: boolean;
  proxyEnabled: boolean;
  isPlaying: boolean;
  isDraggingPlayhead: boolean;
  isRamPreviewing: boolean;
  currentlyGeneratingProxyId: string | null;
  inPoint: number | null;
  outPoint: number | null;
  ramPreviewRange: { start: number; end: number } | null;
  clips: TimelineClip[];
  startRamPreview: () => void;
  cancelRamPreview: () => void;
}

/**
 * Auto-start RAM Preview after idle (like After Effects)
 * Auto-generate proxies after idle
 */
export function useAutoFeatures({
  ramPreviewEnabled,
  proxyEnabled,
  isPlaying,
  isDraggingPlayhead,
  isRamPreviewing,
  currentlyGeneratingProxyId,
  inPoint,
  outPoint,
  ramPreviewRange,
  clips,
  startRamPreview,
  cancelRamPreview,
}: UseAutoFeaturesProps) {
  const ramIdleTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const proxyIdleTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Cancel RAM preview when user starts playing or scrubbing (keep cached frames)
  useEffect(() => {
    if ((isPlaying || isDraggingPlayhead) && isRamPreviewing) {
      cancelRamPreview();
    }
  }, [isPlaying, isDraggingPlayhead, isRamPreviewing, cancelRamPreview]);

  // Auto-start RAM Preview after idle
  useEffect(() => {
    if (ramIdleTimerRef.current) {
      clearTimeout(ramIdleTimerRef.current);
      ramIdleTimerRef.current = null;
    }

    if (
      !ramPreviewEnabled ||
      isPlaying ||
      isRamPreviewing ||
      isDraggingPlayhead ||
      clips.length === 0
    ) {
      return;
    }

    const renderStart = inPoint ?? 0;
    const renderEnd =
      outPoint ?? Math.max(...clips.map((c) => c.startTime + c.duration));

    if (renderEnd - renderStart < 0.1) {
      return;
    }

    if (
      ramPreviewRange &&
      ramPreviewRange.start <= renderStart &&
      ramPreviewRange.end >= renderEnd
    ) {
      return;
    }

    ramIdleTimerRef.current = setTimeout(() => {
      const state = useTimelineStore.getState();
      if (state.ramPreviewEnabled && !state.isPlaying && !state.isRamPreviewing) {
        startRamPreview();
      }
    }, RAM_PREVIEW_IDLE_DELAY);

    return () => {
      if (ramIdleTimerRef.current) {
        clearTimeout(ramIdleTimerRef.current);
        ramIdleTimerRef.current = null;
      }
    };
  }, [
    ramPreviewEnabled,
    isPlaying,
    isRamPreviewing,
    isDraggingPlayhead,
    inPoint,
    outPoint,
    ramPreviewRange,
    clips,
    startRamPreview,
  ]);

  // Auto-generate proxies after idle
  useEffect(() => {
    if (proxyIdleTimerRef.current) {
      clearTimeout(proxyIdleTimerRef.current);
      proxyIdleTimerRef.current = null;
    }

    if (!proxyEnabled || isPlaying || currentlyGeneratingProxyId || isDraggingPlayhead) {
      return;
    }

    proxyIdleTimerRef.current = setTimeout(() => {
      const mediaStore = useMediaStore.getState();
      if (mediaStore.proxyEnabled && !mediaStore.currentlyGeneratingProxyId) {
        const nextFile = mediaStore.getNextFileNeedingProxy();
        if (nextFile) {
          console.log('[Proxy] Auto-starting proxy generation for:', nextFile.name);
          mediaStore.generateProxy(nextFile.id);
        }
      }
    }, PROXY_IDLE_DELAY);

    return () => {
      if (proxyIdleTimerRef.current) {
        clearTimeout(proxyIdleTimerRef.current);
        proxyIdleTimerRef.current = null;
      }
    };
  }, [
    proxyEnabled,
    isPlaying,
    currentlyGeneratingProxyId,
    isDraggingPlayhead,
    clips,
  ]);
}
```

---

### Step 5: Create `hooks/useExternalDrop.ts`

**File:** `src/components/timeline/hooks/useExternalDrop.ts`

**Action:** Extract all external drag/drop handlers from lines 879-1335. This is the largest extraction.

```typescript
// External file drag & drop handling for timeline

import { useCallback, useRef } from 'react';
import { useMediaStore } from '../../../stores/mediaStore';
import { useTimelineStore } from '../../../stores/timeline';
import {
  isVideoFile,
  isAudioFile,
  isMediaFile,
  getVideoDurationQuick,
} from '../utils/fileTypeHelpers';
import type { ExternalDragState } from '../types';
import type { TimelineTrack, TimelineClip } from '../../../types';

interface UseExternalDropProps {
  timelineRef: React.RefObject<HTMLDivElement>;
  scrollX: number;
  tracks: TimelineTrack[];
  clips: TimelineClip[];
  pixelToTime: (pixel: number) => number;
  addTrack: (type: 'video' | 'audio') => string | undefined;
  addClip: (trackId: string, file: File, startTime: number, duration?: number, mediaFileId?: string) => void;
  addCompClip: (trackId: string, comp: any, startTime: number) => void;
}

interface UseExternalDropReturn {
  externalDrag: ExternalDragState | null;
  setExternalDrag: React.Dispatch<React.SetStateAction<ExternalDragState | null>>;
  dragCounterRef: React.MutableRefObject<number>;
  handleTrackDragEnter: (e: React.DragEvent, trackId: string) => void;
  handleTrackDragOver: (e: React.DragEvent, trackId: string) => void;
  handleTrackDragLeave: (e: React.DragEvent) => void;
  handleTrackDrop: (e: React.DragEvent, trackId: string) => Promise<void>;
  handleNewTrackDragOver: (e: React.DragEvent, trackType: 'video' | 'audio') => void;
  handleNewTrackDrop: (e: React.DragEvent, trackType: 'video' | 'audio') => Promise<void>;
}

export function useExternalDrop({
  timelineRef,
  scrollX,
  tracks,
  clips,
  pixelToTime,
  addTrack,
  addClip,
  addCompClip,
}: UseExternalDropProps): UseExternalDropReturn {
  const [externalDrag, setExternalDrag] = useState<ExternalDragState | null>(null);
  const dragCounterRef = useRef(0);
  const dragDurationCacheRef = useRef<{ url: string; duration: number } | null>(null);

  // Handle external file drag enter on track
  const handleTrackDragEnter = useCallback(
    (e: React.DragEvent, trackId: string) => {
      e.preventDefault();
      dragCounterRef.current++;

      const rect = e.currentTarget.getBoundingClientRect();
      const x = e.clientX - rect.left + scrollX;
      const startTime = pixelToTime(x);

      if (e.dataTransfer.types.includes('application/x-composition-id')) {
        setExternalDrag({ trackId, startTime, x: e.clientX, y: e.clientY, duration: 5, isVideo: true });
        return;
      }

      if (e.dataTransfer.types.includes('application/x-media-file-id')) {
        setExternalDrag({ trackId, startTime, x: e.clientX, y: e.clientY, duration: 5, isVideo: true });
        return;
      }

      if (e.dataTransfer.types.includes('Files')) {
        let dur: number | undefined;
        const items = e.dataTransfer.items;
        if (items && items.length > 0) {
          for (let i = 0; i < items.length; i++) {
            const item = items[i];
            if (item.kind === 'file') {
              const file = item.getAsFile();
              if (file && isVideoFile(file)) {
                const cacheKey = `${file.name}_${file.size}`;
                if (dragDurationCacheRef.current?.url === cacheKey) {
                  dur = dragDurationCacheRef.current.duration;
                } else {
                  getVideoDurationQuick(file).then((d) => {
                    if (d) {
                      dragDurationCacheRef.current = { url: cacheKey, duration: d };
                      setExternalDrag((prev) =>
                        prev ? { ...prev, duration: d } : null
                      );
                    }
                  });
                }
                break;
              }
            }
          }
        }

        setExternalDrag({ trackId, startTime, x: e.clientX, y: e.clientY, duration: dur });
      }
    },
    [scrollX, pixelToTime]
  );

  // Handle external file drag over track
  const handleTrackDragOver = useCallback(
    (e: React.DragEvent, trackId: string) => {
      e.preventDefault();
      e.dataTransfer.dropEffect = 'copy';

      const isCompDrag = e.dataTransfer.types.includes('application/x-composition-id');
      const isMediaPanelDrag = e.dataTransfer.types.includes('application/x-media-file-id');
      const isFileDrag = e.dataTransfer.types.includes('Files');

      if ((isCompDrag || isMediaPanelDrag || isFileDrag) && timelineRef.current) {
        const rect = timelineRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left + scrollX;
        const startTime = pixelToTime(x);

        const targetTrack = tracks.find((t) => t.id === trackId);
        const isVideoTrack = targetTrack?.type === 'video';

        const previewDuration =
          externalDrag?.duration ?? dragDurationCacheRef.current?.duration ?? 5;

        let audioTrackId: string | undefined;
        if (isVideoTrack) {
          const audioTracks = tracks.filter((t) => t.type === 'audio');
          const endTime = startTime + previewDuration;

          for (const aTrack of audioTracks) {
            const trackClips = clips.filter((c) => c.trackId === aTrack.id);
            const hasOverlap = trackClips.some((clip) => {
              const clipEnd = clip.startTime + clip.duration;
              return !(endTime <= clip.startTime || startTime >= clipEnd);
            });
            if (!hasOverlap) {
              audioTrackId = aTrack.id;
              break;
            }
          }
          if (!audioTrackId) {
            audioTrackId = '__new_audio_track__';
          }
        }

        setExternalDrag((prev) => ({
          trackId,
          startTime,
          x: e.clientX,
          y: e.clientY,
          audioTrackId,
          isVideo: isVideoTrack,
          duration: prev?.duration ?? dragDurationCacheRef.current?.duration,
        }));
      }
    },
    [scrollX, pixelToTime, tracks, clips, externalDrag, timelineRef]
  );

  // Handle external file drag leave
  const handleTrackDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    dragCounterRef.current--;

    if (dragCounterRef.current === 0) {
      setExternalDrag(null);
    }
  }, []);

  // Handle drag over "new track" drop zone
  const handleNewTrackDragOver = useCallback(
    (e: React.DragEvent, trackType: 'video' | 'audio') => {
      e.preventDefault();
      e.stopPropagation();
      e.dataTransfer.dropEffect = 'copy';

      if (timelineRef.current) {
        const rect = timelineRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left + scrollX;
        const startTime = pixelToTime(x);

        setExternalDrag((prev) => ({
          trackId: '__new_track__',
          startTime,
          x: e.clientX,
          y: e.clientY,
          duration: prev?.duration ?? dragDurationCacheRef.current?.duration ?? 5,
          newTrackType: trackType,
          isVideo: trackType === 'video',
          isAudio: trackType === 'audio',
        }));
      }
    },
    [scrollX, pixelToTime, timelineRef]
  );

  // Helper to extract file path from drag event
  const extractFilePath = (e: React.DragEvent): string | undefined => {
    // Try text/uri-list (Nautilus, Dolphin)
    const uriList = e.dataTransfer.getData('text/uri-list');
    if (uriList) {
      const uri = uriList.split('\n')[0]?.trim();
      if (uri?.startsWith('file://')) {
        return decodeURIComponent(uri.replace('file://', ''));
      }
    }

    // Try text/plain (some file managers)
    const plainText = e.dataTransfer.getData('text/plain');
    if (plainText?.startsWith('/') || plainText?.startsWith('file://')) {
      return plainText.startsWith('file://')
        ? decodeURIComponent(plainText.replace('file://', ''))
        : plainText;
    }

    // Try text/x-moz-url (Firefox)
    const mozUrl = e.dataTransfer.getData('text/x-moz-url');
    if (mozUrl?.startsWith('file://')) {
      return decodeURIComponent(mozUrl.split('\n')[0].replace('file://', ''));
    }

    return undefined;
  };

  // Handle drop on "new track" zone - creates new track and adds clip
  const handleNewTrackDrop = useCallback(
    async (e: React.DragEvent, trackType: 'video' | 'audio') => {
      e.preventDefault();
      e.stopPropagation();

      const cachedDuration =
        externalDrag?.duration ?? dragDurationCacheRef.current?.duration;

      dragCounterRef.current = 0;
      setExternalDrag(null);

      // Validate file type matches track type BEFORE creating track
      const mediaFileId = e.dataTransfer.getData('application/x-media-file-id');
      if (mediaFileId) {
        const mediaStore = useMediaStore.getState();
        const mediaFile = mediaStore.files.find((f) => f.id === mediaFileId);
        if (mediaFile?.file) {
          const fileIsAudio = isAudioFile(mediaFile.file);
          if (fileIsAudio && trackType === 'video') {
            console.log('[Timeline] Audio files can only be dropped on audio tracks');
            return;
          }
          if (!fileIsAudio && trackType === 'audio') {
            console.log('[Timeline] Video/image files can only be dropped on video tracks');
            return;
          }
        }
      }

      if (e.dataTransfer.files.length > 0) {
        const file = e.dataTransfer.files[0];
        const fileIsAudio = isAudioFile(file);
        if (fileIsAudio && trackType === 'video') {
          console.log('[Timeline] Audio files can only be dropped on audio tracks');
          return;
        }
        if (!fileIsAudio && trackType === 'audio') {
          console.log('[Timeline] Video/image files can only be dropped on video tracks');
          return;
        }
      }

      // Create a new track
      const newTrackId = addTrack(trackType);
      if (!newTrackId) return;

      const rect = timelineRef.current?.getBoundingClientRect();
      if (!rect) return;

      const x = e.clientX - rect.left + scrollX;
      const startTime = Math.max(0, pixelToTime(x));
      const filePath = extractFilePath(e);

      // Handle composition drag
      const compositionId = e.dataTransfer.getData('application/x-composition-id');
      if (compositionId) {
        const mediaStore = useMediaStore.getState();
        const comp = mediaStore.compositions.find((c) => c.id === compositionId);
        if (comp) {
          addCompClip(newTrackId, comp, startTime);
          return;
        }
      }

      // Handle media panel drag
      if (mediaFileId) {
        const mediaStore = useMediaStore.getState();
        const mediaFile = mediaStore.files.find((f) => f.id === mediaFileId);
        if (mediaFile?.file) {
          addClip(newTrackId, mediaFile.file, startTime, mediaFile.duration, mediaFileId);
          return;
        }
      }

      // Handle external file drop
      const items = e.dataTransfer.items;
      if (items && items.length > 0) {
        const item = items[0];
        if (item.kind === 'file') {
          const mediaStore = useMediaStore.getState();

          // Try to get file handle (File System Access API)
          if ('getAsFileSystemHandle' in item) {
            try {
              const handle = await (item as any).getAsFileSystemHandle();
              if (handle && handle.kind === 'file') {
                const file = await handle.getFile();
                if (filePath) (file as any).path = filePath;
                if (isMediaFile(file)) {
                  const imported = await mediaStore.importFilesWithHandles([
                    { file, handle, absolutePath: filePath },
                  ]);
                  if (imported.length > 0) {
                    addClip(newTrackId, file, startTime, cachedDuration, imported[0].id);
                  }
                  return;
                }
              }
            } catch (err) {
              console.warn('[Timeline] Could not get file handle, falling back:', err);
            }
          }

          // Fallback to regular file (no handle)
          const file = item.getAsFile();
          if (file && filePath) (file as any).path = filePath;
          if (file && isMediaFile(file)) {
            const importedFile = await mediaStore.importFile(file);
            addClip(newTrackId, file, startTime, cachedDuration, importedFile?.id);
          }
        }
      }
    },
    [scrollX, pixelToTime, addTrack, addCompClip, addClip, externalDrag, timelineRef]
  );

  // Handle external file drop on track
  const handleTrackDrop = useCallback(
    async (e: React.DragEvent, trackId: string) => {
      e.preventDefault();

      const cachedDuration =
        externalDrag?.duration ?? dragDurationCacheRef.current?.duration;

      dragCounterRef.current = 0;
      setExternalDrag(null);

      // Get track type for validation
      const targetTrack = tracks.find((t) => t.id === trackId);
      const isVideoTrack = targetTrack?.type === 'video';
      const isAudioTrack = targetTrack?.type === 'audio';

      const compositionId = e.dataTransfer.getData('application/x-composition-id');
      if (compositionId) {
        const mediaStore = useMediaStore.getState();
        const comp = mediaStore.compositions.find((c) => c.id === compositionId);
        if (comp) {
          const rect = e.currentTarget.getBoundingClientRect();
          const x = e.clientX - rect.left + scrollX;
          const startTime = pixelToTime(x);
          addCompClip(trackId, comp, Math.max(0, startTime));
          return;
        }
      }

      const mediaFileId = e.dataTransfer.getData('application/x-media-file-id');
      if (mediaFileId) {
        const mediaStore = useMediaStore.getState();
        const mediaFile = mediaStore.files.find((f) => f.id === mediaFileId);
        if (mediaFile?.file) {
          const fileIsAudio = isAudioFile(mediaFile.file);
          if (fileIsAudio && isVideoTrack) {
            console.log('[Timeline] Audio files can only be dropped on audio tracks');
            return;
          }
          if (!fileIsAudio && isAudioTrack) {
            console.log('[Timeline] Video/image files can only be dropped on video tracks');
            return;
          }

          const rect = e.currentTarget.getBoundingClientRect();
          const x = e.clientX - rect.left + scrollX;
          const startTime = pixelToTime(x);
          addClip(trackId, mediaFile.file, Math.max(0, startTime), mediaFile.duration, mediaFileId);
          return;
        }
      }

      // Handle external file drop
      const items = e.dataTransfer.items;
      const filePath = extractFilePath(e);

      if (items && items.length > 0) {
        const item = items[0];
        if (item.kind === 'file') {
          const rect = e.currentTarget.getBoundingClientRect();
          const x = e.clientX - rect.left + scrollX;
          const startTime = Math.max(0, pixelToTime(x));
          const mediaStore = useMediaStore.getState();

          // Try to get file handle (File System Access API)
          if ('getAsFileSystemHandle' in item) {
            try {
              const handle = await (item as any).getAsFileSystemHandle();
              if (handle && handle.kind === 'file') {
                const file = await handle.getFile();
                if (filePath) (file as any).path = filePath;
                if (isMediaFile(file)) {
                  const fileIsAudio = isAudioFile(file);
                  if (fileIsAudio && isVideoTrack) {
                    console.log('[Timeline] Audio files can only be dropped on audio tracks');
                    return;
                  }
                  if (!fileIsAudio && isAudioTrack) {
                    console.log('[Timeline] Video/image files can only be dropped on video tracks');
                    return;
                  }

                  const imported = await mediaStore.importFilesWithHandles([
                    { file, handle, absolutePath: filePath },
                  ]);
                  if (imported.length > 0) {
                    addClip(trackId, file, startTime, cachedDuration, imported[0].id);
                  }
                  return;
                }
              }
            } catch (err) {
              console.warn('[Timeline] Could not get file handle, falling back:', err);
            }
          }

          // Fallback to regular file (no handle)
          const file = item.getAsFile();
          if (file && filePath) (file as any).path = filePath;
          if (file && isMediaFile(file)) {
            const fileIsAudio = isAudioFile(file);
            if (fileIsAudio && isVideoTrack) {
              console.log('[Timeline] Audio files can only be dropped on audio tracks');
              return;
            }
            if (!fileIsAudio && isAudioTrack) {
              console.log('[Timeline] Video/image files can only be dropped on video tracks');
              return;
            }

            const importedFile = await mediaStore.importFile(file);
            addClip(trackId, file, startTime, cachedDuration, importedFile?.id);
          }
        }
      }
    },
    [scrollX, pixelToTime, addCompClip, addClip, externalDrag, tracks, timelineRef]
  );

  return {
    externalDrag,
    setExternalDrag,
    dragCounterRef,
    handleTrackDragEnter,
    handleTrackDragOver,
    handleTrackDragLeave,
    handleTrackDrop,
    handleNewTrackDragOver,
    handleNewTrackDrop,
  };
}
```

**Note:** Add missing import at top:
```typescript
import { useState } from 'react';
```

---

### Step 6: Create `components/NewTrackDropZone.tsx`

**File:** `src/components/timeline/components/NewTrackDropZone.tsx`

**Action:** Create reusable drop zone component to eliminate duplication.

```typescript
// Drop zone for creating new tracks

import React from 'react';
import type { ExternalDragState } from '../types';

interface NewTrackDropZoneProps {
  type: 'video' | 'audio';
  externalDrag: ExternalDragState | null;
  timeToPixel: (time: number) => number;
  onDragOver: (e: React.DragEvent) => void;
  onDragEnter: (e: React.DragEvent) => void;
  onDragLeave: (e: React.DragEvent) => void;
  onDrop: (e: React.DragEvent) => void;
}

export function NewTrackDropZone({
  type,
  externalDrag,
  timeToPixel,
  onDragOver,
  onDragEnter,
  onDragLeave,
  onDrop,
}: NewTrackDropZoneProps) {
  if (!externalDrag) return null;

  const isActive = externalDrag.newTrackType === type;

  return (
    <div
      className={`new-track-drop-zone ${type} ${isActive ? 'active' : ''}`}
      onDragOver={onDragOver}
      onDragEnter={onDragEnter}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
    >
      <span className="drop-zone-label">
        + Drop to create new {type === 'video' ? 'Video' : 'Audio'} Track
      </span>
      {isActive && (
        <div
          className={`timeline-clip-preview ${type}`}
          style={{
            left: timeToPixel(externalDrag.startTime),
            width: timeToPixel(externalDrag.duration ?? 5),
          }}
        >
          <div className="clip-content">
            <span className="clip-name">New clip</span>
          </div>
        </div>
      )}
    </div>
  );
}
```

---

### Step 7: Create `components/TimelineOverlays.tsx`

**File:** `src/components/timeline/components/TimelineOverlays.tsx`

**Action:** Extract all overlay elements (markers, work area, progress, etc.)

```typescript
// Timeline overlay elements (markers, work area, cache indicators, etc.)

import React from 'react';
import type { ClipDragState } from '../types';

interface TimelineOverlaysProps {
  // Time conversion
  timeToPixel: (time: number) => number;
  formatTime: (seconds: number) => string;

  // In/Out points
  inPoint: number | null;
  outPoint: number | null;
  duration: number;
  markerDrag: { type: 'in' | 'out' } | null;
  onMarkerMouseDown: (e: React.MouseEvent, type: 'in' | 'out') => void;

  // Clip drag
  clipDrag: ClipDragState | null;

  // RAM preview
  isRamPreviewing: boolean;
  ramPreviewProgress: number | null;
  playheadPosition: number;

  // Export
  isExporting: boolean;
  exportProgress: number | null;
  exportRange: { start: number; end: number } | null;

  // Cache
  getCachedRanges: () => { start: number; end: number }[];
}

export function TimelineOverlays({
  timeToPixel,
  formatTime,
  inPoint,
  outPoint,
  duration,
  markerDrag,
  onMarkerMouseDown,
  clipDrag,
  isRamPreviewing,
  ramPreviewProgress,
  playheadPosition,
  isExporting,
  exportProgress,
  exportRange,
  getCachedRanges,
}: TimelineOverlaysProps) {
  return (
    <>
      {/* Snap line */}
      {clipDrag?.isSnapping && clipDrag.snappedTime !== null && (
        <div className="snap-line" style={{ left: timeToPixel(clipDrag.snappedTime) }} />
      )}

      {/* Work area overlays */}
      {(inPoint !== null || outPoint !== null) && (
        <>
          {inPoint !== null && inPoint > 0 && (
            <div
              className="work-area-overlay before"
              style={{
                left: 0,
                width: timeToPixel(inPoint),
              }}
            />
          )}
          {outPoint !== null && (
            <div
              className="work-area-overlay after"
              style={{
                left: timeToPixel(outPoint),
                width: timeToPixel(duration - outPoint),
              }}
            />
          )}
        </>
      )}

      {/* RAM preview progress */}
      {isRamPreviewing && ramPreviewProgress !== null && (
        <div
          className="ram-preview-progress-text"
          style={{
            left: timeToPixel(playheadPosition) + 10,
          }}
        >
          {Math.round(ramPreviewProgress)}%
        </div>
      )}

      {/* Export progress overlay */}
      {isExporting && exportRange && (
        <>
          <div
            className="timeline-export-overlay"
            style={{
              left: timeToPixel(exportRange.start),
              width: timeToPixel(
                (exportRange.end - exportRange.start) * ((exportProgress ?? 0) / 100)
              ),
            }}
          />
          <div
            className="timeline-export-text"
            style={{
              left:
                timeToPixel(
                  exportRange.start +
                    (exportRange.end - exportRange.start) * ((exportProgress ?? 0) / 100)
                ) - 10,
              transform: 'translateX(-100%)',
            }}
          >
            {Math.round(exportProgress ?? 0)}%
          </div>
        </>
      )}

      {/* Cache indicators */}
      {getCachedRanges().map((range, i) => (
        <div
          key={i}
          className="playback-cache-indicator"
          style={{
            left: timeToPixel(range.start),
            width: Math.max(2, timeToPixel(range.end - range.start)),
          }}
          title={`Cached: ${formatTime(range.start)} - ${formatTime(range.end)}`}
        />
      ))}

      {/* In marker */}
      {inPoint !== null && (
        <div
          className={`in-out-marker in-marker ${markerDrag?.type === 'in' ? 'dragging' : ''}`}
          style={{ left: timeToPixel(inPoint) }}
          title={`In: ${formatTime(inPoint)} (drag to move)`}
        >
          <div
            className="marker-flag"
            onMouseDown={(e) => onMarkerMouseDown(e, 'in')}
          >
            I
          </div>
          <div className="marker-line" />
        </div>
      )}

      {/* Out marker */}
      {outPoint !== null && (
        <div
          className={`in-out-marker out-marker ${markerDrag?.type === 'out' ? 'dragging' : ''}`}
          style={{ left: timeToPixel(outPoint) }}
          title={`Out: ${formatTime(outPoint)} (drag to move)`}
        >
          <div
            className="marker-flag"
            onMouseDown={(e) => onMarkerMouseDown(e, 'out')}
          >
            O
          </div>
          <div className="marker-line" />
        </div>
      )}
    </>
  );
}
```

---

### Step 8: Create `components/PickWhipOverlay.tsx`

**File:** `src/components/timeline/components/PickWhipOverlay.tsx`

**Action:** Create reusable pick whip drag overlay.

```typescript
// Pick whip drag overlay for layer parenting

import React from 'react';
import { PhysicsCable } from '../PhysicsCable';
import type { PickWhipDragState } from '../types';

interface PickWhipOverlayProps {
  dragState: PickWhipDragState | null;
}

export function PickWhipOverlay({ dragState }: PickWhipOverlayProps) {
  if (!dragState) return null;

  return (
    <svg
      className="pick-whip-drag-overlay"
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100vw',
        height: '100vh',
        pointerEvents: 'none',
        zIndex: 9999,
      }}
    >
      <PhysicsCable
        startX={dragState.startX}
        startY={dragState.startY}
        endX={dragState.currentX}
        endY={dragState.currentY}
        isPreview={true}
      />
    </svg>
  );
}
```

---

### Step 9: Update `Timeline.tsx`

**File:** `src/components/timeline/Timeline.tsx`

**Action:** Update imports and replace extracted code with hook/component calls.

**Changes to make:**

1. **Add new imports:**
```typescript
// Add at top of imports
import { usePlaybackLoop } from './hooks/usePlaybackLoop';
// useVideoPreload REMOVED — VideoSyncManager.preloadUpcomingClips handles preloading
import { useAutoFeatures } from './hooks/useAutoFeatures';
import { useExternalDrop } from './hooks/useExternalDrop';
import { NewTrackDropZone } from './components/NewTrackDropZone';
import { TimelineOverlays } from './components/TimelineOverlays';
import { PickWhipOverlay } from './components/PickWhipOverlay';
```

2. **Remove lines 375-481** (RAM preview and proxy auto-start effects)
   Replace with:
```typescript
// Auto-start RAM preview and proxy generation
useAutoFeatures({
  ramPreviewEnabled,
  proxyEnabled,
  isPlaying,
  isDraggingPlayhead,
  isRamPreviewing,
  currentlyGeneratingProxyId,
  inPoint,
  outPoint,
  ramPreviewRange,
  clips,
  startRamPreview,
  cancelRamPreview,
});
```

3. **~~Remove lines 509-598~~ (video preloading)** — ALREADY REMOVED
   useVideoPreload was deleted (caused stutter). VideoSyncManager handles preloading.
   Replace with comment:
```typescript
// Video preloading is handled by VideoSyncManager.preloadUpcomingClips()
// in the render loop — it correctly skips shared video elements (split clips)
// and uses smart two-phase preroll. No React hook needed.
```

4. **Remove lines 600-708** (playback loop)
   Replace with:
```typescript
// Audio master clock playback loop
usePlaybackLoop({ isPlaying });
```

5. **Remove lines 815-877** (file type helpers)
   They're now in `utils/fileTypeHelpers.ts`

6. **Remove lines 879-1335** (external drag handlers)
   Replace with:
```typescript
// External file drag & drop
const {
  externalDrag,
  setExternalDrag,
  dragCounterRef,
  handleTrackDragEnter,
  handleTrackDragOver,
  handleTrackDragLeave,
  handleTrackDrop,
  handleNewTrackDragOver,
  handleNewTrackDrop,
} = useExternalDrop({
  timelineRef,
  scrollX,
  tracks,
  clips,
  pixelToTime,
  addTrack,
  addClip,
  addCompClip,
});
```

7. **Remove state declarations that moved to hooks:**
```typescript
// REMOVE these lines (now in useExternalDrop):
// const [externalDrag, setExternalDrag] = useState<ExternalDragState | null>(null);
// const dragCounterRef = useRef(0);
// const dragDurationCacheRef = useRef<{ url: string; duration: number } | null>(null);
```

8. **Replace JSX drop zones** (lines 1687-1712, 1784-1810) with:
```tsx
<NewTrackDropZone
  type="video"
  externalDrag={externalDrag}
  timeToPixel={timeToPixel}
  onDragOver={(e) => handleNewTrackDragOver(e, 'video')}
  onDragEnter={(e) => {
    e.preventDefault();
    dragCounterRef.current++;
  }}
  onDragLeave={handleTrackDragLeave}
  onDrop={(e) => handleNewTrackDrop(e, 'video')}
/>

{/* ... tracks.map ... */}

<NewTrackDropZone
  type="audio"
  externalDrag={externalDrag}
  timeToPixel={timeToPixel}
  onDragOver={(e) => handleNewTrackDragOver(e, 'audio')}
  onDragEnter={(e) => {
    e.preventDefault();
    dragCounterRef.current++;
  }}
  onDragLeave={handleTrackDragLeave}
  onDrop={(e) => handleNewTrackDrop(e, 'audio')}
/>
```

9. **Replace overlay JSX** (lines 1812-1916) with:
```tsx
<TimelineOverlays
  timeToPixel={timeToPixel}
  formatTime={formatTime}
  inPoint={inPoint}
  outPoint={outPoint}
  duration={duration}
  markerDrag={markerDrag}
  onMarkerMouseDown={handleMarkerMouseDown}
  clipDrag={clipDrag}
  isRamPreviewing={isRamPreviewing}
  ramPreviewProgress={ramPreviewProgress}
  playheadPosition={playheadPosition}
  isExporting={isExporting}
  exportProgress={exportProgress}
  exportRange={exportRange}
  getCachedRanges={getCachedRanges}
/>
```

10. **Replace pick whip overlays** (lines 2039-2084) with:
```tsx
<PickWhipOverlay dragState={pickWhipDrag} />
<PickWhipOverlay dragState={trackPickWhipDrag} />
```

---

## Post-Refactor Validation

### Step 10: Verify Compilation

```bash
npx tsc --noEmit
```

### Step 11: Manual Testing

1. `npm run dev`
2. Verify:
   - [ ] Timeline renders correctly
   - [ ] Playback works (play/pause/stop)
   - [ ] Drag files onto timeline (existing tracks)
   - [ ] Drag files onto new track drop zones
   - [ ] In/Out markers drag correctly
   - [ ] RAM preview auto-starts after idle
   - [ ] Proxy auto-generates after idle
   - [ ] Pick whip layer parenting works
   - [x] Video preloading (handled by VideoSyncManager — useVideoPreload removed)

### Step 12: Commit

```bash
git add -A
git commit -m "refactor: Split Timeline.tsx into focused modules

- Extract useExternalDrop hook (drag & drop handling)
- Extract usePlaybackLoop hook (audio master clock)
- REMOVED useVideoPreload hook (caused stutter at cuts, VideoSyncManager handles it)
- Extract useAutoFeatures hook (RAM preview & proxy auto-start)
- Extract fileTypeHelpers utils (remove duplicate code)
- Create NewTrackDropZone component (remove JSX duplication)
- Create TimelineOverlays component (group overlay elements)
- Create PickWhipOverlay component (remove duplication)

Timeline.tsx: 2109 LOC → ~900 LOC
No functional changes."

git push origin staging
```

---

## File Size Summary

| File | LOC | Purpose |
|------|-----|---------|
| `Timeline.tsx` | ~900 | Main component |
| `hooks/useExternalDrop.ts` | ~400 | Drag & drop |
| `hooks/usePlaybackLoop.ts` | ~110 | Audio master clock |
| ~~`hooks/useVideoPreload.ts`~~ | ~~90~~ | ~~REMOVED: caused stutter~~ |
| `hooks/useAutoFeatures.ts` | ~80 | Auto-start features |
| `utils/fileTypeHelpers.ts` | ~60 | File type detection |
| `components/NewTrackDropZone.tsx` | ~50 | Drop zone UI |
| `components/TimelineOverlays.tsx` | ~150 | Overlay elements |
| `components/PickWhipOverlay.tsx` | ~30 | Pick whip UI |
| **Total** | **~1870** | (was 2109) |

**Net reduction:** ~240 LOC of duplicate code removed

---

## Duplicate Code Removed

| Duplicate | Occurrences | Lines Saved |
|-----------|-------------|-------------|
| `isAudioFile` function | 2 → 1 | ~10 |
| `audioExtensions` array | 2 → 1 | ~5 |
| `NewTrackDropZone` JSX | 2 → 1 | ~25 |
| `PickWhipOverlay` JSX | 2 → 1 | ~20 |
| File type helpers | 3 funcs inline → utils | ~60 |
| **Total** | | **~120 LOC** |

---

## Rollback

If issues occur:

```bash
git checkout refactor/timeline-backup
git branch -D refactor/timeline-split
```

---

## Notes for AI Agent

1. **Create utils/fileTypeHelpers.ts first** - other files depend on it
2. **Test after each hook extraction** - run `npm run dev` and verify functionality
3. **The JSX changes are the trickiest** - match prop names exactly
4. **useExternalDrop needs useState import** - it's missing in the template above
5. **Keep existing hooks unchanged** - only add new ones
6. **Verify drag & drop works** - this is the highest-risk extraction
7. **Commit after validation** - don't commit broken state
