// AI Tools Utilities

import { useTimelineStore } from '../../stores/timeline';
import { engine } from '../../engine/WebGPUEngine';
import type { TimelineClip, TimelineTrack } from '../../stores/timeline/types';
import type { ToolResult } from './types';

// Helper to capture frames at multiple times and combine into a grid image
export async function captureFrameGrid(
  times: number[],
  columns: number,
  timelineStore: ReturnType<typeof useTimelineStore.getState>
): Promise<ToolResult> {
  const frameWidth = 320; // Thumbnail size
  const frameHeight = 180;
  const rows = Math.ceil(times.length / columns);

  // Create canvas for the grid
  const gridCanvas = document.createElement('canvas');
  gridCanvas.width = frameWidth * columns;
  gridCanvas.height = frameHeight * rows;
  const gridCtx = gridCanvas.getContext('2d');

  if (!gridCtx) {
    return { success: false, error: 'Failed to create canvas context' };
  }

  // Fill with dark background
  gridCtx.fillStyle = '#1a1a1a';
  gridCtx.fillRect(0, 0, gridCanvas.width, gridCanvas.height);

  const { width: outputWidth, height: outputHeight } = engine.getOutputDimensions();
  const originalPosition = timelineStore.playheadPosition;

  // Capture each frame
  for (let i = 0; i < times.length; i++) {
    const time = times[i];
    const col = i % columns;
    const row = Math.floor(i / columns);

    // Move playhead and wait for render
    timelineStore.setPlayheadPosition(Math.max(0, time));
    await new Promise(resolve => setTimeout(resolve, 50)); // Wait for render

    // Capture frame from engine
    const pixels = await engine.readPixels();
    if (pixels) {
      // Create temp canvas for the frame
      const frameCanvas = document.createElement('canvas');
      frameCanvas.width = outputWidth;
      frameCanvas.height = outputHeight;
      const frameCtx = frameCanvas.getContext('2d');

      if (frameCtx) {
        const imageData = new ImageData(new Uint8ClampedArray(pixels), outputWidth, outputHeight);
        frameCtx.putImageData(imageData, 0, 0);

        // Draw scaled frame onto grid
        gridCtx.drawImage(
          frameCanvas,
          col * frameWidth,
          row * frameHeight,
          frameWidth,
          frameHeight
        );
      }
    }

    // Draw time label
    gridCtx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    gridCtx.fillRect(col * frameWidth, row * frameHeight + frameHeight - 20, frameWidth, 20);
    gridCtx.fillStyle = '#ffffff';
    gridCtx.font = '12px monospace';
    gridCtx.fillText(
      `${time.toFixed(2)}s`,
      col * frameWidth + 5,
      row * frameHeight + frameHeight - 6
    );

    // Draw separator line between "before" and "after" rows if this is a cut preview
    if (i === columns - 1 && rows === 2) {
      gridCtx.strokeStyle = '#ff4444';
      gridCtx.lineWidth = 2;
      gridCtx.beginPath();
      gridCtx.moveTo(0, frameHeight);
      gridCtx.lineTo(gridCanvas.width, frameHeight);
      gridCtx.stroke();
    }
  }

  // Restore original playhead position
  timelineStore.setPlayheadPosition(originalPosition);

  // Convert to PNG
  const dataUrl = gridCanvas.toDataURL('image/png');

  return {
    success: true,
    data: {
      width: gridCanvas.width,
      height: gridCanvas.height,
      frameCount: times.length,
      gridSize: `${columns}x${rows}`,
      dataUrl,
    },
  };
}

// Helper to format clip info for AI
export function formatClipInfo(clip: TimelineClip, track: TimelineTrack | undefined) {
  return {
    id: clip.id,
    name: clip.name,
    trackId: clip.trackId,
    trackName: track?.name || 'Unknown',
    trackType: track?.type || 'unknown',
    startTime: clip.startTime,
    endTime: clip.startTime + clip.duration,
    duration: clip.duration,
    inPoint: clip.inPoint,
    outPoint: clip.outPoint,
    sourceType: clip.source?.type,
    hasAnalysis: clip.analysisStatus === 'ready',
    hasTranscript: !!clip.transcript?.length,
    // Transform info
    transform: clip.transform,
    // Effects count
    effectsCount: clip.effects?.length || 0,
  };
}

// Helper to format track info for AI
export function formatTrackInfo(track: TimelineTrack, clips: TimelineClip[]) {
  const trackClips = clips.filter(c => c.trackId === track.id);
  return {
    id: track.id,
    name: track.name,
    type: track.type,
    visible: track.visible,
    muted: track.muted,
    solo: track.solo,
    clipCount: trackClips.length,
    clips: trackClips.map(c => ({
      id: c.id,
      name: c.name,
      startTime: c.startTime,
      endTime: c.startTime + c.duration,
      duration: c.duration,
      hasAnalysis: c.analysisStatus === 'ready',
      hasTranscript: !!c.transcript?.length,
    })),
  };
}

// Helper to get a quick summary for AI context
export function getQuickTimelineSummary(): string {
  const { tracks, clips, playheadPosition, duration, selectedClipIds } = useTimelineStore.getState();

  const videoTracks = tracks.filter(t => t.type === 'video');
  const audioTracks = tracks.filter(t => t.type === 'audio');
  const videoClips = clips.filter(c => videoTracks.some(t => t.id === c.trackId));
  const audioClips = clips.filter(c => audioTracks.some(t => t.id === c.trackId));

  // Selected clip info
  const selectedCount = selectedClipIds.size;
  let selectedInfo = '';
  if (selectedCount > 0) {
    const selectedClip = clips.find(c => selectedClipIds.has(c.id));
    if (selectedClip) {
      const track = tracks.find(t => t.id === selectedClip.trackId);
      selectedInfo = ` Selected: "${selectedClip.name}" on ${track?.name || 'unknown track'}.`;
      if (selectedCount > 1) {
        selectedInfo = ` ${selectedCount} clips selected, first: "${selectedClip.name}" on ${track?.name || 'unknown track'}.`;
      }
    }
  }

  return `Timeline: ${videoTracks.length} video tracks (${videoClips.length} clips), ${audioTracks.length} audio tracks (${audioClips.length} clips). Playhead at ${playheadPosition.toFixed(2)}s, duration ${duration.toFixed(2)}s.${selectedInfo}`;
}
