// Analysis & Transcript Tool Handlers

import { useTimelineStore } from '../../../stores/timeline';
import type { ToolResult } from '../types';

type TimelineStore = ReturnType<typeof useTimelineStore.getState>;

export async function handleGetClipAnalysis(
  args: Record<string, unknown>,
  timelineStore: TimelineStore
): Promise<ToolResult> {
  const clipId = args.clipId as string;
  const clip = timelineStore.clips.find(c => c.id === clipId);
  if (!clip) {
    return { success: false, error: `Clip not found: ${clipId}` };
  }

  if (clip.analysisStatus !== 'ready' || !clip.analysis) {
    return {
      success: true,
      data: {
        hasAnalysis: false,
        status: clip.analysisStatus,
        message: clip.analysisStatus === 'analyzing'
          ? 'Analysis in progress'
          : 'No analysis data. Run analysis on this clip first.',
      },
    };
  }

  // Summarize analysis data
  const frames = clip.analysis.frames;
  const avgMotion = frames.reduce((sum, f) => sum + f.motion, 0) / frames.length;
  const avgBrightness = frames.reduce((sum, f) => sum + f.brightness, 0) / frames.length;
  const avgFocus = frames.reduce((sum, f) => sum + (f.focus || 0), 0) / frames.length;
  const totalFaces = frames.reduce((sum, f) => sum + (f.faceCount || 0), 0);

  return {
    success: true,
    data: {
      hasAnalysis: true,
      frameCount: frames.length,
      sampleInterval: clip.analysis.sampleInterval,
      summary: {
        averageMotion: avgMotion,
        averageBrightness: avgBrightness,
        averageFocus: avgFocus,
        maxMotion: Math.max(...frames.map(f => f.motion)),
        minMotion: Math.min(...frames.map(f => f.motion)),
        maxFocus: Math.max(...frames.map(f => f.focus || 0)),
        minFocus: Math.min(...frames.map(f => f.focus || 0)),
        totalFacesDetected: totalFaces,
      },
      // Include detailed frame data for specific queries
      frames: frames.map(f => ({
        time: f.timestamp,
        motion: f.motion,
        brightness: f.brightness,
        focus: f.focus || 0,
        faces: f.faceCount || 0,
      })),
    },
  };
}

export async function handleGetClipTranscript(
  args: Record<string, unknown>,
  timelineStore: TimelineStore
): Promise<ToolResult> {
  const clipId = args.clipId as string;
  const clip = timelineStore.clips.find(c => c.id === clipId);
  if (!clip) {
    return { success: false, error: `Clip not found: ${clipId}` };
  }

  if (!clip.transcript?.length) {
    return {
      success: true,
      data: {
        hasTranscript: false,
        message: 'No transcript available. Generate a transcript for this clip first.',
      },
    };
  }

  return {
    success: true,
    data: {
      hasTranscript: true,
      segmentCount: clip.transcript.length,
      segments: clip.transcript.map(word => ({
        start: word.start,
        end: word.end,
        text: word.text,
      })),
      // Full text for easy reading
      fullText: clip.transcript.map(w => w.text).join(' '),
    },
  };
}

export async function handleFindSilentSections(
  args: Record<string, unknown>,
  timelineStore: TimelineStore
): Promise<ToolResult> {
  const clipId = args.clipId as string;
  const minDuration = (args.minDuration as number) || 0.5;

  const clip = timelineStore.clips.find(c => c.id === clipId);
  if (!clip) {
    return { success: false, error: `Clip not found: ${clipId}` };
  }

  if (!clip.transcript?.length) {
    return {
      success: false,
      error: 'No transcript available to analyze for silence.',
    };
  }

  // Only consider the visible range of the clip
  const sourceStart = clip.inPoint;
  const sourceEnd = clip.outPoint;

  // Filter segments to those within the visible range
  const allSegments = clip.transcript;
  const segments = allSegments.filter(seg => seg.end > sourceStart && seg.start < sourceEnd);

  const silentSections: Array<{ sourceStart: number; sourceEnd: number; duration: number }> = [];

  // Check for silence at the beginning (from inPoint to first segment)
  const firstSegStart = segments.length > 0 ? Math.max(segments[0].start, sourceStart) : sourceEnd;
  if (firstSegStart - sourceStart >= minDuration) {
    silentSections.push({
      sourceStart: sourceStart,
      sourceEnd: firstSegStart,
      duration: firstSegStart - sourceStart,
    });
  }

  // Check gaps between segments
  for (let i = 0; i < segments.length - 1; i++) {
    const gapStart = Math.max(segments[i].end, sourceStart);
    const gapEnd = Math.min(segments[i + 1].start, sourceEnd);
    const gapDuration = gapEnd - gapStart;

    if (gapDuration >= minDuration) {
      silentSections.push({
        sourceStart: gapStart,
        sourceEnd: gapEnd,
        duration: gapDuration,
      });
    }
  }

  // Check for silence at the end (from last segment to outPoint)
  if (segments.length > 0) {
    const lastSegEnd = Math.min(segments[segments.length - 1].end, sourceEnd);
    if (sourceEnd - lastSegEnd >= minDuration) {
      silentSections.push({
        sourceStart: lastSegEnd,
        sourceEnd: sourceEnd,
        duration: sourceEnd - lastSegEnd,
      });
    }
  }

  // Convert source time to timeline time
  // Source time t maps to timeline time: clip.startTime + (t - clip.inPoint)
  const timelineSilentSections = silentSections.map(s => ({
    sourceStart: s.sourceStart,
    sourceEnd: s.sourceEnd,
    duration: s.duration,
    timelineStart: clip.startTime + (s.sourceStart - clip.inPoint),
    timelineEnd: clip.startTime + (s.sourceEnd - clip.inPoint),
  }));

  return {
    success: true,
    data: {
      clipId,
      minDuration,
      clipTimelineRange: { start: clip.startTime, end: clip.startTime + clip.duration },
      silentSections: timelineSilentSections,
      totalSilentTime: silentSections.reduce((sum, s) => sum + s.duration, 0),
      count: silentSections.length,
    },
  };
}

export async function handleFindLowQualitySections(
  args: Record<string, unknown>,
  timelineStore: TimelineStore
): Promise<ToolResult> {
  const clipId = args.clipId as string;
  const metric = (args.metric as string) || 'focus';
  const threshold = (args.threshold as number) ?? 0.7;
  const minDuration = (args.minDuration as number) || 0.5;

  const clip = timelineStore.clips.find(c => c.id === clipId);
  if (!clip) {
    return { success: false, error: `Clip not found: ${clipId}` };
  }

  if (clip.analysisStatus !== 'ready' || !clip.analysis?.frames?.length) {
    return {
      success: false,
      error: 'No analysis data available. Run analysis on this clip first.',
    };
  }

  // Only consider frames within the clip's visible range (inPoint to outPoint)
  const sourceStart = clip.inPoint;
  const sourceEnd = clip.outPoint;
  const allFrames = clip.analysis.frames;
  const frames = allFrames.filter(f => f.timestamp >= sourceStart && f.timestamp <= sourceEnd);

  if (frames.length === 0) {
    return {
      success: true,
      data: {
        clipId,
        metric,
        threshold,
        minDuration,
        clipTimelineRange: { start: clip.startTime, end: clip.startTime + clip.duration },
        sections: [],
        totalLowQualityTime: 0,
        count: 0,
        note: 'No analysis frames within the visible clip range.',
      },
    };
  }

  const lowQualitySections: Array<{ start: number; end: number; duration: number; avgValue: number }> = [];

  // Find contiguous sections below threshold
  let sectionStart: number | null = null;
  let sectionValues: number[] = [];

  for (let i = 0; i < frames.length; i++) {
    const frame = frames[i];
    const value = metric === 'focus' ? (frame.focus || 0)
                : metric === 'motion' ? frame.motion
                : frame.brightness;

    if (value < threshold) {
      if (sectionStart === null) {
        sectionStart = frame.timestamp;
      }
      sectionValues.push(value);
    } else {
      // End of low quality section
      if (sectionStart !== null) {
        const sectionEnd = frames[i - 1]?.timestamp ?? frame.timestamp;
        const sectionDuration = sectionEnd - sectionStart;
        if (sectionDuration >= minDuration) {
          lowQualitySections.push({
            start: sectionStart,
            end: sectionEnd,
            duration: sectionDuration,
            avgValue: sectionValues.reduce((a, b) => a + b, 0) / sectionValues.length,
          });
        }
        sectionStart = null;
        sectionValues = [];
      }
    }
  }

  // Handle section at the end
  if (sectionStart !== null) {
    const sectionEnd = frames[frames.length - 1].timestamp;
    const sectionDuration = sectionEnd - sectionStart;
    if (sectionDuration >= minDuration) {
      lowQualitySections.push({
        start: sectionStart,
        end: sectionEnd,
        duration: sectionDuration,
        avgValue: sectionValues.reduce((a, b) => a + b, 0) / sectionValues.length,
      });
    }
  }

  // Convert source time to timeline time
  // Source time t maps to timeline time: clip.startTime + (t - clip.inPoint)
  const timelineSections = lowQualitySections.map(s => ({
    sourceStart: s.start,
    sourceEnd: s.end,
    duration: s.duration,
    avgValue: s.avgValue,
    timelineStart: clip.startTime + (s.start - clip.inPoint),
    timelineEnd: clip.startTime + (s.end - clip.inPoint),
  }));

  return {
    success: true,
    data: {
      clipId,
      metric,
      threshold,
      minDuration,
      clipTimelineRange: { start: clip.startTime, end: clip.startTime + clip.duration },
      sections: timelineSections,
      totalLowQualityTime: lowQualitySections.reduce((sum, s) => sum + s.duration, 0),
      count: lowQualitySections.length,
    },
  };
}

export async function handleStartClipAnalysis(
  args: Record<string, unknown>,
  timelineStore: TimelineStore
): Promise<ToolResult> {
  const clipId = args.clipId as string;
  const clip = timelineStore.clips.find(c => c.id === clipId);
  if (!clip) {
    return { success: false, error: `Clip not found: ${clipId}` };
  }

  if (clip.analysisStatus === 'analyzing') {
    return { success: false, error: 'Analysis already in progress for this clip' };
  }

  // Import and start analysis (runs in background)
  const { analyzeClip } = await import('../../clipAnalyzer');
  analyzeClip(clipId); // Don't await - runs in background

  return {
    success: true,
    data: {
      clipId,
      clipName: clip.name,
      message: 'Analysis started. Check clip details later for results.',
    },
  };
}

export async function handleStartClipTranscription(
  args: Record<string, unknown>,
  timelineStore: TimelineStore
): Promise<ToolResult> {
  const clipId = args.clipId as string;
  const clip = timelineStore.clips.find(c => c.id === clipId);
  if (!clip) {
    return { success: false, error: `Clip not found: ${clipId}` };
  }

  // Import and start transcription (runs in background)
  const { transcribeClip } = await import('../../clipTranscriber');
  transcribeClip(clipId, 'de'); // Don't await - runs in background

  return {
    success: true,
    data: {
      clipId,
      clipName: clip.name,
      message: 'Transcription started. Check clip details later for results.',
    },
  };
}
