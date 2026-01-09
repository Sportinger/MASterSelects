// AI Tools Service - Provides tools for AI chat to control timeline editing
// Uses OpenAI function calling format

import { useTimelineStore } from '../stores/timeline';
import { useMediaStore } from '../stores/mediaStore';
import { engine } from '../engine/WebGPUEngine';
import type { TimelineClip, TimelineTrack } from '../stores/timeline/types';

// ============ TOOL DEFINITIONS (OpenAI Function Calling Format) ============

export const AI_TOOLS = [
  // === TIMELINE STATE TOOLS ===
  {
    type: 'function' as const,
    function: {
      name: 'getTimelineState',
      description: 'Get the current state of the timeline including all tracks, clips, playhead position, and duration. Always call this first to understand the current state before making changes.',
      parameters: {
        type: 'object',
        properties: {},
        required: [],
      },
    },
  },
  {
    type: 'function' as const,
    function: {
      name: 'getClipDetails',
      description: 'Get detailed information about a specific clip including its analysis data, transcript, effects, and transform properties.',
      parameters: {
        type: 'object',
        properties: {
          clipId: {
            type: 'string',
            description: 'The ID of the clip to get details for',
          },
        },
        required: ['clipId'],
      },
    },
  },
  {
    type: 'function' as const,
    function: {
      name: 'getClipsInTimeRange',
      description: 'Get all clips that overlap with a specific time range.',
      parameters: {
        type: 'object',
        properties: {
          startTime: {
            type: 'number',
            description: 'Start time in seconds',
          },
          endTime: {
            type: 'number',
            description: 'End time in seconds',
          },
          trackType: {
            type: 'string',
            enum: ['video', 'audio', 'all'],
            description: 'Filter by track type (default: all)',
          },
        },
        required: ['startTime', 'endTime'],
      },
    },
  },

  // === PLAYBACK TOOLS ===
  {
    type: 'function' as const,
    function: {
      name: 'setPlayhead',
      description: 'Move the playhead to a specific time position.',
      parameters: {
        type: 'object',
        properties: {
          time: {
            type: 'number',
            description: 'Time in seconds to move the playhead to',
          },
        },
        required: ['time'],
      },
    },
  },
  {
    type: 'function' as const,
    function: {
      name: 'setInOutPoints',
      description: 'Set the in and out points for playback range or export.',
      parameters: {
        type: 'object',
        properties: {
          inPoint: {
            type: 'number',
            description: 'In point time in seconds (null to clear)',
          },
          outPoint: {
            type: 'number',
            description: 'Out point time in seconds (null to clear)',
          },
        },
        required: [],
      },
    },
  },

  // === CLIP EDITING TOOLS ===
  {
    type: 'function' as const,
    function: {
      name: 'splitClip',
      description: 'Split a clip at a specific time, creating two separate clips.',
      parameters: {
        type: 'object',
        properties: {
          clipId: {
            type: 'string',
            description: 'The ID of the clip to split',
          },
          splitTime: {
            type: 'number',
            description: 'The time in seconds (timeline time, not clip-relative) where to split',
          },
        },
        required: ['clipId', 'splitTime'],
      },
    },
  },
  {
    type: 'function' as const,
    function: {
      name: 'deleteClip',
      description: 'Delete a clip from the timeline.',
      parameters: {
        type: 'object',
        properties: {
          clipId: {
            type: 'string',
            description: 'The ID of the clip to delete',
          },
        },
        required: ['clipId'],
      },
    },
  },
  {
    type: 'function' as const,
    function: {
      name: 'deleteClips',
      description: 'Delete multiple clips from the timeline at once.',
      parameters: {
        type: 'object',
        properties: {
          clipIds: {
            type: 'array',
            items: { type: 'string' },
            description: 'Array of clip IDs to delete',
          },
        },
        required: ['clipIds'],
      },
    },
  },
  {
    type: 'function' as const,
    function: {
      name: 'moveClip',
      description: 'Move a clip to a new position and/or track.',
      parameters: {
        type: 'object',
        properties: {
          clipId: {
            type: 'string',
            description: 'The ID of the clip to move',
          },
          newStartTime: {
            type: 'number',
            description: 'New start time in seconds',
          },
          newTrackId: {
            type: 'string',
            description: 'ID of the track to move the clip to (optional, keeps current track if not specified)',
          },
        },
        required: ['clipId', 'newStartTime'],
      },
    },
  },
  {
    type: 'function' as const,
    function: {
      name: 'trimClip',
      description: 'Trim a clip by adjusting its in and out points (relative to the source media).',
      parameters: {
        type: 'object',
        properties: {
          clipId: {
            type: 'string',
            description: 'The ID of the clip to trim',
          },
          inPoint: {
            type: 'number',
            description: 'New in point in seconds (relative to source media start)',
          },
          outPoint: {
            type: 'number',
            description: 'New out point in seconds (relative to source media start)',
          },
        },
        required: ['clipId', 'inPoint', 'outPoint'],
      },
    },
  },

  // === TRACK TOOLS ===
  {
    type: 'function' as const,
    function: {
      name: 'createTrack',
      description: 'Create a new video or audio track.',
      parameters: {
        type: 'object',
        properties: {
          type: {
            type: 'string',
            enum: ['video', 'audio'],
            description: 'Type of track to create',
          },
        },
        required: ['type'],
      },
    },
  },
  {
    type: 'function' as const,
    function: {
      name: 'deleteTrack',
      description: 'Delete a track and all clips on it.',
      parameters: {
        type: 'object',
        properties: {
          trackId: {
            type: 'string',
            description: 'The ID of the track to delete',
          },
        },
        required: ['trackId'],
      },
    },
  },
  {
    type: 'function' as const,
    function: {
      name: 'setTrackVisibility',
      description: 'Show or hide a video track.',
      parameters: {
        type: 'object',
        properties: {
          trackId: {
            type: 'string',
            description: 'The ID of the track',
          },
          visible: {
            type: 'boolean',
            description: 'Whether the track should be visible',
          },
        },
        required: ['trackId', 'visible'],
      },
    },
  },
  {
    type: 'function' as const,
    function: {
      name: 'setTrackMuted',
      description: 'Mute or unmute an audio track.',
      parameters: {
        type: 'object',
        properties: {
          trackId: {
            type: 'string',
            description: 'The ID of the track',
          },
          muted: {
            type: 'boolean',
            description: 'Whether the track should be muted',
          },
        },
        required: ['trackId', 'muted'],
      },
    },
  },

  // === VISUAL CAPTURE TOOLS ===
  {
    type: 'function' as const,
    function: {
      name: 'captureFrame',
      description: 'Capture the current composition output as a still image (screenshot). Returns a base64-encoded PNG.',
      parameters: {
        type: 'object',
        properties: {
          time: {
            type: 'number',
            description: 'Time in seconds to capture (optional, uses current playhead if not specified)',
          },
        },
        required: [],
      },
    },
  },

  // === SELECTION TOOLS ===
  {
    type: 'function' as const,
    function: {
      name: 'selectClips',
      description: 'Select one or more clips in the timeline.',
      parameters: {
        type: 'object',
        properties: {
          clipIds: {
            type: 'array',
            items: { type: 'string' },
            description: 'Array of clip IDs to select',
          },
        },
        required: ['clipIds'],
      },
    },
  },
  {
    type: 'function' as const,
    function: {
      name: 'clearSelection',
      description: 'Clear the current clip selection.',
      parameters: {
        type: 'object',
        properties: {},
        required: [],
      },
    },
  },

  // === ANALYSIS & TRANSCRIPT TOOLS ===
  {
    type: 'function' as const,
    function: {
      name: 'getClipAnalysis',
      description: 'Get the analysis data for a clip (motion, faces, colors, etc.). Returns null if not analyzed.',
      parameters: {
        type: 'object',
        properties: {
          clipId: {
            type: 'string',
            description: 'The ID of the clip to get analysis for',
          },
        },
        required: ['clipId'],
      },
    },
  },
  {
    type: 'function' as const,
    function: {
      name: 'getClipTranscript',
      description: 'Get the transcript/subtitles for a clip with word-level timestamps. Returns null if no transcript.',
      parameters: {
        type: 'object',
        properties: {
          clipId: {
            type: 'string',
            description: 'The ID of the clip to get transcript for',
          },
        },
        required: ['clipId'],
      },
    },
  },
  {
    type: 'function' as const,
    function: {
      name: 'findSilentSections',
      description: 'Find sections in a clip where there is no speech (based on transcript). Useful for finding pauses to cut.',
      parameters: {
        type: 'object',
        properties: {
          clipId: {
            type: 'string',
            description: 'The ID of the clip to analyze',
          },
          minDuration: {
            type: 'number',
            description: 'Minimum silence duration in seconds to include (default: 0.5)',
          },
        },
        required: ['clipId'],
      },
    },
  },
  {
    type: 'function' as const,
    function: {
      name: 'findLowQualitySections',
      description: 'Find sections in a clip where focus, motion, or brightness is below a threshold. Useful for finding blurry/unfocused parts to cut.',
      parameters: {
        type: 'object',
        properties: {
          clipId: {
            type: 'string',
            description: 'The ID of the clip to analyze',
          },
          metric: {
            type: 'string',
            enum: ['focus', 'motion', 'brightness'],
            description: 'Which metric to check (default: focus)',
          },
          threshold: {
            type: 'number',
            description: 'Threshold value 0-1. Sections BELOW this value are returned. (default: 0.7)',
          },
          minDuration: {
            type: 'number',
            description: 'Minimum section duration in seconds to include (default: 0.5)',
          },
        },
        required: ['clipId'],
      },
    },
  },
];

// ============ TOOL HANDLERS ============

export interface ToolResult {
  success: boolean;
  data?: unknown;
  error?: string;
}

// Helper to format clip info for AI
function formatClipInfo(clip: TimelineClip, track: TimelineTrack | undefined) {
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
    sourceType: clip.source.type,
    hasAnalysis: clip.analysisStatus === 'ready',
    hasTranscript: !!clip.transcript?.segments?.length,
    // Transform info
    transform: clip.transform,
    // Effects count
    effectsCount: clip.effects?.length || 0,
  };
}

// Helper to format track info for AI
function formatTrackInfo(track: TimelineTrack, clips: TimelineClip[]) {
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
      hasTranscript: !!c.transcript?.segments?.length,
    })),
  };
}

export async function executeAITool(toolName: string, args: Record<string, unknown>): Promise<ToolResult> {
  const timelineStore = useTimelineStore.getState();
  const mediaStore = useMediaStore.getState();

  try {
    switch (toolName) {
      // === TIMELINE STATE ===
      case 'getTimelineState': {
        const { tracks, clips, playheadPosition, duration, inPoint, outPoint, zoom, selectedClipIds } = timelineStore;

        const videoTracks = tracks.filter(t => t.type === 'video').map(t => formatTrackInfo(t, clips));
        const audioTracks = tracks.filter(t => t.type === 'audio').map(t => formatTrackInfo(t, clips));

        // Get details of selected clips
        const selectedClipIdsArray = Array.from(selectedClipIds);
        const selectedClips = selectedClipIdsArray.map(id => {
          const clip = clips.find(c => c.id === id);
          if (!clip) return null;
          const track = tracks.find(t => t.id === clip.trackId);
          return {
            id: clip.id,
            name: clip.name,
            trackId: clip.trackId,
            trackName: track?.name || 'Unknown',
            startTime: clip.startTime,
            endTime: clip.startTime + clip.duration,
            duration: clip.duration,
            hasAnalysis: clip.analysisStatus === 'ready',
            hasTranscript: !!clip.transcript?.segments?.length,
          };
        }).filter(Boolean);

        return {
          success: true,
          data: {
            playheadPosition,
            duration,
            inPoint,
            outPoint,
            zoom,
            totalClips: clips.length,
            // Selected clips info
            selectedClipIds: selectedClipIdsArray,
            selectedClips,
            hasSelection: selectedClipIdsArray.length > 0,
            // Tracks with their clips
            videoTracks,
            audioTracks,
          },
        };
      }

      case 'getClipDetails': {
        const clipId = args.clipId as string;
        const clip = timelineStore.clips.find(c => c.id === clipId);
        if (!clip) {
          return { success: false, error: `Clip not found: ${clipId}` };
        }
        const track = timelineStore.tracks.find(t => t.id === clip.trackId);

        return {
          success: true,
          data: {
            ...formatClipInfo(clip, track),
            effects: clip.effects || [],
            masks: clip.masks || [],
            transcript: clip.transcript,
            analysisStatus: clip.analysisStatus,
          },
        };
      }

      case 'getClipsInTimeRange': {
        const startTime = args.startTime as number;
        const endTime = args.endTime as number;
        const trackType = (args.trackType as string) || 'all';

        const { clips, tracks } = timelineStore;

        const filteredClips = clips.filter(clip => {
          const clipEnd = clip.startTime + clip.duration;
          const overlaps = clip.startTime < endTime && clipEnd > startTime;
          if (!overlaps) return false;

          if (trackType === 'all') return true;
          const track = tracks.find(t => t.id === clip.trackId);
          return track?.type === trackType;
        });

        return {
          success: true,
          data: {
            clips: filteredClips.map(c => {
              const track = tracks.find(t => t.id === c.trackId);
              return formatClipInfo(c, track);
            }),
            count: filteredClips.length,
          },
        };
      }

      // === PLAYBACK ===
      case 'setPlayhead': {
        const time = args.time as number;
        timelineStore.setPlayheadPosition(Math.max(0, time));
        return { success: true, data: { newPosition: Math.max(0, time) } };
      }

      case 'setInOutPoints': {
        const inPoint = args.inPoint as number | undefined;
        const outPoint = args.outPoint as number | undefined;

        if (inPoint !== undefined) {
          timelineStore.setInPoint(inPoint);
        }
        if (outPoint !== undefined) {
          timelineStore.setOutPoint(outPoint);
        }

        return { success: true, data: { inPoint, outPoint } };
      }

      // === CLIP EDITING ===
      case 'splitClip': {
        const clipId = args.clipId as string;
        const splitTime = args.splitTime as number;

        const clip = timelineStore.clips.find(c => c.id === clipId);
        if (!clip) {
          return { success: false, error: `Clip not found: ${clipId}` };
        }

        const clipEnd = clip.startTime + clip.duration;
        if (splitTime <= clip.startTime || splitTime >= clipEnd) {
          return { success: false, error: `Split time ${splitTime}s is outside clip range (${clip.startTime}s - ${clipEnd}s)` };
        }

        timelineStore.splitClip(clipId, splitTime);
        return { success: true, data: { splitAt: splitTime, originalClipId: clipId } };
      }

      case 'deleteClip': {
        const clipId = args.clipId as string;
        const clip = timelineStore.clips.find(c => c.id === clipId);
        if (!clip) {
          return { success: false, error: `Clip not found: ${clipId}` };
        }

        timelineStore.removeClip(clipId);
        return { success: true, data: { deletedClipId: clipId, clipName: clip.name } };
      }

      case 'deleteClips': {
        const clipIds = args.clipIds as string[];
        const deleted: string[] = [];
        const notFound: string[] = [];

        for (const clipId of clipIds) {
          const clip = timelineStore.clips.find(c => c.id === clipId);
          if (clip) {
            timelineStore.removeClip(clipId);
            deleted.push(clipId);
          } else {
            notFound.push(clipId);
          }
        }

        return {
          success: true,
          data: { deleted, notFound, deletedCount: deleted.length },
        };
      }

      case 'moveClip': {
        const clipId = args.clipId as string;
        const newStartTime = args.newStartTime as number;
        const newTrackId = args.newTrackId as string | undefined;

        const clip = timelineStore.clips.find(c => c.id === clipId);
        if (!clip) {
          return { success: false, error: `Clip not found: ${clipId}` };
        }

        if (newTrackId) {
          const track = timelineStore.tracks.find(t => t.id === newTrackId);
          if (!track) {
            return { success: false, error: `Track not found: ${newTrackId}` };
          }
        }

        timelineStore.moveClip(clipId, newStartTime, newTrackId);
        return {
          success: true,
          data: {
            clipId,
            newStartTime,
            newTrackId: newTrackId || clip.trackId,
          },
        };
      }

      case 'trimClip': {
        const clipId = args.clipId as string;
        const inPoint = args.inPoint as number;
        const outPoint = args.outPoint as number;

        const clip = timelineStore.clips.find(c => c.id === clipId);
        if (!clip) {
          return { success: false, error: `Clip not found: ${clipId}` };
        }

        if (inPoint >= outPoint) {
          return { success: false, error: 'In point must be less than out point' };
        }

        timelineStore.trimClip(clipId, inPoint, outPoint);
        return { success: true, data: { clipId, inPoint, outPoint, newDuration: outPoint - inPoint } };
      }

      // === TRACKS ===
      case 'createTrack': {
        const type = args.type as 'video' | 'audio';
        const trackId = timelineStore.addTrack(type);
        const track = timelineStore.tracks.find(t => t.id === trackId);

        return {
          success: true,
          data: {
            trackId,
            trackName: track?.name,
            trackType: type,
          },
        };
      }

      case 'deleteTrack': {
        const trackId = args.trackId as string;
        const track = timelineStore.tracks.find(t => t.id === trackId);
        if (!track) {
          return { success: false, error: `Track not found: ${trackId}` };
        }

        timelineStore.removeTrack(trackId);
        return { success: true, data: { deletedTrackId: trackId, trackName: track.name } };
      }

      case 'setTrackVisibility': {
        const trackId = args.trackId as string;
        const visible = args.visible as boolean;

        const track = timelineStore.tracks.find(t => t.id === trackId);
        if (!track) {
          return { success: false, error: `Track not found: ${trackId}` };
        }

        timelineStore.setTrackVisible(trackId, visible);
        return { success: true, data: { trackId, visible } };
      }

      case 'setTrackMuted': {
        const trackId = args.trackId as string;
        const muted = args.muted as boolean;

        const track = timelineStore.tracks.find(t => t.id === trackId);
        if (!track) {
          return { success: false, error: `Track not found: ${trackId}` };
        }

        timelineStore.setTrackMuted(trackId, muted);
        return { success: true, data: { trackId, muted } };
      }

      // === VISUAL CAPTURE ===
      case 'captureFrame': {
        const time = args.time as number | undefined;

        // If time specified, move playhead there first
        if (time !== undefined) {
          timelineStore.setPlayheadPosition(time);
          // Wait a frame for render to update
          await new Promise(resolve => setTimeout(resolve, 100));
        }

        const pixels = await engine.readPixels();
        if (!pixels) {
          return { success: false, error: 'Failed to capture frame - engine not ready' };
        }

        const { width, height } = engine.getOutputDimensions();

        // Convert to PNG using canvas
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        if (!ctx) {
          return { success: false, error: 'Failed to create canvas context' };
        }

        const imageData = new ImageData(pixels, width, height);
        ctx.putImageData(imageData, 0, 0);

        const dataUrl = canvas.toDataURL('image/png');

        return {
          success: true,
          data: {
            capturedAt: time ?? timelineStore.playheadPosition,
            width,
            height,
            dataUrl,
          },
        };
      }

      // === SELECTION ===
      case 'selectClips': {
        const clipIds = args.clipIds as string[];
        timelineStore.selectClips(clipIds);
        return { success: true, data: { selectedClipIds: clipIds } };
      }

      case 'clearSelection': {
        timelineStore.clearClipSelection();
        return { success: true, data: { message: 'Selection cleared' } };
      }

      // === ANALYSIS & TRANSCRIPT ===
      case 'getClipAnalysis': {
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

      case 'getClipTranscript': {
        const clipId = args.clipId as string;
        const clip = timelineStore.clips.find(c => c.id === clipId);
        if (!clip) {
          return { success: false, error: `Clip not found: ${clipId}` };
        }

        if (!clip.transcript?.segments?.length) {
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
            language: clip.transcript.language,
            segmentCount: clip.transcript.segments.length,
            segments: clip.transcript.segments.map(seg => ({
              start: seg.start,
              end: seg.end,
              text: seg.text,
              words: seg.words,
            })),
            // Full text for easy reading
            fullText: clip.transcript.segments.map(s => s.text).join(' '),
          },
        };
      }

      case 'findSilentSections': {
        const clipId = args.clipId as string;
        const minDuration = (args.minDuration as number) || 0.5;

        const clip = timelineStore.clips.find(c => c.id === clipId);
        if (!clip) {
          return { success: false, error: `Clip not found: ${clipId}` };
        }

        if (!clip.transcript?.segments?.length) {
          return {
            success: false,
            error: 'No transcript available to analyze for silence.',
          };
        }

        const silentSections: Array<{ start: number; end: number; duration: number }> = [];
        const segments = clip.transcript.segments;

        // Check for silence at the beginning
        if (segments[0].start > minDuration) {
          silentSections.push({
            start: 0,
            end: segments[0].start,
            duration: segments[0].start,
          });
        }

        // Check gaps between segments
        for (let i = 0; i < segments.length - 1; i++) {
          const gapStart = segments[i].end;
          const gapEnd = segments[i + 1].start;
          const gapDuration = gapEnd - gapStart;

          if (gapDuration >= minDuration) {
            silentSections.push({
              start: gapStart,
              end: gapEnd,
              duration: gapDuration,
            });
          }
        }

        // Check for silence at the end
        const lastSegment = segments[segments.length - 1];
        const clipDuration = clip.outPoint - clip.inPoint;
        if (clipDuration - lastSegment.end > minDuration) {
          silentSections.push({
            start: lastSegment.end,
            end: clipDuration,
            duration: clipDuration - lastSegment.end,
          });
        }

        // Convert to timeline time (add clip start time)
        const timelineSilentSections = silentSections.map(s => ({
          ...s,
          timelineStart: clip.startTime + s.start,
          timelineEnd: clip.startTime + s.end,
        }));

        return {
          success: true,
          data: {
            clipId,
            minDuration,
            silentSections: timelineSilentSections,
            totalSilentTime: silentSections.reduce((sum, s) => sum + s.duration, 0),
            count: silentSections.length,
          },
        };
      }

      case 'findLowQualitySections': {
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

        const frames = clip.analysis.frames;
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

        // Convert to timeline time
        const timelineSections = lowQualitySections.map(s => ({
          ...s,
          timelineStart: clip.startTime + s.start,
          timelineEnd: clip.startTime + s.end,
        }));

        return {
          success: true,
          data: {
            clipId,
            metric,
            threshold,
            minDuration,
            sections: timelineSections,
            totalLowQualityTime: lowQualitySections.reduce((sum, s) => sum + s.duration, 0),
            count: lowQualitySections.length,
          },
        };
      }

      default:
        return { success: false, error: `Unknown tool: ${toolName}` };
    }
  } catch (error) {
    console.error(`[AI Tool] Error executing ${toolName}:`, error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error occurred',
    };
  }
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
