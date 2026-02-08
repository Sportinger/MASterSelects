// Timeline store constants and default values

import type { ClipTransform, TimelineTrack, TextClipProperties } from '../../types';

// Default transform for new clips
export const DEFAULT_TRANSFORM: ClipTransform = {
  opacity: 1,
  blendMode: 'normal',
  position: { x: 0, y: 0, z: 0 },
  scale: { x: 1, y: 1 },
  rotation: { x: 0, y: 0, z: 0 },
};

// Default timeline tracks
// Note: Video tracks are numbered so that the highest number is at the top (first in array)
// This matches compositing order where higher layers render on top
export const DEFAULT_TRACKS: TimelineTrack[] = [
  { id: 'video-2', name: 'Video 2', type: 'video', height: 60, muted: false, visible: true, solo: false },
  { id: 'video-1', name: 'Video 1', type: 'video', height: 60, muted: false, visible: true, solo: false },
  { id: 'audio-1', name: 'Audio', type: 'audio', height: 40, muted: false, visible: true, solo: false },
];

// Snap threshold in seconds (clips will snap when within this distance)
export const SNAP_THRESHOLD_SECONDS = 0.1;

// Resistance threshold - how far past a clip edge the user must drag to "break through"
// and be allowed to overlap (in PIXELS). Higher = harder to overlap.
// 100 pixels means user must drag about 2 inches on screen to force an overlap.
export const OVERLAP_RESISTANCE_PIXELS = 100;

// Property row heights for expanded tracks
export const PROPERTY_ROW_HEIGHT = 18;
export const GROUP_HEADER_HEIGHT = 20;

// Curve editor constants
export const CURVE_EDITOR_HEIGHT = 250;
export const MIN_CURVE_EDITOR_HEIGHT = 80;
export const MAX_CURVE_EDITOR_HEIGHT = 600;
export const BEZIER_HANDLE_SIZE = 8;

// Default durations
export const DEFAULT_TIMELINE_DURATION = 60;
export const DEFAULT_IMAGE_DURATION = 5;

// Zoom limits (pixels per second)
// MIN_ZOOM = 0.1 allows viewing ~10000 seconds (~2.7 hours) in a 1000px wide timeline
export const MIN_ZOOM = 0.1;
export const MAX_ZOOM = 200;

// Track height limits
export const MIN_TRACK_HEIGHT = 20;
export const MAX_TRACK_HEIGHT = 200;

// RAM Preview settings
export const RAM_PREVIEW_FPS = 30;

// Frame tolerance for position verification (at 30fps)
export const FRAME_TOLERANCE = 0.04;

// Default text clip duration
export const DEFAULT_TEXT_DURATION = 5;

// Default text properties for new text clips
export const DEFAULT_TEXT_PROPERTIES: TextClipProperties = {
  text: 'Enter text',
  fontFamily: 'Roboto',
  fontSize: 72,
  fontWeight: 400,
  fontStyle: 'normal',
  color: '#ffffff',
  textAlign: 'center',
  verticalAlign: 'middle',
  lineHeight: 1.2,
  letterSpacing: 0,
  strokeEnabled: false,
  strokeColor: '#000000',
  strokeWidth: 2,
  shadowEnabled: false,
  shadowColor: 'rgba(0, 0, 0, 0.5)',
  shadowOffsetX: 4,
  shadowOffsetY: 4,
  shadowBlur: 8,
  pathEnabled: false,
  pathPoints: [],
};
