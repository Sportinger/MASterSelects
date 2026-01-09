// Core types for WebVJ Mixer

export interface Layer {
  id: string;
  name: string;
  visible: boolean;
  opacity: number;
  blendMode: BlendMode;
  source: LayerSource | null;
  effects: Effect[];
  position: { x: number; y: number; z: number };
  scale: { x: number; y: number };
  rotation: number | { x: number; y: number; z: number };  // Single value (z only) or full 3D rotation
  // Mask properties (passed from timeline clip masks for GPU processing)
  maskFeather?: number;  // Blur radius in pixels (0-50), handled in GPU shader
  maskFeatherQuality?: number;  // Blur quality: 0=low (9 samples), 1=medium (17), 2=high (25)
  maskInvert?: boolean;  // Whether to invert the mask, handled in GPU shader
}

export type BlendMode =
  // Normal
  | 'normal'
  | 'dissolve'
  | 'dancing-dissolve'
  // Darken
  | 'darken'
  | 'multiply'
  | 'color-burn'
  | 'classic-color-burn'
  | 'linear-burn'
  | 'darker-color'
  // Lighten
  | 'add'
  | 'lighten'
  | 'screen'
  | 'color-dodge'
  | 'classic-color-dodge'
  | 'linear-dodge'
  | 'lighter-color'
  // Contrast
  | 'overlay'
  | 'soft-light'
  | 'hard-light'
  | 'linear-light'
  | 'vivid-light'
  | 'pin-light'
  | 'hard-mix'
  // Inversion
  | 'difference'
  | 'classic-difference'
  | 'exclusion'
  | 'subtract'
  | 'divide'
  // Component
  | 'hue'
  | 'saturation'
  | 'color'
  | 'luminosity'
  // Stencil
  | 'stencil-alpha'
  | 'stencil-luma'
  | 'silhouette-alpha'
  | 'silhouette-luma'
  | 'alpha-add';

export interface LayerSource {
  type: 'video' | 'image' | 'camera' | 'color';
  file?: File;
  videoElement?: HTMLVideoElement;
  imageElement?: HTMLImageElement;
  color?: string;
  texture?: GPUTexture;
  // WebCodecs support for hardware-accelerated video decode
  webCodecsPlayer?: import('../engine/WebCodecsPlayer').WebCodecsPlayer;
  videoFrame?: VideoFrame;
}

export interface Effect {
  id: string;
  name: string;
  type: EffectType;
  enabled: boolean;
  params: Record<string, number | boolean | string>;
}

export type EffectType =
  | 'hue-shift'
  | 'saturation'
  | 'brightness'
  | 'contrast'
  | 'blur'
  | 'pixelate'
  | 'kaleidoscope'
  | 'mirror'
  | 'invert'
  | 'rgb-split'
  | 'levels';

export interface OutputWindow {
  id: string;
  name: string;
  window: Window | null;
  canvas: HTMLCanvasElement | null;
  context: GPUCanvasContext | null;
  isFullscreen: boolean;
}

export interface Project {
  id: string;
  name: string;
  layers: Layer[];
  outputResolution: { width: number; height: number };
  fps: number;
}

export interface MIDIMapping {
  channel: number;
  control: number;
  target: string;
  min: number;
  max: number;
}

export interface EngineStats {
  fps: number;
  frameTime: number;
  gpuMemory: number;
  // Detailed timing (ms)
  timing: {
    rafGap: number;        // Time between rAF callbacks (should be ~16.67ms for 60fps)
    importTexture: number; // Time to import video textures
    renderPass: number;    // Time for GPU render passes
    submit: number;        // Time for GPU queue submit
    total: number;         // Total render time
  };
  // Frame drop stats
  drops: {
    count: number;         // Total dropped frames this session
    lastSecond: number;    // Drops in last second
    reason: 'none' | 'slow_raf' | 'slow_render' | 'slow_import';
  };
  // Current frame info
  layerCount: number;
  targetFps: number;
  // Decoder info
  decoder: 'WebCodecs' | 'HTMLVideo' | 'HTMLVideo(cached)' | 'none';
}

// Timeline types
export interface ClipTransform {
  opacity: number;          // 0-1
  blendMode: BlendMode;
  position: { x: number; y: number; z: number };
  scale: { x: number; y: number };
  rotation: { x: number; y: number; z: number };  // degrees
}

// Transcript word/chunk for speech-to-text
export interface TranscriptWord {
  id: string;
  text: string;
  start: number;        // Start time in seconds (relative to clip source)
  end: number;          // End time in seconds (relative to clip source)
  confidence?: number;  // 0-1 confidence score
  speaker?: string;     // Speaker label if diarization available
}

// Transcript status
export type TranscriptStatus = 'none' | 'transcribing' | 'ready' | 'error';

// Analysis types for focus/motion/face detection
export type AnalysisStatus = 'none' | 'analyzing' | 'ready' | 'error';

export interface FrameAnalysisData {
  timestamp: number;      // Time in seconds (relative to clip source)
  motion: number;         // 0-1 overall motion score (legacy, kept for compatibility)
  globalMotion: number;   // 0-1 camera/scene motion (whole frame changes uniformly)
  localMotion: number;    // 0-1 object motion (localized changes within frame)
  focus: number;          // 0-1 focus/sharpness score
  faceCount: number;      // Number of faces detected
  isSceneCut?: boolean;   // True if this frame is likely a scene cut
}

export interface ClipAnalysis {
  frames: FrameAnalysisData[];
  sampleInterval: number; // Milliseconds between samples
}

export interface TimelineClip {
  id: string;
  trackId: string;
  name: string;
  file: File;
  startTime: number;      // Start position on timeline (seconds)
  duration: number;       // Clip duration (seconds)
  inPoint: number;        // Trim in point within source (seconds)
  outPoint: number;       // Trim out point within source (seconds)
  source: {
    type: 'video' | 'audio' | 'image';
    videoElement?: HTMLVideoElement;
    audioElement?: HTMLAudioElement;
    imageElement?: HTMLImageElement;
    webCodecsPlayer?: import('../engine/WebCodecsPlayer').WebCodecsPlayer;
    naturalDuration?: number;
    mediaFileId?: string;  // Reference to MediaFile for proxy lookup
  } | null;
  thumbnails?: string[];  // Array of data URLs for filmstrip preview
  linkedClipId?: string;  // ID of linked clip (e.g., audio linked to video)
  linkedGroupId?: string; // ID of multicam group (clips synced together)
  waveform?: number[];    // Array of normalized amplitude values (0-1) for audio waveform
  waveformGenerating?: boolean;  // True while waveform is being generated
  waveformProgress?: number;     // 0-100 progress of waveform generation
  transform: ClipTransform;  // Visual transform properties
  effects: Effect[];      // Effects applied to this clip
  isLoading?: boolean;    // True while media is being loaded
  reversed?: boolean;     // True if clip plays in reverse
  // Nested composition support
  isComposition?: boolean;  // True if this clip is a nested composition
  compositionId?: string;   // ID of the nested composition
  nestedClips?: TimelineClip[];  // Loaded clips from the nested composition
  nestedTracks?: TimelineTrack[];  // Tracks from the nested composition
  // Mask support
  masks?: ClipMask[];     // Array of masks applied to this clip
  // Transcript support
  transcript?: TranscriptWord[];  // Speech-to-text transcript
  transcriptStatus?: TranscriptStatus;  // Transcription status
  transcriptProgress?: number;  // 0-100 progress
  transcriptMessage?: string;  // Status message during transcription
  // Analysis support (focus/motion/face)
  analysis?: ClipAnalysis;
  analysisStatus?: AnalysisStatus;
  analysisProgress?: number;  // 0-100 progress
}

export interface TimelineTrack {
  id: string;
  name: string;
  type: 'video' | 'audio';
  height: number;
  muted: boolean;
  visible: boolean;
  solo: boolean;
}

export interface TimelineState {
  tracks: TimelineTrack[];
  clips: TimelineClip[];
  playheadPosition: number;  // Current time in seconds
  duration: number;          // Total timeline duration
  zoom: number;              // Pixels per second
  scrollX: number;           // Horizontal scroll position
  isPlaying: boolean;
  selectedClipId: string | null;
}

// Serializable clip data for storage (without DOM elements)
export interface SerializableClip {
  id: string;
  trackId: string;
  name: string;
  mediaFileId: string;       // Reference to MediaFile in mediaStore
  startTime: number;
  duration: number;
  inPoint: number;
  outPoint: number;
  sourceType: 'video' | 'audio' | 'image';
  naturalDuration?: number;
  thumbnails?: string[];
  linkedClipId?: string;
  linkedGroupId?: string;  // Multicam group ID
  waveform?: number[];
  transform: ClipTransform;
  effects: Effect[];         // Effects applied to this clip
  keyframes?: Keyframe[];    // Animation keyframes for this clip
  // Nested composition support
  isComposition?: boolean;
  compositionId?: string;
  // Mask support
  masks?: ClipMask[];        // Masks applied to this clip
  // Transcript data
  transcript?: TranscriptWord[];
  transcriptStatus?: TranscriptStatus;
  // Analysis data
  analysis?: ClipAnalysis;
  analysisStatus?: AnalysisStatus;
  // Playback
  reversed?: boolean;
}

// Serializable timeline data for composition storage
export interface CompositionTimelineData {
  tracks: TimelineTrack[];
  clips: SerializableClip[];
  playheadPosition: number;
  duration: number;
  durationLocked?: boolean;  // When true, duration won't auto-update based on clips
  zoom: number;
  scrollX: number;
  inPoint: number | null;
  outPoint: number | null;
  loopPlayback: boolean;
}

// Keyframe animation types
export type EasingType = 'linear' | 'ease-in' | 'ease-out' | 'ease-in-out';

// Transform properties that can be animated
export type TransformProperty =
  | 'opacity'
  | 'position.x' | 'position.y' | 'position.z'
  | 'scale.x' | 'scale.y'
  | 'rotation.x' | 'rotation.y' | 'rotation.z';

// Effect property format: effect.{effectId}.{paramName}
// Example: effect.effect_123456.shift, effect.effect_123456.amount
export type EffectProperty = `effect.${string}.${string}`;

// Combined animatable property type
export type AnimatableProperty = TransformProperty | EffectProperty;

// Helper to check if a property is an effect property
export function isEffectProperty(property: string): property is EffectProperty {
  return property.startsWith('effect.');
}

// Helper to parse effect property into parts
export function parseEffectProperty(property: EffectProperty): { effectId: string; paramName: string } | null {
  const parts = property.split('.');
  if (parts.length === 3 && parts[0] === 'effect') {
    return { effectId: parts[1], paramName: parts[2] };
  }
  return null;
}

// Helper to create effect property string
export function createEffectProperty(effectId: string, paramName: string): EffectProperty {
  return `effect.${effectId}.${paramName}` as EffectProperty;
}

// Mask types for After Effects-style clip masking
export interface MaskVertex {
  id: string;
  x: number;              // Position relative to clip (0-1 normalized)
  y: number;
  handleIn: { x: number; y: number };   // Bezier control handle (relative to vertex)
  handleOut: { x: number; y: number };  // Bezier control handle (relative to vertex)
}

export type MaskMode = 'add' | 'subtract' | 'intersect';

export interface ClipMask {
  id: string;
  name: string;
  vertices: MaskVertex[];
  closed: boolean;        // Is the path closed
  opacity: number;        // 0-1
  feather: number;        // Blur amount in pixels
  featherQuality: number; // 0=low (fast), 1=medium, 2=high (smooth)
  inverted: boolean;
  mode: MaskMode;
  expanded: boolean;      // UI state - expanded in properties panel
  position: { x: number; y: number };  // Offset in normalized coords (0-1)
  visible: boolean;       // Toggle outline visibility
}

export interface Keyframe {
  id: string;
  clipId: string;
  time: number;           // Time relative to clip start (seconds)
  property: AnimatableProperty;
  value: number;
  easing: EasingType;     // Easing for interpolation TO the next keyframe
}
