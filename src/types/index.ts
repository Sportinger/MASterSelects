// Core types for WebVJ Mixer

export interface Layer {
  id: string;
  name: string;
  visible: boolean;
  opacity: number;
  blendMode: BlendMode;
  source: LayerSource | null;
  effects: Effect[];
  position: { x: number; y: number };
  scale: { x: number; y: number };
  rotation: number;
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
  | 'rgb-split';

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
  decoder: 'WebCodecs' | 'HTMLVideo' | 'none';
}

// Timeline types
export interface ClipTransform {
  opacity: number;          // 0-1
  blendMode: BlendMode;
  position: { x: number; y: number; z: number };
  scale: { x: number; y: number };
  rotation: { x: number; y: number; z: number };  // degrees
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
  waveform?: number[];    // Array of normalized amplitude values (0-1) for audio waveform
  transform: ClipTransform;  // Visual transform properties
  isLoading?: boolean;    // True while media is being loaded
  reversed?: boolean;     // True if clip plays in reverse
  // Nested composition support
  isComposition?: boolean;  // True if this clip is a nested composition
  compositionId?: string;   // ID of the nested composition
  nestedClips?: TimelineClip[];  // Loaded clips from the nested composition
  nestedTracks?: TimelineTrack[];  // Tracks from the nested composition
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
  waveform?: number[];
  transform: ClipTransform;
  keyframes?: Keyframe[];    // Animation keyframes for this clip
  // Nested composition support
  isComposition?: boolean;
  compositionId?: string;
}

// Serializable timeline data for composition storage
export interface CompositionTimelineData {
  tracks: TimelineTrack[];
  clips: SerializableClip[];
  playheadPosition: number;
  duration: number;
  zoom: number;
  scrollX: number;
  inPoint: number | null;
  outPoint: number | null;
  loopPlayback: boolean;
}

// Keyframe animation types
export type EasingType = 'linear' | 'ease-in' | 'ease-out' | 'ease-in-out';

export type AnimatableProperty =
  | 'opacity'
  | 'position.x' | 'position.y' | 'position.z'
  | 'scale.x' | 'scale.y'
  | 'rotation.x' | 'rotation.y' | 'rotation.z';

export interface Keyframe {
  id: string;
  clipId: string;
  time: number;           // Time relative to clip start (seconds)
  property: AnimatableProperty;
  value: number;
  easing: EasingType;     // Easing for interpolation TO the next keyframe
}
