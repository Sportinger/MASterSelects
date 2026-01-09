// Engine-specific types and interfaces
// Re-export common types from main types file for convenience
export type { Layer, BlendMode, OutputWindow, EngineStats, Effect, EffectType } from '../../types';
import type { Layer } from '../../types';

// Blend mode to shader index mapping
export const BLEND_MODE_MAP: Record<string, number> = {
  // Normal
  'normal': 0,
  'dissolve': 1,
  'dancing-dissolve': 2,
  // Darken
  'darken': 3,
  'multiply': 4,
  'color-burn': 5,
  'classic-color-burn': 6,
  'linear-burn': 7,
  'darker-color': 8,
  // Lighten
  'add': 9,
  'lighten': 10,
  'screen': 11,
  'color-dodge': 12,
  'classic-color-dodge': 13,
  'linear-dodge': 14,
  'lighter-color': 15,
  // Contrast
  'overlay': 16,
  'soft-light': 17,
  'hard-light': 18,
  'linear-light': 19,
  'vivid-light': 20,
  'pin-light': 21,
  'hard-mix': 22,
  // Inversion
  'difference': 23,
  'classic-difference': 24,
  'exclusion': 25,
  'subtract': 26,
  'divide': 27,
  // Component
  'hue': 28,
  'saturation': 29,
  'color': 30,
  'luminosity': 31,
  // Stencil
  'stencil-alpha': 32,
  'stencil-luma': 33,
  'silhouette-alpha': 34,
  'silhouette-luma': 35,
  'alpha-add': 36,
};

// Layer render data prepared for GPU rendering
export interface LayerRenderData {
  layer: Layer;
  isVideo: boolean;
  externalTexture: GPUExternalTexture | null;
  textureView: GPUTextureView | null;
  sourceWidth: number;
  sourceHeight: number;
}

// Effect pipeline configuration
export interface EffectConfig {
  entryPoint: string;
  needsUniform: boolean;
  uniformSize: number;
}

// Effect configurations for pipeline creation
export const EFFECT_CONFIGS: Record<string, EffectConfig> = {
  'hue-shift': { entryPoint: 'hueShiftFragment', needsUniform: true, uniformSize: 16 },
  'brightness': { entryPoint: 'colorAdjustFragment', needsUniform: true, uniformSize: 16 },
  'contrast': { entryPoint: 'colorAdjustFragment', needsUniform: true, uniformSize: 16 },
  'saturation': { entryPoint: 'colorAdjustFragment', needsUniform: true, uniformSize: 16 },
  'pixelate': { entryPoint: 'pixelateFragment', needsUniform: true, uniformSize: 16 },
  'kaleidoscope': { entryPoint: 'kaleidoscopeFragment', needsUniform: true, uniformSize: 16 },
  'mirror': { entryPoint: 'mirrorFragment', needsUniform: true, uniformSize: 16 },
  'rgb-split': { entryPoint: 'rgbSplitFragment', needsUniform: true, uniformSize: 16 },
  'invert': { entryPoint: 'invertFragment', needsUniform: false, uniformSize: 0 },
  'levels': { entryPoint: 'levelsFragment', needsUniform: true, uniformSize: 32 },
};

// Detailed timing stats
export interface DetailedStats {
  rafGap: number;
  importTexture: number;
  renderPass: number;
  submit: number;
  total: number;
  dropsTotal: number;
  dropsLastSecond: number;
  dropsThisSecond: number;
  lastDropReason: 'none' | 'slow_raf' | 'slow_render' | 'slow_import';
  lastRafTime: number;
  decoder: 'WebCodecs' | 'HTMLVideo' | 'HTMLVideo(cached)' | 'none';
}

// Profile data for performance tracking
export interface ProfileData {
  importTexture: number;
  createBindGroup: number;
  renderPass: number;
  submit: number;
  total: number;
}

// GPU frame cache entry for RAM preview
export interface GpuFrameCacheEntry {
  texture: GPUTexture;
  view: GPUTextureView;
  bindGroup: GPUBindGroup;
}
