// Effect system types and interfaces

import type { ComponentType } from 'react';

/**
 * Parameter definition for an effect
 */
export interface EffectParam {
  type: 'number' | 'boolean' | 'select' | 'color' | 'point';
  label: string;
  default: number | boolean | string;
  // For number type:
  min?: number;
  max?: number;
  step?: number;
  // For select type:
  options?: { value: string; label: string }[];
  // Keyframe support:
  animatable?: boolean;
  // Quality parameter (shown in collapsible Quality section):
  quality?: boolean;
}

/**
 * Props for custom effect control components
 */
export interface EffectControlProps {
  effectId: string;
  params: Record<string, number | boolean | string>;
  onChange: (params: Record<string, number | boolean | string>) => void;
  clipId?: string;
}

/**
 * Complete effect definition - each effect module exports this
 */
export interface EffectDefinition {
  // Identification
  id: string;                          // e.g. 'gaussian-blur'
  name: string;                        // e.g. 'Gaussian Blur'
  category: EffectCategory;            // e.g. 'blur'

  // GPU Configuration
  shader: string;                      // WGSL code as string
  entryPoint: string;                  // e.g. 'gaussianBlurFragment'
  uniformSize: number;                 // Bytes, 16-byte aligned

  // Parameters
  params: Record<string, EffectParam>;

  // Uniform packing function (params → Float32Array for GPU)
  packUniforms: (
    params: Record<string, number | boolean | string>,
    width: number,
    height: number
  ) => Float32Array | null;

  // Optional: Multi-pass for complex effects (blur, glow, etc.)
  passes?: number;

  // Optional: Custom UI component for special controls
  customControls?: ComponentType<EffectControlProps>;
}

/**
 * Effect categories for organization
 */
export type EffectCategory =
  | 'color'
  | 'blur'
  | 'distort'
  | 'stylize'
  | 'generate'
  | 'keying'
  | 'time'
  | 'transition';

/**
 * Category metadata for UI display
 */
export interface CategoryInfo {
  id: EffectCategory;
  name: string;
  icon?: string;
}

export const CATEGORY_INFO: CategoryInfo[] = [
  { id: 'color', name: 'Color Correction' },
  { id: 'blur', name: 'Blur & Sharpen' },
  { id: 'distort', name: 'Distort' },
  { id: 'stylize', name: 'Stylize' },
  { id: 'generate', name: 'Generate' },
  { id: 'keying', name: 'Keying' },
  { id: 'time', name: 'Time' },
  { id: 'transition', name: 'Transition' },
];

/**
 * Runtime effect instance (attached to a clip/layer)
 */
export interface EffectInstance {
  id: string;
  type: string;                        // References EffectDefinition.id
  name: string;
  enabled: boolean;
  params: Record<string, number | boolean | string>;
}
