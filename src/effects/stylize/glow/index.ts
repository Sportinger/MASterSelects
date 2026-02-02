// Glow Effect - High Quality

import shader from './shader.wgsl?raw';
import type { EffectDefinition } from '../../types';

export const glow: EffectDefinition = {
  id: 'glow',
  name: 'Glow',
  category: 'stylize',

  shader,
  entryPoint: 'glowFragment',
  uniformSize: 32, // 8 floats

  params: {
    amount: {
      type: 'number',
      label: 'Amount',
      default: 1,
      min: 0,
      max: 5,
      step: 0.1,
      animatable: true,
    },
    threshold: {
      type: 'number',
      label: 'Threshold',
      default: 0.6,
      min: 0,
      max: 1,
      step: 0.01,
      animatable: true,
    },
    radius: {
      type: 'number',
      label: 'Radius',
      default: 20,
      min: 1,
      max: 100,
      step: 1,
      animatable: true,
    },
    softness: {
      type: 'number',
      label: 'Softness',
      default: 0.5,
      min: 0.1,
      max: 1,
      step: 0.05,
      animatable: true,
    },
    rings: {
      type: 'number',
      label: 'Rings',
      default: 4,
      min: 1,
      max: 32,
      step: 1,
      animatable: false,
      quality: true,
    },
    samplesPerRing: {
      type: 'number',
      label: 'Samples/Ring',
      default: 16,
      min: 4,
      max: 64,
      step: 1,
      animatable: false,
      quality: true,
    },
  },

  packUniforms: (params, width, height) => {
    return new Float32Array([
      params.amount as number ?? 1,
      params.threshold as number ?? 0.6,
      params.radius as number ?? 20,
      params.softness as number ?? 0.5,
      width,
      height,
      params.rings as number ?? 4,
      params.samplesPerRing as number ?? 16,
    ]);
  },
};
