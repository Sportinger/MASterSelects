// Vignette Effect

import shader from './shader.wgsl?raw';
import type { EffectDefinition } from '../../types';

export const vignette: EffectDefinition = {
  id: 'vignette',
  name: 'Vignette',
  category: 'stylize',

  shader,
  entryPoint: 'vignetteFragment',
  uniformSize: 16,

  params: {
    amount: {
      type: 'number',
      label: 'Amount',
      default: 0.5,
      min: 0,
      max: 1,
      step: 0.01,
      animatable: true,
    },
    size: {
      type: 'number',
      label: 'Size',
      default: 0.5,
      min: 0,
      max: 1.5,
      step: 0.01,
      animatable: true,
    },
    softness: {
      type: 'number',
      label: 'Softness',
      default: 0.5,
      min: 0,
      max: 1,
      step: 0.01,
      animatable: true,
    },
    roundness: {
      type: 'number',
      label: 'Roundness',
      default: 1,
      min: 0.5,
      max: 2,
      step: 0.01,
      animatable: true,
    },
  },

  packUniforms: (params) => {
    return new Float32Array([
      params.amount as number ?? 0.5,
      params.size as number ?? 0.5,
      params.softness as number ?? 0.5,
      params.roundness as number ?? 1,
    ]);
  },
};
