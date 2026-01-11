// Glow Effect

import shader from './shader.wgsl?raw';
import type { EffectDefinition } from '../../types';

export const glow: EffectDefinition = {
  id: 'glow',
  name: 'Glow',
  category: 'stylize',

  shader,
  entryPoint: 'glowFragment',
  uniformSize: 16,

  params: {
    amount: {
      type: 'number',
      label: 'Amount',
      default: 1,
      min: 0,
      max: 3,
      step: 0.1,
      animatable: true,
    },
    threshold: {
      type: 'number',
      label: 'Threshold',
      default: 0.5,
      min: 0,
      max: 1,
      step: 0.01,
      animatable: true,
    },
    radius: {
      type: 'number',
      label: 'Radius',
      default: 5,
      min: 1,
      max: 20,
      step: 0.5,
      animatable: true,
    },
  },

  packUniforms: (params) => {
    return new Float32Array([
      params.amount as number || 1,
      params.threshold as number || 0.5,
      params.radius as number || 5,
      0,
    ]);
  },
};
