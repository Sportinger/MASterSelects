// Threshold Effect

import shader from './shader.wgsl?raw';
import type { EffectDefinition } from '../../types';

export const threshold: EffectDefinition = {
  id: 'threshold',
  name: 'Threshold',
  category: 'stylize',

  shader,
  entryPoint: 'thresholdFragment',
  uniformSize: 16,

  params: {
    level: {
      type: 'number',
      label: 'Level',
      default: 0.5,
      min: 0,
      max: 1,
      step: 0.01,
      animatable: true,
    },
  },

  packUniforms: (params) => {
    return new Float32Array([
      params.level as number ?? 0.5,
      0, 0, 0,
    ]);
  },
};
