// Exposure Effect

import shader from './shader.wgsl?raw';
import type { EffectDefinition } from '../../types';

export const exposure: EffectDefinition = {
  id: 'exposure',
  name: 'Exposure',
  category: 'color',

  shader,
  entryPoint: 'exposureFragment',
  uniformSize: 16,

  params: {
    exposure: {
      type: 'number',
      label: 'Exposure (EV)',
      default: 0,
      min: -3,
      max: 3,
      step: 0.1,
      animatable: true,
    },
    offset: {
      type: 'number',
      label: 'Offset',
      default: 0,
      min: -0.5,
      max: 0.5,
      step: 0.01,
      animatable: true,
    },
    gamma: {
      type: 'number',
      label: 'Gamma',
      default: 1,
      min: 0.2,
      max: 3,
      step: 0.01,
      animatable: true,
    },
  },

  packUniforms: (params) => {
    return new Float32Array([
      params.exposure as number || 0,
      params.offset as number || 0,
      params.gamma as number || 1,
      0,
    ]);
  },
};
