// Color Temperature Effect

import shader from './shader.wgsl?raw';
import type { EffectDefinition } from '../../types';

export const temperature: EffectDefinition = {
  id: 'temperature',
  name: 'Temperature',
  category: 'color',

  shader,
  entryPoint: 'temperatureFragment',
  uniformSize: 16,

  params: {
    temperature: {
      type: 'number',
      label: 'Temperature',
      default: 0,
      min: -1,
      max: 1,
      step: 0.01,
      animatable: true,
    },
    tint: {
      type: 'number',
      label: 'Tint',
      default: 0,
      min: -1,
      max: 1,
      step: 0.01,
      animatable: true,
    },
  },

  packUniforms: (params) => {
    return new Float32Array([
      params.temperature as number || 0,
      params.tint as number || 0,
      0, 0,
    ]);
  },
};
