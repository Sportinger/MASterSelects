// Kaleidoscope Effect

import shader from './shader.wgsl?raw';
import type { EffectDefinition } from '../../types';

export const kaleidoscope: EffectDefinition = {
  id: 'kaleidoscope',
  name: 'Kaleidoscope',
  category: 'distort',

  shader,
  entryPoint: 'kaleidoscopeFragment',
  uniformSize: 16,

  params: {
    segments: {
      type: 'number',
      label: 'Segments',
      default: 6,
      min: 2,
      max: 16,
      step: 1,
      animatable: true,
    },
    rotation: {
      type: 'number',
      label: 'Rotation',
      default: 0,
      min: 0,
      max: 6.28318, // TAU
      step: 0.01,
      animatable: true,
    },
  },

  packUniforms: (params) => {
    return new Float32Array([
      params.segments as number ?? 6,
      params.rotation as number ?? 0,
      0, 0, // padding
    ]);
  },
};
