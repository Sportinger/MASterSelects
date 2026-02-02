// Posterize Effect

import shader from './shader.wgsl?raw';
import type { EffectDefinition } from '../../types';

export const posterize: EffectDefinition = {
  id: 'posterize',
  name: 'Posterize',
  category: 'stylize',

  shader,
  entryPoint: 'posterizeFragment',
  uniformSize: 16,

  params: {
    levels: {
      type: 'number',
      label: 'Levels',
      default: 6,
      min: 2,
      max: 32,
      step: 1,
      animatable: true,
    },
  },

  packUniforms: (params) => {
    return new Float32Array([
      params.levels as number ?? 6,
      0, 0, 0,
    ]);
  },
};
