// Contrast Effect

import shader from './shader.wgsl?raw';
import type { EffectDefinition } from '../../types';

export const contrast: EffectDefinition = {
  id: 'contrast',
  name: 'Contrast',
  category: 'color',

  shader,
  entryPoint: 'contrastFragment',
  uniformSize: 16,

  params: {
    amount: {
      type: 'number',
      label: 'Amount',
      default: 1,
      min: 0,
      max: 3,
      step: 0.01,
      animatable: true,
    },
  },

  packUniforms: (params) => {
    return new Float32Array([
      params.amount as number ?? 1,
      0, 0, 0, // padding
    ]);
  },
};
