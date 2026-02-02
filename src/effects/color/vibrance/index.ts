// Vibrance Effect

import shader from './shader.wgsl?raw';
import type { EffectDefinition } from '../../types';

export const vibrance: EffectDefinition = {
  id: 'vibrance',
  name: 'Vibrance',
  category: 'color',

  shader,
  entryPoint: 'vibranceFragment',
  uniformSize: 16,

  params: {
    amount: {
      type: 'number',
      label: 'Amount',
      default: 0,
      min: -1,
      max: 1,
      step: 0.01,
      animatable: true,
    },
  },

  packUniforms: (params) => {
    return new Float32Array([
      params.amount as number ?? 0,
      0, 0, 0,
    ]);
  },
};
