// RGB Split Effect

import shader from './shader.wgsl?raw';
import type { EffectDefinition } from '../../types';

export const rgbSplit: EffectDefinition = {
  id: 'rgb-split',
  name: 'RGB Split',
  category: 'distort',

  shader,
  entryPoint: 'rgbSplitFragment',
  uniformSize: 16,

  params: {
    amount: {
      type: 'number',
      label: 'Amount',
      default: 0.01,
      min: 0,
      max: 0.1,
      step: 0.001,
      animatable: true,
    },
    angle: {
      type: 'number',
      label: 'Angle',
      default: 0,
      min: 0,
      max: 6.28318, // TAU
      step: 0.01,
      animatable: true,
    },
  },

  packUniforms: (params) => {
    return new Float32Array([
      params.amount as number || 0.01,
      params.angle as number || 0,
      0, 0, // padding
    ]);
  },
};
