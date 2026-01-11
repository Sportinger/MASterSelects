// Motion Blur Effect

import shader from './shader.wgsl?raw';
import type { EffectDefinition } from '../../types';

export const motionBlur: EffectDefinition = {
  id: 'motion-blur',
  name: 'Motion Blur',
  category: 'blur',

  shader,
  entryPoint: 'motionBlurFragment',
  uniformSize: 16,

  params: {
    amount: {
      type: 'number',
      label: 'Amount',
      default: 0.05,
      min: 0,
      max: 0.2,
      step: 0.001,
      animatable: true,
    },
    angle: {
      type: 'number',
      label: 'Angle',
      default: 0,
      min: 0,
      max: 6.28318,
      step: 0.01,
      animatable: true,
    },
  },

  packUniforms: (params) => {
    return new Float32Array([
      params.amount as number || 0.05,
      params.angle as number || 0,
      0, 0,
    ]);
  },
};
