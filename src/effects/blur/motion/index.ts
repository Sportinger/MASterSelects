// Motion Blur Effect - High Quality

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
      max: 0.3,
      step: 0.005,
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
    samples: {
      type: 'number',
      label: 'Samples',
      default: 24,
      min: 4,
      max: 128,
      step: 1,
      animatable: false,
      quality: true,
    },
  },

  packUniforms: (params) => {
    return new Float32Array([
      params.amount as number || 0.05,
      params.angle as number || 0,
      params.samples as number || 24,
      0,
    ]);
  },
};
