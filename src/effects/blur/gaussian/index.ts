// Gaussian Blur Effect - High Quality

import shader from './shader.wgsl?raw';
import type { EffectDefinition } from '../../types';

export const gaussianBlur: EffectDefinition = {
  id: 'gaussian-blur',
  name: 'Gaussian Blur',
  category: 'blur',

  shader,
  entryPoint: 'gaussianBlurFragment',
  uniformSize: 16,

  params: {
    radius: {
      type: 'number',
      label: 'Radius',
      default: 10,
      min: 0,
      max: 50,
      step: 1,
      animatable: true,
    },
    samples: {
      type: 'number',
      label: 'Samples',
      default: 5,
      min: 1,
      max: 64,
      step: 1,
      animatable: false,
      quality: true, // Marks as quality parameter
    },
  },

  packUniforms: (params, width, height) => {
    return new Float32Array([
      params.radius as number || 10,
      width,
      height,
      params.samples as number || 5,
    ]);
  },
};
