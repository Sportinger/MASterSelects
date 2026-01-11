// Gaussian Blur Effect

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
      max: 100,
      step: 1,
      animatable: true,
    },
  },

  packUniforms: (params, width, height) => {
    return new Float32Array([
      params.radius as number || 10,
      width,
      height,
      0, // direction (0 = combined pass)
    ]);
  },
};
