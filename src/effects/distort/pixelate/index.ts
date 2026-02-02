// Pixelate Effect

import shader from './shader.wgsl?raw';
import type { EffectDefinition } from '../../types';

export const pixelate: EffectDefinition = {
  id: 'pixelate',
  name: 'Pixelate',
  category: 'distort',

  shader,
  entryPoint: 'pixelateFragment',
  uniformSize: 16,

  params: {
    size: {
      type: 'number',
      label: 'Pixel Size',
      default: 8,
      min: 1,
      max: 64,
      step: 1,
      animatable: true,
    },
  },

  packUniforms: (params, width, height) => {
    return new Float32Array([
      params.size as number ?? 8,
      width,
      height,
      0, // padding
    ]);
  },
};
