// Box Blur Effect

import shader from './shader.wgsl?raw';
import type { EffectDefinition } from '../../types';

export const boxBlur: EffectDefinition = {
  id: 'box-blur',
  name: 'Box Blur',
  category: 'blur',

  shader,
  entryPoint: 'boxBlurFragment',
  uniformSize: 16,

  params: {
    radius: {
      type: 'number',
      label: 'Radius',
      default: 5,
      min: 0,
      max: 20,
      step: 1,
      animatable: true,
    },
  },

  packUniforms: (params, width, height) => {
    return new Float32Array([
      params.radius as number ?? 5,
      width,
      height,
      0,
    ]);
  },
};
