// Radial Blur Effect - High Quality

import shader from './shader.wgsl?raw';
import type { EffectDefinition } from '../../types';

export const radialBlur: EffectDefinition = {
  id: 'radial-blur',
  name: 'Radial Blur',
  category: 'blur',

  shader,
  entryPoint: 'radialBlurFragment',
  uniformSize: 16,

  params: {
    amount: {
      type: 'number',
      label: 'Amount',
      default: 0.5,
      min: 0,
      max: 2,
      step: 0.01,
      animatable: true,
    },
    centerX: {
      type: 'number',
      label: 'Center X',
      default: 0.5,
      min: 0,
      max: 1,
      step: 0.01,
      animatable: true,
    },
    centerY: {
      type: 'number',
      label: 'Center Y',
      default: 0.5,
      min: 0,
      max: 1,
      step: 0.01,
      animatable: true,
    },
    samples: {
      type: 'number',
      label: 'Samples',
      default: 32,
      min: 4,
      max: 256,
      step: 1,
      animatable: false,
      quality: true,
    },
  },

  packUniforms: (params) => {
    return new Float32Array([
      params.amount as number || 0.5,
      params.centerX as number || 0.5,
      params.centerY as number || 0.5,
      params.samples as number || 32,
    ]);
  },
};
