// Sharpen Effect - High Quality Unsharp Mask

import shader from './shader.wgsl?raw';
import type { EffectDefinition } from '../../types';

export const sharpen: EffectDefinition = {
  id: 'sharpen',
  name: 'Sharpen',
  category: 'stylize',

  shader,
  entryPoint: 'sharpenFragment',
  uniformSize: 16,

  params: {
    amount: {
      type: 'number',
      label: 'Amount',
      default: 1,
      min: 0,
      max: 5,
      step: 0.1,
      animatable: true,
    },
    radius: {
      type: 'number',
      label: 'Radius',
      default: 1,
      min: 0.5,
      max: 5,
      step: 0.1,
      animatable: true,
    },
  },

  packUniforms: (params, width, height) => {
    return new Float32Array([
      params.amount as number ?? 1,
      params.radius as number ?? 1,
      width,
      height,
    ]);
  },
};
