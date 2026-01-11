// Bulge/Pinch Effect

import shader from './shader.wgsl?raw';
import type { EffectDefinition } from '../../types';

export const bulge: EffectDefinition = {
  id: 'bulge',
  name: 'Bulge/Pinch',
  category: 'distort',

  shader,
  entryPoint: 'bulgeFragment',
  uniformSize: 16,

  params: {
    amount: {
      type: 'number',
      label: 'Amount',
      default: 0.5,
      min: 0.1,
      max: 3,
      step: 0.1,
      animatable: true,
    },
    radius: {
      type: 'number',
      label: 'Radius',
      default: 0.5,
      min: 0.1,
      max: 1,
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
  },

  packUniforms: (params) => {
    return new Float32Array([
      params.amount as number || 0.5,
      params.radius as number || 0.5,
      params.centerX as number || 0.5,
      params.centerY as number || 0.5,
    ]);
  },
};
