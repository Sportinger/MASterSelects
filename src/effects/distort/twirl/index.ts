// Twirl Effect

import shader from './shader.wgsl?raw';
import type { EffectDefinition } from '../../types';

export const twirl: EffectDefinition = {
  id: 'twirl',
  name: 'Twirl',
  category: 'distort',

  shader,
  entryPoint: 'twirlFragment',
  uniformSize: 16,

  params: {
    amount: {
      type: 'number',
      label: 'Amount',
      default: 1,
      min: -10,
      max: 10,
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
      params.amount as number ?? 1,
      params.radius as number ?? 0.5,
      params.centerX as number ?? 0.5,
      params.centerY as number ?? 0.5,
    ]);
  },
};
