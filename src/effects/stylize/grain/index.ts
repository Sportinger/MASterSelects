// Film Grain Effect

import shader from './shader.wgsl?raw';
import type { EffectDefinition } from '../../types';

export const grain: EffectDefinition = {
  id: 'grain',
  name: 'Film Grain',
  category: 'stylize',

  shader,
  entryPoint: 'grainFragment',
  uniformSize: 16,

  params: {
    amount: {
      type: 'number',
      label: 'Amount',
      default: 0.1,
      min: 0,
      max: 0.5,
      step: 0.01,
      animatable: true,
    },
    size: {
      type: 'number',
      label: 'Size',
      default: 1,
      min: 0.5,
      max: 5,
      step: 0.1,
      animatable: true,
    },
    speed: {
      type: 'number',
      label: 'Speed',
      default: 1,
      min: 0,
      max: 5,
      step: 0.1,
      animatable: false,
    },
  },

  packUniforms: (params) => {
    // Use current time for animation
    const time = performance.now() / 1000;
    return new Float32Array([
      params.amount as number ?? 0.1,
      params.size as number ?? 1,
      params.speed as number ?? 1,
      time,
    ]);
  },
};
