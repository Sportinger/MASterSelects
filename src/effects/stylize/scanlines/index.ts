// CRT Scanlines Effect

import shader from './shader.wgsl?raw';
import type { EffectDefinition } from '../../types';

export const scanlines: EffectDefinition = {
  id: 'scanlines',
  name: 'Scanlines',
  category: 'stylize',

  shader,
  entryPoint: 'scanlinesFragment',
  uniformSize: 16,

  params: {
    density: {
      type: 'number',
      label: 'Density',
      default: 5,
      min: 1,
      max: 20,
      step: 0.5,
      animatable: true,
    },
    opacity: {
      type: 'number',
      label: 'Opacity',
      default: 0.3,
      min: 0,
      max: 1,
      step: 0.01,
      animatable: true,
    },
    speed: {
      type: 'number',
      label: 'Scroll Speed',
      default: 0,
      min: 0,
      max: 5,
      step: 0.1,
      animatable: false,
    },
  },

  packUniforms: (params) => {
    const time = performance.now() / 1000;
    return new Float32Array([
      params.density as number ?? 5,
      params.opacity as number ?? 0.3,
      params.speed as number ?? 0,
      time,
    ]);
  },
};
