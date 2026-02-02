// Hue Shift Effect

import shader from './shader.wgsl?raw';
import type { EffectDefinition } from '../../types';

export const hueShift: EffectDefinition = {
  id: 'hue-shift',
  name: 'Hue Shift',
  category: 'color',

  shader,
  entryPoint: 'hueShiftFragment',
  uniformSize: 16,

  params: {
    shift: {
      type: 'number',
      label: 'Shift',
      default: 0,
      min: 0,
      max: 1,
      step: 0.01,
      animatable: true,
    },
  },

  packUniforms: (params) => {
    return new Float32Array([
      params.shift as number ?? 0,
      0, 0, 0, // padding
    ]);
  },
};
