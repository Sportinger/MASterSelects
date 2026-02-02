// Edge Detection Effect

import shader from './shader.wgsl?raw';
import type { EffectDefinition } from '../../types';

export const edgeDetect: EffectDefinition = {
  id: 'edge-detect',
  name: 'Edge Detect',
  category: 'stylize',

  shader,
  entryPoint: 'edgeDetectFragment',
  uniformSize: 16,

  params: {
    strength: {
      type: 'number',
      label: 'Strength',
      default: 1,
      min: 0,
      max: 5,
      step: 0.1,
      animatable: true,
    },
    invert: {
      type: 'boolean',
      label: 'Invert',
      default: false,
    },
  },

  packUniforms: (params, width, height) => {
    return new Float32Array([
      params.strength as number ?? 1,
      width,
      height,
      params.invert ? 1 : 0,
    ]);
  },
};
