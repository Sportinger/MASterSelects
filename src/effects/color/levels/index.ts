// Levels Effect

import shader from './shader.wgsl?raw';
import type { EffectDefinition } from '../../types';

export const levels: EffectDefinition = {
  id: 'levels',
  name: 'Levels',
  category: 'color',

  shader,
  entryPoint: 'levelsFragment',
  uniformSize: 32, // 8 floats

  params: {
    inputBlack: {
      type: 'number',
      label: 'Input Black',
      default: 0,
      min: 0,
      max: 1,
      step: 0.01,
      animatable: true,
    },
    inputWhite: {
      type: 'number',
      label: 'Input White',
      default: 1,
      min: 0,
      max: 1,
      step: 0.01,
      animatable: true,
    },
    gamma: {
      type: 'number',
      label: 'Gamma',
      default: 1,
      min: 0.1,
      max: 3,
      step: 0.01,
      animatable: true,
    },
    outputBlack: {
      type: 'number',
      label: 'Output Black',
      default: 0,
      min: 0,
      max: 1,
      step: 0.01,
      animatable: true,
    },
    outputWhite: {
      type: 'number',
      label: 'Output White',
      default: 1,
      min: 0,
      max: 1,
      step: 0.01,
      animatable: true,
    },
  },

  packUniforms: (params) => {
    return new Float32Array([
      params.inputBlack as number ?? 0,
      params.inputWhite as number ?? 1,
      params.gamma as number ?? 1,
      params.outputBlack as number ?? 0,
      params.outputWhite as number ?? 1,
      0, 0, 0, // padding
    ]);
  },
};
