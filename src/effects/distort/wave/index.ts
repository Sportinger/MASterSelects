// Wave Distortion Effect

import shader from './shader.wgsl?raw';
import type { EffectDefinition } from '../../types';

export const wave: EffectDefinition = {
  id: 'wave',
  name: 'Wave',
  category: 'distort',

  shader,
  entryPoint: 'waveFragment',
  uniformSize: 16,

  params: {
    amplitudeX: {
      type: 'number',
      label: 'Horizontal Amp',
      default: 0.02,
      min: 0,
      max: 0.1,
      step: 0.001,
      animatable: true,
    },
    amplitudeY: {
      type: 'number',
      label: 'Vertical Amp',
      default: 0.02,
      min: 0,
      max: 0.1,
      step: 0.001,
      animatable: true,
    },
    frequencyX: {
      type: 'number',
      label: 'Horizontal Freq',
      default: 5,
      min: 1,
      max: 20,
      step: 0.5,
      animatable: true,
    },
    frequencyY: {
      type: 'number',
      label: 'Vertical Freq',
      default: 5,
      min: 1,
      max: 20,
      step: 0.5,
      animatable: true,
    },
  },

  packUniforms: (params) => {
    return new Float32Array([
      params.amplitudeX as number ?? 0.02,
      params.amplitudeY as number ?? 0.02,
      params.frequencyX as number ?? 5,
      params.frequencyY as number ?? 5,
    ]);
  },
};
