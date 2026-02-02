// Chroma Key Effect

import shader from './shader.wgsl?raw';
import type { EffectDefinition } from '../../types';

export const chromaKey: EffectDefinition = {
  id: 'chroma-key',
  name: 'Chroma Key',
  category: 'keying',

  shader,
  entryPoint: 'chromaKeyFragment',
  uniformSize: 32, // 8 floats

  params: {
    keyColor: {
      type: 'select',
      label: 'Key Color',
      default: 'green',
      options: [
        { value: 'green', label: 'Green Screen' },
        { value: 'blue', label: 'Blue Screen' },
        { value: 'custom', label: 'Custom' },
      ],
    },
    tolerance: {
      type: 'number',
      label: 'Tolerance',
      default: 0.2,
      min: 0,
      max: 1,
      step: 0.01,
      animatable: true,
    },
    softness: {
      type: 'number',
      label: 'Edge Softness',
      default: 0.1,
      min: 0,
      max: 0.5,
      step: 0.01,
      animatable: true,
    },
    spillSuppression: {
      type: 'number',
      label: 'Spill Suppression',
      default: 0.5,
      min: 0,
      max: 1,
      step: 0.01,
      animatable: true,
    },
  },

  packUniforms: (params) => {
    // Get key color RGB values based on preset
    let keyR = 0, keyG = 1, keyB = 0; // Default green

    const keyColor = params.keyColor as string;
    if (keyColor === 'blue') {
      keyR = 0; keyG = 0; keyB = 1;
    } else if (keyColor === 'green') {
      keyR = 0; keyG = 1; keyB = 0;
    }

    return new Float32Array([
      keyR,
      keyG,
      keyB,
      params.tolerance as number ?? 0.2,
      params.softness as number ?? 0.1,
      params.spillSuppression as number ?? 0.5,
      0, 0, // padding
    ]);
  },
};
