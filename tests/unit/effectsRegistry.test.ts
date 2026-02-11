/**
 * Tests for the Effects Registry system.
 *
 * Imports directly from src/effects/index.ts and validates that all effects
 * are properly registered with correct structure and parameter definitions.
 */

import { describe, it, expect } from 'vitest';
import {
  EFFECT_REGISTRY,
  EFFECT_CATEGORIES,
  getEffect,
  getDefaultParams,
  getAllEffects,
  getEffectsByCategory,
  getCategoriesWithEffects,
  hasEffect,
  getEffectConfig,
} from '../../src/effects/index';
import type { EffectDefinition, EffectCategory, EffectParam } from '../../src/effects/types';

// ---- Category registration -------------------------------------------------

describe('Effect category registration', () => {
  const expectedPopulatedCategories: EffectCategory[] = ['color', 'blur', 'distort', 'stylize', 'keying'];
  const expectedEmptyCategories: EffectCategory[] = ['generate', 'time', 'transition'];

  it('should have all eight categories defined', () => {
    const allCategories: EffectCategory[] = ['color', 'blur', 'distort', 'stylize', 'generate', 'keying', 'time', 'transition'];
    for (const cat of allCategories) {
      expect(EFFECT_CATEGORIES).toHaveProperty(cat);
      expect(Array.isArray(EFFECT_CATEGORIES[cat])).toBe(true);
    }
  });

  it('should have effects registered in populated categories', () => {
    for (const cat of expectedPopulatedCategories) {
      expect(EFFECT_CATEGORIES[cat].length).toBeGreaterThan(0);
    }
  });

  it('should have empty arrays for categories with no effects yet', () => {
    for (const cat of expectedEmptyCategories) {
      expect(EFFECT_CATEGORIES[cat]).toHaveLength(0);
    }
  });

  it('getCategoriesWithEffects should only return non-empty categories', () => {
    const populated = getCategoriesWithEffects();
    const categoryNames = populated.map(c => c.category);

    for (const cat of expectedPopulatedCategories) {
      expect(categoryNames).toContain(cat);
    }
    for (const cat of expectedEmptyCategories) {
      expect(categoryNames).not.toContain(cat);
    }
  });

  it('getEffectsByCategory should return effects for a given category', () => {
    const colorEffects = getEffectsByCategory('color');
    expect(colorEffects.length).toBeGreaterThan(0);
    for (const effect of colorEffects) {
      expect(effect.category).toBe('color');
    }
  });

  it('getEffectsByCategory should return empty array for empty category', () => {
    const timeEffects = getEffectsByCategory('time');
    expect(timeEffects).toHaveLength(0);
  });
});

// ---- Expected effects per category -----------------------------------------

describe('Expected effects per category', () => {
  const expectedColorEffects = [
    'brightness', 'contrast', 'saturation', 'hue-shift',
    'levels', 'invert', 'vibrance', 'temperature', 'exposure',
  ];

  const expectedBlurEffects = [
    'gaussian-blur', 'box-blur', 'radial-blur', 'zoom-blur', 'motion-blur',
  ];

  const expectedDistortEffects = [
    'pixelate', 'kaleidoscope', 'mirror', 'rgb-split', 'twirl', 'wave', 'bulge',
  ];

  const expectedStylizeEffects = [
    'vignette', 'grain', 'sharpen', 'posterize', 'glow', 'edge-detect', 'scanlines', 'threshold',
  ];

  it('should register all color effects', () => {
    for (const id of expectedColorEffects) {
      expect(hasEffect(id)).toBe(true);
    }
  });

  it('should register all blur effects', () => {
    for (const id of expectedBlurEffects) {
      expect(hasEffect(id)).toBe(true);
    }
  });

  it('should register all distort effects', () => {
    for (const id of expectedDistortEffects) {
      expect(hasEffect(id)).toBe(true);
    }
  });

  it('should register all stylize effects', () => {
    for (const id of expectedStylizeEffects) {
      expect(hasEffect(id)).toBe(true);
    }
  });

  it('should register chroma-key in keying category', () => {
    expect(hasEffect('chroma-key')).toBe(true);
    const chromaKey = getEffect('chroma-key')!;
    expect(chromaKey.category).toBe('keying');
  });
});

// ---- Effect structure validation -------------------------------------------

describe('Effect required properties', () => {
  it('every effect should have id, name, category, shader, entryPoint, params, packUniforms', () => {
    const allEffects = getAllEffects();
    expect(allEffects.length).toBeGreaterThan(0);

    for (const effect of allEffects) {
      expect(typeof effect.id).toBe('string');
      expect(effect.id.length).toBeGreaterThan(0);

      expect(typeof effect.name).toBe('string');
      expect(effect.name.length).toBeGreaterThan(0);

      expect(typeof effect.category).toBe('string');

      expect(typeof effect.shader).toBe('string');
      expect(effect.shader.length).toBeGreaterThan(0);

      expect(typeof effect.entryPoint).toBe('string');
      expect(effect.entryPoint.length).toBeGreaterThan(0);

      expect(typeof effect.params).toBe('object');
      expect(effect.params).not.toBeNull();

      expect(typeof effect.packUniforms).toBe('function');
    }
  });

  it('every effect should have a non-negative uniformSize', () => {
    for (const effect of getAllEffects()) {
      expect(typeof effect.uniformSize).toBe('number');
      expect(effect.uniformSize).toBeGreaterThanOrEqual(0);
    }
  });

  it('every effect uniformSize should be 16-byte aligned', () => {
    for (const effect of getAllEffects()) {
      expect(effect.uniformSize % 16).toBe(0);
    }
  });
});

// ---- No duplicate effect IDs -----------------------------------------------

describe('No duplicate effect IDs', () => {
  it('should have unique IDs across all effects', () => {
    const allEffects = getAllEffects();
    const ids = allEffects.map(e => e.id);
    const uniqueIds = new Set(ids);

    expect(uniqueIds.size).toBe(ids.length);
  });

  it('registry size should match getAllEffects length', () => {
    expect(EFFECT_REGISTRY.size).toBe(getAllEffects().length);
  });
});

// ---- Parameter validation --------------------------------------------------

describe('Effect parameter validation', () => {
  it('every parameter should have type, label, and default', () => {
    for (const effect of getAllEffects()) {
      for (const [paramKey, param] of Object.entries(effect.params)) {
        expect(typeof param.type).toBe('string');
        expect(['number', 'boolean', 'select', 'color', 'point']).toContain(param.type);

        expect(typeof param.label).toBe('string');
        expect(param.label.length).toBeGreaterThan(0);

        expect(param.default).toBeDefined();
      }
    }
  });

  it('number parameters should have min, max, step with min <= default <= max', () => {
    for (const effect of getAllEffects()) {
      for (const [paramKey, param] of Object.entries(effect.params)) {
        if (param.type !== 'number') continue;

        expect(typeof param.min).toBe('number');
        expect(typeof param.max).toBe('number');
        expect(typeof param.step).toBe('number');

        expect(param.min!).toBeLessThanOrEqual(param.max!);
        expect(param.default as number).toBeGreaterThanOrEqual(param.min!);
        expect(param.default as number).toBeLessThanOrEqual(param.max!);
        expect(param.step!).toBeGreaterThan(0);
      }
    }
  });

  it('select parameters should have options array with value and label', () => {
    for (const effect of getAllEffects()) {
      for (const [paramKey, param] of Object.entries(effect.params)) {
        if (param.type !== 'select') continue;

        expect(Array.isArray(param.options)).toBe(true);
        expect(param.options!.length).toBeGreaterThan(0);

        for (const option of param.options!) {
          expect(typeof option.value).toBe('string');
          expect(typeof option.label).toBe('string');
        }

        // Default should be one of the option values
        const optionValues = param.options!.map(o => o.value);
        expect(optionValues).toContain(param.default);
      }
    }
  });

  it('boolean parameters should have boolean default', () => {
    for (const effect of getAllEffects()) {
      for (const [paramKey, param] of Object.entries(effect.params)) {
        if (param.type !== 'boolean') continue;
        expect(typeof param.default).toBe('boolean');
      }
    }
  });
});

// ---- Helper functions ------------------------------------------------------

describe('Registry helper functions', () => {
  it('getEffect should return a valid definition for known effect', () => {
    const effect = getEffect('brightness');
    expect(effect).toBeDefined();
    expect(effect!.id).toBe('brightness');
    expect(effect!.name).toBe('Brightness');
    expect(effect!.category).toBe('color');
  });

  it('getEffect should return undefined for unknown effect', () => {
    expect(getEffect('nonexistent-effect')).toBeUndefined();
  });

  it('hasEffect should return true for registered effects', () => {
    expect(hasEffect('gaussian-blur')).toBe(true);
    expect(hasEffect('brightness')).toBe(true);
    expect(hasEffect('chroma-key')).toBe(true);
  });

  it('hasEffect should return false for unregistered effects', () => {
    expect(hasEffect('fake-effect')).toBe(false);
    expect(hasEffect('')).toBe(false);
  });

  it('getDefaultParams should return correct defaults', () => {
    const defaults = getDefaultParams('brightness');
    expect(defaults).toHaveProperty('amount');
    expect(defaults.amount).toBe(0);
  });

  it('getDefaultParams should return empty object for unknown effect', () => {
    const defaults = getDefaultParams('nonexistent');
    expect(defaults).toEqual({});
  });

  it('getEffectConfig should return pipeline config for known effects', () => {
    const config = getEffectConfig('gaussian-blur');
    expect(config).toBeDefined();
    expect(config!.entryPoint).toBe('gaussianBlurFragment');
    expect(config!.uniformSize).toBe(16);
    expect(config!.needsUniform).toBe(true);
  });

  it('getEffectConfig should return undefined for unknown effects', () => {
    expect(getEffectConfig('nonexistent')).toBeUndefined();
  });
});

// ---- packUniforms ----------------------------------------------------------

describe('packUniforms function', () => {
  it('should return Float32Array for brightness with default params', () => {
    const effect = getEffect('brightness')!;
    const defaults = getDefaultParams('brightness');
    const uniforms = effect.packUniforms(defaults, 1920, 1080);

    expect(uniforms).toBeInstanceOf(Float32Array);
    expect(uniforms!.length).toBeGreaterThan(0);
  });

  it('should return Float32Array for gaussian-blur with custom params', () => {
    const effect = getEffect('gaussian-blur')!;
    const uniforms = effect.packUniforms({ radius: 20, samples: 10 }, 1920, 1080);

    expect(uniforms).toBeInstanceOf(Float32Array);
    // First element should be the radius value
    expect(uniforms![0]).toBe(20);
  });

  it('every effect packUniforms should not throw with default params', () => {
    for (const effect of getAllEffects()) {
      const defaults = getDefaultParams(effect.id);
      expect(() => {
        effect.packUniforms(defaults, 1920, 1080);
      }).not.toThrow();
    }
  });

  it('every effect packUniforms should return Float32Array or null', () => {
    for (const effect of getAllEffects()) {
      const defaults = getDefaultParams(effect.id);
      const result = effect.packUniforms(defaults, 1920, 1080);
      expect(result === null || result instanceof Float32Array).toBe(true);
    }
  });
});
