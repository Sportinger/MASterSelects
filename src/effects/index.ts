// Effect Registry System
// Auto-discovers and registers all effects from category folders

import type { EffectDefinition, EffectCategory } from './types';
export * from './types';

// Import all effects from each category
import * as colorEffects from './color';
import * as distortEffects from './distort';
import * as blurEffects from './blur';
import * as stylizeEffects from './stylize';
import * as generateEffects from './generate';
import * as keyingEffects from './keying';
import * as timeEffects from './time';
import * as transitionEffects from './transition';

// Main effect registry
export const EFFECT_REGISTRY = new Map<string, EffectDefinition>();

// Effects organized by category
export const EFFECT_CATEGORIES: Record<EffectCategory, EffectDefinition[]> = {
  color: [],
  blur: [],
  distort: [],
  stylize: [],
  generate: [],
  keying: [],
  time: [],
  transition: [],
};

/**
 * Register effects from a module export
 */
function registerEffects(effects: Record<string, unknown>) {
  Object.values(effects).forEach(effect => {
    if (isEffectDefinition(effect)) {
      EFFECT_REGISTRY.set(effect.id, effect);
      EFFECT_CATEGORIES[effect.category]?.push(effect);
    }
  });
}

/**
 * Type guard to check if an object is an EffectDefinition
 */
function isEffectDefinition(obj: unknown): obj is EffectDefinition {
  return (
    typeof obj === 'object' &&
    obj !== null &&
    'id' in obj &&
    'name' in obj &&
    'category' in obj &&
    'shader' in obj &&
    'entryPoint' in obj &&
    'params' in obj &&
    'packUniforms' in obj
  );
}

// Register all effects
registerEffects(colorEffects);
registerEffects(distortEffects);
registerEffects(blurEffects);
registerEffects(stylizeEffects);
registerEffects(generateEffects);
registerEffects(keyingEffects);
registerEffects(timeEffects);
registerEffects(transitionEffects);

// ==================== Helper Functions ====================

/**
 * Get an effect definition by ID
 */
export function getEffect(id: string): EffectDefinition | undefined {
  return EFFECT_REGISTRY.get(id);
}

/**
 * Get default parameters for an effect
 */
export function getDefaultParams(id: string): Record<string, number | boolean | string> {
  const effect = EFFECT_REGISTRY.get(id);
  if (!effect) return {};

  const defaults: Record<string, number | boolean | string> = {};
  Object.entries(effect.params).forEach(([key, param]) => {
    defaults[key] = param.default;
  });
  return defaults;
}

/**
 * Get all registered effects
 */
export function getAllEffects(): EffectDefinition[] {
  return Array.from(EFFECT_REGISTRY.values());
}

/**
 * Get effects by category
 */
export function getEffectsByCategory(category: EffectCategory): EffectDefinition[] {
  return EFFECT_CATEGORIES[category] || [];
}

/**
 * Get all non-empty categories with their effects
 */
export function getCategoriesWithEffects(): { category: EffectCategory; effects: EffectDefinition[] }[] {
  return Object.entries(EFFECT_CATEGORIES)
    .filter(([, effects]) => effects.length > 0)
    .map(([category, effects]) => ({
      category: category as EffectCategory,
      effects,
    }));
}

/**
 * Check if an effect type exists
 */
export function hasEffect(id: string): boolean {
  return EFFECT_REGISTRY.has(id);
}

/**
 * Get effect config for pipeline creation (compatibility layer)
 */
export function getEffectConfig(id: string): { entryPoint: string; needsUniform: boolean; uniformSize: number } | undefined {
  const effect = EFFECT_REGISTRY.get(id);
  if (!effect) return undefined;

  return {
    entryPoint: effect.entryPoint,
    needsUniform: effect.uniformSize > 0,
    uniformSize: effect.uniformSize,
  };
}

// Log registered effects in development
if (import.meta.env.DEV) {
  console.log(`[Effects] Registered ${EFFECT_REGISTRY.size} effects:`,
    Array.from(EFFECT_REGISTRY.keys()).join(', '));
}
