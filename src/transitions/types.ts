// Transition Types
// Defines the structure for timeline transitions between clips

export type TransitionType = 'crossfade' | 'dip-to-black' | 'dip-to-white' | 'wipe-left' | 'wipe-right';

export type TransitionCategory = 'dissolve' | 'wipe' | 'slide' | 'zoom';

/**
 * Transition definition - each transition type implements this interface
 */
export interface TransitionDefinition {
  /** Unique identifier for this transition type */
  id: TransitionType;

  /** Display name shown in UI */
  name: string;

  /** Category for grouping in panel */
  category: TransitionCategory;

  /** Default duration in seconds */
  defaultDuration: number;

  /** Minimum duration in seconds */
  minDuration: number;

  /** Maximum duration in seconds */
  maxDuration: number;

  /** Description shown in tooltip */
  description: string;
}

/**
 * Transition instance stored on a clip
 */
export interface ClipTransition {
  /** Unique ID for this transition instance */
  id: string;
  /** Type of transition */
  type: TransitionType;
  /** Duration in seconds */
  duration: number;
  /** ID of the other clip in the transition */
  linkedClipId: string;
}
