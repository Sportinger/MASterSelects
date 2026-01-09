// Transform composition utility for parent-child clip relationships
// Composes parent and child transforms like After Effects parenting

import type { ClipTransform } from '../types';

/**
 * Composes parent and child transforms.
 * - Position: Child position is relative to parent, rotated by parent rotation
 * - Scale: Child scale is multiplied by parent scale
 * - Rotation: Child rotation is added to parent rotation
 * - Opacity: Child opacity is multiplied by parent opacity
 */
export function composeTransforms(
  parent: ClipTransform,
  child: ClipTransform
): ClipTransform {
  // Convert parent Z rotation to radians
  const parentRotZ = (parent.rotation.z * Math.PI) / 180;

  // Rotate child position by parent's Z rotation
  const rotatedX = child.position.x * Math.cos(parentRotZ) - child.position.y * Math.sin(parentRotZ);
  const rotatedY = child.position.x * Math.sin(parentRotZ) + child.position.y * Math.cos(parentRotZ);

  return {
    // Multiply opacities
    opacity: parent.opacity * child.opacity,

    // Child's blend mode takes precedence
    blendMode: child.blendMode,

    // Position: Parent position + rotated child position (scaled by parent scale)
    position: {
      x: parent.position.x + rotatedX * parent.scale.x,
      y: parent.position.y + rotatedY * parent.scale.y,
      z: parent.position.z + child.position.z,
    },

    // Scale: Multiply parent and child scales
    scale: {
      x: parent.scale.x * child.scale.x,
      y: parent.scale.y * child.scale.y,
    },

    // Rotation: Add parent and child rotations
    rotation: {
      x: parent.rotation.x + child.rotation.x,
      y: parent.rotation.y + child.rotation.y,
      z: parent.rotation.z + child.rotation.z,
    },
  };
}

/**
 * Checks if setting parentId as parent of clipId would create a cycle.
 * Returns true if it would create a cycle (invalid), false if safe.
 */
export function wouldCreateCycle(
  clipId: string,
  parentId: string,
  getParentId: (id: string) => string | undefined
): boolean {
  let currentId: string | undefined = parentId;

  // Walk up the parent chain
  while (currentId) {
    if (currentId === clipId) {
      // Found the clip in the parent chain - would create cycle
      return true;
    }
    currentId = getParentId(currentId);
  }

  return false;
}
