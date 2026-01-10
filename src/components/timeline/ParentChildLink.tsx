// ParentChildLink component - Visual connection line between parent and child clips
// Renders a physics-based cable from child clip to parent clip (like Reason's cables)

import { useRef, useEffect, useState } from 'react';
import type { TimelineClip, TimelineTrack } from '../../types';

interface ParentChildLinkProps {
  childClip: TimelineClip;
  parentClip: TimelineClip;
  tracks: TimelineTrack[];
  zoom: number;
  scrollX: number;
  trackHeaderWidth: number;
  getTrackYPosition: (trackId: string) => number;
}

// Physics constants for cable simulation
const SPRING_STIFFNESS = 0.15;  // How quickly cable catches up
const DAMPING = 0.75;           // Energy loss (0-1, higher = more damping)
const GRAVITY = 0.5;            // Downward sag force
const MAX_SAG = 60;             // Maximum sag in pixels

interface PhysicsState {
  // Control point position (the "sag" point)
  cx: number;
  cy: number;
  // Velocity
  vx: number;
  vy: number;
  // Previous endpoint positions for velocity calculation
  prevChildX: number;
  prevChildY: number;
  prevParentX: number;
  prevParentY: number;
}

export function ParentChildLink({
  childClip,
  parentClip,
  tracks: _tracks,
  zoom,
  scrollX,
  trackHeaderWidth,
  getTrackYPosition,
}: ParentChildLinkProps) {
  // Calculate endpoint positions (both at start of clips)
  const childX = trackHeaderWidth + (childClip.startTime * zoom) - scrollX;
  const childY = getTrackYPosition(childClip.trackId);

  const parentX = trackHeaderWidth + (parentClip.startTime * zoom) - scrollX;
  const parentY = getTrackYPosition(parentClip.trackId);

  // Physics state
  const physicsRef = useRef<PhysicsState | null>(null);
  const animationRef = useRef<number | null>(null);
  const [controlPoint, setControlPoint] = useState({ x: 0, y: 0 });

  // Initialize physics state
  if (!physicsRef.current) {
    const midX = (childX + parentX) / 2;
    const midY = (childY + parentY) / 2;
    physicsRef.current = {
      cx: midX,
      cy: midY + 20, // Initial sag
      vx: 0,
      vy: 0,
      prevChildX: childX,
      prevChildY: childY,
      prevParentX: parentX,
      prevParentY: parentY,
    };
    setControlPoint({ x: midX, y: midY + 20 });
  }

  // Physics simulation
  useEffect(() => {
    const physics = physicsRef.current;
    if (!physics) return;

    const simulate = () => {
      // Target position (midpoint between endpoints)
      const targetX = (childX + parentX) / 2;
      const targetY = (childY + parentY) / 2;

      // Calculate endpoint velocities (how fast the clips are moving)
      const childVelX = childX - physics.prevChildX;
      const childVelY = childY - physics.prevChildY;
      const parentVelX = parentX - physics.prevParentX;
      const parentVelY = parentY - physics.prevParentY;

      // Average endpoint velocity affects the control point
      const endpointVelX = (childVelX + parentVelX) / 2;
      const endpointVelY = (childVelY + parentVelY) / 2;

      // Spring force towards target
      const dx = targetX - physics.cx;
      const _dy = targetY - physics.cy;

      // Calculate natural sag based on distance between endpoints
      const distance = Math.sqrt(Math.pow(parentX - childX, 2) + Math.pow(parentY - childY, 2));
      const naturalSag = Math.min(MAX_SAG, distance * 0.15 + 15);

      // Target Y includes gravity sag
      const sagTargetY = targetY + naturalSag;
      const dySag = sagTargetY - physics.cy;

      // Apply spring force
      physics.vx += dx * SPRING_STIFFNESS;
      physics.vy += dySag * SPRING_STIFFNESS;

      // Add endpoint movement influence (cable "drags" behind moving endpoints)
      physics.vx += endpointVelX * 0.3;
      physics.vy += endpointVelY * 0.3;

      // Apply gravity
      physics.vy += GRAVITY;

      // Apply damping
      physics.vx *= DAMPING;
      physics.vy *= DAMPING;

      // Update position
      physics.cx += physics.vx;
      physics.cy += physics.vy;

      // Constrain sag (don't go above the line connecting endpoints)
      const minY = Math.min(childY, parentY) - 10;
      const maxY = Math.max(childY, parentY) + MAX_SAG + 20;
      physics.cy = Math.max(minY, Math.min(maxY, physics.cy));

      // Store previous positions
      physics.prevChildX = childX;
      physics.prevChildY = childY;
      physics.prevParentX = parentX;
      physics.prevParentY = parentY;

      // Update state for render
      setControlPoint({ x: physics.cx, y: physics.cy });

      // Continue animation if there's still movement
      const totalVelocity = Math.abs(physics.vx) + Math.abs(physics.vy);
      const distanceFromTarget = Math.abs(dx) + Math.abs(dySag - naturalSag);

      if (totalVelocity > 0.1 || distanceFromTarget > 1) {
        animationRef.current = requestAnimationFrame(simulate);
      }
    };

    // Start simulation
    animationRef.current = requestAnimationFrame(simulate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [childX, childY, parentX, parentY]);

  // Don't render if both are off-screen
  if (childX < -100 && parentX < -100) return null;
  if (childX > window.innerWidth + 100 && parentX > window.innerWidth + 100) return null;

  // Create curved path through control point
  const pathD = `M ${childX} ${childY} Q ${controlPoint.x} ${controlPoint.y} ${parentX} ${parentY}`;

  return (
    <g className="parent-child-link-group">
      {/* Cable shadow for depth */}
      <path
        className="parent-child-link-shadow"
        d={pathD}
        fill="none"
      />
      {/* Main cable */}
      <path
        className="parent-child-link"
        d={pathD}
        fill="none"
      />
      {/* Connector at parent end */}
      <circle
        cx={parentX}
        cy={parentY}
        r="5"
        className="parent-child-link-endpoint"
      />
      {/* Connector at child end */}
      <circle
        cx={childX}
        cy={childY}
        r="5"
        className="parent-child-link-start"
      />
    </g>
  );
}
