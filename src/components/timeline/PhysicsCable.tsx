// PhysicsCable component - Reusable physics-based cable (like Reason's cables)
// Used for both established parent-child links and drag previews

import { useRef, useEffect, useState } from 'react';

interface PhysicsCableProps {
  startX: number;
  startY: number;
  endX: number;
  endY: number;
  isPreview?: boolean; // Lighter style for drag preview
}

// Physics constants for cable simulation
const SPRING_STIFFNESS = 0.15;
const DAMPING = 0.75;
const GRAVITY = 0.5;
const MAX_SAG = 60;

interface PhysicsState {
  cx: number;
  cy: number;
  vx: number;
  vy: number;
  prevStartX: number;
  prevStartY: number;
  prevEndX: number;
  prevEndY: number;
}

export function PhysicsCable({
  startX,
  startY,
  endX,
  endY,
  isPreview = false,
}: PhysicsCableProps) {
  const physicsRef = useRef<PhysicsState | null>(null);
  const animationRef = useRef<number | null>(null);
  const [controlPoint, setControlPoint] = useState({ x: 0, y: 0 });

  // Initialize physics state
  if (!physicsRef.current) {
    const midX = (startX + endX) / 2;
    const midY = (startY + endY) / 2;
    physicsRef.current = {
      cx: midX,
      cy: midY + 20,
      vx: 0,
      vy: 0,
      prevStartX: startX,
      prevStartY: startY,
      prevEndX: endX,
      prevEndY: endY,
    };
    setControlPoint({ x: midX, y: midY + 20 });
  }

  useEffect(() => {
    const physics = physicsRef.current;
    if (!physics) return;

    const simulate = () => {
      const targetX = (startX + endX) / 2;
      const targetY = (startY + endY) / 2;

      // Endpoint velocities
      const startVelX = startX - physics.prevStartX;
      const startVelY = startY - physics.prevStartY;
      const endVelX = endX - physics.prevEndX;
      const endVelY = endY - physics.prevEndY;

      const endpointVelX = (startVelX + endVelX) / 2;
      const endpointVelY = (startVelY + endVelY) / 2;

      const dx = targetX - physics.cx;

      // Natural sag based on distance
      const distance = Math.sqrt(Math.pow(endX - startX, 2) + Math.pow(endY - startY, 2));
      const naturalSag = Math.min(MAX_SAG, distance * 0.15 + 15);

      const sagTargetY = targetY + naturalSag;
      const dySag = sagTargetY - physics.cy;

      // Spring force
      physics.vx += dx * SPRING_STIFFNESS;
      physics.vy += dySag * SPRING_STIFFNESS;

      // Endpoint movement influence
      physics.vx += endpointVelX * 0.3;
      physics.vy += endpointVelY * 0.3;

      // Gravity
      physics.vy += GRAVITY;

      // Damping
      physics.vx *= DAMPING;
      physics.vy *= DAMPING;

      // Update position
      physics.cx += physics.vx;
      physics.cy += physics.vy;

      // Constrain
      const minY = Math.min(startY, endY) - 10;
      const maxY = Math.max(startY, endY) + MAX_SAG + 20;
      physics.cy = Math.max(minY, Math.min(maxY, physics.cy));

      // Store previous
      physics.prevStartX = startX;
      physics.prevStartY = startY;
      physics.prevEndX = endX;
      physics.prevEndY = endY;

      setControlPoint({ x: physics.cx, y: physics.cy });

      const totalVelocity = Math.abs(physics.vx) + Math.abs(physics.vy);
      const distanceFromTarget = Math.abs(dx) + Math.abs(dySag - naturalSag);

      if (totalVelocity > 0.1 || distanceFromTarget > 1) {
        animationRef.current = requestAnimationFrame(simulate);
      }
    };

    animationRef.current = requestAnimationFrame(simulate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [startX, startY, endX, endY]);

  const pathD = `M ${startX} ${startY} Q ${controlPoint.x} ${controlPoint.y} ${endX} ${endY}`;

  if (isPreview) {
    // Lighter preview style during drag
    return (
      <g className="physics-cable-preview">
        <path
          d={pathD}
          fill="none"
          stroke="rgba(0,0,0,0.3)"
          strokeWidth="4"
          strokeLinecap="round"
          style={{ filter: 'blur(2px)', transform: 'translate(1px, 2px)' }}
        />
        <path
          d={pathD}
          fill="none"
          stroke="#e5c07b"
          strokeWidth="3"
          strokeLinecap="round"
          style={{ filter: 'drop-shadow(0 1px 2px rgba(0,0,0,0.3))' }}
        />
        <circle
          cx={endX}
          cy={endY}
          r="8"
          fill="#e5c07b"
          stroke="#b8922f"
          strokeWidth="2"
          style={{ filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.4))' }}
        />
        <circle
          cx={startX}
          cy={startY}
          r="5"
          fill="#d4a84b"
          stroke="#b8922f"
          strokeWidth="1.5"
        />
      </g>
    );
  }

  // Full cable style for established connections
  return (
    <g className="parent-child-link-group">
      <path
        className="parent-child-link-shadow"
        d={pathD}
        fill="none"
      />
      <path
        className="parent-child-link"
        d={pathD}
        fill="none"
      />
      <circle
        cx={endX}
        cy={endY}
        r="5"
        className="parent-child-link-endpoint"
      />
      <circle
        cx={startX}
        cy={startY}
        r="5"
        className="parent-child-link-start"
      />
    </g>
  );
}
