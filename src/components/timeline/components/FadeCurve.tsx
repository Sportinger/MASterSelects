// FadeCurve - Renders SVG bezier curve showing opacity fade
// Note: Not using memo() here to ensure re-render on keyframe changes

export function FadeCurve({
  keyframes,
  clipDuration,
  width,
  height,
}: {
  keyframes: Array<{
    time: number;
    value: number;
    easing: string;
    handleIn?: { x: number; y: number };
    handleOut?: { x: number; y: number };
  }>;
  clipDuration: number;
  width: number;
  height: number;
}) {
  if (keyframes.length < 2 || width <= 0 || height <= 0) return null;

  // Sort keyframes by time
  const sorted = [...keyframes].sort((a, b) => a.time - b.time);

  // Build SVG path
  const timeToX = (t: number) => (t / clipDuration) * width;
  const valueToY = (v: number) => height - v * height; // Invert Y (0 at bottom, 1 at top)

  // Generate path segments between keyframes
  const pathSegments: string[] = [];

  // Start from the first keyframe
  const firstKf = sorted[0];
  pathSegments.push(`M ${timeToX(firstKf.time)} ${valueToY(firstKf.value)}`);

  for (let i = 0; i < sorted.length - 1; i++) {
    const kf1 = sorted[i];
    const kf2 = sorted[i + 1];

    const x1 = timeToX(kf1.time);
    const y1 = valueToY(kf1.value);
    const x2 = timeToX(kf2.time);
    const y2 = valueToY(kf2.value);

    const duration = kf2.time - kf1.time;

    // Determine control points based on easing type
    let cp1x: number, cp1y: number, cp2x: number, cp2y: number;

    if (kf1.easing === 'bezier' && kf1.handleOut && kf2.handleIn) {
      // Custom bezier handles
      cp1x = timeToX(kf1.time + kf1.handleOut.x);
      cp1y = valueToY(kf1.value + kf1.handleOut.y);
      cp2x = timeToX(kf2.time + kf2.handleIn.x);
      cp2y = valueToY(kf2.value + kf2.handleIn.y);
    } else {
      // Standard easing curves (cubic-bezier approximations)
      switch (kf1.easing) {
        case 'ease-in':
          cp1x = x1 + duration * 0.42 * (width / clipDuration);
          cp1y = y1;
          cp2x = x2;
          cp2y = y2;
          break;
        case 'ease-out':
          cp1x = x1;
          cp1y = y1;
          cp2x = x1 + duration * 0.58 * (width / clipDuration);
          cp2y = y2;
          break;
        case 'ease-in-out':
          cp1x = x1 + duration * 0.42 * (width / clipDuration);
          cp1y = y1;
          cp2x = x1 + duration * 0.58 * (width / clipDuration);
          cp2y = y2;
          break;
        case 'linear':
        default:
          cp1x = x1 + (x2 - x1) / 3;
          cp1y = y1 + (y2 - y1) / 3;
          cp2x = x1 + (x2 - x1) * 2 / 3;
          cp2y = y1 + (y2 - y1) * 2 / 3;
          break;
      }
    }

    pathSegments.push(`C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${x2} ${y2}`);
  }

  const curvePath = pathSegments.join(' ');

  // Create filled area path (curve + bottom edge)
  const lastKf = sorted[sorted.length - 1];
  const fillPath = `${curvePath} L ${timeToX(lastKf.time)} ${height} L ${timeToX(firstKf.time)} ${height} Z`;

  return (
    <svg
      className="fade-curve-svg"
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
    >
      {/* Filled area under curve */}
      <path
        d={fillPath}
        fill="rgba(0, 0, 0, 0.4)"
        stroke="none"
      />
      {/* Curve line */}
      <path
        d={curvePath}
        fill="none"
        stroke="rgba(140, 180, 220, 0.8)"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* Keyframe dots */}
      {sorted.map((kf, i) => (
        <circle
          key={i}
          cx={timeToX(kf.time)}
          cy={valueToY(kf.value)}
          r="3"
          fill="rgba(140, 180, 220, 1)"
        />
      ))}
    </svg>
  );
}
