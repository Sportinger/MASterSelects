export interface PopupPlacementBounds {
  screenX: number;
  screenY: number;
  outerWidth: number;
  outerHeight: number;
}

export interface PopupPlacement {
  left: number;
  top: number;
}

const HORIZONTAL_MARGIN = 24;
const VERTICAL_MARGIN = 48;
const CENTER_EXCLUSION_RATIO = 0.18;
const MAX_ATTEMPTS = 24;

function randomInt(min: number, max: number, rng: () => number): number {
  if (max <= min) return min;
  return Math.round(min + rng() * (max - min));
}

function isOutsideCenterZone(
  left: number,
  top: number,
  popupWidth: number,
  popupHeight: number,
  bounds: PopupPlacementBounds,
): boolean {
  const popupCenterX = left + popupWidth / 2;
  const popupCenterY = top + popupHeight / 2;
  const boundsCenterX = bounds.screenX + bounds.outerWidth / 2;
  const boundsCenterY = bounds.screenY + bounds.outerHeight / 2;
  const exclusionHalfWidth = Math.max(80, bounds.outerWidth * CENTER_EXCLUSION_RATIO);
  const exclusionHalfHeight = Math.max(60, bounds.outerHeight * CENTER_EXCLUSION_RATIO);

  return (
    Math.abs(popupCenterX - boundsCenterX) > exclusionHalfWidth ||
    Math.abs(popupCenterY - boundsCenterY) > exclusionHalfHeight
  );
}

export function getRandomPopupPlacement(
  bounds: PopupPlacementBounds,
  popupWidth: number,
  popupHeight: number,
  rng: () => number = Math.random,
): PopupPlacement {
  const maxLeft = Math.round(bounds.screenX + Math.max(HORIZONTAL_MARGIN, bounds.outerWidth - popupWidth - HORIZONTAL_MARGIN));
  const minLeft = Math.round(bounds.screenX + HORIZONTAL_MARGIN);
  const maxTop = Math.round(bounds.screenY + Math.max(VERTICAL_MARGIN, bounds.outerHeight - popupHeight - VERTICAL_MARGIN));
  const minTop = Math.round(bounds.screenY + VERTICAL_MARGIN);

  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt += 1) {
    const left = randomInt(minLeft, maxLeft, rng);
    const top = randomInt(minTop, maxTop, rng);
    if (isOutsideCenterZone(left, top, popupWidth, popupHeight, bounds)) {
      return { left, top };
    }
  }

  const randomEdgeLeft = randomInt(minLeft, maxLeft, rng);
  const randomEdgeTop = randomInt(minTop, maxTop, rng);
  const fallbacks: PopupPlacement[] = [
    { left: minLeft, top: minTop },
    { left: maxLeft, top: minTop },
    { left: minLeft, top: maxTop },
    { left: maxLeft, top: maxTop },
    { left: minLeft, top: randomEdgeTop },
    { left: maxLeft, top: randomEdgeTop },
    { left: randomEdgeLeft, top: minTop },
    { left: randomEdgeLeft, top: maxTop },
  ];

  const validFallbacks = fallbacks.filter((placement) =>
    isOutsideCenterZone(placement.left, placement.top, popupWidth, popupHeight, bounds),
  );

  return validFallbacks[Math.floor(rng() * validFallbacks.length)] ?? { left: minLeft, top: minTop };
}
