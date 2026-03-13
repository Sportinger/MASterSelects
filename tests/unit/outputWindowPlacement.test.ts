import { describe, expect, it } from 'vitest';
import { getRandomPopupPlacement } from '../../src/engine/managers/outputWindowPlacement';

describe('getRandomPopupPlacement', () => {
  it('keeps new popups out of the center zone', () => {
    const bounds = { screenX: 100, screenY: 50, outerWidth: 1600, outerHeight: 900 };
    const popupWidth = 960;
    const popupHeight = 540;
    const samples = Array.from({ length: 32 }, (_, index) =>
      getRandomPopupPlacement(bounds, popupWidth, popupHeight, () => ((index * 37) % 100) / 100),
    );

    for (const sample of samples) {
      const centerX = sample.left + popupWidth / 2;
      const centerY = sample.top + popupHeight / 2;
      const boundsCenterX = bounds.screenX + bounds.outerWidth / 2;
      const boundsCenterY = bounds.screenY + bounds.outerHeight / 2;

      expect(Math.abs(centerX - boundsCenterX) > bounds.outerWidth * 0.15 ||
        Math.abs(centerY - boundsCenterY) > bounds.outerHeight * 0.15).toBe(true);
    }
  });

  it('falls back to a valid edge position when there is barely any space', () => {
    const bounds = { screenX: 0, screenY: 0, outerWidth: 1000, outerHeight: 620 };
    const placement = getRandomPopupPlacement(bounds, 960, 540, () => 0.5);

    expect(placement.left).toBeGreaterThanOrEqual(24);
    expect(placement.top).toBeGreaterThanOrEqual(48);
    expect(placement.left).toBeLessThanOrEqual(40);
    expect(placement.top).toBeLessThanOrEqual(80);
  });
});
