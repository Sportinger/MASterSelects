import { describe, expect, it } from 'vitest';

import {
  clearExternalDragPayload,
  getExternalDragPayload,
  setExternalDragPayload,
} from '../../src/components/timeline/utils/externalDragSession';

describe('externalDragSession', () => {
  it('stores and clears the current drag payload', () => {
    clearExternalDragPayload();

    setExternalDragPayload({
      kind: 'media-file',
      id: 'media-1',
      duration: 12.5,
      hasAudio: true,
      isAudio: false,
      isVideo: true,
    });

    expect(getExternalDragPayload()).toEqual({
      kind: 'media-file',
      id: 'media-1',
      duration: 12.5,
      hasAudio: true,
      isAudio: false,
      isVideo: true,
    });

    clearExternalDragPayload();

    expect(getExternalDragPayload()).toBeNull();
  });
});
