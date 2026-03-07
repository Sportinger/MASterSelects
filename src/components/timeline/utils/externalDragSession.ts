export interface ExternalDragPayload {
  kind: 'media-file' | 'composition' | 'text' | 'solid';
  id: string;
  duration?: number;
  hasAudio?: boolean;
  isAudio: boolean;
  isVideo: boolean;
  file?: File;
}

let currentExternalDragPayload: ExternalDragPayload | null = null;

export function setExternalDragPayload(payload: ExternalDragPayload | null): void {
  currentExternalDragPayload = payload;
}

export function getExternalDragPayload(): ExternalDragPayload | null {
  return currentExternalDragPayload;
}

export function clearExternalDragPayload(): void {
  currentExternalDragPayload = null;
}
