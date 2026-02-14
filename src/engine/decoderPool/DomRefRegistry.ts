// DomRefRegistry — centralized ownership of DOM media elements.
// DOM elements (HTMLVideoElement, HTMLAudioElement, HTMLImageElement) live here,
// not on serialized clip state. Undo/redo re-links via this registry.

import type { DomRefRegistryInterface } from '../structuralSharing/types.ts';

class DomRefRegistryImpl implements DomRefRegistryInterface {
  private videoElements = new Map<string, HTMLVideoElement>();
  private audioElements = new Map<string, HTMLAudioElement>();
  private imageElements = new Map<string, HTMLImageElement>();
  private textCanvases = new Map<string, HTMLCanvasElement>();

  // === Getters ===

  getVideoElement(mediaFileId: string): HTMLVideoElement | undefined {
    return this.videoElements.get(mediaFileId);
  }

  getAudioElement(mediaFileId: string): HTMLAudioElement | undefined {
    return this.audioElements.get(mediaFileId);
  }

  getImageElement(mediaFileId: string): HTMLImageElement | undefined {
    return this.imageElements.get(mediaFileId);
  }

  getTextCanvas(clipId: string): HTMLCanvasElement | undefined {
    return this.textCanvases.get(clipId);
  }

  // === Registration ===

  registerVideoElement(mediaFileId: string, element: HTMLVideoElement): void {
    this.videoElements.set(mediaFileId, element);
  }

  registerAudioElement(mediaFileId: string, element: HTMLAudioElement): void {
    this.audioElements.set(mediaFileId, element);
  }

  registerImageElement(mediaFileId: string, element: HTMLImageElement): void {
    this.imageElements.set(mediaFileId, element);
  }

  registerTextCanvas(clipId: string, canvas: HTMLCanvasElement): void {
    this.textCanvases.set(clipId, canvas);
  }

  // === Cleanup ===

  unregister(mediaFileId: string): void {
    this.videoElements.delete(mediaFileId);
    this.audioElements.delete(mediaFileId);
    this.imageElements.delete(mediaFileId);
  }

  unregisterTextCanvas(clipId: string): void {
    this.textCanvases.delete(clipId);
  }

  // === Stats ===

  getStats(): { video: number; audio: number; image: number; canvas: number } {
    return {
      video: this.videoElements.size,
      audio: this.audioElements.size,
      image: this.imageElements.size,
      canvas: this.textCanvases.size,
    };
  }

  clear(): void {
    this.videoElements.clear();
    this.audioElements.clear();
    this.imageElements.clear();
    this.textCanvases.clear();
  }
}

/** Singleton instance */
export const domRefRegistry = new DomRefRegistryImpl();
