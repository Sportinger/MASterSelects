import type { RenderTarget } from '../types/renderTarget';

function isDocumentVisible(doc: Document | null | undefined): boolean {
  return !!doc && doc.visibilityState !== 'hidden';
}

function hasVisibleCanvasArea(canvas: HTMLCanvasElement | null): boolean {
  if (!canvas || !canvas.isConnected) return false;
  if (!isDocumentVisible(canvas.ownerDocument)) return false;

  const rect = canvas.getBoundingClientRect();
  return rect.width > 0 && rect.height > 0;
}

function isWindowVisible(win: Window | null): boolean {
  if (!win || win.closed) return false;

  try {
    return isDocumentVisible(win.document);
  } catch {
    return false;
  }
}

export function isRenderTargetRenderable(target: RenderTarget): boolean {
  if (!target.enabled || !target.context) return false;

  if (target.destinationType === 'window' || target.destinationType === 'tab') {
    if (!isWindowVisible(target.window)) return false;
  }

  return hasVisibleCanvasArea(target.canvas);
}
