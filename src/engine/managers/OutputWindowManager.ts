// Manages external output windows (fullscreen, secondary displays)
// Simplified: only handles window lifecycle. State lives in renderTargetStore.

import { useRenderTargetStore } from '../../stores/renderTargetStore';
import { useSliceStore } from '../../stores/sliceStore';
import { Logger } from '../../services/logger';

const log = Logger.create('OutputWindowManager');
const OPEN_WINDOWS_KEY = 'masterselects-output-windows-open';

/** Track which output window IDs are currently open (survives page refresh within same tab session) */
function markWindowOpen(id: string): void {
  try {
    const ids: string[] = JSON.parse(sessionStorage.getItem(OPEN_WINDOWS_KEY) || '[]');
    if (!ids.includes(id)) ids.push(id);
    sessionStorage.setItem(OPEN_WINDOWS_KEY, JSON.stringify(ids));
  } catch { /* ignore */ }
}

function markWindowClosed(id: string): void {
  try {
    const ids: string[] = JSON.parse(sessionStorage.getItem(OPEN_WINDOWS_KEY) || '[]');
    sessionStorage.setItem(OPEN_WINDOWS_KEY, JSON.stringify(ids.filter(i => i !== id)));
  } catch { /* ignore */ }
}

function isWindowKnownOpen(id: string): boolean {
  try {
    const ids: string[] = JSON.parse(sessionStorage.getItem(OPEN_WINDOWS_KEY) || '[]');
    return ids.includes(id);
  } catch { return false; }
}

export class OutputWindowManager {
  private outputWidth: number;
  private outputHeight: number;

  constructor(width: number, height: number) {
    this.outputWidth = width;
    this.outputHeight = height;
  }

  /**
   * Creates a popup window with a canvas element.
   * Does NOT configure WebGPU - that's done by engine.registerTargetCanvas().
   * Returns the window + canvas for the caller to wire up.
   * Optional geometry restores previous position/size/screen.
   */
  createWindow(id: string, name: string, geometry?: {
    screenX?: number; screenY?: number; outerWidth?: number; outerHeight?: number;
  }): { window: Window; canvas: HTMLCanvasElement } | null {
    const w = geometry?.outerWidth ?? 960;
    const h = geometry?.outerHeight ?? 540;
    let features = `width=${w},height=${h},menubar=no,toolbar=no,location=no,status=no`;
    if (geometry?.screenX != null && geometry?.screenY != null) {
      features += `,left=${geometry.screenX},top=${geometry.screenY}`;
    }
    const outputWindow = window.open('', `output_${id}`, features);

    if (!outputWindow) {
      log.error('Failed to open window (popup blocked?)');
      return null;
    }

    outputWindow.document.title = `WebVJ Output - ${name}`;
    outputWindow.document.body.style.cssText =
      'margin:0;padding:0;background:#000;overflow:hidden;width:100vw;height:100vh;';

    const canvas = outputWindow.document.createElement('canvas');
    canvas.width = this.outputWidth;
    canvas.height = this.outputHeight;
    canvas.style.cssText = 'display:block;background:#000;width:100%;height:100%;';
    outputWindow.document.body.appendChild(canvas);

    // Aspect ratio locking
    const aspectRatio = this.outputWidth / this.outputHeight;
    let lastWidth = outputWindow.innerWidth;
    let lastHeight = outputWindow.innerHeight;
    let resizing = false;

    const enforceAspectRatio = () => {
      if (resizing) return;
      resizing = true;

      const currentWidth = outputWindow.innerWidth;
      const currentHeight = outputWindow.innerHeight;
      const widthDelta = Math.abs(currentWidth - lastWidth);
      const heightDelta = Math.abs(currentHeight - lastHeight);

      let newWidth: number;
      let newHeight: number;

      if (widthDelta >= heightDelta) {
        newWidth = currentWidth;
        newHeight = Math.round(currentWidth / aspectRatio);
      } else {
        newHeight = currentHeight;
        newWidth = Math.round(currentHeight * aspectRatio);
      }

      if (newWidth !== currentWidth || newHeight !== currentHeight) {
        outputWindow.resizeTo(
          newWidth + (outputWindow.outerWidth - currentWidth),
          newHeight + (outputWindow.outerHeight - currentHeight)
        );
      }

      canvas.style.width = '100%';
      canvas.style.height = '100%';

      lastWidth = newWidth;
      lastHeight = newHeight;

      setTimeout(() => { resizing = false; }, 50);
    };

    outputWindow.addEventListener('resize', enforceAspectRatio);

    // Fullscreen button
    const fullscreenBtn = outputWindow.document.createElement('button');
    fullscreenBtn.textContent = 'Fullscreen';
    fullscreenBtn.style.cssText =
      'position:fixed;top:10px;right:10px;padding:8px 16px;cursor:pointer;z-index:1000;opacity:0.7;';
    fullscreenBtn.onclick = () => {
      canvas.requestFullscreen();
    };
    outputWindow.document.body.appendChild(fullscreenBtn);

    outputWindow.document.addEventListener('fullscreenchange', () => {
      fullscreenBtn.style.display = outputWindow.document.fullscreenElement ? 'none' : 'block';
    });

    // Track that this window is open (for reconnection guard after refresh)
    markWindowOpen(id);

    // When window is closed by user, save geometry then deactivate (keep entry grayed out)
    outputWindow.onbeforeunload = () => {
      markWindowClosed(id);
      // Trigger a save so geometry is captured with current window position/size
      try { useSliceStore.getState().saveToLocalStorage(); } catch { /* ignore */ }
      useRenderTargetStore.getState().deactivateTarget(id);
    };

    // Ensure popup gets foreground activation on Windows:
    // Blur the parent first to release foreground lock, then focus the popup
    // from its own context so the OS allows it to become the foreground window.
    window.blur();
    outputWindow.focus();
    outputWindow.setTimeout(() => outputWindow.focus(), 50);
    outputWindow.setTimeout(() => outputWindow.focus(), 200);

    log.info('Created output window', { id, name });
    return { window: outputWindow, canvas };
  }

  /**
   * Try to reconnect to an existing output window after page refresh.
   * Returns the window + its existing canvas, or null if not found.
   */
  reconnectWindow(id: string): { window: Window; canvas: HTMLCanvasElement } | null {
    // Only attempt reconnection if we know this window was open in this session.
    // Without this guard, window.open creates a blank popup that steals focus
    // and causes the dock tabs to jump on page refresh.
    if (!isWindowKnownOpen(id)) {
      return null;
    }

    const existing = window.open('', `output_${id}`);
    if (!existing || existing.closed) {
      markWindowClosed(id);
      return null;
    }

    // Check if this window has our canvas (means it was previously opened by us)
    const canvas = existing.document.querySelector('canvas');
    if (!canvas) {
      // It's a freshly opened blank popup — close it
      existing.close();
      markWindowClosed(id);
      return null;
    }

    // Re-setup the close handler (save geometry before deactivating)
    existing.onbeforeunload = () => {
      markWindowClosed(id);
      try { useSliceStore.getState().saveToLocalStorage(); } catch { /* ignore */ }
      useRenderTargetStore.getState().deactivateTarget(id);
    };

    log.info('Reconnected to existing output window', { id });
    return { window: existing, canvas };
  }

  updateResolution(width: number, height: number): void {
    this.outputWidth = width;
    this.outputHeight = height;
  }

  destroy(): void {
    // Close all output windows via the render target store
    const store = useRenderTargetStore.getState();
    for (const target of store.targets.values()) {
      if (target.destinationType === 'window' && target.window && !target.window.closed) {
        target.window.close();
      }
    }
  }
}
