// Manages external output windows (fullscreen, secondary displays)
// Simplified: only handles window lifecycle. State lives in renderTargetStore.

import { useRenderTargetStore } from '../../stores/renderTargetStore';
import { Logger } from '../../services/logger';

const log = Logger.create('OutputWindowManager');

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
   */
  createWindow(id: string, name: string): { window: Window; canvas: HTMLCanvasElement } | null {
    const outputWindow = window.open(
      '',
      `output_${id}`,
      'width=960,height=540,menubar=no,toolbar=no,location=no,status=no'
    );

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

    // When window is closed by user, clean up from render target store
    outputWindow.onbeforeunload = () => {
      useRenderTargetStore.getState().unregisterTarget(id);
    };

    log.info('Created output window', { id, name });
    return { window: outputWindow, canvas };
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
