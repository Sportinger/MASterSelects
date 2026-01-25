// Manages external output windows (fullscreen, secondary displays)

import type { OutputWindow } from '../core/types';
import { Logger } from '../../services/logger';

const log = Logger.create('OutputWindowManager');

export class OutputWindowManager {
  private outputWindows: Map<string, OutputWindow> = new Map();
  private outputWidth: number;
  private outputHeight: number;

  constructor(width: number, height: number) {
    this.outputWidth = width;
    this.outputHeight = height;
  }

  createOutputWindow(id: string, name: string, device: GPUDevice): OutputWindow | null {
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
    canvas.style.cssText = 'display:block;background:#000;';
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

    canvas.style.width = '100%';
    canvas.style.height = '100%';
    outputWindow.addEventListener('resize', enforceAspectRatio);

    let context: GPUCanvasContext | null = null;

    if (device) {
      context = canvas.getContext('webgpu');
      if (context) {
        context.configure({
          device,
          format: 'bgra8unorm',
          alphaMode: 'premultiplied',
        });
      }
    }

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

    outputWindow.onbeforeunload = () => {
      this.outputWindows.delete(id);
    };

    const output: OutputWindow = {
      id,
      name,
      window: outputWindow,
      canvas,
      context,
      isFullscreen: false,
    };

    this.outputWindows.set(id, output);
    return output;
  }

  closeOutputWindow(id: string): void {
    const output = this.outputWindows.get(id);
    if (output?.window) {
      output.window.close();
    }
    this.outputWindows.delete(id);
  }

  getOutputWindows(): Map<string, OutputWindow> {
    return this.outputWindows;
  }

  updateResolution(width: number, height: number): void {
    this.outputWidth = width;
    this.outputHeight = height;
  }

  destroy(): void {
    for (const output of this.outputWindows.values()) {
      output.window?.close();
    }
    this.outputWindows.clear();
  }
}
