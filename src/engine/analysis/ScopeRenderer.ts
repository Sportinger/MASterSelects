/**
 * GPU-accelerated scope renderer.
 * Delegates to specialized scope classes for each mode.
 */

import { WaveformScope } from './WaveformScope';
import { HistogramScope } from './HistogramScope';
import { VectorscopeScope } from './VectorscopeScope';

export class ScopeRenderer {
  private waveform: WaveformScope;
  private histogram: HistogramScope;
  private vectorscope: VectorscopeScope;

  constructor(device: GPUDevice, format: GPUTextureFormat) {
    this.waveform = new WaveformScope(device, format);
    this.histogram = new HistogramScope(device, format);
    this.vectorscope = new VectorscopeScope(device, format);
  }

  renderWaveform(sourceTexture: GPUTexture, ctx: GPUCanvasContext, mode: number = 0) {
    this.waveform.render(sourceTexture, ctx, mode);
  }

  renderHistogram(sourceTexture: GPUTexture, ctx: GPUCanvasContext, mode: number = 0) {
    this.histogram.render(sourceTexture, ctx, mode);
  }

  renderVectorscope(sourceTexture: GPUTexture, ctx: GPUCanvasContext) {
    this.vectorscope.render(sourceTexture, ctx);
  }

  destroy() {
    this.waveform.destroy();
    this.histogram.destroy();
    this.vectorscope.destroy();
  }
}
