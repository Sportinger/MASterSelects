import { describe, expect, it, vi } from 'vitest';
import { Compositor } from '../../src/engine/render/Compositor';

function makeRenderPass() {
  return {
    setPipeline: vi.fn(),
    setBindGroup: vi.fn(),
    draw: vi.fn(),
    end: vi.fn(),
  };
}

function makeLayerData() {
  return [{
    layer: {
      id: 'layer-1',
      maskClipId: undefined,
      effects: [
        {
          id: 'fx-brightness',
          name: 'Brightness',
          type: 'brightness',
          enabled: true,
          params: { amount: 0.4 },
        },
        {
          id: 'fx-blur',
          name: 'Blur',
          type: 'blur',
          enabled: true,
          params: { radius: 12 },
        },
      ],
    },
    isVideo: false,
    externalTexture: null,
    textureView: { label: 'source-view' },
    sourceWidth: 1920,
    sourceHeight: 1080,
  }] as any;
}

describe('Compositor scrub fast path', () => {
  it('skips inline and complex effects while scrubbing', () => {
    const updateLayerUniforms = vi.fn();
    const applyEffects = vi.fn(() => ({
      finalView: { label: 'effect-view' },
      swapped: false,
    }));

    const compositor = new Compositor(
      {
        getOrCreateUniformBuffer: vi.fn(() => ({ label: 'ubo' })),
        updateLayerUniforms,
        getCompositePipeline: vi.fn(() => ({ label: 'pipeline' })),
        createCompositeBindGroup: vi.fn(() => ({ label: 'bind-group' })),
        getExternalCompositePipeline: vi.fn(),
        createExternalCompositeBindGroup: vi.fn(),
        invalidateBindGroupCache: vi.fn(),
      } as any,
      { applyEffects } as any,
      {
        getMaskInfo: vi.fn(() => ({ hasMask: false, view: { label: 'mask' } })),
        logMaskState: vi.fn(),
      } as any
    );

    const commandEncoder = {
      beginRenderPass: vi.fn(() => makeRenderPass()),
    } as any;

    compositor.composite(makeLayerData(), commandEncoder, {
      device: {} as any,
      sampler: {} as any,
      pingView: { label: 'ping' } as any,
      pongView: { label: 'pong' } as any,
      outputWidth: 1920,
      outputHeight: 1080,
      skipEffects: true,
      effectTempView: { label: 'tmp-a' } as any,
      effectTempView2: { label: 'tmp-b' } as any,
    });

    expect(updateLayerUniforms.mock.calls[0][5]).toEqual({
      brightness: 0,
      contrast: 1,
      saturation: 1,
      invert: false,
    });
    expect(applyEffects).not.toHaveBeenCalled();
  });

  it('still applies layer effects when scrub fast path is disabled', () => {
    const updateLayerUniforms = vi.fn();
    const applyEffects = vi.fn(() => ({
      finalView: { label: 'effect-view' },
      swapped: false,
    }));

    const compositor = new Compositor(
      {
        getOrCreateUniformBuffer: vi.fn(() => ({ label: 'ubo' })),
        updateLayerUniforms,
        getCompositePipeline: vi.fn(() => ({ label: 'pipeline' })),
        createCompositeBindGroup: vi.fn(() => ({ label: 'bind-group' })),
        getExternalCompositePipeline: vi.fn(),
        createExternalCompositeBindGroup: vi.fn(),
        invalidateBindGroupCache: vi.fn(),
      } as any,
      { applyEffects } as any,
      {
        getMaskInfo: vi.fn(() => ({ hasMask: false, view: { label: 'mask' } })),
        logMaskState: vi.fn(),
      } as any
    );

    const commandEncoder = {
      beginRenderPass: vi.fn(() => makeRenderPass()),
    } as any;

    compositor.composite(makeLayerData(), commandEncoder, {
      device: {} as any,
      sampler: {} as any,
      pingView: { label: 'ping' } as any,
      pongView: { label: 'pong' } as any,
      outputWidth: 1920,
      outputHeight: 1080,
      skipEffects: false,
      effectTempView: { label: 'tmp-a' } as any,
      effectTempView2: { label: 'tmp-b' } as any,
    });

    expect(updateLayerUniforms.mock.calls[0][5]).toEqual({
      brightness: 0.4,
      contrast: 1,
      saturation: 1,
      invert: false,
    });
    expect(applyEffects).toHaveBeenCalledTimes(1);
  });
});
