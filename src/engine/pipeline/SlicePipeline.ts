// SlicePipeline - renders warped slices using CPU-computed vertex positions
// Corner Pin: 16x16 subdivision per slice for perspective-correct warping
// Mesh Grid (Phase 2): each grid cell is a quad, same code path

import type { OutputSlice, Point2D } from '../../types/outputSlice';
import sliceShader from '../../shaders/slice.wgsl?raw';

const SUBDIVISIONS = 16;

export class SlicePipeline {
  private device: GPUDevice;
  private pipeline: GPURenderPipeline | null = null;
  private bindGroupLayout: GPUBindGroupLayout | null = null;
  private vertexBuffer: GPUBuffer | null = null;
  private vertexCount = 0;

  constructor(device: GPUDevice) {
    this.device = device;
  }

  async createPipeline(): Promise<void> {
    this.bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
      ],
    });

    const module = this.device.createShaderModule({ code: sliceShader });

    this.pipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [this.bindGroupLayout],
      }),
      vertex: {
        module,
        entryPoint: 'vertexMain',
        buffers: [
          {
            arrayStride: 16, // 4 floats * 4 bytes
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x2' },  // position
              { shaderLocation: 1, offset: 8, format: 'float32x2' },  // uv
            ],
          },
        ],
      },
      fragment: {
        module,
        entryPoint: 'fragmentMain',
        targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }],
      },
      primitive: { topology: 'triangle-list' },
    });
  }

  getBindGroupLayout(): GPUBindGroupLayout | null {
    return this.bindGroupLayout;
  }

  /**
   * Build vertex buffer from enabled slices.
   * Each corner pin slice is subdivided into 16x16 quads (2 triangles each = 3072 vertices per slice).
   */
  buildVertexBuffer(slices: OutputSlice[]): void {
    const enabledSlices = slices.filter((s) => s.enabled);
    if (enabledSlices.length === 0) {
      this.vertexCount = 0;
      return;
    }

    // 16x16 sub-quads * 2 triangles * 3 vertices = 1536 vertices per slice
    // Actually: 16*16 * 6 = 1536 vertices per slice
    const verticesPerSlice = SUBDIVISIONS * SUBDIVISIONS * 6;
    const totalVertices = enabledSlices.length * verticesPerSlice;
    const floatsPerVertex = 4; // position.xy + uv.xy
    const data = new Float32Array(totalVertices * floatsPerVertex);

    let offset = 0;

    for (const slice of enabledSlices) {
      if (slice.warp.mode === 'cornerPin') {
        offset = this.buildCornerPinVertices(data, offset, slice);
      }
      // meshGrid support added in Phase 2
    }

    this.vertexCount = offset / floatsPerVertex;

    // Recreate buffer if needed
    const byteSize = this.vertexCount * floatsPerVertex * 4;
    if (byteSize === 0) return;

    if (!this.vertexBuffer || this.vertexBuffer.size < byteSize) {
      this.vertexBuffer?.destroy();
      this.vertexBuffer = this.device.createBuffer({
        size: byteSize,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
    }

    this.device.queue.writeBuffer(this.vertexBuffer, 0, data, 0, offset);
  }

  private buildCornerPinVertices(data: Float32Array, offset: number, slice: OutputSlice): number {
    const corners = (slice.warp as { mode: 'cornerPin'; corners: [Point2D, Point2D, Point2D, Point2D] }).corners;
    const [tl, tr, br, bl] = corners;
    const rect = slice.inputRect;

    for (let row = 0; row < SUBDIVISIONS; row++) {
      for (let col = 0; col < SUBDIVISIONS; col++) {
        const t0 = col / SUBDIVISIONS;
        const t1 = (col + 1) / SUBDIVISIONS;
        const s0 = row / SUBDIVISIONS;
        const s1 = (row + 1) / SUBDIVISIONS;

        // Output positions (bilinear interpolation of corners, mapped to clip space -1..1)
        const p00 = bilinear(tl, tr, br, bl, t0, s0);
        const p10 = bilinear(tl, tr, br, bl, t1, s0);
        const p11 = bilinear(tl, tr, br, bl, t1, s1);
        const p01 = bilinear(tl, tr, br, bl, t0, s1);

        // Input UVs (lerp within inputRect)
        const uv00 = lerpUV(rect, t0, s0);
        const uv10 = lerpUV(rect, t1, s0);
        const uv11 = lerpUV(rect, t1, s1);
        const uv01 = lerpUV(rect, t0, s1);

        // Triangle 1: p00, p10, p11
        offset = pushVertex(data, offset, p00, uv00);
        offset = pushVertex(data, offset, p10, uv10);
        offset = pushVertex(data, offset, p11, uv11);

        // Triangle 2: p00, p11, p01
        offset = pushVertex(data, offset, p00, uv00);
        offset = pushVertex(data, offset, p11, uv11);
        offset = pushVertex(data, offset, p01, uv01);
      }
    }

    return offset;
  }

  /**
   * Render all sliced output to a canvas context using the source texture.
   */
  renderSlicedOutput(
    commandEncoder: GPUCommandEncoder,
    context: GPUCanvasContext,
    sourceView: GPUTextureView,
    sampler: GPUSampler
  ): void {
    if (!this.pipeline || !this.bindGroupLayout || !this.vertexBuffer || this.vertexCount === 0) return;

    let canvasView: GPUTextureView;
    try {
      canvasView = context.getCurrentTexture().createView();
    } catch {
      return; // Canvas context lost
    }

    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: sampler },
        { binding: 1, resource: sourceView },
      ],
    });

    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: canvasView,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });

    renderPass.setPipeline(this.pipeline);
    renderPass.setBindGroup(0, bindGroup);
    renderPass.setVertexBuffer(0, this.vertexBuffer);
    renderPass.draw(this.vertexCount);
    renderPass.end();
  }

  destroy(): void {
    this.vertexBuffer?.destroy();
    this.vertexBuffer = null;
    this.vertexCount = 0;
  }
}

// === Helpers ===

/** Bilinear interpolation of 4 corners, result mapped to clip space (-1..1, Y-flipped) */
function bilinear(tl: Point2D, tr: Point2D, br: Point2D, bl: Point2D, s: number, t: number): Point2D {
  const x = (1 - s) * (1 - t) * tl.x + s * (1 - t) * tr.x + s * t * br.x + (1 - s) * t * bl.x;
  const y = (1 - s) * (1 - t) * tl.y + s * (1 - t) * tr.y + s * t * br.y + (1 - s) * t * bl.y;
  // Convert from 0-1 normalized to clip space: x → [-1, 1], y → [1, -1] (Y-flipped for WebGPU)
  return { x: x * 2 - 1, y: 1 - y * 2 };
}

/** Interpolate UV coordinates within an input rect */
function lerpUV(rect: { x: number; y: number; width: number; height: number }, s: number, t: number): Point2D {
  return {
    x: rect.x + s * rect.width,
    y: rect.y + t * rect.height,
  };
}

/** Push a vertex (position + UV) into the data array */
function pushVertex(data: Float32Array, offset: number, pos: Point2D, uv: Point2D): number {
  data[offset++] = pos.x;
  data[offset++] = pos.y;
  data[offset++] = uv.x;
  data[offset++] = uv.y;
  return offset;
}
