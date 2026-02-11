// SlicePipeline - renders warped slices and masks using CPU-computed vertex positions
// Corner Pin: 16x16 subdivision per slice for perspective-correct warping
// Masks: inverted (2 tris) or non-inverted (4 complement strips of 2 tris each)

import type { OutputSlice, Point2D } from '../../types/outputSlice';
import sliceShader from '../../shaders/slice.wgsl?raw';

const SUBDIVISIONS = 16;
const FLOATS_PER_VERTEX = 5; // position.xy + uv.xy + maskFlag

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
            arrayStride: 20, // 5 floats * 4 bytes
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x2' },   // position
              { shaderLocation: 1, offset: 8, format: 'float32x2' },   // uv
              { shaderLocation: 2, offset: 16, format: 'float32' },    // maskFlag
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
   * Build vertex buffer from enabled slices and masks.
   * Slices: 16x16 subdivided quads (1536 verts each)
   * Inverted masks: 2 triangles (6 verts)
   * Non-inverted masks: 4 complement strips (24 verts)
   */
  buildVertexBuffer(slices: OutputSlice[]): void {
    const enabledItems = slices.filter((s) => s.enabled);
    if (enabledItems.length === 0) {
      this.vertexCount = 0;
      return;
    }

    // Estimate max vertices: 1536 per slice + 24 per mask
    const maxVertices = enabledItems.reduce((sum, item) => {
      return sum + (item.type === 'mask' ? 24 : SUBDIVISIONS * SUBDIVISIONS * 6);
    }, 0);
    const data = new Float32Array(maxVertices * FLOATS_PER_VERTEX);

    let offset = 0;

    for (const item of enabledItems) {
      if (item.type === 'mask') {
        offset = this.buildMaskVertices(data, offset, item);
      } else {
        if (item.warp.mode === 'cornerPin') {
          offset = this.buildCornerPinVertices(data, offset, item);
        }
      }
    }

    this.vertexCount = offset / FLOATS_PER_VERTEX;

    const byteSize = this.vertexCount * FLOATS_PER_VERTEX * 4;
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
    const [itl, itr, ibr, ibl] = slice.inputCorners;

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

        // Input UVs (bilinear interpolation of input corners)
        const uv00 = bilinearUV(itl, itr, ibr, ibl, t0, s0);
        const uv10 = bilinearUV(itl, itr, ibr, ibl, t1, s0);
        const uv11 = bilinearUV(itl, itr, ibr, ibl, t1, s1);
        const uv01 = bilinearUV(itl, itr, ibr, ibl, t0, s1);

        // Triangle 1: p00, p10, p11
        offset = pushVertex(data, offset, p00, uv00, 0);
        offset = pushVertex(data, offset, p10, uv10, 0);
        offset = pushVertex(data, offset, p11, uv11, 0);

        // Triangle 2: p00, p11, p01
        offset = pushVertex(data, offset, p00, uv00, 0);
        offset = pushVertex(data, offset, p11, uv11, 0);
        offset = pushVertex(data, offset, p01, uv01, 0);
      }
    }

    return offset;
  }

  private buildMaskVertices(data: Float32Array, offset: number, mask: OutputSlice): number {
    if (mask.warp.mode !== 'cornerPin') return offset;

    const corners = mask.warp.corners;
    const [tl, tr, br, bl] = corners;

    // Convert corners to clip space
    const cTL = toClip(tl);
    const cTR = toClip(tr);
    const cBR = toClip(br);
    const cBL = toClip(bl);

    // Dummy UV (masks don't sample texture)
    const dummyUV: Point2D = { x: 0, y: 0 };

    if (mask.inverted) {
      // Inverted: render inside the mask shape as black (2 triangles, 6 verts)
      offset = pushVertex(data, offset, cTL, dummyUV, 1);
      offset = pushVertex(data, offset, cTR, dummyUV, 1);
      offset = pushVertex(data, offset, cBR, dummyUV, 1);

      offset = pushVertex(data, offset, cTL, dummyUV, 1);
      offset = pushVertex(data, offset, cBR, dummyUV, 1);
      offset = pushVertex(data, offset, cBL, dummyUV, 1);
    } else {
      // Non-inverted: render everything OUTSIDE the mask shape as black
      // 4 complement strips connecting screen edges to mask corners (4 * 2 tris = 24 verts)
      const sTL: Point2D = { x: -1, y: 1 };   // screen top-left in clip space
      const sTR: Point2D = { x: 1, y: 1 };    // screen top-right
      const sBR: Point2D = { x: 1, y: -1 };   // screen bottom-right
      const sBL: Point2D = { x: -1, y: -1 };  // screen bottom-left

      // Top strip: sTL → sTR → cTR → cTL
      offset = pushVertex(data, offset, sTL, dummyUV, 1);
      offset = pushVertex(data, offset, sTR, dummyUV, 1);
      offset = pushVertex(data, offset, cTR, dummyUV, 1);
      offset = pushVertex(data, offset, sTL, dummyUV, 1);
      offset = pushVertex(data, offset, cTR, dummyUV, 1);
      offset = pushVertex(data, offset, cTL, dummyUV, 1);

      // Right strip: sTR → sBR → cBR → cTR
      offset = pushVertex(data, offset, sTR, dummyUV, 1);
      offset = pushVertex(data, offset, sBR, dummyUV, 1);
      offset = pushVertex(data, offset, cBR, dummyUV, 1);
      offset = pushVertex(data, offset, sTR, dummyUV, 1);
      offset = pushVertex(data, offset, cBR, dummyUV, 1);
      offset = pushVertex(data, offset, cTR, dummyUV, 1);

      // Bottom strip: sBR → sBL → cBL → cBR
      offset = pushVertex(data, offset, sBR, dummyUV, 1);
      offset = pushVertex(data, offset, sBL, dummyUV, 1);
      offset = pushVertex(data, offset, cBL, dummyUV, 1);
      offset = pushVertex(data, offset, sBR, dummyUV, 1);
      offset = pushVertex(data, offset, cBL, dummyUV, 1);
      offset = pushVertex(data, offset, cBR, dummyUV, 1);

      // Left strip: sBL → sTL → cTL → cBL
      offset = pushVertex(data, offset, sBL, dummyUV, 1);
      offset = pushVertex(data, offset, sTL, dummyUV, 1);
      offset = pushVertex(data, offset, cTL, dummyUV, 1);
      offset = pushVertex(data, offset, sBL, dummyUV, 1);
      offset = pushVertex(data, offset, cTL, dummyUV, 1);
      offset = pushVertex(data, offset, cBL, dummyUV, 1);
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

/** Convert normalized 0-1 coords to clip space (-1..1, Y-flipped for WebGPU) */
function toClip(p: Point2D): Point2D {
  return { x: p.x * 2 - 1, y: 1 - p.y * 2 };
}

/** Bilinear interpolation of 4 corners, result mapped to clip space (-1..1, Y-flipped) */
function bilinear(tl: Point2D, tr: Point2D, br: Point2D, bl: Point2D, s: number, t: number): Point2D {
  const x = (1 - s) * (1 - t) * tl.x + s * (1 - t) * tr.x + s * t * br.x + (1 - s) * t * bl.x;
  const y = (1 - s) * (1 - t) * tl.y + s * (1 - t) * tr.y + s * t * br.y + (1 - s) * t * bl.y;
  // Convert from 0-1 normalized to clip space: x → [-1, 1], y → [1, -1] (Y-flipped for WebGPU)
  return { x: x * 2 - 1, y: 1 - y * 2 };
}

/** Bilinear interpolation of 4 input corners for UV coordinates (stays in 0-1 UV space) */
function bilinearUV(tl: Point2D, tr: Point2D, br: Point2D, bl: Point2D, s: number, t: number): Point2D {
  return {
    x: (1 - s) * (1 - t) * tl.x + s * (1 - t) * tr.x + s * t * br.x + (1 - s) * t * bl.x,
    y: (1 - s) * (1 - t) * tl.y + s * (1 - t) * tr.y + s * t * br.y + (1 - s) * t * bl.y,
  };
}

/** Push a vertex (position + UV + maskFlag) into the data array */
function pushVertex(data: Float32Array, offset: number, pos: Point2D, uv: Point2D, maskFlag: number): number {
  data[offset++] = pos.x;
  data[offset++] = pos.y;
  data[offset++] = uv.x;
  data[offset++] = uv.y;
  data[offset++] = maskFlag;
  return offset;
}
