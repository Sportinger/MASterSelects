/**
 * GPU-accelerated scope renderer.
 * Reads directly from the composition texture — no readPixels() needed.
 * Uses compute shaders for accumulation and render shaders for visualization.
 */

// ──────────────── WGSL SHADERS ────────────────

const WAVEFORM_COMPUTE = /* wgsl */ `
struct Params {
  outW: u32,
  outH: u32,
  srcW: u32,
  srcH: u32,
}

@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> accumR: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> accumG: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> accumB: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.srcW || gid.y >= params.srcH) { return; }

  let pixel = textureLoad(inputTex, vec2i(gid.xy), 0);
  let r = min(u32(pixel.r * 255.0), 255u);
  let g = min(u32(pixel.g * 255.0), 255u);
  let b = min(u32(pixel.b * 255.0), 255u);

  // Sub-pixel X: distribute weight across 2 adjacent columns (scale 256)
  let fxPos = f32(gid.x) * f32(params.outW) / f32(params.srcW);
  let x0 = u32(fxPos);
  let x1 = min(x0 + 1u, params.outW - 1u);
  let frac = fxPos - f32(x0);
  let w0 = u32((1.0 - frac) * 256.0);
  let w1 = 256u - w0;

  let hm1 = params.outH - 1u;

  let ry = (hm1 - r * hm1 / 255u) * params.outW;
  atomicAdd(&accumR[ry + x0], w0);
  atomicAdd(&accumR[ry + x1], w1);

  let gyp = (hm1 - g * hm1 / 255u) * params.outW;
  atomicAdd(&accumG[gyp + x0], w0);
  atomicAdd(&accumG[gyp + x1], w1);

  let byp = (hm1 - b * hm1 / 255u) * params.outW;
  atomicAdd(&accumB[byp + x0], w0);
  atomicAdd(&accumB[byp + x1], w1);
}
`;

const WAVEFORM_RENDER = /* wgsl */ `
struct VertexOutput {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> VertexOutput {
  var p = array<vec2f, 3>(vec2f(-1,-1), vec2f(3,-1), vec2f(-1,3));
  var out: VertexOutput;
  out.pos = vec4f(p[vid], 0, 1);
  out.uv = vec2f((p[vid].x + 1.0) * 0.5, (1.0 - p[vid].y) * 0.5);
  return out;
}

struct RenderParams {
  outW: f32,
  outH: f32,
  refValue: f32,
  intensity: f32,
}

@group(0) @binding(0) var<storage, read> accumR: array<u32>;
@group(0) @binding(1) var<storage, read> accumG: array<u32>;
@group(0) @binding(2) var<storage, read> accumB: array<u32>;
@group(0) @binding(3) var<uniform> params: RenderParams;

// Bilinear sample helper: reads accumulator with interpolation
fn sampleAccum(acc: ptr<storage, array<u32>, read>, fx: f32, fy: f32, w: u32, h: u32) -> f32 {
  let x0 = u32(clamp(fx, 0.0, f32(w - 1u)));
  let y0 = u32(clamp(fy, 0.0, f32(h - 1u)));
  let x1 = min(x0 + 1u, w - 1u);
  let y1 = min(y0 + 1u, h - 1u);
  let dx = fract(fx);
  let dy = fract(fy);
  let v00 = f32((*acc)[y0 * w + x0]);
  let v10 = f32((*acc)[y0 * w + x1]);
  let v01 = f32((*acc)[y1 * w + x0]);
  let v11 = f32((*acc)[y1 * w + x1]);
  return mix(mix(v00, v10, dx), mix(v01, v11, dx), dy);
}

@fragment
fn fs(in: VertexOutput) -> @location(0) vec4f {
  let uv = in.uv;
  if (uv.x < 0.0 || uv.x >= 1.0 || uv.y < 0.0 || uv.y >= 1.0) {
    return vec4f(0.04, 0.04, 0.04, 1.0);
  }

  let w = u32(params.outW);
  let h = u32(params.outH);

  // Floating-point grid position for bilinear sampling
  let fx = uv.x * params.outW - 0.5;
  let fy = uv.y * params.outH - 0.5;

  let rCount = sampleAccum(&accumR, fx, fy, w, h);
  let gCount = sampleAccum(&accumG, fx, fy, w, h);
  let bCount = sampleAccum(&accumB, fx, fy, w, h);

  let rv = params.refValue;
  let rN = clamp(sqrt(rCount) / rv, 0.0, 1.0);
  let gN = clamp(sqrt(gCount) / rv, 0.0, 1.0);
  let bN = clamp(sqrt(bCount) / rv, 0.0, 1.0);

  // Conservative gamma — keep dim traces subtle, only dense areas glow
  let s = params.intensity;
  var color = vec3f(
    pow(rN, 0.75) * s,
    pow(gN, 0.75) * s,
    pow(bN, 0.75) * s,
  );

  // Grid: every 10 IRE (10% of height)
  let gridY = fract(uv.y * 10.0);
  let dGrid = min(gridY, 1.0 - gridY) * params.outH * 0.5;
  if (dGrid < 0.8) {
    let a = 0.15 * (1.0 - dGrid / 0.8);
    color = max(color, vec3f(0.55, 0.45, 0.12) * a);
  }

  return vec4f(clamp(color, vec3f(0.0), vec3f(1.0)), 1.0);
}
`;

const HISTOGRAM_COMPUTE = /* wgsl */ `
struct Params { srcW: u32, srcH: u32, _pad0: u32, _pad1: u32 }

@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> histR: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> histG: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> histB: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> histL: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.srcW || gid.y >= params.srcH) { return; }
  let pixel = textureLoad(inputTex, vec2i(gid.xy), 0);
  let r = min(u32(pixel.r * 255.0), 255u);
  let g = min(u32(pixel.g * 255.0), 255u);
  let b = min(u32(pixel.b * 255.0), 255u);
  let luma = min(u32(0.2126 * pixel.r * 255.0 + 0.7152 * pixel.g * 255.0 + 0.0722 * pixel.b * 255.0), 255u);
  atomicAdd(&histR[r], 1u);
  atomicAdd(&histG[g], 1u);
  atomicAdd(&histB[b], 1u);
  atomicAdd(&histL[luma], 1u);
}
`;

const HISTOGRAM_RENDER = /* wgsl */ `
struct VertexOutput { @builtin(position) pos: vec4f, @location(0) uv: vec2f }

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> VertexOutput {
  var p = array<vec2f, 3>(vec2f(-1,-1), vec2f(3,-1), vec2f(-1,3));
  var out: VertexOutput;
  out.pos = vec4f(p[vid], 0, 1);
  out.uv = vec2f((p[vid].x + 1.0) * 0.5, (1.0 - p[vid].y) * 0.5);
  return out;
}

struct Params { totalPixels: f32, _pad0: f32, _pad1: f32, _pad2: f32 }

@group(0) @binding(0) var<storage, read> histR: array<u32>;
@group(0) @binding(1) var<storage, read> histG: array<u32>;
@group(0) @binding(2) var<storage, read> histB: array<u32>;
@group(0) @binding(3) var<storage, read> histL: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;

@fragment
fn fs(in: VertexOutput) -> @location(0) vec4f {
  let uv = in.uv;
  if (uv.x < 0.0 || uv.x >= 1.0 || uv.y < 0.0 || uv.y >= 1.0) {
    return vec4f(0.04, 0.04, 0.04, 1.0);
  }

  let bin = min(u32(uv.x * 256.0), 255u);
  let rVal = f32(histR[bin]);
  let gVal = f32(histG[bin]);
  let bVal = f32(histB[bin]);
  let lVal = f32(histL[bin]);

  // Sqrt scaling, normalized to total pixels
  let scale = 1.0 / sqrt(params.totalPixels * 0.012);
  let rH = sqrt(rVal) * scale;
  let gH = sqrt(gVal) * scale;
  let bH = sqrt(bVal) * scale;
  let lH = sqrt(lVal) * scale;

  // Y coordinate: 0 = top (highest count), 1 = bottom (zero)
  let y = 1.0 - uv.y;

  // Filled area: pixel is lit if y < normalized height
  var color = vec3f(0.0);
  if (y < lH) { color += vec3f(0.10, 0.10, 0.10); }
  if (y < rH) { color += vec3f(0.55, 0.07, 0.07); }
  if (y < gH) { color += vec3f(0.07, 0.50, 0.07); }
  if (y < bH) { color += vec3f(0.07, 0.07, 0.55); }

  // Edge glow at the top of each bar
  let edgeW = 0.008;
  if (abs(y - rH) < edgeW && y <= rH + edgeW) { color += vec3f(0.5, 0.1, 0.1); }
  if (abs(y - gH) < edgeW && y <= gH + edgeW) { color += vec3f(0.1, 0.5, 0.1); }
  if (abs(y - bH) < edgeW && y <= bH + edgeW) { color += vec3f(0.1, 0.1, 0.5); }

  // Grid lines at 64, 128, 192
  let gridBins = array<f32, 3>(64.0, 128.0, 192.0);
  for (var i = 0u; i < 3u; i++) {
    let gx = gridBins[i] / 256.0;
    if (abs(uv.x - gx) < 0.002) {
      color = max(color, vec3f(0.12));
    }
  }
  // Horizontal grid at 25%, 50%, 75%
  for (var i = 1u; i < 4u; i++) {
    let gy = f32(i) * 0.25;
    if (abs(y - gy) < 0.003) {
      color = max(color, vec3f(0.08));
    }
  }

  return vec4f(clamp(color, vec3f(0.0), vec3f(1.0)), 1.0);
}
`;

const VECTORSCOPE_COMPUTE = /* wgsl */ `
struct Params { outSize: u32, srcW: u32, srcH: u32, _pad: u32 }

@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> accumR: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> accumG: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> accumB: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.srcW || gid.y >= params.srcH) { return; }
  let pixel = textureLoad(inputTex, vec2i(gid.xy), 0);

  let r = pixel.r * 255.0;
  let g = pixel.g * 255.0;
  let b = pixel.b * 255.0;

  // BT.709 Cb/Cr
  let cb = -0.1687 * r - 0.3313 * g + 0.5 * b;
  let cr = 0.5 * r - 0.4187 * g - 0.0813 * b;

  let center = f32(params.outSize) * 0.5;
  let scale = center * 0.82;
  let px = u32(clamp(center + cb / 128.0 * scale, 0.0, f32(params.outSize - 1u)));
  let py = u32(clamp(center - cr / 128.0 * scale, 0.0, f32(params.outSize - 1u)));

  let idx = py * params.outSize + px;
  atomicAdd(&accumR[idx], u32(r) + 40u);
  atomicAdd(&accumG[idx], u32(g) + 40u);
  atomicAdd(&accumB[idx], u32(b) + 40u);
}
`;

const VECTORSCOPE_RENDER = /* wgsl */ `
struct VertexOutput { @builtin(position) pos: vec4f, @location(0) uv: vec2f }

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> VertexOutput {
  var p = array<vec2f, 3>(vec2f(-1,-1), vec2f(3,-1), vec2f(-1,3));
  var out: VertexOutput;
  out.pos = vec4f(p[vid], 0, 1);
  out.uv = vec2f((p[vid].x + 1.0) * 0.5, (1.0 - p[vid].y) * 0.5);
  return out;
}

struct Params { outSize: f32, refValue: f32, _p0: f32, _p1: f32 }

@group(0) @binding(0) var<storage, read> accumR: array<u32>;
@group(0) @binding(1) var<storage, read> accumG: array<u32>;
@group(0) @binding(2) var<storage, read> accumB: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

@fragment
fn fs(in: VertexOutput) -> @location(0) vec4f {
  let uv = in.uv;
  let size = params.outSize;
  let center = 0.5;
  let d = distance(uv, vec2f(center));

  // Background
  var color = vec3f(0.04);

  // Graticule circles at 25% and 75% saturation
  let radius75 = 0.82 * 0.5 * 0.75;
  let radius25 = 0.82 * 0.5 * 0.25;
  if (abs(d - radius75) < 0.002) { color = vec3f(0.18); }
  if (abs(d - radius25) < 0.002) { color = vec3f(0.12); }

  // Crosshair
  if ((abs(uv.x - center) < 0.001 && d < radius75 + 0.02) ||
      (abs(uv.y - center) < 0.001 && d < radius75 + 0.02)) {
    color = max(color, vec3f(0.12));
  }

  // Skin tone line (~123 degrees from Cb+ axis = ~33 degrees in standard orientation)
  let angle = atan2(-(uv.y - center), uv.x - center);
  let skinAngle = radians(123.0);
  if (abs(angle - skinAngle) < 0.008 && d < radius75 + 0.02 && d > 0.01) {
    color = max(color, vec3f(0.25, 0.18, 0.08));
  }

  // BT.709 color targets (R, MG, B, CY, G, YL)
  let targetAngles = array<f32, 6>(
    radians(103.0), radians(61.0), radians(-13.0),
    radians(-77.0), radians(-119.0), radians(167.0)
  );
  let targetColors = array<vec3f, 6>(
    vec3f(0.6, 0.15, 0.15), vec3f(0.5, 0.15, 0.5), vec3f(0.15, 0.15, 0.6),
    vec3f(0.15, 0.5, 0.5), vec3f(0.15, 0.5, 0.15), vec3f(0.5, 0.5, 0.1)
  );
  for (var i = 0u; i < 6u; i++) {
    let ta = targetAngles[i];
    let tx = center + cos(ta) * radius75;
    let ty = center - sin(ta) * radius75;
    let td = distance(uv, vec2f(tx, ty));
    if (td < 0.018) {
      color = max(color, targetColors[i] * 0.8);
      if (td > 0.013) { color = max(color, targetColors[i]); }
    }
  }

  // Data: read accumulator
  if (uv.x >= 0.0 && uv.x < 1.0 && uv.y >= 0.0 && uv.y < 1.0) {
    let gx = min(u32(uv.x * size), u32(size) - 1u);
    let gy = min(u32(uv.y * size), u32(size) - 1u);
    let idx = gy * u32(size) + gx;
    let rVal = f32(accumR[idx]);
    let gVal = f32(accumG[idx]);
    let bVal = f32(accumB[idx]);
    if (rVal > 0.0 || gVal > 0.0 || bVal > 0.0) {
      let rv = params.refValue;
      let rN = pow(clamp(sqrt(rVal) / rv, 0.0, 1.0), 0.55);
      let gN = pow(clamp(sqrt(gVal) / rv, 0.0, 1.0), 0.55);
      let bN = pow(clamp(sqrt(bVal) / rv, 0.0, 1.0), 0.55);
      color = max(color, vec3f(rN, gN, bN));
    }
  }

  return vec4f(color, 1.0);
}
`;

// ──────────────── SCOPE RENDERER CLASS ────────────────

const OUT_W = 1024;
const OUT_H = 512;
const VS_SIZE = 320; // vectorscope grid size

export class ScopeRenderer {
  private device: GPUDevice;
  private format: GPUTextureFormat;

  // Waveform
  private wfComputePipeline!: GPUComputePipeline;
  private wfRenderPipeline!: GPURenderPipeline;
  private wfComputeBGL!: GPUBindGroupLayout;
  private wfRenderBGL!: GPUBindGroupLayout;
  private wfAccumR!: GPUBuffer;
  private wfAccumG!: GPUBuffer;
  private wfAccumB!: GPUBuffer;
  private wfComputeParams!: GPUBuffer;
  private wfRenderParams!: GPUBuffer;

  // Histogram
  private histComputePipeline!: GPUComputePipeline;
  private histRenderPipeline!: GPURenderPipeline;
  private histComputeBGL!: GPUBindGroupLayout;
  private histRenderBGL!: GPUBindGroupLayout;
  private histR!: GPUBuffer;
  private histG!: GPUBuffer;
  private histB!: GPUBuffer;
  private histL!: GPUBuffer;
  private histComputeParams!: GPUBuffer;
  private histRenderParams!: GPUBuffer;

  // Vectorscope
  private vsComputePipeline!: GPUComputePipeline;
  private vsRenderPipeline!: GPURenderPipeline;
  private vsComputeBGL!: GPUBindGroupLayout;
  private vsRenderBGL!: GPUBindGroupLayout;
  private vsAccumR!: GPUBuffer;
  private vsAccumG!: GPUBuffer;
  private vsAccumB!: GPUBuffer;
  private vsComputeParams!: GPUBuffer;
  private vsRenderParams!: GPUBuffer;

  constructor(device: GPUDevice, format: GPUTextureFormat) {
    this.device = device;
    this.format = format;
    this.initWaveform();
    this.initHistogram();
    this.initVectorscope();
  }

  // ──── WAVEFORM ────

  private initWaveform() {
    const d = this.device;
    const bufSize = OUT_W * OUT_H * 4;

    this.wfAccumR = d.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.wfAccumG = d.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.wfAccumB = d.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.wfComputeParams = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.wfRenderParams = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    this.wfComputeBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });

    const computeModule = d.createShaderModule({ code: WAVEFORM_COMPUTE });
    this.wfComputePipeline = d.createComputePipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.wfComputeBGL] }),
      compute: { module: computeModule, entryPoint: 'main' },
    });

    this.wfRenderBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });

    const renderModule = d.createShaderModule({ code: WAVEFORM_RENDER });
    this.wfRenderPipeline = d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.wfRenderBGL] }),
      vertex: { module: renderModule, entryPoint: 'vs' },
      fragment: { module: renderModule, entryPoint: 'fs', targets: [{ format: this.format }] },
    });
  }

  renderWaveform(sourceTexture: GPUTexture, ctx: GPUCanvasContext) {
    const d = this.device;
    const srcW = sourceTexture.width;
    const srcH = sourceTexture.height;

    d.queue.writeBuffer(this.wfComputeParams, 0, new Uint32Array([OUT_W, OUT_H, srcW, srcH]));
    // Weights are 256x scaled from sub-pixel distribution → sqrt(256)=16 factor
    const refValue = Math.sqrt(srcH / OUT_H) * 40.0;
    d.queue.writeBuffer(this.wfRenderParams, 0, new Float32Array([OUT_W, OUT_H, refValue, 0.9]));

    const encoder = d.createCommandEncoder();

    // Clear accumulators
    encoder.clearBuffer(this.wfAccumR);
    encoder.clearBuffer(this.wfAccumG);
    encoder.clearBuffer(this.wfAccumB);

    // Compute pass
    const computeBG = d.createBindGroup({
      layout: this.wfComputeBGL,
      entries: [
        { binding: 0, resource: sourceTexture.createView() },
        { binding: 1, resource: { buffer: this.wfAccumR } },
        { binding: 2, resource: { buffer: this.wfAccumG } },
        { binding: 3, resource: { buffer: this.wfAccumB } },
        { binding: 4, resource: { buffer: this.wfComputeParams } },
      ],
    });

    const cp = encoder.beginComputePass();
    cp.setPipeline(this.wfComputePipeline);
    cp.setBindGroup(0, computeBG);
    cp.dispatchWorkgroups(Math.ceil(srcW / 16), Math.ceil(srcH / 16));
    cp.end();

    // Render pass
    const renderBG = d.createBindGroup({
      layout: this.wfRenderBGL,
      entries: [
        { binding: 0, resource: { buffer: this.wfAccumR } },
        { binding: 1, resource: { buffer: this.wfAccumG } },
        { binding: 2, resource: { buffer: this.wfAccumB } },
        { binding: 3, resource: { buffer: this.wfRenderParams } },
      ],
    });

    const rp = encoder.beginRenderPass({
      colorAttachments: [{
        view: ctx.getCurrentTexture().createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0.04, g: 0.04, b: 0.04, a: 1 },
      }],
    });
    rp.setPipeline(this.wfRenderPipeline);
    rp.setBindGroup(0, renderBG);
    rp.draw(3);
    rp.end();

    d.queue.submit([encoder.finish()]);
  }

  // ──── HISTOGRAM ────

  private initHistogram() {
    const d = this.device;
    const histBufSize = 256 * 4; // 256 bins * u32

    this.histR = d.createBuffer({ size: histBufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.histG = d.createBuffer({ size: histBufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.histB = d.createBuffer({ size: histBufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.histL = d.createBuffer({ size: histBufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.histComputeParams = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.histRenderParams = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    this.histComputeBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });

    const computeModule = d.createShaderModule({ code: HISTOGRAM_COMPUTE });
    this.histComputePipeline = d.createComputePipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.histComputeBGL] }),
      compute: { module: computeModule, entryPoint: 'main' },
    });

    this.histRenderBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });

    const renderModule = d.createShaderModule({ code: HISTOGRAM_RENDER });
    this.histRenderPipeline = d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.histRenderBGL] }),
      vertex: { module: renderModule, entryPoint: 'vs' },
      fragment: { module: renderModule, entryPoint: 'fs', targets: [{ format: this.format }] },
    });
  }

  renderHistogram(sourceTexture: GPUTexture, ctx: GPUCanvasContext) {
    const d = this.device;
    const srcW = sourceTexture.width;
    const srcH = sourceTexture.height;

    d.queue.writeBuffer(this.histComputeParams, 0, new Uint32Array([srcW, srcH, 0, 0]));
    d.queue.writeBuffer(this.histRenderParams, 0, new Float32Array([srcW * srcH, 0, 0, 0]));

    const encoder = d.createCommandEncoder();

    encoder.clearBuffer(this.histR);
    encoder.clearBuffer(this.histG);
    encoder.clearBuffer(this.histB);
    encoder.clearBuffer(this.histL);

    const computeBG = d.createBindGroup({
      layout: this.histComputeBGL,
      entries: [
        { binding: 0, resource: sourceTexture.createView() },
        { binding: 1, resource: { buffer: this.histR } },
        { binding: 2, resource: { buffer: this.histG } },
        { binding: 3, resource: { buffer: this.histB } },
        { binding: 4, resource: { buffer: this.histL } },
        { binding: 5, resource: { buffer: this.histComputeParams } },
      ],
    });

    const cp = encoder.beginComputePass();
    cp.setPipeline(this.histComputePipeline);
    cp.setBindGroup(0, computeBG);
    cp.dispatchWorkgroups(Math.ceil(srcW / 16), Math.ceil(srcH / 16));
    cp.end();

    const renderBG = d.createBindGroup({
      layout: this.histRenderBGL,
      entries: [
        { binding: 0, resource: { buffer: this.histR } },
        { binding: 1, resource: { buffer: this.histG } },
        { binding: 2, resource: { buffer: this.histB } },
        { binding: 3, resource: { buffer: this.histL } },
        { binding: 4, resource: { buffer: this.histRenderParams } },
      ],
    });

    const rp = encoder.beginRenderPass({
      colorAttachments: [{
        view: ctx.getCurrentTexture().createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0.04, g: 0.04, b: 0.04, a: 1 },
      }],
    });
    rp.setPipeline(this.histRenderPipeline);
    rp.setBindGroup(0, renderBG);
    rp.draw(3);
    rp.end();

    d.queue.submit([encoder.finish()]);
  }

  // ──── VECTORSCOPE ────

  private initVectorscope() {
    const d = this.device;
    const bufSize = VS_SIZE * VS_SIZE * 4;

    this.vsAccumR = d.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.vsAccumG = d.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.vsAccumB = d.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.vsComputeParams = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.vsRenderParams = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    this.vsComputeBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });

    const computeModule = d.createShaderModule({ code: VECTORSCOPE_COMPUTE });
    this.vsComputePipeline = d.createComputePipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.vsComputeBGL] }),
      compute: { module: computeModule, entryPoint: 'main' },
    });

    this.vsRenderBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });

    const renderModule = d.createShaderModule({ code: VECTORSCOPE_RENDER });
    this.vsRenderPipeline = d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.vsRenderBGL] }),
      vertex: { module: renderModule, entryPoint: 'vs' },
      fragment: { module: renderModule, entryPoint: 'fs', targets: [{ format: this.format }] },
    });
  }

  renderVectorscope(sourceTexture: GPUTexture, ctx: GPUCanvasContext) {
    const d = this.device;
    const srcW = sourceTexture.width;
    const srcH = sourceTexture.height;

    d.queue.writeBuffer(this.vsComputeParams, 0, new Uint32Array([VS_SIZE, srcW, srcH, 0]));
    const refValue = Math.sqrt(srcH * srcW / (VS_SIZE * VS_SIZE)) * 3.0;
    d.queue.writeBuffer(this.vsRenderParams, 0, new Float32Array([VS_SIZE, refValue, 0, 0]));

    const encoder = d.createCommandEncoder();

    encoder.clearBuffer(this.vsAccumR);
    encoder.clearBuffer(this.vsAccumG);
    encoder.clearBuffer(this.vsAccumB);

    const computeBG = d.createBindGroup({
      layout: this.vsComputeBGL,
      entries: [
        { binding: 0, resource: sourceTexture.createView() },
        { binding: 1, resource: { buffer: this.vsAccumR } },
        { binding: 2, resource: { buffer: this.vsAccumG } },
        { binding: 3, resource: { buffer: this.vsAccumB } },
        { binding: 4, resource: { buffer: this.vsComputeParams } },
      ],
    });

    const cp = encoder.beginComputePass();
    cp.setPipeline(this.vsComputePipeline);
    cp.setBindGroup(0, computeBG);
    cp.dispatchWorkgroups(Math.ceil(srcW / 16), Math.ceil(srcH / 16));
    cp.end();

    const renderBG = d.createBindGroup({
      layout: this.vsRenderBGL,
      entries: [
        { binding: 0, resource: { buffer: this.vsAccumR } },
        { binding: 1, resource: { buffer: this.vsAccumG } },
        { binding: 2, resource: { buffer: this.vsAccumB } },
        { binding: 3, resource: { buffer: this.vsRenderParams } },
      ],
    });

    const rp = encoder.beginRenderPass({
      colorAttachments: [{
        view: ctx.getCurrentTexture().createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0.04, g: 0.04, b: 0.04, a: 1 },
      }],
    });
    rp.setPipeline(this.vsRenderPipeline);
    rp.setBindGroup(0, renderBG);
    rp.draw(3);
    rp.end();

    d.queue.submit([encoder.finish()]);
  }

  // ──── CLEANUP ────

  destroy() {
    const bufs = [
      this.wfAccumR, this.wfAccumG, this.wfAccumB, this.wfComputeParams, this.wfRenderParams,
      this.histR, this.histG, this.histB, this.histL, this.histComputeParams, this.histRenderParams,
      this.vsAccumR, this.vsAccumG, this.vsAccumB, this.vsComputeParams, this.vsRenderParams,
    ];
    for (const b of bufs) b?.destroy();
  }
}
