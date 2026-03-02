/**
 * GPU-accelerated vectorscope.
 * Compute shader maps pixels to CbCr space, render shader visualizes with graticule.
 */

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
  let scale = center * 0.92;
  let px = u32(clamp(center + cb / 128.0 * scale, 0.0, f32(params.outSize - 1u)));
  let py = u32(clamp(center - cr / 128.0 * scale, 0.0, f32(params.outSize - 1u)));

  let idx = py * params.outSize + px;
  // Accumulate raw pixel color (no bias, preserves color ratios)
  atomicAdd(&accumR[idx], u32(max(r, 1.0)));
  atomicAdd(&accumG[idx], u32(max(g, 1.0)));
  atomicAdd(&accumB[idx], u32(max(b, 1.0)));
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

// Bilinear sample from accumulator
fn sampleVS(acc: ptr<storage, array<u32>, read>, fx: f32, fy: f32, sz: u32) -> f32 {
  let x0 = u32(clamp(fx, 0.0, f32(sz - 1u)));
  let y0 = u32(clamp(fy, 0.0, f32(sz - 1u)));
  let x1 = min(x0 + 1u, sz - 1u);
  let y1 = min(y0 + 1u, sz - 1u);
  let dx = fract(fx);
  let dy = fract(fy);
  let v00 = f32((*acc)[y0 * sz + x0]);
  let v10 = f32((*acc)[y0 * sz + x1]);
  let v01 = f32((*acc)[y1 * sz + x0]);
  let v11 = f32((*acc)[y1 * sz + x1]);
  return mix(mix(v00, v10, dx), mix(v01, v11, dx), dy);
}

// Nearest read for bloom
fn readVS(acc: ptr<storage, array<u32>, read>, x: i32, y: i32, sz: i32) -> f32 {
  return f32((*acc)[u32(clamp(y, 0, sz - 1)) * u32(sz) + u32(clamp(x, 0, sz - 1))]);
}

@fragment
fn fs(in: VertexOutput) -> @location(0) vec4f {
  let uv = in.uv;
  let size = params.outSize;
  let sz = u32(size);
  let isz = i32(sz);
  let center = 0.5;
  let d = distance(uv, vec2f(center));
  let gratScale = 0.92;

  // Background
  var color = vec3f(0.04);

  // Graticule: outer circle (100% saturation boundary) + 75% + 25%
  let radiusFull = gratScale * 0.5;
  let radius75 = gratScale * 0.5 * 0.75;
  let radius25 = gratScale * 0.5 * 0.25;
  let lineW = 1.2 / size; // ~1.2px anti-aliased
  let aa = smoothstep(0.0, lineW, abs(d - radiusFull));
  color = mix(vec3f(0.20), color, aa);
  let aa75 = smoothstep(0.0, lineW, abs(d - radius75));
  color = mix(vec3f(0.14), color, aa75);
  let aa25 = smoothstep(0.0, lineW, abs(d - radius25));
  color = mix(vec3f(0.10), color, aa25);

  // Crosshair (anti-aliased)
  let crossW = 0.8 / size;
  if (d < radiusFull + 0.01) {
    let axH = smoothstep(0.0, crossW, abs(uv.y - center));
    let axV = smoothstep(0.0, crossW, abs(uv.x - center));
    color = mix(vec3f(0.12), color, axH);
    color = mix(vec3f(0.12), color, axV);
  }

  // Skin tone line (~123 degrees)
  let angle = atan2(-(uv.y - center), uv.x - center);
  let skinAngle = radians(123.0);
  let skinAA = smoothstep(0.0, crossW, abs(angle - skinAngle));
  if (d < radiusFull + 0.01 && d > 0.01) {
    color = mix(vec3f(0.28, 0.20, 0.08), color, skinAA);
  }

  // BT.709 color targets (R, MG, B, CY, G, YL) — placed on 75% ring
  let targetAngles = array<f32, 6>(
    radians(103.0), radians(61.0), radians(-13.0),
    radians(-77.0), radians(-119.0), radians(167.0)
  );
  let targetColors = array<vec3f, 6>(
    vec3f(0.6, 0.15, 0.15), vec3f(0.5, 0.15, 0.5), vec3f(0.15, 0.15, 0.6),
    vec3f(0.15, 0.5, 0.5), vec3f(0.15, 0.5, 0.15), vec3f(0.5, 0.5, 0.1)
  );
  let dotR = 8.0 / size;
  let ringW = 2.0 / size;
  for (var i = 0u; i < 6u; i++) {
    let ta = targetAngles[i];
    let tx = center + cos(ta) * radius75;
    let ty = center - sin(ta) * radius75;
    let td = distance(uv, vec2f(tx, ty));
    // Filled dot with ring outline
    let dotAA = smoothstep(dotR, dotR - ringW, td);
    let ringAA = smoothstep(ringW * 0.5, 0.0, abs(td - dotR));
    color = mix(color, targetColors[i] * 0.5, dotAA);
    color = mix(color, targetColors[i], ringAA);
  }

  // Data: bilinear center + bloom glow
  if (uv.x >= 0.0 && uv.x < 1.0 && uv.y >= 0.0 && uv.y < 1.0) {
    let fx = uv.x * size - 0.5;
    let fy = uv.y * size - 0.5;

    // Sharp center (bilinear)
    let rCenter = sampleVS(&accumR, fx, fy, sz);
    let gCenter = sampleVS(&accumG, fx, fy, sz);
    let bCenter = sampleVS(&accumB, fx, fy, sz);

    // Bloom: 3x3 gaussian at 3px step
    let ix = i32(fx + 0.5);
    let iy = i32(fy + 0.5);
    var rBloom = 0.0; var gBloom = 0.0; var bBloom = 0.0;
    let bK = array<f32, 3>(0.25, 0.50, 0.25);
    for (var by: i32 = -1; by <= 1; by += 1) {
      for (var bx: i32 = -1; bx <= 1; bx += 1) {
        let bw = bK[u32(bx + 1)] * bK[u32(by + 1)];
        rBloom += readVS(&accumR, ix + bx * 3, iy + by * 3, isz) * bw;
        gBloom += readVS(&accumG, ix + bx * 3, iy + by * 3, isz) * bw;
        bBloom += readVS(&accumB, ix + bx * 3, iy + by * 3, isz) * bw;
      }
    }

    let rv = params.refValue;

    // Density-based brightness (sum of channels)
    let totalCenter = rCenter + gCenter + bCenter;
    let totalBloom = rBloom + gBloom + bBloom;
    let density = pow(clamp(sqrt(totalCenter / 3.0) / rv, 0.0, 1.0), 0.7);
    let bloomD = pow(clamp(sqrt(totalBloom / 3.0) / rv, 0.0, 1.0), 0.6) * 0.18;

    // Color ratios from accumulated data (preserves hue)
    if (totalCenter > 0.0) {
      let rRatio = rCenter / totalCenter;
      let gRatio = gCenter / totalCenter;
      let bRatio = bCenter / totalCenter;
      // Scale ratios to visible range (neutral = 0.33 each, pure channel = 1.0)
      let chromaColor = vec3f(rRatio, gRatio, bRatio) * 3.0;
      // Blend: at low density show saturated color, at high density tend toward white
      let whiteMix = density * density * 0.5;
      let traceColor = mix(chromaColor, vec3f(1.0), whiteMix) * (density + bloomD);
      color = max(color, clamp(traceColor, vec3f(0.0), vec3f(1.0)));
    }
  }

  return vec4f(color, 1.0);
}
`;

export const VS_SIZE = 512;

export class VectorscopeScope {
  private device: GPUDevice;

  private computePipeline!: GPUComputePipeline;
  private renderPipeline!: GPURenderPipeline;
  private computeBGL!: GPUBindGroupLayout;
  private renderBGL!: GPUBindGroupLayout;
  private accumR!: GPUBuffer;
  private accumG!: GPUBuffer;
  private accumB!: GPUBuffer;
  private computeParams!: GPUBuffer;
  private renderParams!: GPUBuffer;

  constructor(device: GPUDevice, format: GPUTextureFormat) {
    this.device = device;
    this.init(format);
  }

  private init(format: GPUTextureFormat) {
    const d = this.device;
    const bufSize = VS_SIZE * VS_SIZE * 4;

    this.accumR = d.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.accumG = d.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.accumB = d.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.computeParams = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.renderParams = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    this.computeBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });

    const computeModule = d.createShaderModule({ code: VECTORSCOPE_COMPUTE });
    this.computePipeline = d.createComputePipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.computeBGL] }),
      compute: { module: computeModule, entryPoint: 'main' },
    });

    this.renderBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });

    const renderModule = d.createShaderModule({ code: VECTORSCOPE_RENDER });
    this.renderPipeline = d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.renderBGL] }),
      vertex: { module: renderModule, entryPoint: 'vs' },
      fragment: { module: renderModule, entryPoint: 'fs', targets: [{ format }] },
    });
  }

  render(sourceTexture: GPUTexture, ctx: GPUCanvasContext) {
    const d = this.device;
    const srcW = sourceTexture.width;
    const srcH = sourceTexture.height;

    d.queue.writeBuffer(this.computeParams, 0, new Uint32Array([VS_SIZE, srcW, srcH, 0]));
    const refValue = Math.sqrt(srcH * srcW / (VS_SIZE * VS_SIZE)) * 18.0;
    d.queue.writeBuffer(this.renderParams, 0, new Float32Array([VS_SIZE, refValue, 0, 0]));

    const encoder = d.createCommandEncoder();

    encoder.clearBuffer(this.accumR);
    encoder.clearBuffer(this.accumG);
    encoder.clearBuffer(this.accumB);

    const computeBG = d.createBindGroup({
      layout: this.computeBGL,
      entries: [
        { binding: 0, resource: sourceTexture.createView() },
        { binding: 1, resource: { buffer: this.accumR } },
        { binding: 2, resource: { buffer: this.accumG } },
        { binding: 3, resource: { buffer: this.accumB } },
        { binding: 4, resource: { buffer: this.computeParams } },
      ],
    });

    const cp = encoder.beginComputePass();
    cp.setPipeline(this.computePipeline);
    cp.setBindGroup(0, computeBG);
    cp.dispatchWorkgroups(Math.ceil(srcW / 16), Math.ceil(srcH / 16));
    cp.end();

    const renderBG = d.createBindGroup({
      layout: this.renderBGL,
      entries: [
        { binding: 0, resource: { buffer: this.accumR } },
        { binding: 1, resource: { buffer: this.accumG } },
        { binding: 2, resource: { buffer: this.accumB } },
        { binding: 3, resource: { buffer: this.renderParams } },
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
    rp.setPipeline(this.renderPipeline);
    rp.setBindGroup(0, renderBG);
    rp.draw(3);
    rp.end();

    d.queue.submit([encoder.finish()]);
  }

  destroy() {
    const bufs = [
      this.accumR, this.accumG, this.accumB,
      this.computeParams, this.renderParams,
    ];
    for (const b of bufs) b?.destroy();
  }
}
