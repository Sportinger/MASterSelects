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
@group(0) @binding(5) var<storage, read_write> accumL: array<atomic<u32>>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.srcW || gid.y >= params.srcH) { return; }

  let pixel = textureLoad(inputTex, vec2i(gid.xy), 0);

  // Sub-pixel X: distribute weight across 2 adjacent columns (scale 256)
  let fxPos = f32(gid.x) * f32(params.outW) / f32(params.srcW);
  let x0 = u32(fxPos);
  let x1 = min(x0 + 1u, params.outW - 1u);
  let frac = fxPos - f32(x0);
  let w0 = u32((1.0 - frac) * 256.0);
  let w1 = 256u - w0;

  let hm1 = f32(params.outH - 1u);
  let maxY = i32(params.outH - 1u);

  // Gaussian vertical spread kernel — 5 rows for smooth DaVinci-style traces
  let gK = array<f32, 5>(0.06, 0.24, 0.40, 0.24, 0.06);

  // ── Red ──
  let ryC = i32(hm1 - clamp(pixel.r, 0.0, 1.0) * hm1);
  for (var d: i32 = -2; d <= 2; d += 1) {
    let y = u32(clamp(ryC + d, 0, maxY));
    let yw = gK[u32(d + 2)];
    let idx = y * params.outW;
    let wA = u32(f32(w0) * yw);
    let wB = u32(f32(w1) * yw);
    if (wA > 0u) { atomicAdd(&accumR[idx + x0], wA); }
    if (wB > 0u) { atomicAdd(&accumR[idx + x1], wB); }
  }

  // ── Green ──
  let gyC = i32(hm1 - clamp(pixel.g, 0.0, 1.0) * hm1);
  for (var d: i32 = -2; d <= 2; d += 1) {
    let y = u32(clamp(gyC + d, 0, maxY));
    let yw = gK[u32(d + 2)];
    let idx = y * params.outW;
    let wA = u32(f32(w0) * yw);
    let wB = u32(f32(w1) * yw);
    if (wA > 0u) { atomicAdd(&accumG[idx + x0], wA); }
    if (wB > 0u) { atomicAdd(&accumG[idx + x1], wB); }
  }

  // ── Blue ──
  let byC = i32(hm1 - clamp(pixel.b, 0.0, 1.0) * hm1);
  for (var d: i32 = -2; d <= 2; d += 1) {
    let y = u32(clamp(byC + d, 0, maxY));
    let yw = gK[u32(d + 2)];
    let idx = y * params.outW;
    let wA = u32(f32(w0) * yw);
    let wB = u32(f32(w1) * yw);
    if (wA > 0u) { atomicAdd(&accumB[idx + x0], wA); }
    if (wB > 0u) { atomicAdd(&accumB[idx + x1], wB); }
  }

  // ── Luma (BT.709) ──
  let luma = 0.2126 * clamp(pixel.r, 0.0, 1.0) + 0.7152 * clamp(pixel.g, 0.0, 1.0) + 0.0722 * clamp(pixel.b, 0.0, 1.0);
  let lyC = i32(hm1 - luma * hm1);
  for (var d: i32 = -2; d <= 2; d += 1) {
    let y = u32(clamp(lyC + d, 0, maxY));
    let yw = gK[u32(d + 2)];
    let idx = y * params.outW;
    let wA = u32(f32(w0) * yw);
    let wB = u32(f32(w1) * yw);
    if (wA > 0u) { atomicAdd(&accumL[idx + x0], wA); }
    if (wB > 0u) { atomicAdd(&accumL[idx + x1], wB); }
  }
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
  mode: u32,   // 0=RGB, 1=R, 2=G, 3=B, 4=Luma
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> accumR: array<u32>;
@group(0) @binding(1) var<storage, read> accumG: array<u32>;
@group(0) @binding(2) var<storage, read> accumB: array<u32>;
@group(0) @binding(3) var<uniform> params: RenderParams;
@group(0) @binding(4) var<storage, read> accumL: array<u32>;

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

// Nearest-neighbor read for bloom sampling
fn readAccum(acc: ptr<storage, array<u32>, read>, x: i32, y: i32, w: i32, h: i32) -> f32 {
  return f32((*acc)[u32(clamp(y, 0, h - 1)) * u32(w) + u32(clamp(x, 0, w - 1))]);
}

@fragment
fn fs(in: VertexOutput) -> @location(0) vec4f {
  let uv = in.uv;
  if (uv.x < 0.0 || uv.x >= 1.0 || uv.y < 0.0 || uv.y >= 1.0) {
    return vec4f(0.04, 0.04, 0.04, 1.0);
  }

  let w = u32(params.outW);
  let h = u32(params.outH);
  let iw = i32(w);
  let ih = i32(h);
  let mode = params.mode;

  // Floating-point grid position for bilinear sampling
  let fx = uv.x * params.outW - 0.5;
  let fy = uv.y * params.outH - 0.5;

  // Center value (bilinear — sharp trace)
  let rCenter = sampleAccum(&accumR, fx, fy, w, h);
  let gCenter = sampleAccum(&accumG, fx, fy, w, h);
  let bCenter = sampleAccum(&accumB, fx, fy, w, h);
  let lCenter = sampleAccum(&accumL, fx, fy, w, h);

  // Phosphor bloom: 3x3 gaussian at 4px step for soft glow
  let ix = i32(fx + 0.5);
  let iy = i32(fy + 0.5);
  var rBloom = 0.0; var gBloom = 0.0; var bBloom = 0.0; var lBloom = 0.0;
  let bK = array<f32, 3>(0.25, 0.50, 0.25);
  for (var by: i32 = -1; by <= 1; by += 1) {
    for (var bx: i32 = -1; bx <= 1; bx += 1) {
      let bw = bK[u32(bx + 1)] * bK[u32(by + 1)];
      rBloom += readAccum(&accumR, ix + bx * 4, iy + by * 4, iw, ih) * bw;
      gBloom += readAccum(&accumG, ix + bx * 4, iy + by * 4, iw, ih) * bw;
      bBloom += readAccum(&accumB, ix + bx * 4, iy + by * 4, iw, ih) * bw;
      lBloom += readAccum(&accumL, ix + bx * 4, iy + by * 4, iw, ih) * bw;
    }
  }

  let rv = params.refValue;
  let s = params.intensity;

  // Tone-map: main trace (sharp) + subtle bloom halo
  let rT = pow(clamp(sqrt(rCenter) / rv, 0.0, 1.0), 0.75) * s;
  let gT = pow(clamp(sqrt(gCenter) / rv, 0.0, 1.0), 0.75) * s;
  let bT = pow(clamp(sqrt(bCenter) / rv, 0.0, 1.0), 0.75) * s;
  let lT = pow(clamp(sqrt(lCenter) / rv, 0.0, 1.0), 0.75) * s;

  let rG = pow(clamp(sqrt(rBloom) / rv, 0.0, 1.0), 0.65) * 0.12;
  let gG = pow(clamp(sqrt(gBloom) / rv, 0.0, 1.0), 0.65) * 0.12;
  let bG = pow(clamp(sqrt(bBloom) / rv, 0.0, 1.0), 0.65) * 0.12;
  let lG = pow(clamp(sqrt(lBloom) / rv, 0.0, 1.0), 0.65) * 0.12;

  // Additive phosphor composite based on mode
  var color: vec3f;
  if (mode == 0u) {
    // RGB: all channels
    color = clamp(vec3f(rT + rG, gT + gG, bT + bG), vec3f(0.0), vec3f(1.0));
  } else if (mode == 1u) {
    // Red only
    let v = clamp(rT + rG, 0.0, 1.0);
    color = vec3f(v, v * 0.15, v * 0.15);
  } else if (mode == 2u) {
    // Green only
    let v = clamp(gT + gG, 0.0, 1.0);
    color = vec3f(v * 0.15, v, v * 0.15);
  } else if (mode == 3u) {
    // Blue only
    let v = clamp(bT + bG, 0.0, 1.0);
    color = vec3f(v * 0.15, v * 0.15, v);
  } else {
    // Luma: white trace
    let v = clamp(lT + lG, 0.0, 1.0);
    color = vec3f(v);
  }

  // Grid: every 10 IRE (10% of height)
  let gridY = fract(uv.y * 10.0);
  let dGrid = min(gridY, 1.0 - gridY) * params.outH * 0.5;
  if (dGrid < 0.8) {
    let a = 0.15 * (1.0 - dGrid / 0.8);
    color = max(color, vec3f(0.55, 0.45, 0.12) * a);
  }

  return vec4f(color, 1.0);
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

struct Params {
  totalPixels: f32,
  mode: f32,    // 0=RGB, 1=R, 2=G, 3=B, 4=Luma (as float for alignment)
  _pad1: f32,
  _pad2: f32,
}

@group(0) @binding(0) var<storage, read> histR: array<u32>;
@group(0) @binding(1) var<storage, read> histG: array<u32>;
@group(0) @binding(2) var<storage, read> histB: array<u32>;
@group(0) @binding(3) var<storage, read> histL: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;

// Linear interpolation between bins for smooth curves
fn sampleHist(hist: ptr<storage, array<u32>, read>, fx: f32) -> f32 {
  let b0 = u32(clamp(fx, 0.0, 255.0));
  let b1 = min(b0 + 1u, 255u);
  let t = fract(fx);
  return mix(f32((*hist)[b0]), f32((*hist)[b1]), t);
}

@fragment
fn fs(in: VertexOutput) -> @location(0) vec4f {
  let uv = in.uv;
  if (uv.x < 0.0 || uv.x >= 1.0 || uv.y < 0.0 || uv.y >= 1.0) {
    return vec4f(0.04, 0.04, 0.04, 1.0);
  }

  let mode = u32(params.mode);

  // Smooth bin position (linear interpolation between adjacent bins)
  let fx = uv.x * 255.0;
  let rVal = sampleHist(&histR, fx);
  let gVal = sampleHist(&histG, fx);
  let bVal = sampleHist(&histB, fx);
  let lVal = sampleHist(&histL, fx);

  // Sqrt scaling, normalized to total pixels (0.08 = expect peak bin ~8% of pixels)
  let scale = 1.0 / sqrt(params.totalPixels * 0.08);
  let rH = sqrt(rVal) * scale;
  let gH = sqrt(gVal) * scale;
  let bH = sqrt(bVal) * scale;
  let lH = sqrt(lVal) * scale;

  // Y coordinate: 0 = bottom (zero), 1 = top (highest count)
  let y = 1.0 - uv.y;

  // Anti-aliased edge width (in normalized Y units)
  let aaW = 0.004;

  // Filled area with soft anti-aliased top edge
  var color = vec3f(0.0);

  if (mode == 0u) {
    // RGB overlay: soft semi-transparent fills with additive blending
    let lFill = smoothstep(lH, lH - aaW, y);
    let rFill = smoothstep(rH, rH - aaW, y);
    let gFill = smoothstep(gH, gH - aaW, y);
    let bFill = smoothstep(bH, bH - aaW, y);
    // Gradient: brighter near top edge, dimmer at bottom
    let rGrad = 0.35 + 0.35 * (y / max(rH, 0.001));
    let gGrad = 0.35 + 0.35 * (y / max(gH, 0.001));
    let bGrad = 0.35 + 0.35 * (y / max(bH, 0.001));
    color += vec3f(0.08) * lFill;
    color += vec3f(rGrad, 0.05, 0.05) * rFill;
    color += vec3f(0.05, gGrad, 0.05) * gFill;
    color += vec3f(0.05, 0.05, bGrad) * bFill;
  } else if (mode == 1u) {
    let fill = smoothstep(rH, rH - aaW, y);
    let grad = 0.3 + 0.5 * (y / max(rH, 0.001));
    color = vec3f(grad, 0.08, 0.08) * fill;
  } else if (mode == 2u) {
    let fill = smoothstep(gH, gH - aaW, y);
    let grad = 0.3 + 0.5 * (y / max(gH, 0.001));
    color = vec3f(0.08, grad, 0.08) * fill;
  } else if (mode == 3u) {
    let fill = smoothstep(bH, bH - aaW, y);
    let grad = 0.3 + 0.5 * (y / max(bH, 0.001));
    color = vec3f(0.08, 0.08, grad) * fill;
  } else {
    let fill = smoothstep(lH, lH - aaW, y);
    let grad = 0.3 + 0.4 * (y / max(lH, 0.001));
    color = vec3f(grad) * fill;
  }

  // Bright edge glow at the top of each fill (phosphor-style)
  let edgeW = 0.006;
  if (mode == 0u) {
    let rEdge = smoothstep(edgeW, 0.0, abs(y - rH)) * step(y, rH + edgeW);
    let gEdge = smoothstep(edgeW, 0.0, abs(y - gH)) * step(y, gH + edgeW);
    let bEdge = smoothstep(edgeW, 0.0, abs(y - bH)) * step(y, bH + edgeW);
    color += vec3f(0.6, 0.12, 0.12) * rEdge;
    color += vec3f(0.12, 0.55, 0.12) * gEdge;
    color += vec3f(0.12, 0.12, 0.6) * bEdge;
  } else if (mode == 1u) {
    let e = smoothstep(edgeW, 0.0, abs(y - rH)) * step(y, rH + edgeW);
    color += vec3f(0.7, 0.18, 0.18) * e;
  } else if (mode == 2u) {
    let e = smoothstep(edgeW, 0.0, abs(y - gH)) * step(y, gH + edgeW);
    color += vec3f(0.18, 0.65, 0.18) * e;
  } else if (mode == 3u) {
    let e = smoothstep(edgeW, 0.0, abs(y - bH)) * step(y, bH + edgeW);
    color += vec3f(0.18, 0.18, 0.7) * e;
  } else {
    let e = smoothstep(edgeW, 0.0, abs(y - lH)) * step(y, lH + edgeW);
    color += vec3f(0.6) * e;
  }

  // Grid lines at 64, 128, 192 (anti-aliased)
  let gridBins = array<f32, 3>(64.0, 128.0, 192.0);
  for (var i = 0u; i < 3u; i++) {
    let gx = gridBins[i] / 256.0;
    let gAA = smoothstep(0.003, 0.001, abs(uv.x - gx));
    color = max(color, vec3f(0.10) * gAA);
  }
  // Horizontal grid at 25%, 50%, 75%
  for (var i = 1u; i < 4u; i++) {
    let gy = f32(i) * 0.25;
    let hAA = smoothstep(0.004, 0.001, abs(y - gy));
    color = max(color, vec3f(0.07) * hAA);
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

const DECAY_COMPUTE = /* wgsl */ `
struct Params { len: u32, _p0: u32, _p1: u32, _p2: u32 }

@group(0) @binding(0) var<storage, read_write> buf: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.len) { return; }
  // Temporal decay: no persistence (full clear)
  buf[gid.x] = 0u;
}
`;

// ──────────────── SCOPE RENDERER CLASS ────────────────

const OUT_W = 1024;
const OUT_H = 512;
const VS_SIZE = 512; // vectorscope grid size (high-res)

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
  private wfAccumL!: GPUBuffer;
  private wfComputeParams!: GPUBuffer;
  private wfRenderParams!: GPUBuffer;

  // Temporal decay
  private decayPipeline!: GPUComputePipeline;
  private decayBGL!: GPUBindGroupLayout;
  private decayParams!: GPUBuffer;

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
    this.initDecay();
  }

  // ──── WAVEFORM ────

  private initWaveform() {
    const d = this.device;
    const bufSize = OUT_W * OUT_H * 4;

    this.wfAccumR = d.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.wfAccumG = d.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.wfAccumB = d.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.wfAccumL = d.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.wfComputeParams = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    // 32 bytes: outW(f32) + outH(f32) + refValue(f32) + intensity(f32) + mode(u32) + 3x pad(u32)
    this.wfRenderParams = d.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    this.wfComputeBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
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
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      ],
    });

    const renderModule = d.createShaderModule({ code: WAVEFORM_RENDER });
    this.wfRenderPipeline = d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.wfRenderBGL] }),
      vertex: { module: renderModule, entryPoint: 'vs' },
      fragment: { module: renderModule, entryPoint: 'fs', targets: [{ format: this.format }] },
    });
  }

  private initDecay() {
    const d = this.device;
    this.decayBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });
    const module = d.createShaderModule({ code: DECAY_COMPUTE });
    this.decayPipeline = d.createComputePipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.decayBGL] }),
      compute: { module, entryPoint: 'main' },
    });
    this.decayParams = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  renderWaveform(sourceTexture: GPUTexture, ctx: GPUCanvasContext, mode: number = 0) {
    const d = this.device;
    const srcW = sourceTexture.width;
    const srcH = sourceTexture.height;

    d.queue.writeBuffer(this.wfComputeParams, 0, new Uint32Array([OUT_W, OUT_H, srcW, srcH]));
    // Vertical spread, no temporal accumulation
    const refValue = Math.sqrt(srcH / OUT_H) * 40.0;
    // Write 32 bytes: 4 floats + 1 u32 mode + 3 u32 padding
    const paramsData = new ArrayBuffer(32);
    new Float32Array(paramsData, 0, 4).set([OUT_W, OUT_H, refValue, 0.9]);
    new Uint32Array(paramsData, 16, 4).set([mode, 0, 0, 0]);
    d.queue.writeBuffer(this.wfRenderParams, 0, paramsData);

    const encoder = d.createCommandEncoder();

    // Temporal decay: fade previous frame instead of clearing (smooth persistence)
    const bufLen = OUT_W * OUT_H;
    d.queue.writeBuffer(this.decayParams, 0, new Uint32Array([bufLen, 0, 0, 0]));

    const decayBG_R = d.createBindGroup({
      layout: this.decayBGL,
      entries: [
        { binding: 0, resource: { buffer: this.wfAccumR } },
        { binding: 1, resource: { buffer: this.decayParams } },
      ],
    });
    const decayBG_G = d.createBindGroup({
      layout: this.decayBGL,
      entries: [
        { binding: 0, resource: { buffer: this.wfAccumG } },
        { binding: 1, resource: { buffer: this.decayParams } },
      ],
    });
    const decayBG_B = d.createBindGroup({
      layout: this.decayBGL,
      entries: [
        { binding: 0, resource: { buffer: this.wfAccumB } },
        { binding: 1, resource: { buffer: this.decayParams } },
      ],
    });
    const decayBG_L = d.createBindGroup({
      layout: this.decayBGL,
      entries: [
        { binding: 0, resource: { buffer: this.wfAccumL } },
        { binding: 1, resource: { buffer: this.decayParams } },
      ],
    });

    const dp = encoder.beginComputePass();
    dp.setPipeline(this.decayPipeline);
    dp.setBindGroup(0, decayBG_R);
    dp.dispatchWorkgroups(Math.ceil(bufLen / 256));
    dp.setBindGroup(0, decayBG_G);
    dp.dispatchWorkgroups(Math.ceil(bufLen / 256));
    dp.setBindGroup(0, decayBG_B);
    dp.dispatchWorkgroups(Math.ceil(bufLen / 256));
    dp.setBindGroup(0, decayBG_L);
    dp.dispatchWorkgroups(Math.ceil(bufLen / 256));
    dp.end();

    // Compute pass
    const computeBG = d.createBindGroup({
      layout: this.wfComputeBGL,
      entries: [
        { binding: 0, resource: sourceTexture.createView() },
        { binding: 1, resource: { buffer: this.wfAccumR } },
        { binding: 2, resource: { buffer: this.wfAccumG } },
        { binding: 3, resource: { buffer: this.wfAccumB } },
        { binding: 4, resource: { buffer: this.wfComputeParams } },
        { binding: 5, resource: { buffer: this.wfAccumL } },
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
        { binding: 4, resource: { buffer: this.wfAccumL } },
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

  renderHistogram(sourceTexture: GPUTexture, ctx: GPUCanvasContext, mode: number = 0) {
    const d = this.device;
    const srcW = sourceTexture.width;
    const srcH = sourceTexture.height;

    d.queue.writeBuffer(this.histComputeParams, 0, new Uint32Array([srcW, srcH, 0, 0]));
    d.queue.writeBuffer(this.histRenderParams, 0, new Float32Array([srcW * srcH, mode, 0, 0]));

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
    const refValue = Math.sqrt(srcH * srcW / (VS_SIZE * VS_SIZE)) * 18.0;
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
      this.wfAccumR, this.wfAccumG, this.wfAccumB, this.wfAccumL, this.wfComputeParams, this.wfRenderParams,
      this.histR, this.histG, this.histB, this.histL, this.histComputeParams, this.histRenderParams,
      this.vsAccumR, this.vsAccumG, this.vsAccumB, this.vsComputeParams, this.vsRenderParams,
      this.decayParams,
    ];
    for (const b of bufs) b?.destroy();
  }
}
