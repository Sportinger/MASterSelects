// WebGPU Rendering Engine for WebVJ Mixer - Optimized

import type { Layer, BlendMode, OutputWindow, EngineStats, Effect } from '../types';
import compositeShader from '../shaders/composite.wgsl?raw';
import outputShader from '../shaders/output.wgsl?raw';
import effectsShader from '../shaders/effects.wgsl?raw';

const BLEND_MODE_MAP: Record<BlendMode, number> = {
  // Normal
  'normal': 0,
  'dissolve': 1,
  'dancing-dissolve': 2,
  // Darken
  'darken': 3,
  'multiply': 4,
  'color-burn': 5,
  'classic-color-burn': 6,
  'linear-burn': 7,
  'darker-color': 8,
  // Lighten
  'add': 9,
  'lighten': 10,
  'screen': 11,
  'color-dodge': 12,
  'classic-color-dodge': 13,
  'linear-dodge': 14,
  'lighter-color': 15,
  // Contrast
  'overlay': 16,
  'soft-light': 17,
  'hard-light': 18,
  'linear-light': 19,
  'vivid-light': 20,
  'pin-light': 21,
  'hard-mix': 22,
  // Inversion
  'difference': 23,
  'classic-difference': 24,
  'exclusion': 25,
  'subtract': 26,
  'divide': 27,
  // Component
  'hue': 28,
  'saturation': 29,
  'color': 30,
  'luminosity': 31,
  // Stencil
  'stencil-alpha': 32,
  'stencil-luma': 33,
  'silhouette-alpha': 34,
  'silhouette-luma': 35,
  'alpha-add': 36,
};

// Composite shader for external video textures (true zero-copy)
const externalCompositeShader = `
struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

struct LayerUniforms {
  opacity: f32,
  blendMode: u32,
  posX: f32,
  posY: f32,
  scaleX: f32,
  scaleY: f32,
  rotation: f32,
  sourceAspect: f32,
  outputAspect: f32,
  time: f32,
  hasMask: u32,
  maskInvert: u32,
};

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var positions = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
  );
  var uvs = array<vec2f, 6>(
    vec2f(0.0, 1.0), vec2f(1.0, 1.0), vec2f(0.0, 0.0),
    vec2f(0.0, 0.0), vec2f(1.0, 1.0), vec2f(1.0, 0.0)
  );
  var output: VertexOutput;
  output.position = vec4f(positions[vertexIndex], 0.0, 1.0);
  output.uv = uvs[vertexIndex];
  return output;
}

@group(0) @binding(0) var texSampler: sampler;
@group(0) @binding(1) var baseTexture: texture_2d<f32>;
@group(0) @binding(2) var videoTexture: texture_external;
@group(0) @binding(3) var<uniform> layer: LayerUniforms;
@group(0) @binding(4) var maskTexture: texture_2d<f32>;

// ============ Utility Functions ============
fn hash(p: vec2f) -> f32 {
  var p3 = fract(vec3f(p.xyx) * 0.1031);
  p3 = p3 + dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

fn getLuminosity(c: vec3f) -> f32 {
  return 0.299 * c.r + 0.587 * c.g + 0.114 * c.b;
}

fn clipColor(c: vec3f) -> vec3f {
  let l = getLuminosity(c);
  let n = min(min(c.r, c.g), c.b);
  let x = max(max(c.r, c.g), c.b);
  var result = c;
  if (n < 0.0) { result = l + (((c - l) * l) / (l - n)); }
  if (x > 1.0) { result = l + (((c - l) * (1.0 - l)) / (x - l)); }
  return result;
}

fn setLuminosity(c: vec3f, l: f32) -> vec3f {
  let d = l - getLuminosity(c);
  return clipColor(c + vec3f(d));
}

fn rgbToHsl(c: vec3f) -> vec3f {
  let cMax = max(max(c.r, c.g), c.b);
  let cMin = min(min(c.r, c.g), c.b);
  let delta = cMax - cMin;
  var h: f32 = 0.0;
  var s: f32 = 0.0;
  let l = (cMax + cMin) / 2.0;
  if (delta > 0.0) {
    s = select(delta / (2.0 - cMax - cMin), delta / (cMax + cMin), l < 0.5);
    if (cMax == c.r) {
      h = ((c.g - c.b) / delta) + select(0.0, 6.0, c.g < c.b);
    } else if (cMax == c.g) {
      h = ((c.b - c.r) / delta) + 2.0;
    } else {
      h = ((c.r - c.g) / delta) + 4.0;
    }
    h = h / 6.0;
  }
  return vec3f(h, s, l);
}

fn hueToRgb(p: f32, q: f32, t: f32) -> f32 {
  var tt = t;
  if (tt < 0.0) { tt = tt + 1.0; }
  if (tt > 1.0) { tt = tt - 1.0; }
  if (tt < 1.0/6.0) { return p + (q - p) * 6.0 * tt; }
  if (tt < 1.0/2.0) { return q; }
  if (tt < 2.0/3.0) { return p + (q - p) * (2.0/3.0 - tt) * 6.0; }
  return p;
}

fn hslToRgb(hsl: vec3f) -> vec3f {
  if (hsl.y == 0.0) { return vec3f(hsl.z); }
  let q = select(hsl.z + hsl.y - hsl.z * hsl.y, hsl.z * (1.0 + hsl.y), hsl.z < 0.5);
  let p = 2.0 * hsl.z - q;
  return vec3f(
    hueToRgb(p, q, hsl.x + 1.0/3.0),
    hueToRgb(p, q, hsl.x),
    hueToRgb(p, q, hsl.x - 1.0/3.0)
  );
}

// ============ Blend Mode Functions ============
fn blendNormal(base: vec3f, blend: vec3f) -> vec3f { return blend; }
fn blendDissolve(base: vec3f, blend: vec3f, uv: vec2f, opacity: f32) -> vec3f {
  return select(base, blend, hash(uv * 1000.0) < opacity);
}
fn blendDancingDissolve(base: vec3f, blend: vec3f, uv: vec2f, opacity: f32, time: f32) -> vec3f {
  return select(base, blend, hash(uv * 1000.0 + vec2f(time * 60.0)) < opacity);
}
fn blendDarken(base: vec3f, blend: vec3f) -> vec3f { return min(base, blend); }
fn blendMultiply(base: vec3f, blend: vec3f) -> vec3f { return base * blend; }
fn blendColorBurn(base: vec3f, blend: vec3f) -> vec3f {
  return vec3f(
    select(1.0 - min(1.0, (1.0 - base.r) / blend.r), 0.0, blend.r == 0.0),
    select(1.0 - min(1.0, (1.0 - base.g) / blend.g), 0.0, blend.g == 0.0),
    select(1.0 - min(1.0, (1.0 - base.b) / blend.b), 0.0, blend.b == 0.0)
  );
}
fn blendClassicColorBurn(base: vec3f, blend: vec3f) -> vec3f {
  return 1.0 - (1.0 - base) / max(blend, vec3f(0.001));
}
fn blendLinearBurn(base: vec3f, blend: vec3f) -> vec3f {
  return max(base + blend - 1.0, vec3f(0.0));
}
fn blendDarkerColor(base: vec3f, blend: vec3f) -> vec3f {
  return select(blend, base, getLuminosity(base) < getLuminosity(blend));
}
fn blendAdd(base: vec3f, blend: vec3f) -> vec3f { return min(base + blend, vec3f(1.0)); }
fn blendLighten(base: vec3f, blend: vec3f) -> vec3f { return max(base, blend); }
fn blendScreen(base: vec3f, blend: vec3f) -> vec3f { return 1.0 - (1.0 - base) * (1.0 - blend); }
fn blendColorDodge(base: vec3f, blend: vec3f) -> vec3f {
  return vec3f(
    select(min(1.0, base.r / (1.0 - blend.r)), 1.0, blend.r == 1.0),
    select(min(1.0, base.g / (1.0 - blend.g)), 1.0, blend.g == 1.0),
    select(min(1.0, base.b / (1.0 - blend.b)), 1.0, blend.b == 1.0)
  );
}
fn blendClassicColorDodge(base: vec3f, blend: vec3f) -> vec3f {
  return base / max(1.0 - blend, vec3f(0.001));
}
fn blendLinearDodge(base: vec3f, blend: vec3f) -> vec3f { return min(base + blend, vec3f(1.0)); }
fn blendLighterColor(base: vec3f, blend: vec3f) -> vec3f {
  return select(blend, base, getLuminosity(base) > getLuminosity(blend));
}
fn blendOverlay(base: vec3f, blend: vec3f) -> vec3f {
  return vec3f(
    select(1.0 - 2.0 * (1.0 - base.r) * (1.0 - blend.r), 2.0 * base.r * blend.r, base.r < 0.5),
    select(1.0 - 2.0 * (1.0 - base.g) * (1.0 - blend.g), 2.0 * base.g * blend.g, base.g < 0.5),
    select(1.0 - 2.0 * (1.0 - base.b) * (1.0 - blend.b), 2.0 * base.b * blend.b, base.b < 0.5)
  );
}
fn blendSoftLight(base: vec3f, blend: vec3f) -> vec3f {
  let d = select(sqrt(base), ((16.0 * base - 12.0) * base + 4.0) * base, base <= vec3f(0.25));
  return select(base + (2.0 * blend - 1.0) * (d - base), base - (1.0 - 2.0 * blend) * base * (1.0 - base), blend <= vec3f(0.5));
}
fn blendHardLight(base: vec3f, blend: vec3f) -> vec3f {
  return vec3f(
    select(1.0 - 2.0 * (1.0 - base.r) * (1.0 - blend.r), 2.0 * base.r * blend.r, blend.r < 0.5),
    select(1.0 - 2.0 * (1.0 - base.g) * (1.0 - blend.g), 2.0 * base.g * blend.g, blend.g < 0.5),
    select(1.0 - 2.0 * (1.0 - base.b) * (1.0 - blend.b), 2.0 * base.b * blend.b, blend.b < 0.5)
  );
}
fn blendLinearLight(base: vec3f, blend: vec3f) -> vec3f {
  return clamp(base + 2.0 * blend - 1.0, vec3f(0.0), vec3f(1.0));
}
fn blendVividLight(base: vec3f, blend: vec3f) -> vec3f {
  return vec3f(
    select(blendColorDodge(base, 2.0 * (blend - 0.5)).r, blendColorBurn(base, 2.0 * blend).r, blend.r <= 0.5),
    select(blendColorDodge(base, 2.0 * (blend - 0.5)).g, blendColorBurn(base, 2.0 * blend).g, blend.g <= 0.5),
    select(blendColorDodge(base, 2.0 * (blend - 0.5)).b, blendColorBurn(base, 2.0 * blend).b, blend.b <= 0.5)
  );
}
fn blendPinLight(base: vec3f, blend: vec3f) -> vec3f {
  return vec3f(
    select(max(base.r, 2.0 * (blend.r - 0.5)), min(base.r, 2.0 * blend.r), blend.r <= 0.5),
    select(max(base.g, 2.0 * (blend.g - 0.5)), min(base.g, 2.0 * blend.g), blend.g <= 0.5),
    select(max(base.b, 2.0 * (blend.b - 0.5)), min(base.b, 2.0 * blend.b), blend.b <= 0.5)
  );
}
fn blendHardMix(base: vec3f, blend: vec3f) -> vec3f {
  return vec3f(
    select(0.0, 1.0, base.r + blend.r >= 1.0),
    select(0.0, 1.0, base.g + blend.g >= 1.0),
    select(0.0, 1.0, base.b + blend.b >= 1.0)
  );
}
fn blendDifference(base: vec3f, blend: vec3f) -> vec3f { return abs(base - blend); }
fn blendClassicDifference(base: vec3f, blend: vec3f) -> vec3f { return abs(base - blend); }
fn blendExclusion(base: vec3f, blend: vec3f) -> vec3f { return base + blend - 2.0 * base * blend; }
fn blendSubtract(base: vec3f, blend: vec3f) -> vec3f { return max(base - blend, vec3f(0.0)); }
fn blendDivide(base: vec3f, blend: vec3f) -> vec3f { return base / max(blend, vec3f(0.001)); }
fn blendHue(base: vec3f, blend: vec3f) -> vec3f {
  let baseHsl = rgbToHsl(base);
  let blendHsl = rgbToHsl(blend);
  return hslToRgb(vec3f(blendHsl.x, baseHsl.y, baseHsl.z));
}
fn blendSaturation(base: vec3f, blend: vec3f) -> vec3f {
  let baseHsl = rgbToHsl(base);
  let blendHsl = rgbToHsl(blend);
  return hslToRgb(vec3f(baseHsl.x, blendHsl.y, baseHsl.z));
}
fn blendColor(base: vec3f, blend: vec3f) -> vec3f {
  let baseHsl = rgbToHsl(base);
  let blendHsl = rgbToHsl(blend);
  return hslToRgb(vec3f(blendHsl.x, blendHsl.y, baseHsl.z));
}
fn blendLuminosity(base: vec3f, blend: vec3f) -> vec3f {
  let baseHsl = rgbToHsl(base);
  let blendHsl = rgbToHsl(blend);
  return hslToRgb(vec3f(baseHsl.x, baseHsl.y, blendHsl.z));
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  // Calculate video UV coordinates with all transformations
  var uv = input.uv;
  uv = uv - vec2f(0.5);

  let cosR = cos(layer.rotation);
  let sinR = sin(layer.rotation);
  uv = vec2f(uv.x * cosR - uv.y * sinR, uv.x * sinR + uv.y * cosR);
  uv = uv / vec2f(layer.scaleX, layer.scaleY);

  let aspectRatio = layer.sourceAspect / layer.outputAspect;
  if (aspectRatio > 1.0) {
    uv.y = uv.y * aspectRatio;
  } else {
    uv.x = uv.x / aspectRatio;
  }

  uv = uv + vec2f(0.5) - vec2f(layer.posX, layer.posY);

  let clampedUV = clamp(uv, vec2f(0.0), vec2f(1.0));
  let baseColor = textureSample(baseTexture, texSampler, input.uv);
  let layerColor = textureSampleBaseClampToEdge(videoTexture, texSampler, clampedUV);

  let outOfBounds = uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0;
  let maskAlpha = select(layerColor.a, 0.0, outOfBounds);

  var blended: vec3f;
  var finalAlpha: f32 = maskAlpha * layer.opacity;

  switch (layer.blendMode) {
    case 0u: { blended = blendNormal(baseColor.rgb, layerColor.rgb); }
    case 1u: { blended = blendDissolve(baseColor.rgb, layerColor.rgb, input.uv, layer.opacity); finalAlpha = 1.0; }
    case 2u: { blended = blendDancingDissolve(baseColor.rgb, layerColor.rgb, input.uv, layer.opacity, layer.time); finalAlpha = 1.0; }
    case 3u: { blended = blendDarken(baseColor.rgb, layerColor.rgb); }
    case 4u: { blended = blendMultiply(baseColor.rgb, layerColor.rgb); }
    case 5u: { blended = blendColorBurn(baseColor.rgb, layerColor.rgb); }
    case 6u: { blended = blendClassicColorBurn(baseColor.rgb, layerColor.rgb); }
    case 7u: { blended = blendLinearBurn(baseColor.rgb, layerColor.rgb); }
    case 8u: { blended = blendDarkerColor(baseColor.rgb, layerColor.rgb); }
    case 9u: { blended = blendAdd(baseColor.rgb, layerColor.rgb); }
    case 10u: { blended = blendLighten(baseColor.rgb, layerColor.rgb); }
    case 11u: { blended = blendScreen(baseColor.rgb, layerColor.rgb); }
    case 12u: { blended = blendColorDodge(baseColor.rgb, layerColor.rgb); }
    case 13u: { blended = blendClassicColorDodge(baseColor.rgb, layerColor.rgb); }
    case 14u: { blended = blendLinearDodge(baseColor.rgb, layerColor.rgb); }
    case 15u: { blended = blendLighterColor(baseColor.rgb, layerColor.rgb); }
    case 16u: { blended = blendOverlay(baseColor.rgb, layerColor.rgb); }
    case 17u: { blended = blendSoftLight(baseColor.rgb, layerColor.rgb); }
    case 18u: { blended = blendHardLight(baseColor.rgb, layerColor.rgb); }
    case 19u: { blended = blendLinearLight(baseColor.rgb, layerColor.rgb); }
    case 20u: { blended = blendVividLight(baseColor.rgb, layerColor.rgb); }
    case 21u: { blended = blendPinLight(baseColor.rgb, layerColor.rgb); }
    case 22u: { blended = blendHardMix(baseColor.rgb, layerColor.rgb); }
    case 23u: { blended = blendDifference(baseColor.rgb, layerColor.rgb); }
    case 24u: { blended = blendClassicDifference(baseColor.rgb, layerColor.rgb); }
    case 25u: { blended = blendExclusion(baseColor.rgb, layerColor.rgb); }
    case 26u: { blended = blendSubtract(baseColor.rgb, layerColor.rgb); }
    case 27u: { blended = blendDivide(baseColor.rgb, layerColor.rgb); }
    case 28u: { blended = blendHue(baseColor.rgb, layerColor.rgb); }
    case 29u: { blended = blendSaturation(baseColor.rgb, layerColor.rgb); }
    case 30u: { blended = blendColor(baseColor.rgb, layerColor.rgb); }
    case 31u: { blended = blendLuminosity(baseColor.rgb, layerColor.rgb); }
    case 32u: { blended = baseColor.rgb; finalAlpha = layerColor.a * layer.opacity; }
    case 33u: { blended = baseColor.rgb; finalAlpha = getLuminosity(layerColor.rgb) * layer.opacity; }
    case 34u: { blended = baseColor.rgb; finalAlpha = (1.0 - layerColor.a) * layer.opacity; }
    case 35u: { blended = baseColor.rgb; finalAlpha = (1.0 - getLuminosity(layerColor.rgb)) * layer.opacity; }
    case 36u: { blended = layerColor.rgb; finalAlpha = min(baseColor.a + layerColor.a * layer.opacity, 1.0); }
    default: { blended = layerColor.rgb; }
  }

  // Apply mask if present
  // Mask is in output frame coordinates, sample with input.uv
  if (layer.hasMask == 1u) {
    let maskSample = textureSample(maskTexture, texSampler, input.uv);
    var maskValue = maskSample.r;
    if (layer.maskInvert == 1u) {
      maskValue = 1.0 - maskValue;
    }
    finalAlpha = finalAlpha * maskValue;
  }

  if (layer.blendMode >= 32u && layer.blendMode <= 35u) {
    return vec4f(blended * finalAlpha, finalAlpha);
  }
  if (layer.blendMode == 36u) {
    return vec4f(mix(baseColor.rgb, blended, layerColor.a * layer.opacity), finalAlpha);
  }

  let alpha = select(finalAlpha, 0.0, outOfBounds);
  let result = mix(baseColor.rgb, blended, alpha);
  return vec4f(result, max(baseColor.a, alpha));
}
`;

export class WebGPUEngine {
  private device: GPUDevice | null = null;
  private adapter: GPUAdapter | null = null;

  // Main preview canvas
  private previewContext: GPUCanvasContext | null = null;

  // Render targets - ping pong buffers
  private pingTexture: GPUTexture | null = null;
  private pongTexture: GPUTexture | null = null;
  private blackTexture: GPUTexture | null = null;
  private whiteMaskTexture: GPUTexture | null = null;  // Fallback mask texture (fully white = no masking)
  private whiteMaskView: GPUTextureView | null = null;

  // Mask textures per layer
  private maskTextures: Map<string, GPUTexture> = new Map();
  private maskTextureViews: Map<string, GPUTextureView> = new Map();
  private lastMaskDebugLog: number | null = null;

  // Cached texture views
  private pingView: GPUTextureView | null = null;
  private pongView: GPUTextureView | null = null;

  // Pipelines
  private compositePipeline: GPURenderPipeline | null = null;
  private externalCompositePipeline: GPURenderPipeline | null = null;
  private outputPipeline: GPURenderPipeline | null = null;

  // Effect pipelines
  private effectPipelines: Map<string, GPURenderPipeline> = new Map();
  private effectBindGroupLayouts: Map<string, GPUBindGroupLayout> = new Map();

  // Resources
  private sampler: GPUSampler | null = null;
  private layerUniformBuffer: GPUBuffer | null = null;

  // Pre-allocated uniform data (12 x 4 bytes = 48 bytes)
  // Using ArrayBuffer with typed views to handle mixed float/uint data
  private uniformBuffer = new ArrayBuffer(48);
  private uniformData = new Float32Array(this.uniformBuffer);
  private uniformDataU32 = new Uint32Array(this.uniformBuffer);

  // Output windows
  private outputWindows: Map<string, OutputWindow> = new Map();

  // Stats
  private frameCount = 0;
  private fps = 0;
  private fpsUpdateTime = 0;

  // Detailed stats tracking
  private detailedStats = {
    rafGap: 0,
    importTexture: 0,
    renderPass: 0,
    submit: 0,
    total: 0,
    dropsTotal: 0,
    dropsLastSecond: 0,
    dropsThisSecond: 0,
    lastDropReason: 'none' as 'none' | 'slow_raf' | 'slow_render' | 'slow_import',
    lastRafTime: 0,
    decoder: 'none' as 'WebCodecs' | 'HTMLVideo' | 'HTMLVideo(cached)' | 'none',
  };
  private readonly TARGET_FRAME_TIME = 16.67; // 60fps target

  // Video frame textures (rendered from external textures)
  private videoFrameTextures: Map<string, GPUTexture> = new Map();
  private videoFrameViews: Map<string, GPUTextureView> = new Map();

  // Animation
  private animationId: number | null = null;
  private isRunning = false;
  private isInitialized = false;
  private initPromise: Promise<boolean> | null = null;

  // Performance profiling
  private profileData = {
    importTexture: 0,
    createBindGroup: 0,
    renderPass: 0,
    submit: 0,
    total: 0,
  };
  private profileCounter = 0;
  private lastProfileTime = 0;
  private lastRenderCall = 0;

  // Resolution
  private outputWidth = 1920;
  private outputHeight = 1080;

  // Bind group layout cache
  private compositeBindGroupLayout: GPUBindGroupLayout | null = null;
  private externalCompositeBindGroupLayout: GPUBindGroupLayout | null = null;
  private outputBindGroupLayout: GPUBindGroupLayout | null = null;

  // Cached bind groups to avoid per-frame creation
  private cachedCompositeBindGroups: Map<string, GPUBindGroup> = new Map();
  private cachedOutputBindGroupPing: GPUBindGroup | null = null;
  private cachedOutputBindGroupPong: GPUBindGroup | null = null;

  // Per-layer uniform buffers (fixes the shared buffer bug!)
  private layerUniformBuffers: Map<string, GPUBuffer> = new Map();

  // Cached image texture views
  private cachedImageViews: Map<GPUTexture, GPUTextureView> = new Map();

  // Cached image textures (created from HTMLImageElement)
  private imageTextures: Map<HTMLImageElement, GPUTexture> = new Map();

  // Ring buffer for frame times (avoids O(n) shift)
  private frameTimeBuffer = new Float32Array(60);
  private frameTimeIndex = 0;
  private frameTimeCount = 0;

  // Track which texture has the final composited frame (for export)
  private lastRenderWasPing = false;

  // === FRAME CACHING FOR SMOOTH SCRUBBING ===
  // Last valid frame cache - keeps last frame visible during seeks
  private lastFrameTextures: Map<HTMLVideoElement, GPUTexture> = new Map();
  private lastFrameViews: Map<HTMLVideoElement, GPUTextureView> = new Map();
  private lastFrameSizes: Map<HTMLVideoElement, { width: number; height: number }> = new Map();
  private lastCaptureTime: Map<HTMLVideoElement, number> = new Map();

  // Scrubbing frame cache - pre-decoded frames for instant access
  // Key: "videoSrc:frameTime" -> GPUTexture
  private scrubbingCache: Map<string, GPUTexture> = new Map();
  private scrubbingCacheViews: Map<string, GPUTextureView> = new Map();
  private scrubbingCacheOrder: string[] = []; // LRU order
  private maxScrubbingCacheFrames = 300; // ~10 seconds at 30fps, ~2.4GB VRAM at 1080p

  // RAM Preview cache - fully composited frames for instant playback
  // Key: time (quantized to frame) -> ImageData (CPU-side for memory efficiency)
  private compositeCache: Map<number, ImageData> = new Map();
  private compositeCacheOrder: number[] = []; // LRU order
  private maxCompositeCacheFrames = 900; // 30 seconds at 30fps

  // GPU texture cache for instant RAM Preview playback (no CPU->GPU upload needed)
  // Limited size to conserve VRAM (~500MB at 1080p for 60 frames)
  private gpuFrameCache: Map<number, { texture: GPUTexture; view: GPUTextureView; bindGroup: GPUBindGroup }> = new Map();
  private gpuFrameCacheOrder: number[] = []; // LRU order
  private maxGpuCacheFrames = 60; // ~500MB at 1080p

  // Reusable resources for RAM Preview playback (avoid creating per-frame)
  private ramPlaybackCanvas: HTMLCanvasElement | null = null;
  private ramPlaybackCtx: CanvasRenderingContext2D | null = null;
  private ramPlaybackTexture: GPUTexture | null = null;
  private ramPlaybackTextureView: GPUTextureView | null = null;
  private ramPlaybackBindGroup: GPUBindGroup | null = null;
  private ramPlaybackTextureSize: { width: number; height: number } | null = null;

  async initialize(): Promise<boolean> {
    // Prevent multiple initializations with promise-based lock
    if (this.isInitialized && this.device) {
      console.log('[WebGPU] Already initialized, skipping');
      return true;
    }

    // If initialization is in progress, wait for it
    if (this.initPromise) {
      console.log('[WebGPU] Initialization in progress, waiting...');
      return this.initPromise;
    }

    if (!navigator.gpu) {
      console.error('WebGPU not supported');
      return false;
    }

    // Create the initialization promise
    this.initPromise = this.doInitialize();
    return this.initPromise;
  }

  private async doInitialize(): Promise<boolean> {
    try {
      this.adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance',
      });

      if (!this.adapter) {
        console.error('Failed to get GPU adapter');
        return false;
      }

      this.device = await this.adapter.requestDevice({
        requiredFeatures: [],
        requiredLimits: {
          maxTextureDimension2D: 4096,
        },
      });

      this.device.lost.then((info) => {
        console.error('WebGPU device lost:', info.message);
        this.isInitialized = false;
      });

      await this.createResources();
      this.isInitialized = true;
      console.log('[WebGPU] Engine initialized successfully');
      return true;
    } catch (error) {
      console.error('Failed to initialize WebGPU:', error);
      this.initPromise = null;
      return false;
    }
  }

  private async createResources(): Promise<void> {
    if (!this.device) return;

    // Create sampler
    this.sampler = this.device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    // Create layer uniform buffer (48 bytes for 12 floats)
    this.layerUniformBuffer = this.device.createBuffer({
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Create black texture
    this.blackTexture = this.device.createTexture({
      size: [1, 1],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    this.device.queue.writeTexture(
      { texture: this.blackTexture },
      new Uint8Array([0, 0, 0, 255]),
      { bytesPerRow: 4 },
      [1, 1]
    );

    // Create white mask texture (fallback for layers without masks)
    this.whiteMaskTexture = this.device.createTexture({
      size: [1, 1],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    this.device.queue.writeTexture(
      { texture: this.whiteMaskTexture },
      new Uint8Array([255, 255, 255, 255]),  // Pure white = fully visible
      { bytesPerRow: 4 },
      [1, 1]
    );

    this.whiteMaskView = this.whiteMaskTexture.createView();

    this.createPingPongTextures();
    await this.createPipelines();
  }

  private createPingPongTextures(): void {
    if (!this.device) return;

    this.pingTexture?.destroy();
    this.pongTexture?.destroy();

    this.pingTexture = this.device.createTexture({
      size: [this.outputWidth, this.outputHeight],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });

    this.pongTexture = this.device.createTexture({
      size: [this.outputWidth, this.outputHeight],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });

    // Cache views
    this.pingView = this.pingTexture.createView();
    this.pongView = this.pongTexture.createView();

    // Invalidate bind group caches (they reference old textures)
    this.cachedCompositeBindGroups.clear();
    this.cachedOutputBindGroupPing = null;
    this.cachedOutputBindGroupPong = null;
  }

  private async createPipelines(): Promise<void> {
    if (!this.device) return;

    // Composite bind group layout
    this.compositeBindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: {} },  // Mask texture
      ],
    });

    // Composite pipeline
    const compositeModule = this.device.createShaderModule({ code: compositeShader });

    this.compositePipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [this.compositeBindGroupLayout],
      }),
      vertex: { module: compositeModule, entryPoint: 'vertexMain' },
      fragment: {
        module: compositeModule,
        entryPoint: 'fragmentMain',
        targets: [{ format: 'rgba8unorm' }],
      },
      primitive: { topology: 'triangle-list' },
    });

    // Output bind group layout
    this.outputBindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
      ],
    });

    // Output pipeline
    const outputModule = this.device.createShaderModule({ code: outputShader });

    this.outputPipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [this.outputBindGroupLayout],
      }),
      vertex: { module: outputModule, entryPoint: 'vertexMain' },
      fragment: {
        module: outputModule,
        entryPoint: 'fragmentMain',
        targets: [{ format: 'bgra8unorm' }],
      },
      primitive: { topology: 'triangle-list' },
    });

    // External composite pipeline (for direct video texture compositing - zero copy!)
    this.externalCompositeBindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, externalTexture: {} },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: {} },  // Mask texture
      ],
    });

    const externalCompositeModule = this.device.createShaderModule({ code: externalCompositeShader });

    this.externalCompositePipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [this.externalCompositeBindGroupLayout],
      }),
      vertex: { module: externalCompositeModule, entryPoint: 'vertexMain' },
      fragment: {
        module: externalCompositeModule,
        entryPoint: 'fragmentMain',
        targets: [{ format: 'rgba8unorm' }],
      },
      primitive: { topology: 'triangle-list' },
    });

    // Create effect pipelines
    await this.createEffectPipelines();
  }

  private async createEffectPipelines(): Promise<void> {
    if (!this.device) return;

    const effectModule = this.device.createShaderModule({ code: effectsShader });

    // Define effect configurations: [entryPoint, needsUniform, uniformSize]
    const effectConfigs: Record<string, [string, boolean, number]> = {
      'hue-shift': ['hueShiftFragment', true, 16],
      'brightness': ['colorAdjustFragment', true, 16],
      'contrast': ['colorAdjustFragment', true, 16],
      'saturation': ['colorAdjustFragment', true, 16],
      'pixelate': ['pixelateFragment', true, 16],
      'kaleidoscope': ['kaleidoscopeFragment', true, 16],
      'mirror': ['mirrorFragment', true, 16],
      'rgb-split': ['rgbSplitFragment', true, 16],
      'invert': ['invertFragment', false, 0],
      'levels': ['levelsFragment', true, 32],
    };

    for (const [effectType, [entryPoint, needsUniform, uniformSize]] of Object.entries(effectConfigs)) {
      // Create bind group layout
      const entries: GPUBindGroupLayoutEntry[] = [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
      ];

      if (needsUniform) {
        entries.push({ binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } });
      }

      const bindGroupLayout = this.device.createBindGroupLayout({ entries });
      this.effectBindGroupLayouts.set(effectType, bindGroupLayout);

      // Create pipeline
      const pipeline = this.device.createRenderPipeline({
        layout: this.device.createPipelineLayout({
          bindGroupLayouts: [bindGroupLayout],
        }),
        vertex: { module: effectModule, entryPoint: 'vertexMain' },
        fragment: {
          module: effectModule,
          entryPoint,
          targets: [{ format: 'rgba8unorm' }],
        },
        primitive: { topology: 'triangle-list' },
      });

      this.effectPipelines.set(effectType, pipeline);
    }
  }

  setPreviewCanvas(canvas: HTMLCanvasElement): void {
    if (!this.device) return;

    this.previewContext = canvas.getContext('webgpu');

    if (this.previewContext) {
      this.previewContext.configure({
        device: this.device,
        format: 'bgra8unorm',
        alphaMode: 'premultiplied',
      });
    }
  }

  // Update mask texture for a layer
  updateMaskTexture(layerId: string, imageData: ImageData | null): void {
    if (!this.device) return;

    // Remove existing mask texture
    const existingTexture = this.maskTextures.get(layerId);
    if (existingTexture) {
      existingTexture.destroy();
      this.maskTextures.delete(layerId);
      this.maskTextureViews.delete(layerId);
    }

    // If no imageData, layer will use white fallback (no masking)
    if (!imageData) {
      console.log(`[Engine] No mask data for layer ${layerId}, using white fallback`);
      return;
    }

    // Create new mask texture
    const maskTexture = this.device.createTexture({
      size: [imageData.width, imageData.height],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    // Upload mask data
    this.device.queue.writeTexture(
      { texture: maskTexture },
      imageData.data,
      {
        bytesPerRow: imageData.width * 4,
        rowsPerImage: imageData.height,
      },
      [imageData.width, imageData.height]
    );

    // Cache texture and view
    this.maskTextures.set(layerId, maskTexture);
    this.maskTextureViews.set(layerId, maskTexture.createView());

    console.log(`[Engine] Uploaded mask texture for layer ${layerId}: ${imageData.width}x${imageData.height}`);
  }

  // Remove mask texture for a layer
  removeMaskTexture(layerId: string): void {
    const existingTexture = this.maskTextures.get(layerId);
    if (existingTexture) {
      existingTexture.destroy();
      this.maskTextures.delete(layerId);
      this.maskTextureViews.delete(layerId);
    }
  }

  createOutputWindow(id: string, name: string): OutputWindow | null {
    const outputWindow = window.open(
      '',
      `output_${id}`,
      'width=960,height=540,menubar=no,toolbar=no,location=no,status=no'
    );

    if (!outputWindow) {
      console.error('Failed to open output window (popup blocked?)');
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

      // Determine which dimension changed more
      const widthDelta = Math.abs(currentWidth - lastWidth);
      const heightDelta = Math.abs(currentHeight - lastHeight);

      let newWidth: number;
      let newHeight: number;

      if (widthDelta >= heightDelta) {
        // Width changed - adjust height to match
        newWidth = currentWidth;
        newHeight = Math.round(currentWidth / aspectRatio);
      } else {
        // Height changed - adjust width to match
        newHeight = currentHeight;
        newWidth = Math.round(currentHeight * aspectRatio);
      }

      // Resize window to enforce aspect ratio
      if (newWidth !== currentWidth || newHeight !== currentHeight) {
        outputWindow.resizeTo(newWidth + (outputWindow.outerWidth - currentWidth),
                              newHeight + (outputWindow.outerHeight - currentHeight));
      }

      // Update canvas to fill window
      canvas.style.width = '100%';
      canvas.style.height = '100%';

      lastWidth = newWidth;
      lastHeight = newHeight;

      setTimeout(() => { resizing = false; }, 50);
    };

    // Initial setup
    canvas.style.width = '100%';
    canvas.style.height = '100%';

    // Enforce on resize
    outputWindow.addEventListener('resize', enforceAspectRatio);

    let context: GPUCanvasContext | null = null;

    if (this.device) {
      context = canvas.getContext('webgpu');
      if (context) {
        context.configure({
          device: this.device,
          format: 'bgra8unorm',
          alphaMode: 'premultiplied',
        });
      }
    }

    const fullscreenBtn = outputWindow.document.createElement('button');
    fullscreenBtn.textContent = 'Fullscreen';
    fullscreenBtn.style.cssText =
      'position:fixed;top:10px;right:10px;padding:8px 16px;cursor:pointer;z-index:1000;opacity:0.7;';
    fullscreenBtn.onclick = () => {
      canvas.requestFullscreen();
    };
    outputWindow.document.body.appendChild(fullscreenBtn);

    // Hide button in fullscreen
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

  // Create GPU texture from HTMLImageElement
  createImageTexture(image: HTMLImageElement): GPUTexture | null {
    // Use naturalWidth/naturalHeight for images not added to DOM (like proxy frames)
    const width = image.naturalWidth || image.width;
    const height = image.naturalHeight || image.height;

    if (!this.device || width === 0 || height === 0) return null;

    // Check cache first
    const cached = this.imageTextures.get(image);
    if (cached) return cached;

    try {
      const texture = this.device.createTexture({
        size: [width, height],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      });

      this.device.queue.copyExternalImageToTexture(
        { source: image },
        { texture },
        [width, height]
      );

      this.imageTextures.set(image, texture);
      return texture;
    } catch (e) {
      console.error('Failed to create image texture:', e);
      return null;
    }
  }

  // Import external texture - true zero-copy from video decoder
  // Supports both HTMLVideoElement and VideoFrame (from WebCodecs)
  importVideoTexture(source: HTMLVideoElement | VideoFrame): GPUExternalTexture | null {
    if (!this.device) return null;

    // Check if source is valid
    if (source instanceof HTMLVideoElement) {
      // readyState >= 2 means HAVE_CURRENT_DATA (has at least one frame)
      // Also check we're not in middle of seeking which can cause blank frames
      if (source.readyState < 2 || source.videoWidth === 0 || source.videoHeight === 0) {
        return null;
      }
      // Skip if video is seeking - frame might not be ready
      if (source.seeking) {
        return null;
      }
    } else if (source instanceof VideoFrame) {
      if (source.codedWidth === 0 || source.codedHeight === 0) {
        return null;
      }
    } else {
      return null;
    }

    try {
      return this.device.importExternalTexture({ source });
    } catch {
      // Silently fail - video may not be ready yet
      return null;
    }
  }

  // Capture current video frame to a persistent GPU texture (for last-frame cache)
  private captureVideoFrame(video: HTMLVideoElement): void {
    if (!this.device || video.videoWidth === 0 || video.videoHeight === 0) return;

    const width = video.videoWidth;
    const height = video.videoHeight;

    // Get or create texture for this video
    let texture = this.lastFrameTextures.get(video);
    const existingSize = this.lastFrameSizes.get(video);

    // Recreate if size changed
    if (!texture || !existingSize || existingSize.width !== width || existingSize.height !== height) {
      texture?.destroy();
      texture = this.device.createTexture({
        size: [width, height],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      this.lastFrameTextures.set(video, texture);
      this.lastFrameSizes.set(video, { width, height });
      this.lastFrameViews.set(video, texture.createView());
    }

    // Copy current frame to texture
    try {
      this.device.queue.copyExternalImageToTexture(
        { source: video },
        { texture },
        [width, height]
      );
    } catch {
      // Video might not be ready - ignore
    }
  }

  // Get last cached frame for a video (used during seeks)
  private getLastFrame(video: HTMLVideoElement): { view: GPUTextureView; width: number; height: number } | null {
    const view = this.lastFrameViews.get(video);
    const size = this.lastFrameSizes.get(video);
    if (view && size) {
      return { view, width: size.width, height: size.height };
    }
    return null;
  }

  // Cleanup resources for a video that's no longer used
  cleanupVideo(video: HTMLVideoElement): void {
    // Destroy GPU textures
    const texture = this.lastFrameTextures.get(video);
    if (texture) {
      texture.destroy();
      this.lastFrameTextures.delete(video);
    }
    this.lastFrameViews.delete(video);
    this.lastFrameSizes.delete(video);
    this.lastCaptureTime.delete(video);

    // Cleanup frame tracking
    this.videoFrameReady.delete(video);
    this.videoLastTime.delete(video);
    this.videoCallbackActive.delete(video);

    console.log('[WebGPU] Cleaned up video resources');
  }

  // Clear all caches (call periodically to prevent memory buildup)
  clearCaches(): void {
    // Clear scrubbing cache
    for (const texture of this.scrubbingCache.values()) {
      texture.destroy();
    }
    this.scrubbingCache.clear();
    this.scrubbingCacheViews.clear();
    this.scrubbingCacheOrder.length = 0;

    // Clear composite cache
    this.compositeCache.clear();
    this.compositeCacheOrder.length = 0;

    // Clear GPU frame cache
    for (const entry of this.gpuFrameCache.values()) {
      entry.texture.destroy();
    }
    this.gpuFrameCache.clear();
    this.gpuFrameCacheOrder.length = 0;

    console.log('[WebGPU] Cleared all caches');
  }

  // === SCRUBBING FRAME CACHE ===
  // Cache a frame at a specific time for instant scrubbing access
  cacheFrameAtTime(video: HTMLVideoElement, time: number): void {
    if (!this.device || video.videoWidth === 0 || video.readyState < 2) return;

    const key = `${video.src}:${time.toFixed(3)}`;
    if (this.scrubbingCache.has(key)) return; // Already cached

    const width = video.videoWidth;
    const height = video.videoHeight;

    // Create texture for this frame
    const texture = this.device.createTexture({
      size: [width, height],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });

    try {
      this.device.queue.copyExternalImageToTexture(
        { source: video },
        { texture },
        [width, height]
      );

      // Add to cache
      this.scrubbingCache.set(key, texture);
      this.scrubbingCacheViews.set(key, texture.createView());
      this.scrubbingCacheOrder.push(key);

      // LRU eviction
      while (this.scrubbingCacheOrder.length > this.maxScrubbingCacheFrames) {
        const oldKey = this.scrubbingCacheOrder.shift()!;
        const oldTexture = this.scrubbingCache.get(oldKey);
        oldTexture?.destroy();
        this.scrubbingCache.delete(oldKey);
        this.scrubbingCacheViews.delete(oldKey);
      }
    } catch {
      texture.destroy();
    }
  }

  // Get cached frame for scrubbing
  getCachedFrame(videoSrc: string, time: number): GPUTextureView | null {
    const key = `${videoSrc}:${time.toFixed(3)}`;
    const view = this.scrubbingCacheViews.get(key);
    if (view) {
      // Move to end of LRU list
      const idx = this.scrubbingCacheOrder.indexOf(key);
      if (idx > -1) {
        this.scrubbingCacheOrder.splice(idx, 1);
        this.scrubbingCacheOrder.push(key);
      }
      return view;
    }
    return null;
  }

  // Pre-cache frames around a time point for smooth scrubbing
  preCacheFrames(video: HTMLVideoElement, centerTime: number, radiusSeconds: number = 2, fps: number = 30): void {
    if (!this.device || !video.src) return;

    const frameInterval = 1 / fps;
    const startTime = Math.max(0, centerTime - radiusSeconds);
    const endTime = Math.min(video.duration || 0, centerTime + radiusSeconds);

    // Queue frames to cache (will be cached as video seeks to each position)
    const framesToCache: number[] = [];
    for (let t = startTime; t <= endTime; t += frameInterval) {
      const key = `${video.src}:${t.toFixed(3)}`;
      if (!this.scrubbingCache.has(key)) {
        framesToCache.push(t);
      }
    }

    // Return the list for external caching logic
    return;
  }

  // Get scrubbing cache stats
  getScrubbingCacheStats(): { count: number; maxCount: number } {
    return {
      count: this.scrubbingCache.size,
      maxCount: this.maxScrubbingCacheFrames,
    };
  }

  // Clear scrubbing cache for a specific video
  clearScrubbingCache(videoSrc?: string): void {
    if (videoSrc) {
      // Clear only frames from this video
      const keysToRemove = this.scrubbingCacheOrder.filter(k => k.startsWith(videoSrc));
      keysToRemove.forEach(key => {
        const texture = this.scrubbingCache.get(key);
        texture?.destroy();
        this.scrubbingCache.delete(key);
        this.scrubbingCacheViews.delete(key);
      });
      this.scrubbingCacheOrder = this.scrubbingCacheOrder.filter(k => !k.startsWith(videoSrc));
    } else {
      // Clear all
      this.scrubbingCache.forEach(t => t.destroy());
      this.scrubbingCache.clear();
      this.scrubbingCacheViews.clear();
      this.scrubbingCacheOrder = [];
    }
  }

  // === RAM PREVIEW COMPOSITE CACHE ===
  // Cache fully composited frames for instant scrubbing

  // Quantize time to frame number at 30fps for cache key
  private quantizeTime(time: number): number {
    return Math.round(time * 30) / 30;
  }

  // Cache the current composited frame at a specific time
  async cacheCompositeFrame(time: number): Promise<void> {
    if (!this.device || !this.pingTexture || !this.pongTexture) return;

    const key = this.quantizeTime(time);
    if (this.compositeCache.has(key)) return; // Already cached

    // Read pixels from current composite result
    const pixels = await this.readPixels();
    if (!pixels) return;

    // Debug: check if pixels have data
    if (this.compositeCache.size === 0) {
      let nonZero = 0;
      for (let i = 0; i < Math.min(1000, pixels.length); i++) {
        if (pixels[i] !== 0) nonZero++;
      }
      console.log(`[RAM Preview] First frame: ${nonZero} non-zero pixels in first 1000, size: ${this.outputWidth}x${this.outputHeight}`);
    }

    // Create ImageData for CPU-side storage
    const imageData = new ImageData(
      new Uint8ClampedArray(pixels),
      this.outputWidth,
      this.outputHeight
    );

    // Add to cache with LRU eviction
    this.compositeCache.set(key, imageData);
    this.compositeCacheOrder.push(key);

    // Evict old frames if over limit
    while (this.compositeCacheOrder.length > this.maxCompositeCacheFrames) {
      const oldKey = this.compositeCacheOrder.shift()!;
      this.compositeCache.delete(oldKey);
    }
  }

  // Get cached composite frame if available
  getCachedCompositeFrame(time: number): ImageData | null {
    const key = this.quantizeTime(time);
    const imageData = this.compositeCache.get(key);

    if (imageData) {
      // Move to end of LRU order
      const idx = this.compositeCacheOrder.indexOf(key);
      if (idx > -1) {
        this.compositeCacheOrder.splice(idx, 1);
        this.compositeCacheOrder.push(key);
      }
      return imageData;
    }
    return null;
  }

  // Check if a frame is cached
  hasCompositeCacheFrame(time: number): boolean {
    return this.compositeCache.has(this.quantizeTime(time));
  }

  // Clear composite cache
  clearCompositeCache(): void {
    this.compositeCache.clear();
    this.compositeCacheOrder = [];

    // Clear GPU frame cache
    for (const entry of this.gpuFrameCache.values()) {
      entry.texture.destroy();
    }
    this.gpuFrameCache.clear();
    this.gpuFrameCacheOrder = [];

    // Also clear reusable playback resources
    this.ramPlaybackTexture?.destroy();
    this.ramPlaybackTexture = null;
    this.ramPlaybackTextureView = null;
    this.ramPlaybackBindGroup = null;
    this.ramPlaybackTextureSize = null;
    this.ramPlaybackCanvas = null;
    this.ramPlaybackCtx = null;
    console.log('[WebGPU] Composite cache cleared');
  }

  // Get composite cache stats
  getCompositeCacheStats(): { count: number; maxFrames: number; memoryMB: number } {
    const count = this.compositeCache.size;
    // Each frame is width * height * 4 bytes (RGBA)
    const bytesPerFrame = this.outputWidth * this.outputHeight * 4;
    const memoryMB = (count * bytesPerFrame) / (1024 * 1024);
    return { count, maxFrames: this.maxCompositeCacheFrames, memoryMB };
  }

  // Flag to skip preview updates during RAM preview generation
  private isGeneratingRamPreview = false;

  setGeneratingRamPreview(generating: boolean): void {
    this.isGeneratingRamPreview = generating;
  }

  // Render cached frame to preview canvas if available
  // Returns true if cached frame was used, false if live render needed
  renderCachedFrame(time: number): boolean {
    const key = this.quantizeTime(time);

    if (!this.previewContext || !this.device) {
      return false;
    }

    // First, check GPU cache for instant playback (no upload needed)
    const gpuCached = this.gpuFrameCache.get(key);
    if (gpuCached) {
      // Update LRU order
      const idx = this.gpuFrameCacheOrder.indexOf(key);
      if (idx > -1) {
        this.gpuFrameCacheOrder.splice(idx, 1);
        this.gpuFrameCacheOrder.push(key);
      }

      // Instant render from GPU cache
      const commandEncoder = this.device.createCommandEncoder();
      this.renderToCanvasCached(commandEncoder, this.previewContext, gpuCached.bindGroup);
      for (const output of this.outputWindows.values()) {
        if (output.context) {
          this.renderToCanvasCached(commandEncoder, output.context, gpuCached.bindGroup);
        }
      }
      this.device.queue.submit([commandEncoder.finish()]);
      return true;
    }

    // Fall back to CPU cache and upload to GPU
    const imageData = this.compositeCache.get(key);
    if (!imageData) {
      return false;
    }

    try {
      const width = imageData.width;
      const height = imageData.height;

      // Reuse or create canvas for ImageData -> GPU transfer
      if (!this.ramPlaybackCanvas || !this.ramPlaybackCtx) {
        this.ramPlaybackCanvas = document.createElement('canvas');
        this.ramPlaybackCanvas.width = width;
        this.ramPlaybackCanvas.height = height;
        this.ramPlaybackCtx = this.ramPlaybackCanvas.getContext('2d', { willReadFrequently: false });
      } else if (this.ramPlaybackCanvas.width !== width || this.ramPlaybackCanvas.height !== height) {
        this.ramPlaybackCanvas.width = width;
        this.ramPlaybackCanvas.height = height;
      }

      if (!this.ramPlaybackCtx) return false;

      // Put imageData to canvas
      this.ramPlaybackCtx.putImageData(imageData, 0, 0);

      // Create a new GPU texture for this frame and add to GPU cache
      const texture = this.device.createTexture({
        size: [width, height],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      });

      // Copy canvas to GPU texture
      this.device.queue.copyExternalImageToTexture(
        { source: this.ramPlaybackCanvas },
        { texture },
        [width, height]
      );

      const view = texture.createView();
      const bindGroup = this.device.createBindGroup({
        layout: this.outputBindGroupLayout!,
        entries: [
          { binding: 0, resource: this.sampler! },
          { binding: 1, resource: view },
        ],
      });

      // Add to GPU cache with LRU eviction
      this.gpuFrameCache.set(key, { texture, view, bindGroup });
      this.gpuFrameCacheOrder.push(key);

      // Evict oldest GPU cached frames if over limit
      while (this.gpuFrameCacheOrder.length > this.maxGpuCacheFrames) {
        const oldKey = this.gpuFrameCacheOrder.shift()!;
        const oldEntry = this.gpuFrameCache.get(oldKey);
        oldEntry?.texture.destroy();
        this.gpuFrameCache.delete(oldKey);
      }

      // Render texture to preview
      const commandEncoder = this.device.createCommandEncoder();
      this.renderToCanvasCached(commandEncoder, this.previewContext, bindGroup);

      // Also render to output windows
      for (const output of this.outputWindows.values()) {
        if (output.context) {
          this.renderToCanvasCached(commandEncoder, output.context, bindGroup);
        }
      }

      this.device.queue.submit([commandEncoder.finish()]);

      return true;
    } catch (e) {
      console.warn('[WebGPU] Failed to render cached frame:', e);
      return false;
    }
  }

  // Reusable layer data array to avoid allocations
  private layerRenderData: Array<{
    layer: Layer;
    isVideo: boolean;
    externalTexture: GPUExternalTexture | null;
    textureView: GPUTextureView | null;
    sourceWidth: number;
    sourceHeight: number;
  }> = [];

  render(layers: Layer[]): void {
    if (!this.device || !this.externalCompositePipeline || !this.outputPipeline) return;
    if (!this.pingView || !this.pongView) return;

    const t0 = performance.now();
    const timeSinceLastRender = this.lastRenderCall > 0 ? t0 - this.lastRenderCall : 0;
    this.lastRenderCall = t0;

    // Reuse array, just clear it
    this.layerRenderData.length = 0;

    // Prepare layer data - import textures (reverse order: lower slots render on top)
    const t1 = performance.now();
    for (let i = layers.length - 1; i >= 0; i--) {
      const layer = layers[i];
      if (!layer || !layer.visible || !layer.source) continue;

      // Try WebCodecs VideoFrame first (if available)
      if (layer.source.webCodecsPlayer) {
        const frame = layer.source.webCodecsPlayer.getCurrentFrame();
        if (frame) {
          const extTex = this.importVideoTexture(frame);
          if (extTex) {
            this.detailedStats.decoder = 'WebCodecs';
            this.layerRenderData.push({
              layer,
              isVideo: true,
              externalTexture: extTex,
              textureView: null,
              sourceWidth: frame.displayWidth,
              sourceHeight: frame.displayHeight,
            });
            continue;
          }
        }
      }

      // HTMLVideoElement - optimized with frame tracking
      if (layer.source.videoElement) {
        const video = layer.source.videoElement;

        // Log video state occasionally for debugging
        if (this.profileCounter === 0) {
          console.log(`[VIDEO] ${video.videoWidth}x${video.videoHeight} readyState=${video.readyState} paused=${video.paused} seeking=${video.seeking} buffered=${video.buffered.length > 0 ? (video.buffered.end(0) - video.buffered.start(0)).toFixed(1) + 's' : '0s'}`);
        }

        if (video.readyState >= 2) {
          // Always try to import external texture (zero-copy GPU path)
          const extTex = this.importVideoTexture(video);
          if (extTex) {
            // Cache frame occasionally for seek/pause fallback (not every frame)
            const now = performance.now();
            const lastCapture = this.lastCaptureTime.get(video) || 0;
            if (now - lastCapture > 500) { // Cache every 500ms
              this.captureVideoFrame(video);
              this.lastCaptureTime.set(video, now);
            }
            this.detailedStats.decoder = 'HTMLVideo';

            this.layerRenderData.push({
              layer,
              isVideo: true,
              externalTexture: extTex,
              textureView: null,
              sourceWidth: video.videoWidth,
              sourceHeight: video.videoHeight,
            });
            continue;
          }

          // Import failed (seeking, not ready) - use cached frame
          const lastFrame = this.getLastFrame(video);
          if (lastFrame) {
            this.detailedStats.decoder = 'HTMLVideo(cached)';
            this.layerRenderData.push({
              layer,
              isVideo: false,
              externalTexture: null,
              textureView: lastFrame.view,
              sourceWidth: lastFrame.width,
              sourceHeight: lastFrame.height,
            });
            continue;
          }
        } else {
          // Video not ready - try last frame cache
          const lastFrame = this.getLastFrame(video);
          if (lastFrame) {
            this.layerRenderData.push({
              layer,
              isVideo: false,
              externalTexture: null,
              textureView: lastFrame.view,
              sourceWidth: lastFrame.width,
              sourceHeight: lastFrame.height,
            });
            continue;
          }
        }
      }

      // Images
      if (layer.source.imageElement) {
        const img = layer.source.imageElement;
        let texture = this.imageTextures.get(img);
        if (!texture) {
          texture = this.createImageTexture(img) ?? undefined;
        }
        if (texture) {
          let imageView = this.cachedImageViews.get(texture);
          if (!imageView) {
            imageView = texture.createView();
            this.cachedImageViews.set(texture, imageView);
          }
          this.layerRenderData.push({
            layer,
            isVideo: false,
            externalTexture: null,
            textureView: imageView,
            sourceWidth: img.naturalWidth,
            sourceHeight: img.naturalHeight,
          });
        }
      }
    }
    this.profileData.importTexture = performance.now() - t1;

    // Update video flag for frame rate limiting
    this.hasActiveVideo = this.layerRenderData.some(d => d.isVideo);

    // Early exit if nothing to render - save CPU!
    if (this.layerRenderData.length === 0) {
      // Just clear the preview to black
      if (this.previewContext) {
        const commandEncoder = this.device.createCommandEncoder();
        const pass = commandEncoder.beginRenderPass({
          colorAttachments: [{
            view: this.previewContext.getCurrentTexture().createView(),
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: 'clear',
            storeOp: 'store',
          }],
        });
        pass.end();
        this.device.queue.submit([commandEncoder.finish()]);
      }
      this.lastLayerCount = 0;
      return;
    }

    const t2 = performance.now();
    const commandEncoder = this.device.createCommandEncoder();

    // Ping-pong compositing
    let readView = this.pingView;
    let writeView = this.pongView;
    let usePing = true;

    // Clear first buffer
    const clearPass = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: readView,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });
    clearPass.end();

    // Composite each visible layer
    for (let i = 0; i < this.layerRenderData.length; i++) {
      const data = this.layerRenderData[i];
      const layer = data.layer;

      // Get or create per-layer uniform buffer
      let uniformBuffer = this.layerUniformBuffers.get(layer.id);
      if (!uniformBuffer) {
        uniformBuffer = this.device.createBuffer({
          size: 48,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.layerUniformBuffers.set(layer.id, uniformBuffer);
      }

      // Calculate aspect ratios
      const sourceAspect = data.sourceWidth / data.sourceHeight;
      const outputAspect = this.outputWidth / this.outputHeight;

      // Get mask texture view for this layer (if any)
      const maskTextureView = this.maskTextureViews.get(layer.id) || this.whiteMaskView!;
      const hasMask = this.maskTextureViews.has(layer.id) ? 1 : 0;

      // Debug logging for mask state (throttled to avoid spam)
      if (hasMask === 1 && (!this.lastMaskDebugLog || Date.now() - this.lastMaskDebugLog > 1000)) {
        console.log(`[Engine] Rendering layer ${layer.id} WITH mask (hasMask=${hasMask})`);
        this.lastMaskDebugLog = Date.now();
      }

      // Update uniforms
      this.uniformData[0] = layer.opacity;
      this.uniformDataU32[1] = BLEND_MODE_MAP[layer.blendMode]; // blendMode is u32 in shader
      this.uniformData[2] = layer.position.x;
      this.uniformData[3] = layer.position.y;
      this.uniformData[4] = layer.scale.x;
      this.uniformData[5] = layer.scale.y;
      this.uniformData[6] = layer.rotation;
      this.uniformData[7] = sourceAspect;
      this.uniformData[8] = outputAspect;
      this.uniformData[9] = 0;  // time (for dissolve effects)
      this.uniformDataU32[10] = hasMask;  // hasMask
      this.uniformDataU32[11] = 0; // maskInvert (handled in mask texture generation)
      this.device.queue.writeBuffer(uniformBuffer, 0, this.uniformData);

      let pipeline: GPURenderPipeline;
      let bindGroup: GPUBindGroup;

      if (data.isVideo && data.externalTexture) {
        // External textures are ephemeral - must create bind group each frame
        pipeline = this.externalCompositePipeline!;
        bindGroup = this.device.createBindGroup({
          layout: this.externalCompositeBindGroupLayout!,
          entries: [
            { binding: 0, resource: this.sampler! },
            { binding: 1, resource: readView },
            { binding: 2, resource: data.externalTexture },
            { binding: 3, resource: { buffer: uniformBuffer } },
            { binding: 4, resource: maskTextureView },
          ],
        });
      } else if (data.textureView) {
        // Images - create bind group each frame since texture may change (proxy frames)
        // External textures (video) are ephemeral by nature, but image textures can also
        // change when using proxy frames, so we always create fresh bind groups
        pipeline = this.compositePipeline!;
        bindGroup = this.device.createBindGroup({
          layout: this.compositeBindGroupLayout!,
          entries: [
            { binding: 0, resource: this.sampler! },
            { binding: 1, resource: readView },
            { binding: 2, resource: data.textureView },
            { binding: 3, resource: { buffer: uniformBuffer } },
            { binding: 4, resource: maskTextureView },
          ],
        });
      } else {
        continue;
      }

      const compositePass = commandEncoder.beginRenderPass({
        colorAttachments: [{
          view: writeView,
          loadOp: 'clear',
          storeOp: 'store',
        }],
      });
      compositePass.setPipeline(pipeline);
      compositePass.setBindGroup(0, bindGroup);
      compositePass.draw(6);
      compositePass.end();

      // Apply effects to the layer if any
      if (layer.effects && layer.effects.length > 0) {
        const enabledEffects = layer.effects.filter(e => e.enabled);
        if (enabledEffects.length > 0) {
          // writeView contains the composited result
          // We'll ping-pong between writeView and readView for each effect
          let effectInput = writeView;
          let effectOutput = readView;

          for (const effect of enabledEffects) {
            const pipeline = this.effectPipelines.get(effect.type);
            const bindGroupLayout = this.effectBindGroupLayouts.get(effect.type);

            if (!pipeline || !bindGroupLayout) continue;

            // Create uniform buffer for effect parameters
            const effectParams = this.createEffectUniformData(effect);
            const effectUniformBuffer = effectParams ? this.device.createBuffer({
              size: effectParams.byteLength,
              usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            }) : null;

            if (effectUniformBuffer && effectParams) {
              this.device.queue.writeBuffer(effectUniformBuffer, 0, effectParams);
            }

            // Create bind group
            const entries: GPUBindGroupEntry[] = [
              { binding: 0, resource: this.sampler! },
              { binding: 1, resource: effectInput },
            ];

            if (effectUniformBuffer) {
              entries.push({ binding: 2, resource: { buffer: effectUniformBuffer } });
            }

            const effectBindGroup = this.device.createBindGroup({
              layout: bindGroupLayout,
              entries,
            });

            // Render effect
            const effectPass = commandEncoder.beginRenderPass({
              colorAttachments: [{
                view: effectOutput,
                loadOp: 'clear',
                storeOp: 'store',
              }],
            });
            effectPass.setPipeline(pipeline);
            effectPass.setBindGroup(0, effectBindGroup);
            effectPass.draw(6);
            effectPass.end();

            // Swap for next effect
            const tempView = effectInput;
            effectInput = effectOutput;
            effectOutput = tempView;
          }

          // After all effects, effectInput contains the final result
          // Make sure it's in writeView for the buffer swap below
          if (effectInput !== writeView) {
            // Need to copy back to writeView
            // The result is in readView, and we want it in writeView
            // So we swap them conceptually by adjusting which is which
            const tempView = readView;
            readView = writeView;
            writeView = tempView;
            usePing = !usePing;
          }
        }
      }

      // Swap buffers
      const temp = readView;
      readView = writeView;
      writeView = temp;
      usePing = !usePing;
    }
    this.profileData.renderPass = performance.now() - t2;

    this.lastLayerCount = this.layerRenderData.length;
    // Track which buffer has the final composited result (readView after loop)
    // After swapping, readView contains the result. usePing tracks if readView is ping.
    this.lastRenderWasPing = usePing;

    // Get cached output bind group
    const finalIsPing = !usePing;
    let outputBindGroup = finalIsPing ? this.cachedOutputBindGroupPing : this.cachedOutputBindGroupPong;
    if (!outputBindGroup) {
      outputBindGroup = this.device.createBindGroup({
        layout: this.outputBindGroupLayout!,
        entries: [
          { binding: 0, resource: this.sampler! },
          { binding: 1, resource: readView },
        ],
      });
      if (finalIsPing) {
        this.cachedOutputBindGroupPing = outputBindGroup;
      } else {
        this.cachedOutputBindGroupPong = outputBindGroup;
      }
    }

    // Render to preview (skip during RAM preview generation for efficiency)
    if (this.previewContext && !this.isGeneratingRamPreview) {
      this.renderToCanvasCached(commandEncoder, this.previewContext, outputBindGroup);
    }

    // Render to output windows (also skip during RAM preview)
    if (!this.isGeneratingRamPreview) {
      for (const output of this.outputWindows.values()) {
        if (output.context) {
          this.renderToCanvasCached(commandEncoder, output.context, outputBindGroup);
        }
      }
    }

    const t3 = performance.now();
    this.device.queue.submit([commandEncoder.finish()]);
    this.profileData.submit = performance.now() - t3;

    this.profileData.total = performance.now() - t0;

    // Update detailed stats (smoothed averages)
    this.detailedStats.importTexture = this.detailedStats.importTexture * 0.9 + this.profileData.importTexture * 0.1;
    this.detailedStats.renderPass = this.detailedStats.renderPass * 0.9 + this.profileData.renderPass * 0.1;
    this.detailedStats.submit = this.detailedStats.submit * 0.9 + this.profileData.submit * 0.1;
    this.detailedStats.total = this.detailedStats.total * 0.9 + this.profileData.total * 0.1;

    // Detect drops caused by slow render (if render takes > 1 frame time)
    if (this.profileData.total > this.TARGET_FRAME_TIME) {
      if (this.profileData.importTexture > this.TARGET_FRAME_TIME * 0.5) {
        this.detailedStats.lastDropReason = 'slow_import';
      } else {
        this.detailedStats.lastDropReason = 'slow_render';
      }
    }

    // Log profile every second (based on time, not frame count)
    this.profileCounter++;
    const now = performance.now();
    if (now - this.lastProfileTime >= 1000) {
      const actualFps = this.profileCounter;
      this.profileCounter = 0;
      this.lastProfileTime = now;

      // Calculate where time is being lost
      const jsTime = this.profileData.total;
      const gapTime = timeSinceLastRender;
      const unaccountedTime = gapTime - jsTime; // Time lost outside our code

      // Detailed breakdown with bottleneck indicator
      let bottleneck = 'ok';
      if (gapTime > 20) {
        if (unaccountedTime > 15) bottleneck = 'BROWSER/DECODE';
        else if (this.profileData.importTexture > 5) bottleneck = 'TEXTURE_IMPORT';
        else if (this.profileData.renderPass > 5) bottleneck = 'GPU_RENDER';
        else bottleneck = 'JS_OTHER';
      }

      console.log(`[PROFILE] FPS=${actualFps} | gap=${gapTime.toFixed(0)}ms | videos=${this.layerRenderData.filter(l => l.isVideo).length} | decoder=${this.detailedStats.decoder} | js=${jsTime.toFixed(1)}ms (import=${this.profileData.importTexture.toFixed(1)} rp=${this.profileData.renderPass.toFixed(1)}) | outside_js=${unaccountedTime.toFixed(0)}ms | drops=${this.detailedStats.dropsLastSecond}/s | ${bottleneck}`);
    }

    this.updateStats();
  }

  private createEffectUniformData(effect: Effect): Float32Array | null {
    const params = effect.params;

    switch (effect.type) {
      case 'hue-shift':
        return new Float32Array([
          params.shift as number || 0,
          0, 0, 0, // padding
        ]);

      case 'brightness':
      case 'contrast':
      case 'saturation': {
        // ColorAdjust shader uses: brightness, contrast, saturation
        const brightness = effect.type === 'brightness' ? (params.amount as number || 0) : 0;
        const contrast = effect.type === 'contrast' ? (params.amount as number || 1) : 1;
        const saturation = effect.type === 'saturation' ? (params.amount as number || 1) : 1;
        return new Float32Array([
          brightness,
          contrast,
          saturation,
          0, // padding
        ]);
      }

      case 'pixelate':
        return new Float32Array([
          params.size as number || 8,
          this.outputWidth,
          this.outputHeight,
          0, // padding
        ]);

      case 'kaleidoscope':
        return new Float32Array([
          params.segments as number || 6,
          params.rotation as number || 0,
          0, 0, // padding
        ]);

      case 'mirror':
        return new Float32Array([
          params.horizontal ? 1 : 0,
          params.vertical ? 1 : 0,
          0, 0, // padding
        ]);

      case 'rgb-split':
        return new Float32Array([
          params.amount as number || 0.01,
          params.angle as number || 0,
          0, 0, // padding
        ]);

      case 'levels':
        return new Float32Array([
          params.inputBlack as number || 0,
          params.inputWhite as number || 1,
          params.gamma as number || 1,
          params.outputBlack as number || 0,
          params.outputWhite as number || 1,
          0, 0, 0, // padding
        ]);

      case 'invert':
        return null; // No uniforms needed

      default:
        return null;
    }
  }

  private renderToCanvasCached(
    commandEncoder: GPUCommandEncoder,
    context: GPUCanvasContext,
    bindGroup: GPUBindGroup
  ): void {
    if (!this.outputPipeline) return;

    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });

    renderPass.setPipeline(this.outputPipeline);
    renderPass.setBindGroup(0, bindGroup);
    renderPass.draw(6);
    renderPass.end();
  }

  // Performance tracking - optimized
  private lastFrameStart = 0;
  private statsCounter = 0;

  private updateStats(): void {
    this.frameCount++;
    this.statsCounter++;

    // Only calculate timing every 10 frames to reduce overhead
    if (this.statsCounter >= 10) {
      this.statsCounter = 0;
      const now = performance.now();

      if (this.lastFrameStart > 0) {
        const frameTime = (now - this.lastFrameStart) / 10; // Average over 10 frames
        this.frameTimeBuffer[this.frameTimeIndex] = frameTime;
        this.frameTimeIndex = (this.frameTimeIndex + 1) % 60;
        if (this.frameTimeCount < 60) this.frameTimeCount++;
      }
      this.lastFrameStart = now;

      if (now - this.fpsUpdateTime >= 1000) {
        this.fps = this.frameCount;
        this.frameCount = 0;
        this.fpsUpdateTime = now;
      }
    }
  }

  private lastLayerCount = 0;

  getStats(): EngineStats {
    let sum = 0;
    for (let i = 0; i < this.frameTimeCount; i++) {
      sum += this.frameTimeBuffer[i];
    }
    const avgFrameTime = this.frameTimeCount > 0 ? sum / this.frameTimeCount : 0;
    return {
      fps: this.fps,
      frameTime: avgFrameTime,
      gpuMemory: 0,
      timing: {
        rafGap: this.detailedStats.rafGap,
        importTexture: this.detailedStats.importTexture,
        renderPass: this.detailedStats.renderPass,
        submit: this.detailedStats.submit,
        total: this.detailedStats.total,
      },
      drops: {
        count: this.detailedStats.dropsTotal,
        lastSecond: this.detailedStats.dropsLastSecond,
        reason: this.detailedStats.lastDropReason,
      },
      layerCount: this.lastLayerCount,
      targetFps: 60,
      decoder: this.detailedStats.decoder,
    };
  }

  // Track active video for requestVideoFrameCallback
  private activeVideo: HTMLVideoElement | null = null;
  private videoFrameCallbackId: number | null = null;

  // Track video frame readiness - only import texture when new frame is available
  private videoFrameReady: Map<HTMLVideoElement, boolean> = new Map();
  private videoLastTime: Map<HTMLVideoElement, number> = new Map();
  private videoCallbackActive: Map<HTMLVideoElement, boolean> = new Map();

  // Register a video to track frame readiness
  registerVideo(video: HTMLVideoElement): void {
    // Already fully registered
    if (this.videoFrameReady.has(video) && this.videoCallbackActive.get(video)) {
      // Re-register callback if video is playing but callback isn't active
      if (!video.paused && !this.videoCallbackActive.get(video)) {
        this.startVideoFrameCallback(video);
      }
      return;
    }

    this.videoFrameReady.set(video, true); // First frame is ready
    this.videoLastTime.set(video, -1);
    this.videoCallbackActive.set(video, false);

    // Start frame callback if video is playing
    if (!video.paused) {
      this.startVideoFrameCallback(video);
    }

    // Listen for play event to restart callback
    video.addEventListener('play', () => {
      this.startVideoFrameCallback(video);
    });
  }

  private startVideoFrameCallback(video: HTMLVideoElement): void {
    if (!('requestVideoFrameCallback' in video)) return;
    if (this.videoCallbackActive.get(video)) return;

    this.videoCallbackActive.set(video, true);

    const onFrame = () => {
      this.videoFrameReady.set(video, true);
      if (!video.paused) {
        (video as any).requestVideoFrameCallback(onFrame);
      } else {
        this.videoCallbackActive.set(video, false);
      }
    };
    (video as any).requestVideoFrameCallback(onFrame);
  }

  // Check if a new frame is available (for non-rVFC browsers, check currentTime)
  private hasNewFrame(video: HTMLVideoElement): boolean {
    const lastTime = this.videoLastTime.get(video) ?? -1;
    const currentTime = video.currentTime;

    // When video is paused, no new frames are being decoded
    // Return true only if time changed (e.g., user seeked) or first frame
    if (video.paused) {
      if (lastTime === -1 || Math.abs(currentTime - lastTime) > 0.001) {
        this.videoLastTime.set(video, currentTime);
        return true;
      }
      return false; // Same frame, use cache
    }

    // Video is playing - use requestVideoFrameCallback if available
    if ('requestVideoFrameCallback' in video) {
      const ready = this.videoFrameReady.get(video) ?? false;
      if (ready) {
        this.videoFrameReady.set(video, false);
        this.videoLastTime.set(video, currentTime);
        return true;
      }
      return false;
    }

    // Fallback: check if time changed (at least 1ms difference for ~1000fps max)
    if (Math.abs(currentTime - lastTime) > 0.001) {
      this.videoLastTime.set(video, currentTime);
      return true;
    }
    return false;
  }

  // Track if we have active video to enable frame limiting
  private hasActiveVideo = false;
  private lastRenderTime = 0;
  private readonly VIDEO_FRAME_TIME = 33.33; // ~30fps when video is playing

  setHasActiveVideo(hasVideo: boolean): void {
    this.hasActiveVideo = hasVideo;
  }

  start(renderCallback: () => void): void {
    if (this.isRunning) return;
    this.isRunning = true;
    console.log('[WebGPU] Starting render loop');

    let lastTimestamp = 0;
    let lastFpsReset = 0;

    const loop = (timestamp: number) => {
      if (!this.isRunning) return;

      // Measure time since last rAF callback
      const rafGap = lastTimestamp > 0 ? timestamp - lastTimestamp : 0;
      lastTimestamp = timestamp;

      // Frame rate limiting when video is playing
      // Skip frames to reduce importExternalTexture overhead
      if (this.hasActiveVideo) {
        const timeSinceLastRender = timestamp - this.lastRenderTime;
        if (timeSinceLastRender < this.VIDEO_FRAME_TIME) {
          // Skip this frame, schedule next
          this.animationId = requestAnimationFrame(loop);
          return;
        }
        this.lastRenderTime = timestamp;
      }

      // Update RAF gap stat (smoothed)
      this.detailedStats.rafGap = this.detailedStats.rafGap * 0.9 + rafGap * 0.1;

      // Detect frame drops based on RAF gap
      // A drop is when we miss more than 1.5 frames (>25ms for 60fps)
      if (rafGap > this.TARGET_FRAME_TIME * 1.5 && lastTimestamp > 0) {
        const missedFrames = Math.floor(rafGap / this.TARGET_FRAME_TIME) - 1;
        this.detailedStats.dropsTotal += missedFrames;
        this.detailedStats.dropsThisSecond += missedFrames;
        this.detailedStats.lastDropReason = 'slow_raf';
      }

      // Reset per-second drop counter every second
      if (timestamp - lastFpsReset >= 1000) {
        this.detailedStats.dropsLastSecond = this.detailedStats.dropsThisSecond;
        this.detailedStats.dropsThisSecond = 0;
        lastFpsReset = timestamp;
      }

      renderCallback();

      this.animationId = requestAnimationFrame(loop);
    };

    this.animationId = requestAnimationFrame(loop);
  }

  // Set the active video to sync rendering with its frame delivery
  setActiveVideo(video: HTMLVideoElement | null): void {
    // Clean up old callback
    if (this.activeVideo && this.videoFrameCallbackId !== null) {
      if ('cancelVideoFrameCallback' in this.activeVideo) {
        (this.activeVideo as any).cancelVideoFrameCallback(this.videoFrameCallbackId);
      }
      this.videoFrameCallbackId = null;
    }
    this.activeVideo = video;
  }

  stop(): void {
    this.isRunning = false;
    if (this.animationId !== null) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
  }

  setResolution(width: number, height: number): void {
    this.outputWidth = width;
    this.outputHeight = height;
    this.createPingPongTextures();
  }

  getDevice(): GPUDevice | null {
    return this.device;
  }

  // Get output dimensions (for export)
  getOutputDimensions(): { width: number; height: number } {
    return { width: this.outputWidth, height: this.outputHeight };
  }

  // Read pixels from the final composited frame (for video export)
  // Returns RGBA pixel data as Uint8ClampedArray
  async readPixels(): Promise<Uint8ClampedArray | null> {
    if (!this.device || !this.pingTexture || !this.pongTexture) return null;

    // Determine which texture has the final composited frame
    const sourceTexture = this.lastRenderWasPing ? this.pingTexture : this.pongTexture;

    // WebGPU requires bytesPerRow to be aligned to 256 bytes
    const bytesPerPixel = 4;
    const unalignedBytesPerRow = this.outputWidth * bytesPerPixel;
    const bytesPerRow = Math.ceil(unalignedBytesPerRow / 256) * 256;
    const bufferSize = bytesPerRow * this.outputHeight;

    // Create staging buffer for GPU -> CPU transfer
    const stagingBuffer = this.device.createBuffer({
      size: bufferSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    // Copy texture to staging buffer
    const commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyTextureToBuffer(
      { texture: sourceTexture },
      { buffer: stagingBuffer, bytesPerRow, rowsPerImage: this.outputHeight },
      [this.outputWidth, this.outputHeight]
    );
    this.device.queue.submit([commandEncoder.finish()]);

    // Wait for GPU to finish and map buffer
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const arrayBuffer = stagingBuffer.getMappedRange();

    // Copy data, removing row padding if any
    const result = new Uint8ClampedArray(this.outputWidth * this.outputHeight * bytesPerPixel);
    const srcView = new Uint8Array(arrayBuffer);

    if (bytesPerRow === unalignedBytesPerRow) {
      // No padding, direct copy
      result.set(srcView.subarray(0, result.length));
    } else {
      // Remove row padding
      for (let y = 0; y < this.outputHeight; y++) {
        const srcOffset = y * bytesPerRow;
        const dstOffset = y * unalignedBytesPerRow;
        result.set(srcView.subarray(srcOffset, srcOffset + unalignedBytesPerRow), dstOffset);
      }
    }

    // Cleanup
    stagingBuffer.unmap();
    stagingBuffer.destroy();

    return result;
  }

  destroy(): void {
    this.stop();

    for (const output of this.outputWindows.values()) {
      output.window?.close();
    }
    this.outputWindows.clear();

    this.pingTexture?.destroy();
    this.pongTexture?.destroy();
    this.blackTexture?.destroy();

    for (const texture of this.videoFrameTextures.values()) {
      texture.destroy();
    }
    this.videoFrameTextures.clear();
    this.videoFrameViews.clear();

    // Clear caches
    this.cachedCompositeBindGroups.clear();
    this.cachedOutputBindGroupPing = null;
    this.cachedOutputBindGroupPong = null;
    this.cachedImageViews.clear();

    // Destroy image textures
    for (const texture of this.imageTextures.values()) {
      texture.destroy();
    }
    this.imageTextures.clear();

    // Destroy per-layer uniform buffers
    for (const buffer of this.layerUniformBuffers.values()) {
      buffer.destroy();
    }
    this.layerUniformBuffers.clear();

    // Destroy last-frame cache textures
    for (const texture of this.lastFrameTextures.values()) {
      texture.destroy();
    }
    this.lastFrameTextures.clear();
    this.lastFrameViews.clear();
    this.lastFrameSizes.clear();
    this.lastCaptureTime.clear();

    // Destroy scrubbing cache
    this.clearScrubbingCache();

    // Destroy composite cache and RAM playback resources
    this.clearCompositeCache();

    this.layerUniformBuffer?.destroy();
    this.device?.destroy();
  }
}

// Preserve singleton across HMR to prevent multiple GPU device creation
let engineInstance: WebGPUEngine;

// Check if we're in a Vite HMR context
declare const import_meta_hot: { data: Record<string, unknown> } | undefined;
const hot = typeof import.meta !== 'undefined' ? (import.meta as { hot?: { data: Record<string, unknown> } }).hot : undefined;

if (hot) {
  // Preserve engine instance across HMR
  const existing = hot.data.engine as WebGPUEngine | undefined;
  if (existing) {
    console.log('[WebGPU] Reusing existing engine instance from HMR');
    engineInstance = existing;
  } else {
    console.log('[WebGPU] Creating new engine instance');
    engineInstance = new WebGPUEngine();
    hot.data.engine = engineInstance;
  }
} else {
  engineInstance = new WebGPUEngine();
}

export const engine = engineInstance;
