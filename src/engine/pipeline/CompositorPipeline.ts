// Ping-pong compositing pipeline for layer blending

import compositeShader from '../../shaders/composite.wgsl?raw';
import { BLEND_MODE_MAP } from '../core/types';
import type { Layer } from '../core/types';

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
  rotationZ: f32,
  sourceAspect: f32,
  outputAspect: f32,
  time: f32,
  hasMask: u32,
  maskInvert: u32,
  rotationX: f32,
  rotationY: f32,
  perspective: f32,
  maskFeather: f32,
  maskFeatherQuality: u32,
  _pad1: f32,
  _pad2: f32,
  _pad3: f32,
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

  // Apply user scale first
  uv = uv / vec2f(layer.scaleX, layer.scaleY);

  // For 3D rotation, work in world coordinates where the panel has its actual aspect ratio
  var p = vec3f(uv.x, uv.y / layer.outputAspect, 0.0);

  // X rotation (tilt forward/back)
  if (abs(layer.rotationX) > 0.0001) {
    let cosX = cos(-layer.rotationX);
    let sinX = sin(-layer.rotationX);
    p = vec3f(p.x, p.y * cosX - p.z * sinX, p.y * sinX + p.z * cosX);
  }

  // Y rotation (turn left/right)
  if (abs(layer.rotationY) > 0.0001) {
    let cosY = cos(-layer.rotationY);
    let sinY = sin(-layer.rotationY);
    p = vec3f(p.x * cosY + p.z * sinY, p.y, -p.x * sinY + p.z * cosY);
  }

  // Z rotation (spin)
  if (abs(layer.rotationZ) > 0.0001) {
    let cosZ = cos(layer.rotationZ);
    let sinZ = sin(layer.rotationZ);
    p = vec3f(p.x * cosZ - p.y * sinZ, p.x * sinZ + p.y * cosZ, p.z);
  }

  // Perspective projection using homogeneous coordinates
  let perspectiveDist = max(layer.perspective, 0.5);
  let w = 1.0 - p.z / perspectiveDist;
  let projectedX = p.x / w;
  let projectedY = p.y / w;

  // Convert back from world coordinates to UV coordinates
  uv = vec2f(projectedX, projectedY * layer.outputAspect);

  // Apply source aspect ratio correction
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
    var maskValue: f32;

    // GPU blur for feather effect with quality levels (1-100 mapped to low/med/high)
    if (layer.maskFeather > 0.5) {
      let maskDim = vec2f(textureDimensions(maskTexture));
      let texelSize = 1.0 / maskDim;
      let r = layer.maskFeather * texelSize;

      // Map quality 1-100: 1-33=low, 34-66=medium, 67-100=high
      if (layer.maskFeatherQuality >= 67u) {
        // HIGH QUALITY: 61-tap blur with dense sampling
        var blur: f32 = 0.0;
        // Center
        blur += textureSample(maskTexture, texSampler, input.uv).r * 0.08;
        // Ring at 0.2r (8 samples)
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(r.x * 0.2, 0.0)).r * 0.04;
        blur += textureSample(maskTexture, texSampler, input.uv - vec2f(r.x * 0.2, 0.0)).r * 0.04;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(0.0, r.y * 0.2)).r * 0.04;
        blur += textureSample(maskTexture, texSampler, input.uv - vec2f(0.0, r.y * 0.2)).r * 0.04;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(r.x * 0.14, r.y * 0.14)).r * 0.03;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(-r.x * 0.14, r.y * 0.14)).r * 0.03;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(r.x * 0.14, -r.y * 0.14)).r * 0.03;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(-r.x * 0.14, -r.y * 0.14)).r * 0.03;
        // Ring at 0.4r (8 samples)
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(r.x * 0.4, 0.0)).r * 0.035;
        blur += textureSample(maskTexture, texSampler, input.uv - vec2f(r.x * 0.4, 0.0)).r * 0.035;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(0.0, r.y * 0.4)).r * 0.035;
        blur += textureSample(maskTexture, texSampler, input.uv - vec2f(0.0, r.y * 0.4)).r * 0.035;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(r.x * 0.28, r.y * 0.28)).r * 0.028;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(-r.x * 0.28, r.y * 0.28)).r * 0.028;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(r.x * 0.28, -r.y * 0.28)).r * 0.028;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(-r.x * 0.28, -r.y * 0.28)).r * 0.028;
        // Ring at 0.6r (8 samples)
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(r.x * 0.6, 0.0)).r * 0.03;
        blur += textureSample(maskTexture, texSampler, input.uv - vec2f(r.x * 0.6, 0.0)).r * 0.03;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(0.0, r.y * 0.6)).r * 0.03;
        blur += textureSample(maskTexture, texSampler, input.uv - vec2f(0.0, r.y * 0.6)).r * 0.03;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(r.x * 0.42, r.y * 0.42)).r * 0.024;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(-r.x * 0.42, r.y * 0.42)).r * 0.024;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(r.x * 0.42, -r.y * 0.42)).r * 0.024;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(-r.x * 0.42, -r.y * 0.42)).r * 0.024;
        // Ring at 0.8r (8 samples)
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(r.x * 0.8, 0.0)).r * 0.025;
        blur += textureSample(maskTexture, texSampler, input.uv - vec2f(r.x * 0.8, 0.0)).r * 0.025;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(0.0, r.y * 0.8)).r * 0.025;
        blur += textureSample(maskTexture, texSampler, input.uv - vec2f(0.0, r.y * 0.8)).r * 0.025;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(r.x * 0.56, r.y * 0.56)).r * 0.02;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(-r.x * 0.56, r.y * 0.56)).r * 0.02;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(r.x * 0.56, -r.y * 0.56)).r * 0.02;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(-r.x * 0.56, -r.y * 0.56)).r * 0.02;
        // Ring at 1.0r (8 samples)
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(r.x, 0.0)).r * 0.02;
        blur += textureSample(maskTexture, texSampler, input.uv - vec2f(r.x, 0.0)).r * 0.02;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(0.0, r.y)).r * 0.02;
        blur += textureSample(maskTexture, texSampler, input.uv - vec2f(0.0, r.y)).r * 0.02;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(r.x * 0.7, r.y * 0.7)).r * 0.016;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(-r.x * 0.7, r.y * 0.7)).r * 0.016;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(r.x * 0.7, -r.y * 0.7)).r * 0.016;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(-r.x * 0.7, -r.y * 0.7)).r * 0.016;
        // Ring at 1.2r (8 samples)
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(r.x * 1.2, 0.0)).r * 0.012;
        blur += textureSample(maskTexture, texSampler, input.uv - vec2f(r.x * 1.2, 0.0)).r * 0.012;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(0.0, r.y * 1.2)).r * 0.012;
        blur += textureSample(maskTexture, texSampler, input.uv - vec2f(0.0, r.y * 1.2)).r * 0.012;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(r.x * 0.85, r.y * 0.85)).r * 0.01;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(-r.x * 0.85, r.y * 0.85)).r * 0.01;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(r.x * 0.85, -r.y * 0.85)).r * 0.01;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(-r.x * 0.85, -r.y * 0.85)).r * 0.01;
        // Ring at 1.4r (4 samples)
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(r.x * 1.4, 0.0)).r * 0.008;
        blur += textureSample(maskTexture, texSampler, input.uv - vec2f(r.x * 1.4, 0.0)).r * 0.008;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(0.0, r.y * 1.4)).r * 0.008;
        blur += textureSample(maskTexture, texSampler, input.uv - vec2f(0.0, r.y * 1.4)).r * 0.008;
        maskValue = blur;
      } else if (layer.maskFeatherQuality >= 34u) {
        // MEDIUM QUALITY: 33-tap blur
        var blur: f32 = 0.0;
        blur += textureSample(maskTexture, texSampler, input.uv).r * 0.12;
        // Ring at 0.33r
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(r.x * 0.33, 0.0)).r * 0.06;
        blur += textureSample(maskTexture, texSampler, input.uv - vec2f(r.x * 0.33, 0.0)).r * 0.06;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(0.0, r.y * 0.33)).r * 0.06;
        blur += textureSample(maskTexture, texSampler, input.uv - vec2f(0.0, r.y * 0.33)).r * 0.06;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(r.x * 0.23, r.y * 0.23)).r * 0.045;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(-r.x * 0.23, r.y * 0.23)).r * 0.045;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(r.x * 0.23, -r.y * 0.23)).r * 0.045;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(-r.x * 0.23, -r.y * 0.23)).r * 0.045;
        // Ring at 0.66r
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(r.x * 0.66, 0.0)).r * 0.045;
        blur += textureSample(maskTexture, texSampler, input.uv - vec2f(r.x * 0.66, 0.0)).r * 0.045;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(0.0, r.y * 0.66)).r * 0.045;
        blur += textureSample(maskTexture, texSampler, input.uv - vec2f(0.0, r.y * 0.66)).r * 0.045;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(r.x * 0.47, r.y * 0.47)).r * 0.035;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(-r.x * 0.47, r.y * 0.47)).r * 0.035;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(r.x * 0.47, -r.y * 0.47)).r * 0.035;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(-r.x * 0.47, -r.y * 0.47)).r * 0.035;
        // Ring at 1.0r
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(r.x, 0.0)).r * 0.03;
        blur += textureSample(maskTexture, texSampler, input.uv - vec2f(r.x, 0.0)).r * 0.03;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(0.0, r.y)).r * 0.03;
        blur += textureSample(maskTexture, texSampler, input.uv - vec2f(0.0, r.y)).r * 0.03;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(r.x * 0.7, r.y * 0.7)).r * 0.024;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(-r.x * 0.7, r.y * 0.7)).r * 0.024;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(r.x * 0.7, -r.y * 0.7)).r * 0.024;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(-r.x * 0.7, -r.y * 0.7)).r * 0.024;
        // Ring at 1.33r
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(r.x * 1.33, 0.0)).r * 0.015;
        blur += textureSample(maskTexture, texSampler, input.uv - vec2f(r.x * 1.33, 0.0)).r * 0.015;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(0.0, r.y * 1.33)).r * 0.015;
        blur += textureSample(maskTexture, texSampler, input.uv - vec2f(0.0, r.y * 1.33)).r * 0.015;
        maskValue = blur;
      } else {
        // LOW QUALITY: 17-tap blur (fast)
        var blur: f32 = 0.0;
        blur += textureSample(maskTexture, texSampler, input.uv).r * 0.18;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(r.x * 0.5, 0.0)).r * 0.08;
        blur += textureSample(maskTexture, texSampler, input.uv - vec2f(r.x * 0.5, 0.0)).r * 0.08;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(0.0, r.y * 0.5)).r * 0.08;
        blur += textureSample(maskTexture, texSampler, input.uv - vec2f(0.0, r.y * 0.5)).r * 0.08;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(r.x, 0.0)).r * 0.06;
        blur += textureSample(maskTexture, texSampler, input.uv - vec2f(r.x, 0.0)).r * 0.06;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(0.0, r.y)).r * 0.06;
        blur += textureSample(maskTexture, texSampler, input.uv - vec2f(0.0, r.y)).r * 0.06;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(r.x * 0.35, r.y * 0.35)).r * 0.04;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(-r.x * 0.35, r.y * 0.35)).r * 0.04;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(r.x * 0.35, -r.y * 0.35)).r * 0.04;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(-r.x * 0.35, -r.y * 0.35)).r * 0.04;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(r.x * 0.7, r.y * 0.7)).r * 0.03;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(-r.x * 0.7, r.y * 0.7)).r * 0.03;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(r.x * 0.7, -r.y * 0.7)).r * 0.03;
        blur += textureSample(maskTexture, texSampler, input.uv + vec2f(-r.x * 0.7, -r.y * 0.7)).r * 0.03;
        maskValue = blur;
      }
    } else {
      maskValue = textureSample(maskTexture, texSampler, input.uv).r;
    }

    // Apply inversion in shader (not CPU)
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

export class CompositorPipeline {
  private device: GPUDevice;

  // Pipelines
  private compositePipeline: GPURenderPipeline | null = null;
  private externalCompositePipeline: GPURenderPipeline | null = null;

  // Bind group layouts
  private compositeBindGroupLayout: GPUBindGroupLayout | null = null;
  private externalCompositeBindGroupLayout: GPUBindGroupLayout | null = null;

  // Uniform buffer and data
  private uniformBuffer = new ArrayBuffer(80); // 20 floats for extended uniforms (including mask quality)
  private uniformData = new Float32Array(this.uniformBuffer);
  private uniformDataU32 = new Uint32Array(this.uniformBuffer);

  // Per-layer uniform buffers
  private layerUniformBuffers: Map<string, GPUBuffer> = new Map();

  constructor(device: GPUDevice) {
    this.device = device;
  }

  async createPipelines(): Promise<void> {
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
  }

  getCompositePipeline(): GPURenderPipeline | null {
    return this.compositePipeline;
  }

  getExternalCompositePipeline(): GPURenderPipeline | null {
    return this.externalCompositePipeline;
  }

  getCompositeBindGroupLayout(): GPUBindGroupLayout | null {
    return this.compositeBindGroupLayout;
  }

  getExternalCompositeBindGroupLayout(): GPUBindGroupLayout | null {
    return this.externalCompositeBindGroupLayout;
  }

  // Get or create per-layer uniform buffer
  getOrCreateUniformBuffer(layerId: string): GPUBuffer {
    let uniformBuffer = this.layerUniformBuffers.get(layerId);
    if (!uniformBuffer) {
      uniformBuffer = this.device.createBuffer({
        size: 80, // 20 floats for extended uniforms (incl. mask quality)
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.layerUniformBuffers.set(layerId, uniformBuffer);
    }
    return uniformBuffer;
  }

  // Update layer uniforms
  updateLayerUniforms(
    layer: Layer,
    sourceAspect: number,
    outputAspect: number,
    hasMask: boolean,
    uniformBuffer: GPUBuffer
  ): void {
    // Get rotation values (layer.rotation can be number or {x,y,z} object)
    let rotX = 0, rotY = 0, rotZ = 0;
    if (typeof layer.rotation === 'number') {
      rotZ = layer.rotation;
    } else if (layer.rotation && typeof layer.rotation === 'object') {
      rotX = (layer.rotation as { x?: number; y?: number; z?: number }).x || 0;
      rotY = (layer.rotation as { x?: number; y?: number; z?: number }).y || 0;
      rotZ = (layer.rotation as { x?: number; y?: number; z?: number }).z || 0;
    }

    // Update uniforms
    this.uniformData[0] = layer.opacity;
    this.uniformDataU32[1] = BLEND_MODE_MAP[layer.blendMode]; // blendMode is u32 in shader
    this.uniformData[2] = layer.position.x;
    this.uniformData[3] = layer.position.y;
    this.uniformData[4] = layer.scale.x;
    this.uniformData[5] = layer.scale.y;
    this.uniformData[6] = rotZ;         // rotationZ
    this.uniformData[7] = sourceAspect;
    this.uniformData[8] = outputAspect;
    this.uniformData[9] = 0;  // time (for dissolve effects)
    this.uniformDataU32[10] = hasMask ? 1 : 0;  // hasMask
    this.uniformDataU32[11] = layer.maskInvert ? 1 : 0; // maskInvert (now handled in shader)
    this.uniformData[12] = rotX;        // rotationX
    this.uniformData[13] = rotY;        // rotationY
    this.uniformData[14] = 2.0;         // perspective distance (lower = stronger 3D effect)
    this.uniformData[15] = layer.maskFeather || 0;      // maskFeather (blur radius in pixels)
    this.uniformDataU32[16] = layer.maskFeatherQuality || 0; // maskFeatherQuality (0=low, 1=med, 2=high)
    this.uniformData[17] = 0;           // _pad1
    this.uniformData[18] = 0;           // _pad2
    this.uniformData[19] = 0;           // _pad3
    this.device.queue.writeBuffer(uniformBuffer, 0, this.uniformData);
  }

  // Create bind group for regular textures
  createCompositeBindGroup(
    sampler: GPUSampler,
    baseView: GPUTextureView,
    layerView: GPUTextureView,
    uniformBuffer: GPUBuffer,
    maskTextureView: GPUTextureView
  ): GPUBindGroup {
    return this.device.createBindGroup({
      layout: this.compositeBindGroupLayout!,
      entries: [
        { binding: 0, resource: sampler },
        { binding: 1, resource: baseView },
        { binding: 2, resource: layerView },
        { binding: 3, resource: { buffer: uniformBuffer } },
        { binding: 4, resource: maskTextureView },
      ],
    });
  }

  // Create bind group for external video textures
  createExternalCompositeBindGroup(
    sampler: GPUSampler,
    baseView: GPUTextureView,
    externalTexture: GPUExternalTexture,
    uniformBuffer: GPUBuffer,
    maskTextureView: GPUTextureView
  ): GPUBindGroup {
    return this.device.createBindGroup({
      layout: this.externalCompositeBindGroupLayout!,
      entries: [
        { binding: 0, resource: sampler },
        { binding: 1, resource: baseView },
        { binding: 2, resource: externalTexture },
        { binding: 3, resource: { buffer: uniformBuffer } },
        { binding: 4, resource: maskTextureView },
      ],
    });
  }

  destroy(): void {
    // Destroy per-layer uniform buffers
    for (const buffer of this.layerUniformBuffers.values()) {
      buffer.destroy();
    }
    this.layerUniformBuffers.clear();
  }
}
