// Chroma Key Effect Shader

struct ChromaKeyParams {
  keyR: f32,
  keyG: f32,
  keyB: f32,
  tolerance: f32,
  softness: f32,
  spillSuppression: f32,
  _p1: f32,
  _p2: f32,
};

@group(0) @binding(0) var texSampler: sampler;
@group(0) @binding(1) var inputTex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> params: ChromaKeyParams;

@fragment
fn chromaKeyFragment(input: VertexOutput) -> @location(0) vec4f {
  let color = textureSample(inputTex, texSampler, input.uv);
  let keyColor = vec3f(params.keyR, params.keyG, params.keyB);

  // Calculate distance in YCbCr space for better keying
  let colorYCbCr = rgb2ycbcr(color.rgb);
  let keyYCbCr = rgb2ycbcr(keyColor);

  // Only compare chrominance (Cb, Cr), ignore luminance
  let cbcrDist = length(colorYCbCr.yz - keyYCbCr.yz);

  // Calculate alpha based on distance from key color
  let innerTolerance = params.tolerance;
  let outerTolerance = params.tolerance + params.softness;

  var alpha = smoothstep(innerTolerance, outerTolerance, cbcrDist);

  // Spill suppression - reduce key color influence
  var finalColor = color.rgb;
  if (params.spillSuppression > 0.0) {
    // For green screen: reduce green, boost red/blue slightly
    if (params.keyG > params.keyR && params.keyG > params.keyB) {
      let spillAmount = max(0.0, finalColor.g - max(finalColor.r, finalColor.b)) * params.spillSuppression;
      finalColor.g -= spillAmount;
      finalColor.r += spillAmount * 0.5;
      finalColor.b += spillAmount * 0.5;
    }
    // For blue screen: reduce blue
    else if (params.keyB > params.keyR && params.keyB > params.keyG) {
      let spillAmount = max(0.0, finalColor.b - max(finalColor.r, finalColor.g)) * params.spillSuppression;
      finalColor.b -= spillAmount;
      finalColor.r += spillAmount * 0.5;
      finalColor.g += spillAmount * 0.5;
    }
  }

  return vec4f(finalColor, color.a * alpha);
}

// RGB to YCbCr conversion
fn rgb2ycbcr(rgb: vec3f) -> vec3f {
  let y = 0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b;
  let cb = 0.564 * (rgb.b - y);
  let cr = 0.713 * (rgb.r - y);
  return vec3f(y, cb, cr);
}
