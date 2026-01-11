// Film Grain Effect Shader

struct GrainParams {
  amount: f32,
  size: f32,
  speed: f32,
  time: f32,
};

@group(0) @binding(0) var texSampler: sampler;
@group(0) @binding(1) var inputTex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> params: GrainParams;

// Simplex-like noise
fn grainNoise(uv: vec2f, t: f32) -> f32 {
  let seed = uv + vec2f(t * 0.1, t * 0.07);
  return fract(sin(dot(seed, vec2f(12.9898, 78.233))) * 43758.5453);
}

@fragment
fn grainFragment(input: VertexOutput) -> @location(0) vec4f {
  let color = textureSample(inputTex, texSampler, input.uv);

  // Scale UV for grain size
  let grainUV = input.uv * (100.0 / params.size);

  // Generate animated noise
  let noise = grainNoise(grainUV, params.time * params.speed) * 2.0 - 1.0;

  // Apply grain with luminance-aware intensity
  let luma = luminance(color.rgb);
  let grainIntensity = params.amount * (1.0 - luma * 0.5); // Less grain in highlights

  let grainColor = color.rgb + vec3f(noise * grainIntensity);

  return vec4f(clamp(grainColor, vec3f(0.0), vec3f(1.0)), color.a);
}
