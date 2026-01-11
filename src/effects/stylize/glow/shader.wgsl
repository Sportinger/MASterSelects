// Glow Effect Shader

struct GlowParams {
  amount: f32,
  threshold: f32,
  radius: f32,
  _pad: f32,
};

@group(0) @binding(0) var texSampler: sampler;
@group(0) @binding(1) var inputTex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> params: GlowParams;

@fragment
fn glowFragment(input: VertexOutput) -> @location(0) vec4f {
  let color = textureSample(inputTex, texSampler, input.uv);

  // Extract bright areas
  let brightness = luminance(color.rgb);
  let brightPass = max(color.rgb - params.threshold, vec3f(0.0)) / (1.0 - params.threshold + 0.001);

  // Simple blur for glow (box filter)
  var glow = vec3f(0.0);
  let samples = 8;
  let radius = params.radius * 0.01;

  for (var i = 0; i < samples; i++) {
    let angle = f32(i) * TAU / f32(samples);
    let offset = vec2f(cos(angle), sin(angle)) * radius;
    let sampleColor = textureSample(inputTex, texSampler, input.uv + offset);
    let sampleBright = max(sampleColor.rgb - params.threshold, vec3f(0.0));
    glow += sampleBright;
  }
  glow /= f32(samples);

  // Combine original with glow
  let result = color.rgb + glow * params.amount;

  return vec4f(clamp(result, vec3f(0.0), vec3f(1.0)), color.a);
}
