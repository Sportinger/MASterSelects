// Glow Effect Shader - High Quality Multi-Ring Blur

struct GlowParams {
  amount: f32,
  threshold: f32,
  radius: f32,
  softness: f32,
  width: f32,
  height: f32,
  quality: f32,
  _p1: f32,
};

@group(0) @binding(0) var texSampler: sampler;
@group(0) @binding(1) var inputTex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> params: GlowParams;

@fragment
fn glowFragment(input: VertexOutput) -> @location(0) vec4f {
  let color = textureSample(inputTex, texSampler, input.uv);
  let texelSize = vec2f(1.0 / params.width, 1.0 / params.height);

  // Multi-ring gaussian blur for glow
  var glow = vec3f(0.0);
  var totalWeight = 0.0;

  // Quality: 1=2 rings/8 samples, 2=4 rings/16 samples, 3=6 rings/24 samples
  let rings = i32(params.quality * 2.0);
  let samplesPerRing = i32(params.quality * 8.0);

  for (var ring = 1; ring <= rings; ring++) {
    let ringRadius = f32(ring) * params.radius * texelSize.x * 10.0;
    let ringWeight = gaussian(f32(ring) / f32(rings), params.softness + 0.3);

    for (var i = 0; i < samplesPerRing; i++) {
      let angle = f32(i) * TAU / f32(samplesPerRing) + f32(ring) * 0.5; // Offset each ring
      let offset = vec2f(cos(angle), sin(angle)) * ringRadius;

      let sampleColor = textureSample(inputTex, texSampler, input.uv + offset);
      let sampleLuma = luminance(sampleColor.rgb);

      // Soft threshold with smoothstep
      let brightFactor = smoothstep(params.threshold - 0.1, params.threshold + 0.1, sampleLuma);
      let brightColor = sampleColor.rgb * brightFactor;

      glow += brightColor * ringWeight;
      totalWeight += ringWeight;
    }
  }

  // Also sample center
  let centerLuma = luminance(color.rgb);
  let centerBright = smoothstep(params.threshold - 0.1, params.threshold + 0.1, centerLuma);
  glow += color.rgb * centerBright * 2.0;
  totalWeight += 2.0;

  glow /= totalWeight;

  // Combine: original + glow (additive)
  let result = color.rgb + glow * params.amount * 2.0;

  return vec4f(clamp(result, vec3f(0.0), vec3f(1.0)), color.a);
}
