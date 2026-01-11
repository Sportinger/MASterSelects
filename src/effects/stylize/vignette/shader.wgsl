// Vignette Effect Shader

struct VignetteParams {
  amount: f32,
  size: f32,
  softness: f32,
  roundness: f32,
};

@group(0) @binding(0) var texSampler: sampler;
@group(0) @binding(1) var inputTex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> params: VignetteParams;

@fragment
fn vignetteFragment(input: VertexOutput) -> @location(0) vec4f {
  let color = textureSample(inputTex, texSampler, input.uv);

  // Center-relative coordinates
  let center = input.uv - 0.5;

  // Distance from center (adjusted for roundness)
  let aspect = vec2f(1.0, params.roundness);
  let dist = length(center * aspect) * 2.0;

  // Vignette factor with softness
  let vignette = 1.0 - smoothstep(params.size, params.size + params.softness, dist);

  // Apply vignette (lerp to black based on amount)
  let vignetteColor = mix(vec3f(0.0), color.rgb, mix(1.0, vignette, params.amount));

  return vec4f(vignetteColor, color.a);
}
