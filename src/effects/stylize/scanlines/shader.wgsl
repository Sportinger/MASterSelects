// CRT Scanlines Effect Shader

struct ScanlinesParams {
  density: f32,
  opacity: f32,
  speed: f32,
  time: f32,
};

@group(0) @binding(0) var texSampler: sampler;
@group(0) @binding(1) var inputTex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> params: ScanlinesParams;

@fragment
fn scanlinesFragment(input: VertexOutput) -> @location(0) vec4f {
  let color = textureSample(inputTex, texSampler, input.uv);

  // Animated scanline position
  let scrollOffset = params.time * params.speed * 0.1;
  let scanline = sin((input.uv.y + scrollOffset) * params.density * 100.0) * 0.5 + 0.5;

  // Apply scanline darkening
  let darken = 1.0 - params.opacity * (1.0 - scanline);

  return vec4f(color.rgb * darken, color.a);
}
