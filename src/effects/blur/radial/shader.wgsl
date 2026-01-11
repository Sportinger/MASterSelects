// Radial Blur Effect Shader

struct RadialBlurParams {
  amount: f32,
  centerX: f32,
  centerY: f32,
  _pad: f32,
};

@group(0) @binding(0) var texSampler: sampler;
@group(0) @binding(1) var inputTex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> params: RadialBlurParams;

@fragment
fn radialBlurFragment(input: VertexOutput) -> @location(0) vec4f {
  let center = vec2f(params.centerX, params.centerY);
  let dir = input.uv - center;
  let dist = length(dir);

  var color = vec4f(0.0);
  let samples = 16;
  let amount = params.amount * 0.1;

  for (var i = 0; i < samples; i++) {
    let scale = 1.0 - amount * (f32(i) / f32(samples)) * dist;
    let samplePos = center + dir * scale;
    color += textureSample(inputTex, texSampler, samplePos);
  }

  return color / f32(samples);
}
