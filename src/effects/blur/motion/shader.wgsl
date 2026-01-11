// Motion Blur Effect Shader

struct MotionBlurParams {
  amount: f32,
  angle: f32,
  _p1: f32,
  _p2: f32,
};

@group(0) @binding(0) var texSampler: sampler;
@group(0) @binding(1) var inputTex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> params: MotionBlurParams;

@fragment
fn motionBlurFragment(input: VertexOutput) -> @location(0) vec4f {
  let direction = vec2f(cos(params.angle), sin(params.angle));

  var color = vec4f(0.0);
  let samples = 16;

  for (var i = 0; i < samples; i++) {
    let t = (f32(i) / f32(samples - 1) - 0.5) * 2.0;
    let offset = direction * t * params.amount;
    color += textureSample(inputTex, texSampler, input.uv + offset);
  }

  return color / f32(samples);
}
