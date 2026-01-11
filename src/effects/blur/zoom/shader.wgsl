// Zoom Blur Effect Shader

struct ZoomBlurParams {
  amount: f32,
  centerX: f32,
  centerY: f32,
  samples: f32,
};

@group(0) @binding(0) var texSampler: sampler;
@group(0) @binding(1) var inputTex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> params: ZoomBlurParams;

@fragment
fn zoomBlurFragment(input: VertexOutput) -> @location(0) vec4f {
  let center = vec2f(params.centerX, params.centerY);
  let dir = input.uv - center;

  var color = vec4f(0.0);
  // Direct sample count from params
  let samples = i32(clamp(params.samples, 4.0, 256.0));
  let amount = params.amount * 0.5;

  for (var i = 0; i < samples; i++) {
    let t = f32(i) / f32(samples - 1);
    let scale = 1.0 + amount * t;
    let samplePos = center + dir * scale;
    color += textureSample(inputTex, texSampler, samplePos);
  }

  return color / f32(samples);
}
