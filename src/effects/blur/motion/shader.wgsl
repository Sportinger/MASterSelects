// Motion Blur Effect Shader - High Quality

struct MotionBlurParams {
  amount: f32,
  angle: f32,
  samples: f32,
  _pad: f32,
};

@group(0) @binding(0) var texSampler: sampler;
@group(0) @binding(1) var inputTex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> params: MotionBlurParams;

@fragment
fn motionBlurFragment(input: VertexOutput) -> @location(0) vec4f {
  if (params.amount < 0.001) {
    return textureSample(inputTex, texSampler, input.uv);
  }

  let direction = vec2f(cos(params.angle), sin(params.angle));
  let samples = i32(clamp(params.samples, 4.0, 128.0));

  var color = vec4f(0.0);
  var totalWeight = 0.0;

  for (var i = 0; i < samples; i++) {
    let t = (f32(i) / f32(samples - 1) - 0.5) * 2.0;
    let offset = direction * t * params.amount;

    // Gaussian-like weight (center samples weighted more)
    let weight = exp(-t * t * 2.0);

    color += textureSample(inputTex, texSampler, input.uv + offset) * weight;
    totalWeight += weight;
  }

  return color / totalWeight;
}
