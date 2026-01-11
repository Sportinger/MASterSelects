// Radial Blur Effect Shader - High Quality

struct RadialBlurParams {
  amount: f32,
  centerX: f32,
  centerY: f32,
  samples: f32,
};

@group(0) @binding(0) var texSampler: sampler;
@group(0) @binding(1) var inputTex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> params: RadialBlurParams;

@fragment
fn radialBlurFragment(input: VertexOutput) -> @location(0) vec4f {
  let center = vec2f(params.centerX, params.centerY);
  let dir = input.uv - center;
  let dist = length(dir);

  if (params.amount < 0.01) {
    return textureSample(inputTex, texSampler, input.uv);
  }

  var color = vec4f(0.0);
  let samples = i32(clamp(params.samples, 4.0, 256.0));
  let amount = params.amount * 0.2;

  var totalWeight = 0.0;

  for (var i = 0; i < samples; i++) {
    let t = f32(i) / f32(samples - 1);
    let scale = 1.0 - amount * t * dist;
    let weight = 1.0 - t * 0.5; // Weight decreases towards outer samples

    let samplePos = center + dir * scale;
    color += textureSample(inputTex, texSampler, samplePos) * weight;
    totalWeight += weight;
  }

  return color / totalWeight;
}
