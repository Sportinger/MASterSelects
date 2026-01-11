// Gaussian Blur Effect Shader
// Single-pass approximation using weighted sampling

struct GaussianBlurParams {
  radius: f32,
  width: f32,
  height: f32,
  direction: f32, // 0 = horizontal, 1 = vertical
};

@group(0) @binding(0) var texSampler: sampler;
@group(0) @binding(1) var inputTex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> params: GaussianBlurParams;

@fragment
fn gaussianBlurFragment(input: VertexOutput) -> @location(0) vec4f {
  if (params.radius < 0.5) {
    return textureSample(inputTex, texSampler, input.uv);
  }

  let texelSize = vec2f(1.0 / params.width, 1.0 / params.height);
  let direction = select(vec2f(1.0, 0.0), vec2f(0.0, 1.0), params.direction > 0.5);

  var color = vec4f(0.0);
  var totalWeight = 0.0;

  let samples = 9;
  let sigma = params.radius / 3.0;

  for (var i = -samples; i <= samples; i++) {
    let offset = f32(i) * texelSize * direction * (params.radius / f32(samples));
    let weight = exp(-f32(i * i) / (2.0 * sigma * sigma));

    color += textureSample(inputTex, texSampler, input.uv + offset) * weight;
    totalWeight += weight;
  }

  return color / totalWeight;
}
