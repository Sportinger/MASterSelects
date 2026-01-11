// Gaussian Blur Effect Shader - High Quality
// Uses 2D kernel sampling for smooth results

struct GaussianBlurParams {
  radius: f32,
  width: f32,
  height: f32,
  samples: f32, // Direct sample radius control (3-32)
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

  // Direct sample radius from params
  let sampleRadius = i32(clamp(params.samples, 1.0, 64.0));

  var color = vec4f(0.0);
  var totalWeight = 0.0;

  let sigma = params.radius / 3.0;
  let twoSigmaSq = 2.0 * sigma * sigma;

  // 2D Gaussian kernel
  for (var x = -sampleRadius; x <= sampleRadius; x++) {
    for (var y = -sampleRadius; y <= sampleRadius; y++) {
      let offset = vec2f(f32(x), f32(y)) * texelSize * (params.radius / f32(sampleRadius));

      // Gaussian weight
      let distSq = f32(x * x + y * y);
      let weight = exp(-distSq / twoSigmaSq);

      color += textureSample(inputTex, texSampler, input.uv + offset) * weight;
      totalWeight += weight;
    }
  }

  return color / totalWeight;
}
