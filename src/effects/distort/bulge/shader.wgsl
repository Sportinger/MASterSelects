// Bulge/Pinch Effect Shader

struct BulgeParams {
  amount: f32,    // positive = bulge, negative = pinch
  radius: f32,
  centerX: f32,
  centerY: f32,
};

@group(0) @binding(0) var texSampler: sampler;
@group(0) @binding(1) var inputTex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> params: BulgeParams;

@fragment
fn bulgeFragment(input: VertexOutput) -> @location(0) vec4f {
  let center = vec2f(params.centerX, params.centerY);
  let delta = input.uv - center;
  let dist = length(delta);

  if (dist < params.radius && dist > 0.0) {
    let normalizedDist = dist / params.radius;

    // Bulge/pinch factor
    let factor = pow(normalizedDist, params.amount);

    let newDist = factor * params.radius;
    let direction = normalize(delta);

    return textureSample(inputTex, texSampler, center + direction * newDist);
  }

  return textureSample(inputTex, texSampler, input.uv);
}
