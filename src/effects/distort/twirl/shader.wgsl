// Twirl Effect Shader

struct TwirlParams {
  amount: f32,
  radius: f32,
  centerX: f32,
  centerY: f32,
};

@group(0) @binding(0) var texSampler: sampler;
@group(0) @binding(1) var inputTex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> params: TwirlParams;

@fragment
fn twirlFragment(input: VertexOutput) -> @location(0) vec4f {
  let center = vec2f(params.centerX, params.centerY);
  let delta = input.uv - center;
  let dist = length(delta);

  if (dist < params.radius) {
    let factor = 1.0 - (dist / params.radius);
    let angle = params.amount * factor * factor;

    let s = sin(angle);
    let c = cos(angle);
    let rotated = vec2f(
      delta.x * c - delta.y * s,
      delta.x * s + delta.y * c
    );

    return textureSample(inputTex, texSampler, center + rotated);
  }

  return textureSample(inputTex, texSampler, input.uv);
}
