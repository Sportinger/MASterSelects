// Sharpen Effect Shader

struct SharpenParams {
  amount: f32,
  width: f32,
  height: f32,
  _pad: f32,
};

@group(0) @binding(0) var texSampler: sampler;
@group(0) @binding(1) var inputTex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> params: SharpenParams;

@fragment
fn sharpenFragment(input: VertexOutput) -> @location(0) vec4f {
  let texelSize = vec2f(1.0 / params.width, 1.0 / params.height);

  // Sample center and neighbors
  let center = textureSample(inputTex, texSampler, input.uv);
  let top = textureSample(inputTex, texSampler, input.uv + vec2f(0.0, -texelSize.y));
  let bottom = textureSample(inputTex, texSampler, input.uv + vec2f(0.0, texelSize.y));
  let left = textureSample(inputTex, texSampler, input.uv + vec2f(-texelSize.x, 0.0));
  let right = textureSample(inputTex, texSampler, input.uv + vec2f(texelSize.x, 0.0));

  // Unsharp mask: original + (original - blur) * amount
  let blur = (top + bottom + left + right) * 0.25;
  let sharpened = center.rgb + (center.rgb - blur.rgb) * params.amount;

  return vec4f(clamp(sharpened, vec3f(0.0), vec3f(1.0)), center.a);
}
