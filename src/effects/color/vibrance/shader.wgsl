// Vibrance Effect Shader
// Intelligent saturation that avoids oversaturating already saturated colors

struct VibranceParams {
  amount: f32,
  _p1: f32,
  _p2: f32,
  _p3: f32,
};

@group(0) @binding(0) var texSampler: sampler;
@group(0) @binding(1) var inputTex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> params: VibranceParams;

@fragment
fn vibranceFragment(input: VertexOutput) -> @location(0) vec4f {
  let color = textureSample(inputTex, texSampler, input.uv);

  // Calculate current saturation
  let maxC = max(max(color.r, color.g), color.b);
  let minC = min(min(color.r, color.g), color.b);
  let sat = (maxC - minC) / (maxC + 0.001);

  // Less saturation increase for already saturated colors
  let vibrance = params.amount * (1.0 - sat);

  let gray = luminance601(color.rgb);
  let adjusted = mix(vec3f(gray), color.rgb, 1.0 + vibrance);

  return vec4f(clamp(adjusted, vec3f(0.0), vec3f(1.0)), color.a);
}
