// Exposure Effect Shader

struct ExposureParams {
  exposure: f32,  // EV stops
  offset: f32,    // Shadow lift
  gamma: f32,     // Mid-tone adjustment
  _pad: f32,
};

@group(0) @binding(0) var texSampler: sampler;
@group(0) @binding(1) var inputTex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> params: ExposureParams;

@fragment
fn exposureFragment(input: VertexOutput) -> @location(0) vec4f {
  let color = textureSample(inputTex, texSampler, input.uv);

  // Apply exposure (EV stops: 2^exposure)
  var adjusted = color.rgb * pow(2.0, params.exposure);

  // Add offset (shadow lift)
  adjusted += params.offset;

  // Apply gamma
  adjusted = pow(max(adjusted, vec3f(0.0)), vec3f(1.0 / params.gamma));

  return vec4f(clamp(adjusted, vec3f(0.0), vec3f(1.0)), color.a);
}
