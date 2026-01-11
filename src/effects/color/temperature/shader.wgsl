// Color Temperature Effect Shader

struct TemperatureParams {
  temperature: f32, // -1 = cool, +1 = warm
  tint: f32,        // -1 = green, +1 = magenta
  _p1: f32,
  _p2: f32,
};

@group(0) @binding(0) var texSampler: sampler;
@group(0) @binding(1) var inputTex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> params: TemperatureParams;

@fragment
fn temperatureFragment(input: VertexOutput) -> @location(0) vec4f {
  let color = textureSample(inputTex, texSampler, input.uv);

  // Temperature shifts orange-blue axis
  // Tint shifts green-magenta axis
  var adjusted = color.rgb;

  // Warm (positive) adds orange, removes blue
  // Cool (negative) adds blue, removes orange
  adjusted.r += params.temperature * 0.1;
  adjusted.b -= params.temperature * 0.1;

  // Tint adjustment
  adjusted.g -= params.tint * 0.1;
  adjusted.r += params.tint * 0.05;
  adjusted.b += params.tint * 0.05;

  return vec4f(clamp(adjusted, vec3f(0.0), vec3f(1.0)), color.a);
}
