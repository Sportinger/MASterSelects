// Wave Distortion Effect Shader

struct WaveParams {
  amplitudeX: f32,
  amplitudeY: f32,
  frequencyX: f32,
  frequencyY: f32,
};

@group(0) @binding(0) var texSampler: sampler;
@group(0) @binding(1) var inputTex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> params: WaveParams;

@fragment
fn waveFragment(input: VertexOutput) -> @location(0) vec4f {
  var uv = input.uv;

  // Horizontal wave (affects Y position based on X)
  uv.y += sin(uv.x * params.frequencyX * TAU) * params.amplitudeX;

  // Vertical wave (affects X position based on Y)
  uv.x += sin(uv.y * params.frequencyY * TAU) * params.amplitudeY;

  return textureSample(inputTex, texSampler, uv);
}
