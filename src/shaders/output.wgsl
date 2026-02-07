// Output shader - renders composited result to canvas
// When showTransparencyGrid is enabled, renders a checkerboard pattern
// behind transparent areas (like After Effects transparency grid)

struct OutputUniforms {
  showTransparencyGrid: u32,  // 1 = show checkerboard behind transparent areas
  outputWidth: f32,
  outputHeight: f32,
  _padding: f32,
};

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var positions = array<vec2f, 6>(
    vec2f(-1.0, -1.0),
    vec2f(1.0, -1.0),
    vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0),
    vec2f(1.0, -1.0),
    vec2f(1.0, 1.0)
  );

  var uvs = array<vec2f, 6>(
    vec2f(0.0, 1.0),
    vec2f(1.0, 1.0),
    vec2f(0.0, 0.0),
    vec2f(0.0, 0.0),
    vec2f(1.0, 1.0),
    vec2f(1.0, 0.0)
  );

  var output: VertexOutput;
  output.position = vec4f(positions[vertexIndex], 0.0, 1.0);
  output.uv = uvs[vertexIndex];
  return output;
}

@group(0) @binding(0) var texSampler: sampler;
@group(0) @binding(1) var inputTexture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> uniforms: OutputUniforms;

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let color = textureSample(inputTexture, texSampler, input.uv);

  if (uniforms.showTransparencyGrid == 1u && color.a < 1.0) {
    // Generate checkerboard pattern in pixel coordinates
    let pixelX = input.uv.x * uniforms.outputWidth;
    let pixelY = input.uv.y * uniforms.outputHeight;
    let checkerSize = 24.0;
    let cx = floor(pixelX / checkerSize);
    let cy = floor(pixelY / checkerSize);
    let checker = (u32(cx) + u32(cy)) % 2u;
    let light = 0.25;  // #404040
    let dark = 0.19;   // #303030
    let bg = select(dark, light, checker == 0u);
    let checkerColor = vec3f(bg);

    // Blend composited content over checkerboard using alpha
    let result = mix(checkerColor, color.rgb, color.a);
    return vec4f(result, 1.0);
  }

  // No transparency grid: composite over black
  return vec4f(color.rgb, 1.0);
}
