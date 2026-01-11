// Output shader with optional transparency grid (checkerboard pattern)

struct OutputUniforms {
  showTransparencyGrid: u32,  // 1 = show checkerboard, 0 = black background
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

// Generate checkerboard pattern for transparency visualization
fn checkerboard(uv: vec2f, outputSize: vec2f) -> vec3f {
  // 10x10 pixel squares
  let squareSize = 10.0;
  let pixelCoord = uv * outputSize;
  let checker = floor(pixelCoord.x / squareSize) + floor(pixelCoord.y / squareSize);
  let isLight = (i32(checker) % 2) == 0;

  // Light gray and dark gray
  return select(vec3f(0.3, 0.3, 0.3), vec3f(0.5, 0.5, 0.5), isLight);
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let color = textureSample(inputTexture, texSampler, input.uv);

  // If transparency grid is enabled and there's transparency, blend with checkerboard
  if (uniforms.showTransparencyGrid == 1u && color.a < 1.0) {
    let outputSize = vec2f(uniforms.outputWidth, uniforms.outputHeight);
    let checker = checkerboard(input.uv, outputSize);
    // Blend: checkerboard * (1 - alpha) + color * alpha
    let blended = checker * (1.0 - color.a) + color.rgb * color.a;
    return vec4f(blended, 1.0);
  }

  // No transparency grid - composite over black
  return vec4f(color.rgb * color.a, 1.0);
}
