// GPU-accelerated proxy frame resize shader
// Renders VideoFrame (texture_external) to a scaled rgba8unorm texture
// Used for fast proxy frame generation with batch processing

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

struct TileParams {
  // Atlas tile position (which tile in the grid)
  tileX: u32,
  tileY: u32,
  // Tile dimensions
  tileWidth: f32,
  tileHeight: f32,
  // Atlas dimensions
  atlasWidth: f32,
  atlasHeight: f32,
  // Source dimensions (for aspect ratio correction)
  srcWidth: f32,
  srcHeight: f32,
}

@group(0) @binding(0) var texSampler: sampler;
@group(0) @binding(1) var videoTexture: texture_external;
@group(0) @binding(2) var<uniform> params: TileParams;

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  // Fullscreen triangle pair for the specific tile in the atlas
  // Each tile covers a portion of the atlas based on tileX, tileY

  // Calculate normalized tile position in atlas (0-1 range)
  let tileStartX = f32(params.tileX) * params.tileWidth / params.atlasWidth;
  let tileStartY = f32(params.tileY) * params.tileHeight / params.atlasHeight;
  let tileEndX = tileStartX + params.tileWidth / params.atlasWidth;
  let tileEndY = tileStartY + params.tileHeight / params.atlasHeight;

  // Convert to clip space (-1 to 1)
  let clipStartX = tileStartX * 2.0 - 1.0;
  let clipStartY = 1.0 - tileEndY * 2.0;  // Flip Y
  let clipEndX = tileEndX * 2.0 - 1.0;
  let clipEndY = 1.0 - tileStartY * 2.0;  // Flip Y

  // 6 vertices for 2 triangles (quad)
  var positions = array<vec2f, 6>(
    vec2f(clipStartX, clipStartY), vec2f(clipEndX, clipStartY), vec2f(clipStartX, clipEndY),
    vec2f(clipStartX, clipEndY), vec2f(clipEndX, clipStartY), vec2f(clipEndX, clipEndY)
  );

  var uvs = array<vec2f, 6>(
    vec2f(0.0, 1.0), vec2f(1.0, 1.0), vec2f(0.0, 0.0),
    vec2f(0.0, 0.0), vec2f(1.0, 1.0), vec2f(1.0, 0.0)
  );

  var output: VertexOutput;
  output.position = vec4f(positions[vertexIndex], 0.0, 1.0);
  output.uv = uvs[vertexIndex];
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  // Sample from external video texture with bilinear filtering
  // textureSampleBaseClampToEdge is required for texture_external
  let color = textureSampleBaseClampToEdge(videoTexture, texSampler, input.uv);
  return color;
}
