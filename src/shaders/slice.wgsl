// Slice warp shader — vertex-driven warping
// All warp computation happens on CPU; GPU just renders the triangles
// maskFlag: 0 = normal slice (sample texture), >0.5 = mask (opaque black)

struct VertexInput {
  @location(0) position: vec2f,
  @location(1) uv: vec2f,
  @location(2) maskFlag: f32,
};

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
  @location(1) maskFlag: f32,
};

@vertex fn vertexMain(input: VertexInput) -> VertexOutput {
  var out: VertexOutput;
  out.position = vec4f(input.position, 0.0, 1.0);
  out.uv = input.uv;
  out.maskFlag = input.maskFlag;
  return out;
}

@group(0) @binding(0) var texSampler: sampler;
@group(0) @binding(1) var inputTexture: texture_2d<f32>;

@fragment fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  if (input.maskFlag > 0.5) {
    return vec4f(0.0, 0.0, 0.0, 1.0);
  }
  return textureSample(inputTexture, texSampler, input.uv);
}
