// Optical Flow Compute Shaders for Motion Analysis
// Lucas-Kanade with Gaussian Pyramid

// ==================== CONSTANTS ====================
const PI: f32 = 3.14159265359;

// Luminance weights (BT.601)
const LUMA_R: f32 = 0.299;
const LUMA_G: f32 = 0.587;
const LUMA_B: f32 = 0.114;

// ==================== GRAYSCALE CONVERSION ====================
// Converts RGBA texture to grayscale luminance

@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var outputTexture: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn grayscaleMain(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(inputTexture);
    if (id.x >= dims.x || id.y >= dims.y) { return; }

    let color = textureLoad(inputTexture, vec2i(id.xy), 0);
    let gray = dot(color.rgb, vec3f(LUMA_R, LUMA_G, LUMA_B));
    textureStore(outputTexture, vec2i(id.xy), vec4f(gray, 0.0, 0.0, 1.0));
}

// ==================== GAUSSIAN BLUR (5x5) ====================
// Used for pyramid generation and noise reduction

@group(0) @binding(0) var blurInput: texture_2d<f32>;
@group(0) @binding(1) var blurOutput: texture_storage_2d<r32float, write>;

// Gaussian kernel 5x5, sigma=1.0
const GAUSS_KERNEL = array<f32, 25>(
    0.003765, 0.015019, 0.023792, 0.015019, 0.003765,
    0.015019, 0.059912, 0.094907, 0.059912, 0.015019,
    0.023792, 0.094907, 0.150342, 0.094907, 0.023792,
    0.015019, 0.059912, 0.094907, 0.059912, 0.015019,
    0.003765, 0.015019, 0.023792, 0.015019, 0.003765
);

@compute @workgroup_size(8, 8)
fn gaussianBlurMain(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(blurInput);
    if (id.x >= dims.x || id.y >= dims.y) { return; }

    var sum: f32 = 0.0;
    let center = vec2i(id.xy);

    for (var ky: i32 = -2; ky <= 2; ky++) {
        for (var kx: i32 = -2; kx <= 2; kx++) {
            let coord = clamp(center + vec2i(kx, ky), vec2i(0), vec2i(dims) - 1);
            let weight = GAUSS_KERNEL[(ky + 2) * 5 + (kx + 2)];
            sum += textureLoad(blurInput, coord, 0).r * weight;
        }
    }

    textureStore(blurOutput, center, vec4f(sum, 0.0, 0.0, 1.0));
}

// ==================== PYRAMID DOWNSAMPLE ====================
// Downsamples by 2x with Gaussian filtering

@group(0) @binding(0) var pyramidSrc: texture_2d<f32>;
@group(0) @binding(1) var pyramidDst: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn pyramidDownsampleMain(@builtin(global_invocation_id) id: vec3u) {
    let dstDims = textureDimensions(pyramidDst);
    if (id.x >= dstDims.x || id.y >= dstDims.y) { return; }

    let srcCoord = vec2i(id.xy) * 2;
    let srcDims = vec2i(textureDimensions(pyramidSrc));

    // 4x4 Gaussian-weighted average for 2x downsample
    var sum: f32 = 0.0;
    var weightSum: f32 = 0.0;

    // Simplified 4x4 Gaussian weights
    let weights = array<f32, 16>(
        0.0625, 0.125, 0.125, 0.0625,
        0.125,  0.25,  0.25,  0.125,
        0.125,  0.25,  0.25,  0.125,
        0.0625, 0.125, 0.125, 0.0625
    );

    for (var ky: i32 = 0; ky < 4; ky++) {
        for (var kx: i32 = 0; kx < 4; kx++) {
            let coord = clamp(srcCoord + vec2i(kx - 1, ky - 1), vec2i(0), srcDims - 1);
            let weight = weights[ky * 4 + kx];
            sum += textureLoad(pyramidSrc, coord, 0).r * weight;
            weightSum += weight;
        }
    }

    textureStore(pyramidDst, vec2i(id.xy), vec4f(sum / weightSum, 0.0, 0.0, 1.0));
}

// ==================== SPATIAL GRADIENTS (SCHARR) ====================
// Computes Ix and Iy using Scharr operator (more accurate than Sobel)

@group(0) @binding(0) var gradInput: texture_2d<f32>;
@group(0) @binding(1) var gradIx: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var gradIy: texture_storage_2d<r32float, write>;

// Scharr kernels (scaled by 1/32 for normalization)
// Kx = [-3, 0, 3; -10, 0, 10; -3, 0, 3] / 32
// Ky = [-3, -10, -3; 0, 0, 0; 3, 10, 3] / 32

@compute @workgroup_size(8, 8)
fn spatialGradientsMain(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(gradInput);
    if (id.x >= dims.x || id.y >= dims.y) { return; }

    let center = vec2i(id.xy);
    let maxCoord = vec2i(dims) - 1;

    // Load 3x3 neighborhood
    let tl = textureLoad(gradInput, clamp(center + vec2i(-1, -1), vec2i(0), maxCoord), 0).r;
    let tc = textureLoad(gradInput, clamp(center + vec2i( 0, -1), vec2i(0), maxCoord), 0).r;
    let tr = textureLoad(gradInput, clamp(center + vec2i( 1, -1), vec2i(0), maxCoord), 0).r;
    let ml = textureLoad(gradInput, clamp(center + vec2i(-1,  0), vec2i(0), maxCoord), 0).r;
    let mr = textureLoad(gradInput, clamp(center + vec2i( 1,  0), vec2i(0), maxCoord), 0).r;
    let bl = textureLoad(gradInput, clamp(center + vec2i(-1,  1), vec2i(0), maxCoord), 0).r;
    let bc = textureLoad(gradInput, clamp(center + vec2i( 0,  1), vec2i(0), maxCoord), 0).r;
    let br = textureLoad(gradInput, clamp(center + vec2i( 1,  1), vec2i(0), maxCoord), 0).r;

    // Scharr X gradient: [-3, 0, 3; -10, 0, 10; -3, 0, 3] / 32
    let ix = ((-3.0 * tl + 3.0 * tr) +
              (-10.0 * ml + 10.0 * mr) +
              (-3.0 * bl + 3.0 * br)) / 32.0;

    // Scharr Y gradient: [-3, -10, -3; 0, 0, 0; 3, 10, 3] / 32
    let iy = ((-3.0 * tl - 10.0 * tc - 3.0 * tr) +
              (3.0 * bl + 10.0 * bc + 3.0 * br)) / 32.0;

    textureStore(gradIx, center, vec4f(ix, 0.0, 0.0, 1.0));
    textureStore(gradIy, center, vec4f(iy, 0.0, 0.0, 1.0));
}

// ==================== TEMPORAL GRADIENT ====================
// Computes It = I(t) - I(t-1)

@group(0) @binding(0) var currFrame: texture_2d<f32>;
@group(0) @binding(1) var prevFrame: texture_2d<f32>;
@group(0) @binding(2) var temporalGrad: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn temporalGradientMain(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(currFrame);
    if (id.x >= dims.x || id.y >= dims.y) { return; }

    let coord = vec2i(id.xy);
    let curr = textureLoad(currFrame, coord, 0).r;
    let prev = textureLoad(prevFrame, coord, 0).r;
    let it = curr - prev;

    textureStore(temporalGrad, coord, vec4f(it, 0.0, 0.0, 1.0));
}

// ==================== LUCAS-KANADE OPTICAL FLOW ====================
// Solves for (Vx, Vy) using least squares over a window

struct LKParams {
    windowRadius: u32,     // Half-size of window (e.g., 2 for 5x5)
    minEigenvalue: f32,    // Threshold for reliable flow
    pyramidScale: f32,     // Scale factor from previous level
    _pad: u32,
};

@group(0) @binding(0) var lkIx: texture_2d<f32>;
@group(0) @binding(1) var lkIy: texture_2d<f32>;
@group(0) @binding(2) var lkIt: texture_2d<f32>;
@group(0) @binding(3) var<uniform> lkParams: LKParams;
@group(0) @binding(4) var flowOutput: texture_storage_2d<rg32float, write>;
@group(0) @binding(5) var prevFlow: texture_2d<f32>;  // Flow from coarser level (or zero)

@compute @workgroup_size(8, 8)
fn lucasKanadeMain(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(lkIx);
    if (id.x >= dims.x || id.y >= dims.y) { return; }

    let center = vec2i(id.xy);
    let maxCoord = vec2i(dims) - 1;
    let radius = i32(lkParams.windowRadius);

    // Get initial flow estimate from coarser level (scaled up)
    var initFlow = vec2f(0.0);
    let prevDims = textureDimensions(prevFlow);
    if (prevDims.x > 1u) {
        // Sample from coarser level and scale
        let prevCoord = vec2i(id.xy / 2u);
        let prevCoordClamped = clamp(prevCoord, vec2i(0), vec2i(prevDims) - 1);
        initFlow = textureLoad(prevFlow, prevCoordClamped, 0).rg * lkParams.pyramidScale;
    }

    // Accumulate structure tensor components over window
    // [sum(Ix*Ix), sum(Ix*Iy)]   [Vx]   [-sum(Ix*It)]
    // [sum(Ix*Iy), sum(Iy*Iy)] * [Vy] = [-sum(Iy*It)]
    var sumIxIx: f32 = 0.0;
    var sumIyIy: f32 = 0.0;
    var sumIxIy: f32 = 0.0;
    var sumIxIt: f32 = 0.0;
    var sumIyIt: f32 = 0.0;

    for (var wy: i32 = -radius; wy <= radius; wy++) {
        for (var wx: i32 = -radius; wx <= radius; wx++) {
            let coord = clamp(center + vec2i(wx, wy), vec2i(0), maxCoord);

            let ix = textureLoad(lkIx, coord, 0).r;
            let iy = textureLoad(lkIy, coord, 0).r;
            let it = textureLoad(lkIt, coord, 0).r;

            sumIxIx += ix * ix;
            sumIyIy += iy * iy;
            sumIxIy += ix * iy;
            sumIxIt += ix * it;
            sumIyIt += iy * it;
        }
    }

    // Solve 2x2 linear system using Cramer's rule
    // A = [sumIxIx, sumIxIy; sumIxIy, sumIyIy]
    // b = [-sumIxIt, -sumIyIt]
    let det = sumIxIx * sumIyIy - sumIxIy * sumIxIy;

    // Check if matrix is well-conditioned (eigenvalue check)
    // For 2x2 symmetric matrix, smallest eigenvalue is bounded by det/trace
    let trace = sumIxIx + sumIyIy;
    let minEig = (trace - sqrt(max(0.0, trace * trace - 4.0 * det))) * 0.5;

    var flow = initFlow;

    if (abs(det) > 1e-6 && minEig > lkParams.minEigenvalue) {
        // Solve: [Vx, Vy] = A^-1 * b
        let invDet = 1.0 / det;
        let vx = (sumIyIy * (-sumIxIt) - sumIxIy * (-sumIyIt)) * invDet;
        let vy = (sumIxIx * (-sumIyIt) - sumIxIy * (-sumIxIt)) * invDet;

        // Add to initial estimate from coarser level
        flow = initFlow + vec2f(vx, vy);
    }

    textureStore(flowOutput, center, vec4f(flow, 0.0, 1.0));
}

// ==================== FLOW STATISTICS REDUCTION ====================
// Computes motion statistics from flow field using atomics

struct FlowStats {
    sumMagnitude: atomic<u32>,      // Fixed-point (x1000)
    sumMagnitudeSq: atomic<u32>,    // For variance
    sumVx: atomic<i32>,             // Fixed-point (x1000)
    sumVy: atomic<i32>,             // Fixed-point (x1000)
    pixelCount: atomic<u32>,
    significantPixels: atomic<u32>, // Pixels with magnitude > threshold
    maxMagnitude: atomic<u32>,      // Fixed-point (x1000)
    directionHistogram: array<atomic<u32>, 8>,  // 8 direction bins (45 deg each)
};

struct StatsParams {
    magnitudeThreshold: f32,  // Minimum flow magnitude to count as motion
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
};

@group(0) @binding(0) var statsFlow: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> stats: FlowStats;
@group(0) @binding(2) var<uniform> statsParams: StatsParams;

@compute @workgroup_size(8, 8)
fn flowStatisticsMain(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(statsFlow);
    if (id.x >= dims.x || id.y >= dims.y) { return; }

    let flow = textureLoad(statsFlow, vec2i(id.xy), 0).rg;
    let magnitude = length(flow);

    // Convert to fixed-point for atomic operations (scale by 1000)
    let magFixed = u32(clamp(magnitude * 1000.0, 0.0, 1000000.0));
    let magSqFixed = u32(clamp(magnitude * magnitude * 1000.0, 0.0, 1000000.0));
    let vxFixed = i32(clamp(flow.x * 1000.0, -1000000.0, 1000000.0));
    let vyFixed = i32(clamp(flow.y * 1000.0, -1000000.0, 1000000.0));

    // Accumulate statistics
    atomicAdd(&stats.sumMagnitude, magFixed);
    atomicAdd(&stats.sumMagnitudeSq, magSqFixed);
    atomicAdd(&stats.sumVx, vxFixed);
    atomicAdd(&stats.sumVy, vyFixed);
    atomicAdd(&stats.pixelCount, 1u);

    // Track max magnitude
    atomicMax(&stats.maxMagnitude, magFixed);

    // Count significant motion pixels
    if (magnitude > statsParams.magnitudeThreshold) {
        atomicAdd(&stats.significantPixels, 1u);

        // Direction histogram (8 bins, 45 degrees each)
        let angle = atan2(flow.y, flow.x);  // -PI to PI
        let normalizedAngle = (angle + PI) / (2.0 * PI);  // 0 to 1
        let bin = u32(clamp(normalizedAngle * 8.0, 0.0, 7.0));
        atomicAdd(&stats.directionHistogram[bin], 1u);
    }
}

// ==================== STATS BUFFER CLEAR ====================
// Clears the statistics buffer before each frame

@group(0) @binding(0) var<storage, read_write> clearStats: FlowStats;

@compute @workgroup_size(1)
fn clearStatsMain() {
    atomicStore(&clearStats.sumMagnitude, 0u);
    atomicStore(&clearStats.sumMagnitudeSq, 0u);
    atomicStore(&clearStats.sumVx, 0i);
    atomicStore(&clearStats.sumVy, 0i);
    atomicStore(&clearStats.pixelCount, 0u);
    atomicStore(&clearStats.significantPixels, 0u);
    atomicStore(&clearStats.maxMagnitude, 0u);
    for (var i: u32 = 0u; i < 8u; i++) {
        atomicStore(&clearStats.directionHistogram[i], 0u);
    }
}
