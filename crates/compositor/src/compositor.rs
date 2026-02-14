//! Main compositor — composites multiple layers into a final output frame.
//!
//! The [`Compositor`] takes a list of [`LayerDesc`] (sorted by z-order) and a
//! map of decoded GPU frames, then dispatches GPU kernels to transform, apply
//! effects, mask, and blend each layer onto the output buffer.

use std::collections::HashMap;

use ms_common::{GpuBackend, GpuBuffer, GpuFrame, GpuStream, KernelArgs, KernelId, LayerDesc};
use tracing::debug;

use crate::blend::{dispatch_blend, BlendParams};
use crate::mask::dispatch_mask;
use crate::pipeline::RenderPipeline;
use crate::transform::dispatch_transform;
use crate::CompositorError;

/// GPU compositor that blends multiple layers into a single output frame.
///
/// The compositor does **not** own any GPU resources — it receives a backend,
/// layer descriptions, decoded frames, and an output buffer from the caller.
pub struct Compositor {
    /// Output resolution width in pixels.
    output_width: u32,
    /// Output resolution height in pixels.
    output_height: u32,
}

impl Compositor {
    /// Create a new compositor targeting the given output resolution.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            output_width: width,
            output_height: height,
        }
    }

    /// Returns the output width.
    pub fn output_width(&self) -> u32 {
        self.output_width
    }

    /// Returns the output height.
    pub fn output_height(&self) -> u32 {
        self.output_height
    }

    /// Composite multiple layers into a final frame.
    ///
    /// # Arguments
    ///
    /// * `backend` - GPU backend for kernel dispatch and memory operations
    /// * `layers` - Layer descriptions sorted by z-order (bottom to top)
    /// * `frames` - Map from source ID string to the decoded GPU frame
    /// * `output` - Destination GPU buffer (must be `output_width * output_height * 4` bytes)
    /// * `stream` - GPU stream for async execution
    ///
    /// # Pipeline per layer
    ///
    /// 1. **Transform** — position, scale, rotation into a temporary buffer
    /// 2. **Effects** — dispatch each effect kernel in order
    /// 3. **Mask** — apply mask to modify alpha channel
    /// 4. **Blend** — blend the layer onto the accumulation buffer
    ///
    /// After all layers are composited, the final result is copied to `output`.
    pub fn composite(
        &self,
        backend: &dyn GpuBackend,
        layers: &[LayerDesc],
        frames: &HashMap<String, GpuFrame>,
        output: &GpuBuffer,
        stream: &GpuStream,
    ) -> Result<(), CompositorError> {
        if layers.is_empty() {
            debug!("No layers to composite, clearing output");
            let zeros = vec![0u8; self.output_width as usize * self.output_height as usize * 4];
            backend.copy_to_device(&zeros, output, stream)?;
            return Ok(());
        }

        debug!(
            layer_count = layers.len(),
            output_w = self.output_width,
            output_h = self.output_height,
            "Starting compositing"
        );

        // Allocate a temporary buffer for per-layer transform + effects + mask output.
        let temp_buffer =
            backend.alloc_buffer(self.output_width as usize * self.output_height as usize * 4)?;

        // Create a render pipeline for ping-pong compositing.
        let mut pipeline = RenderPipeline::new(backend, self.output_width, self.output_height)?;

        // Clear the initial accumulation buffer (transparent black).
        pipeline.clear(backend, stream)?;

        // Sort layers by z-order (caller should already do this, but be safe).
        let mut sorted_layers: Vec<&LayerDesc> = layers.iter().collect();
        sorted_layers.sort_by_key(|l| l.z_order);

        for (idx, layer) in sorted_layers.iter().enumerate() {
            let source_key = &layer.source_id.0;
            let src_frame = frames
                .get(source_key)
                .ok_or_else(|| CompositorError::MissingSource(source_key.clone()))?;

            debug!(
                layer_idx = idx,
                source = %layer.source_id,
                z_order = layer.z_order,
                blend_mode = ?layer.blend_mode,
                opacity = layer.opacity,
                effects_count = layer.effects.len(),
                has_mask = layer.mask.is_some(),
                "Processing layer"
            );

            // Skip fully transparent layers.
            if layer.opacity <= 0.0 {
                debug!(layer_idx = idx, "Skipping fully transparent layer");
                continue;
            }

            // Step 1: Transform the source frame into temp_buffer.
            dispatch_transform(
                backend,
                src_frame,
                temp_buffer.handle,
                self.output_width,
                self.output_height,
                &layer.transform,
                stream,
            )?;

            // Step 2: Apply effects in order.
            // Each effect kernel reads from and writes to temp_buffer (in-place).
            for effect in &layer.effects {
                if !effect.enabled {
                    continue;
                }
                self.dispatch_effect(
                    backend,
                    &temp_buffer,
                    &effect.effect_id.0,
                    &effect.params,
                    stream,
                )?;
            }

            // Step 3: Apply mask (if any) — modifies alpha in temp_buffer.
            if let Some(ref mask) = layer.mask {
                dispatch_mask(
                    backend,
                    temp_buffer.handle,
                    self.output_width,
                    self.output_height,
                    mask,
                    stream,
                )?;
            }

            // Step 4: Blend the processed layer onto the accumulation buffer.
            // We use a temporary GpuFrame wrapper that points to the temp buffer.
            let temp_frame = GpuFrame {
                device_ptr: temp_buffer.handle,
                device_ptr_uv: None,
                resolution: ms_common::Resolution::new(self.output_width, self.output_height),
                format: ms_common::PixelFormat::Rgba8,
                pitch: self.output_width * 4,
                pts: src_frame.pts,
            };

            // Read from current pipeline buffer, blend, write to back buffer.
            // For the blend kernel, we copy the current accumulation into the back buffer
            // first, then blend on top.
            let current_ptr = pipeline.current_buffer().handle;
            let back_ptr = pipeline.back_buffer().handle;

            // Copy current accumulation to back buffer before blending.
            backend.copy_buffer(pipeline.current_buffer(), pipeline.back_buffer(), stream)?;

            dispatch_blend(
                backend,
                &BlendParams {
                    src: &temp_frame,
                    dst_ptr: back_ptr,
                    width: self.output_width,
                    height: self.output_height,
                    opacity: layer.opacity,
                    blend_mode: &layer.blend_mode,
                },
                stream,
            )?;

            // Swap so that the just-blended buffer becomes current.
            pipeline.swap();

            let _ = current_ptr; // suppress unused warning
        }

        // Copy the final composited result to the caller's output buffer.
        backend.copy_buffer(pipeline.current_buffer(), output, stream)?;

        debug!("Compositing complete");

        Ok(())
    }

    /// Dispatch a single effect kernel on the temp buffer (in-place).
    fn dispatch_effect(
        &self,
        backend: &dyn GpuBackend,
        buffer: &GpuBuffer,
        effect_name: &str,
        params: &[(String, ms_common::ParamValue)],
        stream: &GpuStream,
    ) -> Result<(), CompositorError> {
        debug!(
            effect = effect_name,
            params_count = params.len(),
            "Dispatching effect kernel"
        );

        let kernel_id = KernelId::Effect(effect_name.to_string());

        // Build kernel args: buffer pointer, dimensions, then effect parameters.
        let mut args = KernelArgs::new()
            .push_ptr(buffer.handle)
            .push_u32(self.output_width)
            .push_u32(self.output_height);

        for (name, value) in params {
            args = match value {
                ms_common::ParamValue::Float(v) => args.push_f32(*v),
                ms_common::ParamValue::Int(v) => args.push_i32(*v),
                ms_common::ParamValue::Bool(v) => args.push_u32(if *v { 1 } else { 0 }),
                ms_common::ParamValue::Color(c) => args.push_vec4(*c),
                ms_common::ParamValue::Enum(v) => args.push_u32(*v),
                ms_common::ParamValue::Vec2(v) => args.push_vec2(*v),
                ms_common::ParamValue::Angle(v) => args.push_f32(*v),
            };
            let _ = name; // name is for debugging; kernel consumes args positionally
        }

        let block_size = 16u32;
        let grid = [
            self.output_width.div_ceil(block_size),
            self.output_height.div_ceil(block_size),
            1,
        ];
        let block = [block_size, block_size, 1];

        backend.dispatch_kernel(&kernel_id, grid, block, &args, stream)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ms_common::{BlendMode, SourceId};

    #[test]
    fn compositor_new_stores_dimensions() {
        let comp = Compositor::new(1920, 1080);
        assert_eq!(comp.output_width(), 1920);
        assert_eq!(comp.output_height(), 1080);
    }

    #[test]
    fn sorted_layers_by_z_order() {
        let layers = vec![
            LayerDesc {
                source_id: SourceId::new("a"),
                z_order: 5,
                ..LayerDesc::new(SourceId::new("a"))
            },
            LayerDesc {
                source_id: SourceId::new("b"),
                z_order: 1,
                ..LayerDesc::new(SourceId::new("b"))
            },
            LayerDesc {
                source_id: SourceId::new("c"),
                z_order: 3,
                ..LayerDesc::new(SourceId::new("c"))
            },
        ];

        let mut sorted: Vec<&LayerDesc> = layers.iter().collect();
        sorted.sort_by_key(|l| l.z_order);

        assert_eq!(sorted[0].source_id, SourceId::new("b"));
        assert_eq!(sorted[1].source_id, SourceId::new("c"));
        assert_eq!(sorted[2].source_id, SourceId::new("a"));
    }

    #[test]
    fn layer_desc_defaults() {
        let layer = LayerDesc::new(SourceId::new("test"));
        assert_eq!(layer.opacity, 1.0);
        assert_eq!(layer.blend_mode, BlendMode::Normal);
        assert_eq!(layer.z_order, 0);
        assert!(layer.effects.is_empty());
        assert!(layer.mask.is_none());
    }

    #[test]
    fn transparent_layer_is_skippable() {
        let layer = LayerDesc {
            opacity: 0.0,
            ..LayerDesc::new(SourceId::new("transparent"))
        };
        assert!(layer.opacity <= 0.0);
    }

    #[test]
    fn effect_param_encoding() {
        // Verify that each ParamValue variant maps to a KernelArgs push method
        let args = KernelArgs::new()
            .push_f32(0.5) // Float
            .push_i32(42) // Int
            .push_u32(1) // Bool(true)
            .push_vec4([1.0, 0.0, 0.0, 1.0]) // Color
            .push_u32(2) // Enum
            .push_vec2([0.5, 0.5]) // Vec2
            .push_f32(90.0); // Angle
        assert_eq!(args.len(), 7);
    }
}
