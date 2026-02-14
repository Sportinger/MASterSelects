//! Layer description — interface between timeline-eval and compositor.

use serde::{Deserialize, Serialize};

use crate::blend::BlendMode;
use crate::effect::EffectInstance;
use crate::types::SourceId;

/// 2D transform applied to a layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Transform2D {
    /// Position in pixels (from composition top-left).
    pub position: [f32; 2],
    /// Scale factors (1.0 = original size).
    pub scale: [f32; 2],
    /// Rotation in degrees.
    pub rotation: f32,
    /// Anchor point (0.5, 0.5 = center).
    pub anchor: [f32; 2],
}

impl Default for Transform2D {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0],
            scale: [1.0, 1.0],
            rotation: 0.0,
            anchor: [0.5, 0.5],
        }
    }
}

/// Mask type for a layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MaskShape {
    Rect {
        x: f32,
        y: f32,
        width: f32,
        height: f32,
    },
    Ellipse {
        cx: f32,
        cy: f32,
        rx: f32,
        ry: f32,
    },
    Path {
        points: Vec<[f32; 2]>,
        closed: bool,
    },
}

/// Mask applied to a layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MaskDesc {
    pub shape: MaskShape,
    /// Feather radius in pixels.
    pub feather: f32,
    /// Opacity of the mask (0..1).
    pub opacity: f32,
    /// Invert mask.
    pub inverted: bool,
    /// Expansion in pixels (positive = grow, negative = shrink).
    pub expansion: f32,
}

impl Default for MaskDesc {
    fn default() -> Self {
        Self {
            shape: MaskShape::Rect {
                x: 0.0,
                y: 0.0,
                width: 1.0,
                height: 1.0,
            },
            feather: 0.0,
            opacity: 1.0,
            inverted: false,
            expansion: 0.0,
        }
    }
}

/// Complete description of a layer to render at a given time.
/// Produced by timeline-eval, consumed by compositor.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerDesc {
    /// Source media identifier.
    pub source_id: SourceId,
    /// 2D transform.
    pub transform: Transform2D,
    /// Layer opacity (0..1).
    pub opacity: f32,
    /// Blend mode for compositing.
    pub blend_mode: BlendMode,
    /// Applied effects (in order).
    pub effects: Vec<EffectInstance>,
    /// Optional mask.
    pub mask: Option<MaskDesc>,
    /// Z-order (lower = behind).
    pub z_order: i32,
}

impl LayerDesc {
    pub fn new(source_id: SourceId) -> Self {
        Self {
            source_id,
            transform: Transform2D::default(),
            opacity: 1.0,
            blend_mode: BlendMode::default(),
            effects: Vec::new(),
            mask: None,
            z_order: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_transform_is_identity() {
        let t = Transform2D::default();
        assert_eq!(t.position, [0.0, 0.0]);
        assert_eq!(t.scale, [1.0, 1.0]);
        assert_eq!(t.rotation, 0.0);
        assert_eq!(t.anchor, [0.5, 0.5]);
    }

    #[test]
    fn layer_desc_builder() {
        let layer = LayerDesc::new(SourceId::new("clip_001"));
        assert_eq!(layer.opacity, 1.0);
        assert_eq!(layer.blend_mode, BlendMode::Normal);
        assert!(layer.effects.is_empty());
        assert!(layer.mask.is_none());
    }
}
