//! Keying effects.
//!
//! - [`ChromaKeyEffect`] — Chroma key (green/blue screen)
//! - [`LumaKeyEffect`] — Luma key (luminance-based)

pub mod chroma_key;
pub mod luma_key;

pub use chroma_key::ChromaKeyEffect;
pub use luma_key::LumaKeyEffect;
