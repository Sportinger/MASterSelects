//! Video decoder module

mod ffmpeg;
mod pool;

pub use ffmpeg::{VideoDecoder, DecodedFrameData, detect_hw_accel};
pub use pool::{DecoderPool, SharedDecoderPool};
