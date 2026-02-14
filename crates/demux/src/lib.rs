//! `ms-demux` — Custom MP4/MKV container parser.
//!
//! Extracts NAL units from video containers for hardware decoding.
//! No FFmpeg dependency — fully custom parser.

pub mod mp4;
pub mod nal;
pub mod packet;
pub mod probe;
pub mod traits;
