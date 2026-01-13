//! Protocol module - message types and encoding

mod commands;
pub mod frame;

pub use commands::*;
pub use frame::{DecodedFrame, FrameHeader, encode_frame_message, parse_frame_message};

/// Magic bytes for binary messages
pub const MAGIC: &[u8; 2] = b"MH";

/// Message types
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageType {
    Command = 0x01,
    Frame = 0x02,
    Response = 0x03,
    Error = 0x04,
    Progress = 0x05,
}

impl From<u8> for MessageType {
    fn from(v: u8) -> Self {
        match v {
            0x01 => MessageType::Command,
            0x02 => MessageType::Frame,
            0x03 => MessageType::Response,
            0x04 => MessageType::Error,
            0x05 => MessageType::Progress,
            _ => MessageType::Error,
        }
    }
}

/// Frame flags
pub mod flags {
    pub const COMPRESSED: u8 = 0x01;
    pub const SCALED: u8 = 0x02;
    pub const DELTA: u8 = 0x04;
}
