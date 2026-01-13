//! Binary frame message encoding/decoding

use super::{flags, MessageType, MAGIC};
use crate::protocol::Compression;

/// Decoded frame data
#[derive(Debug, Clone)]
pub struct DecodedFrame {
    pub width: u32,
    pub height: u32,
    pub frame_num: u32,
    pub data: Vec<u8>,
}

/// Frame message header (16 bytes)
#[derive(Debug, Clone, Copy)]
pub struct FrameHeader {
    pub msg_type: MessageType,
    pub flags: u8,
    pub width: u16,
    pub height: u16,
    pub frame_num: u32,
    pub request_id: u32,
}

impl FrameHeader {
    pub const SIZE: usize = 16;

    pub fn new(width: u32, height: u32, frame_num: u32, request_id: u32) -> Self {
        Self {
            msg_type: MessageType::Frame,
            flags: 0,
            width: width as u16,
            height: height as u16,
            frame_num,
            request_id,
        }
    }

    pub fn with_compression(mut self) -> Self {
        self.flags |= flags::COMPRESSED;
        self
    }

    pub fn with_scaled(mut self) -> Self {
        self.flags |= flags::SCALED;
        self
    }

    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];

        // Magic (2 bytes)
        buf[0..2].copy_from_slice(MAGIC);

        // Type (1 byte)
        buf[2] = self.msg_type as u8;

        // Flags (1 byte)
        buf[3] = self.flags;

        // Width (2 bytes, little endian)
        buf[4..6].copy_from_slice(&self.width.to_le_bytes());

        // Height (2 bytes, little endian)
        buf[6..8].copy_from_slice(&self.height.to_le_bytes());

        // Frame number (4 bytes, little endian)
        buf[8..12].copy_from_slice(&self.frame_num.to_le_bytes());

        // Request ID (4 bytes, little endian)
        buf[12..16].copy_from_slice(&self.request_id.to_le_bytes());

        buf
    }

    pub fn from_bytes(buf: &[u8]) -> Option<Self> {
        if buf.len() < Self::SIZE {
            return None;
        }

        // Check magic
        if &buf[0..2] != MAGIC {
            return None;
        }

        Some(Self {
            msg_type: MessageType::from(buf[2]),
            flags: buf[3],
            width: u16::from_le_bytes([buf[4], buf[5]]),
            height: u16::from_le_bytes([buf[6], buf[7]]),
            frame_num: u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]),
            request_id: u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]),
        })
    }

    pub fn is_compressed(&self) -> bool {
        self.flags & flags::COMPRESSED != 0
    }

    pub fn is_scaled(&self) -> bool {
        self.flags & flags::SCALED != 0
    }
}

/// Encode a frame into a binary message
pub fn encode_frame_message(
    frame: &DecodedFrame,
    compression: Option<Compression>,
    request_id: u32,
    scaled: bool,
) -> Vec<u8> {
    let mut header = FrameHeader::new(frame.width, frame.height, frame.frame_num, request_id);

    if scaled {
        header = header.with_scaled();
    }

    let payload = match compression {
        Some(Compression::Lz4) => {
            header = header.with_compression();
            lz4_flex::compress_prepend_size(&frame.data)
        }
        None => frame.data.clone(),
    };

    let mut msg = Vec::with_capacity(FrameHeader::SIZE + payload.len());
    msg.extend_from_slice(&header.to_bytes());
    msg.extend_from_slice(&payload);

    msg
}

/// Parse a binary frame message
pub fn parse_frame_message(data: &[u8]) -> Option<(FrameHeader, Vec<u8>)> {
    let header = FrameHeader::from_bytes(data)?;

    let payload_start = FrameHeader::SIZE;
    if data.len() <= payload_start {
        return None;
    }

    let payload = if header.is_compressed() {
        // Decompress LZ4
        lz4_flex::decompress_size_prepended(&data[payload_start..]).ok()?
    } else {
        data[payload_start..].to_vec()
    };

    Some((header, payload))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_roundtrip() {
        let header = FrameHeader::new(1920, 1080, 42, 123)
            .with_compression()
            .with_scaled();

        let bytes = header.to_bytes();
        let parsed = FrameHeader::from_bytes(&bytes).unwrap();

        assert_eq!(parsed.width, 1920);
        assert_eq!(parsed.height, 1080);
        assert_eq!(parsed.frame_num, 42);
        assert_eq!(parsed.request_id, 123);
        assert!(parsed.is_compressed());
        assert!(parsed.is_scaled());
    }

    #[test]
    fn test_frame_message_uncompressed() {
        let frame = DecodedFrame {
            width: 100,
            height: 100,
            frame_num: 0,
            data: vec![0u8; 100 * 100 * 4],
        };

        let msg = encode_frame_message(&frame, None, 1, false);

        let (header, payload) = parse_frame_message(&msg).unwrap();
        assert_eq!(header.width, 100);
        assert_eq!(header.height, 100);
        assert!(!header.is_compressed());
        assert_eq!(payload.len(), 100 * 100 * 4);
    }

    #[test]
    fn test_frame_message_compressed() {
        let frame = DecodedFrame {
            width: 100,
            height: 100,
            frame_num: 5,
            data: vec![128u8; 100 * 100 * 4], // Uniform data compresses well
        };

        let msg = encode_frame_message(&frame, Some(Compression::Lz4), 2, true);

        let (header, payload) = parse_frame_message(&msg).unwrap();
        assert_eq!(header.width, 100);
        assert_eq!(header.height, 100);
        assert_eq!(header.frame_num, 5);
        assert!(header.is_compressed());
        assert!(header.is_scaled());
        // Decompressed payload should match original
        assert_eq!(payload.len(), 100 * 100 * 4);
        assert_eq!(payload[0], 128);
    }
}
