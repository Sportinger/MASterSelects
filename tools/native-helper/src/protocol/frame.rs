//! Binary frame message encoding/decoding

use super::{flags, MessageType, MAGIC};
use super::commands::Compression;

/// Default JPEG quality (1-100). 85 gives good quality at ~150KB for 1080p
const JPEG_QUALITY: u8 = 85;

/// Decoded frame data (raw RGBA pixels)
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

    pub fn with_jpeg(mut self) -> Self {
        self.flags |= flags::JPEG;
        self
    }

    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..2].copy_from_slice(MAGIC);
        buf[2] = self.msg_type as u8;
        buf[3] = self.flags;
        buf[4..6].copy_from_slice(&self.width.to_le_bytes());
        buf[6..8].copy_from_slice(&self.height.to_le_bytes());
        buf[8..12].copy_from_slice(&self.frame_num.to_le_bytes());
        buf[12..16].copy_from_slice(&self.request_id.to_le_bytes());
        buf
    }

    pub fn from_bytes(buf: &[u8]) -> Option<Self> {
        if buf.len() < Self::SIZE {
            return None;
        }
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

    pub fn is_jpeg(&self) -> bool {
        self.flags & flags::JPEG != 0
    }
}

/// Encode RGBA pixels to JPEG bytes
fn rgba_to_jpeg(data: &[u8], width: u32, height: u32, quality: u8) -> Option<Vec<u8>> {
    use image::{ImageBuffer, Rgba, codecs::jpeg::JpegEncoder};
    use std::io::Cursor;

    let img: ImageBuffer<Rgba<u8>, _> = ImageBuffer::from_raw(width, height, data.to_vec())?;
    // Convert RGBA to RGB (JPEG doesn't support alpha)
    let rgb_img = image::DynamicImage::ImageRgba8(img).into_rgb8();

    let mut jpeg_buf = Vec::with_capacity((width * height) as usize / 4); // rough estimate
    let mut cursor = Cursor::new(&mut jpeg_buf);
    let encoder = JpegEncoder::new_with_quality(&mut cursor, quality);
    rgb_img.write_with_encoder(encoder).ok()?;
    Some(jpeg_buf)
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
        Some(Compression::Jpeg) => {
            // Encode as JPEG — typically 50x smaller than raw RGBA
            if let Some(jpeg) = rgba_to_jpeg(&frame.data, frame.width, frame.height, JPEG_QUALITY) {
                header = header.with_jpeg();
                jpeg
            } else {
                // Fallback to raw if JPEG encoding fails
                frame.data.clone()
            }
        }
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

    // JPEG frames: return raw JPEG bytes (browser decodes natively)
    if header.is_jpeg() {
        return Some((header, data[payload_start..].to_vec()));
    }

    let payload = if header.is_compressed() {
        lz4_flex::decompress_size_prepended(&data[payload_start..]).ok()?
    } else {
        data[payload_start..].to_vec()
    };

    Some((header, payload))
}
