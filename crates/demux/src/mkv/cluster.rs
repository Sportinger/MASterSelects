//! Cluster and block parsing for Matroska/WebM containers.
//!
//! Parses `SimpleBlock` and `Block` elements to extract frame data,
//! track numbers, timecode offsets, and keyframe flags.
//!
//! ## SimpleBlock format
//!
//! ```text
//! [track_number: vint] [timecode: int16, relative to cluster] [flags: u8] [frame_data...]
//! ```
//!
//! Flag bits:
//! - bit 7 (0x80): keyframe
//! - bit 3 (0x08): invisible
//! - bits 1-2 (0x06): lacing type (00=none, 01=Xiph, 11=EBML, 10=fixed-size)

use ms_common::DemuxError;

/// Parsed information from a SimpleBlock or Block element.
#[derive(Clone, Debug)]
pub struct SimpleBlockInfo {
    /// Track number this block belongs to.
    pub track_number: u64,
    /// Timecode offset relative to the cluster timecode (signed 16-bit).
    pub timecode_offset: i16,
    /// Whether this frame is a keyframe (only meaningful for SimpleBlock).
    pub is_keyframe: bool,
    /// Whether this frame should not be rendered (invisible flag).
    pub is_invisible: bool,
    /// Raw frame data (single frame, lacing not yet supported for multi-frame blocks).
    pub frame_data: Vec<u8>,
}

/// Lacing type for Matroska blocks.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum LacingType {
    /// No lacing: single frame per block.
    None,
    /// Xiph lacing.
    Xiph,
    /// EBML lacing.
    Ebml,
    /// Fixed-size lacing.
    FixedSize,
}

/// Parse a SimpleBlock element body.
///
/// The `data` slice starts immediately after the element header (ID + size),
/// i.e., it contains: `[track_vint][timecode_i16][flags_u8][frame_data...]`
pub fn parse_simple_block(data: &[u8]) -> Result<SimpleBlockInfo, DemuxError> {
    if data.is_empty() {
        return Err(DemuxError::InvalidStructure {
            offset: 0,
            reason: "Empty SimpleBlock data".into(),
        });
    }

    let mut offset = 0;

    // Read track number (VINT without stripping the leading-1 marker,
    // just like a normal VINT — but for blocks it's a "vint" that keeps
    // the marker bit stripped, like a size vint).
    let (track_number, vint_len) = read_block_vint(&data[offset..])?;
    offset += vint_len;

    // Read timecode offset (signed 16-bit, big-endian)
    if offset + 2 > data.len() {
        return Err(DemuxError::TruncatedData {
            expected: offset + 2,
            got: data.len(),
        });
    }
    let timecode_offset = i16::from_be_bytes([data[offset], data[offset + 1]]);
    offset += 2;

    // Read flags byte
    if offset >= data.len() {
        return Err(DemuxError::TruncatedData {
            expected: offset + 1,
            got: data.len(),
        });
    }
    let flags = data[offset];
    offset += 1;

    let is_keyframe = flags & 0x80 != 0;
    let is_invisible = flags & 0x08 != 0;
    let lacing_type = match (flags >> 1) & 0x03 {
        0b00 => LacingType::None,
        0b01 => LacingType::Xiph,
        0b11 => LacingType::Ebml,
        0b10 => LacingType::FixedSize,
        _ => unreachable!(),
    };

    // Extract frame data.
    // For non-laced blocks: everything remaining is one frame.
    // For laced blocks: parse lacing header + extract first frame.
    let frame_data = match lacing_type {
        LacingType::None => data[offset..].to_vec(),
        LacingType::Xiph => parse_xiph_laced_first_frame(&data[offset..])?,
        LacingType::FixedSize => parse_fixed_laced_first_frame(&data[offset..])?,
        LacingType::Ebml => parse_ebml_laced_first_frame(&data[offset..])?,
    };

    Ok(SimpleBlockInfo {
        track_number,
        timecode_offset,
        is_keyframe,
        is_invisible,
        frame_data,
    })
}

/// Parse a Block element body (same format as SimpleBlock but keyframe
/// is determined by the enclosing BlockGroup's ReferenceBlock elements).
///
/// `has_reference` indicates whether a ReferenceBlock was present in the
/// BlockGroup. If false, the block is a keyframe.
pub fn parse_block(data: &[u8], has_reference: bool) -> Result<SimpleBlockInfo, DemuxError> {
    let mut info = parse_simple_block(data)?;
    // In a Block (not SimpleBlock), the keyframe flag in the flags byte is unused.
    // The keyframe status is derived from the absence of ReferenceBlock elements.
    info.is_keyframe = !has_reference;
    Ok(info)
}

// ─── Internal: block VINT parsing ────────────────────────────────────

/// Read a VINT from a block header (track number encoding).
///
/// Unlike EBML element IDs, the track number VINT in blocks strips the
/// leading-1 marker (same as the size VINT encoding).
///
/// Returns `(value, bytes_consumed)`.
fn read_block_vint(data: &[u8]) -> Result<(u64, usize), DemuxError> {
    if data.is_empty() {
        return Err(DemuxError::TruncatedData {
            expected: 1,
            got: 0,
        });
    }

    let first = data[0];
    let width = block_vint_width(first)?;

    if data.len() < width {
        return Err(DemuxError::TruncatedData {
            expected: width,
            got: data.len(),
        });
    }

    // Strip the leading-1 marker from the first byte
    let mask = 0xFF >> width;
    let mut value = (first & mask) as u64;

    for &byte in data.iter().take(width).skip(1) {
        value = (value << 8) | byte as u64;
    }

    Ok((value, width))
}

/// Determine the width of a VINT in a block header from its first byte.
fn block_vint_width(first: u8) -> Result<usize, DemuxError> {
    if first & 0x80 != 0 {
        Ok(1)
    } else if first & 0x40 != 0 {
        Ok(2)
    } else if first & 0x20 != 0 {
        Ok(3)
    } else if first & 0x10 != 0 {
        Ok(4)
    } else {
        Err(DemuxError::InvalidStructure {
            offset: 0,
            reason: format!(
                "Invalid block VINT leading byte: 0x{first:02X}"
            ),
        })
    }
}

// ─── Lacing helpers ──────────────────────────────────────────────────

/// Parse Xiph lacing and return the first frame.
///
/// Xiph lacing header: `[num_frames_minus_1: u8] [frame_sizes...] [frame_data...]`
/// Each frame size is encoded as a sequence of 0xFF bytes followed by a
/// final byte < 0xFF.
fn parse_xiph_laced_first_frame(data: &[u8]) -> Result<Vec<u8>, DemuxError> {
    if data.is_empty() {
        return Err(DemuxError::TruncatedData {
            expected: 1,
            got: 0,
        });
    }

    let num_frames = data[0] as usize + 1;
    let mut offset = 1;

    if num_frames == 1 {
        // Only one frame, no sizes encoded
        return Ok(data[offset..].to_vec());
    }

    // Read sizes for the first (num_frames - 1) frames
    let mut sizes = Vec::with_capacity(num_frames - 1);
    for _ in 0..(num_frames - 1) {
        let mut size: usize = 0;
        loop {
            if offset >= data.len() {
                return Err(DemuxError::TruncatedData {
                    expected: offset + 1,
                    got: data.len(),
                });
            }
            let b = data[offset] as usize;
            offset += 1;
            size += b;
            if b < 255 {
                break;
            }
        }
        sizes.push(size);
    }

    // The first frame starts at `offset` and has size `sizes[0]`
    if !sizes.is_empty() {
        let first_size = sizes[0];
        if offset + first_size > data.len() {
            return Err(DemuxError::TruncatedData {
                expected: offset + first_size,
                got: data.len(),
            });
        }
        Ok(data[offset..offset + first_size].to_vec())
    } else {
        Ok(data[offset..].to_vec())
    }
}

/// Parse fixed-size lacing and return the first frame.
///
/// Fixed-size lacing header: `[num_frames_minus_1: u8]`
/// All frames have equal size: `remaining_data / num_frames`.
fn parse_fixed_laced_first_frame(data: &[u8]) -> Result<Vec<u8>, DemuxError> {
    if data.is_empty() {
        return Err(DemuxError::TruncatedData {
            expected: 1,
            got: 0,
        });
    }

    let num_frames = data[0] as usize + 1;
    let remaining = data.len() - 1;

    if num_frames == 0 || !remaining.is_multiple_of(num_frames) {
        return Err(DemuxError::InvalidStructure {
            offset: 0,
            reason: format!(
                "Fixed-size lacing: {remaining} bytes not evenly divisible by {num_frames} frames"
            ),
        });
    }

    let frame_size = remaining / num_frames;
    Ok(data[1..1 + frame_size].to_vec())
}

/// Parse EBML lacing and return the first frame.
///
/// EBML lacing header: `[num_frames_minus_1: u8] [first_size: vint] [subsequent_size_diffs: svint...]`
fn parse_ebml_laced_first_frame(data: &[u8]) -> Result<Vec<u8>, DemuxError> {
    if data.is_empty() {
        return Err(DemuxError::TruncatedData {
            expected: 1,
            got: 0,
        });
    }

    let num_frames = data[0] as usize + 1;
    let mut offset = 1;

    if num_frames == 1 {
        return Ok(data[offset..].to_vec());
    }

    // Read the first frame size as a VINT (unsigned)
    let (first_size, vint_len) = read_block_vint(&data[offset..])?;
    offset += vint_len;
    let first_size = first_size as usize;

    // Skip the remaining size diffs (we only need the first frame)
    // We need to skip (num_frames - 2) signed-vint diffs
    // But we just need to know where the frame data starts.
    // Each subsequent size is stored as a signed VINT diff from the previous.
    let mut _prev_size = first_size;
    for _ in 0..(num_frames - 2) {
        if offset >= data.len() {
            return Err(DemuxError::TruncatedData {
                expected: offset + 1,
                got: data.len(),
            });
        }
        // Read a VINT (used as signed diff)
        let (_val, vint_len) = read_block_vint(&data[offset..])?;
        offset += vint_len;
    }

    // The frame data starts at `offset`; the first frame is `first_size` bytes.
    if offset + first_size > data.len() {
        return Err(DemuxError::TruncatedData {
            expected: offset + first_size,
            got: data.len(),
        });
    }

    Ok(data[offset..offset + first_size].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a SimpleBlock with no lacing:
    /// [track_vint=1 (0x81)] [timecode=0 (0x00 0x00)] [flags] [frame_data]
    fn make_simple_block(flags: u8, frame_data: &[u8]) -> Vec<u8> {
        let mut data = Vec::new();
        data.push(0x81); // track number 1 as VINT
        data.extend_from_slice(&[0x00, 0x00]); // timecode offset = 0
        data.push(flags);
        data.extend_from_slice(frame_data);
        data
    }

    #[test]
    fn test_parse_simple_block_keyframe() {
        let frame = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let data = make_simple_block(0x80, &frame); // keyframe flag
        let info = parse_simple_block(&data).unwrap();

        assert_eq!(info.track_number, 1);
        assert_eq!(info.timecode_offset, 0);
        assert!(info.is_keyframe);
        assert!(!info.is_invisible);
        assert_eq!(info.frame_data, frame);
    }

    #[test]
    fn test_parse_simple_block_non_keyframe() {
        let frame = vec![0x01, 0x02, 0x03];
        let data = make_simple_block(0x00, &frame);
        let info = parse_simple_block(&data).unwrap();

        assert_eq!(info.track_number, 1);
        assert!(!info.is_keyframe);
        assert!(!info.is_invisible);
        assert_eq!(info.frame_data, frame);
    }

    #[test]
    fn test_parse_simple_block_invisible() {
        let frame = vec![0xAA];
        let data = make_simple_block(0x08, &frame); // invisible flag
        let info = parse_simple_block(&data).unwrap();

        assert!(info.is_invisible);
        assert!(!info.is_keyframe);
    }

    #[test]
    fn test_parse_simple_block_with_timecode_offset() {
        let mut data = Vec::new();
        data.push(0x81); // track 1
        data.extend_from_slice(&[0x00, 0x0A]); // timecode offset = 10
        data.push(0x80); // keyframe
        data.extend_from_slice(&[0xFF, 0xFE]);

        let info = parse_simple_block(&data).unwrap();
        assert_eq!(info.timecode_offset, 10);
    }

    #[test]
    fn test_parse_simple_block_negative_timecode() {
        let mut data = Vec::new();
        data.push(0x81); // track 1
        data.extend_from_slice(&(-5i16).to_be_bytes()); // timecode offset = -5
        data.push(0x80); // keyframe
        data.push(0xAA);

        let info = parse_simple_block(&data).unwrap();
        assert_eq!(info.timecode_offset, -5);
    }

    #[test]
    fn test_parse_simple_block_track_2() {
        let mut data = Vec::new();
        data.push(0x82); // track 2 as VINT (1000_0010)
        data.extend_from_slice(&[0x00, 0x00]); // timecode offset = 0
        data.push(0x80); // keyframe
        data.extend_from_slice(&[0x01, 0x02]);

        let info = parse_simple_block(&data).unwrap();
        assert_eq!(info.track_number, 2);
    }

    #[test]
    fn test_parse_simple_block_empty_data() {
        let data: Vec<u8> = vec![];
        assert!(parse_simple_block(&data).is_err());
    }

    #[test]
    fn test_parse_simple_block_truncated() {
        // Only 2 bytes, not enough for vint + timecode + flags
        let data = vec![0x81, 0x00];
        assert!(parse_simple_block(&data).is_err());
    }

    #[test]
    fn test_parse_block_no_reference() {
        let frame = vec![0xDE, 0xAD];
        let data = make_simple_block(0x00, &frame);
        let info = parse_block(&data, false).unwrap();

        // No reference block = keyframe
        assert!(info.is_keyframe);
    }

    #[test]
    fn test_parse_block_with_reference() {
        let frame = vec![0xDE, 0xAD];
        let data = make_simple_block(0x80, &frame); // flags say keyframe...
        let info = parse_block(&data, true).unwrap();

        // Has reference block = NOT a keyframe (overrides flag)
        assert!(!info.is_keyframe);
    }

    #[test]
    fn test_read_block_vint_1byte() {
        let data = [0x81]; // 1000_0001 -> value = 1
        let (val, len) = read_block_vint(&data).unwrap();
        assert_eq!(val, 1);
        assert_eq!(len, 1);
    }

    #[test]
    fn test_read_block_vint_2byte() {
        // 0x40 0x80 = 01_000000 10000000 -> value = 0x0080 = 128
        let data = [0x40, 0x80];
        let (val, len) = read_block_vint(&data).unwrap();
        assert_eq!(val, 128);
        assert_eq!(len, 2);
    }

    #[test]
    fn test_fixed_laced_first_frame() {
        // 2 frames (value = 1), 4 bytes remaining -> 2 bytes each
        let data = vec![0x01, 0xAA, 0xBB, 0xCC, 0xDD];
        let first = parse_fixed_laced_first_frame(&data).unwrap();
        assert_eq!(first, vec![0xAA, 0xBB]);
    }

    #[test]
    fn test_xiph_laced_first_frame() {
        // 2 frames (value = 1), first frame size = 3 (byte = 0x03)
        let mut data = vec![0x01, 0x03]; // num_frames-1=1, first frame size=3
        data.extend_from_slice(&[0xAA, 0xBB, 0xCC]); // first frame (3 bytes)
        data.extend_from_slice(&[0xDD, 0xEE]); // second frame (remaining)
        let first = parse_xiph_laced_first_frame(&data).unwrap();
        assert_eq!(first, vec![0xAA, 0xBB, 0xCC]);
    }
}
