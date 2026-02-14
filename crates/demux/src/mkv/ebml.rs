//! EBML (Extensible Binary Meta Language) parser.
//!
//! Provides variable-size integer reading and typed element value parsing
//! for Matroska/WebM containers. All integers in EBML are big-endian.
//!
//! EBML uses a leading-1 encoding for variable-size integers:
//! - 1 byte:  `1xxx xxxx`                (7 data bits)
//! - 2 bytes: `01xx xxxx xxxx xxxx`       (14 data bits)
//! - 3 bytes: `001x xxxx ...`             (21 data bits)
//! - 4 bytes: `0001 xxxx ...`             (28 data bits)

use ms_common::DemuxError;
use std::io::{Read, Seek, SeekFrom};

/// An EBML element header: the ID, data size, and position info.
#[derive(Clone, Debug)]
pub struct EbmlElement {
    /// The EBML element ID (1-4 bytes, encoded as u32).
    pub id: u32,
    /// The data size in bytes (may be unknown = `u64::MAX`).
    pub size: u64,
    /// How many bytes the header (ID + size) consumed.
    pub header_size: u64,
    /// Byte position in the stream where this element header starts.
    pub position: u64,
}

impl EbmlElement {
    /// Byte offset where the element's data (payload) begins.
    pub fn data_offset(&self) -> u64 {
        self.position + self.header_size
    }

    /// Byte offset just past the end of this element (position + header + size).
    /// Returns `None` if the element has unknown size.
    pub fn end_offset(&self) -> Option<u64> {
        if self.size == u64::MAX {
            None
        } else {
            Some(self.position + self.header_size + self.size)
        }
    }
}

/// Read a variable-size EBML ID from the reader.
///
/// Unlike data sizes, the ID keeps the leading-1 marker bit,
/// so the raw bytes form the ID directly.
pub fn read_vint_id<R: Read>(reader: &mut R) -> Result<u32, DemuxError> {
    let first = read_one_byte(reader)?;
    let width = vint_width(first)?;

    let mut id = first as u32;
    for _ in 1..width {
        let b = read_one_byte(reader)?;
        id = (id << 8) | b as u32;
    }

    Ok(id)
}

/// Read a variable-size EBML data size from the reader.
///
/// The leading-1 marker bit is stripped from the value.
/// Returns `u64::MAX` for the all-ones "unknown size" sentinel.
pub fn read_vint_size<R: Read>(reader: &mut R) -> Result<u64, DemuxError> {
    let first = read_one_byte(reader)?;
    let width = vint_width(first)?;

    // Mask out the leading-1 marker
    let mask = 0xFF >> width;
    let mut value = (first & mask) as u64;

    for _ in 1..width {
        let b = read_one_byte(reader)?;
        value = (value << 8) | b as u64;
    }

    // Check for the "unknown size" sentinel (all data bits set to 1).
    let max_for_width: u64 = (1u64 << (7 * width)) - 1;
    if value == max_for_width {
        return Ok(u64::MAX);
    }

    Ok(value)
}

/// Read a complete EBML element header (ID + data size) from the current
/// stream position.
pub fn read_element<R: Read + Seek>(reader: &mut R) -> Result<EbmlElement, DemuxError> {
    let position = reader.stream_position().map_err(DemuxError::Io)?;

    let id = read_vint_id(reader)?;
    let size = read_vint_size(reader)?;

    let after = reader.stream_position().map_err(DemuxError::Io)?;
    let header_size = after - position;

    Ok(EbmlElement {
        id,
        size,
        header_size,
        position,
    })
}

/// Read an unsigned integer element value (1-8 bytes, big-endian).
pub fn read_uint<R: Read>(reader: &mut R, size: u64) -> Result<u64, DemuxError> {
    if size == 0 || size > 8 {
        return Err(DemuxError::InvalidStructure {
            offset: 0,
            reason: format!("Invalid uint size: {size}"),
        });
    }
    let mut val: u64 = 0;
    for _ in 0..size {
        let b = read_one_byte(reader)?;
        val = (val << 8) | b as u64;
    }
    Ok(val)
}

/// Read a signed integer element value (1-8 bytes, big-endian, two's complement).
pub fn read_sint<R: Read>(reader: &mut R, size: u64) -> Result<i64, DemuxError> {
    if size == 0 || size > 8 {
        return Err(DemuxError::InvalidStructure {
            offset: 0,
            reason: format!("Invalid sint size: {size}"),
        });
    }

    let first = read_one_byte(reader)?;
    // Sign-extend the first byte
    let mut val: i64 = if first & 0x80 != 0 {
        -1i64 ^ 0xFF | first as i64
    } else {
        first as i64
    };

    for _ in 1..size {
        let b = read_one_byte(reader)?;
        val = (val << 8) | b as i64;
    }
    Ok(val)
}

/// Read a float element value (must be 0, 4, or 8 bytes).
pub fn read_float<R: Read>(reader: &mut R, size: u64) -> Result<f64, DemuxError> {
    match size {
        0 => Ok(0.0),
        4 => {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf).map_err(DemuxError::Io)?;
            Ok(f32::from_be_bytes(buf) as f64)
        }
        8 => {
            let mut buf = [0u8; 8];
            reader.read_exact(&mut buf).map_err(DemuxError::Io)?;
            Ok(f64::from_be_bytes(buf))
        }
        _ => Err(DemuxError::InvalidStructure {
            offset: 0,
            reason: format!("Invalid float size: {size} (must be 0, 4, or 8)"),
        }),
    }
}

/// Read a UTF-8 string element value.
pub fn read_string<R: Read>(reader: &mut R, size: u64) -> Result<String, DemuxError> {
    if size == 0 {
        return Ok(String::new());
    }
    let data = read_binary(reader, size)?;

    // Matroska strings may be null-terminated; strip trailing nulls
    let end = data.iter().position(|&b| b == 0).unwrap_or(data.len());
    String::from_utf8(data[..end].to_vec()).map_err(|e| DemuxError::InvalidStructure {
        offset: 0,
        reason: format!("Invalid UTF-8 string: {e}"),
    })
}

/// Read raw binary data of the given size.
pub fn read_binary<R: Read>(reader: &mut R, size: u64) -> Result<Vec<u8>, DemuxError> {
    let size_usize = size as usize;
    let mut buf = vec![0u8; size_usize];
    reader.read_exact(&mut buf).map_err(DemuxError::Io)?;
    Ok(buf)
}

/// Skip past `size` bytes in the reader.
pub fn skip_element<R: Read + Seek>(reader: &mut R, size: u64) -> Result<(), DemuxError> {
    if size == u64::MAX {
        return Err(DemuxError::InvalidStructure {
            offset: 0,
            reason: "Cannot skip element with unknown size".into(),
        });
    }
    reader
        .seek(SeekFrom::Current(size as i64))
        .map_err(DemuxError::Io)?;
    Ok(())
}

// ─── Internal helpers ────────────────────────────────────────────────

/// Read exactly one byte from the reader.
fn read_one_byte<R: Read>(reader: &mut R) -> Result<u8, DemuxError> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf).map_err(DemuxError::Io)?;
    Ok(buf[0])
}

/// Determine the width (1-4 bytes) of a VINT from its first byte.
fn vint_width(first: u8) -> Result<u8, DemuxError> {
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
                "Invalid VINT leading byte: 0x{first:02X} (no leading-1 in top 4 bits)"
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_read_vint_id_1byte() {
        // 0x83 = 1000_0011 -> ID = 0x83
        let mut cursor = Cursor::new(vec![0x83]);
        let id = read_vint_id(&mut cursor).unwrap();
        assert_eq!(id, 0x83);
    }

    #[test]
    fn test_read_vint_id_2byte() {
        // TrackEntry: 0xAE = 10101110 -> 1-byte ID (high bit set)
        let mut cursor = Cursor::new(vec![0xAE]);
        let id = read_vint_id(&mut cursor).unwrap();
        assert_eq!(id, 0xAE);
    }

    #[test]
    fn test_read_vint_id_4byte() {
        // EBML Header: 0x1A45DFA3
        // First byte 0x1A = 0001_1010 -> 4-byte ID
        let mut cursor = Cursor::new(vec![0x1A, 0x45, 0xDF, 0xA3]);
        let id = read_vint_id(&mut cursor).unwrap();
        assert_eq!(id, 0x1A45DFA3);
    }

    #[test]
    fn test_read_vint_id_segment() {
        // Segment: 0x18538067
        let mut cursor = Cursor::new(vec![0x18, 0x53, 0x80, 0x67]);
        let id = read_vint_id(&mut cursor).unwrap();
        assert_eq!(id, 0x18538067);
    }

    #[test]
    fn test_read_vint_size_1byte() {
        // 0x85 = 1000_0101 -> data bits: 000_0101 = 5
        let mut cursor = Cursor::new(vec![0x85]);
        let size = read_vint_size(&mut cursor).unwrap();
        assert_eq!(size, 5);
    }

    #[test]
    fn test_read_vint_size_2byte() {
        // 0x40 0x03 = 01_000000 00000011 -> data bits: 00_0000_0000_0011 = 3
        let mut cursor = Cursor::new(vec![0x40, 0x03]);
        let size = read_vint_size(&mut cursor).unwrap();
        assert_eq!(size, 3);
    }

    #[test]
    fn test_read_vint_size_unknown() {
        // Unknown 1-byte size: 0xFF -> all data bits = 1 -> u64::MAX
        let mut cursor = Cursor::new(vec![0xFF]);
        let size = read_vint_size(&mut cursor).unwrap();
        assert_eq!(size, u64::MAX);
    }

    #[test]
    fn test_read_vint_size_unknown_2byte() {
        // Unknown 2-byte size: 0x7F 0xFF -> all data bits = 1
        let mut cursor = Cursor::new(vec![0x7F, 0xFF]);
        let size = read_vint_size(&mut cursor).unwrap();
        assert_eq!(size, u64::MAX);
    }

    #[test]
    fn test_read_uint() {
        // 2-byte uint: 0x03E8 = 1000
        let mut cursor = Cursor::new(vec![0x03, 0xE8]);
        let val = read_uint(&mut cursor, 2).unwrap();
        assert_eq!(val, 1000);
    }

    #[test]
    fn test_read_uint_1byte() {
        let mut cursor = Cursor::new(vec![0x2A]);
        let val = read_uint(&mut cursor, 1).unwrap();
        assert_eq!(val, 42);
    }

    #[test]
    fn test_read_uint_3byte() {
        // 0x0F4240 = 1000000
        let mut cursor = Cursor::new(vec![0x0F, 0x42, 0x40]);
        let val = read_uint(&mut cursor, 3).unwrap();
        assert_eq!(val, 1_000_000);
    }

    #[test]
    fn test_read_uint_invalid_size_0() {
        let mut cursor = Cursor::new(vec![]);
        assert!(read_uint(&mut cursor, 0).is_err());
    }

    #[test]
    fn test_read_uint_invalid_size_9() {
        let mut cursor = Cursor::new(vec![0; 9]);
        assert!(read_uint(&mut cursor, 9).is_err());
    }

    #[test]
    fn test_read_sint_positive() {
        // 0x2A = 42
        let mut cursor = Cursor::new(vec![0x2A]);
        let val = read_sint(&mut cursor, 1).unwrap();
        assert_eq!(val, 42);
    }

    #[test]
    fn test_read_sint_negative() {
        // 0xFF = -1 as a signed byte
        let mut cursor = Cursor::new(vec![0xFF]);
        let val = read_sint(&mut cursor, 1).unwrap();
        assert_eq!(val, -1);
    }

    #[test]
    fn test_read_sint_negative_2byte() {
        // 0xFF 0xFE = -2 as a 2-byte signed int
        let mut cursor = Cursor::new(vec![0xFF, 0xFE]);
        let val = read_sint(&mut cursor, 2).unwrap();
        assert_eq!(val, -2);
    }

    #[test]
    fn test_read_float_4byte() {
        // IEEE 754 single: 42.0 = 0x42280000
        let bytes = 42.0_f32.to_be_bytes();
        let mut cursor = Cursor::new(bytes.to_vec());
        let val = read_float(&mut cursor, 4).unwrap();
        assert!((val - 42.0).abs() < 1e-6);
    }

    #[test]
    fn test_read_float_8byte() {
        // IEEE 754 double: 12345.6789
        let bytes = 12345.6789_f64.to_be_bytes();
        let mut cursor = Cursor::new(bytes.to_vec());
        let val = read_float(&mut cursor, 8).unwrap();
        assert!((val - 12345.6789).abs() < 1e-6);
    }

    #[test]
    fn test_read_float_0byte() {
        let mut cursor = Cursor::new(vec![]);
        let val = read_float(&mut cursor, 0).unwrap();
        assert_eq!(val, 0.0);
    }

    #[test]
    fn test_read_float_invalid_size() {
        let mut cursor = Cursor::new(vec![0; 3]);
        assert!(read_float(&mut cursor, 3).is_err());
    }

    #[test]
    fn test_read_string_basic() {
        let data = b"matroska".to_vec();
        let mut cursor = Cursor::new(data);
        let s = read_string(&mut cursor, 8).unwrap();
        assert_eq!(s, "matroska");
    }

    #[test]
    fn test_read_string_null_terminated() {
        let data = vec![b'h', b'i', 0x00, 0x00];
        let mut cursor = Cursor::new(data);
        let s = read_string(&mut cursor, 4).unwrap();
        assert_eq!(s, "hi");
    }

    #[test]
    fn test_read_string_empty() {
        let mut cursor = Cursor::new(vec![]);
        let s = read_string(&mut cursor, 0).unwrap();
        assert_eq!(s, "");
    }

    #[test]
    fn test_read_binary() {
        let data = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let mut cursor = Cursor::new(data.clone());
        let result = read_binary(&mut cursor, 4).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_read_element() {
        // Construct: EBML header ID (0x1A45DFA3) + size=5 (0x85)
        let data = vec![0x1A, 0x45, 0xDF, 0xA3, 0x85];
        let mut cursor = Cursor::new(data);
        let elem = read_element(&mut cursor).unwrap();
        assert_eq!(elem.id, 0x1A45DFA3);
        assert_eq!(elem.size, 5);
        assert_eq!(elem.position, 0);
        assert_eq!(elem.header_size, 5);
        assert_eq!(elem.data_offset(), 5);
        assert_eq!(elem.end_offset(), Some(10));
    }

    #[test]
    fn test_skip_element() {
        let data = vec![0x00; 100];
        let mut cursor = Cursor::new(data);
        skip_element(&mut cursor, 50).unwrap();
        let pos = cursor.stream_position().unwrap();
        assert_eq!(pos, 50);
    }

    #[test]
    fn test_vint_width() {
        assert_eq!(vint_width(0x80).unwrap(), 1);
        assert_eq!(vint_width(0xFE).unwrap(), 1);
        assert_eq!(vint_width(0x40).unwrap(), 2);
        assert_eq!(vint_width(0x7F).unwrap(), 2);
        assert_eq!(vint_width(0x20).unwrap(), 3);
        assert_eq!(vint_width(0x10).unwrap(), 4);
        assert!(vint_width(0x00).is_err());
        assert!(vint_width(0x08).is_err());
    }
}
