//! Low-level MP4 atom/box writing primitives.
//!
//! MP4 files are structured as nested boxes (atoms). Each box has:
//! - 4-byte big-endian size (includes header)
//! - 4-byte ASCII type (e.g. "ftyp", "moov", "mdat")
//!
//! "Full boxes" additionally have:
//! - 1-byte version
//! - 3-byte flags

use byteorder::{BigEndian, WriteBytesExt};
use std::io::{Seek, SeekFrom, Write};

use crate::error::MuxResult;

/// Write a standard box header: 4-byte size + 4-byte type.
///
/// `size` is the total box size including the 8-byte header.
/// If `size` is 0, it means the box extends to end of file.
/// If `size` is 1, an extended 64-bit size follows (not handled here).
pub fn write_box_header<W: Write>(writer: &mut W, box_type: &[u8; 4], size: u32) -> MuxResult<()> {
    writer.write_u32::<BigEndian>(size)?;
    writer.write_all(box_type)?;
    Ok(())
}

/// Write a "full box" header: 4-byte size + 4-byte type + 1-byte version + 3-byte flags.
///
/// Total header is 12 bytes.
pub fn write_full_box_header<W: Write>(
    writer: &mut W,
    box_type: &[u8; 4],
    size: u32,
    version: u8,
    flags: u32,
) -> MuxResult<()> {
    writer.write_u32::<BigEndian>(size)?;
    writer.write_all(box_type)?;
    // version (1 byte) + flags (3 bytes) = 4 bytes total
    let version_flags = ((version as u32) << 24) | (flags & 0x00FF_FFFF);
    writer.write_u32::<BigEndian>(version_flags)?;
    Ok(())
}

/// Write a box size placeholder (4 bytes of zeros) and return the stream position
/// where the size should be patched later.
///
/// Usage pattern:
/// ```ignore
/// let pos = box_size_placeholder(&mut writer)?;
/// writer.write_all(b"moov")?;
/// // ... write box content ...
/// fill_box_size(&mut writer, pos)?;
/// ```
pub fn box_size_placeholder<W: Write + Seek>(writer: &mut W) -> MuxResult<u64> {
    let pos = writer.stream_position()?;
    writer.write_u32::<BigEndian>(0)?; // placeholder
    Ok(pos)
}

/// Patch the box size at the given position with the actual size
/// (from `pos` to current position).
pub fn fill_box_size<W: Write + Seek>(writer: &mut W, size_pos: u64) -> MuxResult<()> {
    let current = writer.stream_position()?;
    let size = current - size_pos;

    // Standard box size is u32; if > u32::MAX we would need extended size (co64).
    // For now, assert it fits in u32.
    if size > u32::MAX as u64 {
        return Err(crate::error::MuxError::BufferFull(format!(
            "Box size {} exceeds 32-bit limit",
            size
        )));
    }

    writer.seek(SeekFrom::Start(size_pos))?;
    writer.write_u32::<BigEndian>(size as u32)?;
    writer.seek(SeekFrom::Start(current))?;
    Ok(())
}

/// Write a 64-bit box header for large boxes (size == 1 signals extended size).
pub fn write_large_box_header<W: Write>(
    writer: &mut W,
    box_type: &[u8; 4],
    large_size: u64,
) -> MuxResult<()> {
    writer.write_u32::<BigEndian>(1)?; // size=1 means "look at largesize"
    writer.write_all(box_type)?;
    writer.write_u64::<BigEndian>(large_size)?; // 8-byte extended size
    Ok(())
}

/// Write a placeholder for a 64-bit box header and return the position
/// where the extended size field starts.
pub fn large_box_size_placeholder<W: Write + Seek>(
    writer: &mut W,
    box_type: &[u8; 4],
) -> MuxResult<u64> {
    writer.write_u32::<BigEndian>(1)?; // size=1 signals extended size
    writer.write_all(box_type)?;
    let size_pos = writer.stream_position()?;
    writer.write_u64::<BigEndian>(0)?; // placeholder for extended size
    Ok(size_pos)
}

/// Fill in a 64-bit extended size. `size_pos` points to the 8-byte extended size field
/// (after the 8-byte standard header: 4-byte size=1 + 4-byte type).
pub fn fill_large_box_size<W: Write + Seek>(
    writer: &mut W,
    size_pos: u64,
) -> MuxResult<()> {
    let current = writer.stream_position()?;
    // The total box size includes the 8-byte standard header before size_pos
    let total_size = current - (size_pos - 8);
    writer.seek(SeekFrom::Start(size_pos))?;
    writer.write_u64::<BigEndian>(total_size)?;
    writer.seek(SeekFrom::Start(current))?;
    Ok(())
}

/// Convert a time in seconds to a timescale-based integer.
pub fn seconds_to_timescale(seconds: f64, timescale: u32) -> u64 {
    (seconds * timescale as f64).round() as u64
}

/// Convert a timescale-based integer back to seconds.
pub fn timescale_to_seconds(ticks: u64, timescale: u32) -> f64 {
    ticks as f64 / timescale as f64
}

/// Standard video timescale (90kHz, same as MPEG-TS).
pub const VIDEO_TIMESCALE: u32 = 90_000;

/// Movie-level timescale (1000 = millisecond precision).
pub const MOVIE_TIMESCALE: u32 = 1000;

/// Write a fixed-point 16.16 number.
pub fn write_fixed_point_16_16<W: Write>(writer: &mut W, value: f64) -> MuxResult<()> {
    let fixed = (value * 65536.0).round() as i32;
    writer.write_i32::<BigEndian>(fixed)?;
    Ok(())
}

/// Write a fixed-point 8.8 number.
pub fn write_fixed_point_8_8<W: Write>(writer: &mut W, value: f64) -> MuxResult<()> {
    let fixed = (value * 256.0).round() as i16;
    writer.write_i16::<BigEndian>(fixed)?;
    Ok(())
}

/// Write zero padding bytes.
pub fn write_zeros<W: Write>(writer: &mut W, count: usize) -> MuxResult<()> {
    let zeros = vec![0u8; count];
    writer.write_all(&zeros)?;
    Ok(())
}

/// Write a null-terminated string, padded to a specific length.
pub fn write_fixed_string<W: Write>(writer: &mut W, s: &str, len: usize) -> MuxResult<()> {
    let bytes = s.as_bytes();
    let to_write = bytes.len().min(len);
    writer.write_all(&bytes[..to_write])?;
    // Pad with zeros
    for _ in to_write..len {
        writer.write_u8(0)?;
    }
    Ok(())
}

/// ISO 639-2/T language code packed into 3x5 bits.
/// Default is "und" (undetermined).
pub fn encode_language(lang: &str) -> u16 {
    let bytes = lang.as_bytes();
    if bytes.len() < 3 {
        // "und" = undetermined
        return encode_language("und");
    }
    let a = (bytes[0] - 0x60) as u16;
    let b = (bytes[1] - 0x60) as u16;
    let c = (bytes[2] - 0x60) as u16;
    (a << 10) | (b << 5) | c
}

/// Convert seconds since 1904-01-01 (MP4 epoch) from a Unix timestamp.
/// MP4 uses seconds since 1904-01-01 00:00:00 UTC.
/// Unix epoch is 1970-01-01 00:00:00 UTC.
/// Difference: 66 years worth of seconds (including leap years).
pub const MP4_EPOCH_OFFSET: u64 = 2_082_844_800;

/// Get current time as MP4 creation time (seconds since 1904).
/// Returns a reasonable default for now.
pub fn mp4_creation_time() -> u64 {
    // Use a fixed value for reproducibility in tests;
    // in production this would use system time.
    // 2024-01-01 00:00:00 UTC as MP4 time
    MP4_EPOCH_OFFSET + 1_704_067_200
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_write_box_header() {
        let mut buf = Vec::new();
        write_box_header(&mut buf, b"ftyp", 20).unwrap();
        assert_eq!(buf.len(), 8);
        // Size: 20 = 0x00000014
        assert_eq!(&buf[0..4], &[0x00, 0x00, 0x00, 0x14]);
        // Type: "ftyp"
        assert_eq!(&buf[4..8], b"ftyp");
    }

    #[test]
    fn test_write_box_header_zero_size() {
        let mut buf = Vec::new();
        write_box_header(&mut buf, b"mdat", 0).unwrap();
        assert_eq!(&buf[0..4], &[0x00, 0x00, 0x00, 0x00]);
        assert_eq!(&buf[4..8], b"mdat");
    }

    #[test]
    fn test_write_full_box_header() {
        let mut buf = Vec::new();
        write_full_box_header(&mut buf, b"mvhd", 120, 1, 0).unwrap();
        assert_eq!(buf.len(), 12);
        // Size: 120
        assert_eq!(&buf[0..4], &[0x00, 0x00, 0x00, 120]);
        // Type: "mvhd"
        assert_eq!(&buf[4..8], b"mvhd");
        // Version 1, flags 0 → 0x01000000
        assert_eq!(&buf[8..12], &[0x01, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_full_box_header_with_flags() {
        let mut buf = Vec::new();
        write_full_box_header(&mut buf, b"tkhd", 100, 0, 0x000003).unwrap();
        // Version 0, flags 3 → 0x00000003
        assert_eq!(&buf[8..12], &[0x00, 0x00, 0x00, 0x03]);
    }

    #[test]
    fn test_box_size_placeholder_and_fill() {
        let mut cursor = Cursor::new(Vec::new());
        let pos = box_size_placeholder(&mut cursor).unwrap();
        assert_eq!(pos, 0);

        cursor.write_all(b"moov").unwrap();
        // Write some content (20 bytes)
        cursor.write_all(&[0xAA; 20]).unwrap();

        fill_box_size(&mut cursor, pos).unwrap();

        let buf = cursor.into_inner();
        // Total size = 4 (size) + 4 (type) + 20 (content) = 28 bytes
        // But we only wrote 4 (placeholder) + 4 (type) + 20 (content) = 28
        assert_eq!(buf.len(), 28);
        // Size field should be 28
        assert_eq!(&buf[0..4], &[0x00, 0x00, 0x00, 28]);
    }

    #[test]
    fn test_seconds_to_timescale() {
        assert_eq!(seconds_to_timescale(1.0, 90_000), 90_000);
        assert_eq!(seconds_to_timescale(0.5, 90_000), 45_000);
        assert_eq!(seconds_to_timescale(2.0, 44_100), 88_200);
        assert_eq!(seconds_to_timescale(0.0, 90_000), 0);
    }

    #[test]
    fn test_timescale_to_seconds() {
        let secs = timescale_to_seconds(90_000, 90_000);
        assert!((secs - 1.0).abs() < 1e-9);

        let secs = timescale_to_seconds(45_000, 90_000);
        assert!((secs - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_timescale_roundtrip() {
        let original = 7.53921;
        let ticks = seconds_to_timescale(original, VIDEO_TIMESCALE);
        let recovered = timescale_to_seconds(ticks, VIDEO_TIMESCALE);
        assert!((original - recovered).abs() < 0.001);
    }

    #[test]
    fn test_write_fixed_point_16_16() {
        let mut buf = Vec::new();
        write_fixed_point_16_16(&mut buf, 1.0).unwrap();
        // 1.0 * 65536 = 0x00010000
        assert_eq!(&buf, &[0x00, 0x01, 0x00, 0x00]);
    }

    #[test]
    fn test_write_fixed_point_8_8() {
        let mut buf = Vec::new();
        write_fixed_point_8_8(&mut buf, 1.0).unwrap();
        // 1.0 * 256 = 0x0100
        assert_eq!(&buf, &[0x01, 0x00]);
    }

    #[test]
    fn test_encode_language_und() {
        let code = encode_language("und");
        // u=0x15, n=0x0E, d=0x04
        // (0x15 << 10) | (0x0E << 5) | 0x04 = 0x55C4
        assert_eq!(code, 0x55C4);
    }

    #[test]
    fn test_encode_language_eng() {
        let code = encode_language("eng");
        // e=5, n=14, g=7
        // (5 << 10) | (14 << 5) | 7 = 5120 + 448 + 7 = 5575
        assert_eq!(code, 5575);
    }

    #[test]
    fn test_write_zeros() {
        let mut buf = Vec::new();
        write_zeros(&mut buf, 8).unwrap();
        assert_eq!(buf, vec![0u8; 8]);
    }

    #[test]
    fn test_write_fixed_string() {
        let mut buf = Vec::new();
        write_fixed_string(&mut buf, "vid", 8).unwrap();
        assert_eq!(buf.len(), 8);
        assert_eq!(&buf[0..3], b"vid");
        assert_eq!(&buf[3..8], &[0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_write_large_box_header() {
        let mut buf = Vec::new();
        write_large_box_header(&mut buf, b"mdat", 0x1_0000_0000).unwrap();
        assert_eq!(buf.len(), 16);
        // size field = 1 (signals extended size)
        assert_eq!(&buf[0..4], &[0x00, 0x00, 0x00, 0x01]);
        assert_eq!(&buf[4..8], b"mdat");
        // extended size = 0x100000000
        assert_eq!(
            &buf[8..16],
            &[0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00]
        );
    }

    #[test]
    fn test_large_box_placeholder_and_fill() {
        let mut cursor = Cursor::new(Vec::new());
        let size_pos = large_box_size_placeholder(&mut cursor, b"mdat").unwrap();
        // Write 32 bytes of content
        cursor.write_all(&[0xBB; 32]).unwrap();
        fill_large_box_size(&mut cursor, size_pos).unwrap();

        let buf = cursor.into_inner();
        // Total = 4 (size=1) + 4 (type) + 8 (extended size) + 32 (data) = 48
        assert_eq!(buf.len(), 48);
        // Extended size field at offset 8
        let extended_size = u64::from_be_bytes(buf[8..16].try_into().unwrap());
        assert_eq!(extended_size, 48);
    }

    #[test]
    fn test_mp4_creation_time() {
        let t = mp4_creation_time();
        assert!(t > MP4_EPOCH_OFFSET);
    }
}
