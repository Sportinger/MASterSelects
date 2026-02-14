//! File probing — detect container format and extract stream info.

use ms_common::{ContainerFormat, DemuxError};
use std::io::Read;
use std::path::Path;

/// Magic bytes for Matroska/WebM files (EBML header element ID).
const EBML_MAGIC: [u8; 4] = [0x1A, 0x45, 0xDF, 0xA3];

/// Detect container format from file extension.
pub fn detect_format(path: &Path) -> Result<ContainerFormat, DemuxError> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .unwrap_or_default();

    match ext.as_str() {
        "mp4" | "m4v" | "mov" => Ok(ContainerFormat::Mp4),
        "mkv" => Ok(ContainerFormat::Mkv),
        "webm" => Ok(ContainerFormat::WebM),
        _ => Err(DemuxError::UnsupportedContainer),
    }
}

/// Detect container format from magic bytes (first 4+ bytes of the file).
///
/// This probes the file content rather than relying on the extension,
/// which is more reliable for files with incorrect or missing extensions.
pub fn detect_format_from_magic<R: Read>(reader: &mut R) -> Result<ContainerFormat, DemuxError> {
    let mut header = [0u8; 12];
    let bytes_read = reader.read(&mut header).map_err(DemuxError::Io)?;

    if bytes_read < 4 {
        return Err(DemuxError::UnsupportedContainer);
    }

    // Check for EBML magic (MKV/WebM)
    if header[..4] == EBML_MAGIC {
        // Both MKV and WebM start with the EBML header.
        // To distinguish, we'd need to parse the DocType, but for probing
        // purposes we return Mkv (the caller can refine after full parse).
        return Ok(ContainerFormat::Mkv);
    }

    // Check for MP4/MOV: ftyp box
    if bytes_read >= 8 {
        // An MP4 file typically starts with an ftyp box:
        // [4 bytes size][4 bytes "ftyp"]
        if &header[4..8] == b"ftyp" {
            return Ok(ContainerFormat::Mp4);
        }
        // Some MOV files start with other boxes (e.g., "moov", "mdat", "wide")
        if &header[4..8] == b"moov"
            || &header[4..8] == b"mdat"
            || &header[4..8] == b"wide"
            || &header[4..8] == b"free"
            || &header[4..8] == b"skip"
        {
            return Ok(ContainerFormat::Mp4);
        }
    }

    Err(DemuxError::UnsupportedContainer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::path::PathBuf;

    #[test]
    fn test_detect_format_mp4() {
        let path = PathBuf::from("video.mp4");
        assert_eq!(detect_format(&path).unwrap(), ContainerFormat::Mp4);
    }

    #[test]
    fn test_detect_format_mov() {
        let path = PathBuf::from("video.mov");
        assert_eq!(detect_format(&path).unwrap(), ContainerFormat::Mp4);
    }

    #[test]
    fn test_detect_format_mkv() {
        let path = PathBuf::from("video.mkv");
        assert_eq!(detect_format(&path).unwrap(), ContainerFormat::Mkv);
    }

    #[test]
    fn test_detect_format_webm() {
        let path = PathBuf::from("video.webm");
        assert_eq!(detect_format(&path).unwrap(), ContainerFormat::WebM);
    }

    #[test]
    fn test_detect_format_unsupported() {
        let path = PathBuf::from("video.avi");
        assert!(detect_format(&path).is_err());
    }

    #[test]
    fn test_detect_format_from_magic_ebml() {
        let data = vec![0x1A, 0x45, 0xDF, 0xA3, 0x00, 0x00, 0x00, 0x00];
        let mut cursor = Cursor::new(data);
        assert_eq!(
            detect_format_from_magic(&mut cursor).unwrap(),
            ContainerFormat::Mkv
        );
    }

    #[test]
    fn test_detect_format_from_magic_mp4_ftyp() {
        // [size=20][ftyp]
        let data = vec![
            0x00, 0x00, 0x00, 0x14, b'f', b't', b'y', b'p', b'i', b's', b'o', b'm',
        ];
        let mut cursor = Cursor::new(data);
        assert_eq!(
            detect_format_from_magic(&mut cursor).unwrap(),
            ContainerFormat::Mp4
        );
    }

    #[test]
    fn test_detect_format_from_magic_too_short() {
        let data = vec![0x00, 0x01];
        let mut cursor = Cursor::new(data);
        assert!(detect_format_from_magic(&mut cursor).is_err());
    }

    #[test]
    fn test_detect_format_from_magic_unknown() {
        let data = vec![0x52, 0x49, 0x46, 0x46, 0x00, 0x00, 0x00, 0x00]; // RIFF (AVI)
        let mut cursor = Cursor::new(data);
        assert!(detect_format_from_magic(&mut cursor).is_err());
    }
}

