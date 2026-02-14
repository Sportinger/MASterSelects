//! Audio error types (thiserror-based).

use thiserror::Error;

/// Audio subsystem error type.
#[derive(Error, Debug)]
pub enum AudioError {
    /// Failed to open or read an audio file.
    #[error("Failed to open audio file: {0}")]
    FileOpen(String),

    /// The audio format/codec is not supported.
    #[error("Unsupported audio format: {0}")]
    UnsupportedFormat(String),

    /// Decoding error from Symphonia.
    #[error("Decode error: {0}")]
    Decode(String),

    /// Seeking to a specific time failed.
    #[error("Seek error at {time}s: {reason}")]
    Seek { time: f64, reason: String },

    /// No audio track found in the container.
    #[error("No audio track found in file")]
    NoAudioTrack,

    /// Audio output device error.
    #[error("Audio output error: {0}")]
    Output(String),

    /// Audio output stream build error.
    #[error("Failed to build audio stream: {0}")]
    StreamBuild(String),

    /// Audio output stream play error.
    #[error("Failed to play audio stream: {0}")]
    StreamPlay(String),

    /// Ring buffer is full, cannot write more samples.
    #[error("Audio ring buffer full")]
    BufferFull,

    /// Ring buffer is empty, underrun occurred.
    #[error("Audio buffer underrun")]
    BufferUnderrun,

    /// Resampler configuration error.
    #[error("Resampler error: {0}")]
    Resampler(String),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display() {
        let err = AudioError::FileOpen("test.mp3".to_string());
        assert_eq!(err.to_string(), "Failed to open audio file: test.mp3");
    }

    #[test]
    fn error_seek_display() {
        let err = AudioError::Seek {
            time: 5.0,
            reason: "out of range".to_string(),
        };
        assert_eq!(err.to_string(), "Seek error at 5s: out of range");
    }

    #[test]
    fn error_io_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let audio_err: AudioError = io_err.into();
        assert!(matches!(audio_err, AudioError::Io(_)));
    }
}
