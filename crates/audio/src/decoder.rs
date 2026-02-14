//! Audio file decoding via Symphonia.
//!
//! Wraps Symphonia to provide a simple interface for decoding audio files
//! into f32 interleaved sample buffers. Supports AAC, MP3, FLAC, WAV, and Opus.

use std::fs::File;
use std::path::Path;

use ms_common::{AudioCodec, AudioStreamInfo, TimeCode};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::{FormatOptions, SeekMode, SeekTo};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::core::units::Time;
use tracing::{debug, warn};

use crate::error::AudioError;

/// Decoded audio samples in f32 interleaved format.
#[derive(Clone, Debug)]
pub struct DecodedAudio {
    /// Interleaved f32 samples.
    pub data: Vec<f32>,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u16,
    /// Presentation timestamp of the first sample.
    pub pts: TimeCode,
}

/// Audio file decoder backed by Symphonia.
///
/// Opens an audio file and decodes it packet-by-packet into f32 interleaved
/// sample buffers. Supports seeking and full-file decoding.
pub struct AudioDecoder {
    reader: Box<dyn symphonia::core::formats::FormatReader>,
    decoder: Box<dyn symphonia::core::codecs::Decoder>,
    track_id: u32,
    stream_info: AudioStreamInfo,
}

impl AudioDecoder {
    /// Open an audio file for decoding.
    ///
    /// Probes the file to determine format and codec, then initializes
    /// the appropriate Symphonia decoder.
    pub fn open(path: &Path) -> Result<Self, AudioError> {
        let file = File::open(path).map_err(|e| AudioError::FileOpen(format!("{path:?}: {e}")))?;
        let mss = MediaSourceStream::new(Box::new(file), Default::default());

        // Build a hint from the file extension
        let mut hint = Hint::new();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            hint.with_extension(ext);
        }

        let format_opts = FormatOptions {
            enable_gapless: true,
            ..Default::default()
        };
        let metadata_opts = MetadataOptions::default();

        let probed = symphonia::default::get_probe()
            .format(&hint, mss, &format_opts, &metadata_opts)
            .map_err(|e| AudioError::UnsupportedFormat(format!("{e}")))?;

        let reader = probed.format;

        // Find the first audio track
        let track = reader
            .tracks()
            .iter()
            .find(|t| {
                t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL
                    && t.codec_params.channels.is_some()
            })
            .ok_or(AudioError::NoAudioTrack)?;

        let track_id = track.id;
        let codec_params = track.codec_params.clone();

        let sample_rate = codec_params
            .sample_rate
            .ok_or_else(|| AudioError::Decode("No sample rate in codec params".to_string()))?;

        let channels = codec_params
            .channels
            .map(|c| c.count() as u16)
            .ok_or_else(|| AudioError::Decode("No channel info in codec params".to_string()))?;

        let duration = if let Some(n_frames) = codec_params.n_frames {
            TimeCode::from_secs(n_frames as f64 / sample_rate as f64)
        } else if let Some(_tb) = codec_params.time_base {
            // Time base available but no frame count; cannot compute duration
            TimeCode::from_secs(0.0)
        } else {
            TimeCode::from_secs(0.0)
        };

        let codec = symphonia_codec_to_audio_codec(&codec_params.codec);

        let stream_info = AudioStreamInfo {
            codec,
            sample_rate,
            channels,
            duration,
            bitrate: 0,
        };

        let decoder_opts = DecoderOptions::default();
        let decoder = symphonia::default::get_codecs()
            .make(&codec_params, &decoder_opts)
            .map_err(|e| AudioError::UnsupportedFormat(format!("Codec init failed: {e}")))?;

        debug!(
            codec = ?stream_info.codec,
            sample_rate = sample_rate,
            channels = channels,
            "Opened audio file"
        );

        Ok(Self {
            reader,
            decoder,
            track_id,
            stream_info,
        })
    }

    /// Get information about the audio stream.
    pub fn stream_info(&self) -> &AudioStreamInfo {
        &self.stream_info
    }

    /// Seek to a specific time in the audio file.
    pub fn seek(&mut self, time: TimeCode) -> Result<(), AudioError> {
        let seek_to = SeekTo::Time {
            time: Time::from(time.as_secs()),
            track_id: Some(self.track_id),
        };

        self.reader
            .seek(SeekMode::Coarse, seek_to)
            .map_err(|e| AudioError::Seek {
                time: time.as_secs(),
                reason: format!("{e}"),
            })?;

        // Reset the decoder state after seeking
        self.decoder.reset();

        debug!(time_secs = time.as_secs(), "Seeked audio decoder");
        Ok(())
    }

    /// Decode the next packet of audio, returning interleaved f32 samples.
    ///
    /// Returns `Ok(None)` when the end of the stream is reached.
    pub fn decode_next(&mut self) -> Result<Option<DecodedAudio>, AudioError> {
        loop {
            let packet = match self.reader.next_packet() {
                Ok(p) => p,
                Err(symphonia::core::errors::Error::IoError(ref e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    return Ok(None);
                }
                Err(e) => {
                    return Err(AudioError::Decode(format!("{e}")));
                }
            };

            // Skip packets that don't belong to our track
            if packet.track_id() != self.track_id {
                continue;
            }

            let pts_ts = packet.ts();

            let decoded = match self.decoder.decode(&packet) {
                Ok(d) => d,
                Err(symphonia::core::errors::Error::DecodeError(msg)) => {
                    warn!(error = %msg, "Skipping corrupted audio packet");
                    continue;
                }
                Err(e) => {
                    return Err(AudioError::Decode(format!("{e}")));
                }
            };

            let spec = *decoded.spec();
            let num_frames = decoded.frames();

            if num_frames == 0 {
                continue;
            }

            let num_channels = spec.channels.count();
            let mut sample_buf = SampleBuffer::<f32>::new(num_frames as u64, spec);
            sample_buf.copy_interleaved_ref(decoded);

            let samples = sample_buf.samples().to_vec();

            // Compute PTS from timestamp
            let pts = if let Some(tb) = self
                .reader
                .tracks()
                .iter()
                .find(|t| t.id == self.track_id)
                .and_then(|t| t.codec_params.time_base)
            {
                let time = tb.calc_time(pts_ts);
                TimeCode::from_secs(time.seconds as f64 + time.frac)
            } else {
                TimeCode::from_secs(pts_ts as f64 / self.stream_info.sample_rate as f64)
            };

            return Ok(Some(DecodedAudio {
                data: samples,
                sample_rate: spec.rate,
                channels: num_channels as u16,
                pts,
            }));
        }
    }

    /// Decode the entire audio file into a single interleaved f32 sample buffer.
    ///
    /// This is useful for waveform generation or short audio files.
    /// For long files, prefer streaming via [`decode_next`](Self::decode_next).
    pub fn decode_all(&mut self) -> Result<Vec<f32>, AudioError> {
        let mut all_samples = Vec::new();

        while let Some(decoded) = self.decode_next()? {
            all_samples.extend_from_slice(&decoded.data);
        }

        debug!(
            total_samples = all_samples.len(),
            "Decoded entire audio file"
        );

        Ok(all_samples)
    }
}

/// Map a Symphonia codec type to our `AudioCodec` enum.
fn symphonia_codec_to_audio_codec(codec: &symphonia::core::codecs::CodecType) -> AudioCodec {
    use symphonia::core::codecs;

    if *codec == codecs::CODEC_TYPE_AAC {
        AudioCodec::Aac
    } else if *codec == codecs::CODEC_TYPE_MP3 {
        AudioCodec::Mp3
    } else if *codec == codecs::CODEC_TYPE_FLAC {
        AudioCodec::Flac
    } else if *codec == codecs::CODEC_TYPE_VORBIS {
        AudioCodec::Vorbis
    } else if *codec == codecs::CODEC_TYPE_OPUS {
        AudioCodec::Opus
    } else if *codec == codecs::CODEC_TYPE_PCM_F32LE
        || *codec == codecs::CODEC_TYPE_PCM_S16LE
        || *codec == codecs::CODEC_TYPE_PCM_S24LE
        || *codec == codecs::CODEC_TYPE_PCM_S32LE
    {
        AudioCodec::Wav
    } else {
        AudioCodec::Aac // fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_mapping() {
        use symphonia::core::codecs;
        assert_eq!(
            symphonia_codec_to_audio_codec(&codecs::CODEC_TYPE_AAC),
            AudioCodec::Aac
        );
        assert_eq!(
            symphonia_codec_to_audio_codec(&codecs::CODEC_TYPE_MP3),
            AudioCodec::Mp3
        );
        assert_eq!(
            symphonia_codec_to_audio_codec(&codecs::CODEC_TYPE_FLAC),
            AudioCodec::Flac
        );
        assert_eq!(
            symphonia_codec_to_audio_codec(&codecs::CODEC_TYPE_VORBIS),
            AudioCodec::Vorbis
        );
    }

    #[test]
    fn decoded_audio_struct() {
        let audio = DecodedAudio {
            data: vec![0.0, 0.5, -0.5, 1.0],
            sample_rate: 44100,
            channels: 2,
            pts: TimeCode::from_secs(0.0),
        };
        assert_eq!(audio.data.len(), 4);
        assert_eq!(audio.sample_rate, 44100);
        assert_eq!(audio.channels, 2);
    }

    #[test]
    fn open_nonexistent_file() {
        let result = AudioDecoder::open(Path::new("/nonexistent/file.mp3"));
        assert!(result.is_err());
        let err = result.err().expect("Expected error");
        match err {
            AudioError::FileOpen(msg) => assert!(msg.contains("nonexistent")),
            other => panic!("Expected FileOpen error, got: {other}"),
        }
    }
}
