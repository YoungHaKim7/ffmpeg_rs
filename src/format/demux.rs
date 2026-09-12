//! Demuxer registry — port of `libavformat/demux.h` (`FFInputFormat`) and
//! the static list in `demuxer_list.c`.
//!
//! C keeps a public metadata struct (`AVInputFormat`) plus a hidden vtable
//! (`FFInputFormat` with `read_probe`/`read_header`/`read_packet`). Rust
//! folds both into one [`InputFormat`] row: metadata plus two constructor
//! hooks, one plain and one fed demuxer options (the AVOptions of the C
//! priv_data class, typed here instead).

use crate::{
    codec::packet::Packet,
    util::{error::Result, pixfmt::PixelFormat, rational::Rational},
};

use super::{Stream, io::IoContext};

/// `AVPROBE_SCORE_MAX`.
pub const PROBE_SCORE_MAX: u32 = 100;

/// The demuxer list (a hand-maintanded `demuxer_list.c`).
pub static INPUT_FORMATS: &[InputFormat] = &[
    InputFormat {
        name: "yuv4mpegpipe",
        long_name: "YUV4MPEG pipe",
        extensions: &["y4m"],
        probe: Some(super::y4m::probe),
        make: |_| Box::new(super::y4m::Y4mDemuxer::new()),
    },
    InputFormat {
        name: "rawvideo",
        long_name: "raw video",
        extensions: &["yuv", "rgb"],
        // C marks rawvideo as not auto-detectable: every file would match.
        probe: None,
        make: |opts| Box::new(super::rawvideo::RawVideoDemuxer::new(&opts.raw_video)),
    },
    InputFormat {
        name: "wav",
        long_name: "WAV / WAVE (Waveform Audio)",
        extensions: &["wav"],
        probe: Some(super::wav::probe),
        make: |_| Box::new(super::wav::WavDemuxer::new()),
    },
    InputFormat {
        name: "nut",
        long_name: "NUT",
        extensions: &["nut"],
        probe: Some(super::nut::probe),
        make: |_| Box::new(super::nut::NutDemuxer::new()),
    },
];

/// Options only the rawvideo demuxer consumes (its AVOptions
/// `pixel_format`/`video_size`/`framerate`, rawvideodec.c:211-227, typed).
#[derive(Debug, Clone)]
pub struct RawVideoDemuxOptions {
    pub pixel_format: PixelFormat,
    /// Required — C's `av_image_check_size` rejects 0x0.
    pub video_size: Option<(u32, u32)>,
    pub framerate: Rational,
}

impl Default for RawVideoDemuxOptions {
    fn default() -> Self {
        // C defaults: pix_fmt YUV420P, framerate 25, size unset.
        RawVideoDemuxOptions {
            pixel_format: PixelFormat::Yuv420p,
            video_size: None,
            framerate: Rational::new(25, 1),
        }
    }
}

/// Demuxer-specific options bucket (the union of every demuxer's AVOptions;
/// Phase 1 has exactly one consumer).
#[derive(Debug, Clone, Default)]
pub struct DemuxOptions {
    pub raw_video: RawVideoDemuxOptions,
}

/// `FFInputFormat` — one row of the demuxer registry.
pub struct InputFormat {
    /// `p.name` — what `-f` expects.
    pub name: &'static str,
    pub long_name: &'static str,
    /// `p.extensions` split on ','.
    pub extensions: &'static [&'static str],
    /// `read_probe`; `None` = C's "must be forced with -f" (rawvideo).
    pub probe: Option<fn(&[u8]) -> u32>,
    /// `read_header` + `read_packet` owner.
    pub make: fn(&DemuxOptions) -> Box<dyn Demuxer>,
}

/// The demuxer vtable (`FFInputFormat` callbacks).
pub trait Demuxer {
    /// `read_header` — parse the stream header, return the (single, video)
    /// stream description.
    fn read_header(&mut self, io: &mut IoContext) -> Result<Stream>;

    /// `read_packet` — next packet or `Err(Error::Eof)`.
    fn read_packet(&mut self, io: &mut IoContext) -> Result<Packet>;
}

/// `av_find_input_format` — registry lookup by `-f` name.
pub fn find_input_format(name: &str) -> Option<&'static InputFormat> {
    INPUT_FORMATS.iter().find(|f| f.name == name)
}

/// Format detection by file extension (`av_match_ext` path in
/// `avformat_open_input`).
pub fn find_input_format_by_extension(url: &str) -> Option<&'static InputFormat> {
    let ext = url.rsplit('.').next()?;
    INPUT_FORMATS.iter().find(|f| f.extensions.contains(&ext))
}
