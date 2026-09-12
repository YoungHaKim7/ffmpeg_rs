//! Muxer registry — port of `libavformat/mux.h` (`FFOutputFormat`) and the
//! static list in `muxer_list.c`.

use crate::{
    codec::{packet::Packet, params::CodecId},
    util::error::Result,
};

use super::{Stream, io::IoContext};

/// The muxer list.
pub static OUTPUT_FORMATS: &[OutputFormat] = &[
    OutputFormat {
        name: "yuv4mpegpipe",
        long_name: "YUV4MPEG pipe",
        extensions: &["y4m"],
        video_codec: CodecId::Rawvideo,
        make: || Box::new(super::y4m::Y4mMuxer::new()),
    },
    OutputFormat {
        name: "rawvideo",
        long_name: "raw video",
        extensions: &["yuv", "rgb"],
        video_codec: CodecId::Rawvideo,
        make: || Box::new(super::rawvideo::RawVideoMuxer),
    },
    OutputFormat {
        name: "wav",
        long_name: "WAV / WAVE (Waveform Audio)",
        extensions: &["wav"],
        // C's ff_wav_muxer declares video_codec NONE + audio_codec
        // PCM_S16LE (wavenc.c:533-534); the registry row has an audio
        // counterpart only as this None — the stream gate lives in
        // WavMuxer::init/write_header.
        video_codec: CodecId::None,
        make: || Box::new(super::wav::WavMuxer::new()),
    },
    OutputFormat {
        name: "nut",
        long_name: "NUT",
        extensions: &["nut"],
        // C's ff_nut_muxer declares video_codec MPEG4 + audio_codec
        // VORBIS/MP3/MP2 (nutenc.c:1248-1250) as stream-creation *defaults*
        // for avformat_alloc_output_context2, not a gate; the port's
        // create() takes an explicit Stream, so the row stays None and the
        // codec gate lives where C's does — write_streamheader's
        // "No codec tag defined" (nutenc.c:470-471).
        video_codec: CodecId::None,
        make: || Box::new(super::nut::NutMuxer::new()),
    },
];

/// `FFOutputFormat` — one row of the muxer registry.
pub struct OutputFormat {
    pub name: &'static str,
    pub long_name: &'static str,
    pub extensions: &'static [&'static str],
    /// `p.video_codec` — what the muxer expects to be fed.
    pub video_codec: CodecId,
    pub make: fn() -> Box<dyn Muxer>,
}

/// The muxer vtable (`FFOutputFormat` callbacks).
pub trait Muxer {
    /// `init` — validate streams before writing (y4m_init's codec/format
    /// gate).
    fn init(&mut self, streams: &[Stream]) -> Result<()>;

    /// `write_header`.
    fn write_header(&mut self, io: &mut IoContext, streams: &[Stream]) -> Result<()>;

    /// `write_packet`.
    fn write_packet(&mut self, io: &mut IoContext, streams: &[Stream], pkt: &Packet) -> Result<()>;

    /// `write_trailer` — default: nothing (raw formats have no trailer).
    fn write_trailer(&mut self, _io: &mut IoContext, _streams: &[Stream]) -> Result<()> {
        Ok(())
    }
}

/// `av_guess_format` by `-f` name.
pub fn find_output_format(name: &str) -> Option<&'static OutputFormat> {
    OUTPUT_FORMATS.iter().find(|f| f.name == name)
}

/// `av_guess_format(NULL, filename, NULL)` — extension match.
pub fn find_output_format_by_extension(url: &str) -> Option<&'static OutputFormat> {
    let ext = url.rsplit('.').next()?;
    OUTPUT_FORMATS.iter().find(|f| f.extensions.contains(&ext))
}
