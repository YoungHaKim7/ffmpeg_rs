//! Muxer registry — port of `libavformat/mux.h` (`FFOutputFormat`) and the
//! static list in `muxer_list.c`.

use crate::codec::packet::Packet;
use crate::codec::params::CodecId;
use crate::util::error::Result;

use super::io::IoContext;
use super::Stream;

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
];

/// `av_guess_format` by `-f` name.
pub fn find_output_format(name: &str) -> Option<&'static OutputFormat> {
    OUTPUT_FORMATS.iter().find(|f| f.name == name)
}

/// `av_guess_format(NULL, filename, NULL)` — extension match.
pub fn find_output_format_by_extension(url: &str) -> Option<&'static OutputFormat> {
    let ext = url.rsplit('.').next()?;
    OUTPUT_FORMATS.iter().find(|f| f.extensions.contains(&ext))
}
