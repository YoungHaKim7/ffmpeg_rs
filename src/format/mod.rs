//! `libavformat` — container (de)muxing.
//!
//! Port map:
//!
//! | FFmpeg | here | status |
//! |---|---|---|
//! | `avio.h` + `aviobuf.c` | [`io`] | file protocol + buffered core |
//! | `demux.h` + `demuxer_list.c` | [`demux`] | 2 demuxers |
//! | `mux.h` + `muxer_list.c` | [`mux`] | 2 muxers |
//! | `avformat.h` (contexts) | this module | input/output contexts, `Stream` |
//! | `yuv4mpeg{dec,enc}.c` | [`y4m`] | full subset |
//! | `rawvideodec.c` + `rawenc.c` | [`rawvideo`] | compact-stride subset |
//!
//! Lifecycle mirrors C exactly:
//!
//! ```text
//! open_input (probe → pick format → read_header)
//!   → find_stream_info (self-describing formats: validation only)
//!   → loop read_frame  … Err(Eof)
//! out: create → write_header → write_frame* → write_trailer
//! ```
//!
//! Not ported in Phase 1: probing score ladders beyond magic checks,
//! interleaving (single stream), seeking, metadata, chapters, multiple
//! streams, `pipe:` URLs.

pub mod demux;
pub mod io;
pub mod mux;
pub mod rawvideo;
#[cfg(test)]
pub mod testutil;
pub mod wav;
pub mod y4m;

use crate::{
    NOPTS,
    codec::{
        packet::Packet,
        params::{CodecParameters, MediaType},
    },
    util::{
        error::{Error, Result},
        rational::Rational,
    },
};

pub use demux::{DemuxOptions, Demuxer, InputFormat, PROBE_SCORE_MAX};
pub use mux::{Muxer, OutputFormat};

/// `AVStream` subset (`avformat.h:768`) — one (video) stream of a file.
#[derive(Debug, Clone)]
pub struct Stream {
    pub index: u32,
    /// `codecpar` — the description exchanged with libavcodec.
    pub codecpar: CodecParameters,
    /// `avpriv_set_pts_info` result: packet timestamps are in these units.
    pub time_base: Rational,
    pub avg_frame_rate: Rational,
    pub r_frame_rate: Rational,
    pub sample_aspect_ratio: Rational,
    pub start_time: i64,
    /// In `time_base` units (`NOPTS` when the container can't know).
    pub duration: i64,
    pub nb_frames: i64,
}

impl Stream {
    /// A fresh video stream (`avformat_new_stream` + type).
    pub fn new_video(index: u32) -> Stream {
        Stream {
            index,
            codecpar: CodecParameters {
                codec_type: MediaType::Video,
                ..CodecParameters::default()
            },
            time_base: Rational::UNKNOWN,
            avg_frame_rate: Rational::UNKNOWN,
            r_frame_rate: Rational::UNKNOWN,
            sample_aspect_ratio: Rational::UNKNOWN,
            start_time: NOPTS,
            duration: NOPTS,
            nb_frames: 0,
        }
    }

    /// `avpriv_set_pts_info(st, 64, m, n)` — time_base `m/n` reduced, and
    /// `avg_frame_rate` its inverse.
    pub fn set_pts_info(&mut self, num: i64, den: i64) {
        let (reduced, _) = Rational::reduce(num, den, (1u64 << 31) as i64 - 1);
        self.time_base = reduced;
        self.avg_frame_rate = reduced.inv();
        self.r_frame_rate = self.avg_frame_rate;
    }
}

/// `AVFormatContext` — input side.
pub struct InputFormatContext {
    pub url: String,
    pub iformat: &'static InputFormat,
    pub streams: Vec<Stream>,
    io: io::IoContext,
    demuxer: Box<dyn Demuxer>,
}

impl InputFormatContext {
    /// `avformat_open_input(url, fmt, opts)` — open I/O, pick the format
    /// (explicit `-f` > extension > probe score), run `read_header`.
    pub fn open(url: &str, format_name: Option<&str>, opts: &DemuxOptions) -> Result<Self> {
        let mut io = io::IoContext::open_input(url)?;

        // Format selection (avformat_open_input → io_open + init_input).
        let iformat: &'static InputFormat = if let Some(name) = format_name {
            demux::find_input_format(name).ok_or_else(|| {
                Error::NotFound(format!(
                    "demuxer '{name}' (known: {})",
                    demux::INPUT_FORMATS
                        .iter()
                        .map(|f| f.name)
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            })?
        } else if let Some(f) = demux::find_input_format_by_extension(url) {
            f
        } else {
            // Probe: highest score above 50 wins (C's threshold).
            let head = io.peek(2048)?;
            let mut best: Option<(&'static InputFormat, u32)> = None;
            for f in demux::INPUT_FORMATS.iter() {
                if let Some(probe) = f.probe {
                    let score = probe(&head);
                    if score > 50 && best.map(|(_, s)| score > s).unwrap_or(true) {
                        best = Some((f, score));
                    }
                }
            }
            best.map(|(f, _)| f).ok_or_else(|| {
                Error::InvalidData(format!(
                    "Unable to find a suitable input format for '{url}'; \
                     force one with -f (known: {})",
                    demux::INPUT_FORMATS
                        .iter()
                        .map(|f| f.name)
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            })?
        };

        let mut demuxer = (iformat.make)(opts);
        let stream = demuxer.read_header(&mut io)?;

        Ok(InputFormatContext {
            url: url.to_string(),
            iformat,
            streams: vec![stream],
            io,
            demuxer,
        })
    }

    /// `avformat_find_stream_info` — the ported containers are
    /// self-describing, so this is the lifecycle hook plus per-media sanity
    /// checks (C would decode frames to fill in gaps; we have none).
    pub fn find_stream_info(&mut self) -> Result<()> {
        let st = &self.streams[0];
        match st.codecpar.codec_type {
            MediaType::Video => {
                if st.codecpar.width == 0 || st.codecpar.height == 0 {
                    return Err(Error::InvalidData("stream dimensions unknown".into()));
                }
            }
            MediaType::Audio => {
                // WAV fills sample_rate/ch_layout/sample_fmt at header
                // parse; a zero rate or zero channels is malformed.
                if st.codecpar.sample_rate <= 0 {
                    return Err(Error::InvalidData("stream sample rate unknown".into()));
                }
                if st.codecpar.ch_layout.nb_channels == 0 {
                    return Err(Error::InvalidData("stream channel count unknown".into()));
                }
            }
            other => {
                return Err(Error::Unsupported(format!("non-A/V streams: {other:?}")));
            }
        }
        Ok(())
    }

    /// `av_read_frame` — next packet of any stream (Phase 1: stream 0 only),
    /// `Err(Error::Eof)` at end.
    pub fn read_frame(&mut self) -> Result<Packet> {
        self.demuxer.read_packet(&mut self.io)
    }

    /// Underlying I/O (used by the dump code for size queries).
    pub fn io(&self) -> &io::IoContext {
        &self.io
    }
}

/// `AVFormatContext` — output side.
pub struct OutputFormatContext {
    pub url: String,
    pub oformat: &'static OutputFormat,
    pub streams: Vec<Stream>,
    io: io::IoContext,
    muxer: Box<dyn Muxer>,
    header_written: bool,
    trailer_written: bool,
}

impl OutputFormatContext {
    /// `avformat_alloc_output_context2` + `avio_open` + `avformat_init`
    /// (single video stream in Phase 1).
    pub fn create(url: &str, format_name: Option<&str>, stream: Stream) -> Result<Self> {
        let oformat: &'static OutputFormat = if let Some(name) = format_name {
            mux::find_output_format(name).ok_or_else(|| {
                Error::NotFound(format!(
                    "muxer '{name}' (known: {})",
                    mux::OUTPUT_FORMATS
                        .iter()
                        .map(|f| f.name)
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            })?
        } else if let Some(f) = mux::find_output_format_by_extension(url) {
            f
        } else {
            return Err(Error::InvalidArgument(format!(
                "Unable to infer output format from '{url}'; force one with -f \
                 (known: {})",
                mux::OUTPUT_FORMATS
                    .iter()
                    .map(|f| f.name)
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        };

        let io = io::IoContext::open_output(url)?;
        let mut muxer = (oformat.make)();
        let streams = vec![stream];
        muxer.init(&streams)?;

        Ok(OutputFormatContext {
            url: url.to_string(),
            oformat,
            streams,
            io,
            muxer,
            header_written: false,
            trailer_written: false,
        })
    }

    /// `avformat_write_header`.
    pub fn write_header(&mut self) -> Result<()> {
        if self.header_written {
            return Err(Error::InvalidArgument("header already written".into()));
        }
        self.muxer.write_header(&mut self.io, &self.streams)?;
        self.header_written = true;
        Ok(())
    }

    /// `av_write_frame`.
    pub fn write_frame(&mut self, pkt: &Packet) -> Result<()> {
        if !self.header_written || self.trailer_written {
            return Err(Error::InvalidArgument(
                "write_frame outside header/trailer".into(),
            ));
        }
        self.muxer.write_packet(&mut self.io, &self.streams, pkt)
    }

    /// `av_write_trailer` (+ implicit flush).
    pub fn write_trailer(&mut self) -> Result<()> {
        if self.trailer_written {
            return Err(Error::InvalidArgument("trailer already written".into()));
        }
        self.muxer.write_trailer(&mut self.io, &self.streams)?;
        self.io.flush()?;
        self.trailer_written = true;
        Ok(())
    }
}
