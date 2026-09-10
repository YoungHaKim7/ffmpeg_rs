//! Codec parameters — port of `libavcodec/codec_par.h` (`AVCodecParameters`)
//! plus the small enums it references (`AVMediaType`, `AVCodecID`,
//! `AVFieldOrder` from `codec.h`/`avcodec.h`).
//!
//! `AVCodecParameters` is the *serializable* stream description exchanged
//! between libavformat and libavcodec — what a demuxer learns from a
//! container header and what a decoder/encoder is configured from
//! (`avcodec_parameters_to_context`). Field names kept verbatim for
//! grep-ability against C.

use crate::util::{
    color::{ChromaLocation, ColorPrimaries, ColorRange, ColorSpace, ColorTrc},
    pixfmt::PixelFormat,
    rational::Rational,
};

/// `enum AVMediaType`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MediaType {
    /// Usually treated like data if no specific media type.
    #[default]
    Unknown,
    Video,
    Audio,
    Data,
    Subtitle,
}

/// `enum AVCodecID` — supported subset.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CodecId {
    #[default]
    None,
    /// `AV_CODEC_ID_RAWVIDEO` — uncompressed frames.
    Rawvideo,
    /// `AV_CODEC_ID_WRAPPED_AVFRAME`.
    WrappedAvframe,
    /// The PCM family (`AV_CODEC_ID_PCM_*`) as far as WAV needs it.
    /// alaw/mulaw are recognized by the demuxer but their companding
    /// decoders are not ported (Error::Unsupported at open).
    PcmU8,
    PcmS16le,
    PcmS16be,
    PcmS24le,
    PcmS24be,
    PcmS32le,
    PcmS32be,
    PcmF32le,
    PcmF32be,
    PcmF64le,
    PcmF64be,
    PcmAlaw,
    PcmMulaw,
}

impl CodecId {
    /// `avcodec_get_name` subset.
    pub const fn name(self) -> &'static str {
        match self {
            CodecId::None => "none",
            CodecId::Rawvideo => "rawvideo",
            CodecId::WrappedAvframe => "wrapped_avframe",
            CodecId::PcmU8 => "pcm_u8",
            CodecId::PcmS16le => "pcm_s16le",
            CodecId::PcmS16be => "pcm_s16be",
            CodecId::PcmS24le => "pcm_s24le",
            CodecId::PcmS24be => "pcm_s24be",
            CodecId::PcmS32le => "pcm_s32le",
            CodecId::PcmS32be => "pcm_s32be",
            CodecId::PcmF32le => "pcm_f32le",
            CodecId::PcmF32be => "pcm_f32be",
            CodecId::PcmF64le => "pcm_f64le",
            CodecId::PcmF64be => "pcm_f64be",
            CodecId::PcmAlaw => "pcm_alaw",
            CodecId::PcmMulaw => "pcm_mulaw",
        }
    }
}

/// `enum AVFieldOrder`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FieldOrder {
    #[default]
    Unknown,
    Progressive,
    /// Top field coded first, top displayed first.
    Tt,
    /// Bottom field coded and displayed first.
    Bb,
    /// Bottom coded first, top displayed first.
    Tb,
    /// Top coded first, bottom displayed first.
    Bt,
}

/// `AVCodecParameters` — video subset (`codec_par.h:49`).
#[derive(Debug, Clone)]
pub struct CodecParameters {
    pub codec_type: MediaType,
    pub codec_id: CodecId,
    /// Pixel format (`codecpar->format`, an `AVPixelFormat` for video).
    pub format: PixelFormat,
    pub width: u32,
    pub height: u32,
    /// `AVRational 0/1` when unspecified.
    pub sample_aspect_ratio: Rational,
    /// `AVRational 0/1` when unspecified (fields != frames).
    pub framerate: Rational,
    pub field_order: FieldOrder,
    pub color_range: ColorRange,
    pub color_primaries: ColorPrimaries,
    pub color_trc: ColorTrc,
    pub color_space: ColorSpace,
    pub chroma_location: ChromaLocation,
    /// Bits per second, 0 when unknown.
    pub bit_rate: i64,
    // ---- audio (codecpar's audio fields; `format` stays the video union
    // member — C stores the AVSampleFormat in the same int, the port
    // keeps a separate field) ----
    /// `sample_rate`.
    pub sample_rate: i32,
    /// `ch_layout`.
    pub ch_layout: crate::util::channel_layout::ChannelLayout,
    /// `format` for audio — the `AVSampleFormat`.
    pub sample_fmt: crate::util::samplefmt::SampleFormat,
    /// `block_align` — bytes per sample frame (channels · bps).
    pub block_align: i32,
    /// `frame_size` — samples per packet (PCM: 1 conceptually; C uses it
    /// for the demuxer's per-packet sample count bookkeeping).
    pub frame_size: i32,
}

impl Default for CodecParameters {
    fn default() -> Self {
        CodecParameters {
            codec_type: MediaType::Unknown,
            codec_id: CodecId::None,
            format: PixelFormat::Gray8,
            width: 0,
            height: 0,
            sample_aspect_ratio: Rational::UNKNOWN,
            framerate: Rational::UNKNOWN,
            field_order: FieldOrder::Unknown,
            color_range: ColorRange::Unspecified,
            color_primaries: ColorPrimaries::Unspecified,
            color_trc: ColorTrc::Unspecified,
            color_space: ColorSpace::Unspecified,
            chroma_location: ChromaLocation::Unspecified,
            bit_rate: 0,
            sample_rate: 0,
            ch_layout: crate::util::channel_layout::ChannelLayout::default(),
            sample_fmt: crate::util::samplefmt::SampleFormat::S16,
            block_align: 0,
            frame_size: 0,
        }
    }
}
