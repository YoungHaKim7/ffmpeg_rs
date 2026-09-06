//! Codec parameters — port of `libavcodec/codec_par.h` (`AVCodecParameters`)
//! plus the small enums it references (`AVMediaType`, `AVCodecID`,
//! `AVFieldOrder` from `codec.h`/`avcodec.h`).
//!
//! `AVCodecParameters` is the *serializable* stream description exchanged
//! between libavformat and libavcodec — what a demuxer learns from a
//! container header and what a decoder/encoder is configured from
//! (`avcodec_parameters_to_context`). Field names kept verbatim for
//! grep-ability against C.

use crate::{
    util::color::{ChromaLocation, ColorPrimaries, ColorRange, ColorSpace, ColorTrc},
    util::pixfmt::PixelFormat,
    util::rational::Rational,
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
}

impl CodecId {
    /// `avcodec_get_name` subset.
    pub const fn name(self) -> &'static str {
        match self {
            CodecId::None => "none",
            CodecId::Rawvideo => "rawvideo",
            CodecId::WrappedAvframe => "wrapped_avframe",
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_c_unspecified() {
        let p = CodecParameters::default();
        assert_eq!(p.codec_type, MediaType::Unknown);
        assert_eq!(p.codec_id, CodecId::None);
        assert_eq!(p.width, 0);
        assert_eq!(p.sample_aspect_ratio, Rational::UNKNOWN);
        assert_eq!(p.framerate, Rational::UNKNOWN);
    }

    #[test]
    fn codec_names() {
        assert_eq!(CodecId::Rawvideo.name(), "rawvideo");
        assert_eq!(CodecId::WrappedAvframe.name(), "wrapped_avframe");
    }
}
