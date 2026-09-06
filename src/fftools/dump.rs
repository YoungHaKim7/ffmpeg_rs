//! Stream dumping — port of `libavformat/dump.c`'s `av_dump_format` (video
//! parts) + the `avcodec_string` pieces that print codec/pixfmt lines.
//!
//! Output is matched against real ffmpeg 8.1.2 byte-shape where visible:
//!
//! ```text
//! Input #0, yuv4mpegpipe, from 'in.y4m':
//!   Duration: 00:00:01.00, start: 0.000000, bitrate: 1475 kb/s
//!   Stream #0:0: Video: rawvideo (I420 / 0x30323449), yuv420p(progressive), 128x96, SAR 1:1 DAR 4:3, 10 fps, 10 tbr, 10 tbn
//! Stream mapping:
//!   Stream #0:0 -> #0:0 (rawvideo (native) -> rawvideo (native))
//! Output #0, rawvideo, to 'out.rgb':
//!   Stream #0:0: Video: rawvideo (RGB[24] / 0x18424752), rgb24(pc, progressive), 128x96 [SAR 1:1 DAR 4:3], q=2-31, 2949 kb/s, 10 fps, 10 tbn
//! ```
//!
//! Documented divergences (8.1.2 prints them, we don't): the `Metadata:`
//! blocks, and the `gbr/unknown/unknown` colorspace triplet — our port only
//! carries range + field order into the parenthetical.

use crate::{
    codec::params::{CodecId, FieldOrder},
    format::{InputFormatContext, OutputFormatContext, Stream},
    log_info,
    util::{color::ColorRange, mathematics, pixfmt::PixelFormat, rational::Rational},
};

/// `avcodec_pix_fmt_to_codec_tag` subset — the FIRST matching row of
/// `libavcodec/raw_pix_fmt_tags.h` per format (formats absent here have no
/// tag in C either and print without the parenthetical).
fn codec_tag(fmt: PixelFormat) -> Option<u32> {
    Some(match fmt {
        PixelFormat::Yuv420p => tag(b"I420"),
        PixelFormat::Yuv422p => tag(b"Y42B"),
        PixelFormat::Yuv444p => tag(b"444P"),
        PixelFormat::Yuv420p10le => tag(b"Y3\x0b\x0a"),
        PixelFormat::Yuv422p10le => tag(b"Y3\x0a\x0a"),
        PixelFormat::Yuv444p10le => tag(b"Y3\x00\x0a"),
        PixelFormat::Yuv420p16le => tag(b"Y3\x0b\x10"),
        PixelFormat::Yuv444p16le => tag(b"Y3\x00\x10"),
        PixelFormat::Gray8 => tag(b"Y800"),
        PixelFormat::Gray16le => tag(b"Y1\x00\x10"),
        PixelFormat::Rgb24 => tag(b"RGB\x18"),
        PixelFormat::Bgr24 => tag(b"BGR\x18"),
        PixelFormat::Rgba => tag(b"RGBA"),
        PixelFormat::Bgra => tag(b"BGRA"),
        PixelFormat::Argb => tag(b"ARGB"),
        PixelFormat::Abgr => tag(b"ABGR"),
        PixelFormat::Rgb565le => tag(b"RGB\x10"),
        PixelFormat::Nv12 => tag(b"NV12"),
        PixelFormat::Nv21 => tag(b"NV21"),
        PixelFormat::Yuyv422 => tag(b"YUY2"),
        PixelFormat::Uyvy422 => tag(b"UYVY"),
        // GBRP/GBRAP carry no first-row tag in raw_pix_fmt_tags.h.
        _ => return None,
    })
}

/// `MKTAG(a, b, c, d)` = a | b<<8 | c<<16 | d<<24.
fn tag(t: &[u8; 4]) -> u32 {
    u32::from_le_bytes(*t)
}

/// `av_fourcc_make_string` for our tags: printable bytes as-is, others as
/// `[n]` — that's what turns `RGB\x18` into `RGB[24]`.
fn tag_string(t: u32) -> String {
    let bytes = t.to_le_bytes();
    let mut out = String::new();
    for b in bytes {
        if (0x20..0x7f).contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("[{b}]"));
        }
    }
    out
}

/// `dump.c`'s `print_fps` — integral rates without decimals, fractional ones
/// with two.
fn fps_string(r: Rational) -> String {
    if r.den == 0 {
        return "0.0".into();
    }
    let d = r.to_f64();
    let v = (d * 100.0).round() as i64;
    if v % 100 != 0 {
        format!("{d:3.2}")
    } else if v % 10_000 != 0 {
        format!("{d:1.0}")
    } else {
        format!("{d:6.0}")
    }
}

/// `HH:MM:SS.cc` for a duration in seconds (dump.c's duration print).
fn time_string(secs: f64) -> String {
    let total_cs = (secs * 100.0).round() as u64;
    let cs = total_cs % 100;
    let s = (total_cs / 100) % 60;
    let m = (total_cs / 6000) % 60;
    let h = total_cs / 360_000;
    format!("{h:02}:{m:02}:{s:02}.{cs:02}")
}

/// `av_reduce`-based DAR: `SAR · W/H`.
fn dar_string(st: &Stream) -> Option<String> {
    if st.sample_aspect_ratio.num == 0 {
        return None; // C skips the whole SAR/DAR clause
    }
    let num = st.sample_aspect_ratio.num as i64 * st.codecpar.width as i64;
    let den = st.sample_aspect_ratio.den as i64 * st.codecpar.height as i64;
    let (dar, _) = Rational::reduce(num, den, i32::MAX as i64);
    Some(format!(
        "SAR {}:{} DAR {}:{}",
        st.sample_aspect_ratio.num, st.sample_aspect_ratio.den, dar.num, dar.den
    ))
}

/// The `(pixfmt…, field order)` parenthetical: `(pc, progressive)`.
fn pixfmt_parenthetical(st: &Stream) -> String {
    let mut parts: Vec<String> = Vec::new();
    match st.codecpar.color_range {
        ColorRange::Jpeg => parts.push("pc".into()),
        ColorRange::Mpeg => parts.push("tv".into()),
        ColorRange::Unspecified => {}
    }
    if st.codecpar.field_order != FieldOrder::Unknown {
        parts.push("progressive".into());
    }
    format!("{}({})", st.codecpar.format.name(), parts.join(", "))
}

/// The codec-name prefix + fourcc parenthetical: `rawvideo (I420 / 0x30323449)`.
fn codec_prefix(st: &Stream) -> String {
    let name = match st.codecpar.codec_id {
        CodecId::Rawvideo => "rawvideo",
        CodecId::WrappedAvframe => "wrapped_avframe",
        CodecId::None => "none",
    };
    match codec_tag(st.codecpar.format) {
        Some(t) => format!("{name} ({} / 0x{t:08X})", tag_string(t)),
        None => name.into(),
    }
}

/// `Input #0, …` block (`av_dump_format(…, 0, …, 0)`).
pub fn dump_input(ctx: &InputFormatContext) {
    log_info!(None, "Input #0, {}, from '{}':", ctx.iformat.name, ctx.url);
    let st = &ctx.streams[0];

    let duration_secs = if st.duration != crate::NOPTS && st.time_base.den != 0 {
        st.duration as f64 * st.time_base.to_f64()
    } else {
        0.0
    };
    // Bitrate from the file size when the duration is known.
    let size = ctx.io().size();
    let bitrate = if duration_secs > 0.0 {
        format!(
            "{} kb/s",
            (size as f64 * 8.0 / duration_secs / 1000.0) as i64
        )
    } else {
        "N/A".into()
    };
    log_info!(
        None,
        "  Duration: {}, start: 0.000000, bitrate: {}",
        time_string(duration_secs),
        bitrate
    );

    let mut extras: Vec<String> = Vec::new();
    if let Some(sar) = dar_string(st) {
        extras.push(sar);
    }
    let fps = fps_string(st.avg_frame_rate);
    log_info!(
        None,
        "  Stream #0:0: Video: {}, {}, {}x{}{}, {} fps, {} tbr, {} tbn",
        codec_prefix(st),
        pixfmt_parenthetical(st),
        st.codecpar.width,
        st.codecpar.height,
        if extras.is_empty() {
            String::new()
        } else {
            format!(", {}", extras.join(" "))
        },
        fps,
        fps,
        st.time_base.den,
    );
}

/// `Output #0, …` block (`av_dump_format(…, 1, …, 0)`).
pub fn dump_output(ctx: &OutputFormatContext) {
    log_info!(None, "Output #0, {}, to '{}':", ctx.oformat.name, ctx.url);
    let st = &ctx.streams[0];
    let mut extras: Vec<String> = Vec::new();
    if let Some(sar) = dar_string(st) {
        extras.push(format!("[{sar}]"));
    }
    extras.push("q=2-31".into());
    if st.codecpar.bit_rate > 0 {
        extras.push(format!("{} kb/s", (st.codecpar.bit_rate + 500) / 1000));
    }
    // dump.c separates size from a bracketed SAR by a space but everything
    // else by commas: `128x96 [SAR 1:1 DAR 4:3], q=2-31, 2949 kb/s`.
    let extras_str = match extras.first() {
        None => String::new(),
        Some(first) if first.starts_with('[') => format!(" {}", extras.join(", ")),
        Some(_) => format!(", {}", extras.join(", ")),
    };
    log_info!(
        None,
        "  Stream #0:0: Video: {}, {}, {}x{}{}, {} fps, {} tbn",
        codec_prefix(st),
        pixfmt_parenthetical(st),
        st.codecpar.width,
        st.codecpar.height,
        extras_str,
        fps_string(st.avg_frame_rate),
        st.time_base.den,
    );
}

/// ffmpeg's mapping banner.
pub fn dump_stream_mapping() {
    log_info!(None, "Stream mapping:");
    log_info!(
        None,
        "  Stream #0:0 -> #0:0 (rawvideo (native) -> rawvideo (native))"
    );
}

/// Progress/summary time (`time=00:00:01.00` in `print_report`), from an
/// output timestamp.
pub fn progress_time_string(pts: i64, tb: Rational) -> String {
    let secs = pts as f64 * tb.to_f64();
    time_string(secs)
}

/// Rescale helper re-exported for the transcode loop (the same call fftools
/// makes: NEAR_INF | PASS_MINMAX).
pub fn rescale_ts(pts: i64, from: Rational, to: Rational) -> i64 {
    mathematics::rescale_ts(pts, from, to)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fourcc_formatting_matches_ffmpeg() {
        assert_eq!(tag_string(tag(b"I420")), "I420");
        assert_eq!(tag_string(tag(b"RGB\x18")), "RGB[24]");
        assert_eq!(tag_string(tag(b"RGB\x10")), "RGB[16]");
        assert_eq!(tag(b"RGB\x18"), 0x18424752);
        assert_eq!(tag(b"I420"), 0x30323449);
    }

    #[test]
    fn fps_strings() {
        assert_eq!(fps_string(Rational::new(10, 1)), "10");
        assert_eq!(fps_string(Rational::new(25, 1)), "25");
        // 30000/1001 → 29.97
        let r = Rational::reduce(30000, 1001, i32::MAX as i64).0;
        assert_eq!(fps_string(r), "29.97");
    }

    #[test]
    fn time_strings() {
        assert_eq!(time_string(1.0), "00:00:01.00");
        assert_eq!(time_string(61.23), "00:01:01.23");
        assert_eq!(time_string(3661.0), "01:01:01.00");
    }
}
