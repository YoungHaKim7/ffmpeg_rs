//! YUV4MPEG (Y4M) demuxer + muxer — port of `libavformat/yuv4mpegdec.c` +
//! `yuv4mpegenc.c` + `yuv4mpeg.h`.
//!
//! Y4M is the raw-film exchange format: one text stream header
//! `YUV4MPEG2 W128 H96 F10:1 Ip A0:0 C420jpeg XYSCSS=420JPEG`, then per frame
//! `FRAME\n` + the compact planar payload. Byte-level fidelity matters
//! because files written here must round-trip through real ffmpeg unchanged
//! (the golden test pins this).
//!
//! ## Ported subset of the `C` colorspace table
//!
//! The C table maps 27 `C` tokens to pixel formats; this port carries the
//! rows whose format exists in [`crate::util::pixfmt`] and turns the rest
//! into explicit errors *including rows whose name is a prefix of a
//! supported one* (`420p9` vs `420`, `mono16` vs `mono`, `444alpha` vs
//! `444`) — without those guard rows a naive prefix match would silently
//! mis-parse frame sizes. Missing-but-unambiguous rows (`411`, `422p16`)
//! simply fail with "unknown pixel format", the C behavior for tokens
//! outside its table.
//!
//! BE-format note: `mono16` is GRAY16BE in C and stays un-ported (LE-only
//! subset).

use crate::{
    codec::{
        packet::{Packet, PacketFlags},
        params::{CodecId, FieldOrder},
    },
    imgutils, log_error,
    util::{
        color::{ChromaLocation, ColorRange},
        error::{Error, Result},
        pixfmt::PixelFormat,
        rational::Rational,
    },
};

use super::{
    Stream,
    demux::{Demuxer, PROBE_SCORE_MAX},
    io::IoContext,
    mux::Muxer,
};

/// `Y4M_MAGIC` (yuv4mpeg.h).
const Y4M_MAGIC: &[u8] = b"YUV4MPEG2";
/// `Y4M_FRAME_MAGIC`.
const Y4M_FRAME_MAGIC: &[u8] = b"FRAME";
/// `Y4M_FRAME_MAGIC_LEN`.
const Y4M_FRAME_MAGIC_LEN: usize = Y4M_FRAME_MAGIC.len() + 1; // "FRAME\n"
/// `MAX_YUV4_HEADER`.
const MAX_YUV4_HEADER: usize = 128;
/// `MAX_FRAME_HEADER`.
const MAX_FRAME_HEADER: usize = 80;

/// `yuv4_read_header`'s colorspace table rows that map into our format
/// subset, in C order (order matters: `av_strstart` prefix matching).
/// `None` rows are the guard entries explained in the module docs.
const COLORSPACE_TABLE: &[(&str, Option<(PixelFormat, ChromaLocation)>)] = &[
    (
        "420jpeg",
        Some((PixelFormat::Yuv420p, ChromaLocation::Center)),
    ),
    (
        "420mpeg2",
        Some((PixelFormat::Yuv420p, ChromaLocation::Left)),
    ),
    (
        "420paldv",
        Some((PixelFormat::Yuv420p, ChromaLocation::TopLeft)),
    ),
    (
        "420p16",
        Some((PixelFormat::Yuv420p16le, ChromaLocation::Unspecified)),
    ),
    ("422p16", None), // yuv422p16 — not in subset (guard: prefix "422" exists)
    (
        "444p16",
        Some((PixelFormat::Yuv444p16le, ChromaLocation::Unspecified)),
    ),
    ("420p14", None), // guard: prefix "420"
    ("422p14", None), // guard: prefix "422"
    ("444p14", None), // guard: prefix "444"
    ("420p12", None), // guard
    ("422p12", None), // guard
    ("444p12", None), // guard
    (
        "420p10",
        Some((PixelFormat::Yuv420p10le, ChromaLocation::Unspecified)),
    ),
    (
        "422p10",
        Some((PixelFormat::Yuv422p10le, ChromaLocation::Unspecified)),
    ),
    (
        "444p10",
        Some((PixelFormat::Yuv444p10le, ChromaLocation::Unspecified)),
    ),
    ("420p9", None), // guard
    ("422p9", None), // guard
    ("444p9", None), // guard
    ("420", Some((PixelFormat::Yuv420p, ChromaLocation::Center))),
    ("411", None), // yuv411p — unambiguous, plain "unknown" error
    (
        "422",
        Some((PixelFormat::Yuv422p, ChromaLocation::Unspecified)),
    ),
    ("444alpha", None), // yuva444p — guard: prefix "444"
    (
        "444",
        Some((PixelFormat::Yuv444p, ChromaLocation::Unspecified)),
    ),
    ("mono16", None), // gray16be — guard: prefix "mono"
    ("mono12", None), // guard
    ("mono10", None), // guard
    ("mono9", None),  // guard
    (
        "mono",
        Some((PixelFormat::Gray8, ChromaLocation::Unspecified)),
    ),
];

/// `YSCSS=` legacy table (uppercase names), subset.
const YSCSS_TABLE: &[(&str, PixelFormat)] = &[
    ("420JPEG", PixelFormat::Yuv420p),
    ("420MPEG2", PixelFormat::Yuv420p),
    ("420PALDV", PixelFormat::Yuv420p),
    ("420P10", PixelFormat::Yuv420p10le),
    ("444P10", PixelFormat::Yuv444p10le),
    ("420P16", PixelFormat::Yuv420p16le),
    ("444P16", PixelFormat::Yuv444p16le),
    ("422", PixelFormat::Yuv422p),
    ("444", PixelFormat::Yuv444p),
];

/// `ff_yuv4mpegpipe_demuxer`'s private state.
pub struct Y4mDemuxer {
    /// `s->packet_size` — frame header + payload.
    packet_size: usize,
    /// `ffformatcontext(s)->data_offset` — first byte after the stream
    /// header; pts math divides by packet_size from here.
    data_offset: u64,
    /// Stream timebase (set by read_header; stamped on packets like
    /// av_read_frame does).
    time_base: Rational,
}

impl Y4mDemuxer {
    pub fn new() -> Self {
        Y4mDemuxer {
            packet_size: 0,
            data_offset: 0,
            time_base: Rational::UNKNOWN,
        }
    }
}

/// Leading decimal integer of a string, `sscanf("%d")` style (0 when absent).
fn leading_int(s: &str) -> i32 {
    let digits: String = s
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '-')
        .collect();
    digits.parse().unwrap_or(0)
}

/// Parse `"%d:%d"` — the `F`/`A` header values. Like `sscanf`, parsing stops
/// at the first non-numeric character ("1 Ip A1:1" → denominator 1).
fn parse_ratio(tok: &[u8]) -> (i32, i32) {
    let s = std::str::from_utf8(tok).unwrap_or("");
    let mut parts = s.splitn(2, ':');
    let n = parts.next().map(leading_int).unwrap_or(0);
    let d = parts.next().map(leading_int).unwrap_or(0);
    (n, d)
}

/// `strtol`-style leading integer of a byte token (None when no digits —
/// the caller keeps its -1 sentinel, like an untouched C out-param).
fn parse_i32(tok: &[u8]) -> Option<i32> {
    let s = std::str::from_utf8(tok).ok()?;
    let digits: String = s
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '-')
        .collect();
    digits.parse().ok()
}

impl Demuxer for Y4mDemuxer {
    /// `yuv4_read_header` (yuv4mpegdec.c:34).
    fn read_header(&mut self, io: &mut IoContext) -> Result<Stream> {
        let mut width: i32 = -1;
        let mut height: i32 = -1;
        let mut raten: i32 = 0;
        let mut rated: i32 = 0;
        let mut aspectn: i32 = 0;
        let mut aspectd: i32 = 0;
        let mut pix_fmt: Option<PixelFormat> = None;
        let mut alt_pix_fmt: Option<PixelFormat> = None;
        let mut chroma_sample_location = ChromaLocation::Unspecified;
        let mut field_order = FieldOrder::Unknown;
        let mut color_range = ColorRange::Unspecified;

        // Read the header line. The loop mirrors C including the synthetic
        // trailing space that disambiguates "444" from "444alpha".
        let mut header = [0u8; MAX_YUV4_HEADER + 2];
        let mut i = 0usize;
        loop {
            header[i] = io.r8()?;
            if header[i] == b'\n' {
                header[i + 1] = 0x20; // synthetic space after last option
                break;
            }
            i += 1;
            if i == MAX_YUV4_HEADER {
                log_error!(Some("yuv4mpegpipe"), "Header too large.");
                return Err(Error::InvalidArgument("header too large".into()));
            }
        }
        let header_end = i + 1; // index of the synthetic space

        if &header[..Y4M_MAGIC.len()] != Y4M_MAGIC {
            log_error!(Some("yuv4mpegpipe"), "Invalid magic number for yuv4mpeg.");
            return Err(Error::InvalidData(
                "invalid magic number for yuv4mpeg".into(),
            ));
        }

        // Token walk: `tokstart` scans option characters; each option's value
        // runs to the next space. Kept as a byte-index walk (not
        // split_whitespace!) because prefixes are significant.
        let mut tok = Y4M_MAGIC.len() + 1;
        while tok < header_end {
            if header[tok] == 0x20 {
                tok += 1;
                continue;
            }
            let opt = header[tok];
            tok += 1;
            match opt {
                b'W' => {
                    width = parse_i32(&header[tok..header_end]).unwrap_or(-1);
                    // C's strtol also consumes the digits; the outer loop
                    // skips to the next space anyway.
                }
                b'H' => {
                    height = parse_i32(&header[tok..header_end]).unwrap_or(-1);
                }
                b'C' => {
                    let mut matched = false;
                    for (name, entry) in COLORSPACE_TABLE {
                        if header[tok..header_end].starts_with(name.as_bytes()) {
                            match entry {
                                Some((fmt, loc)) => {
                                    pix_fmt = Some(*fmt);
                                    if *loc != ChromaLocation::Unspecified {
                                        chroma_sample_location = *loc;
                                    }
                                }
                                None => {
                                    log_error!(
                                        Some("yuv4mpegpipe"),
                                        "YUV4MPEG stream contains an unsupported pixel \
                                         format ({} is outside the ffmpeg_rs subset).",
                                        name
                                    );
                                    return Err(Error::Unsupported(format!(
                                        "colorspace '{name}' not in the ported subset"
                                    )));
                                }
                            }
                            matched = true;
                            break;
                        }
                    }
                    if !matched {
                        log_error!(
                            Some("yuv4mpegpipe"),
                            "YUV4MPEG stream contains an unknown pixel format."
                        );
                        return Err(Error::InvalidData(
                            "YUV4MPEG stream contains an unknown pixel format.".into(),
                        ));
                    }
                    while tok < header_end && header[tok] != 0x20 {
                        tok += 1;
                    }
                }
                b'I' => {
                    field_order = match header[tok] {
                        b'?' => FieldOrder::Unknown,
                        b'p' => FieldOrder::Progressive,
                        b't' => FieldOrder::Tt,
                        b'b' => FieldOrder::Bb,
                        b'm' => {
                            log_error!(
                                Some("yuv4mpegpipe"),
                                "YUV4MPEG stream contains mixed interlaced and \
                                 non-interlaced frames."
                            );
                            return Err(Error::Unsupported(
                                "mixed interlaced and non-interlaced frames".into(),
                            ));
                        }
                        _ => {
                            log_error!(Some("yuv4mpegpipe"), "YUV4MPEG has invalid header.");
                            return Err(Error::InvalidData("YUV4MPEG has invalid header".into()));
                        }
                    };
                    tok += 1;
                }
                b'F' => {
                    let (n, d) = parse_ratio(&header[tok..header_end]);
                    raten = n;
                    rated = d;
                    while tok < header_end && header[tok] != 0x20 {
                        tok += 1;
                    }
                }
                b'A' => {
                    let (n, d) = parse_ratio(&header[tok..header_end]);
                    aspectn = n;
                    aspectd = d;
                    while tok < header_end && header[tok] != 0x20 {
                        tok += 1;
                    }
                }
                b'X' => {
                    let rest = &header[tok..header_end];
                    if rest.starts_with(b"YSCSS=") {
                        let legacy = &rest[6..];
                        for (name, fmt) in YSCSS_TABLE {
                            if legacy.starts_with(name.as_bytes()) {
                                alt_pix_fmt = Some(*fmt);
                                break;
                            }
                        }
                    } else if rest.starts_with(b"COLORRANGE=") {
                        let val = &rest[11..];
                        if val.starts_with(b"FULL") {
                            color_range = ColorRange::Jpeg;
                        } else if val.starts_with(b"LIMITED") {
                            color_range = ColorRange::Mpeg;
                        }
                    }
                    while tok < header_end && header[tok] != 0x20 {
                        tok += 1;
                    }
                }
                _ => {
                    // Unknown option letter: C's switch falls through and the
                    // scan continues past the token.
                    while tok < header_end && header[tok] != 0x20 {
                        tok += 1;
                    }
                }
            }
        }

        if width == -1 || height == -1 {
            log_error!(Some("yuv4mpegpipe"), "YUV4MPEG has invalid header.");
            return Err(Error::InvalidData("YUV4MPEG has invalid header".into()));
        }

        let pix_fmt = match pix_fmt.or(alt_pix_fmt) {
            Some(f) => f,
            None => PixelFormat::Yuv420p,
        };

        if raten <= 0 || rated <= 0 {
            // Frame rate unknown
            raten = 25;
            rated = 1;
        }

        if aspectn == 0 && aspectd == 0 {
            // Pixel aspect unknown
            aspectd = 1;
        }

        let mut st = Stream::new_video(0);
        st.codecpar.width = width as u32;
        st.codecpar.height = height as u32;
        st.set_pts_info(rated as i64, raten as i64);
        st.codecpar.format = pix_fmt;
        st.codecpar.codec_id = CodecId::Rawvideo;
        st.sample_aspect_ratio = Rational::new(aspectn, aspectd);
        st.codecpar.sample_aspect_ratio = st.sample_aspect_ratio;
        st.codecpar.framerate = st.avg_frame_rate;
        st.codecpar.chroma_location = chroma_sample_location;
        st.codecpar.color_range = color_range;
        st.codecpar.field_order = field_order;

        self.packet_size = imgutils::get_buffer_size(pix_fmt, width as u32, height as u32, 1)?
            + Y4M_FRAME_MAGIC_LEN;
        self.data_offset = io.tell();
        self.time_base = st.time_base;
        st.duration =
            ((io.size() as i64 - self.data_offset as i64) / self.packet_size as i64).max(0);

        Ok(st)
    }

    /// `yuv4_read_packet` (yuv4mpegdec.c:267).
    fn read_packet(&mut self, io: &mut IoContext) -> Result<Packet> {
        let off = io.tell();

        let mut header = [0u8; MAX_FRAME_HEADER + 1];
        let mut i = 0usize;
        while i < MAX_FRAME_HEADER {
            header[i] = io.r8()?;
            if header[i] == b'\n' {
                break;
            }
            i += 1;
        }
        // C's check order: error, then EOF, then overlong header.
        if let Some(err) = io.take_error() {
            return Err(Error::Io(err));
        } else if io.is_eof() {
            return Err(Error::Eof);
        } else if i == MAX_FRAME_HEADER {
            return Err(Error::InvalidData("frame header too large".into()));
        }

        if &header[..Y4M_FRAME_MAGIC.len()] != Y4M_FRAME_MAGIC {
            return Err(Error::InvalidData("frame header magic mismatch".into()));
        }

        let want = self.packet_size - Y4M_FRAME_MAGIC_LEN;
        let payload = match io.get_packet(want) {
            Ok(data) => data,
            Err(Error::Eof) => return Err(Error::Eof),
            Err(e) => return Err(e),
        };
        if payload.len() != want {
            return if io.is_eof() {
                Err(Error::Eof)
            } else {
                Err(Error::InvalidData("short frame".into()))
            };
        }

        let mut pkt = Packet::from_vec(payload);
        pkt.stream_index = 0;
        pkt.pts = (off - self.data_offset) as i64 / self.packet_size as i64;
        pkt.duration = 1;
        pkt.time_base = self.time_base;
        pkt.flags = pkt.flags.union(PacketFlags::KEY);
        Ok(pkt)
    }
}

/// `yuv4_probe` — magic match.
pub fn probe(buf: &[u8]) -> u32 {
    if buf.starts_with(Y4M_MAGIC) {
        PROBE_SCORE_MAX
    } else {
        0
    }
}

/// `ff_yuv4mpegpipe_muxer`'s private state (none — kept for symmetry).
pub struct Y4mMuxer;

impl Y4mMuxer {
    pub fn new() -> Self {
        Y4mMuxer
    }
}

impl Muxer for Y4mMuxer {
    /// `yuv4_init` (yuv4mpegenc.c:225) — codec gate + format gate. Formats
    /// outside the "official" set behave like `-strict -1` (warning +
    /// proceed), documented divergence: C errors at normal compliance.
    fn init(&mut self, streams: &[Stream]) -> Result<()> {
        let par = &streams[0].codecpar;
        if par.codec_id != CodecId::WrappedAvframe && par.codec_id != CodecId::Rawvideo {
            log_error!(Some("yuv4mpegpipe"), "ERROR: Codec not supported.");
            return Err(Error::InvalidData(
                "codec not supported by yuv4mpegpipe".into(),
            ));
        }
        let unofficial = !matches!(
            par.format,
            PixelFormat::Gray8 | PixelFormat::Yuv420p | PixelFormat::Yuv422p | PixelFormat::Yuv444p
        );
        if unofficial {
            crate::log_warning!(
                Some("yuv4mpegpipe"),
                "Warning: generating non standard YUV stream. Mjpegtools will not work."
            );
        }
        Ok(())
    }

    /// `yuv4_write_header` (yuv4mpegenc.c:29).
    fn write_header(&mut self, io: &mut IoContext, streams: &[Stream]) -> Result<()> {
        let st = &streams[0];
        let width = st.codecpar.width;
        let height = st.codecpar.height;

        // F from the stream timebase (reduced), like C.
        let (fps, _) = Rational::reduce(
            st.time_base.den as i64,
            st.time_base.num as i64,
            (1u64 << 31) as i64 - 1,
        );

        let aspectn = st.sample_aspect_ratio.num;
        let mut aspectd = st.sample_aspect_ratio.den;
        if aspectn == 0 && aspectd == 1 {
            aspectd = 0; // 0:0 means unknown
        }

        let colorrange = match st.codecpar.color_range {
            ColorRange::Mpeg => " XCOLORRANGE=LIMITED",
            ColorRange::Jpeg => " XCOLORRANGE=FULL",
            ColorRange::Unspecified => "",
        };

        let inter = match st.codecpar.field_order {
            FieldOrder::Tt | FieldOrder::Tb => 't',
            FieldOrder::Bb | FieldOrder::Bt => 'b',
            _ => 'p',
        };

        // Note: C's YUVJ* formats would also force " XCOLORRANGE=FULL" here;
        // those deprecated aliases are not part of the ported format set.
        let colorspace = colorspace_token(st);

        let header = format!(
            "{} W{width} H{height} F{}:{} I{inter} A{aspectn}:{aspectd}{}{}\n",
            std::str::from_utf8(Y4M_MAGIC).unwrap(),
            fps.num,
            fps.den,
            colorspace,
            colorrange,
        );
        io.write_all(header.as_bytes()).map_err(|e| {
            log_error!(
                Some("yuv4mpegpipe"),
                "Error. YUV4MPEG stream header write failed."
            );
            e
        })
    }

    /// `yuv4_write_packet` (yuv4mpegenc.c:181) — rawvideo path only
    /// (the WRAPPED_AVFRAME plane-walking path is unused by this pipeline).
    fn write_packet(
        &mut self,
        io: &mut IoContext,
        _streams: &[Stream],
        pkt: &Packet,
    ) -> Result<()> {
        io.write_all(b"FRAME\n")?;
        io.write_all(pkt.as_slice())?;
        Ok(())
    }
}

/// The colorspace token for `write_header`'s format switch, including the
/// chroma-location-selected 420 triplet and the `XYSCSS=` legacy suffix C
/// appends to every token it knows.
fn colorspace_token(st: &Stream) -> &'static str {
    match st.codecpar.format {
        PixelFormat::Gray8 => " Cmono",
        PixelFormat::Yuv420p => match st.codecpar.chroma_location {
            ChromaLocation::TopLeft => " C420paldv XYSCSS=420PALDV",
            ChromaLocation::Left => " C420mpeg2 XYSCSS=420MPEG2",
            // Default matches CENTER and UNSPECIFIED alike (C's default arm).
            _ => " C420jpeg XYSCSS=420JPEG",
        },
        PixelFormat::Yuv422p => " C422 XYSCSS=422",
        PixelFormat::Yuv444p => " C444 XYSCSS=444",
        PixelFormat::Yuv420p10le => " C420p10 XYSCSS=420P10",
        PixelFormat::Yuv422p10le => " C422p10 XYSCSS=422P10",
        PixelFormat::Yuv444p10le => " C444p10 XYSCSS=444P10",
        PixelFormat::Yuv420p16le => " C420p16 XYSCSS=420P16",
        PixelFormat::Yuv444p16le => " C444p16 XYSCSS=444P16",
        // rgb formats and friends: C writes an empty token (its switch falls
        // through), producing "… A0:0\n" — reproduce that.
        _ => "",
    }
}

// The transcode loop rescales packet timestamps between stream/codec
// timebases via `crate::util::mathematics::rescale_ts`, like fftools does
// with av_rescale_q_rnd(…, AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX).

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::testutil::MemHandler;

    fn io_with(data: &[u8]) -> IoContext {
        MemHandler::io(data)
    }

    fn demux_header(header: &str) -> Result<Stream> {
        let mut io = io_with(header.as_bytes());
        let mut dem = Y4mDemuxer::new();
        dem.read_header(&mut io)
    }

    #[test]
    fn parses_full_featured_header() {
        let st = demux_header("YUV4MPEG2 W128 H96 F10:1 Ip A1:1 C420mpeg2 XCOLORRANGE=LIMITED\n")
            .unwrap();
        assert_eq!((st.codecpar.width, st.codecpar.height), (128, 96));
        assert_eq!(st.codecpar.format, PixelFormat::Yuv420p);
        assert_eq!(st.time_base, Rational::new(1, 10));
        assert_eq!(st.avg_frame_rate, Rational::new(10, 1));
        assert_eq!(st.sample_aspect_ratio, Rational::new(1, 1));
        assert_eq!(st.codecpar.chroma_location, ChromaLocation::Left);
        assert_eq!(st.codecpar.color_range, ColorRange::Mpeg);
        assert_eq!(st.codecpar.field_order, FieldOrder::Progressive);
    }

    #[test]
    fn minimal_header_defaults() {
        // No C → yuv420p but chroma location stays UNSPECIFIED (only table
        // rows set it in C); no F → 25:1; no A → 0:1.
        let st = demux_header("YUV4MPEG2 W64 H48\n").unwrap();
        assert_eq!(st.codecpar.format, PixelFormat::Yuv420p);
        assert_eq!(st.codecpar.chroma_location, ChromaLocation::Unspecified);
        assert_eq!(st.avg_frame_rate, Rational::new(25, 1));
        assert_eq!(st.sample_aspect_ratio, Rational::new(0, 1));
        // F 0:0 also means unknown.
        let st = demux_header("YUV4MPEG2 W64 H48 F0:0\n").unwrap();
        assert_eq!(st.avg_frame_rate, Rational::new(25, 1));
    }

    #[test]
    fn missing_dimensions_rejected() {
        assert!(demux_header("YUV4MPEG2 H48\n").is_err());
        assert!(demux_header("YUV4MPEG2 W64\n").is_err());
    }

    #[test]
    fn bad_magic_rejected() {
        assert!(demux_header("YUV4MPEG9 W64 H48\n").is_err());
    }

    #[test]
    fn unknown_colorspace_rejected() {
        assert!(demux_header("YUV4MPEG2 W64 H48 C999\n").is_err());
        // And the guard rows error instead of mis-parsing as their prefix.
        let err = demux_header("YUV4MPEG2 W64 H48 C420p9\n").unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
        assert!(demux_header("YUV4MPEG2 W64 H48 Cmono16\n").is_err());
        assert!(demux_header("YUV4MPEG2 W64 H48 C444alpha\n").is_err());
    }

    #[test]
    fn prefix_disambiguation_via_synthetic_space() {
        // "C444" must parse as yuv444p…
        let st = demux_header("YUV4MPEG2 W64 H48 C444\n").unwrap();
        assert_eq!(st.codecpar.format, PixelFormat::Yuv444p);
        // …and "C444p10" as 10-bit (not caught by the bare "444" row).
        let st = demux_header("YUV4MPEG2 W64 H48 C444p10\n").unwrap();
        assert_eq!(st.codecpar.format, PixelFormat::Yuv444p10le);
    }

    #[test]
    fn legacy_yscss_parsed() {
        let st = demux_header("YUV4MPEG2 W64 H48 XYSCSS=420MPEG2\n").unwrap();
        assert_eq!(st.codecpar.format, PixelFormat::Yuv420p);
        let st = demux_header("YUV4MPEG2 W64 H48 XYSCSS=444P10\n").unwrap();
        assert_eq!(st.codecpar.format, PixelFormat::Yuv444p10le);
    }

    #[test]
    fn mixed_interlace_rejected() {
        assert!(matches!(
            demux_header("YUV4MPEG2 W64 H48 Im\n"),
            Err(Error::Unsupported(_))
        ));
    }

    #[test]
    fn header_too_large_rejected() {
        let long = format!("YUV4MPEG2 W64 H48 {}", "X".repeat(200));
        assert!(demux_header(&long).is_err());
    }

    #[test]
    fn packet_read_pts_and_eof() {
        let payload = vec![0xABu8; 64 * 48 * 3 / 2];
        let mut file = b"YUV4MPEG2 W64 H48 F25:1 C420jpeg\n".to_vec();
        for n in 0..3u8 {
            file.extend_from_slice(b"FRAME\n");
            file.extend(payload.iter().map(|&b| b ^ n));
        }
        let mut io = io_with(&file);
        let mut dem = Y4mDemuxer::new();
        let st = dem.read_header(&mut io).unwrap();
        assert_eq!(st.duration, 3);

        for n in 0..3u8 {
            let pkt = dem.read_packet(&mut io).unwrap();
            assert_eq!(pkt.pts, n as i64);
            assert_eq!(pkt.duration, 1);
            assert!(pkt.flags.contains(PacketFlags::KEY));
            assert_eq!(pkt.as_slice()[0], 0xAB ^ n);
        }
        assert!(matches!(dem.read_packet(&mut io), Err(Error::Eof)));
    }

    #[test]
    fn truncated_final_frame_is_eof() {
        let mut file = b"YUV4MPEG2 W64 H48\n".to_vec();
        file.extend_from_slice(b"FRAME\n");
        file.extend_from_slice(&vec![0u8; 100]); // short
        let mut io = io_with(&file);
        let mut dem = Y4mDemuxer::new();
        dem.read_header(&mut io).unwrap();
        assert!(matches!(dem.read_packet(&mut io), Err(Error::Eof)));
    }

    #[test]
    fn muxer_header_matches_ffmpeg_layout() {
        let mut st = Stream::new_video(0);
        st.codecpar.codec_id = CodecId::Rawvideo;
        st.codecpar.width = 128;
        st.codecpar.height = 96;
        st.codecpar.format = PixelFormat::Yuv420p;
        st.codecpar.chroma_location = ChromaLocation::Center;
        st.set_pts_info(1, 10);
        st.sample_aspect_ratio = Rational::new(0, 1);

        // Writes land in the handler's shared buffer; read them back.
        let mut io = MemHandler::io(b"");
        let mut mux = Y4mMuxer::new();
        mux.init(&[st.clone()]).unwrap();
        mux.write_header(&mut io, &[st.clone()]).unwrap();
        io.seek(0).unwrap();
        let out = io.get_packet(4096).unwrap();
        let out = String::from_utf8(out).unwrap();
        assert_eq!(
            out,
            "YUV4MPEG2 W128 H96 F10:1 Ip A0:0 C420jpeg XYSCSS=420JPEG\n"
        );
    }

    #[test]
    fn muxer_420_chroma_location_triplet() {
        for (loc, want) in [
            (ChromaLocation::Left, " C420mpeg2 XYSCSS=420MPEG2"),
            (ChromaLocation::TopLeft, " C420paldv XYSCSS=420PALDV"),
            (ChromaLocation::Center, " C420jpeg XYSCSS=420JPEG"),
        ] {
            let mut st = Stream::new_video(0);
            st.codecpar.width = 8;
            st.codecpar.height = 8;
            st.codecpar.format = PixelFormat::Yuv420p;
            st.codecpar.chroma_location = loc;
            st.set_pts_info(1, 25);
            let token = colorspace_token(&st);
            assert_eq!(token, want);
        }
    }
}
