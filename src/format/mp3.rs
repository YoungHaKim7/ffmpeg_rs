//! MP3 demuxer — port of `libavformat/mp3dec.c` (probe, header, packets)
//! fused with the frame-splitting core of `libavcodec/mpegaudio_parser.c`.
//!
//! C's design SPLITS the work: `mp3_read_packet` hands out fixed 1024-byte
//! chunks and `AVSTREAM_PARSE_FULL_RAW` + the mpegaudio parser re-splits
//! them into whole frames downstream. The port has no generic parser layer,
//! so this demuxer emits frame-accurate packets directly: read header →
//! `frame_size` bytes → next. The observable packet stream is identical to
//! C's demuxer+parser composition.
//!
//! ## C → Rust map
//!
//! | C | here |
//! |---|---|
//! | `mp3_read_probe` (mp3dec.c:70-137) | [`probe`] — the frame-streak scan, four score tiers |
//! | ID3v2 skip (`ff_id3v2_read`/`_skip`) | [`skip_id3v2`] — syncsafe length, contents dropped (no metadata dictionary) |
//! | junk skip (mp3dec.c:483-500) | two-consecutive-frame check with `MP3_MASK` match, ≤64 KiB window |
//! | `mp3_read_header` (442-517) | [`Mp3Demuxer::read_header`] — codecpar filled from the first header (C defers to the parser; find_stream_info wants the fields now) |
//! | `mp3_read_packet` + `mpegaudio_parse` (parser.c) | [`Mp3Demuxer::read_packet`] — one whole frame per packet, resync on garbage |
//! | Xing/Info/LAME + VBRI (`mp3_parse_vbr_tags`, 267-441) | [`Mp3Demuxer::parse_vbr_tags`] — tag frame skipped, gapless (`start_skip_samples` / `first_discard_sample`) and duration exported to the stream |
//! | iTunSMPB (`mp3_parse_itunes_smpb`) | not ported (ID3v2 contents are dropped, so the comment is never seen) |
//! | ID3v1, replaygain, seek index | not ported |
//!
//! Timestamps: C sets tb = 1/14112000 (LCM of all mp3 rates) and lets the
//! parser stamp pts; the port uses tb = 1/sample_rate with per-frame
//! pts/duration advanced by 1152 (or 576 lsf) samples — identical wall
//! times, simpler arithmetic.

use crate::{
    codec::{
        audio::mp3::{MpaDecodeHeader, avpriv_mpegaudio_decode_header, ff_mpa_check_header},
        packet::{Packet, PacketFlags},
        params::CodecId,
    },
    util::{
        error::{Error, Result},
        rational::Rational,
    },
};

use super::{
    Stream,
    demux::{DemuxOptions, Demuxer, InputFormat},
    id3,
    io::IoContext,
};

/// `MP3_MASK` (mp3dec.c:38): the header bits that must MATCH between
/// consecutive frames of one stream (everything except bitrate/padding/
/// private/copyright/original/emphasis).
const MP3_MASK: u32 = 0xfffe_0cc0;

/// `MPA_MAX_CODED_FRAME_SIZE` (mpegaudio.h): free-format read cap.
const MPA_MAX_CODED_FRAME_SIZE: usize = 2880;

/// The registry row (`ff_mp3_demuxer`).
pub static MP3_INPUT_FORMAT: InputFormat = InputFormat {
    name: "mp3",
    long_name: "MP3 (MPEG audio layer 3)",
    extensions: &["mp3"],
    probe: Some(probe),
    make: |_opts: &DemuxOptions| Box::new(Mp3Demuxer::new()),
};

/// The demuxer state: first header's mask + running sample stamps.
pub struct Mp3Demuxer {
    /// First valid header (mask-match reference for resync).
    header: u32,
    /// Samples per frame (1152, or 576 lsf).
    frame_samples: i64,
    sample_rate: i32,
    /// Next packet's pts (in stream tb units).
    next_pts: i64,
    started: bool,
}

impl Default for Mp3Demuxer {
    fn default() -> Self {
        Self::new()
    }
}

impl Mp3Demuxer {
    pub fn new() -> Self {
        Mp3Demuxer {
            header: 0,
            frame_samples: 0,
            sample_rate: 0,
            next_pts: 0,
            started: false,
        }
    }

    /// `mp3_parse_vbr_tags` + `mp3_parse_info_tag` + `mp3_parse_vbri_tag`
    /// (mp3dec.c:267-441): read the frame at `base`; if it carries a
    /// Xing/Info (LAME) or VBRI tag, export gapless and duration onto the
    /// stream and seek past the tag frame. `Ok(false)` = no usable tag;
    /// the caller restores the position and the frame demuxes as audio.
    ///
    /// Empirically (LAME-encoded fixture, vs system ffmpeg): the tag's
    /// `frames` counts AUDIO frames — the tag frame itself is excluded —
    /// so these formulas stay in the same sample timeline as the packets
    /// emitted after the skip: first audio packet pts 0, `frames·spf`
    /// samples total.
    fn parse_vbr_tags(&mut self, io: &mut IoContext, st: &mut Stream, base: u64) -> Result<bool> {
        // mp3_parse_vbr_tags (383-397): decode the header at base — a
        // non-frame, free-format frame, or non-layer-3 means no tag.
        let (header, vbrtag_size) = match check_at(io, base) {
            Ok(v) => v,
            Err(_) => return Ok(false),
        };
        let mut c = MpaDecodeHeader::default();
        let _free = avpriv_mpegaudio_decode_header(&mut c, header);
        if c.layer != 3 {
            return Ok(false);
        }
        let spf: i64 = if c.lsf != 0 { 576 } else { 1152 };
        st.set_pts_info(1, c.sample_rate as i64);
        st.codecpar.sample_rate = c.sample_rate;

        // The whole frame in memory (C streams it; reads past a truncated
        // frame yield zeros there, and be32()/be24() below mirror that).
        io.seek(base)?;
        let mut buf = vec![0u8; vbrtag_size];
        read_upto(io, &mut buf);

        let mut frames = 0i64;
        let mut header_filesize = 0i64;
        let mut is_cbr = false;
        let mut start_pad = 0i64;
        let mut end_pad = 0i64;

        // ---- mp3_parse_info_tag (157-265): Xing/Info ----
        // xing_offtbl[lsf][mono] (mp3dec.c:166): C measures from AFTER
        // the 4 consumed header bytes; `buf` includes them, hence +4.
        let xing_off =
            4 + [[32usize, 17], [17, 9]][(c.lsf != 0) as usize][(c.nb_channels == 1) as usize];
        let magic = be32(&buf, xing_off);
        if magic == u32::from_be_bytes(*b"Info") {
            is_cbr = true;
        }
        if is_cbr || magic == u32::from_be_bytes(*b"Xing") {
            let flags = be32(&buf, xing_off + 4);
            let mut q = xing_off + 8;
            if flags & 1 != 0 {
                frames = be32(&buf, q) as i64;
                q += 4; // frames
            }
            if flags & 2 != 0 {
                header_filesize = be32(&buf, q) as i64;
                q += 4; // bytes
            }
            if flags & 4 != 0 {
                q += 100; // TOC (seek index not ported)
            }
            if flags & 8 != 0 {
                q += 4; // quality
            }
            // Encoder short version string — only LAME-family tags carry
            // a usable delay field (241-253).
            let version = &buf[q.min(buf.len())..(q + 9).min(buf.len())];
            q += 9 + 1 + 1 + 4 + 2 + 2 + 1 + 1; // ver, rev+vbr, lowpass, peak, rgains, flags+ath, abr
            if version.starts_with(b"LAME")
                || version.starts_with(b"Lavf")
                || version.starts_with(b"Lavc")
            {
                let v = be24(&buf, q);
                start_pad = (v >> 12) as i64;
                end_pad = (v & 4095) as i64;
                st.start_skip_samples = start_pad + 528 + 1;
                if frames != 0 {
                    st.first_discard_sample = -end_pad + 528 + 1 + frames * spf;
                    st.last_discard_sample = frames * spf;
                }
            }
        }

        // ---- mp3_parse_vbri_tag (267-282): always base+4+32 ----
        if frames == 0 && header_filesize == 0 {
            let p = 4 + 32;
            if be32(&buf, p) == u32::from_be_bytes(*b"VBRI") && be16(&buf, p + 4) == 1 {
                header_filesize = be32(&buf, p + 10) as i64;
                frames = be32(&buf, p + 14) as i64;
            }
        }

        // "Packets keep the skipped samples, so shift the timeline
        // instead" (431-434): start_time = start_skip_samples in tb —
        // identity here, tb is 1/sample_rate.
        if st.start_skip_samples != 0 {
            st.start_time = st.start_skip_samples;
        }

        if frames == 0 && header_filesize == 0 {
            return Ok(false);
        }

        // Skip the vbr tag frame (437).
        io.seek(base + vbrtag_size as u64)?;

        if frames != 0 {
            if st.duration == crate::NOPTS {
                st.duration = frames * spf - start_pad - end_pad;
            }
            if header_filesize != 0 && !is_cbr {
                st.codecpar.bit_rate = header_filesize * 8 * c.sample_rate as i64 / (frames * spf);
            }
        }
        Ok(true)
    }
}

impl Demuxer for Mp3Demuxer {
    /// `mp3_read_header` (442-517): ID3v2 skip, VBR-tag parse off the
    /// frame at the current offset (tag frame skipped past), junk skip to
    /// two consecutive matching frames, codecpar from the first header.
    fn read_header(&mut self, io: &mut IoContext) -> Result<Stream> {
        let head = io.peek(10)?;
        if head.len() >= 10 && id3::id3v2_match(&head) {
            id3::skip_id3v2(io)?;
        }

        let mut st = Stream::new_audio(0);

        // mp3_parse_vbr_tags first (mp3_read_header 481-482): C parses
        // the tag off the frame at the post-ID3 offset BEFORE the junk
        // scan; on failure it seeks back and the frame demuxes normally.
        let base = io.tell();
        if !self.parse_vbr_tags(io, &mut st, base)? {
            io.seek(base)?;
        }

        // Junk skip (483-500): find i where headers at off+i and
        // off+i+frame_size mask-match.
        let base = io.tell();
        let mut found = None;
        for i in 0..64 * 1024u64 {
            match check_at(io, base + i) {
                Ok((header, fs)) => match check_at(io, base + i + fs as u64) {
                    Ok((h2, _)) if (header & MP3_MASK) == (h2 & MP3_MASK) => {
                        found = Some((base + i, header));
                        break;
                    }
                    _ => continue,
                },
                Err(Error::Eof) => break,
                Err(_) => continue,
            }
        }
        let (off, header) = found.ok_or_else(|| {
            Error::InvalidData("Failed to find two consecutive MPEG audio frames.".into())
        })?;
        io.seek(off)?;

        let mut h = MpaDecodeHeader::default();
        let _free = avpriv_mpegaudio_decode_header(&mut h, header);
        self.header = header;
        self.frame_samples = if h.lsf != 0 { 576 } else { 1152 };
        self.sample_rate = h.sample_rate;
        self.started = true;

        // codecpar from the header (C: the parser fills these later; our
        // find_stream_info wants them immediately). These overwrite the
        // same fields parse_vbr_tags set — same stream, same values,
        // except bit_rate, which the tag computes more precisely for VBR.
        let tag_bit_rate = st.codecpar.bit_rate;
        st.codecpar.codec_id = match h.layer {
            1 => CodecId::Mp1,
            2 => CodecId::Mp2,
            _ => CodecId::Mp3,
        };
        st.codecpar.sample_rate = h.sample_rate;
        st.codecpar.ch_layout = if h.nb_channels == 1 {
            crate::util::channel_layout::ChannelLayout::MONO
        } else {
            crate::util::channel_layout::ChannelLayout::STEREO
        };
        st.codecpar.sample_fmt = crate::util::samplefmt::SampleFormat::Fltp;
        st.codecpar.bit_rate = if tag_bit_rate != 0 {
            tag_bit_rate
        } else {
            h.bit_rate as i64
        };
        st.set_pts_info(1, h.sample_rate as i64);
        if st.start_time == crate::NOPTS {
            st.start_time = 0;
        }
        Ok(st)
    }

    /// One whole MPEG-audio frame per packet (mp3_read_packet + the
    /// mpegaudio parser's split, fused). Resync on garbage: scan forward
    /// for a header matching `MP3_MASK` (parser.c's ff_mpa_resync).
    fn read_packet(&mut self, io: &mut IoContext) -> Result<Packet> {
        loop {
            let pos = io.tell();
            let mut hb = [0u8; 4];
            if id3::read_full(io, &mut hb).is_err() {
                return Err(Error::Eof);
            }
            let header = u32::from_be_bytes(hb);
            let mut h = MpaDecodeHeader::default();
            if ff_mpa_check_header(header).is_err()
                || matches!(avpriv_mpegaudio_decode_header(&mut h, header), Ok(true))
                || (self.started && (header & MP3_MASK) != (self.header & MP3_MASK))
            {
                // Not a frame of our stream: advance one byte and rescan.
                io.seek(pos + 1)?;
                continue;
            }
            let fs = (h.frame_size as usize).min(MPA_MAX_CODED_FRAME_SIZE + 512);
            let mut data = vec![0u8; fs];
            data[..4].copy_from_slice(&hb);
            if id3::read_full(io, &mut data[4..]).is_err() {
                return Err(Error::Eof);
            }
            let mut pkt = Packet::from_vec(data);
            pkt.pts = self.next_pts;
            pkt.duration = self.frame_samples;
            pkt.time_base = Rational::new(1, self.sample_rate);
            pkt.flags = PacketFlags::KEY;
            self.next_pts += self.frame_samples;
            return Ok(pkt);
        }
    }
}

// ID3v2 helpers moved to the shared `super::id3` module (id3v2.c is
// shared by every raw-audio demuxer in C too).

/// Read what `io` will give without failing at EOF — the tag-frame
/// buffer's short-read shape (a truncated frame parses as zeros, as it
/// does through C's avio).
fn read_upto(io: &mut IoContext, buf: &mut [u8]) {
    let mut got = 0;
    while got < buf.len() {
        match io.read(&mut buf[got..]) {
            Ok(0) | Err(_) => break,
            Ok(n) => got += n,
        }
    }
}

/// `avio_rb32` over an in-memory frame; out-of-range reads 0 (past-EOF).
fn be32(b: &[u8], o: usize) -> u32 {
    if o + 4 <= b.len() {
        u32::from_be_bytes(b[o..o + 4].try_into().unwrap())
    } else {
        0
    }
}

/// `avio_rb24`.
fn be24(b: &[u8], o: usize) -> u32 {
    if o + 3 <= b.len() {
        ((b[o] as u32) << 16) | ((b[o + 1] as u32) << 8) | b[o + 2] as u32
    } else {
        0
    }
}

/// `avio_rb16`.
fn be16(b: &[u8], o: usize) -> u16 {
    if o + 2 <= b.len() {
        u16::from_be_bytes([b[o], b[o + 1]])
    } else {
        0
    }
}

/// `check()` (mp3dec.c:536-559): header at `pos` → (header, frame size),
/// Err(Eof) past the end, Err(_) for a non-frame.
fn check_at(io: &mut IoContext, pos: u64) -> Result<(u32, usize)> {
    io.seek(pos)?;
    let mut hb = [0u8; 4];
    id3::read_full(io, &mut hb)?;
    let header = u32::from_be_bytes(hb);
    ff_mpa_check_header(header)?;
    let mut h = MpaDecodeHeader::default();
    if avpriv_mpegaudio_decode_header(&mut h, header)? {
        return Err(Error::InvalidData("free-format frame".into()));
    }
    Ok((header, h.frame_size as usize))
}

/// `mp3_read_probe` (mp3dec.c:70-137): scan for runs of consecutive
/// valid frame headers; score by run length. The header-emulation guard
/// (positions inside the frame that mask-match the header) is simplified
/// to a length-only check — the guard exists to reject MPEG *video*
/// streams, which none of this port's other demuxers produce.
pub fn probe(buf: &[u8]) -> u32 {
    if buf.len() < 4 {
        return 0;
    }
    let end = buf.len().saturating_sub(4);
    let mut buf0 = 0;
    while buf0 < end && buf[buf0] == 0 {
        buf0 += 1;
    }
    let mut max_frames = 0usize;
    let mut max_framesizes = 0usize;
    let mut first_frames = 0usize;
    let mut b = buf0;
    while b < end {
        let mut b2 = b;
        let mut frames = 0usize;
        let mut framesizes = 0usize;
        while b2 < end {
            let header = u32::from_be_bytes([buf[b2], buf[b2 + 1], buf[b2 + 2], buf[b2 + 3]]);
            let mut h = MpaDecodeHeader::default();
            if !matches!(avpriv_mpegaudio_decode_header(&mut h, header), Ok(false)) {
                break;
            }
            framesizes += h.frame_size as usize;
            frames += 1;
            if (h.frame_size as usize) > end - b2 {
                break; // frame would run past the buffer
            }
            b2 += h.frame_size as usize;
        }
        if b == buf0 {
            first_frames = frames;
        }
        max_frames = max_frames.max(frames);
        max_framesizes = max_framesizes.max(framesizes);
        b += 1;
    }
    // Score tiers (mp3dec.c:127-137); the registry's >50 threshold makes
    // the extension+2 tier the one that actually wins.
    if first_frames >= 7 {
        52
    } else if max_frames > 200 && buf.len() < 2 * max_framesizes {
        50
    } else if max_frames >= 4 && buf.len() < 2 * max_framesizes {
        25
    } else if id3::id3v2_match(&buf[buf0.min(buf.len() - 10)..]) {
        13
    } else if max_frames >= 1 && buf.len() < 10 * max_framesizes {
        1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        codec::{audio::mp3::Mp3Decoder, traits::AudioDecoder},
        format::testutil::MemHandler,
    };

    /// Build a minimal valid MPEG-1 L3 mono 44100 32k frame: header +
    /// side info (17 bytes mono) + main data, padded to frame_size (104).
    fn frame() -> Vec<u8> {
        let mut f = Vec::new();
        // 0xFFFB 1000: MPEG1 L3, no CRC, 32 kbps (idx 1 → 104-byte frames),
        // 44100, no pad, mono.
        f.extend_from_slice(&[0xFF, 0xFB, 0x10, 0x00]);
        // side info: mdb=0, private=0, scfsi=0, 2 granules of zeros.
        f.extend_from_slice(&[0u8; 17]);
        f.resize(104, 0);
        f
    }

    fn file() -> Vec<u8> {
        let mut v = Vec::new();
        for _ in 0..8 {
            v.extend_from_slice(&frame());
        }
        v
    }

    /// A 128 kbps MPEG-1 L3 mono frame (417 bytes) whose main data is a
    /// full LAME `Info` tag — the metadata frame LAME writes as frame 0.
    /// `mp3_parse_info_tag`'s walk, byte for byte (mono ⇒ tag at 4+17).
    fn tag_frame(frames_field: u32, start_pad: u32, end_pad: u32) -> Vec<u8> {
        let mut f = vec![0u8; 417];
        f[..4].copy_from_slice(&[0xFF, 0xFB, 0x90, 0xC0]); // MPEG1 L3 128k 44.1k mono
        let put32 = |f: &mut Vec<u8>, o: usize, v: u32| {
            f[o..o + 4].copy_from_slice(&v.to_be_bytes());
        };
        f[21..25].copy_from_slice(b"Info"); // xing_off = 4+17
        put32(&mut f, 25, 0xF); // frames | bytes | TOC | quality
        put32(&mut f, 29, frames_field);
        put32(&mut f, 33, 417 * (frames_field + 1)); // bytes
        f[141..150].copy_from_slice(b"LAME99999"); // encoder version (9 bytes)
        // rev(150) lowpass(151) peak(152) rgains(156..160) flags(160) abr(161)
        let v = (start_pad << 12) | end_pad;
        f[162..165].copy_from_slice(&[(v >> 16) as u8, (v >> 8) as u8, v as u8]);
        f
    }

    #[test]
    fn probe_scores_multi_frame_buffer() {
        assert!(probe(&file()) >= 50, "8 clean frames ⇒ extension score");
        assert_eq!(probe(b"not an mp3 at all"), 0);
    }

    #[test]
    fn id3v2_length_syncsafe() {
        let tag = vec![b'I', b'D', b'3', 4, 0, 0, 0, 0, 0, 5];
        assert_eq!(id3::id3v2_tag_len(&tag), 15);
    }

    #[test]
    fn header_skips_junk_and_fills_stream() {
        let mut data = vec![0x7Fu8; 33];
        data.extend_from_slice(&file());
        let mut io = MemHandler::io(&data);
        let mut d = Mp3Demuxer::new();
        let st = d.read_header(&mut io).unwrap();
        assert_eq!(st.codecpar.codec_id, CodecId::Mp3);
        assert_eq!(st.codecpar.sample_rate, 44100);
        assert_eq!(st.time_base, Rational::new(1, 44100));
        assert_eq!(io.tell(), 33, "junk skipped");
    }

    #[test]
    fn id3v2_tag_is_skipped() {
        let mut data = vec![b'I', b'D', b'3', 4, 0, 0, 0, 0, 0, 5];
        data.extend_from_slice(&[0u8; 5]); // tag body
        data.extend_from_slice(&file());
        let mut io = MemHandler::io(&data);
        let mut d = Mp3Demuxer::new();
        d.read_header(&mut io).unwrap();
        assert_eq!(io.tell(), 15, "positioned after the tag");
    }

    #[test]
    fn packets_are_whole_frames_with_pts() {
        let mut io = MemHandler::io(&file());
        let mut d = Mp3Demuxer::new();
        d.read_header(&mut io).unwrap();
        let p0 = d.read_packet(&mut io).unwrap();
        assert_eq!(p0.size(), 104);
        assert_eq!(p0.pts, 0);
        assert_eq!(p0.duration, 1152);
        let p1 = d.read_packet(&mut io).unwrap();
        assert_eq!(p1.pts, 1152);
        let mut seen = 2;
        while d.read_packet(&mut io).is_ok() {
            seen += 1;
        }
        assert_eq!(seen, 8);
        assert!(matches!(d.read_packet(&mut io), Err(Error::Eof)));
    }

    #[test]
    fn garbage_between_frames_resyncs() {
        let mut data = file();
        data.extend_from_slice(&[0xAA; 40]);
        data.extend_from_slice(&frame());
        let mut io = MemHandler::io(&data);
        let mut d = Mp3Demuxer::new();
        d.read_header(&mut io).unwrap();
        let mut seen = 0;
        while let Ok(p) = d.read_packet(&mut io) {
            assert_eq!(p.size(), 104);
            seen += 1;
        }
        assert_eq!(seen, 9, "all frames recovered across the garbage");
    }

    #[test]
    fn two_consecutive_required() {
        let data = frame();
        let mut io = MemHandler::io(&data);
        let mut d = Mp3Demuxer::new();
        let err = d.read_header(&mut io).unwrap_err();
        assert!(err.to_string().contains("two consecutive"));
    }

    /// `mp3_parse_vbr_tags` + `mp3_parse_info_tag` (mp3dec.c:383-441): the
    /// tag frame is skipped, gapless lands on the stream, duration and
    /// start_time reflect the trimmed audio.
    #[test]
    fn vbr_tag_frame_skipped_and_gapless_exported() {
        let mut data = tag_frame(8, 576, 756);
        for _ in 0..8 {
            data.extend_from_slice(&frame());
        }
        let mut io = MemHandler::io(&data);
        let mut d = Mp3Demuxer::new();
        let st = d.read_header(&mut io).unwrap();
        assert_eq!(st.start_skip_samples, 576 + 528 + 1);
        assert_eq!(st.first_discard_sample, -756 + 528 + 1 + 8 * 1152);
        assert_eq!(st.last_discard_sample, 8 * 1152);
        assert_eq!(st.duration, 8 * 1152 - 576 - 756);
        assert_eq!(st.start_time, 576 + 528 + 1, "timeline shifted by the skip");
        assert_eq!(io.tell(), 417, "tag frame skipped past");
        let p0 = d.read_packet(&mut io).unwrap();
        assert_eq!(p0.pts, 0, "first audio frame restarts the timeline");
    }

    /// The demux.c gapless application (read_frame_internal 1536-1557):
    /// the stream fields become packet `skip_samples` (first packet) and
    /// `discard_padding` (the packet that crosses `first_discard_sample`).
    #[test]
    fn read_frame_injects_skip_and_discard() {
        let mut data = tag_frame(8, 576, 756);
        for _ in 0..8 {
            data.extend_from_slice(&frame());
        }
        let path = std::env::temp_dir().join("ffmpeg_rs_mp3_gapless_test.mp3");
        std::fs::write(&path, &data).unwrap();
        let mut ictx = crate::format::InputFormatContext::open(
            path.to_str().unwrap(),
            None,
            &DemuxOptions::default(),
        )
        .unwrap();
        ictx.find_stream_info().unwrap();

        let mut n = 0usize;
        let mut first_skip = None;
        let mut discards = Vec::new();
        while let Ok(p) = ictx.read_frame() {
            if n == 0 {
                first_skip = Some(p.skip_samples);
            }
            discards.push(p.discard_padding);
            n += 1;
        }
        assert_eq!(n, 8, "the tag frame itself never becomes a packet");
        assert_eq!(first_skip, Some(576 + 528 + 1));
        // Only the last packet crosses first_discard_sample
        // (-end_pad + 529 + 8·1152); it loses end_pad - 529 samples.
        assert_eq!(&discards[..7], &[0u32; 7]);
        assert_eq!(discards[7], 756 - 528 - 1);
    }

    /// End-to-end: packets through Mp3Decoder produce FLTP frames.
    #[test]
    fn packets_decode_through_mp3decoder() {
        let mut io = MemHandler::io(&file());
        let mut d = Mp3Demuxer::new();
        let st = d.read_header(&mut io).unwrap();
        let mut dec = Mp3Decoder::new();
        dec.init(&st.codecpar).unwrap();
        let mut frames = 0;
        while let Ok(p) = d.read_packet(&mut io) {
            dec.send_packet(Some(&p)).unwrap();
            while let Ok(f) = dec.receive_frame() {
                assert_eq!(f.nb_samples, 1152);
                frames += 1;
            }
        }
        assert_eq!(
            frames, 8,
            "one 1152-sample frame per packet (2 granules inside)"
        );
    }
}
