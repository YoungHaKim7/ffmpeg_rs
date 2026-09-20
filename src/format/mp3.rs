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
//! | Xing/VBRI (`mp3_parse_vbr_tags`, 383-441) | not ported (gapless/duration metadata; documented) |
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
    io::IoContext,
};

/// `MP3_MASK` (mp3dec.c:38): the header bits that must MATCH between
/// consecutive frames of one stream (everything except bitrate/padding/
/// private/copyright/original/emphasis).
const MP3_MASK: u32 = 0xfffe_0cc0;

/// `MPA_MAX_CODED_FRAME_SIZE` (mpegaudio.h): free-format read cap.
const MPA_MAX_CODED_FRAME_SIZE: usize = 2880;

/// ID3v2 magic `ff_id3v2_match` (id3v2.h/c): "ID3" + version ≠ 0xff.
fn id3v2_match(buf: &[u8]) -> bool {
    buf.len() >= 10
        && buf[0] == b'I'
        && buf[1] == b'D'
        && buf[2] == b'3'
        && buf[3] != 0xff
        && buf[4] != 0xff
}

/// `get_size` (id3v2.c:210-217): 7-bit-per-byte syncsafe size.
fn id3v2_tag_len(buf: &[u8]) -> usize {
    ((buf[6] as usize & 0x7f) << 21)
        | ((buf[7] as usize & 0x7f) << 14)
        | ((buf[8] as usize & 0x7f) << 7)
        | (buf[9] as usize & 0x7f) + 10
}

/// `ff_id3v2_skip` shape: consume the whole tag (header + body + footer).
/// Returns the total tag length.
fn skip_id3v2(io: &mut IoContext) -> Result<usize> {
    let mut head = [0u8; 10];
    read_full(io, &mut head)?;
    let len = id3v2_tag_len(&head);
    let footer = if head[5] & 0x10 != 0 { 10 } else { 0 };
    io.seek(io.tell() + (len - 10 + footer) as u64)?;
    Ok(len + footer)
}

/// `io.read` until the buffer is full; Err(Eof) if it runs dry mid-way
/// (C's avio_read + short-check).
fn read_full(io: &mut IoContext, buf: &mut [u8]) -> Result<()> {
    let mut got = 0;
    while got < buf.len() {
        match io.read(&mut buf[got..])? {
            0 => return Err(Error::Eof),
            n => got += n,
        }
    }
    Ok(())
}

/// `check()` (mp3dec.c:536-559): header at `pos` → (header, frame size),
/// Err(Eof) past the end, Err(_) for a non-frame.
fn check_at(io: &mut IoContext, pos: u64) -> Result<(u32, usize)> {
    io.seek(pos)?;
    let mut hb = [0u8; 4];
    read_full(io, &mut hb)?;
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
    } else if id3v2_match(&buf[buf0.min(buf.len() - 10)..]) {
        13
    } else if max_frames >= 1 && buf.len() < 10 * max_framesizes {
        1
    } else {
        0
    }
}

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
}

impl Demuxer for Mp3Demuxer {
    /// `mp3_read_header` (442-517): ID3v2 skip, junk skip to two
    /// consecutive matching frames, codecpar from the first header.
    fn read_header(&mut self, io: &mut IoContext) -> Result<Stream> {
        let head = io.peek(10)?;
        if head.len() >= 10 && id3v2_match(&head) {
            skip_id3v2(io)?;
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
        // find_stream_info wants them immediately).
        let mut st = Stream::new_audio(0);
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
        st.codecpar.bit_rate = h.bit_rate as i64;
        st.set_pts_info(1, h.sample_rate as i64);
        st.start_time = 0;
        Ok(st)
    }

    /// One whole MPEG-audio frame per packet (mp3_read_packet + the
    /// mpegaudio parser's split, fused). Resync on garbage: scan forward
    /// for a header matching `MP3_MASK` (parser.c's ff_mpa_resync).
    fn read_packet(&mut self, io: &mut IoContext) -> Result<Packet> {
        loop {
            let pos = io.tell();
            let mut hb = [0u8; 4];
            if read_full(io, &mut hb).is_err() {
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
            if read_full(io, &mut data[4..]).is_err() {
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

/// The registry row (`ff_mp3_demuxer`).
pub static MP3_INPUT_FORMAT: InputFormat = InputFormat {
    name: "mp3",
    long_name: "MP3 (MPEG audio layer 3)",
    extensions: &["mp3"],
    probe: Some(probe),
    make: |_opts: &DemuxOptions| Box::new(Mp3Demuxer::new()),
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::audio::mp3::Mp3Decoder;
    use crate::codec::traits::AudioDecoder;
    use crate::format::testutil::MemHandler;

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

    #[test]
    fn probe_scores_multi_frame_buffer() {
        assert!(probe(&file()) >= 50, "8 clean frames ⇒ extension score");
        assert_eq!(probe(b"not an mp3 at all"), 0);
    }

    #[test]
    fn id3v2_length_syncsafe() {
        let tag = vec![b'I', b'D', b'3', 4, 0, 0, 0, 0, 0, 5];
        assert_eq!(id3v2_tag_len(&tag), 15);
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
