//! raw ADTS AAC demuxer — port of `libavformat/aacdec.c`.
//!
//! C's demuxer emits the WHOLE ADTS frame (7-byte header included) per
//! packet and lets the AAC parser re-split downstream; the port keeps the
//! packet shape (the decoder's `aac_decode_frame_int` parses the inline
//! ADTS header itself, exactly as C's does when handed a full frame).
//!
//! ## C → Rust map
//!
//! | C (`libavformat/aacdec.c`) | here |
//! |---|---|
//! | `adts_aac_probe` (35-68) | [`probe`] — the frame-streak scan, four score tiers |
//! | ID3v2 auto-skip (`FF_INFMT_FLAG_ID3V2_AUTO`) + `handle_id3` (81-115) | shared [`super::id3`] helpers; tags between frames are skipped in [`Mp3StyleDemuxer::read_packet`] |
//! | `adts_aac_resync` (70-92) | the 0xFFF state scan in [`AdtsDemuxer::read_packet`] |
//! | `adts_aac_read_header` (94-141) | [`AdtsDemuxer::read_header`] — resync + first-frame peek for codecpar (C leaves that to find_stream_info's parser; the port wants the fields now) |
//! | `adts_aac_read_packet` (117-174) | [`AdtsDemuxer::read_packet`] — one ADTS frame per packet |
//! | ID3v1 / APE tag reads | not ported |
//!
//! Timestamps: C sets tb = 1/28224000 (LCM of AAC rates) and lets the
//! parser stamp; the port uses tb = 1/sample_rate from the first frame's
//! header, pts advancing by 1024·num_raw_data_blocks per packet.

use crate::{
    codec::{
        packet::{Packet, PacketFlags},
        params::CodecId,
    },
    util::{
        channel_layout::ChannelLayout,
        error::{Error, Result},
        rational::Rational,
    },
};

use super::{
    id3,
    demux::{DemuxOptions, Demuxer, InputFormat},
    io::IoContext,
    Stream,
};

/// `ADTS_HEADER_SIZE` (aacdec.c:33).
const ADTS_HEADER_SIZE: usize = 7;

/// `ff_mpeg4audio_sample_rates` (mpeg4audio_sample_rates.h:30) — the
/// sampling_index → rate table (index 12 is the last valid rate).
const SAMPLE_RATES: [i32; 13] = [
    96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350,
];

/// The registry row (`ff_aac_demuxer`).
pub static AAC_INPUT_FORMAT: InputFormat = InputFormat {
    name: "aac",
    long_name: "raw ADTS AAC (Advanced Audio Coding)",
    extensions: &["aac"],
    probe: Some(probe),
    make: |_opts: &DemuxOptions| Box::new(AdtsDemuxer::new()),
};

/// The demuxer state: the running sample clock from the first frame's
/// header.
pub struct AdtsDemuxer {
    sample_rate: i32,
    /// Next packet's pts, in 1/sample_rate units.
    next_pts: i64,
}

impl Default for AdtsDemuxer {
    fn default() -> Self {
        Self::new()
    }
}

/// The 7-byte ADTS header fields the demuxer reads
/// (`ff_adts_header_parse`, adts_header.c:33-73 — demuxer subset).
struct AdtsHeaderInfo {
    frame_length: usize,
    sampling_index: usize,
    sample_rate: i32,
    chan_config: u8,
    /// number_of_raw_data_blocks + 1.
    num_rdb: u32,
}

fn parse_adts(data: &[u8]) -> Result<AdtsHeaderInfo> {
    if data.len() < ADTS_HEADER_SIZE {
        return Err(Error::InvalidData("short ADTS header".into()));
    }
    if (u16::from_be_bytes([data[0], data[1]]) >> 4) != 0xfff {
        return Err(Error::InvalidData("ADTS syncword missing".into()));
    }
    let frame_length = ((u32::from_be_bytes([data[3], data[4], data[5], data[6]]) >> 13)
        & 0x1fff) as usize;
    if frame_length < ADTS_HEADER_SIZE {
        return Err(Error::InvalidData("ADTS frame too short".into()));
    }
    let sampling_index = ((data[2] >> 2) & 0xf) as usize;
    if sampling_index >= SAMPLE_RATES.len() || SAMPLE_RATES[sampling_index] == 0 {
        return Err(Error::InvalidData("invalid ADTS sampling index".into()));
    }
    Ok(AdtsHeaderInfo {
        frame_length,
        sampling_index,
        sample_rate: SAMPLE_RATES[sampling_index],
        chan_config: ((data[2] & 1) << 2) | (data[3] >> 6),
        num_rdb: (data[6] & 3) + 1,
    })
}

impl AdtsDemuxer {
    pub fn new() -> Self {
        AdtsDemuxer {
            sample_rate: 0,
            next_pts: 0,
        }
    }

    /// `adts_aac_resync` (aacdec.c:70-92): scan forward byte-by-byte until
    /// two consecutive bytes read as a syncword prefix.
    fn resync(io: &mut IoContext) -> Result<()> {
        let mut state = 0u16;
        loop {
            let mut b = [0u8; 1];
            match io.read(&mut b) {
                Ok(0) | Err(_) => return Err(Error::Eof),
                Ok(_) => {}
            }
            state = (state << 8) | b[0] as u16;
            if state >> 4 == 0xfff {
                io.seek(io.tell() - 2)?;
                return Ok(());
            }
        }
    }
}

impl Demuxer for AdtsDemuxer {
    /// `adts_aac_read_header`: ID3v2 skip, resync to the first ADTS
    /// frame, codecpar from its header.
    fn read_header(&mut self, io: &mut IoContext) -> Result<Stream> {
        let head = io.peek(10)?;
        if head.len() >= 10 && id3::id3v2_match(&head) {
            id3::skip_id3v2(io)?;
        }
        Self::resync(io)?;

        // Peek the first frame's header (C: codecpar is filled by
        // find_stream_info's parser pass; the port reads it here).
        let mut hdr = [0u8; ADTS_HEADER_SIZE];
        id3::read_full(io, &mut hdr)?;
        let info = parse_adts(&hdr)?;
        io.seek(io.tell() - ADTS_HEADER_SIZE as u64)?;

        self.sample_rate = info.sample_rate;
        self.next_pts = 0;

        // ff_mpeg4audio_channels inverse (mpeg4audio.c): counts for the
        // default configurations.
        let channels: u8 = match info.chan_config {
            1..=6 => info.chan_config,
            7 => 8,
            _ => 0,
        };
        let mut st = Stream::new_audio(0);
        st.codecpar.codec_id = CodecId::Aac;
        st.codecpar.sample_rate = info.sample_rate;
        st.codecpar.ch_layout = match info.chan_config {
            1 => ChannelLayout::MONO,
            2 => ChannelLayout::STEREO,
            c if (1..=7).contains(&c) => ChannelLayout::unspecified(channels),
            _ => ChannelLayout::default(),
        };
        st.set_pts_info(1, info.sample_rate as i64);
        st.start_time = 0;
        Ok(st)
    }

    /// `adts_aac_read_packet`: whole ADTS frames; ID3v2 tags between
    /// frames are consumed (contents dropped, like C's handle_id3) and
    /// the read retried.
    fn read_packet(&mut self, io: &mut IoContext) -> Result<Packet> {
        loop {
            let pos = io.tell();
            let mut hdr = [0u8; ADTS_HEADER_SIZE];
            if id3::read_full(io, &mut hdr).is_err() {
                return Err(Error::Eof);
            }
            if (u16::from_be_bytes([hdr[0], hdr[1]]) >> 4) != 0xfff {
                // Not sync: ID3v2 between frames, or junk → resync.
                let mut tag = hdr.to_vec();
                let mut more = [0u8; 3];
                if id3::read_full(io, &mut more).is_ok() {
                    tag.extend_from_slice(&more);
                }
                if id3::id3v2_match(&tag) {
                    let len = id3::id3v2_tag_len(&tag);
                    if len > tag.len() {
                        io.seek(io.tell() + (len - tag.len()) as u64)?;
                    }
                } else {
                    io.seek(pos + 1)?;
                    Self::resync(io)?;
                }
                continue;
            }
            let info = parse_adts(&hdr)?;
            let mut data = vec![0u8; info.frame_length];
            data[..ADTS_HEADER_SIZE].copy_from_slice(&hdr);
            if id3::read_full(io, &mut data[ADTS_HEADER_SIZE..]).is_err() {
                return Err(Error::Eof);
            }
            let mut pkt = Packet::from_vec(data);
            pkt.pts = self.next_pts;
            pkt.duration = 1024 * info.num_rdb as i64;
            pkt.time_base = Rational::new(1, self.sample_rate.max(1) as i64);
            pkt.flags = PacketFlags::KEY;
            self.next_pts += pkt.duration;
            return Ok(pkt);
        }
    }
}

/// `adts_aac_probe` (aacdec.c:35-68): count runs of consecutive ADTS
/// frames (sync + plausible frame length chaining), score by run length.
pub fn probe(buf: &[u8]) -> u32 {
    let end = buf.len().saturating_sub(7);
    let mut max_frames = 0usize;
    let mut first_frames = 0usize;
    let mut start = 0usize;
    loop {
        if start >= end {
            break;
        }
        let mut b2 = start;
        let mut frames = 0usize;
        while b2 < end {
            let header = u16::from_be_bytes([buf[b2], buf[b2 + 1]]);
            if (header & 0xfff6) != 0xfff0 {
                if start != 0 {
                    // False positive mid-buffer: discard the run.
                    frames = 0;
                }
                break;
            }
            let fsize = ((u32::from_be_bytes([
                buf[b2 + 3],
                buf[b2 + 4],
                buf[b2 + 5],
                buf[b2 + 6],
            ]) >> 13) & 0x1fff) as usize;
            if fsize < ADTS_HEADER_SIZE {
                break;
            }
            let fsize = fsize.min(end - b2);
            b2 += fsize;
            frames += 1;
        }
        if start == 0 {
            first_frames = frames;
        }
        max_frames = max_frames.max(frames);
        if b2 >= end {
            break;
        }
        start = b2 + 1;
    }
    if first_frames >= 3 {
        51 // AVPROBE_SCORE_EXTENSION + 1
    } else if max_frames > 100 {
        50 // AVPROBE_SCORE_EXTENSION
    } else if max_frames >= 3 {
        25 // AVPROBE_SCORE_EXTENSION / 2
    } else if first_frames >= 1 {
        1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::testutil::MemHandler;

    /// Build a synthetic ADTS frame: 7-byte header (MPEG-4 AAC-LC,
    /// 44.1 kHz, stereo) + `payload` bytes.
    fn frame(payload: usize, chan_config: u8) -> Vec<u8> {
        let fsize = ADTS_HEADER_SIZE + payload;
        let mut f = Vec::with_capacity(fsize);
        f.push(0xff);
        f.push(0xf1); // sync, MPEG-4, layer 0, no CRC
        f.push((1 << 6) | (4 << 2) | 0); // profile LC(1), sf_index 4, no private
        // chan_config occupies stream bits 23..25 → byte2 bit0 + byte3 bits 7..6
        let b2_low = (chan_config >> 2) & 1;
        let b3_high = (chan_config & 3) << 6;
        f[2] |= b2_low;
        let fl = fsize as u32;
        let b3 = b3_high | ((fl >> 11) as u8 & 3);
        f.push(b3);
        f.push((fl >> 3) as u8);
        f.push(((fl & 7) as u8) << 5);
        f.push(0x1f); // buffer fullness high + rdb-1 = 0
        f.resize(fsize, 0x55);
        f
    }

    fn file(n: usize) -> Vec<u8> {
        let mut v = Vec::new();
        for _ in 0..n {
            v.extend_from_slice(&frame(64, 2));
        }
        v
    }

    #[test]
    fn probe_scores_frame_runs() {
        assert!(probe(&file(8)) >= 50, "8 clean frames ⇒ extension score");
        assert_eq!(probe(b"not an aac file at all...."), 0);
    }

    #[test]
    fn header_fills_stream_and_positions() {
        let data = {
            let mut d = vec![b'I', b'D', b'3', 4, 0, 0, 0, 0, 0, 5];
            d.extend_from_slice(&[0u8; 5]);
            d.extend_from_slice(&file(4));
            d
        };
        let mut io = MemHandler::io(&data);
        let mut d = AdtsDemuxer::new();
        let st = d.read_header(&mut io).unwrap();
        assert_eq!(st.codecpar.codec_id, CodecId::Aac);
        assert_eq!(st.codecpar.sample_rate, 44100);
        assert_eq!(st.codecpar.ch_layout.nb_channels, 2);
        assert_eq!(st.time_base, Rational::new(1, 44100));
        assert_eq!(io.tell(), 15, "ID3 skipped, at first frame");
    }

    #[test]
    fn packets_are_whole_adts_frames() {
        let data = file(5);
        let mut io = MemHandler::io(&data);
        let mut d = AdtsDemuxer::new();
        d.read_header(&mut io).unwrap();
        let p0 = d.read_packet(&mut io).unwrap();
        assert_eq!(p0.size(), 7 + 64);
        assert_eq!(p0.pts, 0);
        assert_eq!(p0.duration, 1024);
        assert_eq!(p0.data[0], 0xff, "header included (decoder re-parses)");
        let p1 = d.read_packet(&mut io).unwrap();
        assert_eq!(p1.pts, 1024);
        let mut seen = 2;
        while d.read_packet(&mut io).is_ok() {
            seen += 1;
        }
        assert_eq!(seen, 5);
    }

    #[test]
    fn junk_between_frames_resyncs() {
        let mut data = file(3);
        let at = data.len();
        data.extend_from_slice(&[0xAA; 40]);
        data.extend_from_slice(&frame(64, 2));
        let mut io = MemHandler::io(&data);
        let mut d = AdtsDemuxer::new();
        d.read_header(&mut io).unwrap();
        let mut seen = 0;
        while let Ok(p) = d.read_packet(&mut io) {
            assert_eq!(p.size(), 7 + 64);
            seen += 1;
        }
        assert_eq!(seen, 4, "all frames recovered past the junk at {at}");
    }

    #[test]
    fn id3_between_frames_is_skipped() {
        let mut data = file(2);
        // A 10+5-byte ID3v2 tag between frames.
        data.extend_from_slice(&[b'I', b'D', b'3', 4, 0, 0, 0, 0, 0, 5]);
        data.extend_from_slice(&[0u8; 5]);
        data.extend_from_slice(&frame(64, 2));
        let mut io = MemHandler::io(&data);
        let mut d = AdtsDemuxer::new();
        d.read_header(&mut io).unwrap();
        let mut seen = 0;
        while let Ok(p) = d.read_packet(&mut io) {
            assert_eq!(p.data[0], 0xff);
            seen += 1;
        }
        assert_eq!(seen, 3, "tag consumed, frames on both sides kept");
    }
}
