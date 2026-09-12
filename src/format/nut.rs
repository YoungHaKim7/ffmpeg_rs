//! NUT demuxer — port of `libavformat/nutdec.c` plus the framing core the
//! container's two halves share (`nut.c`/`nut.h`, the `ffio_read_varlen`
//! vint of `aviobuf.c:919-928`, and the `AV_CRC_32_IEEE` table of
//! `libavutil/crc.c`). The write side (`nutenc.c`) lands next and reuses
//! [`put_v`]/[`put_s`]/[`put_packet`] and the CRC core from here.
//!
//! ## C → Rust map
//!
//! | C | here |
//! |---|---|
//! | startcodes + `Flag` enum (`nut.h:29-56`) | [`MAIN_STARTCODE`] … and the [`flag`] consts |
//! | `FrameCode` (`nut.h:65-73`) | [`FrameCode`] |
//! | `StreamContext` (`nut.h:75-85`) | [`StreamContext`] (`keyframe_pts` out — index only) |
//! | `NUTContext` (`nut.h:91-118`) | [`NutDemuxer`] (+ `stream_params`, see below) |
//! | `ffio_read_varlen` (`aviobuf.c:919-928`) | [`NutReader::get_v`] |
//! | `put_v`/`put_s` (`nutenc.c:306-348`) | [`put_v`]/[`put_s`] |
//! | `av_crc` + `av_crc_init` for `AV_CRC_32_IEEE` (`crc.c:341, 366-376, 421-452`) | [`CRC_TABLE`] + [`crc04c11db7_update`] |
//! | `ff_crc04C11DB7_update` (`aviobuf.c:568-572`) | [`crc04c11db7_update`] |
//! | `ffio_init_checksum`/`ffio_get_checksum` (`avio_internal.h`) | [`NutReader::init_checksum`]/[`NutReader::get_checksum`] |
//! | `get_str` (`nutdec.c:44-66`) | [`get_str`] |
//! | `get_s` (`nutdec.c:68-76`) | [`NutReader::get_s`] |
//! | `get_fourcc` (`nutdec.c:78-90`) | [`get_fourcc`] |
//! | `get_packetheader` (`nutdec.c:92-110`) | [`get_packetheader`] |
//! | `put_packet` (`nutenc.c:351-370`) | [`put_packet`] (Vec form; the muxer flushes the Vec to I/O) |
//! | `find_any_startcode`/`find_startcode` (`nutdec.c:112-153`) | [`find_any_startcode`]/[`find_startcode`] |
//! | `nut_probe` (`nutdec.c:155-166`) | [`probe`] |
//! | `GET_V` macro (`nutdec.c:168-177`) | [`get_v_check`] (same `"Error <dst> is (<v>)"` text) |
//! | `skip_reserved` (`nutdec.c:179-193`) | [`skip_reserved`] |
//! | `decode_main_header` (`nutdec.c:195-379`) | [`NutDemuxer::decode_main_header`] |
//! | `decode_stream_header` (`nutdec.c:381-488`) | [`NutDemuxer::decode_stream_header`] |
//! | `decode_info_header` (`nutdec.c:505-626`) | [`NutDemuxer::decode_info_header`] |
//! | `decode_syncpoint` (`nutdec.c:628-669`) | [`NutDemuxer::decode_syncpoint`] |
//! | `ff_nut_reset_ts` (`nut.c:266-275`) | [`NutDemuxer::reset_ts`] over [`rescale_rnd_down`] |
//! | `ff_lsb2full` (`nut.c:277-282`) | [`ff_lsb2full`] |
//! | `read_sm_data` (`nutdec.c:880-995`) | [`read_sm_data`] (parse-only, see divergences) |
//! | `decode_frame_header` (`nutdec.c:997-1078`) | [`NutDemuxer::decode_frame_header`] |
//! | `decode_frame` (`nutdec.c:1080-1145`) | [`NutDemuxer::decode_frame`] |
//! | `nut_read_header` (`nutdec.c:812-878`) | [`NutDemuxer::read_header`] |
//! | `nut_read_packet` (`nutdec.c:1147-1205`) | [`NutDemuxer::read_packet`] |
//! | `ff_nut_video_tags`/`ff_codec_bmp_tags` RAWVIDEO rows (`nut.c:44-220`, `riff.c`) | [`RAWVIDEO_TAGS`] |
//! | `ff_nut_audio_tags` + `ff_codec_wav_tags` + `ff_nut_audio_extra_tags` (`nut.c:222-264`, `riff.c`) | [`NUT_AUDIO_TAGS`] |
//!
//! ## The byte format, exactly as the demuxer walks it
//!
//! ```text
//! packet        := startcode(8, BE) [forward_crc(4, LE) if length>4096]
//!                  v: forward_ptr (= body + 4)   ─┬ CRC-32/IEEE-802.3 (poly
//!                  body                            │ 0x04C11DB7, MSB-first, no
//!                  crc32(body)(4, LE)            ─┘  reflect, init 0) — a
//! main header   := v: version [v: minor_version if v>3]  stored checksum is
//!                  v: stream_count  v: max_distance        valid exactly when
//!                  v: time_base_count { v:num v:den }      folding its own 4
//!                  frame-code rows (fill all 256, 'N'→invalid)  bytes into
//!                  [v: header_count-1 { v:len bytes } if more than 4 left]   the running
//!                  [v: flags if version>3 and more than 4 left]       CRC hits 0
//! stream header := v: stream_id  v: class  fourcc(v:2|4 + LE bytes)
//!                  v: time_base_id  v: msb_pts_shift  v: max_pts_distance
//!                  v: decode_delay  v: flags  v: extradata_len [bytes]
//!                  video: v:w v:h v:sar_num v:sar_den v:csp
//!                  audio: v:rate v:rate_den v:channels
//! syncpoint     := v: ts*tb_count+tb_id (put_tt)  v: back_ptr/16
//! frame         := frame_code byte [+coded flags xor] [+v:stream_id]
//!                  [+v:coded_pts] [+v:size_msb] [+s:match_time]
//!                  [+v:header_idx] [+v:reserved_count {v}] [+crc32(4, LE)]
//!                  [side/meta data] payload
//! ```
//!
//! ## Multi-stream degradation (port constraint)
//!
//! NUT is multi-stream by design; the port's [`Demuxer`] trait returns one
//! [`Stream`] and [`InputFormatContext`](super::InputFormatContext) holds
//! one. The demuxer therefore keeps C's *full* multi-stream header state
//! (all [`StreamContext`]s for per-stream pts bookkeeping, all learned
//! codec parameters in `stream_params`) but **exposes only stream 0**:
//!
//! * `read_header` returns stream 0's description (a 0-stream file is
//!   impossible — the main header's `GET_V(stream_count, tmp > 0 …)`
//!   rejects it with C's `"Error stream_count is (0)"` text).
//! * `read_packet` decodes frames of *every* stream — `decode_frame_header`
//!   must run for all of them to keep `last_pts` honest — then **drops the
//!   packets of streams ≠ 0** (C's `discard` path, `nutdec.c:1098-1107`,
//!   does the same when a stream is marked AVDISCARD_ALL; the port applies
//!   it unconditionally to the streams it cannot surface).
//!
//! ## Skipped C paths (documented divergences, with the C guard)
//!
//! | C path | Guard / reason |
//! |---|---|
//! | `find_and_decode_index` (`nutdec.c:684-796`), `nut_read_timestamp`, `read_seek`, `ff_nut_add_sp` syncpoint tree (`nut.c:296-333`) | seek-by-index not ported; C tolerates a missing index (`\"no index at the end\"`, `nutdec.c:702` is a warning) and continues — the port simply never looks, so `Stream::duration` stays `NOPTS` |
//! | metadata/chapters (`decode_info_header` storage, `set_disposition_bits`, `ff_nut_metadata_conv`) | no metadata dictionary in the port; the info packet is still parsed byte-exactly and checksum-verified |
//! | `r_frame_rate` from the info header (`nutdec.c:603-609`) | per-stream AVStream not kept beyond stream 0; dropped with the metadata |
//! | packet side data (`read_sm_data` output: palette/new-extradata/param-change/skip-samples) | `Packet` has no side-data fields; entries are parsed (byte-exact, warnings like C) and dropped |
//! | per-stream `discard` levels + `skip_until_key_frame` after seek (`nutdec.c:1098-1107`, `read_seek:1299-1300`) | no seeking, no per-stream discard policy in the port; `skip_until_key_frame` stays `false` (C's zero-init) so the discard branch never fires |
//! | `pkt->duration` | C's `decode_frame` leaves it 0 and the generic demux layer derives it from the *next* packet's dts; the port has no generic layer, so the frame code's `pts_delta` (the container's declared delta) is used |
//! | extradata storage (`ff_get_extradata`, `nutdec.c:447-453`) | `CodecParameters` has no extradata field; bytes are read (and checksummed) then dropped |
//! | `st->codecpar->video_delay` (`nutdec.c:444`) | no such field; the value is validated (< 1000) then dropped |
//! | codec tags outside the `CodecId` family (GIF/VP9/HEVC/MP3/Opus/… and RAWVIDEO tags whose pixel format the port cannot represent: RGB15/12, PAL8, Y3 11 9, G3 0 10, …) | `CodecId` has no variant; they resolve to `CodecId::None` with C's `\"Unknown codec tag\"` log. C reports the real codec id (RAWVIDEO with a pix fmt the port's decoder family lacks) — both sides fail identically at decode time |
//! | `PCM_S8` (`'P','S','D',8`, `nut.c:245`) and the `U16/U24/U32/S64/*_PLANAR` PCM tags | not in the `CodecId` family → `CodecId::None` |
//! | `get_fourcc` lengths ≠ 2/4 (`nutdec.c:87`) | C logs `\"Unsupported fourcc length\"` and continues with tag `-1`; port does the same (no error) |
//! | `AVERROR(ENOMEM)` early-out of the main-header retry loop (`nutdec.c:825-826`) | Rust allocation failure aborts; the loop retries on parse errors only, like C |
//! | ELI (elision) headers | fully ported (main header table + `FLAG_HEADER_IDX` + `size > 4096` reset, `nutdec.c:322-346, 1055-1061`) |

use crate::{
    NOPTS,
    codec::{
        packet::{Packet, PacketFlags},
        params::{CodecId, MediaType},
        pcm,
    },
    log_error, log_verbose, log_warning,
    util::{
        channel_layout::ChannelLayout,
        error::{Error, Result},
        pixfmt::PixelFormat,
        rational::Rational,
    },
};

use super::{
    Stream,
    demux::{Demuxer, PROBE_SCORE_MAX},
    io::IoContext,
};

// ---------------------------------------------------------------------
// nut.h:29-56 — startcodes, limits, frame flags
// ---------------------------------------------------------------------

/// `nut.h:29` — `0x7A561F5F04AD + ((uint64_t)('N'<<8) + 'M')<<48`.
pub const MAIN_STARTCODE: u64 =
    0x7A56_1F5F_04AD + ((((b'N' as u64) << 8) + (b'M' as u64)) << 48);
/// `nut.h:30` — `…('N'<<8) + 'S'…`.
pub const STREAM_STARTCODE: u64 =
    0x1140_5BF2_F9DB + ((((b'N' as u64) << 8) + (b'S' as u64)) << 48);
/// `nut.h:31` — `…('N'<<8) + 'K'…`.
pub const SYNCPOINT_STARTCODE: u64 =
    0xE4AD_EECA_4569 + ((((b'N' as u64) << 8) + (b'K' as u64)) << 48);
/// `nut.h:32` — `…('N'<<8) + 'X'…`.
pub const INDEX_STARTCODE: u64 =
    0xDD67_2F23_E64E + ((((b'N' as u64) << 8) + (b'X' as u64)) << 48);
/// `nut.h:33` — `…('N'<<8) + 'I'…`.
pub const INFO_STARTCODE: u64 =
    0xAB68_B596_BA78 + ((((b'N' as u64) << 8) + (b'I' as u64)) << 48);

/// `nut.h:37`.
pub const MAX_DISTANCE: u32 = 1024 * 32 - 1;
/// `nut.h:39-41`.
pub const NUT_MAX_VERSION: u64 = 4;
/// `nut.h:40`.
pub const NUT_STABLE_VERSION: u64 = 3;
/// `nut.h:41`.
pub const NUT_MIN_VERSION: u64 = 2;

/// `nut.h:43-56` (`Flag`) — frame-code / frame-header flags.
pub mod flag {
    /// if set, frame is keyframe
    pub const KEY: u32 = 1;
    /// if set, stream has no relevance on presentation. (EOR)
    pub const EOR: u32 = 2;
    /// if set, coded_pts is in the frame header
    pub const CODED_PTS: u32 = 8;
    /// if set, stream_id is coded in the frame header
    pub const STREAM_ID: u32 = 16;
    /// if set, data_size_msb is at frame header, otherwise data_size_msb is 0
    pub const SIZE_MSB: u32 = 32;
    /// if set, the frame header contains a checksum
    pub const CHECKSUM: u32 = 64;
    /// if set, reserved_count is coded in the frame header
    pub const RESERVED: u32 = 128;
    /// if set, side / meta data is stored in the frame header.
    pub const SM_DATA: u32 = 256;
    /// If set, header_idx is coded in the frame header.
    pub const HEADER_IDX: u32 = 1024;
    /// If set, match_time_delta is coded in the frame header
    pub const MATCH_TIME: u32 = 2048;
    /// if set, coded_flags are stored in the frame header
    pub const CODED: u32 = 4096;
    /// if set, frame_code is invalid
    pub const INVALID: u32 = 8192;
}

/// `NUT_BROADCAST` (`nut.h:113`) — bit 0 of the version-4 main-header flags.
pub const NUT_BROADCAST: u64 = 1;
/// `NUT_PIPE` (`nut.h:114`) — bit 1: no syncpoints/checksums in the stream.
pub const NUT_PIPE: u64 = 2;

/// `nut.h:65-73`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FrameCode {
    pub flags: u16,
    pub stream_id: u8,
    pub size_mul: u16,
    pub size_lsb: u16,
    pub pts_delta: i16,
    pub reserved_count: u8,
    pub header_idx: u8,
}

/// `nut.h:75-85` — per-stream demux state. `time_base` is C's borrowed
/// pointer; `None` doubles as C's NULL check that rejects a duplicate
/// stream header for the same id (`nutdec.c:393`). `Default` is C's
/// av_calloc: everything zero, no time base yet.
#[derive(Debug, Clone, Default)]
pub struct StreamContext {
    pub last_flags: u32,
    pub skip_until_key_frame: bool,
    pub last_pts: i64,
    pub time_base_id: usize,
    pub msb_pts_shift: u32,
    pub max_pts_distance: i64,
    pub decode_delay: u32,
    pub time_base: Option<Rational>,
}

// ---------------------------------------------------------------------
// libavutil/crc.c — the AV_CRC_32_IEEE table (poly 0x04C11DB7, MSB-first)
// ---------------------------------------------------------------------

/// `av_crc_init(ctx, le=0, bits=32, poly=0x04C11DB7)` (`crc.c:341` declared,
/// built at `crc.c:366-376`): the MSB-first table entry *stored
/// byteswapped*, which is the representation `av_crc`'s update loop
/// (`crc.c:449-452`, `crc = ctx[(uint8_t)crc ^ byte] ^ crc >> 8`) computes
/// with. FFmpeg's checksum values are `bswap32` of the textbook MSB-first
/// CRC — and `avio_wl32` of that value re-emits the textbook BE byte order,
/// which is why appending a stored checksum drives the running CRC to zero.
pub const CRC_TABLE: [u32; 256] = build_crc_table();

const fn build_crc_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        // crc.c:372-374: c = i << 24; 8× (c << 1) ^ (poly & (c >> 31 sign));
        let mut c = (i as u32) << 24;
        let mut j = 0;
        while j < 8 {
            let mask = if (c as i32) < 0 { 0x04C1_1DB7 } else { 0 };
            c = (c << 1) ^ mask;
            j += 1;
        }
        table[i] = c.swap_bytes(); // crc.c:375: ctx[i] = av_bswap32(c);
        i += 1;
    }
    table
}

/// `ff_crc04C11DB7_update` (`aviobuf.c:568-572`) = `av_crc(AV_CRC_32_IEEE)`
/// (`crc.c:421-452`) over one slice. Init value passed by the caller (0 for
/// fresh checksums, the startcode CRC for packet headers).
pub fn crc04c11db7_update(crc: u32, buf: &[u8]) -> u32 {
    let mut crc = crc;
    for &b in buf {
        crc = CRC_TABLE[((crc as u8) ^ b) as usize] ^ (crc >> 8); // crc.c:450
    }
    crc
}

// ---------------------------------------------------------------------
// vint / svint — nutenc.c:306-348 (put) + aviobuf.c:919-928 (get)
// ---------------------------------------------------------------------

/// `get_v_length` + `put_v` (`nutenc.c:306-327`): 7 bits per byte, most
/// significant group first; bit 7 set on every byte but the last.
pub fn put_v(out: &mut Vec<u8>, val: u64) {
    let mut i = 1usize; // get_v_length (nutenc.c:306-313)
    {
        let mut v = val;
        while {
            v >>= 7;
            v != 0
        } {
            i += 1;
        }
    }
    while {
        i -= 1;
        i > 0
    } {
        out.push(128 | (val >> (7 * i)) as u8); // nutenc.c:324
    }
    out.push((val & 127) as u8); // nutenc.c:326
}

/// `put_s` (`nutenc.c:345-348`): `put_v(2·|val| − (val > 0))`.
pub fn put_s(out: &mut Vec<u8>, val: i64) {
    let v = 2 * (val as i128).abs() - i128::from(val > 0);
    put_v(out, v as u64);
}

/// A checksum-aware reader — `AVIOContext` plus the `ffio_init_checksum`
/// state NUT layers over it (`avio_internal.h`). Every read method folds
/// consumed bytes into the running CRC while a checksum is active, exactly
/// as C's buffered reader does; zero-fill bytes at EOF are *not* folded in
/// (C's `checksum_ptr..buf_ptr` range never covers them).
pub struct NutReader<'a> {
    io: &'a mut IoContext,
    crc: u32,
    checksumming: bool,
}

impl<'a> NutReader<'a> {
    pub fn new(io: &'a mut IoContext) -> Self {
        NutReader {
            io,
            crc: 0,
            checksumming: false,
        }
    }

    /// The raw stream (for the few C paths that read with the checksum
    /// explicitly off — payload reads after `get_checksum`).
    pub fn io(&mut self) -> &mut IoContext {
        self.io
    }

    /// `avio_r8`.
    pub fn r8(&mut self) -> Result<u8> {
        let b = self.io.r8()?;
        if self.checksumming && !self.io.is_eof() {
            self.crc = crc04c11db7_update(self.crc, &[b]);
        }
        Ok(b)
    }

    /// `avio_read` — returns the short count at EOF.
    pub fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        let got = self.io.read(buf)?;
        if self.checksumming {
            self.crc = crc04c11db7_update(self.crc, &buf[..got]);
        }
        Ok(got)
    }

    /// `ffio_read_varlen` (`aviobuf.c:919-928`).
    pub fn get_v(&mut self) -> Result<u64> {
        let mut val = 0u64;
        loop {
            let tmp = self.r8()?;
            val = (val << 7).wrapping_add(u64::from(tmp & 127));
            if tmp & 128 == 0 {
                break;
            }
        }
        Ok(val)
    }

    /// `get_s` (`nutdec.c:68-76`) — zigzag-coded signed vint.
    pub fn get_s(&mut self) -> Result<i64> {
        let v = (self.get_v()? as i64).wrapping_add(1);
        if v & 1 != 0 {
            Ok(-(v >> 1))
        } else {
            Ok(v >> 1)
        }
    }

    /// `avio_rb16`.
    pub fn rb16(&mut self) -> Result<u16> {
        let mut b = [0u8; 2];
        for slot in &mut b {
            *slot = self.r8()?;
        }
        Ok(u16::from_be_bytes(b))
    }

    /// `avio_rb32`.
    pub fn rb32(&mut self) -> Result<u32> {
        let mut b = [0u8; 4];
        for slot in &mut b {
            *slot = self.r8()?;
        }
        Ok(u32::from_be_bytes(b))
    }

    /// `avio_rl16` (get_fourcc's 2-byte arm).
    pub fn rl16(&mut self) -> Result<u16> {
        let mut b = [0u8; 2];
        for slot in &mut b {
            *slot = self.r8()?;
        }
        Ok(u16::from_le_bytes(b))
    }

    /// `avio_rl32` (get_fourcc's 4-byte arm).
    pub fn rl32(&mut self) -> Result<u32> {
        let mut b = [0u8; 4];
        for slot in &mut b {
            *slot = self.r8()?;
        }
        Ok(u32::from_le_bytes(b))
    }

    /// `avio_tell`.
    pub fn tell(&self) -> u64 {
        self.io.tell()
    }

    /// `avio_seek` — never checksummed (C's seek flushes the checksum
    /// range; NUT never seeks with a checksum active).
    pub fn seek(&mut self, pos: u64) -> Result<()> {
        self.io.seek(pos)
    }

    /// `avio_skip` (read-out form). Not checksummed — C's skip is used
    /// only with the checksum off (packet bodies in the skip arm).
    pub fn skip(&mut self, n: u64) -> Result<()> {
        self.io.skip(n)
    }

    /// `pb->eof_reached`.
    pub fn is_eof(&self) -> bool {
        self.io.is_eof()
    }

    /// `ffio_init_checksum` (`checksum != 0` selects an active checksum —
    /// C passes a NULL update fn when the packet body is unchecked,
    /// `nutdec.c:107`).
    pub fn init_checksum(&mut self, on: bool, init: u32) {
        self.checksumming = on;
        self.crc = init;
    }

    /// `ffio_get_checksum` — returns the running CRC and deactivates it.
    pub fn get_checksum(&mut self) -> u32 {
        self.checksumming = false;
        self.crc
    }
}

// ---------------------------------------------------------------------
// nutdec.c:92-110 / nutenc.c:351-370 — the packet header, both directions
// ---------------------------------------------------------------------

/// `get_packetheader` (`nutdec.c:92-110`). Called with the 8 startcode
/// bytes already consumed: seeds the CRC over them, reads the forward
/// pointer (and the stored forward checksum when the pointer exceeds
/// 4096 — folding it in must zero the CRC), then arms (or explicitly
/// disarms, `calculate_checksum == false`) the body checksum. Returns the
/// forward pointer: body length *including* its trailing 4 checksum bytes.
pub fn get_packetheader(
    r: &mut NutReader,
    calculate_checksum: bool,
    startcode: u64,
) -> Result<u64> {
    // nutdec.c:97-98 — CRC over the startcode's 8 big-endian bytes (the
    // bytes as they lie in the file).
    let sc = crc04c11db7_update(0, &startcode.to_be_bytes());
    r.init_checksum(true, sc);

    let size = r.get_v()?;
    if size > 4096 {
        r.rb32()?; // the stored forward checksum, consumed through the CRC
    }
    let crc = r.get_checksum();
    if crc != 0 && size > 4096 {
        // nutdec.c:104-105 (C: return -1 → AVERROR_INVALIDDATA at the
        // caller; C prints nothing).
        return Err(Error::InvalidData(
            "main header forward checksum mismatch".into(),
        ));
    }
    r.init_checksum(calculate_checksum, 0);
    Ok(size)
}

/// `put_packet` (`nutenc.c:351-370`) — the muxer's mirror of
/// [`get_packetheader`]: big-endian startcode, vint forward pointer
/// (`payload + 4`), forward checksum when the pointer exceeds 4096
/// (little-endian, over startcode + forward pointer), payload, then the
/// little-endian body checksum. Writing the checksum LE is what lets the
/// demuxer's zero-check verify it (see [`CRC_TABLE`]).
pub fn put_packet(out: &mut Vec<u8>, payload: &[u8], startcode: u64) {
    let forw_ptr = payload.len() as u64 + 4; // nutenc.c:357
    let mut head = Vec::with_capacity(16);
    head.extend_from_slice(&startcode.to_be_bytes()); // avio_wb64
    put_v(&mut head, forw_ptr);
    if forw_ptr > 4096 {
        let crc = crc04c11db7_update(0, &head);
        head.extend_from_slice(&crc.to_le_bytes()); // avio_wl32
    }
    out.extend_from_slice(&head);
    out.extend_from_slice(payload);
    out.extend_from_slice(&crc04c11db7_update(0, payload).to_le_bytes());
}

// ---------------------------------------------------------------------
// nutdec.c small readers: get_str, get_fourcc, skip_reserved
// ---------------------------------------------------------------------

/// `get_str` (`nutdec.c:44-66`) — a vint length + raw bytes, Capped at
/// `maxlen` (C's stack buffer size; 256 at both call sites). `Err(Eof)` is
/// C's `AVERROR_EOF`, `Err(InvalidData)` C's `-1` "exactly filled" return.
fn get_str(r: &mut NutReader, maxlen: usize) -> Result<String> {
    let mut len = r.get_v()?;
    let mut bytes = vec![0u8; maxlen];
    if len != 0 && maxlen > 0 {
        let take = (len.min(maxlen as u64)) as usize;
        r.read(&mut bytes[..take])?;
    }
    while len > maxlen as u64 {
        r.r8()?;
        len -= 1;
        if r.is_eof() {
            len = maxlen as u64;
        }
    }
    if r.is_eof() {
        return Err(Error::Eof);
    }
    if maxlen as u64 == len {
        return Err(Error::InvalidData("get_str: string too long".into()));
    }
    let n = (len.min(maxlen as u64 - 1)) as usize;
    Ok(String::from_utf8_lossy(&bytes[..n]).into_owned())
}

/// `get_fourcc` (`nutdec.c:78-90`). Lengths other than 2/4 log C's
/// `"Unsupported fourcc length"` and return the `-1` sentinel (C returns
/// -1 cast to uint64_t; callers treat it as an unknown tag).
fn get_fourcc(r: &mut NutReader) -> Result<u64> {
    let len = r.get_v()?;
    if len == 2 {
        Ok(u64::from(r.rl16()?))
    } else if len == 4 {
        Ok(u64::from(r.rl32()?))
    } else {
        log_error!(None, "Unsupported fourcc length {}", len);
        Ok(u64::MAX)
    }
}

/// `skip_reserved` (`nutdec.c:179-193`) — consume (checksummed) bytes up
/// to the packet end `end`. A negative remainder seeks back and fails.
fn skip_reserved(r: &mut NutReader, end: u64) -> Result<()> {
    let pos = end as i64 - r.tell() as i64;
    if pos < 0 {
        // C: avio_seek(bc, pos, SEEK_CUR) then AVERROR_INVALIDDATA — the
        // seek's failure is irrelevant, the packet is doomed either way.
        let _ = r.seek((r.tell() as i64 + pos).max(0) as u64);
        Err(Error::InvalidData("reserved data before packet end".into()))
    } else {
        let mut pos = pos;
        while pos > 0 {
            if r.is_eof() {
                return Err(Error::InvalidData("EOF in reserved data".into()));
            }
            r.r8()?;
            pos -= 1;
        }
        Ok(())
    }
}

/// The `GET_V` macro (`nutdec.c:168-177`) — read a vint, `check` it, on
/// failure log and return `AVERROR_INVALIDDATA` with C's message
/// `"Error <dst> is (<value>)"`. `dst` keeps C's spelled-out lvalue text
/// (`"stream_count"`, `"nut->header_len[i]"`, …) so logs grep the same.
fn get_v_check(r: &mut NutReader, dst: &str, check: impl Fn(u64) -> bool) -> Result<u64> {
    let tmp = r.get_v()?;
    if !check(tmp) {
        log_error!(Some("nut"), "Error {} is ({})", dst, tmp as i64);
        return Err(Error::InvalidData(format!(
            "Error {} is ({})",
            dst,
            tmp as i64
        )));
    }
    Ok(tmp)
}

// ---------------------------------------------------------------------
// nut.c:266-282 — timestamp helpers
// ---------------------------------------------------------------------

/// `ff_lsb2full` (`nut.c:277-282`): reconstruct the full pts from its
/// `msb_pts_shift` low bits around the previous pts.
pub fn ff_lsb2full(msb_pts_shift: u32, last_pts: i64, lsb: i64) -> i64 {
    let mask = (1i64 << msb_pts_shift) - 1;
    let delta = last_pts.wrapping_sub(mask / 2);
    (lsb.wrapping_sub(delta) & mask).wrapping_add(delta)
}

/// `av_rescale_rnd(…, AV_ROUND_DOWN)` as `ff_nut_reset_ts` needs it
/// (`nut.c:270-274`): `a · b / c` with 128-bit intermediates, floored.
pub fn rescale_rnd_down(a: i64, b: i64, c: i64) -> i64 {
    if c == 0 {
        return 0;
    }
    let n = a as i128 * b as i128;
    let d = c as i128;
    let q = if (n < 0) == (d < 0) || n % d == 0 {
        n / d
    } else {
        n / d - 1 // floor toward zero's negative side
    };
    q.clamp(i64::MIN as i128, i64::MAX as i128) as i64
}

// ---------------------------------------------------------------------
// codec tag tables — nut.c:222-264 + riff.c + raw_pix_fmt_tags.h
// ---------------------------------------------------------------------

/// `MKTAG(a,b,c,d)` — the u32 `get_fourcc`'s little-endian reads produce
/// for the four file bytes `a b c d`.
const fn tag(a: u8, b: u8, c: u8, d: u8) -> u32 {
    u32::from_le_bytes([a, b, c, d])
}

/// The RAWVIDEO codec-tag rows the port can resolve to a family pixel
/// format: `ff_nut_video_tags` (`nut.c:44-220`) first, then the rawvideo
/// rows of `ff_codec_bmp_tags` (`riff.c`), tag → the pixel format
/// `find_pix_fmt(raw_pix_fmt_tags)` gives (`rawdec.c:33-44` over
/// `raw_pix_fmt_tags.h`). A hit means `CodecId::Rawvideo` + this format;
/// tags C maps to RAWVIDEO *outside* the family (RGB15/12, PAL8, the
/// p9/p12/p14 Y/G variants, …) resolve to `CodecId::None` here — both
/// sides fail identically at decode time (see the module map).
const RAWVIDEO_TAGS: &[(u32, PixelFormat)] = &[
    // ff_nut_video_tags order (nut.c:62-219)
    (tag(b'R', b'G', b'B', b'A'), PixelFormat::Rgba),          // nut.c:62
    (tag(b'B', b'G', b'R', b'A'), PixelFormat::Bgra),          // nut.c:64
    (tag(b'A', b'B', b'G', b'R'), PixelFormat::Abgr),          // nut.c:66
    (tag(b'A', b'R', b'G', b'B'), PixelFormat::Argb),          // nut.c:68
    (tag(b'R', b'G', b'B', 24), PixelFormat::Rgb24),           // nut.c:70
    (tag(b'B', b'G', b'R', 24), PixelFormat::Bgr24),           // nut.c:71
    (tag(b'4', b'2', b'2', b'P'), PixelFormat::Yuv422p),       // nut.c:73
    (tag(b'4', b'4', b'4', b'P'), PixelFormat::Yuv444p),       // nut.c:77
    (tag(b'Y', b'3', 11, 10), PixelFormat::Yuv420p10le),       // nut.c:103
    (tag(b'Y', b'3', 10, 10), PixelFormat::Yuv422p10le),       // nut.c:105
    (tag(b'Y', b'3', 0, 10), PixelFormat::Yuv444p10le),        // nut.c:99
    (tag(b'Y', b'3', 11, 16), PixelFormat::Yuv420p16le),       // nut.c:121
    (tag(b'Y', b'3', 0, 16), PixelFormat::Yuv444p16le),        // nut.c:125
    (tag(b'Y', b'1', 0, 16), PixelFormat::Gray16le),           // nut.c:119
    (tag(b'G', b'3', 0, 8), PixelFormat::Gbrp),                // nut.c:169
    (tag(b'G', b'4', 0, 8), PixelFormat::Gbrap),               // nut.c:186
    // ff_codec_bmp_tags rawvideo rows (riff.c), first-match order as in
    // the port's wav.rs WAV_CODEC_TAGS
    (tag(b'I', b'4', b'2', b'0'), PixelFormat::Yuv420p),       // raw:29
    (tag(b'I', b'Y', b'U', b'V'), PixelFormat::Yuv420p),       // raw:30
    (tag(b'y', b'v', b'1', b'2'), PixelFormat::Yuv420p),       // raw:31
    (tag(b'Y', b'V', b'1', b'2'), PixelFormat::Yuv420p),       // raw:31
    (tag(b'Y', b'4', b'2', b'B'), PixelFormat::Yuv422p),       // raw:36
    (tag(b'P', b'4', b'2', b'2'), PixelFormat::Yuv422p),       // raw:37
    (tag(b'Y', b'V', b'1', b'6'), PixelFormat::Yuv422p),       // raw:38
    (tag(b'I', b'4', b'2', b'2'), PixelFormat::Yuv422p),       // raw:270
    (tag(b'I', b'4', b'4', b'4'), PixelFormat::Yuv444p),       // raw:272
    (tag(b'Y', b'8', b'0', b'0'), PixelFormat::Gray8),         // raw:46
    (tag(b'Y', b'8', b' ', b' '), PixelFormat::Gray8),         // raw:47
    (tag(b'G', b'R', b'E', b'Y'), PixelFormat::Gray8),         // raw:69
    (tag(b'Y', b'U', b'Y', b'2'), PixelFormat::Yuyv422),       // raw:49
    (tag(b'Y', b'4', b'2', b'2'), PixelFormat::Yuyv422),       // raw:50
    (tag(b'Y', b'U', b'Y', b'V'), PixelFormat::Yuyv422),       // raw:54
    (tag(b'U', b'Y', b'V', b'Y'), PixelFormat::Uyvy422),       // raw:56
    (tag(b'N', b'V', b'1', b'2'), PixelFormat::Nv12),          // raw:70
    (tag(b'N', b'V', b'2', b'1'), PixelFormat::Nv21),          // raw:71
    (tag(b'R', b'G', b'B', 16), PixelFormat::Rgb565le),        // raw:81
];

/// `av_codec_get_id` over `[ff_nut_audio_tags, ff_codec_wav_tags,
/// ff_nut_audio_extra_tags]` (`nutdec.c:415-420`) — family rows in C's
/// table order (first match wins). `PCM_S8` (`'P','S','D',8`, nut.c:245),
/// the `U16/U24/U32/S64/PLANAR` flavors and the compressed codecs
/// (MP3/Opus/WavPack/comfort-noise) are not in the `CodecId` family and
/// resolve to `None` — C's `AV_CODEC_ID_NONE` path.
const NUT_AUDIO_TAGS: &[(CodecId, u32)] = &[
    // ff_nut_audio_tags (nut.c:232-258)
    (CodecId::PcmF32be, tag(32, b'D', b'F', b'P')), // nut.c:233
    (CodecId::PcmF32le, tag(b'P', b'F', b'D', 32)), // nut.c:234
    (CodecId::PcmF64be, tag(64, b'D', b'F', b'P')), // nut.c:235
    (CodecId::PcmF64le, tag(b'P', b'F', b'D', 64)), // nut.c:236
    (CodecId::PcmS16be, tag(16, b'D', b'S', b'P')), // nut.c:237
    (CodecId::PcmS16le, tag(b'P', b'S', b'D', 16)), // nut.c:238
    (CodecId::PcmS24be, tag(24, b'D', b'S', b'P')), // nut.c:239
    (CodecId::PcmS24le, tag(b'P', b'S', b'D', 24)), // nut.c:240
    (CodecId::PcmS32be, tag(32, b'D', b'S', b'P')), // nut.c:241
    (CodecId::PcmS32le, tag(b'P', b'S', b'D', 32)), // nut.c:242
    (CodecId::PcmU8, tag(b'P', b'U', b'D', 8)),     // nut.c:252
    // ff_codec_wav_tags family rows (riff.c:526-537, 602) — same rows as
    // the port's wav.rs WAV_CODEC_TAGS
    (CodecId::PcmS16le, 0x0001),
    (CodecId::PcmF32le, 0x0003),
    (CodecId::PcmAlaw, 0x0006),
    (CodecId::PcmMulaw, 0x0007),
    (CodecId::PcmMulaw, 0x6c75), // ('u'<<8)|'l'
    // ff_nut_audio_extra_tags (nut.c:222-230)
    (CodecId::PcmAlaw, tag(b'A', b'L', b'A', b'W')),  // nut.c:224
    (CodecId::PcmMulaw, tag(b'U', b'L', b'A', b'W')), // nut.c:225
];

/// `av_codec_get_id` for a video-class fourcc (`nutdec.c:404-412`):
/// the RAWVIDEO rows of `ff_nut_video_tags` / `ff_codec_bmp_tags` /
/// `ff_codec_movvideo_tags`, collapsed to "in the family or not".
fn video_codec_id(codec_tag: u32) -> (CodecId, Option<PixelFormat>) {
    match RAWVIDEO_TAGS.iter().find(|&&(t, _)| t == codec_tag) {
        Some(&(_, fmt)) => (CodecId::Rawvideo, Some(fmt)),
        None => (CodecId::None, None),
    }
}

/// `av_codec_get_id` for an audio-class fourcc (`nutdec.c:415-420`).
fn audio_codec_id(codec_tag: u32) -> CodecId {
    NUT_AUDIO_TAGS
        .iter()
        .find(|&&(_, t)| t == codec_tag)
        .map(|&(id, _)| id)
        .unwrap_or(CodecId::None)
}

// ---------------------------------------------------------------------
// nutdec.c:39 + NUTContext — demuxer state
// ---------------------------------------------------------------------

/// `NUT_MAX_STREAMS` (`nutdec.c:39`).
const NUT_MAX_STREAMS: u64 = 256;

/// What one stream header taught the demuxer (C scatters this over the
/// freshly allocated `AVStream`/`AVCodecParameters`, `nutdec.c:395-470`).
/// The port keeps the full set so frame decoding can span every stream
/// while only stream 0 is surfaced (see the module map).
#[derive(Debug, Clone, Default)]
struct StreamParams {
    codec_type: MediaType,
    codec_id: CodecId,
    codec_tag: u32,
    format: Option<PixelFormat>,
    width: u32,
    height: u32,
    sample_aspect_ratio: Rational,
    sample_rate: i32,
    channels: usize,
}

/// `NUTContext` (`nut.h:91-118`) — the read-side fields.
pub struct NutDemuxer {
    frame_code: [FrameCode; 256],
    /// `nut->header[]` — elision headers; `[0]` empty by definition
    /// (`nutdec.c:345`).
    headers: Vec<Vec<u8>>,
    /// `nut->next_startcode` — a startcode already consumed by resync.
    next_startcode: u64,
    /// `nut->stream[]`.
    streams: Vec<StreamContext>,
    /// Port addition: per-stream codec parameters from the stream headers.
    stream_params: Vec<StreamParams>,
    max_distance: u32,
    time_bases: Vec<Rational>,
    last_syncpoint_pos: i64,
    last_resync_pos: i64,
    header_count: usize,
    /// Main-header flags (`nutdec.c:349-351`, version 4+).
    flags: u64,
    version: u64,
    minor_version: u64,
}

impl NutDemuxer {
    pub fn new() -> Self {
        NutDemuxer {
            frame_code: [FrameCode::default(); 256],
            headers: vec![Vec::new(); 128],
            next_startcode: 0,
            streams: Vec::new(),
            stream_params: Vec::new(),
            max_distance: 0,
            time_bases: Vec::new(),
            last_syncpoint_pos: 0,
            last_resync_pos: 0,
            header_count: 0,
            flags: 0,
            version: 0,
            minor_version: 0,
        }
    }

    /// `ff_nut_reset_ts` (`nut.c:266-275`) — rescale the syncpoint's
    /// timestamp (in its own time base) into every stream's time base.
    fn reset_ts(&mut self, time_base: Rational, val: i64) {
        for stc in &mut self.streams {
            if let Some(stb) = stc.time_base {
                stc.last_pts = rescale_rnd_down(
                    val,
                    i64::from(time_base.num) * i64::from(stb.den),
                    i64::from(time_base.den) * i64::from(stb.num),
                );
            }
        }
    }

    /// `decode_main_header` (`nutdec.c:195-379`).
    fn decode_main_header(&mut self, r: &mut NutReader) -> Result<()> {
        let length = get_packetheader(r, true, MAIN_STARTCODE)?;
        let end = length + r.tell();

        self.version = r.get_v()?;
        if !(NUT_MIN_VERSION..=NUT_MAX_VERSION).contains(&self.version) {
            // nutdec.c:210-215.
            log_error!(Some("nut"), "Version {} not supported.", self.version);
            return Err(Error::Unsupported(format!(
                "Version {} not supported.",
                self.version
            )));
        }
        if self.version > 3 {
            self.minor_version = r.get_v()?;
        }

        let stream_count = get_v_check(r, "stream_count", |t| {
            t > 0 && t <= NUT_MAX_STREAMS
        })?;

        self.max_distance = r.get_v()? as u32; // nutdec.c:221 (int field)
        if self.max_distance > 65536 {
            // nutdec.c:222-225.
            log_verbose!(Some("nut"), "max_distance {}", self.max_distance);
            self.max_distance = 65536;
        }

        // nutdec.c:227 — C also bounds by INT_MAX/sizeof(AVRational).
        let time_base_count = get_v_check(r, "nut->time_base_count", |t| {
            t > 0 && t < (i32::MAX as u64) / 8 && t < length / 2
        })? as usize;

        self.time_bases = Vec::with_capacity(time_base_count);
        for _ in 0..time_base_count {
            let num = get_v_check(r, "nut->time_base[i].num", |t| t > 0 && t < (1 << 31))? as i32;
            let den = get_v_check(r, "nut->time_base[i].den", |t| t > 0 && t < (1 << 31))? as i32;
            if crate::util::mathematics::gcd(i64::from(num), i64::from(den)) != 1 {
                // nutdec.c:235-241.
                log_error!(Some("nut"), "invalid time base {}/{}", num, den);
                return Err(Error::InvalidData(format!(
                    "invalid time base {}/{}",
                    num, den
                )));
            }
            self.time_bases.push(Rational::new(num, den));
        }

        // The frame-code table (nutdec.c:243-319).
        let mut tmp_pts: i64 = 0;
        let mut tmp_mul: i32 = 1;
        let mut tmp_stream: u32 = 0;
        let mut tmp_head_idx: u32 = 0;
        let mut i = 0usize;
        while i < 256 {
            let tmp_flags = r.get_v()? as u32;
            let raw_fields = r.get_v()?;
            let fields_i32 = (raw_fields as u32) as i32;
            if fields_i32 < 0 {
                // nutdec.c:250-253 — the int-cast negative check.
                log_error!(Some("nut"), "fields {} is invalid", fields_i32);
                return Err(Error::InvalidData(format!(
                    "fields {} is invalid",
                    fields_i32
                )));
            }
            let tmp_fields = raw_fields;
            let mut tmp_size: i32 = 0;
            let mut tmp_res: u32 = 0;

            if tmp_fields > 0 {
                tmp_pts = r.get_s()?;
            }
            if tmp_fields > 1 {
                tmp_mul = r.get_v()? as u32 as i32;
            }
            if tmp_fields > 2 {
                tmp_stream = r.get_v()? as u32;
            }
            if tmp_fields > 3 {
                tmp_size = r.get_v()? as u32 as i32;
            }
            if tmp_fields > 4 {
                tmp_res = r.get_v()? as u32;
            }
            let count = if tmp_fields > 5 {
                r.get_v()? as u32 as i32
            } else {
                // nutdec.c:273 — `count = tmp_mul - (unsigned)tmp_size`.
                ((tmp_mul as u32).wrapping_sub(tmp_size as u32)) as i32
            };
            if tmp_fields > 6 {
                r.get_s()?; // match_time_delta — unused by the reader
            }
            if tmp_fields > 7 {
                tmp_head_idx = r.get_v()? as u32;
            }
            let mut extra = tmp_fields;
            while extra > 8 {
                if r.is_eof() {
                    // nutdec.c:280-284.
                    log_error!(
                        Some("nut"),
                        "reached EOF while decoding main header"
                    );
                    return Err(Error::InvalidData(
                        "reached EOF while decoding main header".into(),
                    ));
                }
                r.get_v()?;
                extra -= 1;
            }

            let bound = 256 - i32::from(u64::from(b'N') >= i as u64) - i as i32;
            if count <= 0 || count > bound {
                // nutdec.c:288-292.
                log_error!(Some("nut"), "illegal count {} at {}", count, i);
                return Err(Error::InvalidData(format!(
                    "illegal count {} at {}",
                    count, i
                )));
            }
            if tmp_stream as u64 >= stream_count {
                // nutdec.c:293-297.
                log_error!(
                    Some("nut"),
                    "illegal stream number {} >= {}",
                    tmp_stream,
                    stream_count
                );
                return Err(Error::InvalidData(format!(
                    "illegal stream number {} >= {}",
                    tmp_stream, stream_count
                )));
            }
            if tmp_size < 0 || tmp_size > i32::MAX - count {
                // nutdec.c:299-303.
                log_error!(Some("nut"), "illegal size");
                return Err(Error::InvalidData("illegal size".into()));
            }

            let mut j = 0i32;
            while j < count {
                if i == b'N' as usize {
                    // nutdec.c:306-309 — 'N' can never start a frame code.
                    self.frame_code[i].flags = flag::INVALID as u16;
                    j -= 1; // cancelled by the loop increment, like C
                } else {
                    self.frame_code[i] = FrameCode {
                        flags: tmp_flags as u16,
                        pts_delta: tmp_pts as i16,
                        stream_id: tmp_stream as u8,
                        size_mul: tmp_mul as u16,
                        size_lsb: (tmp_size + j) as u16,
                        reserved_count: tmp_res as u8,
                        header_idx: tmp_head_idx as u8,
                    };
                }
                j += 1;
                i += 1;
            }
        }
        debug_assert_eq!(self.frame_code[b'N' as usize].flags, flag::INVALID as u16);

        // Elision headers (nutdec.c:322-346).
        if end > r.tell() + 4 {
            let mut rem = 1024i32;
            let header_count =
                get_v_check(r, "nut->header_count", |t| t < 128)? as usize;
            self.header_count = header_count + 1; // nutdec.c:325
            for i in 1..self.header_count {
                let len =
                    get_v_check(r, "nut->header_len[i]", |t| t > 0 && t < 256)? as usize;
                if rem < len as i32 {
                    // nutdec.c:329-335.
                    log_error!(
                        Some("nut"),
                        "invalid elision header {} : {} > {}",
                        i,
                        len,
                        rem
                    );
                    return Err(Error::InvalidData(format!(
                        "invalid elision header {} : {} > {}",
                        i, len, rem
                    )));
                }
                rem -= len as i32;
                let mut hdr = vec![0u8; len];
                r.read(&mut hdr)?;
                self.headers[i] = hdr;
            }
            debug_assert_eq!(self.headers[0].len(), 0); // nutdec.c:345
        }

        // flags had been effectively introduced in version 4 (nutdec.c:349).
        if self.version > 3 && end > r.tell() + 4 {
            self.flags = r.get_v()?;
        }

        // nutdec.c:353-357.
        let crc_ok = match skip_reserved(r, end) {
            Ok(()) => r.get_checksum() == 0,
            Err(_) => false,
        };
        if !crc_ok {
            log_error!(Some("nut"), "main header checksum mismatch");
            return Err(Error::InvalidData(
                "main header checksum mismatch".into(),
            ));
        }

        // nutdec.c:359-369 — create the stream contexts (the port's
        // stream objects are built in read_header from stream_params).
        self.streams = vec![StreamContext::default(); stream_count as usize];
        self.stream_params = vec![StreamParams::default(); stream_count as usize];
        Ok(())
    }

    /// `decode_stream_header` (`nutdec.c:381-488`).
    fn decode_stream_header(&mut self, r: &mut NutReader) -> Result<()> {
        let end = get_packetheader(r, true, STREAM_STARTCODE)? + r.tell();

        let stream_count = self.streams.len() as u64;
        let stream_id = get_v_check(r, "stream_id", |t| {
            t < stream_count && self.streams[t as usize].time_base.is_none()
        })? as usize;
        let stc = &mut self.streams[stream_id];
        let params = &mut self.stream_params[stream_id];

        let class = r.get_v()?;
        let codec_tag = get_fourcc(r)? as u32;
        params.codec_tag = codec_tag;
        match class {
            0 => {
                params.codec_type = MediaType::Video;
                let (id, fmt) = video_codec_id(codec_tag);
                params.codec_id = id;
                params.format = fmt;
            }
            1 => {
                params.codec_type = MediaType::Audio;
                params.codec_id = audio_codec_id(codec_tag);
            }
            2 => {
                params.codec_type = MediaType::Subtitle; // nutdec.c:423-426
                params.codec_id = CodecId::None; // ff_nut_subtitle_tags: none in family
            }
            3 => {
                params.codec_type = MediaType::Data; // nutdec.c:427-430
                params.codec_id = CodecId::None;
            }
            _ => {
                // nutdec.c:431-434.
                log_error!(Some("nut"), "unknown stream class ({})", class);
                return Err(Error::Unsupported(format!(
                    "unknown stream class ({})",
                    class
                )));
            }
        }
        if class < 3 && params.codec_id == CodecId::None {
            // nutdec.c:435-438.
            log_error!(
                Some("nut"),
                "Unknown codec tag '0x{:04x}' for stream number {}",
                codec_tag,
                stream_id
            );
        }

        let tbc = self.time_bases.len() as u64;
        stc.time_base_id = get_v_check(r, "stc->time_base_id", |t| t < tbc)? as usize;
        stc.msb_pts_shift = get_v_check(r, "stc->msb_pts_shift", |t| t < 16)? as u32;
        stc.max_pts_distance = r.get_v()? as u32 as i32 as i64; // int field
        stc.decode_delay = get_v_check(r, "stc->decode_delay", |t| t < 1000)? as u32;
        // st->codecpar->video_delay = stc->decode_delay (nutdec.c:444) — no
        // such field in the port's CodecParameters.
        r.get_v()?; // stream flags

        let extradata_size = get_v_check(r, "st->codecpar->extradata_size", |t| {
            t < (1 << 30)
        })? as usize;
        if extradata_size > 0 {
            // ff_get_extradata (nutdec.c:448-453) — no extradata field in
            // the port; consume (checksummed) and drop.
            let mut extra = vec![0u8; extradata_size];
            r.read(&mut extra)?;
        }

        match params.codec_type {
            MediaType::Video => {
                params.width = get_v_check(r, "st->codecpar->width", |t| t > 0)? as u32;
                params.height = get_v_check(r, "st->codecpar->height", |t| t > 0)? as u32;
                let num = r.get_v()? as i32;
                let den = r.get_v()? as i32;
                if (num == 0) != (den == 0) {
                    // nutdec.c:460-465.
                    log_error!(Some("nut"), "invalid aspect ratio {}/{}", num, den);
                    return Err(Error::InvalidData(format!(
                        "invalid aspect ratio {}/{}",
                        num, den
                    )));
                }
                params.sample_aspect_ratio = Rational::new(num, den);
                r.get_v()?; /* csp type */
            }
            MediaType::Audio => {
                params.sample_rate =
                    get_v_check(r, "st->codecpar->sample_rate", |t| t > 0)? as i32;
                r.get_v()?; // samplerate_den
                params.channels = get_v_check(
                    r,
                    "st->codecpar->ch_layout.nb_channels",
                    |t| t > 0,
                )? as usize;
            }
            _ => {}
        }

        // nutdec.c:472-477.
        let crc_ok = match skip_reserved(r, end) {
            Ok(()) => r.get_checksum() == 0,
            Err(_) => false,
        };
        if !crc_ok {
            log_error!(
                Some("nut"),
                "stream header {} checksum mismatch",
                stream_id
            );
            return Err(Error::InvalidData(format!(
                "stream header {} checksum mismatch",
                stream_id
            )));
        }

        // nutdec.c:478-480 — bind the time base (the duplicate guard).
        stc.time_base = Some(self.time_bases[stc.time_base_id]);
        Ok(())
    }

    /// `decode_info_header` (`nutdec.c:505-626`) — parsed byte-exactly and
    /// checksum-verified; the metadata/chapter/disposition storage is out
    /// (no dictionary in the port, see the module map).
    fn decode_info_header(&mut self, r: &mut NutReader) -> Result<()> {
        let end = get_packetheader(r, true, INFO_STARTCODE)? + r.tell();
        let nb_streams = self.streams.len() as u64;

        let stream_id_plus1 = get_v_check(r, "stream_id_plus1", |t| t <= nb_streams)?;
        let chapter_id = r.get_s()?;
        let _chapter_start = r.get_v()?;
        let _chapter_len = r.get_v()?;
        let count = r.get_v()?; // unsigned int in C

        // Chapter/metadata target selection (nutdec.c:530-550) — the port
        // keeps only the parse; chapter_id is still read by the entry loop.
        let _ = chapter_id;

        for _ in 0..count {
            let name = match get_str(r, 256) {
                Ok(s) => s,
                Err(e) => {
                    // nutdec.c:553-557.
                    log_error!(
                        Some("nut"),
                        "get_str failed while decoding info header"
                    );
                    return Err(e);
                }
            };
            let mut value = r.get_s()?;
            let mut type_str = String::new();
            let mut str_value = String::new();

            if value == -1 {
                // type = "UTF-8"
                str_value = get_str(r, 1024).inspect_err(|_| {
                    log_error!(
                        Some("nut"),
                        "get_str failed while decoding info header"
                    );
                })?;
            } else if value == -2 {
                type_str = get_str(r, 256).inspect_err(|_| {
                    log_error!(
                        Some("nut"),
                        "get_str failed while decoding info header"
                    );
                })?;
                str_value = get_str(r, 1024).inspect_err(|_| {
                    log_error!(
                        Some("nut"),
                        "get_str failed while decoding info header"
                    );
                })?;
            } else if value == -3 {
                // type = "s"
                value = r.get_s()?;
            } else if value == -4 {
                // type = "t"
                value = r.get_v()? as i64;
            } else if value < -4 {
                // type = "r" — rational, second component skipped
                r.get_s()?;
            }
            // else type = "v" (plain integer already in `value`)

            if stream_id_plus1 > nb_streams {
                // nutdec.c:590-595.
                log_warning!(
                    Some("nut"),
                    "invalid stream id {} for info packet",
                    stream_id_plus1
                );
                continue;
            }

            // Metadata dictionary, Disposition bits and r_frame_rate
            // (nutdec.c:597-617) — not ported; the (name, value) pairs are
            // dropped here.
            let _ = (name, str_value, type_str, value);
        }

        // nutdec.c:620-623.
        let crc_ok = match skip_reserved(r, end) {
            Ok(()) => r.get_checksum() == 0,
            Err(_) => false,
        };
        if !crc_ok {
            log_error!(Some("nut"), "info header checksum mismatch");
            return Err(Error::InvalidData("info header checksum mismatch".into()));
        }
        Ok(())
    }

    /// `decode_syncpoint` (`nutdec.c:628-669`). Returns `(ts, back_ptr)` —
    /// the AV_TIME_BASE ts C computes for the syncpoint tree; the port has
    /// no seek index so callers only use the state side effects
    /// (`last_syncpoint_pos`, `reset_ts`).
    fn decode_syncpoint(&mut self, r: &mut NutReader) -> Result<(i64, i64)> {
        self.last_syncpoint_pos = r.tell() as i64 - 8; // nutdec.c:636

        let end = get_packetheader(r, true, SYNCPOINT_STARTCODE)? + r.tell();

        let tbc = self.time_bases.len() as u64;
        let mut tmp = r.get_v()?;
        let back_ptr = self.last_syncpoint_pos - 16 * r.get_v()? as i64; // nutdec.c:642
        if back_ptr < 0 {
            return Err(Error::InvalidData(
                "sync point back pointer before start of file".into(),
            ));
        }

        // ff_nut_reset_ts (nut.c:266-275).
        let tb = self.time_bases[(tmp % tbc) as usize];
        self.reset_ts(tb, (tmp / tbc) as i64);

        if self.flags & NUT_BROADCAST != 0 {
            // nutdec.c:649-655 — C reads the wallclock *into tmp* and then
            // computes *ts from the overwritten value; ported as-is.
            tmp = r.get_v()?;
            log_verbose!(Some("nut"), "Syncpoint wallclock {}", tmp / tbc);
        }

        // nutdec.c:657-660.
        let crc_ok = match skip_reserved(r, end) {
            Ok(()) => r.get_checksum() == 0,
            Err(_) => false,
        };
        if !crc_ok {
            log_error!(Some("nut"), "sync point checksum mismatch");
            return Err(Error::InvalidData("sync point checksum mismatch".into()));
        }

        // nutdec.c:662-663 — `tmp/tbc * av_q2d(tb) * AV_TIME_BASE` (double
        // math, exactly as C; ff_nut_add_sp is out with the index).
        let ts = (tmp / tbc) as f64 * tb.to_f64() * 1_000_000f64; // AV_TIME_BASE
        Ok((ts as i64, back_ptr))
    }

    /// `decode_frame_header` (`nutdec.c:997-1078`). Returns
    /// `(size, pts, stream_id, header_idx)`; updates the stream's
    /// `last_pts`/`last_flags` like C does at nutdec.c:1072-1073.
    fn decode_frame_header(
        &mut self,
        r: &mut NutReader,
        frame_code: u8,
    ) -> Result<(i64, i64, usize, usize)> {
        if self.flags & NUT_PIPE == 0
            && r.tell() as i64 > self.last_syncpoint_pos + i64::from(self.max_distance)
        {
            // nutdec.c:1006-1011.
            log_error!(
                Some("nut"),
                "Last frame must have been damaged {} > {} + {}",
                r.tell(),
                self.last_syncpoint_pos,
                self.max_distance
            );
            return Err(Error::InvalidData(format!(
                "Last frame must have been damaged {} > {} + {}",
                r.tell(),
                self.last_syncpoint_pos,
                self.max_distance
            )));
        }

        let fc = self.frame_code[frame_code as usize];
        let mut flags = u32::from(fc.flags);
        let size_mul = u32::from(fc.size_mul);
        let mut size = i64::from(fc.size_lsb);
        let mut stream_id = fc.stream_id as usize;
        let pts_delta = i64::from(fc.pts_delta);
        let mut reserved_count = fc.reserved_count as u64;
        let mut header_idx = fc.header_idx as usize;

        if flags & flag::INVALID != 0 {
            // nutdec.c:1022-1023 — C returns AVERROR_INVALIDDATA silently.
            return Err(Error::InvalidData(format!(
                "invalid frame code {frame_code}"
            )));
        }
        if flags & flag::CODED != 0 {
            flags ^= r.get_v()? as u32; // nutdec.c:1024-1025
        }
        if flags & flag::STREAM_ID != 0 {
            let nb = self.streams.len() as u64;
            stream_id = get_v_check(r, "*stream_id", |t| t < nb)? as usize;
        }
        let stc = &mut self.streams[stream_id];
        let pts;
        if flags & flag::CODED_PTS != 0 {
            let coded_pts = r.get_v()? as i64;
            // nutdec.c:1030-1036.
            if coded_pts < (1i64 << stc.msb_pts_shift) {
                pts = ff_lsb2full(stc.msb_pts_shift, stc.last_pts, coded_pts);
            } else {
                pts = coded_pts - (1i64 << stc.msb_pts_shift);
            }
        } else {
            pts = stc.last_pts + pts_delta; // nutdec.c:1037-1038
        }
        if flags & flag::SIZE_MSB != 0 {
            size += i64::from(size_mul) * r.get_v()? as i64; // nutdec.c:1040
        }
        if flags & flag::MATCH_TIME != 0 {
            r.get_s()?; // nutdec.c:1042
        }
        if flags & flag::HEADER_IDX != 0 {
            header_idx = r.get_v()? as usize; // nutdec.c:1044
        }
        if flags & flag::RESERVED != 0 {
            reserved_count = r.get_v()?; // nutdec.c:1046
        }
        for _ in 0..reserved_count {
            if r.is_eof() {
                // nutdec.c:1048-1051.
                log_error!(Some("nut"), "reached EOF while decoding frame header");
                return Err(Error::InvalidData(
                    "reached EOF while decoding frame header".into(),
                ));
            }
            r.get_v()?;
        }

        if header_idx >= self.header_count {
            // nutdec.c:1055-1058.
            log_error!(Some("nut"), "header_idx invalid");
            return Err(Error::InvalidData("header_idx invalid".into()));
        }
        if size > 4096 {
            header_idx = 0; // nutdec.c:1059-1060
        }
        size -= self.headers[header_idx].len() as i64; // nutdec.c:1061

        if flags & flag::CHECKSUM != 0 {
            r.rb32()?; // nutdec.c:1063-1064 — FIXME check this (C doesn't)
        } else if self.flags & NUT_PIPE == 0
            && (size > 2 * i64::from(self.max_distance)
                || (stc.last_pts - pts).abs() > stc.max_pts_distance)
        {
            // nutdec.c:1065-1070.
            log_error!(
                Some("nut"),
                "frame size > 2max_distance and no checksum"
            );
            return Err(Error::InvalidData(
                "frame size > 2max_distance and no checksum".into(),
            ));
        }

        stc.last_pts = pts; // nutdec.c:1072
        stc.last_flags = flags; // nutdec.c:1073

        if size < 0 {
            // decode_frame's `if (size < 0) return size` (nutdec.c:1090-1091)
            // — a negative size (over-long elision header) is a decode error.
            return Err(Error::InvalidData("frame size negative".into()));
        }
        Ok((size, pts, stream_id, header_idx))
    }

    /// `decode_frame` (`nutdec.c:1080-1145`). `Ok(None)` is C's `return 1`
    /// ("packet decoded but discarded" — the port discards streams ≠ 0).
    fn decode_frame(&mut self, r: &mut NutReader, frame_code: u8) -> Result<Option<Packet>> {
        let (mut size, pts, stream_id, header_idx) =
            self.decode_frame_header(r, frame_code)?;

        let stc = &mut self.streams[stream_id];
        if stc.last_flags & flag::KEY != 0 {
            stc.skip_until_key_frame = false; // nutdec.c:1095-1096
        }
        // The discard checks (nutdec.c:1098-1107) never fire with C's
        // defaults (discard = AVDISCARD_DEFAULT, skip_until_key_frame 0
        // without a seek); the port's permanent degradation — dropping
        // packets of streams it cannot surface — lives in the same spot.
        if stream_id != 0 {
            let _ = r.skip(size as u64); // C's avio_skip in the discard arm
            return Ok(None);
        }

        let mut data = Vec::with_capacity((size + self.headers[header_idx].len() as i64) as usize);
        data.extend_from_slice(&self.headers[header_idx]); // nutdec.c:1112-1113
        let pos = r.tell(); // pkt->pos (nutdec.c:1114)
        let last_flags = self.streams[stream_id].last_flags;
        if last_flags & flag::SM_DATA != 0 {
            // nutdec.c:1115-1127.
            if read_sm_data(r, pos + size as u64, false).is_err() {
                return Err(Error::InvalidData("invalid side/meta data".into()));
            }
            if read_sm_data(r, pos + size as u64, true).is_err() {
                return Err(Error::InvalidData("invalid side/meta data".into()));
            }
            let sm_size = r.tell() - pos;
            size -= sm_size as i64;
        }

        // nutdec.c:1129-1134 — short reads shrink the packet, like C.
        let mut payload = vec![0u8; size as usize];
        let ret = r.read(&mut payload)?;
        data.extend_from_slice(&payload[..ret]);

        let mut pkt = Packet::from_vec(data);
        pkt.stream_index = 0; // only stream 0's packets surface (see above)
        pkt.pos = pos;
        pkt.pts = pts; // nutdec.c:1139
        // dts stays NOPTS (decode_frame sets only pts; b-frame reordering
        // is the generic layer's job, which the port does not have).
        pkt.dts = NOPTS;
        // C leaves duration 0 for the generic layer (which derives it from
        // the next packet's dts); the port uses the frame code's pts_delta
        // — the container's declared delta to the next frame of the stream.
        pkt.duration = i64::from(self.frame_code[frame_code as usize].pts_delta);
        pkt.time_base = self.streams[stream_id].time_base.unwrap_or(Rational::UNKNOWN);
        if last_flags & flag::KEY != 0 {
            pkt.flags = pkt.flags.union(PacketFlags::KEY); // nutdec.c:1137-1138
        }
        Ok(Some(pkt))
    }

    /// The resync arm of `nut_read_packet` (nutdec.c:1195-1202): scan for
    /// the next startcode and stash it. `Ok(false)` = C's `tmp == 0`
    /// (AVERROR_INVALIDDATA at the call site).
    fn resync(&mut self, r: &mut NutReader, pos: i64) -> Result<bool> {
        log_verbose!(Some("nut"), "syncing from {}", pos);
        let from = self.last_syncpoint_pos.max(self.last_resync_pos) + 1;
        let tmp = find_any_startcode(r, from);
        self.last_resync_pos = r.tell() as i64;
        if tmp == 0 {
            return Ok(false);
        }
        log_verbose!(Some("nut"), "sync");
        self.next_startcode = tmp;
        Ok(true)
    }

    /// `nut_read_header` (`nutdec.c:812-878`).
    fn read_header_impl(&mut self, r: &mut NutReader) -> Result<Stream> {
        /* main header */
        let mut pos = 0i64;
        loop {
            pos = match find_startcode(r, MAIN_STARTCODE, pos) {
                Some(p) => p as i64 + 1,
                None => {
                    // nutdec.c:829-831.
                    log_error!(Some("nut"), "No main startcode found.");
                    return Err(Error::InvalidData("No main startcode found.".into()));
                }
            };
            // C's do/while retries the main header on the next startcode
            // after any failure (except ENOMEM — allocation failure aborts
            // in Rust), nutdec.c:824-833.
            if self.decode_main_header(r).is_ok() {
                break;
            }
        }

        /* stream headers */
        pos = 0;
        let mut initialized_stream_count = 0usize;
        while initialized_stream_count < self.streams.len() {
            pos = match find_startcode(r, STREAM_STARTCODE, pos) {
                Some(p) => p as i64 + 1,
                None => {
                    // nutdec.c:839-841.
                    log_error!(Some("nut"), "Not all stream headers found.");
                    return Err(Error::InvalidData(
                        "Not all stream headers found.".into(),
                    ));
                }
            };
            if self.decode_stream_header(r).is_ok() {
                initialized_stream_count += 1;
            }
        }

        /* info headers */
        pos = 0;
        loop {
            let startcode = find_any_startcode(r, pos);
            pos = r.tell() as i64;

            if startcode == 0 {
                // nutdec.c:853-855.
                log_error!(Some("nut"), "EOF before video frames");
                return Err(Error::InvalidData("EOF before video frames".into()));
            } else if startcode == SYNCPOINT_STARTCODE {
                self.next_startcode = startcode;
                break;
            } else if startcode != INFO_STARTCODE {
                continue;
            }

            let _ = self.decode_info_header(r); // C ignores errors here too
        }

        // ffformatcontext(s)->data_offset = pos - 8 (nutdec.c:866) — seek
        // bookkeeping, out with the index. find_and_decode_index
        // (nutdec.c:868-872) likewise (see the module map).
        debug_assert_eq!(self.next_startcode, SYNCPOINT_STARTCODE);

        // Build the exposed stream: stream 0 of the port's single-Stream
        // contract, with everything its header learned (see the module
        // map's multi-stream degradation).
        let p = &self.stream_params[0];
        let mut st = Stream::new_video(0);
        st.codecpar.codec_type = p.codec_type;
        st.codecpar.codec_id = p.codec_id;
        match p.codec_type {
            MediaType::Video => {
                st.codecpar.width = p.width;
                st.codecpar.height = p.height;
                st.codecpar.format = p.format.unwrap_or(PixelFormat::Gray8);
                st.sample_aspect_ratio = p.sample_aspect_ratio;
                st.codecpar.sample_aspect_ratio = p.sample_aspect_ratio;
            }
            MediaType::Audio => {
                st.codecpar.sample_rate = p.sample_rate;
                st.codecpar.ch_layout = ChannelLayout::unspecified(p.channels);
                if let Some(f) = pcm::sample_fmt(p.codec_id) {
                    st.codecpar.sample_fmt = f;
                }
            }
            _ => {}
        }
        // avpriv_set_pts_info(st, 63, tb.num, tb.den) (nutdec.c:479-480) —
        // the NUT time base is already reduced (gcd checked in the main
        // header), and avg_frame_rate stays unknown (C leaves it for the
        // generic estimation layer).
        st.time_base = self.streams[0].time_base.unwrap_or(Rational::UNKNOWN);
        Ok(st)
    }

    /// `nut_read_packet` (`nutdec.c:1147-1205`).
    fn read_packet_impl(&mut self, r: &mut NutReader) -> Result<Packet> {
        loop {
            let pos = r.tell() as i64; // C keeps `pos` for the resync log
            let mut tmp = self.next_startcode;
            self.next_startcode = 0;

            let mut frame_code = 0u8;
            if tmp == 0 {
                frame_code = r.r8()?;
                if r.is_eof() {
                    return Err(Error::Eof); // nutdec.c:1163-1164
                }
                if frame_code == b'N' {
                    tmp = u64::from(frame_code);
                    for _ in 1..8 {
                        tmp = (tmp << 8) + u64::from(r.r8()?); // nutdec.c:1167-1169
                    }
                }
            }

            match tmp {
                MAIN_STARTCODE | STREAM_STARTCODE | INDEX_STARTCODE => {
                    // nutdec.c:1175-1177 — skip the packet, unchecked body.
                    let skip = get_packetheader(r, false, tmp)?;
                    if let Err(Error::Eof) = r.skip(skip) {
                        return Err(Error::Eof); // eof latched, like C
                    }
                }
                INFO_STARTCODE => {
                    if self.decode_info_header(r).is_err() {
                        // goto resync
                        if !self.resync(r, pos)? {
                            return Err(Error::InvalidData("no startcode to sync to".into()));
                        }
                    }
                }
                SYNCPOINT_STARTCODE => {
                    match self.decode_syncpoint(r) {
                        Ok(_) => {
                            let frame_code = r.r8()?; // nutdec.c:1185
                            match self.decode_frame(r, frame_code) {
                                Ok(Some(pkt)) => return Ok(pkt),
                                Ok(None) => {} // discarded, keep reading
                                Err(_) => {
                                    if !self.resync(r, pos)? {
                                        return Err(Error::InvalidData(
                                            "no startcode to sync to".into(),
                                        ));
                                    }
                                }
                            }
                        }
                        Err(_) => {
                            if !self.resync(r, pos)? {
                                return Err(Error::InvalidData(
                                    "no startcode to sync to".into(),
                                ));
                            }
                        }
                    }
                }
                0 => {
                    match self.decode_frame(r, frame_code) {
                        Ok(Some(pkt)) => return Ok(pkt),
                        Ok(None) => {}
                        Err(_) => {
                            if !self.resync(r, pos)? {
                                return Err(Error::InvalidData(
                                    "no startcode to sync to".into(),
                                ));
                            }
                        }
                    }
                }
                _ => {
                    if !self.resync(r, pos)? {
                        return Err(Error::InvalidData("no startcode to sync to".into()));
                    }
                }
            }
        }
    }
}

impl Default for NutDemuxer {
    fn default() -> Self {
        NutDemuxer::new()
    }
}

impl Demuxer for NutDemuxer {
    /// `nut_read_header` (`nutdec.c:812-878`) — see
    /// [`NutDemuxer::read_header_impl`].
    fn read_header(&mut self, io: &mut IoContext) -> Result<Stream> {
        let mut r = NutReader::new(io);
        self.read_header_impl(&mut r)
    }

    /// `nut_read_packet` (`nutdec.c:1147-1205`) — see
    /// [`NutDemuxer::read_packet_impl`].
    fn read_packet(&mut self, io: &mut IoContext) -> Result<Packet> {
        let mut r = NutReader::new(io);
        self.read_packet_impl(&mut r)
    }
}

// ---------------------------------------------------------------------
// read_sm_data — nutdec.c:880-995
// ---------------------------------------------------------------------

/// `read_sm_data` (`nutdec.c:880-995`) — the side/metadata block parser.
/// `is_meta` selects C's second call; the parse is identical (C never
/// reads the flag). The port keeps the byte-exact parse (including the
/// `maxpos` bounds and C's warnings) but drops the resulting side data —
/// `Packet` has no side-data fields (see the module map).
fn read_sm_data(r: &mut NutReader, maxpos: u64, _is_meta: bool) -> Result<()> {
    let count = r.get_v()? as i64; // int in C
    let mut skip_start: i64 = 0;
    let mut skip_end: i64 = 0;
    let mut sample_rate: i64 = 0;
    let mut width: i64 = 0;
    let mut height: i64 = 0;

    for _ in 0..count {
        if r.tell() >= maxpos {
            return Err(Error::InvalidData("sm data beyond frame".into())); // nutdec.c:893-894
        }
        let name = match get_str(r, 256) {
            Ok(s) => s,
            Err(e) => {
                log_error!(Some("nut"), "get_str failed while reading sm data");
                return Err(e);
            }
        };
        let value = r.get_s()?;

        if value == -1 {
            let str_value = match get_str(r, 256) {
                Ok(s) => s,
                Err(e) => {
                    log_error!(Some("nut"), "get_str failed while reading sm data");
                    return Err(e);
                }
            };
            log_warning!(Some("nut"), "Unknown string {} / {}", name, str_value);
        } else if value == -2 {
            let type_str = match get_str(r, 256) {
                Ok(s) => s,
                Err(e) => {
                    log_error!(Some("nut"), "get_str failed while reading sm data");
                    return Err(e);
                }
            };
            let value_len = r.get_v()? as i64;
            if value_len < 0 || value_len as u64 >= maxpos - r.tell() {
                return Err(Error::InvalidData("sm data value too long".into())); // nutdec.c:919
            }
            // Palette/Extradata/CodecSpecificSide% would become packet side
            // data (nutdec.c:921-941); the port consumes the bytes and
            // keeps C's "Unknown data" warning for the rest.
            log_warning!(Some("nut"), "Unknown data {} / {}", name, type_str);
            r.skip(value_len as u64)?;
        } else if value == -3 {
            let _typed = r.get_s()?; // type "s" — dropped with the side data
        } else if value == -4 {
            let _typed = r.get_v()? as i64; // type "t"
        } else if value < -4 {
            r.get_s()?;
        } else {
            match name.as_str() {
                "SkipStart" => skip_start = value,
                "SkipEnd" => skip_end = value,
                "Channels" => {} // Ignored (nutdec.c:953-954)
                "SampleRate" => sample_rate = value,
                "Width" => width = value,
                "Height" => height = value,
                _ => log_warning!(Some("nut"), "Unknown integer {}", name),
            }
        }
    }

    // The PARAM_CHANGE / SKIP_SAMPLES side-data blocks (nutdec.c:967-989)
    // are dropped with the rest of the side data.
    let _ = (skip_start, skip_end, sample_rate, width, height);

    if r.tell() >= maxpos {
        return Err(Error::InvalidData("sm data consumed the frame".into())); // nutdec.c:991-992
    }
    Ok(())
}

// ---------------------------------------------------------------------
// startcode scanning + probe — nutdec.c:112-166
// ---------------------------------------------------------------------

/// `find_any_startcode` (`nutdec.c:112-135`) — scan forward (optionally
/// after seeking to `pos`; -1 = stay put) for the next known startcode.
/// Returns 0 (C's sentinel) when EOF is hit first.
fn find_any_startcode(r: &mut NutReader, pos: i64) -> u64 {
    if pos >= 0 {
        // "this may fail if the stream is not seekable, but that should
        // not matter" (nutdec.c:117-119) — failure ignored, keep reading.
        let _ = r.seek(pos as u64);
    }
    let mut state = 0u64;
    while !r.is_eof() {
        match r.r8() {
            Ok(b) => state = (state << 8) | u64::from(b),
            Err(_) => return 0,
        }
        if (state >> 56) != u64::from(b'N') {
            continue;
        }
        match state {
            MAIN_STARTCODE | STREAM_STARTCODE | SYNCPOINT_STARTCODE | INFO_STARTCODE
            | INDEX_STARTCODE => return state, // nutdec.c:125-130
            _ => {}
        }
    }
    0
}

/// `find_startcode` (`nutdec.c:143-153`) — position of `code`'s first byte
/// or `None` (C's -1).
fn find_startcode(r: &mut NutReader, code: u64, pos: i64) -> Option<u64> {
    let mut pos = pos;
    loop {
        let startcode = find_any_startcode(r, pos);
        if startcode == code {
            return Some(r.tell() - 8);
        } else if startcode == 0 {
            return None;
        }
        pos = -1; // continue from the current position
    }
}

/// `nut_probe` (`nutdec.c:155-166`) — the big-endian main startcode
/// anywhere in the first buffer wins outright.
pub fn probe(buf: &[u8]) -> u32 {
    if buf.len() < 9 {
        return 0; // C: `i < buf_size - 8` never true
    }
    let hi = (MAIN_STARTCODE >> 32) as u32;
    let lo = (MAIN_STARTCODE & 0xFFFF_FFFF) as u32;
    for i in 0..buf.len() - 8 {
        if u32::from_be_bytes([buf[i], buf[i + 1], buf[i + 2], buf[i + 3]]) != hi {
            continue;
        }
        if u32::from_be_bytes([buf[i + 4], buf[i + 5], buf[i + 6], buf[i + 7]]) == lo {
            return PROBE_SCORE_MAX;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        codec::pcm::PcmDecoder,
        codec::traits::AudioDecoder,
        format::{testutil::MemHandler, DemuxOptions, InputFormatContext},
        util::samplefmt::SampleFormat,
    };

    // ------------------------------------------------------------------
    // framing: vint/svint round trips (nutenc.c:306-348 + aviobuf.c:919-928)
    // ------------------------------------------------------------------

    fn get_v_from(bytes: &[u8]) -> u64 {
        let mut io = MemHandler::io(bytes);
        let mut r = NutReader::new(&mut io);
        r.get_v().unwrap()
    }

    fn get_s_from(bytes: &[u8]) -> i64 {
        let mut io = MemHandler::io(bytes);
        let mut r = NutReader::new(&mut io);
        r.get_s().unwrap()
    }

    #[test]
    fn vint_round_trips_and_shapes() {
        // Byte shapes: 7 bits per byte, high bit = "more bytes follow".
        let mut v = Vec::new();
        put_v(&mut v, 0);
        assert_eq!(v, vec![0x00]);
        v.clear();
        put_v(&mut v, 127);
        assert_eq!(v, vec![0x7F]);
        v.clear();
        put_v(&mut v, 128);
        assert_eq!(v, vec![0x81, 0x00]); // (128>>7)<<7 | 128&127
        v.clear();
        put_v(&mut v, 255);
        assert_eq!(v, vec![0x81, 0x7F]);
        v.clear();
        put_v(&mut v, 2097152); // 2^21 → 1 + three zero groups
        assert_eq!(v, vec![0x81, 0x80, 0x80, 0x00]);

        // get_v decodes them back (ffio_read_varlen's shift/add loop).
        assert_eq!(get_v_from(&[0x00]), 0);
        assert_eq!(get_v_from(&[0x7F]), 127);
        assert_eq!(get_v_from(&[0x81, 0x00]), 128);
        assert_eq!(get_v_from(&[0x81, 0x7F]), 255);
        assert_eq!(get_v_from(&[0x81, 0x80, 0x80, 0x00]), 2097152);

        for val in [
            0u64, 1, 2, 126, 127, 128, 129, 16383, 16384, 1 << 31,
            (1 << 35) + 12345, u64::MAX - 1, u64::MAX,
        ] {
            let mut v = Vec::new();
            put_v(&mut v, val);
            assert_eq!(get_v_from(&v), val, "round trip of {val}");
        }
    }

    #[test]
    fn svint_zigzag_round_trips() {
        // get_s (nutdec.c:68-76): v = vint+1; odd → -(v>>1), even → v>>1.
        // put_s (nutenc.c:345-348): vint of 2|v| - (v>0).
        assert_eq!(get_s_from(&[0x00]), 0); // u=0 → v=1 odd → 0
        assert_eq!(get_s_from(&[0x01]), 1); // u=1 → v=2 even → 1
        assert_eq!(get_s_from(&[0x02]), -1); // u=2 → v=3 odd → -1
        assert_eq!(get_s_from(&[0x03]), 2);
        assert_eq!(get_s_from(&[0x04]), -2);

        for val in [-1_000_000i64, -3, -2, -1, 0, 1, 2, 3, 1_000_000] {
            let mut v = Vec::new();
            put_s(&mut v, val);
            assert_eq!(get_s_from(&v), val, "round trip of {val}");
        }
    }


    /// Drive `decode_main_header` alone (the reader sits after the
    /// startcode) — read_header's C retry loop (nutdec.c:824-833) would
    /// otherwise wrap the error into "No main startcode found.".
    fn main_header_error(bytes: &[u8]) -> Error {
        let mut io = MemHandler::io(bytes);
        let mut r = NutReader::new(&mut io);
        r.seek(8).unwrap();
        let mut dem = NutDemuxer::new();
        dem.decode_main_header(&mut r).unwrap_err()
    }

    /// Parse the main header, then drive `decode_stream_header` on the
    /// first stream-header packet.
    fn stream_header_error(bytes: &[u8]) -> Error {
        let mut io = MemHandler::io(bytes);
        let mut r = NutReader::new(&mut io);
        let mut dem = NutDemuxer::new();
        assert!(find_startcode(&mut r, MAIN_STARTCODE, 0).is_some());
        dem.decode_main_header(&mut r).unwrap();
        assert!(find_startcode(&mut r, STREAM_STARTCODE, 0).is_some());
        dem.decode_stream_header(&mut r).unwrap_err()
    }

    /// Full read_header, then one syncpoint + one frame through the inner
    /// decoders (read_packet's resync arm would otherwise swallow the
    /// frame error, exactly like C's goto resync).
    fn frame_header_error(bytes: &[u8]) -> Error {
        let mut io = MemHandler::io(bytes);
        let mut r = NutReader::new(&mut io);
        let mut dem = NutDemuxer::new();
        dem.read_header_impl(&mut r).unwrap();
        dem.decode_syncpoint(&mut r).unwrap();
        let code = r.r8().unwrap();
        dem.decode_frame_header(&mut r, code).unwrap_err()
    }

    /// read_header leaves the reader on the (undecoded) syncpoint packet;
    /// drive `decode_syncpoint` alone.
    fn syncpoint_error(bytes: &[u8]) -> Error {
        let mut io = MemHandler::io(bytes);
        let mut r = NutReader::new(&mut io);
        let mut dem = NutDemuxer::new();
        dem.read_header_impl(&mut r).unwrap();
        dem.decode_syncpoint(&mut r).unwrap_err()
    }

    /// Drive `decode_info_header` on the first info packet.
    fn info_header_error(bytes: &[u8]) -> Error {
        let mut io = MemHandler::io(bytes);
        let mut r = NutReader::new(&mut io);
        let mut dem = NutDemuxer::new();
        dem.read_header_impl(&mut r).unwrap();
        // The reader sits on the syncpoint; rewind to the info packet.
        assert!(find_startcode(&mut r, INFO_STARTCODE, 0).is_some());
        dem.decode_info_header(&mut r).unwrap_err()
    }

    // ------------------------------------------------------------------
    // CRC — the AV_CRC_32_IEEE table (crc.c:341, 366-376, 421-452)
    // ------------------------------------------------------------------

    /// Independent reference: the textbook MSB-first CRC-32 (poly
    /// 0x04C11DB7, no reflection, no final xor). FFmpeg's `av_crc` runs the
    /// *byteswapped* register (its table stores `av_bswap32` of the
    /// MSB-first entries, crc.c:375), so its result is `bswap` of the
    /// MSB-first CRC whose *init* is `bswap(init)`.
    fn crc_msb_ref(init: u32, buf: &[u8]) -> u32 {
        let mut crc = init;
        for &b in buf {
            crc ^= (b as u32) << 24;
            for _ in 0..8 {
                crc = if crc & 0x8000_0000 != 0 {
                    (crc << 1) ^ 0x04C1_1DB7
                } else {
                    crc << 1
                };
            }
        }
        crc
    }

    #[test]
    fn crc_matches_ffmpeg_representation() {
        // FFmpeg's value is bswap32 of the MSB-first CRC with the same init.
        for init in [0u32, 0xFFFF_FFFF, 0x1234_5678] {
            for msg in [&b"123456789"[..], &b"nut/multimedia container"[..], &[0u8][..]] {
                assert_eq!(
                    crc04c11db7_update(init, msg),
                    crc_msb_ref(init.swap_bytes(), msg).swap_bytes(),
                    "init {init:#x}"
                );
            }
        }
        // External vector: CRC-32/MPEG-2 (= MSB-first, poly 0x04C11DB7,
        // init 0xFFFFFFFF, no xorout) of "123456789" is 0x0376E6E7; in
        // FFmpeg's representation that is the byteswap.
        assert_eq!(crc_msb_ref(0xFFFF_FFFF, b"123456789"), 0x0376_E6E7);
        assert_eq!(crc04c11db7_update(0xFFFF_FFFF, b"123456789"), 0xE7E6_7603);
        // The table itself: ctx[i] = bswap(MSB-first entry).
        assert_eq!(CRC_TABLE[1], 0x04C1_1DB7u32.swap_bytes());
    }

    #[test]
    fn crc_zero_closure_is_the_packet_checksum_rule() {
        // Writing the running FFmpeg-CRC as 4 little-endian bytes (C's
        // avio_wl32, nutenc.c:363/367) drives the CRC over the extended
        // message to zero — the property get_packetheader's
        // `ffio_get_checksum(bc)` check (nutdec.c:104) relies on.
        for msg in [&b""[..], b"j", b"header body", &[0xABu8; 5000][..]] {
            let crc = crc04c11db7_update(0, msg);
            let mut closed = msg.to_vec();
            closed.extend_from_slice(&crc.to_le_bytes());
            assert_eq!(crc04c11db7_update(0, &closed), 0, "len {}", msg.len());
        }
    }

    // ------------------------------------------------------------------
    // packet header round trip (nutdec.c:92-110 vs nutenc.c:351-370)
    // ------------------------------------------------------------------

    #[test]
    fn put_packet_get_packetheader_round_trip() {
        for size in [4usize, 100, 4092, 4093, 5000] {
            let body: Vec<u8> = (0..size).map(|i| (i % 251) as u8).collect();
            let mut file = Vec::new();
            put_packet(&mut file, &body, MAIN_STARTCODE);

            let mut io = MemHandler::io(&file);
            let mut r = NutReader::new(&mut io);
            r.seek(8).unwrap(); // startcode consumed by the scanner
            let len = get_packetheader(&mut r, true, MAIN_STARTCODE).unwrap();
            assert_eq!(len, body.len() as u64 + 4, "forward ptr (size {size})");

            let mut back = vec![0u8; body.len()];
            r.read(&mut back).unwrap();
            assert_eq!(back, body);
            r.rb32().unwrap(); // the stored checksum, folded in
            assert_eq!(r.get_checksum(), 0, "checksum closes (size {size})");
        }
    }

    #[test]
    fn corrupt_forward_checksum_rejected() {
        let body = vec![7u8; 5000]; // > 4096 → forward checksum present
        let mut file = Vec::new();
        put_packet(&mut file, &body, MAIN_STARTCODE);
        file[10] ^= 0xFF; // inside the forward-checksum region

        let mut io = MemHandler::io(&file);
        let mut r = NutReader::new(&mut io);
        r.seek(8).unwrap();
        assert!(get_packetheader(&mut r, true, MAIN_STARTCODE).is_err());
    }

    #[test]
    fn lsb2full_wraps_around_last_pts() {
        // ff_lsb2full (nut.c:277-282) with shift 4: mask 15, delta =
        // last_pts - 7; result = ((lsb - delta) & 15) + delta.
        assert_eq!(ff_lsb2full(4, 100, 3), 99);
        assert_eq!(ff_lsb2full(4, 100, 4), 100);
        assert_eq!(ff_lsb2full(4, 100, 5), 101);
        // wrap: last_pts 100, lsb 0 → the multiple of 16 nearest below 93.
        assert_eq!(ff_lsb2full(4, 100, 0), 96);
        assert_eq!(ff_lsb2full(4, 100, 15), 95);
        // shift 0: mask 0 → always delta == last_pts.
        assert_eq!(ff_lsb2full(0, 1234, 0), 1234);
    }

    // ------------------------------------------------------------------
    // byte fixtures — a hand-built NUT file, per the nut.h/nutdec.c spec
    // ------------------------------------------------------------------

    /// One frame-code table row as `write_mainheader` (nutenc.c:403-422)
    /// writes it.
    struct FcRow {
        flags: u64,
        fields: u64,
        pts: i64,
        mul: u64,
        stream: u64,
        size: u64,
        res: u64,
        count: Option<u64>,
        head_idx: u64,
    }

    fn write_row(out: &mut Vec<u8>, row: &FcRow) {
        put_v(out, row.flags);
        put_v(out, row.fields);
        if row.fields > 0 {
            put_s(out, row.pts);
        }
        if row.fields > 1 {
            put_v(out, row.mul);
        }
        if row.fields > 2 {
            put_v(out, row.stream);
        }
        if row.fields > 3 {
            put_v(out, row.size);
        }
        if row.fields > 4 {
            put_v(out, row.res);
        }
        if row.fields > 5 {
            put_v(out, row.count.unwrap_or(0));
        }
        if row.fields > 6 {
            put_v(out, 0); // match_time — decoder reads it as get_s
        }
        if row.fields > 7 {
            put_v(out, row.head_idx);
        }
    }

    /// Codes 0..7: KEY|SIZE_MSB|CHECKSUM, pts_delta 1, size_mul 8,
    /// stream `stream`, size_lsb j (payload%8 picked per frame).
    /// Codes 8..255: unused filler.
    fn std_rows(stream: u64, flags: u32, pts_delta: i64) -> Vec<FcRow> {
        vec![
            FcRow {
                flags: flags as u64,
                fields: 4,
                pts: pts_delta,
                mul: 8,
                stream,
                size: 0,
                res: 0,
                count: None,
                head_idx: 0,
            },
            FcRow {
                flags: 0,
                fields: 6,
                pts: 0,
                mul: 1,
                stream: 0,
                size: 0,
                res: 0,
                count: Some(247),
                head_idx: 0,
            },
        ]
    }

    fn write_main(
        out: &mut Vec<u8>,
        streams: u64,
        time_bases: &[(u64, u64)],
        rows: &[FcRow],
        elision: &[&[u8]],
        max_distance: u64,
    ) {
        let mut body = Vec::new();
        put_v(&mut body, 3); // version (NUT_STABLE_VERSION)
        put_v(&mut body, streams);
        put_v(&mut body, max_distance);
        put_v(&mut body, time_bases.len() as u64);
        for &(num, den) in time_bases {
            put_v(&mut body, num);
            put_v(&mut body, den);
        }
        for row in rows {
            write_row(&mut body, row);
        }
        put_v(&mut body, elision.len() as u64); // header_count - 1
        for h in elision {
            put_v(&mut body, h.len() as u64);
            body.extend_from_slice(h);
        }
        put_packet(out, &body, MAIN_STARTCODE);
    }

    fn write_stream_video(out: &mut Vec<u8>, sar: (u64, u64)) {
        let mut body = Vec::new();
        put_v(&mut body, 0); // stream_id
        put_v(&mut body, 0); // class: video
        put_v(&mut body, 4); // fourcc length
        body.extend_from_slice(b"I420");
        put_v(&mut body, 0); // time_base_id
        put_v(&mut body, 15); // msb_pts_shift
        put_v(&mut body, 10240); // max_pts_distance
        put_v(&mut body, 0); // decode_delay
        put_v(&mut body, 0); // stream flags
        put_v(&mut body, 0); // extradata_size
        put_v(&mut body, 64); // width
        put_v(&mut body, 48); // height
        put_v(&mut body, sar.0);
        put_v(&mut body, sar.1);
        put_v(&mut body, 0); // csp type
        put_packet(out, &body, STREAM_STARTCODE);
    }

    fn write_stream_audio(out: &mut Vec<u8>, stream_id: u64) {
        let mut body = Vec::new();
        put_v(&mut body, stream_id);
        put_v(&mut body, 1); // class: audio
        put_v(&mut body, 4); // fourcc length
        body.extend_from_slice(&[b'P', b'S', b'D', 16]); // PCM_S16LE
        put_v(&mut body, 0); // time_base_id
        put_v(&mut body, 15); // msb_pts_shift
        put_v(&mut body, 1 << 20); // max_pts_distance
        put_v(&mut body, 0); // decode_delay
        put_v(&mut body, 0); // stream flags
        put_v(&mut body, 0); // extradata_size
        put_v(&mut body, 48000); // sample_rate
        put_v(&mut body, 1); // samplerate_den
        put_v(&mut body, 2); // channels
        put_packet(out, &body, STREAM_STARTCODE);
    }

    fn write_syncpoint(out: &mut Vec<u8>, ts: u64) {
        let mut body = Vec::new();
        put_v(&mut body, ts); // put_tt: ts * time_base_count + tb_id
        put_v(&mut body, 0); // back_ptr / 16
        put_packet(out, &body, SYNCPOINT_STARTCODE);
    }

    /// A frame for the std table (codes 0..7 of `std_rows`): header is
    /// [code, v: size/8] + crc32, then the payload. `size` (code + 8*msb)
    /// must cover exactly the on-disk payload.
    fn write_frame_std(out: &mut Vec<u8>, payload: &[u8]) {
        debug_assert!(payload.len() % 8 < 8);
        let mut h = vec![payload.len() as u8 % 8]; // frame_code = size_lsb
        put_v(&mut h, payload.len() as u64 / 8); // FLAG_SIZE_MSB
        h.extend_from_slice(&crc04c11db7_update(0, &h).to_le_bytes());
        out.extend_from_slice(&h);
        out.extend_from_slice(payload);
    }

    /// A frame with per-frame coded stream_id + coded_pts + size_msb +
    /// checksum, for a table row with all those flags.
    fn write_frame_coded(out: &mut Vec<u8>, payload: &[u8], stream_id: u64, pts: u64) {
        let mut h = vec![0u8]; // frame_code 0
        put_v(&mut h, stream_id); // FLAG_STREAM_ID
        put_v(&mut h, pts + (1 << 15)); // FLAG_CODED_PTS (else branch)
        put_v(&mut h, payload.len() as u64 / 8); // FLAG_SIZE_MSB (mul 8)
        h.extend_from_slice(&crc04c11db7_update(0, &h).to_le_bytes());
        out.extend_from_slice(&h);
        out.extend_from_slice(payload);
    }

    /// The canonical single-stream rawvideo fixture: 25 fps 64x48 yuv420p,
    /// sar 16/11, syncpoint at ts 0, three key frames.
    fn nut_video_bytes() -> Vec<u8> {
        let mut out = Vec::new();
        write_main(
            &mut out,
            1,
            &[(1, 25)],
            &std_rows(0, flag::KEY | flag::SIZE_MSB | flag::CHECKSUM, 1),
            &[],
            u64::from(MAX_DISTANCE),
        );
        write_stream_video(&mut out, (16, 11));
        write_syncpoint(&mut out, 0);
        write_frame_std(&mut out, &[0x11; 16]);
        write_frame_std(&mut out, &[0x22; 17]);
        write_frame_std(&mut out, &[0x33; 24]);
        out
    }

    // ------------------------------------------------------------------
    // probe (nutdec.c:155-166)
    // ------------------------------------------------------------------

    #[test]
    fn probe_hits_main_startcode() {
        assert_eq!(probe(&nut_video_bytes()), PROBE_SCORE_MAX);
        // The startcode may sit anywhere in the probe buffer.
        let mut buf = vec![0u8; 300];
        buf[100..108].copy_from_slice(&MAIN_STARTCODE.to_be_bytes());
        assert_eq!(probe(&buf), PROBE_SCORE_MAX);
    }

    #[test]
    fn probe_rejects_without_startcode() {
        assert_eq!(probe(b"not a nut file at all........"), 0);
        assert_eq!(probe(&[0x4E, 0x4D]), 0); // too short
        // partial startcode
        let mut buf = vec![0u8; 64];
        buf[..7].copy_from_slice(&MAIN_STARTCODE.to_be_bytes()[..7]);
        assert_eq!(probe(&buf), 0);
        // other startcodes do not count
        let mut buf = vec![0u8; 64];
        buf[..8].copy_from_slice(&INFO_STARTCODE.to_be_bytes());
        assert_eq!(probe(&buf), 0);
    }

    // ------------------------------------------------------------------
    // read_header (nutdec.c:812-878)
    // ------------------------------------------------------------------

    #[test]
    fn header_fills_video_codecpar() {
        let mut io = MemHandler::io(&nut_video_bytes());
        let mut dem = NutDemuxer::new();
        let st = dem.read_header_impl(&mut NutReader::new(&mut io)).unwrap();

        assert_eq!(st.codecpar.codec_type, MediaType::Video);
        assert_eq!(st.codecpar.codec_id, CodecId::Rawvideo);
        assert_eq!(st.codecpar.format, PixelFormat::Yuv420p);
        assert_eq!(st.codecpar.width, 64);
        assert_eq!(st.codecpar.height, 48);
        assert_eq!(st.sample_aspect_ratio, Rational::new(16, 11));
        assert_eq!(st.codecpar.sample_aspect_ratio, Rational::new(16, 11));
        // avpriv_set_pts_info(st, 63, 1, 25) (nutdec.c:479-480).
        assert_eq!(st.time_base, Rational::new(1, 25));
        // no index read → duration unknown.
        assert_eq!(st.duration, NOPTS);
    }

    #[test]
    fn header_fills_audio_fields() {
        let mut out = Vec::new();
        write_main(
            &mut out,
            1,
            &[(1, 48000)],
            &std_rows(0, flag::KEY | flag::SIZE_MSB | flag::CHECKSUM, 1024),
            &[],
            u64::from(MAX_DISTANCE),
        );
        write_stream_audio(&mut out, 0);
        write_syncpoint(&mut out, 0);
        write_frame_std(&mut out, &[0x55; 4096]); // 1024 stereo samples

        let mut io = MemHandler::io(&out);
        let mut dem = NutDemuxer::new();
        let st = dem.read_header_impl(&mut NutReader::new(&mut io)).unwrap();

        assert_eq!(st.codecpar.codec_type, MediaType::Audio);
        assert_eq!(st.codecpar.codec_id, CodecId::PcmS16le);
        assert_eq!(st.codecpar.sample_rate, 48000);
        assert_eq!(st.codecpar.ch_layout, ChannelLayout::unspecified(2));
        assert_eq!(st.codecpar.sample_fmt, SampleFormat::S16);
        assert_eq!(st.time_base, Rational::new(1, 48000));

        // PCM pipeline: packet decodes as 1024 stereo samples.
        let mut dec = PcmDecoder::new();
        dec.init(&st.codecpar).unwrap();
        let pkt = dem
            .read_packet_impl(&mut NutReader::new(&mut io))
            .unwrap();
        assert_eq!(pkt.pts, 1024); // syncpoint reset to 0, delta 1024
        assert_eq!(pkt.duration, 1024);
        dec.send_packet(Some(&pkt)).unwrap();
        let f = dec.receive_frame().unwrap();
        assert_eq!(f.nb_samples, 1024);
        assert_eq!(f.sample_rate, 48000);
    }

    #[test]
    fn frames_carry_pts_duration_flags_payload() {
        let mut io = MemHandler::io(&nut_video_bytes());
        let mut dem = NutDemuxer::new();
        dem.read_header_impl(&mut NutReader::new(&mut io)).unwrap();

        // Syncpoint reset last_pts to 0; every frame is last + pts_delta(1).
        let p0 = dem
            .read_packet_impl(&mut NutReader::new(&mut io))
            .unwrap();
        assert_eq!(p0.size(), 16);
        assert_eq!(p0.pts, 1);
        assert_eq!(p0.dts, NOPTS);
        assert_eq!(p0.duration, 1);
        assert!(p0.flags.contains(PacketFlags::KEY));
        assert_eq!(p0.stream_index, 0);
        assert_eq!(p0.time_base, Rational::new(1, 25));
        assert_eq!(p0.as_slice(), &[0x11; 16]);

        let p1 = dem
            .read_packet_impl(&mut NutReader::new(&mut io))
            .unwrap();
        assert_eq!(p1.size(), 17); // code 1, size 1 + 8*2
        assert_eq!(p1.pts, 2);
        assert_eq!(p1.as_slice(), &[0x22; 17]);

        let p2 = dem
            .read_packet_impl(&mut NutReader::new(&mut io))
            .unwrap();
        assert_eq!(p2.pts, 3);
        assert_eq!(p2.size(), 24);

        // Clean EOF at the end of the file (nutdec.c:1163-1164).
        assert!(matches!(
            dem.read_packet_impl(&mut NutReader::new(&mut io)),
            Err(Error::Eof)
        ));
    }

    #[test]
    fn syncpoint_resets_pts() {
        let mut out = Vec::new();
        write_main(
            &mut out,
            1,
            &[(1, 25)],
            &std_rows(0, flag::KEY | flag::SIZE_MSB | flag::CHECKSUM, 1),
            &[],
            u64::from(MAX_DISTANCE),
        );
        write_stream_video(&mut out, (0, 0));
        write_syncpoint(&mut out, 100); // ts 100 in the 1/25 time base
        write_frame_std(&mut out, &[1; 8]);

        let mut io = MemHandler::io(&out);
        let mut dem = NutDemuxer::new();
        dem.read_header_impl(&mut NutReader::new(&mut io)).unwrap();
        let p = dem
            .read_packet_impl(&mut NutReader::new(&mut io))
            .unwrap();
        assert_eq!(p.pts, 101); // 100 (reset) + pts_delta 1
    }

    // ------------------------------------------------------------------
    // checksum mismatches carry C's texts (nutdec.c:353-357, 472-477, …)
    // ------------------------------------------------------------------

    #[test]
    fn main_header_checksum_mismatch() {
        let mut bytes = nut_video_bytes();
        // Flip a bit of the time-base denominator (body offset 7: the
        // den vint 25 → 24 — still a valid parse, so the checksum is
        // what rejects it).
        bytes[8 + 1 + 7] ^= 0x01;
        // read_header's retry loop would wrap this as "No main startcode
        // found." (C's do/while, nutdec.c:824-833) — pin the inner text.
        assert_eq!(
            main_header_error(&bytes),
            Error::InvalidData("main header checksum mismatch".into())
        );
    }

    #[test]
    fn stream_header_checksum_mismatch() {
        let mut bytes = nut_video_bytes();
        // Corrupt the first stream-header body byte (its startcode is the
        // second startcode in the file).
        let needle = STREAM_STARTCODE.to_be_bytes();
        let pos = bytes
            .windows(8)
            .position(|w| w == needle)
            .expect("stream header present");
        bytes[pos + 12] ^= 0x01; // fourcc '0' → '1': parse stays valid
        assert_eq!(
            stream_header_error(&bytes),
            Error::InvalidData("stream header 0 checksum mismatch".into())
        );
    }

    #[test]
    fn syncpoint_checksum_mismatch() {
        let mut bytes = nut_video_bytes();
        // Corrupt a byte of the syncpoint body (found via its startcode).
        let needle = SYNCPOINT_STARTCODE.to_be_bytes();
        let pos = bytes
            .windows(8)
            .position(|w| w == needle)
            .expect("syncpoint present");
        bytes[pos + 9] ^= 0xFF; // inside the syncpoint body

        // read_header only *looks* for the syncpoint startcode
        // (nutdec.c:856-858); the checksum surfaces when read_packet
        // decodes it — and its resync arm then fails at EOF.
        assert_eq!(
            syncpoint_error(&bytes),
            Error::InvalidData("sync point checksum mismatch".into())
        );
        let mut io = MemHandler::io(&bytes);
        let mut dem = NutDemuxer::new();
        dem.read_header_impl(&mut NutReader::new(&mut io)).unwrap();
        assert_eq!(
            dem.read_packet_impl(&mut NutReader::new(&mut io)).unwrap_err(),
            Error::InvalidData("no startcode to sync to".into())
        );
    }

    #[test]
    fn info_header_is_parsed_and_checksummed() {
        let mut out = Vec::new();
        write_main(
            &mut out,
            1,
            &[(1, 25)],
            &std_rows(0, flag::KEY | flag::SIZE_MSB | flag::CHECKSUM, 1),
            &[],
            u64::from(MAX_DISTANCE),
        );
        write_stream_video(&mut out, (0, 0));
        // info: stream_id_plus1 0, chapter 0, start 0, len 0, two entries.
        let mut info = Vec::new();
        put_v(&mut info, 0);
        put_v(&mut info, 0);
        put_v(&mut info, 0);
        put_v(&mut info, 0);
        put_v(&mut info, 2);
        // UTF-8 entry (add_info, nutenc.c:527-531): name, -1, value.
        put_v(&mut info, 5);
        info.extend_from_slice(b"Title");
        put_s(&mut info, -1);
        put_v(&mut info, 4);
        info.extend_from_slice(b"test");
        // "s" integer entry: name, -3, value.
        put_v(&mut info, 6);
        info.extend_from_slice(b"Marker");
        put_s(&mut info, -3);
        put_s(&mut info, 42);
        put_packet(&mut out, &info, INFO_STARTCODE);
        write_syncpoint(&mut out, 0);
        write_frame_std(&mut out, &[9; 8]);

        let mut io = MemHandler::io(&out);
        let mut dem = NutDemuxer::new();
        dem.read_header_impl(&mut NutReader::new(&mut io)).unwrap();
        let p = dem
            .read_packet_impl(&mut NutReader::new(&mut io))
            .unwrap();
        assert_eq!(p.as_slice(), &[9; 8]); // info consumed, frames follow

        // Corrupt one entry byte → the checksum must reject the packet.
        let needle = INFO_STARTCODE.to_be_bytes();
        let pos = out
            .windows(8)
            .position(|w| w == needle)
            .expect("info present");
        // Flip a bit inside the first entry's name (past the packet's
        // vint fields, so the parse stays valid and only the checksum
        // can catch it).
        out[pos + 15] ^= 0x01; // 'T' of "Title"
        assert_eq!(
            info_header_error(&out),
            Error::InvalidData("info header checksum mismatch".into())
        );
        // read_header tolerates info errors like C (nutdec.c:863 ignores
        // the return); the first frame still demuxes.
        let mut io = MemHandler::io(&out);
        let mut dem = NutDemuxer::new();
        dem.read_header_impl(&mut NutReader::new(&mut io)).unwrap();
        let p = dem
            .read_packet_impl(&mut NutReader::new(&mut io))
            .unwrap();
        assert_eq!(p.as_slice(), &[9; 8]);
    }

    // ------------------------------------------------------------------
    // header error paths with C's texts
    // ------------------------------------------------------------------

    #[test]
    fn no_main_startcode_found() {
        let mut io = MemHandler::io(b"garbage bytes, no startcode here...");
        let mut dem = NutDemuxer::new();
        assert_eq!(
            dem.read_header_impl(&mut NutReader::new(&mut io))
                .unwrap_err(),
            Error::InvalidData("No main startcode found.".into())
        );
    }

    #[test]
    fn not_all_stream_headers_found() {
        let mut out = Vec::new();
        write_main(
            &mut out,
            1,
            &[(1, 25)],
            &std_rows(0, flag::KEY | flag::SIZE_MSB | flag::CHECKSUM, 1),
            &[],
            u64::from(MAX_DISTANCE),
        );
        // no stream header follows
        let mut io = MemHandler::io(&out);
        let mut dem = NutDemuxer::new();
        assert_eq!(
            dem.read_header_impl(&mut NutReader::new(&mut io))
                .unwrap_err(),
            Error::InvalidData("Not all stream headers found.".into())
        );
    }

    #[test]
    fn eof_before_video_frames() {
        let mut out = Vec::new();
        write_main(
            &mut out,
            1,
            &[(1, 25)],
            &std_rows(0, flag::KEY | flag::SIZE_MSB | flag::CHECKSUM, 1),
            &[],
            u64::from(MAX_DISTANCE),
        );
        write_stream_video(&mut out, (0, 0));
        // no syncpoint after the headers
        let mut io = MemHandler::io(&out);
        let mut dem = NutDemuxer::new();
        assert_eq!(
            dem.read_header_impl(&mut NutReader::new(&mut io))
                .unwrap_err(),
            Error::InvalidData("EOF before video frames".into())
        );
    }

    #[test]
    fn version_gate() {
        let mut body = Vec::new();
        put_v(&mut body, 1); // version < NUT_MIN_VERSION
        put_v(&mut body, 1);
        put_v(&mut body, 1024 * 32 - 1);
        put_v(&mut body, 1);
        put_v(&mut body, 1);
        put_v(&mut body, 25);
        let mut out = Vec::new();
        put_packet(&mut out, &body, MAIN_STARTCODE);

        assert_eq!(
            main_header_error(&out),
            Error::Unsupported("Version 1 not supported.".into())
        );
    }

    #[test]
    fn zero_stream_count_rejected() {
        let mut body = Vec::new();
        put_v(&mut body, 3);
        put_v(&mut body, 0); // stream_count — GET_V requires > 0
        let mut out = Vec::new();
        put_packet(&mut out, &body, MAIN_STARTCODE);
        assert_eq!(
            main_header_error(&out),
            Error::InvalidData("Error stream_count is (0)".into())
        );
    }

    #[test]
    fn non_reduced_time_base_rejected() {
        let mut out = Vec::new();
        write_main(
            &mut out,
            1,
            &[(2, 4)], // gcd 2 ≠ 1
            &std_rows(0, flag::KEY | flag::SIZE_MSB | flag::CHECKSUM, 1),
            &[],
            u64::from(MAX_DISTANCE),
        );
        assert_eq!(
            main_header_error(&out),
            Error::InvalidData("invalid time base 2/4".into())
        );
    }

    #[test]
    fn illegal_frame_code_count_rejected() {
        let rows = vec![FcRow {
            flags: 0,
            fields: 6,
            pts: 0,
            mul: 1,
            stream: 0,
            size: 0,
            res: 0,
            count: Some(300), // > 255 available at i = 0
            head_idx: 0,
        }];
        let mut out = Vec::new();
        write_main(&mut out, 1, &[(1, 25)], &rows, &[], 1024 * 32 - 1);
        assert_eq!(
            main_header_error(&out),
            Error::InvalidData("illegal count 300 at 0".into())
        );
    }

    #[test]
    fn unknown_stream_class_rejected() {
        let mut out = Vec::new();
        write_main(
            &mut out,
            1,
            &[(1, 25)],
            &std_rows(0, flag::KEY | flag::SIZE_MSB | flag::CHECKSUM, 1),
            &[],
            u64::from(MAX_DISTANCE),
        );
        let mut body = Vec::new();
        put_v(&mut body, 0); // stream_id
        put_v(&mut body, 7); // class: unknown
        put_v(&mut body, 4);
        body.extend_from_slice(b"I420");
        put_packet(&mut out, &body, STREAM_STARTCODE);

        assert_eq!(
            stream_header_error(&out),
            Error::Unsupported("unknown stream class (7)".into())
        );
    }

    #[test]
    fn invalid_aspect_ratio_rejected() {
        let mut out = Vec::new();
        write_main(
            &mut out,
            1,
            &[(1, 25)],
            &std_rows(0, flag::KEY | flag::SIZE_MSB | flag::CHECKSUM, 1),
            &[],
            u64::from(MAX_DISTANCE),
        );
        write_stream_video(&mut out, (16, 0)); // num set, den zero
        assert_eq!(
            stream_header_error(&out),
            Error::InvalidData("invalid aspect ratio 16/0".into())
        );
    }

    // ------------------------------------------------------------------
    // frame decoding: coded pts, elision headers, guards
    // ------------------------------------------------------------------

    #[test]
    fn coded_pts_and_stream_id_frames() {
        let mut out = Vec::new();
        write_main(
            &mut out,
            1,
            &[(1, 25)],
            &std_rows(
                0,
                flag::KEY | flag::CODED_PTS | flag::STREAM_ID | flag::SIZE_MSB | flag::CHECKSUM,
                7,
            ),
            &[],
            u64::from(MAX_DISTANCE),
        );
        write_stream_video(&mut out, (0, 0));
        write_syncpoint(&mut out, 0);
        write_frame_coded(&mut out, &[0xAA; 16], 0, 40); // pts 40, size 0+8*2
        write_frame_coded(&mut out, &[0xBB; 24], 0, 41);
        // A frame whose coded pts < (1<<15) exercises ff_lsb2full.
        let mut h = vec![0u8];
        put_v(&mut h, 0); // stream id
        put_v(&mut h, 39); // coded_pts < 1<<15 → lsb path, last_pts 41
        put_v(&mut h, 24 / 8);
        h.extend_from_slice(&crc04c11db7_update(0, &h).to_le_bytes());
        out.extend_from_slice(&h);
        out.extend_from_slice(&[0xCC; 24]);

        let mut io = MemHandler::io(&out);
        let mut dem = NutDemuxer::new();
        dem.read_header_impl(&mut NutReader::new(&mut io)).unwrap();
        let p0 = dem
            .read_packet_impl(&mut NutReader::new(&mut io))
            .unwrap();
        assert_eq!(p0.pts, 40);
        assert_eq!(p0.duration, 7); // frame-code pts_delta
        let p1 = dem
            .read_packet_impl(&mut NutReader::new(&mut io))
            .unwrap();
        assert_eq!(p1.pts, 41);
        // lsb path: last_pts 41, shift 15 → mask 32767, delta = 41 - 16383;
        // ((39 - delta) & mask) + delta = 39.
        let p2 = dem
            .read_packet_impl(&mut NutReader::new(&mut io))
            .unwrap();
        assert_eq!(p2.pts, 39);
        assert_eq!(p2.as_slice(), &[0xCC; 24]);
    }

    #[test]
    fn elision_headers_are_prepended() {
        let elision: &[&[u8]] = &[&[0xDE, 0xAD, 0xBE, 0xEF]];
        let mut out = Vec::new();
        // codes 0..7 carry header_idx 1 (fields = 8).
        let rows = vec![
            FcRow {
                flags: flag::KEY as u64 | flag::SIZE_MSB as u64 | flag::CHECKSUM as u64,
                fields: 8,
                pts: 1,
                mul: 8,
                stream: 0,
                size: 0,
                res: 0,
                count: Some(8),
                head_idx: 1,
            },
            FcRow {
                flags: 0,
                fields: 6,
                pts: 0,
                mul: 1,
                stream: 0,
                size: 0,
                res: 0,
                count: Some(247),
                head_idx: 0,
            },
        ];
        write_main(&mut out, 1, &[(1, 25)], &rows, elision, u64::from(MAX_DISTANCE));
        write_stream_video(&mut out, (0, 0));
        write_syncpoint(&mut out, 0);

        // Coded size covers elision (4) + on-disk payload (12): 16.
        let mut h = vec![0u8]; // code 0: lsb 0, size 0 + 8*2
        put_v(&mut h, 16 / 8);
        h.extend_from_slice(&crc04c11db7_update(0, &h).to_le_bytes());
        out.extend_from_slice(&h);
        out.extend_from_slice(&[0x77; 12]);

        let mut io = MemHandler::io(&out);
        let mut dem = NutDemuxer::new();
        dem.read_header_impl(&mut NutReader::new(&mut io)).unwrap();
        let p = dem
            .read_packet_impl(&mut NutReader::new(&mut io))
            .unwrap();
        assert_eq!(p.size(), 16); // 4 elided + 12 on disk
        assert_eq!(
            p.as_slice(),
            &[0xDE, 0xAD, 0xBE, 0xEF, 0x77, 0x77, 0x77, 0x77, 0x77, 0x77, 0x77, 0x77, 0x77, 0x77, 0x77, 0x77]
        );
    }

    #[test]
    fn header_idx_invalid_rejected() {
        let mut out = Vec::new();
        // Row with FLAG_HEADER_IDX coded per frame.
        let rows = vec![
            FcRow {
                flags: flag::KEY as u64
                    | flag::SIZE_MSB as u64
                    | flag::CHECKSUM as u64
                    | flag::HEADER_IDX as u64,
                fields: 4,
                pts: 1,
                mul: 8,
                stream: 0,
                size: 0,
                res: 0,
                count: None,
                head_idx: 0,
            },
            FcRow {
                flags: 0,
                fields: 6,
                pts: 0,
                mul: 1,
                stream: 0,
                size: 0,
                res: 0,
                count: Some(247),
                head_idx: 0,
            },
        ];
        write_main(&mut out, 1, &[(1, 25)], &rows, &[], u64::from(MAX_DISTANCE));
        write_stream_video(&mut out, (0, 0));
        write_syncpoint(&mut out, 0);
        // header_count is 1 → any coded idx ≥ 1 is invalid.
        let mut h = vec![0u8];
        put_v(&mut h, 16 / 8); // size_msb
        put_v(&mut h, 1); // FLAG_HEADER_IDX
        h.extend_from_slice(&crc04c11db7_update(0, &h).to_le_bytes());
        out.extend_from_slice(&h);
        out.extend_from_slice(&[0x88; 16]);

        assert_eq!(
            frame_header_error(&out),
            Error::InvalidData("header_idx invalid".into())
        );
    }

    #[test]
    fn oversize_frame_without_checksum_rejected() {
        let mut out = Vec::new();
        let rows = std_rows(0, flag::KEY | flag::SIZE_MSB, 1); // no CHECKSUM
        write_main(&mut out, 1, &[(1, 25)], &rows, &[], u64::from(MAX_DISTANCE));
        write_stream_video(&mut out, (0, 0));
        write_syncpoint(&mut out, 0);
        // Claim size 0 + 8*10000 = 80000 > 2*max_distance (65534).
        let mut h = vec![0u8];
        put_v(&mut h, 10000);
        out.extend_from_slice(&h);
        out.extend_from_slice(&[0u8; 16]);

        assert_eq!(
            frame_header_error(&out),
            Error::InvalidData("frame size > 2max_distance and no checksum".into())
        );
    }

    #[test]
    fn damaged_distance_guard_fires() {
        let mut out = Vec::new();
        write_main(
            &mut out,
            1,
            &[(1, 25)],
            &std_rows(0, flag::KEY | flag::SIZE_MSB | flag::CHECKSUM, 1),
            &[],
            0, // max_distance 0: everything beyond the syncpoint is "damaged"
        );
        write_stream_video(&mut out, (0, 0));
        write_syncpoint(&mut out, 0);
        write_frame_std(&mut out, &[1; 8]);

        let err = frame_header_error(&out);
        match err {
            Error::InvalidData(msg) => {
                assert!(msg.starts_with("Last frame must have been damaged"), "{msg}")
            }
            other => panic!("unexpected error {other:?}"),
        }
    }

    #[test]
    fn frame_code_n_is_invalid_and_resyncs_to_eof() {
        let mut out = nut_video_bytes();
        out.push(b'N'); // 0x4E — reserved, FLAG_INVALID
        let mut io = MemHandler::io(&out);
        let mut dem = NutDemuxer::new();
        dem.read_header_impl(&mut NutReader::new(&mut io)).unwrap();
        // The three frames decode, then 'N' fails and resync finds nothing.
        for _ in 0..3 {
            dem.read_packet_impl(&mut NutReader::new(&mut io))
                .unwrap();
        }
        assert_eq!(
            dem.read_packet_impl(&mut NutReader::new(&mut io))
                .unwrap_err(),
            Error::InvalidData("no startcode to sync to".into())
        );
    }

    #[test]
    fn sm_data_frames_parse_and_drop_side_data() {
        let mut out = Vec::new();
        write_main(
            &mut out,
            1,
            &[(1, 25)],
            &std_rows(
                0,
                flag::KEY | flag::SM_DATA | flag::SIZE_MSB | flag::CHECKSUM,
                1,
            ),
            &[],
            u64::from(MAX_DISTANCE),
        );
        write_stream_video(&mut out, (0, 0));
        write_syncpoint(&mut out, 0);

        // sm block 1: one integer entry ("SkipStart" = 5); block 2: empty.
        let mut sm = Vec::new();
        put_v(&mut sm, 1);
        put_v(&mut sm, 9);
        sm.extend_from_slice(b"SkipStart");
        put_s(&mut sm, 5);
        let mut sm2 = Vec::new();
        put_v(&mut sm2, 0);
        let payload = [0x99; 8];
        let total = sm.len() + sm2.len() + payload.len();

        let mut h = vec![total as u8 % 8];
        put_v(&mut h, total as u64 / 8);
        h.extend_from_slice(&crc04c11db7_update(0, &h).to_le_bytes());
        out.extend_from_slice(&h);
        out.extend_from_slice(&sm);
        out.extend_from_slice(&sm2);
        out.extend_from_slice(&payload);

        let mut io = MemHandler::io(&out);
        let mut dem = NutDemuxer::new();
        dem.read_header_impl(&mut NutReader::new(&mut io)).unwrap();
        let p = dem
            .read_packet_impl(&mut NutReader::new(&mut io))
            .unwrap();
        // The sm bytes are consumed out of the coded size; the packet
        // carries only the payload (side data dropped, see module map).
        assert_eq!(p.as_slice(), &payload);
        assert_eq!(p.pts, 1);
    }

    // ------------------------------------------------------------------
    // multi-stream degradation: stream 0 surfaced, others dropped
    // ---------------------------------------------------------------------

    #[test]
    fn multi_stream_file_surfaces_stream0_only() {
        let mut out = Vec::new();
        // Two streams (video + audio tbs), codes 0..7 → stream 0,
        // codes 8..15 → stream 1, rest filler.
        let rows = vec![
            FcRow {
                flags: flag::KEY as u64 | flag::SIZE_MSB as u64 | flag::CHECKSUM as u64,
                fields: 4,
                pts: 1,
                mul: 8,
                stream: 0,
                size: 0,
                res: 0,
                count: None,
                head_idx: 0,
            },
            FcRow {
                flags: flag::KEY as u64 | flag::SIZE_MSB as u64 | flag::CHECKSUM as u64,
                fields: 4,
                pts: 3,
                mul: 8,
                stream: 1,
                size: 0,
                res: 0,
                count: None,
                head_idx: 0,
            },
            FcRow {
                flags: 0,
                fields: 6,
                pts: 0,
                mul: 1,
                stream: 0,
                size: 0,
                res: 0,
                count: Some(239),
                head_idx: 0,
            },
        ];
        write_main(&mut out, 2, &[(1, 25)], &rows, &[], u64::from(MAX_DISTANCE));
        write_stream_video(&mut out, (0, 0));
        write_stream_audio(&mut out, 1);
        write_syncpoint(&mut out, 0);

        // frame(stream0), frame(stream1), frame(stream0).
        let f = |out: &mut Vec<u8>, base: u8, payload: &[u8]| {
            let code = 0u8 + payload.len() as u8 % 8; // stream 0 row
            let mut h = vec![code];
            put_v(&mut h, payload.len() as u64 / 8);
            h.extend_from_slice(&crc04c11db7_update(0, &h).to_le_bytes());
            out.extend_from_slice(&h);
            out.extend_from_slice(payload);
            let _ = base;
        };
        f(&mut out, 0, &[0xA0; 16]);
        // stream 1 frame: codes 8..15 (size_lsb = 8 + payload%8).
        {
            let mut h = vec![8 + 16 % 8];
            put_v(&mut h, 16 / 8);
            h.extend_from_slice(&crc04c11db7_update(0, &h).to_le_bytes());
            out.extend_from_slice(&h);
            out.extend_from_slice(&[0xB0; 16]);
        }
        f(&mut out, 0, &[0xC0; 24]);

        let mut io = MemHandler::io(&out);
        let mut dem = NutDemuxer::new();
        let st = dem.read_header_impl(&mut NutReader::new(&mut io)).unwrap();
        assert_eq!(st.codecpar.codec_type, MediaType::Video); // stream 0

        // Stream-1 frames are decoded (their last_pts advances) but their
        // packets dropped; stream 0's own pts sequence is untouched by them.
        let p0 = dem
            .read_packet_impl(&mut NutReader::new(&mut io))
            .unwrap();
        assert_eq!(p0.as_slice(), &[0xA0; 16]);
        assert_eq!(p0.pts, 1);
        let p1 = dem
            .read_packet_impl(&mut NutReader::new(&mut io))
            .unwrap();
        assert_eq!(p1.as_slice(), &[0xC0; 24]);
        assert_eq!(p1.pts, 2); // stream 0's second frame, delta 1

        assert!(matches!(
            dem.read_packet_impl(&mut NutReader::new(&mut io)),
            Err(Error::Eof)
        ));
    }

    // ------------------------------------------------------------------
    // the full pipeline through the registry (probe → open → packets)
    // ------------------------------------------------------------------

    #[test]
    fn pipeline_open_probe_and_read() {
        let bytes = nut_video_bytes();
        assert_eq!(probe(&bytes), PROBE_SCORE_MAX);

        // No .nut extension → forces the probe path in
        // InputFormatContext::open.
        let path = std::env::temp_dir().join(format!(
            "ffmpeg_rs_nut_demux_{}.bin",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).unwrap();

        let mut ictx = InputFormatContext::open(
            path.to_str().unwrap(),
            None,
            &DemuxOptions::default(),
        )
        .unwrap();
        assert_eq!(ictx.iformat.name, "nut");
        let st = &ictx.streams[0];
        assert_eq!(st.codecpar.codec_id, CodecId::Rawvideo);
        assert_eq!(st.codecpar.format, PixelFormat::Yuv420p);
        assert_eq!(st.codecpar.width, 64);
        assert_eq!(st.codecpar.height, 48);

        let p = ictx.read_frame().unwrap();
        assert_eq!(p.size(), 16);
        assert_eq!(p.pts, 1);
        assert_eq!(p.duration, 1);
        assert!(p.flags.contains(PacketFlags::KEY));
        let p = ictx.read_frame().unwrap();
        assert_eq!(p.pts, 2);
        let p = ictx.read_frame().unwrap();
        assert_eq!(p.pts, 3);

        assert!(matches!(ictx.read_frame(), Err(Error::Eof)));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn forced_format_by_name() {
        let path = std::env::temp_dir().join(format!(
            "ffmpeg_rs_nut_forced_{}.nut",
            std::process::id()
        ));
        std::fs::write(&path, nut_video_bytes()).unwrap();
        let ictx = InputFormatContext::open(
            path.to_str().unwrap(),
            Some("nut"),
            &DemuxOptions::default(),
        )
        .unwrap();
        assert_eq!(ictx.iformat.name, "nut");
        std::fs::remove_file(&path).ok();
    }
}
