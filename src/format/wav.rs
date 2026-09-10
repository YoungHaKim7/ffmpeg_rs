//! WAV demuxer — port of `libavformat/wavdec.c` (the RIFF/WAVE core) plus
//! the pieces it leans on: `ff_get_wav_header` (`riffdec.c`), the WAV
//! codec-tag map (`ff_codec_wav_tags`, `riff.c`) and `ff_get_pcm_codec_id`
//! (`libavformat/utils.c`).
//!
//! ## C → Rust map
//!
//! | C | here |
//! |---|---|
//! | `WAVDemuxContext` (`wavdec.c:51-69`) | [`WavDemuxer`] (the RIFF-reachable fields) |
//! | `wav_probe` (`wavdec.c:161-178`) | [`probe`] |
//! | `wav_read_header` (`wavdec.c:362-700`) | [`WavDemuxer::read_header`] |
//! | `wav_parse_fmt_tag` (`wavdec.c:189-206`) | folded into `read_header` |
//! | `ff_get_wav_header` (`riffdec.c:141-275`) | [`get_wav_header`] |
//! | `parse_waveformatex` (`riffdec.c:62-92`) | [`parse_waveformatex`] |
//! | `compute_bitrate` (`riffdec.c:99-138`) | [`compute_bitrate`] |
//! | `ff_wav_codec_get_id` (`riffdec.c:277-292`) | [`wav_codec_get_id`] |
//! | `ff_codec_wav_tags` (`riff.c:525-625`) | [`WAV_CODEC_TAGS`] (subset) |
//! | `ff_get_pcm_codec_id` (`utils.c:154-198`) | [`get_pcm_codec_id`] |
//! | `set_max_size` (`wavdec.c:82-88`) | end of `read_header` |
//! | `ff_pcm_default_packet_size` (`libavformat/pcm.c:29-55`) | [`pcm_default_packet_size`] |
//! | `wav_read_packet` (`wavdec.c:724-815`) | [`WavDemuxer::read_packet`] |
//!
//! ## The core path, exactly as C walks it
//!
//! ```text
//! RIFF <u32 size> WAVE
//!   fmt  <u32 size> WAVEFORMATEX[/EXTENSIBLE]   → get_wav_header:
//!        tag ─► codec_id (tag table + bps refinement),
//!        channels/rate/byte_rate/block_align, mask ─► ch_layout
//!   …unknown chunks skipped at even offsets…
//!   data <u32 size> → data_end bounds
//! → packets: block-aligned slices of the data chunk, sized by
//!   ff_pcm_default_packet_size (~1/10 s, power-of-two sample blocks)
//! ```
//!
//! ## Out of scope (documented divergences, all guarded in C)
//!
//! | C path | Guard / reason |
//! |---|---|
//! | W64 (`w64_read_header`, `wavdec.c:885-1021`), RF64/BW64 `ds64` (`wavdec.c:409-428, 469-471`) | separate container dialects; `read_header` returns `Error::Unsupported`, and `probe` does not match them (C's `wav_probe` arm at `wavdec.c:172-175`) |
//! | RIFX big-endian params (`wav->rifx`, `wavdec.c:385-387, 503`; `ff_get_wav_header`'s `big_endian` arms, `riffdec.c:163-177`) | `probe` still matches (C does), `read_header` rejects with `Error::Unsupported` |
//! | metadata: `LIST`/`INFO` (`wavdec.c:542-579`), `bext` (`wavdec.c:254-351`), `ID3 ` (`wavdec.c:580-591`), `cue` chapters (`wavdec.c:592-612`), metadata conversion (`wavdec.c:693-694`) | chunks are skipped (seeked past) like any unknown tag; no metadata dictionary in the port yet |
//! | SMV video-in-wav (`wavdec.c:499-541, 734-781`) | single-stream `Demuxer` trait |
//! | `XMA2` chunk (`wavdec.c:208-252`) and fmt tag `0x0165` (`riffdec.c:157-162, 223-238`) | `CodecId` has no XMA; `0x0165` → `Error::Unsupported`, an `XMA2`-only file fails C's "no fmt tag" check the same way |
//! | HEAAC `0x1610` (`riffdec.c:198-212`), extradata storage (`ff_get_extradata`, `riffdec.c:214`) | `CodecParameters` has no extradata field; leftover `cbSize` bytes are skipped |
//! | `fact` chunk (`wavdec.c:491-494`) | sample count is recomputed from `data_size` (C's primary path for PCM, `wavdec.c:662-669`), identical result |
//! | SPDIF probe (`wavdec.c:90-119`), `ignore_length`/`max_size` AVOptions (`wavdec.c:73-80`) | default behavior only |
//! | chained `data` chunks (`find_tag`, `wavdec.c:142-159, 786-800`) | EOF at `data_end` instead |
//! | `WAVEFORMATEXTENSIBLE` GUID subformats (`ff_codec_wav_guids`, `riff.c:648-657`) | none of those codec ids is in the family → `CodecId::None` (C's `AV_CODEC_ID_NONE` + "unknown subformat" warning) |
//! | sample-count sanity checks vs `fact`/bit_rate (`wavdec.c:640-660`), F16LE/F24LE/XMA/ADPCM block fix-ups (`wavdec.c:674-691`) | need `fact`/extradata/absent codec ids |
//! | `wav->unaligned` odd-offset handling (`wavdec.c:67, 134-139, 147-148`) | ID3-prefixed inputs are out; chunks are opened at offset 0, so the even-align adjustment reduces to `next_tag_ofs` (already padded by `size + (size & 1)`) |
//!
//! Packet timing note: C's `wav_read_packet` leaves `pts`/`duration` unset
//! (the generic demux layer fills them from the stream's `cur_dts`, which
//! `ff_pcm_read_seek` computes as `pos * time_base.den / byte_rate` =
//! `pos / block_align`, `libavformat/pcm.c:103`). This port sets
//! `pts = (pos - data_ofs) / block_align` and
//! `duration = size / block_align` directly — the same numbers, without
//! the generic layer. `AV_PKT_FLAG_KEY` stays unset, as in C.

use crate::{
    NOPTS,
    codec::{
        packet::Packet,
        params::{CodecId, CodecParameters, MediaType},
        pcm,
    },
    log_warning,
    util::{
        channel_layout::ChannelLayout,
        error::{Error, Result},
        rational::Rational,
    },
};

use super::{
    Stream,
    demux::{Demuxer, PROBE_SCORE_MAX},
    io::IoContext,
};

// ---------------------------------------------------------------------
// RIFF fourccs (MKTAG / av_fourcc2str equivalents)
// ---------------------------------------------------------------------

/// `MKTAG(a,b,c,d)` — little-endian fourcc, as C compares the u32s.
const fn fourcc(s: &[u8; 4]) -> u32 {
    u32::from_le_bytes(*s)
}

/// The fourcc rendered C-style (`av_fourcc2str`) for error messages.
fn fourcc_str(tag: u32) -> String {
    tag.to_le_bytes().iter().map(|&b| b as char).collect()
}

const TAG_RIFF: u32 = fourcc(b"RIFF");
const TAG_RIFX: u32 = fourcc(b"RIFX");
const TAG_RF64: u32 = fourcc(b"RF64");
const TAG_BW64: u32 = fourcc(b"BW64");
const TAG_WAVE: u32 = fourcc(b"WAVE");
const TAG_FMT: u32 = fourcc(b"fmt ");
const TAG_DATA: u32 = fourcc(b"data");

// ---------------------------------------------------------------------
// avio little-endian readers (zero-fill at EOF, like avio_rl*)
// ---------------------------------------------------------------------

fn rl16(io: &mut IoContext) -> Result<u16> {
    let mut b = [0u8; 2];
    io.read(&mut b)?;
    Ok(u16::from_le_bytes(b))
}

fn rl32(io: &mut IoContext) -> Result<u32> {
    let mut b = [0u8; 4];
    io.read(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

// ---------------------------------------------------------------------
// Codec tag map — riff.c:525-625 + riffdec.c:277-292
// ---------------------------------------------------------------------

/// `ff_codec_wav_tags` (`riff.c:525-625`) — every row whose codec id is in
/// the `CodecId` family. First match wins (`ff_codec_get_id`), so the C
/// table order (`PCM_S16LE` before `PCM_U8` under tag 0x0001, etc.) is
/// preserved. All other C rows (ADPCM, MP3, WMA, …) map to
/// `CodecId::None`, exactly C's `AV_CODEC_ID_NONE` for unknown tags.
const WAV_CODEC_TAGS: &[(CodecId, u32)] = &[
    (CodecId::PcmS16le, 0x0001),
    (CodecId::PcmF32le, 0x0003),
    (CodecId::PcmAlaw, 0x0006),
    (CodecId::PcmMulaw, 0x0007),
    // ('u' << 8) | 'l' — the "rogue" mu-law tag (riff.c:602).
    (CodecId::PcmMulaw, 0x6c75),
];

/// `ff_get_pcm_codec_id` (`libavformat/utils.c:154-198`) — LE arms only
/// (the WAV call sites pass `be = 0`). Where C's answer is a PCM flavor
/// outside the `CodecId` family the function returns `CodecId::None` —
/// notably `PCM_S8` (signed 1-byte, unreachable from the WAV tag path
/// because `sflags = ~1` clears the 1-byte sign bit) and `PCM_S64LE`
/// (bps 64), plus the unsigned 2/3/4-byte flavors (same reason).
fn get_pcm_codec_id(bps: i32, flt: bool, sflags: i32) -> CodecId {
    if bps <= 0 || bps > 64 {
        return CodecId::None;
    }
    if flt {
        return match bps {
            32 => CodecId::PcmF32le,
            64 => CodecId::PcmF64le,
            _ => CodecId::None,
        };
    }
    let bytes = (bps + 7) >> 3; // 1..=8
    if sflags & (1 << (bytes - 1)) != 0 {
        match bytes {
            1 => CodecId::None, // AV_CODEC_ID_PCM_S8 — not in the family
            2 => CodecId::PcmS16le,
            3 => CodecId::PcmS24le,
            4 => CodecId::PcmS32le,
            8 => CodecId::None, // AV_CODEC_ID_PCM_S64LE — not in the family
            _ => CodecId::None,
        }
    } else {
        match bytes {
            1 => CodecId::PcmU8,
            // U16/U24/U32 LE — not in the family (unreachable from WAV)
            _ => CodecId::None,
        }
    }
}

/// `ff_wav_codec_get_id` (`riffdec.c:277-292`): tag-table lookup, then the
/// two bps refinements (`PCM_S16LE` → signed integer flavor by bps,
/// `PCM_F32LE` → float flavor by bps). The `ADPCM_ZORK` twist is out with
/// ADPCM.
pub fn wav_codec_get_id(tag: u32, bps: i32) -> CodecId {
    let Some(&(id, _)) = WAV_CODEC_TAGS.iter().find(|&&(_, t)| t == tag) else {
        return CodecId::None;
    };
    match id {
        CodecId::PcmS16le => get_pcm_codec_id(bps, false, !1),
        CodecId::PcmF32le => get_pcm_codec_id(bps, true, 0),
        other => other,
    }
}

// ---------------------------------------------------------------------
// ff_get_wav_header — riffdec.c:141-275
// ---------------------------------------------------------------------

/// What `ff_get_wav_header` learned (C scatters it over `AVCodecParameters`
/// + locals; `bits_per_coded_sample`/`codec_tag` have no codecpar field in
/// the port and live only inside [`get_wav_header`], where the tag map and
/// the bit-rate consistency check need them).
#[derive(Debug, Clone)]
struct WavFormat {
    codec_id: CodecId,
    sample_rate: i32,
    channels: usize,
    block_align: i32,
    bit_rate: i64,
    ch_layout: ChannelLayout,
}

/// `parse_waveformatex` (`riffdec.c:62-92`) — the tail 22 bytes of a
/// `WAVEFORMATEXTENSIBLE`: valid-bits override, channel mask, subformat
/// GUID. Returns `(mask, codec_id)`.
fn parse_waveformatex(io: &mut IoContext, bits: &mut i32) -> Result<(u64, CodecId)> {
    // wValidBitsPerSample — 0 keeps the container's value (riffdec.c:68-70).
    let bps = rl16(io)? as i32;
    if bps != 0 {
        *bits = bps;
    }
    let mask = rl32(io)? as u64; // dwChannelMask

    let mut subformat = [0u8; 16];
    io.read(&mut subformat)?;

    // The three 12-byte "base GUID" tails C accepts (riff.h:115-122).
    const MEDIASUBTYPE_BASE: [u8; 12] = [
        0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
    ];
    const AMBISONIC_BASE: [u8; 12] = [
        0x21, 0x07, 0xD3, 0x11, 0x86, 0x44, 0xC8, 0xC1, 0xCA, 0x00, 0x00, 0x00,
    ];
    const BROKEN_BASE: [u8; 12] = [
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA,
    ];
    let tail: &[u8] = &subformat[4..];
    if *tail == MEDIASUBTYPE_BASE || *tail == AMBISONIC_BASE || *tail == BROKEN_BASE {
        // codec_tag = the GUID's first u32 (little-endian), then the
        // ordinary tag→id path with the (possibly overridden) bps
        // (riffdec.c:82-84).
        let tag = u32::from_le_bytes([subformat[0], subformat[1], subformat[2], subformat[3]]);
        Ok((mask, wav_codec_get_id(tag, *bits)))
    } else {
        // ff_codec_wav_guids (riff.c:648-657) — AC3/EAC3/MP2/ATRAC/DFPWM,
        // none in the family → C's AV_CODEC_ID_NONE + warning.
        log_warning!(Some("wav"), "unknown subformat GUID");
        Ok((mask, CodecId::None))
    }
}

/// `compute_bitrate` (`riffdec.c:99-138`) — the deterministic-PCM arm: the
/// byte rate a consistent PCM header must carry. 0 = "not handled" (C's
/// `default:` and the mismatch bail-out).
fn compute_bitrate(
    codec_id: CodecId,
    sample_rate: i32,
    block_align: i32,
    bits: i32,
    channels: usize,
) -> i64 {
    if sample_rate <= 0 || block_align <= 0 || channels == 0 {
        return 0;
    }
    match codec_id {
        CodecId::PcmU8
        | CodecId::PcmS16le
        | CodecId::PcmS16be
        | CodecId::PcmS24le
        | CodecId::PcmS24be
        | CodecId::PcmS32le
        | CodecId::PcmS32be
        | CodecId::PcmF32le
        | CodecId::PcmF32be
        | CodecId::PcmF64le
        | CodecId::PcmF64be
        | CodecId::PcmAlaw
        | CodecId::PcmMulaw => {
            let expected_align = ((bits + 7) / 8) * channels as i32;
            if block_align != expected_align {
                return 0;
            }
            sample_rate as i64 * block_align as i64 * 8
        }
        _ => 0,
    }
}

/// `ff_get_wav_header` (`riffdec.c:141-275`), little-endian subset.
/// `size` is the fmt chunk's declared size (bytes after the chunk header):
/// 14 = plain `WAVEFORMAT`, 16 = `PCMWAVEFORMAT`, ≥ 18 = `WAVEFORMATEX`
/// (`cbSize`), 18+22 with tag `0xFFFE` = `WAVEFORMATEXTENSIBLE`.
fn get_wav_header(io: &mut IoContext, size: i64) -> Result<WavFormat> {
    if size < 14 {
        // avpriv_request_sample + AVERROR_INVALIDDATA (riffdec.c:147-150).
        return Err(Error::InvalidData("wav header size < 14".into()));
    }

    let id = rl16(io)? as u32;
    if id == 0x0165 {
        // XMA fmt — riffdec.c:157-162, 223-238.
        return Err(Error::Unsupported(
            "XMA fmt tag 0x0165 is not ported".into(),
        ));
    }
    let channels = rl16(io)? as usize;
    let sample_rate = rl32(io)? as i32;
    let byte_rate = rl32(io)? as i64; // nAvgBytesPerSec
    let block_align = rl16(io)? as i32;

    // Plain WAVEFORMAT has no wBitsPerSample field (riffdec.c:170-178).
    let mut bits = if size == 14 { 8 } else { rl16(io)? as i32 };

    let mut codec_id = CodecId::None;
    let mut ch_layout = ChannelLayout::default();

    // The tag→codec refinement happens before cbSize parsing (riffdec.c:179-185);
    // 0xFFFE is deferred to the subformat GUID below.
    if id != 0xFFFE {
        codec_id = wav_codec_get_id(id, bits);
    }

    if size >= 18 {
        let mut cb_size = rl16(io)? as i64; // cbSize
        let mut size = size - 18;
        cb_size = cb_size.min(size); // FFMIN(size, cbSize) (riffdec.c:192-193)
        if cb_size >= 22 && id == 0xfffe {
            // WAVEFORMATEXTENSIBLE (riffdec.c:194-197).
            let (mask, cid) = parse_waveformatex(io, &mut bits)?;
            codec_id = cid;
            // av_channel_layout_from_mask (riffdec.c:72-73); mask 0 fails
            // in C and leaves the layout unset — unwrap_or_default.
            ch_layout = ChannelLayout::from_mask(mask).unwrap_or_default();
            cb_size -= 22;
            size -= 22;
        }
        if cb_size > 0 {
            // ff_get_extradata (riffdec.c:213-218) — no extradata field in
            // the port's CodecParameters; skip the bytes.
            io.skip(cb_size as u64).ok();
        }
        // "It is possible for the chunk to contain garbage at the end"
        // (riffdec.c:220-222).
        if size > 0 {
            io.skip(size as u64).ok();
        }
    }

    let mut bit_rate = byte_rate * 8; // riffdec.c:160 (bitrate = rl32 * 8LL)

    if sample_rate <= 0 {
        // riffdec.c:242-246.
        return Err(Error::InvalidData(format!(
            "Invalid sample rate: {sample_rate}"
        )));
    }

    // "ignore WAVEFORMATEXTENSIBLE layout if different from channel count"
    // (riffdec.c:258-263): a mask whose popcount (or 0) disagrees with
    // nChannels falls back to an UNSPEC layout of nChannels.
    if ch_layout.nb_channels != channels {
        ch_layout = ChannelLayout::unspecified(channels);
    }

    // The nAvgBytesPerSec-consistency override (riffdec.c:265-272).
    let expected = compute_bitrate(codec_id, sample_rate, block_align, bits, channels);
    if expected != 0 && bit_rate / 8 != expected / 8 {
        log_warning!(
            Some("wav"),
            "nAvgBytesPerSec {} inconsistent with other fields (expected {}), \
             overriding.",
            bit_rate / 8,
            expected / 8
        );
        bit_rate = expected;
    }

    Ok(WavFormat {
        codec_id,
        sample_rate,
        channels,
        block_align,
        bit_rate,
        ch_layout,
    })
}

// ---------------------------------------------------------------------
// Packet sizing — libavformat/pcm.c:29-55
// ---------------------------------------------------------------------

/// `ff_pcm_default_packet_size` (`libavformat/pcm.c:29-55`) — the demux
/// target is ~1/10 s of audio (`PCM_DEMUX_TARGET_FPS`, pcm.c:27), rounded
/// down to a power-of-two sample blocks, times `block_align`.
/// `None` = C's `AVERROR(EINVAL)` (no block_align).
fn pcm_default_packet_size(
    block_align: i32,
    codec_id: CodecId,
    sample_rate: i32,
    channels: usize,
    bit_rate: i64,
) -> Option<usize> {
    const PCM_DEMUX_TARGET_FPS: i64 = 10;

    if block_align <= 0 {
        return None;
    }
    let max_samples = i32::MAX as i64 / block_align as i64;
    let bits_per_sample = pcm::bits_per_sample(codec_id);
    let mut bitrate = bit_rate;
    // "Don't trust the codecpar bitrate if we can calculate it ourselves"
    // (pcm.c:41-44).
    if bits_per_sample > 0 && sample_rate > 0 && channels > 0 {
        bitrate = bits_per_sample as i64 * sample_rate as i64 * channels as i64;
    }
    let nb_samples = if bitrate > 0 {
        let n = (bitrate / 8 / PCM_DEMUX_TARGET_FPS / block_align as i64).clamp(1, max_samples);
        // 1 << av_log2(n) (pcm.c:48) — round down to a power of two.
        1i64 << (63 - n.leading_zeros())
    } else {
        // Size-based fallback for an unknown-rate codec (pcm.c:50-51).
        (4096 / block_align as i64).clamp(1, max_samples)
    };
    Some((block_align as i64 * nb_samples) as usize)
}

// ---------------------------------------------------------------------
// The demuxer
// ---------------------------------------------------------------------

/// `WAVDemuxContext` (`wavdec.c:51-69`) — the RIFF-reachable fields.
pub struct WavDemuxer {
    /// `wav->data_end` — absolute offset one past the data chunk
    /// (`u64::MAX` = C's `INT64_MAX` "length unknown" sentinel,
    /// `wavdec.c:480`).
    data_end: u64,
    /// Where the data payload starts (C's `data_ofs` local).
    data_ofs: u64,
    /// `wav->max_size`, set by `set_max_size` (`wavdec.c:82-88`).
    max_size: usize,
    /// `st->codecpar->block_align`, cached for packet slicing.
    block_align: i32,
    /// Stream time base (`1/sample_rate`) for packet timestamps.
    time_base: Rational,
}

impl WavDemuxer {
    pub fn new() -> Self {
        WavDemuxer {
            data_end: u64::MAX,
            data_ofs: 0,
            max_size: 0,
            block_align: 0,
            time_base: Rational::UNKNOWN,
        }
    }
}

impl Default for WavDemuxer {
    fn default() -> Self {
        WavDemuxer::new()
    }
}

impl Demuxer for WavDemuxer {
    /// `wav_read_header` (`wavdec.c:362-700`), RIFF subset.
    fn read_header(&mut self, io: &mut IoContext) -> Result<Stream> {
        let filesize = io.size();

        // Chunk ID (wavdec.c:381-398).
        let tag = rl32(io)?;
        match tag {
            TAG_RIFF => {}
            TAG_RIFX => {
                return Err(Error::Unsupported(
                    "RIFX (big-endian RIFF) is not ported".into(),
                ));
            }
            TAG_RF64 | TAG_BW64 => {
                return Err(Error::Unsupported("RF64/BW64 (ds64) is not ported".into()));
            }
            _ => {
                return Err(Error::InvalidData(format!(
                    "invalid start code {} in RIFF header",
                    fourcc_str(tag)
                )));
            }
        }
        let _riff_size = rl32(io)?; // chunk size, unused (wavdec.c:401)
        if rl32(io)? != TAG_WAVE {
            return Err(Error::InvalidData("invalid format in RIFF header".into()));
        }

        // The audio stream exists from the start so its index is 0
        // (wavdec.c:430-433); the port's Stream is built at the end once
        // the fmt chunk has the parameters.
        let mut got_fmt = false;
        let mut fmt: Option<WavFormat> = None;
        let mut data_ofs = u64::MAX; // C's -1 sentinel
        let mut data_size: u64 = 0;

        // The chunk walk (wavdec.c:435-620).
        loop {
            let tag = rl32(io)?;
            let size = rl32(io)? as u64;
            let next_tag_ofs = io.tell() + size + (size & 1); // wavdec.c:438

            if io.is_eof() {
                break; // avio_feof (wavdec.c:440-441)
            }

            match tag {
                TAG_FMT => {
                    // "only parse the first 'fmt ' tag found" (wavdec.c:444-451).
                    if !got_fmt {
                        fmt = Some(get_wav_header(io, size as i64)?);
                    } else {
                        log_warning!(Some("wav"), "found more than one 'fmt ' tag");
                    }
                    got_fmt = true;
                    // wavdec.c:201-203 (need_parsing / pts info) happen at
                    // stream build time below.
                }
                TAG_DATA => {
                    // wavdec.c:462-490. Unseekable-input guard is N/A
                    // (IoContext always seeks); data-before-fmt falls
                    // through to the "no 'fmt ' tag" error just like C's
                    // seekable path.
                    if size > 0 && size != 0xFFFF_FFFF {
                        data_size = size;
                        self.data_end = io.tell() + size;
                    } else {
                        // "Ignoring maximum wav data size" (wavdec.c:477-480).
                        log_warning!(
                            Some("wav"),
                            "Ignoring maximum wav data size, file may be invalid"
                        );
                        data_size = 0;
                        self.data_end = u64::MAX;
                    }
                    data_ofs = io.tell();
                    // Seekable input: stop scanning at 'data' (wavdec.c:485-490).
                    break;
                }
                // 'fact' (recomputed duration), 'bext', 'LIST', 'ID3 ',
                // 'cue ', 'SMV0', 'XMA2', anything else: skipped whole via
                // next_tag_ofs (see the module map for the C sites).
                _ => {}
            }

            // Seek to the next tag unless that would run into EOF
            // (wavdec.c:615-619).
            if filesize > 0 && next_tag_ofs >= filesize {
                break;
            }
            // wav_seek_tag's even-offset adjustment (wavdec.c:134-139);
            // next_tag_ofs is already even (size + size&1 from an even
            // base), so this is belt-and-braces like C.
            let mut seek_to = next_tag_ofs;
            if seek_to & 1 != 0 {
                seek_to += 1;
            }
            if io.seek(seek_to).is_err() {
                break;
            }
        }

        // break_loop checks (wavdec.c:622-631).
        let Some(fmt) = fmt else {
            return Err(Error::InvalidData("no 'fmt ' tag found".into()));
        };
        if data_ofs == u64::MAX {
            return Err(Error::InvalidData("no 'data' tag found".into()));
        }

        io.seek(data_ofs)?; // wavdec.c:633
        self.data_ofs = data_ofs;

        // Duration: sample_count = data_size·8 / (channels · bits)
        // (wavdec.c:662-669 — for PCM codecs av_get_exact_bits_per_sample
        // > 0 forces this recomputation over any 'fact' claim; 'fact' is
        // out, so this is the only path, C's primary one).
        let mut sample_count: u64 = 0;
        let bits = pcm::bits_per_sample(fmt.codec_id);
        if fmt.channels > 0 && data_size > 0 && bits > 0 && self.data_end <= filesize {
            sample_count = data_size * 8 / (fmt.channels as u64 * bits as u64);
        }

        // Build the stream (avformat_new_stream + wav_parse_fmt_tag's
        // avpriv_set_pts_info(st, 64, 1, sample_rate), wavdec.c:203).
        let mut st = Stream {
            index: 0,
            codecpar: CodecParameters {
                codec_type: MediaType::Audio,
                codec_id: fmt.codec_id,
                sample_rate: fmt.sample_rate,
                ch_layout: fmt.ch_layout,
                block_align: fmt.block_align,
                bit_rate: fmt.bit_rate,
                ..CodecParameters::default()
            },
            time_base: Rational::UNKNOWN,
            avg_frame_rate: Rational::UNKNOWN,
            r_frame_rate: Rational::UNKNOWN,
            sample_aspect_ratio: Rational::UNKNOWN,
            start_time: NOPTS,
            duration: NOPTS,
            nb_frames: 0,
        };
        // sample_fmt from the codec id (C fills codecpar->format at
        // find_stream_info via the decoder; the port has no probing layer,
        // so the demuxer resolves it — pcm.c:268-292's table).
        if let Some(f) = pcm::sample_fmt(fmt.codec_id) {
            st.codecpar.sample_fmt = f;
        }
        st.set_pts_info(1, fmt.sample_rate as i64);
        if sample_count > 0 {
            st.duration = sample_count as i64; // wavdec.c:671-672
        }
        self.time_base = st.time_base;
        self.block_align = fmt.block_align;

        // set_max_size (wavdec.c:82-88): the AVOption defaults to 0, so
        // always ff_pcm_default_packet_size, 4096 on failure.
        self.max_size = pcm_default_packet_size(
            fmt.block_align,
            fmt.codec_id,
            fmt.sample_rate,
            fmt.channels,
            fmt.bit_rate,
        )
        .unwrap_or(4096);

        Ok(st)
    }

    /// `wav_read_packet` (`wavdec.c:724-815`) — SPDIF/SMV/chained-data
    /// paths out; what remains is the PCM block slicing.
    fn read_packet(&mut self, io: &mut IoContext) -> Result<Packet> {
        let pos = io.tell();

        let left = self.data_end.saturating_sub(pos); // wavdec.c:783
        if left == 0 {
            // C would find_tag() the next 'data' chunk of a chained file
            // (wavdec.c:786-800) — out; EOF like C's find_tag failure.
            return Err(Error::Eof);
        }

        // wavdec.c:802-808 — snap the read size down to whole blocks.
        let mut size = self.max_size;
        if self.block_align > 1 {
            let ba = self.block_align as usize;
            if size < ba {
                size = ba;
            }
            size = size / ba * ba;
        }
        let size = (size as u64).min(left) as usize; // FFMIN(size, left)

        let payload = io.get_packet(size)?; // av_get_packet (Eof when empty)
        let mut pkt = Packet::from_vec(payload);
        pkt.stream_index = 0; // wavdec.c:812
        pkt.pos = pos;
        // The generic layer's cur_dts for PCM (ff_pcm_read_seek, pcm.c:103)
        // is pos/byte_rate·rate = pos/block_align samples — see the module
        // doc's timing note.
        if self.block_align > 0 {
            pkt.pts = ((pos - self.data_ofs) / self.block_align as u64) as i64;
            pkt.dts = pkt.pts;
            pkt.duration = (pkt.size() / self.block_align as usize) as i64;
        }
        pkt.time_base = self.time_base;
        Ok(pkt)
    }
}

/// `wav_probe` (`wavdec.c:161-178`) — RIFF/RIFX magic at 0, `WAVE` at 8,
/// score `AVPROBE_SCORE_MAX - 1` (kept below MAX so the ACT demuxer can
/// still win — see the C comment at wavdec.c:168-171). The RF64/BW64 arm
/// (wavdec.c:172-175) is out with the ds64 support.
pub fn probe(buf: &[u8]) -> u32 {
    if buf.len() <= 32 {
        return 0;
    }
    if buf[8..12] == *b"WAVE" && (buf[..4] == *b"RIFF" || buf[..4] == *b"RIFX") {
        PROBE_SCORE_MAX - 1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::pcm::PcmDecoder;
    use crate::codec::traits::AudioDecoder;
    use crate::format::demux::DemuxOptions;
    use crate::format::testutil::MemHandler;
    use crate::util::channel_layout::Order;
    use crate::util::samplefmt::SampleFormat;

    // ---- fixture builders ----

    /// A minimal 16-byte `PCMWAVEFORMAT` fmt chunk body.
    fn fmt_pcm(
        tag: u16,
        channels: u16,
        rate: u32,
        byte_rate: u32,
        block_align: u16,
        bits: u16,
    ) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&tag.to_le_bytes());
        v.extend_from_slice(&channels.to_le_bytes());
        v.extend_from_slice(&rate.to_le_bytes());
        v.extend_from_slice(&byte_rate.to_le_bytes());
        v.extend_from_slice(&block_align.to_le_bytes());
        v.extend_from_slice(&bits.to_le_bytes());
        v
    }

    /// A 40-byte WAVEFORMATEXTENSIBLE fmt body (cbSize 22).
    fn fmt_extensible(
        channels: u16,
        rate: u32,
        byte_rate: u32,
        block_align: u16,
        bits: u16,
        valid_bits: u16,
        mask: u32,
        subtag: u32,
    ) -> Vec<u8> {
        let mut v = fmt_pcm(0xFFFE, channels, rate, byte_rate, block_align, bits);
        v.extend_from_slice(&22u16.to_le_bytes()); // cbSize
        v.extend_from_slice(&valid_bits.to_le_bytes());
        v.extend_from_slice(&mask.to_le_bytes());
        // Subformat GUID: tag LE u32 + the MEDIASUBTYPE base tail.
        v.extend_from_slice(&subtag.to_le_bytes());
        v.extend_from_slice(&[
            0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
        ]);
        v
    }

    fn chunk(id: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut v = id.to_vec();
        v.extend_from_slice(&(body.len() as u32).to_le_bytes());
        v.extend_from_slice(body);
        if body.len() % 2 == 1 {
            v.push(0); // RIFF chunks pad to even (wavdec.c:438)
        }
        v
    }

    fn wav_bytes(chunks: &[(&[u8; 4], Vec<u8>)]) -> Vec<u8> {
        let mut body = Vec::new();
        for (id, c) in chunks {
            body.extend(chunk(id, c));
        }
        let mut v = b"RIFF".to_vec();
        v.extend_from_slice(&((body.len() + 4) as u32).to_le_bytes());
        v.extend_from_slice(b"WAVE");
        v.extend(body);
        v
    }

    fn stereo_s16_wav(data_len: usize) -> Vec<u8> {
        stereo_s16_wav_with(&vec![0xAB; data_len])
    }

    fn stereo_s16_wav_with(payload: &[u8]) -> Vec<u8> {
        let fmt = fmt_pcm(1, 2, 44100, 176400, 4, 16);
        wav_bytes(&[(&b"fmt ", fmt), (&b"data", payload.to_vec())])
    }

    // ---- probe (wavdec.c:161-178) ----

    #[test]
    fn probe_matches_riff_and_rifx_wave() {
        let f = stereo_s16_wav(64);
        assert_eq!(probe(&f), PROBE_SCORE_MAX - 1);
        let mut rifx = f.clone();
        rifx[0..4].copy_from_slice(b"RIFX");
        assert_eq!(probe(&rifx), PROBE_SCORE_MAX - 1);
    }

    #[test]
    fn probe_rejects_short_wrong_and_rf64() {
        assert_eq!(probe(&stereo_s16_wav(64)[..32]), 0); // <= 32 bytes
        let mut f = stereo_s16_wav(64);
        f[8..12].copy_from_slice(b"AVI "); // not WAVE
        assert_eq!(probe(&f), 0);
        let mut f = stereo_s16_wav(64);
        f[0..4].copy_from_slice(b"RF64"); // ds64 dialect: out
        assert_eq!(probe(&f), 0);
    }

    // ---- the fmt-tag map (riff.c:525-625 + riffdec.c:277-292 + utils.c:154-198) ----

    #[test]
    fn tag_0001_refines_by_bps() {
        // ff_get_pcm_codec_id(bps, 0, 0, ~1): widths >= 2 signed, 8-bit
        // unsigned (bit 0 of ~1 is clear), bps 64 → PCM_S64LE (not in the
        // family → None).
        assert_eq!(wav_codec_get_id(0x0001, 8), CodecId::PcmU8);
        assert_eq!(wav_codec_get_id(0x0001, 16), CodecId::PcmS16le);
        assert_eq!(wav_codec_get_id(0x0001, 24), CodecId::PcmS24le);
        assert_eq!(wav_codec_get_id(0x0001, 32), CodecId::PcmS32le);
        assert_eq!(wav_codec_get_id(0x0001, 64), CodecId::None);
        assert_eq!(wav_codec_get_id(0x0001, 0), CodecId::None);
        assert_eq!(wav_codec_get_id(0x0001, 65), CodecId::None);
        // Non-multiple-of-8 bps rounds up: 9 → 2 bytes → S16LE.
        assert_eq!(wav_codec_get_id(0x0001, 9), CodecId::PcmS16le);
    }

    #[test]
    fn tag_0003_refines_to_float_family() {
        assert_eq!(wav_codec_get_id(0x0003, 32), CodecId::PcmF32le);
        assert_eq!(wav_codec_get_id(0x0003, 64), CodecId::PcmF64le);
        assert_eq!(wav_codec_get_id(0x0003, 16), CodecId::None);
    }

    #[test]
    fn tags_0006_0007_ul_and_unknown() {
        assert_eq!(wav_codec_get_id(0x0006, 8), CodecId::PcmAlaw);
        assert_eq!(wav_codec_get_id(0x0007, 8), CodecId::PcmMulaw);
        assert_eq!(wav_codec_get_id(0x6c75, 8), CodecId::PcmMulaw); // 'ul'
        // ADPCM_MS (0x0002), MP3 (0x0055), … — codec not in the family.
        assert_eq!(wav_codec_get_id(0x0002, 4), CodecId::None);
        assert_eq!(wav_codec_get_id(0x0055, 0), CodecId::None);
    }

    // ---- read_header ----

    #[test]
    fn header_fills_codecpar_audio_fields() {
        let mut io = MemHandler::io(&stereo_s16_wav(32768));
        let mut dem = WavDemuxer::new();
        let st = dem.read_header(&mut io).unwrap();

        assert_eq!(st.codecpar.codec_type, MediaType::Audio);
        assert_eq!(st.codecpar.codec_id, CodecId::PcmS16le);
        assert_eq!(st.codecpar.sample_rate, 44100);
        assert_eq!(st.codecpar.block_align, 4);
        assert_eq!(st.codecpar.sample_fmt, SampleFormat::S16);
        assert_eq!(st.codecpar.bit_rate, 1_411_200); // 176400 B/s · 8
        // No extensible mask → UNSPEC(2) (riffdec.c:258-263 path).
        assert_eq!(st.codecpar.ch_layout, ChannelLayout::unspecified(2));
        // avpriv_set_pts_info(st, 64, 1, 44100) (wavdec.c:203).
        assert_eq!(st.time_base, Rational::new(1, 44100));
        assert_eq!(st.avg_frame_rate, Rational::new(44100, 1));
        // Duration: 32768·8 / (2 ch · 16 bits) = 8192 samples.
        assert_eq!(st.duration, 8192);
    }

    #[test]
    fn header_overrides_inconsistent_byte_rate() {
        // compute_bitrate (riffdec.c:99-138): the fields say
        // 44100·4·8 = 1411200; a lying nAvgBytesPerSec is overridden.
        let fmt = fmt_pcm(1, 2, 44100, 1000, 4, 16);
        let f = wav_bytes(&[(&b"fmt ", fmt), (&b"data", vec![0; 100])]);
        let mut io = MemHandler::io(&f);
        let st = WavDemuxer::new().read_header(&mut io).unwrap();
        assert_eq!(st.codecpar.bit_rate, 1_411_200);
    }

    #[test]
    fn header_skips_unknown_chunks_between_fmt_and_data() {
        let fmt = fmt_pcm(1, 1, 8000, 8000, 1, 8);
        let f = wav_bytes(&[
            (&b"fmt ", fmt),
            (&b"LIST", vec![b'I', b'N', b'F', b'O', 1, 2, 3]), // odd body → padded
            (&b"junk", vec![0u8; 10]),
            (&b"data", vec![0x55; 300]),
        ]);
        let mut io = MemHandler::io(&f);
        let mut dem = WavDemuxer::new();
        let st = dem.read_header(&mut io).unwrap();
        assert_eq!(st.codecpar.codec_id, CodecId::PcmU8);
        assert_eq!(st.codecpar.sample_rate, 8000);
        // u8 mono: 300 bytes = 300 samples.
        assert_eq!(st.duration, 300);
    }

    #[test]
    fn header_rejects_missing_fmt_or_data() {
        let f = wav_bytes(&[(&b"data", vec![0; 16])]);
        let err = WavDemuxer::new()
            .read_header(&mut MemHandler::io(&f))
            .unwrap_err();
        assert_eq!(err, Error::InvalidData("no 'fmt ' tag found".into()));

        let fmt = fmt_pcm(1, 1, 8000, 8000, 1, 8);
        let f = wav_bytes(&[(&b"fmt ", fmt)]);
        let err = WavDemuxer::new()
            .read_header(&mut MemHandler::io(&f))
            .unwrap_err();
        assert_eq!(err, Error::InvalidData("no 'data' tag found".into()));
    }

    #[test]
    fn header_rejects_bad_magic_and_header() {
        let mut f = stereo_s16_wav(16);
        f[0..4].copy_from_slice(b"JUNK");
        assert!(matches!(
            WavDemuxer::new().read_header(&mut MemHandler::io(&f)),
            Err(Error::InvalidData(_))
        ));
        let mut f = stereo_s16_wav(16);
        f[8..12].copy_from_slice(b"AVI ");
        let err = WavDemuxer::new()
            .read_header(&mut MemHandler::io(&f))
            .unwrap_err();
        assert_eq!(
            err,
            Error::InvalidData("invalid format in RIFF header".into())
        );
        // RIFX/RF64: probed (RIFX) or force-opened — reading is Unsupported.
        let mut f = stereo_s16_wav(16);
        f[0..4].copy_from_slice(b"RIFX");
        assert!(matches!(
            WavDemuxer::new().read_header(&mut MemHandler::io(&f)),
            Err(Error::Unsupported(_))
        ));
        let mut f = stereo_s16_wav(16);
        f[0..4].copy_from_slice(b"RF64");
        assert!(matches!(
            WavDemuxer::new().read_header(&mut MemHandler::io(&f)),
            Err(Error::Unsupported(_))
        ));
    }

    #[test]
    fn header_rejects_short_fmt_and_zero_rate() {
        // fmt body of 10 bytes < 14 (riffdec.c:147-150).
        let f = wav_bytes(&[(&b"fmt ", vec![0u8; 10]), (&b"data", vec![0; 4])]);
        let err = WavDemuxer::new()
            .read_header(&mut MemHandler::io(&f))
            .unwrap_err();
        assert_eq!(err, Error::InvalidData("wav header size < 14".into()));
        // sample_rate 0 (riffdec.c:242-246).
        let fmt = fmt_pcm(1, 1, 0, 0, 1, 8);
        let f = wav_bytes(&[(&b"fmt ", fmt), (&b"data", vec![0; 4])]);
        let err = WavDemuxer::new()
            .read_header(&mut MemHandler::io(&f))
            .unwrap_err();
        assert_eq!(err, Error::InvalidData("Invalid sample rate: 0".into()));
    }

    #[test]
    fn header_rejects_xma_fmt_tag() {
        let fmt = fmt_pcm(0x0165, 2, 44100, 176400, 4, 16);
        let f = wav_bytes(&[(&b"fmt ", fmt), (&b"data", vec![0; 4])]);
        assert!(matches!(
            WavDemuxer::new().read_header(&mut MemHandler::io(&f)),
            Err(Error::Unsupported(_))
        ));
    }

    #[test]
    fn header_plain_waveformat_size_14_bits_default_8() {
        // WAVEFORMAT (no bits field) → bits 8 → PCM_U8 under tag 1
        // (riffdec.c:170-171 + the bps refinement).
        let mut body = Vec::new();
        body.extend_from_slice(&1u16.to_le_bytes());
        body.extend_from_slice(&1u16.to_le_bytes()); // channels
        body.extend_from_slice(&8000u32.to_le_bytes());
        body.extend_from_slice(&8000u32.to_le_bytes());
        body.extend_from_slice(&1u16.to_le_bytes()); // block_align
        assert_eq!(body.len(), 14);
        let f = wav_bytes(&[(&b"fmt ", body), (&b"data", vec![7; 8])]);
        let st = WavDemuxer::new()
            .read_header(&mut MemHandler::io(&f))
            .unwrap();
        assert_eq!(st.codecpar.codec_id, CodecId::PcmU8);
        assert_eq!(st.codecpar.sample_fmt, SampleFormat::U8);
    }

    // ---- WAVEFORMATEXTENSIBLE (riffdec.c:62-92, 194-197) ----

    #[test]
    fn extensible_subtag_and_mask() {
        // Subtag 1 (PCM) + stereo mask 0x3 → S16LE + native STEREO.
        let fmt = fmt_extensible(2, 44100, 176400, 4, 16, 16, 0x3, 1);
        let f = wav_bytes(&[(&b"fmt ", fmt), (&b"data", vec![0; 64])]);
        let st = WavDemuxer::new()
            .read_header(&mut MemHandler::io(&f))
            .unwrap();
        assert_eq!(st.codecpar.codec_id, CodecId::PcmS16le);
        assert_eq!(st.codecpar.ch_layout, ChannelLayout::STEREO);
        assert_eq!(st.codecpar.ch_layout.order, Order::Native);

        // Subtag 3 (IEEE float), 64-bit → F64LE; mask 0x4 = front center
        // = MONO.
        let fmt = fmt_extensible(1, 48000, 384000, 8, 64, 64, 0x4, 3);
        let f = wav_bytes(&[(&b"fmt ", fmt), (&b"data", vec![0; 64])]);
        let st = WavDemuxer::new()
            .read_header(&mut MemHandler::io(&f))
            .unwrap();
        assert_eq!(st.codecpar.codec_id, CodecId::PcmF64le);
        assert_eq!(st.codecpar.sample_fmt, SampleFormat::Dbl);
        assert_eq!(st.codecpar.ch_layout, ChannelLayout::MONO);
    }

    #[test]
    fn extensible_mask_mismatch_falls_back_to_unspec() {
        // 2 channels but a mono mask (popcount 1 ≠ 2) → UNSPEC(2)
        // (riffdec.c:258-263). Same for mask 0.
        for mask in [0x1u32, 0x0] {
            let fmt = fmt_extensible(2, 44100, 176400, 4, 16, 16, mask, 1);
            let f = wav_bytes(&[(&b"fmt ", fmt), (&b"data", vec![0; 64])]);
            let st = WavDemuxer::new()
                .read_header(&mut MemHandler::io(&f))
                .unwrap();
            assert_eq!(
                st.codecpar.ch_layout,
                ChannelLayout::unspecified(2),
                "{mask}"
            );
        }
    }

    #[test]
    fn extensible_unknown_guid_and_alaw_subtag() {
        // GUID tail not a base GUID → codec None (C's guid table is out).
        let mut fmt = fmt_extensible(2, 44100, 176400, 4, 16, 16, 0x3, 1);
        let n = fmt.len();
        fmt[n - 1] = 0x72; // break the tail
        let f = wav_bytes(&[(&b"fmt ", fmt), (&b"data", vec![0; 64])]);
        let st = WavDemuxer::new()
            .read_header(&mut MemHandler::io(&f))
            .unwrap();
        assert_eq!(st.codecpar.codec_id, CodecId::None);
        assert_eq!(st.duration, NOPTS); // bits 0 → no duration (wavdec.c:662-669)

        // Subtag 6 = A-law: recognized, decoder open will refuse.
        let fmt = fmt_extensible(1, 8000, 8000, 1, 8, 8, 0x4, 6);
        let f = wav_bytes(&[(&b"fmt ", fmt), (&b"data", vec![0; 8])]);
        let st = WavDemuxer::new()
            .read_header(&mut MemHandler::io(&f))
            .unwrap();
        assert_eq!(st.codecpar.codec_id, CodecId::PcmAlaw);
    }

    // ---- read_packet: the pcm.c sample-block math ----

    #[test]
    fn packets_are_pow2_sample_blocks() {
        // 44100 stereo s16: bitrate-derived target 4410 samples →
        // 1<<log2 → 4096 samples → 16384-byte packets (pcm.c:46-48).
        let mut io = MemHandler::io(&stereo_s16_wav(32768));
        let mut dem = WavDemuxer::new();
        dem.read_header(&mut io).unwrap();
        let p0 = dem.read_packet(&mut io).unwrap();
        assert_eq!(p0.size(), 16384);
        assert_eq!(p0.pts, 0);
        assert_eq!(p0.dts, 0);
        assert_eq!(p0.duration, 4096);
        assert_eq!(p0.stream_index, 0);
        assert_eq!(p0.time_base, Rational::new(1, 44100));
        assert_eq!(p0.pos, 44); // 12 RIFF + 8+16 fmt + 8 data header
        let p1 = dem.read_packet(&mut io).unwrap();
        assert_eq!(p1.size(), 16384);
        assert_eq!(p1.pts, 4096);
        assert_eq!(p1.duration, 4096);
        assert_eq!(p1.pos, 44 + 16384);
        assert!(matches!(dem.read_packet(&mut io), Err(Error::Eof)));
    }

    #[test]
    fn block_align_math_u8_mono() {
        // u8 mono 8000 Hz: 64000 bit/s → 64000/8/10/1 = 800 samples →
        // 1<<log2(800) = 512 → 512-byte packets; block_align 1 skips the
        // snapping branch (wavdec.c:803-807).
        let fmt = fmt_pcm(1, 1, 8000, 8000, 1, 8);
        let f = wav_bytes(&[(&b"fmt ", fmt), (&b"data", vec![0x11; 1000])]);
        let mut io = MemHandler::io(&f);
        let mut dem = WavDemuxer::new();
        dem.read_header(&mut io).unwrap();
        let p = dem.read_packet(&mut io).unwrap();
        assert_eq!(p.size(), 512);
        assert_eq!(p.duration, 512); // 1 byte = 1 sample
        let p = dem.read_packet(&mut io).unwrap();
        assert_eq!(p.size(), 488); // tail: min(512, left=488)
        assert_eq!(p.pts, 512);
        assert!(matches!(dem.read_packet(&mut io), Err(Error::Eof)));
    }

    #[test]
    fn block_align_snaps_packet_size() {
        // 8000 Hz stereo s16: bitrate 16·8000·2 = 256000 → target samples
        // 256000/8/10/4 = 800 → 1<<log2(800) = 512 → 2048-byte packets,
        // already a multiple of block_align 4.
        let fmt = fmt_pcm(1, 2, 8000, 32000, 4, 16);
        let f = wav_bytes(&[(&b"fmt ", fmt), (&b"data", vec![0; 5000])]);
        let mut io = MemHandler::io(&f);
        let mut dem = WavDemuxer::new();
        dem.read_header(&mut io).unwrap();
        let p = dem.read_packet(&mut io).unwrap();
        assert_eq!(p.size(), 2048);

        // A raised (broken) block_align 2048: ff_pcm_default_packet_size
        // gives 2048·1 = 2048, and the wavdec.c:803-806 snapping keeps it.
        let fmt = fmt_pcm(1, 2, 8000, 32000, 2048, 16);
        let f = wav_bytes(&[(&b"fmt ", fmt), (&b"data", vec![0; 5000])]);
        let mut io = MemHandler::io(&f);
        let mut dem = WavDemuxer::new();
        dem.read_header(&mut io).unwrap();
        // block_align 2048 vs PCM math (bits 16 → expected 4): bit_rate
        // stays the header's 256000 (compute_bitrate mismatch → 0).
        assert_eq!(io.tell(), 44); // data payload start
        let p = dem.read_packet(&mut io).unwrap();
        assert_eq!(p.size(), 2048);
        assert_eq!(p.duration, 1); // 2048/2048
    }

    #[test]
    fn unknown_data_length_reads_to_eof() {
        // data size 0xFFFFFFFF → "Ignoring maximum wav data size",
        // data_end = sentinel (wavdec.c:476-481): read everything left,
        // in packet-sized chunks, duration unknown (data_size 0).
        let fmt = fmt_pcm(1, 1, 8000, 8000, 1, 8);
        let mut f = b"RIFF".to_vec();
        f.extend_from_slice(&((4 + 8 + 8 + 600) as u32).to_le_bytes());
        f.extend_from_slice(b"WAVE");
        f.extend_from_slice(b"fmt ");
        f.extend_from_slice(&16u32.to_le_bytes());
        f.extend_from_slice(&fmt);
        f.extend_from_slice(b"data");
        f.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        f.extend_from_slice(&vec![0x22; 600]);

        let mut io = MemHandler::io(&f);
        let mut dem = WavDemuxer::new();
        let st = dem.read_header(&mut io).unwrap();
        assert_eq!(st.duration, NOPTS);
        let p = dem.read_packet(&mut io).unwrap();
        assert_eq!(p.size(), 512);
        let p = dem.read_packet(&mut io).unwrap();
        assert_eq!(p.size(), 88); // 600 - 512, at real EOF
        assert!(matches!(dem.read_packet(&mut io), Err(Error::Eof)));
    }

    #[test]
    fn short_data_chunk_single_packet() {
        // 20 bytes < 16384: one 20-byte packet (FFMIN with left), tail not
        // block-trimmed (C passes short packets through, wavdec.c:808-809).
        let mut io = MemHandler::io(&stereo_s16_wav(20));
        let mut dem = WavDemuxer::new();
        dem.read_header(&mut io).unwrap();
        let p = dem.read_packet(&mut io).unwrap();
        assert_eq!(p.size(), 20);
        assert_eq!(p.duration, 5); // 20/4
        assert!(matches!(dem.read_packet(&mut io), Err(Error::Eof)));
    }

    // ---- the full pipeline: probe → open → packets → PCM decode ----

    #[test]
    fn file_round_trip_probe_header_packets_decode() {
        // Byte-level fixture: 44100 Hz stereo s16le, 8 stereo sample
        // frames (32 bytes), sample values 0..16.
        let payload: Vec<u8> = (0u16..16).flat_map(|v| v.to_le_bytes()).collect();
        let bytes = stereo_s16_wav_with(&payload);
        assert_eq!(probe(&bytes), PROBE_SCORE_MAX - 1);

        // Write to a temp file (no .wav extension → forces the probe path
        // in InputFormatContext::open).
        let path = std::env::temp_dir().join(format!(
            "ffmpeg_rs_wav_roundtrip_{}.bin",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).unwrap();

        let mut ictx = super::super::InputFormatContext::open(
            path.to_str().unwrap(),
            None,
            &DemuxOptions::default(),
        )
        .unwrap();
        assert_eq!(ictx.iformat.name, "wav");
        let st = &ictx.streams[0];
        assert_eq!(st.codecpar.codec_id, CodecId::PcmS16le);
        assert_eq!(st.codecpar.sample_rate, 44100);
        assert_eq!(st.codecpar.block_align, 4);
        assert_eq!(st.duration, 8); // 32 bytes / 4 per sample frame

        let mut dec = PcmDecoder::new();
        dec.init(&st.codecpar).unwrap();

        let pkt = ictx.read_frame().unwrap();
        assert_eq!(pkt.as_slice(), &payload[..]);

        let f = {
            dec.send_packet(Some(&pkt)).unwrap();
            dec.receive_frame().unwrap()
        };
        assert_eq!(f.format, SampleFormat::S16);
        assert_eq!(f.nb_samples, 8);
        assert_eq!(f.channels(), 2);
        assert_eq!(f.sample_rate, 44100);
        assert_eq!(f.plane(0), &payload[..]);
        assert_eq!(f.pts, 0);
        assert_eq!(f.duration, 8);

        // 32-byte data chunk is one packet; then EOF.
        assert!(matches!(ictx.read_frame(), Err(Error::Eof)));
        dec.send_packet(None).unwrap();
        assert!(matches!(dec.receive_frame(), Err(Error::Eof)));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn alaw_file_demuxes_but_decoder_refuses() {
        let fmt = fmt_pcm(6, 1, 8000, 8000, 1, 8);
        let f = wav_bytes(&[(&b"fmt ", fmt), (&b"data", vec![0xD5; 16])]);
        let mut ictx_io = MemHandler::io(&f);
        let mut dem = WavDemuxer::new();
        let st = dem.read_header(&mut ictx_io).unwrap();
        assert_eq!(st.codecpar.codec_id, CodecId::PcmAlaw);
        assert_eq!(st.codecpar.sample_fmt, SampleFormat::S16); // lut row
        let mut dec = PcmDecoder::new();
        assert!(matches!(dec.init(&st.codecpar), Err(Error::Unsupported(_))));
    }
}
