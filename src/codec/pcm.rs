//! PCM decoder — port of the decode side of `libavcodec/pcm.c`.
//!
//! ## C → Rust map
//!
//! | C | here |
//! |---|---|
//! | `codec_id_to_samplefmt[]` (`pcm.c:268-292`) | [`PCM_TABLE`] (+ [`sample_fmt`]/[`sample_size`]/[`bits_per_sample`]) |
//! | `pcm_decode_init` (`pcm.c:265-306`) | [`PcmDecoder::init`] |
//! | `pcm_decode_frame` (`pcm.c:403-624`) | [`PcmDecoder::send_packet`] |
//! | `av_get_bits_per_sample` PCM arms (`libavutil/utils.c`) | [`bits_per_sample`] |
//!
//! ## What "decoding" PCM is
//!
//! For every PCM codec, a packet is a flat run of interleaved coded samples
//! and a frame is the same samples in an [`AudioFrame`] — so the decoder's
//! whole job (`pcm_decode_frame`) is: validate that `buf_size` is a whole
//! number of per-sample-frame blocks (`n = channels * sample_size`,
//! `pcm.c:431-441`), compute `nb_samples = buf_size / sample_size /
//! channels` (`pcm.c:443-446`), and move the bytes. Three byte shapes exist
//! in the ported subset:
//!
//! * **passthrough** — output sample format's in-memory layout equals the
//!   coded layout (`U8`; every `*LE` type on a little-endian target, C's
//!   `HAVE_BIGENDIAN == 0` arm, `pcm.c:509-556`): the packet `Arc` is
//!   adopted directly, zero copy.
//! * **byte swap** — the `*BE` twins of the above (`DECODE(size, beX, …)`
//!   with `shift 0, offset 0`): same size, each sample word reversed.
//! * **3→4 expansion** — `PCM_S24LE/BE` decode into `AV_SAMPLE_FMT_S32`
//!   via `DECODE(32, le24/be24, src, dst, n, 8, 0)` (`pcm.c:458-466`):
//!   the zero-extended 24-bit word is shifted left 8 into an `i32` (so
//!   `0xFFFFFF` = −1 becomes `0xFFFFFF00` = −256), buffer grows 3→4 bytes
//!   per sample.
//!
//! ## Skipped C paths (documented, with the C guard that keeps them out)
//!
//! | C path | Guard | Ported? |
//! |---|---|---|
//! | A-law/µ-law/VIDC LUT decode (`pcm.c:341-372, 572-584`) | `PCMLUTDecode` init | no — [`PcmDecoder::init`] returns `Error::Unsupported` for `PcmAlaw`/`PcmMulaw` (the G.711 tables are a later phase) |
//! | planar codecs `S8/S16/S32/S24_LE_PLANAR` (`pcm.c:461-462, 500-508, 563-571`) | `DECODE_PLANAR` | no — no planar `CodecId` in the family (`params.rs`) |
//! | `PCM_LXF` 40-bit blocks (`pcm.c:416-419, 585-608`) | `samples_per_block = 2` | no |
//! | `PCM_SGA` sign/magnitude (`pcm.c:492-499`), `PCM_S24DAUD` bit-reversal (`pcm.c:473-481`) | codec ids | no — not in the `CodecId` family |
//! | `PCM_F16LE/F24LE` float scaling (`pcm.c:613-619`, `pcm_scale_decode_init`) | `PCMScaleDecode` | no — WAV's F16/F24 rewrite (`wavdec.c:674-684`) needs `extradata`, also out |
//! | `PCM_S64LE/BE` (`pcm.c:510-511, 535-537`) | codec ids | no — not in the `CodecId` family (WAV bps=64 maps to `CodecId::None`) |
//! | unsigned `U16/U24/U32` (`pcm.c:482-487` etc.) | codec ids | no — unreachable from the WAV tag map (`ff_get_pcm_codec_id` is only called with `sflags = ~1`, which signs every width ≥ 2) |
//! | encoders (`pcm_encode_*`, `pcm.c:41-259`) | — | no — Phase 4b is decode-only |
//! | `avctx->codec_id != avctx->codec->id` check (`pcm.c:426-429`) | decoder/packet id mismatch | unrepresentable — `init` pins the id the table matched |
//!
//! One deviation: `PCM_U8` passthrough is `memcpy` in C (`pcm.c:557-559`);
//! here the packet's `Arc<[u8]>` is shared instead of copied — same bytes,
//! one allocation fewer (the same trick `RawVideoDecoder` uses).

use crate::{
    codec::{
        packet::Packet,
        params::{CodecId, CodecParameters, MediaType},
        traits::AudioDecoder,
    },
    util::{
        audio_frame::AudioFrame,
        error::{Error, Result},
        samplefmt::SampleFormat,
    },
};

/// One row of C's `codec_id_to_samplefmt[]` (`pcm.c:268-292`): codec →
/// output `AVSampleFormat`, coded bytes per sample (`BITS/8` in the C
/// `ENTRY` macro) and bits per coded sample. Only rows whose codec id is
/// in the [`CodecId`] family; the two LUT codecs are listed so the table
/// pins their (unused-here) shape too — [`PcmDecoder::init`] rejects them.
pub const PCM_TABLE: &[(CodecId, SampleFormat, u8, u8)] = &[
    // (codec, sample_fmt, sample_size bytes, bits_per_sample)
    (CodecId::PcmU8, SampleFormat::U8, 1, 8),
    (CodecId::PcmS16le, SampleFormat::S16, 2, 16),
    (CodecId::PcmS16be, SampleFormat::S16, 2, 16),
    (CodecId::PcmS24le, SampleFormat::S32, 3, 24),
    (CodecId::PcmS24be, SampleFormat::S32, 3, 24),
    (CodecId::PcmS32le, SampleFormat::S32, 4, 32),
    (CodecId::PcmS32be, SampleFormat::S32, 4, 32),
    (CodecId::PcmF32le, SampleFormat::Flt, 4, 32),
    (CodecId::PcmF32be, SampleFormat::Flt, 4, 32),
    (CodecId::PcmF64le, SampleFormat::Dbl, 8, 64),
    (CodecId::PcmF64be, SampleFormat::Dbl, 8, 64),
    // pcm_lut_decode_init (pcm.c:341-372): S16 out, 1 coded byte/sample.
    // Decode itself is NOT ported (init → Error::Unsupported).
    (CodecId::PcmAlaw, SampleFormat::S16, 1, 8),
    (CodecId::PcmMulaw, SampleFormat::S16, 1, 8),
];

/// `codec_id_to_samplefmt[]` lookup (`pcm.c:294-303`) → the output format.
pub fn sample_fmt(codec_id: CodecId) -> Option<SampleFormat> {
    PCM_TABLE
        .iter()
        .find(|(id, ..)| *id == codec_id)
        .map(|&(_, fmt, ..)| fmt)
}

/// `s->sample_size` (`pcm.c:296`) — coded bytes per sample
/// (`BITS_PER_SAMPLE / 8` of the C `ENTRY` macro).
pub fn sample_size(codec_id: CodecId) -> Option<usize> {
    PCM_TABLE
        .iter()
        .find(|(id, ..)| *id == codec_id)
        .map(|&(_, _, size, _)| size as usize)
}

/// `av_get_bits_per_sample` for the PCM family (`libavutil/utils.c`: the
/// `PCM_CODEC`-registered ids return their table bits; `ALAW`/`MULAW` are
/// 8). Non-PCM/unknown ids return 0, exactly like C.
pub const fn bits_per_sample(codec_id: CodecId) -> i32 {
    match codec_id {
        CodecId::PcmU8 | CodecId::PcmAlaw | CodecId::PcmMulaw => 8,
        CodecId::PcmS16le | CodecId::PcmS16be => 16,
        CodecId::PcmS24le | CodecId::PcmS24be => 24,
        CodecId::PcmS32le | CodecId::PcmS32be | CodecId::PcmF32le | CodecId::PcmF32be => 32,
        CodecId::PcmF64le | CodecId::PcmF64be => 64,
        _ => 0,
    }
}

/// `ff_pcm_s16le_decoder` and family — one struct for every table row
/// (C differs only in the private-data init function).
#[derive(Debug, Default)]
pub struct PcmDecoder {
    params: CodecParameters,
    /// One-frame output queue (`AV_CODEC_CAP_VARIABLE_FRAME_SIZE`: PCM
    /// emits exactly one frame per packet).
    pending: Option<AudioFrame>,
    /// Drain requested (`avcodec_send_packet(avctx, NULL)`).
    eof: bool,
}

impl PcmDecoder {
    pub fn new() -> Self {
        Self::default()
    }
}

impl AudioDecoder for PcmDecoder {
    /// `pcm_decode_init` (`pcm.c:265-306`) — resolve the table row; the
    /// LUT codecs' `pcm_lut_decode_init` is unported → `Unsupported`.
    fn init(&mut self, params: &CodecParameters) -> Result<()> {
        match params.codec_id {
            CodecId::PcmAlaw => {
                return Err(Error::Unsupported(
                    "pcm_alaw (G.711 A-law companding table) is not ported".into(),
                ));
            }
            CodecId::PcmMulaw => {
                return Err(Error::Unsupported(
                    "pcm_mulaw (G.711 mu-law companding table) is not ported".into(),
                ));
            }
            id if PCM_TABLE.iter().any(|&(t, ..)| t == id) => {}
            id => {
                return Err(Error::Unsupported(format!(
                    "codec '{}' is not a ported PCM decoder",
                    id.name()
                )));
            }
        }
        self.params = params.clone();
        self.params.codec_type = MediaType::Audio;
        Ok(())
    }

    /// `pcm_decode_frame` (`pcm.c:403-624`) — the subset table above.
    fn send_packet(&mut self, pkt: Option<&Packet>) -> Result<()> {
        let Some(pkt) = pkt else {
            self.eof = true;
            return Ok(());
        };
        if self.eof {
            // C rejects input after drain in avcodec_send_packet.
            return Err(Error::Eof);
        }

        let channels = self.params.ch_layout.nb_channels;
        // "Invalid number of channels" (pcm.c:421-424).
        if channels == 0 {
            return Err(Error::InvalidData("Invalid number of channels".into()));
        }
        let codec_id = self.params.codec_id;
        let sample_size = sample_size(codec_id).expect("init pinned a table row");
        let sample_fmt = sample_fmt(codec_id).expect("init pinned a table row");

        // pcm.c:431-441 — a packet must hold whole per-sample-frame blocks.
        let block = channels * sample_size;
        let mut buf_size = pkt.size();
        if block > 0 && buf_size % block != 0 {
            if buf_size < block {
                return Err(Error::InvalidData(format!(
                    "Invalid PCM packet, data has size {buf_size} but at least a \
                     size of {block} was expected"
                )));
            }
            buf_size -= buf_size % block;
        }

        // pcm.c:443-446 — n = buf_size / sample_size, nb_samples = n /
        // channels (samples_per_block is 1; only LXF uses 2).
        let src = &pkt.as_slice()[..buf_size];
        let nb_samples = buf_size / sample_size / channels;

        let buf: std::sync::Arc<[u8]> = match codec_id {
            // memcpy / DECODE(size, leX, …, 0, 0) on a little-endian target
            // (pcm.c:509-559): the coded layout IS the native layout —
            // adopt the packet buffer, zero copy.
            CodecId::PcmU8
            | CodecId::PcmS16le
            | CodecId::PcmS32le
            | CodecId::PcmF32le
            | CodecId::PcmF64le => std::sync::Arc::clone(&pkt.data),

            // DECODE(16/32/64, beX, src, dst, n, 0, 0) (pcm.c:535-547):
            // v = beX(); store native — a per-word byte swap.
            CodecId::PcmS16be | CodecId::PcmS32be | CodecId::PcmF32be | CodecId::PcmF64be => {
                let w = sample_fmt.bytes_per_sample();
                let mut out = Vec::with_capacity(buf_size);
                for chunk in src.chunks_exact(w) {
                    out.extend(chunk.iter().rev());
                }
                std::sync::Arc::from(out)
            }

            // DECODE(32, le24/be24, src, samples, n, 8, 0) (pcm.c:458-466):
            // v = 24-bit word (zero-extended by AV_RL24/AV_RB24), stored as
            // (v - 0) << 8 — an i32 whose top 24 bits are the sample.
            CodecId::PcmS24le | CodecId::PcmS24be => {
                let mut out = Vec::with_capacity(nb_samples * channels * 4);
                for chunk in src.chunks_exact(3) {
                    let v: u32 = if codec_id == CodecId::PcmS24le {
                        u32::from(chunk[0])
                            | (u32::from(chunk[1]) << 8)
                            | (u32::from(chunk[2]) << 16)
                    } else {
                        (u32::from(chunk[0]) << 16)
                            | (u32::from(chunk[1]) << 8)
                            | u32::from(chunk[2])
                    };
                    out.extend_from_slice(&v.wrapping_shl(8).to_le_bytes());
                }
                std::sync::Arc::from(out)
            }

            id => unreachable!("init pinned a table row, but {id:?} missed"),
        };

        let mut frame =
            AudioFrame::wrap_buffer(buf, sample_fmt, self.params.ch_layout, nb_samples)?;
        // ff_decode_frame_props_from_pkt: timing comes off the packet.
        frame.pts = pkt.pts;
        frame.duration = pkt.duration;
        frame.time_base = pkt.time_base;
        // The codec context's stream parameters (set from codecpar at open).
        frame.sample_rate = self.params.sample_rate;

        self.pending = Some(frame);
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<AudioFrame> {
        match self.pending.take() {
            Some(frame) => Ok(frame),
            None if self.eof => Err(Error::Eof),
            None => Err(Error::Again),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::packet::PacketFlags;
    use crate::util::channel_layout::ChannelLayout;
    use crate::util::rational::Rational;

    fn params(id: CodecId, channels: usize, rate: i32) -> CodecParameters {
        let mut p = CodecParameters::default();
        p.codec_type = MediaType::Audio;
        p.codec_id = id;
        p.sample_rate = rate;
        p.ch_layout = ChannelLayout::unspecified(channels);
        p.sample_fmt = SampleFormat::S16;
        p.block_align = (sample_size(id).unwrap_or(1) * channels) as i32;
        p
    }

    fn pkt(bytes: &[u8]) -> Packet {
        let mut p = Packet::from_vec(bytes.to_vec());
        p.pts = 7;
        p.duration = 3;
        p.time_base = Rational::new(1, 48000);
        p.flags = PacketFlags::KEY;
        p
    }

    // ---- the table pin: every PCM codec id's format/size/bits ----

    #[test]
    fn table_pins_every_pcm_codec_id() {
        // C rows (pcm.c:277-290) restricted to the CodecId family.
        let expect: &[(CodecId, SampleFormat, usize, i32)] = &[
            (CodecId::PcmU8, SampleFormat::U8, 1, 8),
            (CodecId::PcmS16le, SampleFormat::S16, 2, 16),
            (CodecId::PcmS16be, SampleFormat::S16, 2, 16),
            (CodecId::PcmS24le, SampleFormat::S32, 3, 24),
            (CodecId::PcmS24be, SampleFormat::S32, 3, 24),
            (CodecId::PcmS32le, SampleFormat::S32, 4, 32),
            (CodecId::PcmS32be, SampleFormat::S32, 4, 32),
            (CodecId::PcmF32le, SampleFormat::Flt, 4, 32),
            (CodecId::PcmF32be, SampleFormat::Flt, 4, 32),
            (CodecId::PcmF64le, SampleFormat::Dbl, 8, 64),
            (CodecId::PcmF64be, SampleFormat::Dbl, 8, 64),
            (CodecId::PcmAlaw, SampleFormat::S16, 1, 8),
            (CodecId::PcmMulaw, SampleFormat::S16, 1, 8),
        ];
        // Every row's three lookups agree, and every row is found.
        assert_eq!(PCM_TABLE.len(), expect.len());
        for &(id, fmt, size, bits) in expect {
            assert_eq!(sample_fmt(id), Some(fmt), "{id:?}");
            assert_eq!(sample_size(id), Some(size), "{id:?}");
            assert_eq!(bits_per_sample(id), bits, "{id:?}");
        }
        // Non-PCM ids: no row, 0 bits (av_get_bits_per_sample contract).
        assert_eq!(sample_fmt(CodecId::Rawvideo), None);
        assert_eq!(sample_size(CodecId::None), None);
        assert_eq!(bits_per_sample(CodecId::Rawvideo), 0);
    }

    // ---- handshake ----

    #[test]
    fn handshake_again_then_frame_then_eof() {
        let mut dec = PcmDecoder::new();
        dec.init(&params(CodecId::PcmS16le, 2, 48000)).unwrap();
        // Nothing queued: EAGAIN (C's receive_frame with no buffer).
        assert!(matches!(dec.receive_frame(), Err(Error::Again)));
        dec.send_packet(Some(&pkt(&[0x01, 0x02, 0x03, 0x04])))
            .unwrap();
        let f = dec.receive_frame().unwrap();
        assert_eq!(f.nb_samples, 1);
        // Drain: send None → frames done → Eof.
        dec.send_packet(None).unwrap();
        assert!(matches!(dec.receive_frame(), Err(Error::Eof)));
        // Input after drain is rejected (avcodec_send_packet contract).
        assert!(matches!(
            dec.send_packet(Some(&pkt(&[0, 0]))),
            Err(Error::Eof)
        ));
    }

    // ---- passthrough / swap / expand ----

    #[test]
    fn s16le_stereo_passthrough_zero_copy() {
        let mut dec = PcmDecoder::new();
        dec.init(&params(CodecId::PcmS16le, 2, 44100)).unwrap();
        let p = pkt(&[0x10, 0xF0, 0x20, 0x00, 0x30, 0x11, 0x40, 0x22]);
        dec.send_packet(Some(&p)).unwrap();
        let f = dec.receive_frame().unwrap();
        assert_eq!(f.format, SampleFormat::S16);
        assert_eq!(f.nb_samples, 2); // 8 bytes / 2 per sample / 2 ch
        assert_eq!(f.channels(), 2);
        assert_eq!(
            f.plane(0),
            &[0x10, 0xF0, 0x20, 0x00, 0x30, 0x11, 0x40, 0x22]
        );
        // Timing and stream params carried onto the frame.
        assert_eq!(f.pts, 7);
        assert_eq!(f.duration, 3);
        assert_eq!(f.time_base, Rational::new(1, 48000));
        assert_eq!(f.sample_rate, 44100);
        // Zero copy: the plane shares the packet's Arc.
        assert!(std::sync::Arc::ptr_eq(&f.planes[0].buf, &p.data));
    }

    #[test]
    fn u8_mono_passthrough() {
        let mut dec = PcmDecoder::new();
        dec.init(&params(CodecId::PcmU8, 1, 8000)).unwrap();
        dec.send_packet(Some(&pkt(&[0x80, 0x00, 0xFF]))).unwrap();
        let f = dec.receive_frame().unwrap();
        assert_eq!(f.format, SampleFormat::U8);
        assert_eq!(f.nb_samples, 3);
        assert_eq!(f.plane(0), &[0x80, 0x00, 0xFF]);
    }

    #[test]
    fn s16be_swaps_each_word() {
        let mut dec = PcmDecoder::new();
        dec.init(&params(CodecId::PcmS16be, 1, 8000)).unwrap();
        // BE words 0x0001 (1) and 0xFFFE (-2) → native LE bytes swapped.
        dec.send_packet(Some(&pkt(&[0x00, 0x01, 0xFF, 0xFE])))
            .unwrap();
        let f = dec.receive_frame().unwrap();
        assert_eq!(f.format, SampleFormat::S16);
        assert_eq!(f.nb_samples, 2);
        assert_eq!(f.plane(0), &[0x01, 0x00, 0xFE, 0xFF]);
        // Value check through the sample view.
        let s16 = |b: &[u8]| i16::from_le_bytes([b[0], b[1]]);
        assert_eq!(s16(&f.plane(0)[..2]), 1);
        assert_eq!(s16(&f.plane(0)[2..]), -2);
    }

    #[test]
    fn s32be_f64be_swap_words() {
        let mut dec = PcmDecoder::new();
        dec.init(&params(CodecId::PcmS32be, 1, 8000)).unwrap();
        dec.send_packet(Some(&pkt(&[0, 0, 0, 1]))).unwrap();
        let f = dec.receive_frame().unwrap();
        assert_eq!(f.format, SampleFormat::S32);
        assert_eq!(f.plane(0), &[1, 0, 0, 0]);

        let mut dec = PcmDecoder::new();
        dec.init(&params(CodecId::PcmF64be, 1, 8000)).unwrap();
        let word: [u8; 8] = [0, 0, 0, 0, 0, 0, 0, 1];
        dec.send_packet(Some(&pkt(&word))).unwrap();
        let f = dec.receive_frame().unwrap();
        assert_eq!(f.format, SampleFormat::Dbl);
        assert_eq!(f.plane(0), &[1, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn f32le_passthrough() {
        let mut dec = PcmDecoder::new();
        dec.init(&params(CodecId::PcmF32le, 2, 48000)).unwrap();
        let bytes: Vec<u8> = [1.0f32, -0.5]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        dec.send_packet(Some(&pkt(&bytes))).unwrap();
        let f = dec.receive_frame().unwrap();
        assert_eq!(f.format, SampleFormat::Flt);
        assert_eq!(f.nb_samples, 1);
        assert_eq!(f.plane(0), &bytes[..]);
    }

    #[test]
    fn s24le_expands_three_to_four_with_shift_8() {
        let mut dec = PcmDecoder::new();
        dec.init(&params(CodecId::PcmS24le, 2, 48000)).unwrap();
        // Stereo, one sample frame: L = 0xFFFFFF (-1), R = 0x000001 (1).
        // DECODE(32, le24, …, 8, 0): (v << 8) as i32 → 0xFFFFFF00 = -256.
        dec.send_packet(Some(&pkt(&[0xFF, 0xFF, 0xFF, 0x01, 0x00, 0x00])))
            .unwrap();
        let f = dec.receive_frame().unwrap();
        assert_eq!(f.format, SampleFormat::S32);
        assert_eq!(f.nb_samples, 1);
        assert_eq!(f.plane(0).len(), 8); // 2 ch x 4 bytes
        let l = i32::from_le_bytes(f.plane(0)[..4].try_into().unwrap());
        let r = i32::from_le_bytes(f.plane(0)[4..].try_into().unwrap());
        assert_eq!(l, -256);
        assert_eq!(r, 256);
    }

    #[test]
    fn s24be_expands_big_endian_word() {
        let mut dec = PcmDecoder::new();
        dec.init(&params(CodecId::PcmS24be, 1, 48000)).unwrap();
        // BE word 0x00004D (77) → (77 << 8) stored native LE.
        dec.send_packet(Some(&pkt(&[0x00, 0x00, 0x4D]))).unwrap();
        let f = dec.receive_frame().unwrap();
        assert_eq!(
            i32::from_le_bytes(f.plane(0)[..4].try_into().unwrap()),
            77 << 8
        );
    }

    // ---- size validation (pcm.c:431-441) ----

    #[test]
    fn short_packet_rejected() {
        let mut dec = PcmDecoder::new();
        dec.init(&params(CodecId::PcmS16le, 2, 48000)).unwrap();
        // 3 bytes < block of 4 → the pcm.c:436-438 error.
        let err = dec.send_packet(Some(&pkt(&[1, 2, 3]))).unwrap_err();
        assert_eq!(
            err,
            Error::InvalidData(
                "Invalid PCM packet, data has size 3 but at least a size of 4 \
                 was expected"
                    .into()
            )
        );
    }

    #[test]
    fn partial_block_truncated_not_rejected() {
        let mut dec = PcmDecoder::new();
        dec.init(&params(CodecId::PcmS16le, 2, 48000)).unwrap();
        // 6 bytes: 6 % 4 = 2 → buf_size 4 → one stereo sample frame.
        dec.send_packet(Some(&pkt(&[1, 2, 3, 4, 5, 6]))).unwrap();
        let f = dec.receive_frame().unwrap();
        assert_eq!(f.nb_samples, 1);
        assert_eq!(f.plane(0), &[1, 2, 3, 4]);
    }

    #[test]
    fn zero_channels_rejected() {
        let mut p = params(CodecId::PcmS16le, 0, 48000);
        p.ch_layout = ChannelLayout::default();
        let mut dec = PcmDecoder::new();
        dec.init(&p).unwrap();
        assert_eq!(
            dec.send_packet(Some(&pkt(&[1, 2]))),
            Err(Error::InvalidData("Invalid number of channels".into()))
        );
    }

    // ---- init gates ----

    #[test]
    fn alaw_mulaw_open_is_unsupported() {
        for id in [CodecId::PcmAlaw, CodecId::PcmMulaw] {
            let mut dec = PcmDecoder::new();
            let err = dec.init(&params(id, 1, 8000)).unwrap_err();
            assert!(matches!(err, Error::Unsupported(_)), "{err}");
        }
    }

    #[test]
    fn non_pcm_codec_rejected() {
        let mut dec = PcmDecoder::new();
        let err = dec.init(&params(CodecId::Rawvideo, 1, 8000)).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
    }
}
