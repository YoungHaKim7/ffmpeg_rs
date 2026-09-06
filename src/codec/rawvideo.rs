//! Raw video decoder/encoder — port of `libavcodec/rawdec.c` + `rawenc.c`
//! (the reachable subset).
//!
//! Raw video "coding" is re-labeling: the packet payload *is* the frame in a
//! compact layout. So the decoder's whole job is validating the size and
//! attaching the packet buffer to a frame — zero copy
//! (`av_buffer_ref(avpkt->buf)` in C, `Arc` clone here) — and the encoder's
//! job is packing the frame's planes back into one contiguous buffer
//! (`av_image_copy_to_buffer`).
//!
//! ## Skipped C paths, and why they are unreachable here
//!
//! `raw_decode` carries a zoo of legacy codec-tag fix-ups that only fire for
//! AVI/MOV-tagged streams (`rawdec.c` guards on `avctx->codec_tag`):
//!
//! | C branch | Gate | Ported? |
//! |---|---|---|
//! | 1/2/4/8-bpp row repacking | `bits_per_coded_sample` + `raw ` tag | no — Y4M/rawvideo set neither |
//! | `yuv2` chroma sign flip | codec_tag `yuv2` | no |
//! | `b64a` word rotate | codec_tag `b64a` | no — RGBA64 not in subset |
//! | BottomUp vertical flip (negative linesize) | `BottomUp` tag | no — Phase 1 bans negative linesize |
//! | YV12/YV16/YV24 plane swap | `AV_RL32("YV12")` etc. tags | no |
//! | PAL8 palette / extradata | `pal8` pixfmt | no — not in subset |
//!
//! The port is faithful *for the subset the Y4M/rawvideo pipeline can
//! reach*, which is the Phase 1 contract.

use crate::{
    NOPTS,
    codec::{
        packet::{Packet, PacketFlags},
        params::{CodecId, CodecParameters, MediaType},
        traits::{Decoder, Encoder},
    },
    imgutils, log_error, mathematics,
    util::{
        error::{Error, Result},
        frame::{Frame, FrameFlags, PictureType},
        rational::Rational,
    },
};

/// `ff_rawvideo_decoder` — decode rawvideo packets into frames.
#[derive(Debug, Default)]
pub struct RawVideoDecoder {
    params: CodecParameters,
    /// One-frame output queue (rawvideo has no reordering delay).
    pending: Option<Frame>,
    /// Drain requested (`send_packet(NULL)`).
    eof: bool,
}

impl RawVideoDecoder {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Decoder for RawVideoDecoder {
    fn init(&mut self, params: &CodecParameters) -> Result<()> {
        if params.width == 0 || params.height == 0 {
            // C: "width is not set" / "height is not set" (rawdec.c:159-165).
            return Err(Error::InvalidData("width is not set".into()));
        }
        self.params = params.clone();
        self.params.codec_type = MediaType::Video;
        self.params.codec_id = CodecId::Rawvideo;
        Ok(())
    }

    fn send_packet(&mut self, pkt: Option<&Packet>) -> Result<()> {
        let Some(pkt) = pkt else {
            self.eof = true;
            return Ok(());
        };
        if self.eof {
            // C rejects input after drain in avcodec_send_packet.
            return Err(Error::Eof);
        }
        let w = self.params.width;
        let h = self.params.height;

        // C's stride sanity (rawdec.c:174-181): stride = size/height, and the
        // packet must cover stride*height.
        let stride = pkt.size() / h as usize;
        if stride == 0 || pkt.size() < stride * h as usize {
            log_error!(Some("rawvideo"), "Packet too small ({})", pkt.size());
            return Err(Error::InvalidData("packet too small".into()));
        }

        // Compact frame size for this format (rawdec.c:196-198 path without
        // the 1/2/4/8-bpp branch).
        let frame_size = imgutils::get_buffer_size(self.params.format, w, h, 1)?;
        if pkt.size() < frame_size {
            log_error!(Some("rawvideo"), "Packet too small ({})", pkt.size());
            return Err(Error::InvalidData("packet too small".into()));
        }

        // Zero-copy: adopt the packet buffer as the frame backing store.
        let mut frame = Frame::wrap_buffer(pkt.data.clone(), self.params.format, w, h)?;

        // ff_decode_frame_props_from_pkt: timing comes off the packet.
        frame.pts = pkt.pts;
        frame.duration = pkt.duration;
        frame.time_base = pkt.time_base;
        if pkt.flags.contains(PacketFlags::KEY) {
            frame.flags = frame.flags.union(FrameFlags::KEY);
        }

        // fill_frame_props: color/SAR come off the codec context, which was
        // itself filled from the stream's codecpar.
        frame.sample_aspect_ratio = self.params.sample_aspect_ratio;
        frame.color_range = self.params.color_range;
        frame.color_primaries = self.params.color_primaries;
        frame.color_trc = self.params.color_trc;
        frame.color_space = self.params.color_space;
        frame.chroma_location = self.params.chroma_location;
        frame.pict_type = PictureType::I;

        self.pending = Some(frame);
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        match self.pending.take() {
            Some(frame) => Ok(frame),
            None if self.eof => Err(Error::Eof),
            None => Err(Error::Again),
        }
    }
}

/// `ff_rawvideo_encoder` — pack frames into rawvideo packets.
#[derive(Debug, Default)]
pub struct RawVideoEncoder {
    params: CodecParameters,
    /// Encoder timebase — C uses avctx->time_base; the CLI sets it from the
    /// output framerate. Defaults to 1/25 and is overwritten by `init`.
    time_base: Rational,
    pending: Option<Packet>,
    eof: bool,
}

impl RawVideoEncoder {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Encoder for RawVideoEncoder {
    fn init(&mut self, params: &CodecParameters) -> Result<()> {
        if params.width == 0 || params.height == 0 {
            return Err(Error::InvalidData("width is not set".into()));
        }
        self.params = params.clone();
        self.params.codec_type = MediaType::Video;
        self.params.codec_id = CodecId::Rawvideo;
        // Encoder timebase follows the framerate (what ffmpeg's CLI does for
        // rawvideo: -framerate 25 → time_base 1/25).
        self.time_base = if params.framerate.num > 0 {
            params.framerate.inv()
        } else {
            Rational::new(1, 25)
        };
        // bits_per_coded_sample (raw_encode_init) is only used for tags; skip.
        Ok(())
    }

    fn send_frame(&mut self, frame: Option<&Frame>) -> Result<()> {
        let Some(frame) = frame else {
            self.eof = true;
            return Ok(());
        };
        if self.eof {
            return Err(Error::Eof);
        }

        // raw_encode (rawenc.c:49): size = compact buffer size, then
        // av_image_copy_to_buffer with align 1.
        let planes: Vec<&[u8]> = frame.planes.iter().map(|p| p.data()).collect();
        let mut linesizes = [0usize; 4];
        for (i, p) in frame.planes.iter().enumerate().take(4) {
            linesizes[i] = p.linesize;
        }
        let data = imgutils::copy_to_buffer(
            frame.format,
            frame.width,
            frame.height,
            1,
            &planes,
            &linesizes,
        )?;

        let mut pkt = Packet::from_vec(data);
        // encode.c's generic path: packet takes the frame's timing, rescaled
        // into the encoder timebase.
        pkt.pts = if frame.pts != NOPTS {
            mathematics::rescale_q(frame.pts, frame.time_base, self.time_base)
        } else {
            NOPTS
        };
        pkt.duration = if frame.duration > 0 {
            mathematics::rescale_q(frame.duration, frame.time_base, self.time_base).max(1)
        } else {
            1
        };
        pkt.time_base = self.time_base;
        pkt.flags = pkt.flags.union(PacketFlags::KEY);

        self.pending = Some(pkt);
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        match self.pending.take() {
            Some(pkt) => Ok(pkt),
            None if self.eof => Err(Error::Eof),
            None => Err(Error::Again),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::pixfmt::PixelFormat;

    fn stream_params() -> CodecParameters {
        let mut p = CodecParameters::default();
        p.codec_type = MediaType::Video;
        p.codec_id = CodecId::Rawvideo;
        p.format = PixelFormat::Yuv420p;
        p.width = 64;
        p.height = 48;
        p.framerate = Rational::new(25, 1);
        p
    }

    fn sample_packet(data: Vec<u8>, pts: i64) -> Packet {
        let mut pkt = Packet::from_vec(data);
        pkt.pts = pts;
        pkt.duration = 1;
        pkt.time_base = Rational::new(1, 25);
        pkt.flags = pkt.flags.union(PacketFlags::KEY);
        pkt
    }

    #[test]
    fn decode_is_zero_copy_and_carries_props() {
        let mut dec = RawVideoDecoder::new();
        dec.init(&stream_params()).unwrap();

        let payload: Vec<u8> = (0..64 * 48 * 3 / 2).map(|i| (i % 251) as u8).collect();
        let pkt = sample_packet(payload, 7);
        dec.send_packet(Some(&pkt)).unwrap();
        let frame = dec.receive_frame().unwrap();

        assert_eq!((frame.width, frame.height), (64, 48));
        assert_eq!(frame.pts, 7);
        assert_eq!(frame.time_base, Rational::new(1, 25));
        assert!(frame.flags.contains(FrameFlags::KEY));
        assert_eq!(frame.pict_type, PictureType::I);
        // Plane shares the packet buffer.
        assert!(std::sync::Arc::ptr_eq(&frame.planes[0].buf, &pkt.data));
        assert_eq!(frame.plane(0)[5], 5);

        assert!(matches!(dec.receive_frame(), Err(Error::Again)));
        dec.send_packet(None).unwrap();
        assert!(matches!(dec.receive_frame(), Err(Error::Eof)));
    }

    #[test]
    fn decode_rejects_short_packets() {
        let mut dec = RawVideoDecoder::new();
        dec.init(&stream_params()).unwrap();
        let pkt = sample_packet(vec![0u8; 100], 0);
        assert!(matches!(
            dec.send_packet(Some(&pkt)),
            Err(Error::InvalidData(_))
        ));
    }

    #[test]
    fn encode_decode_round_trip_is_byte_exact() {
        let params = stream_params();
        let mut enc = RawVideoEncoder::new();
        enc.init(&params).unwrap();

        let mut frame = Frame::alloc(PixelFormat::Yuv420p, 64, 48).unwrap();
        frame
            .plane_mut(0)
            .iter_mut()
            .enumerate()
            .for_each(|(i, b)| *b = (i % 251) as u8);
        frame
            .plane_mut(1)
            .iter_mut()
            .enumerate()
            .for_each(|(i, b)| *b = (i % 13) as u8);
        frame
            .plane_mut(2)
            .iter_mut()
            .enumerate()
            .for_each(|(i, b)| *b = (i % 7) as u8);
        frame.pts = 3;
        frame.duration = 1;
        frame.time_base = Rational::new(1, 25);

        enc.send_frame(Some(&frame)).unwrap();
        let pkt = enc.receive_packet().unwrap();
        assert_eq!(pkt.size(), 64 * 48 * 3 / 2);
        assert_eq!(pkt.pts, 3);
        assert!(pkt.flags.contains(PacketFlags::KEY));
        assert!(matches!(enc.receive_packet(), Err(Error::Again)));
        enc.send_frame(None).unwrap();
        assert!(matches!(enc.receive_packet(), Err(Error::Eof)));

        let mut dec = RawVideoDecoder::new();
        dec.init(&params).unwrap();
        dec.send_packet(Some(&pkt)).unwrap();
        let back = dec.receive_frame().unwrap();
        assert_eq!(back.plane(0), frame.plane(0));
        assert_eq!(back.plane(1), frame.plane(1));
        assert_eq!(back.plane(2), frame.plane(2));
        assert_eq!(back.pts, 3);
    }
}
