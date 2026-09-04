//! Raw video demuxer + muxer — port of `libavformat/rawvideodec.c` +
//! `rawenc.c`'s `ff_rawvideo_muxer`.
//!
//! The rawvideo demuxer has no container structure: the user *tells* it the
//! pixel format, size and framerate (C: AVOptions `pixel_format`,
//! `video_size`, `framerate`), it computes the fixed frame stride, and
//! `read_packet` chops the file into frames with `pts = pos / frame_size`.
//!
//! Skipped from C: the `linesize[]`/stride padding input option (the whole
//! `has_padding` branch), the `bitpacked`/`v210`/`v210x` sibling demuxers,
//! and zero-fill-with-CORRUPT for short trailing packets (a short tail
//! packet is passed through; the rawvideo *decoder* rejects it, matching
//! C's effective behavior through raw_decode's size check).

use crate::codec::packet::{Packet, PacketFlags};
use crate::codec::params::{CodecId, MediaType};
use crate::imgutils;
use crate::log_error;
use crate::util::error::{Error, Result};
use crate::util::mathematics;
use crate::util::rational::Rational;

use super::demux::{Demuxer, RawVideoDemuxOptions};
use super::io::IoContext;
use super::mux::Muxer;
use super::Stream;

/// `ff_rawvideo_demuxer`'s private state (`RawVideoDemuxerContext`).
pub struct RawVideoDemuxer {
    pix_fmt: crate::util::pixfmt::PixelFormat,
    width: u32,
    height: u32,
    framerate: Rational,
    /// `ctx->packet_size` — bytes per frame (compact layout).
    packet_size: usize,
    time_base: Rational,
}

impl RawVideoDemuxer {
    pub fn new(opts: &RawVideoDemuxOptions) -> Self {
        RawVideoDemuxer {
            pix_fmt: opts.pixel_format,
            width: opts.video_size.map(|(w, _)| w).unwrap_or(0),
            height: opts.video_size.map(|(_, h)| h).unwrap_or(0),
            framerate: opts.framerate,
            packet_size: 0,
            time_base: Rational::UNKNOWN,
        }
    }
}

impl Demuxer for RawVideoDemuxer {
    /// `rawvideo_read_header` (rawvideodec.c:36).
    fn read_header(&mut self, io: &mut IoContext) -> Result<Stream> {
        let mut st = Stream::new_video(0);
        st.codecpar.codec_id = CodecId::Rawvideo;

        // avpriv_set_pts_info(st, 64, framerate.den, framerate.num)
        st.set_pts_info(self.framerate.den as i64, self.framerate.num as i64);

        imgutils::check_size(self.width, self.height)?;
        st.codecpar.width = self.width;
        st.codecpar.height = self.height;
        st.codecpar.format = self.pix_fmt;
        st.sample_aspect_ratio = Rational::new(0, 1);
        st.codecpar.sample_aspect_ratio = st.sample_aspect_ratio;
        st.codecpar.framerate = self.framerate;

        let packet_size = imgutils::get_buffer_size(self.pix_fmt, self.width, self.height, 1)?;
        if packet_size == 0 {
            log_error!(Some("rawvideo"), "Invalid frame size {}x{}.",
                self.width, self.height);
            return Err(Error::InvalidArgument("invalid frame size".into()));
        }
        self.packet_size = packet_size;
        self.time_base = st.time_base;

        // bit_rate = size·8 / time_base (rawvideodec.c:96).
        st.codecpar.bit_rate = mathematics::rescale_q(
            packet_size as i64 * 8,
            Rational::ONE,
            st.time_base,
        );

        // Duration is knowable for seekable inputs; C leaves it unset for
        // pipes. We set it when the size is real (regular files).
        let file_size = io.size();
        if file_size > 0 {
            st.duration = (file_size as i64) / packet_size as i64;
        }
        Ok(st)
    }

    /// `rawvideo_read_packet` (rawvideodec.c, no-padding path).
    fn read_packet(&mut self, io: &mut IoContext) -> Result<Packet> {
        let pos = io.tell();
        let payload = io.get_packet(self.packet_size)?;
        let mut pkt = Packet::from_vec(payload);
        pkt.pos = pos;
        pkt.pts = (pos / self.packet_size as u64) as i64;
        pkt.dts = pkt.pts;
        pkt.duration = 1;
        pkt.time_base = self.time_base;
        pkt.flags = pkt.flags.union(PacketFlags::KEY);
        pkt.stream_index = 0;
        Ok(pkt)
    }
}

/// `ff_rawvideo_muxer` — `ff_raw_write_packet` = plain `avio_write`; no
/// header, no trailer (`AVFMT_NOTIMESTAMPS`).
pub struct RawVideoMuxer;

impl Muxer for RawVideoMuxer {
    fn init(&mut self, streams: &[Stream]) -> Result<()> {
        if streams[0].codecpar.codec_id != CodecId::Rawvideo {
            return Err(Error::InvalidData(
                "rawvideo muxer accepts only rawvideo packets".into(),
            ));
        }
        if streams[0].codecpar.codec_type != MediaType::Video {
            return Err(Error::InvalidData("rawvideo muxer is video-only".into()));
        }
        Ok(())
    }

    fn write_header(&mut self, _io: &mut IoContext, _streams: &[Stream]) -> Result<()> {
        Ok(())
    }

    fn write_packet(&mut self, io: &mut IoContext, _streams: &[Stream], pkt: &Packet) -> Result<()> {
        io.write_all(pkt.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::testutil::MemHandler;
    use crate::util::pixfmt::PixelFormat;

    fn io_with(data: &[u8]) -> IoContext {
        MemHandler::io(data)
    }

    fn opts_64x48() -> RawVideoDemuxOptions {
        RawVideoDemuxOptions {
            pixel_format: PixelFormat::Yuv420p,
            video_size: Some((64, 48)),
            framerate: Rational::new(10, 1),
        }
    }

    #[test]
    fn header_computes_geometry_and_rate() {
        let mut io = io_with(&vec![0u8; 64 * 48 * 3 / 2 * 4]);
        let mut dem = RawVideoDemuxer::new(&opts_64x48());
        let st = dem.read_header(&mut io).unwrap();
        assert_eq!(st.codecpar.format, PixelFormat::Yuv420p);
        assert_eq!(st.time_base, Rational::new(1, 10));
        assert_eq!(st.avg_frame_rate, Rational::new(10, 1));
        assert_eq!(st.duration, 4);
        // 4608 bytes/frame · 8 · 10 fps = 368640 bit/s.
        assert_eq!(st.codecpar.bit_rate, 368_640);
    }

    #[test]
    fn missing_size_rejected() {
        let mut io = io_with(b"");
        let mut dem = RawVideoDemuxer::new(&RawVideoDemuxOptions::default());
        assert!(dem.read_header(&mut io).is_err());
    }

    #[test]
    fn packets_get_sequential_pts() {
        let frame = 64 * 48 * 3 / 2;
        let mut io = io_with(&vec![7u8; frame * 3]);
        let mut dem = RawVideoDemuxer::new(&opts_64x48());
        dem.read_header(&mut io).unwrap();
        for n in 0..3i64 {
            let pkt = dem.read_packet(&mut io).unwrap();
            assert_eq!(pkt.pts, n);
            assert_eq!(pkt.dts, n);
            assert_eq!(pkt.size(), frame);
            assert!(pkt.flags.contains(PacketFlags::KEY));
        }
        assert!(matches!(dem.read_packet(&mut io), Err(Error::Eof)));
    }

    #[test]
    fn muxer_writes_payload_only() {
        let mut io = MemHandler::io(b"");
        let mut st = Stream::new_video(0);
        st.codecpar.codec_id = CodecId::Rawvideo;
        let mut mux = RawVideoMuxer;
        mux.init(&[st.clone()]).unwrap();
        mux.write_header(&mut io, &[st.clone()]).unwrap();
        let mut pkt = Packet::from_vec(vec![1, 2, 3]);
        pkt.stream_index = 0;
        mux.write_packet(&mut io, &[st], &pkt).unwrap();
        mux.write_trailer(&mut io, &[]).unwrap();
        io.seek(0).unwrap();
        assert_eq!(io.get_packet(16).unwrap(), vec![1, 2, 3]);
    }
}
