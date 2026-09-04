//! Codec vtable traits — port of `libavcodec/codec_internal.h` (`FFCodec`)
//! and the `libavcodec/decode.c`/`encode.c` push-pull drivers.
//!
//! FFmpeg 8's codec API is a handshake:
//!
//! ```text
//! decode:  send_packet(pkt) → receive_frame()*   (Err(Again) = feed more,
//!          send_packet(None) → drain → Err(Eof)    like C's AVERROR(EAGAIN))
//! encode:  send_frame(frame) → receive_packet()*
//!          send_frame(None) → flush → Err(Eof)
//! ```
//!
//! C expresses this with a `union` of callbacks plus `AVCodecContext` soup;
//! Rust splits it into [`Decoder`] / [`Encoder`] traits with a one-slot
//! output queue inside each implementation (rawvideo emits at most one
//! frame per packet, but the trait keeps the general shape so later codecs
//! — PCM, FFV1 — drop in unchanged).

use crate::codec::packet::Packet;
use crate::codec::params::CodecParameters;
use crate::util::error::Result;
use crate::util::frame::Frame;

/// Decoder interface (`FFCodec` with `FF_CODEC_DECODE_CB`).
pub trait Decoder {
    /// `avcodec_open2` — configure from stream parameters
    /// (`avcodec_parameters_to_context` + open).
    fn init(&mut self, params: &CodecParameters) -> Result<()>;

    /// `avcodec_send_packet` — queue one packet for decoding.
    /// `None` enters drain mode (C: `avcodec_send_packet(avctx, NULL)`).
    fn send_packet(&mut self, pkt: Option<&Packet>) -> Result<()>;

    /// `avcodec_receive_frame` — pull the next decoded frame.
    ///
    /// * `Ok(frame)` — a frame is ready
    /// * `Err(Error::Again)` — needs more input first
    /// * `Err(Error::Eof)` — input ended and all frames were emitted
    fn receive_frame(&mut self) -> Result<Frame>;
}

/// Encoder interface (`FFCodec` with `FF_CODEC_ENCODE_CB`).
pub trait Encoder {
    /// `avcodec_open2` — configure output parameters.
    fn init(&mut self, params: &CodecParameters) -> Result<()>;

    /// `avcodec_send_frame` — queue one frame; `None` flushes
    /// (C: `avcodec_send_frame(avctx, NULL)`).
    fn send_frame(&mut self, frame: Option<&Frame>) -> Result<()>;

    /// `avcodec_receive_packet` — same handshake as the decoder's frames.
    fn receive_packet(&mut self) -> Result<Packet>;
}
