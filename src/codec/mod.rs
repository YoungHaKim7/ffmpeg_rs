//! `libavcodec` — decode/encode layer.
//!
//! Port map:
//!
//! | FFmpeg | here | status |
//! |---|---|---|
//! | `packet.h` | [`packet`] | full (subset surface) |
//! | `codec_par.h` | [`params`] | video subset |
//! | `codec_internal.h` (`FFCodec`) | [`traits`] | send/receive handshake |
//! | `rawdec.c` + `rawenc.c` | [`rawvideo`] | reachable subset |
//!
//! Not ported in Phase 1: parsers, bitstream filters, threading, hardware
//! paths, every real (lossy) codec.

pub mod packet;
pub mod params;
pub mod rawvideo;
pub mod traits;

pub use packet::{Packet, PacketFlags};
pub use params::{CodecId, CodecParameters, FieldOrder, MediaType};
pub use rawvideo::{RawVideoDecoder, RawVideoEncoder};
