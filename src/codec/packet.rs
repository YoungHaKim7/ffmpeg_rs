//! Packets — port of `libavcodec/packet.h` (the compressed-data container
//! that flows demuxer → decoder and encoder → muxer).
//!
//! C's `AVPacket` is `{AVBufferRef *buf; uint8_t *data; int size; …}` where
//! `buf` owns, `data`/`size` view. Rust collapses the triple into one
//! `Arc<[u8]>`; `Packet::size()` and `as_slice()` keep the C accessors.
//!
//! Timestamps (`pts`, `dts`, `duration`) are `i64` in `time_base` units with
//! [`crate::util::NOPTS`] for "none" — exactly the C contract.

use std::sync::Arc;

use crate::util::rational::Rational;

pub use crate::util::NOPTS;

/// `AV_PKT_FLAG_*` (`packet.h:650`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PacketFlags(pub u32);

impl PacketFlags {
    /// `AV_PKT_FLAG_KEY` (1 << 0) — decodable without prior data. Y4M and
    /// rawvideo mark every packet KEY.
    pub const KEY: PacketFlags = PacketFlags(1 << 0);
    /// `AV_PKT_FLAG_CORRUPT` (1 << 1).
    pub const CORRUPT: PacketFlags = PacketFlags(1 << 1);

    pub const fn contains(self, other: PacketFlags) -> bool {
        self.0 & other.0 == other.0
    }
    pub const fn union(self, other: PacketFlags) -> PacketFlags {
        PacketFlags(self.0 | other.0)
    }
}

/// `AVPacket` — one chunk of coded data plus its timing.
#[derive(Debug, Clone)]
pub struct Packet {
    /// The payload; `Arc` so a demuxer can hand ownership to a decoder
    /// without copying (and several consumers can share).
    pub data: Arc<[u8]>,
    /// In `time_base` units; `NOPTS` when the container carries none.
    pub pts: i64,
    /// Decompression timestamp; `NOPTS` when absent.
    pub dts: i64,
    /// Which stream of the format context this packet belongs to.
    pub stream_index: u32,
    pub flags: PacketFlags,
    /// Duration in `time_base` units.
    pub duration: i64,
    /// Unit of the timestamps above (stream timebase while in the demuxer,
    /// codec timebase after decode-side rescaling).
    pub time_base: Rational,
    /// Byte position of the packet in the input (0 = unknown in Phase 1).
    pub pos: u64,
}

impl Default for Packet {
    fn default() -> Self {
        Packet {
            data: Arc::from(Vec::new()),
            pts: NOPTS,
            dts: NOPTS,
            stream_index: 0,
            flags: PacketFlags(0),
            duration: 0,
            time_base: Rational::UNKNOWN,
            pos: 0,
        }
    }
}

impl Packet {
    /// `av_new_packet` analog — take ownership of a payload `Vec`.
    pub fn from_vec(data: Vec<u8>) -> Packet {
        Packet {
            data: Arc::from(data),
            ..Packet::default()
        }
    }

    /// `pkt->size`.
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// View of `pkt->data[0..size]`.
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }
}
