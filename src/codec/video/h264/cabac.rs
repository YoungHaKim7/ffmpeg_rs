//! CABAC arithmetic decoder — port of FFmpeg's `cabac_functions.h`
//! (CABAC_BITS = 16 path) on the tables from `cabac.c`. The engine is a
//! C-verbatim translation: `low` keeps the offset in its high bits and a
//! 16-bit lookahead plus a sentinel bit (the lowest set bit of `low`)
//! below; refills append two bytes at the sentinel position.
//!
//! One deliberate deviation: `ff_init_cabac_decoder` (cabac.c:162) reads
//! a third init byte only when the bytestream pointer is odd-aligned —
//! a C micro-optimization for unaligned loads. Both branches feed the
//! engine the same bit sequence (they only change *when* bytes are
//! fetched), so the port always takes the even path (`low += 1<<9`,
//! consume 2 bytes). The PCM byte-position recovery below accounts for
//! the sentinel layout either way.
//!
//! Reads past `bytestream_end` return 0 (C reads its padded buffer
//! garbage; the slice loop's `bytestream > end + 4` check errors out
//! long before it matters for valid streams). The slice payload is
//! copied in — it is small and this keeps refills borrow-free.

use super::cabac_tables::FF_H264_CABAC_TABLES;
use crate::fferror::{Error, Result};

// cabac.h offsets into ff_h264_cabac_tables.
const NORM_SHIFT_OFF: usize = 0;
const LPS_RANGE_OFF: usize = 512;
const MLPS_STATE_OFF: usize = 1024;
#[allow(dead_code)] // 8x8 residual (Phase D)
pub(super) const LAST_COEFF_FLAG_OFFSET_8X8_OFF: usize = 1280;

#[inline]
fn norm_shift(i: usize) -> i32 {
    FF_H264_CABAC_TABLES[NORM_SHIFT_OFF + i] as i32
}
#[inline]
fn lps_range(i: usize) -> i32 {
    FF_H264_CABAC_TABLES[LPS_RANGE_OFF + i] as i32
}
#[inline]
fn mlps_state(i: usize) -> u8 {
    FF_H264_CABAC_TABLES[MLPS_STATE_OFF + i]
}

const CABAC_BITS: i32 = 16;
const CABAC_MASK: i32 = (1 << CABAC_BITS) - 1;

pub(super) struct Cabac {
    buf: Vec<u8>,
    pub(super) low: i32,
    pub(super) range: i32,
    /// Byte index of the next unfetched byte (C's bytestream; start = 0).
    pub(super) bytestream: usize,
    pub(super) bytestream_end: usize,
}

impl Cabac {
    /// `ff_init_cabac_decoder` (cabac.c:162), even-aligned path.
    pub(super) fn new(buf: &[u8]) -> Result<Cabac> {
        let b = |i: usize| -> i32 { *buf.get(i).unwrap_or(&0) as i32 };
        let low = (b(0) << 18) + (b(1) << 10) + (1 << 9);
        let range = 0x1FE;
        if (range << (CABAC_BITS + 1)) < low {
            return Err(Error::InvalidData("cabac init: offset >= range".into()));
        }
        Ok(Cabac {
            buf: buf.to_vec(),
            low,
            range,
            bytestream: 2,
            bytestream_end: buf.len(),
        })
    }

    /// The slice payload from byte `pos` on (PCM reads / re-init).
    pub(super) fn data_from(&self, pos: usize) -> &[u8] {
        &self.buf[pos.min(self.buf.len())..]
    }

    /// `refill` (cabac_functions.h:64) — bypass path, two bytes.
    fn refill(&mut self) {
        let b = |i: usize| -> i32 { *self.buf.get(i).unwrap_or(&0) as i32 };
        self.low += (b(self.bytestream) << 9) + (b(self.bytestream + 1) << 1);
        self.low -= CABAC_MASK;
        self.bytestream += 2;
    }

    /// `refill2` (cabac_functions.h:89) — decision path; appends two
    /// bytes at the sentinel position (`i = ctz(low) − 16`.
    fn refill2(&mut self) {
        let b = |i: usize| -> i32 { *self.buf.get(i).unwrap_or(&0) as i32 };
        let i = self.low.trailing_zeros() as i32 - CABAC_BITS;
        let x = -CABAC_MASK + (b(self.bytestream) << 9) + (b(self.bytestream + 1) << 1);
        self.low = self.low.wrapping_add(x.wrapping_shl(i as u32));
        self.bytestream += 2;
    }

    /// `get_cabac_inline` (cabac_functions.h:116) — one decision bin.
    #[inline]
    pub(super) fn get(&mut self, state: &mut u8) -> u32 {
        let s = *state as i32;
        let range_lps = lps_range(2 * (self.range as usize & 0xC0) + s as usize);

        self.range -= range_lps;
        // lps_mask = ((range<<17) - low) >> 31  (arithmetic): -1 on LPS.
        let lps_mask = ((self.range << (CABAC_BITS + 1)).wrapping_sub(self.low)) >> 31;

        self.low -= (self.range << (CABAC_BITS + 1)) & lps_mask;
        self.range += (range_lps - self.range) & lps_mask;

        let s = s ^ lps_mask;
        // C: (ff_h264_mlps_state+128)[s] — s in [0..=125] (MPS) or the
        // two's-complement ~state (LPS) mapping to 127-state.
        *state = mlps_state((128i32 + s) as usize);
        let bit = (s & 1) as u32;

        let shift = norm_shift(self.range as usize);
        self.range <<= shift;
        self.low <<= shift;
        if self.low & CABAC_MASK == 0 {
            self.refill2();
        }
        bit
    }

    /// `get_cabac_bypass` (cabac_functions.h:149).
    #[inline]
    pub(super) fn bypass(&mut self) -> u32 {
        self.low += self.low;
        if self.low & CABAC_MASK == 0 {
            self.refill();
        }
        let range = self.range << (CABAC_BITS + 1);
        if self.low < range {
            0
        } else {
            self.low -= range;
            1
        }
    }

    /// `get_cabac_bypass_sign` (cabac_functions.h:167): reads one bypass
    /// bin and returns ±val.
    #[inline]
    pub(super) fn bypass_sign(&mut self, val: i32) -> i32 {
        self.low += self.low;
        if self.low & CABAC_MASK == 0 {
            self.refill();
        }
        let range = self.range << (CABAC_BITS + 1);
        self.low -= range;
        let mask = self.low >> 31;
        self.low += range & mask;
        (val ^ mask) - mask
    }

    /// `get_cabac_terminate` (cabac_functions.h:187): returns non-zero
    /// when the terminate bin decoded as 1.
    #[inline]
    pub(super) fn terminate(&mut self) -> usize {
        self.range -= 2;
        if self.low < self.range << (CABAC_BITS + 1) {
            // renorm_cabac_decoder_once
            let shift = ((self.range as u32).wrapping_sub(0x100) >> 31) as i32;
            self.range <<= shift;
            self.low <<= shift;
            if self.low & CABAC_MASK == 0 {
                self.refill();
            }
            0
        } else {
            self.bytestream
        }
    }

    /// Intra PCM byte-position recovery (h264_cabac.c:2044): the current
    /// bytestream index minus the partially-consumed bytes still in `low`.
    pub(super) fn pcm_ptr(&self) -> usize {
        let mut ptr = self.bytestream;
        if self.low & 0x1 != 0 {
            ptr -= 1;
        }
        if self.low & 0x1FF != 0 {
            ptr -= 1;
        }
        ptr
    }

    /// C's `sl->cabac.bytestream > sl->cabac.bytestream_end + 4` guard.
    pub(super) fn overread(&self) -> bool {
        self.bytestream > self.bytestream_end + 4
    }
}

#[cfg(test)]
mod tests {
    use super::super::cabac_tables::CTX_INIT_I;
    use super::*;

    /// Twin of /tmp/cabac_probe.c: same bytes, same init states, same
    /// ctx schedule — the engines must agree bin for bin.
    #[test]
    fn engine_matches_c_probe() {
        let Ok(buf) = std::fs::read("/tmp/cabac_slice.bin") else {
            eprintln!("skip: no /tmp/cabac_slice.bin");
            return;
        };
        let qp = 23i32;
        let mut state = [0u8; 1024];
        for (i, st) in state.iter_mut().enumerate() {
            let t = CTX_INIT_I[i];
            let mut pre = 2 * (((t[0] as i32 * qp) >> 4) + t[1] as i32) - 127;
            pre ^= pre >> 31;
            if pre > 124 {
                pre = 124 + (pre & 1);
            }
            *st = pre as u8;
        }
        eprintln!(
            "states: 3={} 64={} 67={} 68={} 69={} 73={}",
            state[3], state[64], state[67], state[68], state[69], state[73]
        );
        let mut cab = Cabac::new(&buf).unwrap();
        let ctxs = [
            3, 5, 6, 7, 8, 9, 10, 64, 67, 67, 67, 68, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69,
            69, 69, 69, 69, 69, 69, 69, 69, 69,
        ];
        for (k, &ctx) in ctxs.iter().enumerate() {
            let b = cab.get(&mut state[ctx]);
            eprintln!(
                "bin {k:2} ctx {ctx:3} -> {b} (state now {}, range {}, low {:08x}, bs {})",
                state[ctx], cab.range, cab.low, cab.bytestream
            );
        }
    }
}
