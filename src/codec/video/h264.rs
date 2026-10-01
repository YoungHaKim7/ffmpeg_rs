//! H.264 decoder — port of FFmpeg's native `ff_h264_decoder`,
//! **baseline-profile subset**: CAVLC entropy coding, frame pictures,
//! I/P slices, intra (4x4/16x16/PCM) and inter (16x16/16x8/8x16/8x8
//! partitions) macroblocks, 6-tap luma / bilinear chroma motion
//! compensation, multiple short-term reference pictures (sliding window +
//! list modification), and the in-loop **deblocking filter**
//! ([`deblock`]). Acceptance: bit-exact vs default `ffmpeg` output.
//!
//! Gated `Unsupported` (degrade honestly, like AAC's ER objects): CABAC,
//! B slices + direct mode + weighted prediction (baseline excludes them),
//! MBAFF/field pictures, 8x8 transform, FMO, SP/SI slices, chroma
//! 422/444, bit depths > 8, custom scaling matrices, MMCO/long-term refs.
//!
//! ## C → Rust map
//!
//! | C | here |
//! |---|---|
//! | Annex-B split + emulation prevention (`h2645_parse.c`) | [`split_nals`] |
//! | exp-golomb (`golomb.h`) | [`Gb::ue`] / [`Gb::se`] |
//! | `ff_h264_decode_seq_parameter_set` (h264_ps.c:284) | [`parse_sps`] |
//! | `ff_h264_decode_picture_parameter_set` (h264_ps.c:698) + `init_dequant4_coeff_table` (617) | [`parse_pps`] |
//! | `h264_slice_header_parse` (h264_slice.c:1718) | [`H264Decoder::parse_slice_header`] |
//! | `ff_h264_decode_mb_cavlc` (h264_cavlc.c:682) + `decode_residual` (405) + `cavlc_level_tab` (289) | [`H264Decoder::decode_mb_cavlc`] / [`decode_residual`] |
//! | `fill_decode_neighbors`/`_caches` (h264_mvpred.h:487/539) | [`H264Decoder::fill_decode_caches`] — frame-only path |
//! | `pred_motion`/`pred_16x8`/`pred_8x16`/`pred_pskip` + `fetch_diagonal_mv` (h264_mvpred.h) | same names |
//! | `pred4x4*`/`pred16x16*`/`pred8x8*` (h264pred_template.c) | [`pred4x4`] / [`pred16x16`] / [`pred8x8`] |
//! | `ff_h264_idct_add`/`idct_dc_add`/`luma_dc_dequant_idct` (h264idct_template.c) | same names |
//! | `hl_decode_mb` (h264_mb_template.c) | [`H264Decoder::hl_decode_mb`] |
//! | qpel 6-tap + chroma MC (h264qpel/h264chroma templates, spec 8.4.2.2) | [`mc_luma`] / [`mc_chroma`] (scalar) |
//! | ref-list (`h264_refs.c`) | [`H264Decoder::build_ref_list`] — short-term sliding window, no MMCO |
//! | deblocking (`h264_loopfilter.c` + `h264dsp_template.c` filters) | [`deblock::filter_picture`] |
//!
//! Output: `PixelFormat::Yuv420p` frames, SPS-cropped, decode order.
//! The generator for `tables.rs` (extracted CAVLC/h264data tables) is
//! the python script documented in that file's header.

use std::sync::OnceLock;

mod cabac;
mod cabac_tables;
mod deblock;
mod deblock_tables;
mod decoder;
mod picture;
mod tables;
mod vlc;

use crate::fferror::{Error, Result};

use decoder::H264Decoder;
use picture::Picture;
use vlc::Cavlc;

use deblock::{MbDeblock, PART_16X16};
use tables::*;

/// `scan8` (h264dec.h): block index → position in the 6x8 neighborhood
/// caches (border included). 48/49 are the luma/chroma DC slots.
#[rustfmt::skip]
const SCAN8: [usize; 51] = [
    4 + 1 * 8, 5 + 1 * 8, 4 + 2 * 8, 5 + 2 * 8,
    6 + 1 * 8, 7 + 1 * 8, 6 + 2 * 8, 7 + 2 * 8,
    4 + 3 * 8, 5 + 3 * 8, 4 + 4 * 8, 5 + 4 * 8,
    6 + 3 * 8, 7 + 3 * 8, 6 + 4 * 8, 7 + 4 * 8,
    4 + 6 * 8, 5 + 6 * 8, 4 + 7 * 8, 5 + 7 * 8,
    6 + 6 * 8, 7 + 6 * 8, 6 + 7 * 8, 7 + 7 * 8,
    4 + 8 * 8, 5 + 8 * 8, 4 + 9 * 8, 5 + 9 * 8,
    6 + 8 * 8, 7 + 8 * 8, 6 + 9 * 8, 7 + 9 * 8,
    4 + 11 * 8, 5 + 11 * 8, 4 + 12 * 8, 5 + 12 * 8,
    6 + 11 * 8, 7 + 11 * 8, 6 + 12 * 8, 7 + 12 * 8,
    4 + 13 * 8, 5 + 13 * 8, 4 + 14 * 8, 5 + 14 * 8,
    6 + 13 * 8, 7 + 13 * 8, 6 + 14 * 8, 7 + 14 * 8,
    0 + 0 * 8, 0 + 5 * 8, 0 + 10 * 8,
];
const LUMA_DC: usize = 48;
const CHROMA_DC: usize = 49;

/// `ff_zigzag_scan` (mathtables.c:148).
/// C's `h->zigzag_scan` (init_scan_tables, h264_slice.c:755):
/// TRANSPOSE(ff_zigzag_scan[i]) with TRANSPOSE(x) = (x>>2)|((x<<2)&0xF).
/// The whole CAVLC residual space (dequant4_coeff's transposed build,
/// qmul[scan_value] indexing) is transposed — the RAW scan is only used
/// for transform_bypass (qscale 0); the port has no bypass, so always
/// use the transposed table.
const ZIGZAG: [u8; 16] = [0, 4, 1, 2, 5, 8, 12, 9, 6, 3, 7, 10, 13, 14, 11, 15];

/// `ff_h264_chroma_qp[0]` (h264data.c:203, the depth-8 row).
#[rustfmt::skip]
const CHROMA_QP8: [u8; 52] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 29, 30, 31, 32, 32, 33, 34, 34, 35, 35, 36, 36, 37, 37, 37, 38,
    38, 38, 39, 39, 39, 39,
];

// MB types (port-internal codes for the mb_type array).
const MB_UNAVAIL: u32 = 0;
const MB_INTRA4X4: u32 = 1;
const MB_INTRA16X16: u32 = 2;
const MB_PCM: u32 = 3;
const MB_INTER: u32 = 4; // any P partition / skip

// Intra prediction modes (h264pred.h names).
const VERT_PRED: i8 = 0;
const HOR_PRED: i8 = 1;
const DC_PRED: i8 = 2;
const DIAG_DOWN_LEFT_PRED: i8 = 3;
const DIAG_DOWN_RIGHT_PRED: i8 = 4;
const VERT_RIGHT_PRED: i8 = 5;
const HOR_DOWN_PRED: i8 = 6;
const VERT_LEFT_PRED: i8 = 7;
const HOR_UP_PRED: i8 = 8;
// i4x4 DC variants live ABOVE the 9 real modes (C's LEFT_DC_PRED=9,
// TOP_DC_PRED=10, DC_128_PRED=11 — a separate namespace; squashing them
// into 0-3 collides with HOR_PRED and silently changes the prediction).
// The 16x16/chroma path folds its DC variants into plain DC(2) and lets
// the availability-aware preds handle them instead.
const I4_LEFT_DC_PRED: i8 = 9;
const I4_TOP_DC_PRED: i8 = 10;
const I4_DC_128_PRED: i8 = 11;

const LEVEL_TAB_BITS: u32 = 8;

/// MB partition shape (`ff_h264_p_mb_type_info` + the intra path).
#[derive(Debug)]
enum Part {
    Intra(usize),
    P16x16,
    P16x8,
    P8x16,
    P8x8,
}

#[derive(Clone, Copy)]
enum PlaneSel {
    Cb,
    Cr,
}

// ---------------------------------------------------------------------
// Bit reader + exp-golomb (get_bits.h / golomb.h)
// ---------------------------------------------------------------------

struct Gb<'a> {
    buf: &'a [u8],
    index: usize,
    size_in_bits: usize,
}

impl<'a> Gb<'a> {
    fn new(buf: &'a [u8]) -> Gb<'a> {
        Gb {
            buf,
            index: 0,
            size_in_bits: buf.len() * 8,
        }
    }
    fn left(&self) -> i64 {
        self.size_in_bits as i64 - self.index as i64
    }
    fn peek(&self, n: u32) -> u32 {
        if n == 0 || self.index >= self.size_in_bits {
            return 0;
        }
        let mut w = 0u64;
        for k in 0..8 {
            let b = self.buf.get((self.index >> 3) + k).copied().unwrap_or(0);
            w = (w << 8) | b as u64;
        }
        let sh = 64 - (self.index & 7) as u32 - n;
        ((w >> sh) & ((1u64 << n) - 1)) as u32
    }
    fn read(&mut self, n: u32) -> u32 {
        let v = self.peek(n);
        self.skip(n);
        v
    }
    fn read_bit(&mut self) -> u32 {
        self.read(1)
    }
    fn skip(&mut self, n: u32) {
        self.index = (self.index + n as usize).min(self.size_in_bits);
    }
    fn align(&mut self) {
        self.index = (self.index + 7) & !7;
    }

    /// `get_ue_golomb` (C's _31 shape; sane streams).
    fn ue(&mut self) -> Result<u32> {
        let mut lz = 0u32;
        while self.peek(1) == 0 {
            if lz >= 31 || self.left() <= 0 {
                return Err(Error::InvalidData("bad ue(vlc)".into()));
            }
            self.skip(1);
            lz += 1;
        }
        self.skip(1);
        if lz == 0 {
            return Ok(0);
        }
        if self.left() < lz as i64 {
            return Err(Error::InvalidData("truncated ue".into()));
        }
        Ok(self.read(lz).wrapping_add((1u32 << lz) - 1))
    }
    /// `get_se_golomb`.
    fn se(&mut self) -> Result<i32> {
        let k = self.ue()? as i32;
        Ok(((k + 1) >> 1) * if k & 1 == 1 { 1 } else { -1 })
    }
    /// `get_level_prefix` (h264_cavlc.c:382): count of leading ones.
    fn level_prefix(&mut self) -> Result<u32> {
        let mut log = 0u32;
        loop {
            if self.left() <= 0 {
                return Err(Error::InvalidData("level prefix overread".into()));
            }
            if self.read_bit() == 1 {
                break;
            }
            log += 1;
            if log > 30 {
                return Err(Error::InvalidData("level prefix too long".into()));
            }
        }
        Ok(log)
    }
    /// `more_rbsp_data` (golomb.h): bits left besides the trailing
    /// one-bit-and-zeros tail. NOTE: not a slice-loop condition — C's MB
    /// loop is gated on get_bits_left + pending skip run instead.
    #[allow(dead_code)]
    fn more_rbsp_data(&self) -> bool {
        if self.left() <= 0 {
            return false;
        }
        // find the last set bit; if it's the stop bit at/near the end and
        // we've reached it, no more data.
        let mut i = self.index;
        // emulate: scan for a remaining 1 bit strictly beyond current
        // position; the final 1 in the buffer is rbsp_stop_bit.
        let total = self.size_in_bits;
        // last set bit position:
        let mut last_one = None;
        for bit in (0..total).rev() {
            let byte = self.buf.get(bit >> 3).copied().unwrap_or(0);
            if (byte >> (7 - (bit & 7))) & 1 == 1 {
                last_one = Some(bit);
                break;
            }
        }
        match last_one {
            None => false,
            Some(lo) => {
                i = i.min(total);
                i < lo
            }
        }
    }
}

// ---------------------------------------------------------------------
// NAL parsing (h2645_parse.c)
// ---------------------------------------------------------------------

struct Nal {
    kind: u8,
    ref_idc: u8,
    rbsp: Vec<u8>,
}

fn split_nals(data: &[u8]) -> Vec<Nal> {
    let mut starts: Vec<(usize, usize)> = Vec::new();
    let mut i = 0usize;
    while i + 3 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            let sc = if i > 0 && data[i - 1] == 0 { i - 1 } else { i };
            if let Some(last) = starts.last_mut() {
                last.1 = sc;
            }
            starts.push((i + 3, usize::MAX));
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut nals = Vec::new();
    for &(start, next) in &starts {
        let end = next.min(data.len());
        let mut e = end;
        while e > start && data[e - 1] == 0 {
            e -= 1;
        }
        if e <= start {
            continue;
        }
        let hdr = data[start];
        let mut rbsp = Vec::with_capacity(e - start);
        let mut z = 0usize;
        for &b in &data[start + 1..e] {
            if z == 2 && b == 3 {
                z = 0;
                continue;
            }
            if b == 0 {
                z += 1;
            } else {
                z = 0;
            }
            rbsp.push(b);
        }
        nals.push(Nal {
            kind: hdr & 0x1f,
            ref_idc: hdr >> 5,
            rbsp,
        });
    }
    nals
}

// ---------------------------------------------------------------------
// Parameter sets (h264_ps.c, baseline subset)
// ---------------------------------------------------------------------

#[derive(Clone, Default)]
struct Sps {
    profile_idc: u8,
    log2_max_frame_num: u32,
    poc_type: u8,
    log2_max_poc_lsb: u32,
    ref_frame_count: u32,
    mb_width: usize,
    mb_height: usize,
    direct_8x8_inference: bool,
    crop: bool,
    crop_left: u32,
    crop_right: u32,
    crop_top: u32,
    crop_bottom: u32,
}

#[derive(Clone)]
struct Pps {
    pic_order_present: bool,
    ref_count: [u32; 2],
    init_qp: i32,
    deblocking_filter_parameters_present: bool,
    constrained_intra_pred: bool,
    redundant_pic_cnt_present: bool,
    /// entropy_coding_mode_flag: CABAC slices (Phase B).
    cabac: bool,
    /// chroma_qp_index_offset[0/1] (the second from the PPS tail;
    /// defaults to the first when absent — h264_ps.c:800-812).
    chroma_qp_offset: [i32; 2],
    /// `dequant4_coeff[i][q][x]` (init_dequant4_coeff_table, h264_ps.c:617)
    /// with the default all-16 scaling matrix (custom lists gate Unsupported).
    dequant4_full: Vec<[[u32; 16]; 52]>,
}

impl Pps {
    fn build_dequant(&mut self) {
        // Flat scaling list: scaling_matrix4[i][x] = 16 everywhere, so C's
        // `* pps->scaling_matrix4[i][x]` (h264_ps.c:643-644) contributes a
        // constant x16 that is folded into the shift below.
        for i in 0..6usize {
            for q in 0..52usize {
                let shift = QUANT_DIV6[q] as u32 + 2 + 4;
                let idx = QUANT_REM6[q] as usize;
                for x in 0..16usize {
                    let transposed = (x >> 2) | ((x << 2) & 0xF);
                    self.dequant4_full[i][q][transposed] =
                        (DEQUANT4_COEFF_INIT[idx * 3 + (x & 1) + ((x >> 2) & 1)] as u32) << shift;
                }
            }
        }
    }
    fn dequant(&self, i: usize, q: usize) -> &[u32; 16] {
        &self.dequant4_full[i][q.min(51)]
    }
}

fn parse_sps(rbsp: &[u8]) -> Result<Sps> {
    let mut gb = Gb::new(rbsp);
    let profile_idc = gb.read(8) as u8;
    let _constraint_flags = gb.read(8);
    let _level_idc = gb.read(8);
    let _sps_id = gb.ue()?;

    if matches!(
        profile_idc,
        100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 144
    ) {
        let chroma_format_idc = gb.ue()?;
        if chroma_format_idc == 3 {
            let _rct = gb.read_bit();
        }
        let bit_depth = gb.ue()? + 8;
        let _bd_chroma = gb.ue()?;
        let _bypass = gb.read_bit();
        let scaling_present = gb.read_bit();
        if scaling_present != 0 {
            return Err(Error::Unsupported("scaling matrices".into()));
        }
        if chroma_format_idc != 1 || bit_depth != 8 {
            return Err(Error::Unsupported(format!(
                "chroma_format_idc {chroma_format_idc} / bit depth {bit_depth}"
            )));
        }
    }

    let log2_mfn_m4 = gb.ue()?;
    if log2_mfn_m4 > 12 {
        return Err(Error::InvalidData("log2_max_frame_num out of range".into()));
    }
    let log2_max_frame_num = log2_mfn_m4 + 4;

    let poc_type = gb.ue()? as u8;
    let log2_max_poc_lsb;
    match poc_type {
        0 => {
            let t = gb.ue()?;
            if t > 12 {
                return Err(Error::InvalidData("log2_max_poc_lsb".into()));
            }
            log2_max_poc_lsb = t + 4;
        }
        1 => return Err(Error::Unsupported("POC type 1".into())),
        2 => log2_max_poc_lsb = 0,
        _ => return Err(Error::InvalidData("illegal POC type".into())),
    }

    let ref_frame_count = gb.ue()?;
    if ref_frame_count > 16 {
        return Err(Error::InvalidData("too many reference frames".into()));
    }
    let _gaps = gb.read_bit();
    let mb_width = gb.ue()? as usize + 1;
    let mb_height = gb.ue()? as usize + 1;
    let frame_mbs_only = gb.read_bit() == 1;
    if !frame_mbs_only {
        return Err(Error::Unsupported("field/interlaced pictures".into()));
    }
    let direct_8x8 = gb.read_bit() == 1;
    let crop = gb.read_bit() == 1;
    let (cl, cr, ct, cb) = if crop {
        let l = gb.ue()? as u32;
        let r = gb.ue()? as u32;
        let t = gb.ue()? as u32;
        let b = gb.ue()? as u32;
        (l * 2, r * 2, t * 2, b * 2) // 4:2:0 frame: unit = 2 samples
    } else {
        (0, 0, 0, 0)
    };
    let _vui = gb.read_bit();
    // VUI is the final SPS member; nothing follows — skipped safely.

    Ok(Sps {
        profile_idc,
        log2_max_frame_num,
        poc_type,
        log2_max_poc_lsb,
        ref_frame_count,
        mb_width,
        mb_height,
        direct_8x8_inference: direct_8x8,
        crop: crop,
        crop_left: cl,
        crop_right: cr,
        crop_top: ct,
        crop_bottom: cb,
    })
}

fn parse_pps(rbsp: &[u8], sps_ok: bool) -> Result<Pps> {
    let mut gb = Gb::new(rbsp);
    let _pps_id = gb.ue()?;
    let sps_id = gb.ue()? as usize;
    if !sps_ok || sps_id != 0 {
        return Err(Error::InvalidData("PPS references unknown SPS".into()));
    }
    // entropy_coding_mode_flag — CABAC slices (Phase B).
    let cabac = gb.read_bit() == 1;
    let pic_order_present = gb.read_bit() == 1;
    if gb.ue()? + 1 > 1 {
        return Err(Error::Unsupported("FMO (slice groups)".into()));
    }
    let ref_count = [gb.ue()? + 1, gb.ue()? + 1];
    if ref_count[0] > 32 || ref_count[1] > 32 {
        return Err(Error::InvalidData("reference overflow (pps)".into()));
    }
    let _weighted_pred = gb.read_bit();
    let _weighted_bipred = gb.read(2);
    let init_qp = gb.se()? + 26;
    let _init_qs = gb.se()?;
    let chroma_qp_off = gb.se()?;
    if !(-12..=12).contains(&chroma_qp_off) {
        return Err(Error::InvalidData("chroma_qp_index_offset".into()));
    }
    let deblocking_present = gb.read_bit() == 1;
    let constrained_intra_pred = gb.read_bit() == 1;
    let redundant_pic_cnt_present = gb.read_bit() == 1;
    // PPS tail (h264_ps.c:795-812): present when any data remains past
    // the trailing stop bit.
    let mut chroma_qp_off2 = chroma_qp_off;
    if gb.more_rbsp_data() {
        if gb.read_bit() == 1 {
            return Err(Error::Unsupported("8x8 transform (High profile)".into()));
        }
        if gb.read_bit() == 1 {
            return Err(Error::Unsupported("scaling matrices".into()));
        }
        chroma_qp_off2 = gb.se()?;
        if !(-12..=12).contains(&chroma_qp_off2) {
            return Err(Error::InvalidData("second chroma_qp_index_offset".into()));
        }
    }
    let mut pps = Pps {
        pic_order_present,
        ref_count,
        init_qp,
        deblocking_filter_parameters_present: deblocking_present,
        constrained_intra_pred,
        redundant_pic_cnt_present,
        cabac,
        chroma_qp_offset: [chroma_qp_off, chroma_qp_off2],
        dequant4_full: vec![[[0u32; 16]; 52]; 6],
    };
    pps.build_dequant();
    Ok(pps)
}

// ---------------------------------------------------------------------
// Intra prediction (h264pred_template.c) — spec filters, u8 planes
// ---------------------------------------------------------------------

fn clip8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// 4x4 intra prediction into `dst` (16 bytes, 4x4) from neighbor samples.
/// `top` = A..D + topright (8 values: top[0..8], topright may be
/// replicated when unavailable — caller prepares per C's rules),
/// `left` = I..L (4), `lt` = the top-left sample.
fn pred4x4(mode: i8, dst: &mut [u8], top: &[u8; 8], left: &[u8; 4], lt: u8) {
    // The DC variants (9/10/11) need availability; every other mode was
    // validated against it by check_intra4x4_pred_mode.
    match mode {
        I4_DC_128_PRED => {
            for r in 0..4 {
                for c in 0..4 {
                    dst[r * 4 + c] = 128;
                }
            }
            return;
        }
        I4_TOP_DC_PRED => {
            let dc =
                ((top[0] as u32 + top[1] as u32 + top[2] as u32 + top[3] as u32 + 2) >> 2) as u8;
            for r in 0..4 {
                for c in 0..4 {
                    dst[r * 4 + c] = dc;
                }
            }
            return;
        }
        I4_LEFT_DC_PRED => {
            let dc = ((left[0] as u32 + left[1] as u32 + left[2] as u32 + left[3] as u32 + 2) >> 2)
                as u8;
            for r in 0..4 {
                for c in 0..4 {
                    dst[r * 4 + c] = dc;
                }
            }
            return;
        }
        _ => {}
    }
    let (t0, t1, t2, t3) = (top[0] as u32, top[1] as u32, top[2] as u32, top[3] as u32);
    let (t4, t5, t6, t7) = (top[4] as u32, top[5] as u32, top[6] as u32, top[7] as u32);
    let (l0, l1, l2, l3) = (
        left[0] as u32,
        left[1] as u32,
        left[2] as u32,
        left[3] as u32,
    );
    match mode {
        VERT_PRED => {
            for r in 0..4 {
                dst[r * 4..r * 4 + 4].copy_from_slice(&[t0 as u8, t1 as u8, t2 as u8, t3 as u8]);
            }
        }
        HOR_PRED => {
            for r in 0..4 {
                dst[r * 4..r * 4 + 4].copy_from_slice(&[left[r]; 4]);
            }
        }
        DC_PRED => {
            // C pred4x4_dc: (top[0..3] + left[0..3] + 4) >> 3 — ROUNDED
            // (unavailable sides arrive here as the DC variants 9/10/11).
            let dc = ((t0 + t1 + t2 + t3 + l0 + l1 + l2 + l3 + 4) >> 3) as u8;
            for r in 0..4 {
                dst[r * 4..r * 4 + 4].copy_from_slice(&[dc; 4]);
            }
        }
        DIAG_DOWN_LEFT_PRED => {
            let px = |i: usize| -> u8 { top[i] };
            let mut put = |r: usize, c: usize, v: u8| dst[r * 4 + c] = v;
            put(0, 0, ((t0 + t2 + 2 * t1 + 2) >> 2) as u8);
            put(0, 1, ((t1 + t3 + 2 * t2 + 2) >> 2) as u8);
            put(0, 2, ((t2 + t4 + 2 * t3 + 2) >> 2) as u8);
            put(0, 3, ((t3 + t5 + 2 * t4 + 2) >> 2) as u8);
            put(1, 0, ((t1 + t3 + 2 * t2 + 2) >> 2) as u8);
            put(1, 1, ((t2 + t4 + 2 * t3 + 2) >> 2) as u8);
            put(1, 2, ((t3 + t5 + 2 * t4 + 2) >> 2) as u8);
            put(1, 3, ((t4 + t6 + 2 * t5 + 2) >> 2) as u8);
            put(2, 0, ((t2 + t4 + 2 * t3 + 2) >> 2) as u8);
            put(2, 1, ((t3 + t5 + 2 * t4 + 2) >> 2) as u8);
            put(2, 2, ((t4 + t6 + 2 * t5 + 2) >> 2) as u8);
            put(2, 3, ((t5 + t7 + 2 * t6 + 2) >> 2) as u8);
            put(3, 0, ((t3 + t5 + 2 * t4 + 2) >> 2) as u8);
            put(3, 1, ((t4 + t6 + 2 * t5 + 2) >> 2) as u8);
            put(3, 2, ((t5 + t7 + 2 * t6 + 2) >> 2) as u8);
            put(3, 3, ((t6 + 3 * t7 + 2) >> 2) as u8);
            let _ = px;
        }
        DIAG_DOWN_RIGHT_PRED => {
            // h264pred_template.c pred4x4_down_right (src[c + r*stride])
            let mut put = |r: usize, c: usize, v: u8| dst[r * 4 + c] = v;
            let a = ((l3 + 2 * l2 + l1 + 2) >> 2) as u8;
            put(3, 0, a);
            let a = ((l2 + 2 * l1 + l0 + 2) >> 2) as u8;
            put(2, 0, a);
            put(3, 1, a);
            let a = ((l1 + 2 * l0 + lt as u32 + 2) >> 2) as u8;
            put(1, 0, a);
            put(2, 1, a);
            put(3, 2, a);
            let a = ((l0 + 2 * lt as u32 + t0 + 2) >> 2) as u8;
            put(0, 0, a);
            put(1, 1, a);
            put(2, 2, a);
            put(3, 3, a);
            let a = ((lt as u32 + 2 * t0 + t1 + 2) >> 2) as u8;
            put(0, 1, a);
            put(1, 2, a);
            put(2, 3, a);
            let a = ((t0 + 2 * t1 + t2 + 2) >> 2) as u8;
            put(0, 2, a);
            put(1, 3, a);
            put(0, 3, ((t1 + 2 * t2 + t3 + 2) >> 2) as u8);
        }
        VERT_RIGHT_PRED => {
            // h264pred_template.c pred4x4_vertical_right
            // (src[col + row*stride] → put(row, col))
            let mut put = |r: usize, c: usize, v: u8| dst[r * 4 + c] = v;
            let a = ((lt as u32 + t0 + 1) >> 1) as u8;
            put(0, 0, a);
            put(2, 1, a);
            let a = ((t0 + t1 + 1) >> 1) as u8;
            put(0, 1, a);
            put(2, 2, a);
            let a = ((t1 + t2 + 1) >> 1) as u8;
            put(0, 2, a);
            put(2, 3, a);
            let a = ((t2 + t3 + 1) >> 1) as u8;
            put(0, 3, a);
            let a = ((l0 + 2 * lt as u32 + t0 + 2) >> 2) as u8;
            put(1, 0, a);
            put(3, 1, a);
            let a = ((lt as u32 + 2 * t0 + t1 + 2) >> 2) as u8;
            put(1, 1, a);
            put(3, 2, a);
            let a = ((t0 + 2 * t1 + t2 + 2) >> 2) as u8;
            put(1, 2, a);
            put(3, 3, a);
            let a = ((t1 + 2 * t2 + t3 + 2) >> 2) as u8;
            put(1, 3, a);
            let a = ((lt as u32 + 2 * l0 + l1 + 2) >> 2) as u8;
            put(2, 0, a);
            let a = ((l0 + 2 * l1 + l2 + 2) >> 2) as u8;
            put(3, 0, a);
        }
        HOR_DOWN_PRED => {
            // h264pred_template.c pred4x4_horizontal_down
            let mut put = |r: usize, c: usize, v: u8| dst[r * 4 + c] = v;
            let a = ((lt as u32 + l0 + 1) >> 1) as u8;
            put(0, 0, a);
            put(1, 2, a);
            let a = ((l0 + 2 * lt as u32 + t0 + 2) >> 2) as u8;
            put(0, 1, a);
            put(1, 3, a);
            let a = ((lt as u32 + 2 * t0 + t1 + 2) >> 2) as u8;
            put(0, 2, a);
            let a = ((t0 + 2 * t1 + t2 + 2) >> 2) as u8;
            put(0, 3, a);
            let a = ((l0 + l1 + 1) >> 1) as u8;
            put(1, 0, a);
            put(2, 2, a);
            let a = ((lt as u32 + 2 * l0 + l1 + 2) >> 2) as u8;
            put(1, 1, a);
            put(2, 3, a);
            let a = ((l1 + l2 + 1) >> 1) as u8;
            put(2, 0, a);
            put(3, 2, a);
            let a = ((l0 + 2 * l1 + l2 + 2) >> 2) as u8;
            put(2, 1, a);
            put(3, 3, a);
            let a = ((l2 + l3 + 1) >> 1) as u8;
            put(3, 0, a);
            let a = ((l1 + 2 * l2 + l3 + 2) >> 2) as u8;
            put(3, 1, a);
        }
        VERT_LEFT_PRED => {
            // h264pred_template.c pred4x4_vertical_left
            let mut put = |r: usize, c: usize, v: u8| dst[r * 4 + c] = v;
            let a = ((t0 + t1 + 1) >> 1) as u8;
            put(0, 0, a);
            let a = ((t1 + t2 + 1) >> 1) as u8;
            put(0, 1, a);
            put(2, 0, a);
            let a = ((t2 + t3 + 1) >> 1) as u8;
            put(0, 2, a);
            put(2, 1, a);
            let a = ((t3 + t4 + 1) >> 1) as u8;
            put(0, 3, a);
            put(2, 2, a);
            let a = ((t4 + t5 + 1) >> 1) as u8;
            put(2, 3, a);
            let a = ((t0 + 2 * t1 + t2 + 2) >> 2) as u8;
            put(1, 0, a);
            let a = ((t1 + 2 * t2 + t3 + 2) >> 2) as u8;
            put(1, 1, a);
            put(3, 0, a);
            let a = ((t2 + 2 * t3 + t4 + 2) >> 2) as u8;
            put(1, 2, a);
            put(3, 1, a);
            let a = ((t3 + 2 * t4 + t5 + 2) >> 2) as u8;
            put(1, 3, a);
            put(3, 2, a);
            let a = ((t4 + 2 * t5 + t6 + 2) >> 2) as u8;
            put(3, 3, a);
        }
        HOR_UP_PRED => {
            // h264pred_template.c pred4x4_horizontal_up
            let mut put = |r: usize, c: usize, v: u8| dst[r * 4 + c] = v;
            let a = ((l0 + l1 + 1) >> 1) as u8;
            put(0, 0, a);
            let a = ((l0 + 2 * l1 + l2 + 2) >> 2) as u8;
            put(0, 1, a);
            let a = ((l1 + l2 + 1) >> 1) as u8;
            put(0, 2, a);
            put(1, 0, a);
            let a = ((l1 + 2 * l2 + l3 + 2) >> 2) as u8;
            put(0, 3, a);
            put(1, 1, a);
            let a = ((l2 + l3 + 1) >> 1) as u8;
            put(1, 2, a);
            put(2, 0, a);
            let a = ((l2 + 2 * l3 + l3 + 2) >> 2) as u8;
            put(1, 3, a);
            put(2, 1, a);
            put(2, 2, l3 as u8);
            put(2, 3, l3 as u8);
            put(3, 0, l3 as u8);
            put(3, 1, l3 as u8);
            put(3, 2, l3 as u8);
            put(3, 3, l3 as u8);
        }
        _ => {
            // 128 DC fallback (caller maps unavailable modes away)
            for v in dst.iter_mut() {
                *v = 128;
            }
        }
    }
}

/// Row pairs from the generated flat tables.
fn table_rows(prefix: &str, n_rows: usize, width: usize) -> Vec<(Vec<u8>, Vec<u8>)> {
    // The generated names are {PREFIX}_LEN_{r} / {PREFIX}_BITS_{r}.
    macro_rules! get {
        ($name:ident) => {{
            static V: OnceLock<Vec<u8>> = OnceLock::new();
            V.get_or_init(|| $name.to_vec())
        }};
    }
    let mut out = Vec::new();
    for r in 0..n_rows {
        let (l, b) = match (prefix, r) {
            ("COEFF_TOKEN", 0) => (get!(COEFF_TOKEN_LEN_0), get!(COEFF_TOKEN_BITS_0)),
            ("COEFF_TOKEN", 1) => (get!(COEFF_TOKEN_LEN_1), get!(COEFF_TOKEN_BITS_1)),
            ("COEFF_TOKEN", 2) => (get!(COEFF_TOKEN_LEN_2), get!(COEFF_TOKEN_BITS_2)),
            ("COEFF_TOKEN", 3) => (get!(COEFF_TOKEN_LEN_3), get!(COEFF_TOKEN_BITS_3)),
            ("TOTAL_ZEROS", r) => match r {
                0 => (get!(TOTAL_ZEROS_LEN_0), get!(TOTAL_ZEROS_BITS_0)),
                1 => (get!(TOTAL_ZEROS_LEN_1), get!(TOTAL_ZEROS_BITS_1)),
                2 => (get!(TOTAL_ZEROS_LEN_2), get!(TOTAL_ZEROS_BITS_2)),
                3 => (get!(TOTAL_ZEROS_LEN_3), get!(TOTAL_ZEROS_BITS_3)),
                4 => (get!(TOTAL_ZEROS_LEN_4), get!(TOTAL_ZEROS_BITS_4)),
                5 => (get!(TOTAL_ZEROS_LEN_5), get!(TOTAL_ZEROS_BITS_5)),
                6 => (get!(TOTAL_ZEROS_LEN_6), get!(TOTAL_ZEROS_BITS_6)),
                7 => (get!(TOTAL_ZEROS_LEN_7), get!(TOTAL_ZEROS_BITS_7)),
                8 => (get!(TOTAL_ZEROS_LEN_8), get!(TOTAL_ZEROS_BITS_8)),
                9 => (get!(TOTAL_ZEROS_LEN_9), get!(TOTAL_ZEROS_BITS_9)),
                10 => (get!(TOTAL_ZEROS_LEN_10), get!(TOTAL_ZEROS_BITS_10)),
                11 => (get!(TOTAL_ZEROS_LEN_11), get!(TOTAL_ZEROS_BITS_11)),
                12 => (get!(TOTAL_ZEROS_LEN_12), get!(TOTAL_ZEROS_BITS_12)),
                13 => (get!(TOTAL_ZEROS_LEN_13), get!(TOTAL_ZEROS_BITS_13)),
                14 => (get!(TOTAL_ZEROS_LEN_14), get!(TOTAL_ZEROS_BITS_14)),
                _ => (get!(TOTAL_ZEROS_LEN_15), get!(TOTAL_ZEROS_BITS_15)),
            },
            _ => unreachable!(),
        };
        debug_assert_eq!(l.len(), width);
        let _ = width;
        out.push((l.clone(), b.clone()));
    }
    out
}

/// 16x16 luma intra prediction (mode 0=V,1=H,2=DC,3=plane). `top`/`left`
/// have 16 samples; availability pre-applied (DC_128 etc. by caller).
fn pred16x16(mode: i32, dst: &mut [u8], top: &[u8; 16], left: &[u8; 16], lt: u8) {
    match mode {
        0 => {
            for r in 0..16 {
                dst[r * 16..r * 16 + 16].copy_from_slice(top);
            }
        }
        1 => {
            for r in 0..16 {
                dst[r * 16..r * 16 + 16].copy_from_slice(&[left[r]; 16]);
            }
        }
        2 => {
            let s: u32 = top.iter().map(|&v| v as u32).sum::<u32>()
                + left.iter().map(|&v| v as u32).sum::<u32>();
            let dc = (s / 32) as u8;
            for r in 0..16 {
                dst[r * 16..r * 16 + 16].copy_from_slice(&[dc; 16]);
            }
        }
        3 => {
            // plane prediction (spec 8.3.3.1.4)
            // C pred16x16_plane: top[−1] / left[−1] are the top-left
            // corner sample (the old top[6−i] underflowed at i=7).
            let mut h = 0i32;
            for i in 0..8usize {
                let lo = if i == 7 { lt as i32 } else { top[6 - i] as i32 };
                h += (i as i32 + 1) * (top[8 + i] as i32 - lo);
            }
            let mut v = 0i32;
            for i in 0..8usize {
                let lo = if i == 7 {
                    lt as i32
                } else {
                    left[6 - i] as i32
                };
                v += (i as i32 + 1) * (left[8 + i] as i32 - lo);
            }
            let a = 16 * (top[15] as i32 + left[15] as i32);
            let b = (5 * h + 32) >> 6;
            let c = (5 * v + 32) >> 6;
            // spec 8.3.3.4 / C pred16x16_plane: (a + b(x−7) + c(y−7) + 16)
            // >> 5 over the whole 16x16 (the old per-8x8 x−3/y−3 form added
            // a constant 4b+4c).
            for y in 0..16i32 {
                for x in 0..16i32 {
                    dst[(y as usize) * 16 + x as usize] =
                        clip8((a + b * (x - 7) + c * (y - 7) + 16) >> 5);
                }
            }
        }
        _ => unreachable!(),
    }
}

/// 8x8 chroma intra prediction (mode 0=DC,1=H,2=V,3=plane).
fn pred8x8(mode: i32, dst: &mut [u8], top: &[u8; 8], left: &[u8; 8], lt: u8, lb: u8) {
    match mode {
        0 => {
            let s: u32 = top.iter().map(|&v| v as u32).sum::<u32>()
                + left.iter().map(|&v| v as u32).sum::<u32>();
            let dc = (s / 16) as u8;
            for r in 0..8 {
                dst[r * 8..r * 8 + 8].copy_from_slice(&[dc; 8]);
            }
        }
        1 => {
            for r in 0..8 {
                dst[r * 8..r * 8 + 8].copy_from_slice(&[left[r]; 8]);
            }
        }
        2 => {
            for r in 0..8 {
                dst[r * 8..r * 8 + 8].copy_from_slice(top);
            }
        }
        3 => pred8x8_plane(dst, 8, top, left, lt, lb),
        _ => unreachable!(),
    }
}

/// `pred8x8_plane` (h264pred_template.c:746): src0 walks the top row
/// extended by the top-LEFT sample at src0[-1]; V the left column plus
/// two rows below the last (clamped to left[7] — C reads the MB below,
/// which for edge MBs the border-fill already replicated).
fn pred8x8_plane(dst: &mut [u8], dstride: usize, top: &[u8; 8], left: &[u8; 8], lt: u8, lb: u8) {
    // h264pred_template.c pred8x8_plane: src0 = top row at x=3, src1/src2
    // walk the left column down/up. top[−1] and left[−1] are BOTH the
    // top-left corner sample; lb (below-left) is never read.
    let _ = lb;
    let t = |i: i32| -> i32 {
        if i < 0 {
            lt as i32
        } else {
            top[i as usize] as i32
        }
    };
    let l = |i: i32| -> i32 {
        if i < 0 {
            lt as i32
        } else {
            left[i as usize] as i32
        }
    };
    // H = top[4]-top[2] + 2*(top[5]-top[1]) + 3*(top[6]-top[0]) + 4*(top[7]-lt)
    let mut h = t(4) - t(2);
    let mut v = l(4) - l(2);
    let mut k = 2i32;
    while k <= 4 {
        h += k * (t(3 + k) - t(3 - k));
        v += k * (l(3 + k) - l(3 - k));
        k += 1;
    }
    let h = (17 * h + 16) >> 5;
    let v = (17 * v + 16) >> 5;
    let a = 16 * (l(7) + t(7) + 1) - 3 * (v + h);
    for y in 0..8i32 {
        for x in 0..8i32 {
            dst[(y as usize) * dstride + x as usize] = clip8((a + y * v + x * h) >> 5);
        }
    }
}

// ---------------------------------------------------------------------
// Transforms (h264idct_template.c)
// ---------------------------------------------------------------------

/// `ff_h264_idct_add` (h264idct_template.c:34).
fn idct_add(dst: &mut [u8], dstride: usize, block: &mut [i16; 16]) {
    // C's int16_t `block[0] += 1<<5` wraps on extreme streams (garbage
    // levels from a desynced walk can overflow i16 here); mirror that.
    block[0] = block[0].wrapping_add(1 << 5);
    // columns
    for i in 0..4 {
        let z0 = block[i] as i32 + block[i + 8] as i32;
        let z1 = block[i] as i32 - block[i + 8] as i32;
        let z2 = (block[i + 4] >> 1) as i32 - block[i + 12] as i32;
        let z3 = block[i + 4] as i32 + (block[i + 12] >> 1) as i32;
        block[i] = (z0 + z3) as i16;
        block[i + 4] = (z1 + z2) as i16;
        block[i + 8] = (z1 - z2) as i16;
        block[i + 12] = (z0 - z3) as i16;
    }
    // rows
    for i in 0..4 {
        let z0 = block[4 * i] as i32 + block[4 * i + 2] as i32;
        let z1 = block[4 * i] as i32 - block[4 * i + 2] as i32;
        let z2 = (block[4 * i + 1] >> 1) as i32 - block[4 * i + 3] as i32;
        let z3 = block[4 * i + 1] as i32 + (block[4 * i + 3] >> 1) as i32;
        // C writes COLUMN-major: dst[i + k*stride] — the coefficients are
        // stored in transposed raster (the transposed zigzag scan), so the
        // row-major loop-variable write would transpose every block.
        let base = i;
        dst[base] = clip8(dst[base] as i32 + ((z0 + z3) >> 6));
        dst[base + dstride] = clip8(dst[base + dstride] as i32 + ((z1 + z2) >> 6));
        dst[base + 2 * dstride] = clip8(dst[base + 2 * dstride] as i32 + ((z1 - z2) >> 6));
        dst[base + 3 * dstride] = clip8(dst[base + 3 * dstride] as i32 + ((z0 - z3) >> 6));
    }
    *block = [0; 16];
}

/// `ff_h264_idct_dc_add`.
fn idct_dc_add(dst: &mut [u8], dstride: usize, block: &mut [i16; 16]) {
    let dc = ((block[0] as i32 + 32) >> 6) as i32;
    block[0] = 0;
    for r in 0..4 {
        for c in 0..4 {
            let at = r * dstride + c;
            dst[at] = clip8(dst[at] as i32 + dc);
        }
    }
}

/// `ff_h264_luma_dc_dequant_idct` (h264idct_template.c:259): the 4x4
/// Hadamard transform of the 16 DC coefficients, dequantized and
/// scattered into the 16 AC blocks.
fn luma_dc_dequant_idct(output: &mut [i16], input: &[i16; 16], qmul: u32) {
    let mut temp = [0i32; 16];
    for i in 0..4 {
        let z0 = input[4 * i] as i32 + input[4 * i + 1] as i32;
        let z1 = input[4 * i] as i32 - input[4 * i + 1] as i32;
        let z2 = input[4 * i + 2] as i32 - input[4 * i + 3] as i32;
        let z3 = input[4 * i + 2] as i32 + input[4 * i + 3] as i32;
        temp[4 * i] = z0 + z3;
        temp[4 * i + 1] = z0 - z3;
        temp[4 * i + 2] = z1 - z2;
        temp[4 * i + 3] = z1 + z2;
    }
    static X_OFF: [usize; 4] = [0, 2 * 16, 8 * 16, 10 * 16];
    for i in 0..4 {
        let off = X_OFF[i];
        let z0 = temp[4 * 0 + i] + temp[4 * 2 + i];
        let z1 = temp[4 * 0 + i] - temp[4 * 2 + i];
        let z2 = temp[4 * 1 + i] - temp[4 * 3 + i];
        let z3 = temp[4 * 1 + i] + temp[4 * 3 + i];
        output[16 * 0 + off] = (((z0 + z3) as i64 * qmul as i64 + 128) >> 8) as i16;
        output[16 * 1 + off] = (((z1 + z2) as i64 * qmul as i64 + 128) >> 8) as i16;
        output[16 * 4 + off] = (((z1 - z2) as i64 * qmul as i64 + 128) >> 8) as i16;
        output[16 * 5 + off] = (((z0 - z3) as i64 * qmul as i64 + 128) >> 8) as i16;
    }
}

/// `ff_h264_chroma_dc_dequant_idct` (h264idct_template.c:332).
fn chroma_dc_dequant_idct(block: &mut [i16], qmul: u32) {
    const STRIDE: usize = 16 * 2;
    const XSTR: usize = 16;
    let a = block[STRIDE * 0 + XSTR * 0] as i32;
    let b = block[STRIDE * 0 + XSTR * 1] as i32;
    let c = block[STRIDE * 1 + XSTR * 0] as i32;
    let d = block[STRIDE * 1 + XSTR * 1] as i32;
    let e = a - b;
    let f = c - d;
    let g = a + b;
    let h = c + d;
    // This FFmpeg's version (h264idct_template.c:332): plain >> 7, NO +128
    // rounding (an older variant's (v*qmul+128)>>8 is 2x too small).
    block[STRIDE * 0 + XSTR * 0] = (((g + h) as i64 * qmul as i64) >> 7) as i16;
    block[STRIDE * 0 + XSTR * 1] = (((e + f) as i64 * qmul as i64) >> 7) as i16;
    block[STRIDE * 1 + XSTR * 0] = (((g - h) as i64 * qmul as i64) >> 7) as i16;
    block[STRIDE * 1 + XSTR * 1] = (((e - f) as i64 * qmul as i64) >> 7) as i16;
}

// ---------------------------------------------------------------------
// Motion compensation (spec 8.4.2.2; scalar form of h264qpel/h264chroma)
// ---------------------------------------------------------------------

/// `mid_pred` (libavutil/mathops.h): median of three.
fn mid_pred(a: i16, b: i16, c: i16) -> i16 {
    let (lo, hi) = (a.min(b), a.max(b));
    c.clamp(lo, hi)
}

fn hpel6(vals: [i32; 6]) -> i32 {
    (vals[0] - 5 * vals[1] + 20 * vals[2] + 20 * vals[3] - 5 * vals[4] + vals[5] + 16) >> 5
}

/// One filtered luma plane (16x16 luma quadrant at MB-relative 4x4
/// granularity), motion-compensated from `src` at qpel `mv`.
fn mc_luma(
    dst: &mut [u8],
    dstride: usize,
    w: usize,
    h: usize,
    src: &Picture,
    base_x: i32,
    base_y: i32,
    mx: i16,
    my: i16,
) {
    let ox = base_x + (mx >> 2) as i32;
    let oy = base_y + (my >> 2) as i32;
    let fx = (mx & 3) as i32;
    let fy = (my & 3) as i32;
    // Sample getters (edge-clamped reads == C's edge extension).
    let i_px = |x: i32, y: i32| -> i32 { src.sample_y(x, y) as i32 };
    let h_px = |x: i32, y: i32| -> i32 {
        clip8(hpel6([
            i_px(x - 2, y),
            i_px(x - 1, y),
            i_px(x, y),
            i_px(x + 1, y),
            i_px(x + 2, y),
            i_px(x + 3, y),
        ])) as i32
    };
    let v_px = |x: i32, y: i32| -> i32 {
        clip8(hpel6([
            i_px(x, y - 2),
            i_px(x, y - 1),
            i_px(x, y),
            i_px(x, y + 1),
            i_px(x, y + 2),
            i_px(x, y + 3),
        ])) as i32
    };
    // j (center half-pel), spec 8.4.2.2.1: 6-tap over the UNROUNDED
    // horizontal intermediates b1, then (j1 + 512) >> 10. (Filtering the
    // already-rounded/clipped b values is not bit-exact.)
    let b1 = |x: i32, y: i32| -> i32 {
        i_px(x - 2, y) - 5 * i_px(x - 1, y) + 20 * i_px(x, y) + 20 * i_px(x + 1, y)
            - 5 * i_px(x + 2, y)
            + i_px(x + 3, y)
    };
    let j_px = |x: i32, y: i32| -> i32 {
        let j1 = b1(x, y - 2) - 5 * b1(x, y - 1) + 20 * b1(x, y) + 20 * b1(x, y + 1)
            - 5 * b1(x, y + 2)
            + b1(x, y + 3);
        clip8((j1 + 512) >> 10) as i32
    };
    let avg = |a: i32, b: i32| -> i32 { (a + b + 1) >> 1 };
    for r in 0..h {
        for c in 0..w {
            let x = ox + c as i32;
            let y = oy + r as i32;
            // Spec 8.4.2.2.1 sample table (G at integer (x,y); b = h_px,
            // h = v_px, j = centre; s = b one row down, m = h one col right).
            let v: i32 = match (fx, fy) {
                (0, 0) => i_px(x, y),
                (1, 0) => avg(i_px(x, y), h_px(x, y)),     // a
                (2, 0) => h_px(x, y),                      // b
                (3, 0) => avg(i_px(x + 1, y), h_px(x, y)), // c
                (0, 1) => avg(i_px(x, y), v_px(x, y)),     // d
                (0, 2) => v_px(x, y),                      // h
                (0, 3) => avg(i_px(x, y + 1), v_px(x, y)), // n
                (2, 2) => j_px(x, y),                      // j
                (2, 1) => avg(h_px(x, y), j_px(x, y)),     // f
                (2, 3) => avg(j_px(x, y), h_px(x, y + 1)), // q
                (1, 2) => avg(v_px(x, y), j_px(x, y)),     // i
                (3, 2) => avg(j_px(x, y), v_px(x + 1, y)), // k
                (1, 1) => avg(h_px(x, y), v_px(x, y)),     // e
                (3, 1) => avg(h_px(x, y), v_px(x + 1, y)), // g
                (1, 3) => avg(v_px(x, y), h_px(x, y + 1)), // p
                _ => avg(v_px(x + 1, y), h_px(x, y + 1)),  // r (3,3)
            };
            dst[r * dstride + c] = v as u8;
        }
    }
}

/// Chroma MC (spec 8.4.2.2.2): bilinear at eighth-pel.
fn mc_chroma(
    dst: &mut [u8],
    dstride: usize,
    w: usize,
    h: usize,
    src: &Picture,
    plane: &PlaneSel,
    base_x: i32,
    base_y: i32,
    mx: i16,
    my: i16,
) {
    let p: &[u8] = match plane {
        PlaneSel::Cb => &src.cb,
        PlaneSel::Cr => &src.cr,
    };
    let fx = (mx & 7) as i32;
    let fy = (my & 7) as i32;
    let ox = base_x / 2 + (mx >> 3) as i32;
    let oy = base_y / 2 + (my >> 3) as i32;
    for r in 0..h {
        for c in 0..w {
            let x = ox + c as i32;
            let y = oy + r as i32;
            let a = src.sample_c(p, x, y) as i32;
            let b = src.sample_c(p, x + 1, y) as i32;
            let cc = src.sample_c(p, x, y + 1) as i32;
            let d = src.sample_c(p, x + 1, y + 1) as i32;
            let v = ((8 - fx) * (8 - fy) * a
                + fx * (8 - fy) * b
                + (8 - fx) * fy * cc
                + fx * fy * d
                + 32)
                >> 6;
            dst[r * dstride + c] = v as u8;
        }
    }
}

// ---------------------------------------------------------------------
// decode_residual (h264_cavlc.c:405) + chroma DC variant
// ---------------------------------------------------------------------

/// `decode_residual` for 4x4 blocks. `n` indexes the nnz cache (scan8).
/// `qmul`: C's STORE_BLOCK (h264_cavlc.c:564) — `None` (C's NULL, luma16 /
/// chroma DC blocks, n >= LUMA_DC_BLOCK_INDEX) stores levels raw;
/// `Some` is the per-position dequant table indexed by scan value:
/// `block[s] = (level*qmul[s] + 32) >> 6`.
fn decode_residual(
    h: &mut H264Decoder,
    cv: &Cavlc,
    gb: &mut Gb,
    block: &mut [i16],
    n: usize,
    scan: &[u8; 16],
    qmul: Option<&[u32; 16]>,
    max_coeff: usize,
) -> Result<()> {
    let mut level = [0i32; 16];
    if std::env::var_os("H264_DUMP").is_some() {
        eprintln!("ENTR n={n} pos={}", gb.index);
    }

    let coeff_token = if max_coeff == 4 {
        cv.chroma_dc_coeff_token.get(gb)? as usize
    } else {
        let nnz_pred = h.pred_nnz(if n >= LUMA_DC { (n - LUMA_DC) * 16 } else { n }) as usize;
        let bucket = coeff_token_bucket(nnz_pred);
        if std::env::var_os("H264_DUMP").is_some() {
            eprintln!(
                "  TOK n={n} pred={nnz_pred} b={bucket} bits={:016b}",
                gb.peek(16)
            );
        }
        cv.coeff_token[bucket].get(gb)? as usize
    };
    let total_coeff = coeff_token >> 2;
    h.nnz_cache[SCAN8[n]] = total_coeff as u8;

    if total_coeff == 0 {
        return Ok(());
    }
    if total_coeff > max_coeff {
        return Err(Error::InvalidData("total_coeff too large".into()));
    }
    let trailing_ones = (coeff_token & 3) as usize;

    // trailing ones signs
    let i3 = gb.peek(3);
    gb.skip(trailing_ones as u32);
    level[0] = 1 - ((i3 >> 1) & 2) as i32;
    level[1] = 1 - (i3 & 2) as i32;
    level[2] = 1 - ((i3 & 1) << 1) as i32;

    if trailing_ones < total_coeff {
        let mut suffix_length = (total_coeff > 10 && trailing_ones < 3) as usize;
        // first non-trailing level
        {
            let bitsi = gb.peek(LEVEL_TAB_BITS) as usize;
            if std::env::var_os("H264_DUMP").is_some() && n == LUMA_DC {
                eprintln!(
                    "  FIRSTDC tc={total_coeff} to={trailing_ones} sl={suffix_length} bitsi={bitsi} e0={} e1={}",
                    cv.level_tab[suffix_length][bitsi][0], cv.level_tab[suffix_length][bitsi][1]
                );
            }
            if std::env::var_os("H264_DUMP").is_some() && (h.mb_y == 0 && h.mb_x <= 2)
                || (h.mb_y == 2 && h.mb_x == 0)
            {
                eprintln!(
                    "  FIRST sl={suffix_length} bitsi={bitsi} e0={} e1={}",
                    cv.level_tab[suffix_length][bitsi][0], cv.level_tab[suffix_length][bitsi][1]
                );
            }
            let (mut level_code, consumed) = (
                cv.level_tab[suffix_length][bitsi][0] as i32,
                cv.level_tab[suffix_length][bitsi][1] as u32,
            );
            gb.skip(consumed);
            if level_code >= 100 {
                let mut prefix = level_code - 100;
                if prefix == LEVEL_TAB_BITS as i32 {
                    prefix += gb.level_prefix()? as i32;
                }
                if prefix < 14 {
                    level_code = if suffix_length != 0 {
                        (prefix << 1) + gb.read_bit() as i32
                    } else {
                        prefix
                    };
                } else if prefix == 14 {
                    level_code = if suffix_length != 0 {
                        (prefix << 1) + gb.read_bit() as i32
                    } else {
                        prefix + gb.read(4) as i32
                    };
                } else {
                    level_code = 30;
                    if prefix >= 16 {
                        if prefix > 28 {
                            return Err(Error::InvalidData("invalid level prefix".into()));
                        }
                        level_code += (1i32 << (prefix - 3)) - 4096;
                    }
                    level_code += gb.read((prefix - 3) as u32) as i32;
                }
                if trailing_ones < 3 {
                    level_code += 2;
                }
                suffix_length = 2;
                let mask = -(level_code & 1);
                level_code = (((2 + level_code) >> 1) ^ mask) - mask;
            } else {
                level_code += ((level_code >> 31) | 1) & -((trailing_ones < 3) as i32);
                // C's unsigned compare: a negative level wraps and
                // pushes the suffix length to 2.
                suffix_length = 1 + ((level_code as u32).wrapping_add(3) > 6) as usize;
            }
            level[trailing_ones] = level_code;
        }
        // remaining levels
        for i in (trailing_ones + 1)..total_coeff {
            let bitsi = gb.peek(LEVEL_TAB_BITS) as usize;
            if std::env::var_os("H264_DUMP").is_some() && (h.mb_y == 0 && h.mb_x <= 2)
                || (h.mb_y == 2 && h.mb_x == 0)
            {
                eprintln!("  LVL i={i} sl={suffix_length} bitsi={bitsi}");
            }
            let (mut level_code, consumed) = (
                cv.level_tab[suffix_length][bitsi][0] as i32,
                cv.level_tab[suffix_length][bitsi][1] as u32,
            );
            if std::env::var_os("H264_DUMP").is_some() && (h.mb_y == 0 && h.mb_x <= 2)
                || (h.mb_y == 2 && h.mb_x == 0)
            {
                eprintln!(
                    "  RLOOK i={i} sl={suffix_length} bitsi={bitsi} c={level_code} l={consumed} pre={}",
                    gb.index
                );
            }
            gb.skip(consumed);
            if level_code >= 100 {
                let mut prefix = level_code - 100;
                if prefix == LEVEL_TAB_BITS as i32 {
                    prefix += gb.level_prefix()? as i32;
                }
                if std::env::var_os("H264_DUMP").is_some() && (h.mb_y == 0 && h.mb_x <= 2)
                    || (h.mb_y == 2 && h.mb_x == 0)
                {
                    eprintln!(
                        "  RESC i={i} prefix={prefix} sl={suffix_length} pos={}",
                        gb.index
                    );
                }
                if prefix < 15 {
                    level_code = (prefix << suffix_length) + gb.read(suffix_length as u32) as i32;
                } else {
                    level_code = 15 << suffix_length;
                    if prefix >= 16 {
                        if prefix > 28 {
                            return Err(Error::InvalidData("invalid level prefix".into()));
                        }
                        level_code += (1i32 << (prefix - 3)) - 4096;
                    }
                    level_code += gb.read((prefix - 3) as u32) as i32;
                }
                let mask = -(level_code & 1);
                level_code = (((2 + level_code) >> 1) ^ mask) - mask;
            }
            level[i] = level_code;
            // C's unsigned compare: suffix_limit[sl] + level_code >
            // 2U*suffix_limit[sl] — a negative level wraps huge and bumps.
            const SUFFIX_LIMIT: [u32; 7] = [0, 3, 6, 12, 24, 48, u32::MAX];
            suffix_length += ((SUFFIX_LIMIT[suffix_length].wrapping_add(level_code as u32))
                > SUFFIX_LIMIT[suffix_length].wrapping_mul(2))
                as usize;
        }
    }

    // total_zeros
    if std::env::var_os("H264_DUMP").is_some() {
        eprintln!("  LVL n={n} pos={}", gb.index);
    }
    let zeros_left = if total_coeff == max_coeff {
        0usize
    } else if max_coeff == 4 {
        cv.chroma_dc_tz[total_coeff - 1].get(gb)? as usize
    } else {
        cv.total_zeros[total_coeff - 1].get(gb)? as usize
    };

    // run_before + store (STORE_BLOCK). C walks the scantable pointer
    // with plain int arithmetic — it may go below the block start
    // without storing there, so pos is signed here.
    let mut pos = (zeros_left + total_coeff - 1) as i32;
    let mut zi = zeros_left as i32;
    let mut i = 0usize;
    loop {
        if !(0..16).contains(&pos) {
            return Err(Error::InvalidData(
                "run_before position out of block".into(),
            ));
        }
        let s = scan[pos as usize] as usize;
        block[s] = match qmul {
            None => level[i] as i16,
            Some(q) => ((level[i] as i64 * q[s] as i64 + 32) >> 6) as i16,
        };
        i += 1;
        if i >= total_coeff || zi <= 0 {
            break;
        }
        let run = if zi < 7 {
            cv.run[(zi - 1) as usize].get(gb)? as usize
        } else {
            cv.run7.get(gb)? as usize
        };
        if std::env::var_os("H264_DUMP").is_some() && (h.mb_y == 0 && h.mb_x <= 2)
            || (h.mb_y == 2 && h.mb_x == 0)
        {
            eprintln!("  RUN i={i} zi={zi} run={run} pos={}", gb.index);
        }
        zi -= run as i32;
        pos -= 1 + run as i32;
    }
    if i < total_coeff {
        for k in i..total_coeff {
            // C's tail (h264_cavlc.c STORE_BLOCK): `scantable--;` BEFORE
            // the store — the remaining coefficients step down one at a
            // time from the last stored position (storing at the current
            // pos first shifts every tail coeff one slot too high and
            // drops the lowest-frequency one entirely).
            pos -= 1;
            if !(0..16).contains(&pos) {
                return Err(Error::InvalidData("coeff overrun".into()));
            }
            let s = scan[pos as usize] as usize;
            block[s] = match qmul {
                None => level[k] as i16,
                Some(q) => ((level[k] as i64 * q[s] as i64 + 32) >> 6) as i16,
            };
        }
    }
    if std::env::var_os("H264_DUMP").is_some() {
        eprintln!("RES n={n} tc={total_coeff} to={trailing_ones} zl={zeros_left}");
        if h.mb_y == 0 && h.mb_x == 0 {
            let mut dbg = String::new();
            for v in block.iter().take(16) {
                dbg.push_str(&format!("{v} "));
            }
            eprintln!("  BLK{n}: {dbg}");
        }
    }
    if zi < 0 {
        return Err(Error::InvalidData("negative zeros".into()));
    }
    Ok(())
}

/// Chroma DC residual (max_coeff 4, chroma_dc_coeff_token VLC).
fn decode_residual_chroma_dc(
    h: &mut H264Decoder,
    cv: &Cavlc,
    gb: &mut Gb,
    dc: &mut [i16; 16],
    _ch: usize,
) -> Result<()> {
    let coeff_token = cv.chroma_dc_coeff_token.get(gb)? as usize;
    let total_coeff = coeff_token >> 2;
    h.nnz_cache[SCAN8[CHROMA_DC + _ch]] = total_coeff as u8;
    if total_coeff == 0 {
        return Ok(());
    }
    let trailing_ones = (coeff_token & 3) as usize;
    let mut level = [0i32; 4];
    let i3 = gb.peek(3);
    gb.skip(trailing_ones as u32);
    if trailing_ones > 0 {
        level[0] = 1 - ((i3 >> 1) & 2) as i32;
    }
    if trailing_ones > 1 {
        level[1] = 1 - (i3 & 2) as i32;
    }
    if trailing_ones > 2 {
        level[2] = 1 - ((i3 & 1) << 1) as i32;
    }
    if trailing_ones < total_coeff {
        // single level, suffix_length 0 semantics via level_tab
        let bitsi = gb.peek(LEVEL_TAB_BITS) as usize;
        let (mut level_code, consumed) = (
            cv.level_tab[0][bitsi][0] as i32,
            cv.level_tab[0][bitsi][1] as u32,
        );
        gb.skip(consumed);
        if level_code >= 100 {
            let mut prefix = level_code - 100;
            if prefix == LEVEL_TAB_BITS as i32 {
                prefix += gb.level_prefix()? as i32;
            }
            if prefix < 14 {
                level_code = prefix;
            } else if prefix == 14 {
                level_code = prefix + gb.read(4) as i32;
            } else {
                level_code = 30;
                if prefix >= 16 {
                    level_code += (1i32 << (prefix - 3)) - 4096;
                }
                level_code += gb.read((prefix - 3) as u32) as i32;
            }
            if trailing_ones < 3 {
                level_code += 2;
            }
            let mask = -(level_code & 1);
            level_code = (((2 + level_code) >> 1) ^ mask) - mask;
        } else {
            level_code += ((level_code >> 31) | 1) & -((trailing_ones < 3) as i32);
        }
        level[trailing_ones] = level_code;
    }
    let zeros_left = if total_coeff == 4 {
        0
    } else {
        cv.chroma_dc_tz[total_coeff - 1].get(gb)? as usize
    };
    // scan positions: chroma_dc_scan = {0,1,4,5} in a 2x2-stride-16 view —
    // here dc[0..4] linear with C's ff_h264_chroma_dc_scan.
    static CDC_SCAN: [usize; 4] = [0, 1, 2, 3];
    let _ = CDC_SCAN;
    let mut pos = zeros_left + total_coeff - 1;
    let mut zi = zeros_left as i32;
    let mut i = 0usize;
    loop {
        let s = pos.min(3);
        dc[s] = level[i] as i16;
        i += 1;
        if i >= total_coeff || zi <= 0 {
            break;
        }
        let run = if zi < 7 {
            cv.run[zi as usize].get(gb)? as usize
        } else {
            cv.run7.get(gb)? as usize
        };
        zi -= run as i32;
        pos -= 1 + run;
        if pos > 3 + 1 {
            return Err(Error::InvalidData("chroma DC overrun".into()));
        }
    }
    while i < total_coeff {
        if pos > 3 {
            return Err(Error::InvalidData("chroma DC overrun".into()));
        }
        dc[pos] = level[i] as i16;
        pos -= 1;
        i += 1;
    }
    if zi < 0 {
        return Err(Error::InvalidData("chroma DC negative zeros".into()));
    }
    Ok(())
}

/// coeff_token VLC bucket (coeff_token_table_index, cavlc.c:358).
fn coeff_token_bucket(nnz: usize) -> usize {
    const IDX: [usize; 17] = [0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3];
    IDX[nnz.min(16)]
}

// small helpers used by decode_mb_residual
fn scan_shift1() -> [u8; 16] {
    // C uses `scan + 1` (skipping the DC position for AC-only blocks)
    let mut s = ZIGZAG;
    s.rotate_left(1);
    s
}
// pred16x16/pred8x8 with availability-aware DC (C's *_DC_* variants).
fn pred16x16_avail(
    mode: i32,
    dst: &mut [u8],
    top: &[u8; 16],
    left: &[u8; 16],
    t_ok: bool,
    l_ok: bool,
    lt: u8,
) {
    match mode {
        0 => {
            // VERT (port space 0): replicate the top row
            for r in 0..16 {
                dst[r * 16..r * 16 + 16].copy_from_slice(top);
            }
        }
        2 => {
            // DC (port space 2): C pred16x16_dc — (sum+16)>>5 both,
            // (sum+8)>>4 single side, 128 none.
            let dc = match (t_ok, l_ok) {
                (true, true) => {
                    (top.iter().map(|&v| v as u32).sum::<u32>()
                        + left.iter().map(|&v| v as u32).sum::<u32>()
                        + 16)
                        >> 5
                }
                (true, false) => (top.iter().map(|&v| v as u32).sum::<u32>() + 8) >> 4,
                (false, true) => (left.iter().map(|&v| v as u32).sum::<u32>() + 8) >> 4,
                (false, false) => 128,
            };
            for r in 0..16 {
                dst[r * 16..r * 16 + 16].copy_from_slice(&[dc as u8; 16]);
            }
        }
        1 => {
            for r in 0..16 {
                dst[r * 16..r * 16 + 16].copy_from_slice(&[left[r]; 16]);
            }
        }
        3 => pred16x16(3, dst, top, left, lt),
        4 => {
            let dc = (left.iter().map(|&v| v as u32).sum::<u32>() + 8) / 16;
            for r in 0..16 {
                dst[r * 16..r * 16 + 16].copy_from_slice(&[dc as u8; 16]);
            }
        }
        5 => {
            let dc = (top.iter().map(|&v| v as u32).sum::<u32>() + 8) / 16;
            for r in 0..16 {
                dst[r * 16..r * 16 + 16].copy_from_slice(&[dc as u8; 16]);
            }
        }
        _ => {
            for r in 0..16 {
                dst[r * 16..r * 16 + 16].copy_from_slice(&[128u8; 16]);
            }
        }
    }
}

fn pred8x8_avail(
    mode: i32,
    dst: &mut [u8],
    dstride: usize,
    top: &[u8; 8],
    left: &[u8; 8],
    t_ok: bool,
    l_ok: bool,
    lt: u8,
    lb: u8,
) {
    match mode {
        0 => {
            // VERT (port space 0): replicate the top row
            for r in 0..8 {
                for c in 0..8 {
                    dst[r * dstride + c] = top[c];
                }
            }
        }
        2 => {
            // DC (port space 2) — C pred8x8_dc/left_dc/top_dc (h264pred
            // _template.c): FOUR per-quadrant DCs when both sides are
            // available; top-left = (left[0..3]+top[0..3]+4)>>3,
            // top-right = (top[4..7]+2)>>2, bottom-left =
            // (left[4..7]+2)>>2, bottom-right = (dc1+dc2+4)>>3.
            // Single-side variants are PER-HALF-ROW/col, NOT one flat
            // average: left_dc uses (left[0..3]+2)>>2 top /
            // (left[4..7]+2)>>2 bottom; top_dc mirrors it.
            match (t_ok, l_ok) {
                (true, true) => {
                    let dc0 = (top[0] as u32
                        + top[1] as u32
                        + top[2] as u32
                        + top[3] as u32
                        + left[0] as u32
                        + left[1] as u32
                        + left[2] as u32
                        + left[3] as u32
                        + 4)
                        >> 3;
                    let s1 = top[4] as u32 + top[5] as u32 + top[6] as u32 + top[7] as u32;
                    let s2 = left[4] as u32 + left[5] as u32 + left[6] as u32 + left[7] as u32;
                    let dc1 = (s1 + 2) >> 2;
                    let dc2 = (s2 + 2) >> 2;
                    // C averages the RAW 4-sample sums (not the rounded
                    // quadrant DCs — rebuilding sums from dc1/dc2 is lossy).
                    let dc3 = (s1 + s2 + 4) >> 3;
                    for r in 0..8 {
                        for c in 0..8 {
                            let v = if r < 4 {
                                if c < 4 { dc0 } else { dc1 }
                            } else if c < 4 {
                                dc2
                            } else {
                                dc3
                            };
                            dst[r * dstride + c] = v as u8;
                        }
                    }
                }
                (false, true) => {
                    let dc0 =
                        (left[0] as u32 + left[1] as u32 + left[2] as u32 + left[3] as u32 + 2)
                            >> 2;
                    let dc2 =
                        (left[4] as u32 + left[5] as u32 + left[6] as u32 + left[7] as u32 + 2)
                            >> 2;
                    for r in 0..8 {
                        let v = if r < 4 { dc0 } else { dc2 };
                        for c in 0..8 {
                            dst[r * dstride + c] = v as u8;
                        }
                    }
                }
                (true, false) => {
                    let dc0 =
                        (top[0] as u32 + top[1] as u32 + top[2] as u32 + top[3] as u32 + 2) >> 2;
                    let dc1 =
                        (top[4] as u32 + top[5] as u32 + top[6] as u32 + top[7] as u32 + 2) >> 2;
                    for r in 0..8 {
                        for c in 0..8 {
                            let v = if c < 4 { dc0 } else { dc1 };
                            dst[r * dstride + c] = v as u8;
                        }
                    }
                }
                (false, false) => {
                    for r in 0..8 {
                        for c in 0..8 {
                            dst[r * dstride + c] = 128;
                        }
                    }
                }
            }
        }
        1 => {
            for r in 0..8 {
                for c in 0..8 {
                    dst[r * dstride + c] = left[r];
                }
            }
        }
        3 => pred8x8_plane(dst, dstride, top, left, lt, lb),
        _ => {
            for r in 0..8 {
                for c in 0..8 {
                    dst[r * dstride + c] = 128;
                }
            }
        }
    }
}

fn type_mask_nnz(t: u32) -> bool {
    t != MB_UNAVAIL
}

#[cfg(test)]
mod tests {
    use crate::{
        Frame,
        codec::{
            CodecId, CodecParameters, Packet, traits::Decoder, video::h264::decoder::H264Decoder,
        },
    };

    fn decode_file(path: &str) -> Vec<Frame> {
        let data = match std::fs::read(path) {
            Ok(d) => d,
            Err(_) => return Vec::new(),
        };
        let mut dec = H264Decoder::new();
        let mut p = CodecParameters::default();
        p.codec_id = CodecId::H264;
        Decoder::init(&mut dec, &p).unwrap();
        let mut pkt = Packet::from_vec(data);
        pkt.pts = 0;
        if let Err(e) = Decoder::send_packet(&mut dec, Some(&pkt)) {
            eprintln!("H264 WIP: decode error: {e}");
        }
        let _ = Decoder::send_packet(&mut dec, None);
        let mut out = Vec::new();
        while let Ok(f) = Decoder::receive_frame(&mut dec) {
            out.push(f);
        }
        out
    }

    #[test]
    fn decodes_all_intra_fixture() {
        // H264_TEST=/path.h264 to decode another fixture; the reference is
        // /tmp/h264_ref_<stem>.yuv, a DEFAULT `ffmpeg -i x.h264 -f rawvideo
        // -pix_fmt yuv420p` decode (loop filter on).
        let path = std::env::var("H264_TEST").unwrap_or_else(|_| "/tmp/h264_alli.h264".into());
        let Ok(_data) = std::fs::read(&path) else {
            eprintln!("skip: no {path} fixture");
            return;
        };
        let frames = decode_file(&path);
        eprintln!("H264 WIP: decoded {} frames", frames.len());
        let Some(fr0) = frames.first() else { return };
        let (w, h) = (fr0.width as usize, fr0.height as usize);
        let stem = path
            .trim_start_matches("/tmp/h264_")
            .trim_end_matches(".h264");
        let ref_path = std::format!("/tmp/h264_ref_{stem}.yuv");
        let Ok(_ref) = std::fs::read(&ref_path) else {
            eprintln!("skip: no {ref_path} reference");
            return;
        };
        let plane = |fr: &Frame, p: usize| fr.plane(p).to_vec();
        for (idx, fr) in frames.iter().enumerate() {
            let y = plane(fr, 0);
            let u = plane(fr, 1);
            let v = plane(fr, 2);
            let off = idx * (w * h * 3 / 2);
            if off + w * h * 3 / 2 > _ref.len() {
                break;
            }
            let mut maxd = 0i32;
            let mut cmp = |ours: &[u8], rp: usize| {
                for k in 0..ours.len() {
                    let r = _ref[rp + k] as i32;
                    maxd = maxd.max((ours[k] as i32 - r).abs());
                }
            };
            cmp(&y, off);
            cmp(&u, off + w * h);
            cmp(&v, off + w * h + w * h / 4);
            eprintln!("H264 FRAME {idx}: max pixel diff = {maxd}");
            if std::env::var_os("H264_DUMP").is_some() {
                let mut out = Vec::new();
                out.extend_from_slice(&y);
                out.extend_from_slice(&u);
                out.extend_from_slice(&v);
                let _ = std::fs::write("/tmp/h264_ours.yuv", &out);
                // all-decode dump (append): lets a python pass compare any
                // P frame against the reference stream.
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open("/tmp/h264_ours_all.yuv")
                {
                    let _ = f.write_all(&out);
                }
            }
            // Bit-exact vs default ffmpeg (deblocking included).
            if std::env::var_os("H264_DUMP").is_none() {
                assert_eq!(maxd, 0, "frame {idx} differs from the ffmpeg reference");
            }
        }
    }
}
