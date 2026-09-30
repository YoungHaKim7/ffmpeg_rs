// ---------------------------------------------------------------------
// The decoder
// ---------------------------------------------------------------------

use crate::{
    PixelFormat,
    codec::{
        packet::Packet,
        params::{CodecId, CodecParameters, MediaType},
        traits::Decoder,
    },
    util::error::{Error, Result},
    util::frame::Frame,
};

use super::vlc::cavlc;
use super::{
    CHROMA_DC, CHROMA_QP8, DC_PRED, Gb, I4_DC_128_PRED, I4_LEFT_DC_PRED, I4_TOP_DC_PRED, LUMA_DC,
    MB_INTER, MB_INTRA4X4, MB_INTRA16X16, MB_PCM, MB_UNAVAIL, MbDeblock, Nal, PART_16X16, Part,
    PlaneSel, Pps, SCAN8, Sps, ZIGZAG, chroma_dc_dequant_idct, deblock,
    deblock::{PART_8X8, PART_8X16, PART_16X8},
    decode_residual, idct_add, idct_dc_add, luma_dc_dequant_idct, mc_chroma, mc_luma, mid_pred,
    parse_pps, parse_sps,
    picture::Picture,
    pred4x4, pred8x8_avail, pred16x16_avail, scan_shift1, split_nals,
    tables::{
        CHROMA_DC_SCAN, GOLOMB_TO_INTER_CBP, GOLOMB_TO_INTRA4X4_CBP, GOLOMB_TO_PICT_TYPE,
        I_MB_TYPE_INFO,
    },
    type_mask_nnz,
};

pub struct H264Decoder {
    pub(super) sps: Option<Sps>,
    pub(super) pps: Option<Pps>,
    pub(super) params: CodecParameters,
    pub(super) pending: std::collections::VecDeque<Frame>,
    pub(super) eof: bool,
    // picture state
    pub(super) cur: Option<Picture>,
    /// Short-term reference DPB, newest first (sliding window of
    /// max_num_ref_frames; C's h264_refs.c without MMCO/long-term).
    pub(super) refs: Vec<Picture>,
    /// RefPicList0 for the current slice: indices into `refs`.
    pub(super) ref_list: Vec<usize>,
    /// num_ref_idx_l0_active for the current slice (PPS default or the
    /// slice-header override).
    ref_count_l0: u32,
    /// Parsed ref_pic_list_modification ops (idc, value), applied after
    /// the previous picture enters the DPB.
    pub(super) reorder_ops: Vec<(u32, u32)>,
    /// frame_num from the slice header being decoded.
    pub(super) slice_frame_num: u32,
    /// nal_ref_idc of the current picture (non-ref pictures stay out of
    /// the DPB).
    pub(super) cur_is_ref: bool,
    /// P_8x8ref0 (mb_type 4): every sub-8x8 uses ref 0, no ref_idx.
    p8x8_ref0: bool,
    pub(super) got_mb: bool, // cur has decoded MBs (start-of-picture detection)
    pub(super) frame_num: u32,
    // slice state
    pub(super) slice_type_nos: u8, // 0=P, 2=I
    pub(super) qscale: i32,
    pub(super) chroma_qp: [i32; 2],
    pub(super) mb_skip_run: i64,
    pub(super) mb_x: usize,
    pub(super) mb_y: usize,
    pub(super) mb_width: usize,
    pub(super) mb_height: usize,
    pub(super) slice_num: usize,
    pub(super) slice_table: Vec<usize>,
    pub(super) prev_mb_skipped: bool,
    /// Deblocking parameters of the slice being decoded.
    pub(super) slice_dbk: MbDeblock,
    /// Inter partition shape of the MB being decoded (PART_*).
    pub(super) cur_part: u8,
    pub(super) next_pic_id: u64,
    // per-MB scratch
    pub(super) mb: [i16; 48 * 16],
    pub(super) mb_luma_dc: [i16; 16],
    intra4x4_pred_mode_cache: [i8; 15 * 8],
    pub(super) nnz_cache: [u8; 15 * 8],
    pub(super) mv_cache: [[i16; 2]; 15 * 8],
    pub(super) ref_cache: [i8; 15 * 8],
    pub(super) top_samples_available: u16,
    pub(super) left_samples_available: u16,
    pub(super) topright_samples_available: u16,
    // neighbor types (0 = unavailable)
    pub(super) n_top: u32,
    pub(super) n_left: u32,
    pub(super) n_topleft: u32,
    pub(super) n_topright: u32,
    // per-MB decoded info for reconstruction
    pub(super) mb_type: u32,
    intra16x16_pred_mode: i32,
    pub(super) chroma_pred_mode: i32,
    pub(super) cbp: u32,
    // inter info for this MB: per 4x4 mv/ref in cache; partitions kept
    // implicitly via mv/ref caches (write-back per 4x4).
    pub(super) intra_pcm: Vec<u8>,
    pub(super) frame_count: usize,
    // applied crop
    pub(super) out_w: usize,
    pub(super) out_h: usize,
}

impl Default for H264Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl H264Decoder {
    pub fn new() -> Self {
        H264Decoder {
            sps: None,
            pps: None,
            params: CodecParameters::default(),
            pending: std::collections::VecDeque::new(),
            eof: false,
            cur: None,
            refs: Vec::new(),
            ref_list: Vec::new(),
            ref_count_l0: 1,
            reorder_ops: Vec::new(),
            slice_frame_num: 0,
            cur_is_ref: true,
            p8x8_ref0: false,
            got_mb: false,
            frame_num: u32::MAX,
            slice_type_nos: 2,
            qscale: 0,
            chroma_qp: [0, 0],
            mb_skip_run: -1,
            mb_x: 0,
            mb_y: 0,
            mb_width: 0,
            mb_height: 0,
            slice_num: 0,
            slice_table: Vec::new(),
            prev_mb_skipped: false,
            slice_dbk: MbDeblock::default(),
            cur_part: PART_16X16,
            next_pic_id: 1,
            mb: [0; 48 * 16],
            mb_luma_dc: [0; 16],
            intra4x4_pred_mode_cache: [-1; 120],
            nnz_cache: [0; 120],
            mv_cache: [[0, 0]; 120],
            ref_cache: [-2; 120],
            top_samples_available: 0,
            left_samples_available: 0,
            topright_samples_available: 0,
            n_top: 0,
            n_left: 0,
            n_topleft: 0,
            n_topright: 0,
            mb_type: 0,
            intra16x16_pred_mode: 0,
            chroma_pred_mode: 0,
            cbp: 0,
            intra_pcm: Vec::new(),
            frame_count: 0,
            out_w: 0,
            out_h: 0,
        }
    }

    pub fn flush(&mut self) {
        self.cur = None;
        self.refs.clear();
        self.ref_list.clear();
        self.slice_table.clear();
        self.pending.clear();
        self.got_mb = false;
    }

    // ---------------- slice header ----------------

    fn parse_slice_header(&mut self, nal: &Nal, gb: &mut Gb) -> Result<(usize, bool)> {
        let sps = self
            .sps
            .clone()
            .ok_or_else(|| Error::InvalidData("no SPS".into()))?;
        let pps = self
            .pps
            .clone()
            .ok_or_else(|| Error::InvalidData("no PPS".into()))?;

        let first_mb = gb.ue()? as usize;
        let mut slice_type = gb.ue()?;
        if slice_type > 9 {
            return Err(Error::InvalidData("slice type too large".into()));
        }
        if slice_type > 4 {
            slice_type -= 5;
        }
        // C's AV_PICTURE_TYPE codes: I=1, P=2, B=3, SP=4, SI=5 — remapped
        // here to 2=I / 0=P (the port's slice_type_nos).
        let st = match GOLOMB_TO_PICT_TYPE[slice_type as usize] {
            1 => 2u8, // I
            2 => 0u8, // P
            3 => return Err(Error::Unsupported("B slices".into())),
            _ => return Err(Error::Unsupported("SP/SI slices".into())),
        };
        if nal.kind == 5 && st != 2 {
            return Err(Error::InvalidData("non-intra slice in IDR NAL".into()));
        }
        self.slice_type_nos = st;

        let pps_id = gb.ue()?;
        if pps_id != 0 {
            return Err(Error::InvalidData("pps_id != 0 (single-PPS subset)".into()));
        }

        let frame_num = gb.read(sps.log2_max_frame_num);
        // frame pictures; no field flags.
        let _idr_pic_id = if nal.kind == 5 { gb.ue()? } else { 0 };
        if std::env::var_os("H264_DUMP").is_some() {
            eprintln!(
                "  H1 idr={} pos={} log2poc={} log2fn={}",
                _idr_pic_id, gb.index, sps.log2_max_poc_lsb, sps.log2_max_frame_num
            );
        }

        let poc_lsb_dbg = match sps.poc_type {
            0 => {
                let v = gb.read(sps.log2_max_poc_lsb);
                if std::env::var_os("H264_DUMP").is_some() {
                    eprintln!(
                        "  H2 poc={v} pop={} pos={}",
                        pps.pic_order_present, gb.index
                    );
                }
                let _dpb = gb.se()?;
                v
            }
            _ => 0,
        };

        if pps.redundant_pic_cnt_present {
            let _rpc = gb.ue()?;
        }

        self.slice_frame_num = frame_num;
        self.ref_count_l0 = pps.ref_count[0];
        self.reorder_ops.clear();
        if self.slice_type_nos != 2 {
            // ff_h264_parse_ref_count (h264_parse.c:237): the l1 count
            // is only read for B slices.
            if gb.read_bit() == 1 {
                let l0 = gb.ue()? + 1;
                if l0 > 32 {
                    return Err(Error::InvalidData("reference overflow".into()));
                }
                self.ref_count_l0 = l0;
                if self.slice_type_nos == 1 {
                    let _l1 = gb.ue()?;
                }
            }
            // ff_h264_decode_ref_pic_list_reordering (h264_refs.c:431):
            // op/value pairs until op == 3. Single-ref DPB ⇒ reordering
            // is a parse-only no-op (it cannot move the only picture).
            if gb.read_bit() == 1 {
                loop {
                    let op = gb.ue()?;
                    if op == 3 {
                        break;
                    }
                    if op > 2 {
                        return Err(Error::InvalidData(format!(
                            "illegal modification_of_pic_nums_idc {op}"
                        )));
                    }
                    let val = gb.ue()?;
                    if op == 2 {
                        return Err(Error::Unsupported("long-term reference reordering".into()));
                    }
                    self.reorder_ops.push((op, val));
                }
            }
        }

        if nal.ref_idc != 0 {
            if nal.kind == 5 {
                let _noop = gb.read_bit();
                let _ltr = gb.read_bit();
            } else if gb.read_bit() == 1 {
                return Err(Error::Unsupported("MMCO".into()));
            }
        }
        if std::env::var_os("H264_DUMP").is_some() {
            eprintln!("  H3 post-marking pos={}", gb.index);
        }

        let qp = pps.init_qp + gb.se()?;
        if std::env::var_os("H264_DUMP").is_some() {
            eprintln!(
                "HDR fm={first_mb} st_nos={} pps={pps_id} fn={frame_num} poc={poc_lsb_dbg} pos={}",
                self.slice_type_nos, gb.index
            );
        }
        if !(0..=51).contains(&qp) {
            return Err(Error::InvalidData(format!("QP {qp} out of range")));
        }
        self.qscale = qp;
        let off = self.pps.as_ref().map(|p| p.chroma_qp_offset).unwrap_or(0);
        let cqp = CHROMA_QP8[(qp + off).clamp(0, 51) as usize] as i32;
        self.chroma_qp[0] = cqp;
        self.chroma_qp[1] = cqp;

        // Deblocking (h264_slice.c:1900-1928): C keeps the idc with 0/1
        // swapped (1 = on, 0 = off, 2 = on but not across slice edges) and
        // the offsets doubled.
        let mut dbk = MbDeblock {
            mode: 1,
            cqp_off: [pps.chroma_qp_offset; 2],
            ..MbDeblock::default()
        };
        if pps.deblocking_filter_parameters_present {
            let idc = gb.ue()?;
            if idc > 2 {
                return Err(Error::InvalidData("deblocking_filter_idc".into()));
            }
            let mut on = idc as u8;
            if on < 2 {
                on ^= 1;
            }
            dbk.mode = on;
            if on != 0 {
                let a = gb.se()?;
                let b = gb.se()?;
                if !(-6..=6).contains(&a) || !(-6..=6).contains(&b) {
                    return Err(Error::InvalidData("deblocking offsets".into()));
                }
                dbk.alpha = a * 2;
                dbk.beta = b * 2;
            }
        }
        self.slice_dbk = dbk;
        Ok((
            first_mb,
            frame_num != self.frame_num || nal.kind == 5 || first_mb == 0 && self.got_mb,
        ))
    }

    // ---------------- MB layer ----------------

    /// `pred_non_zero_count` (h264_cavlc.c:275).
    pub(super) fn pred_nnz(&self, n: usize) -> u32 {
        let idx = SCAN8[n];
        let left = self.nnz_cache[idx - 1] as u32;
        let top = self.nnz_cache[idx - 8] as u32;
        let mut i = left + top;
        if i < 64 {
            i = (i + 1) >> 1;
        }
        i & 31
    }

    /// `pred_intra_mode` (h264_mvpred.h:42).
    fn pred_intra_mode(&self, n: usize) -> i8 {
        let idx = SCAN8[n];
        let left = self.intra4x4_pred_mode_cache[idx - 1];
        let top = self.intra4x4_pred_mode_cache[idx - 8];
        let min = left.min(top);
        if min < 0 { DC_PRED } else { min }
    }

    /// `ff_h264_check_intra4x4_pred_mode` (h264_parse.c:134) — returns
    /// the (possibly DC-fallback) modes for the 16 blocks.
    fn check_intra4x4_pred_mode(&mut self) -> Result<()> {
        // C (h264_parse.c:134): top[] = {-1,0,LEFT_DC_PRED,-1,-1,-1,-1,-1,0}
        // left[] = {0,-1,TOP_DC_PRED,0,-1,-1,-1,0,-1,DC_128_PRED} — the DC
        // variants are the i4x4-namespace codes 9/10/11 (NOT the 0-3
        // 16x16 codes); pred4x4 handles them with availability.
        static TOP: [i8; 12] = [-1, 0, I4_LEFT_DC_PRED, -1, -1, -1, -1, -1, 0, -1, -1, -1];
        static LEFT: [i8; 12] = [
            0,
            -1,
            I4_TOP_DC_PRED,
            0,
            -1,
            -1,
            -1,
            0,
            -1,
            I4_DC_128_PRED,
            -1,
            -1,
        ];
        if self.top_samples_available & 0x8000 == 0 {
            for i in 0..4 {
                let m = self.intra4x4_pred_mode_cache[SCAN8[0] + i];
                let status = TOP[m as usize];
                if status < 0 {
                    return Err(Error::InvalidData(
                        "top block unavailable for requested intra mode".into(),
                    ));
                } else if status != 0 {
                    self.intra4x4_pred_mode_cache[SCAN8[0] + i] = status;
                }
            }
        }
        if self.left_samples_available & 0x8888 != 0x8888 {
            static MASK: [u16; 4] = [0x8000, 0x2000, 0x80, 0x20];
            for i in 0..4 {
                if self.left_samples_available & MASK[i] == 0 {
                    let m = self.intra4x4_pred_mode_cache[SCAN8[0] + 8 * i];
                    let status = LEFT[m as usize];
                    if status < 0 {
                        return Err(Error::InvalidData(
                            "left block unavailable for requested intra4x4 mode".into(),
                        ));
                    } else if status != 0 {
                        self.intra4x4_pred_mode_cache[SCAN8[0] + 8 * i] = status;
                    }
                }
            }
        }
        Ok(())
    }

    /// `ff_h264_check_intra_pred_mode` (h264_parse.c:182) for 16x16/chroma.
    /// Luma16 mode space: 0=V,1=H,2=DC,3=plane; chroma: 0=DC,1=H,2=V,3=plane.
    fn check_intra_pred_mode(&self, mode: i32, is_chroma: bool) -> Result<i32> {
        // The C tables use mode codes where V=0,H=1,DC=2 for both luma16
        // and chroma(H=1). Chroma mode from bitstream: 0=DC,1=H,2=V,3=P —
        // remap to the V/H/DC/plane space first.
        let m = if is_chroma {
            match mode {
                0 => 2, // DC
                1 => 1, // H
                2 => 0, // V
                _ => 3, // plane
            }
        } else {
            mode
        };
        if std::env::var_os("H264_DUMP").is_some() {
            eprintln!(
                "CHK mb={}:{} m={m} is_chroma={is_chroma} top={:#06x} left={:#06x}",
                self.mb_x, self.mb_y, self.top_samples_available, self.left_samples_available
            );
        }
        if m > 3 {
            return Err(Error::InvalidData(
                "out of range intra chroma pred mode".into(),
            ));
        }
        // Port space (0=V,1=H,2=DC,3=plane): top unavailable →
        // V becomes DC (plane is illegal there per C's table); left
        // unavailable → H becomes DC.
        static TOP: [i32; 4] = [2, 1, 2, -1];
        static LEFT: [i32; 5] = [0, 2, 2, -1, -1];
        // C table indices (luma16 space): top[] = {LEFT_DC_PRED8x8,1,-1,-1}
        // left[] = {TOP_DC_PRED8x8,-1,2,-1,DC_128_PRED8x8}; in V/H/DC
        // coding: DC from left only = 1 (H? no...). Ported by value:
        let top_map: [i32; 4] = [1, 2, -1, -1]; // V→H? no — see below
        let _ = top_map;
        // C: top[] = { LEFT_DC_PRED8x8=1, 1, -1, -1 }: mode V(0)→1(H??)
        // — actually LEFT_DC_PRED8x8 means "DC using left only". In the
        // V/H/DC space used by pred16x16: 0=V,1=H,2=DC. "DC-left-only"
        // is a variant we fold into DC with availability handled at
        // prediction time (the neighbor arrays are pre-filled), so the
        // fallbacks here only need to reject unavailable modes:
        let mut mm = m;
        if self.top_samples_available & 0x8000 == 0 {
            let t = TOP[mm as usize];
            if t < 0 {
                return Err(Error::InvalidData("top unavailable".into()));
            }
            mm = t;
        }
        if self.left_samples_available & 0x8080 != 0x8080 {
            let l = LEFT[mm as usize];
            if l < 0 {
                return Err(Error::InvalidData("left unavailable".into()));
            }
            mm = l;
        }
        Ok(mm)
    }

    /// fill_decode_neighbors + fill_decode_caches, frame-only path
    /// (h264_mvpred.h:487/539, the non-MBAFF branches).
    fn fill_decode_caches(&mut self, mb_type: u32) {
        let mb_xy = self.mb_x + self.mb_y * self.mb_width;
        let top_xy = mb_xy as isize - self.mb_width as isize;
        let top_ok = self.mb_y > 0;
        let left_ok = self.mb_x > 0;
        let topleft_ok = self.mb_x > 0 && self.mb_y > 0;
        let topright_ok = self.mb_x + 1 < self.mb_width && self.mb_y > 0;
        let in_slice = |xy: isize| -> bool {
            xy >= 0
                && (xy as usize) < self.slice_table.len()
                && self.slice_table[xy as usize] == self.slice_num
        };
        let (t, l, tl, tr) = {
            let pic = self.cur.as_ref().unwrap();
            let type_of = |xy: isize, ok: bool| -> u32 {
                if ok && in_slice(xy) {
                    pic.mb_type[xy as usize]
                } else {
                    0
                }
            };
            (
                type_of(top_xy, top_ok),
                type_of(mb_xy as isize - 1, left_ok),
                type_of(top_xy - 1, topleft_ok),
                type_of(top_xy + 1, topright_ok),
            )
        };
        self.n_top = t;
        self.n_left = l;
        self.n_topleft = tl;
        self.n_topright = tr;

        // ---- intra sample availability (the frame-only branches) ----
        let constrained = self.pps.as_ref().unwrap().constrained_intra_pred;
        let type_mask = |t: u32| -> bool {
            if constrained {
                t == MB_INTRA4X4 || t == MB_INTRA16X16 || t == MB_PCM
            } else {
                t != MB_UNAVAIL
            }
        };
        let is_intra = mb_type == MB_INTRA4X4 || mb_type == MB_INTRA16X16 || mb_type == MB_PCM;
        if is_intra {
            self.top_samples_available = 0xFFFF;
            self.left_samples_available = 0xFFFF;
            self.topright_samples_available = 0xEEEA;
            if !type_mask(self.n_top) {
                self.topleft_samples_available_hack(0xB3FF);
                self.top_samples_available = 0x33FF;
                self.topright_samples_available = 0x26EA;
            }
            if !type_mask(self.n_left) {
                self.topleft_samples_available_hack(0xDF5F & 0xB3FF.max(1));
                self.left_samples_available &= 0x5F5F;
            }
            if !type_mask(self.n_topleft) {
                self.topleft_samples_available_hack(0x7FFF);
            }
            if !type_mask(self.n_topright) {
                self.topright_samples_available &= 0xFBFF;
            }
            if mb_type == MB_INTRA4X4 {
                // intra4x4_pred_mode cache borders
                let top4 = if self.n_top == MB_INTRA4X4 {
                    None
                } else {
                    Some(2 - 3 * (!type_mask(self.n_top)) as i8)
                };
                let _ = top4;
                // (populated below via write-back arrays)
                self.fill_intra4x4_mode_cache();
                let left4 = if self.n_left == MB_INTRA4X4 {
                    None
                } else {
                    Some(2 - 3 * (!type_mask(self.n_left)) as i8)
                };
                let _ = left4;
            }
        } else {
            self.fill_intra4x4_mode_cache_inter();
        }

        // ---- nnz cache ----
        {
            let pic = self.cur.as_ref().unwrap();
            let b_stride = self.mb_width * 4 + 1;
            if type_mask_nnz(self.n_top) {
                let nnz = &pic.nnz[(self.mb_x + (self.mb_y - 1).max(0) * self.mb_width).max(0)];
                for c in 0..4 {
                    self.nnz_cache[4 + 0 * 8 + c] = nnz[12 + c];
                    self.nnz_cache[4 + 5 * 8 + c] = nnz[20 + c];
                    self.nnz_cache[4 + 10 * 8 + c] = nnz[36 + c];
                }
            } else {
                let v = if self.n_top != MB_UNAVAIL { 0u8 } else { 64 };
                for c in 0..4 {
                    self.nnz_cache[4 + c] = v;
                    self.nnz_cache[4 + 5 * 8 + c] = v;
                    self.nnz_cache[4 + 10 * 8 + c] = v;
                }
            }
            for i in 0..2usize {
                if type_mask_nnz(self.n_left) {
                    let nnz =
                        &pic.nnz[(self.mb_x.saturating_sub(1) + self.mb_y * self.mb_width).max(0)];
                    self.nnz_cache[3 + 8 * 1 + 2 * 8 * i] = nnz[[3usize, 11][i]];
                    self.nnz_cache[3 + 8 * 2 + 2 * 8 * i] = nnz[[7usize, 15][i]];
                    self.nnz_cache[3 + 8 * 6 + 8 * i] = nnz[[17usize, 21][i]];
                    self.nnz_cache[3 + 8 * 11 + 8 * i] = nnz[[33usize, 37][i]];
                } else {
                    let v = if self.n_left != MB_UNAVAIL { 0u8 } else { 64 };
                    self.nnz_cache[3 + 8 * 1 + 2 * 8 * i] = v;
                    self.nnz_cache[3 + 8 * 2 + 2 * 8 * i] = v;
                    self.nnz_cache[3 + 8 * 6 + 8 * i] = v;
                    self.nnz_cache[3 + 8 * 11 + 8 * i] = v;
                }
            }
            let _ = b_stride;
        }

        // ---- mv/ref cache (list 0) ----
        // C fill_decode_caches (h264_mvpred.h:740-800, frame pictures):
        // top = BOTTOM row of the MB above (ref[4*top+2 + c/2]),
        // topright cell = scan8[0]-8+4 (= 8; the 8-wide cache wraps), from
        // the bottom-left 4x4 of the top-right MB (ref +2), topleft cell =
        // scan8[0]-8-1 from the bottom-right 4x4 of the top-left MB (ref +3).
        if mb_type == MB_INTER {
            let pic = self.cur.as_ref().unwrap();
            let b_stride = self.mb_width * 4 + 1;
            let top_xy = self.mb_x + self.mb_y.saturating_sub(1) * self.mb_width;
            let left_xy = self.mb_x.saturating_sub(1) + self.mb_y * self.mb_width;
            let uses = |t: u32| -> bool { t == MB_INTER };
            let fill_edge = |t: u32| -> i8 { if t != MB_UNAVAIL { -1 } else { -2 } };
            let top0 = SCAN8[0] - 8;
            if uses(self.n_top) {
                let bxy = 4 * self.mb_x + 4 * (self.mb_y - 1) * b_stride + 3 * b_stride;
                for c in 0..4 {
                    self.mv_cache[top0 + c] = pic.mv[bxy + c];
                    self.ref_cache[top0 + c] = pic.ref_index[4 * top_xy + 2 + (c >> 1)];
                }
            } else {
                for c in 0..4 {
                    self.mv_cache[top0 + c] = [0, 0];
                    self.ref_cache[top0 + c] = fill_edge(self.n_top);
                }
            }
            let tr = SCAN8[0] - 8 + 4;
            if uses(self.n_topright) {
                let bxy = 4 * (self.mb_x + 1) + 4 * (self.mb_y - 1) * b_stride + 3 * b_stride;
                self.mv_cache[tr] = pic.mv[bxy];
                self.ref_cache[tr] = pic.ref_index[4 * (top_xy + 1) + 2];
            } else {
                self.mv_cache[tr] = [0, 0];
                self.ref_cache[tr] = fill_edge(self.n_topright);
            }
            let tl = SCAN8[0] - 8 - 1;
            if uses(self.n_topleft) {
                let bxy = 4 * (self.mb_x - 1) + 4 * (self.mb_y - 1) * b_stride + 3 + 3 * b_stride;
                self.mv_cache[tl] = pic.mv[bxy];
                self.ref_cache[tl] = pic.ref_index[4 * (top_xy - 1) + 3];
            } else {
                self.mv_cache[tl] = [0, 0];
                self.ref_cache[tl] = fill_edge(self.n_topleft);
            }
            for i in 0..4 {
                if uses(self.n_left) {
                    let bxy = 4 * self.mb_x.saturating_sub(1)
                        + 4 * self.mb_y * b_stride
                        + 3
                        + i * b_stride;
                    self.mv_cache[3 + 8 * (1 + i)] = pic.mv[bxy];
                    self.ref_cache[3 + 8 * (1 + i)] = pic.ref_index[4 * left_xy + 1 + 2 * (i >> 1)];
                } else if self.n_left != MB_UNAVAIL {
                    self.mv_cache[3 + 8 * (1 + i)] = [0, 0];
                    self.ref_cache[3 + 8 * (1 + i)] = -1;
                } else {
                    self.mv_cache[3 + 8 * (1 + i)] = [0, 0];
                    self.ref_cache[3 + 8 * (1 + i)] = -2;
                }
            }
        }
    }

    // helpers used above (kept tiny to avoid a big rewrite of masks)
    fn topleft_samples_available_hack(&mut self, mask: u16) {
        // C masks topleft via the same field; we keep a separate implicit
        // topleft availability inside top/left handling (folded).
        self.top_samples_available |= mask & !0xFFFF; // no-op placeholder
    }
    fn fill_intra4x4_mode_cache(&mut self) {
        let pic = self.cur.as_ref().unwrap();
        let constrained = self.pps.as_ref().unwrap().constrained_intra_pred;
        let type_mask = |t: u32| -> bool {
            if constrained {
                t == MB_INTRA4X4
            } else {
                t != MB_UNAVAIL
            }
        };
        if self.n_top == MB_INTRA4X4 {
            let modes = &pic.mb_i4x4[self.mb_x + (self.mb_y - 1).max(0) * self.mb_width];
            // C (h264_mvpred.h): AV_COPY32(cache+4+8*0, stored) — the
            // above MB's stored[0..3] = its BOTTOM luma row.
            for c in 0..4 {
                self.intra4x4_pred_mode_cache[4 + c] = modes[c];
            }
        } else {
            let v = 2 - 3 * (!type_mask(self.n_top)) as i8;
            for c in 0..4 {
                self.intra4x4_pred_mode_cache[4 + c] = v;
            }
        }
        {
            // C: cache[3+8*(1..4)] = mode[6 - left_block[0..3]] with the
            // progressive left_block {0,1,2,3} → stored[6]=b5 (row 1),
            // stored[5]=b7 (row 2), stored[4]=b13 (row 3), stored[3]=b15
            // (row 4) — the left MB's right-column blocks bottom-to-top.
            if self.n_left == MB_INTRA4X4 {
                let modes = &pic.mb_i4x4[self.mb_x.saturating_sub(1) + self.mb_y * self.mb_width];
                self.intra4x4_pred_mode_cache[3 + 8 * 1] = modes[6];
                self.intra4x4_pred_mode_cache[3 + 8 * 2] = modes[5];
                self.intra4x4_pred_mode_cache[3 + 8 * 3] = modes[4];
                self.intra4x4_pred_mode_cache[3 + 8 * 4] = modes[3];
            } else {
                let v = 2 - 3 * (!type_mask(self.n_left)) as i8;
                for r in 1..5usize {
                    self.intra4x4_pred_mode_cache[3 + 8 * r] = v;
                }
            }
        }
    }
    fn fill_intra4x4_mode_cache_inter(&mut self) {}

    // write-backs (h264_mvpred.h)
    fn write_back_intra_pred_mode(&mut self, mb_xy: usize) {
        let c = &self.intra4x4_pred_mode_cache;
        let stored: [i8; 8] = [
            c[4 + 8 * 4],
            c[5 + 8 * 4],
            c[6 + 8 * 4],
            c[7 + 8 * 4],
            c[7 + 8 * 3],
            c[7 + 8 * 2],
            c[7 + 8 * 1],
            c[7 + 8 * 0],
        ];
        // mb_i4x4 stores the bottom row (4) + right column (next 4) in
        // C's mb2br layout shape [4+4]
        self.cur.as_mut().unwrap().mb_i4x4[mb_xy] = stored;
    }

    fn write_back_non_zero_count(&mut self, mb_xy: usize) {
        let c = &self.nnz_cache;
        let nnz = &mut self.cur.as_mut().unwrap().nnz[mb_xy];
        for i in 0..4 {
            nnz[0 + i] = c[4 + 8 * 1 + i];
            nnz[4 + i] = c[4 + 8 * 2 + i];
            nnz[8 + i] = c[4 + 8 * 3 + i];
            nnz[12 + i] = c[4 + 8 * 4 + i];
            nnz[16 + i] = c[4 + 8 * 6 + i];
            nnz[20 + i] = c[4 + 8 * 7 + i];
            nnz[32 + i] = c[4 + 8 * 11 + i];
            nnz[36 + i] = c[4 + 8 * 12 + i];
        }
    }

    fn write_back_motion(&mut self, mb_xy: usize) {
        let b_stride = self.mb_width * 4 + 1;
        let b_xy = 4 * self.mb_x + 4 * self.mb_y * b_stride;
        let pic = self.cur.as_mut().unwrap();
        for r in 0..4 {
            for c in 0..4 {
                pic.mv[b_xy + r * b_stride + c] = self.mv_cache[SCAN8[0] + 8 * r + c];
            }
        }
        pic.ref_index[4 * mb_xy] = self.ref_cache[SCAN8[0]];
        pic.ref_index[4 * mb_xy + 1] = self.ref_cache[SCAN8[4]];
        pic.ref_index[4 * mb_xy + 2] = self.ref_cache[SCAN8[8]];
        pic.ref_index[4 * mb_xy + 3] = self.ref_cache[SCAN8[12]];
        // C's ref2frm: the loop filter compares the referenced PICTURES.
        for (k, blk) in [0usize, 4, 8, 12].into_iter().enumerate() {
            let ri = self.ref_cache[SCAN8[blk]];
            let id = if ri < 0 {
                -1
            } else {
                self.ref_list
                    .get(ri as usize)
                    .and_then(|&i| self.refs.get(i))
                    .map_or(-1, |p| p.id as i32)
            };
            self.cur.as_mut().unwrap().ref_pic[4 * mb_xy + k] = id;
        }
    }

    /// Per-MB state kept in the picture after decoding: qscale (0 for
    /// PCM — C's `qscale_table`, the running slice QP is untouched), cbp,
    /// partition shape and the slice's deblocking parameters.
    fn record_mb(&mut self, mb_xy: usize) {
        let pcm = self.mb_type == MB_PCM;
        let pic = self.cur.as_mut().unwrap();
        pic.qscale[mb_xy] = if pcm { 0 } else { self.qscale as u8 };
        pic.cbp[mb_xy] = if self.cbp == u32::MAX {
            0
        } else {
            self.cbp as u8
        };
        pic.part[mb_xy] = self.cur_part;
        pic.dbk[mb_xy] = self.slice_dbk;
        self.slice_table[mb_xy] = self.slice_num;
    }

    // ---------------- MV prediction (h264_mvpred.h) ----------------

    /// `fetch_diagonal_mv` frame path.
    fn fetch_diagonal_mv(&self, i: usize, part_width: usize) -> (i8, [i16; 2]) {
        let tr = self.ref_cache[i - 8 + part_width];
        if tr != -2 {
            (tr, self.mv_cache[i - 8 + part_width])
        } else {
            (self.ref_cache[i - 8 - 1], self.mv_cache[i - 8 - 1])
        }
    }

    /// `pred_motion`.
    fn pred_motion(&self, n: usize, part_width: usize, r: i8) -> (i16, i16) {
        let idx = SCAN8[n];
        let left_ref = self.ref_cache[idx - 1];
        let a = self.mv_cache[idx - 1];
        let top_ref = self.ref_cache[idx - 8];
        let b = self.mv_cache[idx - 8];
        let (diag_ref, c) = self.fetch_diagonal_mv(idx, part_width);
        let match_count = (diag_ref == r) as i32 + (top_ref == r) as i32 + (left_ref == r) as i32;
        let mid3 = |x: i16, y: i16, z: i16| -> i16 {
            // mid_pred
            let (lo, hi) = (x.min(y), x.max(y));
            if z < lo {
                lo
            } else if z > hi {
                hi
            } else {
                z
            }
        };
        if match_count > 1 {
            (mid3(a[0], b[0], c[0]), mid3(a[1], b[1], c[1]))
        } else if match_count == 1 {
            if left_ref == r {
                (a[0], a[1])
            } else if top_ref == r {
                (b[0], b[1])
            } else {
                (c[0], c[1])
            }
        } else if top_ref == -2 && diag_ref == -2 && left_ref != -2 {
            (a[0], a[1])
        } else {
            (mid3(a[0], b[0], c[0]), mid3(a[1], b[1], c[1]))
        }
    }

    /// `pred_16x8_motion` (n = 0 top half, 1 bottom half).
    fn pred_16x8_motion(&self, n: usize, r: i8) -> (i16, i16) {
        if n == 0 {
            let top_ref = self.ref_cache[SCAN8[0] - 8];
            let b = self.mv_cache[SCAN8[0] - 8];
            if top_ref == r {
                return (b[0], b[1]);
            }
        } else {
            let left_ref = self.ref_cache[SCAN8[8] - 1];
            let a = self.mv_cache[SCAN8[8] - 1];
            if left_ref == r {
                return (a[0], a[1]);
            }
        }
        // C passes the BLOCK index (8*n), not the partition number.
        self.pred_motion(8 * n, 4, r)
    }

    /// `pred_8x16_motion` (n = 0 left half, 1 right half).
    fn pred_8x16_motion(&self, n: usize, r: i8) -> (i16, i16) {
        if n == 0 {
            let left_ref = self.ref_cache[SCAN8[0] - 1];
            let a = self.mv_cache[SCAN8[0] - 1];
            if left_ref == r {
                return (a[0], a[1]);
            }
        } else {
            let (d, c) = self.fetch_diagonal_mv(SCAN8[4], 2);
            if d == r {
                return (c[0], c[1]);
            }
        }
        // C passes the BLOCK index (4*n), not the partition number.
        self.pred_motion(4 * n, 2, r)
    }

    /// `pred_pskip_motion` (h264_mvpred.h:390) — C-verbatim for frame
    /// pictures: neighbours read from the written-back picture arrays
    /// (bottom row of top/topright, right column of left, bottom-right of
    /// topleft), zero-MV early outs, then the pred_motion match_count rule.
    fn pred_pskip_motion(&mut self) {
        const NOT_USED: i8 = -1; // LIST_NOT_USED
        const NOT_AVAIL: i8 = -2; // PART_NOT_AVAILABLE
        let b_stride = self.mb_width * 4 + 1;
        let pic = self.cur.as_ref().unwrap();
        let zero = [0i16, 0];
        let b_xy = |x: usize, y: usize| 4 * x + 4 * y * b_stride;
        let mw = self.mb_width;
        let (mx_, my_) = (self.mb_x, self.mb_y);
        let mv = 'pred: {
            // A: left
            let (left_ref, a) = if self.n_left == MB_INTER {
                let xy = mx_ - 1 + my_ * mw;
                let r = pic.ref_index[4 * xy + 1];
                let a = pic.mv[b_xy(mx_ - 1, my_) + 3];
                if r == 0 && a == zero {
                    break 'pred zero;
                }
                (r, a)
            } else if self.n_left != MB_UNAVAIL {
                (NOT_USED, zero)
            } else {
                break 'pred zero;
            };
            // B: top
            let (top_ref, bm) = if self.n_top == MB_INTER {
                let xy = mx_ + (my_ - 1) * mw;
                let r = pic.ref_index[4 * xy + 2];
                let bm = pic.mv[b_xy(mx_, my_ - 1) + 3 * b_stride];
                if r == 0 && bm == zero {
                    break 'pred zero;
                }
                (r, bm)
            } else if self.n_top != MB_UNAVAIL {
                (NOT_USED, zero)
            } else {
                break 'pred zero;
            };
            // C: topright, else topleft
            let (diag_ref, c) = if self.n_topright == MB_INTER {
                let xy = mx_ + 1 + (my_ - 1) * mw;
                (
                    pic.ref_index[4 * xy + 2],
                    pic.mv[b_xy(mx_ + 1, my_ - 1) + 3 * b_stride],
                )
            } else if self.n_topright != MB_UNAVAIL {
                (NOT_USED, zero)
            } else if self.n_topleft == MB_INTER {
                let xy = mx_ - 1 + (my_ - 1) * mw;
                (
                    pic.ref_index[4 * xy + 3],
                    pic.mv[b_xy(mx_ - 1, my_ - 1) + 3 + 3 * b_stride],
                )
            } else if self.n_topleft != MB_UNAVAIL {
                (NOT_USED, zero)
            } else {
                (NOT_AVAIL, zero)
            };
            let match_count =
                (diag_ref == 0) as i32 + (top_ref == 0) as i32 + (left_ref == 0) as i32;
            if match_count > 1 {
                [mid_pred(a[0], bm[0], c[0]), mid_pred(a[1], bm[1], c[1])]
            } else if match_count == 1 {
                if left_ref == 0 {
                    a
                } else if top_ref == 0 {
                    bm
                } else {
                    c
                }
            } else {
                [mid_pred(a[0], bm[0], c[0]), mid_pred(a[1], bm[1], c[1])]
            }
        };
        for r in 0..4 {
            for c in 0..4 {
                self.mv_cache[SCAN8[0] + 8 * r + c] = mv;
                self.ref_cache[SCAN8[0] + 8 * r + c] = 0;
            }
        }
    }
    // ---------------- CAVLC MB decode (h264_cavlc.c:682) ----------------

    fn decode_mb_cavlc(&mut self, gb: &mut Gb) -> Result<()> {
        let mb_xy = self.mb_x + self.mb_y * self.mb_width;

        // mb_skip_run (P slices; C: `if (sl->mb_skip_run--)`)
        if self.slice_type_nos == 0 {
            if self.mb_skip_run == -1 {
                self.mb_skip_run = gb.ue()? as i64;
            }
            let run = self.mb_skip_run;
            self.mb_skip_run -= 1;
            if run > 0 {
                self.decode_mb_skip(mb_xy);
                // P_Skip still reconstructs (MC from the reference with
                // the predicted skip MV — cavlc path returns early but
                // hl_decode_mb runs for every MB, C: h264dec.c:101).
                self.hl_decode_mb(mb_xy)?;
                return Ok(());
            }
            // run == 0: falls through with the counter at -1 so the next
            // MB re-reads (C's postfix-decrement semantics).
        }
        self.prev_mb_skipped = false;

        let raw = gb.ue()?;
        let is_i = self.slice_type_nos == 2;
        if std::env::var_os("H264_DUMP").is_some() {
            eprintln!(
                "MBT mb={}:{} raw={raw} pos={}",
                self.mb_x, self.mb_y, gb.index
            );
        }
        let part = if is_i {
            if raw > 25 {
                return Err(Error::InvalidData(format!("mb_type {raw} too large")));
            }
            Part::Intra(raw as usize)
        } else if raw < 5 {
            match raw {
                0 => Part::P16x16,
                1 => Part::P16x8,
                2 => Part::P8x16,
                _ => {
                    // 3 = P_8x8, 4 = P_8x8ref0 (no ref_idx; all ref 0)
                    self.p8x8_ref0 = raw == 4;
                    Part::P8x8
                }
            }
        } else {
            if raw - 5 > 25 {
                return Err(Error::InvalidData("mb_type too large".into()));
            }
            Part::Intra((raw - 5) as usize)
        };

        if std::env::var_os("H264_DUMP").is_some() {
            eprintln!(
                "PMB {}:{} skip_run={} pos={}",
                self.mb_x, self.mb_y, self.mb_skip_run, gb.index
            );
        }
        match part {
            Part::Intra(row) => {
                let row = row as usize;
                let (mbt, cbp, pred) = (
                    I_MB_TYPE_INFO[row * 3],
                    I_MB_TYPE_INFO[row * 3 + 1],
                    I_MB_TYPE_INFO[row * 3 + 2] as i32,
                );
                if std::env::var_os("H264_DUMP").is_some() {
                    eprintln!(
                        "  TBL row={row} t={} cbp={} pred={pred}",
                        I_MB_TYPE_INFO[row * 3],
                        I_MB_TYPE_INFO[row * 3 + 1],
                    );
                }
                // C type codes → port-internal (0=4x4→1, 1=16x16→2,
                // 25=PCM→3); cbp 255 = C's -1 (only 16x16 cbp implied).
                let mbt = match mbt {
                    0 => MB_INTRA4X4,
                    25 => MB_PCM,
                    _ => MB_INTRA16X16,
                };
                self.mb_type = mbt as u32;
                self.cbp = if cbp == 255 { u32::MAX } else { cbp as u32 }; // 255 = -1 (none)
                // The table's pred column is in C's 16x16 namespace
                // (pred16x16[] indexing: 0=DC, 1=H, 2=V, 3=plane —
                // h264pred.h PRED8x8 codes), NOT this port's
                // (0=V, 1=H, 2=DC, 3=plane): remap (black/gray never
                // noticed — VERT and DC coincide at 128 when both
                // neighbors are unavailable).
                self.intra16x16_pred_mode = match pred {
                    0 => 2, // C DC → port DC
                    1 => 1, // H
                    2 => 0, // C VERT → port VERT
                    _ => 3, // plane
                };
                self.decode_mb_intra(gb, mb_xy)?;
            }
            Part::P16x16 | Part::P16x8 | Part::P8x16 | Part::P8x8 => {
                self.mb_type = MB_INTER;
                self.cbp = 0;
                self.decode_mb_inter(gb, mb_xy, &part)?;
            }
        }

        self.record_mb(mb_xy);
        self.hl_decode_mb(mb_xy)?;
        Ok(())
    }

    /// The intra MB path of `ff_h264_decode_mb_cavlc` (cavlc.c:784-831
    /// + the residual tail).
    fn decode_mb_intra(&mut self, gb: &mut Gb, mb_xy: usize) -> Result<()> {
        if self.mb_type == MB_PCM {
            // IS_INTRA_PCM: byte-aligned 384 samples (cavlc.c:759).
            gb.align();
            if gb.left() < 384 * 8 {
                return Err(Error::InvalidData("not enough data for intra PCM".into()));
            }
            self.intra_pcm.clear();
            for _ in 0..384 {
                self.intra_pcm.push(gb.read(8) as u8);
            }
            let pic = self.cur.as_mut().unwrap();
            pic.nnz[mb_xy] = [16; 48];
            pic.mb_type[mb_xy] = self.mb_type;
            // C stores qscale_table[mb_xy] = 0 (record_mb) but leaves the
            // slice's running QP alone for the MBs that follow.
            return Ok(());
        }

        self.fill_decode_caches(self.mb_type);

        if self.mb_type == MB_INTRA4X4 {
            for i in 0..16usize {
                let mut mode = self.pred_intra_mode(i);
                if gb.read_bit() == 0 {
                    let rem = gb.read(3) as i8;
                    mode = rem + (rem >= mode) as i8;
                }
                self.intra4x4_pred_mode_cache[SCAN8[i]] = mode;
            }
            self.write_back_intra_pred_mode(mb_xy);
            self.check_intra4x4_pred_mode()?;
        } else {
            // intra16x16: the pred mode came from the mb_type table
            self.intra16x16_pred_mode =
                self.check_intra_pred_mode(self.intra16x16_pred_mode, false)?;
        }
        let pos_before_cp = gb.index;
        let cmode = gb.ue()? as i32;
        let pos_after_cp = gb.index;
        self.chroma_pred_mode = self.check_intra_pred_mode(cmode, true)?;

        // cbp: the I-table's -1 (255) means "read from bitstream" for
        // non-16x16 intra (cavlc.c:1053-1064).
        if self.cbp == u32::MAX {
            let mut cbp = gb.ue()?;
            if cbp > 47 {
                return Err(Error::InvalidData("cbp too large".into()));
            }
            cbp = GOLOMB_TO_INTRA4X4_CBP[cbp as usize] as u32;
            self.cbp = cbp;
        }

        // residual
        if std::env::var_os("H264_DUMP").is_some() {
            eprintln!(
                "MB {}:{} t={} i16={} cpraw={} cp={} cbp={:#04x} q={} pcp={} pacp={} pos={}",
                self.mb_x,
                self.mb_y,
                self.mb_type,
                self.intra16x16_pred_mode,
                cmode,
                self.chroma_pred_mode,
                self.cbp,
                self.qscale,
                pos_before_cp,
                pos_after_cp,
                gb.index
            );
        }
        self.decode_mb_residual(gb, mb_xy)?;
        let pic = self.cur.as_mut().unwrap();
        pic.mb_type[mb_xy] = self.mb_type;
        Ok(())
    }

    /// The inter MB path (cavlc.c:832-1082 — the subset without B,
    /// weighting, MBAFF; single reference ⇒ no ref_idx reads when
    /// ref_count == 1, which the baseline single-ref DPB always is).
    /// ref_idx → picture id. C's ref_cache lives in ref2frm space and
    /// pred_motion's `r` comes FROM the cache; this port uses picture ids
    /// for that one space (borders are seeded from `ref_pic` ids), so the
    /// per-partition ref must be converted before any compare.
    fn ref_frm(&self, ri: i8) -> i8 {
        if ri < 0 {
            return -1; // LIST_NOT_USED
        }
        self.ref_list
            .get(ri as usize)
            .and_then(|&k| self.refs.get(k))
            .map_or(-1, |pic| pic.id as i8)
    }

    fn decode_mb_inter(&mut self, gb: &mut Gb, mb_xy: usize, part: &Part) -> Result<()> {
        self.cur_part = match part {
            Part::P16x8 => PART_16X8,
            Part::P8x16 => PART_8X16,
            Part::P8x8 => PART_8X8,
            _ => PART_16X16,
        };
        self.fill_decode_caches(MB_INTER);
        // ref_idx_l0 = te(v) (h264_cavlc.c:944): absent for one active
        // ref, one inverted bit for two, ue otherwise. C reads ALL of an
        // MB's ref_idx before any mvd and fills ref_cache with them —
        // pred_motion of later partitions compares against these refs.
        let rc = self.ref_count_l0;
        let read_ref = |gb: &mut Gb| -> Result<i8> {
            match rc {
                0 | 1 => Ok(0),
                2 => Ok((gb.read_bit() ^ 1) as i8),
                _ => {
                    let v = gb.ue()?;
                    if v >= rc {
                        return Err(Error::InvalidData(format!("ref {v} overflow")));
                    }
                    Ok(v as i8)
                }
            }
        };
        let set_ref = |cache: &mut [i8], x: usize, y: usize, w: usize, h: usize, r: i8| {
            for yy in 0..h {
                for xx in 0..w {
                    cache[SCAN8[0] + 8 * (y + yy) + (x + xx)] = r;
                }
            }
        };

        let read_mvd = |gb: &mut Gb| -> Result<(i16, i16)> {
            let dx = gb.se()?;
            let dy = gb.se()?;
            Ok((dx as i16, dy as i16))
        };
        match part {
            Part::P16x16 => {
                let r0 = read_ref(gb)?;
                let f0 = self.ref_frm(r0);
                set_ref(&mut self.ref_cache, 0, 0, 4, 4, f0);
                let (mx, my) = self.pred_motion(0, 4, f0);
                let (dx, dy) = read_mvd(gb)?;
                if std::env::var_os("H264_DUMP").is_some()
                    && ((self.slice_type_nos == 0 && self.mb_y == 0)
                        || (self.mb_y == 7 && self.mb_x <= 1 && self.slice_frame_num == 5))
                {
                    eprintln!(
                        "  MV16 mb={}:{} pred=({mx},{my}) mvd=({dx},{dy})",
                        self.mb_x, self.mb_y
                    );
                }
                let (mx, my) = (mx + dx, my + dy);
                self.fill_mv_rect(0, 0, 4, 4, mx, my);
            }
            Part::P16x8 => {
                let mut r = [0i8; 2];
                for (n, rn) in r.iter_mut().enumerate() {
                    *rn = self.ref_frm(read_ref(gb)?);
                    set_ref(&mut self.ref_cache, 0, 2 * n, 4, 2, *rn);
                }
                for n in 0..2usize {
                    let (mx, my) = self.pred_16x8_motion(n, r[n]);
                    let (dx, dy) = read_mvd(gb)?;
                    let (mx, my) = (mx + dx, my + dy);
                    self.fill_mv_rect(0, 2 * n, 4, 2, mx, my);
                }
            }
            Part::P8x16 => {
                let mut r = [0i8; 2];
                for (n, rn) in r.iter_mut().enumerate() {
                    *rn = self.ref_frm(read_ref(gb)?);
                    set_ref(&mut self.ref_cache, 2 * n, 0, 2, 4, *rn);
                }
                for n in 0..2usize {
                    let (mx, my) = self.pred_8x16_motion(n, r[n]);
                    let (dx, dy) = read_mvd(gb)?;
                    let (mx, my) = (mx + dx, my + dy);
                    self.fill_mv_rect(2 * n, 0, 2, 4, mx, my);
                }
            }
            Part::P8x8 | Part::Intra(_) => {
                // C (h264_cavlc.c:853): ALL four sub_mb_types first, then
                // ref_idx (none here: ref0 / single ref), THEN the mvds.
                let mut subs = [0u32; 4];
                for sub in subs.iter_mut() {
                    *sub = gb.ue()?;
                    if *sub > 3 {
                        return Err(Error::InvalidData("P sub_mb_type out of range".into()));
                    }
                }
                // ref_idx per 8x8 (quadrant i: x = 2*(i&1), y = 2*(i>>1)),
                // skipped for P_8x8ref0.
                let mut refs8 = [0i8; 4];
                for (i, ri) in refs8.iter_mut().enumerate() {
                    let raw = if self.p8x8_ref0 { 0 } else { read_ref(gb)? };
                    *ri = self.ref_frm(raw);
                    set_ref(&mut self.ref_cache, 2 * (i & 1), 2 * (i >> 1), 2, 2, *ri);
                }
                for (i, &sub) in subs.iter().enumerate() {
                    // sub: 0=sub8x8(1 part), 1=sub8x4(2), 2=sub4x8(2), 3=sub4x4(4)
                    let (bw, bh, count) = match sub {
                        0 => (2usize, 2usize, 1usize),
                        1 => (2, 1, 2),
                        2 => (1, 2, 2),
                        _ => (1, 1, 4),
                    };
                    for j in 0..count {
                        // C index (scan8 quadrant order) → grid position
                        let block = 4 * i + bw * j;
                        let dbgN = std::env::var_os("H264_DUMP").is_some()
                            && self.mb_y == 7
                            && self.mb_x <= 1
                            && self.slice_frame_num == 5;
                        if dbgN {
                            let idx = SCAN8[block];
                            eprintln!(
                                "  NB blk{block} idx{idx} L=({},{}) rL={} T=({},{}) rT={} C=({},{}) rC={}",
                                self.mv_cache[idx - 1][0],
                                self.mv_cache[idx - 1][1],
                                self.ref_cache[idx - 1],
                                self.mv_cache[idx - 8][0],
                                self.mv_cache[idx - 8][1],
                                self.ref_cache[idx - 8],
                                self.mv_cache[idx - 8 + bw][0],
                                self.mv_cache[idx - 8 + bw][1],
                                self.ref_cache[idx - 8 + bw]
                            );
                        }
                        let (mx, my) = self.pred_motion(block, bw, refs8[i]);
                        let (dx, dy) = read_mvd(gb)?;
                        if dbgN {
                            eprintln!(
                                "  SUBMB mb=1:7 sub{i} j{j} blk{block} sub{sub} bw{bw} ref={} pred=({mx},{my}) mvd=({dx},{dy}) final=({},{})",
                                refs8[i],
                                mx + dx,
                                my + dy
                            );
                        }
                        if std::env::var_os("H264_DUMP").is_some()
                            && self.slice_type_nos == 0
                            && self.mb_y == 0
                        {
                            eprintln!(
                                "  MV8 mb={}:{} sub{i} blk{block} pred=({mx},{my}) mvd=({dx},{dy})",
                                self.mb_x, self.mb_y
                            );
                        }
                        let (mx, my) = (mx + dx, my + dy);
                        let g = SCAN8[block];
                        self.fill_mv_rect((g & 7) - 4, (g >> 3) - 1, bw, bh, mx, my);
                    }
                }
            }
        }
        self.write_back_motion(mb_xy);

        // cbp
        let mut cbp = gb.ue()?;
        if cbp > 47 {
            return Err(Error::InvalidData("cbp too large".into()));
        }
        cbp = GOLOMB_TO_INTER_CBP[cbp as usize] as u32;
        self.cbp = cbp;
        if std::env::var_os("H264_DUMP").is_some() {
            eprintln!(
                "  IEND mb={}:{} cbp={:#04x} pos={}",
                self.mb_x, self.mb_y, cbp, gb.index
            );
        }

        self.decode_mb_residual(gb, mb_xy)?;
        if std::env::var_os("H264_DUMP").is_some()
            && self.mb_x == 1
            && self.mb_y == 7
            && self.slice_frame_num == 5
        {
            eprintln!(
                "  RESIDPOST mb=1:7 blocks8..11={:?}",
                &self.mb[8 * 16..12 * 16]
            );
        }
        let pic = self.cur.as_mut().unwrap();
        pic.mb_type[mb_xy] = self.mb_type;
        Ok(())
    }

    /// Fill an mv rectangle in the cache (fill_rectangle on mv_cache).
    fn fill_mv_rect(&mut self, x: usize, y: usize, w: usize, h: usize, mx: i16, my: i16) {
        for r in 0..h {
            for c in 0..w {
                // (x, y) are 4x4-grid coords; cache is 8 wide with the
                // MB's top-left cell at SCAN8[0]. (Raster→SCAN8 would
                // mis-place partitions: SCAN8 is quadrant-ordered.)
                self.mv_cache[SCAN8[0] + 8 * (y + r) + (x + c)] = [mx, my];
            }
        }
    }

    /// `decode_mb_skip` (h264_mvpred.h:950): P_Skip = zero-out + pskip mv.
    fn decode_mb_skip(&mut self, mb_xy: usize) {
        self.fill_decode_caches(MB_INTER);
        // hl_decode_mb reads self.mb_type (C reads
        // cur_pic.mb_type[mb_xy]); set it or the stale intra type from
        // the previous MB leaks into the skip reconstruction.
        self.mb_type = MB_INTER;
        // C: a skipped MB carries no residual (cbp 0, all nnz 0) — without
        // this, hl_decode_mb re-adds the PREVIOUS MB's stale coefficients.
        self.cbp = 0;
        for i in 0..48usize {
            self.nnz_cache[SCAN8[i]] = 0;
        }
        self.pred_pskip_motion();
        for i in 0..16usize {
            self.ref_cache[SCAN8[i]] = 0;
        }
        self.write_back_motion(mb_xy);
        let pic = self.cur.as_mut().unwrap();
        pic.mb_type[mb_xy] = MB_INTER;
        pic.nnz[mb_xy] = [0; 48];
        self.cur_part = PART_16X16;
        self.record_mb(mb_xy);
        self.prev_mb_skipped = true;
    }

    /// The residual tail of `ff_h264_decode_mb_cavlc` (cavlc.c:1087-1172):
    /// mb_qp_delta + luma/chroma residuals into `self.mb`.
    fn decode_mb_residual(&mut self, gb: &mut Gb, mb_xy: usize) -> Result<()> {
        let cv = cavlc();
        if self.cbp != u32::MAX && (self.cbp != 0 || self.mb_type == MB_INTRA16X16) {
            let dq = gb.se()?;
            self.qscale += dq;
            if self.qscale < 0 {
                self.qscale += 52;
            } else if self.qscale > 51 {
                self.qscale -= 52;
            }
            if !(0..=51).contains(&self.qscale) {
                return Err(Error::InvalidData("dquant out of range".into()));
            }
            let off = self.pps.as_ref().map(|p| p.chroma_qp_offset).unwrap_or(0);
            let cqp = CHROMA_QP8[(self.qscale + off).clamp(0, 51) as usize] as i32;
            self.chroma_qp[0] = cqp;
            self.chroma_qp[1] = cqp;
        }
        self.mb = [0; 48 * 16];

        let scan: [u8; 16] = ZIGZAG;
        if std::env::var_os("H264_DUMP").is_some()
            && self.mb_x == 1
            && self.mb_y == 7
            && self.slice_frame_num == 5
        {
            eprintln!(
                "  RESID mb=1:7 cbp={:#04x} q={} blocks8..11={:?}",
                self.cbp,
                self.qscale,
                &self.mb[8 * 16..12 * 16]
            );
        }
        // Luma DC (intra16x16)
        if self.mb_type == MB_INTRA16X16 {
            self.mb_luma_dc = [0; 16];
            let mut dc = self.mb_luma_dc;
            decode_residual(self, &cv, gb, &mut dc, LUMA_DC, &scan, None, 16)?;
            if std::env::var_os("H264_DUMP").is_some() {
                eprintln!("  LUMADC {:?}", &dc[..8]);
            }
            self.mb_luma_dc = dc;
            if self.cbp & 15 != 0 {
                let qm = *self.pps.as_ref().unwrap().dequant(0, self.qscale as usize);
                for i in 0..16usize {
                    let mut blk = [0i16; 16];
                    decode_residual(self, &cv, gb, &mut blk, i, &scan_shift1(), Some(&qm), 15)?;
                    // scan+1 path: positions 1..15 only (C decodes in
                    // place; the DC slot is written by the later scatter).
                    self.mb[i * 16 + 1..(i + 1) * 16].copy_from_slice(&blk[1..16]);
                }
            } else {
                for i in 0..16usize {
                    self.nnz_cache[SCAN8[i]] = 0;
                }
            }
        } else if self.cbp & 15 != 0 {
            // C's cqm for luma non-16x16: (IS_INTRA ? 0 : 3) + p.
            let cqm = if self.mb_type == MB_INTRA4X4 || self.mb_type == MB_PCM {
                0
            } else {
                3
            };
            let qm = *self
                .pps
                .as_ref()
                .unwrap()
                .dequant(cqm, self.qscale as usize);
            for i8x8 in 0..4usize {
                if self.cbp & (1 << i8x8) != 0 {
                    for i4x4 in 0..4usize {
                        let index = i4x4 + 4 * i8x8;
                        let mut blk = [0i16; 16];
                        decode_residual(self, &cv, gb, &mut blk, index, &scan, Some(&qm), 16)?;
                        self.mb[index * 16..(index + 1) * 16].copy_from_slice(&blk);
                    }
                } else {
                    for i4x4 in 0..4usize {
                        self.nnz_cache[SCAN8[4 * i8x8 + i4x4]] = 0;
                    }
                }
            }
        } else {
            for i in 0..16usize {
                self.nnz_cache[SCAN8[i]] = 0;
            }
        }

        // Chroma DC + AC
        if self.cbp != u32::MAX {
            if self.cbp & 0x30 != 0 {
                for ch in 0..2usize {
                    // C routes chroma DC through the SAME decode_residual
                    // (max_coeff 4 picks the chroma DC VLCs), storing raw
                    // levels at ff_h264_chroma_dc_scan = {0,16,32,48} into
                    // the 4-block chroma region of mb.
                    let mut scan_cdc = [0u8; 16];
                    scan_cdc[..4].copy_from_slice(&CHROMA_DC_SCAN);
                    let mut dc64 = [0i16; 64];
                    decode_residual(self, &cv, gb, &mut dc64, CHROMA_DC + ch, &scan_cdc, None, 4)?;
                    // C's block pointer is mb + 16*(16+16*ch); the scan put
                    // the four coefficients at slots 0/16/32/48 of that.
                    let base = 16 * (16 + 16 * ch);
                    for s in [0usize, 16, 32, 48] {
                        self.mb[base + s] = dc64[s];
                    }
                }
            }
            if self.cbp & 0x20 != 0 {
                for ch in 0..2usize {
                    // C's chroma AC qmul set: chroma_idx+1 + (IS_INTRA?0:3).
                    let set = ch
                        + 1
                        + if self.mb_type == MB_INTRA4X4
                            || self.mb_type == MB_INTRA16X16
                            || self.mb_type == MB_PCM
                        {
                            0
                        } else {
                            3
                        };
                    let qm = *self
                        .pps
                        .as_ref()
                        .unwrap()
                        .dequant(set, self.chroma_qp[ch] as usize);
                    // 4:2:0: one 8x8 per component (num_c8x8 = 1)
                    for i8 in 0..1usize {
                        for i4 in 0..4usize {
                            // Block index for nnz/SCAN8 (0..47); the mb
                            // offset C uses is 16*(16+16*ch) + 16*block
                            // (`+ i4` would scatter all four blocks into
                            // block 0's first bytes).
                            let block_idx = 16 + 16 * ch + i4;
                            let index = 16 * (16 + 16 * ch) + 16 * i4;
                            let mut blk = [0i16; 16];
                            decode_residual(
                                self,
                                &cv,
                                gb,
                                &mut blk,
                                block_idx,
                                &scan_shift1(),
                                Some(&qm),
                                15,
                            )?;
                            // C decodes ACs IN PLACE (scan+1 writes positions
                            // 1..15 only, preserving the DC slots that the
                            // chroma-DC decode left at {0,16,32,48}) — copy
                            // positions 1..16, never clobbering blk[0].
                            self.mb[index + 1..index + 16].copy_from_slice(&blk[1..16]);
                        }
                    }
                }
            } else {
                for ch in 0..2usize {
                    for i in 0..4usize {
                        self.nnz_cache[SCAN8[16 + 16 * ch + i]] = 0;
                        self.nnz_cache[SCAN8[20 + 16 * ch + i]] = 0;
                    }
                }
            }
        }

        self.write_back_non_zero_count(mb_xy);
        Ok(())
    }

    // ---------------- Reconstruction (h264_mb_template.c) ----------------

    /// `hl_decode_mb`: predict + add residual for one MB (or copy PCM /
    /// MC for inter). Loops the loop filter — not ported.
    fn hl_decode_mb(&mut self, mb_xy: usize) -> Result<()> {
        let w = self.mb_width * 16;
        let y0 = self.mb_y * 16 * w + self.mb_x * 16;
        let c0 = self.mb_y * 8 * (w / 2) + self.mb_x * 8;

        if self.mb_type == MB_PCM {
            let pic = self.cur.as_mut().unwrap();
            for r in 0..16 {
                for c in 0..16 {
                    pic.y[y0 + r * w + c] = self.intra_pcm[r * 16 + c];
                }
            }
            for r in 0..8 {
                for c in 0..8 {
                    pic.cb[c0 + r * (w / 2) + c] = self.intra_pcm[256 + r * 8 + c];
                    pic.cr[c0 + r * (w / 2) + c] = self.intra_pcm[320 + r * 8 + c];
                }
            }
            return Ok(());
        }

        if self.mb_type == MB_INTER {
            // MC per 4x4 block: ref_cache → RefPicList0 → DPB picture.
            let refs = &self.refs;
            let ref_list = &self.ref_list;
            let pick = |ri: i8| -> Result<&Picture> {
                ref_list
                    .get(ri.max(0) as usize)
                    .and_then(|&k| refs.get(k))
                    .ok_or_else(|| {
                        Error::InvalidData("inter MB references a missing picture".into())
                    })
            };
            let cur = self.cur.as_mut().unwrap();
            for i in 0..16usize {
                let mv = self.mv_cache[SCAN8[i]];
                let prev = pick(self.ref_cache[SCAN8[i]])?;
                // scan8 (quadrant) order → pixel position, C's block_offset.
                let grid = SCAN8[i];
                let bc = ((grid & 7) - 4) as i32;
                let br = ((grid >> 3) - 1) as i32;
                let bx = self.mb_x as i32 * 16 + 4 * bc;
                let by = self.mb_y as i32 * 16 + 4 * br;
                // luma
                let mut luma = [0u8; 16];
                mc_luma(&mut luma, 4, 4, 4, prev, bx, by, mv[0], mv[1]);
                for dr in 0..4usize {
                    for dc in 0..4usize {
                        cur.y[y0
                            + (4 * br + dr as i32) as usize * w
                            + (4 * bc + dc as i32) as usize] = luma[dr * 4 + dc];
                    }
                }
            }
            for r in 0..4usize {
                for c in 0..4usize {
                    // raster block (r, c) ↔ cache cell SCAN8[0] + 8r + c
                    let mv = self.mv_cache[SCAN8[0] + 8 * r + c];
                    let prev = pick(self.ref_cache[SCAN8[0] + 8 * r + c])?;
                    let bx = self.mb_x as i32 * 16 + 4 * c as i32;
                    let by = self.mb_y as i32 * 16 + 4 * r as i32;
                    let mut cb = [0u8; 16];
                    let mut cr = [0u8; 16];
                    mc_chroma(&mut cb, 2, 2, 2, prev, &PlaneSel::Cb, bx, by, mv[0], mv[1]);
                    mc_chroma(&mut cr, 2, 2, 2, prev, &PlaneSel::Cr, bx, by, mv[0], mv[1]);
                    let cur = self.cur.as_mut().unwrap();
                    for dr in 0..2 {
                        for dc in 0..2 {
                            cur.cb[c0 + (2 * r + dr) * (w / 2) + 2 * c + dc] = cb[dr * 2 + dc];
                            cur.cr[c0 + (2 * r + dr) * (w / 2) + 2 * c + dc] = cr[dr * 2 + dc];
                        }
                    }
                }
            }
            // residual (if coded): add over the MC'd area
            if self.cbp & 0x3f != 0 {
                self.add_residual(y0, c0, w);
            }
            return Ok(());
        }

        // ---- intra ----
        {
            // chroma prediction first (C order: chroma, then luma)
            let (top_ok, left_ok) = (self.n_top != 0, self.n_left != 0);
            let pic = self.cur.as_ref().unwrap();
            let cw = w / 2;
            let mut top = [0u8; 8];
            let mut left = [0u8; 8];
            let mut lt_cb = 128u8;
            let mut lt_cr = 128u8;
            if top_ok {
                for i in 0..8 {
                    top[i] = pic.cb[c0 - cw + i];
                }
            }
            if left_ok {
                for i in 0..8 {
                    left[i] = pic.cb[c0 + i * cw - 1];
                }
            }
            if top_ok && left_ok {
                lt_cb = pic.cb[c0 - cw - 1];
            }
            let mode = self.chroma_pred_mode;
            let pic = self.cur.as_mut().unwrap();
            let lb_cb = if left_ok && self.mb_y + 1 <= self.mb_height {
                pic.cb[c0 + cw - 1]
            } else {
                lt_cb
            };
            pred8x8_avail(
                mode,
                &mut pic.cb[c0..],
                cw,
                &top,
                &left,
                top_ok,
                left_ok,
                lt_cb,
                lb_cb,
            );
            if top_ok {
                for i in 0..8 {
                    top[i] = pic.cr[c0 - cw + i];
                }
            }
            if left_ok {
                for i in 0..8 {
                    left[i] = pic.cr[c0 + i * cw - 1];
                }
            }
            if top_ok && left_ok {
                lt_cr = pic.cr[c0 - cw - 1];
            }
            let lb_cr = if left_ok && self.mb_y + 1 <= self.mb_height {
                pic.cr[c0 + cw - 1]
            } else {
                lt_cr
            };
            pred8x8_avail(
                mode,
                &mut pic.cr[c0..],
                cw,
                &top,
                &left,
                top_ok,
                left_ok,
                lt_cr,
                lb_cr,
            );
        }

        if self.mb_type == MB_INTRA16X16 {
            let pic = self.cur.as_ref().unwrap();
            let mut top = [0u8; 16];
            let mut left = [0u8; 16];
            let top_ok = self.n_top != 0;
            let left_ok = self.n_left != 0;
            if top_ok {
                for i in 0..16 {
                    top[i] = pic.y[y0 - w + i];
                }
            }
            if left_ok {
                for i in 0..16 {
                    left[i] = pic.y[y0 + i * w - 1];
                }
            }
            let mode = self.intra16x16_pred_mode;
            let lt = if top_ok && left_ok {
                pic.y[y0 - w - 1]
            } else {
                128
            };
            let pic = self.cur.as_mut().unwrap();
            let mut mb_dst = [0u8; 256];
            pred16x16_avail(mode, &mut mb_dst, &top, &left, top_ok, left_ok, lt);
            for r in 0..16 {
                for c in 0..16 {
                    pic.y[y0 + r * w + c] = mb_dst[r * 16 + c];
                }
            }
            // luma DC hadamard scatter into self.mb, then IDCT each block
            if std::env::var_os("H264_DUMP").is_some() && self.mb_x == 0 && self.mb_y == 0 {
                eprintln!(
                    "  PRE-SCATTER nnzDC={} qmul={} scattered0..1 will follow",
                    self.nnz_cache[SCAN8[LUMA_DC]],
                    self.pps.as_ref().unwrap().dequant(0, self.qscale as usize)[0]
                );
            }
            if self.nnz_cache[SCAN8[LUMA_DC]] != 0 {
                let qmul = self.pps.as_ref().unwrap().dequant(0, self.qscale as usize)[0];
                // C writes value(loop-i, stride-slot k) at
                // output[16·k + x_off[i]] — i.e. into mb block
                // k + {0,2,8,10}[i] with k ∈ {0,1,4,5} (the function
                // writes straight into sl->mb). Map each transform output
                // to exactly that block; a per-(r,c) ROW/XOFF composition
                // lands on the transposed block for non-symmetric DC
                // matrices (uniform per-block pixel offsets).
                let mut scattered = [0i16; 256];
                let dc = self.mb_luma_dc;
                luma_dc_dequant_idct(&mut scattered, &dc, qmul);
                const XOFF: [usize; 4] = [0, 32, 128, 160];
                for k in [0usize, 1, 4, 5] {
                    for i in 0..4usize {
                        let block = k + [0usize, 2, 8, 10][i];
                        self.mb[block * 16] = scattered[16 * k + XOFF[i]];
                    }
                }
                if std::env::var_os("H264_DUMP").is_some() && self.mb_x == 0 && self.mb_y == 0 {
                    eprintln!(
                        "  POST-SCATTER mbDC0={} mbDC5={}",
                        self.mb[0],
                        self.mb[5 * 16]
                    );
                }
            }
            // residual AC: idct_add16intra semantics
            self.add_residual_intra16(y0, w);
            if std::env::var_os("H264_DUMP").is_some() && self.mb_x == 0 && self.mb_y == 0 {
                let pic = self.cur.as_ref().unwrap();
                eprintln!(
                    "  RECON16 y00={} y15,15={}",
                    pic.y[y0],
                    pic.y[y0 + 15 * w + 15]
                );
            }
            // chroma residual (C gates the whole section on cbp & 0x30;
            // idct_add8's DC-only branch needs it too)
            if self.cbp & 0x30 != 0 {
                self.apply_chroma_dc();
                self.apply_chroma_ac(c0, w);
            }
            return Ok(());
        }

        // intra4x4: predict each block from reconstructed pixels, add IDCT.
        // Block index i is in C's scan8 order — QUADRANT-based (blocks
        // 0-3 = top-left 8x8's 2x2), NOT raster: block 2 sits at (br1,bc0).
        // Pixel positions derive from the scan8 grid like C's block_offset
        // (h264_slice.c:557).
        for i in 0..16usize {
            let mode = self.intra4x4_pred_mode_cache[SCAN8[i]] as i32;
            let grid = SCAN8[i];
            let bx = ((grid & 7) - 4) * 4;
            let by = ((grid >> 3) - 1) * 4;
            let (br, bc) = ((grid >> 3) - 1, (grid & 7) - 4);
            let at = y0 + by * w + bx;
            let pic = self.cur.as_ref().unwrap();
            let mut top = [0u8; 8];
            let mut left = [0u8; 4];
            let mut lt = 0u8;
            // C reads i4x4 neighbors straight from the reconstructed
            // frame (no per-block gating — modes are pre-validated by
            // ff_h264_check_intra_pred_mode); availability reduces to
            // "does the source exist": within-MB blocks (br>0 / bc>0)
            // are already reconstructed (scan8 order guarantees the
            // up-left quadrant blocks exist before their right/below
            // neighbors).
            let top_ok = br > 0 || self.mb_y > 0;
            let left_ok = bc > 0 || self.mb_x > 0;
            // C's topright gate (h264_mb.c:678):
            // (topright_samples_available << i) & 0x8000 — else the
            // block's own top-right pixel is replicated 4x.
            let tr_ok = ((self.topright_samples_available << i) & 0x8000) != 0;
            if std::env::var_os("H264_DUMP").is_some() && (self.mb_y == 0 && self.mb_x == 0) {
                eprintln!(
                    "  I4 mb{}:{} blk{i} mode={mode} t={top_ok} l={left_ok} tr={tr_ok} trmask={:04x}",
                    self.mb_x, self.mb_y, self.topright_samples_available
                );
            }
            if top_ok {
                for k in 0..4 {
                    top[k] = pic.y[at - w + k];
                }
                if tr_ok {
                    for k in 0..4 {
                        top[4 + k] = pic.y[at - w + 4 + k];
                    }
                } else {
                    // C: topright = ptr[3 - linesize] replicated 4x
                    for k in 0..4 {
                        top[4 + k] = top[3];
                    }
                }
            } else {
                top = [128; 8];
            }
            if left_ok {
                for k in 0..4 {
                    left[k] = pic.y[at + k * w - 1];
                }
            } else {
                left = [128; 4];
            }
            lt = if top_ok && left_ok && (i % 4 == 0) {
                pic.y[at - w - 1]
            } else if top_ok && left_ok {
                pic.y[at - w - 1]
            } else {
                128
            };
            let pic = self.cur.as_mut().unwrap();
            let mut dst = [0u8; 16];
            pred4x4(mode as i8, &mut dst, &top, &left, lt);
            for r in 0..4 {
                for c in 0..4 {
                    pic.y[at + r * w + c] = dst[r * 4 + c];
                }
            }
            // residual
            let nnz = self.nnz_cache[SCAN8[i]] as usize;
            if nnz > 0 {
                let mut blk = [0i16; 16];
                blk.copy_from_slice(&self.mb[i * 16..(i + 1) * 16]);
                let base = at;
                let pic = self.cur.as_mut().unwrap();
                let mut dst4: [u8; 16] = {
                    let mut d = [0u8; 16];
                    for r in 0..4 {
                        for c in 0..4 {
                            d[r * 4 + c] = pic.y[base + r * w + c];
                        }
                    }
                    d
                };
                if nnz == 1 && blk[0] != 0 {
                    idct_dc_add(&mut dst4, 4, &mut blk);
                } else {
                    idct_add(&mut dst4, 4, &mut blk);
                }
                let pic = self.cur.as_mut().unwrap();
                for r in 0..4 {
                    for c in 0..4 {
                        pic.y[base + r * w + c] = dst4[r * 4 + c];
                    }
                }
            }
        }

        // chroma residual for intra4x4 (AC + DC)
        if self.cbp & 0x30 != 0 {
            self.apply_chroma_dc();
        }
        // C's idct_add8 runs for cbp & 0x30 (ANY chroma bit) — DC-only
        // blocks reach the pixels through its nnz==0/DC!=0 branch.
        if self.cbp & 0x30 != 0 {
            self.apply_chroma_ac(c0, w);
        }
        Ok(())
    }

    /// Add coded luma/chroma residual over an inter MB (idct_add16).
    fn add_residual(&mut self, y0: usize, c0: usize, w: usize) {
        for i in 0..16usize {
            if self.cbp & (1 << (i / 4)) == 0 && self.cbp & 15 != 0 {
                if self.cbp & (1 << (i / 4)) == 0 {
                    continue;
                }
            }
            if self.cbp & 15 == 0 {
                continue;
            }
            if self.cbp & (1 << (i / 4)) == 0 {
                continue;
            }
            let nnz = self.nnz_cache[SCAN8[i]] as usize;
            if nnz == 0 {
                continue;
            }
            let mut blk = [0i16; 16];
            blk.copy_from_slice(&self.mb[i * 16..(i + 1) * 16]);
            // scan8 (quadrant) order → pixel position, C's block_offset.
            let grid = SCAN8[i];
            let bx = ((grid & 7) - 4) * 4;
            let by = ((grid >> 3) - 1) * 4;
            let pic = self.cur.as_mut().unwrap();
            let base = y0 + by * w + bx;
            let mut d = [0u8; 16];
            for r in 0..4 {
                for c in 0..4 {
                    d[r * 4 + c] = pic.y[base + r * w + c];
                }
            }
            if nnz == 1 && blk[0] != 0 {
                idct_dc_add(&mut d, 4, &mut blk);
            } else {
                idct_add(&mut d, 4, &mut blk);
            }
            for r in 0..4 {
                for c in 0..4 {
                    pic.y[base + r * w + c] = d[r * 4 + c];
                }
            }
        }
        if self.cbp & 0x30 != 0 {
            self.apply_chroma_dc();
        }
        // C's idct_add8 runs for cbp & 0x30 (ANY chroma bit) — DC-only
        // blocks reach the pixels through its nnz==0/DC!=0 branch.
        if self.cbp & 0x30 != 0 {
            self.apply_chroma_ac(c0, w);
        }
    }

    /// idct_add16intra (h264_mb.c:756) over an intra16x16 MB.
    fn add_residual_intra16(&mut self, y0: usize, w: usize) {
        for i in 0..16usize {
            let nnz = self.nnz_cache[SCAN8[i]] as usize;
            if nnz == 0 && self.mb[i * 16] == 0 {
                continue;
            }
            let mut blk = [0i16; 16];
            blk.copy_from_slice(&self.mb[i * 16..(i + 1) * 16]);
            // scan8 (quadrant) order → pixel position, C's block_offset.
            let grid = SCAN8[i];
            let bx = ((grid & 7) - 4) * 4;
            let by = ((grid >> 3) - 1) * 4;
            let pic = self.cur.as_mut().unwrap();
            let base = y0 + by * w + bx;
            let mut d = [0u8; 16];
            for r in 0..4 {
                for c in 0..4 {
                    d[r * 4 + c] = pic.y[base + r * w + c];
                }
            }
            if nnz != 0 {
                idct_add(&mut d, 4, &mut blk);
            } else {
                // nnz == 0 with a scattered DC → C's idct_add16intra
                // idct_dc_add fast path.
                idct_dc_add(&mut d, 4, &mut blk);
            }
            for r in 0..4 {
                for c in 0..4 {
                    pic.y[base + r * w + c] = d[r * 4 + c];
                }
            }
        }
    }

    /// Chroma DC dequant-idct (h264_mb_template.c:236): gated on the DC
    /// block's nnz, in place over the shared 64-coeff chroma region — the
    /// four dequantized DCs land in each 4x4 block's [0] slot (the
    /// ff_h264_chroma_dc_scan positions {0,16,32,48}) and are spread over
    /// the pixels by apply_chroma_ac (idct_add8's branches). Dequant set:
    /// intra 1/2, inter 4/5.
    fn apply_chroma_dc(&mut self) {
        for ch in 0..2usize {
            if self.nnz_cache[SCAN8[CHROMA_DC + ch]] == 0 {
                continue;
            }
            let q = self.chroma_qp[ch] as usize;
            let intra = self.mb_type == MB_INTRA4X4
                || self.mb_type == MB_INTRA16X16
                || self.mb_type == MB_PCM;
            let set = if intra { ch + 1 } else { ch + 4 };
            let qmul = self.pps.as_ref().unwrap().dequant(set, q)[0];
            let base = 16 * (16 + 16 * ch);
            let mut block64 = [0i16; 64];
            block64.copy_from_slice(&self.mb[base..base + 64]);
            if std::env::var_os("H264_DUMP").is_some() && self.mb_x == 1 && self.mb_y == 0 {
                eprintln!(
                    "  CDC ch{ch} raw a,b,c,d = {},{},{},{} q={q} qmul={qmul}",
                    block64[0], block64[16], block64[32], block64[48]
                );
            }
            chroma_dc_dequant_idct(&mut block64, qmul);
            if std::env::var_os("H264_DUMP").is_some() && self.mb_x == 1 && self.mb_y == 0 {
                eprintln!(
                    "  CDC ch{ch} out {},{},{},{}",
                    block64[0], block64[16], block64[32], block64[48]
                );
            }
            self.mb[base..base + 64].copy_from_slice(&block64);
        }
    }

    /// Chroma residual apply = C's ff_h264_idct_add8 (h264idct_template.c):
    /// per 4x4 block, nnz != 0 → full idct (DC at [0] + AC); nnz == 0 with
    /// a scattered DC → idct_dc_add; else nothing.
    fn apply_chroma_ac(&mut self, c0: usize, w: usize) {
        let cw = w / 2;
        for ch in 0..2usize {
            for b in 0..4usize {
                let block_idx = 16 + 16 * ch + b;
                let nnz = self.nnz_cache[SCAN8[block_idx]] as usize;
                let mb_off = 16 * (16 + 16 * ch) + 16 * b;
                let mut blk = [0i16; 16];
                blk.copy_from_slice(&self.mb[mb_off..mb_off + 16]);
                if std::env::var_os("H264_DUMP").is_some() && self.mb_x == 0 && self.mb_y == 0 {
                    eprintln!("  CAC ch{ch} b{b} nnz={nnz} blk0..3={:?}", &blk[..8]);
                }
                if nnz == 0 && blk[0] == 0 {
                    continue;
                }
                let bx = (b % 2) * 4;
                let by = (b / 2) * 4;
                let pic = self.cur.as_mut().unwrap();
                let base = c0 + by * cw + bx;
                let plane: &mut [u8] = if ch == 0 { &mut pic.cb } else { &mut pic.cr };
                let mut d = [0u8; 16];
                for r in 0..4 {
                    for c in 0..4 {
                        d[r * 4 + c] = plane[base + r * cw + c];
                    }
                }
                if nnz != 0 {
                    idct_add(&mut d, 4, &mut blk);
                } else {
                    idct_dc_add(&mut d, 4, &mut blk);
                }
                for r in 0..4 {
                    for c in 0..4 {
                        plane[base + r * cw + c] = d[r * 4 + c];
                    }
                }
            }
        }
    }

    // ---------------- Picture / slice driver ----------------

    fn start_new_picture(&mut self, is_idr: bool) {
        // (the caller emitted the old picture via finish_picture, which
        // also moved it into `prev` — the single-ref DPB)
        self.cur = None;
        let mut pic = Picture::new(self.mb_width, self.mb_height);
        pic.id = self.next_pic_id;
        self.next_pic_id += 1;
        self.cur = Some(pic);
        self.slice_table = vec![0; self.mb_width * self.mb_height];
        self.got_mb = false;
        self.mb_skip_run = -1;
        if is_idr {
            self.refs.clear(); // IDR clears the DPB
        }
    }

    fn decode_slice(&mut self, nal: &Nal, next_slice_idx: usize) -> Result<bool> {
        let mut gb = Gb::new(&nal.rbsp);
        let (first_mb, new_pic) = self.parse_slice_header(nal, &mut gb).map_err(|e| {
            eprintln!(
                "H264DBG: slice hdr err kind={} ref_idc={} len={} : {e}",
                nal.kind,
                nal.ref_idc,
                nal.rbsp.len()
            );
            e
        })?;
        if std::env::var_os("H264_DUMP").is_some() {
            eprintln!(
                "H264DBG: slice kind={} first_mb={first_mb} new_pic={new_pic} hdr_bits={} left={}",
                nal.kind,
                gb.index,
                gb.left()
            );
        }
        if new_pic {
            self.finish_picture()?; // emit the finished picture first
            self.start_new_picture(nal.kind == 5);
            self.cur_is_ref = nal.ref_idc != 0;
            self.frame_num = self.slice_frame_num;
            self.cur.as_mut().unwrap().frame_num = self.slice_frame_num;
        }
        // One number per slice (C's current_slice / slice_table): MBs of
        // other slices are unavailable for prediction, and deblocking mode 2
        // stops at slice edges.
        self.slice_num += 1;
        self.slice_dbk.slice = self.slice_num;
        self.build_ref_list()?;
        // (frame_num kept from header for next comparison)
        self.got_mb = true;

        let mb_num = self.mb_width * self.mb_height;
        let mut mb_abs = first_mb;
        // C's CAVLC slice loop (h264_slice.c:2784): decode MB, advance,
        // finish at end-of-picture; stop on exhausted bits ONLY when no
        // skip run is pending — an all-skip slice has zero bits left for
        // the remaining MBs (skip MBs consume no bits), so more_rbsp_data
        // must NOT gate the loop. C also stops each slice at
        // next_slice_idx (ff_h264_execute_decode_slices): the first MB of
        // the following slice — a slice's rbsp trailing bits leave a few
        // readable bits, so bit exhaustion alone would overread into the
        // padding (multi-slice pictures).
        loop {
            if mb_abs >= next_slice_idx {
                break;
            }
            self.mb_x = mb_abs % self.mb_width;
            self.mb_y = mb_abs / self.mb_width;
            self.decode_mb_cavlc(&mut gb).map_err(|e| {
                eprintln!(
                    "H264DBG: MB decode err slice#{} kind={} first_mb={first_mb} at mb({},{}) bits={}: {e}",
                    self.slice_num,
                    nal.kind,
                    self.mb_x,
                    self.mb_y,
                    gb.index
                );
                e
            })?;
            mb_abs += 1;
            if mb_abs >= mb_num {
                break;
            }
            if gb.left() <= 0 && self.mb_skip_run <= 0 {
                if gb.left() == 0 {
                    break;
                }
                return Err(Error::InvalidData("slice overread".into()));
            }
        }
        Ok(true)
    }

    /// Emit the finished picture as an output Frame (cropped).
    fn finish_picture(&mut self) -> Result<()> {
        let mut pic = match self.cur.take() {
            Some(p) => p,
            None => return Ok(()),
        };
        // In-loop deblocking: the filtered picture is both the output and
        // the reference for later P slices.
        deblock::filter_picture(&mut pic, self.mb_width, self.mb_height);
        // Sliding-window short-term marking (h264_refs.c, no MMCO): the
        // newest reference goes first; the oldest drops past
        // max_num_ref_frames. A non-reference picture is output only.
        let max_refs = self
            .sps
            .as_ref()
            .map(|s| s.ref_frame_count.max(1))
            .unwrap_or(1) as usize;
        let keep = self.cur_is_ref;
        self.refs.insert(0, pic);
        let pic_idx = 0usize;
        let pic = &self.refs[pic_idx];

        let sps = self.sps.as_ref().unwrap();
        let w = self.mb_width * 16;
        let h = self.mb_height * 16;
        let ow = (w - sps.crop_left as usize - sps.crop_right as usize).min(w);
        let oh = (h - sps.crop_top as usize - sps.crop_bottom as usize).min(h);
        let mut frame = Frame::alloc(PixelFormat::Yuv420p, ow as u32, oh as u32)?;
        let ls = frame.linesize(0);
        let cs = frame.linesize(1);
        for r in 0..oh {
            let src = sps.crop_top as usize * w + sps.crop_left as usize + r * w;
            frame.plane_mut(0)[r * ls..r * ls + ow].copy_from_slice(&pic.y[src..src + ow]);
        }
        for r in 0..oh / 2 {
            let src =
                sps.crop_top as usize / 2 * (w / 2) + sps.crop_left as usize / 2 + r * (w / 2);
            frame.plane_mut(1)[r * cs..r * cs + ow / 2].copy_from_slice(&pic.cb[src..src + ow / 2]);
            frame.plane_mut(2)[r * cs..r * cs + ow / 2].copy_from_slice(&pic.cr[src..src + ow / 2]);
        }
        self.frame_count += 1;
        self.cur = None; // picture moved into the DPB above
        if keep {
            self.refs.truncate(max_refs);
        } else {
            self.refs.remove(pic_idx);
        }
        self.pending.push_back(frame);
        self.got_mb = false;
        Ok(())
    }

    /// RefPicList0 init + modification for a P slice (h264_refs.c:
    /// ff_h264_build_ref_list, short-term only). Default order is
    /// descending PicNum (PicNum = FrameNumWrap: frame_num, minus
    /// MaxFrameNum when it exceeds the current frame_num); the
    /// modification ops then move the named picture to each index.
    fn build_ref_list(&mut self) -> Result<()> {
        self.ref_list.clear();
        if self.slice_type_nos == 2 {
            return Ok(());
        }
        let max_fn = 1i64 << self.sps.as_ref().unwrap().log2_max_frame_num;
        let cur_fn = self.slice_frame_num as i64;
        let pic_num = |f: u32| -> i64 {
            let f = f as i64;
            if f > cur_fn { f - max_fn } else { f }
        };
        let mut list: Vec<usize> = (0..self.refs.len()).collect();
        list.sort_by_key(|&i| std::cmp::Reverse(pic_num(self.refs[i].frame_num)));
        let mut pred = cur_fn;
        for (idx, &(op, val)) in self.reorder_ops.iter().enumerate() {
            let abs_diff = val as i64 + 1;
            if op == 0 {
                pred -= abs_diff;
                if pred < 0 {
                    pred += max_fn;
                }
            } else {
                pred += abs_diff;
                if pred >= max_fn {
                    pred -= max_fn;
                }
            }
            let want = if pred > cur_fn { pred - max_fn } else { pred };
            let Some(pos) = list
                .iter()
                .position(|&i| pic_num(self.refs[i].frame_num) == want)
            else {
                return Err(Error::InvalidData(
                    "reference picture missing during reorder".into(),
                ));
            };
            let r = list.remove(pos);
            list.insert(idx.min(list.len()), r);
        }
        list.truncate(self.ref_count_l0 as usize);
        self.ref_list = list;
        Ok(())
    }
}

impl Decoder for H264Decoder {
    fn init(&mut self, params: &CodecParameters) -> Result<()> {
        match params.codec_id {
            CodecId::H264 => {}
            other => {
                return Err(Error::Unsupported(format!(
                    "codec '{}' is not the H.264 decoder",
                    other.name()
                )));
            }
        }
        self.params = params.clone();
        self.params.codec_type = MediaType::Video;
        Ok(())
    }

    fn send_packet(&mut self, pkt: Option<&Packet>) -> Result<()> {
        let Some(pkt) = pkt else {
            self.eof = true;
            // finish the trailing picture
            if self.got_mb {
                self.finish_picture()?;
            }
            return Ok(());
        };
        if self.eof {
            return Err(Error::Eof);
        }
        self.pending.clear();

        let nals = split_nals(pkt.as_slice());
        // first_mb_in_slice is the slice header's very first ue(v); peek it
        // so each slice knows where the next one begins (C's next_slice_idx).
        let slice_first: Vec<usize> = nals
            .iter()
            .filter(|n| n.kind == 1 || n.kind == 5)
            .map(|n| {
                let mut gb = Gb::new(&n.rbsp);
                gb.ue().map(|v| v as usize).unwrap_or(usize::MAX)
            })
            .collect();
        let mut si = 0usize;
        for nal in &nals {
            match nal.kind {
                7 => {
                    self.sps = Some(parse_sps(&nal.rbsp)?);
                    if self.mb_width == 0 {
                        self.mb_width = self.sps.as_ref().unwrap().mb_width;
                        self.mb_height = self.sps.as_ref().unwrap().mb_height;
                    }
                }
                8 => {
                    self.pps = Some(parse_pps(&nal.rbsp, self.sps.is_some())?);
                }
                5 | 1 => {
                    // A slice whose successor starts at a later MB belongs
                    // to the same picture; otherwise the picture must end
                    // by itself (bit exhaustion).
                    let mb_num = self.mb_width * self.mb_height;
                    let next = slice_first
                        .get(si + 1)
                        .copied()
                        .filter(|&f| f > slice_first[si] && f <= mb_num)
                        .unwrap_or(mb_num);
                    self.decode_slice(nal, next)?;
                    si += 1;
                }
                _ => {} // SEI etc. skipped
            }
        }
        // Raw .h264 access units: finish the picture at the end of each
        // packet when it carries whole frames (the demuxer's shape).
        if self.got_mb {
            self.finish_picture()?;
        }
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        match self.pending.pop_front() {
            Some(frame) => Ok(frame),
            None if self.eof => Err(Error::Eof),
            None => Err(Error::Again),
        }
    }
}
