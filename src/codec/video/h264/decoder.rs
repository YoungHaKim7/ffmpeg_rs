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
    fferror::{Error, Result},
    util::frame::Frame,
};

use super::{
    cabac::Cabac,
    cabac_tables::{CTX_INIT_I, CTX_INIT_PB_0, CTX_INIT_PB_1, CTX_INIT_PB_2},
    vlc::cavlc,
    {
        CHROMA_DC, CHROMA_QP8, DC_PRED, Gb, I4_DC_128_PRED, I4_LEFT_DC_PRED, I4_TOP_DC_PRED,
        LUMA_DC, MB_INTER, MB_INTRA4X4, MB_INTRA16X16, MB_PCM, MB_UNAVAIL, MbDeblock, Nal,
        PART_16X16, Part, PlaneSel, Pps, SCAN8, Sps, ZIGZAG, chroma_dc_dequant_idct, deblock,
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
    },
};

/// `H264PredWeightTable` (h264_ps.h) — explicit weights/offsets per
/// list/ref plus the implicit-weights table for B (frame pictures use
/// the [0] field-pair slot only).
#[derive(Clone)]
struct PredWeightTable {
    use_weight: u8,
    use_weight_chroma: u8,
    luma_log2_denom: i32,
    chroma_log2_denom: i32,
    /// luma_weight[i][list][0/1] = weight, offset.
    luma_w: [[i32; 32]; 2],
    luma_o: [[i32; 32]; 2],
    /// chroma_weight[i][list][cbcr][0/1].
    chroma_w: [[[i32; 32]; 2]; 2],
    chroma_o: [[[i32; 32]; 2]; 2],
    /// implicit_weight[ref0][ref1] (B, weighted_bipred_idc == 2).
    implicit: [[i32; 32]; 32],
}

impl Default for PredWeightTable {
    fn default() -> Self {
        PredWeightTable {
            use_weight: 0,
            use_weight_chroma: 0,
            luma_log2_denom: 0,
            chroma_log2_denom: 0,
            luma_w: [[0; 32]; 2],
            luma_o: [[0; 32]; 2],
            chroma_w: [[[0; 32]; 2]; 2],
            chroma_o: [[[0; 32]; 2]; 2],
            implicit: [[32; 32]; 32],
        }
    }
}

/// A finished frame held for output reordering (C's delayed_pic entry:
/// picture + poc + KEY/mmco_reset markers).
struct DelayedFrame {
    frame: Frame,
    poc: i32,
    key: bool,
    mmco_reset: bool,
}

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
    /// RefPicList[2] for the current slice: indices into `refs` (list 1
    /// built for B slices only).
    pub(super) ref_lists: [Vec<usize>; 2],
    /// num_ref_idx_active for the current slice (PPS default or the
    /// slice-header override), per list.
    ref_counts: [u32; 2],
    /// list_count (1 for P, 2 for B).
    list_count: usize,
    /// direct_spatial_mv_pred_flag of the current B slice.
    direct_spatial: bool,
    /// Parsed ref_pic_list_modification ops per list (idc, value),
    /// applied after the previous picture enters the DPB.
    pub(super) reorder_ops0: Vec<(u32, u32)>,
    pub(super) reorder_ops1: Vec<(u32, u32)>,
    /// PredWeightTable (ff_h264_pred_weight_table) + implicit weights.
    pwt: PredWeightTable,
    /// frame_num from the slice header being decoded.
    pub(super) slice_frame_num: u32,
    /// nal_ref_idc of the current picture (non-ref pictures stay out of
    /// the DPB).
    pub(super) cur_is_ref: bool,
    /// Current picture is a B picture (reorder heuristic, C's
    /// cur->f->pict_type == AV_PICTURE_TYPE_B).
    pub(super) cur_is_b: bool,
    /// nal_ref_idc of the current picture (POC type 1/2 arithmetic).
    pub(super) cur_nal_ref_idc: u8,
    // ---- POC (C's H264POCContext + sl->poc_* slice carriers) ----
    poc_prev_frame_num: i32,
    poc_prev_frame_num_offset: i32,
    poc_prev_msb: i32,
    poc_prev_lsb: i32,
    poc_frame_num_offset: i32,
    poc_msb: i32,
    sl_poc_lsb: i32,
    sl_delta_poc_bottom: i32,
    sl_delta_poc: [i32; 2],
    // ---- output reordering (C's delayed_pic + last_pocs + has_b_frames) ----
    delayed: Vec<DelayedFrame>,
    last_pocs: [i32; 16],
    next_outputed_poc: i32,
    has_b_frames: usize,
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
    /// Per-list motion caches (C's mv_cache[2] etc; list 1 = B only).
    pub(super) mv_cache: [[[i16; 2]; 15 * 8]; 2],
    pub(super) ref_cache: [[i8; 15 * 8]; 2],
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
    /// RAW chroma pred mode as decoded (0..3, C's chroma_pred_mode_table).
    chroma_pred_raw: u8,
    pub(super) cbp: u32,
    /// P_Skip flag of the MB being decoded (CABAC skip ctx + IS_SKIP).
    cur_skip: bool,
    // inter info for this MB: per 4x4 mv/ref in cache; partitions kept
    // implicitly via mv/ref caches (write-back per 4x4).
    pub(super) intra_pcm: Vec<u8>,
    // ---- CABAC (Phase B) ----
    /// Context states (sl->cabac_state, 1024 entries).
    cabac_state: Box<[u8; 1024]>,
    /// cabac_init_idc of the slice being decoded (P/B only).
    cabac_init_idc: usize,
    /// last_qscale_diff of the slice (mb_qp_delta ctx).
    last_qscale_diff: i32,
    /// |mvd| cache per list, scan8 layout — CABAC mvd ctx.
    mvd_cache: [[[u8; 2]; 15 * 8]; 2],
    /// fill_decode_caches CABAC section (h264_mvpred.h:736-750).
    left_cbp: u16,
    top_cbp: u16,
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
            ref_lists: [Vec::new(), Vec::new()],
            ref_counts: [1, 1],
            list_count: 1,
            direct_spatial: true,
            reorder_ops0: Vec::new(),
            reorder_ops1: Vec::new(),
            pwt: PredWeightTable::default(),
            slice_frame_num: 0,
            cur_is_ref: true,
            cur_is_b: false,
            cur_nal_ref_idc: 0,
            // C's POC init (h264dec.c:442-445 / 301-304): prev_poc_msb
            // = 1<<16, prev_poc_lsb = prev_frame_num = -1, offsets 0.
            poc_prev_frame_num: -1,
            poc_prev_frame_num_offset: 0,
            poc_prev_msb: 1 << 16,
            poc_prev_lsb: -1,
            poc_frame_num_offset: 0,
            poc_msb: 0,
            sl_poc_lsb: 0,
            sl_delta_poc_bottom: 0,
            sl_delta_poc: [0, 0],
            delayed: Vec::new(),
            last_pocs: [i32::MIN; 16],
            next_outputed_poc: i32::MIN + 1,
            has_b_frames: 0,
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
            mv_cache: [[[0, 0]; 120], [[0, 0]; 120]],
            ref_cache: [[-2; 120], [-2; 120]],
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
            chroma_pred_raw: 0,
            cbp: 0,
            cur_skip: false,
            cabac_state: Box::new([0; 1024]),
            cabac_init_idc: 0,
            last_qscale_diff: 0,
            mvd_cache: [[[0; 2]; 15 * 8], [[0; 2]; 15 * 8]],
            left_cbp: 0,
            top_cbp: 0,
            intra_pcm: Vec::new(),
            frame_count: 0,
            out_w: 0,
            out_h: 0,
        }
    }

    pub fn flush(&mut self) {
        self.cur = None;
        self.refs.clear();
        self.ref_lists[0].clear();
        self.ref_lists[1].clear();
        self.slice_table.clear();
        self.pending.clear();
        self.delayed.clear();
        self.got_mb = false;
        self.poc_prev_frame_num = -1;
        self.poc_prev_frame_num_offset = 0;
        self.poc_prev_msb = 1 << 16;
        self.poc_prev_lsb = -1;
        self.poc_frame_num_offset = 0;
        self.last_pocs = [i32::MIN; 16];
        self.next_outputed_poc = i32::MIN + 1;
        self.has_b_frames = 0;
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
            3 => 1u8, // B
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

        // POC slice fields (h264_slice.c:1816-1830; frame pictures):
        // type 0: poc_lsb (+ delta_poc_bottom with pic_order_present);
        // type 1: delta_poc[0] (+ [1]) unless always-zero.
        let (poc_lsb_dbg, poc_lsb, dpb, dp0, dp1) = match sps.poc_type {
            0 => {
                let v = gb.read(sps.log2_max_poc_lsb);
                if std::env::var_os("H264_DUMP").is_some() {
                    eprintln!(
                        "  H2 poc={v} pop={} pos={}",
                        pps.pic_order_present, gb.index
                    );
                }
                let d = if pps.pic_order_present { gb.se()? } else { 0 };
                (v, v, d, 0, 0)
            }
            1 => {
                let d0 = if sps.delta_pic_order_always_zero {
                    0
                } else {
                    gb.se()?
                };
                let d1 = if sps.delta_pic_order_always_zero || !pps.pic_order_present {
                    0
                } else {
                    gb.se()?
                };
                (0, 0, 0, d0, d1)
            }
            _ => (0, 0, 0, 0, 0),
        };

        if pps.redundant_pic_cnt_present {
            let _rpc = gb.ue()?;
        }

        self.slice_frame_num = frame_num;
        self.sl_poc_lsb = poc_lsb as i32;
        self.sl_delta_poc_bottom = dpb;
        self.sl_delta_poc = [dp0, dp1];
        self.ref_counts = pps.ref_count;
        self.reorder_ops0.clear();
        self.reorder_ops1.clear();
        self.list_count = 0;
        self.direct_spatial = true;
        if self.slice_type_nos != 2 {
            // direct_spatial_mv_pred_flag — B slices, before the ref
            // counts (h264_slice.c:1833).
            if self.slice_type_nos == 1 {
                self.direct_spatial = gb.read_bit() == 1;
            }
            // ff_h264_parse_ref_count (h264_parse.c:237): the l1 count
            // is only read for B slices.
            if gb.read_bit() == 1 {
                let l0 = gb.ue()? + 1;
                if l0 > 32 {
                    return Err(Error::InvalidData("reference overflow".into()));
                }
                self.ref_counts[0] = l0;
                if self.slice_type_nos == 1 {
                    let l1 = gb.ue()? + 1;
                    if l1 > 32 {
                        return Err(Error::InvalidData("reference overflow l1".into()));
                    }
                    self.ref_counts[1] = l1;
                }
            }
            self.list_count = if self.slice_type_nos == 1 { 2 } else { 1 };
            // ff_h264_decode_ref_pic_list_reordering (h264_refs.c:431):
            // op/value pairs until op == 3, per list.
            for list in 0..self.list_count {
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
                            return Err(Error::Unsupported(
                                "long-term reference reordering".into(),
                            ));
                        }
                        if list == 0 {
                            self.reorder_ops0.push((op, val));
                        } else {
                            self.reorder_ops1.push((op, val));
                        }
                    }
                }
            }
            // pred_weight_table (ff_h264_pred_weight_table, h264_parse.c:30):
            // explicit weights for weighted P slices / B with idc 1.
            let want_wt = (self.slice_type_nos == 0 && pps.weighted_pred)
                || (self.slice_type_nos == 1 && pps.weighted_bipred_idc == 1);
            if want_wt {
                self.parse_pred_weight_table(&pps, gb)?;
            }
        } else {
            self.list_count = 0;
        }

        if nal.ref_idc != 0 {
            if nal.kind == 5 {
                let _noop = gb.read_bit();
                let _ltr = gb.read_bit();
            } else if gb.read_bit() == 1 {
                return Err(Error::Unsupported("MMCO".into()));
            }
        }
        // cabac_init_idc (h264_slice.c:1876): P/B slices of a CABAC PPS,
        // between dec_ref_pic_marking and slice_qp_delta.
        if self.slice_type_nos != 2 && pps.cabac {
            let idc = gb.ue()?;
            if idc > 2 {
                return Err(Error::InvalidData("cabac_init_idc overflow".into()));
            }
            self.cabac_init_idc = idc as usize;
        }
        self.last_qscale_diff = 0;
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
        let off = self
            .pps
            .as_ref()
            .map(|p| p.chroma_qp_offset)
            .unwrap_or([0, 0]);
        self.chroma_qp[0] = CHROMA_QP8[(qp + off[0]).clamp(0, 51) as usize] as i32;
        self.chroma_qp[1] = CHROMA_QP8[(qp + off[1]).clamp(0, 51) as usize] as i32;

        // Deblocking (h264_slice.c:1900-1928): C keeps the idc with 0/1
        // swapped (1 = on, 0 = off, 2 = on but not across slice edges) and
        // the offsets doubled.
        let mut dbk = MbDeblock {
            mode: 1,
            cqp_off: pps.chroma_qp_offset,
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
        // New-picture detection per C (ff_h264_queue_decode_slice,
        // h264_slice.c:2190): ONLY a first_mb==0 slice starts a new
        // picture — later slices of a multi-slice picture (IDR included)
        // continue the current one. The frame_num change is a port-level
        // safety net for stream breaks.
        Ok((
            first_mb,
            frame_num != self.frame_num
                || (nal.kind == 5 && first_mb == 0)
                || (first_mb == 0 && self.got_mb),
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

    /// fill_decode_neighbors (h264_mvpred.h:487) — the frame-only type
    /// computation, split out so CABAC syntax elements that run BEFORE
    /// fill_decode_caches (mb_skip_run, mb_type) see the neighbor types.
    fn fill_decode_neighbors(&mut self) {
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
    }

    /// fill_decode_neighbors + fill_decode_caches, frame-only path
    /// (h264_mvpred.h:487/539, the non-MBAFF branches).
    fn fill_decode_caches(&mut self, mb_type: u32) {
        self.fill_decode_neighbors();

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
        // C (h264_mvpred.h:698/725): for CABAC INTER macroblocks an
        // unavailable neighbor contributes 0 (never 64) — the coded_
        // block_flag ctx then sees "no coefficients" instead of "all".
        let nnz_unavail = if self.pps.as_ref().is_some_and(|p| p.cabac) && !is_intra {
            0u8
        } else {
            64
        };
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
                let v = if self.n_top != MB_UNAVAIL {
                    0u8
                } else {
                    nnz_unavail
                };
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
                    let v = if self.n_left != MB_UNAVAIL {
                        0u8
                    } else {
                        nnz_unavail
                    };
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
                    self.mv_cache[0][top0 + c] = pic.mv[0][bxy + c];
                    self.ref_cache[0][top0 + c] = pic.ref_pic[0][4 * top_xy + 2 + (c >> 1)] as i8;
                }
            } else {
                for c in 0..4 {
                    self.mv_cache[0][top0 + c] = [0, 0];
                    self.ref_cache[0][top0 + c] = fill_edge(self.n_top);
                }
            }
            let tr = SCAN8[0] - 8 + 4;
            if uses(self.n_topright) {
                let bxy = 4 * (self.mb_x + 1) + 4 * (self.mb_y - 1) * b_stride + 3 * b_stride;
                self.mv_cache[0][tr] = pic.mv[0][bxy];
                self.ref_cache[0][tr] = pic.ref_pic[0][4 * (top_xy + 1) + 2] as i8;
            } else {
                self.mv_cache[0][tr] = [0, 0];
                self.ref_cache[0][tr] = fill_edge(self.n_topright);
            }
            let tl = SCAN8[0] - 8 - 1;
            if uses(self.n_topleft) {
                let bxy = 4 * (self.mb_x - 1) + 4 * (self.mb_y - 1) * b_stride + 3 + 3 * b_stride;
                self.mv_cache[0][tl] = pic.mv[0][bxy];
                self.ref_cache[0][tl] = pic.ref_pic[0][4 * (top_xy - 1) + 3] as i8;
            } else {
                self.mv_cache[0][tl] = [0, 0];
                self.ref_cache[0][tl] = fill_edge(self.n_topleft);
            }
            for i in 0..4 {
                if uses(self.n_left) {
                    let bxy = 4 * self.mb_x.saturating_sub(1)
                        + 4 * self.mb_y * b_stride
                        + 3
                        + i * b_stride;
                    self.mv_cache[0][3 + 8 * (1 + i)] = pic.mv[0][bxy];
                    self.ref_cache[0][3 + 8 * (1 + i)] =
                        pic.ref_pic[0][4 * left_xy + 1 + 2 * (i >> 1)] as i8;
                } else if self.n_left != MB_UNAVAIL {
                    self.mv_cache[0][3 + 8 * (1 + i)] = [0, 0];
                    self.ref_cache[0][3 + 8 * (1 + i)] = -1;
                } else {
                    self.mv_cache[0][3 + 8 * (1 + i)] = [0, 0];
                    self.ref_cache[0][3 + 8 * (1 + i)] = -2;
                }
            }

            // ---- CABAC caches (h264_mvpred.h:836-869) ----
            // mvd borders: the top MB's bottom row / left MB's right
            // column of |mvd| values (C stores only the br 4x4 per MB —
            // the port keeps the full b_stride grid, same cells read).
            {
                let top0 = SCAN8[0] - 8;
                if uses(self.n_top) {
                    let bxy = 4 * self.mb_x + 4 * (self.mb_y - 1) * b_stride + 3 * b_stride;
                    for c in 0..4 {
                        self.mvd_cache[0][top0 + c] = pic.mvd[0][bxy + c];
                    }
                } else {
                    for c in 0..4 {
                        self.mvd_cache[0][top0 + c] = [0, 0];
                    }
                }
                for i in 0..4 {
                    if uses(self.n_left) {
                        let bxy = 4 * self.mb_x.saturating_sub(1)
                            + 4 * self.mb_y * b_stride
                            + 3
                            + i * b_stride;
                        self.mvd_cache[0][3 + 8 * (1 + i)] = pic.mvd[0][bxy];
                    } else {
                        self.mvd_cache[0][3 + 8 * (1 + i)] = [0, 0];
                    }
                }
                // AV_ZERO16(mvd_cache[2 + 8*0]) / [2 + 8*2]
                self.mvd_cache[0][SCAN8[0] + 2] = [0, 0];
                self.mvd_cache[0][SCAN8[0] + 2 + 16] = [0, 0];
            }
        }

        // ---- CABAC cbp contexts (h264_mvpred.h:736-750) ----
        if self.pps.as_ref().is_some_and(|p| p.cabac) {
            let is_intra_cur =
                mb_type == MB_INTRA4X4 || mb_type == MB_INTRA16X16 || mb_type == MB_PCM;
            let pic = self.cur.as_ref().unwrap();
            let top_xy = self.mb_x + self.mb_y.saturating_sub(1) * self.mb_width;
            let left_xy = self.mb_x.saturating_sub(1) + self.mb_y * self.mb_width;
            if self.n_top != 0 {
                self.top_cbp = pic.cbp[top_xy];
            } else {
                self.top_cbp = if is_intra_cur { 0x7CF } else { 0x00F };
            }
            if self.n_left != 0 {
                // frame left_block {0,1,2,3}: keep the left 8x8's luma
                // bits 1/3 plus chroma/DC bits 4..10.
                let cbp = pic.cbp[left_xy];
                self.left_cbp = (cbp & 0x7F0) | (cbp & 2) | (cbp & 8);
            } else {
                self.left_cbp = if is_intra_cur { 0x7CF } else { 0x00F };
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
                pic.mv[0][b_xy + r * b_stride + c] = self.mv_cache[0][SCAN8[0] + 8 * r + c];
                // CABAC mvd table (write_back_motion_list, h264_mvpred.h:
                // 106-117): C keeps only the MB's br 4x4 (bottom row +
                // right column is all the borders ever read); the port's
                // full-grid store covers the same cells.
                pic.mvd[0][b_xy + r * b_stride + c] = self.mvd_cache[0][SCAN8[0] + 8 * r + c];
            }
        }
        // ref_cache is in PICTURE-ID space (C's ref2frm). Store the id
        // directly for the loop filter, and recover the RAW ref_idx (the
        // position of that picture in RefPicList0) for ref_index —
        // pred_pskip_motion compares ref_index against raw 0.
        for (k, blk) in [0usize, 4, 8, 12].into_iter().enumerate() {
            let f = self.ref_cache[0][SCAN8[blk]];
            self.cur.as_mut().unwrap().ref_pic[0][4 * mb_xy + k] = f as i32;
            let raw = if f < 0 {
                -1
            } else {
                self.ref_lists[0]
                    .iter()
                    .position(|&i| self.refs.get(i).is_some_and(|p| p.id as i8 == f))
                    .map_or(-1, |p| p as i8)
            };
            self.cur.as_mut().unwrap().ref_index[0][4 * mb_xy + k] = raw;
        }
    }

    /// Per-MB state kept in the picture after decoding: qscale (0 for
    /// PCM — C's `qscale_table`, the running slice QP is untouched), cbp
    /// (CABAC ORs the DC-coded bits in during the residual), partition
    /// shape, raw chroma pred mode + skip flag (CABAC contexts) and the
    /// slice's deblocking parameters.
    fn record_mb(&mut self, mb_xy: usize) {
        let pcm = self.mb_type == MB_PCM;
        let pic = self.cur.as_mut().unwrap();
        pic.qscale[mb_xy] = if pcm { 0 } else { self.qscale as u8 };
        pic.cbp[mb_xy] = if self.cbp == u32::MAX {
            0
        } else {
            self.cbp as u16
        };
        pic.part[mb_xy] = self.cur_part;
        pic.chroma_pred[mb_xy] = self.chroma_pred_raw;
        pic.skip[mb_xy] = self.cur_skip;
        pic.dbk[mb_xy] = self.slice_dbk;
        self.slice_table[mb_xy] = self.slice_num;
    }

    // ---------------- MV prediction (h264_mvpred.h) ----------------

    /// `fetch_diagonal_mv` frame path.
    fn fetch_diagonal_mv(&self, i: usize, part_width: usize) -> (i8, [i16; 2]) {
        let tr = self.ref_cache[0][i - 8 + part_width];
        if tr != -2 {
            (tr, self.mv_cache[0][i - 8 + part_width])
        } else {
            (self.ref_cache[0][i - 8 - 1], self.mv_cache[0][i - 8 - 1])
        }
    }

    /// `pred_motion`.
    fn pred_motion(&self, n: usize, part_width: usize, r: i8) -> (i16, i16) {
        let idx = SCAN8[n];
        let left_ref = self.ref_cache[0][idx - 1];
        let a = self.mv_cache[0][idx - 1];
        let top_ref = self.ref_cache[0][idx - 8];
        let b = self.mv_cache[0][idx - 8];
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
            let top_ref = self.ref_cache[0][SCAN8[0] - 8];
            let b = self.mv_cache[0][SCAN8[0] - 8];
            if top_ref == r {
                return (b[0], b[1]);
            }
        } else {
            let left_ref = self.ref_cache[0][SCAN8[8] - 1];
            let a = self.mv_cache[0][SCAN8[8] - 1];
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
            let left_ref = self.ref_cache[0][SCAN8[0] - 1];
            let a = self.mv_cache[0][SCAN8[0] - 1];
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
                let r = pic.ref_index[0][4 * xy + 1];
                let a = pic.mv[0][b_xy(mx_ - 1, my_) + 3];
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
                let r = pic.ref_index[0][4 * xy + 2];
                let bm = pic.mv[0][b_xy(mx_, my_ - 1) + 3 * b_stride];
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
                    pic.ref_index[0][4 * xy + 2],
                    pic.mv[0][b_xy(mx_ + 1, my_ - 1) + 3 * b_stride],
                )
            } else if self.n_topright != MB_UNAVAIL {
                (NOT_USED, zero)
            } else if self.n_topleft == MB_INTER {
                let xy = mx_ - 1 + (my_ - 1) * mw;
                (
                    pic.ref_index[0][4 * xy + 3],
                    pic.mv[0][b_xy(mx_ - 1, my_ - 1) + 3 + 3 * b_stride],
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
        // C: fill_rectangle(ref_cache, 4,4,8, 0) — frm of ref_idx 0, i.e.
        // id space: RefPicList0[0]'s picture id.
        let f0 = self.ref_frm(0);
        for r in 0..4 {
            for c in 0..4 {
                self.mv_cache[0][SCAN8[0] + 8 * r + c] = mv;
                self.ref_cache[0][SCAN8[0] + 8 * r + c] = f0;
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
        self.ref_lists[0]
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
        let rc = self.ref_counts[0];
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
                if std::env::var_os("H264_DUMP").is_some()
                    && self.slice_frame_num == 5
                    && self.mb_y == 7
                    && self.mb_x <= 1
                {
                    let i0 = SCAN8[0];
                    eprintln!(
                        "  P16 mb={}:{} r0={r0} f0={f0} L=({},{}) rL={} T=({},{}) rT={} C=({},{}) rC={}",
                        self.mb_x,
                        self.mb_y,
                        self.mv_cache[0][i0 - 1][0],
                        self.mv_cache[0][i0 - 1][1],
                        self.ref_cache[0][i0 - 1],
                        self.mv_cache[0][i0 - 8][0],
                        self.mv_cache[0][i0 - 8][1],
                        self.ref_cache[0][i0 - 8],
                        self.mv_cache[0][i0 - 4][0],
                        self.mv_cache[0][i0 - 4][1],
                        self.ref_cache[0][i0 - 4]
                    );
                }
                set_ref(&mut self.ref_cache[0], 0, 0, 4, 4, f0);
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
                    set_ref(&mut self.ref_cache[0], 0, 2 * n, 4, 2, *rn);
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
                    set_ref(&mut self.ref_cache[0], 2 * n, 0, 2, 4, *rn);
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
                    set_ref(&mut self.ref_cache[0], 2 * (i & 1), 2 * (i >> 1), 2, 2, *ri);
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
                                self.mv_cache[0][idx - 1][0],
                                self.mv_cache[0][idx - 1][1],
                                self.ref_cache[0][idx - 1],
                                self.mv_cache[0][idx - 8][0],
                                self.mv_cache[0][idx - 8][1],
                                self.ref_cache[0][idx - 8],
                                self.mv_cache[0][idx - 8 + bw][0],
                                self.mv_cache[0][idx - 8 + bw][1],
                                self.ref_cache[0][idx - 8 + bw]
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
                self.mv_cache[0][SCAN8[0] + 8 * (y + r) + (x + c)] = [mx, my];
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
        self.cur_skip = true;
        self.chroma_pred_raw = 0;
        for i in 0..48usize {
            self.nnz_cache[SCAN8[i]] = 0;
        }
        // write_back_motion_list zeroes the mvd table for IS_SKIP MBs.
        for r in 0..4 {
            for c in 0..4 {
                self.mvd_cache[0][SCAN8[0] + 8 * r + c] = [0, 0];
            }
        }
        self.pred_pskip_motion();
        // C fills ref_cache with ref2frm[0] (= reference 0); in the port's
        // id space that's the id of RefPicList0[0]. A raw 0 here poisons
        // every later neighbour comparison (0 is not a picture id).
        let f0 = self.ref_frm(0);
        for i in 0..16usize {
            self.ref_cache[0][SCAN8[i]] = f0;
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
            let off = self
                .pps
                .as_ref()
                .map(|p| p.chroma_qp_offset)
                .unwrap_or([0, 0]);
            self.chroma_qp[0] = CHROMA_QP8[(self.qscale + off[0]).clamp(0, 51) as usize] as i32;
            self.chroma_qp[1] = CHROMA_QP8[(self.qscale + off[1]).clamp(0, 51) as usize] as i32;
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

    // ---------------- CABAC MB decode (h264_cabac.c:1920) ----------------

    /// `ff_h264_init_cabac_states` (h264_cabac.c:1262).
    fn init_cabac_states(&mut self) {
        let slice_qp = self.qscale.clamp(0, 51);
        let table: &[[i8; 2]; 1024] = if self.slice_type_nos == 2 {
            &CTX_INIT_I
        } else {
            match self.cabac_init_idc {
                0 => &CTX_INIT_PB_0,
                1 => &CTX_INIT_PB_1,
                _ => &CTX_INIT_PB_2,
            }
        };
        for i in 0..1024usize {
            let mut pre = 2 * (((table[i][0] as i32 * slice_qp) >> 4) + table[i][1] as i32) - 127;
            pre ^= pre >> 31; // abs
            if pre > 124 {
                pre = 124 + (pre & 1);
            }
            self.cabac_state[i] = pre as u8;
        }
    }

    /// `decode_cabac_mb_skip` (h264_cabac.c:1336), frame path. C's ctx
    /// tests slice_table + IS_SKIP at mba = mb_xy-1 / mbb = mb_xy-stride
    /// with NO mb_x guard (a row-start MB sees the previous row's last
    /// MB as its "left") — ported verbatim.
    fn cabac_mb_skip(&mut self, cab: &mut Cabac) -> u32 {
        // C computes mba/mbb with RAW mb_xy arithmetic over mb_stride =
        // mb_width+1: at a row start, mba = mb_xy-1 lands in the PADDING
        // column (slice_table there is never this slice → no ctx), and
        // for the top row mbb goes negative. The port's flat mb_xy must
        // guard explicitly instead of wrapping to the previous row's
        // last MB.
        let mb_xy = self.mb_x + self.mb_y * self.mb_width;
        let mba = if self.mb_x > 0 {
            Some(mb_xy as isize - 1)
        } else {
            None
        };
        let mbb = if self.mb_y > 0 {
            Some(mb_xy as isize - self.mb_width as isize)
        } else {
            None
        };
        let in_slice = |xy: isize| -> bool {
            xy >= 0
                && (xy as usize) < self.slice_table.len()
                && self.slice_table[xy as usize] == self.slice_num
        };
        let mut ctx = 0usize;
        {
            let pic = self.cur.as_ref().unwrap();
            if mba.is_some_and(|xy| in_slice(xy) && !pic.skip[xy as usize]) {
                ctx += 1;
            }
            if mbb.is_some_and(|xy| in_slice(xy) && !pic.skip[xy as usize]) {
                ctx += 1;
            }
        }
        cab.get(&mut self.cabac_state[11 + ctx])
    }

    /// `decode_cabac_intra_mb_type` (h264_cabac.c:1304) — returns the
    /// ff_h264_i_mb_type_info row (0=I4x4, 1..24=I16x16, 25=PCM).
    fn cabac_intra_mb_type(
        &mut self,
        cab: &mut Cabac,
        ctx_base: usize,
        intra_slice: bool,
    ) -> Result<usize> {
        let mut base = ctx_base;
        if intra_slice {
            let mut ctx = 0usize;
            if matches!(self.n_left, MB_INTRA16X16 | MB_PCM) {
                ctx += 1;
            }
            if matches!(self.n_top, MB_INTRA16X16 | MB_PCM) {
                ctx += 1;
            }
            if cab.get(&mut self.cabac_state[base + ctx]) == 0 {
                return Ok(0); // I4x4
            }
            base += 2;
        } else if cab.get(&mut self.cabac_state[base]) == 0 {
            return Ok(0); // I4x4
        }
        // pcm_flag rides on a terminate bin (spec 9.3.2.4).
        if cab.terminate() != 0 {
            return Ok(25); // PCM
        }
        let is = intra_slice as usize;
        let mut mb_type = 1usize; // I16x16
        mb_type += 12 * cab.get(&mut self.cabac_state[base + 1]) as usize;
        if cab.get(&mut self.cabac_state[base + 2]) == 1 {
            mb_type += 4 + 4 * cab.get(&mut self.cabac_state[base + 2 + is]) as usize;
        }
        mb_type += 2 * cab.get(&mut self.cabac_state[base + 3 + is]) as usize;
        mb_type += cab.get(&mut self.cabac_state[base + 3 + 2 * is]) as usize;
        Ok(mb_type)
    }

    /// `decode_cabac_mb_intra4x4_pred_mode` (h264_cabac.c:1373).
    fn cabac_intra4x4_pred_mode(&mut self, cab: &mut Cabac, pred_mode: i8) -> i8 {
        if cab.get(&mut self.cabac_state[68]) == 1 {
            return pred_mode;
        }
        let mut mode = 0i8;
        mode += cab.get(&mut self.cabac_state[69]) as i8;
        mode += 2 * cab.get(&mut self.cabac_state[69]) as i8;
        mode += 4 * cab.get(&mut self.cabac_state[69]) as i8;
        mode + (mode >= pred_mode) as i8
    }

    /// `decode_cabac_mb_chroma_pre_mode` (h264_cabac.c:1387) — RAW mode
    /// 0..3 (the port's check_intra_pred_mode remaps afterwards).
    fn cabac_chroma_pre_mode(&mut self, cab: &mut Cabac) -> u32 {
        let mb_xy = self.mb_x + self.mb_y * self.mb_width;
        let left_xy = mb_xy.wrapping_sub(1);
        let top_xy = mb_xy.wrapping_sub(self.mb_width);
        let mut ctx = 0usize;
        {
            let pic = self.cur.as_ref().unwrap();
            if self.n_left != 0 && pic.chroma_pred[left_xy] != 0 {
                ctx += 1;
            }
            if self.n_top != 0 && pic.chroma_pred[top_xy] != 0 {
                ctx += 1;
            }
        }
        if cab.get(&mut self.cabac_state[64 + ctx]) == 0 {
            return 0;
        }
        if cab.get(&mut self.cabac_state[64 + 3]) == 0 {
            return 1;
        }
        if cab.get(&mut self.cabac_state[64 + 3]) == 0 {
            return 2;
        }
        3
    }

    /// `decode_cabac_mb_cbp_luma` (h264_cabac.c:1412).
    fn cabac_cbp_luma(&mut self, cab: &mut Cabac) -> u32 {
        let (cbp_a, cbp_b) = (self.left_cbp, self.top_cbp);
        let mut cbp = 0u32;
        let mut ctx = ((cbp_a & 0x02) == 0) as usize + 2 * ((cbp_b & 0x04) == 0) as usize;
        cbp += cab.get(&mut self.cabac_state[73 + ctx]) as u32;
        ctx = (cbp & 0x01 == 0) as usize + 2 * ((cbp_b & 0x08) == 0) as usize;
        cbp += (cab.get(&mut self.cabac_state[73 + ctx]) as u32) << 1;
        ctx = ((cbp_a & 0x08) == 0) as usize + 2 * (cbp & 0x01 == 0) as usize;
        cbp += (cab.get(&mut self.cabac_state[73 + ctx]) as u32) << 2;
        ctx = (cbp & 0x04 == 0) as usize + 2 * (cbp & 0x02 == 0) as usize;
        cbp += (cab.get(&mut self.cabac_state[73 + ctx]) as u32) << 3;
        cbp
    }

    /// `decode_cabac_mb_cbp_chroma` (h264_cabac.c:1429).
    fn cabac_cbp_chroma(&mut self, cab: &mut Cabac) -> u32 {
        let cbp_a = (self.left_cbp >> 4) & 0x03;
        let cbp_b = (self.top_cbp >> 4) & 0x03;
        let mut ctx = 0usize;
        if cbp_a > 0 {
            ctx += 1;
        }
        if cbp_b > 0 {
            ctx += 2;
        }
        if cab.get(&mut self.cabac_state[77 + ctx]) == 0 {
            return 0;
        }
        ctx = 4;
        if cbp_a == 2 {
            ctx += 1;
        }
        if cbp_b == 2 {
            ctx += 2;
        }
        1 + cab.get(&mut self.cabac_state[77 + ctx]) as u32
    }

    /// `decode_cabac_p_mb_sub_type` (h264_cabac.c:1449): 0=sub8x8,
    /// 1=sub8x4, 2=sub4x8, 3=sub4x4.
    fn cabac_p_mb_sub_type(&mut self, cab: &mut Cabac) -> u32 {
        if cab.get(&mut self.cabac_state[21]) == 1 {
            return 0; // 8x8
        }
        if cab.get(&mut self.cabac_state[22]) == 0 {
            return 1; // 8x4
        }
        if cab.get(&mut self.cabac_state[23]) == 1 {
            return 2; // 4x8
        }
        3 // 4x4
    }

    /// picture-id (this port's ref_cache space) → RAW ref_idx (position
    /// in RefPicList0). C's ref_cache is raw ref_idx and the ctx test is
    /// `refa > 0`; the port carries picture ids, so convert for the ctx.
    fn id_to_raw(&self, id: i8) -> i8 {
        if id < 0 {
            return id; // -1 unused / -2 unavailable — both fail > 0
        }
        self.ref_lists[0]
            .iter()
            .position(|&k| self.refs.get(k).is_some_and(|p| p.id as i8 == id))
            .map_or(-1, |p| p as i8)
    }

    /// `decode_cabac_mb_ref` (h264_cabac.c:1477), P slice.
    fn cabac_mb_ref(&mut self, cab: &mut Cabac, n: usize) -> Result<i8> {
        let refa = self.id_to_raw(self.ref_cache[0][SCAN8[n] - 1]);
        let refb = self.id_to_raw(self.ref_cache[0][SCAN8[n] - 8]);
        let mut ctx = 0usize;
        if refa > 0 {
            ctx += 1;
        }
        if refb > 0 {
            ctx += 2;
        }
        let mut r = 0i8;
        while cab.get(&mut self.cabac_state[54 + ctx]) == 1 {
            r += 1;
            ctx = (ctx >> 2) + 4;
            if r >= 32 {
                return Err(Error::InvalidData("cabac ref overflow".into()));
            }
        }
        Ok(r)
    }

    /// `decode_cabac_mb_mvd` (h264_cabac.c:1506): returns (mvd, |mvd|
    /// capped at 70 — C's *mvda for the mvd_cache).
    fn cabac_mb_mvd(&mut self, cab: &mut Cabac, ctxbase: usize, amvd: i32) -> Result<(i32, u8)> {
        // C: ctxbase + ((amvd-3)>>31) + ((amvd-33)>>31) + 2 — the sign
        // shifts contribute -1 while amvd is BELOW the threshold:
        // amvd<3 → base+0, 3..32 → base+1, ≥33 → base+2.
        let ctx = ctxbase + 2 - (amvd < 3) as usize - (amvd < 33) as usize;
        if cab.get(&mut self.cabac_state[ctx]) == 0 {
            return Ok((0, 0));
        }
        let mut mvd = 1i32;
        let mut ctx = ctxbase + 3;
        while mvd < 9 && cab.get(&mut self.cabac_state[ctx]) == 1 {
            if mvd < 4 {
                ctx += 1;
            }
            mvd += 1;
        }
        if mvd >= 9 {
            let mut k = 3i32;
            while cab.bypass() == 1 {
                mvd += 1 << k;
                k += 1;
                if k > 24 {
                    return Err(Error::InvalidData("cabac mvd overflow".into()));
                }
            }
            loop {
                // C's `while (k--)`: test-then-decrement, body sees k-1.
                let t = k;
                k -= 1;
                if t == 0 {
                    break;
                }
                mvd += (cab.bypass() as i32) << k;
            }
        }
        let abs = mvd.min(70) as u8;
        Ok((cab.bypass_sign(-mvd), abs))
    }

    /// DECODE_CABAC_MB_MVD (h264_cabac.c:1543): both components + the
    /// mvd_cache amvd sums (list 0).
    fn cabac_mvd_xy(&mut self, cab: &mut Cabac, n: usize) -> Result<(i32, i32, u8, u8)> {
        let idx = SCAN8[n];
        let amvd0 = self.mvd_cache[0][idx - 1][0] as i32 + self.mvd_cache[0][idx - 8][0] as i32;
        let amvd1 = self.mvd_cache[0][idx - 1][1] as i32 + self.mvd_cache[0][idx - 8][1] as i32;
        let (mxd, mpx) = self.cabac_mb_mvd(cab, 40, amvd0)?;
        let (myd, mpy) = self.cabac_mb_mvd(cab, 47, amvd1)?;
        Ok((mxd, myd, mpx, mpy))
    }

    /// Fill an |mvd| rectangle in the cache (fill_rectangle, 2-byte cells).
    fn fill_mvd_rect(&mut self, x: usize, y: usize, w: usize, h: usize, px: u8, py: u8) {
        for r in 0..h {
            for c in 0..w {
                self.mvd_cache[0][SCAN8[0] + 8 * (y + r) + (x + c)] = [px, py];
            }
        }
    }

    /// `get_cabac_cbf_ctx` (h264_cabac.c:1558).
    fn cabac_cbf_ctx(&self, cat: usize, idx: usize, is_dc: bool) -> usize {
        static BASE_CTX: [usize; 14] = [
            85, 89, 93, 97, 101, 1012, 460, 464, 468, 1016, 472, 476, 480, 1020,
        ];
        let (nza, nzb) = if is_dc {
            if cat == 3 {
                let i = idx - CHROMA_DC;
                (
                    (self.left_cbp >> (6 + i)) & 1,
                    (self.top_cbp >> (6 + i)) & 1,
                )
            } else {
                let i = idx - LUMA_DC;
                (
                    (self.left_cbp >> (8 + i)) & 1,
                    (self.top_cbp >> (8 + i)) & 1,
                )
            }
        } else {
            (
                self.nnz_cache[SCAN8[idx] - 1] as u16,
                self.nnz_cache[SCAN8[idx] - 8] as u16,
            )
        };
        let mut ctx = 0usize;
        if nza > 0 {
            ctx += 1;
        }
        if nzb > 0 {
            ctx += 2;
        }
        BASE_CTX[cat] + ctx
    }

    /// `decode_cabac_mb_dqp` (h264_cabac.c:2399).
    fn cabac_mb_dqp(&mut self, cab: &mut Cabac) -> Result<()> {
        if cab.get(&mut self.cabac_state[60 + (self.last_qscale_diff != 0) as usize]) == 1 {
            let mut val = 1i32;
            let mut ctx = 2usize;
            while cab.get(&mut self.cabac_state[60 + ctx]) == 1 {
                ctx = 3;
                val += 1;
                if val > 2 * 51 {
                    return Err(Error::InvalidData("cabac dqp overflow".into()));
                }
            }
            let val = if val & 1 != 0 {
                (val + 1) >> 1
            } else {
                -((val + 1) >> 1)
            };
            self.last_qscale_diff = val;
            self.qscale += val;
            if self.qscale < 0 {
                self.qscale += 52;
            } else if self.qscale > 51 {
                self.qscale -= 52;
            }
            if !(0..=51).contains(&self.qscale) {
                return Err(Error::InvalidData("dquant out of range".into()));
            }
            let off = self
                .pps
                .as_ref()
                .map(|p| p.chroma_qp_offset)
                .unwrap_or([0, 0]);
            self.chroma_qp[0] = CHROMA_QP8[(self.qscale + off[0]).clamp(0, 51) as usize] as i32;
            self.chroma_qp[1] = CHROMA_QP8[(self.qscale + off[1]).clamp(0, 51) as usize] as i32;
        } else {
            self.last_qscale_diff = 0;
        }
        Ok(())
    }

    /// `decode_cabac_residual_internal` (h264_cabac.c:1590) — the 420
    /// frame subset: cats 0-4, no 8x8/422. `qmul == None` is C's is_dc.
    /// The coded_block_flag (decode_cabac_residual_{dc,nondc}) is folded
    /// in at the top.
    #[allow(clippy::too_many_arguments)]
    fn cabac_residual(
        &mut self,
        cab: &mut Cabac,
        block: &mut [i16],
        cat: usize,
        n: usize,
        scan: &[u8; 16],
        qmul: Option<&[u32; 16]>,
        max_coeff: usize,
    ) -> Result<()> {
        // significant_coeff_flag_offset / last_coeff_flag_offset /
        // coeff_abs_level_m1_offset, MB_FIELD = 0 row (h264_cabac.c:1597).
        static SIG_OFF: [usize; 14] = [
            105, 120, 134, 149, 152, 402, 484, 499, 513, 660, 528, 543, 557, 718,
        ];
        static LAST_OFF: [usize; 14] = [
            166, 181, 195, 210, 213, 417, 572, 587, 601, 690, 616, 631, 645, 748,
        ];
        static ABSM1_OFF: [usize; 14] = [
            227, 237, 247, 257, 266, 426, 952, 962, 972, 708, 982, 992, 1002, 766,
        ];
        static L1_CTX: [usize; 8] = [1, 2, 3, 4, 0, 0, 0, 0];
        static GT1_CTX: [usize; 8] = [5, 5, 5, 5, 6, 7, 8, 9];
        static TRANS0: [usize; 8] = [1, 2, 3, 3, 4, 5, 6, 7];
        static TRANS1: [usize; 8] = [4, 4, 4, 4, 5, 6, 7, 7];

        let is_dc = qmul.is_none();

        // coded_block_flag
        let ctx = self.cabac_cbf_ctx(cat, n, is_dc);
        if cab.get(&mut self.cabac_state[ctx]) == 0 {
            self.nnz_cache[SCAN8[n]] = 0;
            return Ok(());
        }

        // significance map: positions 0..max_coeff-2 explicit, the final
        // scan position is significant by elimination when the walk
        // exhausts (DECODE_SIGNIFICANCE, h264_cabac.c:1669).
        let mut index = [0usize; 64];
        let mut coeff_count = 0usize;
        let sig_base = SIG_OFF[cat];
        let last_base = LAST_OFF[cat];
        let mut last = 0usize;
        while last < max_coeff - 1 {
            if cab.get(&mut self.cabac_state[sig_base + last]) == 1 {
                index[coeff_count] = last;
                coeff_count += 1;
                if cab.get(&mut self.cabac_state[last_base + last]) == 1 {
                    last = max_coeff; // break marker
                    break;
                }
            }
            last += 1;
        }
        if last == max_coeff - 1 {
            index[coeff_count] = last;
            coeff_count += 1;
        }

        // nnz + DC-coded cbp bits (C ORs into cbp_table during decode;
        // the port ORs into self.cbp — record_mb persists it after).
        self.nnz_cache[SCAN8[n]] = coeff_count as u8;
        if std::env::var_os("H264_DUMP").is_some() {
            eprintln!(
                "  RES cat={cat} n={n} max={max_coeff} cc={coeff_count} q={}",
                self.qscale
            );
        }
        if is_dc {
            if cat == 3 {
                self.cbp |= 0x40 << (n - CHROMA_DC);
            } else {
                self.cbp |= 0x100 << (n - LUMA_DC);
            }
        }

        // STORE_BLOCK (h264_cabac.c:1722) — reverse scan order.
        let abs_base = ABSM1_OFF[cat];
        let mut node_ctx = 0usize;
        while coeff_count > 0 {
            coeff_count -= 1;
            let j = scan[index[coeff_count]] as usize;
            let ctx = L1_CTX[node_ctx] + abs_base;
            let v = if cab.get(&mut self.cabac_state[ctx]) == 0 {
                node_ctx = TRANS0[node_ctx];
                match qmul {
                    None => cab.bypass_sign(-1),
                    Some(q) => (cab.bypass_sign(-(q[j] as i32)) + 32) >> 6,
                }
            } else {
                let ctx = GT1_CTX[node_ctx] + abs_base;
                node_ctx = TRANS1[node_ctx];
                let mut coeff_abs = 2u32;
                while coeff_abs < 15 && cab.get(&mut self.cabac_state[ctx]) == 1 {
                    coeff_abs += 1;
                }
                if coeff_abs >= 15 {
                    let mut jk = 0usize;
                    while cab.bypass() == 1 && jk < 16 + 7 {
                        jk += 1;
                    }
                    coeff_abs = 1;
                    while jk > 0 {
                        jk -= 1;
                        coeff_abs = coeff_abs + coeff_abs + cab.bypass() as u32;
                    }
                    coeff_abs += 14;
                }
                match qmul {
                    None => cab.bypass_sign(-(coeff_abs as i32)),
                    Some(q) => {
                        ((cab.bypass_sign(-(coeff_abs as i32)) as i64 * q[j] as i64 + 32) >> 6)
                            as i32
                    }
                }
            };
            block[j] = v as i16;
        }
        Ok(())
    }

    /// The residual tail of `ff_h264_decode_mb_cabac` (h264_cabac.c:2436)
    /// for 420: luma (+ intra16 DC) then chroma DC/AC.
    fn decode_mb_residual_cabac(&mut self, cab: &mut Cabac, mb_xy: usize) -> Result<()> {
        self.mb = [0; 48 * 16];
        let scan: [u8; 16] = ZIGZAG;
        let scan1 = scan_shift1();
        // ff_h264_chroma_dc_scan {0,16,32,48} into the 64-slot region.
        let mut scan_cdc = [0u8; 16];
        scan_cdc[..4].copy_from_slice(&CHROMA_DC_SCAN);

        // ---- luma ----
        if self.mb_type == MB_INTRA16X16 {
            self.mb_luma_dc = [0; 16];
            let mut dc = [0i16; 16];
            self.cabac_residual(cab, &mut dc, 0, LUMA_DC, &scan, None, 16)?;
            self.mb_luma_dc = dc;
            if self.cbp & 15 != 0 {
                let qm = *self.pps.as_ref().unwrap().dequant(0, self.qscale as usize);
                for i in 0..16usize {
                    let mut blk = [0i16; 16];
                    self.cabac_residual(cab, &mut blk, 1, i, &scan1, Some(&qm), 15)?;
                    // scan+1 writes positions 1..15 (DC slot = the scatter)
                    self.mb[i * 16 + 1..(i + 1) * 16].copy_from_slice(&blk[1..16]);
                }
            } else {
                for i in 0..16usize {
                    self.nnz_cache[SCAN8[i]] = 0;
                }
            }
        } else if self.cbp & 15 != 0 {
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
                        self.cabac_residual(cab, &mut blk, 2, index, &scan, Some(&qm), 16)?;
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

        // ---- chroma DC ----
        if self.cbp & 0x30 != 0 {
            for ch in 0..2usize {
                let mut dc64 = [0i16; 64];
                self.cabac_residual(cab, &mut dc64, 3, CHROMA_DC + ch, &scan_cdc, None, 4)?;
                let base = 16 * (16 + 16 * ch);
                for s in [0usize, 16, 32, 48] {
                    self.mb[base + s] = dc64[s];
                }
            }
        }
        // ---- chroma AC ----
        if self.cbp & 0x20 != 0 {
            for ch in 0..2usize {
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
                for i4 in 0..4usize {
                    let block_idx = 16 + 16 * ch + i4;
                    let index = 16 * (16 + 16 * ch) + 16 * i4;
                    let mut blk = [0i16; 16];
                    self.cabac_residual(cab, &mut blk, 4, block_idx, &scan1, Some(&qm), 15)?;
                    self.mb[index + 1..index + 16].copy_from_slice(&blk[1..16]);
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

        self.write_back_non_zero_count(mb_xy);
        Ok(())
    }

    /// `ff_h264_decode_mb_cabac` (h264_cabac.c:1920), the frame I/P 420
    /// subset: no B/direct/weighting/MBAFF/8x8 (all gated Unsupported).
    /// Skip and PCM MBs are fully handled here (record_mb included); the
    /// caller runs hl_decode_mb + end_of_slice afterwards either way.
    fn decode_mb_cabac(&mut self, cab: &mut Cabac) -> Result<()> {
        let mb_xy = self.mb_x + self.mb_y * self.mb_width;
        self.cur_skip = false;
        self.chroma_pred_raw = 0;

        // ---- mb_skip_flag (P slices) ----
        if self.slice_type_nos != 2 && self.cabac_mb_skip(cab) == 1 {
            if std::env::var_os("H264_DUMP").is_some() {
                eprintln!("CMB {} {} SKIP", self.mb_x, self.mb_y);
            }
            self.decode_mb_skip(mb_xy);
            self.last_qscale_diff = 0;
            return Ok(());
        }
        self.prev_mb_skipped = false;
        self.fill_decode_neighbors();

        // ---- mb_type ----
        let part = if self.slice_type_nos == 0 {
            if cab.get(&mut self.cabac_state[14]) == 0 {
                // P-type (no P_8x8ref0 in CABAC)
                let mt = if cab.get(&mut self.cabac_state[15]) == 0 {
                    3 * cab.get(&mut self.cabac_state[16]) as usize // 0 / 3 (P_8x8)
                } else {
                    2 - cab.get(&mut self.cabac_state[17]) as usize // 1 (16x8) / 2 (8x16)
                };
                match mt {
                    0 => Part::P16x16,
                    1 => Part::P16x8,
                    2 => Part::P8x16,
                    _ => Part::P8x8,
                }
            } else {
                Part::Intra(self.cabac_intra_mb_type(cab, 17, false)?)
            }
        } else {
            Part::Intra(self.cabac_intra_mb_type(cab, 3, true)?)
        };

        // ---- intra PCM (before fill_decode_caches, C order) ----
        if let Part::Intra(25) = part {
            let ptr = cab.pcm_ptr();
            let data = cab.data_from(ptr);
            if data.len() < 384 {
                return Err(Error::InvalidData("not enough data for intra PCM".into()));
            }
            self.intra_pcm = data[..384].to_vec();
            *cab = Cabac::new(cab.data_from(ptr + 384))?;
            self.mb_type = MB_PCM;
            self.cbp = 0xf7ef; // C: cbp_table[mb_xy] = 0xf7ef
            self.cur_part = PART_16X16;
            let pic = self.cur.as_mut().unwrap();
            pic.nnz[mb_xy] = [16; 48];
            pic.mb_type[mb_xy] = MB_PCM;
            self.record_mb(mb_xy);
            return Ok(());
        }

        // ---- port type/cbp/pred-mode mapping (the CAVLC table logic) ----
        if std::env::var_os("H264_DUMP").is_some() {
            eprintln!("CMB {} {} type={:?} cbp-tail", self.mb_x, self.mb_y, part);
        }
        match part {
            Part::Intra(row) => {
                let row = row as usize;
                let (mbt, cbp, pred) = (
                    I_MB_TYPE_INFO[row * 3],
                    I_MB_TYPE_INFO[row * 3 + 1],
                    I_MB_TYPE_INFO[row * 3 + 2] as i32,
                );
                let mbt = match mbt {
                    0 => MB_INTRA4X4,
                    25 => MB_PCM,
                    _ => MB_INTRA16X16,
                };
                self.mb_type = mbt as u32;
                self.cbp = if cbp == 255 { u32::MAX } else { cbp as u32 };
                // C's 16x16 pred namespace (0=DC,1=H,2=V,3=plane) → port.
                self.intra16x16_pred_mode = match pred {
                    0 => 2,
                    1 => 1,
                    2 => 0,
                    _ => 3,
                };
            }
            _ => {
                self.mb_type = MB_INTER;
                self.cbp = 0;
            }
        }

        self.fill_decode_caches(self.mb_type);
        let pic = self.cur.as_mut().unwrap();
        pic.mb_type[mb_xy] = self.mb_type;

        // ---- intra prediction modes ----
        if self.mb_type == MB_INTRA4X4 || self.mb_type == MB_INTRA16X16 {
            if self.mb_type == MB_INTRA4X4 {
                for i in 0..16usize {
                    let pred = self.pred_intra_mode(i);
                    let mode = self.cabac_intra4x4_pred_mode(cab, pred);
                    if std::env::var_os("H264_DUMP").is_some() {
                        eprintln!("  i4x4 {pred} {mode}");
                    }
                    self.intra4x4_pred_mode_cache[SCAN8[i]] = mode;
                }
                self.write_back_intra_pred_mode(mb_xy);
                self.check_intra4x4_pred_mode()?;
            } else {
                self.intra16x16_pred_mode =
                    self.check_intra_pred_mode(self.intra16x16_pred_mode, false)?;
            }
            let raw = self.cabac_chroma_pre_mode(cab);
            self.chroma_pred_raw = raw as u8;
            if std::env::var_os("H264_DUMP").is_some() {
                eprintln!("  chroma_pred={raw}");
            }
            self.chroma_pred_mode = self.check_intra_pred_mode(raw as i32, true)?;
        } else {
            // ---- inter: partitions, refs, mvds ----
            self.cur_part = match part {
                Part::P16x8 => PART_16X8,
                Part::P8x16 => PART_8X16,
                Part::P8x8 => PART_8X8,
                _ => PART_16X16,
            };
            self.p8x8_ref0 = false;
            let set_ref = |s: &mut Self, x: usize, y: usize, w: usize, h: usize, r: i8| {
                for yy in 0..h {
                    for xx in 0..w {
                        s.ref_cache[0][SCAN8[0] + 8 * (y + yy) + (x + xx)] = r;
                    }
                }
            };
            let rc = self.ref_counts[0];
            let read_ref = |s: &mut Self, cab: &mut Cabac, n: usize| -> Result<i8> {
                if rc <= 1 {
                    return Ok(0);
                }
                let r = s.cabac_mb_ref(cab, n)?;
                if r as u32 >= rc {
                    return Err(Error::InvalidData(format!("reference {r} overflow")));
                }
                Ok(r)
            };
            match part {
                Part::P16x16 => {
                    let r0 = read_ref(self, cab, 0)?;
                    let f0 = self.ref_frm(r0);
                    set_ref(self, 0, 0, 4, 4, f0);
                    let (mx, my) = self.pred_motion(0, 4, f0);
                    let (mxd, myd, mpx, mpy) = self.cabac_mvd_xy(cab, 0)?;
                    let (mx, my) = (mx + mxd as i16, my + myd as i16);
                    self.fill_mv_rect(0, 0, 4, 4, mx, my);
                    self.fill_mvd_rect(0, 0, 4, 4, mpx, mpy);
                }
                Part::P16x8 => {
                    let mut r = [0i8; 2];
                    for n in 0..2usize {
                        let rn = read_ref(self, cab, 8 * n)?;
                        r[n] = self.ref_frm(rn);
                        set_ref(self, 0, 2 * n, 4, 2, r[n]);
                    }
                    for n in 0..2usize {
                        let (mx, my) = self.pred_16x8_motion(n, r[n]);
                        let (mxd, myd, mpx, mpy) = self.cabac_mvd_xy(cab, 8 * n)?;
                        let (mx, my) = (mx + mxd as i16, my + myd as i16);
                        self.fill_mv_rect(0, 2 * n, 4, 2, mx, my);
                        self.fill_mvd_rect(0, 2 * n, 4, 2, mpx, mpy);
                    }
                }
                Part::P8x16 => {
                    let mut r = [0i8; 2];
                    for n in 0..2usize {
                        let rn = read_ref(self, cab, 4 * n)?;
                        r[n] = self.ref_frm(rn);
                        set_ref(self, 2 * n, 0, 2, 4, r[n]);
                    }
                    for n in 0..2usize {
                        let (mx, my) = self.pred_8x16_motion(n, r[n]);
                        let (mxd, myd, mpx, mpy) = self.cabac_mvd_xy(cab, 4 * n)?;
                        let (mx, my) = (mx + mxd as i16, my + myd as i16);
                        self.fill_mv_rect(2 * n, 0, 2, 4, mx, my);
                        self.fill_mvd_rect(2 * n, 0, 2, 4, mpx, mpy);
                    }
                }
                Part::P8x8 | Part::Intra(_) => {
                    // sub_mb_types first, then per-quadrant refs, then mvds.
                    let mut subs = [0u32; 4];
                    for sub in subs.iter_mut() {
                        *sub = self.cabac_p_mb_sub_type(cab);
                    }
                    let mut refs8 = [0i8; 4];
                    for i in 0..4usize {
                        let raw = read_ref(self, cab, 4 * i)?;
                        refs8[i] = self.ref_frm(raw);
                        set_ref(self, 2 * (i & 1), 2 * (i >> 1), 2, 2, refs8[i]);
                    }
                    for (i, &sub) in subs.iter().enumerate() {
                        let (bw, bh, count) = match sub {
                            0 => (2usize, 2usize, 1usize),
                            1 => (2, 1, 2),
                            2 => (1, 2, 2),
                            _ => (1, 1, 4),
                        };
                        for j in 0..count {
                            let block = 4 * i + bw * j;
                            let (mx, my) = self.pred_motion(block, bw, refs8[i]);
                            let (mxd, myd, mpx, mpy) = self.cabac_mvd_xy(cab, block)?;
                            let (mx, my) = (mx + mxd as i16, my + myd as i16);
                            let g = SCAN8[block];
                            let (gx, gy) = ((g & 7) - 4, (g >> 3) - 1);
                            self.fill_mv_rect(gx, gy, bw, bh, mx, my);
                            self.fill_mvd_rect(gx, gy, bw, bh, mpx, mpy);
                        }
                    }
                }
            }
            self.write_back_motion(mb_xy);
        }

        // ---- cbp ----
        if self.mb_type != MB_INTRA16X16 {
            let mut cbp = self.cabac_cbp_luma(cab);
            cbp |= self.cabac_cbp_chroma(cab) << 4;
            self.cbp = cbp;
        }
        if std::env::var_os("H264_DUMP").is_some() {
            eprintln!("  cbp={:#04x}", self.cbp);
        }

        // ---- residual ----
        if self.cbp != 0 || self.mb_type == MB_INTRA16X16 {
            self.cabac_mb_dqp(cab)?;
            if std::env::var_os("H264_DUMP").is_some() {
                eprintln!("  qscale={}", self.qscale);
            }
            self.decode_mb_residual_cabac(cab, mb_xy)?;
        } else {
            for i in 0..16usize {
                self.nnz_cache[SCAN8[i]] = 0;
            }
            for ch in 0..2usize {
                for i in 0..4usize {
                    self.nnz_cache[SCAN8[16 + 16 * ch + i]] = 0;
                    self.nnz_cache[SCAN8[20 + 16 * ch + i]] = 0;
                }
            }
            self.last_qscale_diff = 0;
        }

        self.record_mb(mb_xy);
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
            // MC per 4x4 block. ref_cache holds PICTURE IDS (C's ref2frm
            // space) since the pred_motion unification — resolve the DPB
            // picture by id; negative = no reference (use list0[0]).
            let refs = &self.refs;
            let fallback = self.ref_lists[0]
                .first()
                .and_then(|&k| self.refs.get(k))
                .map_or(0, |p| p.id);
            let pick = |ri: i8| -> Result<&Picture> {
                let want = if ri < 0 { fallback } else { ri as u64 };
                refs.iter()
                    .find(|p| p.id == want)
                    .or_else(|| refs.first())
                    .ok_or_else(|| {
                        Error::InvalidData("inter MB references a missing picture".into())
                    })
            };
            let cur = self.cur.as_mut().unwrap();
            for i in 0..16usize {
                let mv = self.mv_cache[0][SCAN8[i]];
                let prev = pick(self.ref_cache[0][SCAN8[i]])?;
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
                    let mv = self.mv_cache[0][SCAN8[0] + 8 * r + c];
                    let prev = pick(self.ref_cache[0][SCAN8[0] + 8 * r + c])?;
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
        pic.frame_num = self.slice_frame_num;
        pic.key = is_idr;
        pic.poc = self.compute_poc(is_idr);
        self.cur = Some(pic);
        self.slice_table = vec![0; self.mb_width * self.mb_height];
        self.got_mb = false;
        self.mb_skip_run = -1;
        if is_idr {
            self.refs.clear(); // IDR clears the DPB
        }
    }

    /// `ff_h264_pred_weight_table` (h264_parse.c:30) — explicit
    /// weights/offsets per list/ref; frame pictures, 420 chroma.
    fn parse_pred_weight_table(&mut self, pps: &Pps, gb: &mut Gb) -> Result<()> {
        let mut pwt = PredWeightTable::default();
        pwt.use_weight = 0;
        pwt.use_weight_chroma = 0;
        pwt.luma_log2_denom = gb.ue()? as i32;
        if pwt.luma_log2_denom > 7 {
            return Err(Error::InvalidData("luma_log2_weight_denom".into()));
        }
        let luma_def = 1i32 << pwt.luma_log2_denom;
        pwt.chroma_log2_denom = gb.ue()? as i32;
        if pwt.chroma_log2_denom > 7 {
            return Err(Error::InvalidData("chroma_log2_weight_denom".into()));
        }
        let chroma_def = 1i32 << pwt.chroma_log2_denom;
        let mut luma_w_any = [false; 2];
        let mut chroma_w_any = [false; 2];
        for list in 0..2usize {
            for i in 0..self.ref_counts[list] as usize {
                if gb.read_bit() == 1 {
                    let w = gb.se()?;
                    let o = gb.se()?;
                    if !(-128..=127).contains(&w) || !(-128..=127).contains(&o) {
                        return Err(Error::InvalidData("weight out of range".into()));
                    }
                    pwt.luma_w[list][i] = w;
                    pwt.luma_o[list][i] = o;
                    if w != luma_def || o != 0 {
                        luma_w_any[list] = true;
                    }
                } else {
                    pwt.luma_w[list][i] = luma_def;
                    pwt.luma_o[list][i] = 0;
                }
                for j in 0..2usize {
                    if gb.read_bit() == 1 {
                        let w = gb.se()?;
                        let o = gb.se()?;
                        if !(-128..=127).contains(&w) || !(-128..=127).contains(&o) {
                            return Err(Error::InvalidData("chroma weight".into()));
                        }
                        pwt.chroma_w[list][j][i] = w;
                        pwt.chroma_o[list][j][i] = o;
                        if w != chroma_def || o != 0 {
                            chroma_w_any[list] = true;
                        }
                    } else {
                        pwt.chroma_w[list][j][i] = chroma_def;
                        pwt.chroma_o[list][j][i] = 0;
                    }
                }
            }
            if self.slice_type_nos != 1 {
                break; // P: list 0 only
            }
        }
        pwt.use_weight_chroma = chroma_w_any.iter().any(|&b| b) as u8;
        pwt.use_weight = (luma_w_any.iter().any(|&b| b) || pwt.use_weight_chroma != 0) as u8;
        self.pwt = pwt;
        let _ = pps;
        Ok(())
    }

    /// `implicit_weight_table` (h264_slice.c:691), frame path: fills
    /// pwt.implicit[ref0][ref1] and flags use_weight = 2 (implicit).
    /// Call after build_ref_list (needs the POCs).
    fn implicit_weight_table(&mut self) {
        let cur_poc = self.cur.as_ref().map(|p| p.poc).unwrap_or(0);
        let r0 = &self.ref_lists[0];
        let r1 = &self.ref_lists[1];
        // Single-picture trivial case (C's early out).
        if self.ref_counts[0] == 1 && self.ref_counts[1] == 1 && !r0.is_empty() && !r1.is_empty() {
            let p0 = self.refs[r0[0]].poc as i64;
            let p1 = self.refs[r1[0]].poc as i64;
            if p0 + p1 == 2 * cur_poc as i64 {
                self.pwt.use_weight = 0;
                self.pwt.use_weight_chroma = 0;
                return;
            }
        }
        self.pwt.use_weight = 2;
        self.pwt.use_weight_chroma = 2;
        self.pwt.luma_log2_denom = 5;
        self.pwt.chroma_log2_denom = 5;
        self.pwt.implicit = [[32; 32]; 32];
        for (i0, &a) in r0.iter().enumerate() {
            let poc0 = self.refs[a].poc;
            for (i1, &b) in r1.iter().enumerate() {
                let poc1 = self.refs[b].poc;
                let mut w = 32i32;
                let td = (poc1 - poc0).clamp(-128, 127);
                if td != 0 {
                    let tb = (cur_poc - poc0).clamp(-128, 127);
                    let tx = (16384 + (td.abs() >> 1)) / td;
                    let dsf = (tb * tx + 32) >> 8;
                    if (-64..=128).contains(&dsf) {
                        w = 64 - dsf;
                    }
                }
                self.pwt.implicit[i0][i1] = w;
            }
        }
    }

    /// `ff_h264_init_poc` (h264_parse.c:280), frame-picture path. Uses
    /// and updates the decoder's POC state; the caller snapshots
    /// `poc_msb`/`lsb` for the prev_* updates at picture end (C does
    /// them in ff_h264_field_end).
    fn compute_poc(&mut self, _is_idr: bool) -> i32 {
        let sps = self.sps.clone().unwrap();
        let max_frame_num = 1i32 << sps.log2_max_frame_num;
        let frame_num = self.slice_frame_num as i32;

        let mut offset = self.poc_prev_frame_num_offset;
        if frame_num < self.poc_prev_frame_num {
            offset += max_frame_num;
        }
        self.poc_frame_num_offset = offset;

        let field_poc0;
        if sps.poc_type == 0 {
            let max_poc_lsb = 1i32 << sps.log2_max_poc_lsb;
            let mut prev_lsb = self.poc_prev_lsb;
            if prev_lsb < 0 {
                prev_lsb = self.sl_poc_lsb;
            }
            if self.sl_poc_lsb < prev_lsb && prev_lsb - self.sl_poc_lsb >= max_poc_lsb / 2 {
                self.poc_msb = self.poc_prev_msb + max_poc_lsb;
            } else if self.sl_poc_lsb > prev_lsb && prev_lsb - self.sl_poc_lsb < -max_poc_lsb / 2 {
                self.poc_msb = self.poc_prev_msb - max_poc_lsb;
            } else {
                self.poc_msb = self.poc_prev_msb;
            }
            field_poc0 = self.poc_msb + self.sl_poc_lsb;
            // field_poc[1] = field_poc[0] + delta_poc_bottom (frame);
            // pic_poc = min of the two.
            let f1 = field_poc0 + self.sl_delta_poc_bottom;
            return field_poc0.min(f1);
        } else if sps.poc_type == 1 {
            let mut abs_frame_num = if !sps.offset_for_ref_frame.is_empty() {
                offset + frame_num
            } else {
                0
            };
            let nal_ref_idc = self.cur_nal_ref_idc;
            if nal_ref_idc == 0 && abs_frame_num > 0 {
                abs_frame_num -= 1;
            }
            let cycle = sps.offset_for_ref_frame.len() as i32;
            let expected_delta_per_cycle = sps
                .offset_for_ref_frame
                .iter()
                .fold(0i64, |a, &v| a + v as i64);
            let mut expected: i64 = 0;
            if abs_frame_num > 0 {
                let cycle_cnt = (abs_frame_num - 1) / cycle;
                let in_cycle = (abs_frame_num - 1) % cycle;
                expected = cycle_cnt as i64 * expected_delta_per_cycle;
                for i in 0..=in_cycle {
                    expected += sps.offset_for_ref_frame[i as usize] as i64;
                }
            }
            if nal_ref_idc == 0 {
                expected += sps.offset_for_non_ref_pic as i64;
            }
            let f0 = expected + self.sl_delta_poc[0] as i64;
            let f1 = f0 + sps.offset_for_top_to_bottom as i64 + self.sl_delta_poc[1] as i64;
            return (f0.min(f1)) as i32;
        } else {
            let mut poc = 2 * (offset + frame_num);
            if self.cur_nal_ref_idc == 0 {
                poc -= 1;
            }
            field_poc0 = poc;
        }
        field_poc0
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
            self.cur_nal_ref_idc = nal.ref_idc;
            self.cur_is_b = self.slice_type_nos == 1;
            self.start_new_picture(nal.kind == 5);
            self.cur_is_ref = nal.ref_idc != 0;
            self.frame_num = self.slice_frame_num;
        }
        // One number per slice (C's current_slice / slice_table): MBs of
        // other slices are unavailable for prediction, and deblocking mode 2
        // stops at slice edges.
        self.slice_num += 1;
        self.slice_dbk.slice = self.slice_num;
        self.build_ref_list()?;
        // C resets the skip run per SLICE (decode_slice, h264_slice.c:2681)
        // — a later slice of the same picture starts with a fresh run.
        self.mb_skip_run = -1;
        // (frame_num kept from header for next comparison)
        self.got_mb = true;

        let mb_num = self.mb_width * self.mb_height;
        let mut mb_abs = first_mb;
        if self.pps.as_ref().is_some_and(|p| p.cabac) {
            // C's CABAC slice loop (h264_slice.c:2701-2782): align, init
            // the engine + states, per MB decode → hl_decode_mb →
            // end_of_slice (terminate bin).
            gb.align();
            let start = gb.index / 8;
            let bytes = ((gb.left() + 7) / 8) as usize;
            let end = (start + bytes).min(nal.rbsp.len());
            let mut cab = Cabac::new(&nal.rbsp[start..end])?;
            self.init_cabac_states();
            if std::env::var_os("H264_DUMP").is_some() {
                eprintln!(
                    "CABACSTART qp={} start={start} bytes={} nos={} idc={} rc0={}",
                    self.qscale,
                    end - start,
                    self.slice_type_nos,
                    self.cabac_init_idc,
                    self.ref_counts[0]
                );
            }
            loop {
                if mb_abs >= next_slice_idx || mb_abs >= mb_num {
                    break;
                }
                self.mb_x = mb_abs % self.mb_width;
                self.mb_y = mb_abs / self.mb_width;
                self.decode_mb_cabac(&mut cab).map_err(|e| {
                    eprintln!(
                        "H264DBG: CABAC MB err slice#{} kind={} first_mb={first_mb} at mb({},{}) bytes={}: {e}",
                        self.slice_num,
                        nal.kind,
                        self.mb_x,
                        self.mb_y,
                        cab.bytestream
                    );
                    e
                })?;
                self.hl_decode_mb(mb_abs)?;
                let eos = cab.terminate() != 0;
                if cab.overread() {
                    return Err(Error::InvalidData("cabac slice overread".into()));
                }
                mb_abs += 1;
                if eos {
                    break;
                }
            }
            return Ok(true);
        }
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
        let cur_poc = pic.poc;
        let cur_key = pic.key;

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
        // Picture-end POC state (ff_h264_field_end, h264_slice.c:459-463):
        // prev msb/lsb only for reference pictures, offsets always.
        if self.cur_nal_ref_idc != 0 {
            self.poc_prev_msb = self.poc_msb;
            self.poc_prev_lsb = self.sl_poc_lsb;
        }
        self.poc_prev_frame_num_offset = self.poc_frame_num_offset;
        self.poc_prev_frame_num = self.slice_frame_num as i32;
        // Output reordering (h264_select_output_frame, h264_slice.c:1313).
        self.select_output_frame(frame, cur_poc, cur_key);
        self.got_mb = false;
        Ok(())
    }

    /// `h264_select_output_frame` — push the finished frame into the
    /// delayed buffer and emit whatever is ready. `mmco_reset` is always
    /// false here (explicit marking is Phase E).
    fn select_output_frame(&mut self, frame: Frame, cur_poc: i32, cur_key: bool) {
        let sps = self.sps.clone().unwrap();
        if sps.bitstream_restriction {
            self.has_b_frames = self.has_b_frames.max(sps.num_reorder_frames);
        }

        // last_pocs shift (C's running window of recent POCs).
        let mut i = 0usize;
        let mut out_of_order;
        loop {
            if i == self.last_pocs.len() || cur_poc < self.last_pocs[i] {
                if i != 0 {
                    self.last_pocs[i - 1] = cur_poc;
                }
                break;
            } else if i != 0 {
                self.last_pocs[i - 1] = self.last_pocs[i];
            }
            i += 1;
        }
        out_of_order = self.last_pocs.len() - i;
        // C: B pictures or a POC gap > 2 raise the reorder need.
        if self.cur_is_b
            || (self.last_pocs[self.last_pocs.len() - 2] > i32::MIN
                && self.last_pocs[self.last_pocs.len() - 1] as i64
                    - self.last_pocs[self.last_pocs.len() - 2] as i64
                    > 2)
        {
            out_of_order = out_of_order.max(1);
        }
        if out_of_order == self.last_pocs.len() {
            // Invalid POC ordering — reset the window.
            for p in self.last_pocs.iter_mut().skip(1) {
                *p = i32::MIN;
            }
            self.last_pocs[0] = cur_poc;
        } else if self.has_b_frames < out_of_order && !sps.bitstream_restriction {
            self.has_b_frames = out_of_order;
        }

        self.delayed.push(DelayedFrame {
            frame,
            poc: cur_poc,
            key: cur_key,
            mmco_reset: false,
        });
        let pics = self.delayed.len();

        // Pick the minimum-POC candidate (stopping at key/reset frames).
        let mut out_idx = 0usize;
        for j in 1..pics {
            if self.delayed[j].key || self.delayed[j].mmco_reset {
                break;
            }
            if self.delayed[j].poc < self.delayed[out_idx].poc {
                out_idx = j;
            }
        }
        if self.has_b_frames == 0 && (self.delayed[0].key || self.delayed[0].mmco_reset) {
            self.next_outputed_poc = i32::MIN + 1;
        }
        let out_of_order = self.delayed[out_idx].poc < self.next_outputed_poc;

        if out_of_order || pics > self.has_b_frames {
            let out = self.delayed.remove(out_idx);
            if !out_of_order {
                if out_idx == 0 && self.delayed.first().is_some_and(|d| d.key || d.mmco_reset) {
                    self.next_outputed_poc = i32::MIN + 1;
                } else {
                    self.next_outputed_poc = out.poc;
                }
            }
            self.pending.push_back(out.frame);
        }
    }

    /// RefPicList init + modification (h264_refs.c:
    /// h264_initialise_ref_list + ff_h264_build_ref_list, short-term
    /// only). P: descending PicNum. B: POC-sorted — list 0 = past
    /// pictures nearest-first then future nearest-first; list 1 the
    /// mirror image; if both lists come out identical and non-trivial,
    /// list1[0]/[1] are swapped so list1[0] differs from list0[0]. The
    /// per-list modification ops then move the named picture to each
    /// index (PicNum-pred walk identical for both lists).
    fn build_ref_list(&mut self) -> Result<()> {
        self.ref_lists[0].clear();
        self.ref_lists[1].clear();
        if self.slice_type_nos == 2 {
            return Ok(());
        }
        let max_fn = 1i64 << self.sps.as_ref().unwrap().log2_max_frame_num;
        let cur_fn = self.slice_frame_num as i64;
        let pic_num = |f: u32| -> i64 {
            let f = f as i64;
            if f > cur_fn { f - max_fn } else { f }
        };
        let mut lists: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
        if self.slice_type_nos == 1 {
            // B: add_sorted (h264_refs.c:103) — dir=1 picks the largest
            // POC ≤ cur first (past, nearest first); dir=0 the smallest
            // above cur (future, nearest first).
            let cur_poc = self.cur.as_ref().map(|p| p.poc).unwrap_or(0);
            for list in 0..2usize {
                let mut sorted: Vec<usize> = Vec::new();
                for dir in [1 ^ list, 0 ^ list] {
                    loop {
                        let mut best = if dir == 1 { i32::MIN } else { i32::MAX };
                        let mut best_i = None;
                        for (i, r) in self.refs.iter().enumerate() {
                            let poc = r.poc;
                            if ((poc > cur_poc) ^ (dir == 1)) == false {
                                continue;
                            }
                            if (poc < best) ^ (dir == 1) == false {
                                continue;
                            }
                            best = poc;
                            best_i = Some(i);
                        }
                        match best_i {
                            Some(i) => sorted.push(i),
                            None => break,
                        }
                    }
                }
                lists[list] = sorted;
            }
            // C: identical non-trivial lists → swap list1[0]/[1].
            let (l0, l1) = (&lists[0], &lists[1]);
            if l0.len() == l1.len() && l1.len() > 1 && l0 == l1 {
                lists[1].swap(0, 1);
            }
        } else {
            lists[0] = (0..self.refs.len()).collect();
            lists[0].sort_by_key(|&i| std::cmp::Reverse(pic_num(self.refs[i].frame_num)));
        }

        // Per-list modification (ff_h264_build_ref_list's pred walk).
        for list in 0..self.list_count {
            let ops = if list == 0 {
                &self.reorder_ops0
            } else {
                &self.reorder_ops1
            };
            let mut list_v = lists[list].clone();
            let mut pred = cur_fn;
            for (idx, &(op, val)) in ops.iter().enumerate() {
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
                let Some(pos) = list_v
                    .iter()
                    .position(|&i| pic_num(self.refs[i].frame_num) == want)
                else {
                    return Err(Error::InvalidData(
                        "reference picture missing during reorder".into(),
                    ));
                };
                let r = list_v.remove(pos);
                list_v.insert(idx.min(list_v.len()), r);
            }
            list_v.truncate(self.ref_counts[list] as usize);
            self.ref_lists[list] = list_v;
        }
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
            // EOF (send_next_delayed_frame): emit the reorder buffer in
            // min-POC order (scan stops at key/reset frames).
            while !self.delayed.is_empty() {
                let mut out_idx = 0usize;
                for j in 1..self.delayed.len() {
                    if self.delayed[j].key || self.delayed[j].mmco_reset {
                        break;
                    }
                    if self.delayed[j].poc < self.delayed[out_idx].poc {
                        out_idx = j;
                    }
                }
                let out = self.delayed.remove(out_idx);
                self.pending.push_back(out.frame);
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
