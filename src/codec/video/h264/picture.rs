// ---------------------------------------------------------------------
// Picture
// ---------------------------------------------------------------------

use crate::codec::video::h264::{MbDeblock, PART_16X16};

#[derive(Clone)]
pub(crate) struct Picture {
    pub(crate) w: usize,
    pub(crate) h: usize,
    pub(crate) y: Vec<u8>,
    pub(crate) cb: Vec<u8>,
    pub(crate) cr: Vec<u8>,
    pub mb_type: Vec<u32>,
    pub nnz: Vec<[u8; 48]>,
    /// Per-list MVs on the b_stride grid (C's motion_val[2]; list 1 only
    /// written for B pictures).
    pub mv: [Vec<[i16; 2]>; 2],
    /// Per-list RAW ref_idx, 4 per MB (C's ref_index[2]).
    pub ref_index: [Vec<i8>; 2],
    /// |mvd| per 4x4 per list (CABAC mvd_cache borders, h264_mvpred.h:
    /// 838) — the b_stride grid like `mv`; only the bottom row / right
    /// column of each MB are ever read back.
    pub mvd: [Vec<[u8; 2]>; 2],
    pub(crate) qscale: Vec<u8>,
    /// intra4x4 modes: bottom row (4) + right column (4) per MB — C's
    /// `mb2br`-indexed `intra4x4_pred_mode` slots the caches read.
    pub mb_i4x4: Vec<[i8; 8]>,
    /// frame_num of this picture (PicNum derivation for the ref list).
    pub(crate) frame_num: u32,
    /// Picture order count (top field for frame pictures).
    pub(crate) poc: i32,
    /// IDR (AV_FRAME_FLAG_KEY) — the reorder buffer flushes on it.
    pub(crate) key: bool,
    /// Unique id — the deblocking filter compares references by picture
    /// identity (C's `ref2frm`), not by per-slice ref_idx.
    pub id: u64,
    /// ff_h264_direct_ref_list_init stores the slice's ref counts and
    /// RefPicList "POCs" (4*frame_num + reference&3) so a future B
    /// picture's temporal direct can map its colocated references.
    pub ref_count_pic: [usize; 2],
    pub ref_poc: [[i32; 32]; 2],
    /// Per-list per 8x8 (4 per MB): id of the referenced picture, -1 =
    /// none (C's ref2frm-mapped ref_pic).
    pub ref_pic: [Vec<i32>; 2],
    /// Per MB: coded_block_pattern (C's `cbp_table`; CABAC ORs the
    /// DC-coded bits 0x40/0x80/0x100 into it — h264_cabac.c:1707),
    /// inter partition shape, RAW chroma pred mode (CABAC ctx) and the
    /// P_Skip flag (CABAC skip ctx), plus the slice's deblocking params.
    pub(crate) cbp: Vec<u16>,
    pub(crate) part: Vec<u8>,
    pub(crate) chroma_pred: Vec<u8>,
    pub(crate) skip: Vec<bool>,
    /// B_Direct MB marker (C's MB_TYPE_DIRECT2 flag — the direct-mode
    /// mb_type ctx and the loopfilter read it).
    pub(crate) direct: Vec<bool>,
    /// Per 8x8-quadrant direct flags (C's direct_table): all-true for
    /// B_Direct_16x16, per-sub for B_8x8, else false.
    pub(crate) direct8: Vec<[bool; 4]>,
    /// list_count of the picture's slices (1 = P, 2 = B) — the loop
    /// filter's check_mv reads list 1 only for B pictures.
    pub(crate) list_count: usize,
    pub(crate) dbk: Vec<MbDeblock>,
}

impl Picture {
    pub fn new(mb_w: usize, mb_h: usize) -> Picture {
        let w = mb_w * 16;
        let h = mb_h * 16;
        let b_stride = mb_w * 4 + 1;
        Picture {
            w,
            h,
            y: vec![0u8; w * h],
            cb: vec![0u8; (w / 2) * (h / 2)],
            cr: vec![0u8; (w / 2) * (h / 2)],
            mb_type: vec![0; mb_w * mb_h + 1],
            nnz: vec![[0; 48]; mb_w * mb_h + 1],
            mv: [
                vec![[0, 0]; b_stride * (mb_h * 4 + 1)],
                vec![[0, 0]; b_stride * (mb_h * 4 + 1)],
            ],
            ref_index: [
                vec![-1; (mb_w * 4 + 1) * (mb_h * 4 + 1)],
                vec![-1; (mb_w * 4 + 1) * (mb_h * 4 + 1)],
            ],
            mvd: [
                vec![[0, 0]; b_stride * (mb_h * 4 + 1)],
                vec![[0, 0]; b_stride * (mb_h * 4 + 1)],
            ],
            qscale: vec![0; mb_w * mb_h + 1],
            mb_i4x4: vec![[-1; 8]; mb_w * mb_h + 1],
            frame_num: 0,
            poc: 0,
            key: false,
            id: 0,
            ref_count_pic: [0, 0],
            ref_poc: [[0; 32]; 2],
            ref_pic: [
                vec![-1; 4 * (mb_w * mb_h + 1)],
                vec![-1; 4 * (mb_w * mb_h + 1)],
            ],
            cbp: vec![0; mb_w * mb_h + 1],
            part: vec![PART_16X16; mb_w * mb_h + 1],
            chroma_pred: vec![0; mb_w * mb_h + 1],
            skip: vec![false; mb_w * mb_h + 1],
            direct: vec![false; mb_w * mb_h + 1],
            direct8: vec![[false; 4]; mb_w * mb_h + 1],
            list_count: 1,
            dbk: vec![MbDeblock::default(); mb_w * mb_h + 1],
        }
    }
    pub fn sample_y(&self, x: i32, y: i32) -> u8 {
        *self
            .y
            .get(
                y.clamp(0, self.h as i32 - 1) as usize * self.w
                    + x.clamp(0, self.w as i32 - 1) as usize,
            )
            .unwrap_or(&0)
    }

    pub fn sample_c(&self, p: &[u8], x: i32, y: i32) -> u8 {
        *p.get(
            y.clamp(0, self.h as i32 / 2 - 1) as usize * (self.w / 2)
                + x.clamp(0, self.w as i32 / 2 - 1) as usize,
        )
        .unwrap_or(&0)
    }
}
