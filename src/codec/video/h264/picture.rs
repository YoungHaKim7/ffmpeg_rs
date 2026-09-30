// ---------------------------------------------------------------------
// Picture
// ---------------------------------------------------------------------

use crate::codec::video::h264::{MbDeblock, PART_16X16};

pub(crate) struct Picture {
    pub(crate) w: usize,
    pub(crate) h: usize,
    pub(crate) y: Vec<u8>,
    pub(crate) cb: Vec<u8>,
    pub(crate) cr: Vec<u8>,
    pub mb_type: Vec<u32>,
    pub nnz: Vec<[u8; 48]>,
    pub mv: Vec<[i16; 2]>,  // b_stride = mb_w*4 (+1 padding row)
    pub ref_index: Vec<i8>, // 4 per MB
    pub(crate) qscale: Vec<u8>,
    /// intra4x4 modes: bottom row (4) + right column (4) per MB — C's
    /// `mb2br`-indexed `intra4x4_pred_mode` slots the caches read.
    pub mb_i4x4: Vec<[i8; 8]>,
    /// frame_num of this picture (PicNum derivation for the ref list).
    pub(crate) frame_num: u32,
    /// Unique id — the deblocking filter compares references by picture
    /// identity (C's `ref2frm`), not by per-slice ref_idx.
    pub id: u64,
    /// Per 8x8 (4 per MB): id of the referenced picture, -1 = none.
    pub ref_pic: Vec<i32>,
    /// Per MB: coded_block_pattern (C's `cbp_table`), inter partition
    /// shape, and the slice's deblocking parameters.
    pub(crate) cbp: Vec<u8>,
    pub(crate) part: Vec<u8>,
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
            mv: vec![[0, 0]; b_stride * (mb_h * 4 + 1)],
            ref_index: vec![-1; (mb_w * 4 + 1) * (mb_h * 4 + 1)],
            qscale: vec![0; mb_w * mb_h + 1],
            mb_i4x4: vec![[-1; 8]; mb_w * mb_h + 1],
            frame_num: 0,
            id: 0,
            ref_pic: vec![-1; 4 * (mb_w * mb_h + 1)],
            cbp: vec![0; mb_w * mb_h + 1],
            part: vec![PART_16X16; mb_w * mb_h + 1],
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
