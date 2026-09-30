// 16x16/chroma mode space: 0=DC,1=plane? — chroma: 0=DC,1=H,2=V,3=plane;
// luma16: 0=V,1=H,2=DC,3=plane. The check tables map through; see
// check_intra_pred_mode.

// ---------------------------------------------------------------------
// VLCs + level table (ff_h264_decode_init_vlc, h264_cavlc.c:329-380)
// ---------------------------------------------------------------------

use std::sync::OnceLock;

use super::table_rows;
use super::tables::*;
use super::{Gb, LEVEL_TAB_BITS};
use crate::fferror::{Error, Result};

pub(super) struct Vlc {
    max_len: u32,
    tab: Vec<u32>,
}

impl Vlc {
    fn new(lens: &[u8], codes: &[u8]) -> Vlc {
        let max_len = lens.iter().copied().max().unwrap_or(0) as u32;
        let mut tab = vec![0u32; 1usize << max_len];
        for i in 0..lens.len() {
            let l = lens[i] as u32;
            if l == 0 || l > max_len {
                continue;
            }
            let lo = (codes[i] as usize) << (max_len - l);
            for t in &mut tab[lo..lo + (1usize << (max_len - l))] {
                *t = (l << 16) | i as u32;
            }
        }
        Vlc { max_len, tab }
    }
    pub(super) fn get(&self, gb: &mut Gb) -> Result<u32> {
        let e = self.tab[gb.peek(self.max_len) as usize];
        let len = e >> 16;
        if len == 0 {
            return Err(Error::InvalidData("invalid CAVLC code".into()));
        }
        gb.skip(len);
        Ok(e & 0xffff)
    }
}

pub(super) struct Cavlc {
    pub(super) coeff_token: [Vlc; 4],
    pub(super) chroma_dc_coeff_token: Vlc,
    pub(super) total_zeros: Vec<Vlc>,  // index by total_coeff 1..15
    pub(super) chroma_dc_tz: Vec<Vlc>, // 1..3
    pub(super) run: Vec<Vlc>,          // 1..6
    pub(super) run7: Vlc,
    /// `cavlc_level_tab` (h264_cavlc.c:289): [suffix][peek8] = (code, len).
    pub(super) level_tab: Vec<[[i16; 2]; 256]>,
}

pub(super) fn cavlc() -> &'static Cavlc {
    static T: OnceLock<Cavlc> = OnceLock::new();
    T.get_or_init(|| {
        let ct = table_rows("COEFF_TOKEN", 4, 68);
        let coeff_token = std::array::from_fn(|i| Vlc::new(&ct[i].0, &ct[i].1));

        let chroma_dc_coeff_token = {
            let mut l = CHROMA_DC_TOKEN_LEN.to_vec();
            let mut b = CHROMA_DC_TOKEN_BITS.to_vec();
            l.truncate(20);
            b.truncate(20);
            Vlc::new(&l, &b)
        };

        let tz_rows = table_rows("TOTAL_ZEROS", 16, 16);
        let total_zeros: Vec<Vlc> = (1..16)
            // C: total_zeros_vlc[i + 1] built from row i ⇒ tc uses row tc−1.
            .map(|tc| Vlc::new(&tz_rows[tc - 1].0, &tz_rows[tc - 1].1))
            .collect();

        let chroma_dc_tz: Vec<Vlc> = (1..4)
            .map(|tc| {
                // rows of the [3][4] table
                let (l, b): (Vec<u8>, Vec<u8>) = match tc {
                    1 => (CHROMA_DC_TZ_LEN_0.to_vec(), CHROMA_DC_TZ_BITS_0.to_vec()),
                    2 => (CHROMA_DC_TZ_LEN_1.to_vec(), CHROMA_DC_TZ_BITS_1.to_vec()),
                    _ => (CHROMA_DC_TZ_LEN_2.to_vec(), CHROMA_DC_TZ_BITS_2.to_vec()),
                };
                Vlc::new(&l, &b)
            })
            .collect();

        // C: run_vlc[i + 1] from row i ⇒ indexed by zeros_left directly
        // (run[z] decodes zeros_left = z + 1's runs; see the use site).
        let run: Vec<Vlc> = (0..6)
            .map(|r| {
                let (l, b): (Vec<u8>, Vec<u8>) = match r {
                    0 => (RUN_LEN_0.to_vec(), RUN_BITS_0.to_vec()),
                    1 => (RUN_LEN_1.to_vec(), RUN_BITS_1.to_vec()),
                    2 => (RUN_LEN_2.to_vec(), RUN_BITS_2.to_vec()),
                    3 => (RUN_LEN_3.to_vec(), RUN_BITS_3.to_vec()),
                    4 => (RUN_LEN_4.to_vec(), RUN_BITS_4.to_vec()),
                    _ => (RUN_LEN_5.to_vec(), RUN_BITS_5.to_vec()),
                };
                Vlc::new(&l, &b)
            })
            .collect();
        let run7 = Vlc::new(&RUN_LEN_6, &RUN_BITS_6);

        // init_cavlc_level_tab (h264_cavlc.c:289)
        let mut level_tab = vec![[[0i16; 2]; 256]; 7];
        for sl in 0..7usize {
            for i in 0..256usize {
                // prefix = LEVEL_TAB_BITS - av_log2(2*i); av_log2(0) is
                // undefined in C but i>=1 semantics: av_log2(2*i) with
                // i=0 → log2(0) → -inf; C's av_log2(0) returns 0. Then
                // prefix = 8. Use the same convention.
                // C: prefix = LEVEL_TAB_BITS - av_log2(2*i); ff's
                // av_log2(0) = 0 ⇒ i=0 → prefix 8.
                let av_log2_2i = if i == 0 {
                    0
                } else {
                    31 - ((2 * i as u32).leading_zeros() as i32)
                };
                let prefix = LEVEL_TAB_BITS as i32 - av_log2_2i;
                let (mut code, len) = if prefix + 1 + sl as i32 <= LEVEL_TAB_BITS as i32 {
                    let log2i = if i == 0 {
                        0
                    } else {
                        31 - ((i as u32).leading_zeros() as i32)
                    };
                    let c = ((prefix as i64) << sl) + ((i >> (log2i - sl as i32).max(0)) as i64)
                        - (1i64 << sl);
                    let ci = c as i32;
                    let mask = -(ci & 1);
                    let ci = (((2 + ci) >> 1) ^ mask) - mask;
                    (ci as i16, (prefix + 1 + sl as i32) as i16)
                } else if prefix + 1 <= LEVEL_TAB_BITS as i32 {
                    ((prefix + 100) as i16, (prefix + 1) as i16)
                } else {
                    ((LEVEL_TAB_BITS as i32 + 100) as i16, LEVEL_TAB_BITS as i16)
                };
                let _ = &mut code;
                level_tab[sl][i] = [code, len];
            }
        }

        Cavlc {
            coeff_token,
            chroma_dc_coeff_token,
            total_zeros,
            chroma_dc_tz,
            run,
            run7,
            level_tab,
        }
    })
}
