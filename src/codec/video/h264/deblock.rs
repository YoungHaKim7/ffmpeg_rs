//! H.264 in-loop deblocking filter — port of `h264_loopfilter.c`
//! (`ff_h264_filter_mb`, `filter_mb_dir`, `check_mv`, the `filter_mb_edge*`
//! helpers and their alpha/beta/tc0 tables), the caches it reads
//! (`fill_filter_caches` / `fill_filter_caches_inter`, h264_slice.c), and
//! the four pixel kernels of `h264dsp_template.c`. Frame pictures, 8-bit
//! 4:2:0, one reference list (P/I slices) — the scope of the decoder.
//!
//! C filters each MB row right after decoding it, saving the unfiltered
//! top/left borders (`backup_mb_border`) so intra prediction of the next
//! row still sees pre-filter samples. That is exactly the spec's model —
//! intra prediction uses samples *before* deblocking — so this port
//! filters the whole picture once all its MBs are decoded, in the same
//! raster MB order and with the same per-MB edge order (vertical edges,
//! then horizontal), which yields identical output.
//!
//! The asm-only `loop_filter_strength` fast path (`h264_filter_mb_fast`)
//! is an optimisation of this same computation; C falls back to
//! `ff_h264_filter_mb` without it, which is what is ported here.

use super::{
    deblock_tables::{ALPHA_TABLE, BETA_TABLE, TC0_TABLE},
    picture::Picture,
    {CHROMA_QP8, MB_INTER, MB_INTRA4X4, MB_INTRA16X16, MB_PCM},
};

/// Inter partition shape of an MB, for the edge masks (C's
/// `MB_TYPE_16x16 / 16x8 / 8x16 / 8x8` bits). P_Skip is 16x16.
pub(super) const PART_16X16: u8 = 0;
pub(super) const PART_16X8: u8 = 1;
pub(super) const PART_8X16: u8 = 2;
pub(super) const PART_8X8: u8 = 3;

/// Per-MB copy of the slice state the filter needs. `slice == 0` marks an
/// MB that was never decoded (C's `slice_table == 0xFFFF`).
#[derive(Clone, Copy, Default)]
pub(super) struct MbDeblock {
    /// C's internal `sl->deblocking_filter`: 0 off, 1 on (also across
    /// slice edges), 2 on but not across slice edges. The bitstream's
    /// `disable_deblocking_filter_idc` 0/1 is swapped into 1/0.
    pub mode: u8,
    /// `slice_alpha_c0_offset` / `slice_beta_offset` (the ×2 values).
    pub alpha: i32,
    pub beta: i32,
    /// PPS `chroma_qp_index_offset` (Cb, Cr).
    pub cqp_off: [i32; 2],
    /// Slice identity for the mode-2 slice-edge test.
    pub slice: usize,
}

fn is_intra(t: u32) -> bool {
    t == MB_INTRA4X4 || t == MB_INTRA16X16 || t == MB_PCM
}

fn chroma_qp(off: i32, qp: i32) -> i32 {
    CHROMA_QP8[(qp + off).clamp(0, 51) as usize] as i32
}

// ---------------------------------------------------------------------
// Pixel kernels (h264dsp_template.c, BIT_DEPTH 8). `xs` steps across the
// edge, `ys` along it; `inner` = samples per bS segment (4 luma, 2 chroma).
// ---------------------------------------------------------------------

fn at(pos: isize, k: isize, s: isize) -> usize {
    (pos + k * s) as usize
}

fn loop_filter_luma(
    p: &mut [u8],
    pos: isize,
    xs: isize,
    ys: isize,
    alpha: i32,
    beta: i32,
    tc0: &[i8; 4],
) {
    let mut pix = pos;
    for &t in tc0 {
        let tc_orig = t as i32;
        if tc_orig < 0 {
            pix += 4 * ys;
            continue;
        }
        for _ in 0..4 {
            let p0 = p[at(pix, -1, xs)] as i32;
            let p1 = p[at(pix, -2, xs)] as i32;
            let p2 = p[at(pix, -3, xs)] as i32;
            let q0 = p[at(pix, 0, xs)] as i32;
            let q1 = p[at(pix, 1, xs)] as i32;
            let q2 = p[at(pix, 2, xs)] as i32;
            if (p0 - q0).abs() < alpha && (p1 - p0).abs() < beta && (q1 - q0).abs() < beta {
                let mut tc = tc_orig;
                if (p2 - p0).abs() < beta {
                    if tc_orig != 0 {
                        p[at(pix, -2, xs)] = (p1
                            + (((p2 + ((p0 + q0 + 1) >> 1)) >> 1) - p1).clamp(-tc_orig, tc_orig))
                            as u8;
                    }
                    tc += 1;
                }
                if (q2 - q0).abs() < beta {
                    if tc_orig != 0 {
                        p[at(pix, 1, xs)] = (q1
                            + (((q2 + ((p0 + q0 + 1) >> 1)) >> 1) - q1).clamp(-tc_orig, tc_orig))
                            as u8;
                    }
                    tc += 1;
                }
                let delta = ((((q0 - p0) * 4) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
                p[at(pix, -1, xs)] = (p0 + delta).clamp(0, 255) as u8;
                p[at(pix, 0, xs)] = (q0 - delta).clamp(0, 255) as u8;
            }
            pix += ys;
        }
    }
}

fn loop_filter_luma_intra(p: &mut [u8], pos: isize, xs: isize, ys: isize, alpha: i32, beta: i32) {
    let mut pix = pos;
    for _ in 0..16 {
        let p2 = p[at(pix, -3, xs)] as i32;
        let p1 = p[at(pix, -2, xs)] as i32;
        let p0 = p[at(pix, -1, xs)] as i32;
        let q0 = p[at(pix, 0, xs)] as i32;
        let q1 = p[at(pix, 1, xs)] as i32;
        let q2 = p[at(pix, 2, xs)] as i32;
        if (p0 - q0).abs() < alpha && (p1 - p0).abs() < beta && (q1 - q0).abs() < beta {
            if (p0 - q0).abs() < ((alpha >> 2) + 2) {
                if (p2 - p0).abs() < beta {
                    let p3 = p[at(pix, -4, xs)] as i32;
                    p[at(pix, -1, xs)] = ((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3) as u8;
                    p[at(pix, -2, xs)] = ((p2 + p1 + p0 + q0 + 2) >> 2) as u8;
                    p[at(pix, -3, xs)] = ((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3) as u8;
                } else {
                    p[at(pix, -1, xs)] = ((2 * p1 + p0 + q1 + 2) >> 2) as u8;
                }
                if (q2 - q0).abs() < beta {
                    let q3 = p[at(pix, 3, xs)] as i32;
                    p[at(pix, 0, xs)] = ((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3) as u8;
                    p[at(pix, 1, xs)] = ((p0 + q0 + q1 + q2 + 2) >> 2) as u8;
                    p[at(pix, 2, xs)] = ((2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) >> 3) as u8;
                } else {
                    p[at(pix, 0, xs)] = ((2 * q1 + q0 + p1 + 2) >> 2) as u8;
                }
            } else {
                p[at(pix, -1, xs)] = ((2 * p1 + p0 + q1 + 2) >> 2) as u8;
                p[at(pix, 0, xs)] = ((2 * q1 + q0 + p1 + 2) >> 2) as u8;
            }
        }
        pix += ys;
    }
}

fn loop_filter_chroma(
    p: &mut [u8],
    pos: isize,
    xs: isize,
    ys: isize,
    alpha: i32,
    beta: i32,
    tc0: &[i8; 4],
) {
    let mut pix = pos;
    for &t in tc0 {
        // C: tc = ((tc0[i] - 1U) << 0) + 1 — the edge helper already added
        // the +1, so tc <= 0 exactly when bS was 0.
        let tc = t as i32;
        if tc <= 0 {
            pix += 2 * ys;
            continue;
        }
        for _ in 0..2 {
            let p0 = p[at(pix, -1, xs)] as i32;
            let p1 = p[at(pix, -2, xs)] as i32;
            let q0 = p[at(pix, 0, xs)] as i32;
            let q1 = p[at(pix, 1, xs)] as i32;
            if (p0 - q0).abs() < alpha && (p1 - p0).abs() < beta && (q1 - q0).abs() < beta {
                let delta = (((q0 - p0) * 4 + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
                p[at(pix, -1, xs)] = (p0 + delta).clamp(0, 255) as u8;
                p[at(pix, 0, xs)] = (q0 - delta).clamp(0, 255) as u8;
            }
            pix += ys;
        }
    }
}

fn loop_filter_chroma_intra(p: &mut [u8], pos: isize, xs: isize, ys: isize, alpha: i32, beta: i32) {
    let mut pix = pos;
    for _ in 0..8 {
        let p0 = p[at(pix, -1, xs)] as i32;
        let p1 = p[at(pix, -2, xs)] as i32;
        let q0 = p[at(pix, 0, xs)] as i32;
        let q1 = p[at(pix, 1, xs)] as i32;
        if (p0 - q0).abs() < alpha && (p1 - p0).abs() < beta && (q1 - q0).abs() < beta {
            p[at(pix, -1, xs)] = ((2 * p1 + p0 + q1 + 2) >> 2) as u8;
            p[at(pix, 0, xs)] = ((2 * q1 + q0 + p1 + 2) >> 2) as u8;
        }
        pix += ys;
    }
}

// ---------------------------------------------------------------------
// Edge helpers (filter_mb_edge{v,h} / edgec{v,h}): `a`/`b` carry the +52
// table bias plus the slice offset, exactly as in C.
// ---------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn edge(
    p: &mut [u8],
    pos: isize,
    xs: isize,
    ys: isize,
    bs: &[i16; 4],
    qp: i32,
    a: i32,
    b: i32,
    intra: bool,
    chroma: bool,
) {
    let index_a = (qp + a) as usize;
    let alpha = ALPHA_TABLE[index_a] as i32;
    let beta = BETA_TABLE[(qp + b) as usize] as i32;
    if alpha == 0 || beta == 0 {
        return;
    }
    if bs[0] < 4 || !intra {
        let add = chroma as i8;
        let tc = [
            TC0_TABLE[index_a][bs[0] as usize] + add,
            TC0_TABLE[index_a][bs[1] as usize] + add,
            TC0_TABLE[index_a][bs[2] as usize] + add,
            TC0_TABLE[index_a][bs[3] as usize] + add,
        ];
        if chroma {
            loop_filter_chroma(p, pos, xs, ys, alpha, beta, &tc);
        } else {
            loop_filter_luma(p, pos, xs, ys, alpha, beta, &tc);
        }
    } else if chroma {
        loop_filter_chroma_intra(p, pos, xs, ys, alpha, beta);
    } else {
        loop_filter_luma_intra(p, pos, xs, ys, alpha, beta);
    }
}

// ---------------------------------------------------------------------
// Per-MB filter (ff_h264_filter_mb + filter_mb_dir, frame pictures).
// ---------------------------------------------------------------------

/// The 8-wide filter caches (C's scan8 layout): row 0 = top neighbour's
/// bottom row, column 3 = left neighbour's right column, rows 1..4 ×
/// cols 4..7 = this MB.
struct Caches {
    nnz: [u8; 40],
    refs: [[i32; 40]; 2],
    mv: [[[i16; 2]; 40]; 2],
    list_count: usize,
}

fn check_mv(c: &Caches, b: usize, bn: usize, mvy_limit: i32) -> i16 {
    let mvdiff = |list: usize| -> bool {
        ((c.mv[list][b][0] as i32 - c.mv[list][bn][0] as i32 + 3) as u32 >= 7)
            | ((c.mv[list][b][1] as i32 - c.mv[list][bn][1] as i32).abs() >= mvy_limit)
    };
    let mut v = c.refs[0][b] != c.refs[0][bn];
    if !v && c.refs[0][b] != -1 {
        v = mvdiff(0);
    }
    if c.list_count == 2 {
        if !v {
            v = (c.refs[1][b] != c.refs[1][bn]) | mvdiff(1);
        }
        if v {
            // Different-picture fallback: same picture across lists
            // (b_list0 vs bn_list1) still needs the MV comparison.
            if c.refs[0][b] != c.refs[1][bn] || c.refs[1][b] != c.refs[0][bn] {
                return 1;
            }
            let cross = |la: usize, lb: usize| -> bool {
                ((c.mv[la][b][0] as i32 - c.mv[lb][bn][0] as i32 + 3) as u32 >= 7)
                    | ((c.mv[la][b][1] as i32 - c.mv[lb][bn][1] as i32).abs() >= mvy_limit)
            };
            return (cross(0, 1) | cross(1, 0)) as i16;
        }
    }
    v as i16
}

/// `mask_edge_tab[dir][(mb_type>>3)&7]` over the partition shape.
fn mask_edge(dir: usize, part: u8) -> usize {
    match (dir, part) {
        (_, PART_16X16) => 3,
        (0, PART_16X8) => 3,
        (0, PART_8X16) => 1,
        (1, PART_16X8) => 1,
        (1, PART_8X16) => 3,
        _ => 0, // 8x8 (and intra: masks unused)
    }
}

/// `mb_type & (MB_TYPE_16x16 | (MB_TYPE_8x16 >> dir))` — 16x16 or the
/// partition that keeps whole columns (dir 0: 8x16) / rows (dir 1: 16x8).
fn par0(dir: usize, part: u8) -> bool {
    part == PART_16X16 || (dir == 0 && part == PART_8X16) || (dir == 1 && part == PART_16X8)
}

/// Deblock the whole picture in raster MB order (C's per-row
/// `loop_filter`, `h264_slice.c:2562`).
pub(super) fn filter_picture(pic: &mut Picture, mb_w: usize, mb_h: usize) {
    for mb_y in 0..mb_h {
        for mb_x in 0..mb_w {
            filter_mb(pic, mb_w, mb_x, mb_y);
        }
    }
}

fn filter_mb(pic: &mut Picture, mb_w: usize, mb_x: usize, mb_y: usize) {
    let mb_xy = mb_x + mb_y * mb_w;
    let d = pic.dbk[mb_xy];
    if d.slice == 0 || d.mode == 0 {
        return;
    }
    let mb_type = pic.mb_type[mb_xy];
    let qp = pic.qscale[mb_xy] as i32;
    let top_xy = (mb_y > 0).then(|| mb_xy - mb_w);
    let left_xy = (mb_x > 0).then(|| mb_xy - 1);
    let decoded = |xy: usize| pic.dbk[xy].slice != 0;

    // fill_filter_caches: "for sufficiently low qp, filtering wouldn't do
    // anything" (conservative early out, per the MB's own slice).
    let qp_thresh = 15 - d.alpha.min(d.beta) - 0.max(d.cqp_off[0]).max(d.cqp_off[1]);
    let avg_ok = |n: Option<usize>| match n {
        Some(xy) if decoded(xy) => ((qp + pic.qscale[xy] as i32 + 1) >> 1) <= qp_thresh,
        _ => true,
    };
    if qp <= qp_thresh && avg_ok(left_xy) && avg_ok(top_xy) {
        return;
    }

    // Neighbour types for the filter: unavailable (0) when outside the
    // picture, not decoded, or — mode 2 — in another slice.
    let nb_type = |n: Option<usize>| match n {
        Some(xy) if decoded(xy) && (d.mode != 2 || pic.dbk[xy].slice == d.slice) => pic.mb_type[xy],
        _ => 0,
    };
    let top_type = nb_type(top_xy);
    let left_type = nb_type(left_xy);

    // Inter caches (fill_filter_caches_inter, list 0) + nnz.
    let mut c = Caches {
        nnz: [0; 40],
        refs: [[-1; 40], [-1; 40]],
        mv: [[[0, 0]; 40], [[0, 0]; 40]],
        list_count: pic.list_count,
    };
    let b_stride = mb_w * 4 + 1;
    if !is_intra(mb_type) {
        let bxy = |x: usize, y: usize| 4 * x + 4 * y * b_stride;
        for list in 0..c.list_count {
            if top_type == MB_INTER {
                let t = top_xy.unwrap();
                let b = bxy(mb_x, mb_y - 1) + 3 * b_stride;
                for i in 0..4 {
                    c.mv[list][4 + i] = pic.mv[list][b + i];
                    c.refs[list][4 + i] = pic.ref_pic[list][4 * t + 2 + (i >> 1)];
                }
            }
            if left_type == MB_INTER {
                let l = left_xy.unwrap();
                let b = bxy(mb_x - 1, mb_y) + 3;
                for i in 0..4 {
                    c.mv[list][3 + 8 * (1 + i)] = pic.mv[list][b + i * b_stride];
                    c.refs[list][3 + 8 * (1 + i)] = pic.ref_pic[list][4 * l + 1 + 2 * (i >> 1)];
                }
            }
            let b = bxy(mb_x, mb_y);
            for r in 0..4 {
                for col in 0..4 {
                    let k = 12 + 8 * r + col;
                    c.mv[list][k] = pic.mv[list][b + r * b_stride + col];
                    c.refs[list][k] = pic.ref_pic[list][4 * mb_xy + 2 * (r >> 1) + (col >> 1)];
                }
            }
        }
        let nnz = &pic.nnz[mb_xy];
        for r in 0..4 {
            for col in 0..4 {
                c.nnz[12 + 8 * r + col] = nnz[4 * r + col];
            }
        }
        if top_type != 0 {
            let t = &pic.nnz[top_xy.unwrap()];
            for col in 0..4 {
                c.nnz[4 + col] = t[12 + col];
            }
        }
        if left_type != 0 {
            let l = &pic.nnz[left_xy.unwrap()];
            for r in 0..4 {
                c.nnz[3 + 8 * (1 + r)] = l[3 + 4 * r];
            }
        }
    }

    let a = 52 + d.alpha;
    let b = 52 + d.beta;
    let cqp = [chroma_qp(d.cqp_off[0], qp), chroma_qp(d.cqp_off[1], qp)];
    let cbp = pic.cbp[mb_xy];
    let part = pic.part[mb_xy];
    let nbs = [left_xy, top_xy];
    let nb_types = [left_type, top_type];
    for dir in 0..2usize {
        filter_mb_dir(
            pic,
            mb_w,
            mb_x,
            mb_y,
            mb_type,
            &c,
            dir,
            nbs[dir],
            nb_types[dir],
            qp,
            cqp,
            a,
            b,
            cbp,
            part,
            d,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn filter_mb_dir(
    pic: &mut Picture,
    mb_w: usize,
    mb_x: usize,
    mb_y: usize,
    mb_type: u32,
    c: &Caches,
    dir: usize,
    mbm_xy: Option<usize>,
    mbm_type: u32,
    qp: i32,
    cqp: [i32; 2],
    a: i32,
    b: i32,
    cbp: u16,
    part: u8,
    d: MbDeblock,
) {
    let w = mb_w * 16;
    let cw = w / 2;
    let y0 = (mb_y * 16 * w + mb_x * 16) as isize;
    let c0 = (mb_y * 8 * cw + mb_x * 8) as isize;
    let (w_i, cw_i) = (w as isize, cw as isize);
    // dir 0 = vertical edges (filter across x): xs 1, ys stride.
    // dir 1 = horizontal edges (filter across y): xs stride, ys 1.
    let (lxs, lys, cxs, cys) = if dir == 0 {
        (1, w_i, 1, cw_i)
    } else {
        (w_i, 1, cw_i, 1)
    };
    let intra_mb = is_intra(mb_type);
    let mask_edge = if intra_mb { 0 } else { mask_edge(dir, part) };
    let edges = if mask_edge == 3 && (cbp & 15) == 0 {
        1
    } else {
        4
    };
    let mask_par0 = !intra_mb && par0(dir, part);
    let step = if dir == 0 { 1 } else { 8 };
    let mvy_limit = 4;

    // ---- MB edge (edge 0), against the left / top neighbour ----
    if mbm_type != 0 {
        let mut bs = [0i16; 4];
        if intra_mb || is_intra(mbm_type) {
            bs = [4; 4];
        } else {
            let mbm_part = pic.part[mbm_xy.unwrap()];
            let mv_done = if mask_par0 && par0(dir, mbm_part) {
                let v = check_mv(c, 12, 12 - step, mvy_limit);
                bs = [v; 4];
                true
            } else {
                false
            };
            for (i, s) in bs.iter_mut().enumerate() {
                let (x, y) = if dir == 0 { (0, i) } else { (i, 0) };
                let bi = 12 + x + 8 * y;
                let bn = bi - step;
                if c.nnz[bi] | c.nnz[bn] != 0 {
                    *s = 2;
                } else if !mv_done {
                    *s = check_mv(c, bi, bn, mvy_limit);
                }
            }
        }
        if bs.iter().any(|&v| v != 0) {
            let nq = pic.qscale[mbm_xy.unwrap()] as i32;
            let eqp = (qp + nq + 1) >> 1;
            let cavg = [
                (cqp[0] + chroma_qp(d.cqp_off[0], nq) + 1) >> 1,
                (cqp[1] + chroma_qp(d.cqp_off[1], nq) + 1) >> 1,
            ];
            edge(&mut pic.y, y0, lxs, lys, &bs, eqp, a, b, true, false);
            edge(&mut pic.cb, c0, cxs, cys, &bs, cavg[0], a, b, true, true);
            edge(&mut pic.cr, c0, cxs, cys, &bs, cavg[1], a, b, true, true);
        }
    }

    // ---- internal edges ----
    for e in 1..edges {
        let mut bs = [0i16; 4];
        if intra_mb {
            bs = [3; 4];
        } else {
            let mv_done = if e & mask_edge != 0 {
                true
            } else if mask_par0 {
                let bi = 12 + e * step;
                let v = check_mv(c, bi, bi - step, mvy_limit);
                bs = [v; 4];
                true
            } else {
                false
            };
            for (i, s) in bs.iter_mut().enumerate() {
                let (x, y) = if dir == 0 { (e, i) } else { (i, e) };
                let bi = 12 + x + 8 * y;
                let bn = bi - step;
                if c.nnz[bi] | c.nnz[bn] != 0 {
                    *s = 2;
                } else if !mv_done {
                    *s = check_mv(c, bi, bn, mvy_limit);
                }
            }
            if bs.iter().all(|&v| v == 0) {
                continue;
            }
        }
        let off = 4 * e as isize;
        let (lpos, cpos) = if dir == 0 {
            (y0 + off, c0 + off / 2)
        } else {
            (y0 + off * w_i, c0 + (off / 2) * cw_i)
        };
        edge(&mut pic.y, lpos, lxs, lys, &bs, qp, a, b, false, false);
        if e & 1 == 0 {
            edge(&mut pic.cb, cpos, cxs, cys, &bs, cqp[0], a, b, false, true);
            edge(&mut pic.cr, cpos, cxs, cys, &bs, cqp[1], a, b, false, true);
        }
    }
}
