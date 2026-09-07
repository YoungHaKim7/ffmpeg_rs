//! Table-driven scaling filters — a bit-faithful port of libswscale's
//! `initFilter()` (`libswscale/utils.c:197-612`) and its 8-bit apply path.
//!
//! The float single-pass sampler in [`super::sample_plane`] (nearest /
//! bilinear / bicubic, mirrored by `assets/scale.comp` on the GPU) keeps its
//! fixed tap window; these tables are how real swscale scales: coefficient
//! rows generated once per context in `int64_t` fixed point, reduced,
//! border-folded, then quantized to `int16_t` with error diffusion so every
//! row sums to exactly `one`. Filter width widens in source space on
//! downscale (`utils.c:287-293`) — the behavior the fixed-tap float path
//! deliberately diverges from.
//!
//! ## C → Rust mapping
//!
//! | C | Rust |
//! |---|---|
//! | `initFilter` (`utils.c:197-612`) | [`init_filter`] |
//! | `getSplineCoeff` (`utils.c:155-166`) | [`get_spline_coeff`] |
//! | sizeFactor table + lanczos override (`utils.c:183-195, 278-279`) | [`TableScaler::size_factor`] |
//! | `hScale8To15_c` (`swscale.c:128-142`) | [`h_scale8_to15`] |
//! | `yuv2planeX_8_c` (`output.c:468-483`) | [`yuv2plane_x_8`] |
//! | `yuv2plane1_8_c` fast path (`output.c:485-493`) | not ported — exactly [`yuv2plane_x_8`] with one 4096 tap (`4096·(v+64) = v·4096 + 64<<12`; pinned by a unit test) |
//! | slice.c ring buffer + `fill_ones` sentinels | [`scale_plane`] materializes the whole H intermediate (`src_h × dst_w` i16, ≤ ~4 MB at 1080p) |
//! | `lumXInc`/`chrXInc` (`utils.c:1253-1254, 1431-1432`) | [`x_inc`] |
//! | chroma positions (`ff_sws_chroma_pos` format.c:554 → `av_chroma_location_enum_to_pos` pixdesc.c:3902 → `get_local_pos` utils.c:168) | [`chroma_pos`] |
//!
//! Fixed-point formats (see `utils.c`): increments 16.16; intermediate
//! coefficients in `fone = 1<<(54-min(av_log2(srcW/dstW),8))` units; distance
//! `d` in 2^30 units (×`dstW/srcW` on downscale, one truncating division);
//! **stored H coefficients int16 summing to 1<<14, V to 1<<12**
//! (`utils.c:1686/1716`); H intermediate int16 at 15 bpc (`>>7`); V output
//! `(val>>19)` with the constant `sws_pb_64` dither.
//!
//! ## Deliberately skipped C paths (with the guard that makes them OK)
//!
//! * `filterAlign` machinery (MMX 4 / AltiVec 8 / NEON 4, `utils.c:1678-1713`,
//!   `459-484`): **`filterAlign = 1` always** — exactly what FFmpeg's C build
//!   picks without those CPU flags; `filterSize = minFilterSize` unaligned.
//! * `dstW+3` `filterPos` tail + 3 replicated coefficient rows
//!   (`utils.c:216, 562-599`) and `ff_shuffle_filter_coefficients`
//!   (`utils.c:97-153`): MMX/AVX2 overread layout. Rust reads are
//!   index-bounded; taps at `filterPos+j >= srcW` are provably zero (the
//!   `utils.c:532-560` border-fix invariant) and are skipped instead.
//! * `SWS_POINT` special case (`utils.c:229-243`): semantically our existing
//!   float `Nearest`; intentional non-duplication.
//! * FAST_BILINEAR and the `SWS_X`/BICUBIC/BILINEAR general-branch kernels:
//!   BICUBIC/BILINEAR are served by the float path (GPU-mirrored); X and
//!   fast_bilinear are out of the ported flag subset. The area-upscale 2-tap
//!   loop (`utils.c:244-267`, shared with fast_bilinear) IS ported since
//!   `SWS_AREA` reaches it.
//! * `srcFilter`/`dstFilter` convolution (`utils.c:385-415`): the graph API
//!   always passes NULL (`graph.c:635`); the legacy `SwsVector` API is not
//!   ported.
//! * Cascade split at the geometric mean (`utils.c:1806-1834`): when the
//!   post-reduction `filterSize >= 256` (SWS_MAX_FILTER_SIZE, no
//!   ACCURATE_RND ⇒ threshold exactly 256), C transparently chains two
//!   contexts; we return [`Error::Unsupported`] — a documented divergence
//!   (requires ≥12.75× downscale for sinc/spline at srcW ≥ ~2000).
//! * Ordered-dither tables `ff_dither_8x8_128[dstY&7]` and the 0/3 offsets
//!   (`swscale.c:385-387, 519-522`): `should_dither = isNBPS(src) ||
//!   is16BPS(src)` is false for every 8-bit source accepted here, so the
//!   dither is the constant `sws_pb_64` (all 64). The tables would only go
//!   live for ≥9-bit sources, out of this subset.
//! * MMX vertical-filter packing, FAST_BILINEAR `xInc ± 20` hacks
//!   (`utils.c:1441-1451`), `emms_c()` — SIMD-only.

use crate::util::{
    color::ChromaLocation,
    error::{Error, Result},
};

use super::ScaleAlgorithm;

/// Horizontal rows sum to this (`utils.c:1686`, `one = 1 << 14`).
pub(crate) const ONE_H: i64 = 1 << 14;
/// Vertical rows sum to this (`utils.c:1716/1727`, `one = 1 << 12`).
pub(crate) const ONE_V: i64 = 1 << 12;
/// `SWS_MAX_FILTER_SIZE` (configure default 256); without SWS_ACCURATE_RND
/// the check at `utils.c:492-493` reads `filterSize >= 256`.
pub(crate) const MAX_FILTER_SIZE: usize = 256;

/// The table-driven `SWS_*` scalers — the five kernels this module generates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableScaler {
    /// `SWS_AREA` (1<<5) — downscale trapezoid; upscale takes the 2-tap
    /// bilinear loop (`utils.c:244-267`).
    Area,
    /// `SWS_GAUSS` (1<<7) — `2^(-3·d²)`.
    Gauss,
    /// `SWS_SINC` (1<<8) — `sin(πd)/(πd)`.
    Sinc,
    /// `SWS_LANCZOS` (1<<9) — 3-lobe `sinc·sinc/p`.
    Lanczos,
    /// `SWS_SPLINE` (1<<10) — the recursive cubic-Hermite `getSplineCoeff`
    /// with `p = -2.196152422706632`. One value exactly like ffmpeg's
    /// `-sws_flags spline`: libswscale has no spline16/36/64 tables — the
    /// 16/36/64 character is purely the tap count (`utils.c:287-293`),
    /// reproduced by the widening formula.
    Spline,
}

impl TableScaler {
    /// The `scale_algorithms[]` sizeFactor (`utils.c:183-195`) with the
    /// lanczos override (`utils.c:278-279`): params are never overridden
    /// (always `SWS_PARAM_DEFAULT`), so lanczos is the constant
    /// `ceil(2·3.0) = 6`.
    pub(crate) const fn size_factor(self) -> i32 {
        match self {
            TableScaler::Area => 1, // "downscale only, for upscale it is bilinear"
            TableScaler::Gauss => 8,
            TableScaler::Sinc => 20,
            TableScaler::Lanczos => 6,
            TableScaler::Spline => 20,
        }
    }

    /// Map the public algorithm enum (only valid for table-driven variants —
    /// see [`ScaleAlgorithm::is_table_driven`]).
    pub(crate) fn from_algorithm(alg: ScaleAlgorithm) -> TableScaler {
        match alg {
            ScaleAlgorithm::Area => TableScaler::Area,
            ScaleAlgorithm::Gauss => TableScaler::Gauss,
            ScaleAlgorithm::Sinc => TableScaler::Sinc,
            ScaleAlgorithm::Lanczos => TableScaler::Lanczos,
            ScaleAlgorithm::Spline => TableScaler::Spline,
            _ => unreachable!("not a table-driven algorithm"),
        }
    }
}

/// One plane's coefficient table: `dst_len` rows of `size` int16 taps plus
/// their source start positions (`hLumFilter`/`hLumFilterPos` etc.).
#[derive(Debug)]
pub(crate) struct SwsFilter {
    /// `dst_len * size` int16 taps, row-major, no SIMD padding.
    pub(crate) filter: Vec<i16>,
    /// `dst_len` positions (C allocates `dstW + 3` for MMX overread — skipped).
    pub(crate) filter_pos: Vec<i32>,
    pub(crate) size: usize,
}

impl SwsFilter {
    /// Row `i` of int16 taps (`filter + i * filterSize`).
    pub(crate) fn row(&self, i: usize) -> &[i16] {
        &self.filter[i * self.size..(i + 1) * self.size]
    }
}

/// The four coefficient tables a context owns (`utils.c:1684-1738`).
pub(crate) struct FilterPlan {
    pub(crate) h_lum: SwsFilter,
    pub(crate) h_chr: SwsFilter,
    pub(crate) v_lum: SwsFilter,
    pub(crate) v_chr: SwsFilter,
}

/// `av_log2` — floor log2, `av_log2(0) == 0` (`libavutil/intmath.h`).
#[inline]
fn av_log2(x: i32) -> i32 {
    if x <= 1 {
        0
    } else {
        31 - x.leading_zeros() as i32
    }
}

/// `fone = 1LL << (54 - FFMIN(av_log2(srcW/dstW), 8))` (`utils.c:210`) —
/// integer division BEFORE the log; upscale ⇒ `1<<54` (log2(0)==0); the
/// shift clamps at 8 for ≥256× downscale.
#[inline]
fn fone(src_len: i32, dst_len: i32) -> i64 {
    1i64 << (54 - av_log2(src_len / dst_len).min(8))
}

/// 16.16 increment, round-to-nearest (`utils.c:1253-1254, 1431-1432`).
#[inline]
pub(crate) fn x_inc(src: i32, dst: i32) -> i64 {
    (((src as i64) << 16) + ((dst >> 1) as i64)) / dst as i64
}

/// `getSplineCoeff` (`utils.c:155-166`) — recursive cubic Hermite, one
/// polynomial segment per integer interval of `dist` (this is NOT a
/// spline16/36/64 table; the recursion depth sets the effective width).
fn get_spline_coeff(a: f64, b: f64, c: f64, d: f64, dist: f64) -> f64 {
    if dist <= 1.0 {
        ((d * dist + c) * dist + b) * dist + a
    } else {
        get_spline_coeff(
            0.0,
            b + 2.0 * c + 3.0 * d,
            c + 3.0 * d,
            -b - 3.0 * c - 6.0 * d,
            dist - 1.0,
        )
    }
}

/// The per-kernel coefficient in `fone` units (`utils.c:312-373`), evaluated
/// in f64 exactly as C does (double → int64 = Rust `as i64` truncation);
/// `d != 0` ternaries test the INTEGER `d`, not `floatd`.
fn kernel_coefficient(scaler: TableScaler, d: i64, floatd: f64, x_inc: i64, fone: i64) -> i64 {
    match scaler {
        // utils.c:346-354 — pure i64 trapezoid. d2*xInc is the C product;
        // wrapping keeps degenerate geometries from panicking where C would
        // (unreachable for accepted sizes).
        TableScaler::Area => {
            let d2 = d - (1 << 29);
            let mut coeff = if d2.wrapping_mul(x_inc) < -(1i64 << (29 + 16)) {
                1i64 << (30 + 16)
            } else if d2.wrapping_mul(x_inc) < (1i64 << (29 + 16)) {
                -d2.wrapping_mul(x_inc) + (1i64 << (30 + 16))
            } else {
                0
            };
            coeff *= fone >> (30 + 16);
            coeff
        }
        // utils.c:355-357 — exp2 (2^(-p·d²)), not exp.
        TableScaler::Gauss => {
            let p = 3.0f64;
            (f64::exp2(-p * floatd * floatd) * fone as f64) as i64
        }
        // utils.c:358-359 — the ternary tests the integer d.
        TableScaler::Sinc => {
            let c = if d != 0 {
                (floatd * std::f64::consts::PI).sin() / (floatd * std::f64::consts::PI)
            } else {
                1.0
            };
            (c * fone as f64) as i64
        }
        // utils.c:360-365 — window zeroes floatd > p AFTER the product.
        TableScaler::Lanczos => {
            let p = 3.0f64;
            let pi = std::f64::consts::PI;
            let c = if d != 0 {
                (floatd * pi).sin() * (floatd * pi / p).sin() / (floatd * floatd * pi * pi / p)
            } else {
                1.0
            };
            let mut coeff = (c * fone as f64) as i64;
            if floatd > p {
                coeff = 0;
            }
            coeff
        }
        // utils.c:371-373.
        TableScaler::Spline => {
            let p = -2.196152422706632f64;
            (get_spline_coeff(1.0, 0.0, p, -p - 1.0, floatd) * fone as f64) as i64
        }
    }
}

/// The general branch's raw tap count (`utils.c:287-293`) — pre-reduction,
/// exposed for the unit-test pins. Widening on downscale is the ceil form.
fn general_branch_size(scaler: TableScaler, x_inc: i64, src_len: i32, dst_len: i32) -> usize {
    let size_factor = scaler.size_factor() as i64;
    debug_assert!(size_factor <= 50, "utils.c:282-285 would AVERROR(EINVAL)");
    let filter_size = if x_inc <= 1 << 16 {
        1 + size_factor // upscale
    } else {
        1 + (size_factor * src_len as i64 + dst_len as i64 - 1) / dst_len as i64
    };
    filter_size.min(src_len as i64 - 2).max(1) as usize
}

/// `initFilter()` (`utils.c:197-612`) with `filterAlign = 1`, no
/// srcFilter/dstFilter, `SWS_PARAM_DEFAULT` params, no ACCURATE_RND/BITEXACT.
///
/// `src_pos`/`dst_pos` are the `get_local_pos` values: 128 for luma both
/// ways; chroma tables get the source's siting position vs the destination's
/// center default (see [`chroma_pos`]).
pub(crate) fn init_filter(
    x_inc: i64,
    src_len: i32,
    dst_len: i32,
    one: i64,
    scaler: TableScaler,
    src_pos: i32,
    dst_pos: i32,
) -> Result<SwsFilter> {
    let dst_len = dst_len as usize;
    let src_len_u = src_len as usize;
    let fone = fone(src_len, dst_len as i32);
    let mut filter_pos = vec![0i32; dst_len];

    // ---- coefficient generation ----------------------------------------
    let (mut filter2, filter2_size): (Vec<i64>, usize) = if (x_inc - 0x10000).abs() < 10
        && src_pos == dst_pos
    {
        // utils.c:219-228 — unscaled: single fone tap at identity.
        filter_pos
            .iter_mut()
            .enumerate()
            .for_each(|(i, p)| *p = i as i32);
        (vec![fone; dst_len], 1)
    } else if x_inc <= 1 << 16 && scaler == TableScaler::Area {
        // utils.c:244-267 — area upscale / fast-bilinear 2-tap loop
        // (fast_bilinear itself is not ported; area reaches this).
        let filter_size = 2usize;
        let mut f = vec![0i64; dst_len * filter_size];
        // Note the asymmetric shifts: dstPos term >>8, srcPos term >>7
        // (utils.c:246) — a >>7 here mirrors every row.
        let mut x_dst_in_src = ((dst_pos as i64 * x_inc) >> 8) - ((src_pos as i64 * 0x8000) >> 7);
        for i in 0..dst_len {
            let mut xx = (x_dst_in_src - ((filter_size as i64 - 1) << 15) + (1 << 15)) >> 16;
            filter_pos[i] = xx as i32;
            for j in 0..filter_size {
                let mut coeff = fone - (xx * (1 << 16) - x_dst_in_src).abs() * (fone >> 16);
                if coeff < 0 {
                    coeff = 0;
                }
                f[i * filter_size + j] = coeff;
                xx += 1;
            }
            x_dst_in_src += x_inc;
        }
        (f, filter_size)
    } else {
        // utils.c:268-381 — general branch (all five kernels on
        // downscale; everything but area on upscale).
        let filter_size = general_branch_size(scaler, x_inc, src_len, dst_len as i32);
        let mut f = vec![0i64; dst_len * filter_size];
        let mut x_dst_in_src = ((dst_pos as i64 * x_inc) >> 7) - ((src_pos as i64 * 0x10000) >> 7);
        for i in 0..dst_len {
            // Division (truncates toward zero), NOT a floor shift — they
            // differ on the negative border values (utils.c:299).
            let xx0 = (x_dst_in_src - (filter_size as i64 - 2) * (1 << 16)) / (1 << 17);
            filter_pos[i] = xx0 as i32;
            let mut xx = xx0;
            for j in 0..filter_size {
                let mut d = (xx * (1 << 17) - x_dst_in_src).abs() << 13;
                if x_inc > 1 << 16 {
                    // One i64 multiply then ONE truncating division
                    // (utils.c:308-309) — do not pre-divide.
                    d = d * dst_len as i64 / src_len as i64;
                }
                let floatd = d as f64 * (1.0 / (1 << 30) as f64);
                f[i * filter_size + j] = kernel_coefficient(scaler, d, floatd, x_inc, fone);
                xx += 1;
            }
            x_dst_in_src += 2 * x_inc;
        }
        (f, filter_size)
    };
    // srcFilter/dstFilter convolution (utils.c:385-415) skipped: the graph
    // API always passes NULL, so filter2 == filter and the position
    // recenter `(filterSize-1)/2 - (filter2Size-1)/2` is 0.

    // ---- reduction pass (utils.c:417-457) -------------------------------
    let mut min_filter_size = 0usize;
    for i in (0..dst_len).rev() {
        let mut min = filter2_size;
        let base = i * filter2_size;

        // Shift near-zeros out on the left; cutOff re-reads ELEMENT 0 after
        // each shift; monotonicity must hold or the core cannot apply it.
        let mut cut_off = 0i64;
        for _ in 0..filter2_size {
            cut_off += filter2[base].abs();
            if cut_off as f64 > 0.002 * fone as f64 {
                break;
            }
            if i < dst_len - 1 && filter_pos[i] >= filter_pos[i + 1] {
                break;
            }
            filter2.copy_within(base + 1..base + filter2_size, base);
            filter2[base + filter2_size - 1] = 0;
            filter_pos[i] += 1;
        }

        // Count near-zeros on the right (separate pass, cumulative again).
        let mut cut_off = 0i64;
        for j in (1..filter2_size).rev() {
            cut_off += filter2[base + j].abs();
            if cut_off as f64 > 0.002 * fone as f64 {
                break;
            }
            min -= 1;
        }
        min_filter_size = min_filter_size.max(min);
    }
    debug_assert!(min_filter_size > 0);

    // filterAlign = 1 ⇒ filterSize = minFilterSize (utils.c:487).
    let filter_size = min_filter_size;
    // utils.c:492-493 — the cascade threshold. C splits into two contexts at
    // the geometric mean (utils.c:1806-1834); we refuse — see module docs.
    if filter_size >= MAX_FILTER_SIZE {
        return Err(Error::Unsupported(format!(
            "scaling ratio would need swscale's cascaded two-pass (filterSize {filter_size} \
             >= {MAX_FILTER_SIZE}, utils.c:1806-1834); unsupported in this phase"
        )));
    }

    // ---- step 2: copy to the final width (utils.c:503-510) --------------
    let mut filter = vec![0i64; dst_len * filter_size];
    for i in 0..dst_len {
        for j in 0..filter_size {
            filter[i * filter_size + j] = if j < filter2_size {
                filter2[i * filter2_size + j]
            } else {
                0
            };
        }
        // (SWS_BITEXACT zeroing of j >= minFilterSize skipped: flag not set.)
    }

    // ---- border fix (utils.c:513-560), order exactly as C ---------------
    for i in 0..dst_len {
        let base = i * filter_size;
        if filter_pos[i] < 0 {
            // Fold out-of-range left taps onto their in-range mirror.
            for j in 1..filter_size {
                let left = (j as i32 + filter_pos[i]).max(0) as usize;
                filter[base + left] += filter[base + j];
                filter[base + j] = 0;
            }
            filter_pos[i] = 0;
        }

        if filter_pos[i] as usize + filter_size > src_len_u {
            let shift = filter_pos[i] + (filter_size as i32 - src_len).min(0);
            let mut acc = 0i64;
            for j in (0..filter_size).rev() {
                if filter_pos[i] as usize + j >= src_len_u {
                    acc += filter[base + j];
                    filter[base + j] = 0;
                }
            }
            // Backwards so the sources are still original when read.
            for j in (0..filter_size).rev() {
                filter[base + j] = if (j as i32) < shift {
                    0
                } else {
                    filter[base + j - shift as usize]
                };
            }
            filter_pos[i] -= shift;
            filter[base + src_len_u - 1 - filter_pos[i] as usize] += acc;
        }
        debug_assert!(filter_pos[i] >= 0 && (filter_pos[i] as usize) < src_len_u);
    }

    // ---- normalize + error-diffused quantization (utils.c:562-588) ------
    // ROUNDED_DIV is half-away-from-zero; the residual carries WITHIN the
    // row, so every row sums to exactly `one` (±1 on the last tap).
    let mut out = vec![0i16; dst_len * filter_size];
    for i in 0..dst_len {
        let base = i * filter_size;
        let mut sum: i64 = 0;
        for j in 0..filter_size {
            sum += filter[base + j];
        }
        sum = (sum + one / 2) / one;
        if sum == 0 {
            // C warns "SwScaler: zero vector in scaling" and continues.
            sum = 1;
        }
        let mut error = 0i64;
        for j in 0..filter_size {
            let v = filter[base + j] + error;
            let int_v = rounded_div(v, sum);
            out[base + j] = int_v as i16;
            error = v - int_v * sum;
        }
    }
    // The +3 SIMD tail rows (utils.c:590-599) are skipped — reads are
    // index-bounded here.

    Ok(SwsFilter {
        filter: out,
        filter_pos,
        size: filter_size,
    })
}

/// `ROUNDED_DIV` (`libavutil/common.h:58`): round-half-away-from-zero.
#[inline]
fn rounded_div(a: i64, b: i64) -> i64 {
    if a >= 0 {
        (a + (b >> 1)) / b
    } else {
        (a - (b >> 1)) / b
    }
}

/// The chroma `srcPos`/`dstPos` for one axis: `ff_sws_chroma_pos`
/// (`format.c:554-593`) → `av_chroma_location_enum_to_pos`
/// (`pixdesc.c:3902-3912`) → `get_local_pos` (`utils.c:168-175`),
/// specialized to yuv420p (sub 1 ⇒ the `·((1<<1)-1)` rescale is ×1) and
/// progressive frames. Unspecified defaults to CENTER (`format.c:562-565`).
/// Resulting table:
///
/// | location             | h pos | v pos |
/// |----------------------|-------|-------|
/// | Unspecified / Center | 128   | 128   |
/// | Left                 | 64    | 128   |
/// | TopLeft              | 64    | 64    |
/// | Top                  | 128   | 64    |
/// | BottomLeft           | 64    | 192   |
/// | Bottom               | 128   | 192   |
///
/// (Top/Bottom sit at the horizontally-centered variant — xpos 128 — unlike
/// the *Left flavors; the `pos--` then `(pos&1)*128` decode in
/// `enum_to_pos` is what produces the split.) Destination frames in this
/// pipeline never carry a location, so callers pass the CENTER value 128 —
/// matching C's UNSPECIFIED⇒CENTER default. Note the float path
/// (`sample_plane`) treats Unspecified as LEFT-sited; this table path is the
/// C-faithful mapping (a documented inconsistency between the two paths).
pub(crate) fn chroma_pos(loc: ChromaLocation, horizontal: bool) -> i32 {
    let loc = if loc == ChromaLocation::Unspecified {
        ChromaLocation::Center
    } else {
        loc
    };
    let pos = loc as i32 - 1;
    let (x, y) = ((pos & 1) * 128, ((pos >> 1) ^ i32::from(pos < 4)) * 128);
    let raw = if horizontal { x } else { y };
    // get_local_pos with chr_subsample = 1: pos += 128; pos >> 1.
    (raw + 128) >> 1
}

/// Build the four tables a context needs (`utils.c:1684-1738`): luma and
/// chroma, horizontal (`one = 1<<14`) and vertical (`one = 1<<12`). Chroma
/// geometry is `ceil(dim/2)` (the codebase's `div_ceil(2)` convention,
/// `AV_CEIL_RSHIFT`); luma positions are the constant 128.
pub(crate) fn build_plan(
    alg: TableScaler,
    src: (i32, i32),
    dst: (i32, i32),
    src_loc: ChromaLocation,
) -> Result<FilterPlan> {
    // AV_CEIL_RSHIFT(dim, 1) — `div_ceil(2)` is unstable on this toolchain.
    let ceil2 = |v: i32| (v + 1) / 2;
    let (csw, csh) = (ceil2(src.0), ceil2(src.1));
    let (cdw, cdh) = (ceil2(dst.0), ceil2(dst.1));
    let h_pos = chroma_pos(src_loc, true);
    let v_pos = chroma_pos(src_loc, false);
    Ok(FilterPlan {
        h_lum: init_filter(x_inc(src.0, dst.0), src.0, dst.0, ONE_H, alg, 128, 128)?,
        h_chr: init_filter(x_inc(csw, cdw), csw, cdw, ONE_H, alg, h_pos, 128)?,
        v_lum: init_filter(x_inc(src.1, dst.1), src.1, dst.1, ONE_V, alg, 128, 128)?,
        v_chr: init_filter(x_inc(csh, cdh), csh, cdh, ONE_V, alg, v_pos, 128)?,
    })
}

/// `hScale8To15_c` (`swscale.c:128-142`): 8-bit source → 15-bit i16
/// intermediate. Arithmetic `>>7`, upper clamp ONLY — negative lobes flow
/// into the i16 as-is (two's-complement `as i16` matches C's int16_t store).
pub(crate) fn h_scale8_to15(dst: &mut [i16], src_row: &[u8], f: &SwsFilter) {
    for i in 0..dst.len() {
        let src_pos = f.filter_pos[i] as usize;
        let mut val: i32 = 0;
        let row = f.row(i);
        for j in 0..f.size {
            // Taps at filterPos+j >= srcW are zero by the border-fix
            // invariant (utils.c:532-560 asserts it) — skipping them is
            // bit-identical to multiplying by C's zeros, and our compact
            // planes have no overread padding.
            let idx = src_pos + j;
            if idx >= src_row.len() {
                continue;
            }
            val += src_row[idx] as i32 * row[j] as i32;
        }
        dst[i] = (val >> 7).min((1 << 15) - 1) as i16;
    }
}

/// `yuv2planeX_8_c` (`output.c:468-483`): vertical combine of `filterSize`
/// H-intermediate lines into one 8-bit row. Dither is the constant
/// `sws_pb_64` (all 64) for every 8-bit source (`swscale.c:291-292,
/// 385-387`); the `(unsigned)` cast is two's-complement-identical to a
/// wrapping add.
pub(crate) fn yuv2plane_x_8(dest_row: &mut [u8], lines: &[&[i16]], row: &[i16]) {
    for i in 0..dest_row.len() {
        let mut val: i32 = 64 << 12;
        for (j, line) in lines.iter().enumerate() {
            val = val.wrapping_add(line[i] as i32 * row[j] as i32);
        }
        dest_row[i] = (val >> 19).clamp(0, 255) as u8;
    }
}

/// Two-pass plane driver: H-pass every source line into a materialized i16
/// intermediate (slice.c's ring buffer, single-threaded and whole), then the
/// V-pass per destination row. `first = max(1 - v_size, v_pos[y])`
/// (`swscale.c:417`); lines beyond `src_h` carry zero coefficients (border
/// invariant) and are skipped.
pub(crate) fn scale_plane(
    src_plane: &[u8],
    src_ls: usize,
    src_wh: (i32, i32),
    dst_plane: &mut [u8],
    dst_ls: usize,
    h: &SwsFilter,
    v: &SwsFilter,
) {
    let (src_w, src_h) = (src_wh.0 as usize, src_wh.1 as usize);
    let dst_w = h.filter_pos.len();
    let mut inter = vec![0i16; src_h * dst_w];
    for y in 0..src_h {
        h_scale8_to15(
            &mut inter[y * dst_w..(y + 1) * dst_w],
            &src_plane[y * src_ls..y * src_ls + src_w],
            h,
        );
    }
    let mut lines: Vec<&[i16]> = Vec::with_capacity(v.size);
    for yy in 0..v.filter_pos.len() {
        let first = (1 - v.size as i32).max(v.filter_pos[yy]).max(0) as usize;
        lines.clear();
        for j in 0..v.size {
            let y = first + j;
            if y >= src_h {
                break; // zero coefficient — the H/V skip rule
            }
            lines.push(&inter[y * dst_w..(y + 1) * dst_w]);
        }
        yuv2plane_x_8(
            &mut dst_plane[yy * dst_ls..yy * dst_ls + dst_w],
            &lines,
            v.row(yy),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_scalers() -> [TableScaler; 5] {
        [
            TableScaler::Area,
            TableScaler::Gauss,
            TableScaler::Sinc,
            TableScaler::Lanczos,
            TableScaler::Spline,
        ]
    }

    /// The anchor, fully hand-derived from the C formulas (verified against
    /// the reference C source compiled standalone):
    ///
    /// * `xInc = ((4<<16) + (2>>1))/2 = 262145/2 = 0x20000` (131072);
    /// * pos 128/128 ⇒ `xDstInSrc = (128·xInc)>>7 − (128·0x10000)>>7
    ///   = 131072 − 65536 = 65536`;
    /// * downscale ⇒ general branch; area sizeFactor 1 ⇒ filterSize
    ///   `1 + (1·4 + 1)/2 = 3`, then `FFMIN(3, 4−2) = 2`;
    /// * row 0: `xx = (65536 − 0)/2^17 = 0`, taps {0,1}; each has
    ///   `|xx·2^17 − xDstInSrc|<<13 = 2^29`, ×dstW/srcW ⇒ `d = 2^28`,
    ///   `d2 = −2^28`, `d2·xInc = −2^45` (inside ±2^45) ⇒ coeff
    ///   `2^45 + 2^46 = 3·2^45`, ×`(fone>>46 = 2^53>>46 = 2^7)` = `3·2^52`;
    /// * row sum `3·2^53` ⇒ `sum = (3·2^53 + 2^13)/2^14 = 3·2^39`,
    ///   `ROUNDED_DIV(3·2^52, 3·2^39) = 2^13 = 8192` per tap;
    /// * row 1: `xDstInSrc = 65536 + 2·xInc = 327680`, `xx = 2` (taps {2,3},
    ///   both in range — `pos 2 + size 2 = srcW`, no border fold), same
    ///   symmetric distances ⇒ `[8192, 8192]`.
    #[test]
    fn area_downscale_4_to2_hand_computed() {
        let f = init_filter(x_inc(4, 2), 4, 2, ONE_H, TableScaler::Area, 128, 128).unwrap();
        assert_eq!(f.size, 2);
        assert_eq!(f.filter_pos, vec![0, 2]);
        assert_eq!(f.row(0), &[8192, 8192]);
        assert_eq!(f.row(1), &[8192, 8192]);
    }

    /// § utils.c:287-293 — the raw (pre-reduction) widening formula pins.
    #[test]
    fn filter_sizes_match_c_formulas() {
        let size = |s, sw, dw| general_branch_size(s, x_inc(sw, dw), sw, dw);
        for sw in [64, 100, 1920] {
            let dw = sw / 2;
            assert_eq!(size(TableScaler::Lanczos, sw, dw), 13, "lanczos 2x down");
            assert_eq!(size(TableScaler::Gauss, sw, dw), 17, "gauss 2x down");
            assert_eq!(size(TableScaler::Sinc, sw, dw), 41, "sinc 2x down");
            assert_eq!(size(TableScaler::Spline, sw, dw), 41, "spline 2x down");
            assert_eq!(size(TableScaler::Area, sw, dw), 3, "area 2x down");
        }
        // Upscale: the constant 1+sizeFactor.
        assert_eq!(size(TableScaler::Lanczos, 32, 64), 7);
        assert_eq!(size(TableScaler::Gauss, 32, 64), 9);
        assert_eq!(size(TableScaler::Sinc, 32, 64), 21);
        assert_eq!(size(TableScaler::Spline, 32, 64), 21);
        // The clamps: tiny dst widens to srcW-2; tiny src clamps up to 1.
        assert_eq!(size(TableScaler::Sinc, 1920, 1), 1918);
        assert_eq!(size(TableScaler::Sinc, 2, 4), 1);
    }

    /// The FINAL sizes after the reduction pass (these differ from the raw
    /// formula: near-zero tails get trimmed — measured on the reference C
    /// source compiled standalone). Gauss trims hardest (2^(-3d²) decays
    /// super-polynomially), sinc not at all at 2× (edge taps ~1/(πd) > the
    /// 0.002 cumulative cutoff).
    #[test]
    fn final_sizes_match_c_reference() {
        let size = |s, sw, dw| {
            init_filter(x_inc(sw, dw), sw, dw, ONE_H, s, 128, 128)
                .unwrap()
                .size
        };
        // 64→32: area 2, gauss 6, lanczos 12, sinc 41, spline 20.
        assert_eq!(size(TableScaler::Area, 64, 32), 2);
        assert_eq!(size(TableScaler::Gauss, 64, 32), 6);
        assert_eq!(size(TableScaler::Lanczos, 64, 32), 12);
        assert_eq!(size(TableScaler::Sinc, 64, 32), 41);
        assert_eq!(size(TableScaler::Spline, 64, 32), 20);
        // 32→64 upscale: area takes the special-C 2-tap loop; gauss trims
        // 9→3, lanczos 7→6, spline 21→9, sinc stays 21.
        assert_eq!(size(TableScaler::Area, 32, 64), 2);
        assert_eq!(size(TableScaler::Gauss, 32, 64), 3);
        assert_eq!(size(TableScaler::Lanczos, 32, 64), 6);
        assert_eq!(size(TableScaler::Sinc, 32, 64), 21);
        assert_eq!(size(TableScaler::Spline, 32, 64), 9);
    }

    /// Coefficient rows pinned against the reference C source (bit-exact
    /// double kernels: sin/exp2/spline recursion).
    #[test]
    fn coefficient_rows_match_c_reference() {
        // sinc 5→3 (center siting): pos [0,1,2].
        let f = init_filter(x_inc(5, 3), 5, 3, ONE_H, TableScaler::Sinc, 128, 128).unwrap();
        assert_eq!(f.size, 3);
        assert_eq!(f.filter_pos, vec![0, 1, 2]);
        assert_eq!(f.row(0), &[9057, 7327, 0]);
        assert_eq!(f.row(1), &[4115, 8154, 4115]);
        assert_eq!(f.row(2), &[0, 6437, 9947]);

        // gauss 13→7 with LEFT-sited source chroma (srcPos 64) vs the
        // center default dst (128): pos [0,0,2,...].
        let f = init_filter(x_inc(13, 7), 13, 7, ONE_H, TableScaler::Gauss, 64, 128).unwrap();
        assert_eq!(f.size, 7);
        assert_eq!(f.filter_pos[..3], vec![0, 0, 2]);
        assert_eq!(f.row(0), &[6846, 6745, 2505, 279, 9, 0, 0]);
        assert_eq!(f.row(1), &[149, 1732, 6038, 6304, 1971, 185, 5]);
        assert_eq!(f.row(2), &[227, 2230, 6542, 5750, 1513, 119, 3]);

        // lanczos 13→7 left-sited: 11 taps, negative lobes present.
        let f = init_filter(x_inc(13, 7), 13, 7, ONE_H, TableScaler::Lanczos, 64, 128).unwrap();
        assert_eq!(f.size, 11);
        assert_eq!(
            f.row(0),
            &[6428, 8479, 2871, -1191, -501, 279, 19, 0, 0, 0, 0]
        );
        assert_eq!(
            f.row(1),
            &[-1527, 1574, 7686, 7997, 1991, -1305, -299, 264, 3, 0, 0]
        );
    }

    /// The error-diffusion invariant: every H row sums to exactly 16384 and
    /// every V row to exactly 4096; positions stay inside the source; taps
    /// beyond the source are zero (the border invariant the apply path
    /// relies on). One upscale + one downscale + one odd-geometry config
    /// per algorithm, H and V.
    #[test]
    fn filter_rows_sum_to_one() {
        for scaler in all_scalers() {
            for (sw, dw) in [(64, 32), (32, 64), (13, 7)] {
                for (one, name) in [(ONE_H, "H"), (ONE_V, "V")] {
                    let f = init_filter(x_inc(sw, dw), sw, dw, one, scaler, 128, 128)
                        .unwrap_or_else(|e| panic!("{scaler:?} {sw}->{dw} {name}: {e}"));
                    for i in 0..dw as usize {
                        let sum: i32 = f.row(i).iter().map(|&c| c as i32).sum();
                        assert_eq!(sum, one as i32, "{scaler:?} {sw}->{dw} {name} row {i}");
                        assert!(
                            (0..sw).contains(&f.filter_pos[i]),
                            "{scaler:?} pos in range"
                        );
                        for (j, &c) in f.row(i).iter().enumerate() {
                            if f.filter_pos[i] as usize + j >= sw as usize {
                                assert_eq!(c, 0, "{scaler:?} out-of-range tap must be zero");
                            }
                        }
                    }
                }
            }
        }
    }

    /// Kernel-value pins, hand-computed from the C expressions.
    #[test]
    fn kernel_value_pins() {
        let f2 = fone(4, 2); // 1<<53

        // spline recursion at 0.5 (single segment): hand value 0.600481…
        let p = -2.196152422706632f64;
        let v = get_spline_coeff(1.0, 0.0, p, -p - 1.0, 0.5);
        assert!((v - 0.600480947161671).abs() < 1e-15, "{v}");
        // and past the first segment the recursion shifts coefficients AND
        // dist−1: value at 1.25 equals the value of the shifted polynomial
        // at 0.25.
        let v2 = get_spline_coeff(1.0, 0.0, p, -p - 1.0, 1.25);
        let shifted = get_spline_coeff(
            0.0,
            0.0 + 2.0 * p + 3.0 * (-p - 1.0),
            p + 3.0 * (-p - 1.0),
            -0.0 - 3.0 * p - 6.0 * (-p - 1.0),
            0.25,
        );
        assert!((v2 - shifted).abs() < 1e-12);

        // gauss is exp2: 2^(−3·0.25) = 2^−0.75 = 0.5946035575013605…
        let g = kernel_coefficient(
            TableScaler::Gauss,
            (0.5 * (1 << 30) as f64) as i64,
            0.5,
            0x20000,
            f2,
        );
        assert_eq!(g, (2f64.powf(-0.75) * f2 as f64) as i64);
        assert!((2f64.powf(-0.75) - 0.5946035575013605).abs() < 1e-15);

        // All double kernels hit exactly fone at d == 0 (the ternaries test
        // the integer d), and spline collapses to a = 1.0 at dist 0.
        for scaler in all_scalers() {
            let c = kernel_coefficient(scaler, 0, 0.0, 0x20000, f2);
            assert_eq!(c, f2, "{scaler:?} at d=0");
        }

        // lanczos zeroes floatd > 3.0 after the product.
        let l = kernel_coefficient(
            TableScaler::Lanczos,
            (3.5 * (1 << 30) as f64) as i64,
            3.5,
            0x20000,
            f2,
        );
        assert_eq!(l, 0);
        // …but keeps the small nonzero at 2.5.
        let l = kernel_coefficient(
            TableScaler::Lanczos,
            (2.5 * (1 << 30) as f64) as i64,
            2.5,
            0x20000,
            f2,
        );
        assert!(l > 0 && l < f2 / 4, "{l}");

        // sinc at integer floatd (sin(n·π) ≈ 0 in double) quantizes tiny.
        let s = kernel_coefficient(TableScaler::Sinc, 8 << 30, 8.0, 0x20000, f2);
        assert_eq!(s, 0, "sin(8π) is ~1e-15 ⇒ truncates to 0");

        // area trapezoid boundaries: d2·xInc vs ±2^45 with xInc = 0x20000.
        // At d = 0: d2 = −2^29 ⇒ d2·xInc = −2^46 < −2^45 ⇒ full weight.
        let a = kernel_coefficient(TableScaler::Area, 0, 0.0, 0x20000, f2);
        assert_eq!(a, (1i64 << 46) * (f2 >> 46));
        // At floatd = 2.0: d2 = 3·2^29 ⇒ d2·xInc = 3·2^46 > 2^45 ⇒ zero.
        let a_far = kernel_coefficient(TableScaler::Area, 2 << 30, 2.0, 0x20000, f2);
        assert_eq!(a_far, 0);
    }

    /// x_inc and fone pins.
    #[test]
    fn x_inc_rounding_pins() {
        assert_eq!(x_inc(128, 64), 131072); // exactly 2.0
        assert_eq!(x_inc(100, 64), 102400); // 0x19000
        assert_eq!(x_inc(4, 2), 131072);
        // fone: integer division BEFORE av_log2; upscale ⇒ 1<<54
        // (av_log2(0) == 0); 2× down ⇒ shift 1 ⇒ 1<<53; ≥256× clamps the
        // shift at 8 ⇒ fone ≥ 1<<46.
        assert_eq!(fone(64, 32), 1 << 53);
        assert_eq!(fone(32, 64), 1 << 54);
        assert_eq!(fone(1, 1), 1 << 54);
        assert_eq!(fone(2048, 8), 1 << 46);
        assert_eq!(fone(4096, 8), 1 << 46);
    }

    /// End-to-end row-sum invariant: a constant plane stays constant under
    /// every table algorithm, up and down.
    #[test]
    fn constant_plane_is_identity_under_table_resampling() {
        for scaler in all_scalers() {
            for (sw, sh, dw, dh) in [(9, 7, 17, 13), (32, 24, 16, 12)] {
                let src = vec![137u8; sw * sh];
                let mut dst = vec![0u8; dw * dh];
                let plan = build_plan(
                    scaler,
                    (sw as i32, sh as i32),
                    (dw as i32, dh as i32),
                    ChromaLocation::Unspecified,
                )
                .unwrap_or_else(|e| panic!("{scaler:?}: {e}"));
                scale_plane(
                    &src,
                    sw,
                    (sw as i32, sh as i32),
                    &mut dst,
                    dw,
                    &plan.h_lum,
                    &plan.v_lum,
                );
                assert!(
                    dst.iter().all(|&b| b == 137),
                    "{scaler:?} {sw}x{sh}->{dw}x{dh}: {:?}",
                    {
                        let min = *dst.iter().min().unwrap();
                        let max = *dst.iter().max().unwrap();
                        (min, max)
                    }
                );
            }
        }
    }

    /// yuv2planeX with a single 4096 tap IS yuv2plane1's
    /// `(v + 64) >> 7`: `4096·(v+64) >> 19 == (v+64) >> 7` (output.c:485).
    #[test]
    fn yuv2plane_x_8_matches_plane1() {
        let vals = [0i16, 1, 64, 127, -1, -64, -128, 32767, -32768, 100, -100];
        let mut lines: Vec<&[i16]> = Vec::new();
        let mut line = vec![0i16; vals.len()];
        for (i, &v) in vals.iter().enumerate() {
            line[i] = v;
        }
        lines.push(&line[..]);
        let mut got = vec![0u8; vals.len()];
        yuv2plane_x_8(&mut got, &lines, &[4096]);
        for (i, &v) in vals.iter().enumerate() {
            let expect = ((v as i32 + 64) >> 7).clamp(0, 255) as u8;
            assert_eq!(got[i], expect, "v={v}");
        }
    }

    /// filterSize ≥ 256 refuses instead of cascading (utils.c:492-493 +
    /// 1806-1834; documented divergence — C transparently chains two
    /// contexts at the geometric mean).
    #[test]
    fn cascade_ratio_errors() {
        // sinc 1920→120: raw size 1+ceil(20·16)=321, post-reduction ≥ 256.
        let err = init_filter(
            x_inc(1920, 120),
            1920,
            120,
            ONE_H,
            TableScaler::Sinc,
            128,
            128,
        )
        .unwrap_err();
        match err {
            Error::Unsupported(msg) => assert!(msg.contains("cascaded"), "{msg}"),
            other => panic!("expected Unsupported, got {other:?}"),
        }
        // …while ordinary ratios stay well clear.
        assert!(
            init_filter(
                x_inc(1920, 480),
                1920,
                480,
                ONE_H,
                TableScaler::Sinc,
                128,
                128
            )
            .is_ok()
        );
    }

    /// Chroma siting positions — the §`chroma_pos` table.
    #[test]
    fn chroma_positions() {
        use crate::util::color::ChromaLocation as L;
        let h = |l| chroma_pos(l, true);
        let v = |l| chroma_pos(l, false);
        assert_eq!((h(L::Unspecified), v(L::Unspecified)), (128, 128));
        assert_eq!((h(L::Center), v(L::Center)), (128, 128));
        assert_eq!((h(L::Left), v(L::Left)), (64, 128));
        assert_eq!((h(L::TopLeft), v(L::TopLeft)), (64, 64));
        assert_eq!((h(L::Top), v(L::Top)), (128, 64));
        assert_eq!((h(L::BottomLeft), v(L::BottomLeft)), (64, 192));
        assert_eq!((h(L::Bottom), v(L::Bottom)), (128, 192));
    }

    /// Luma tables of an identity geometry take the unscaled special case
    /// (utils.c:219-228): one 16384 tap per output, identity positions.
    #[test]
    fn identity_geometry_is_special_a() {
        let f = init_filter(x_inc(9, 9), 9, 9, ONE_H, TableScaler::Spline, 128, 128).unwrap();
        assert_eq!(f.size, 1);
        assert_eq!(f.filter_pos, vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(f.filter.iter().all(|&c| c == 16384));
        // Area takes the 2-tap special-C loop on upscale-xInc even at
        // identity size — but only when positions differ; with equal
        // positions special A wins first (branch order, utils.c:219 > 244).
        let f = init_filter(x_inc(9, 9), 9, 9, ONE_H, TableScaler::Area, 64, 128).unwrap();
        assert_eq!(f.size, 2, "area identity-size with differing siting");
        assert_eq!(f.row(0), &[12288, 4096]);
        // Gauss with differing siting goes through the general branch
        // (9 raw taps, trimmed to 3 by the near-zero reduction — both
        // verified against the reference C).
        let f = init_filter(x_inc(9, 9), 9, 9, ONE_H, TableScaler::Gauss, 64, 128).unwrap();
        assert_eq!(f.size, 3);
        assert_eq!(f.row(0), &[12240, 4144, 0]);
    }
}
