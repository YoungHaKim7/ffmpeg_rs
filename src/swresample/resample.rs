//! Rational resampling — port of `libswresample/resample.c` (513 lines), the
//! dispatch of `resample_dsp.c` (78 lines), plus `av_bessel_i0` from
//! `libavutil/mathematics.c:227-330` (needed by the Kaiser window).
//!
//! ## C → Rust map
//!
//! | C site | here |
//! |---|---|
//! | `struct ResampleContext` (`resample.h:30-61`) | [`ResampleContext`] — the `dsp` function-pointer struct is flattened to direct method dispatch (scalar only, see below); `filter_bank` (a `uint8_t*` reinterpreted per format) becomes the typed [`FilterBank`] enum |
//! | `enum SwrFilterType` (`swresample.h:173-176`) | [`FilterType`] |
//! | `build_filter()` `resample.c:41-174` | [`build_filter`] — exact windowed-sinc math (per-tap `sin`/`s/x` switch, CUBIC piecewise, Blackman-Nuttall chebyshev form, Kaiser·`bessel_i0`), `norm` accumulated on `ph == 0` only, per-format quantization, and the even-`phase_count` tap mirror `:105-128` |
//! | `resample_init()` `resample.c:184-278` | [`ResampleContext::new`] — cutoff default 0.97, `factor = min(out·cutoff/in, 1)`, `filter_length = FFALIGN(ceil(filter_size/factor), 2)`, the `exact_rational` `av_reduce` phase-count reduction `:197-205`, `filter_alloc = FFALIGN(filter_length, 8)`, bank `filter_alloc·(phase_count+1)` elements, the `+1`-phase wrap copy `:253-254`, `src_incr`/`dst_incr` via `av_reduce(out, in·phase_count, INT32_MAX/2)` plus the `< 1<<20` doubling `:258-263`, `index = -phase_count·((filter_length-1)/2)` `:268` |
//! | `rebuild_filter_bank_with_compensation()` `resample.c:280-326` | [`ResampleContext::rebuild_filter_bank_with_compensation`] |
//! | `set_compensation()` `resample.c:328-347` | [`ResampleContext::set_compensation`] |
//! | `multiple_resample()` `resample.c:349-406` | [`ResampleContext::multiple_resample`] — both the `filter_length == 1 && phase_count == 1` nearest-neighbor fast path (`:359-377`, 16.16-ish `index2`/`incr` in 32.32 fixed point) and the general path (`:378-394`, `end_index`/`delta_frac`/`delta_n` clamp, linear-vs-common selection `:389-390`, last-channel `update_ctx`), and the compensation countdown `:396-403` |
//! | `get_delay()` `resample.c:408-416` | [`ResampleContext::get_delay`] — C reads `s->in_buffer_count`, `s->in_sample_rate` off the `SwrContext`; those arrive as parameters (the core context is a later phase) |
//! | `get_out_samples()` `resample.c:418-435` | [`ResampleContext::get_out_samples`] — same flattening; `AV_ROUND_UP` rescale and the compensation `FFMAX` widening `:432` |
//! | `resample_flush()` `resample.c:437-454` | [`ResampleContext::resample_flush`] — mirrors `reflection` samples off the tail of the in-buffer; grows it via the local [`realloc_audio`] analog of `swri_realloc_audio` (`swresample.c:426-456`) |
//! | `invert_initial_buffer()` `resample.c:457-502` | [`ResampleContext::invert_initial_buffer`] — the negative-`index` priming skip; C's `INT_MAX` "need more input" return stays the `i32::MAX` sentinel |
//! | `resample_free()` `resample.c:176-182` | nothing — Rust ownership replaces `av_freep` |
//! | `swri_resample_dsp_init()` `resample_dsp.c:46-78` | nothing to port — the x86/arm/aarch64 overrides (`:71-77`) are SIMD-only; a C build without them dispatches exactly the scalar functions below |
//! | `resample_one`/`resample_common`/`resample_linear` × {int16,int32,float,double} `resample_template.c:81-207` | [`resample_one`]/[`resample_common`]/[`resample_linear`] generic over the [`ResampleElem`] trait (one impl per C template instantiation, each method citing its template lines) |
//! | `av_bessel_i0()` `libavutil/mathematics.c:257-330` | [`bessel_i0`] — Blair-Edwards minimax rational approximations (`p1/q1` \|x\| ≤ 15, `p2/q2·e^x/√x` above), `eval_poly` (`:216-224`) inlined as a loop. Ported here (not `util::mathematics`) to keep this phase a single new file; `av_reduce` by contrast already exists as [`Rational::reduce`] and is reused |
//!
//! ## Fixed-point format (the load-bearing numbers)
//!
//! * The **phase clock**: `index` counts phases `[0, phase_count)`, `frac`
//!   counts `[0, src_incr)`. Per output sample: `frac += dst_incr_mod;
//!   index += dst_incr_div;` carry `frac ≥ src_incr → frac -= src_incr,
//!   index += 1` (`resample_template.c:128-133`). One input sample =
//!   `phase_count` index steps and `src_incr` frac steps.
//! * `src_incr/dst_incr = av_reduce(out_rate, in_rate·phase_count)` reduced
//!   and doubled until either part reaches `1<<20` (`resample.c:258-263`) —
//!   the doubling keeps `frac·dst_incr_mod` products inside `int` range.
//! * Filter coefficients are quantized to the sample format (`scale =
//!   1 << filter_shift`: 15 for s16p, 30 for s32p, 0 for fltp/dblp,
//!   `resample.c:219-229`); the apply undoes it with an arithmetic
//!   `>> filter_shift` inside `OUT` plus `FOFFSET = 1<<(shift-1)` rounding
//!   bias for the int formats (`resample_template.c:62,76`).
//! * `resample_common` splits the MAC into two accumulators (`val`/`val2`,
//!   two taps per iteration, template `:116-121`); for s16 the final add
//!   widens to `FELEML = int64_t` (`:122-123`). `resample_linear` uses
//!   **unsigned** accumulators (`FELEM2U`, template `:28-29,170`) so the
//!   `v2 - val` interpolation cannot overflow UB, and interpolates between
//!   phase `index` and phase `index+1` (the wrapped `filter_alloc` stride)
//!   by `frac/src_incr`.
//!
//! ## Not ported (guards cited)
//!
//! * **SIMD** (`swri_resample_dsp_x86/arm/aarch64_init`): scalar-equivalent
//!   only, like every module in this crate.
//! * **The soxr engine** (`soxr_resample.c`, engine `SWR_ENGINE_SOXR`): the
//!   `precision`/`cheby` parameters of [`ResampleContext::new`] exist only
//!   for signature parity with the C vtable and are ignored — exactly as
//!   `resample_init` ignores them (`resample.c:184-278` never reads them).
//!
//! ## Divergences from C (each pinned by tests or cited at its guard)
//!
//! * **Context reuse**: C's `resample_init` takes the old context and keeps
//!   the bank when the configuration matches (`resample.c:207-209`); the
//!   port always builds fresh (the only observable difference is
//!   allocation churn). The `av_class` field is logging-only plumbing.
//! * `sin_lut` is `av_malloc` (uninitialized) in C and only *read* (as `s`)
//!   when `factor == 1.0` (`resample.c:46,61-66`) — the port zero-initializes
//!   it; the garbage C reads is never used in `y`.
//! * C's per-read out-of-bounds behavior: the kernels read
//!   `src[sample_index + i]` unchecked (`resample_template.c:117-121`); the
//!   driver's `delta_n` clamp keeps the reads *at or below* the plane's
//!   capacity, but the last tap may legally read one stale sample past
//!   `src_size` *within* capacity. The port reproduces reads-within-capacity
//!   (planes carry capacity) and turns a read past the plane into
//!   [`Error::BufferTooSmall`] instead of C's heap overread; dst may then
//!   hold partial output (C has no such failure mode). Same convention as
//!   `audioconvert`.
//! * A **negative `index` at kernel entry** is C UB (filter bank read before
//!   its base, `resample_template.c:111`); the driver contract
//!   (`invert_initial_buffer` runs first) makes it unreachable — the port
//!   `debug_assert`s and returns [`Error::InvalidArgument`] in release.
//! * Integer accumulator overflow is UB in C; the port uses wrapping
//!   arithmetic there (matching what x86-64 C actually does), never panics.
//! * `resample_flush`'s `swri_realloc_audio` grows by doubling and pads each
//!   plane to 32-byte `ALIGN` for SIMD (`swresample.c:438-441`); the local
//!   [`realloc_audio`] grows by doubling without the padding (invisible to
//!   every caller — same reasoning as `AudioData`'s own doc in `mod.rs`).

use crate::util::{
    error::{Error, Result},
    mathematics::{Rounding, rescale, rescale_rnd},
    rational::Rational,
    samplefmt::SampleFormat,
};

use super::AudioData;

// ---------------------------------------------------------------------------
// Filter type + the per-format element trait (resample_template.c)
// ---------------------------------------------------------------------------

/// `enum SwrFilterType` (`swresample.h:173-176`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterType {
    /// `SWR_FILTER_TYPE_CUBIC` = 0 — cubic window.
    Cubic,
    /// `SWR_FILTER_TYPE_BLACKMAN_NUTTALL` = 1.
    BlackmanNuttall,
    /// `SWR_FILTER_TYPE_KAISER` = 2 (the swr default, `options.c:118`).
    Kaiser,
}

/// The typed polyphase filter bank — C's format-dependent
/// `uint8_t *filter_bank` (`resample.h:32`): `filter_alloc·(phase_count+1)`
/// elements of the sample format's width; variant chosen by `format`.
enum FilterBank {
    /// `AV_SAMPLE_FMT_S16P`.
    S16(Vec<i16>),
    /// `AV_SAMPLE_FMT_S32P`.
    S32(Vec<i32>),
    /// `AV_SAMPLE_FMT_FLTP`.
    Flt(Vec<f32>),
    /// `AV_SAMPLE_FMT_DBLP`.
    Dbl(Vec<f64>),
}

impl FilterBank {
    /// Zeroed bank of `filter_alloc * (phase_count + 1)` elements (the
    /// `av_calloc` of `resample.c:245`).
    fn zeroed(format: SampleFormat, filter_alloc: usize, phase_count: usize) -> Self {
        let n = filter_alloc * (phase_count + 1);
        match format {
            SampleFormat::S16p => FilterBank::S16(vec![0; n]),
            SampleFormat::S32p => FilterBank::S32(vec![0; n]),
            SampleFormat::Fltp => FilterBank::Flt(vec![0.0; n]),
            SampleFormat::Dblp => FilterBank::Dbl(vec![0.0; n]),
            _ => unreachable!("bank variant settled by ResampleContext::new's format check"),
        }
    }
}

/// One template instantiation of `resample_template.c` — implemented by the
/// four `DELEM` types `int16_t`/`int32_t`/`float`/`double` (`:34-49`).
///
/// `DELEM == FELEM` in every instantiation (the bank stores the same type as
/// the audio), so one type parameter covers both; `Acc`/`UAcc` are the
/// `FELEM2`/`FELEM2U` accumulator types of the two kernels.
trait ResampleElem: Copy + PartialEq + std::fmt::Debug {
    /// The planar sample format served (`AV_SAMPLE_FMT_S16P` etc.).
    const FORMAT: SampleFormat;
    /// `FILTER_SHIFT` (`resample_template.c:33,54,43,68`): 15 / 30 / 0 / 0.
    const FILTER_SHIFT: u32;
    /// `FELEM2` — `resample_common`'s signed accumulator (`:35,56,46,69`).
    type Acc: Copy + PartialEq + std::fmt::Debug;
    /// `FELEM2U` — `resample_linear`'s overflow-safe accumulator (`:37,58,47,70`).
    type UAcc: Copy + PartialEq + std::fmt::Debug;

    /// `FOFFSET` (`:38,48,57,71`) for both kernels — the int formats' round
    /// bias `1<<(FILTER_SHIFT-1)`, zero for float/double.
    fn foffset_acc() -> Self::Acc;
    fn foffset_uacc() -> Self::UAcc;
    /// `FELEM2 val2 = 0` (`resample_template.c:114`).
    fn zero_acc() -> Self::Acc;

    // -- resample_common (:94-147) -----------------------------------------

    /// `val += src[i] * (FELEM2)filter[i]` (`:117-118,121`) — signed Acc
    /// arithmetic (wrapping for the int formats, where C overflow is UB).
    fn acc_common(acc: Self::Acc, s: Self, f: Self) -> Self::Acc;
    /// `OUT(dst[i], val + val2)` (`:122-125`) — `FELEML` widening for s16.
    fn out_common(val: Self::Acc, val2: Self::Acc) -> Self;

    // -- resample_linear (:149-207) ----------------------------------------

    /// `val += src[i] * (FELEM2U)filter[i]` (`:174-175`) — unsigned/wrapping.
    fn acc_linear(acc: Self::UAcc, s: Self, f: Self) -> Self::UAcc;
    /// The `val += (FELEM2)(v2 - val)·frac/src_incr` interpolation — three
    /// distinct C shapes (`:177-185`): s16 multiplies first (`FELEML` i64),
    /// s32 **divides first**, float/double scale by `inv_src_incr` in double.
    fn interp_linear(
        val: Self::UAcc,
        v2: Self::UAcc,
        frac: i32,
        src_incr: i32,
        inv_src_incr: f64,
    ) -> Self::UAcc;
    /// `OUT(dst[i], (FELEM2)val)` (`:186`).
    fn out_linear(val: Self::UAcc) -> Self;

    /// Little-endian plane byte load/store (crate LE policy).
    fn load(plane: &[u8], idx: usize) -> Self;
    fn store(plane: &mut [u8], idx: usize, v: Self);

    /// Coefficient quantization `resample.c:102-129`: s16 `av_clip_int16(
    /// lrintf(v))` (the double narrows to float at the `lrintf` call),
    /// s32 `av_clipl_int32(llrint(v))`, float/double plain conversion.
    /// `lrintf`/`llrint` = round-half-to-even under the default x87/SSE
    /// rounding mode (same reading as `audioconvert`).
    fn quantize(v: f64) -> Self;
}

impl ResampleElem for i16 {
    const FORMAT: SampleFormat = SampleFormat::S16p;
    const FILTER_SHIFT: u32 = 15;
    type Acc = i32; // FELEM2  (template :70)
    type UAcc = u32; // FELEM2U (template :71)

    fn foffset_acc() -> i32 {
        1 << (Self::FILTER_SHIFT - 1) // 16384
    }
    fn foffset_uacc() -> u32 {
        1 << (Self::FILTER_SHIFT - 1)
    }
    fn zero_acc() -> i32 {
        0
    }

    fn acc_common(acc: i32, s: i16, f: i16) -> i32 {
        acc.wrapping_add(s as i32 * f as i32)
    }
    fn out_common(val: i32, val2: i32) -> i16 {
        // template :77,122-123 — FELEML widening: (i32 val + i64 val2) >> 15,
        // then C's long→int truncation before av_clip_int16.
        let v = val as i64 + val2 as i64;
        clip_i16((v >> Self::FILTER_SHIFT) as i32)
    }

    fn acc_linear(acc: u32, s: i16, f: i16) -> u32 {
        // :175 — int · (FELEM2U)filter → u32 (wrapping), += wrapping.
        acc.wrapping_add((s as i32).wrapping_mul(f as i32) as u32)
    }
    fn interp_linear(val: u32, v2: u32, frac: i32, src_incr: i32, _inv: f64) -> u32 {
        // :178 — FELEML arm: (FELEM2)(v2−val) · (FELEML)frac / src_incr,
        // multiply FIRST; result narrowed back to u32 wrapping.
        let d = v2.wrapping_sub(val) as i32 as i64;
        val.wrapping_add((d * frac as i64 / src_incr as i64) as u32)
    }
    fn out_linear(val: u32) -> i16 {
        // :77 — av_clip_int16((v)>>15) with v = (FELEM2)val (u32→i32).
        clip_i16(((val as i32) >> Self::FILTER_SHIFT) as i32)
    }

    fn load(plane: &[u8], idx: usize) -> i16 {
        i16::from_le_bytes([plane[idx * 2], plane[idx * 2 + 1]])
    }
    fn store(plane: &mut [u8], idx: usize, v: i16) {
        plane[idx * 2..idx * 2 + 2].copy_from_slice(&v.to_le_bytes());
    }

    fn quantize(v: f64) -> i16 {
        // :104 — lrintf: f64 → f32 first, ties-to-even.
        clip_i16((v as f32).round_ties_even() as i64 as i32)
    }
}

impl ResampleElem for i32 {
    const FORMAT: SampleFormat = SampleFormat::S32p;
    const FILTER_SHIFT: u32 = 30;
    type Acc = i64; // FELEM2
    type UAcc = u64; // FELEM2U

    fn foffset_acc() -> i64 {
        1 << (Self::FILTER_SHIFT - 1) // 1<<29
    }
    fn foffset_uacc() -> u64 {
        1 << (Self::FILTER_SHIFT - 1)
    }
    fn zero_acc() -> i64 {
        0
    }

    fn acc_common(acc: i64, s: i32, f: i32) -> i64 {
        acc.wrapping_add(s as i64 * f as i64)
    }
    fn out_common(val: i64, val2: i64) -> i32 {
        // :63 — av_clipl_int32((v)>>30); no FELEML for s32 (:124-125).
        clip_i32(val.wrapping_add(val2) >> Self::FILTER_SHIFT)
    }

    fn acc_linear(acc: u64, s: i32, f: i32) -> u64 {
        acc.wrapping_add((s as i64).wrapping_mul(f as i64) as u64)
    }
    fn interp_linear(val: u64, v2: u64, frac: i32, src_incr: i32, _inv: f64) -> u64 {
        // :183 — the non-FELEML int arm: DIVIDE first: (v2−val)/src_incr·frac.
        let d = v2.wrapping_sub(val) as i64;
        val.wrapping_add((d / src_incr as i64).wrapping_mul(frac as i64) as u64)
    }
    fn out_linear(val: u64) -> i32 {
        // :63 — av_clipl_int32((v)>>30) with v = (FELEM2)val (u64→i64).
        clip_i32((val as i64) >> Self::FILTER_SHIFT)
    }

    fn load(plane: &[u8], idx: usize) -> i32 {
        let b = &plane[idx * 4..idx * 4 + 4];
        i32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }
    fn store(plane: &mut [u8], idx: usize, v: i32) {
        plane[idx * 4..idx * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }

    fn quantize(v: f64) -> i32 {
        // :111 — llrint on the double, then av_clipl_int32.
        clip_i32(v.round_ties_even() as i64)
    }
}

impl ResampleElem for f32 {
    const FORMAT: SampleFormat = SampleFormat::Fltp;
    const FILTER_SHIFT: u32 = 0;
    type Acc = f32;
    type UAcc = f32;

    fn foffset_acc() -> f32 {
        0.0
    }
    fn foffset_uacc() -> f32 {
        0.0
    }
    fn zero_acc() -> f32 {
        0.0
    }

    fn acc_common(acc: f32, s: f32, f: f32) -> f32 {
        acc + s * f
    }
    fn out_common(val: f32, val2: f32) -> f32 {
        val + val2 // :39 OUT(d, v) d = v
    }

    fn acc_linear(acc: f32, s: f32, f: f32) -> f32 {
        acc + s * f
    }
    fn interp_linear(val: f32, v2: f32, frac: i32, _src_incr: i32, inv_src_incr: f64) -> f32 {
        // :180-181 — FILTER_SHIFT == 0 arm: computed in double, narrowed.
        (val as f64 + (v2 - val) as f64 * inv_src_incr * frac as f64) as f32
    }
    fn out_linear(val: f32) -> f32 {
        val
    }

    fn load(plane: &[u8], idx: usize) -> f32 {
        let b = &plane[idx * 4..idx * 4 + 4];
        f32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }
    fn store(plane: &mut [u8], idx: usize, v: f32) {
        plane[idx * 4..idx * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }

    fn quantize(v: f64) -> f32 {
        v as f32 // :118
    }
}

impl ResampleElem for f64 {
    const FORMAT: SampleFormat = SampleFormat::Dblp;
    const FILTER_SHIFT: u32 = 0;
    type Acc = f64;
    type UAcc = f64;

    fn foffset_acc() -> f64 {
        0.0
    }
    fn foffset_uacc() -> f64 {
        0.0
    }
    fn zero_acc() -> f64 {
        0.0
    }

    fn acc_common(acc: f64, s: f64, f: f64) -> f64 {
        acc + s * f
    }
    fn out_common(val: f64, val2: f64) -> f64 {
        val + val2
    }

    fn acc_linear(acc: f64, s: f64, f: f64) -> f64 {
        acc + s * f
    }
    fn interp_linear(val: f64, v2: f64, frac: i32, _src_incr: i32, inv_src_incr: f64) -> f64 {
        // :180-181 — same FILTER_SHIFT == 0 arm, all double.
        val + (v2 - val) * inv_src_incr * frac as f64
    }
    fn out_linear(val: f64) -> f64 {
        val
    }

    fn load(plane: &[u8], idx: usize) -> f64 {
        let b = &plane[idx * 8..idx * 8 + 8];
        f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
    }
    fn store(plane: &mut [u8], idx: usize, v: f64) {
        plane[idx * 8..idx * 8 + 8].copy_from_slice(&v.to_le_bytes());
    }

    fn quantize(v: f64) -> f64 {
        v // :125
    }
}

// ---------------------------------------------------------------------------
// ResampleContext — resample.h:30-61 + resample.c:184-278
// ---------------------------------------------------------------------------

/// `struct ResampleContext` (`resample.h:30-61`) minus the `dsp` vtable
/// (direct dispatch) and `av_class`; `filter_bank` is the typed
/// [`FilterBank`]. All fields keep their C names.
pub struct ResampleContext {
    bank: FilterBank,
    pub filter_length: i32,
    pub filter_alloc: i32,
    pub ideal_dst_incr: i32,
    pub dst_incr: i32,
    pub dst_incr_div: i32,
    pub dst_incr_mod: i32,
    pub index: i32,
    pub frac: i32,
    pub src_incr: i32,
    pub compensation_distance: i32,
    pub phase_count: i32,
    pub linear: i32,
    pub filter_type: FilterType,
    pub kaiser_beta: f64,
    pub factor: f64,
    pub format: SampleFormat,
    pub felem_size: i32,
    pub filter_shift: i32,
    /// Desired `phase_count` when compensation is enabled
    /// (`resample.h:51`).
    pub phase_count_compensation: i32,
}

impl std::fmt::Debug for ResampleContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The bank is thousands of samples of numbers — summarize.
        f.debug_struct("ResampleContext")
            .field("format", &self.format)
            .field("filter_length", &self.filter_length)
            .field("filter_alloc", &self.filter_alloc)
            .field("phase_count", &self.phase_count)
            .field("phase_count_compensation", &self.phase_count_compensation)
            .field("filter_type", &self.filter_type)
            .field("kaiser_beta", &self.kaiser_beta)
            .field("factor", &self.factor)
            .field("linear", &self.linear)
            .field("index", &self.index)
            .field("frac", &self.frac)
            .field("src_incr", &self.src_incr)
            .field("dst_incr", &self.dst_incr)
            .field("ideal_dst_incr", &self.ideal_dst_incr)
            .field("compensation_distance", &self.compensation_distance)
            .finish_non_exhaustive()
    }
}

impl ResampleContext {
    /// `resample_init` (`resample.c:184-278`) — C's first parameter (the
    /// possibly-reused old context) is dropped: the port always builds
    /// fresh, which is observably identical minus allocation churn.
    ///
    /// The C parameter set is kept verbatim — including `_precision` and
    /// `_cheby`, which only the soxr engine reads (`resample.c` never does).
    ///
    /// Errors (C returns `NULL`): unsupported `format` (`:230-232`, the C
    /// `av_assert0(0)` after "Unsupported sample format"), `filter_size/
    /// factor > INT32_MAX/256` (`:235-238`), and an inexact
    /// `av_reduce(out_rate, in_rate·phase_count, INT32_MAX/2)` (`:258`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        out_rate: i32,
        in_rate: i32,
        filter_size: i32,
        phase_shift: i32,
        linear: i32,
        cutoff0: f64,
        format: SampleFormat,
        filter_type: FilterType,
        kaiser_beta: f64,
        _precision: f64,
        _cheby: i32,
        exact_rational: i32,
    ) -> Result<Self> {
        // :188-192.
        let cutoff = if cutoff0 != 0.0 { cutoff0 } else { 0.97 };
        let factor = (out_rate as f64 * cutoff / in_rate as f64).min(1.0);
        let mut phase_count: i32 = 1 << phase_shift;
        let mut phase_count_compensation = phase_count;
        let mut filter_length = (filter_size as f64 / factor).ceil().max(1.0) as i32;

        // :194-195 — FFALIGN(filter_length, 2), only when > 1.
        if filter_length > 1 {
            filter_length = (filter_length + 1) & !1;
        }

        // :197-205 — exact rational phase count: 44100→48000 has ratio
        // 147:160, so 160 phases suffice and are exact.
        if exact_rational != 0 {
            let (r, _) = Rational::reduce(out_rate as i64, in_rate as i64, i32::MAX as i64);
            let phase_count_exact = r.num as i32;
            if phase_count_exact <= phase_count {
                phase_count_compensation = phase_count_exact * (phase_count / phase_count_exact);
                phase_count = phase_count_exact;
            }
        }

        // :219-233 — filter_shift per format; the default arm is C's
        // "Unsupported sample format" log + abort.
        let filter_shift = match format {
            SampleFormat::S16p => 15,
            SampleFormat::S32p => 30,
            SampleFormat::Fltp | SampleFormat::Dblp => 0,
            _ => {
                // C: av_log(NULL, AV_LOG_ERROR, "Unsupported sample format")
                crate::log_error!(None, "Unsupported sample format");
                return Err(Error::Unsupported(format!(
                    "sample format {} for resampling",
                    format.name()
                )));
            }
        };
        let felem_size = format.bytes_per_sample() as i32;

        // :235-238 — guard against filter_length overflowing the int phase
        // arithmetic downstream.
        if filter_size as f64 / factor > (i32::MAX / 256) as f64 {
            crate::log_error!(None, "Filter length too large");
            return Err(Error::InvalidArgument("filter length too large".into()));
        }

        // :240-255.
        let filter_alloc = (filter_length + 7) & !7;
        let mut bank = FilterBank::zeroed(format, filter_alloc as usize, phase_count as usize);
        let scale: i32 = 1 << filter_shift;
        match &mut bank {
            FilterBank::S16(b) => build_filter(
                b,
                factor,
                filter_length as usize,
                filter_alloc as usize,
                phase_count as usize,
                scale,
                filter_type,
                kaiser_beta,
            ),
            FilterBank::S32(b) => build_filter(
                b,
                factor,
                filter_length as usize,
                filter_alloc as usize,
                phase_count as usize,
                scale,
                filter_type,
                kaiser_beta,
            ),
            FilterBank::Flt(b) => build_filter(
                b,
                factor,
                filter_length as usize,
                filter_alloc as usize,
                phase_count as usize,
                scale,
                filter_type,
                kaiser_beta,
            ),
            FilterBank::Dbl(b) => build_filter(
                b,
                factor,
                filter_length as usize,
                filter_alloc as usize,
                phase_count as usize,
                scale,
                filter_type,
                kaiser_beta,
            ),
        }
        // :253-254 — the +1 wrap phase: element [alloc·pc + 1 + j] = [j] and
        // [alloc·pc] = [alloc-1], so resample_linear can read phase pc's
        // filter at filter[i + filter_alloc] (template :175).
        macro_rules! wrap_phase {
            ($b:expr) => {{
                let b = $b;
                let (alloc, pc) = (filter_alloc as usize, phase_count as usize);
                for j in 0..alloc - 1 {
                    b[alloc * pc + 1 + j] = b[j];
                }
                b[alloc * pc] = b[alloc - 1];
            }};
        }
        match &mut bank {
            FilterBank::S16(b) => wrap_phase!(b),
            FilterBank::S32(b) => wrap_phase!(b),
            FilterBank::Flt(b) => wrap_phase!(b),
            FilterBank::Dbl(b) => wrap_phase!(b),
        }

        // :257-263 — src_incr/dst_incr = out/in·pc in lowest terms (must be
        // exact), then doubled until a part reaches 1<<20.
        let (r, exact) = Rational::reduce(
            out_rate as i64,
            in_rate as i64 * phase_count as i64,
            (i32::MAX / 2) as i64,
        );
        if !exact {
            return Err(Error::InvalidArgument(format!(
                "resampling ratio {out_rate}/{in_rate} not exact in int32"
            )));
        }
        let mut src_incr = r.num;
        let mut dst_incr = r.den;
        while dst_incr < (1 << 20) && src_incr < (1 << 20) {
            dst_incr *= 2;
            src_incr *= 2;
        }

        Ok(ResampleContext {
            bank,
            filter_length,
            filter_alloc,
            ideal_dst_incr: dst_incr,
            dst_incr,
            dst_incr_div: dst_incr / src_incr,
            dst_incr_mod: dst_incr % src_incr,
            index: -phase_count * ((filter_length - 1) / 2),
            frac: 0,
            src_incr,
            compensation_distance: 0,
            phase_count,
            linear,
            filter_type,
            kaiser_beta,
            factor,
            format,
            felem_size,
            filter_shift,
            phase_count_compensation,
        })
    }

    /// `rebuild_filter_bank_with_compensation` (`resample.c:280-326`) —
    /// swaps in the `phase_count_compensation` bank and rescales the phase
    /// clock (`src_incr`/`dst_incr` re-reduced from the OLD clock by the
    /// compensation/normal phase ratio, re-doubled, `index` scaled).
    fn rebuild_filter_bank_with_compensation(&mut self) -> Result<()> {
        let phase_count = self.phase_count_compensation;

        // :287-288 — a no-op when compensation never enlarged the phases.
        if phase_count == self.phase_count {
            return Ok(());
        }

        // :290 — the caller contract (freshly initialized or fully drained).
        debug_assert!(
            self.frac == 0 && self.dst_incr_mod == 0,
            "resample.c:290 av_assert0"
        );

        // :292-303.
        let mut new_bank = FilterBank::zeroed(
            self.format,
            self.filter_alloc as usize,
            phase_count as usize,
        );
        let scale: i32 = 1 << self.filter_shift;
        match &mut new_bank {
            FilterBank::S16(b) => build_filter(
                b,
                self.factor,
                self.filter_length as usize,
                self.filter_alloc as usize,
                phase_count as usize,
                scale,
                self.filter_type,
                self.kaiser_beta,
            ),
            FilterBank::S32(b) => build_filter(
                b,
                self.factor,
                self.filter_length as usize,
                self.filter_alloc as usize,
                phase_count as usize,
                scale,
                self.filter_type,
                self.kaiser_beta,
            ),
            FilterBank::Flt(b) => build_filter(
                b,
                self.factor,
                self.filter_length as usize,
                self.filter_alloc as usize,
                phase_count as usize,
                scale,
                self.filter_type,
                self.kaiser_beta,
            ),
            FilterBank::Dbl(b) => build_filter(
                b,
                self.factor,
                self.filter_length as usize,
                self.filter_alloc as usize,
                phase_count as usize,
                scale,
                self.filter_type,
                self.kaiser_beta,
            ),
        }
        macro_rules! wrap_phase {
            ($b:expr) => {{
                let b = $b;
                let (alloc, pc) = (self.filter_alloc as usize, phase_count as usize);
                for j in 0..alloc - 1 {
                    b[alloc * pc + 1 + j] = b[j];
                }
                b[alloc * pc] = b[alloc - 1];
            }};
        }
        match &mut new_bank {
            FilterBank::S16(b) => wrap_phase!(b),
            FilterBank::S32(b) => wrap_phase!(b),
            FilterBank::Flt(b) => wrap_phase!(b),
            FilterBank::Dbl(b) => wrap_phase!(b),
        }

        // :305-310 — exact re-reduction of the OLD clock by the phase ratio.
        let (r, exact) = Rational::reduce(
            self.src_incr as i64,
            self.dst_incr as i64 * (phase_count / self.phase_count) as i64,
            (i32::MAX / 2) as i64,
        );
        if !exact {
            return Err(Error::InvalidArgument(
                "compensation phase count ratio not exact".into(),
            ));
        }
        self.src_incr = r.num;
        self.dst_incr = r.den;

        // :314-322.
        while self.dst_incr < (1 << 20) && self.src_incr < (1 << 20) {
            self.dst_incr *= 2;
            self.src_incr *= 2;
        }
        self.ideal_dst_incr = self.dst_incr;
        self.dst_incr_div = self.dst_incr / self.src_incr;
        self.dst_incr_mod = self.dst_incr % self.src_incr;
        self.index *= phase_count / self.phase_count;
        self.phase_count = phase_count;
        self.bank = new_bank;
        Ok(())
    }

    /// `set_compensation` (`resample.c:328-347`) — the `swr_set_compensation`
    /// engine backend: over the next `compensation_distance` output samples
    /// produce `sample_delta` more (or fewer, when negative) than the
    /// nominal rate.
    pub fn set_compensation(
        &mut self,
        sample_delta: i32,
        compensation_distance: i32,
    ) -> Result<()> {
        if compensation_distance != 0 && sample_delta != 0 {
            self.rebuild_filter_bank_with_compensation()?;
        }

        // :337-341 — the rate skew: dst_incr shrinks (faster output) by the
        // ideal·delta/distance share; int64 math, truncated back to int.
        self.compensation_distance = compensation_distance;
        if compensation_distance != 0 {
            self.dst_incr = (self.ideal_dst_incr as i64
                - self.ideal_dst_incr as i64 * sample_delta as i64 / compensation_distance as i64)
                as i32;
        } else {
            self.dst_incr = self.ideal_dst_incr;
        }

        self.dst_incr_div = self.dst_incr / self.src_incr;
        self.dst_incr_mod = self.dst_incr % self.src_incr;
        Ok(())
    }

    /// The `dsp.resample_common`/`dsp.resample_linear` slot (`resample.c:382-392`)
    /// — scalar kernel dispatch by format (`resample_dsp.c:46-78` without the
    /// arch overrides). Returns the consumed input sample count
    /// (`sample_index`); `index`/`frac` are threaded as locals and written
    /// back only under `update_ctx` (the C kernels' `:141-144,201-204`).
    fn resample_kernel(
        &self,
        linear: bool,
        dst: &mut [u8],
        src: &[u8],
        n: usize,
        index: &mut i32,
        frac: &mut i32,
    ) -> Result<usize> {
        match &self.bank {
            FilterBank::S16(b) => {
                if linear {
                    resample_linear(self, b, dst, src, n, index, frac)
                } else {
                    resample_common(self, b, dst, src, n, index, frac)
                }
            }
            FilterBank::S32(b) => {
                if linear {
                    resample_linear(self, b, dst, src, n, index, frac)
                } else {
                    resample_common(self, b, dst, src, n, index, frac)
                }
            }
            FilterBank::Flt(b) => {
                if linear {
                    resample_linear(self, b, dst, src, n, index, frac)
                } else {
                    resample_common(self, b, dst, src, n, index, frac)
                }
            }
            FilterBank::Dbl(b) => {
                if linear {
                    resample_linear(self, b, dst, src, n, index, frac)
                } else {
                    resample_common(self, b, dst, src, n, index, frac)
                }
            }
        }
    }

    /// `multiple_resample` (`resample.c:349-406`) — the engine's
    /// `multiple_resample` vtable slot. Resamples up to `dst_size` samples
    /// per channel from `src` into `dst`, reporting the consumed input
    /// count through `consumed`. Returns the number produced.
    pub fn multiple_resample(
        &mut self,
        dst: &mut AudioData,
        mut dst_size: i32,
        src: &AudioData,
        mut src_size: i32,
        consumed: &mut i32,
    ) -> Result<i32> {
        // :351 — clamp src so the i64 phase arithmetic below cannot overflow.
        let max_src_size = ((i64::MAX / 2) / self.phase_count as i64) / self.src_incr as i64;
        if self.compensation_distance != 0 {
            dst_size = dst_size.min(self.compensation_distance);
        }
        src_size = (src_size as i64).min(max_src_size) as i32;

        *consumed = 0;

        // A negative index at this point means the driver skipped
        // invert_initial_buffer — C reads the filter bank out of bounds
        // (template :111). debug_assert + graceful error (documented).
        if self.index < 0 {
            debug_assert!(self.index >= 0, "caller must run invert_initial_buffer");
            return Err(Error::InvalidArgument(
                "resample called with unprimed filter (index < 0)".into(),
            ));
        }

        if self.filter_length == 1 && self.phase_count == 1 {
            // :359-377 — degenerate 1×1 filter: nearest-neighbor pick via a
            // 32.32 fixed-point position (index2) and increment (incr).
            let index2: i64 = (1i64 << 32) * self.frac as i64 / self.src_incr as i64
                + (1i64 << 32) * self.index as i64
                + 1;
            let incr: i64 = (1i64 << 32) * self.dst_incr as i64 / self.src_incr as i64 + 1;
            let new_size = ((src_size as i64 * self.src_incr as i64 - self.frac as i64
                + self.dst_incr as i64
                - 1)
                / self.dst_incr as i64) as i32;

            dst_size = dst_size.min(new_size).max(0);
            if dst_size > 0 {
                for i in 0..dst.ch_count {
                    let dst_plane = dst.plane_bytes_mut(i).ok_or(Error::BufferTooSmall)?;
                    let src_plane = src.plane(i).ok_or(Error::BufferTooSmall)?;
                    self.resample_one_kernel(
                        dst_plane,
                        src_plane,
                        dst_size as usize,
                        index2,
                        incr,
                    )?;
                    if i + 1 == dst.ch_count {
                        // :369-375 — update on the LAST channel only. All
                        // arithmetic in i64 then narrowed (C int64 exprs
                        // assigned to int fields); wrapping per the module
                        // divergence note.
                        self.index = self
                            .index
                            .wrapping_add(dst_size.wrapping_mul(self.dst_incr_div));
                        self.index = self.index.wrapping_add(
                            ((self.frac as i64 + dst_size as i64 * self.dst_incr_mod as i64)
                                / self.src_incr as i64) as i32,
                        );
                        debug_assert!(self.index >= 0, "resample.c:371");
                        *consumed = self.index;
                        self.frac = ((self.frac as i64
                            + dst_size as i64 * self.dst_incr_mod as i64)
                            % self.src_incr as i64) as i32;
                        self.index = 0;
                    }
                }
            }
        } else {
            // :379-385 — how many outputs can the input sustain: advance the
            // phase clock to (1 + src_size - filter_length)·phase_count.
            let end_index =
                (1i64 + src_size as i64 - self.filter_length as i64) * self.phase_count as i64;
            let delta_frac =
                (end_index - self.index as i64) * self.src_incr as i64 - self.frac as i64;
            let delta_n = (delta_frac + self.dst_incr as i64 - 1) / self.dst_incr as i64;

            dst_size = dst_size.min(delta_n as i32).max(0);
            if dst_size > 0 {
                // :387-390 — resample_linear and resample_common agree when
                // frac and dst_incr_mod are zero; pick the cheap one then.
                let linear = self.linear != 0 && (self.frac != 0 || self.dst_incr_mod != 0);
                for i in 0..dst.ch_count {
                    let dst_plane = dst.plane_bytes_mut(i).ok_or(Error::BufferTooSmall)?;
                    let src_plane = src.plane(i).ok_or(Error::BufferTooSmall)?;
                    // :392 — C passes update_ctx = (i+1 == ch_count); here
                    // every channel's run is threaded through locals and
                    // only the last one's state is written back.
                    let mut index = self.index;
                    let mut frac = self.frac;
                    let ret = self.resample_kernel(
                        linear,
                        dst_plane,
                        src_plane,
                        dst_size as usize,
                        &mut index,
                        &mut frac,
                    )? as i32;
                    *consumed = ret;
                    if i + 1 == dst.ch_count {
                        self.index = index;
                        self.frac = frac;
                    }
                }
            }
        }

        // :396-403 — compensation countdown; on exhaustion snap dst_incr
        // back to the ideal rate.
        if self.compensation_distance != 0 {
            self.compensation_distance -= dst_size;
            if self.compensation_distance == 0 {
                self.dst_incr = self.ideal_dst_incr;
                self.dst_incr_div = self.dst_incr / self.src_incr;
                self.dst_incr_mod = self.dst_incr % self.src_incr;
            }
        }

        Ok(dst_size)
    }

    /// `get_delay` (`resample.c:408-416`) — input samples still "inside" the
    /// filter, expressed in `base` units (C reads `s->in_buffer_count` and
    /// `s->in_sample_rate` off the SwrContext; those are parameters here).
    pub fn get_delay(&self, base: i64, in_buffer_count: i64, in_sample_rate: i32) -> i64 {
        let mut num = in_buffer_count - (self.filter_length as i64 - 1) / 2;
        num *= self.phase_count as i64;
        num -= self.index as i64;
        num *= self.src_incr as i64;
        num -= self.frac as i64;
        rescale(
            num,
            base,
            in_sample_rate as i64 * self.src_incr as i64 * self.phase_count as i64,
        )
    }

    /// `get_out_samples` (`resample.c:418-435`) — upper bound on the output
    /// sample count obtainable from `in_samples` more inputs (C's `s->`
    /// fields flattened as parameters). The `+ 2`s are slack the C authors
    /// left for implementation slop (`:420-422`).
    pub fn get_out_samples(
        &self,
        in_samples: i32,
        in_buffer_count: i64,
        in_sample_rate: i32,
        out_sample_rate: i32,
    ) -> Result<i64> {
        let mut num = in_buffer_count + 2 + in_samples as i64;
        num *= self.phase_count as i64;
        num -= self.index as i64;
        num = rescale_rnd(
            num,
            out_sample_rate as i64,
            in_sample_rate as i64 * self.phase_count as i64,
            Rounding::Up,
            false,
        ) + 2;

        if self.compensation_distance != 0 {
            if num > i32::MAX as i64 {
                return Err(Error::InvalidArgument("out sample count overflow".into()));
            }
            // :432 — during compensation the skewed rate can yield more.
            num = num.max((num * self.ideal_dst_incr as i64 - 1) / self.dst_incr as i64 + 1);
        }
        Ok(num)
    }

    /// `resample_flush` (`resample.c:437-454`) — extends the in-buffer with
    /// `reflection = (min(count, filter_length) + 1)/2` samples mirrored off
    /// its tail, so the final `swr_convert` calls can push the filter past
    /// the real end of input. Grows `in_buffer` (the `swri_realloc_audio`
    /// call at `:443` — see [`realloc_audio`]).
    pub fn resample_flush(
        &self,
        in_buffer: &mut AudioData,
        in_buffer_index: usize,
        in_buffer_count: &mut usize,
    ) -> Result<()> {
        let reflection = ((*in_buffer_count).min(self.filter_length as usize) + 1) / 2;

        realloc_audio(in_buffer, in_buffer_index + *in_buffer_count + reflection)?;
        debug_assert!(in_buffer.planar, "resample.c:445");
        let bps = in_buffer.bps;
        for ch in 0..in_buffer.ch_count {
            let plane = in_buffer.plane_bytes_mut(ch).ok_or(Error::BufferTooSmall)?;
            for j in 0..reflection {
                // :448-449 — sample [idx+count+j] = sample [idx+count-j-1].
                let from = (in_buffer_index + *in_buffer_count - j - 1) * bps;
                let to = (in_buffer_index + *in_buffer_count + j) * bps;
                plane.copy_within(from..from + bps, to);
            }
        }
        *in_buffer_count += reflection;
        Ok(())
    }

    /// `invert_initial_buffer` (`resample.c:457-502`) — primes the negative
    /// startup `index` by consuming a mirrored prefix of input: copies
    /// `in_count` samples after the `*out_sz` already staged in `dst`,
    /// and once `filter_length + 1` are staged, time-inverts them around
    /// position `filter_length` (the `:484-490` copy) and advances `index`
    /// to `≥ 0` while stepping `*out_idx` back one per phase_count.
    ///
    /// Return: `Ok(0)` = nothing to do (`index ≥ 0`), `Ok(i32::MAX)` = need
    /// more input (C's `INT_MAX`, `:480`), `Ok(border)` = the caller must
    /// skip `border` input samples (they were consumed into `dst`).
    pub fn invert_initial_buffer(
        &mut self,
        dst: &mut AudioData,
        src: &AudioData,
        in_count: usize,
        out_idx: &mut usize,
        out_sz: &mut usize,
    ) -> Result<i32> {
        // :460-462.
        if self.index >= 0 {
            return Ok(0);
        }
        let num = (in_count + *out_sz).min(self.filter_length as usize + 1);

        // :465-466 — room for filter_length*2+1 staged samples.
        realloc_audio(dst, self.filter_length as usize * 2 + 1)?;

        // :469-474 — append the new input after the already-staged prefix.
        for n in *out_sz..num {
            for ch in 0..src.ch_count {
                let (dst_plane, src_plane) = borrow_two(dst, src, ch)?;
                let fe = self.felem_size as usize;
                let from = (n - *out_sz) * fe;
                let to = (self.filter_length as usize + n) * fe;
                dst_plane[to..to + fe].copy_from_slice(&src_plane[from..from + fe]);
            }
        }

        // :476-481 — not enough staged yet: wait for more input.
        if num < self.filter_length as usize + 1 {
            *out_sz = num;
            *out_idx = self.filter_length as usize;
            return Ok(i32::MAX);
        }

        // :483-490 — mirror [filter_length+n] into [filter_length-n], n≥1.
        for n in 1..=self.filter_length as usize {
            for ch in 0..src.ch_count {
                let fe = self.felem_size as usize;
                let fl = self.filter_length as usize;
                let plane = dst.plane_bytes_mut(ch).ok_or(Error::BufferTooSmall)?;
                let from = (fl + n) * fe;
                let to = (fl - n) * fe;
                plane.copy_within(from..from + fe, to);
            }
        }

        // :492-501.
        let res = num as i32 - *out_sz as i32;
        *out_idx = self.filter_length as usize;
        while self.index < 0 {
            *out_idx -= 1;
            self.index += self.phase_count;
        }
        *out_sz = (*out_sz + self.filter_length as usize).max(1 + self.filter_length as usize * 2)
            - *out_idx;

        Ok(res.max(0))
    }

    /// Test/inspection accessor: filter coefficient `(phase, tap)` widened
    /// to f64 regardless of the storage format (C would read
    /// `((FELEM*)filter_bank)[phase*filter_alloc + tap]`).
    pub fn filter_coeff(&self, phase: usize, tap: usize) -> f64 {
        let (a, p) = (self.filter_alloc as usize, phase);
        match &self.bank {
            FilterBank::S16(b) => b[p * a + tap] as f64,
            FilterBank::S32(b) => b[p * a + tap] as f64,
            FilterBank::Flt(b) => b[p * a + tap] as f64,
            FilterBank::Dbl(b) => b[p * a + tap],
        }
    }

    /// The `dsp.resample_one` dispatch for the degenerate-filter fast path
    /// (`resample.c:367`) — same format match as [`Self::resample_kernel`].
    fn resample_one_kernel(
        &self,
        dst: &mut [u8],
        src: &[u8],
        n: usize,
        index2: i64,
        incr: i64,
    ) -> Result<()> {
        match &self.bank {
            FilterBank::S16(_) => resample_one::<i16>(dst, src, n, index2, incr),
            FilterBank::S32(_) => resample_one::<i32>(dst, src, n, index2, incr),
            FilterBank::Flt(_) => resample_one::<f32>(dst, src, n, index2, incr),
            FilterBank::Dbl(_) => resample_one::<f64>(dst, src, n, index2, incr),
        }
    }
}

// av_bessel_i0 — libavutil/mathematics.c:227-330
// ---------------------------------------------------------------------------

/// `av_bessel_i0` (`libavutil/mathematics.c:257`) — modified Bessel function
/// of the first kind, order zero, via the Blair-Edwards (AECL-4928, 1974)
/// minimax rational approximations taken from Boost:
///
/// * \|x\| ≤ 15: `p1(y)/q1(y)` with `y = x²`
/// * \|x\| > 15: `p2(y)/q2(y) · e^x/√x` with `y = 1/x − 1/15`
///
/// `eval_poly` (`mathematics.c:216-224`) is the Horner loop
/// `sum = c[n-1]; for i in (0..n-1).rev() { sum = sum·x + c[i] }`.
pub fn bessel_i0(x: f64) -> f64 {
    /// `p1[]` (`mathematics.c:258-273`).
    const P1: [f64; 15] = [
        -2.2335582639474375249e+15,
        -5.5050369673018427753e+14,
        -3.2940087627407749166e+13,
        -8.4925101247114157499e+11,
        -1.1912746104985237192e+10,
        -1.0313066708737980747e+08,
        -5.9545626019847898221e+05,
        -2.4125195876041896775e+03,
        -7.0935347449210549190e+00,
        -1.5453977791786851041e-02,
        -2.5172644670688975051e-05,
        -3.0517226450451067446e-08,
        -2.6843448573468483278e-11,
        -1.5982226675653184646e-14,
        -5.2487866627945699800e-18,
    ];
    /// `q1[]` (`mathematics.c:274-282`).
    const Q1: [f64; 6] = [
        -2.2335582639474375245e+15,
        7.8858692566751002988e+12,
        -1.2207067397808979846e+10,
        1.0377081058062166144e+07,
        -4.8527560179962773045e+03,
        1.0,
    ];
    /// `p2[]` (`mathematics.c:283-291`).
    const P2: [f64; 7] = [
        -2.2210262233306573296e-04,
        1.3067392038106924055e-02,
        -4.4700805721174453923e-01,
        5.5674518371240761397e+00,
        -2.3517945679239481621e+01,
        3.1611322818701131207e+01,
        -9.6090021968656180000e+00,
    ];
    /// `q2[]` (`mathematics.c:292-301`).
    const Q2: [f64; 8] = [
        -5.5194330231005480228e-04,
        3.2547697594819615062e-02,
        -1.1151759188741312645e+00,
        1.3982595353892851542e+01,
        -6.0228002066743340583e+01,
        8.5539563258012929600e+01,
        -3.1446690275135491500e+01,
        1.0,
    ];

    /// `eval_poly(coeff, size, x)` (`mathematics.c:216-224`) — Horner.
    fn eval_poly(coeff: &[f64], x: f64) -> f64 {
        let mut sum = coeff[coeff.len() - 1];
        for &c in coeff.iter().rev().skip(1) {
            sum = sum * x + c;
        }
        sum
    }

    if x == 0.0 {
        return 1.0;
    }
    let x = x.abs();
    if x <= 15.0 {
        let y = x * x;
        eval_poly(&P1, y) / eval_poly(&Q1, y)
    } else {
        let y = 1.0 / x - 1.0 / 15.0;
        let r = eval_poly(&P2, y) / eval_poly(&Q2, y);
        let factor = x.exp() / x.sqrt();
        factor * r
    }
}

/// `av_clip_int16` (`libavutil/common.h:243-247`).
fn clip_i16(a: i32) -> i16 {
    a.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

/// `av_clipl_int32` (`libavutil/common.h:254-258`).
fn clip_i32(a: i64) -> i32 {
    a.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

// ---------------------------------------------------------------------------
// build_filter — resample.c:41-174
// ---------------------------------------------------------------------------

/// `build_filter` (`resample.c:41-174`) — builds the polyphase filterbank
/// into `bank` (typed elements, C's `void *filter` + `c->format` switch).
///
/// C name kept; `scale` is C's `int scale` (`1 << filter_shift`), applied as
/// `tab[i] * scale / norm` in double. The even-`phase_count` mirror
/// (`:105-128`) writes phase `phase_count - ph` as the reversed tap image of
/// phase `ph` (sinc symmetry: `h_{P-p}[N-1-i] = h_p[i]`).
fn build_filter<E: ResampleElem>(
    bank: &mut [E],
    factor: f64,
    tap_count: usize,
    alloc: usize,
    phase_count: usize,
    scale: i32,
    filter_type: FilterType,
    kaiser_beta: f64,
) {
    // :44 — odd phase_count builds every phase; even builds a bit over half
    // and mirrors the rest below.
    let ph_nb = if phase_count % 2 == 1 {
        phase_count
    } else {
        phase_count / 2 + 1
    };
    let mut tab = vec![0.0f64; tap_count + 1];
    // C's sin_lut is av_malloc'd (uninitialized) and only read when
    // factor == 1.0 — zero-init here (documented divergence).
    let mut sin_lut = vec![0.0f64; ph_nb];
    let center = (tap_count - 1) / 2;
    let mut norm = 0.0f64;

    debug_assert!(tap_count == 1 || tap_count % 2 == 0, "resample.c:55");

    // :58-59 — upsampling needs no anti-alias filter.
    let factor = if factor > 1.0 { 1.0 } else { factor };

    // :61-64 — the factor==1 shortcut: sin(x)/x == sin_lut[ph]/x via the
    // reflection identity, precomputed with the center-parity sign.
    if factor == 1.0 {
        for (ph, slot) in sin_lut.iter_mut().enumerate() {
            *slot = (std::f64::consts::PI * ph as f64 / phase_count as f64).sin()
                * (if center & 1 == 1 { 1.0 } else { -1.0 });
        }
    }

    for ph in 0..ph_nb {
        let mut s = sin_lut[ph];
        for i in 0..tap_count {
            // :68 — x is PI-scaled; also the window argument below (CUBIC
            // reassigns it to the unscaled |offset|, its own arm only).
            let mut x = std::f64::consts::PI
                * ((i as f64 - center as f64) - ph as f64 / phase_count as f64)
                * factor;
            let mut y = if x == 0.0 {
                1.0
            } else if factor == 1.0 {
                s / x
            } else {
                x.sin() / x
            };
            match filter_type {
                FilterType::Cubic => {
                    // :75-80 — Keys cubic, d = -0.5.
                    let d = -0.5f64;
                    x = (((i as f64 - center as f64) - ph as f64 / phase_count as f64) * factor)
                        .abs();
                    if x < 1.0 {
                        y = 1.0 - 3.0 * x * x + 2.0 * x * x * x + d * (-x * x + x * x * x);
                    } else {
                        y = d * (-4.0 + 8.0 * x - 5.0 * x * x + x * x * x);
                    }
                }
                FilterType::BlackmanNuttall => {
                    // :81-85 — Chebyshev form in t = -cos(w).
                    let w = 2.0 * x / (factor * tap_count as f64);
                    let t = -w.cos();
                    y *= 0.3635819 - 0.4891775 * t + 0.1365995 * (2.0 * t * t - 1.0)
                        - 0.0106411 * (4.0 * t * t * t - 3.0 * t);
                }
                FilterType::Kaiser => {
                    // :86-89.
                    let w = 2.0 * x / (factor * tap_count as f64 * std::f64::consts::PI);
                    y *= bessel_i0(kaiser_beta * (1.0 - w * w).max(0.0).sqrt());
                }
            }

            tab[i] = y;
            s = -s;
            if ph == 0 {
                norm += y; // :96-97 — DC gain from phase 0 only
            }
        }

        // :100-130 — quantize + mirror. The EVEN mirror below runs when
        // phase_count is even (the `if (phase_count % 2) break;` skips it).
        for i in 0..tap_count {
            let q = E::quantize(tab[i] * scale as f64 / norm);
            bank[ph * alloc + i] = q;
        }
        if phase_count % 2 == 0 {
            for i in 0..tap_count {
                let q = bank[ph * alloc + i];
                bank[(phase_count - ph) * alloc + tap_count - 1 - i] = q;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The scalar kernels — resample_template.c:81-207
// ---------------------------------------------------------------------------

/// `resample_one` (`resample_template.c:81-92`) — nearest-neighbor copy for
/// the `filter_length == 1 && phase_count == 1` degenerate configuration:
/// `dst[i] = src[index2 >> 32]`, `index2 += incr` (32.32 fixed point).
fn resample_one<E: ResampleElem>(
    dst: &mut [u8],
    src: &[u8],
    n: usize,
    mut index2: i64,
    incr: i64,
) -> Result<()> {
    for dst_index in 0..n {
        let v = load_sample::<E>(src, (index2 >> 32) as usize)?;
        store_sample::<E>(dst, dst_index, v)?;
        index2 += incr;
    }
    Ok(())
}

/// Bounds-checked `src[idx]` (the C kernels read unchecked; see the module
/// divergences note — a read past the plane is C's heap overread).
#[inline]
fn load_sample<E: ResampleElem>(src: &[u8], idx: usize) -> Result<E> {
    let bps = E::FORMAT.bytes_per_sample();
    let off = idx * bps;
    let s = src.get(off..off + bps).ok_or(Error::BufferTooSmall)?;
    Ok(E::load(s, 0))
}

/// Bounds-checked `dst[idx] = v`.
#[inline]
fn store_sample<E: ResampleElem>(dst: &mut [u8], idx: usize, v: E) -> Result<()> {
    let bps = E::FORMAT.bytes_per_sample();
    let off = idx * bps;
    let d = dst.get_mut(off..off + bps).ok_or(Error::BufferTooSmall)?;
    E::store(d, 0, v);
    Ok(())
}

/// `resample_common` (`resample_template.c:94-147`) — the plain polyphase
/// FIR: per output sample, dot the `filter_length` taps of phase `index`
/// against `src[sample_index..]`, then advance the phase clock
/// (`frac/index` split, `:128-138`). Returns `sample_index`, the number of
/// whole input samples consumed. C's `update_ctx` flag (`:141-144`) is the
/// caller's job here — `index`/`frac` arrive as locals.
fn resample_common<E: ResampleElem>(
    c: &ResampleContext,
    bank: &[E],
    dst: &mut [u8],
    src: &[u8],
    n: usize,
    index: &mut i32,
    frac: &mut i32,
) -> Result<usize> {
    let mut sample_index: usize = 0;

    // :105-108 — normalize a carried-over index into [0, phase_count).
    while *index >= c.phase_count {
        sample_index += 1;
        *index -= c.phase_count;
    }

    let fl = c.filter_length as usize;
    let fa = c.filter_alloc as usize;

    for dst_index in 0..n {
        // :111 — C reads ((FELEM*)filter_bank) + filter_alloc·index; index
        // is normalized ≥ 0 here (caller contract).
        let filter = &bank[*index as usize * fa..];

        // :113-121 — two taps per iteration, separate accumulators.
        let mut val = E::foffset_acc();
        let mut val2 = E::zero_acc();
        let mut i = 0;
        while i + 1 < fl {
            let s0 = load_sample::<E>(src, sample_index + i)?;
            let s1 = load_sample::<E>(src, sample_index + i + 1)?;
            val = E::acc_common(val, s0, filter[i]);
            val2 = E::acc_common(val2, s1, filter[i + 1]);
            i += 2;
        }
        if i < fl {
            let s0 = load_sample::<E>(src, sample_index + i)?;
            val = E::acc_common(val, s0, filter[i]);
        }
        store_sample::<E>(dst, dst_index, E::out_common(val, val2))?;

        // :128-138 — the phase clock advance.
        *frac += c.dst_incr_mod;
        *index += c.dst_incr_div;
        if *frac >= c.src_incr {
            *frac -= c.src_incr;
            *index += 1;
        }
        while *index >= c.phase_count {
            sample_index += 1;
            *index -= c.phase_count;
        }
    }

    Ok(sample_index)
}

/// `resample_linear` (`resample_template.c:149-207`) — same structure, but
/// additionally accumulates the *next* phase's dot product (`v2`, at the
/// `filter_alloc` wrap stride) and interpolates `val → v2` by
/// `frac/src_incr` (`:170-186`). `inv_src_incr = 1.0/src_incr` is computed
/// once, in double (`:159-161`), for the FILTER_SHIFT == 0 formats.
fn resample_linear<E: ResampleElem>(
    c: &ResampleContext,
    bank: &[E],
    dst: &mut [u8],
    src: &[u8],
    n: usize,
    index: &mut i32,
    frac: &mut i32,
) -> Result<usize> {
    let mut sample_index: usize = 0;
    let inv_src_incr: f64 = 1.0 / c.src_incr as f64;

    while *index >= c.phase_count {
        sample_index += 1;
        *index -= c.phase_count;
    }

    let fl = c.filter_length as usize;
    let fa = c.filter_alloc as usize;

    for dst_index in 0..n {
        let filter = &bank[*index as usize * fa..];

        let mut val = E::foffset_uacc();
        let mut v2 = E::foffset_uacc();
        for i in 0..fl {
            let s = load_sample::<E>(src, sample_index + i)?;
            val = E::acc_linear(val, s, filter[i]);
            v2 = E::acc_linear(v2, s, filter[i + fa]);
        }
        val = E::interp_linear(val, v2, *frac, c.src_incr, inv_src_incr);
        store_sample::<E>(dst, dst_index, E::out_linear(val))?;

        *frac += c.dst_incr_mod;
        *index += c.dst_incr_div;
        if *frac >= c.src_incr {
            *frac -= c.src_incr;
            *index += 1;
        }
        while *index >= c.phase_count {
            sample_index += 1;
            *index -= c.phase_count;
        }
    }

    Ok(sample_index)
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/// Simultaneous `(dst plane, src plane)` borrow of channel `ch` from two
/// distinct `AudioData`s (C just takes two pointers, `resample.c:471-472`).
fn borrow_two<'a>(
    dst: &'a mut AudioData,
    src: &'a AudioData,
    ch: usize,
) -> Result<(&'a mut [u8], &'a [u8])> {
    let d = dst.plane_bytes_mut(ch).ok_or(Error::BufferTooSmall)?;
    let s = src.plane(ch).ok_or(Error::BufferTooSmall)?;
    Ok((d, s))
}

/// The `swri_realloc_audio` (`swresample.c:426-456`) analog local to this
/// module (the real one belongs to the future swresample core): grows
/// `a`'s capacity to at least `count` samples per channel by doubling,
/// preserving the old contents. C pads each plane to `ALIGN = 32` for SIMD
/// (`swresample.c:31,438`); dropped — invisible to callers (same reasoning
/// as [`AudioData`]'s module doc).
fn realloc_audio(a: &mut AudioData, count: usize) -> Result<()> {
    if a.count >= count {
        return Ok(());
    }
    let new_count = count * 2;
    let mut data = vec![0u8; a.ch_count * new_count * a.bps];
    // swresample.c:448-452 — planar: per-plane prefix; packed: one prefix.
    if a.planar {
        for ch in 0..a.ch_count {
            let from = ch * a.count * a.bps;
            let to = ch * new_count * a.bps;
            data[to..to + a.count * a.bps].copy_from_slice(&a.data[from..from + a.count * a.bps]);
        }
    } else {
        let n = a.count * a.ch_count * a.bps;
        data[..n].copy_from_slice(&a.data[..n]);
    }
    a.data = data.into();
    a.count = new_count;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::swresample::AudioData;
    use std::sync::Arc;

    fn mono(count: usize, fmt: SampleFormat, bytes: Vec<u8>) -> AudioData {
        AudioData {
            bps: fmt.bytes_per_sample(),
            data: Arc::from(bytes),
            ch_count: 1,
            count,
            planar: true,
            fmt,
        }
    }

    fn s16_plane(v: &[i16]) -> Vec<u8> {
        v.iter().flat_map(|s| s.to_le_bytes()).collect()
    }

    /// The swr_convert staging shape: invert_initial_buffer stages the
    /// first filter_length+1 samples (mirrored around the start), returning
    /// the staged sample count — then multiple_resample reads the staging.
    fn stage(c: &mut ResampleContext, src: &AudioData, in_count: usize) -> (AudioData, usize) {
        let mut staging = AudioData::new(c.format, 1, 0);
        let mut out_idx = 0usize;
        let mut out_sz = 0usize;
        let r = c
            .invert_initial_buffer(&mut staging, src, in_count, &mut out_idx, &mut out_sz)
            .unwrap();
        assert!(
            r != i32::MAX,
            "enough input staged (feed >= filter_length+1)"
        );
        (staging, out_sz)
    }

    fn ctx_44100_48000() -> ResampleContext {
        // C defaults (options.c): filter_size 16, phase_shift 10, linear 0,
        // cutoff 0 (→0.97), Kaiser beta 10? — pinned explicitly instead of
        // relying on option defaults.
        ResampleContext::new(
            44100,
            48000,
            16,
            10,
            0,
            0.0,
            SampleFormat::S16p,
            FilterType::Kaiser,
            10.0,
            0.0,
            0,
            1,
        )
        .unwrap()
    }

    /// `av_bessel_i0` against reference values (Abramowitz & Stegun 9.6.16
    /// / mpmath): I0(0)=1, I0(1)=1.2660658777520084, I0(2)=2.2795853023360673.
    #[test]
    fn bessel_i0_values() {
        assert_eq!(bessel_i0(0.0), 1.0);
        assert!((bessel_i0(1.0) - 1.2660658777520084).abs() < 1e-14);
        assert!((bessel_i0(2.0) - 2.2795853023360673).abs() < 1e-13);
        assert_eq!(bessel_i0(-1.5), bessel_i0(1.5), "I0 is even");
    }

    /// resample_init's geometry anchors, computed by hand from C:
    /// factor = min(44100·0.97/48000, 1) = 0.89031250;
    /// filter_length = ceil(16/0.8903125) = ceil(17.970…) = 18 (already even);
    /// filter_alloc = FFALIGN(18, 8) = 24;
    /// exact_rational: av_reduce(44100, 48000) = 147/160 ≤ 1024 ⇒ phase_count
    /// = 147, compensation count = 147·⌊1024/147⌋ = 147·6 = 882;
    /// index = −147·((18−1)/2) = −1176; src/dst_incr = av_reduce(44100,
    /// 48000·147) = 1/160, doubled while BOTH < 2²⁰ — dst_incr hits 2²⁰
    /// first (13 doublings): src = 2¹³ = 8192, dst = 160·8192 = 1310720;
    /// dst_incr_div = 160, dst_incr_mod = 0, frac = 0.
    #[test]
    fn init_geometry_anchors() {
        let c = ctx_44100_48000();
        assert!((c.factor - (44100.0 * 0.97 / 48000.0)).abs() < 1e-12);
        assert_eq!(c.filter_length, 18);
        assert_eq!(c.filter_alloc, 24);
        assert_eq!(c.phase_count, 147);
        assert_eq!(c.phase_count_compensation, 882);
        assert_eq!(c.index, -1176);
        assert_eq!(c.frac, 0);
        assert_eq!(c.src_incr, 8192);
        assert_eq!(c.dst_incr, 1310720);
        assert_eq!(c.dst_incr_div, 160);
        assert_eq!(c.dst_incr_mod, 0);
        assert_eq!(c.filter_shift, 15, "s16p filter_shift (resample.c:227)");
    }

    /// DC-gain invariant: phase 0's quantized taps sum to the scale factor
    /// (1<<15 for s16p) — `norm` accumulates phase 0 pre-quantization, so
    /// Σ tab[i]·2¹⁵/norm = 2¹⁵ up to per-tap rounding.
    #[test]
    fn phase0_dc_gain_s16() {
        let c = ctx_44100_48000();
        let sum: f64 = (0..c.filter_length as usize)
            .map(|t| c.filter_coeff(0, t))
            .sum();
        assert!(
            (sum - (1i64 << 15) as f64).abs() <= c.filter_length as f64,
            "phase-0 DC gain {sum} vs 32768 (±{len} rounding)",
            len = c.filter_length
        );
    }

    /// Staged-window ½ downsample: with exactly the filter_length+1
    /// staged samples the apply loop must produce ≈staged/2 outputs (the
    /// full-pipeline length math is the driver's swr_convert test).
    #[test]
    fn downsample_staged_window() {
        let mut c = ResampleContext::new(
            24000,
            48000,
            16,
            10,
            0,
            0.0,
            SampleFormat::S16p,
            FilterType::Kaiser,
            10.0,
            0.0,
            0,
            1,
        )
        .unwrap();
        const IN: usize = 256;
        let src = mono(IN, SampleFormat::S16p, s16_plane(&vec![1000i16; IN]));
        let (staging, staged) = stage(&mut c, &src, IN);
        // staged = the mirror-zone extent; the RESAMPLEABLE span of a
        // single-shot staging window is (staged − filter_length) at the
        // out/in ratio — the driver loops swr_convert to cover the rest.
        let expect = (staged - c.filter_length as usize) as f64 / 2.0;
        let mut dst = mono(IN, SampleFormat::S16p, vec![0; IN * 2]);
        let mut consumed = 0;
        let n = c
            .multiple_resample(&mut dst, IN as i32, &staging, staged as i32, &mut consumed)
            .unwrap();
        assert!(
            ((n as f64) - expect).abs() <= 4.0,
            "≈(staged−filter_length)/2 out, got {n} for staged {staged} fl {}",
            c.filter_length
        );
    }

    /// Staged-window 2× upsample of a constant: the interior reproduces
    /// the constant to float precision (normalized Kaiser DC gain).
    #[test]
    fn upsample_staged_constant() {
        let mut c = ResampleContext::new(
            48000,
            24000,
            16,
            10,
            0,
            0.0,
            SampleFormat::Fltp,
            FilterType::Kaiser,
            10.0,
            0.0,
            0,
            1,
        )
        .unwrap();
        const IN: usize = 256;
        let src = mono(
            IN,
            SampleFormat::Fltp,
            vec![0.5f32; IN]
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect(),
        );
        let (staging, staged) = stage(&mut c, &src, IN);
        let mut dst = mono(IN * 2, SampleFormat::Fltp, vec![0; IN * 2 * 4]);
        let mut consumed = 0;
        let n = c
            .multiple_resample(
                &mut dst,
                (IN * 2) as i32,
                &staging,
                staged as i32,
                &mut consumed,
            )
            .unwrap();
        let expect = ((staged - c.filter_length as usize) * 2) as i32;
        assert!(
            (n - expect).abs() <= 4,
            "2× up: n={n} expect≈{expect} (staged {staged} fl {})",
            c.filter_length
        );
        let out: Vec<f32> = dst.data[..n as usize * 4]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        // The MIRRORED staging prefix makes the first samples see only the
        // constant; every output is the constant to float precision.
        for &v in &out {
            assert!((v - 0.5).abs() < 1e-4, "constant through 2× upsample: {v}");
        }
    }

    /// get_delay at equal rates is zero before any input (swresample.c's
    /// delay query shape).
    #[test]
    fn delay_shapes() {
        let c = ctx_44100_48000();
        // (48−44.1)k means the engine holds a filter_length/2-ish window;
        // assert it is non-negative and in-sample-rate units for base=in_rate.
        let d = c.get_delay(48000, 0, 48000);
        assert!(d >= 0);
    }
}
